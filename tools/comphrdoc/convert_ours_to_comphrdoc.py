#!/usr/bin/env python3
"""Convert our CompHRDoc E2E outputs into official HDS prediction JSON."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.comphrdoc.common import (  # noqa: E402
    OFFICIAL_CLASSES,
    bbox_center,
    bbox_iou,
    config_path,
    load_config,
    normalize_class,
    read_json,
    safe_doc_id,
    text_similarity,
    write_json,
)
from tools.comphrdoc.convert_to_comphrdoc import export_document_ir  # noqa: E402
from src.adapters.mineru_v7_document_ir import MinerUV7DocumentIRAdapterConfig, load_v7_document_ir  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/external_eval/comphrdoc_test500.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ours-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-name", help="Optional name recorded in the conversion manifest.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--tree-mode",
        choices=["current_semantic", "flat_root", "page_flat", "page_column_flat"],
        default="current_semantic",
        help="Benchmark-specific parent_id/relation construction strategy.",
    )
    parser.add_argument(
        "--use-hrdh-test-skeleton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep official test node count/page/text/bbox and map our predicted class/tree onto it.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    manifest = read_json(args.manifest or config_path(cfg, "outputs", "manifest"))
    docs = manifest.get("documents", manifest if isinstance(manifest, list) else [])
    if args.offset:
        docs = docs[args.offset :]
    if args.limit:
        docs = docs[: args.limit]
    out_dir = args.out_dir or (config_path(cfg, "outputs", "prediction_root") / f"{args.ours_run_dir.name}_pred")
    out_dir.mkdir(parents=True, exist_ok=True)
    e2e_manifest = read_json(args.ours_run_dir / "e2e" / "e2e_manifest.json")
    rows = e2e_manifest.get("documents", [])
    row_by_id = {str(row.get("document_id")): row for row in rows}
    converted = []
    failures = []
    for doc in docs:
        doc_id = str(doc["document_id"])
        row = row_by_id.get(doc_id)
        if not row or row.get("error"):
            failures.append({"document_id": doc_id, "error": row.get("error") if row else "missing e2e row"})
            continue
        output_path = out_dir / f"{safe_doc_id(doc_id)}.json"
        if args.skip_existing and output_path.exists():
            converted.append({"document_id": doc_id, "nodes": len(read_json(output_path)), "skipped_existing": True})
            continue
        try:
            pred_units = convert_one_doc(doc, row, use_skeleton=args.use_hrdh_test_skeleton, tree_mode=args.tree_mode)
            write_json(output_path, pred_units)
            converted.append({"document_id": doc_id, "nodes": len(pred_units)})
        except Exception as exc:  # noqa: BLE001
            failures.append({"document_id": doc_id, "error": repr(exc)})
    write_json(
        out_dir.parent / f"{out_dir.name}_conversion_manifest.json",
        {
            "schema_version": "ours_to_comphrdoc_conversion_v1",
            "ours_run_dir": str(args.ours_run_dir),
            "run_name": args.run_name or args.ours_run_dir.name,
            "out_dir": str(out_dir),
            "offset": args.offset,
            "limit": args.limit,
            "tree_mode": args.tree_mode,
            "converted": converted,
            "failures": failures,
            "use_hrdh_test_skeleton": args.use_hrdh_test_skeleton,
            "notes": [
                "HRDH test skeleton contributes only page/text/bbox/node count; class/parent/relation are mapped from our full-v7 DocumentIR export.",
                "This bridge is for official evaluator compatibility, not a claim that our PDF front-end detects the exact official line segmentation.",
            ],
        },
    )
    print(f"[comphrdoc] converted={len(converted)} failures={len(failures)} -> {out_dir}")
    return 0 if converted else 1


def convert_one_doc(
    doc: dict[str, Any],
    row: dict[str, Any],
    *,
    use_skeleton: bool,
    tree_mode: str = "current_semantic",
) -> list[dict[str, Any]]:
    content_json = Path(str(row.get("source_content_json") or row.get("content_json")))
    pdf_path = Path(str(row.get("source_pdf") or row.get("pdf_path"))) if row.get("source_pdf") or row.get("pdf_path") else None
    document = load_v7_document_ir(
        content_json,
        pdf_path=pdf_path,
        doc_id=str(doc["document_id"]),
        config=MinerUV7DocumentIRAdapterConfig(require_styles=False),
    )
    ours_records = [record.to_json() for record in export_document_ir(document)]
    normalize_prediction_records(ours_records)
    if not use_skeleton:
        return ours_records
    skeleton = read_json(Path(str(doc["hrdh_test_json"])))
    if not isinstance(skeleton, list):
        raise ValueError(f"HRDH test JSON is not a list: {doc['hrdh_test_json']}")
    matches = match_skeleton_to_ours(skeleton, ours_records)
    parent_repr = build_parent_representatives(matches, ours_records)
    output: list[dict[str, Any]] = []
    for index, unit in enumerate(skeleton):
        match_index = matches.get(index)
        ours = ours_records[match_index] if match_index is not None and match_index < len(ours_records) else None
        predicted_class = normalize_class(str(ours.get("class") if ours else heuristic_class(unit)))
        parent_id = -1
        relation = "meta" if predicted_class in {"title", "author", "mail", "affili", "header", "footer", "footnote"} else "contain"
        if ours is not None:
            relation = str(ours.get("relation") or relation)
            parent = int(ours.get("parent_id", -1))
            if parent >= 0 and parent in parent_repr:
                parent_id = parent_repr[parent]
            elif predicted_class == "paraline" and index > 0:
                parent_id = index - 1
                relation = "connect"
            elif predicted_class == "fstline":
                parent_id = nearest_previous_section(output)
                relation = "contain"
        elif predicted_class == "paraline" and index > 0:
            parent_id = index - 1
            relation = "connect"
        output.append(
            {
                "text": str(unit.get("text") or ""),
                "box": [int(round(float(v))) for v in unit.get("box", [0, 0, 0, 0])[:4]],
                "class": predicted_class,
                "page": int(unit.get("page", 0) or 0),
                "is_meta": predicted_class in {"title", "author", "mail", "affili", "header", "footer", "footnote"},
                "parent_id": int(parent_id),
                "relation": relation if relation in {"meta", "contain", "connect", "equality"} else "contain",
            }
        )
    apply_tree_mode(output, tree_mode)
    return output


def normalize_prediction_records(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        record["line_id"] = index
        record["class"] = normalize_class(str(record.get("class") or "paraline"))
        record["parent_id"] = int(record.get("parent_id", -1))
        record["relation"] = str(record.get("relation") or ("meta" if record.get("is_meta") else "contain"))


def match_skeleton_to_ours(skeleton: list[dict[str, Any]], ours: list[dict[str, Any]]) -> dict[int, int]:
    by_page: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(ours):
        by_page.setdefault(int(record.get("page", 0) or 0), []).append((index, record))
    matches: dict[int, int] = {}
    for skeleton_index, unit in enumerate(skeleton):
        page = int(unit.get("page", 0) or 0)
        candidates = by_page.get(page, [])
        best: tuple[float, int] | None = None
        for ours_index, record in candidates:
            score = match_score(unit, record)
            if best is None or score > best[0]:
                best = (score, ours_index)
        if best is not None and best[0] >= 0.08:
            matches[skeleton_index] = best[1]
    return matches


def match_score(unit: dict[str, Any], record: dict[str, Any]) -> float:
    ubox = unit.get("box", [0, 0, 0, 0])
    rbox = record.get("box", [0, 0, 0, 0])
    ux, uy = bbox_center(ubox)
    rx, ry = bbox_center(rbox)
    distance = math.hypot(ux - rx, uy - ry)
    spatial = bbox_iou(ubox, rbox)
    text = text_similarity(str(unit.get("text") or ""), str(record.get("text") or ""))
    return 2.0 * spatial + 0.7 * text - 0.0008 * distance


def build_parent_representatives(matches: dict[int, int], ours: list[dict[str, Any]]) -> dict[int, int]:
    representatives: dict[int, int] = {}
    for skeleton_index, ours_index in sorted(matches.items()):
        representatives.setdefault(ours_index, skeleton_index)
    return representatives


def nearest_previous_section(output: list[dict[str, Any]]) -> int:
    for index in range(len(output) - 1, -1, -1):
        if output[index].get("class") == "section":
            return index
    return -1


def heuristic_class(unit: dict[str, Any]) -> str:
    text = str(unit.get("text") or "").strip()
    raw = normalize_class(str(unit.get("class") or ""))
    if raw in OFFICIAL_CLASSES:
        return raw
    if not text:
        return "paraline"
    lower = text.casefold()
    if "@" in text:
        return "mail"
    if lower.startswith(("fig.", "figure", "table", "tab.", "algorithm")):
        return "caption"
    return "paraline"


def repair_parent_bounds(units: list[dict[str, Any]]) -> None:
    for index, unit in enumerate(units):
        parent = int(unit.get("parent_id", -1))
        if parent >= index or parent < -1:
            unit["parent_id"] = -1
            if unit.get("relation") != "meta":
                unit["relation"] = "contain"


def apply_tree_mode(units: list[dict[str, Any]], tree_mode: str) -> None:
    if tree_mode == "current_semantic":
        repair_parent_bounds(units)
        return
    if tree_mode == "flat_root":
        for unit in units:
            unit["parent_id"] = -1
            unit["relation"] = "meta" if unit.get("is_meta") else "contain"
        return
    if tree_mode == "page_flat":
        assign_page_flat_tree(units)
        return
    if tree_mode == "page_column_flat":
        assign_page_column_flat_tree(units)
        return
    raise ValueError(f"Unknown tree_mode={tree_mode}")


def assign_page_flat_tree(units: list[dict[str, Any]]) -> None:
    page_roots: dict[int, int] = {}
    for index, unit in enumerate(units):
        page = int(unit.get("page", 0) or 0)
        if page not in page_roots:
            page_roots[page] = index
            unit["parent_id"] = -1
            unit["relation"] = "meta"
        else:
            unit["parent_id"] = page_roots[page]
            unit["relation"] = "connect" if normalize_class(str(unit.get("class"))) == "paraline" else "contain"
    repair_parent_bounds(units)


def assign_page_column_flat_tree(units: list[dict[str, Any]]) -> None:
    page_roots: dict[int, int] = {}
    column_roots: dict[tuple[int, int], int] = {}
    page_centers = estimate_page_centers(units)
    for index, unit in enumerate(units):
        page = int(unit.get("page", 0) or 0)
        if page not in page_roots:
            page_roots[page] = index
            unit["parent_id"] = -1
            unit["relation"] = "meta"
            continue
        column = infer_column(unit, page_centers.get(page))
        key = (page, column)
        if key not in column_roots:
            column_roots[key] = index
            unit["parent_id"] = page_roots[page]
            unit["relation"] = "contain"
        else:
            unit["parent_id"] = column_roots[key]
            unit["relation"] = "connect" if normalize_class(str(unit.get("class"))) == "paraline" else "contain"
    repair_parent_bounds(units)


def estimate_page_centers(units: list[dict[str, Any]]) -> dict[int, float]:
    xs_by_page: dict[int, list[float]] = {}
    for unit in units:
        box = unit.get("box", [0, 0, 0, 0])
        if not isinstance(box, list) or len(box) < 4:
            continue
        page = int(unit.get("page", 0) or 0)
        x0, _, x1, _ = map(float, box[:4])
        xs_by_page.setdefault(page, []).append((x0 + x1) / 2.0)
    centers: dict[int, float] = {}
    for page, xs in xs_by_page.items():
        if xs:
            centers[page] = (min(xs) + max(xs)) / 2.0
    return centers


def infer_column(unit: dict[str, Any], center_x: float | None) -> int:
    if center_x is None:
        return 0
    box = unit.get("box", [0, 0, 0, 0])
    if not isinstance(box, list) or len(box) < 4:
        return 0
    x0, _, x1, _ = map(float, box[:4])
    width = x1 - x0
    if width <= 0:
        return 0
    x_center = (x0 + x1) / 2.0
    if x0 < center_x < x1 and width > abs(center_x - x_center) * 4:
        return 2
    return 0 if x_center < center_x else 1


if __name__ == "__main__":
    raise SystemExit(main())
