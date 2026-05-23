#!/usr/bin/env python3
"""Evaluate whether projected middle-fragment MERGE edges visibly affect output.

The projection model is trained on fragment graphs.  This audit maps its
cross-owner predicted MERGE pairs back to the full-v7 graph, forces those pairs
inside the normal TreeDecoder, renders both baseline and forced variants, and
compares generated.tex plus structure metrics.  It is deliberately small and
does not alter model weights, graph tensors, v7 source JSON, or the main
decoder policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.batch_visual_qa_inference import (  # noqa: E402
    infer_document_title,
    render_decoded_tree_with_ir_backend,
    safe_filename,
    write_json,
)
from scripts.pipeline.step5_generate_tex import load_node_records  # noqa: E402
from src.evaluation.comparison_structure import latex_file_to_comparison, write_comparison_json  # noqa: E402
from src.evaluation.structure_metrics import evaluate_comparison_structures  # noqa: E402
from src.pipeline.v7_contract import assert_v7_content_json, assert_v7_graph_data  # noqa: E402
from src.reasoning.postprocess import TreeDecoder, TreeDecoderConfig  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-dir", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--doc-ids", nargs="*", default=[])
    parser.add_argument("--max-forced-pairs-per-doc", type=int, default=64)
    parser.add_argument("--match-threshold", type=float, default=0.58)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_rows = {row_doc_id(row): row for row in load_manifest_rows(args.full_manifest)}
    projection_reports = load_projection_reports(args.projection_dir, doc_ids=args.doc_ids, limit=args.limit)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, projection_path in enumerate(projection_reports, start=1):
        payload = load_json(projection_path)
        doc_id = str(payload.get("doc_id") or projection_path.parent.name)
        full_row = full_rows.get(doc_id)
        if not full_row:
            failures.append({"doc_id": doc_id, "error": "missing_full_manifest_row"})
            continue
        try:
            row = process_one_doc(
                doc_id,
                full_row,
                projection_payload=payload,
                index=index,
                output_dir=args.output_dir,
                max_pairs=args.max_forced_pairs_per_doc,
                match_threshold=args.match_threshold,
                torch=torch,
            )
            rows.append(row)
            print(
                f"[{index}/{len(projection_reports)}] {doc_id} forced={row.get('forced_pair_count')} "
                f"tex_changed={row.get('generated_tex_changed')} "
                f"pb_delta={row.get('delta_paragraph_boundary_f1')}"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"doc_id": doc_id, "error": repr(exc)})
            print(f"[{index}/{len(projection_reports)}] {doc_id} ERROR {exc!r}")
    write_summary(args, rows, failures)
    return 0 if not failures else 1


def process_one_doc(
    doc_id: str,
    full_row: dict[str, Any],
    *,
    projection_payload: dict[str, Any],
    index: int,
    output_dir: Path,
    max_pairs: int,
    match_threshold: float,
    torch: Any,
) -> dict[str, Any]:
    doc_dir = output_dir / f"{index:02d}_{safe_filename(doc_id)}"
    base_dir = doc_dir / "base_no_projected_merge"
    forced_dir = doc_dir / "forced_projected_owner_merge"
    base_dir.mkdir(parents=True, exist_ok=True)
    forced_dir.mkdir(parents=True, exist_ok=True)

    content_json = Path(str(full_row["content_json"]))
    graph_path = Path(str(full_row["graph_path"]))
    source_tex = existing_path(full_row.get("tex_path") or full_row.get("main_tex") or full_row.get("source_tex"))
    pdf_path = existing_path(full_row.get("pdf_path"))
    assert_v7_content_json(content_json, require_styles=True)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    assert_v7_graph_data(graph, graph_path)
    node_records = load_node_records(content_json, graph)
    v7_to_gnn = build_v7_to_gnn_index(graph)
    forced_pairs = projected_owner_pairs_to_graph_pairs(
        projection_payload.get("cross_owner_predicted_pairs") or [],
        v7_to_gnn=v7_to_gnn,
        edge_index=graph.edge_index.detach().cpu().long(),
        max_pairs=max_pairs,
    )
    base_tex, base_metrics, base_trace = render_variant(
        doc_id,
        node_records=node_records,
        graph=graph,
        content_json=content_json,
        pdf_path=pdf_path,
        source_tex=source_tex,
        output_dir=base_dir,
        forced_pairs=[],
        match_threshold=match_threshold,
        torch=torch,
    )
    forced_tex, forced_metrics, forced_trace = render_variant(
        doc_id,
        node_records=node_records,
        graph=graph,
        content_json=content_json,
        pdf_path=pdf_path,
        source_tex=source_tex,
        output_dir=forced_dir,
        forced_pairs=forced_pairs,
        match_threshold=match_threshold,
        torch=torch,
    )
    row = {
        "doc_id": doc_id,
        "doc_dir": str(doc_dir),
        "projection_report": str(projection_payload.get("projection_report") or ""),
        "projected_cross_owner_pairs": len(projection_payload.get("cross_owner_predicted_pairs") or []),
        "forced_pair_count": len(forced_pairs),
        "base_generated_tex": str(base_dir / "generated.tex"),
        "forced_generated_tex": str(forced_dir / "generated.tex"),
        "generated_tex_changed": sha256_text(base_tex) != sha256_text(forced_tex),
        "base_paragraph_boundary_f1": metric_value(base_metrics, "paragraph_boundary_f1"),
        "forced_paragraph_boundary_f1": metric_value(forced_metrics, "paragraph_boundary_f1"),
        "base_paragraph_text_coverage_f1": metric_value(base_metrics, "paragraph_text_coverage_f1"),
        "forced_paragraph_text_coverage_f1": metric_value(forced_metrics, "paragraph_text_coverage_f1"),
        "base_macro_structure_score": metric_value(base_metrics, "macro_structure_score"),
        "forced_macro_structure_score": metric_value(forced_metrics, "macro_structure_score"),
        "base_accepted_merge_count": accepted_merge_count(base_trace),
        "forced_accepted_merge_count": accepted_merge_count(forced_trace),
    }
    for key in ("paragraph_boundary_f1", "paragraph_text_coverage_f1", "macro_structure_score"):
        base = row.get(f"base_{key}")
        forced = row.get(f"forced_{key}")
        row[f"delta_{key}"] = None if base is None or forced is None else forced - base
    write_json(doc_dir / "visibility_record.json", row)
    return row


def render_variant(
    doc_id: str,
    *,
    node_records: list[dict[str, Any]],
    graph: Any,
    content_json: Path,
    pdf_path: Path | None,
    source_tex: Path | None,
    output_dir: Path,
    forced_pairs: list[tuple[int, int]],
    match_threshold: float,
    torch: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    logits = torch.zeros((int(graph.edge_index.shape[1]), 3), dtype=torch.float32)
    logits[:, 2] = 8.0
    decoder = TreeDecoder(
        TreeDecoderConfig(
            merge_threshold=0.99,
            parent_threshold=0.99,
            heading_skeleton_mode="stack",
            forced_merge_pairs=tuple(forced_pairs),
        )
    )
    root = decoder.decode(node_records, graph.edge_index.detach().cpu(), logits)
    predicted_relations_path = output_dir / "predicted_relations.json"
    write_json(
        predicted_relations_path,
        {
            "schema_version": "middlefrag_projection_visibility_predicted_relations_v1",
            "relation_source": "none_logits_plus_forced_projected_merges",
            "forced_merge_pairs": [{"source": s, "target": t} for s, t in forced_pairs],
        },
    )
    tex = render_decoded_tree_with_ir_backend(
        root,
        node_records=node_records,
        content_json=content_json,
        pdf_path=pdf_path,
        source_tex_path=source_tex,
        document_id=doc_id,
        title=infer_document_title(node_records),
        document_metadata=getattr(graph, "document_metadata", None),
        predicted_relations_path=predicted_relations_path,
        decoder_trace=decoder.last_trace,
        attribution_output_path=output_dir / "relation_trace_report.json",
        table_asset_output_dir=None,
        figure_asset_output_dir=None,
        asset_latex_prefix="assets",
    )
    tex_path = output_dir / "generated.tex"
    tex_path.write_text(tex, encoding="utf-8")
    metrics: dict[str, Any] = {}
    if source_tex and source_tex.exists():
        gold = latex_file_to_comparison(source_tex, doc_id=doc_id)
        pred = latex_file_to_comparison(tex_path, doc_id=doc_id)
        write_comparison_json(gold, output_dir / "gold_structure.json")
        write_comparison_json(pred, output_dir / "generated_structure.json")
        metrics = evaluate_comparison_structures(gold.to_dict(), pred.to_dict(), match_threshold=match_threshold)
        write_json(output_dir / "structure_metrics.json", metrics)
    return tex, metrics, decoder.last_trace


def projected_owner_pairs_to_graph_pairs(
    pairs: list[dict[str, Any]],
    *,
    v7_to_gnn: dict[str, int],
    edge_index: Any,
    max_pairs: int,
) -> list[tuple[int, int]]:
    edge_pairs = {(int(edge_index[0, pos]), int(edge_index[1, pos])) for pos in range(int(edge_index.shape[1]))}
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in pairs:
        left_ids = [str(value) for value in pair.get("src_owner_v7_ids") or []]
        right_ids = [str(value) for value in pair.get("dst_owner_v7_ids") or []]
        for left_id in left_ids:
            for right_id in right_ids:
                if left_id not in v7_to_gnn or right_id not in v7_to_gnn:
                    continue
                src = v7_to_gnn[left_id]
                dst = v7_to_gnn[right_id]
                candidate = (src, dst)
                reverse = (dst, src)
                selected = candidate if candidate in edge_pairs else reverse if reverse in edge_pairs else None
                if selected is None or selected in seen:
                    continue
                seen.add(selected)
                out.append(selected)
                if len(out) >= max_pairs:
                    return out
    return out


def build_v7_to_gnn_index(graph: Any) -> dict[str, int]:
    values = getattr(graph, "gnn_to_v7_ids", None) or getattr(graph, "gnn_to_v7_id", None)
    if values is None:
        raise ValueError("graph missing gnn_to_v7 bridge")
    out: dict[str, int] = {}
    for idx, value in enumerate(values):
        if isinstance(value, (list, tuple)):
            ids = value
        else:
            ids = [value]
        for v7_id in ids:
            text = str(v7_id)
            if text and text not in out:
                out[text] = idx
    return out


def accepted_merge_count(trace: dict[str, Any]) -> int:
    value = trace.get("merge_components")
    if isinstance(value, list):
        return len(value)
    return int((trace.get("merge") or {}).get("accepted_merge_count") or 0) if isinstance(trace.get("merge"), dict) else 0


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        for nested in ("f1", "score", "accuracy"):
            if nested in value:
                return metric_value({key: value[nested]}, key)
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    summary = {
        "schema_version": "middlefrag_projection_visibility_summary_v1",
        "projection_dir": str(args.projection_dir),
        "full_manifest": str(args.full_manifest),
        "doc_count": len(rows),
        "failure_count": len(failures),
        "aggregate": aggregate(rows),
        "documents": rows,
        "failures": failures,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "summary.csv", rows)
    lines = [
        "# Middle-Fragment Projection Visibility Report",
        "",
        f"- docs: `{len(rows)}`",
        f"- failures: `{len(failures)}`",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in summary["aggregate"].items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- This is a visibility check for projected cross-owner fragment MERGE pairs.",
        "- It does not validate replacing the main structural GNN.",
        "- If generated.tex does not change, projected fragment MERGE is not being exposed at full-v7 output level.",
    ]
    (args.output_dir / "MIDDLEFRAG_PROJECTION_VISIBILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_tex_changed_docs": sum(1 for row in rows if row.get("generated_tex_changed")),
        "forced_pair_count": sum(int(row.get("forced_pair_count") or 0) for row in rows),
        "mean_delta_paragraph_boundary_f1": mean(row.get("delta_paragraph_boundary_f1") for row in rows),
        "mean_delta_paragraph_text_coverage_f1": mean(row.get("delta_paragraph_text_coverage_f1") for row in rows),
        "mean_delta_macro_structure_score": mean(row.get("delta_macro_structure_score") for row in rows),
    }


def load_projection_reports(root: Path, *, doc_ids: list[str], limit: int | None) -> list[Path]:
    wanted = {str(doc_id) for doc_id in doc_ids if str(doc_id)}
    paths = sorted(root.glob("per_doc/*/fragment_to_v7_projection.json"))
    if wanted:
        paths = [path for path in paths if path.parent.name in wanted or projection_doc_id(path) in wanted]
    if limit is not None:
        paths = paths[: max(0, int(limit))]
    return paths


def projection_doc_id(path: Path) -> str:
    try:
        return str(load_json(path).get("doc_id") or "")
    except Exception:
        return ""


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("documents") or payload.get("items") or payload.get("records") or []
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def row_doc_id(row: dict[str, Any]) -> str:
    for key in ("document_id", "doc_id", "id", "paper_id", "arxiv_id"):
        value = str(row.get(key) or "")
        if value:
            return value
    return ""


def existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Any) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
