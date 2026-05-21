#!/usr/bin/env python3
"""Run E2E generation without learned GNN relation predictions.

This is a document-reconstruction baseline, not an edge-classification model
ablation.  It keeps the canonical full-v7 -> DocumentIR -> RenderTreeIR
renderer path, but replaces GNN logits with deterministic relation sources:

* ``rules_only_no_merge``: heading stack + renderer only, no learned MERGE or
  PARENT_CHILD edges.
* ``rules_only_deterministic_merge``: same as above, plus a conservative
  adjacent-text continuation merge heuristic.

The script intentionally does not train, relabel, rebuild graphs, or mutate v7
facts.  It consumes existing full-v7 JSON and graph files only to preserve the
same graph-index -> v7-id bridge used by the production renderer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.batch_visual_qa_inference import (  # noqa: E402
    compile_tex,
    infer_document_title,
    render_decoded_tree_with_ir_backend,
    resolve_device,
    safe_filename,
    select_documents,
    write_json,
)
from scripts.pipeline.run_m05_e2e_comparison import (  # noqa: E402
    evaluate_generated_document,
    print_status,
    summarize_rows,
    write_summary_csv,
)
from scripts.pipeline.step5_generate_tex import load_node_records  # noqa: E402
from src.pipeline.v7_contract import assert_v7_content_json, assert_v7_graph_data  # noqa: E402
from src.reasoning.postprocess import DecodedEdge, MERGE, build_resolved_tree  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rules-mode",
        choices=["rules_only_no_merge", "rules_only_deterministic_merge"],
        default="rules_only_no_merge",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--document-id", action="append", default=[], help="Optional explicit document id; repeatable.")
    parser.add_argument("--match-threshold", type=float, default=0.58)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--pdflatex", default="pdflatex")
    parser.add_argument("--compile-runs", type=int, default=2)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--layout-dpi", type=int, default=72)
    parser.add_argument("--layout-max-pages", type=int, default=5)
    parser.add_argument("--render-table-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-output-dir", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    import torch

    if args.clean_output_dir and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    docs = select_documents(args)
    if not docs:
        raise ValueError("No documents selected for rules-only E2E comparison")

    device = resolve_device(args.device, torch=torch)
    rows: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        try:
            row = run_one_rules_document(
                doc,
                index=index,
                output_dir=args.output_dir,
                rules_mode=args.rules_mode,
                device=device,
                torch=torch,
                pdflatex=args.pdflatex,
                compile_runs=args.compile_runs,
                compile_timeout=args.compile_timeout,
                skip_compile=args.skip_compile,
                render_table_crops=args.render_table_crops,
            )
            row.update(evaluate_generated_document(doc, row, args))
        except Exception as exc:  # noqa: BLE001 - E2E batches should keep moving.
            doc_id = str(doc.get("document_id", f"doc_{index}"))
            doc_dir = args.output_dir / f"{index:02d}_{safe_filename(doc_id)}"
            doc_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "document_id": doc_id,
                "doc_dir": str(doc_dir),
                "generated_pdf_exists": False,
                "rules_mode": args.rules_mode,
                "error": repr(exc),
            }
            write_json(doc_dir / "e2e_record.json", row)
        rows.append(row)
        print_status(index, len(docs), row)

    payload = {
        "schema_version": "rules_only_e2e_comparison_v1",
        "manifest": str(args.manifest),
        "relation_source": args.rules_mode,
        "split": args.split,
        "limit": args.limit,
        "match_threshold": args.match_threshold,
        "render_table_crops": args.render_table_crops,
        "summary": summarize_rows(rows),
        "documents": rows,
    }
    write_json(args.output_dir / "e2e_comparison_manifest.json", payload)
    write_summary_csv(args.output_dir / "e2e_comparison_summary.csv", rows)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_one_rules_document(
    doc: dict[str, Any],
    *,
    index: int,
    output_dir: Path,
    rules_mode: str,
    device: Any,
    torch: Any,
    pdflatex: str,
    compile_runs: int,
    compile_timeout: int,
    skip_compile: bool,
    render_table_crops: bool,
) -> dict[str, Any]:
    document_id = str(doc["document_id"])
    safe_id = safe_filename(document_id)
    doc_dir = output_dir / f"{index:02d}_{safe_id}"
    doc_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(str(doc["pdf_path"]))
    content_json = Path(str(doc["content_json"]))
    graph_path = Path(str(doc["graph_path"]))
    assert_v7_content_json(content_json, require_styles=True)

    original_pdf = doc_dir / "original.pdf"
    shutil.copy2(pdf_path, original_pdf)

    data = torch.load(graph_path, map_location=device, weights_only=False)
    assert_v7_graph_data(data, graph_path)
    node_records = load_node_records(content_json, data)
    decoded_edges = deterministic_decoded_edges(node_records, rules_mode=rules_mode)
    root = build_resolved_tree(node_records, decoded_edges)

    predicted_relations_path = doc_dir / "predicted_relations.json"
    write_json(
        predicted_relations_path,
        {
            "schema_version": "rules_only_predicted_relations_v1",
            "document_id": document_id,
            "graph_path": str(graph_path),
            "relation_source": rules_mode,
            "merge_edges": [
                {"source": edge.source, "target": edge.target, "score": edge.score}
                for edge in decoded_edges
                if edge.label == MERGE
            ],
            "parent_child_edges": [],
            "notes": [
                "No learned GNN logits were used.",
                "Heading scope is built by the deterministic stack in TreeDecoder/build_resolved_tree.",
            ],
        },
    )

    title = infer_document_title(node_records)
    tex = render_decoded_tree_with_ir_backend(
        root,
        node_records=node_records,
        content_json=content_json,
        pdf_path=pdf_path,
        source_tex_path=Path(str(doc["tex_path"])) if doc.get("tex_path") else None,
        document_id=document_id,
        title=title,
        document_metadata=getattr(data, "document_metadata", None),
        predicted_relations_path=predicted_relations_path,
        table_asset_output_dir=doc_dir / "assets" if render_table_crops else None,
        figure_asset_output_dir=doc_dir / "assets" if render_table_crops else None,
        asset_latex_prefix="assets",
    )
    tex_path = doc_dir / "generated.tex"
    tex_path.write_text(tex, encoding="utf-8")

    compile_info: dict[str, Any] = {"skipped": bool(skip_compile)}
    generated_pdf = doc_dir / "generated.pdf"
    if not skip_compile:
        compile_info = compile_tex(
            tex_path,
            pdflatex=pdflatex,
            runs=compile_runs,
            timeout=compile_timeout,
        )

    source_pdf = doc_dir / f"{safe_id}_original.pdf"
    source_generated_pdf = doc_dir / f"{safe_id}_generated.pdf"
    shutil.copy2(original_pdf, source_pdf)
    if generated_pdf.exists():
        shutil.copy2(generated_pdf, source_generated_pdf)

    row = {
        "document_id": document_id,
        "doc_dir": str(doc_dir),
        "source_pdf": str(pdf_path),
        "source_graph": str(graph_path),
        "source_content_json": str(content_json),
        "predicted_relations": str(predicted_relations_path),
        "original_pdf": str(original_pdf),
        "generated_tex": str(tex_path),
        "generated_pdf": str(generated_pdf),
        "paired_original_pdf": str(source_pdf),
        "paired_generated_pdf": str(source_generated_pdf) if source_generated_pdf.exists() else None,
        "generated_pdf_exists": generated_pdf.exists(),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "rules_mode": rules_mode,
        "accepted_rule_merges": sum(1 for edge in decoded_edges if edge.label == MERGE),
        "compile": compile_info,
        "renderer": "ir",
    }
    write_json(doc_dir / "qa_record.json", row)
    write_json(doc_dir / "e2e_record.json", row)
    return row


def deterministic_decoded_edges(node_records: list[dict[str, Any]], *, rules_mode: str) -> list[DecodedEdge]:
    if rules_mode == "rules_only_no_merge":
        return []
    if rules_mode != "rules_only_deterministic_merge":
        raise ValueError(f"Unknown rules mode: {rules_mode}")
    edges: list[DecodedEdge] = []
    ordered = sorted(range(len(node_records)), key=lambda idx: reading_key(node_records[idx], idx))
    for left, right in zip(ordered, ordered[1:]):
        if deterministic_merge_allowed(node_records[left], node_records[right]):
            edges.append(DecodedEdge(source=left, target=right, label=MERGE, score=1.0))
    return edges


def deterministic_merge_allowed(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not merge_candidate_record(left) or not merge_candidate_record(right):
        return False
    left_text = record_text(left)
    right_text = record_text(right)
    if len(left_text) < 3 or len(right_text) < 3:
        return False
    if caption_like(left_text) or caption_like(right_text):
        return False
    page_delta = int(record_page(right) or 0) - int(record_page(left) or 0)
    if page_delta < 0 or page_delta > 1:
        return False
    if not spatially_continuous(left, right, page_delta=page_delta):
        return False
    if ends_with_hyphen(left_text):
        return True
    if not ends_with_terminal_punctuation(left_text) and starts_like_continuation(right_text):
        return True
    return False


def merge_candidate_record(record: dict[str, Any]) -> bool:
    typ = canonical_type(record)
    role = role_text(record)
    if typ not in {"text", "paragraph", "reference"}:
        return False
    if any(token in role for token in ("title", "heading", "caption", "equation", "formula", "figure", "table", "algorithm")):
        return False
    return True


def spatially_continuous(left: dict[str, Any], right: dict[str, Any], *, page_delta: int) -> bool:
    left_box = bbox(left)
    right_box = bbox(right)
    if left_box is None or right_box is None:
        return page_delta == 0
    if page_delta == 1:
        return True
    lx0, ly0, lx1, ly1 = left_box
    rx0, ry0, rx1, ry1 = right_box
    x_overlap = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    min_width = max(1.0, min(lx1 - lx0, rx1 - rx0))
    if x_overlap / min_width < 0.45:
        return False
    line_height = max(6.0, ly1 - ly0, ry1 - ry0)
    gap = ry0 - ly1
    return -0.35 * line_height <= gap <= 2.5 * line_height


def canonical_type(record: dict[str, Any]) -> str:
    for key in ("canonical_type", "type", "block_type"):
        value = str(record.get(key) or "").casefold()
        if value:
            if value in {"plain_text", "body_text"}:
                return "text"
            return value
    return "text"


def role_text(record: dict[str, Any]) -> str:
    values = []
    for key in ("layout_role", "role", "semantic_role", "layout_layer"):
        value = str(record.get(key) or "").casefold()
        if value:
            values.append(value)
    return " ".join(values)


def record_text(record: dict[str, Any]) -> str:
    return str(record.get("merged_text") or record.get("text") or record.get("text_for_embedding") or "").strip()


def record_page(record: dict[str, Any]) -> int | None:
    for key in ("page_idx", "page", "page_index"):
        value = record.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = record.get("bbox")
    if isinstance(value, list) and len(value) >= 4:
        try:
            return tuple(float(item) for item in value[:4])  # type: ignore[return-value]
        except Exception:
            return None
    return None


def reading_key(record: dict[str, Any], fallback: int) -> tuple[int, float, float, int]:
    page = record_page(record)
    box = bbox(record)
    if box is not None:
        return (page if page is not None else 0, box[1], box[0], fallback)
    for key in ("reading_order", "global_order", "order", "_gnn_view_index"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return (page if page is not None else 0, float(value), 0.0, fallback)
    return (page if page is not None else 0, float(fallback), 0.0, fallback)


def caption_like(text: str) -> bool:
    lower = text.strip().casefold()
    return lower.startswith(("figure ", "fig. ", "fig ", "table ", "tab. ", "algorithm ", "alg. "))


def ends_with_hyphen(text: str) -> bool:
    return text.rstrip().endswith(("-", "‐", "‑", "‒", "–", "—"))


def ends_with_terminal_punctuation(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "。", "！", "？", ":", ";", "；"))


def starts_like_continuation(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    first = stripped[0]
    if first.islower() or first in ",;:)]}":
        return True
    first_word = stripped.split(maxsplit=1)[0].casefold().strip(",.;:")
    return first_word in {
        "and",
        "or",
        "where",
        "which",
        "that",
        "while",
        "because",
        "for",
        "in",
        "of",
        "to",
        "the",
        "with",
        "by",
    }


if __name__ == "__main__":
    raise SystemExit(main())
