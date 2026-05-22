#!/usr/bin/env python3
"""Run the current model through E2E generation and structure/layout QA.

This script is intentionally a thin orchestration layer. It reuses the current
visual-QA inference path, then evaluates generated LaTeX against the matching
source TeX through the neutral comparison-structure schema.

The filename is kept for compatibility with older reports. The defaults now
target the current v7 float-proxy adapter-aware model family and the canonical
IR generator path.
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
    load_model,
    resolve_device,
    run_one_document,
    select_documents,
    write_json,
)
from src.evaluation.comparison_structure import latex_file_to_comparison, write_comparison_json  # noqa: E402
from src.evaluation.structure_metrics import evaluate_comparison_structures  # noqa: E402
from src.evaluation.visual_qa import compare_pdf_layouts  # noqa: E402
from src.reasoning.postprocess import TreeDecoder, TreeDecoderConfig  # noqa: E402


DEFAULT_MANIFEST = Path("data/00_manifests/v7_floatproxy_adapter_20260516_205926_trainable_recall98.json")
DEFAULT_CHECKPOINT = Path(
    "data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926/"
    "M06_y_network_plus_merge_gate/seed_7/best_model.pth"
)
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/current_e2e_comparison_hard20_floatcaption_rerun_20260518_132615")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--document-id", action="append", default=[], help="Optional explicit document id; repeatable.")
    parser.add_argument("--merge-threshold", type=float, default=0.37)
    parser.add_argument("--parent-threshold", type=float, default=0.45)
    parser.add_argument("--require-merge-argmax", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-parent-argmax", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--enable-layout-scope-continuation-merge",
        action="store_true",
        help="Experimental: allow a narrow deterministic text continuation merge across layout-band boundaries.",
    )
    parser.add_argument(
        "--enable-list-continuation-merge",
        action="store_true",
        help="Experimental: allow a narrow deterministic split-list-item continuation merge.",
    )
    parser.add_argument(
        "--enable-family-aware-merge-policy",
        action="store_true",
        help="Experimental: use family-specific MERGE thresholds/gates without changing the trained model.",
    )
    parser.add_argument("--family-body-list-merge-threshold", type=float, default=0.05)
    parser.add_argument("--family-reference-merge-threshold", type=float, default=0.82)
    parser.add_argument(
        "--enable-family-aware-missing-candidate-merge",
        action="store_true",
        help="Experimental: add only very narrow deterministic BODY/LIST continuation merges for missing candidate edges.",
    )
    parser.add_argument(
        "--heading-skeleton-mode",
        choices=["stack"],
        default="stack",
        help="Canonical decoder mode. Only stack is supported.",
    )
    parser.add_argument("--renderer", choices=["ir"], default="ir")
    parser.add_argument("--render-table-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--match-threshold", type=float, default=0.58)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pdflatex", default="pdflatex")
    parser.add_argument("--compile-runs", type=int, default=2)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--layout-dpi", type=int, default=72)
    parser.add_argument("--layout-max-pages", type=int, default=5)
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
        raise ValueError("No documents selected for current E2E comparison")

    device = resolve_device(args.device, torch=torch)
    model = load_model(args.checkpoint, device=device, torch=torch)
    decoder = TreeDecoder(
        TreeDecoderConfig(
            merge_threshold=args.merge_threshold,
            parent_threshold=args.parent_threshold,
            require_merge_argmax=args.require_merge_argmax,
            require_parent_argmax=args.require_parent_argmax,
            enable_layout_scope_continuation_merge=args.enable_layout_scope_continuation_merge,
            enable_list_continuation_merge=args.enable_list_continuation_merge,
            enable_family_aware_merge_policy=args.enable_family_aware_merge_policy,
            family_body_list_merge_threshold=args.family_body_list_merge_threshold,
            family_reference_merge_threshold=args.family_reference_merge_threshold,
            enable_family_aware_missing_candidate_merge=args.enable_family_aware_missing_candidate_merge,
            heading_skeleton_mode=args.heading_skeleton_mode,
        )
    )

    rows: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        try:
            row = run_one_document(
                doc,
                index=index,
                output_dir=args.output_dir,
                model=model,
                decoder=decoder,
                device=device,
                torch=torch,
                pdflatex=args.pdflatex,
                compile_runs=args.compile_runs,
                compile_timeout=args.compile_timeout,
                skip_compile=args.skip_compile,
                renderer=args.renderer,
                render_table_crops=args.render_table_crops,
                model_version=str(args.checkpoint),
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
                "error": repr(exc),
            }
            write_json(doc_dir / "e2e_record.json", row)
        rows.append(row)
        print_status(index, len(docs), row)

    payload = {
        "schema_version": "current_e2e_comparison_v1",
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "limit": args.limit,
        "merge_threshold": args.merge_threshold,
        "parent_threshold": args.parent_threshold,
        "heading_skeleton_mode": args.heading_skeleton_mode,
        "renderer": args.renderer,
        "render_table_crops": args.render_table_crops,
        "match_threshold": args.match_threshold,
        "summary": summarize_rows(rows),
        "documents": rows,
    }
    write_json(args.output_dir / "e2e_comparison_manifest.json", payload)
    write_summary_csv(args.output_dir / "e2e_comparison_summary.csv", rows)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if rows else 2


def evaluate_generated_document(doc: dict[str, Any], row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    doc_dir = Path(str(row["doc_dir"]))
    generated_tex = Path(str(row["generated_tex"]))
    source_tex = existing_path(doc.get("tex_path") or doc.get("main_tex") or doc.get("source_tex"))
    result: dict[str, Any] = {}

    if source_tex and generated_tex.exists():
        gold = latex_file_to_comparison(source_tex, doc_id=str(doc.get("document_id") or source_tex.stem))
        pred = latex_file_to_comparison(generated_tex, doc_id=str(doc.get("document_id") or generated_tex.stem))
        gold_path = doc_dir / "gold_structure.json"
        pred_path = doc_dir / "generated_structure.json"
        metrics_path = doc_dir / "structure_metrics.json"
        write_comparison_json(gold, gold_path)
        write_comparison_json(pred, pred_path)
        metrics = evaluate_comparison_structures(
            gold.to_dict(),
            pred.to_dict(),
            match_threshold=args.match_threshold,
        )
        write_json(metrics_path, metrics)
        result.update(
            {
                "source_tex": str(source_tex),
                "gold_structure": str(gold_path),
                "generated_structure": str(pred_path),
                "structure_metrics": str(metrics_path),
                "macro_structure_score": metrics.get("macro_structure_score"),
                "heading_tree_accuracy": (metrics.get("heading_tree_accuracy") or {}).get("score"),
                "reading_order_accuracy": (metrics.get("reading_order_accuracy") or {}).get("score"),
                "paragraph_boundary_f1": (metrics.get("paragraph_boundary_f1") or {}).get("f1"),
                "paragraph_text_coverage_f1": (metrics.get("paragraph_text_coverage_f1") or {}).get("f1"),
                "section_attachment_f1": (metrics.get("section_attachment_f1") or {}).get("f1"),
                "section_attachment_body_no_float_f1": (metrics.get("section_attachment_body_no_float_f1") or {}).get("f1"),
                "section_attachment_oracle_heading_flow_f1": (
                    metrics.get("section_attachment_oracle_heading_flow_f1") or {}
                ).get("f1"),
                "reference_section_completeness": (metrics.get("reference_section_completeness") or {}).get("score"),
                "float_caption_attachment_accuracy": (metrics.get("float_caption_attachment_accuracy") or {}).get("score"),
                "generated_structure_validity": (metrics.get("generated_structure_validity") or {}).get("score"),
            }
        )

    original_pdf = Path(str(row.get("paired_original_pdf") or row.get("original_pdf") or ""))
    generated_pdf = Path(str(row.get("paired_generated_pdf") or row.get("generated_pdf") or ""))
    if original_pdf.exists() and generated_pdf.exists():
        layout = compare_pdf_layouts(
            original_pdf,
            generated_pdf,
            dpi=args.layout_dpi,
            max_pages=args.layout_max_pages,
        )
        layout_path = doc_dir / "layout_metrics.json"
        write_json(layout_path, layout)
        result["layout_metrics"] = str(layout_path)
        result["layout_similarity"] = layout.get("layout_similarity")
        result["layout_page_count_score"] = layout.get("page_count_score")

    row.update(result)
    write_json(doc_dir / "e2e_record.json", row)
    return result


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "documents": len(rows),
        "compiled": sum(1 for row in rows if row.get("generated_pdf_exists")),
        "compile_success_rate": safe_div(sum(1 for row in rows if row.get("generated_pdf_exists")), len(rows)),
        "macro_structure_score": mean_value(row.get("macro_structure_score") for row in rows),
        "heading_tree_accuracy": mean_value(row.get("heading_tree_accuracy") for row in rows),
        "reading_order_accuracy": mean_value(row.get("reading_order_accuracy") for row in rows),
        "paragraph_boundary_f1": mean_value(row.get("paragraph_boundary_f1") for row in rows),
        "paragraph_text_coverage_f1": mean_value(row.get("paragraph_text_coverage_f1") for row in rows),
        "section_attachment_f1": mean_value(row.get("section_attachment_f1") for row in rows),
        "section_attachment_body_no_float_f1": mean_value(row.get("section_attachment_body_no_float_f1") for row in rows),
        "section_attachment_oracle_heading_flow_f1": mean_value(
            row.get("section_attachment_oracle_heading_flow_f1") for row in rows
        ),
        "reference_section_completeness": mean_value(row.get("reference_section_completeness") for row in rows),
        "float_caption_attachment_accuracy": mean_value(row.get("float_caption_attachment_accuracy") for row in rows),
        "generated_structure_validity": mean_value(row.get("generated_structure_validity") for row in rows),
        "layout_similarity": mean_value(row.get("layout_similarity") for row in rows),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "document_id",
        "generated_pdf_exists",
        "macro_structure_score",
        "heading_tree_accuracy",
        "reading_order_accuracy",
        "paragraph_boundary_f1",
        "paragraph_text_coverage_f1",
        "section_attachment_f1",
        "section_attachment_body_no_float_f1",
        "section_attachment_oracle_heading_flow_f1",
        "reference_section_completeness",
        "float_caption_attachment_accuracy",
        "generated_structure_validity",
        "layout_similarity",
        "doc_dir",
        "error",
    ]
    lines = [",".join(fields)]
    for row in rows:
        values = [csv_cell(row.get(field)) for field in fields]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_status(index: int, total: int, row: dict[str, Any]) -> None:
    score = row.get("macro_structure_score")
    layout = row.get("layout_similarity")
    print(
        f"[{index:02d}/{total:02d}] {row.get('document_id')} "
        f"pdf={bool(row.get('generated_pdf_exists'))} "
        f"structure={format_float(score)} layout={format_float(layout)} "
        f"-> {row.get('doc_dir')}",
        flush=True,
    )


def existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def mean_value(values: Any) -> float | None:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace('"', '""')
    if any(ch in text for ch in [",", "\n", '"']):
        return f'"{text}"'
    return text


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))[:120] or "document"


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
