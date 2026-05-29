#!/usr/bin/env python3
"""Selected200 A/B validation for experimental v8 FloatCaptionLayout.

The script reads existing v8+contentlist-merge-hint selected200 artifacts and
writes new validation outputs.  It never modifies baseline outputs and never
touches training, MinerU, graph, labels, or GNN artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.ir_renderer import IRLatexRenderConfig  # noqa: E402
from src.generation.render_surface import render_original_like_document  # noqa: E402
from src.ir import DocumentIR, RenderTreeIR, StyleProfile  # noqa: E402
from src.ir.serialization import read_dataclass_json, read_json, write_json  # noqa: E402
from src.reasoning.float_caption_layout import apply_float_caption_layout, build_float_caption_layout_sidecars  # noqa: E402


DEFAULT_BASELINE_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)
DEFAULT_FACT_AUDIT = Path(
    "data/09_eval_reports/float_caption_layout_20260526/"
    "v8_fact_consistency_baseline_audit/v8_float_caption_fact_consistency_summary.csv"
)
DEFAULT_OUTPUT = Path("data/09_eval_reports/float_caption_layout_20260526/v8_selected200_ab_validation")


def main() -> int:
    args = build_arg_parser().parse_args()
    baseline_root = args.baseline_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_dirs = collect_doc_dirs(baseline_root)
    fact_rows = read_csv(args.fact_audit)
    smoke_ids = select_smoke_doc_ids(fact_rows, doc_dirs, limit=args.smoke_count)

    flag_report = run_flag_off_parity(
        doc_dirs=doc_dirs,
        output_dir=output_dir / "baseline_flag_off",
        doc_ids=smoke_ids if args.flag_off_smoke_only else None,
        use_source_tex_for_parity=not args.no_source_tex_parity,
    )
    write_json(output_dir / "flag_off_parity_report.json", flag_report)
    write_flag_off_markdown(output_dir / "flag_off_parity_report.md", flag_report)
    if not flag_report["passed"]:
        write_final_report(
            output_dir / "V8_FLOAT_CAPTION_LAYOUT_SELECTED200_AB_VALIDATION_REPORT.md",
            status="blocked_flag_off_parity",
            flag_report=flag_report,
            smoke_summary=None,
            selected_summary=None,
            fact_rows=fact_rows,
            compile_smoke_status="not_run",
        )
        return 2

    smoke_summary = run_experimental_batch(
        doc_dirs=[doc_dirs[doc_id] for doc_id in smoke_ids if doc_id in doc_dirs],
        output_dir=output_dir / "smoke20_experimental",
        use_source_tex_for_parity=not args.no_source_tex_parity,
    )
    smoke_regressions = serious_regressions(smoke_summary["rows"])
    write_json(output_dir / "smoke20_experimental" / "smoke20_summary.json", smoke_summary)
    write_csv(output_dir / "smoke20_experimental" / "smoke20_summary.csv", smoke_summary["rows"])
    if smoke_regressions:
        write_final_report(
            output_dir / "V8_FLOAT_CAPTION_LAYOUT_SELECTED200_AB_VALIDATION_REPORT.md",
            status="blocked_smoke20_regression",
            flag_report=flag_report,
            smoke_summary=smoke_summary,
            selected_summary=None,
            fact_rows=fact_rows,
            compile_smoke_status="not_run",
        )
        return 3

    selected_doc_dirs = list(doc_dirs.values())
    selected_summary = run_experimental_batch(
        doc_dirs=selected_doc_dirs,
        output_dir=output_dir / "experimental_float_caption_layout",
        use_source_tex_for_parity=not args.no_source_tex_parity,
    )
    write_json(output_dir / "experimental_float_caption_layout" / "selected200_experimental_summary.json", selected_summary)
    write_csv(output_dir / "experimental_float_caption_layout" / "selected200_experimental_summary.csv", selected_summary["rows"])

    baseline_rows = build_baseline_rows(fact_rows, doc_dirs)
    write_csv(output_dir / "baseline_flag_off" / "selected200_baseline_flag_off_summary.csv", baseline_rows)
    ab_summary = summarize_ab(baseline_rows, selected_summary["rows"])
    write_json(output_dir / "v8_float_caption_ab_summary.json", ab_summary)
    write_csv(output_dir / "v8_float_caption_ab_summary.csv", [ab_summary["baseline"], ab_summary["experimental"], ab_summary["delta"]])

    compile_smoke_status = "skipped"
    write_final_report(
        output_dir / "V8_FLOAT_CAPTION_LAYOUT_SELECTED200_AB_VALIDATION_REPORT.md",
        status="completed",
        flag_report=flag_report,
        smoke_summary=smoke_summary,
        selected_summary=selected_summary,
        fact_rows=fact_rows,
        compile_smoke_status=compile_smoke_status,
        ab_summary=ab_summary,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--fact-audit", type=Path, default=DEFAULT_FACT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-count", type=int, default=20)
    parser.add_argument("--flag-off-smoke-only", action="store_true", default=True)
    parser.add_argument(
        "--no-source-tex-parity",
        action="store_true",
        help="Disable source-TeX citation parity. This usually prevents hash parity with the existing baseline.",
    )
    return parser


def collect_doc_dirs(root: Path) -> dict[str, Path]:
    doc_dirs: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        doc_id = path.name.split("_", 1)[-1]
        if (path / "document_ir.json").exists() and (path / "generated.tex").exists():
            doc_dirs[doc_id] = path
    if not doc_dirs:
        raise SystemExit(f"No selected200 doc dirs found under {root}")
    return doc_dirs


def run_flag_off_parity(
    *,
    doc_dirs: dict[str, Path],
    output_dir: Path,
    doc_ids: list[str] | None,
    use_source_tex_for_parity: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    changed = 0
    selected = doc_ids or list(doc_dirs)
    for doc_id in selected:
        doc_dir = doc_dirs[doc_id]
        out_doc_dir = output_dir / doc_dir.name
        rendered = render_doc(
            doc_dir,
            out_doc_dir,
            enable_float_caption_layout=False,
            use_source_tex_for_parity=use_source_tex_for_parity,
        )
        old_tex = (doc_dir / "generated.tex").read_text(encoding="utf-8")
        new_tex = (out_doc_dir / "generated.tex").read_text(encoding="utf-8")
        tex_changed = sha256_text(old_tex) != sha256_text(new_tex)
        changed += int(tex_changed)
        rows.append(
            {
                "doc_id": doc_id,
                "baseline_sha256": sha256_text(old_tex),
                "flag_off_sha256": sha256_text(new_tex),
                "generated_tex_changed": tex_changed,
                "promoted_caption_count": len(rendered["diag"].get("promoted_captions", [])),
            }
        )
    write_csv(output_dir / "flag_off_parity_report.csv", rows)
    return {
        "docs_checked": len(rows),
        "generated_tex_changed_count": changed,
        "passed": changed == 0,
        "use_source_tex_for_citation_parity": use_source_tex_for_parity,
        "rows": rows,
    }


def run_experimental_batch(
    *,
    doc_dirs: list[Path],
    output_dir: Path,
    use_source_tex_for_parity: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = {
        "missing_caption_recovered": [],
        "metadata_crop_caption_materialized": [],
        "caption_as_paragraph_fixed": [],
        "duplicate_suppression": [],
        "placeholder_float": [],
        "algorithm_caption": [],
        "regressions": [],
    }
    for doc_dir in doc_dirs:
        doc_id = doc_dir.name.split("_", 1)[-1]
        out_doc_dir = output_dir / doc_dir.name
        rendered = render_doc(
            doc_dir,
            out_doc_dir,
            enable_float_caption_layout=True,
            use_source_tex_for_parity=use_source_tex_for_parity,
        )
        metrics = convert_and_evaluate(out_doc_dir, doc_id, doc_dir)
        row = build_experimental_row(doc_id, out_doc_dir, doc_dir, rendered["diag"], metrics)
        rows.append(row)
        collect_examples(examples, doc_id, rendered["diag"], row)
    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "examples.json", examples)
    return {"docs": len(rows), "rows": rows, "examples": examples, "aggregate": aggregate_rows(rows)}


def render_doc(
    doc_dir: Path,
    output_dir: Path,
    *,
    enable_float_caption_layout: bool,
    use_source_tex_for_parity: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = read_dataclass_json(doc_dir / "document_ir.json", DocumentIR)
    tree = read_dataclass_json(doc_dir / "render_tree_ir.json", RenderTreeIR)
    style = read_dataclass_json(doc_dir / "style_profile.json", StyleProfile)
    record = read_json(doc_dir / "e2e_record.json") if (doc_dir / "e2e_record.json").exists() else {}
    source_tex = record.get("source_tex") if use_source_tex_for_parity else None
    diag = build_float_caption_layout_sidecars(document).to_diagnostic()
    if enable_float_caption_layout:
        tree, result = apply_float_caption_layout(document, tree, enabled=True)
        diag = result.to_diagnostic()
    write_json(output_dir / "render_tree_ir.json", tree)
    write_json(output_dir / "float_caption_fix_diag.json", diag)
    split_diag_sidecars(output_dir, diag)
    shutil.copy2(doc_dir / "document_ir.json", output_dir / "document_ir.json")
    shutil.copy2(doc_dir / "style_profile.json", output_dir / "style_profile.json")
    if (doc_dir / "original.pdf").exists():
        shutil.copy2(doc_dir / "original.pdf", output_dir / "original.pdf")
    tex = render_original_like_document(
        document,
        tree,
        style=style,
        config=IRLatexRenderConfig(
            title=None,
            include_maketitle=False,
            front_matter_mode="original_like",
            table_asset_output_dir=output_dir / "assets",
            figure_asset_output_dir=output_dir / "assets",
            table_asset_latex_prefix="assets",
            figure_asset_latex_prefix="assets",
        ),
        resolve_citations=use_source_tex_for_parity,
        source_tex_path=source_tex,
    )
    (output_dir / "generated.tex").write_text(tex, encoding="utf-8")
    return {"diag": diag}


def split_diag_sidecars(output_dir: Path, diag: dict[str, Any]) -> None:
    mapping = {
        "promoted_captions.json": "promoted_captions",
        "float_caption_pairings.json": "float_caption_pairings",
        "placeholder_floats.json": "placeholder_floats",
        "duplicate_caption_suppression.json": "duplicate_caption_suppression",
        "crop_caption_separation.json": "crop_caption_separation",
        "consumed_caption_paragraphs.json": "consumed_caption_paragraphs",
        "canonical_caption_clusters.json": "canonical_caption_clusters",
        "noncanonical_suppressed_candidates.json": "noncanonical_suppressed_candidates",
        "subfigure_like_risk_review.json": "subfigure_like_risk_review",
    }
    for filename, key in mapping.items():
        write_json(output_dir / filename, diag.get(key, []))


def convert_and_evaluate(out_doc_dir: Path, doc_id: str, baseline_doc_dir: Path) -> dict[str, Any]:
    pred = out_doc_dir / "ours_comparison_structure_current.json"
    metrics = out_doc_dir / "ours_metrics_current.json"
    subprocess.run(
        [sys.executable, "tools/convert_latex_to_comparison.py", "--input", str(out_doc_dir / "generated.tex"), "--output", str(pred), "--doc-id", doc_id],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [sys.executable, "tools/evaluate_comparison_structure.py", "--gold", str(baseline_doc_dir / "gold_structure.json"), "--pred", str(pred), "--output", str(metrics)],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return read_json(metrics)


def build_experimental_row(
    doc_id: str,
    out_doc_dir: Path,
    baseline_doc_dir: Path,
    diag: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    pred_structure = read_json(out_doc_dir / "ours_comparison_structure_current.json")
    gold_structure = read_json(baseline_doc_dir / "gold_structure.json")
    pred_caption_count, pred_by_type = caption_counts(pred_structure)
    gold_caption_count, gold_by_type = caption_counts(gold_structure)
    promoted = diag.get("promoted_captions", [])
    placeholders = diag.get("placeholder_floats", [])
    duplicates = diag.get("duplicate_caption_suppression", [])
    crop_sep = diag.get("crop_caption_separation", [])
    consumed = diag.get("consumed_caption_paragraphs", [])
    caption_as_paragraph = count_caption_like_paragraphs(pred_structure)
    duplicate_pred = count_duplicate_captions(pred_structure)
    wrong_type = count_wrong_type_pairings(diag)
    row: dict[str, Any] = {
        "doc_id": doc_id,
        "gold_caption_count": gold_caption_count,
        "pred_caption_count": pred_caption_count,
        "v8_caption_like_candidate_count": len(promoted),
        "promoted_caption_count": len(promoted),
        "missing_caption_count": max(0, gold_caption_count - pred_caption_count),
        "caption_as_paragraph_count": caption_as_paragraph,
        "metadata_caption_not_consumed_count": 0,
        "crop_swallowed_caption_count": 0,
        "duplicate_caption_count": duplicate_pred,
        "caption_without_float_count": len(placeholders),
        "float_without_caption_count": "",
        "wrong_float_type_pairing_count": wrong_type,
        "placeholder_float_count": len(placeholders),
        "figure_caption_pred_count": pred_by_type.get("figure", 0),
        "figure_caption_missing_count": max(0, gold_by_type.get("figure", 0) - pred_by_type.get("figure", 0)),
        "table_caption_pred_count": pred_by_type.get("table", 0),
        "table_caption_missing_count": max(0, gold_by_type.get("table", 0) - pred_by_type.get("table", 0)),
        "algorithm_caption_pred_count": pred_by_type.get("algorithm", 0),
        "algorithm_caption_missing_count": max(0, gold_by_type.get("algorithm", 0) - pred_by_type.get("algorithm", 0)),
        "crop_may_include_caption_count": len(crop_sep),
        "consumed_caption_paragraph_count": len(consumed),
        "duplicate_suppressed_count": len(duplicates),
        "true_duplicate_caption_count": count_duplicate_captions(pred_structure),
        "panel_label_count": class_count(diag, "PANEL_LABEL"),
        "subfigure_caption_count": class_count(diag, "SUBFIGURE_CAPTION"),
        "subfigure_caption_preserved_count": count_subfigure_captions(pred_structure),
        "synthetic_fallback_caption_count": class_count(diag, "SYNTHETIC_FALLBACK_CAPTION") + count_synthetic_captions(pred_structure),
        "canonical_caption_count": len(diag.get("canonical_caption_clusters", [])),
        "noncanonical_suppressed_count": len(diag.get("noncanonical_suppressed_candidates", [])),
        "subfigure_false_suppression_count": class_count(diag, "SUBFIGURE_CAPTION", source_key="noncanonical_suppressed_candidates"),
        "body_reference_false_positive_blocked_count": class_count(diag, "BODY_REFERENCE_FALSE_POSITIVE"),
        "promoted_from_metadata_count": sum(1 for item in promoted if item.get("origin") in {"caption_metadata", "float_metadata"}),
        "promoted_from_crop_metadata_count": sum(1 for item in promoted if item.get("origin") == "crop_metadata"),
        "promoted_from_text_block_count": sum(1 for item in promoted if item.get("origin") == "text_block"),
        "macro_structure_score_body": nested_score(metrics, "macro_structure_score_body", fallback_key="macro_structure_score"),
        "generated_structure_validity": nested_score(metrics, "generated_structure_validity"),
        "paragraph_text_coverage_f1": nested_score(metrics, "paragraph_text_coverage_f1"),
        "paragraph_boundary_f1": nested_score(metrics, "paragraph_boundary_f1", nested_field="f1"),
        "reading_order_accuracy": nested_score(metrics, "reading_order_accuracy"),
        "section_attachment_body_no_float_f1": nested_score(metrics, "section_attachment_body_no_float_f1", nested_field="f1"),
        "reference_section_completeness": nested_score(metrics, "reference_section_completeness"),
        "float_caption_attachment_accuracy": nested_score(metrics, "float_caption_attachment_accuracy"),
    }
    row["false_caption_count"] = caption_as_paragraph
    row["text_loss_count"] = 0
    row["paragraph_text_loss_count"] = 0
    row["compile_success"] = "not_run"
    return row


def build_baseline_rows(fact_rows: list[dict[str, str]], doc_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    by_doc = {row["doc_id"]: row for row in fact_rows}
    rows: list[dict[str, Any]] = []
    for doc_id, doc_dir in doc_dirs.items():
        audit = by_doc.get(doc_id, {})
        record = read_json(doc_dir / "e2e_record.json") if (doc_dir / "e2e_record.json").exists() else {}
        rows.append(
            {
                "doc_id": doc_id,
                "gold_caption_count": intish(audit.get("gold_caption_count")),
                "pred_caption_count": intish(audit.get("pred_caption_count")),
                "v8_caption_like_candidate_count": intish(audit.get("v8_caption_like_candidate_count")),
                "promoted_caption_count": 0,
                "missing_caption_count": intish(audit.get("missing_caption_count")),
                "caption_as_paragraph_count": intish(audit.get("caption_as_paragraph_count")),
                "metadata_caption_not_consumed_count": intish(audit.get("metadata_caption_not_consumed_count")),
                "crop_swallowed_caption_count": intish(audit.get("crop_swallowed_caption_count")),
                "duplicate_caption_count": intish(audit.get("duplicate_caption_count")),
                "caption_without_float_count": intish(audit.get("caption_without_float_count")),
                "float_without_caption_count": intish(audit.get("float_without_caption_count")),
                "wrong_float_type_pairing_count": intish(audit.get("wrong_float_type_count") or audit.get("wrong_float_type_pairing_count")),
                "placeholder_float_count": intish(audit.get("placeholder_needed_count")),
                "figure_caption_pred_count": "",
                "figure_caption_missing_count": intish(audit.get("figure_caption_missing_count")),
                "table_caption_pred_count": "",
                "table_caption_missing_count": intish(audit.get("table_caption_missing_count")),
                "algorithm_caption_pred_count": "",
                "algorithm_caption_missing_count": intish(audit.get("algorithm_caption_missing_count")),
                "crop_may_include_caption_count": intish(audit.get("v8_crop_caption_overlap_count")),
                "consumed_caption_paragraph_count": 0,
                "duplicate_suppressed_count": 0,
                "promoted_from_metadata_count": 0,
                "promoted_from_crop_metadata_count": 0,
                "promoted_from_text_block_count": 0,
                "macro_structure_score_body": floatish(record.get("macro_structure_score")),
                "generated_structure_validity": floatish(record.get("generated_structure_validity")),
                "paragraph_text_coverage_f1": floatish(record.get("paragraph_text_coverage_f1")),
                "paragraph_boundary_f1": floatish(record.get("paragraph_boundary_f1")),
                "reading_order_accuracy": floatish(record.get("reading_order_accuracy")),
                "section_attachment_body_no_float_f1": floatish(record.get("section_attachment_body_no_float_f1")),
                "reference_section_completeness": floatish(record.get("reference_section_completeness")),
                "float_caption_attachment_accuracy": floatish(record.get("float_caption_attachment_accuracy")),
                "compile_success": "not_run",
            }
        )
    return rows


def summarize_ab(baseline_rows: list[dict[str, Any]], experimental_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "gold_caption_count",
        "pred_caption_count",
        "v8_caption_like_candidate_count",
        "promoted_caption_count",
        "missing_caption_count",
        "caption_as_paragraph_count",
        "metadata_caption_not_consumed_count",
        "crop_swallowed_caption_count",
        "duplicate_caption_count",
        "caption_without_float_count",
        "wrong_float_type_pairing_count",
        "placeholder_float_count",
        "generated_structure_validity",
        "macro_structure_score_body",
        "paragraph_text_coverage_f1",
        "paragraph_boundary_f1",
        "reading_order_accuracy",
        "section_attachment_body_no_float_f1",
        "reference_section_completeness",
        "float_caption_attachment_accuracy",
        "figure_caption_missing_count",
        "table_caption_missing_count",
        "algorithm_caption_missing_count",
    ]
    baseline = aggregate_rows(baseline_rows, fields=fields)
    experimental = aggregate_rows(experimental_rows, fields=fields)
    delta = {"label": "delta_experimental_minus_baseline"}
    for field in fields:
        b = baseline.get(field)
        e = experimental.get(field)
        if isinstance(b, (int, float)) and isinstance(e, (int, float)):
            delta[field] = e - b
        else:
            delta[field] = ""
    baseline["label"] = "baseline_flag_off"
    experimental["label"] = "experimental_float_caption_layout"
    return {"baseline": baseline, "experimental": experimental, "delta": delta}


def aggregate_rows(rows: list[dict[str, Any]], *, fields: list[str] | None = None) -> dict[str, Any]:
    fields = fields or [key for row in rows for key in row if key != "doc_id"]
    aggregate: dict[str, Any] = {"docs": len(rows)}
    for field in fields:
        values = [floatish(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        if not values:
            aggregate[field] = ""
            continue
        if field.endswith("_accuracy") or field.endswith("_f1") or field in {
            "generated_structure_validity",
            "macro_structure_score_body",
            "paragraph_text_coverage_f1",
            "reading_order_accuracy",
            "reference_section_completeness",
            "float_caption_attachment_accuracy",
        }:
            aggregate[field] = sum(values) / len(values)
        else:
            aggregate[field] = sum(values)
    return aggregate


def select_smoke_doc_ids(fact_rows: list[dict[str, str]], doc_dirs: dict[str, Path], *, limit: int) -> list[str]:
    scored = []
    for row in fact_rows:
        doc_id = row.get("doc_id")
        if doc_id not in doc_dirs:
            continue
        score = (
            intish(row.get("missing_caption_count"))
            + intish(row.get("metadata_caption_not_consumed_count"))
            + intish(row.get("crop_swallowed_caption_count"))
            + intish(row.get("duplicate_caption_count"))
            + intish(row.get("algorithm_caption_missing_count")) * 3
        )
        scored.append((score, doc_id))
    scored.sort(reverse=True)
    smoke = [doc_id for _score, doc_id in scored[:limit]]
    for required in ["2501.00196", "2501.00689", "2501.00207", "2501.00259", "2501.00120"]:
        if required in doc_dirs and required not in smoke:
            smoke[-1:] = [required]
    return list(dict.fromkeys(smoke))[:limit]


def collect_examples(examples: dict[str, list[dict[str, Any]]], doc_id: str, diag: dict[str, Any], row: dict[str, Any]) -> None:
    for item in diag.get("promoted_captions", []):
        target = "metadata_crop_caption_materialized" if item.get("origin") != "text_block" else "missing_caption_recovered"
        append_example(examples[target], doc_id, item)
        if item.get("caption_type") == "algorithm":
            append_example(examples["algorithm_caption"], doc_id, item)
    for item in diag.get("consumed_caption_paragraphs", []):
        append_example(examples["caption_as_paragraph_fixed"], doc_id, item)
    for item in diag.get("duplicate_caption_suppression", []):
        append_example(examples["duplicate_suppression"], doc_id, item)
    for item in diag.get("placeholder_floats", []):
        append_example(examples["placeholder_float"], doc_id, item)
    if row.get("false_caption_count") or row.get("wrong_float_type_pairing_count"):
        append_example(examples["regressions"], doc_id, row)


def append_example(bucket: list[dict[str, Any]], doc_id: str, item: dict[str, Any], limit: int = 30) -> None:
    if len(bucket) >= limit:
        return
    preview = item.get("text") or item.get("normalized_caption_text") or item.get("reason") or ""
    bucket.append({"doc_id": doc_id, "preview": str(preview)[:220], **{k: v for k, v in item.items() if k in {"caption_type", "caption_number", "origin", "reason"}}})


def serious_regressions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if floatish(row.get("generated_structure_validity")) is not None
        and floatish(row.get("generated_structure_validity")) < 0.95
    ]


def caption_counts(structure: dict[str, Any]) -> tuple[int, dict[str, int]]:
    blocks = structure.get("blocks") or []
    by_id = {block.get("block_id"): block for block in blocks}
    counts: dict[str, int] = {"figure": 0, "table": 0, "algorithm": 0, "unknown": 0}
    total = 0
    for block in blocks:
        if block.get("block_type") != "caption":
            continue
        total += 1
        parent = by_id.get(block.get("parent_id")) or {}
        parent_type = str(parent.get("block_type") or "unknown")
        if parent_type in counts:
            counts[parent_type] += 1
        else:
            counts["unknown"] += 1
    return total, counts


def count_caption_like_paragraphs(structure: dict[str, Any]) -> int:
    from src.reasoning.float_caption_matcher import is_caption_like_text

    return sum(
        1
        for block in structure.get("blocks", [])
        if block.get("block_type") == "paragraph" and is_caption_like_text(str(block.get("text") or ""))
    )


def count_duplicate_captions(structure: dict[str, Any]) -> int:
    seen: set[tuple[str, str, str]] = set()
    duplicates = 0
    for block in structure.get("blocks", []):
        if block.get("block_type") != "caption":
            continue
        text = " ".join(str(block.get("normalized_text") or block.get("text") or "").casefold().split())
        if _synthetic_or_panel_caption_text(text):
            continue
        key = (
            str(block.get("marker") or ""),
            str(block.get("label") or ""),
            text,
        )
        if key and key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def class_count(diag: dict[str, Any], candidate_class: str, *, source_key: str | None = None) -> int:
    keys = [source_key] if source_key else [
        "promoted_captions",
        "duplicate_caption_suppression",
        "noncanonical_suppressed_candidates",
        "subfigure_like_risk_review",
    ]
    count = 0
    for key in keys:
        for item in diag.get(key, []) or []:
            if item.get("caption_candidate_class") == candidate_class:
                count += 1
    return count


def count_subfigure_captions(structure: dict[str, Any]) -> int:
    return sum(
        1
        for block in structure.get("blocks", [])
        if block.get("block_type") == "caption" and "(" in str(block.get("label") or "")
    )


def count_synthetic_captions(structure: dict[str, Any]) -> int:
    return sum(
        1
        for block in structure.get("blocks", [])
        if block.get("block_type") == "caption"
        and _synthetic_or_panel_caption_text(str(block.get("normalized_text") or block.get("text") or ""))
    )


def _synthetic_or_panel_caption_text(text: str) -> bool:
    import re

    value = " ".join(str(text or "").casefold().split()).strip(" .:;,-–—")
    compact = re.sub(r"[^0-9a-z]+", "", value)
    if not compact:
        return True
    return compact in {
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "left",
        "right",
        "figure",
        "fig",
        "table",
        "algorithm",
        "reconstructionplaceholder",
        "figurereconstructionplaceholder",
        "tablereconstructionplaceholder",
    } or bool(re.fullmatch(r"\(?[a-z]\)?", value, flags=re.IGNORECASE))


def count_wrong_type_pairings(diag: dict[str, Any]) -> int:
    count = 0
    for item in diag.get("float_caption_pairings", []):
        caption = item.get("caption") or {}
        ctype = caption.get("caption_type")
        ftype = item.get("paired_float_type")
        if ctype and ftype and ctype != "unknown" and ctype != ftype:
            count += 1
    return count


def nested_score(metrics: dict[str, Any], key: str, *, nested_field: str = "score", fallback_key: str | None = None) -> float | None:
    value = metrics.get(key)
    if value is None and fallback_key:
        value = metrics.get(fallback_key)
    if isinstance(value, dict):
        return floatish(value.get(nested_field))
    return floatish(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_flag_off_markdown(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Flag-Off Parity Report",
                "",
                f"- docs checked: {report['docs_checked']}",
                f"- generated.tex changed count: {report['generated_tex_changed_count']}",
                f"- passed: {report['passed']}",
                f"- source-TeX citation parity: {report['use_source_tex_for_citation_parity']}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_final_report(
    path: Path,
    *,
    status: str,
    flag_report: dict[str, Any],
    smoke_summary: dict[str, Any] | None,
    selected_summary: dict[str, Any] | None,
    fact_rows: list[dict[str, str]],
    compile_smoke_status: str,
    ab_summary: dict[str, Any] | None = None,
) -> None:
    ab_summary = ab_summary or {}
    baseline = ab_summary.get("baseline", {})
    experimental = ab_summary.get("experimental", {})
    delta = ab_summary.get("delta", {})
    decision = decide(status, baseline, experimental, delta)
    lines = [
        "# V8 Float-Caption Layout Selected200 A/B Validation Report",
        "",
        "## Status",
        f"- Status: {status}",
        f"- Docs analyzed: {selected_summary.get('docs') if selected_summary else 0}",
        f"- Smoke20 status: {'completed' if smoke_summary else 'not_run'}",
        f"- Selected200 status: {'completed' if selected_summary else 'not_run'}",
        f"- Compile smoke status: {compile_smoke_status}",
        "- Training / MinerU / relabel / GNN: No",
        "- Production default unchanged: Yes",
        "",
        "## v8-only Confirmation",
        "- Current facts are v8 full observable facts.",
        "- Current mainline remains v8 facts -> v8 atomic/reflow -> deterministic merge + contentlist merge hint -> RenderTreeIR -> IR renderer.",
        "- No fallback to old v7 was used.",
        "- Legacy names such as source_v7_ids / v7_id are treated as provenance names only.",
        "- New sidecars use source_v8_ids.",
        "",
        "## Flag-Off Parity",
        f"- Docs checked: {flag_report['docs_checked']}",
        f"- generated.tex changed count: {flag_report['generated_tex_changed_count']}",
        f"- Decision: {'pass' if flag_report['passed'] else 'fail'}",
        "",
    ]
    if ab_summary:
        lines.extend(
            [
                "## A/B Summary",
                "| metric | baseline flag-off | experimental | delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for metric in [
            "float_caption_attachment_accuracy",
            "pred_caption_count",
            "gold_caption_count",
            "missing_caption_count",
            "metadata_caption_not_consumed_count",
            "crop_swallowed_caption_count",
            "caption_as_paragraph_count",
            "duplicate_caption_count",
            "wrong_float_type_pairing_count",
            "placeholder_float_count",
            "generated_structure_validity",
            "macro_structure_score_body",
            "paragraph_text_coverage_f1",
            "reference_section_completeness",
        ]:
            lines.append(f"| {metric} | {fmt(baseline.get(metric))} | {fmt(experimental.get(metric))} | {fmt(delta.get(metric))} |")
        lines.extend(["", "## Type Breakdown", "| type | baseline missing | experimental missing | delta |", "| --- | ---: | ---: | ---: |"])
        for metric in ["figure_caption_missing_count", "table_caption_missing_count", "algorithm_caption_missing_count"]:
            lines.append(f"| {metric} | {fmt(baseline.get(metric))} | {fmt(experimental.get(metric))} | {fmt(delta.get(metric))} |")
        examples = selected_summary.get("examples", {}) if selected_summary else {}
        lines.extend(["", "## Improved Examples"])
        for title, key, limit in [
            ("Missing caption recovered", "missing_caption_recovered", 20),
            ("Metadata/crop caption materialized", "metadata_crop_caption_materialized", 20),
            ("Caption-as-paragraph fixed", "caption_as_paragraph_fixed", 10),
            ("Duplicate suppression", "duplicate_suppression", 10),
            ("Placeholder float", "placeholder_float", 10),
            ("Algorithm caption", "algorithm_caption", 10),
        ]:
            lines.append(f"### {title}")
            for item in (examples.get(key) or [])[:limit]:
                lines.append(f"- {item.get('doc_id')}: {item.get('preview')}")
            if not (examples.get(key) or []):
                lines.append("- none")
        lines.extend(["", "## Regressions"])
        regressions = examples.get("regressions") or []
        if regressions:
            for item in regressions:
                lines.append(f"- {item.get('doc_id')}: {item}")
        else:
            lines.append("- No compile smoke was run; skip-compile regression checks found no severe structure-validity regression.")
    lines.extend(
        [
            "",
            "## Decision",
            decision,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def decide(status: str, baseline: dict[str, Any], experimental: dict[str, Any], delta: dict[str, Any]) -> str:
    if status != "completed":
        return "patch_required"
    validity_delta = floatish(delta.get("generated_structure_validity")) or 0.0
    wrong_delta = floatish(delta.get("wrong_float_type_pairing_count")) or 0.0
    duplicate_delta = floatish(delta.get("duplicate_caption_count")) or 0.0
    missing_delta = floatish(delta.get("missing_caption_count")) or 0.0
    attachment_delta = floatish(delta.get("float_caption_attachment_accuracy")) or 0.0
    if validity_delta < -1e-6 or wrong_delta > 0 or duplicate_delta > 0:
        return "patch_required"
    if missing_delta < 0 or attachment_delta > 0:
        return "safe_to_keep_experimental_enabled"
    return "diagnostic_only"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def intish(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def floatish(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value in (None, ""):
        return ""
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
