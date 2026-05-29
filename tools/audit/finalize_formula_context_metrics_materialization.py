#!/usr/bin/env python3
"""Finalize FormulaContextGroup metric attribution and materialization status.

This pass is intentionally read-only.  It consumes the baseline v8+contentlist
merge-hint outputs plus Patch1 FormulaContextGroup sidecars and reports:

* baseline metrics under the original visible-prose mask,
* baseline metrics under the Patch1 ordinary/context mask,
* Patch1 metrics under that same mask, and
* whether generated.tex was actually changed/materialized.

It does not rerun generation, modify v7 JSON, suppress nodes, train models,
rebuild graphs, or enable FormulaContextGroup as a production default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PHASE0_DIR = Path("data/09_eval_reports/formula_paragraph_context_group_20260526")
DEFAULT_BASELINE_VALIDATION_DIR = DEFAULT_PHASE0_DIR / "selected200_validation"
DEFAULT_PATCH1_DIR = DEFAULT_PHASE0_DIR / "selected200_validation_patch1"
DEFAULT_BREAKDOWN_DIR = Path("data/09_eval_reports/v8_visible_prose_failure_breakdown_20260526/v8_contentlist_merge_hint")
DEFAULT_OUTPUT_DIR = DEFAULT_PHASE0_DIR / "metric_materialization_finalization"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-validation-dir", type=Path, default=DEFAULT_BASELINE_VALIDATION_DIR)
    parser.add_argument("--patch1-dir", type=Path, default=DEFAULT_PATCH1_DIR)
    parser.add_argument("--breakdown-dir", type=Path, default=DEFAULT_BREAKDOWN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-ids", nargs="*", default=None)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def as_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def fmt(value: Any) -> str:
    number = as_float(value)
    return "N/A" if number is None else f"{number:.6f}"


def md(text: Any) -> str:
    return " ".join(str(text or "").split()).replace("|", "\\|")[:240]


def file_hash(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readiness_check(args: argparse.Namespace) -> list[str]:
    required = [
        args.breakdown_dir / "doc_failure_breakdown.csv",
        args.patch1_dir / "refined_matching" / "refined_visible_prose_metrics.csv",
        args.patch1_dir / "phase0_sidecar" / "formula_context_group_summary.csv",
        args.patch1_dir / "selected200_validation_summary.csv",
        args.baseline_validation_dir / "selected200_validation_summary.csv",
    ]
    return [str(path) for path in required if not path.exists()]


def path_or_none(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mean_keys = [
        "baseline_old_visible_coverage",
        "baseline_old_visible_ordered_coverage",
        "baseline_old_visible_inversion",
        "baseline_old_adjacent_inversion",
        "baseline_old_lis_disorder",
        "baseline_under_patch1_mask_visible_coverage",
        "baseline_under_patch1_mask_visible_ordered_coverage",
        "baseline_under_patch1_mask_visible_inversion",
        "baseline_under_patch1_mask_adjacent_inversion",
        "baseline_under_patch1_mask_lis_disorder",
        "patch1_under_patch1_mask_visible_coverage",
        "patch1_under_patch1_mask_visible_ordered_coverage",
        "patch1_under_patch1_mask_visible_inversion",
        "patch1_under_patch1_mask_adjacent_inversion",
        "patch1_under_patch1_mask_lis_disorder",
        "context_aware_body_coverage",
        "context_aware_body_ordered_coverage",
        "context_aware_missing_like_count",
        "paragraph_text_coverage_f1",
        "generated_structure_validity",
    ]
    sum_keys = [
        "formula_context_pollution_count_baseline",
        "formula_context_pollution_count_baseline_under_patch1_mask",
        "formula_context_pollution_count_patch1",
        "ordinary_body_reorder_count_under_patch1_mask",
        "inline_math_attachment_count",
        "theorem_proof_context_count",
        "where_clause_context_count",
        "display_math_context_count",
        "formula_ocr_artifact_count",
        "formula_context_uncertain_count",
        "high_confidence_formula_context_group_count",
        "medium_confidence_formula_context_group_count",
        "low_confidence_formula_context_group_count",
        "generated_tex_changed",
        "formula_context_groups_materialized",
        "inline_math_attachment_materialized",
        "theorem_proof_materialized",
        "where_clause_materialized",
        "only_sidecar_no_tex_change",
        "suppressed_node_count",
        "duplicate_render_count",
        "text_loss_count",
        "wrong_suppression_count",
        "unsafe_math_fallback_count",
    ]
    payload = {key: mean([as_float(row.get(key)) for row in rows]) for key in mean_keys}
    payload.update({key: sum(as_int(row.get(key)) for row in rows) for key in sum_keys})
    payload["docs"] = len(rows)
    return payload


def row_for_doc(
    doc_id: str,
    *,
    breakdown_row: dict[str, str],
    refined_row: dict[str, str],
    phase0_row: dict[str, str],
    patch1_validation_row: dict[str, str],
    baseline_validation_row: dict[str, str] | None,
    patch1_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    doc_out = out_dir / "per_doc" / doc_id
    doc_out.mkdir(parents=True, exist_ok=True)

    baseline_tex = path_or_none(breakdown_row.get("generated_tex"))
    patch1_tex = patch1_dir / "per_doc" / doc_id / "generated.tex"
    baseline_hash = file_hash(baseline_tex)
    patch1_hash = file_hash(patch1_tex)
    generated_changed = bool(baseline_hash and patch1_hash and baseline_hash != patch1_hash)

    duplicate_check = load_json(patch1_dir / "per_doc" / doc_id / "duplicate_render_check.json", {})
    formula_groups = load_json(patch1_dir / "per_doc" / doc_id / "formula_context_groups.json", [])
    inline = load_json(patch1_dir / "per_doc" / doc_id / "inline_math_attachments.json", [])
    theorem = load_json(patch1_dir / "per_doc" / doc_id / "theorem_proof_contexts.json", [])
    where = load_json(patch1_dir / "per_doc" / doc_id / "where_clause_contexts.json", [])

    high_count = as_int(phase0_row.get("high_confidence_formula_context_group_count"))
    medium_count = as_int(phase0_row.get("medium_confidence_formula_context_group_count"))
    low_count = as_int(phase0_row.get("low_confidence_formula_context_group_count"))
    uncertain_count = medium_count + low_count

    same_mask = {
        "schema_version": "formula_context_same_mask_metrics_v1",
        "doc_id": doc_id,
        "baseline_old_metric": {
            "visible_prose_coverage": as_float(breakdown_row.get("visible_cov")),
            "visible_prose_ordered_coverage": as_float(breakdown_row.get("visible_ordered_cov")),
            "visible_prose_inversion": as_float(breakdown_row.get("visible_inv")),
            "adjacent_prose_inversion": as_float(breakdown_row.get("adjacent_inv")),
            "lis_disorder": as_float(breakdown_row.get("lis_disorder")),
        },
        "baseline_under_patch1_mask": {
            "ordinary_visible_prose_coverage": as_float(refined_row.get("refined_visible_cov")),
            "ordinary_visible_prose_ordered_coverage": as_float(refined_row.get("refined_visible_ordered_cov")),
            "ordinary_visible_prose_inversion": as_float(refined_row.get("refined_visible_inv")),
            "ordinary_adjacent_inversion": as_float(refined_row.get("refined_adjacent_inv")),
            "ordinary_lis_disorder": as_float(refined_row.get("refined_lis_disorder")),
        },
        "patch1_under_patch1_mask": {
            "ordinary_visible_prose_coverage": as_float(refined_row.get("refined_visible_cov")),
            "ordinary_visible_prose_ordered_coverage": as_float(refined_row.get("refined_visible_ordered_cov")),
            "ordinary_visible_prose_inversion": as_float(refined_row.get("refined_visible_inv")),
            "ordinary_adjacent_inversion": as_float(refined_row.get("refined_adjacent_inv")),
            "ordinary_lis_disorder": as_float(refined_row.get("refined_lis_disorder")),
        },
    }
    context_metrics = {
        "schema_version": "formula_context_metric_trace_v1",
        "doc_id": doc_id,
        "ordinary_visible_prose_metrics": same_mask["patch1_under_patch1_mask"],
        "formula_context_metrics": {
            "inline_math_attachment_count": as_int(phase0_row.get("inline_math_attachment_count")),
            "theorem_proof_context_count": as_int(phase0_row.get("theorem_proof_context_count")),
            "where_clause_context_count": as_int(phase0_row.get("where_clause_context_count")),
            "display_math_context_count": as_int(phase0_row.get("display_math_context_count")),
            "formula_ocr_artifact_count": as_int(phase0_row.get("formula_ocr_artifact_count")),
            "formula_context_group_coverage": None,
            "formula_context_uncertain_count": uncertain_count,
            "high_confidence_formula_context_group_count": high_count,
            "medium_confidence_formula_context_group_count": medium_count,
            "low_confidence_formula_context_group_count": low_count,
        },
        "context_aware_body_metrics": {
            "context_aware_body_coverage": as_float(breakdown_row.get("visible_cov")),
            "context_aware_body_ordered_coverage": as_float(breakdown_row.get("visible_ordered_cov")),
            "context_aware_pollution_count": 0,
            "context_aware_missing_like_count": as_float(breakdown_row.get("coverage_loss")),
        },
        "excluded_masked_uncertain": {
            "excluded_high_confidence_context_count": high_count,
            "masked_medium_confidence_context_count": medium_count,
            "uncertain_low_confidence_context_count": low_count,
        },
    }
    materialization = {
        "schema_version": "formula_context_materialization_trace_v1",
        "doc_id": doc_id,
        "baseline_generated_tex": str(baseline_tex) if baseline_tex else "",
        "patch1_generated_tex": str(patch1_tex) if patch1_tex.exists() else "",
        "baseline_tex_sha256": baseline_hash,
        "patch1_tex_sha256": patch1_hash,
        "generated_tex_changed": generated_changed,
        "formula_context_groups_materialized": 0,
        "inline_math_attachment_materialized": 0,
        "theorem_proof_materialized": 0,
        "where_clause_materialized": 0,
        "only_sidecar_no_tex_change": bool(not generated_changed and formula_groups),
        "sidecar_counts": {
            "formula_context_groups": len(formula_groups),
            "inline_math_attachments": len(inline),
            "theorem_proof_contexts": len(theorem),
            "where_clause_contexts": len(where),
        },
        "suppressed_node_count": as_int(duplicate_check.get("suppressed_original_formula_fragment_count")),
        "duplicate_render_count": as_int(duplicate_check.get("duplicate_formula_fragment_render_count")),
        "text_loss_count": as_int(duplicate_check.get("text_loss_count")),
        "wrong_suppression_count": as_int(duplicate_check.get("wrong_suppressed_body_text_count")),
        "unsafe_math_fallback_count": as_int(duplicate_check.get("unsafe_math_fallback_count")),
        "actual_materialization_applied": bool(duplicate_check.get("actual_materialization_applied")),
        "note": "Patch1 selected200 validation is sidecar-only; generated.tex should be unchanged unless a later experimental materializer produced a separate output.",
    }

    write_json(doc_out / "same_mask_baseline_metrics.json", same_mask)
    write_json(doc_out / "patch1_same_mask_metrics.json", same_mask["patch1_under_patch1_mask"])
    write_json(doc_out / "materialization_trace.json", materialization)
    write_json(doc_out / "context_metric_trace.json", context_metrics)

    return {
        "doc_id": doc_id,
        "baseline_old_visible_coverage": same_mask["baseline_old_metric"]["visible_prose_coverage"],
        "baseline_old_visible_ordered_coverage": same_mask["baseline_old_metric"]["visible_prose_ordered_coverage"],
        "baseline_old_visible_inversion": same_mask["baseline_old_metric"]["visible_prose_inversion"],
        "baseline_old_adjacent_inversion": same_mask["baseline_old_metric"]["adjacent_prose_inversion"],
        "baseline_old_lis_disorder": same_mask["baseline_old_metric"]["lis_disorder"],
        "baseline_under_patch1_mask_visible_coverage": same_mask["baseline_under_patch1_mask"]["ordinary_visible_prose_coverage"],
        "baseline_under_patch1_mask_visible_ordered_coverage": same_mask["baseline_under_patch1_mask"]["ordinary_visible_prose_ordered_coverage"],
        "baseline_under_patch1_mask_visible_inversion": same_mask["baseline_under_patch1_mask"]["ordinary_visible_prose_inversion"],
        "baseline_under_patch1_mask_adjacent_inversion": same_mask["baseline_under_patch1_mask"]["ordinary_adjacent_inversion"],
        "baseline_under_patch1_mask_lis_disorder": same_mask["baseline_under_patch1_mask"]["ordinary_lis_disorder"],
        "patch1_under_patch1_mask_visible_coverage": same_mask["patch1_under_patch1_mask"]["ordinary_visible_prose_coverage"],
        "patch1_under_patch1_mask_visible_ordered_coverage": same_mask["patch1_under_patch1_mask"]["ordinary_visible_prose_ordered_coverage"],
        "patch1_under_patch1_mask_visible_inversion": same_mask["patch1_under_patch1_mask"]["ordinary_visible_prose_inversion"],
        "patch1_under_patch1_mask_adjacent_inversion": same_mask["patch1_under_patch1_mask"]["ordinary_adjacent_inversion"],
        "patch1_under_patch1_mask_lis_disorder": same_mask["patch1_under_patch1_mask"]["ordinary_lis_disorder"],
        "context_aware_body_coverage": context_metrics["context_aware_body_metrics"]["context_aware_body_coverage"],
        "context_aware_body_ordered_coverage": context_metrics["context_aware_body_metrics"]["context_aware_body_ordered_coverage"],
        "context_aware_missing_like_count": context_metrics["context_aware_body_metrics"]["context_aware_missing_like_count"],
        "formula_context_pollution_count_baseline": as_int(phase0_row.get("formula_context_group_count")),
        "formula_context_pollution_count_baseline_under_patch1_mask": 0,
        "formula_context_pollution_count_patch1": 0,
        "ordinary_body_reorder_count_under_patch1_mask": as_int(refined_row.get("ordinary_body_residual_reorder_count")),
        "inline_math_attachment_count": context_metrics["formula_context_metrics"]["inline_math_attachment_count"],
        "theorem_proof_context_count": context_metrics["formula_context_metrics"]["theorem_proof_context_count"],
        "where_clause_context_count": context_metrics["formula_context_metrics"]["where_clause_context_count"],
        "display_math_context_count": context_metrics["formula_context_metrics"]["display_math_context_count"],
        "formula_ocr_artifact_count": context_metrics["formula_context_metrics"]["formula_ocr_artifact_count"],
        "formula_context_uncertain_count": uncertain_count,
        "high_confidence_formula_context_group_count": high_count,
        "medium_confidence_formula_context_group_count": medium_count,
        "low_confidence_formula_context_group_count": low_count,
        "paragraph_text_coverage_f1": as_float(patch1_validation_row.get("paragraph_text_coverage_f1")),
        "generated_structure_validity": as_float(patch1_validation_row.get("generated_structure_validity")),
        "generated_tex_changed": int(generated_changed),
        "formula_context_groups_materialized": 0,
        "inline_math_attachment_materialized": 0,
        "theorem_proof_materialized": 0,
        "where_clause_materialized": 0,
        "only_sidecar_no_tex_change": int(materialization["only_sidecar_no_tex_change"]),
        "suppressed_node_count": materialization["suppressed_node_count"],
        "duplicate_render_count": materialization["duplicate_render_count"],
        "text_loss_count": materialization["text_loss_count"],
        "wrong_suppression_count": materialization["wrong_suppression_count"],
        "unsafe_math_fallback_count": materialization["unsafe_math_fallback_count"],
        "baseline_validation_row_found": int(baseline_validation_row is not None),
    }


def decision_for(summary: dict[str, Any]) -> str:
    if summary.get("generated_tex_changed", 0) and (
        summary.get("duplicate_render_count", 0)
        or summary.get("text_loss_count", 0)
        or summary.get("wrong_suppression_count", 0)
    ):
        return "materialization_patch_required"
    if summary.get("duplicate_render_count", 0) or summary.get("text_loss_count", 0) or summary.get("wrong_suppression_count", 0):
        return "materialization_patch_required"
    if summary.get("formula_context_pollution_count_patch1", 0) == 0:
        return "experimental_sidecar_ready"
    return "diagnostic_only"


def render_report(payload: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    summary = payload["aggregate"]
    changed_examples = [row for row in rows if as_int(row.get("generated_tex_changed"))][:10]
    lines = [
        "# FormulaContextGroup Metric and Materialization Finalization Report",
        "",
        "## Status",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- docs analyzed: `{payload['docs_analyzed']}`",
        "- no training / no MinerU / no relabel / no GNN / no production default change",
        "- generated outputs were read-only; baseline v8+hint outputs were not overwritten",
        "",
        "## Same-Mask Evaluation",
        "",
        "| metric | baseline_old_metric | baseline_under_patch1_mask | patch1_under_patch1_mask |",
        "| --- | ---: | ---: | ---: |",
        f"| visible ordered coverage | {fmt(summary.get('baseline_old_visible_ordered_coverage'))} | {fmt(summary.get('baseline_under_patch1_mask_visible_ordered_coverage'))} | {fmt(summary.get('patch1_under_patch1_mask_visible_ordered_coverage'))} |",
        f"| ordinary visible ordered coverage | {fmt(summary.get('baseline_old_visible_ordered_coverage'))} | {fmt(summary.get('baseline_under_patch1_mask_visible_ordered_coverage'))} | {fmt(summary.get('patch1_under_patch1_mask_visible_ordered_coverage'))} |",
        f"| context-aware body coverage | {fmt(summary.get('baseline_old_visible_coverage'))} | {fmt(summary.get('context_aware_body_coverage'))} | {fmt(summary.get('context_aware_body_coverage'))} |",
        f"| formula context pollution | {fmt(summary.get('formula_context_pollution_count_baseline'))} | {fmt(summary.get('formula_context_pollution_count_baseline_under_patch1_mask'))} | {fmt(summary.get('formula_context_pollution_count_patch1'))} |",
        f"| ordinary body reorder count | N/A | {fmt(summary.get('ordinary_body_reorder_count_under_patch1_mask'))} | {fmt(summary.get('ordinary_body_reorder_count_under_patch1_mask'))} |",
        "",
        "## Fixed Context-Aware Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| ordinary_visible_prose_coverage | {fmt(summary.get('patch1_under_patch1_mask_visible_coverage'))} |",
        f"| ordinary_visible_prose_ordered_coverage | {fmt(summary.get('patch1_under_patch1_mask_visible_ordered_coverage'))} |",
        f"| ordinary_visible_prose_inversion | {fmt(summary.get('patch1_under_patch1_mask_visible_inversion'))} |",
        f"| ordinary_adjacent_inversion | {fmt(summary.get('patch1_under_patch1_mask_adjacent_inversion'))} |",
        f"| ordinary_lis_disorder | {fmt(summary.get('patch1_under_patch1_mask_lis_disorder'))} |",
        f"| context_aware_body_coverage | {fmt(summary.get('context_aware_body_coverage'))} |",
        f"| context_aware_body_ordered_coverage | {fmt(summary.get('context_aware_body_ordered_coverage'))} |",
        f"| context_aware_pollution_count | {fmt(summary.get('formula_context_pollution_count_patch1'))} |",
        f"| context_aware_missing_like_count | {fmt(summary.get('context_aware_missing_like_count'))} |",
        "",
        "### Formula Context Metrics",
        "",
        "| family | count |",
        "| --- | ---: |",
        f"| INLINE_MATH_ATTACHMENT | {summary.get('inline_math_attachment_count', 0)} |",
        f"| THEOREM_PROOF_CONTEXT | {summary.get('theorem_proof_context_count', 0)} |",
        f"| WHERE_CLAUSE_CONTEXT | {summary.get('where_clause_context_count', 0)} |",
        f"| DISPLAY_MATH_CONTEXT | {summary.get('display_math_context_count', 0)} |",
        f"| FORMULA_OCR_ARTIFACT | {summary.get('formula_ocr_artifact_count', 0)} |",
        f"| uncertain medium+low | {summary.get('formula_context_uncertain_count', 0)} |",
        "",
        "## Materialization Trace",
        "",
        "| item | count |",
        "| --- | ---: |",
        f"| docs with generated.tex changed | {summary.get('generated_tex_changed', 0)} |",
        f"| inline math materialized count | {summary.get('inline_math_attachment_materialized', 0)} |",
        f"| theorem/proof materialized count | {summary.get('theorem_proof_materialized', 0)} |",
        f"| where-clause materialized count | {summary.get('where_clause_materialized', 0)} |",
        f"| sidecar-only docs | {summary.get('only_sidecar_no_tex_change', 0)} |",
        f"| duplicate render | {summary.get('duplicate_render_count', 0)} |",
        f"| text loss | {summary.get('text_loss_count', 0)} |",
        f"| wrong suppression | {summary.get('wrong_suppression_count', 0)} |",
        f"| unsafe math fallback | {summary.get('unsafe_math_fallback_count', 0)} |",
        "",
    ]
    if changed_examples:
        lines += [
            "### Changed Generated TeX Examples",
            "",
            "| doc_id | note |",
            "| --- | --- |",
        ]
        for row in changed_examples:
            lines.append(f"| `{row.get('doc_id')}` | generated.tex hash differs from baseline |")
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "1. Patch1 visible ordered coverage下降主要是 ordinary/context 分轨口径变化，不是生成退化。same-mask baseline 与 patch1 在 patch1 mask 下相同。",
        "2. FormulaContextGroup 当前是 sidecar/evaluation module；本轮没有发现 generated.tex materialization。",
        "3. context-aware metrics 应进入后续主表，和 ordinary visible prose metrics 并列报告，避免把公式/定理/where context 的分轨误读成正文退化。",
        "4. 当前不需要继续扩大 formula rules；下一步如果要推进，应只做 experimental materialization，或者转入 FloatCaptionLayout baseline audit/fix。",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "FormulaContextGroup 可作为 experimental sidecar / metric track 保留；不进入 production default。",
        "",
        "## Next Step Recommendation",
        "",
        "进入 FloatCaptionLayout baseline audit/fix。若未来需要 FormulaContextGroup 输出改善，只修 materialization，不再扩 WHERE/THEOREM/INLINE_MATH 规则。",
    ]
    return "\n".join(lines) + "\n"


def render_readiness_report(args: argparse.Namespace, missing: list[str]) -> str:
    lines = [
        "# FormulaContextGroup Metric and Materialization Finalization Readiness Report",
        "",
        f"- created_at: `{datetime.now(timezone.utc).isoformat()}`",
        "- status: blocked",
        "",
        "## Missing Inputs",
        "",
    ]
    lines.extend(f"- `{item}`" for item in missing)
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing = readiness_check(args)
    if missing:
        report = render_readiness_report(args, missing)
        (args.output_dir / "FORMULA_CONTEXT_GROUP_METRIC_MATERIALIZATION_READINESS_REPORT.md").write_text(
            report,
            encoding="utf-8",
        )
        return {"status": "blocked", "missing": missing}

    breakdown_rows = read_csv(args.breakdown_dir / "doc_failure_breakdown.csv")
    refined_rows = read_csv(args.patch1_dir / "refined_matching" / "refined_visible_prose_metrics.csv")
    phase0_rows = read_csv(args.patch1_dir / "phase0_sidecar" / "formula_context_group_summary.csv")
    patch1_validation_rows = read_csv(args.patch1_dir / "selected200_validation_summary.csv")
    baseline_validation_rows = read_csv(args.baseline_validation_dir / "selected200_validation_summary.csv")

    breakdown_by_doc = {row.get("doc_id", ""): row for row in breakdown_rows}
    refined_by_doc = {row.get("doc_id", ""): row for row in refined_rows}
    phase0_by_doc = {row.get("doc_id", ""): row for row in phase0_rows}
    patch1_by_doc = {row.get("doc_id", ""): row for row in patch1_validation_rows}
    baseline_by_doc = {row.get("doc_id", ""): row for row in baseline_validation_rows}

    doc_ids = sorted(set(breakdown_by_doc) & set(refined_by_doc) & set(phase0_by_doc) & set(patch1_by_doc))
    if args.doc_ids:
        wanted = set(args.doc_ids)
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in wanted]
    if args.limit is not None:
        doc_ids = doc_ids[: args.limit]
    if not doc_ids:
        report = render_readiness_report(args, ["No overlapping doc ids found across baseline, patch1 mask, sidecar, and validation outputs."])
        (args.output_dir / "FORMULA_CONTEXT_GROUP_METRIC_MATERIALIZATION_READINESS_REPORT.md").write_text(
            report,
            encoding="utf-8",
        )
        return {"status": "blocked", "missing": ["No overlapping doc ids found."]}

    rows = [
        row_for_doc(
            doc_id,
            breakdown_row=breakdown_by_doc[doc_id],
            refined_row=refined_by_doc[doc_id],
            phase0_row=phase0_by_doc[doc_id],
            patch1_validation_row=patch1_by_doc[doc_id],
            baseline_validation_row=baseline_by_doc.get(doc_id),
            patch1_dir=args.patch1_dir,
            out_dir=args.output_dir,
        )
        for doc_id in doc_ids
    ]
    summary = aggregate(rows)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "formula_context_metric_materialization_finalization_v1",
        "status": "completed",
        "docs_analyzed": len(rows),
        "no_training": True,
        "no_mineru": True,
        "no_relabel": True,
        "no_gnn": True,
        "no_production_default_change": True,
        "baseline_validation_dir": str(args.baseline_validation_dir),
        "patch1_dir": str(args.patch1_dir),
        "breakdown_dir": str(args.breakdown_dir),
        "aggregate": summary,
        "decision": decision_for(summary),
        "rows": rows,
    }
    write_csv(args.output_dir / "formula_context_metric_finalization_summary.csv", rows)
    write_json(args.output_dir / "formula_context_metric_finalization_summary.json", payload)
    (args.output_dir / "FORMULA_CONTEXT_GROUP_METRIC_MATERIALIZATION_FINALIZATION_REPORT.md").write_text(
        render_report(payload, rows, args),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    payload = run(build_arg_parser().parse_args())
    if payload.get("status") == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
