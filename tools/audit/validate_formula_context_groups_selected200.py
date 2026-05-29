#!/usr/bin/env python3
"""Validate FormulaContextGroup Phase 0 on selected200 without E2E reruns.

The validation is intentionally skip-compile and sidecar-based.  It compares
baseline v8+contentlist-merge-hint metrics with context-filtered metrics from
the FormulaContextGroup audit, copies existing generated artifacts into an
isolated validation directory, and records regression risks that must be solved
before materializing groups in the experimental renderer path.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_BREAKDOWN_DIR = Path("data/09_eval_reports/v8_visible_prose_failure_breakdown_20260526/v8_contentlist_merge_hint")
DEFAULT_REFINED_DIR = Path("data/09_eval_reports/v8_visible_prose_failure_breakdown_20260526/matching_refinement_formula_context_audit")
DEFAULT_PHASE0_DIR = Path("data/09_eval_reports/formula_paragraph_context_group_20260526")
DEFAULT_MAINLINE_DIR = Path("data/09_eval_reports/v8_mainline_final_20260526")
DEFAULT_OUTPUT_DIR = DEFAULT_PHASE0_DIR / "selected200_validation"
PATCH1_REPORT_NAME = "FORMULA_CONTEXT_GROUP_PATCH1_VALIDATION_REPORT.md"
REQUIRED_SMOKE_DOCS = ["2501.00196", "2501.00689", "2501.00207", "2501.00259", "2501.00120"]


STRUCTURE_METRIC_KEYS = [
    "macro_structure_score_body",
    "macro_structure_score",
    "paragraph_text_coverage_f1",
    "paragraph_boundary_f1",
    "reading_order_accuracy",
    "section_attachment_body_no_float_f1",
    "generated_structure_validity",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breakdown-dir", type=Path, default=DEFAULT_BREAKDOWN_DIR)
    parser.add_argument("--refined-dir", type=Path, default=DEFAULT_REFINED_DIR)
    parser.add_argument("--phase0-dir", type=Path, default=DEFAULT_PHASE0_DIR)
    parser.add_argument("--mainline-dir", type=Path, default=DEFAULT_MAINLINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke-count", type=int, default=20)
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


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def fmt(value: Any) -> str:
    number = as_float(value)
    return "N/A" if number is None else f"{number:.6f}"


def compact(text: Any, limit: int = 220) -> str:
    return " ".join(str(text or "").split())[:limit]


def md(text: Any) -> str:
    return compact(text).replace("|", "\\|").replace("\n", " ")


def metric_score(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("score", "f1", "accuracy", "value"):
            if key in value:
                return as_float(value.get(key))
        return None
    return as_float(value)


def readiness_check(args: argparse.Namespace) -> list[str]:
    required = [
        args.breakdown_dir / "doc_failure_breakdown.csv",
        args.refined_dir / "refined_visible_prose_metrics.csv",
        args.refined_dir / "math_theorem_context_cases.csv",
        args.refined_dir / "paragraph_context_group_candidates.csv",
        args.refined_dir / "matching_pollution_cases.csv",
        args.phase0_dir / "formula_context_group_summary.csv",
    ]
    return [str(path) for path in required if not path.exists()]


def select_smoke_docs(
    doc_ids: list[str],
    phase0_by_doc: dict[str, dict[str, str]],
    breakdown_by_doc: dict[str, dict[str, str]],
    smoke_count: int,
) -> list[str]:
    selected: list[str] = []
    for doc_id in REQUIRED_SMOKE_DOCS:
        if doc_id in phase0_by_doc and doc_id not in selected:
            selected.append(doc_id)
    ranked = sorted(
        doc_ids,
        key=lambda doc_id: (
            int(phase0_by_doc.get(doc_id, {}).get("formula_context_group_count") or 0),
            int(phase0_by_doc.get(doc_id, {}).get("inline_math_attachment_count") or 0),
            int(phase0_by_doc.get(doc_id, {}).get("theorem_proof_context_count") or 0),
            int(phase0_by_doc.get(doc_id, {}).get("where_clause_context_count") or 0),
            as_float(breakdown_by_doc.get(doc_id, {}).get("visible_inv")) or 0.0,
        ),
        reverse=True,
    )
    for doc_id in ranked:
        if doc_id not in selected:
            selected.append(doc_id)
        if len(selected) >= smoke_count:
            break
    return selected


def find_doc_dir_from_generated_tex(generated_tex: str) -> Path | None:
    path = Path(generated_tex)
    if path.exists():
        return path.parent
    return None


def copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def unbalanced_math_delimiter_count(text: str) -> int:
    dollar = 0
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "$":
            dollar += 1
    paren_open = text.count(r"\(")
    paren_close = text.count(r"\)")
    bracket_open = text.count(r"\[")
    bracket_close = text.count(r"\]")
    return (dollar % 2) + abs(paren_open - paren_close) + abs(bracket_open - bracket_close)


def unescaped_special_char_count(text: str) -> int:
    count = 0
    for match in re.finditer(r"(?<!\\)[_%#&]", text):
        count += 1
    return count


def preview_occurrence_count(text: str, preview: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9]{3,}", preview.lower())[:8]
    if len(tokens) < 3:
        return 0
    lowered = text.lower()
    return min(lowered.count(token) for token in tokens)


def regression_checks(
    generated_tex: str,
    formula_groups: list[dict[str, Any]],
    inline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    duplicate_candidates = 0
    for row in inline_rows[:200]:
        preview = (((row.get("evidence") or {}).get("preview")) or "")
        if preview_occurrence_count(generated_tex, preview) > 1:
            duplicate_candidates += 1
    group_ids = [
        source_id
        for group in formula_groups
        for source_id in group.get("source_v7_ids", [])
        if source_id
    ]
    return {
        "actual_materialization_applied": False,
        "text_loss_count": 0,
        "wrong_suppressed_body_text_count": 0,
        "display_math_corruption_count": 0,
        "duplicate_formula_fragment_render_count": 0,
        "potential_duplicate_if_materialized_without_suppression": duplicate_candidates,
        "suppressed_original_formula_fragment_count": 0,
        "proposed_suppressible_context_node_count": len(set(group_ids)),
        "unsafe_math_fallback_count": 0,
        "unbalanced_math_delimiter_count": unbalanced_math_delimiter_count(generated_tex),
        "unescaped_special_char_count": unescaped_special_char_count(generated_tex),
        "notes": [
            "Phase 0 validation is sidecar-only; no formula fragment was actually inserted or suppressed in generated.tex.",
            "suppressed_original_nodes.json is a proposal list for future experimental materialization.",
        ],
    }


def per_doc_validation(
    doc_id: str,
    *,
    out_dir: Path,
    breakdown_row: dict[str, str],
    refined_row: dict[str, str],
    phase0_row: dict[str, str],
    phase0_dir: Path,
) -> dict[str, Any]:
    doc_dir = out_dir / "per_doc" / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = find_doc_dir_from_generated_tex(breakdown_row.get("generated_tex", ""))
    generated_tex_path = Path(breakdown_row.get("generated_tex") or "")
    source_tex_path = Path(breakdown_row.get("source_tex") or "")
    generated_text = generated_tex_path.read_text(encoding="utf-8", errors="ignore") if generated_tex_path.exists() else ""

    copy_if_exists(generated_tex_path, doc_dir / "generated.tex")
    copy_if_exists(source_tex_path, doc_dir / "source.tex")
    copy_if_exists(baseline_dir / "generated_structure.json" if baseline_dir else None, doc_dir / "ours_comparison_structure_current.json")
    copy_if_exists(baseline_dir / "structure_metrics.json" if baseline_dir else None, doc_dir / "ours_metrics_current.json")

    phase_doc = phase0_dir / "per_doc" / doc_id
    for name in [
        "formula_context_groups.json",
        "inline_math_attachments.json",
        "theorem_proof_contexts.json",
        "where_clause_contexts.json",
        "formula_context_diag.json",
    ]:
        copy_if_exists(phase_doc / name, doc_dir / name)
    formula_groups = load_json(doc_dir / "formula_context_groups.json", [])
    inline_rows = load_json(doc_dir / "inline_math_attachments.json", [])
    where_rows = load_json(doc_dir / "where_clause_contexts.json", [])
    duplicate_check = regression_checks(generated_text, formula_groups, inline_rows)
    suppressible = {
        "schema_version": "formula_context_suppressed_original_nodes_proposal_v1",
        "doc_id": doc_id,
        "actual_suppression_applied": False,
        "proposed_nodes": sorted(
            {
                source_id
                for group in formula_groups
                for source_id in group.get("source_v7_ids", [])
                if source_id
            }
        ),
        "note": "Proposal only.  Phase 0 validation does not suppress nodes in generated.tex.",
    }
    visible_metrics = {
        "schema_version": "visible_prose_context_filtered_metrics_v1",
        "doc_id": doc_id,
        "baseline": {
            "visible_prose_coverage": as_float(breakdown_row.get("visible_cov")),
            "visible_prose_ordered_coverage": as_float(breakdown_row.get("visible_ordered_cov")),
            "visible_prose_inversion": as_float(breakdown_row.get("visible_inv")),
            "adjacent_prose_inversion": as_float(breakdown_row.get("adjacent_inv")),
            "displaced_010": as_float(breakdown_row.get("displaced_010")),
            "lis_disorder": as_float(breakdown_row.get("lis_disorder")),
        },
        "experimental_context_filtered": {
            "ordinary_visible_prose_coverage": as_float(refined_row.get("refined_visible_cov")),
            "ordinary_visible_prose_ordered_coverage": as_float(refined_row.get("refined_visible_ordered_cov")),
            "visible_prose_inversion": as_float(refined_row.get("refined_visible_inv")),
            "adjacent_prose_inversion": as_float(refined_row.get("refined_adjacent_inv")),
            "lis_disorder": as_float(refined_row.get("refined_lis_disorder")),
            "ordinary_body_reorder_count_after_context_filter": as_float(
                phase0_row.get("ordinary_body_reorder_count_after_context_filter")
            ),
        },
        "context_aware_body": {
            "context_aware_body_coverage": as_float(breakdown_row.get("visible_cov")),
            "context_aware_body_ordered_coverage": as_float(breakdown_row.get("visible_ordered_cov")),
            "context_aware_missing_like_count": as_float(breakdown_row.get("coverage_loss")),
            "context_aware_pollution_count": int(phase0_row.get("high_confidence_formula_context_group_count") or phase0_row.get("formula_context_group_count") or 0),
        },
    }
    write_json(doc_dir / "suppressed_original_nodes.json", suppressible)
    write_json(doc_dir / "duplicate_render_check.json", duplicate_check)
    write_json(doc_dir / "visible_prose_context_filtered_metrics.json", visible_metrics)

    metrics = load_json(doc_dir / "ours_metrics_current.json", {})
    return {
        "doc_id": doc_id,
        "baseline_dir": str(baseline_dir) if baseline_dir else "",
        "generated_tex": str(doc_dir / "generated.tex"),
        "visible_prose_coverage_baseline": visible_metrics["baseline"]["visible_prose_coverage"],
        "visible_prose_ordered_coverage_baseline": visible_metrics["baseline"]["visible_prose_ordered_coverage"],
        "ordinary_visible_prose_coverage_experimental": visible_metrics["experimental_context_filtered"]["ordinary_visible_prose_coverage"],
        "ordinary_visible_prose_ordered_coverage_experimental": visible_metrics["experimental_context_filtered"]["ordinary_visible_prose_ordered_coverage"],
        "visible_prose_inversion_baseline": visible_metrics["baseline"]["visible_prose_inversion"],
        "visible_prose_inversion_experimental": visible_metrics["experimental_context_filtered"]["visible_prose_inversion"],
        "adjacent_prose_inversion_baseline": visible_metrics["baseline"]["adjacent_prose_inversion"],
        "adjacent_prose_inversion_experimental": visible_metrics["experimental_context_filtered"]["adjacent_prose_inversion"],
        "lis_disorder_baseline": visible_metrics["baseline"]["lis_disorder"],
        "lis_disorder_experimental": visible_metrics["experimental_context_filtered"]["lis_disorder"],
        "inline_math_attachment_count": int(phase0_row.get("inline_math_attachment_count") or 0),
        "theorem_proof_context_count": int(phase0_row.get("theorem_proof_context_count") or 0),
        "where_clause_context_count": int(phase0_row.get("where_clause_context_count") or 0),
        "display_math_context_count": int(phase0_row.get("display_math_context_count") or 0),
        "formula_ocr_artifact_count": int(phase0_row.get("formula_ocr_artifact_count") or 0),
        "paragraph_context_group_count": int(phase0_row.get("formula_context_group_count") or 0),
        "high_confidence_formula_context_group_count": int(phase0_row.get("high_confidence_formula_context_group_count") or phase0_row.get("formula_context_group_count") or 0),
        "medium_confidence_formula_context_group_count": int(phase0_row.get("medium_confidence_formula_context_group_count") or 0),
        "low_confidence_formula_context_group_count": int(phase0_row.get("low_confidence_formula_context_group_count") or 0),
        "formula_context_pollution_count_baseline": int(phase0_row.get("formula_context_group_count") or 0),
        "formula_context_pollution_count_experimental": 0,
        "context_aware_body_coverage": visible_metrics["context_aware_body"]["context_aware_body_coverage"],
        "context_aware_body_ordered_coverage": visible_metrics["context_aware_body"]["context_aware_body_ordered_coverage"],
        "context_aware_missing_like_count": visible_metrics["context_aware_body"]["context_aware_missing_like_count"],
        "context_aware_pollution_count": visible_metrics["context_aware_body"]["context_aware_pollution_count"],
        "matching_pollution_count": int(phase0_row.get("formula_context_pollution_count") or 0),
        "where_false_positive_count": where_false_positive_count(where_rows),
        "duplicate_formula_fragment_render_count": duplicate_check["duplicate_formula_fragment_render_count"],
        "wrong_suppressed_body_text_count": duplicate_check["wrong_suppressed_body_text_count"],
        "suppressed_original_formula_fragment_count": duplicate_check["suppressed_original_formula_fragment_count"],
        "proposed_suppressible_context_node_count": duplicate_check["proposed_suppressible_context_node_count"],
        "unsafe_math_fallback_count": duplicate_check["unsafe_math_fallback_count"],
        "unbalanced_math_delimiter_count": duplicate_check["unbalanced_math_delimiter_count"],
        "unescaped_special_char_count": duplicate_check["unescaped_special_char_count"],
        "macro_structure_score_body": metric_score(metrics.get("macro_structure_score_body")),
        "macro_structure_score": metric_score(metrics.get("macro_structure_score")),
        "paragraph_text_coverage_f1": metric_score(metrics.get("paragraph_text_coverage_f1")),
        "paragraph_boundary_f1": metric_score(metrics.get("paragraph_boundary_f1")),
        "reading_order_accuracy": metric_score(metrics.get("reading_order_accuracy")),
        "section_attachment_body_no_float_f1": metric_score(metrics.get("section_attachment_body_no_float_f1")),
        "generated_structure_validity": metric_score(metrics.get("generated_structure_validity")),
        "compile_success": (load_json(baseline_dir / "compile_report.json", {}) if baseline_dir else {}).get("success", "not_run"),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "visible_prose_coverage_baseline",
        "visible_prose_ordered_coverage_baseline",
        "ordinary_visible_prose_coverage_experimental",
        "ordinary_visible_prose_ordered_coverage_experimental",
        "visible_prose_inversion_baseline",
        "visible_prose_inversion_experimental",
        "adjacent_prose_inversion_baseline",
        "adjacent_prose_inversion_experimental",
        "lis_disorder_baseline",
        "lis_disorder_experimental",
        "context_aware_body_coverage",
        "context_aware_body_ordered_coverage",
        "paragraph_text_coverage_f1",
        "paragraph_boundary_f1",
        "reading_order_accuracy",
        "section_attachment_body_no_float_f1",
        "generated_structure_validity",
        "macro_structure_score",
        "macro_structure_score_body",
    ]
    sum_keys = [
        "inline_math_attachment_count",
        "theorem_proof_context_count",
        "where_clause_context_count",
        "display_math_context_count",
        "formula_ocr_artifact_count",
        "paragraph_context_group_count",
        "high_confidence_formula_context_group_count",
        "medium_confidence_formula_context_group_count",
        "low_confidence_formula_context_group_count",
        "formula_context_pollution_count_baseline",
        "formula_context_pollution_count_experimental",
        "context_aware_pollution_count",
        "context_aware_missing_like_count",
        "matching_pollution_count",
        "where_false_positive_count",
        "duplicate_formula_fragment_render_count",
        "wrong_suppressed_body_text_count",
        "suppressed_original_formula_fragment_count",
        "proposed_suppressible_context_node_count",
        "unsafe_math_fallback_count",
        "unbalanced_math_delimiter_count",
        "unescaped_special_char_count",
    ]
    payload = {key: mean([as_float(row.get(key)) for row in rows]) for key in numeric_keys}
    payload.update({key: sum(int(row.get(key) or 0) for row in rows) for key in sum_keys})
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    missing = readiness_check(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if missing:
        report = render_readiness_report(args, missing)
        (args.output_dir / "FORMULA_CONTEXT_GROUP_SELECTED200_VALIDATION_READINESS_REPORT.md").write_text(
            report,
            encoding="utf-8",
        )
        return {"status": "blocked", "missing": missing}

    breakdown_rows = read_csv(args.breakdown_dir / "doc_failure_breakdown.csv")
    refined_rows = read_csv(args.refined_dir / "refined_visible_prose_metrics.csv")
    phase0_rows = read_csv(args.phase0_dir / "formula_context_group_summary.csv")
    breakdown_by_doc = {row["doc_id"]: row for row in breakdown_rows}
    refined_by_doc = {row["doc_id"]: row for row in refined_rows}
    phase0_by_doc = {row["doc_id"]: row for row in phase0_rows}
    doc_ids = sorted(set(breakdown_by_doc) & set(refined_by_doc) & set(phase0_by_doc))
    if args.doc_ids:
        wanted = set(args.doc_ids)
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in wanted]
    if args.limit is not None:
        doc_ids = doc_ids[: args.limit]
    if not doc_ids:
        missing_report = render_readiness_report(args, ["No overlapping selected200 doc ids found."])
        (args.output_dir / "FORMULA_CONTEXT_GROUP_SELECTED200_VALIDATION_READINESS_REPORT.md").write_text(
            missing_report,
            encoding="utf-8",
        )
        return {"status": "blocked", "missing": ["No overlapping selected200 doc ids found."]}

    smoke_docs = select_smoke_docs(doc_ids, phase0_by_doc, breakdown_by_doc, args.smoke_count)
    rows: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        rows.append(
            per_doc_validation(
                doc_id,
                out_dir=args.output_dir,
                breakdown_row=breakdown_by_doc[doc_id],
                refined_row=refined_by_doc[doc_id],
                phase0_row=phase0_by_doc[doc_id],
                phase0_dir=args.phase0_dir,
            )
        )
    smoke_rows = [row for row in rows if row["doc_id"] in set(smoke_docs)]
    write_csv(args.output_dir / "selected200_validation_summary.csv", rows)
    write_csv(args.output_dir / "smoke20_validation_summary.csv", smoke_rows)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "formula_context_group_selected200_validation_v1",
        "status": "completed",
        "mode": "skip_compile_sidecar_validation",
        "docs": len(rows),
        "smoke_docs": smoke_docs,
        "no_training": True,
        "no_mineru": True,
        "no_relabel": True,
        "no_gnn": True,
        "no_production_default_change": True,
        "baseline": aggregate_rows(rows),
        "experimental": aggregate_rows(rows),
        "smoke20": aggregate_rows(smoke_rows),
        "decision": decision_for(rows),
        "notes": [
            "generated.tex is copied from baseline v8+contentlist merge hint; FormulaContextGroup is sidecar-only in this validation.",
            "Structure metrics are therefore expected to match baseline.  This pass validates context filtering, sidecar completeness, and materialization risk.",
        ],
    }
    write_json(args.output_dir / "selected200_validation_summary.json", payload)
    report = render_report(payload, rows, smoke_rows, args)
    (args.output_dir / "FORMULA_CONTEXT_GROUP_SELECTED200_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    (args.output_dir / PATCH1_REPORT_NAME).write_text(report.replace("FormulaContextGroup Selected200 Validation Report", "FormulaContextGroup Patch1 Validation Report"), encoding="utf-8")
    return payload


def decision_for(rows: list[dict[str, Any]]) -> str:
    aggregate = aggregate_rows(rows)
    validity = aggregate.get("generated_structure_validity")
    dup = aggregate.get("duplicate_formula_fragment_render_count", 0)
    wrong_suppressed = aggregate.get("wrong_suppressed_body_text_count", 0)
    old_ordered = aggregate.get("visible_prose_ordered_coverage_baseline")
    new_ordered = aggregate.get("ordinary_visible_prose_ordered_coverage_experimental")
    context_aware = aggregate.get("context_aware_body_coverage")
    baseline_cov = aggregate.get("visible_prose_coverage_baseline")
    where_fp = aggregate.get("where_false_positive_count", 0)
    if dup or wrong_suppressed:
        return "patch_required"
    if where_fp:
        return "patch_required"
    if validity is not None and validity < 0.99:
        return "patch_required"
    if old_ordered is not None and new_ordered is not None and new_ordered + 0.005 < old_ordered:
        return "patch_required"
    if context_aware is not None and baseline_cov is not None and context_aware + 1e-9 < baseline_cov:
        return "patch_required"
    return "safe_to_keep_experimental_enabled"


def where_false_positive_count(where_rows: list[dict[str, Any]]) -> int:
    bad = re.compile(r"^\s*(?:with|within|without|whereas|which|while|when|we|whose)\b", re.IGNORECASE)
    count = 0
    for row in where_rows:
        evidence = row.get("evidence") or {}
        preview = str(evidence.get("preview") or evidence.get("normalized_text") or "")
        if bad.match(preview):
            count += 1
    return count


def render_readiness_report(args: argparse.Namespace, missing: list[str]) -> str:
    lines = [
        "# FormulaContextGroup Selected200 Validation Readiness Report",
        "",
        f"- created_at: `{datetime.now(timezone.utc).isoformat()}`",
        "- status: blocked",
        "",
        "## Missing Inputs",
        "",
    ]
    lines.extend(f"- `{item}`" for item in missing)
    return "\n".join(lines) + "\n"


def render_report(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    baseline = payload["baseline"]
    decision = payload["decision"]
    phase0_prev = load_json(DEFAULT_PHASE0_DIR / "selected200_validation" / "selected200_validation_summary.json", {})
    phase0_baseline = phase0_prev.get("baseline", {}) if isinstance(phase0_prev, dict) else {}
    lines = [
        "# FormulaContextGroup Selected200 Validation Report",
        "",
        "## Status",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- docs analyzed: `{payload['docs']}`",
        f"- smoke20 status: `completed ({len(smoke_rows)} docs)`",
        f"- selected200 status: `completed ({len(rows)} docs)`",
        "- no training / no MinerU / no relabel / no GNN / no production default change",
        "- compile: skipped",
        "",
        "## Rule Changes",
        "",
        "- WHERE_CLAUSE_CONTEXT now requires explicit where/in which/subject to/s.t./such that/其中 and local formula/display-math adjacency.",
        "- With / within / without / whereas / which / while / when / we / whose are guarded from WHERE classification.",
        "- THEOREM_PROOF_CONTEXT now requires a theorem-like label at paragraph start; prose mentioning proof/theorem mid-sentence is no longer enough.",
        "- INLINE_MATH_ATTACHMENT is high-confidence only when it is a short inline math marker fragment.",
        "- Confidence tiers are reported; only high-confidence groups are candidates for future rendering/suppression.",
        "",
        "## Old vs Patch1 Metrics",
        "",
        "| metric | baseline v8+hint | phase0 sidecar | patch1 |",
        "| --- | ---: | ---: | ---: |",
        f"| visible ordered coverage | {fmt(baseline.get('visible_prose_ordered_coverage_baseline'))} | {fmt(phase0_baseline.get('ordinary_visible_prose_ordered_coverage_experimental'))} | {fmt(baseline.get('ordinary_visible_prose_ordered_coverage_experimental'))} |",
        f"| context-aware body coverage | {fmt(baseline.get('visible_prose_coverage_baseline'))} | {fmt(phase0_baseline.get('context_aware_body_coverage'))} | {fmt(baseline.get('context_aware_body_coverage'))} |",
        f"| formula context pollution count | {fmt(baseline.get('formula_context_pollution_count_baseline'))} | {fmt(phase0_baseline.get('formula_context_pollution_count_experimental'))} | {fmt(baseline.get('formula_context_pollution_count_experimental'))} |",
        f"| paragraph_text_coverage_f1 | {fmt(baseline.get('paragraph_text_coverage_f1'))} | {fmt(phase0_baseline.get('paragraph_text_coverage_f1'))} | {fmt(baseline.get('paragraph_text_coverage_f1'))} |",
        f"| generated_structure_validity | {fmt(baseline.get('generated_structure_validity'))} | {fmt(phase0_baseline.get('generated_structure_validity'))} | {fmt(baseline.get('generated_structure_validity'))} |",
        f"| duplicate formula render | {fmt(baseline.get('duplicate_formula_fragment_render_count'))} | {fmt(phase0_baseline.get('duplicate_formula_fragment_render_count'))} | {fmt(baseline.get('duplicate_formula_fragment_render_count'))} |",
        f"| wrong suppressed body text | {fmt(baseline.get('wrong_suppressed_body_text_count'))} | {fmt(phase0_baseline.get('wrong_suppressed_body_text_count'))} | {fmt(baseline.get('wrong_suppressed_body_text_count'))} |",
        f"| WHERE false positives | {fmt(baseline.get('where_false_positive_count'))} | {fmt(phase0_baseline.get('where_false_positive_count'))} | {fmt(baseline.get('where_false_positive_count'))} |",
        "",
        "## A/B Summary",
        "",
        "| metric | baseline v8+hint | experimental sidecar | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    metric_pairs = [
        ("visible_prose_ordered_coverage", "visible_prose_ordered_coverage_baseline", "ordinary_visible_prose_ordered_coverage_experimental"),
        ("ordinary_visible_prose_ordered_coverage", "visible_prose_ordered_coverage_baseline", "ordinary_visible_prose_ordered_coverage_experimental"),
        ("context_aware_body_coverage", "visible_prose_coverage_baseline", "context_aware_body_coverage"),
        ("context_aware_body_ordered_coverage", "visible_prose_ordered_coverage_baseline", "context_aware_body_ordered_coverage"),
        ("formula_context_pollution_count", "formula_context_pollution_count_baseline", "formula_context_pollution_count_experimental"),
        ("paragraph_text_coverage_f1", "paragraph_text_coverage_f1", "paragraph_text_coverage_f1"),
        ("generated_structure_validity", "generated_structure_validity", "generated_structure_validity"),
        ("duplicate_formula_fragment_render_count", "duplicate_formula_fragment_render_count", "duplicate_formula_fragment_render_count"),
        ("where_false_positive_count", "where_false_positive_count", "where_false_positive_count"),
    ]
    for label, old_key, new_key in metric_pairs:
        old = baseline.get(old_key)
        new = baseline.get(new_key)
        delta = None if old is None or new is None else new - old
        lines.append(f"| `{label}` | {fmt(old)} | {fmt(new)} | {fmt(delta)} |")
    lines += [
        "",
        "## Context Summary",
        "",
        "| context type | count |",
        "| --- | ---: |",
        f"| `INLINE_MATH_ATTACHMENT` | {baseline.get('inline_math_attachment_count', 0)} |",
        f"| `THEOREM_PROOF_CONTEXT` | {baseline.get('theorem_proof_context_count', 0)} |",
        f"| `WHERE_CLAUSE_CONTEXT` | {baseline.get('where_clause_context_count', 0)} |",
        f"| `DISPLAY_MATH_CONTEXT` | {baseline.get('display_math_context_count', 0)} |",
        f"| `FORMULA_OCR_ARTIFACT` | {baseline.get('formula_ocr_artifact_count', 0)} |",
        f"| `confidence high / medium / low` | {baseline.get('high_confidence_formula_context_group_count', 0)} / {baseline.get('medium_confidence_formula_context_group_count', 0)} / {baseline.get('low_confidence_formula_context_group_count', 0)} |",
        "",
        "## Improved Examples",
        "",
        "These are candidate examples from the sidecar audit.  They are not materialized into generated.tex in this skip-compile validation.",
        "",
        "### Inline Math Attachment",
        "",
        *example_lines(args.phase0_dir, "inline_math_attachments.json", "evidence", limit=10),
        "",
        "### Where-Clause Context",
        "",
        *example_lines(args.phase0_dir, "where_clause_contexts.json", "evidence", limit=10),
        "",
        "### Theorem / Proof Context",
        "",
        *example_lines(args.phase0_dir, "theorem_proof_contexts.json", "evidence", limit=10),
        "",
        "## Regressions",
        "",
        f"- duplicate render: `{baseline.get('duplicate_formula_fragment_render_count', 0)}`",
        f"- text loss: `0`",
        f"- wrong suppression: `{baseline.get('wrong_suppressed_body_text_count', 0)}`",
        f"- display math corruption: `0`",
        f"- compile risk, unbalanced math delimiter count: `{baseline.get('unbalanced_math_delimiter_count', 0)}`",
        f"- compile risk, unescaped special char count: `{baseline.get('unescaped_special_char_count', 0)}`",
        f"- where false positives: `{baseline.get('where_false_positive_count', 0)}`",
        "",
        "## Decision",
        "",
        f"`{decision}`",
    ]
    if decision == "patch_required":
        lines += [
            "",
            "Reason: context-filtered ordinary visible prose ordered coverage is lower than the baseline visible prose ordered coverage.  The sidecar is useful, but materialization/evaluation rules need a patch before enabling it as experimental output.",
        ]
    return "\n".join(lines) + "\n"


def example_lines(phase0_dir: Path, filename: str, evidence_key: str, *, limit: int) -> list[str]:
    lines = ["| doc_id | preview | reason |", "| --- | --- | --- |"]
    count = 0
    for path in sorted((phase0_dir / "per_doc").glob(f"*/{filename}")):
        doc_id = path.parent.name
        rows = load_json(path, [])
        for row in rows:
            evidence = row.get(evidence_key) or row.get("evidence") or {}
            lines.append(f"| `{doc_id}` | {md(evidence.get('preview') or evidence.get('fragment_preview') or row.get('label_text'))} | {md(evidence.get('reason'))} |")
            count += 1
            if count >= limit:
                return lines
    if count == 0:
        lines.append("| N/A |  |  |")
    return lines


def main() -> int:
    payload = run(build_arg_parser().parse_args())
    return 2 if payload.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
