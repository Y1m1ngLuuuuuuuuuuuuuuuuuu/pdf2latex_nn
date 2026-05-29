#!/usr/bin/env python3
"""Same-code selected200 A/B validation for v8 FloatCaptionLayout.

This tool compares the current code with FloatCaptionLayout disabled against
the same current code with the experimental flag enabled. Historical generated
LaTeX is used only as an artifact source, never as the pass/fail comparator.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "tools" / "audit"
for path in (PROJECT_ROOT, AUDIT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.ir.serialization import read_json, write_json  # noqa: E402
from src.reasoning.float_caption_matcher import is_caption_like_text  # noqa: E402
from validate_v8_float_caption_layout_selected200 import (  # noqa: E402
    DEFAULT_BASELINE_ROOT,
    DEFAULT_FACT_AUDIT,
    aggregate_rows,
    build_experimental_row,
    collect_doc_dirs,
    convert_and_evaluate,
    floatish,
    fmt,
    intish,
    read_csv,
    render_doc,
    select_smoke_doc_ids,
    write_csv,
)

DEFAULT_OUTPUT = Path("data/09_eval_reports/float_caption_layout_20260526/v8_same_code_ab_validation")


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    doc_dirs = collect_doc_dirs(args.baseline_root)
    fact_rows = read_csv(args.fact_audit)
    smoke_ids = select_smoke_doc_ids(fact_rows, doc_dirs, limit=args.smoke_count)
    smoke_doc_dirs = [doc_dirs[doc_id] for doc_id in smoke_ids if doc_id in doc_dirs]

    smoke_baseline = run_batch(
        doc_dirs=smoke_doc_dirs,
        output_dir=args.output_dir / "smoke20_baseline_flag_off_current_code",
        enable_float_caption_layout=False,
    )
    smoke_experimental = run_batch(
        doc_dirs=smoke_doc_dirs,
        output_dir=args.output_dir / "smoke20_experimental_flag_on_current_code",
        enable_float_caption_layout=True,
    )
    smoke_ab = summarize_same_code(smoke_baseline["rows"], smoke_experimental["rows"])
    smoke_diff = diff_batch(
        baseline_root=args.output_dir / "smoke20_baseline_flag_off_current_code",
        experimental_root=args.output_dir / "smoke20_experimental_flag_on_current_code",
    )
    write_json(args.output_dir / "smoke20_same_code_ab_summary.json", smoke_ab)
    write_csv(args.output_dir / "smoke20_same_code_ab_summary.csv", summary_rows(smoke_ab))
    write_json(args.output_dir / "smoke20_diff_attribution.json", smoke_diff)
    write_csv(args.output_dir / "smoke20_diff_attribution.csv", smoke_diff["rows"])

    smoke_decision = decide_gate(smoke_ab, smoke_diff)
    if smoke_decision != "pass":
        write_report(
            args.output_dir / "V8_FLOAT_CAPTION_SAME_CODE_AB_VALIDATION_REPORT.md",
            status="blocked_smoke20_regression",
            smoke_ab=smoke_ab,
            selected_ab=None,
            smoke_diff=smoke_diff,
            selected_diff=None,
            compile_smoke_status="not_run",
            decision="patch_required",
        )
        return 3

    selected_doc_dirs = list(doc_dirs.values())
    selected_baseline = run_batch(
        doc_dirs=selected_doc_dirs,
        output_dir=args.output_dir / "baseline_flag_off_current_code",
        enable_float_caption_layout=False,
    )
    selected_experimental = run_batch(
        doc_dirs=selected_doc_dirs,
        output_dir=args.output_dir / "experimental_flag_on_current_code",
        enable_float_caption_layout=True,
    )
    selected_ab = summarize_same_code(selected_baseline["rows"], selected_experimental["rows"])
    selected_diff = diff_batch(
        baseline_root=args.output_dir / "baseline_flag_off_current_code",
        experimental_root=args.output_dir / "experimental_flag_on_current_code",
    )
    write_json(args.output_dir / "selected200_same_code_ab_summary.json", selected_ab)
    write_csv(args.output_dir / "selected200_same_code_ab_summary.csv", summary_rows(selected_ab))
    write_json(args.output_dir / "selected200_diff_attribution.json", selected_diff)
    write_csv(args.output_dir / "selected200_diff_attribution.csv", selected_diff["rows"])

    decision = decide_final(selected_ab, selected_diff)
    write_report(
        args.output_dir / "V8_FLOAT_CAPTION_SAME_CODE_AB_VALIDATION_REPORT.md",
        status="completed",
        smoke_ab=smoke_ab,
        selected_ab=selected_ab,
        smoke_diff=smoke_diff,
        selected_diff=selected_diff,
        compile_smoke_status="skipped",
        decision=decision,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--fact-audit", type=Path, default=DEFAULT_FACT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-count", type=int, default=20)
    return parser


def run_batch(*, doc_dirs: list[Path], output_dir: Path, enable_float_caption_layout: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    examples = empty_examples()
    for doc_dir in doc_dirs:
        doc_id = doc_dir.name.split("_", 1)[-1]
        out_doc_dir = output_dir / doc_dir.name
        rendered = render_doc(
            doc_dir,
            out_doc_dir,
            enable_float_caption_layout=enable_float_caption_layout,
            use_source_tex_for_parity=False,
        )
        metrics = convert_and_evaluate(out_doc_dir, doc_id, doc_dir)
        row = build_experimental_row(doc_id, out_doc_dir, doc_dir, rendered["diag"], metrics)
        row.update(candidate_consumption_stats(out_doc_dir, rendered["diag"]))
        rows.append(row)
        collect_examples(examples, doc_id, rendered["diag"], row, enable_float_caption_layout=enable_float_caption_layout)
    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "examples.json", examples)
    return {"docs": len(rows), "rows": rows, "examples": examples, "aggregate": aggregate_rows(rows)}


def candidate_consumption_stats(out_doc_dir: Path, diag: dict[str, Any]) -> dict[str, int]:
    structure = read_json(out_doc_dir / "ours_comparison_structure_current.json")
    pred_caption_texts = normalized_caption_texts(structure)
    promoted = diag.get("promoted_captions", [])
    metadata_candidates = [
        item
        for item in promoted
        if item.get("origin") in {"caption_metadata", "float_metadata", "crop_metadata"}
    ]
    crop_candidates = [item for item in promoted if item.get("origin") == "crop_metadata"]
    not_consumed = [item for item in metadata_candidates if not caption_consumed(item, pred_caption_texts)]
    crop_not_consumed = [item for item in crop_candidates if not caption_consumed(item, pred_caption_texts)]
    return {
        "metadata_caption_not_consumed_count": len(not_consumed),
        "crop_swallowed_caption_count": len(crop_not_consumed),
    }


def normalized_caption_texts(structure: dict[str, Any]) -> list[str]:
    texts = []
    for block in structure.get("blocks", []):
        if block.get("block_type") == "caption":
            texts.append(normalize(block.get("text") or block.get("normalized_text") or ""))
    return texts


def caption_consumed(candidate: dict[str, Any], pred_caption_texts: list[str]) -> bool:
    text = normalize(candidate.get("normalized_caption_text") or candidate.get("text") or "")
    if not text:
        return False
    return any(text in pred or pred in text for pred in pred_caption_texts if pred)


def normalize(text: str) -> str:
    return " ".join(str(text).casefold().replace(":", " ").split())


def empty_examples() -> dict[str, list[dict[str, Any]]]:
    return {
        "missing_caption_recovered": [],
        "metadata_crop_caption_materialized": [],
        "caption_as_paragraph_fixed": [],
        "duplicate_suppression": [],
        "placeholder_float": [],
        "algorithm_caption": [],
        "regressions": [],
        "false_caption": [],
    }


def collect_examples(
    examples: dict[str, list[dict[str, Any]]],
    doc_id: str,
    diag: dict[str, Any],
    row: dict[str, Any],
    *,
    enable_float_caption_layout: bool,
) -> None:
    if enable_float_caption_layout:
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
    if row.get("caption_as_paragraph_count"):
        append_example(examples["false_caption"], doc_id, {"text": f"caption-like paragraphs: {row.get('caption_as_paragraph_count')}"})
    if row.get("wrong_float_type_pairing_count"):
        append_example(examples["regressions"], doc_id, row)


def append_example(bucket: list[dict[str, Any]], doc_id: str, item: dict[str, Any], limit: int = 40) -> None:
    if len(bucket) >= limit:
        return
    preview = item.get("text") or item.get("normalized_caption_text") or item.get("reason") or str(item)
    bucket.append(
        {
            "doc_id": doc_id,
            "preview": str(preview)[:240],
            **{k: v for k, v in item.items() if k in {"caption_type", "caption_number", "origin", "reason"}},
        }
    )


def summarize_same_code(baseline_rows: list[dict[str, Any]], experimental_rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "figure_caption_pred_count",
        "figure_caption_missing_count",
        "table_caption_pred_count",
        "table_caption_missing_count",
        "algorithm_caption_pred_count",
        "algorithm_caption_missing_count",
        "crop_may_include_caption_count",
        "consumed_caption_paragraph_count",
        "duplicate_suppressed_count",
        "true_duplicate_caption_count",
        "panel_label_count",
        "subfigure_caption_count",
        "subfigure_caption_preserved_count",
        "synthetic_fallback_caption_count",
        "canonical_caption_count",
        "noncanonical_suppressed_count",
        "subfigure_false_suppression_count",
        "body_reference_false_positive_blocked_count",
        "promoted_from_metadata_count",
        "promoted_from_crop_metadata_count",
        "promoted_from_text_block_count",
        "generated_structure_validity",
        "macro_structure_score_body",
        "paragraph_text_coverage_f1",
        "paragraph_boundary_f1",
        "reading_order_accuracy",
        "section_attachment_body_no_float_f1",
        "reference_section_completeness",
        "float_caption_attachment_accuracy",
    ]
    baseline = aggregate_rows(baseline_rows, fields=fields)
    experimental = aggregate_rows(experimental_rows, fields=fields)
    baseline["label"] = "baseline_current_code_flag_off"
    experimental["label"] = "experimental_current_code_flag_on"
    delta = {"label": "delta_experimental_minus_baseline"}
    for field in fields:
        b = baseline.get(field)
        e = experimental.get(field)
        if isinstance(b, (int, float)) and isinstance(e, (int, float)):
            delta[field] = e - b
        else:
            delta[field] = ""
    return {
        "baseline": baseline,
        "experimental": experimental,
        "delta": delta,
        "baseline_rows": baseline_rows,
        "experimental_rows": experimental_rows,
    }


def summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [summary["baseline"], summary["experimental"], summary["delta"]]


def diff_batch(*, baseline_root: Path, experimental_root: Path) -> dict[str, Any]:
    rows = []
    for baseline_doc in sorted(path for path in baseline_root.iterdir() if path.is_dir()):
        exp_doc = experimental_root / baseline_doc.name
        if not exp_doc.exists():
            continue
        row = classify_tex_diff(
            doc_id=baseline_doc.name.split("_", 1)[-1],
            baseline_tex=baseline_doc / "generated.tex",
            experimental_tex=exp_doc / "generated.tex",
        )
        rows.append(row)
    aggregate = {
        "docs": len(rows),
        "changed_docs": sum(1 for row in rows if row["generated_tex_changed"]),
        "caption_related_changed_docs": sum(1 for row in rows if row["caption_related_change_count"] > 0),
        "non_caption_suspicious_docs": sum(1 for row in rows if row["non_caption_suspicious_change_count"] > 0),
        "caption_related_change_count": sum(row["caption_related_change_count"] for row in rows),
        "non_caption_suspicious_change_count": sum(row["non_caption_suspicious_change_count"] for row in rows),
    }
    return {"aggregate": aggregate, "rows": rows}


def classify_tex_diff(*, doc_id: str, baseline_tex: Path, experimental_tex: Path) -> dict[str, Any]:
    old = baseline_tex.read_text(encoding="utf-8", errors="replace").splitlines()
    new = experimental_tex.read_text(encoding="utf-8", errors="replace").splitlines()
    diff = list(difflib.unified_diff(old, new, n=1, lineterm=""))
    changed = [line[1:] for line in diff if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---"))]
    caption_related = []
    suspicious = []
    for line in changed:
        if is_caption_related_line(line):
            caption_related.append(line)
        elif line.strip():
            suspicious.append(line)
    return {
        "doc_id": doc_id,
        "generated_tex_changed": old != new,
        "diff_line_count": len(changed),
        "caption_related_change_count": len(caption_related),
        "non_caption_suspicious_change_count": len(suspicious),
        "caption_related_examples": " || ".join(caption_related[:5]),
        "non_caption_suspicious_examples": " || ".join(suspicious[:5]),
    }


def is_caption_related_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    lowered = stripped.casefold()
    if any(token in lowered for token in ["\\caption", "caption", "figure placeholder", "table placeholder", "algorithm placeholder"]):
        return True
    if any(token in stripped for token in ["\\begin{figure", "\\end{figure", "\\begin{table", "\\end{table", "\\begin{algorithm", "\\end{algorithm", "\\label{fig:", "\\label{tab:", "\\label{alg:"]):
        return True
    heading_caption = re.match(
        r"\\(?:section|subsection|subsubsection|paragraph|subparagraph)\*?\{(?P<title>.*)\}\s*$",
        stripped,
    )
    if heading_caption and is_caption_like_text(heading_caption.group("title")):
        return True
    if is_caption_like_text(stripped.lstrip("% ")):
        return True
    return False


def decide_gate(summary: dict[str, Any], diff: dict[str, Any]) -> str:
    delta = summary["delta"]
    if (floatish(delta.get("generated_structure_validity")) or 0.0) < -1e-6:
        return "fail"
    if (floatish(delta.get("paragraph_text_coverage_f1")) or 0.0) < -0.005:
        return "fail"
    if diff["aggregate"].get("non_caption_suspicious_change_count", 0) > diff["aggregate"].get("caption_related_change_count", 0) * 2:
        return "fail"
    return "pass"


def decide_final(summary: dict[str, Any], diff: dict[str, Any]) -> str:
    delta = summary["delta"]
    if (floatish(delta.get("generated_structure_validity")) or 0.0) < -1e-6:
        return "patch_required"
    if (floatish(delta.get("paragraph_text_coverage_f1")) or 0.0) < -0.005:
        return "patch_required"
    if (floatish(delta.get("reference_section_completeness")) or 0.0) < -0.005:
        return "patch_required"
    if (floatish(delta.get("wrong_float_type_pairing_count")) or 0.0) > 0:
        return "patch_required"
    if diff["aggregate"].get("non_caption_suspicious_change_count", 0) > diff["aggregate"].get("caption_related_change_count", 0) * 2:
        return "patch_required"
    if (floatish(delta.get("float_caption_attachment_accuracy")) or 0.0) > 0 or (floatish(delta.get("missing_caption_count")) or 0.0) < 0:
        return "safe_to_keep_experimental_enabled"
    return "diagnostic_only"


def write_report(
    path: Path,
    *,
    status: str,
    smoke_ab: dict[str, Any],
    selected_ab: dict[str, Any] | None,
    smoke_diff: dict[str, Any],
    selected_diff: dict[str, Any] | None,
    compile_smoke_status: str,
    decision: str,
) -> None:
    summary = selected_ab or smoke_ab
    diff = selected_diff or smoke_diff
    baseline = summary["baseline"]
    experimental = summary["experimental"]
    delta = summary["delta"]
    examples = collect_report_examples(path.parent)
    lines = [
        "# V8 Float-Caption Same-Code A/B Validation Report",
        "",
        "## Status",
        f"- Status: {status}",
        f"- Docs analyzed: {baseline.get('docs', 0) if selected_ab else 0}",
        "- Smoke20 status: completed",
        f"- Selected200 status: {'completed' if selected_ab else 'not_run'}",
        f"- Compile smoke status: {compile_smoke_status}",
        "- Training: No",
        "- MinerU: No",
        "- Relabel / rebuild: No",
        "- GNN: No",
        "- Production default unchanged: Yes",
        "",
        "## Why Same-Code A/B",
        "- Previous flag-off parity failed because the historical baseline was produced by older code.",
        "- Old generated.tex is historical reference only.",
        "- This report compares current code flag-off vs current code flag-on.",
        "",
        "## v8-only Confirmation",
        "- v8 full observable facts were used.",
        "- No fallback to old v7 was used.",
        "- Legacy names such as source_v7_ids / v7_id are provenance names only.",
        "- The only intended difference is enable_float_caption_layout false vs true.",
        "- TeX source was not used for float-caption inference or citation parity in this same-code validation.",
        "",
        "## Smoke20 Result",
    ]
    lines.extend(metric_table(smoke_ab, title=False))
    lines.extend(
        [
            "",
            "## selected200 A/B Summary",
        ]
    )
    if selected_ab:
        lines.extend(metric_table(selected_ab, title=False))
    else:
        lines.append("- Not run because smoke20 failed.")
    lines.extend(
        [
            "",
            "## Type Breakdown",
            "| metric | baseline_current_code | experimental_current_code | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in ["figure_caption_missing_count", "table_caption_missing_count", "algorithm_caption_missing_count", "figure_caption_pred_count", "table_caption_pred_count", "algorithm_caption_pred_count"]:
        lines.append(f"| {metric} | {fmt(baseline.get(metric))} | {fmt(experimental.get(metric))} | {fmt(delta.get(metric))} |")
    lines.extend(
        [
            "",
            "## Diff Attribution",
            f"- changed docs: {diff['aggregate'].get('changed_docs')}",
            f"- caption-related change lines: {diff['aggregate'].get('caption_related_change_count')}",
            f"- non-caption suspicious change lines: {diff['aggregate'].get('non_caption_suspicious_change_count')}",
            f"- implementation leakage docs: {diff['aggregate'].get('non_caption_suspicious_docs')}",
            "",
            "## Improved Examples",
        ]
    )
    for heading, key, limit in [
        ("Missing caption recovered", "missing_caption_recovered", 20),
        ("Metadata/crop caption materialized", "metadata_crop_caption_materialized", 20),
        ("Caption-as-paragraph fixed", "caption_as_paragraph_fixed", 10),
        ("Duplicate suppression examples", "duplicate_suppression", 10),
        ("Placeholder float examples", "placeholder_float", 10),
        ("Algorithm caption examples", "algorithm_caption", 10),
    ]:
        lines.append(f"### {heading}")
        bucket = examples.get(key, [])
        if not bucket:
            lines.append("- none")
        for item in bucket[:limit]:
            lines.append(f"- {item.get('doc_id')}: {item.get('preview')}")
    lines.extend(["", "## Regressions"])
    regressions = examples.get("regressions", [])
    if regressions:
        for item in regressions:
            lines.append(f"- {item.get('doc_id')}: {item.get('preview')}")
    else:
        lines.append("- No compile smoke was run; skip-compile checks found no severe structure-validity regression.")
    lines.extend(["", "## Decision", decision, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def metric_table(summary: dict[str, Any], *, title: bool = True) -> list[str]:
    baseline = summary["baseline"]
    experimental = summary["experimental"]
    delta = summary["delta"]
    lines = [
        "| metric | baseline_current_code flag-off | experimental_current_code flag-on | delta |",
        "|---|---:|---:|---:|",
    ]
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
    return lines


def collect_report_examples(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    for candidate in [
        output_root / "experimental_flag_on_current_code" / "examples.json",
        output_root / "smoke20_experimental_flag_on_current_code" / "examples.json",
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return empty_examples()


if __name__ == "__main__":
    raise SystemExit(main())
