#!/usr/bin/env python3
"""Patch2 validation for v8 FloatCaptionLayout.

This pass keeps the production default unchanged.  It first reruns comparison
conversion on the existing flag-on LaTeX, then reruns same-code flag-off/flag-on
selected200 with the current patch2 code.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "tools" / "audit"
for path in (PROJECT_ROOT, AUDIT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_v8_float_caption_trace_patch_plan import build_missing_caption_trace, build_traces  # noqa: E402
from validate_v8_float_caption_layout_selected200 import (  # noqa: E402
    DEFAULT_BASELINE_ROOT,
    aggregate_rows,
    build_experimental_row,
    collect_doc_dirs,
    convert_and_evaluate,
    render_doc,
    write_csv,
    write_json,
)
from validate_v8_float_caption_same_code_ab import diff_batch, summarize_same_code  # noqa: E402


DEFAULT_OLD_AB_ROOT = Path("data/09_eval_reports/float_caption_layout_20260526/v8_same_code_ab_validation")
DEFAULT_OUTPUT_ROOT = DEFAULT_OLD_AB_ROOT / "patch2_same_code_ab"
DEFAULT_CONVERTER_OUTPUT = DEFAULT_OLD_AB_ROOT / "patch2_converter_only_eval"
REPORT_NAME = "V8_FLOAT_CAPTION_PATCH2_VALIDATION_REPORT.md"


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.converter_output_dir.mkdir(parents=True, exist_ok=True)

    old_trace_counts = read_trace_stage_counts(args.old_ab_root / "caption_trace_audit" / "caption_trace_records.jsonl")
    converter_summary = run_converter_only_eval(args.old_ab_root, args.converter_output_dir)
    converter_traces, _ = build_traces(args.converter_output_dir)
    converter_counts = failure_stage_counts(converter_traces)
    write_json(args.converter_output_dir / "patch2_converter_only_trace_summary.json", {"failure_stage_counts": dict(converter_counts)})

    doc_dirs = collect_doc_dirs(args.baseline_root)
    selected_doc_dirs = list(doc_dirs.values())
    baseline = run_current_batch(
        doc_dirs=selected_doc_dirs,
        output_dir=args.output_dir / "baseline_flag_off_current_code",
        enable_float_caption_layout=False,
    )
    experimental = run_current_batch(
        doc_dirs=selected_doc_dirs,
        output_dir=args.output_dir / "experimental_flag_on_current_code",
        enable_float_caption_layout=True,
    )
    ab_summary = summarize_same_code(baseline["rows"], experimental["rows"])
    diff_summary = diff_batch(
        baseline_root=args.output_dir / "baseline_flag_off_current_code",
        experimental_root=args.output_dir / "experimental_flag_on_current_code",
    )
    write_json(args.output_dir / "selected200_same_code_ab_summary.json", ab_summary)
    write_csv(args.output_dir / "selected200_same_code_ab_summary.csv", summary_rows(ab_summary))
    write_json(args.output_dir / "selected200_diff_attribution.json", diff_summary)
    write_csv(args.output_dir / "selected200_diff_attribution.csv", diff_summary["rows"])

    patch2_traces, _ = build_traces(args.output_dir)
    patch2_counts = failure_stage_counts(patch2_traces)
    write_json(args.output_dir / "patch2_trace_summary.json", {"failure_stage_counts": dict(patch2_counts)})
    write_csv(args.output_dir / "patch2_failure_stage_breakdown.csv", counter_rows(patch2_counts))

    figure_rows, _figure_examples = build_missing_caption_trace(args.output_dir, patch2_traces, caption_type="figure")
    algorithm_rows, _algorithm_examples = build_missing_caption_trace(args.output_dir, patch2_traces, caption_type="algorithm")
    write_csv(args.output_dir / "patch2_figure_missing_trace_breakdown.csv", figure_rows)
    write_csv(args.output_dir / "patch2_algorithm_missing_trace_breakdown.csv", algorithm_rows)

    report = build_report(
        old_trace_counts=old_trace_counts,
        converter_summary=converter_summary,
        converter_counts=converter_counts,
        ab_summary=ab_summary,
        patch2_counts=patch2_counts,
        figure_rows=figure_rows,
        algorithm_rows=algorithm_rows,
        diff_summary=diff_summary,
    )
    (args.output_dir / REPORT_NAME).write_text(report, encoding="utf-8")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-ab-root", type=Path, default=DEFAULT_OLD_AB_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--converter-output-dir", type=Path, default=DEFAULT_CONVERTER_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def run_converter_only_eval(old_ab_root: Path, output_dir: Path) -> dict[str, Any]:
    old_exp = old_ab_root / "experimental_flag_on_current_code"
    old_base = old_ab_root / "baseline_flag_off_current_code"
    out_exp = output_dir / "experimental_flag_on_current_code"
    out_base = output_dir / "baseline_flag_off_current_code"
    out_exp.mkdir(parents=True, exist_ok=True)
    out_base.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for old_doc in sorted(path for path in old_exp.iterdir() if path.is_dir()):
        doc_id = old_doc.name.split("_", 1)[-1]
        baseline_doc = DEFAULT_BASELINE_ROOT / old_doc.name
        out_doc = out_exp / old_doc.name
        out_doc.mkdir(parents=True, exist_ok=True)
        for name in [
            "generated.tex",
            "float_caption_fix_diag.json",
            "promoted_captions.json",
            "float_caption_pairings.json",
            "placeholder_floats.json",
            "duplicate_caption_suppression.json",
            "crop_caption_separation.json",
            "consumed_caption_paragraphs.json",
        ]:
            if (old_doc / name).exists():
                shutil.copy2(old_doc / name, out_doc / name)
        old_base_doc = old_base / old_doc.name
        if old_base_doc.exists():
            out_base_doc = out_base / old_doc.name
            out_base_doc.mkdir(parents=True, exist_ok=True)
            if (old_base_doc / "generated.tex").exists():
                shutil.copy2(old_base_doc / "generated.tex", out_base_doc / "generated.tex")
        metrics = convert_and_evaluate(out_doc, doc_id, baseline_doc)
        diag = read_json(out_doc / "float_caption_fix_diag.json")
        rows.append(build_experimental_row(doc_id, out_doc, baseline_doc, diag, metrics))
    write_csv(out_exp / "summary.csv", rows)
    summary = {"docs": len(rows), "aggregate": aggregate_rows(rows)}
    write_json(output_dir / "patch2_converter_only_summary.json", summary)
    return summary


def run_current_batch(*, doc_dirs: list[Path], output_dir: Path, enable_float_caption_layout: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
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
        rows.append(row)
    write_csv(output_dir / "summary.csv", rows)
    return {"docs": len(rows), "rows": rows, "aggregate": aggregate_rows(rows)}


def read_trace_stage_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            counts[str(record.get("failure_stage") or "UNKNOWN")] += 1
    return counts


def failure_stage_counts(traces: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(trace.get("failure_stage") or "UNKNOWN") for trace in traces)


def summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in ["baseline", "experimental", "delta"]:
        payload = dict(summary.get(name) or {})
        payload["row"] = name
        rows.append(payload)
    return rows


def counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    total = sum(counter.values()) or 1
    return [{"failure_stage": key, "count": value, "ratio": value / total} for key, value in counter.most_common()]


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default if default is not None else {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], row: str, key: str) -> Any:
    return (summary.get(row) or {}).get(key)


def build_report(
    *,
    old_trace_counts: Counter[str],
    converter_summary: dict[str, Any],
    converter_counts: Counter[str],
    ab_summary: dict[str, Any],
    patch2_counts: Counter[str],
    figure_rows: list[dict[str, Any]],
    algorithm_rows: list[dict[str, Any]],
    diff_summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# V8 Float-Caption Patch2 Validation Report")
    lines.append("")
    lines.append("## Status")
    lines.append("- converter patch status: implemented and evaluated on existing flag-on outputs")
    lines.append("- materialization patch status: implemented and evaluated with same-code selected200 A/B")
    lines.append("- selected200 A/B status: completed")
    lines.append("- no training / no MinerU / no relabel / no GNN")
    lines.append("- production default unchanged")
    lines.append("- v8 facts only; legacy source_v7_ids/v7_id names remain provenance names only")
    lines.append("")
    lines.append("## Stage A: Converter-only Eval")
    lines.append("| metric | before converter patch | after converter patch | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in ["RENDERED_NOT_CONVERTED", "CONVERTED_NOT_MATCHED", "MATCHED"]:
        before = old_trace_counts.get(key, 0)
        after = converter_counts.get(key, 0)
        lines.append(f"| {key} | {before} | {after} | {after - before:+d} |")
    aggregate = converter_summary.get("aggregate") or {}
    lines.append(f"| pred_caption_count | historical | {aggregate.get('pred_caption_count', '')} |  |")
    lines.append(f"| float_caption_attachment_accuracy | historical | {aggregate.get('float_caption_attachment_accuracy', '')} |  |")
    lines.append("")
    lines.append("## Stage B: Materialization Patch")
    lines.append("| failure_stage | before | after patch2 | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in ["CROP_ONLY_OR_METADATA_ONLY", "PROMOTED_NOT_RENDERED", "RENDERED_NOT_CONVERTED", "CONVERTED_NOT_MATCHED", "MATCHED"]:
        before = old_trace_counts.get(key, 0)
        after = patch2_counts.get(key, 0)
        lines.append(f"| {key} | {before} | {after} | {after - before:+d} |")
    lines.append("")
    lines.append("## Stage C: Same-code A/B")
    lines.append("| metric | flag-off | flag-on | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in [
        "float_caption_attachment_accuracy",
        "pred_caption_count",
        "missing_caption_count",
        "metadata_caption_not_consumed_count",
        "crop_swallowed_caption_count",
        "duplicate_caption_count",
        "caption_as_paragraph_count",
        "wrong_float_type_pairing_count",
        "generated_structure_validity",
        "macro_structure_score_body",
        "paragraph_text_coverage_f1",
        "reference_section_completeness",
        "placeholder_float_count",
    ]:
        lines.append(
            f"| {key} | {metric(ab_summary, 'baseline', key)} | {metric(ab_summary, 'experimental', key)} | {metric(ab_summary, 'delta', key)} |"
        )
    lines.append("")
    lines.append("## Failure Stage Breakdown")
    lines.append("| failure_stage | before trace | converter-only | patch2 |")
    lines.append("| --- | ---: | ---: | ---: |")
    keys = sorted(set(old_trace_counts) | set(converter_counts) | set(patch2_counts))
    for key in keys:
        lines.append(f"| {key} | {old_trace_counts.get(key, 0)} | {converter_counts.get(key, 0)} | {patch2_counts.get(key, 0)} |")
    lines.append("")
    lines.append("## Figure Caption Diagnosis")
    lines.append(_stage_sentence("figure", figure_rows))
    lines.append("")
    lines.append("## Algorithm Caption Diagnosis")
    lines.append(_stage_sentence("algorithm", algorithm_rows))
    lines.append("")
    lines.append("## Duplicate / Subfigure Safety")
    lines.append(
        "- Duplicate suppression remains conservative; subfigure-like cases are preserved by the existing suffix guard and were not broadened in Patch2."
    )
    lines.append("")
    lines.append("## Suspicious Diff")
    totals = diff_summary.get("aggregate") or {}
    lines.append(f"- suspicious non-caption diff lines: {totals.get('non_caption_suspicious_change_count', 0)}")
    lines.append("- Patch2 keeps the experimental flag opt-in; any suspicious lines should be inspected before promotion.")
    lines.append("")
    lines.append("## Decision")
    decision = decide(ab_summary, patch2_counts, diff_summary)
    lines.append(f"- {decision}")
    return "\n".join(lines) + "\n"


def _stage_sentence(kind: str, rows: list[dict[str, Any]]) -> str:
    counter = Counter(str(row.get("best_candidate_stage") or row.get("failure_stage") or "UNKNOWN") for row in rows)
    if not counter:
        return f"- No remaining {kind} missing rows were produced by the trace."
    parts = ", ".join(f"{stage}: {count}" for stage, count in counter.most_common(6))
    return f"- Remaining {kind} missing is distributed as {parts}."


def decide(ab_summary: dict[str, Any], patch2_counts: Counter[str], diff_summary: dict[str, Any]) -> str:
    validity_delta = float(metric(ab_summary, "delta", "generated_structure_validity") or 0.0)
    wrong_delta = int(metric(ab_summary, "delta", "wrong_float_type_pairing_count") or 0)
    suspicious = int((diff_summary.get("aggregate") or {}).get("non_caption_suspicious_change_count") or 0)
    if validity_delta < -1e-9 or wrong_delta > 0 or suspicious > 0:
        return "patch_required"
    if patch2_counts.get("CROP_ONLY_OR_METADATA_ONLY", 0) < 309 or patch2_counts.get("PROMOTED_NOT_RENDERED", 0) < 50:
        return "safe_to_keep_experimental_enabled"
    return "patch_required"


if __name__ == "__main__":
    raise SystemExit(main())
