#!/usr/bin/env python3
"""Patch3 validation for v8 FloatCaptionLayout.

Patch3 keeps the experimental branch opt-in.  It audits Patch2 duplicate
clusters, then reruns same-code selected200 flag-off/flag-on with canonical
caption selection and metadata/crop materialization wiring.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "tools" / "audit"
for path in (PROJECT_ROOT, AUDIT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_v8_float_caption_trace_patch_plan import build_missing_caption_trace, build_traces  # noqa: E402
from run_v8_float_caption_patch2_validation import run_current_batch  # noqa: E402
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


DEFAULT_PATCH2_ROOT = Path("data/09_eval_reports/float_caption_layout_20260526/v8_same_code_ab_validation/patch2_same_code_ab")
DEFAULT_OUTPUT_ROOT = Path("data/09_eval_reports/float_caption_layout_20260526/v8_same_code_ab_validation/patch3_same_code_ab")
REPORT_NAME = "V8_FLOAT_CAPTION_PATCH3_VALIDATION_REPORT.md"


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    duplicate_audit = audit_patch2_duplicate_clusters(args.patch2_root, args.output_dir)

    doc_dirs = collect_doc_dirs(args.baseline_root)
    selected_doc_dirs = list(doc_dirs.values())
    baseline = run_current_batch(
        doc_dirs=selected_doc_dirs,
        output_dir=args.output_dir / "baseline_flag_off_current_code",
        enable_float_caption_layout=False,
        workers=args.workers,
    )
    experimental = run_current_batch(
        doc_dirs=selected_doc_dirs,
        output_dir=args.output_dir / "experimental_flag_on_current_code",
        enable_float_caption_layout=True,
        workers=args.workers,
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

    traces, _context = build_traces(args.output_dir)
    patch3_counts = failure_stage_counts(traces)
    write_json(args.output_dir / "patch3_trace_summary.json", {"failure_stage_counts": dict(patch3_counts)})
    write_csv(args.output_dir / "patch3_failure_stage_breakdown.csv", counter_rows(patch3_counts))

    figure_rows, _figure_examples = build_missing_caption_trace(args.output_dir, traces, caption_type="figure")
    algorithm_rows, _algorithm_examples = build_missing_caption_trace(args.output_dir, traces, caption_type="algorithm")
    write_csv(args.output_dir / "patch3_figure_missing_trace_breakdown.csv", figure_rows)
    write_csv(args.output_dir / "patch3_algorithm_missing_trace_breakdown.csv", algorithm_rows)

    materialization_summary = collect_materialization_summary(args.output_dir / "experimental_flag_on_current_code")
    write_json(args.output_dir / "patch3_materialization_summary.json", materialization_summary)
    write_csv(args.output_dir / "patch3_materialization_summary.csv", [materialization_summary])

    patch2_summary = read_json(args.patch2_root / "selected200_same_code_ab_summary.json", {})
    patch2_counts = read_trace_stage_counts(args.patch2_root / "patch2_trace_summary.json")
    patch2_diff = read_json(args.patch2_root / "selected200_diff_attribution.json", {})
    report = build_report(
        duplicate_audit=duplicate_audit,
        patch2_summary=patch2_summary,
        patch2_counts=patch2_counts,
        patch2_diff=patch2_diff,
        ab_summary=ab_summary,
        patch3_counts=patch3_counts,
        figure_rows=figure_rows,
        algorithm_rows=algorithm_rows,
        diff_summary=diff_summary,
        materialization_summary=materialization_summary,
    )
    (args.output_dir / REPORT_NAME).write_text(report, encoding="utf-8")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--patch2-root", type=Path, default=DEFAULT_PATCH2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def audit_patch2_duplicate_clusters(patch2_root: Path, output_dir: Path) -> dict[str, Any]:
    exp_root = patch2_root / "experimental_flag_on_current_code"
    cluster_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    jsonl_path = output_dir / "duplicate_caption_clusters.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for doc_dir in sorted(path for path in exp_root.iterdir() if path.is_dir()):
            doc_id = doc_dir.name.split("_", 1)[-1]
            structure = read_json(doc_dir / "ours_comparison_structure_current.json", {})
            grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
            for block in structure.get("blocks", []):
                if block.get("block_type") != "caption":
                    continue
                kind = str(block.get("marker") or "")
                label = str(block.get("label") or "")
                text = normalize_text(block.get("normalized_text") or block.get("text") or "")
                grouped[(kind, label, text)].append(block)
            for (kind, label, text), members in grouped.items():
                if len(members) <= 1:
                    continue
                cluster_type = cluster_type_for(kind, label, text)
                row = {
                    "doc_id": doc_id,
                    "caption_type": kind,
                    "caption_number": label,
                    "normalized_caption_text": text,
                    "cluster_size": len(members),
                    "duplicate_count": len(members) - 1,
                    "cluster_type": cluster_type,
                    "block_ids": " ".join(str(member.get("block_id")) for member in members),
                }
                cluster_rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if cluster_type in {"panel_label", "subfigure_like"}:
                    risk_rows.append(row)
    write_csv(output_dir / "duplicate_caption_clusters.csv", cluster_rows)
    write_csv(output_dir / "subfigure_like_risk_review.csv", risk_rows)
    summary = {
        "cluster_count": len(cluster_rows),
        "duplicate_count": sum(int(row["duplicate_count"]) for row in cluster_rows),
        "cluster_type_counts": dict(Counter(row["cluster_type"] for row in cluster_rows)),
        "subfigure_like_risk_count": len(risk_rows),
    }
    write_json(output_dir / "duplicate_cluster_audit_summary.json", summary)
    return summary


def run_current_batch(
    *,
    doc_dirs: list[Path],
    output_dir: Path,
    enable_float_caption_layout: bool,
    workers: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        rows = [
            _run_one_doc((str(doc_dir), str(output_dir), enable_float_caption_layout))
            for doc_dir in doc_dirs
        ]
    else:
        tasks = [(str(doc_dir), str(output_dir), enable_float_caption_layout) for doc_dir in doc_dirs]
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_one_doc, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: str(row.get("doc_id") or ""))
    write_csv(output_dir / "summary.csv", rows)
    return {"docs": len(rows), "rows": rows, "aggregate": aggregate_rows(rows)}


def _run_one_doc(task: tuple[str, str, bool]) -> dict[str, Any]:
    doc_dir = Path(task[0])
    output_dir = Path(task[1])
    enable_float_caption_layout = task[2]
    doc_id = doc_dir.name.split("_", 1)[-1]
    out_doc_dir = output_dir / doc_dir.name
    rendered = render_doc(
        doc_dir,
        out_doc_dir,
        enable_float_caption_layout=enable_float_caption_layout,
        use_source_tex_for_parity=False,
    )
    metrics = convert_and_evaluate(out_doc_dir, doc_id, doc_dir)
    return build_experimental_row(doc_id, out_doc_dir, doc_dir, rendered["diag"], metrics)


def collect_materialization_summary(exp_root: Path) -> dict[str, Any]:
    totals = Counter()
    for doc_dir in sorted(path for path in exp_root.iterdir() if path.is_dir()):
        diag = read_json(doc_dir / "float_caption_fix_diag.json", {})
        promoted = diag.get("promoted_captions", []) or []
        placeholders = diag.get("placeholder_floats", []) or []
        suppressed = diag.get("noncanonical_suppressed_candidates", []) or []
        clusters = diag.get("canonical_caption_clusters", []) or []
        risk = diag.get("subfigure_like_risk_review", []) or []
        totals["promoted_caption_count"] += len(promoted)
        totals["metadata_caption_materialized_count"] += sum(1 for item in promoted if item.get("origin") in {"caption_metadata", "float_metadata"})
        totals["crop_caption_materialized_count"] += sum(1 for item in promoted if item.get("origin") == "crop_metadata")
        totals["placeholder_float_count"] += len(placeholders)
        totals["canonical_caption_count"] += len(clusters)
        totals["noncanonical_suppressed_count"] += len(suppressed)
        totals["subfigure_like_risk_count"] += len(risk)
        totals["subfigure_false_suppression_count"] += sum(1 for item in suppressed if item.get("caption_candidate_class") == "SUBFIGURE_CAPTION")
        totals["panel_label_count"] += sum(1 for item in suppressed if item.get("caption_candidate_class") == "PANEL_LABEL")
        totals["synthetic_fallback_caption_count"] += sum(1 for item in suppressed if item.get("caption_candidate_class") == "SYNTHETIC_FALLBACK_CAPTION")
        totals["body_reference_false_positive_blocked_count"] += sum(1 for item in suppressed if item.get("caption_candidate_class") == "BODY_REFERENCE_FALSE_POSITIVE")
    return dict(totals)


def read_trace_stage_counts(path: Path) -> Counter[str]:
    payload = read_json(path, {})
    return Counter(payload.get("failure_stage_counts") or {})


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


def normalize_text(text: Any) -> str:
    value = " ".join(str(text or "").casefold().split())
    value = re.sub(r"[^0-9a-z()]+", " ", value)
    return " ".join(value.split()).strip()


def cluster_type_for(kind: str, label: str, text: str) -> str:
    compact = re.sub(r"[^0-9a-z]+", "", str(text or "").casefold())
    if compact in {"a", "b", "c", "d", "e", "f", "left", "right", "figure", "fig", "table", "algorithm", "reconstructionplaceholder", "figurereconstructionplaceholder", "tablereconstructionplaceholder"}:
        return "panel_or_synthetic"
    if re.search(r"\([a-z0-9]+\)", str(label or ""), flags=re.IGNORECASE):
        return "subfigure_like"
    if not label:
        return "near_duplicate_no_number"
    return "true_duplicate"


def metric(summary: dict[str, Any], row: str, key: str) -> Any:
    return (summary.get(row) or {}).get(key)


def build_report(
    *,
    duplicate_audit: dict[str, Any],
    patch2_summary: dict[str, Any],
    patch2_counts: Counter[str],
    patch2_diff: dict[str, Any],
    ab_summary: dict[str, Any],
    patch3_counts: Counter[str],
    figure_rows: list[dict[str, Any]],
    algorithm_rows: list[dict[str, Any]],
    diff_summary: dict[str, Any],
    materialization_summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# V8 Float-Caption Patch3 Validation Report")
    lines.append("")
    lines.append("## Status")
    lines.append("- duplicate audit status: completed on Patch2 flag-on outputs")
    lines.append("- materialization patch status: canonical caption selection + metadata/crop wiring evaluated")
    lines.append("- same-code selected200 A/B status: completed")
    lines.append("- no training / no MinerU / no relabel / no rebuild / no GNN")
    lines.append("- production default unchanged; experimental flag remains opt-in")
    lines.append("- v8 full observable facts only; legacy source_v7_ids/v7_id names are provenance names only")
    lines.append("")
    lines.append("## Patch2 Recap")
    lines.append("| metric | Patch2 flag-off | Patch2 flag-on | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in ["float_caption_attachment_accuracy", "pred_caption_count", "missing_caption_count", "duplicate_caption_count", "caption_as_paragraph_count", "wrong_float_type_pairing_count", "generated_structure_validity", "macro_structure_score_body"]:
        lines.append(f"| {key} | {metric(patch2_summary, 'baseline', key)} | {metric(patch2_summary, 'experimental', key)} | {metric(patch2_summary, 'delta', key)} |")
    lines.append("")
    lines.append("## Stage A: Duplicate Cluster Audit")
    lines.append(f"- duplicate clusters: {duplicate_audit.get('cluster_count', 0)}")
    lines.append(f"- duplicate captions inside clusters: {duplicate_audit.get('duplicate_count', 0)}")
    lines.append(f"- cluster type counts: {duplicate_audit.get('cluster_type_counts', {})}")
    lines.append(f"- subfigure/panel risk rows: {duplicate_audit.get('subfigure_like_risk_count', 0)}")
    lines.append("")
    lines.append("## Stage B: Canonical Dedupe")
    lines.append("| metric | flag-off | flag-on Patch3 | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in ["duplicate_caption_count", "true_duplicate_caption_count", "duplicate_suppressed_count", "noncanonical_suppressed_count", "subfigure_false_suppression_count", "panel_label_count", "synthetic_fallback_caption_count", "canonical_caption_count"]:
        lines.append(f"| {key} | {metric(ab_summary, 'baseline', key)} | {metric(ab_summary, 'experimental', key)} | {metric(ab_summary, 'delta', key)} |")
    lines.append("")
    lines.append("## Stage C: Materialization Wiring")
    lines.append("| metric | Patch3 value |")
    lines.append("| --- | ---: |")
    for key in ["metadata_caption_materialized_count", "crop_caption_materialized_count", "promoted_caption_count", "canonical_caption_count", "placeholder_float_count", "subfigure_like_risk_count"]:
        lines.append(f"| {key} | {materialization_summary.get(key, 0)} |")
    lines.append("")
    lines.append("## Stage D: Same-code A/B")
    lines.append("| metric | flag-off | flag-on Patch3 | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in [
        "float_caption_attachment_accuracy",
        "pred_caption_count",
        "missing_caption_count",
        "duplicate_caption_count",
        "caption_as_paragraph_count",
        "wrong_float_type_pairing_count",
        "generated_structure_validity",
        "macro_structure_score_body",
        "paragraph_text_coverage_f1",
        "reference_section_completeness",
        "placeholder_float_count",
    ]:
        lines.append(f"| {key} | {metric(ab_summary, 'baseline', key)} | {metric(ab_summary, 'experimental', key)} | {metric(ab_summary, 'delta', key)} |")
    lines.append("")
    lines.append("## Failure Stage Breakdown")
    lines.append("| failure_stage | Patch2 | Patch3 | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in sorted(set(patch2_counts) | set(patch3_counts)):
        before = patch2_counts.get(key, 0)
        after = patch3_counts.get(key, 0)
        lines.append(f"| {key} | {before} | {after} | {after - before:+d} |")
    lines.append("")
    lines.append("## Figure Caption Diagnosis")
    lines.append(_stage_sentence("figure", figure_rows))
    lines.append("")
    lines.append("## Algorithm Caption Diagnosis")
    lines.append(_stage_sentence("algorithm", algorithm_rows))
    lines.append("- Remaining algorithm misses should be handled by a separate AlgorithmRegion pass if still dominated by NO_V8_CANDIDATE_MATCH.")
    lines.append("")
    lines.append("## Duplicate / Subfigure Safety")
    lines.append(f"- subfigure_false_suppression_count: {metric(ab_summary, 'experimental', 'subfigure_false_suppression_count')}")
    lines.append("- Panel-only and synthetic fallback captions are suppressed before materialization and kept in sidecar review.")
    lines.append("")
    lines.append("## Suspicious Diff")
    before_suspicious = (patch2_diff.get("aggregate") or {}).get("non_caption_suspicious_change_count", 0)
    after_suspicious = (diff_summary.get("aggregate") or {}).get("non_caption_suspicious_change_count", 0)
    lines.append(f"- true suspicious non-caption lines: Patch2 {before_suspicious} -> Patch3 {after_suspicious} ({after_suspicious - before_suspicious:+d})")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- {decide(ab_summary, diff_summary, patch2_summary, patch2_diff)}")
    return "\n".join(lines) + "\n"


def _stage_sentence(kind: str, rows: list[dict[str, Any]]) -> str:
    counter = Counter(str(row.get("best_candidate_stage") or row.get("failure_stage") or "UNKNOWN") for row in rows)
    if not counter:
        return f"- No remaining {kind} missing rows were produced by the trace."
    parts = ", ".join(f"{stage}: {count}" for stage, count in counter.most_common(6))
    return f"- Remaining {kind} missing is distributed as {parts}."


def decide(
    ab_summary: dict[str, Any],
    diff_summary: dict[str, Any],
    patch2_summary: dict[str, Any],
    patch2_diff: dict[str, Any],
) -> str:
    validity_delta = float(metric(ab_summary, "delta", "generated_structure_validity") or 0.0)
    wrong_delta = float(metric(ab_summary, "delta", "wrong_float_type_pairing_count") or 0.0)
    duplicate_on = float(metric(ab_summary, "experimental", "duplicate_caption_count") or 0.0)
    duplicate_off = float(metric(ab_summary, "baseline", "duplicate_caption_count") or 0.0)
    patch2_duplicate_on = float(metric(patch2_summary, "experimental", "duplicate_caption_count") or 10**9)
    subfigure_false = float(metric(ab_summary, "experimental", "subfigure_false_suppression_count") or 0.0)
    suspicious = int((diff_summary.get("aggregate") or {}).get("non_caption_suspicious_change_count") or 0)
    patch2_suspicious = int((patch2_diff.get("aggregate") or {}).get("non_caption_suspicious_change_count") or 10**9)
    if validity_delta < -1e-9 or wrong_delta > 0 or subfigure_false > 0:
        return "patch_required"
    if duplicate_on > duplicate_off or duplicate_on > patch2_duplicate_on:
        return "patch_required"
    if suspicious > patch2_suspicious:
        return "patch_required"
    return "safe_to_keep_experimental_enabled"


if __name__ == "__main__":
    raise SystemExit(main())
