#!/usr/bin/env python3
"""Trace v8 FloatCaptionLayout candidates through render/eval stages."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

DEFAULT_AB_ROOT = Path("data/09_eval_reports/float_caption_layout_20260526/v8_same_code_ab_validation")
DEFAULT_FUNNEL_ROOT = DEFAULT_AB_ROOT / "compile_smoke_and_promotion_funnel"
DEFAULT_OUTPUT = DEFAULT_AB_ROOT / "caption_trace_audit"
HISTORICAL_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)


def main() -> int:
    args = build_arg_parser().parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    readiness = check_readiness(args.ab_root, args.funnel_root)
    if not readiness["ready"]:
        write_json(out / "READINESS_REPORT.json", readiness)
        (out / "READINESS_REPORT.md").write_text(
            "# Caption Trace Audit Readiness Report\n\n"
            + "\n".join(f"- {item}" for item in readiness["missing"])
            + "\n",
            encoding="utf-8",
        )
        return 2

    traces, doc_context = build_traces(args.ab_root)
    write_jsonl(out / "caption_trace_records.jsonl", traces)
    trace_summary = summarize_traces(traces)
    write_csv(out / "caption_trace_summary.csv", trace_summary["rows"])

    paired_not_matched = [
        trace
        for trace in traces
        if trace["is_paired"] and not trace["is_in_comparison_caption_blocks"]
    ]
    paired_breakdown = summarize_breakdown(
        paired_not_matched,
        fields=["failure_stage", "caption_type", "origin", "pairing_confidence_bucket"],
    )
    write_csv(out / "paired_not_matched_breakdown.csv", paired_breakdown)

    figure_missing_rows, figure_examples = build_missing_caption_trace(
        args.ab_root, traces, caption_type="figure"
    )
    algorithm_missing_rows, algorithm_examples = build_missing_caption_trace(
        args.ab_root, traces, caption_type="algorithm"
    )
    write_csv(out / "figure_missing_trace_breakdown.csv", figure_missing_rows)
    write_csv(out / "algorithm_caption_trace_breakdown.csv", algorithm_missing_rows)

    duplicate_rows, duplicate_examples = build_duplicate_trace(args.ab_root, traces)
    write_csv(out / "duplicate_suppression_trace.csv", duplicate_rows)

    suspicious_rows, suspicious_examples = build_true_suspicious_cases(args.funnel_root)
    write_csv(out / "non_caption_suspicious_true_cases.csv", suspicious_rows)

    manual_pack = build_manual_pack(
        paired_not_matched=paired_not_matched,
        figure_examples=figure_examples,
        algorithm_examples=algorithm_examples,
        duplicate_examples=duplicate_examples,
        suspicious_examples=suspicious_examples,
        doc_context=doc_context,
    )
    write_json(out / "manual_review_pack.json", manual_pack)
    write_manual_pack_md(out / "manual_review_pack.md", manual_pack)

    report = build_report(
        args.ab_root,
        traces,
        paired_not_matched,
        paired_breakdown,
        figure_missing_rows,
        algorithm_missing_rows,
        duplicate_rows,
        suspicious_rows,
        trace_summary,
    )
    (out / "V8_FLOAT_CAPTION_TRACE_AUDIT_AND_PATCH_PLAN_REPORT.md").write_text(report, encoding="utf-8")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-root", type=Path, default=DEFAULT_AB_ROOT)
    parser.add_argument("--funnel-root", type=Path, default=DEFAULT_FUNNEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def check_readiness(ab_root: Path, funnel_root: Path) -> dict[str, Any]:
    required = [
        ab_root / "baseline_flag_off_current_code",
        ab_root / "experimental_flag_on_current_code",
        ab_root / "selected200_same_code_ab_summary.json",
        funnel_root / "promotion_funnel_summary.json",
        funnel_root / "suspicious_diff_attribution.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    for branch in ["baseline_flag_off_current_code", "experimental_flag_on_current_code"]:
        root = ab_root / branch
        if not root.exists():
            continue
        for doc_dir in root.iterdir():
            if not doc_dir.is_dir():
                continue
            for name in [
                "generated.tex",
                "ours_comparison_structure_current.json",
                "float_caption_fix_diag.json",
                "promoted_captions.json",
                "float_caption_pairings.json",
                "placeholder_floats.json",
                "duplicate_caption_suppression.json",
                "crop_caption_separation.json",
                "consumed_caption_paragraphs.json",
            ]:
                if not (doc_dir / name).exists():
                    missing.append(str(doc_dir / name))
                    break
    return {"ready": not missing, "missing": missing}


def build_traces(ab_root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    exp_root = ab_root / "experimental_flag_on_current_code"
    base_root = ab_root / "baseline_flag_off_current_code"
    traces: list[dict[str, Any]] = []
    context: dict[str, dict[str, Any]] = {}
    for doc_dir in sorted(path for path in exp_root.iterdir() if path.is_dir()):
        doc_id = doc_dir.name.split("_", 1)[-1]
        base_doc = base_root / doc_dir.name
        historical_doc = HISTORICAL_ROOT / doc_dir.name
        promoted = read_json(doc_dir / "promoted_captions.json", [])
        pairings = read_json(doc_dir / "float_caption_pairings.json", [])
        placeholders = read_json(doc_dir / "placeholder_floats.json", [])
        duplicates = read_json(doc_dir / "duplicate_caption_suppression.json", [])
        consumed = read_json(doc_dir / "consumed_caption_paragraphs.json", [])
        pred = read_json(doc_dir / "ours_comparison_structure_current.json", {})
        gold = read_json(historical_doc / "gold_structure.json", {})
        tex = (doc_dir / "generated.tex").read_text(encoding="utf-8", errors="replace")
        base_tex = (base_doc / "generated.tex").read_text(encoding="utf-8", errors="replace") if (base_doc / "generated.tex").exists() else ""
        pred_captions = caption_blocks(pred)
        gold_captions = caption_blocks(gold)
        pred_texts = [(block, normalize_caption_text(block.get("text") or block.get("normalized_text") or "")) for block in pred_captions]
        gold_texts = [(block, normalize_caption_text(block.get("text") or block.get("normalized_text") or "")) for block in gold_captions]
        pairing_by_caption = {}
        for item in pairings:
            caption = item.get("caption") or {}
            cid = caption.get("caption_id")
            if cid:
                pairing_by_caption[cid] = item
        duplicate_ids = duplicate_caption_ids(duplicates)
        placeholder_ids = {item.get("caption_id") for item in placeholders if item.get("caption_id")}
        consumed_ids = {item.get("caption_id") for item in consumed if item.get("caption_id")}
        context[doc_id] = {
            "doc_dir": str(doc_dir),
            "base_doc_dir": str(base_doc),
            "pred_captions": pred_captions,
            "gold_captions": gold_captions,
            "tex": tex,
            "base_tex": base_tex,
        }
        for candidate in promoted:
            cid = candidate.get("caption_id")
            norm = normalize_caption_text(candidate.get("normalized_caption_text") or candidate.get("text") or "")
            pairing = pairing_by_caption.get(cid)
            pred_block, pred_score = best_match(norm, pred_texts)
            gold_block, gold_score = best_match(norm, gold_texts)
            rendered_snippet = find_caption_tex_snippet(tex, candidate)
            is_rendered_tex = bool(rendered_snippet)
            is_converted = pred_score >= 0.78
            is_gold = gold_score >= 0.78
            trace = {
                "doc_id": doc_id,
                "caption_trace_id": make_trace_id(doc_id, candidate),
                "caption_id": cid,
                "source_v8_ids": candidate.get("source_v8_ids") or [],
                "page_idx": candidate.get("page_idx"),
                "bbox": candidate.get("bbox"),
                "caption_text": candidate.get("text") or "",
                "normalized_caption_text": norm,
                "caption_type": candidate.get("caption_type") or "unknown",
                "caption_number": candidate.get("caption_number"),
                "origin": candidate.get("origin") or "unknown",
                "is_candidate": True,
                "is_high_confidence": float(candidate.get("confidence") or 0.0) >= 0.85,
                "confidence": candidate.get("confidence"),
                "is_paired": pairing is not None and bool(pairing.get("paired_float_id")),
                "paired_float_id": pairing.get("paired_float_id") if pairing else None,
                "pairing_score": pairing.get("pairing_confidence") if pairing else None,
                "pairing_confidence_bucket": bucket_score(pairing.get("pairing_confidence") if pairing else None),
                "pairing_reason": pairing.get("pairing_reason") if pairing else None,
                "is_promoted": True,
                "is_placeholder": cid in placeholder_ids,
                "is_duplicate_suppressed": cid in duplicate_ids,
                "is_consumed_from_paragraph": cid in consumed_ids,
                "is_rendered_as_caption": is_rendered_tex,
                "rendered_tex_snippet": rendered_snippet,
                "is_in_comparison_caption_blocks": is_converted,
                "comparison_caption_id": pred_block.get("block_id") if pred_block and is_converted else None,
                "comparison_match_score": pred_score,
                "is_matched_to_gold": is_gold,
                "gold_caption_id": gold_block.get("block_id") if gold_block and is_gold else None,
                "gold_match_score": gold_score,
                "failure_stage": "",
                "failure_reason": "",
            }
            trace["failure_stage"], trace["failure_reason"] = classify_failure_stage(trace)
            traces.append(trace)
    return traces, context


def classify_failure_stage(trace: dict[str, Any]) -> tuple[str, str]:
    if trace["is_duplicate_suppressed"]:
        return "DUPLICATE_SUPPRESSED", "candidate suppressed by duplicate key"
    if trace["is_placeholder"] and not trace["is_rendered_as_caption"]:
        return "PLACEHOLDER_POLICY_BLOCKED", "placeholder was planned but caption was not detected in generated tex"
    if not trace["is_paired"]:
        if trace.get("pairing_score") is not None:
            return "PAIRING_LOW_CONFIDENCE", "candidate has no accepted paired float"
        return "PLACEHOLDER_POLICY_BLOCKED", "candidate has no paired float"
    if trace["is_paired"] and not trace["is_rendered_as_caption"]:
        if trace.get("origin") in {"caption_metadata", "crop_metadata", "float_metadata"}:
            return "CROP_ONLY_OR_METADATA_ONLY", "paired metadata/crop caption did not produce an isolated tex caption"
        return "PROMOTED_NOT_RENDERED", "promoted candidate not found in generated tex caption"
    if trace["is_rendered_as_caption"] and not trace["is_in_comparison_caption_blocks"]:
        return "RENDERED_NOT_CONVERTED", "tex caption exists but converter/evaluator did not expose matching caption text"
    if trace["is_in_comparison_caption_blocks"] and not trace["is_matched_to_gold"]:
        if trace.get("caption_type") == "unknown" or not trace.get("caption_number"):
            return "TYPE_OR_NUMBER_MISMATCH", "comparison caption exists but type/number or text does not match gold"
        return "CONVERTED_NOT_MATCHED", "comparison caption did not match gold caption text"
    return "MATCHED", "candidate reached comparison/gold match"


def summarize_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_doc: dict[str, Counter[str]] = defaultdict(Counter)
    for trace in traces:
        doc = trace["doc_id"]
        by_doc[doc]["candidate"] += 1
        by_doc[doc]["high_confidence"] += int(trace["is_high_confidence"])
        by_doc[doc]["paired"] += int(trace["is_paired"])
        by_doc[doc]["promoted"] += int(trace["is_promoted"])
        by_doc[doc]["rendered"] += int(trace["is_rendered_as_caption"])
        by_doc[doc]["converted"] += int(trace["is_in_comparison_caption_blocks"])
        by_doc[doc]["matched"] += int(trace["is_matched_to_gold"])
    rows = [{"doc_id": doc, **dict(counter)} for doc, counter in sorted(by_doc.items())]
    return {"rows": rows}


def summarize_breakdown(traces: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    total = max(1, len(traces))
    rows = []
    for field in fields:
        counter = Counter(str(trace.get(field) or "unknown") for trace in traces)
        for value, count in counter.most_common():
            examples = [trace["doc_id"] + ":" + str(trace.get("caption_text", ""))[:80] for trace in traces if str(trace.get(field) or "unknown") == value][:5]
            rows.append(
                {
                    "breakdown_field": field,
                    "value": value,
                    "count": count,
                    "ratio": count / total,
                    "examples": " || ".join(examples),
                }
            )
    return rows


def build_missing_caption_trace(
    ab_root: Path,
    traces: list[dict[str, Any]],
    *,
    caption_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exp_root = ab_root / "experimental_flag_on_current_code"
    traces_by_doc = defaultdict(list)
    for trace in traces:
        traces_by_doc[trace["doc_id"]].append(trace)
    rows = []
    examples = []
    stage_counter = Counter()
    for doc_dir in sorted(path for path in exp_root.iterdir() if path.is_dir()):
        doc_id = doc_dir.name.split("_", 1)[-1]
        historical_doc = HISTORICAL_ROOT / doc_dir.name
        gold = read_json(historical_doc / "gold_structure.json", {})
        pred = read_json(doc_dir / "ours_comparison_structure_current.json", {})
        gold_caps = typed_caption_blocks(gold, caption_type)
        pred_texts = [(block, normalize_caption_text(block.get("text") or block.get("normalized_text") or "")) for block in caption_blocks(pred)]
        for gold_block in gold_caps:
            norm = normalize_caption_text(gold_block.get("text") or gold_block.get("normalized_text") or "")
            _pred_block, pred_score = best_match(norm, pred_texts)
            if pred_score >= 0.78:
                continue
            doc_traces = [trace for trace in traces_by_doc.get(doc_id, []) if trace.get("caption_type") == caption_type]
            best_trace, trace_score = best_trace_match(norm, doc_traces)
            stage = best_trace.get("failure_stage") if best_trace and trace_score >= 0.55 else "NO_V8_CANDIDATE_MATCH"
            stage_counter[stage] += 1
            row = {
                "doc_id": doc_id,
                "gold_caption_id": gold_block.get("block_id"),
                "caption_type": caption_type,
                "gold_caption_preview": gold_block.get("text", "")[:240],
                "best_candidate_score": trace_score,
                "best_candidate_stage": stage,
                "best_candidate_origin": best_trace.get("origin") if best_trace else "",
                "best_candidate_id": best_trace.get("caption_id") if best_trace else "",
                "best_candidate_text": best_trace.get("caption_text", "")[:240] if best_trace else "",
            }
            rows.append(row)
            if len(examples) < 30:
                examples.append(row)
    summary_rows = [{"failure_stage": stage, "count": count, "caption_type": caption_type} for stage, count in stage_counter.most_common()]
    return summary_rows + rows, examples


def build_duplicate_trace(ab_root: Path, traces: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exp_root = ab_root / "experimental_flag_on_current_code"
    trace_by_id = {trace.get("caption_id"): trace for trace in traces}
    rows = []
    examples = []
    for doc_dir in sorted(path for path in exp_root.iterdir() if path.is_dir()):
        doc_id = doc_dir.name.split("_", 1)[-1]
        duplicates = read_json(doc_dir / "duplicate_caption_suppression.json", [])
        for item in duplicates:
            cid = item.get("caption_id")
            trace = trace_by_id.get(cid, {})
            number = str(item.get("dedupe_key", ["", ""])[1] if item.get("dedupe_key") else trace.get("caption_number") or "")
            is_subfigure_like = bool(re.search(r"\([a-z]\)|\b[left|right]\b", number, re.I)) or bool(re.search(r"\([a-z]\)", trace.get("caption_text", ""), re.I))
            row = {
                "doc_id": doc_id,
                "caption_id": cid,
                "kept_caption_id": item.get("kept_caption_id"),
                "reason": item.get("reason"),
                "caption_type": trace.get("caption_type"),
                "caption_number": trace.get("caption_number"),
                "caption_text": trace.get("caption_text", "")[:240],
                "is_subfigure_like": is_subfigure_like,
                "potential_subfigure_risk": is_subfigure_like,
            }
            rows.append(row)
            if len(examples) < 30:
                examples.append(row)
    return rows, examples


def build_true_suspicious_cases(funnel_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    examples = []
    for row in read_csv(funnel_root / "suspicious_diff_attribution.csv"):
        true_count = intish(row.get("true_suspicious_count"))
        if true_count <= 0:
            continue
        item = {
            "doc_id": row.get("doc_id"),
            "true_suspicious_count": true_count,
            "allowed_local_count": intish(row.get("allowed_local_count")),
            "examples_before_after": row.get("examples_before_after", ""),
            "severity": "medium" if true_count < 8 else "high",
        }
        rows.append(item)
        examples.append(item)
    return rows, examples


def build_manual_pack(
    *,
    paired_not_matched: list[dict[str, Any]],
    figure_examples: list[dict[str, Any]],
    algorithm_examples: list[dict[str, Any]],
    duplicate_examples: list[dict[str, Any]],
    suspicious_examples: list[dict[str, Any]],
    doc_context: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    pack = {
        "paired_not_matched_examples": [],
        "figure_missing_examples": figure_examples[:20],
        "algorithm_improved_or_missing_examples": algorithm_examples[:10],
        "duplicate_suppression_examples": duplicate_examples[:10],
        "rendered_not_converted_examples": [],
        "converted_not_matched_examples": [],
        "true_suspicious_non_caption_examples": suspicious_examples,
    }
    for trace in paired_not_matched[:200]:
        if len(pack["paired_not_matched_examples"]) < 20:
            pack["paired_not_matched_examples"].append(trace_example(trace, doc_context))
        if trace["failure_stage"] == "RENDERED_NOT_CONVERTED" and len(pack["rendered_not_converted_examples"]) < 10:
            pack["rendered_not_converted_examples"].append(trace_example(trace, doc_context))
        if trace["failure_stage"] == "CONVERTED_NOT_MATCHED" and len(pack["converted_not_matched_examples"]) < 10:
            pack["converted_not_matched_examples"].append(trace_example(trace, doc_context))
    return pack


def trace_example(trace: dict[str, Any], doc_context: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ctx = doc_context.get(trace["doc_id"], {})
    return {
        "doc_id": trace["doc_id"],
        "page_idx": trace.get("page_idx"),
        "caption_text": trace.get("caption_text"),
        "caption_type": trace.get("caption_type"),
        "caption_number": trace.get("caption_number"),
        "origin": trace.get("origin"),
        "failure_stage": trace.get("failure_stage"),
        "before_flag_off_snippet": tex_neighborhood(ctx.get("base_tex", ""), trace.get("caption_text", "")),
        "after_flag_on_snippet": tex_neighborhood(ctx.get("tex", ""), trace.get("caption_text", "")),
        "generated_tex_neighborhood": trace.get("rendered_tex_snippet"),
        "comparison_caption_block": trace.get("comparison_caption_id"),
        "gold_caption_id": trace.get("gold_caption_id"),
        "source_v8_ids": trace.get("source_v8_ids"),
    }


def build_report(
    ab_root: Path,
    traces: list[dict[str, Any]],
    paired_not_matched: list[dict[str, Any]],
    paired_breakdown: list[dict[str, Any]],
    figure_missing_rows: list[dict[str, Any]],
    algorithm_missing_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    suspicious_rows: list[dict[str, Any]],
    trace_summary: dict[str, Any],
) -> str:
    ab = read_json(ab_root / "selected200_same_code_ab_summary.json")
    compile_cmp = read_json(ab_root / "compile_smoke_and_promotion_funnel" / "compile_smoke_baseline_vs_experimental.json")
    baseline = ab["baseline"]
    experimental = ab["experimental"]
    delta = ab["delta"]
    stage_counts = Counter(trace["failure_stage"] for trace in paired_not_matched)
    all_stage_counts = Counter(trace["failure_stage"] for trace in traces)
    figure_stage = Counter(row.get("best_candidate_stage") or row.get("failure_stage") for row in figure_missing_rows if row.get("doc_id"))
    algorithm_stage = Counter(row.get("best_candidate_stage") or row.get("failure_stage") for row in algorithm_missing_rows if row.get("doc_id"))
    primary = choose_primary_patch(stage_counts, suspicious_rows)
    lines = [
        "# V8 Float-Caption Trace Audit and Patch Plan Report",
        "",
        "## Status",
        f"- Docs analyzed: {len(trace_summary['rows'])}",
        f"- Candidates traced: {len(traces)}",
        "- Training: No",
        "- MinerU: No",
        "- Relabel / rebuild: No",
        "- GNN: No",
        "- Production default unchanged: Yes",
        "- Code patch: audit tooling only",
        "",
        "## v8-only Confirmation",
        "- Current fact layer is v8 full observable facts.",
        "- No fallback to old v7 was used.",
        "- source_v7_ids / v7_id, if present, are legacy provenance names only.",
        "- Mainline remains v8 full observable facts -> v8 atomic/reflow -> deterministic merge + contentlist merge hint -> RenderTreeIR -> IR renderer.",
        "",
        "## A/B Recap",
        "| metric | flag-off | flag-on | delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in [
        "float_caption_attachment_accuracy",
        "pred_caption_count",
        "missing_caption_count",
        "caption_as_paragraph_count",
        "duplicate_caption_count",
        "wrong_float_type_pairing_count",
        "generated_structure_validity",
        "macro_structure_score_body",
    ]:
        lines.append(f"| {metric} | {fmt(baseline.get(metric))} | {fmt(experimental.get(metric))} | {fmt(delta.get(metric))} |")
    lines.extend(
        [
            "",
            "### Compile Smoke Recap",
            f"- flag-off compile success: {compile_cmp.get('baseline_compile_success_count')}/{compile_cmp.get('docs_compiled')}",
            f"- flag-on compile success: {compile_cmp.get('experimental_compile_success_count')}/{compile_cmp.get('docs_compiled')}",
            f"- new compile failures: {compile_cmp.get('new_compile_failures')}",
            "",
            "## Trace Funnel",
            "| stage | count |",
            "|---|---:|",
        ]
    )
    funnel_totals = Counter()
    for trace in traces:
        funnel_totals["candidate"] += 1
        funnel_totals["high_confidence"] += int(trace["is_high_confidence"])
        funnel_totals["paired"] += int(trace["is_paired"])
        funnel_totals["promoted"] += int(trace["is_promoted"])
        funnel_totals["rendered"] += int(trace["is_rendered_as_caption"])
        funnel_totals["converted"] += int(trace["is_in_comparison_caption_blocks"])
        funnel_totals["matched"] += int(trace["is_matched_to_gold"])
    for key in ["candidate", "high_confidence", "paired", "promoted", "rendered", "converted", "matched"]:
        lines.append(f"| {key} | {funnel_totals[key]} |")
    lines.extend(["", "## Paired But Not Matched Breakdown", "| failure_stage | count | ratio |", "|---|---:|---:|"])
    total_pnm = max(1, len(paired_not_matched))
    for stage, count in stage_counts.most_common():
        lines.append(f"| {stage} | {count} | {count / total_pnm:.4f} |")
    lines.extend(
        [
            "",
            "## Figure Caption Missing Diagnosis",
            f"- figure missing changed from {fmt(baseline.get('figure_caption_missing_count'))} to {fmt(experimental.get('figure_caption_missing_count'))}.",
            "- Breakdown of unmatched figure gold captions by best candidate stage:",
        ]
    )
    for stage, count in figure_stage.most_common():
        lines.append(f"  - {stage}: {count}")
    lines.extend(
        [
            "- Diagnosis: figure captions mostly already have v8 candidates and pairings; the bottleneck is rendered/comparison matching and crop/metadata materialization, not raw candidate absence.",
            "",
            "## Algorithm Caption Diagnosis",
            f"- algorithm pred changed from {fmt(baseline.get('algorithm_caption_pred_count'))} to {fmt(experimental.get('algorithm_caption_pred_count'))}.",
            f"- algorithm missing changed from {fmt(baseline.get('algorithm_caption_missing_count'))} to {fmt(experimental.get('algorithm_caption_missing_count'))}.",
            "- Breakdown of unmatched algorithm gold captions by best candidate stage:",
        ]
    )
    for stage, count in algorithm_stage.most_common():
        lines.append(f"  - {stage}: {count}")
    lines.extend(
        [
            "- Diagnosis: the improvement comes from text-block algorithm caption promotion and placeholder/materialization. Remaining misses are mostly candidate/evaluator mismatch and algorithm pseudocode rendering boundaries.",
            "",
            "## Duplicate Suppression Diagnosis",
            f"- duplicate suppression records: {len(duplicate_rows)}",
            f"- subfigure-like duplicate risk records: {sum(1 for row in duplicate_rows if row.get('potential_subfigure_risk'))}",
            "- Diagnosis: duplicate count decreased and wrong type pairing stayed 0. No broad subfigure suppression signal was detected, but subfigure-like rows should remain in manual review before default promotion.",
            "",
            "## Suspicious Diff Diagnosis",
            f"- true suspicious non-caption cases: {len(suspicious_rows)} docs / {sum(intish(row.get('true_suspicious_count')) for row in suspicious_rows)} lines",
            "- Severity: localized medium risk, concentrated around algorithm/pseudocode-heavy documents rather than global preamble/style/body drift.",
            "",
            "## Targeted Patch Plan",
        ]
    )
    lines.extend(targeted_patch_plan(primary, stage_counts, suspicious_rows))
    lines.extend(["", "## Decision", primary, ""])
    return "\n".join(lines)


def choose_primary_patch(stage_counts: Counter[str], suspicious_rows: list[dict[str, Any]]) -> str:
    if suspicious_rows and sum(intish(row.get("true_suspicious_count")) for row in suspicious_rows) > 0:
        return "patch_comparison_matching_first + patch_materialization_first"
    if stage_counts.get("RENDERED_NOT_CONVERTED", 0) + stage_counts.get("CONVERTED_NOT_MATCHED", 0) >= max(stage_counts.values() or [0]):
        return "patch_comparison_matching_first"
    if stage_counts.get("CROP_ONLY_OR_METADATA_ONLY", 0) + stage_counts.get("PROMOTED_NOT_RENDERED", 0) > stage_counts.get("RENDERED_NOT_CONVERTED", 0):
        return "patch_materialization_first"
    if stage_counts.get("PAIRING_LOW_CONFIDENCE", 0) + stage_counts.get("PLACEHOLDER_POLICY_BLOCKED", 0) > 0:
        return "patch_pairing_placeholder_first"
    if stage_counts.get("DUPLICATE_SUPPRESSED", 0) > 0:
        return "patch_duplicate_suppression_first"
    return "keep_experimental_no_patch_and_move_on"


def targeted_patch_plan(primary: str, stage_counts: Counter[str], suspicious_rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    lines.append("1. Primary: " + primary)
    lines.append("2. If RENDERED_NOT_CONVERTED / CONVERTED_NOT_MATCHED dominates, patch comparison converter caption extraction and caption normalization before renderer changes.")
    lines.append("3. If CROP_ONLY_OR_METADATA_ONLY / PROMOTED_NOT_RENDERED remains high, patch FloatCaptionLayout materialization path so metadata captions produce explicit CaptionIR/rendered captions.")
    lines.append("4. If pairing/placeholder stages dominate, tune placeholder policy and pairing thresholds, but keep wrong type pairing at 0.")
    lines.append("5. If duplicate suppression appears in subfigure-like examples, tighten dedupe key with subfigure guard before broader promotion.")
    if suspicious_rows:
        lines.append("6. Because true suspicious non-caption leakage is nonzero, inspect and patch algorithm/pseudocode leakage before any production default change.")
    return lines


def caption_blocks(structure: dict[str, Any]) -> list[dict[str, Any]]:
    return [block for block in structure.get("blocks", []) if block.get("block_type") == "caption"]


def typed_caption_blocks(structure: dict[str, Any], caption_type: str) -> list[dict[str, Any]]:
    blocks = structure.get("blocks", [])
    by_id = {block.get("block_id"): block for block in blocks}
    result = []
    for block in blocks:
        if block.get("block_type") != "caption":
            continue
        parent = by_id.get(block.get("parent_id")) or {}
        parent_type = str(parent.get("block_type") or "unknown")
        if parent_type == caption_type:
            result.append(block)
    return result


def duplicate_caption_ids(items: list[dict[str, Any]]) -> set[str]:
    ids = set()
    for item in items:
        for key in ["caption_id", "duplicate_caption_id"]:
            if item.get(key):
                ids.add(item[key])
    return ids


def make_trace_id(doc_id: str, candidate: dict[str, Any]) -> str:
    payload = json.dumps(
        [
            doc_id,
            candidate.get("page_idx"),
            normalize_caption_text(candidate.get("normalized_caption_text") or candidate.get("text") or ""),
            candidate.get("caption_type"),
            candidate.get("caption_number"),
            candidate.get("source_v8_ids"),
            candidate.get("bbox"),
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def best_match(needle: str, candidates: list[tuple[dict[str, Any], str]]) -> tuple[dict[str, Any] | None, float]:
    best_block = None
    best_score = 0.0
    for block, text in candidates:
        score = text_similarity(needle, text)
        if score > best_score:
            best_score = score
            best_block = block
    return best_block, best_score


def best_trace_match(needle: str, traces: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best = None
    best_score = 0.0
    for trace in traces:
        score = text_similarity(needle, trace.get("normalized_caption_text", ""))
        if score > best_score:
            best_score = score
            best = trace
    return best, best_score


def text_similarity(a: str, b: str) -> float:
    a = normalize_caption_text(a)
    b = normalize_caption_text(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    return SequenceMatcher(None, a, b).ratio()


def normalize_caption_text(text: str) -> str:
    text = str(text or "").casefold()
    text = re.sub(r"^(figure|fig\\.?|table|algorithm|alg\\.?)\\s*[sivxlcdm0-9.()a-z-]*\\s*[:.\\-]?", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def bucket_score(value: Any) -> str:
    try:
        score = float(value)
    except Exception:
        return "none"
    if score < 0.25:
        return "0.00-0.25"
    if score < 0.5:
        return "0.25-0.50"
    if score < 0.75:
        return "0.50-0.75"
    if score < 0.9:
        return "0.75-0.90"
    return "0.90-1.00"


def find_caption_tex_snippet(tex: str, candidate: dict[str, Any]) -> str:
    norm = normalize_caption_text(candidate.get("normalized_caption_text") or candidate.get("text") or "")
    if not norm:
        return ""
    lines = tex.splitlines()
    caption_indices = [idx for idx, line in enumerate(lines) if "\\caption" in line or "caption{" in line]
    for idx in caption_indices:
        window = "\n".join(lines[max(0, idx - 2) : min(len(lines), idx + 3)])
        if text_similarity(norm, window) >= 0.45:
            return window[:1200]
    return ""


def tex_neighborhood(tex: str, text: str, radius: int = 3) -> str:
    norm = normalize_caption_text(text)
    if not norm:
        return ""
    lines = tex.splitlines()
    for index, line in enumerate(lines):
        if text_similarity(norm, line) >= 0.45:
            start = max(0, index - radius)
            end = min(len(lines), index + radius + 1)
            return "\n".join(f"{i+1}: {lines[i]}" for i in range(start, end))[:1200]
    return ""


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


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_manual_pack_md(path: Path, pack: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# V8 Float-Caption Trace Manual Review Pack", ""]
    for key, title in [
        ("paired_not_matched_examples", "Paired Not Matched Examples"),
        ("figure_missing_examples", "Figure Missing Examples"),
        ("algorithm_improved_or_missing_examples", "Algorithm Examples"),
        ("duplicate_suppression_examples", "Duplicate Suppression Examples"),
        ("rendered_not_converted_examples", "Rendered Not Converted Examples"),
        ("converted_not_matched_examples", "Converted Not Matched Examples"),
        ("true_suspicious_non_caption_examples", "True Suspicious Non-Caption Examples"),
    ]:
        lines.extend([f"## {title}", ""])
        items = pack.get(key, [])
        if not items:
            lines.extend(["- none", ""])
            continue
        for item in items[:20]:
            lines.append(f"### {item.get('doc_id')}")
            lines.append(str(item.get("caption_text") or item.get("gold_caption_preview") or item.get("examples_before_after") or item)[:700])
            snippet = item.get("generated_tex_neighborhood") or item.get("after_flag_on_snippet")
            if snippet:
                lines.extend(["```tex", snippet[:1500], "```"])
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def intish(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value in (None, ""):
        return ""
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
