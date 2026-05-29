#!/usr/bin/env python3
"""Posthoc analysis for v8 learned MERGE branches.

This tool is read-only with respect to model/generator outputs.  It consumes
existing paragraph-preservation audit JSONs and produces:

* document-level deltas against deterministic v8;
* wrong-merge category counts/examples;
* an evaluated-variant oracle over already materialized outputs.

The oracle is intentionally named "evaluated-variant" because it does not build
new edge-level oracle outputs.  It only asks: if we could choose the best
existing branch per document under safety constraints, how much headroom is
visible in the outputs we already generated?
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FOCUS_RUNS = {
    "conservative_overlay_no_owner_scope_20260525/residual_target_overlay_selected200_t090_gap4_v2_cap3",
    "conservative_overlay_no_owner_scope_20260525/residual_target_overlay_selected200_t090_gap4_v2_cap5",
    "conservative_overlay_no_owner_scope_20260525/projection_selected200_B_no_owner_scope_t090_gap4_v2_cap10",
    "residual_ranker_overlay_selected200_20260525/hp_cap1_b099_l097",
    "residual_ranker_overlay_selected200_20260525/hp_cap2_b098_l096",
    "residual_ranker_overlay_selected200_20260525/balanced_cap3_b097_l095",
    "residual_ranker_overlay_selected200_20260525/recall_cap5_b095_l093",
    "merge_selector_veto_selected200_20260525/keep_relaxed_add_strict",
}

HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+){0,3}\s+)?"
    r"(?:abstract|introduction|related work|method|methods|approach|experiment|experiments|"
    r"results|discussion|conclusion|references|appendix)\b",
    re.IGNORECASE,
)
CAPTION_RE = re.compile(r"^(?:fig(?:ure)?\.?|table|tab\.?|algorithm|alg\.?)\s*(?:[A-Z]?\d+|[IVX]+)?[:.)\s]", re.I)
REFERENCE_RE = re.compile(r"(?:\[[0-9,\-\s]+\]|\b[A-Z][A-Za-z-]+ et al\.?[, ]+\d{4}|\(\s*[A-Z][A-Za-z-]+.*?\d{4}.*?\))")
FORMULA_RE = re.compile(r"(?:\\(?:frac|sum|prod|mathbf|mathcal|alpha|beta|gamma|theta|lambda)|[=<>≤≥≈∑∏]|[_^]{1,2}|\bwhere\b.*\bdenotes\b)", re.I)
LIST_RE = re.compile(r"^(?:[-*•]|\(?[0-9]+[.)]|\(?[a-zA-Z][.)])\s+")
TABLE_CODE_RE = re.compile(
    r"(?:\\hline|\\toprule|\\midrule|\\bottomrule|&\s*|;.*;|node distance|/\.style|"
    r"\btransformer block\b|\bdense\s*\+\s*relu\b|\binput:\b|\boutput:\b|\bfor each\b|\bend for\b)",
    re.I,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--focus-run", action="append", dest="focus_runs")
    parser.add_argument("--max-examples-per-category", type=int, default=12)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    focus_runs = set(args.focus_runs or DEFAULT_FOCUS_RUNS)

    audits = load_audits(args.root)
    document_rows, document_summary_rows = build_document_deltas(audits, focus_runs)
    wrong_rows, wrong_delta_rows, wrong_examples = build_wrong_merge_categories(
        audits, focus_runs, args.max_examples_per_category
    )
    oracle_rows, oracle_summary = build_evaluated_variant_oracle(audits, focus_runs)

    write_csv(args.output_dir / "document_deltas.csv", document_rows)
    write_json(args.output_dir / "document_deltas.json", document_rows)
    write_csv(args.output_dir / "document_delta_summary.csv", document_summary_rows)
    write_json(args.output_dir / "document_delta_summary.json", document_summary_rows)
    write_csv(args.output_dir / "wrong_merge_category_summary.csv", wrong_rows)
    write_json(args.output_dir / "wrong_merge_category_summary.json", wrong_rows)
    write_csv(args.output_dir / "wrong_merge_category_delta_vs_deterministic.csv", wrong_delta_rows)
    write_json(args.output_dir / "wrong_merge_category_delta_vs_deterministic.json", wrong_delta_rows)
    write_json(args.output_dir / "wrong_merge_category_examples.json", wrong_examples)
    write_csv(args.output_dir / "evaluated_variant_oracle_summary.csv", oracle_rows)
    write_json(args.output_dir / "evaluated_variant_oracle_summary.json", oracle_rows)
    write_json(args.output_dir / "evaluated_variant_oracle_detail.json", oracle_summary)

    report = render_report(
        root=args.root,
        focus_runs=focus_runs,
        document_summary_rows=document_summary_rows,
        wrong_rows=wrong_rows,
        wrong_delta_rows=wrong_delta_rows,
        oracle_rows=oracle_rows,
        oracle_summary=oracle_summary,
    )
    (args.output_dir / "MERGE_BRANCH_POSTHOC_ANALYSIS_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote {args.output_dir / 'MERGE_BRANCH_POSTHOC_ANALYSIS_REPORT.md'}")
    return 0


def load_audits(root: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    audits: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(root.glob("**/paragraph_audit/paragraph_preservation_against_tex.json")):
        if "source_coverage_v2_refresh_20260525" in str(path):
            continue
        try:
            run = str(path.parents[3].relative_to(root))
            doc_id = path.parents[2].name
            variant = path.parents[1].name
            audits[run][doc_id][variant] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return audits


def build_document_deltas(
    audits: dict[str, dict[str, dict[str, dict[str, Any]]]],
    focus_runs: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for run, docs in sorted(audits.items()):
        if run not in focus_runs:
            continue
        for doc_id, variants in sorted(docs.items()):
            det = variants.get("deterministic")
            if not det:
                continue
            det_m = metrics(det)
            for variant, payload in sorted(variants.items()):
                if variant == "deterministic":
                    continue
                var_m = metrics(payload)
                row = {
                    "run": run,
                    "doc_id": doc_id,
                    "variant": variant,
                    **prefixed("det", det_m),
                    **prefixed("variant", var_m),
                    "delta_body_source_coverage": delta(var_m["body_source_coverage"], det_m["body_source_coverage"]),
                    "delta_body_ordered_source_coverage": delta(
                        var_m["body_ordered_source_coverage"], det_m["body_ordered_source_coverage"]
                    ),
                    "delta_body_order_inversion_rate": delta(
                        var_m["body_order_inversion_rate"], det_m["body_order_inversion_rate"]
                    ),
                    "delta_body_missing_merge_rate": delta(var_m["body_missing_merge_rate"], det_m["body_missing_merge_rate"]),
                    "delta_body_wrong_merge_rate": delta(var_m["body_wrong_merge_rate"], det_m["body_wrong_merge_rate"]),
                    "delta_body_paragraph_delta": delta(var_m["body_paragraph_delta"], det_m["body_paragraph_delta"]),
                }
                row.update(classify_doc_delta(row))
                rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    by_run_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run_variant[(row["run"], row["variant"])].append(row)
    for (run, variant), group in sorted(by_run_variant.items()):
        summary_rows.append(
            {
                "run": run,
                "variant": variant,
                "docs": len(group),
                "coverage_up": sum(1 for r in group if gt0(r["delta_body_source_coverage"])),
                "coverage_down": sum(1 for r in group if lt0(r["delta_body_source_coverage"])),
                "missing_down": sum(1 for r in group if lt0(r["delta_body_missing_merge_rate"])),
                "missing_up": sum(1 for r in group if gt0(r["delta_body_missing_merge_rate"])),
                "wrong_down": sum(1 for r in group if lt0(r["delta_body_wrong_merge_rate"])),
                "wrong_up": sum(1 for r in group if gt0(r["delta_body_wrong_merge_rate"])),
                "safe_improved_docs": sum(1 for r in group if r["safe_improved"]),
                "pure_harm_docs": sum(1 for r in group if r["pure_harm"]),
                "coverage_up_wrong_not_up": sum(1 for r in group if r["coverage_up_wrong_not_up"]),
                "mean_delta_body_source_coverage": mean(r["delta_body_source_coverage"] for r in group),
                "mean_delta_body_ordered_source_coverage": mean(r["delta_body_ordered_source_coverage"] for r in group),
                "mean_delta_body_order_inversion_rate": mean(r["delta_body_order_inversion_rate"] for r in group),
                "mean_delta_body_missing_merge_rate": mean(r["delta_body_missing_merge_rate"] for r in group),
                "mean_delta_body_wrong_merge_rate": mean(r["delta_body_wrong_merge_rate"] for r in group),
                "mean_delta_body_paragraph_delta": mean(r["delta_body_paragraph_delta"] for r in group),
            }
        )
    return rows, summary_rows


def build_wrong_merge_categories(
    audits: dict[str, dict[str, dict[str, dict[str, Any]]]],
    focus_runs: set[str],
    max_examples_per_category: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run, docs in sorted(audits.items()):
        if run not in focus_runs:
            continue
        for doc_id, variants in sorted(docs.items()):
            for variant, payload in sorted(variants.items()):
                for ex in payload.get("body_wrong_merge_examples") or payload.get("wrong_merge_examples") or []:
                    category = categorize_wrong_merge(ex)
                    counts[(run, variant, category)] += 1
                    key = f"{run}/{variant}/{category}"
                    if len(examples[key]) < max_examples_per_category:
                        examples[key].append(
                            {
                                "doc_id": doc_id,
                                "generated_line": (ex.get("generated") or {}).get("line"),
                                "generated_preview": (ex.get("generated") or {}).get("preview"),
                                "source_part_previews": [
                                    (part.get("source") or {}).get("preview")
                                    for part in (ex.get("source_parts") or [])[:4]
                                ],
                                "category": category,
                            }
                        )
    rows = [
        {"run": run, "variant": variant, "category": category, "count": count}
        for (run, variant, category), count in sorted(counts.items())
    ]
    delta_rows: list[dict[str, Any]] = []
    run_variants: dict[str, set[str]] = defaultdict(set)
    run_categories: dict[str, set[str]] = defaultdict(set)
    for run, variant, category in counts:
        run_variants[run].add(variant)
        run_categories[run].add(category)
    for run in sorted(run_variants):
        for variant in sorted(run_variants[run]):
            if variant == "deterministic":
                continue
            for category in sorted(run_categories[run]):
                det_count = counts.get((run, "deterministic", category), 0)
                variant_count = counts.get((run, variant, category), 0)
                delta_rows.append(
                    {
                        "run": run,
                        "variant": variant,
                        "category": category,
                        "deterministic_count": det_count,
                        "variant_count": variant_count,
                        "delta_count": variant_count - det_count,
                    }
                )
    return rows, delta_rows, dict(examples)


def build_evaluated_variant_oracle(
    audits: dict[str, dict[str, dict[str, dict[str, Any]]]],
    focus_runs: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {"schema_version": "evaluated_variant_oracle_v1", "runs": {}}
    for run, docs in sorted(audits.items()):
        if run not in focus_runs:
            continue
        chosen_docs: list[dict[str, Any]] = []
        det_metrics: list[dict[str, Any]] = []
        oracle_metrics: list[dict[str, Any]] = []
        for doc_id, variants in sorted(docs.items()):
            det_payload = variants.get("deterministic")
            if not det_payload:
                continue
            det = metrics(det_payload)
            candidates = [("deterministic", det)]
            for variant, payload in variants.items():
                if variant != "deterministic":
                    candidates.append((variant, metrics(payload)))
            safe = [
                (name, m)
                for name, m in candidates
                if leq(m["body_wrong_merge_rate"], det["body_wrong_merge_rate"])
                and geq(m["body_source_coverage"], det["body_source_coverage"])
            ]
            if not safe:
                safe = [("deterministic", det)]
            best_name, best = min(
                safe,
                key=lambda item: (
                    none_high(item[1]["body_missing_merge_rate"]),
                    abs_none_high(item[1]["body_paragraph_delta"]),
                    -none_low(item[1]["body_source_coverage"]),
                ),
            )
            chosen_docs.append(
                {
                    "doc_id": doc_id,
                    "chosen_variant": best_name,
                    "det": det,
                    "oracle": best,
                    "changed": best_name != "deterministic",
                }
            )
            det_metrics.append(det)
            oracle_metrics.append(best)
        row = {
            "run": run,
            "docs": len(chosen_docs),
            "oracle_changed_docs": sum(1 for item in chosen_docs if item["changed"]),
            "det_body_source_coverage": mean(m["body_source_coverage"] for m in det_metrics),
            "oracle_body_source_coverage": mean(m["body_source_coverage"] for m in oracle_metrics),
            "delta_body_source_coverage": delta(
                mean(m["body_source_coverage"] for m in oracle_metrics),
                mean(m["body_source_coverage"] for m in det_metrics),
            ),
            "det_body_ordered_source_coverage": mean(m["body_ordered_source_coverage"] for m in det_metrics),
            "oracle_body_ordered_source_coverage": mean(m["body_ordered_source_coverage"] for m in oracle_metrics),
            "delta_body_ordered_source_coverage": delta(
                mean(m["body_ordered_source_coverage"] for m in oracle_metrics),
                mean(m["body_ordered_source_coverage"] for m in det_metrics),
            ),
            "det_body_order_inversion_rate": mean(m["body_order_inversion_rate"] for m in det_metrics),
            "oracle_body_order_inversion_rate": mean(m["body_order_inversion_rate"] for m in oracle_metrics),
            "delta_body_order_inversion_rate": delta(
                mean(m["body_order_inversion_rate"] for m in oracle_metrics),
                mean(m["body_order_inversion_rate"] for m in det_metrics),
            ),
            "det_body_missing_merge_rate": mean(m["body_missing_merge_rate"] for m in det_metrics),
            "oracle_body_missing_merge_rate": mean(m["body_missing_merge_rate"] for m in oracle_metrics),
            "delta_body_missing_merge_rate": delta(
                mean(m["body_missing_merge_rate"] for m in oracle_metrics),
                mean(m["body_missing_merge_rate"] for m in det_metrics),
            ),
            "det_body_wrong_merge_rate": mean(m["body_wrong_merge_rate"] for m in det_metrics),
            "oracle_body_wrong_merge_rate": mean(m["body_wrong_merge_rate"] for m in oracle_metrics),
            "delta_body_wrong_merge_rate": delta(
                mean(m["body_wrong_merge_rate"] for m in oracle_metrics),
                mean(m["body_wrong_merge_rate"] for m in det_metrics),
            ),
            "det_body_paragraph_delta": mean(m["body_paragraph_delta"] for m in det_metrics),
            "oracle_body_paragraph_delta": mean(m["body_paragraph_delta"] for m in oracle_metrics),
            "delta_body_paragraph_delta": delta(
                mean(m["body_paragraph_delta"] for m in oracle_metrics),
                mean(m["body_paragraph_delta"] for m in det_metrics),
            ),
        }
        rows.append(row)
        detail["runs"][run] = {"summary": row, "chosen_docs": chosen_docs}
    return rows, detail


def metrics(payload: dict[str, Any]) -> dict[str, Any]:
    s = payload.get("summary") or {}
    return {
        "raw_source_coverage": s.get("source_coverage_rate_raw", s.get("source_coverage_rate")),
        "body_source_coverage": s.get("body_source_coverage_rate"),
        "body_ordered_source_coverage": s.get("body_ordered_source_coverage_rate"),
        "body_order_inversion_rate": s.get("body_source_order_inversion_rate"),
        "body_order_kendall_tau": s.get("body_source_order_kendall_tau"),
        "body_missing_merge_rate": s.get("body_missing_merge_rate_among_covered"),
        "body_wrong_merge_rate": s.get("body_wrong_merge_rate_among_generated"),
        "body_paragraph_delta": s.get("body_paragraph_count_delta"),
    }


def categorize_wrong_merge(example: dict[str, Any]) -> str:
    generated = str((example.get("generated") or {}).get("preview") or "")
    source_parts = " ".join(str((part.get("source") or {}).get("preview") or "") for part in example.get("source_parts") or [])
    text = f"{generated} {source_parts}".strip()
    if CAPTION_RE.search(text):
        return "caption_or_float"
    if HEADING_RE.search(text):
        return "heading_boundary"
    if TABLE_CODE_RE.search(text):
        return "table_code_algorithm"
    if FORMULA_RE.search(text):
        return "formula_or_math_context"
    if REFERENCE_RE.search(text):
        return "citation_reference_context"
    if LIST_RE.search(text):
        return "list_boundary"
    return "body_body_or_unknown"


def classify_doc_delta(row: dict[str, Any]) -> dict[str, bool]:
    coverage_delta = row.get("delta_body_source_coverage")
    missing_delta = row.get("delta_body_missing_merge_rate")
    wrong_delta = row.get("delta_body_wrong_merge_rate")
    return {
        "coverage_up_wrong_not_up": gt0(coverage_delta) and not gt0(wrong_delta),
        "safe_improved": gt0(coverage_delta) and lt0(missing_delta) and not gt0(wrong_delta),
        "pure_harm": not gt0(coverage_delta) and gt0(wrong_delta),
    }


def prefixed(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


def delta(new: Any, old: Any) -> float | None:
    if new is None or old is None:
        return None
    return round(float(new) - float(old), 6)


def mean(values: Any) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def gt0(value: Any, eps: float = 1e-12) -> bool:
    return value is not None and float(value) > eps


def lt0(value: Any, eps: float = 1e-12) -> bool:
    return value is not None and float(value) < -eps


def leq(left: Any, right: Any, eps: float = 1e-12) -> bool:
    return left is not None and right is not None and float(left) <= float(right) + eps


def geq(left: Any, right: Any, eps: float = 1e-12) -> bool:
    return left is not None and right is not None and float(left) >= float(right) - eps


def none_high(value: Any) -> float:
    return float(value) if value is not None else 1e9


def none_low(value: Any) -> float:
    return float(value) if value is not None else -1e9


def abs_none_high(value: Any) -> float:
    return abs(float(value)) if value is not None else 1e9


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(
    *,
    root: Path,
    focus_runs: set[str],
    document_summary_rows: list[dict[str, Any]],
    wrong_rows: list[dict[str, Any]],
    wrong_delta_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    oracle_summary: dict[str, Any],
) -> str:
    lines = [
        "# MERGE Branch Posthoc Analysis",
        "",
        "## Status",
        "",
        f"- created_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- root: `{root}`",
        f"- focus runs: {len(focus_runs)}",
        "- scope: read-only posthoc analysis over refreshed paragraph audit JSONs.",
        "- no model training, no generator rerun, no MinerU rerun.",
        "",
        "## Document-Level Delta Summary",
        "",
        "| run | variant | docs | coverage up | wrong up | safe improved | pure harm | mean Δcov | mean Δordered cov | mean Δorder inv | mean Δmissing | mean Δwrong |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in document_summary_rows:
        lines.append(
            f"| `{row['run']}` | `{row['variant']}` | {row['docs']} | {row['coverage_up']} | "
            f"{row['wrong_up']} | {row['safe_improved_docs']} | {row['pure_harm_docs']} | "
            f"{row['mean_delta_body_source_coverage']} | {row.get('mean_delta_body_ordered_source_coverage')} | "
            f"{row.get('mean_delta_body_order_inversion_rate')} | {row['mean_delta_body_missing_merge_rate']} | "
            f"{row['mean_delta_body_wrong_merge_rate']} |"
        )
    lines += [
        "",
        "## Wrong-Merge Category Summary",
        "",
        "| run | variant | category | count |",
        "| --- | --- | --- | ---: |",
    ]
    non_det_wrong_rows = [row for row in wrong_rows if row["variant"] != "deterministic"]
    for row in non_det_wrong_rows[:80]:
        lines.append(f"| `{row['run']}` | `{row['variant']}` | `{row['category']}` | {row['count']} |")
    if len(non_det_wrong_rows) > 80:
        lines.append(f"| ... | ... | ... | {len(non_det_wrong_rows) - 80} more rows |")
    lines += [
        "",
        "## Wrong-Merge Category Delta vs Deterministic",
        "",
        "| run | variant | category | det count | variant count | delta |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(wrong_delta_rows, key=lambda r: (r["run"], r["variant"], -r["delta_count"], r["category"]))[:80]:
        lines.append(
            f"| `{row['run']}` | `{row['variant']}` | `{row['category']}` | "
            f"{row['deterministic_count']} | {row['variant_count']} | {row['delta_count']} |"
        )
    lines += [
        "",
        "## Evaluated-Variant Oracle",
        "",
        "This is not an edge-level oracle. It only chooses the best already-generated variant per document under `wrong <= deterministic` and `coverage >= deterministic`.",
        "",
        "| run | docs | changed docs | Δcoverage | Δordered coverage | Δorder inversion | Δmissing | Δwrong | Δparagraph delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in oracle_rows:
        lines.append(
            f"| `{row['run']}` | {row['docs']} | {row['oracle_changed_docs']} | "
            f"{row['delta_body_source_coverage']} | {row.get('delta_body_ordered_source_coverage')} | "
            f"{row.get('delta_body_order_inversion_rate')} | {row['delta_body_missing_merge_rate']} | "
            f"{row['delta_body_wrong_merge_rate']} | {row['delta_body_paragraph_delta']} |"
        )
    best = best_oracle_row(oracle_rows)
    lines += ["", "## Interpretation", ""]
    if best:
        lines.append(
            f"- Best evaluated-variant oracle by missing reduction under safety: `{best['run']}` "
            f"with Δmissing={best['delta_body_missing_merge_rate']}, "
            f"Δwrong={best['delta_body_wrong_merge_rate']}, "
            f"changed_docs={best['oracle_changed_docs']}."
        )
    lines += [
        "- If the evaluated oracle barely changes documents, the current materialized learned branches have limited safe headroom.",
        "- If wrong-merge categories concentrate in formula/caption/table/code contexts, learned MERGE needs stronger risk gating rather than lower thresholds.",
        "- A true edge-level oracle would require applying gold-safe residual edge choices and re-rendering; this report does not claim that stronger oracle.",
        "",
        "## Artifacts",
        "",
        "- `document_deltas.csv/json`",
        "- `document_delta_summary.csv/json`",
        "- `wrong_merge_category_summary.csv/json`",
        "- `wrong_merge_category_delta_vs_deterministic.csv/json`",
        "- `wrong_merge_category_examples.json`",
        "- `evaluated_variant_oracle_summary.csv/json`",
        "- `evaluated_variant_oracle_detail.json`",
        "",
    ]
    return "\n".join(lines)


def best_oracle_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    viable = [row for row in rows if row.get("delta_body_missing_merge_rate") is not None]
    if not viable:
        return None
    return min(viable, key=lambda row: (row["delta_body_missing_merge_rate"], -row["oracle_changed_docs"]))


if __name__ == "__main__":
    raise SystemExit(main())
