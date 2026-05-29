#!/usr/bin/env python3
"""Aggregate visible-prose order failure causes for a generated-TeX variant.

This is a read-only diagnostic over existing paragraph audit JSON files.  It
does not rerun MinerU, generation, training, or E2E inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--group-prefix", required=True)
    parser.add_argument("--variant-name", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-docs", type=int, default=None)
    return parser


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def safe(value: Any) -> float:
    return as_float(value) or 0.0


def sort_value(value: Any) -> float:
    number = as_float(value)
    return number if number is not None else -1.0


def compact(text: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


def fmt(value: Any) -> str:
    number = as_float(value)
    return "N/A" if number is None else f"{number:.6f}"


def text_from_example(example: dict[str, Any]) -> str:
    parts: list[str] = []
    source = example.get("source")
    if isinstance(source, dict):
        parts.append(source.get("preview", ""))
    for key in ("left_source", "right_source"):
        value = example.get(key)
        if isinstance(value, dict):
            parts.append(value.get("preview", ""))
    for key in ("left_best_match", "right_best_match", "best_match"):
        match = example.get(key)
        if not isinstance(match, dict):
            continue
        for side in ("source", "generated"):
            value = match.get(side)
            if isinstance(value, dict):
                parts.append(value.get("preview", ""))
    return "\n".join(parts)


def classify_visible_failure(example: dict[str, Any]) -> str:
    text = text_from_example(example).lower()
    source_order = example.get("source_order") or []
    source_gap = None
    if isinstance(source_order, list) and len(source_order) == 2:
        try:
            source_gap = abs(int(source_order[1]) - int(source_order[0]))
        except Exception:
            source_gap = None

    if re.search(r"\b(?:figure|fig\.?|table|tab\.?|algorithm|caption|panel|subfigure)\b", text):
        return "float_caption_context"
    if (
        re.search(r"\b(?:equation|where|lemma|theorem|proof|corollary|proposition)\b", text)
        or sum(text.count(ch) for ch in "=<>^_{}[]()+*/|") >= 8
    ):
        return "math_or_theorem_context"
    if re.search(
        r"\b(?:references|proceedings|journal|conference|arxiv|doi|transactions|springer|ieee|acm|vol\.|pp\.)\b",
        text,
    ):
        return "reference_like_pollution"
    if re.search(
        r"@|\b(?:university|institute|department|laboratory|affiliation|abstract|keywords|corresponding author)\b",
        text,
    ):
        return "front_matter_or_abstract_pollution"
    if source_gap == 1:
        return "adjacent_ordinary_body_reorder"
    if source_gap is not None and source_gap <= 4:
        return "local_ordinary_body_reorder"
    return "long_range_or_matching_ambiguity"


def example_row(doc_id: str, example: dict[str, Any], cause: str) -> dict[str, Any]:
    source = example.get("source") if isinstance(example.get("source"), dict) else {}
    best_generated = ((example.get("best_match") or {}).get("generated") or {})
    left_source = example.get("left_source") if isinstance(example.get("left_source"), dict) else {}
    right_source = example.get("right_source") if isinstance(example.get("right_source"), dict) else {}
    left_generated = ((example.get("left_best_match") or {}).get("generated") or {})
    right_generated = ((example.get("right_best_match") or {}).get("generated") or {})
    return {
        "doc_id": doc_id,
        "cause": cause,
        "source_order": example.get("source_order") or example.get("source_index"),
        "generated_order": example.get("generated_order") or example.get("generated_index"),
        "left_source_preview": compact(left_source.get("preview") or source.get("preview")),
        "right_source_preview": compact(right_source.get("preview")),
        "left_generated_preview": compact(left_generated.get("preview") or best_generated.get("preview")),
        "right_generated_preview": compact(right_generated.get("preview")),
    }


def interpretation_for_displacement(displaced: float, inversion: float, adjacent: float, lis: float) -> str:
    if displaced > 0.5 and inversion < 0.02 and lis < 0.08:
        return "mostly_matching_set_or_rank_normalization"
    if displaced > 0.5 and adjacent > 0.1:
        return "real_local_reorder_plus_displacement"
    if displaced > 0.5:
        return "mixed_global_shift_or_matching_ambiguity"
    return "limited_displacement"


def load_rows(summary_json: Path, group_prefix: str, max_docs: int | None) -> list[dict[str, Any]]:
    rows_all = json.loads(summary_json.read_text(encoding="utf-8"))
    if not isinstance(rows_all, list):
        raise ValueError(f"Expected list summary JSON, got {type(rows_all).__name__}: {summary_json}")
    rows = [row for row in rows_all if str(row.get("group", "")).startswith(group_prefix)]
    rows.sort(key=lambda row: row.get("doc_id", ""))
    if max_docs is not None:
        rows = rows[:max_docs]
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.summary_json, args.group_prefix, args.max_docs)
    variant_name = args.variant_name or args.group_prefix.replace("/", "_")

    doc_rows: list[dict[str, Any]] = []
    sampled_examples: list[dict[str, Any]] = []
    case_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_channel_counts: Counter[str] = Counter()
    generated_channel_counts: Counter[str] = Counter()
    source_exclusion_counts: Counter[str] = Counter()

    for row in rows:
        doc_id = row["doc_id"]
        audit_path = Path(row["output_dir"]) / "paragraph_preservation_against_tex.json"
        audit: dict[str, Any] = {}
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        summary = audit.get("summary", {})

        visible_cov = as_float(row.get("visible_prose_source_coverage_rate"))
        visible_ordered = as_float(row.get("visible_prose_ordered_coverage_rate"))
        visible_inv = as_float(row.get("visible_prose_order_inversion_rate"))
        adjacent = as_float(row.get("adjacent_prose_inversion_rate"))
        displaced = as_float(row.get("displaced_prose_paragraph_rate_010"))
        lis = as_float(row.get("visible_prose_lis_disorder_rate"))
        coverage_loss = 1.0 - visible_cov if visible_cov is not None else None
        order_loss = max(visible_cov - visible_ordered, 0.0) if visible_cov is not None and visible_ordered is not None else None

        inv_examples = audit.get("visible_prose_order_inversion_examples", [])[:20]
        cause_counter = Counter(classify_visible_failure(example) for example in inv_examples)
        dominant = cause_counter.most_common(1)[0][0] if cause_counter else "none_sampled"

        doc_row = {
            "doc_id": doc_id,
            "visible_cov": visible_cov,
            "visible_ordered_cov": visible_ordered,
            "visible_order_loss": order_loss,
            "coverage_loss": coverage_loss,
            "visible_inv": visible_inv,
            "adjacent_inv": adjacent,
            "displaced_010": displaced,
            "lis_disorder": lis,
            "dominant_sampled_cause": dominant,
            "sampled_float_caption_context": cause_counter.get("float_caption_context", 0),
            "sampled_math_or_theorem_context": cause_counter.get("math_or_theorem_context", 0),
            "sampled_reference_like_pollution": cause_counter.get("reference_like_pollution", 0),
            "sampled_front_matter_or_abstract_pollution": cause_counter.get("front_matter_or_abstract_pollution", 0),
            "sampled_adjacent_ordinary_body_reorder": cause_counter.get("adjacent_ordinary_body_reorder", 0),
            "sampled_local_ordinary_body_reorder": cause_counter.get("local_ordinary_body_reorder", 0),
            "sampled_long_range_or_matching_ambiguity": cause_counter.get("long_range_or_matching_ambiguity", 0),
            "displaced_interpretation": interpretation_for_displacement(
                safe(displaced),
                safe(visible_inv),
                safe(adjacent),
                safe(lis),
            ),
            "audit_json": str(audit_path),
            "generated_tex": row.get("generated_tex"),
            "source_tex": row.get("source_tex"),
        }
        doc_rows.append(doc_row)

        source_channel_counts.update(summary.get("source_semantic_channel_counts", {}) or {})
        generated_channel_counts.update(summary.get("generated_semantic_channel_counts", {}) or {})
        source_exclusion_counts.update(summary.get("source_body_exclusion_reason_counts", {}) or {})

        for example in inv_examples[:5]:
            cause = classify_visible_failure(example)
            item = example_row(doc_id, example, cause)
            sampled_examples.append(item)
            if len(case_examples[cause]) < 8:
                case_examples[cause].append(item)

        for block in audit.get("generated_paragraphs", []):
            channel = block.get("semantic_channel")
            if channel not in {
                "caption",
                "front_matter",
                "abstract",
                "math_context",
                "display_math",
                "reference_item",
                "url_or_metadata",
                "front_note",
                "metadata",
            }:
                continue
            key = f"generated_channel:{channel}"
            if len(case_examples[key]) < 8:
                case_examples[key].append(
                    {
                        "doc_id": doc_id,
                        "channel": channel,
                        "line": block.get("line"),
                        "preview": compact(block.get("preview")),
                    }
                )

    if not doc_rows:
        raise ValueError(f"No rows matched group prefix: {args.group_prefix}")

    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in doc_rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    sampled_cause_counts: Counter[str] = Counter()
    for row in doc_rows:
        for key, value in row.items():
            if key.startswith("sampled_") and isinstance(value, int):
                sampled_cause_counts[key.replace("sampled_", "")] += value

    top_docs = {
        "visible_order_loss": sorted(doc_rows, key=lambda row: sort_value(row["visible_order_loss"]), reverse=True)[:25],
        "visible_inversion": sorted(doc_rows, key=lambda row: sort_value(row["visible_inv"]), reverse=True)[:25],
        "adjacent_inversion": sorted(doc_rows, key=lambda row: sort_value(row["adjacent_inv"]), reverse=True)[:25],
        "displaced_010": sorted(doc_rows, key=lambda row: sort_value(row["displaced_010"]), reverse=True)[:25],
        "coverage_loss": sorted(doc_rows, key=lambda row: sort_value(row["coverage_loss"]), reverse=True)[:25],
    }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "visible_prose_failure_breakdown_v1",
        "input_summary": str(args.summary_json),
        "group_prefix": args.group_prefix,
        "variant": variant_name,
        "docs": len(doc_rows),
        "aggregate": {
            "visible_prose_source_coverage_rate_mean": mean("visible_cov"),
            "visible_prose_ordered_coverage_rate_mean": mean("visible_ordered_cov"),
            "visible_order_loss_mean": mean("visible_order_loss"),
            "coverage_loss_mean": mean("coverage_loss"),
            "visible_prose_order_inversion_rate_mean": mean("visible_inv"),
            "adjacent_prose_inversion_rate_mean": mean("adjacent_inv"),
            "displaced_prose_paragraph_rate_010_mean": mean("displaced_010"),
            "visible_prose_lis_disorder_rate_mean": mean("lis_disorder"),
        },
        "sampled_inversion_cause_counts": dict(sampled_cause_counts.most_common()),
        "source_semantic_channel_counts": dict(source_channel_counts.most_common()),
        "generated_semantic_channel_counts": dict(generated_channel_counts.most_common()),
        "source_body_exclusion_reason_counts": dict(source_exclusion_counts.most_common()),
        "top_docs": top_docs,
        "case_examples": case_examples,
    }

    write_outputs(args.output_dir, payload, doc_rows, sampled_examples)
    return payload


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    doc_rows: list[dict[str, Any]],
    sampled_examples: list[dict[str, Any]],
) -> None:
    (output_dir / "visible_prose_failure_breakdown_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "doc_failure_breakdown.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(doc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(doc_rows)
    with (output_dir / "sampled_visible_inversion_examples.jsonl").open("w", encoding="utf-8") as file:
        for row in sampled_examples:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "top_failure_docs.json").write_text(
        json.dumps(payload["top_docs"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "VISIBLE_PROSE_FAILURE_BREAKDOWN_REPORT.md").write_text(
        render_report(payload),
        encoding="utf-8",
    )


def render_report(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    top_docs = payload["top_docs"]
    case_examples = payload["case_examples"]
    lines: list[str] = [
        "# v8+hint Visible Prose Failure Breakdown",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- variant: `{payload['variant']}`",
        f"- docs: `{payload['docs']}`",
        f"- input: `{payload['input_summary']}`",
        "- no MinerU / generation / training rerun: Yes",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in aggregate.items():
        lines.append(f"| `{key}` | {fmt(value)} |")

    lines += [
        "",
        "## Where visible_ordered_cov is lost",
        "",
        f"- Mean visible prose coverage loss: `{fmt(aggregate['coverage_loss_mean'])}`. This is visible prose not confidently covered/matched.",
        f"- Mean order penalty after coverage: `{fmt(aggregate['visible_order_loss_mean'])}`. This is the part directly attributable to matched prose order inversions.",
        "- Interpretation: the larger loss is still coverage/matching, while order inversions are concentrated in a smaller set of documents.",
        "",
        "## Sampled inversion cause counts",
        "",
        "These counts are over saved per-doc examples, not every possible pairwise inversion. They are for triage, not exact metric denominators.",
        "",
        "| sampled cause | count |",
        "| --- | ---: |",
    ]
    for key, value in payload["sampled_inversion_cause_counts"].items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Top visible inversion docs",
        "",
        "| doc_id | visible_inv | adjacent_inv | displaced_010 | lis_disorder | dominant sampled cause |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in top_docs["visible_inversion"][:12]:
        lines.append(
            f"| `{row['doc_id']}` | {fmt(row['visible_inv'])} | {fmt(row['adjacent_inv'])} | "
            f"{fmt(row['displaced_010'])} | {fmt(row['lis_disorder'])} | `{row['dominant_sampled_cause']}` |"
        )

    lines += [
        "",
        "## Top adjacent inversion docs",
        "",
        "| doc_id | adjacent_inv | visible_inv | displaced_010 | lis_disorder | dominant sampled cause |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in top_docs["adjacent_inversion"][:12]:
        lines.append(
            f"| `{row['doc_id']}` | {fmt(row['adjacent_inv'])} | {fmt(row['visible_inv'])} | "
            f"{fmt(row['displaced_010'])} | {fmt(row['lis_disorder'])} | `{row['dominant_sampled_cause']}` |"
        )

    lines += [
        "",
        "## Common-Matched Displacement Audit",
        "",
        "| doc_id | displaced_010 | visible_inv | adjacent_inv | lis_disorder | interpretation |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in top_docs["displaced_010"][:12]:
        lines.append(
            f"| `{row['doc_id']}` | {fmt(row['displaced_010'])} | {fmt(row['visible_inv'])} | "
            f"{fmt(row['adjacent_inv'])} | {fmt(row['lis_disorder'])} | `{row['displaced_interpretation']}` |"
        )

    lines += [
        "",
        "## Pollution Tracks",
        "",
        "Generated paragraph channel counts sampled from the current audit JSONs:",
        "",
        "| generated semantic channel | count |",
        "| --- | ---: |",
    ]
    for key, value in payload["generated_semantic_channel_counts"].items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "Source exclusion reasons from body metric filtering:",
        "",
        "| source exclusion reason | count |",
        "| --- | ---: |",
    ]
    for key, value in payload["source_body_exclusion_reason_counts"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.append("")
    lines.append("## Representative Examples")
    for cause, examples in case_examples.items():
        if not examples:
            continue
        lines.append("")
        lines.append(f"### `{cause}`")
        for example in examples[:4]:
            if "left_source_preview" in example:
                lines.append(
                    f"- `{example['doc_id']}` source {example.get('source_order')} -> generated "
                    f"{example.get('generated_order')}: {example['left_source_preview']} || {example['right_source_preview']}"
                )
            else:
                lines.append(f"- `{example.get('doc_id')}` L{example.get('line')}: {example.get('preview')}")

    lines += [
        "",
        "## Diagnosis",
        "",
        "1. `visible_ordered_cov` is still dominated by coverage/matching loss, not only order inversions.",
        "2. The highest visible inversion docs are concentrated and should become v8 reflow case studies before any model work is reopened.",
        "3. `displaced_010` is too high to read alone: many docs combine genuine global shift with normalized-rank/matching-set effects. Use LIS and adjacent inversion together.",
        "4. Residual examples split into ordinary body reflow, math/theorem context, float/caption context, and long-range matching ambiguity. These point to v8 reflow, Formula/ParagraphContextGroup, and FloatCaptionLayout, not GNN merge.",
        "",
        "## Recommended Next Fixes",
        "",
        "1. Inspect top inversion docs with page screenshots and v8 block order.",
        "2. Add a Formula/ParagraphContextGroup track for theorem/proof/where/equation-adjacent prose instead of treating it as ordinary prose displacement.",
        "3. Continue hardening FrontMatterExtractor and FloatCaptionLayout only where generated channel examples show leakage.",
        "4. Keep GNN/learned merge archived unless a future oracle shows residual candidates can improve visible prose order without increasing wrong merges.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = run(args)
    print(Path(args.output_dir) / "VISIBLE_PROSE_FAILURE_BREAKDOWN_REPORT.md")
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
