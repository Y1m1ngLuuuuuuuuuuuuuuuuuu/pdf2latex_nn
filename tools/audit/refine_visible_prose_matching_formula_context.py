#!/usr/bin/env python3
"""Refine visible-prose matching and audit formula/theorem context.

This pass is deliberately read-only over existing generated TeX and source TeX.
It does not modify production v8, renderer output, MinerU artifacts, labels, or
graphs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.check_paragraph_preservation_against_tex import (  # noqa: E402
    ParagraphBlock,
    analyze_preservation,
    channels_can_match,
    extract_paragraph_blocks,
    is_visible_prose_channel,
    pair_scores,
    reindex_blocks,
    split_body_source_blocks,
)
from src.reasoning.formula_context_group import (  # noqa: E402
    classify_formula_context as classify_formula_context_record_text,
    should_exclude_from_ordinary_visible_prose_evidence,
)


CONTEXT_EXCLUSION_FAMILIES = {
    "DISPLAY_MATH_CONTEXT",
    "THEOREM_PROOF_CONTEXT",
    "WHERE_CLAUSE_CONTEXT",
    "FORMULA_OCR_ARTIFACT",
}

PARAGRAPH_CONTEXT_GROUP_FAMILIES = {
    "DISPLAY_MATH_CONTEXT",
    "THEOREM_PROOF_CONTEXT",
    "WHERE_CLAUSE_CONTEXT",
}


@dataclass
class MatchConfig:
    min_tokens: int = 8
    min_common_tokens: int = 5
    candidate_source_recall: float = 0.12
    candidate_generated_precision: float = 0.30
    covered_source_recall: float = 0.55
    covered_combined_recall: float = 0.65
    split_combined_recall: float = 0.60
    split_best_source_recall_max: float = 0.85
    overmerge_generated_coverage: float = 0.60
    include_list_items: bool = False
    max_examples: int = 20


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breakdown-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-ids", nargs="*", default=None)
    return parser


def compact(text: Any, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


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


def block_payload(block: ParagraphBlock, family: str | None = None) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "index": block.index,
        "line": block.line,
        "semantic_channel": block.semantic_channel,
        "context_family": family or classify_formula_context(block),
        "token_count": len(block.tokens),
        "preview": compact(block.text),
    }


def classify_formula_context_with_evidence(
    block: ParagraphBlock,
    *,
    local_formula_context: bool = False,
) -> tuple[str, Any]:
    family, evidence = classify_formula_context_record_text(
        block.text,
        raw_text=block.raw_text,
        semantic_channel=block.semantic_channel,
        local_formula_context=local_formula_context,
    )
    return family, evidence


def classify_formula_context(block: ParagraphBlock) -> str:
    family, _evidence = classify_formula_context_with_evidence(block)
    return family


def is_formula_context_anchor(block: ParagraphBlock) -> bool:
    family, evidence = classify_formula_context_with_evidence(block)
    return family in {"DISPLAY_MATH_CONTEXT", "FORMULA_OCR_ARTIFACT"} or evidence.display_math_env


def classify_block_contexts(blocks: list[ParagraphBlock]) -> dict[str, tuple[str, Any]]:
    mapping: dict[str, tuple[str, Any]] = {}
    for idx, block in enumerate(blocks):
        local = False
        for neighbor_idx in (idx - 1, idx + 1):
            if 0 <= neighbor_idx < len(blocks) and is_formula_context_anchor(blocks[neighbor_idx]):
                local = True
                break
        mapping[block.block_id] = classify_formula_context_with_evidence(block, local_formula_context=local)
    return mapping


def exclude_context_tuple(value: tuple[str, Any] | None) -> bool:
    if value is None:
        return False
    family, evidence = value
    if evidence is None:
        return False
    return should_exclude_from_ordinary_visible_prose_evidence(family, evidence)


def family_reason(family: str) -> str:
    if family == "DISPLAY_MATH_CONTEXT":
        return "display equation or equation-adjacent prose should be evaluated as paragraph context, not ordinary prose order"
    if family == "THEOREM_PROOF_CONTEXT":
        return "theorem/proof-like blocks often interleave display math and should use a structured context group"
    if family == "WHERE_CLAUSE_CONTEXT":
        return "where-clause prose belongs to equation context rather than ordinary paragraph order"
    if family == "INLINE_MATH_ATTACHMENT":
        return "short inline math fragment should be attached as paragraph span, not standalone prose"
    if family == "FORMULA_OCR_ARTIFACT":
        return "formula-like OCR residue should be suppressed or aligned weakly"
    return "ordinary prose residual reorder candidate"


def read_breakdown_rows(breakdown_dir: Path, doc_ids: set[str] | None, limit: int | None) -> list[dict[str, str]]:
    path = breakdown_dir / "doc_failure_breakdown.csv"
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if doc_ids:
        rows = [row for row in rows if row.get("doc_id") in doc_ids]
    rows.sort(key=lambda row: row.get("doc_id", ""))
    if limit is not None:
        rows = rows[:limit]
    return rows


def namespace_from_config(config: MatchConfig) -> argparse.Namespace:
    return argparse.Namespace(**config.__dict__)


def candidate_scores_for(
    source_blocks: list[ParagraphBlock],
    generated_blocks: list[ParagraphBlock],
    config: MatchConfig,
    *,
    type_aware: bool,
) -> list[Any]:
    scores = pair_scores(
        source_blocks,
        generated_blocks,
        min_common_tokens=config.min_common_tokens,
        candidate_source_recall=config.candidate_source_recall,
        candidate_generated_precision=config.candidate_generated_precision,
        type_aware=type_aware,
    )
    return [
        score
        for score in scores
        if score.common_tokens >= config.min_common_tokens
        and (
            score.source_recall >= config.candidate_source_recall
            or score.generated_precision >= config.candidate_generated_precision
        )
    ]


def refined_visible_analysis(
    source_tex: Path,
    generated_tex: Path,
    config: MatchConfig,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_text = source_tex.read_text(encoding="utf-8", errors="ignore")
    generated_text = generated_tex.read_text(encoding="utf-8", errors="ignore")
    source_blocks = extract_paragraph_blocks(
        source_text,
        source="source",
        min_tokens=config.min_tokens,
        include_list_items=config.include_list_items,
    )
    generated_blocks = extract_paragraph_blocks(
        generated_text,
        source="generated",
        min_tokens=config.min_tokens,
        include_list_items=config.include_list_items,
    )
    body_source_blocks, _excluded = split_body_source_blocks(source_blocks)

    source_context = classify_block_contexts(body_source_blocks)
    generated_context = classify_block_contexts(generated_blocks)

    refined_source_blocks = reindex_blocks(
        [
            block
            for block in body_source_blocks
            if is_visible_prose_channel(block.semantic_channel)
            and not exclude_context_tuple(source_context.get(block.block_id))
        ]
    )
    refined_generated_blocks = reindex_blocks(
        [
            block
            for block in generated_blocks
            if is_visible_prose_channel(block.semantic_channel)
            and not exclude_context_tuple(generated_context.get(block.block_id))
        ]
    )
    refined_scores = candidate_scores_for(refined_source_blocks, refined_generated_blocks, config, type_aware=True)
    refined = analyze_preservation(refined_source_blocks, refined_generated_blocks, refined_scores, namespace_from_config(config))

    pollution_cases = matching_pollution_cases(body_source_blocks, generated_blocks, config)
    context_cases = collect_context_cases(body_source_blocks, generated_blocks, source_context, generated_context)
    paragraph_context_candidates = collect_paragraph_context_candidates(
        body_source_blocks,
        generated_blocks,
        source_context,
        generated_context,
    )
    return refined, {
        "raw_source_count": len(source_blocks),
        "body_source_count": len(body_source_blocks),
        "refined_source_count": len(refined_source_blocks),
        "generated_count": len(generated_blocks),
        "refined_generated_count": len(refined_generated_blocks),
        "source_context_counts": dict(Counter(family for family, _ in source_context.values())),
        "generated_context_counts": dict(Counter(family for family, _ in generated_context.values())),
        "source_confidence_tier_counts": dict(Counter(evidence.confidence_tier for _, evidence in source_context.values())),
        "generated_confidence_tier_counts": dict(Counter(evidence.confidence_tier for _, evidence in generated_context.values())),
    }, pollution_cases, context_cases, paragraph_context_candidates


def matching_pollution_cases(
    source_blocks: list[ParagraphBlock],
    generated_blocks: list[ParagraphBlock],
    config: MatchConfig,
) -> list[dict[str, Any]]:
    scores = candidate_scores_for(source_blocks, generated_blocks, config, type_aware=False)
    by_source: dict[str, list[Any]] = {block.block_id: [] for block in source_blocks}
    source_by_id = {block.block_id: block for block in source_blocks}
    generated_by_id = {block.block_id: block for block in generated_blocks}
    for score in scores:
        by_source.setdefault(score.source_id, []).append(score)
    cases: list[dict[str, Any]] = []
    for score_list in by_source.values():
        if not score_list:
            continue
        score_list.sort(key=lambda score: (-score.source_recall, -score.generated_precision))
        score = score_list[0]
        if score.source_recall < config.covered_source_recall and score.generated_precision < 0.65:
            continue
        source = source_by_id[score.source_id]
        generated = generated_by_id[score.generated_id]
        if channels_can_match(source.semantic_channel, generated.semantic_channel):
            continue
        cases.append(
            {
                "source": block_payload(source),
                "generated": block_payload(generated),
                "source_recall": score.source_recall,
                "generated_precision": score.generated_precision,
                "common_tokens": score.common_tokens,
                "pollution_type": f"{source.semantic_channel}->{generated.semantic_channel}",
            }
        )
    return cases


def collect_context_cases(
    source_blocks: list[ParagraphBlock],
    generated_blocks: list[ParagraphBlock],
    source_context: dict[str, tuple[str, Any]],
    generated_context: dict[str, tuple[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for origin, blocks, mapping in (
        ("source", source_blocks, source_context),
        ("generated", generated_blocks, generated_context),
    ):
        for block in blocks:
            family, evidence = mapping.get(block.block_id, ("ORDINARY_BODY_REORDER", None))
            if family == "ORDINARY_BODY_REORDER":
                continue
            cases.append(
                {
                    "origin": origin,
                    "family": family,
                    "confidence": getattr(evidence, "confidence_tier", ""),
                    "reason": family_reason(family),
                    **block_payload(block, family),
                }
            )
    return cases


def collect_paragraph_context_candidates(
    source_blocks: list[ParagraphBlock],
    generated_blocks: list[ParagraphBlock],
    source_context: dict[str, tuple[str, Any]],
    generated_context: dict[str, tuple[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for origin, blocks, mapping in (
        ("source", source_blocks, source_context),
        ("generated", generated_blocks, generated_context),
    ):
        for idx, block in enumerate(blocks):
            family, evidence = mapping.get(block.block_id, ("ORDINARY_BODY_REORDER", None))
            if family not in PARAGRAPH_CONTEXT_GROUP_FAMILIES:
                continue
            if getattr(evidence, "confidence_tier", "low") != "high":
                continue
            before = blocks[idx - 1] if idx > 0 else None
            after = blocks[idx + 1] if idx + 1 < len(blocks) else None
            candidates.append(
                {
                    "origin": origin,
                    "family": family,
                    "confidence": getattr(evidence, "confidence_tier", ""),
                    "block_id": block.block_id,
                    "line": block.line,
                    "suggested_group": "text_before / display_math_or_context / text_after",
                    "reason": family_reason(family),
                    "text_before_preview": compact(before.text if before else ""),
                    "context_preview": compact(block.text),
                    "text_after_preview": compact(after.text if after else ""),
                }
            )
    return candidates


def row_for_doc(
    base_row: dict[str, str],
    refined: dict[str, Any],
    counts: dict[str, Any],
    pollution_cases: list[dict[str, Any]],
    context_cases: list[dict[str, Any]],
    paragraph_context_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = refined["summary"]
    return {
        "doc_id": base_row["doc_id"],
        "old_visible_cov": as_float(base_row.get("visible_cov")),
        "old_visible_ordered_cov": as_float(base_row.get("visible_ordered_cov")),
        "old_visible_inv": as_float(base_row.get("visible_inv")),
        "old_adjacent_inv": as_float(base_row.get("adjacent_inv")),
        "old_displaced_010": as_float(base_row.get("displaced_010")),
        "old_lis_disorder": as_float(base_row.get("lis_disorder")),
        "refined_visible_cov": summary.get("source_coverage_rate"),
        "refined_visible_ordered_cov": summary.get("ordered_source_coverage_rate"),
        "refined_visible_inv": summary.get("source_order_inversion_rate"),
        "refined_adjacent_inv": summary.get("source_order_adjacent_inversion_rate"),
        "refined_displaced_010": summary.get("source_order_displaced_rate_010"),
        "refined_lis_disorder": summary.get("source_order_lis_disorder_rate"),
        "matching_pollution_count": len(pollution_cases),
        "math_theorem_context_count": len(context_cases),
        "paragraph_context_group_candidate_count": len(paragraph_context_candidates),
        "ordinary_body_residual_reorder_count": len(refined.get("source_order_inversion_examples", [])),
        "refined_source_count": counts["refined_source_count"],
        "refined_generated_count": counts["refined_generated_count"],
        "source_context_counts": json.dumps(counts["source_context_counts"], sort_keys=True),
        "generated_context_counts": json.dumps(counts["generated_context_counts"], sort_keys=True),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_breakdown_rows(args.breakdown_dir, set(args.doc_ids) if args.doc_ids else None, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = MatchConfig()

    metric_rows: list[dict[str, Any]] = []
    pollution_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    pcg_rows: list[dict[str, Any]] = []
    ordinary_reorder_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for row in rows:
        doc_id = row["doc_id"]
        try:
            refined, counts, pollution_cases, context_cases, paragraph_context_candidates = refined_visible_analysis(
                Path(row["source_tex"]),
                Path(row["generated_tex"]),
                config,
            )
        except Exception as exc:
            errors.append({"doc_id": doc_id, "error": repr(exc)})
            continue
        metric_rows.append(
            row_for_doc(row, refined, counts, pollution_cases, context_cases, paragraph_context_candidates)
        )
        for case in pollution_cases:
            pollution_rows.append({"doc_id": doc_id, **flatten_case(case)})
        for case in context_cases:
            context_rows.append({"doc_id": doc_id, **case})
        for case in paragraph_context_candidates:
            pcg_rows.append({"doc_id": doc_id, **case})
        for case in refined.get("source_order_inversion_examples", [])[:10]:
            ordinary_reorder_rows.append(
                {
                    "doc_id": doc_id,
                    "source_order": case.get("source_order"),
                    "generated_order": case.get("generated_order"),
                    "left_source_preview": compact((case.get("left_source") or {}).get("preview")),
                    "right_source_preview": compact((case.get("right_source") or {}).get("preview")),
                    "left_generated_preview": compact(((case.get("left_best_match") or {}).get("generated") or {}).get("preview")),
                    "right_generated_preview": compact(((case.get("right_best_match") or {}).get("generated") or {}).get("preview")),
                }
            )

    write_csv(args.output_dir / "refined_visible_prose_metrics.csv", metric_rows)
    write_csv(args.output_dir / "math_theorem_context_cases.csv", context_rows)
    write_csv(args.output_dir / "paragraph_context_group_candidates.csv", pcg_rows)
    write_csv(args.output_dir / "matching_pollution_cases.csv", pollution_rows)
    write_csv(args.output_dir / "ordinary_body_residual_reorder_examples.csv", ordinary_reorder_rows)

    payload = summarize(metric_rows, context_rows, pollution_rows, pcg_rows, ordinary_reorder_rows, errors, args)
    (args.output_dir / "refined_visible_prose_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "VISIBLE_PROSE_MATCHING_REFINEMENT_AND_FORMULA_CONTEXT_AUDIT_REPORT.md").write_text(
        render_report(payload, pcg_rows, ordinary_reorder_rows),
        encoding="utf-8",
    )
    return payload


def flatten_case(case: dict[str, Any]) -> dict[str, Any]:
    source = case.get("source", {})
    generated = case.get("generated", {})
    return {
        "pollution_type": case.get("pollution_type"),
        "source_recall": case.get("source_recall"),
        "generated_precision": case.get("generated_precision"),
        "common_tokens": case.get("common_tokens"),
        "source_block_id": source.get("block_id"),
        "source_channel": source.get("semantic_channel"),
        "source_family": source.get("context_family"),
        "source_line": source.get("line"),
        "source_preview": source.get("preview"),
        "generated_block_id": generated.get("block_id"),
        "generated_channel": generated.get("semantic_channel"),
        "generated_family": generated.get("context_family"),
        "generated_line": generated.get("line"),
        "generated_preview": generated.get("preview"),
    }


def summarize(
    metric_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    pollution_rows: list[dict[str, Any]],
    pcg_rows: list[dict[str, Any]],
    ordinary_reorder_rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    keys = [
        "old_visible_cov",
        "old_visible_ordered_cov",
        "old_visible_inv",
        "old_adjacent_inv",
        "old_displaced_010",
        "old_lis_disorder",
        "refined_visible_cov",
        "refined_visible_ordered_cov",
        "refined_visible_inv",
        "refined_adjacent_inv",
        "refined_displaced_010",
        "refined_lis_disorder",
    ]
    aggregate = {key: mean([as_float(row.get(key)) for row in metric_rows]) for key in keys}
    family_counts = Counter(row.get("family", "UNKNOWN") for row in context_rows)
    pollution_counts = Counter(row.get("pollution_type", "UNKNOWN") for row in pollution_rows)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "visible_prose_matching_refinement_formula_context_audit_v1",
        "breakdown_dir": str(args.breakdown_dir),
        "docs": len(metric_rows),
        "errors": errors,
        "aggregate": aggregate,
        "matching_pollution_count": len(pollution_rows),
        "matching_pollution_by_type": dict(pollution_counts.most_common()),
        "math_theorem_context_count": len(context_rows),
        "math_theorem_context_by_family": dict(family_counts.most_common()),
        "paragraph_context_group_candidate_count": len(pcg_rows),
        "ordinary_body_residual_reorder_example_count": len(ordinary_reorder_rows),
        "top_docs_by_context_count": sorted(
            metric_rows,
            key=lambda row: int(row.get("math_theorem_context_count") or 0),
            reverse=True,
        )[:20],
        "top_docs_by_refined_inversion": sorted(
            metric_rows,
            key=lambda row: as_float(row.get("refined_visible_inv")) or -1.0,
            reverse=True,
        )[:20],
    }


def render_report(
    payload: dict[str, Any],
    pcg_rows: list[dict[str, Any]],
    ordinary_reorder_rows: list[dict[str, Any]],
) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# Visible Prose Matching Refinement and Formula Context Audit",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- docs: `{payload['docs']}`",
        f"- input: `{payload['breakdown_dir']}`",
        "- MinerU / generation / training / E2E rerun: No",
        "",
        "## Original vs Refined Metrics",
        "",
        "| metric | original | refined | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    metric_pairs = [
        ("visible_cov", "old_visible_cov", "refined_visible_cov"),
        ("visible_ordered_cov", "old_visible_ordered_cov", "refined_visible_ordered_cov"),
        ("visible_inv", "old_visible_inv", "refined_visible_inv"),
        ("adjacent_inv", "old_adjacent_inv", "refined_adjacent_inv"),
        ("displaced_010", "old_displaced_010", "refined_displaced_010"),
        ("lis_disorder", "old_lis_disorder", "refined_lis_disorder"),
    ]
    for label, old_key, new_key in metric_pairs:
        old = aggregate.get(old_key)
        new = aggregate.get(new_key)
        delta = None if old is None or new is None else new - old
        lines.append(f"| `{label}` | {fmt(old)} | {fmt(new)} | {fmt(delta)} |")

    lines += [
        "",
        "## Matching Pollution",
        "",
        f"- matching pollution cases: `{payload['matching_pollution_count']}`",
        "",
        "| pollution type | count |",
        "| --- | ---: |",
    ]
    for key, value in payload["matching_pollution_by_type"].items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Math / Theorem / Formula Context",
        "",
        f"- context cases: `{payload['math_theorem_context_count']}`",
        f"- ParagraphContextGroup candidates: `{payload['paragraph_context_group_candidate_count']}`",
        "",
        "| family | count |",
        "| --- | ---: |",
    ]
    for key, value in payload["math_theorem_context_by_family"].items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Top ParagraphContextGroup Examples",
        "",
        "| doc_id | family | line | context | reason |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in pcg_rows[:20]:
        lines.append(
            f"| `{row.get('doc_id')}` | `{row.get('family')}` | {row.get('line') or ''} | "
            f"{md(row.get('context_preview'))} | {md(row.get('reason'))} |"
        )

    lines += [
        "",
        "## Top Ordinary Body Residual Reorder Examples",
        "",
        "| doc_id | source order | generated order | source previews | generated previews |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in ordinary_reorder_rows[:20]:
        source_preview = f"{row.get('left_source_preview')} || {row.get('right_source_preview')}"
        generated_preview = f"{row.get('left_generated_preview')} || {row.get('right_generated_preview')}"
        lines.append(
            f"| `{row.get('doc_id')}` | `{row.get('source_order')}` | `{row.get('generated_order')}` | "
            f"{md(source_preview)} | {md(generated_preview)} |"
        )

    lines += [
        "",
        "## Diagnosis",
        "",
        "1. Refined matching removes equation/theorem/where/formula-like context from ordinary visible prose order and reports it separately.",
        "2. If refined ordered coverage improves materially while context cases are high, the next production work should be Formula/ParagraphContextGroup rather than GNN merge.",
        "3. Matching pollution cases indicate source-generated type mismatches that should be treated as evaluator/matcher contamination or front matter/float/reference leakage.",
        "4. Remaining ordinary-body residual reorder examples are the only part that should drive v8 reflow fixes directly.",
        "",
        "## Next Recommendation",
        "",
        "Implement Formula/ParagraphContextGroup if the context-family counts remain large after this refinement. Keep GNN merge archived.",
        "",
    ]
    return "\n".join(lines)


def md(value: Any) -> str:
    text = compact(value, 180)
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = run(args)
    print(Path(args.output_dir) / "VISIBLE_PROSE_MATCHING_REFINEMENT_AND_FORMULA_CONTEXT_AUDIT_REPORT.md")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
