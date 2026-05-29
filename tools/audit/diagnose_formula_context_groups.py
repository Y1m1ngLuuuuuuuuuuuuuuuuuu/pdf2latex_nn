#!/usr/bin/env python3
"""Build Formula/ParagraphContextGroup Phase 0 sidecars from matching audit CSVs.

This tool is read-only over existing generated/source artifacts.  It consumes
the matching-refinement audit outputs and emits decoder-side context group
sidecars plus summary metrics.  It does not modify v7 JSON, graph.pt, labels,
renderer output, or generated LaTeX.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reasoning.formula_context_group import (  # noqa: E402
    FormulaContextGroup,
    InlineMathAttachment,
    TheoremProofContext,
    WhereClauseContext,
    confidence_for_context,
    render_policy_for_context,
)


FAMILIES = [
    "INLINE_MATH_ATTACHMENT",
    "THEOREM_PROOF_CONTEXT",
    "WHERE_CLAUSE_CONTEXT",
    "DISPLAY_MATH_CONTEXT",
    "FORMULA_OCR_ARTIFACT",
]
CONFIDENCE_TIERS = ["high", "medium", "low"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/09_eval_reports/v8_visible_prose_failure_breakdown_20260526/matching_refinement_formula_context_audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/09_eval_reports/formula_paragraph_context_group_20260526"),
    )
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


def compact(text: Any, limit: int = 240) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit]


def md(text: Any) -> str:
    return compact(text).replace("|", "\\|").replace("\n", " ")


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


def require_inputs(input_dir: Path, output_dir: Path) -> tuple[bool, list[str]]:
    required = [
        "math_theorem_context_cases.csv",
        "paragraph_context_group_candidates.csv",
        "matching_pollution_cases.csv",
        "refined_visible_prose_metrics.json",
        "refined_visible_prose_metrics.csv",
    ]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = [
            "# Formula Paragraph Context Group Phase 0 Readiness Report",
            "",
            f"- created_at: `{datetime.now(timezone.utc).isoformat()}`",
            f"- input_dir: `{input_dir}`",
            "- status: blocked",
            "",
            "## Missing Inputs",
            "",
        ]
        report.extend(f"- `{name}`" for name in missing)
        (output_dir / "FORMULA_PARAGRAPH_CONTEXT_GROUP_PHASE0_READINESS_REPORT.md").write_text(
            "\n".join(report) + "\n",
            encoding="utf-8",
        )
        return False, missing
    return True, []


def group_rows_by_doc(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        doc_id = row.get("doc_id") or "unknown"
        grouped[doc_id].append(row)
    return grouped


def group_from_context_row(doc_id: str, row: dict[str, str], index: int) -> FormulaContextGroup:
    family = row.get("family") or row.get("context_family") or "UNCERTAIN"
    block_id = row.get("block_id") or row.get("source_block_id") or row.get("generated_block_id") or f"{doc_id}_ctx_{index:04d}"
    evidence = {
        "origin": row.get("origin"),
        "line": row.get("line"),
        "semantic_channel": row.get("semantic_channel"),
        "preview": row.get("preview") or row.get("context_preview"),
        "reason": row.get("reason"),
        "confidence_tier": row.get("confidence") or "low",
    }
    confidence = confidence_for_context(family, _evidence_proxy(family))  # type: ignore[arg-type]
    return FormulaContextGroup(
        group_id=f"{doc_id}_formula_context_{index:04d}",
        context_type=family,  # type: ignore[arg-type]
        source_v7_ids=[block_id],
        text_before_ids=[],
        formula_ids=[block_id] if family in {"DISPLAY_MATH_CONTEXT", "FORMULA_OCR_ARTIFACT"} else [],
        text_after_ids=[],
        theorem_label_ids=[block_id] if family == "THEOREM_PROOF_CONTEXT" else [],
        confidence=confidence,
        evidence=evidence,
        render_policy=render_policy_for_context(family),  # type: ignore[arg-type]
        confidence_tier=row.get("confidence") or ("high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low"),
    )


class _evidence_proxy:
    def __init__(self, family: str) -> None:
        self.theorem_like = family == "THEOREM_PROOF_CONTEXT"
        self.starts_where_clause = family == "WHERE_CLAUSE_CONTEXT"
        self.display_math_env = family == "DISPLAY_MATH_CONTEXT"
        self.semantic_channel = "display_math" if family == "DISPLAY_MATH_CONTEXT" else None
        self.inline_math_marker = family == "INLINE_MATH_ATTACHMENT"
        self.short_fragment = family == "FORMULA_OCR_ARTIFACT"
        self.equation_number_like = False
        self.local_formula_context = family == "WHERE_CLAUSE_CONTEXT"
        self.negative_where_start = False


def inline_attachment_from_row(doc_id: str, row: dict[str, str], index: int) -> InlineMathAttachment:
    block_id = row.get("block_id") or f"{doc_id}_inline_{index:04d}"
    return InlineMathAttachment(
        paragraph_node_id=str(row.get("block_id") or block_id),
        inline_formula_node_ids=[str(block_id)],
        insertion_position="uncertain",
        confidence=0.78,
        evidence={
            "origin": row.get("origin"),
            "line": row.get("line"),
            "preview": row.get("preview"),
            "reason": row.get("reason"),
            "confidence_tier": row.get("confidence") or "medium",
        },
    )


def theorem_context_from_row(doc_id: str, row: dict[str, str], index: int) -> TheoremProofContext:
    block_id = row.get("block_id") or f"{doc_id}_theorem_{index:04d}"
    return TheoremProofContext(
        label_text=compact(row.get("preview"), 80),
        body_node_ids=[str(block_id)],
        source_v7_ids=[str(block_id)],
        confidence=0.90,
        evidence={
            "origin": row.get("origin"),
            "line": row.get("line"),
            "reason": row.get("reason"),
            "confidence_tier": row.get("confidence") or "high",
        },
    )


def where_context_from_row(doc_id: str, row: dict[str, str], index: int) -> WhereClauseContext:
    block_id = row.get("block_id") or f"{doc_id}_where_{index:04d}"
    return WhereClauseContext(
        lead_in_node_ids=[],
        display_math_node_ids=[],
        where_clause_node_ids=[str(block_id)],
        confidence=0.86,
        evidence={
            "origin": row.get("origin"),
            "line": row.get("line"),
            "preview": row.get("preview"),
            "reason": row.get("reason"),
            "confidence_tier": row.get("confidence") or "high",
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    ready, missing = require_inputs(args.input_dir, args.output_dir)
    if not ready:
        return {"status": "blocked", "missing": missing}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    context_rows = read_csv(args.input_dir / "math_theorem_context_cases.csv")
    pcg_rows = read_csv(args.input_dir / "paragraph_context_group_candidates.csv")
    pollution_rows = read_csv(args.input_dir / "matching_pollution_cases.csv")
    metric_rows = read_csv(args.input_dir / "refined_visible_prose_metrics.csv")
    refined_payload = json.loads((args.input_dir / "refined_visible_prose_metrics.json").read_text(encoding="utf-8"))

    wanted = set(args.doc_ids) if args.doc_ids else None
    doc_ids = sorted({row.get("doc_id", "") for row in metric_rows if row.get("doc_id")})
    if wanted:
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in wanted]
    if args.limit is not None:
        doc_ids = doc_ids[: args.limit]
    doc_set = set(doc_ids)

    context_by_doc = group_rows_by_doc([row for row in context_rows if row.get("doc_id") in doc_set])
    pcg_by_doc = group_rows_by_doc([row for row in pcg_rows if row.get("doc_id") in doc_set])
    pollution_by_doc = group_rows_by_doc([row for row in pollution_rows if row.get("doc_id") in doc_set])
    metric_by_doc = {row.get("doc_id"): row for row in metric_rows if row.get("doc_id") in doc_set}

    summary_rows: list[dict[str, Any]] = []
    all_groups: list[dict[str, Any]] = []
    all_inline: list[dict[str, Any]] = []
    all_theorem: list[dict[str, Any]] = []
    all_where: list[dict[str, Any]] = []

    for doc_id in doc_ids:
        doc_dir = args.output_dir / "per_doc" / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        rows = context_by_doc.get(doc_id, [])
        family_counts = Counter(row.get("family") or "UNCERTAIN" for row in rows)
        tier_counts = Counter(row.get("confidence") or "low" for row in rows)
        high_rows = [row for row in rows if (row.get("confidence") or "low") == "high"]
        groups = [group_from_context_row(doc_id, row, idx).to_dict() for idx, row in enumerate(high_rows)]
        inline = [
            inline_attachment_from_row(doc_id, row, idx).to_dict()
            for idx, row in enumerate(high_rows)
            if row.get("family") == "INLINE_MATH_ATTACHMENT"
        ]
        theorem = [
            theorem_context_from_row(doc_id, row, idx).to_dict()
            for idx, row in enumerate(high_rows)
            if row.get("family") == "THEOREM_PROOF_CONTEXT"
        ]
        where = [
            where_context_from_row(doc_id, row, idx).to_dict()
            for idx, row in enumerate(high_rows)
            if row.get("family") == "WHERE_CLAUSE_CONTEXT"
        ]
        diag = {
            "schema_version": "formula_context_diag_v1",
            "doc_id": doc_id,
            "family_counts": dict(family_counts),
            "confidence_tier_counts": dict(tier_counts),
            "high_confidence_family_counts": dict(Counter(row.get("family") or "UNCERTAIN" for row in high_rows)),
            "paragraph_context_group_candidate_count": len(pcg_by_doc.get(doc_id, [])),
            "matching_pollution_count": len(pollution_by_doc.get(doc_id, [])),
            "metric_row": metric_by_doc.get(doc_id, {}),
        }
        (doc_dir / "formula_context_groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (doc_dir / "inline_math_attachments.json").write_text(json.dumps(inline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (doc_dir / "theorem_proof_contexts.json").write_text(json.dumps(theorem, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (doc_dir / "where_clause_contexts.json").write_text(json.dumps(where, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (doc_dir / "formula_context_diag.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metric = metric_by_doc.get(doc_id, {})
        summary_rows.append(
            {
                "doc_id": doc_id,
                "inline_math_attachment_count": family_counts.get("INLINE_MATH_ATTACHMENT", 0),
                "theorem_proof_context_count": family_counts.get("THEOREM_PROOF_CONTEXT", 0),
                "where_clause_context_count": family_counts.get("WHERE_CLAUSE_CONTEXT", 0),
                "display_math_context_count": family_counts.get("DISPLAY_MATH_CONTEXT", 0),
                "formula_ocr_artifact_count": family_counts.get("FORMULA_OCR_ARTIFACT", 0),
                "formula_context_group_count": sum(family_counts.values()),
                "high_confidence_formula_context_group_count": len(high_rows),
                "medium_confidence_formula_context_group_count": tier_counts.get("medium", 0),
                "low_confidence_formula_context_group_count": tier_counts.get("low", 0),
                "paragraph_context_group_candidate_count": len(pcg_by_doc.get(doc_id, [])),
                "formula_context_pollution_count": len(pollution_by_doc.get(doc_id, [])),
                "ordinary_body_reorder_count_after_context_filter": metric.get("ordinary_body_residual_reorder_count", ""),
                "refined_visible_cov": metric.get("refined_visible_cov", ""),
                "refined_visible_ordered_cov": metric.get("refined_visible_ordered_cov", ""),
                "refined_visible_inv": metric.get("refined_visible_inv", ""),
                "refined_adjacent_inv": metric.get("refined_adjacent_inv", ""),
                "refined_lis_disorder": metric.get("refined_lis_disorder", ""),
            }
        )
        all_groups.extend({"doc_id": doc_id, **row} for row in groups)
        all_inline.extend({"doc_id": doc_id, **row} for row in inline)
        all_theorem.extend({"doc_id": doc_id, **row} for row in theorem)
        all_where.extend({"doc_id": doc_id, **row} for row in where)

    write_csv(args.output_dir / "formula_context_group_summary.csv", summary_rows)
    payload = summarize(args, summary_rows, refined_payload, context_rows, pcg_rows, pollution_rows)
    (args.output_dir / "formula_context_group_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "FORMULA_PARAGRAPH_CONTEXT_GROUP_PHASE0_REPORT.md").write_text(
        render_report(payload, context_rows, pcg_rows, pollution_rows),
        encoding="utf-8",
    )
    return payload


def summarize(
    args: argparse.Namespace,
    summary_rows: list[dict[str, Any]],
    refined_payload: dict[str, Any],
    context_rows: list[dict[str, str]],
    pcg_rows: list[dict[str, str]],
    pollution_rows: list[dict[str, str]],
) -> dict[str, Any]:
    family_counts = Counter(row.get("family") or "UNCERTAIN" for row in context_rows)
    tier_counts = Counter(row.get("confidence") or "low" for row in context_rows)
    family_tier_counts = Counter((row.get("family") or "UNCERTAIN", row.get("confidence") or "low") for row in context_rows)
    aggregate = refined_payload.get("aggregate", {})
    impact = {
        "old_visible_cov": aggregate.get("old_visible_cov"),
        "context_filtered_visible_cov": aggregate.get("refined_visible_cov"),
        "old_visible_ordered_cov": aggregate.get("old_visible_ordered_cov"),
        "context_filtered_visible_ordered_cov": aggregate.get("refined_visible_ordered_cov"),
        "old_visible_inv": aggregate.get("old_visible_inv"),
        "context_filtered_visible_inv": aggregate.get("refined_visible_inv"),
        "old_adjacent_inv": aggregate.get("old_adjacent_inv"),
        "context_filtered_adjacent_inv": aggregate.get("refined_adjacent_inv"),
        "old_lis_disorder": aggregate.get("old_lis_disorder"),
        "context_filtered_lis_disorder": aggregate.get("refined_lis_disorder"),
        "ordinary_body_reorder_count_after_context_filter": sum(
            int(row.get("ordinary_body_reorder_count_after_context_filter") or 0) for row in summary_rows
        ),
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "formula_paragraph_context_group_phase0_summary_v1",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "docs": len(summary_rows),
        "status": "diagnostic_only",
        "no_training": True,
        "no_mineru": True,
        "no_relabel": True,
        "no_gnn": True,
        "no_full_e2e": True,
        "family_counts": {family: family_counts.get(family, 0) for family in FAMILIES},
        "confidence_tier_counts": {tier: tier_counts.get(tier, 0) for tier in CONFIDENCE_TIERS},
        "family_confidence_tier_counts": {
            family: {tier: family_tier_counts.get((family, tier), 0) for tier in CONFIDENCE_TIERS}
            for family in FAMILIES
        },
        "matching_pollution_count": len(pollution_rows),
        "paragraph_context_group_candidate_count": len(pcg_rows),
        "metric_impact": impact,
        "decision": decision_for_summary(family_counts, len(pcg_rows)),
        "summary_rows": summary_rows,
    }


def decision_for_summary(family_counts: Counter[str], pcg_count: int) -> str:
    context_count = sum(family_counts.values())
    if context_count == 0:
        return "keep diagnostic only"
    if pcg_count > 0 and context_count >= 500:
        return "safe to enable FormulaContextGroup in experimental v8+hint path"
    return "needs rule patch"


def render_report(
    payload: dict[str, Any],
    context_rows: list[dict[str, str]],
    pcg_rows: list[dict[str, str]],
    pollution_rows: list[dict[str, str]],
) -> str:
    impact = payload["metric_impact"]
    lines = [
        "# Formula Paragraph Context Group Phase 0 Report",
        "",
        "## Status",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- input_dir: `{payload['input_dir']}`",
        f"- output_dir: `{payload['output_dir']}`",
        f"- docs analyzed: `{payload['docs']}`",
        "- implemented files: `src/reasoning/formula_context_group.py`, `src/reasoning/paragraph_context_group.py`, `src/generation/ir_renderers/math.py`, `src/generation/ir_renderers/text.py`, `tools/audit/diagnose_formula_context_groups.py`",
        "- tests added: `tests/test_formula_context_group.py`, `tests/test_inline_math_attachment.py`, `tests/test_theorem_proof_context.py`, `tests/test_where_clause_context.py`",
        "- training / MinerU / relabel / GNN / full E2E: No",
        "",
        "## Context Summary",
        "",
        "| context type | count |",
        "| --- | ---: |",
    ]
    for family in FAMILIES:
        tier_counts = payload.get("family_confidence_tier_counts", {}).get(family, {})
        lines.append(
            f"| `{family}` | {payload['family_counts'].get(family, 0)} "
            f"(high={tier_counts.get('high', 0)}, medium={tier_counts.get('medium', 0)}, low={tier_counts.get('low', 0)}) |"
        )
    lines += [
        "",
        "## Metric Impact",
        "",
        "| metric | old visible prose | context-filtered | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    metric_pairs = [
        ("visible coverage", "old_visible_cov", "context_filtered_visible_cov"),
        ("visible ordered coverage", "old_visible_ordered_cov", "context_filtered_visible_ordered_cov"),
        ("visible inversion", "old_visible_inv", "context_filtered_visible_inv"),
        ("adjacent inversion", "old_adjacent_inv", "context_filtered_adjacent_inv"),
        ("LIS disorder", "old_lis_disorder", "context_filtered_lis_disorder"),
    ]
    for label, old_key, new_key in metric_pairs:
        old = impact.get(old_key)
        new = impact.get(new_key)
        delta = None if old is None or new is None else new - old
        lines.append(f"| {label} | {fmt(old)} | {fmt(new)} | {fmt(delta)} |")
    lines += [
        "",
        f"- ordinary_body_reorder_count_after_context_filter: `{impact.get('ordinary_body_reorder_count_after_context_filter')}`",
        f"- formula_context_group_count: `{sum(payload['family_counts'].values())}`",
        f"- matching_pollution_count: `{payload['matching_pollution_count']}`",
        "",
        "## Examples",
        "",
        "### Inline Math Attachment Examples",
        "",
        *example_table(context_rows, "INLINE_MATH_ATTACHMENT", "preview"),
        "",
        "### Theorem / Proof Examples",
        "",
        *example_table(context_rows, "THEOREM_PROOF_CONTEXT", "preview"),
        "",
        "### Where-Clause Examples",
        "",
        *example_table(context_rows, "WHERE_CLAUSE_CONTEXT", "preview"),
        "",
        "### Uncertain / Formula OCR Artifact Examples",
        "",
        *example_table(context_rows, "FORMULA_OCR_ARTIFACT", "preview"),
        "",
        "### ParagraphContextGroup Examples",
        "",
        "| doc_id | family | context | reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in pcg_rows[:20]:
        lines.append(
            f"| `{row.get('doc_id')}` | `{row.get('family')}` | {md(row.get('context_preview'))} | {md(row.get('reason'))} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "Phase 0 remains decoder/IR-side and diagnostic.  It should only be wired into the experimental v8+hint path after spot-checking examples; it should not become a GNN label or ordinary MERGE rule.",
    ]
    return "\n".join(lines) + "\n"


def example_table(rows: list[dict[str, str]], family: str, preview_key: str) -> list[str]:
    lines = ["| doc_id | line | preview | reason |", "| --- | ---: | --- | --- |"]
    selected = [row for row in rows if row.get("family") == family][:20]
    if not selected:
        lines.append("| N/A |  |  |  |")
        return lines
    for row in selected:
        lines.append(
            f"| `{row.get('doc_id')}` | {row.get('line') or ''} | {md(row.get(preview_key))} | {md(row.get('reason'))} |"
        )
    return lines


def main() -> int:
    payload = run(build_arg_parser().parse_args())
    if payload.get("status") == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
