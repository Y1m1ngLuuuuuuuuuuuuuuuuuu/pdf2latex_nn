"""ParagraphContextGroup Phase 0 helpers.

The groups here are decoder-side suggestions.  They do not change v7 records or
GNN graph inputs; they provide a stable sidecar for audits and future
RenderTreeIR materialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.reasoning.formula_context_group import (
    FormulaContextGroup,
    FormulaContextType,
    build_formula_context_groups,
    classify_record_formula_context,
    classify_formula_context,
    has_local_formula_evidence_neighbor,
    has_local_formula_neighbor,
    record_id,
    record_channel,
    record_metadata,
    record_text,
    should_exclude_from_ordinary_visible_prose_evidence,
)


ParagraphContextKind = Literal[
    "formula_context",
    "where_clause_context",
    "theorem_proof_context",
    "inline_math_attachment",
    "uncertain",
]


@dataclass(frozen=True)
class ParagraphContextGroup:
    group_id: str
    context_kind: ParagraphContextKind
    context_type: FormulaContextType
    source_v7_ids: list[str]
    text_before_ids: list[str] = field(default_factory=list)
    context_ids: list[str] = field(default_factory=list)
    text_after_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    render_policy: str = "plain_paragraph_fallback"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paragraph_context_kind_for(context_type: str) -> ParagraphContextKind:
    if context_type == "INLINE_MATH_ATTACHMENT":
        return "inline_math_attachment"
    if context_type == "WHERE_CLAUSE_CONTEXT":
        return "where_clause_context"
    if context_type == "THEOREM_PROOF_CONTEXT":
        return "theorem_proof_context"
    if context_type in {"DISPLAY_MATH_CONTEXT", "FORMULA_OCR_ARTIFACT"}:
        return "formula_context"
    return "uncertain"


def build_paragraph_context_groups(records: list[Any], *, group_prefix: str = "pcg") -> list[ParagraphContextGroup]:
    formula_groups = build_formula_context_groups(records, group_prefix=f"{group_prefix}_formula")
    groups: list[ParagraphContextGroup] = []
    for group in formula_groups:
        groups.append(group_from_formula_context(group, group_id=f"{group_prefix}_{len(groups):04d}"))
    for idx, record in enumerate(records):
        context_type, evidence = classify_formula_context(
            record_text(record),
            raw_text=getattr(record, "raw_text", None) if not isinstance(record, dict) else record.get("raw_text"),
            semantic_channel=record_channel(record),
            local_formula_context=has_local_formula_neighbor(records, idx),
            local_formula_evidence=has_local_formula_evidence_neighbor(records, idx),
            formula_metadata=record_metadata(record),
        )
        if context_type != "INLINE_MATH_ATTACHMENT":
            continue
        if evidence.confidence_tier != "high":
            continue
        current_id = record_id(record, f"record_{idx:04d}")
        before_id = record_id(records[idx - 1], f"record_{idx-1:04d}") if idx > 0 else None
        after_id = record_id(records[idx + 1], f"record_{idx+1:04d}") if idx + 1 < len(records) else None
        paragraph_id = before_id or after_id or current_id
        groups.append(
            ParagraphContextGroup(
                group_id=f"{group_prefix}_{len(groups):04d}",
                context_kind="inline_math_attachment",
                context_type=context_type,
                source_v7_ids=[current_id],
                text_before_ids=[before_id] if before_id else [],
                context_ids=[current_id],
                text_after_ids=[after_id] if after_id else [],
                confidence=0.82,
                evidence={
                    **evidence.to_dict(),
                    "paragraph_node_id": paragraph_id,
                    "fragment_preview": record_text(record)[:160],
                },
                render_policy="inline_math_span_attachment",
            )
        )
    return groups


def group_from_formula_context(group: FormulaContextGroup, *, group_id: str | None = None) -> ParagraphContextGroup:
    return ParagraphContextGroup(
        group_id=group_id or group.group_id,
        context_kind=paragraph_context_kind_for(group.context_type),
        context_type=group.context_type,
        source_v7_ids=list(group.source_v7_ids),
        text_before_ids=list(group.text_before_ids),
        context_ids=list(group.formula_ids or group.theorem_label_ids or group.source_v7_ids),
        text_after_ids=list(group.text_after_ids),
        confidence=group.confidence,
        evidence=dict(group.evidence),
        render_policy=group.render_policy,
    )


def ordinary_visible_prose_context_filter(records: list[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    kept: list[Any] = []
    excluded: list[dict[str, Any]] = []
    for record in records:
        context_type, evidence = classify_record_formula_context(record)
        if should_exclude_from_ordinary_visible_prose_evidence(context_type, evidence):
            excluded.append(
                {
                    "record_id": record_id(record),
                    "context_type": context_type,
                    "reason": evidence.reason,
                    "preview": record_text(record)[:200],
                }
            )
        else:
            kept.append(record)
    return kept, excluded
