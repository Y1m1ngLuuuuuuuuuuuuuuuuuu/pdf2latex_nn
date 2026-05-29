"""Front-matter context helpers backed by P0-E model/page-furniture evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.reasoning.page_furniture_context_group import PageFurnitureContext, contexts_from_document_ir_check


FRONT_MATTER_KINDS = {
    "document_title",
    "model_title",
    "front_matter_author_affiliation_candidate",
    "abstract_title_candidate",
}


@dataclass(frozen=True)
class FrontMatterContext:
    context_id: str
    doc_id: str
    node_id: str
    page_idx: int
    text_preview: str
    front_matter_role: str
    evidence_source: str
    confidence_tier: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "doc_id": self.doc_id,
            "node_id": self.node_id,
            "page_idx": self.page_idx,
            "text_preview": self.text_preview,
            "front_matter_role": self.front_matter_role,
            "evidence_source": self.evidence_source,
            "confidence_tier": self.confidence_tier,
            "reason": self.reason,
        }


def front_matter_contexts_from_document_ir_check(doc_id: str, document_check: dict[str, Any]) -> list[FrontMatterContext]:
    return front_matter_contexts_from_page_contexts(contexts_from_document_ir_check(doc_id, document_check))


def front_matter_contexts_from_page_contexts(contexts: list[PageFurnitureContext]) -> list[FrontMatterContext]:
    front_contexts: list[FrontMatterContext] = []
    for context in contexts:
        if context.page_idx != 0 and context.context_kind not in {"page_header", "page_footer", "page_number"}:
            continue
        if context.context_kind not in FRONT_MATTER_KINDS and not any(mask in context.negative_masks for mask in ("front_matter_body_heading", "title_body_heading", "abstract_title_body_heading")):
            continue
        front_contexts.append(
            FrontMatterContext(
                context_id=context.context_id.replace(":", ":front:", 1),
                doc_id=context.doc_id,
                node_id=context.node_id,
                page_idx=context.page_idx,
                text_preview=context.text_preview,
                front_matter_role=front_matter_role(context),
                evidence_source=context.evidence_source,
                confidence_tier=context.confidence_tier,
                reason="front matter negative evidence derived from P0-E model/page-furniture metadata",
            )
        )
    return front_contexts


def front_matter_role(context: PageFurnitureContext) -> str:
    if context.context_kind == "document_title":
        return "document_title_candidate"
    if context.context_kind == "model_title":
        return "front_matter_title_candidate"
    if context.context_kind == "front_matter_author_affiliation_candidate":
        return "author_affiliation_candidate"
    if context.context_kind == "abstract_title_candidate":
        return "abstract_title_candidate"
    if "abstract_title_body_heading" in context.negative_masks:
        return "abstract_title_candidate"
    if "title_body_heading" in context.negative_masks:
        return "front_matter_title_candidate"
    return "front_matter_candidate"
