"""Page furniture/model-label context classification for audit tracks.

This module consumes P0-E MinerU/model preservation metadata. It does not
perform renderer changes and keeps regex-only page-furniture guesses diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


HIGH_CONFIDENCE_VALUES = {"strong_content_list_role", "strong_middle_discarded", "strong_model_label"}


@dataclass(frozen=True)
class PageFurnitureContext:
    context_id: str
    doc_id: str
    node_id: str
    page_idx: int
    text_preview: str
    context_kind: str
    evidence_source: str
    confidence_tier: str
    model_label: str | None = None
    model_score: float | None = None
    page_furniture_role: str | None = None
    negative_masks: tuple[str, ...] = field(default_factory=tuple)
    source_v8_ids: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "doc_id": self.doc_id,
            "node_id": self.node_id,
            "page_idx": self.page_idx,
            "text_preview": self.text_preview,
            "context_kind": self.context_kind,
            "evidence_source": self.evidence_source,
            "confidence_tier": self.confidence_tier,
            "model_label": self.model_label,
            "model_score": self.model_score,
            "page_furniture_role": self.page_furniture_role,
            "negative_masks": list(self.negative_masks),
            "source_v8_ids": list(self.source_v8_ids),
            "reason": self.reason,
        }


def contexts_from_document_ir_check(doc_id: str, document_check: dict[str, Any]) -> list[PageFurnitureContext]:
    contexts: list[PageFurnitureContext] = []
    for node in document_check.get("after_page_furniture_nodes") or []:
        context = context_from_node(doc_id, node)
        if context is not None:
            contexts.append(context)
    for node in document_check.get("after_model_label_nodes") or []:
        context = context_from_node(doc_id, node)
        if context is not None:
            contexts.append(context)
    deduped: dict[str, PageFurnitureContext] = {}
    for context in contexts:
        deduped.setdefault(context.context_id, context)
    return list(deduped.values())


def context_from_node(doc_id: str, node: dict[str, Any]) -> PageFurnitureContext | None:
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    if not metadata:
        return None
    role = str(metadata.get("mineru_page_furniture_role") or "")
    model_label = metadata.get("model_label")
    has_mask = any(
        bool(metadata.get(key))
        for key in (
            "should_exclude_from_body_order",
            "should_exclude_from_heading_detection",
            "should_exclude_from_visible_prose_metric",
            "front_matter_negative_for_body_heading",
            "title_negative_for_body_heading",
            "abstract_title_negative_for_body_heading",
            "is_document_title_candidate",
            "is_front_matter_candidate",
        )
    )
    if not role and model_label is None and not has_mask:
        return None
    node_id = str(node.get("node_id") or node.get("id") or "")
    context_kind = context_kind_from_metadata(metadata)
    evidence_source = evidence_source_from_metadata(metadata)
    confidence_tier = confidence_tier_from_metadata(metadata)
    return PageFurnitureContext(
        context_id=f"{doc_id}:{node_id}:{context_kind}",
        doc_id=doc_id,
        node_id=node_id,
        page_idx=int(node.get("page_idx") or 0),
        text_preview=" ".join(str(node.get("text_preview") or node.get("text") or "").split())[:240],
        context_kind=context_kind,
        evidence_source=evidence_source,
        confidence_tier=confidence_tier,
        model_label=str(model_label) if model_label is not None else None,
        model_score=_float_or_none(metadata.get("model_score")),
        page_furniture_role=role or None,
        negative_masks=tuple(negative_masks_from_metadata(metadata)),
        source_v8_ids=tuple(source_ids_from_metadata(node_id, metadata)),
        reason=reason_from_metadata(metadata),
    )


def context_kind_from_metadata(metadata: dict[str, Any]) -> str:
    role = str(metadata.get("mineru_page_furniture_role") or "")
    if role in {"page_header", "page_footer", "page_number", "page_footnote", "aside_text", "margin_note", "discarded_block"}:
        return role
    label = str(metadata.get("model_label") or "").casefold()
    if label == "doc_title" or metadata.get("is_document_title_candidate"):
        return "document_title"
    if label in {"title", "paragraph_title"}:
        return "model_title"
    if label in {"header", "footer", "number", "page_number"}:
        return {"header": "page_header", "footer": "page_footer", "number": "page_number", "page_number": "page_number"}[label]
    if metadata.get("is_author_affiliation_candidate"):
        return "front_matter_author_affiliation_candidate"
    if metadata.get("is_abstract_title_candidate"):
        return "abstract_title_candidate"
    return "model_label_context"


def evidence_source_from_metadata(metadata: dict[str, Any]) -> str:
    sources: list[str] = []
    if metadata.get("mineru_page_furniture_role"):
        layer = str(metadata.get("page_furniture_source_layer") or "")
        if layer == "content_list":
            sources.append("mineru_content_list_role")
        elif layer == "middle":
            sources.append("mineru_middle_discarded")
        else:
            sources.append("document_ir_negative_mask")
    if metadata.get("model_label") is not None:
        sources.append("model_label")
    if len(sources) > 1:
        return "mixed"
    return sources[0] if sources else "document_ir_negative_mask"


def confidence_tier_from_metadata(metadata: dict[str, Any]) -> str:
    page_confidence = str(metadata.get("page_furniture_confidence") or "")
    model_confidence = str(metadata.get("model_label_confidence") or "")
    if page_confidence in HIGH_CONFIDENCE_VALUES or model_confidence in HIGH_CONFIDENCE_VALUES:
        return "high"
    if page_confidence or model_confidence:
        return "medium"
    return "diagnostic_only"


def negative_masks_from_metadata(metadata: dict[str, Any]) -> list[str]:
    masks: list[str] = []
    if metadata.get("should_exclude_from_body_order"):
        masks.append("body_order")
    if metadata.get("should_exclude_from_heading_detection"):
        masks.append("heading_detection")
    if metadata.get("should_exclude_from_visible_prose_metric"):
        masks.append("visible_prose")
    if metadata.get("front_matter_negative_for_body_heading"):
        masks.append("front_matter_body_heading")
    if metadata.get("title_negative_for_body_heading"):
        masks.append("title_body_heading")
    if metadata.get("abstract_title_negative_for_body_heading"):
        masks.append("abstract_title_body_heading")
    return masks


def source_ids_from_metadata(node_id: str, metadata: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("page_furniture_source_ids", "source_block_ids", "source_line_ids"):
        value = metadata.get(key)
        if isinstance(value, list):
            ids.extend(str(part) for part in value if str(part))
        elif value:
            ids.append(str(value))
    if node_id:
        ids.append(node_id)
    return list(dict.fromkeys(ids))


def reason_from_metadata(metadata: dict[str, Any]) -> str:
    kind = context_kind_from_metadata(metadata)
    if kind in {"page_header", "page_footer", "page_number", "page_footnote", "aside_text", "margin_note", "discarded_block"}:
        return "strong page furniture evidence preserved from MinerU/model metadata"
    if kind in {"document_title", "model_title", "front_matter_author_affiliation_candidate", "abstract_title_candidate"}:
        return "front matter/title evidence preserved from model/layout metadata"
    return "model label evidence preserved for diagnostic context"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
