"""Reference context diagnostics backed by MinerU reference subtype evidence.

This module is audit/context-track only. It consumes DocumentIR metadata
preserved from v8 full observable facts and keeps regex-only reference guesses
diagnostic. It does not mutate v8 facts, graph views, renderer paths, or labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.ir import BlockType, DocumentIR, DocumentNode


REFERENCE_HEADING_RE = re.compile(r"^\s*(references|bibliography|reference)\s*$", re.IGNORECASE)
BODY_CITATION_RE = re.compile(
    r"\b(?:see|as\s+shown\s+in|shown\s+in|prior\s+work|using|from|in)\s+"
    r"\[\d+(?:\s*[,;-]\s*\d+)*\]",
    re.IGNORECASE,
)
INLINE_BRACKET_CITATION_RE = re.compile(r"\[[0-9]+(?:\s*[,;-]\s*[0-9]+)*\]")
REFERENCE_ITEM_RE = re.compile(
    r"^\s*(?:\[[0-9A-Za-z]+\]|\d+[\).])\s+.{10,}",
    re.DOTALL,
)
REFERENCE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
REFERENCE_HIGH_CONFIDENCE = {"strong_ref_text_subtype", "strong_reference_region"}


@dataclass(frozen=True)
class ReferenceEvidenceContext:
    context_id: str
    text: str
    context_kind: str
    evidence_source: str = "regex_only"
    confidence_tier: str = "diagnostic_only"
    source_v8_ids: list[str] = field(default_factory=list)
    page_idx: int | None = None
    parent_reference_block_id: str | None = None
    list_item_order: int | None = None
    canonical_mineru_reference_id: str | None = None
    source_layers: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "text": self.text,
            "context_kind": self.context_kind,
            "evidence_source": self.evidence_source,
            "confidence_tier": self.confidence_tier,
            "source_v8_ids": list(self.source_v8_ids),
            "page_idx": self.page_idx,
            "parent_reference_block_id": self.parent_reference_block_id,
            "list_item_order": self.list_item_order,
            "canonical_mineru_reference_id": self.canonical_mineru_reference_id,
            "source_layers": list(self.source_layers),
            "evidence": dict(self.evidence),
        }


def reference_evidence_contexts_from_document(document: DocumentIR) -> list[ReferenceEvidenceContext]:
    contexts: list[ReferenceEvidenceContext] = []
    for node in document.nodes:
        contexts.extend(reference_evidence_contexts_from_node(node))
    return contexts


def reference_evidence_contexts_from_node(node: DocumentNode) -> list[ReferenceEvidenceContext]:
    metadata = node.metadata or {}
    contexts: list[ReferenceEvidenceContext] = []
    role = str(metadata.get("mineru_reference_role") or "").strip()
    reference_text = str(metadata.get("reference_text") or "").strip()
    if role and reference_text:
        confidence = str(metadata.get("reference_confidence") or "")
        context_role = str(metadata.get("reference_context_role") or "")
        if context_role == "reference_heading" or role == "reference_heading":
            kind = "reference_heading"
        elif context_role == "bibliography_block" or role == "reference_list":
            kind = "bibliography_block"
        elif bool(metadata.get("is_reference_item")) or role in {"ref_text", "bibliography_item"}:
            kind = "reference_item"
        else:
            kind = "reference_diagnostic"
        contexts.append(
            ReferenceEvidenceContext(
                context_id=f"mineru_reference_{_safe_id(node.node_id)}",
                text=" ".join(reference_text.split()),
                context_kind=kind,
                evidence_source=_reference_evidence_source(metadata),
                confidence_tier="high" if confidence in REFERENCE_HIGH_CONFIDENCE else "medium",
                source_v8_ids=[node.node_id],
                page_idx=node.page_idx,
                parent_reference_block_id=_parent_reference_block_id(metadata),
                list_item_order=_int_or_none(metadata.get("list_item_order", metadata.get("reference_list_item_index"))),
                canonical_mineru_reference_id=canonical_mineru_reference_id(node),
                source_layers=_reference_source_layers(metadata),
                evidence={
                    "mineru_reference_role": role,
                    "reference_confidence": confidence,
                    "reference_source_layer": metadata.get("reference_source_layer"),
                    "reference_context_role": context_role,
                    "reference_source_ids": metadata.get("reference_source_ids") or [],
                    "reference_label": metadata.get("reference_label"),
                    "reference_bbox": metadata.get("reference_bbox"),
                },
            )
        )
        return contexts
    text = str(node.text or "")
    if is_body_citation_text(text):
        contexts.append(
            ReferenceEvidenceContext(
                context_id=f"body_citation_{_safe_id(node.node_id)}",
                text=" ".join(text.split()),
                context_kind="body_citation_guard",
                evidence_source="regex_only",
                confidence_tier="diagnostic_only",
                source_v8_ids=[node.node_id],
                page_idx=node.page_idx,
                evidence={"reason": "body_citation_guard_without_ref_text_evidence"},
            )
        )
    elif is_reference_like_text(text):
        contexts.append(
            ReferenceEvidenceContext(
                context_id=f"regex_reference_{_safe_id(node.node_id)}",
                text=" ".join(text.split()),
                context_kind="reference_like_diagnostic",
                evidence_source="regex_only",
                confidence_tier="diagnostic_only",
                source_v8_ids=[node.node_id],
                page_idx=node.page_idx,
                evidence={"reason": "regex_reference_like_without_mineru_evidence"},
            )
        )
    elif node.node_type == BlockType.LIST:
        contexts.append(
            ReferenceEvidenceContext(
                context_id=f"ordinary_list_{_safe_id(node.node_id)}",
                text=" ".join(text.split()),
                context_kind="ordinary_list",
                evidence_source="document_ir_reference_metadata",
                confidence_tier="diagnostic_only",
                source_v8_ids=[node.node_id],
                page_idx=node.page_idx,
                evidence={"reason": "ordinary_list_without_ref_text_evidence"},
            )
        )
    return contexts


def is_body_citation_text(text: str) -> bool:
    value = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if not value:
        return False
    if REFERENCE_ITEM_RE.match(value):
        return False
    return bool(BODY_CITATION_RE.search(value) or (INLINE_BRACKET_CITATION_RE.search(value) and len(value) > 40))


def is_reference_like_text(text: str) -> bool:
    value = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if not value:
        return False
    if REFERENCE_HEADING_RE.match(value):
        return True
    if REFERENCE_ITEM_RE.match(value) and (REFERENCE_YEAR_RE.search(value) or len(value) > 60):
        return True
    return False


def canonical_mineru_reference_id(node: DocumentNode) -> str | None:
    metadata = node.metadata or {}
    text = str(metadata.get("reference_text") or "")
    if not text:
        return None
    source_ids = metadata.get("reference_source_ids")
    if isinstance(source_ids, list) and source_ids:
        source_key = "|".join(sorted(str(part) for part in source_ids if str(part)))
    else:
        source_key = "|".join(sorted(str(part) for part in metadata.get("source_block_ids") or [] if str(part)))
    parent = _parent_reference_block_id(metadata) or ""
    role = str(metadata.get("mineru_reference_role") or "reference")
    return f"{role}::{parent}::{source_key}::{_norm_text(text)}"


def _reference_evidence_source(metadata: dict[str, Any]) -> str:
    confidence = str(metadata.get("reference_confidence") or "")
    layer = str(metadata.get("reference_source_layer") or "").casefold()
    if confidence == "strong_ref_text_subtype":
        return "mineru_ref_text_subtype"
    if layer == "content_list":
        return "content_list_field"
    if layer == "middle":
        return "mineru_ref_text_subtype"
    if metadata.get("mineru_reference_role"):
        return "document_ir_reference_metadata"
    return "regex_only"


def _reference_source_layers(metadata: dict[str, Any]) -> list[str]:
    layers: list[str] = []
    for key in ("reference_source_layer", "source_layer_hierarchy"):
        value = metadata.get(key)
        if isinstance(value, list):
            layers.extend(str(part) for part in value if str(part))
        elif value:
            layers.extend(str(value).split(","))
    return list(dict.fromkeys(layer.strip() for layer in layers if layer.strip()))


def _parent_reference_block_id(metadata: dict[str, Any]) -> str | None:
    for key in ("parent_reference_block_id", "reference_parent_block_id"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "node")).strip("_") or "node"


def _norm_text(value: str) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
