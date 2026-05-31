"""Caption grammar and float pairing helpers for the v8 observable fact layer.

This module is intentionally decoder-side.  It reads ``DocumentIR`` nodes and
their metadata, but it does not mutate v8 facts, GNN views, graph tensors, or
training labels.  Legacy field names such as ``source_v7_ids`` may still appear
in upstream provenance; this module treats all source ids as opaque provenance
ids for the current v8 path.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.ir import BBox, BlockType, DocumentIR, DocumentNode


CAPTION_LABEL_RE = re.compile(
    r"^\s*"
    r"(?P<label>Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?)"
    r"\s+"
    r"(?P<number>(?:S?\d+(?:\.\d+)*)(?:\([a-zA-Z0-9]+\))?|[IVXLCDM]+(?:\([a-zA-Z0-9]+\))?)"
    r"\s*"
    r"(?P<sep>[:.\-–—]|\s+)"
    r"(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)

CAPTION_REFERENCE_GUARD_RE = re.compile(
    r"^\s*(?:"
    r"(?:as\s+)?shown\s+in\s+|"
    r"see\s+|"
    r"according\s+to\s+|"
    r"in\s+|"
    r"from\s+|"
    r"using\s+|"
    r"the\s+"
    r")?(?:Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?)\s+"
    r"(?:S?\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?|[IVXLCDM]+(?:\([a-zA-Z0-9]+\))?)"
    r"\s+"
    r"(?:shows?|reports?|illustrates?|depicts?|summari[sz]es?|contains?|presents?|is\s+used|"
    r"can\s+be\s+seen|demonstrates?|lists?|compares?)\b",
    re.IGNORECASE,
)

CAPTION_SEE_GUARD_RE = re.compile(
    r"^\s*(?:see|as\s+shown\s+in|shown\s+in|according\s+to|refer\s+to)\s+"
    r"(?:Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?)\s+",
    re.IGNORECASE,
)

FLOAT_METADATA_CAPTION_KEYS = (
    "caption",
    "figure_caption",
    "figure_group_caption",
    "image_group_caption",
    "table_caption",
    "table_group_caption",
    "algorithm_caption",
    "algorithm_group_caption",
    "crop_caption",
    "caption_text",
    "detected_caption",
)

MINERU_CAPTION_CONFIDENCE_HIGH = {"strong_middle_child", "strong_content_list_field", "strong_v2_field"}
MINERU_CAPTION_ROLE_TO_TYPE = {
    "image_caption": "figure",
    "figure_caption": "figure",
    "table_caption": "table",
    "chart_caption": "chart",
    "algorithm_caption": "algorithm",
    "code_caption": "code",
}
MINERU_FOOTNOTE_ROLE_TO_TYPE = {
    "image_footnote": "image_note",
    "figure_footnote": "image_note",
    "table_footnote": "table_note",
    "chart_footnote": "chart_note",
    "code_footnote": "code_note",
    "algorithm_footnote": "code_note",
    "page_footnote": "page_note",
}


@dataclass(frozen=True)
class CaptionCandidate:
    caption_id: str
    caption_type: str
    caption_number: str
    text: str
    normalized_text: str
    source_v8_ids: list[str]
    page_idx: int | None = None
    bbox: list[float] | None = None
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    current_role: str | None = None
    current_logical_owner: str | None = None
    origin: str = "text_block"

    def dedupe_key(self) -> tuple[str, str, str, int | None]:
        subfigure_suffix = _subfigure_suffix(self.caption_number)
        number_key = f"{self.caption_number.casefold()}:{subfigure_suffix or ''}"
        return (self.caption_type, number_key, self.normalized_text, self.page_idx)


@dataclass(frozen=True)
class FloatCandidate:
    float_id: str
    float_type: str
    source_v8_ids: list[str]
    page_idx: int | None = None
    bbox: list[float] | None = None
    caption_text: str | None = None
    caption_number: str | None = None
    logical_owner: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptionPairing:
    caption: CaptionCandidate
    float_candidate: FloatCandidate | None
    confidence: float
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptionMatch:
    caption_type: str
    caption_number: str
    caption_body: str
    label: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptionEvidenceContext:
    context_id: str
    text: str
    context_kind: str
    caption_type: str = "unknown"
    footnote_type: str = "unknown"
    evidence_source: str = "regex_only"
    confidence_tier: str = "diagnostic_only"
    source_v8_ids: list[str] = field(default_factory=list)
    page_idx: int | None = None
    parent_float_id: str | None = None
    canonical_mineru_caption_id: str | None = None
    source_layers: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "text": self.text,
            "context_kind": self.context_kind,
            "caption_type": self.caption_type,
            "footnote_type": self.footnote_type,
            "evidence_source": self.evidence_source,
            "confidence_tier": self.confidence_tier,
            "source_v8_ids": list(self.source_v8_ids),
            "page_idx": self.page_idx,
            "parent_float_id": self.parent_float_id,
            "canonical_mineru_caption_id": self.canonical_mineru_caption_id,
            "source_layers": list(self.source_layers),
            "evidence": dict(self.evidence),
        }


def parse_caption_prefix(text: str) -> CaptionMatch | None:
    value = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if not value:
        return None
    if CAPTION_SEE_GUARD_RE.match(value) or CAPTION_REFERENCE_GUARD_RE.match(value):
        return None
    match = CAPTION_LABEL_RE.match(value)
    if not match:
        return None
    label = match.group("label")
    caption_type = _caption_type_from_label(label)
    number = _normalize_caption_number(match.group("number"))
    body = (match.group("body") or "").strip()
    sep = match.group("sep") or ""
    if sep.isspace() and _looks_like_reference_sentence_after_caption_number(body):
        return None
    if len(body) < 2 and caption_type != "algorithm":
        return None
    confidence = 0.86
    if sep in {":", ".", "-", "–", "—"}:
        confidence += 0.06
    if number:
        confidence += 0.04
    if _subfigure_suffix(number):
        confidence += 0.02
    return CaptionMatch(
        caption_type=caption_type,
        caption_number=number,
        caption_body=body,
        label=label,
        confidence=min(confidence, 0.98),
        evidence={
            "caption_label": label,
            "caption_separator": sep,
            "caption_number": number,
            "grammar": "v8_float_caption_v1",
        },
    )


def is_caption_like_text(text: str) -> bool:
    return parse_caption_prefix(text) is not None


def is_body_reference_text(text: str) -> bool:
    value = " ".join(str(text or "").replace("\n", " ").split()).strip()
    return bool(CAPTION_SEE_GUARD_RE.match(value) or CAPTION_REFERENCE_GUARD_RE.match(value))


def caption_evidence_contexts_from_document(document: DocumentIR) -> list[CaptionEvidenceContext]:
    contexts: list[CaptionEvidenceContext] = []
    for node in document.nodes:
        contexts.extend(caption_evidence_contexts_from_node(node))
    return contexts


def caption_evidence_contexts_from_node(node: DocumentNode) -> list[CaptionEvidenceContext]:
    metadata = node.metadata or {}
    contexts: list[CaptionEvidenceContext] = []
    caption_text = str(metadata.get("caption_text") or "").strip()
    caption_confidence = str(metadata.get("caption_confidence") or "")
    caption_role = str(metadata.get("mineru_caption_role") or metadata.get("raw_caption_type") or "")
    caption_type = str(metadata.get("caption_type") or MINERU_CAPTION_ROLE_TO_TYPE.get(caption_role, "unknown") or "unknown")
    if caption_text and caption_role:
        contexts.append(
            CaptionEvidenceContext(
                context_id=f"mineru_caption_{_safe_id(node.node_id)}",
                text=" ".join(caption_text.split()),
                context_kind="caption",
                caption_type=caption_type,
                evidence_source=_caption_evidence_source(metadata),
                confidence_tier="high" if caption_confidence in MINERU_CAPTION_CONFIDENCE_HIGH else "medium",
                source_v8_ids=[node.node_id],
                page_idx=node.page_idx,
                parent_float_id=_metadata_parent_float_id(metadata),
                canonical_mineru_caption_id=canonical_mineru_caption_id(node),
                source_layers=_metadata_source_layers(metadata, kind="caption"),
                evidence={
                    "mineru_caption_role": caption_role,
                    "caption_confidence": caption_confidence,
                    "caption_source_layer": metadata.get("caption_source_layer"),
                    "caption_source_ids": metadata.get("caption_source_ids") or [],
                    "caption_bbox": metadata.get("caption_bbox"),
                },
            )
        )
    footnote_text = str(metadata.get("footnote_text") or "").strip()
    footnote_confidence = str(metadata.get("footnote_confidence") or "")
    footnote_role = str(metadata.get("mineru_footnote_role") or metadata.get("raw_footnote_type") or "")
    if footnote_text and footnote_role:
        contexts.append(
            CaptionEvidenceContext(
                context_id=f"mineru_footnote_{_safe_id(node.node_id)}",
                text=" ".join(footnote_text.split()),
                context_kind="footnote",
                footnote_type=str(metadata.get("footnote_type") or MINERU_FOOTNOTE_ROLE_TO_TYPE.get(footnote_role, "unknown")),
                evidence_source=_footnote_evidence_source(metadata),
                confidence_tier="high" if footnote_confidence in MINERU_CAPTION_CONFIDENCE_HIGH else "medium",
                source_v8_ids=[node.node_id],
                page_idx=node.page_idx,
                parent_float_id=_metadata_parent_float_id(metadata),
                source_layers=_metadata_source_layers(metadata, kind="footnote"),
                evidence={
                    "mineru_footnote_role": footnote_role,
                    "footnote_confidence": footnote_confidence,
                    "footnote_source_layer": metadata.get("footnote_source_layer"),
                    "footnote_source_ids": metadata.get("footnote_source_ids") or [],
                    "footnote_bbox": metadata.get("footnote_bbox"),
                },
            )
        )
    match = parse_caption_prefix(node.text)
    if match is not None and not caption_text:
        contexts.append(
            CaptionEvidenceContext(
                context_id=f"regex_caption_{_safe_id(node.node_id)}",
                text=" ".join(str(node.text or "").split()),
                context_kind="caption_like_diagnostic",
                caption_type=match.caption_type,
                evidence_source="regex_only",
                confidence_tier="diagnostic_only",
                source_v8_ids=[node.node_id],
                page_idx=node.page_idx,
                evidence={"reason": "regex_caption_like_without_mineru_evidence", **match.evidence},
            )
        )
    elif is_body_reference_text(node.text):
        contexts.append(
            CaptionEvidenceContext(
                context_id=f"body_reference_{_safe_id(node.node_id)}",
                text=" ".join(str(node.text or "").split()),
                context_kind="body_reference_guard",
                evidence_source="regex_only",
                confidence_tier="diagnostic_only",
                source_v8_ids=[node.node_id],
                page_idx=node.page_idx,
                evidence={"reason": "body_reference_guard"},
            )
        )
    return contexts


def canonical_mineru_caption_id(node: DocumentNode) -> str | None:
    metadata = node.metadata or {}
    caption_text = str(metadata.get("caption_text") or "")
    if not caption_text:
        return None
    source_ids = metadata.get("caption_source_ids")
    if isinstance(source_ids, list) and source_ids:
        source_key = "|".join(sorted(str(part) for part in source_ids if str(part)))
    else:
        source_key = "|".join(sorted(str(part) for part in metadata.get("source_block_ids") or [] if str(part)))
    parent = _metadata_parent_float_id(metadata) or ""
    text_key = normalize_caption_text(caption_text, str(metadata.get("caption_type") or "unknown"))
    return f"{metadata.get('caption_type') or 'unknown'}::{parent}::{source_key}::{text_key}"


def caption_candidates_from_document(document: DocumentIR) -> list[CaptionCandidate]:
    candidates: list[CaptionCandidate] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for node in document.nodes:
        for candidate in caption_candidates_from_node(node):
            key = (candidate.origin, tuple(candidate.source_v8_ids), candidate.normalized_text)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def caption_candidates_from_node(node: DocumentNode) -> list[CaptionCandidate]:
    candidates: list[CaptionCandidate] = []
    text_match = parse_caption_prefix(node.text)
    if text_match is not None:
        candidates.append(_candidate_from_match(node, text_match, origin="text_block", text=node.text))
    for key, value in _iter_caption_metadata(node):
        metadata_text = " ".join(str(value).split())
        if not metadata_text:
            continue
        match = parse_caption_prefix(metadata_text)
        if match is None:
            inferred_kind = _caption_type_from_node(node)
            if inferred_kind is None:
                continue
            number = _caption_number_from_metadata_or_text(node, metadata_text)
            normalized = normalize_caption_text(metadata_text, inferred_kind)
            candidates.append(
                CaptionCandidate(
                    caption_id=f"cap_{_safe_id(node.node_id)}_{key}",
                    caption_type=inferred_kind,
                    caption_number=number or "",
                    text=metadata_text,
                    normalized_text=normalized,
                    source_v8_ids=[node.node_id],
                    page_idx=node.page_idx,
                    bbox=_node_bbox(node),
                    confidence=0.82,
                    evidence={"metadata_key": key, "source": "metadata_without_visible_label"},
                    current_role=node.node_type.value,
                    current_logical_owner=_logical_owner(node),
                    origin=_metadata_origin(key),
                )
            )
            continue
        candidates.append(_candidate_from_match(node, match, origin=_metadata_origin(key), text=metadata_text, metadata_key=key))
    return candidates


def float_candidates_from_document(document: DocumentIR) -> list[FloatCandidate]:
    candidates: list[FloatCandidate] = []
    for node in document.nodes:
        kind = _caption_type_from_node(node)
        if kind not in {"figure", "table", "algorithm"}:
            continue
        caption_text = _first_metadata_caption(node)
        caption_number = None
        if caption_text:
            match = parse_caption_prefix(caption_text)
            caption_number = match.caption_number if match else _caption_number_from_metadata_or_text(node, caption_text)
        candidates.append(
            FloatCandidate(
                float_id=f"float_{_safe_id(node.node_id)}",
                float_type=kind,
                source_v8_ids=[node.node_id],
                page_idx=node.page_idx,
                bbox=_node_bbox(node),
                caption_text=caption_text,
                caption_number=caption_number,
                logical_owner=_logical_owner(node),
                evidence={
                    "node_type": node.node_type.value,
                    "raw_type": node.raw_type,
                    "metadata_caption": bool(caption_text),
                },
            )
        )
    return candidates


def pair_caption_candidates(
    captions: Iterable[CaptionCandidate],
    floats: Iterable[FloatCandidate],
) -> list[CaptionPairing]:
    float_list = list(floats)
    pairings: list[CaptionPairing] = []
    for caption in captions:
        best: tuple[float, FloatCandidate, dict[str, Any]] | None = None
        for float_candidate in float_list:
            score, evidence = _pairing_score(caption, float_candidate)
            if best is None or score > best[0]:
                best = (score, float_candidate, evidence)
        if best is None or best[0] < 0.32:
            pairings.append(
                CaptionPairing(
                    caption=caption,
                    float_candidate=None,
                    confidence=caption.confidence,
                    reason="placeholder_float_needed",
                    evidence={"caption_confidence": caption.confidence},
                )
            )
        else:
            score, float_candidate, evidence = best
            pairings.append(
                CaptionPairing(
                    caption=caption,
                    float_candidate=float_candidate,
                    confidence=round(score, 4),
                    reason=evidence.get("reason", "paired_by_layout_score"),
                    evidence=evidence,
                )
            )
    return pairings


def dedupe_caption_candidates(candidates: Iterable[CaptionCandidate]) -> tuple[list[CaptionCandidate], list[dict[str, Any]]]:
    kept: list[CaptionCandidate] = []
    suppressed: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str, int | None], CaptionCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: (-item.confidence, item.caption_id)):
        key = candidate.dedupe_key()
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = candidate
            kept.append(candidate)
            continue
        if _subfigure_suffix(candidate.caption_number) and candidate.caption_number != previous.caption_number:
            kept.append(candidate)
            continue
        suppressed.append(
            {
                "caption_id": candidate.caption_id,
                "kept_caption_id": previous.caption_id,
                "dedupe_key": list(key),
                "reason": "same_type_number_text_page",
            }
        )
    kept.sort(key=lambda item: (item.page_idx if item.page_idx is not None else -1, item.caption_id))
    return kept, suppressed


def normalize_caption_text(text: str, caption_type: str | None = None) -> str:
    value = " ".join(str(text or "").casefold().split())
    value = re.sub(r"\\[a-z]+\s*\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    if caption_type:
        value = re.sub(
            rf"^\s*(?:{_caption_label_pattern(caption_type)})\s+"
            r"(?:s?\d+(?:\.\d+)*(?:\([a-z0-9]+\))?|[ivxlcdm]+(?:\([a-z0-9]+\))?)\s*[:.\-–—]?\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
    return value.strip(" \t\n\r:.-–—")


def caption_to_record(candidate: CaptionCandidate) -> dict[str, Any]:
    return {
        "caption_id": candidate.caption_id,
        "caption_type": candidate.caption_type,
        "caption_number": candidate.caption_number,
        "text": candidate.text,
        "normalized_caption_text": candidate.normalized_text,
        "source_v8_ids": list(candidate.source_v8_ids),
        "page_idx": candidate.page_idx,
        "bbox": candidate.bbox,
        "confidence": candidate.confidence,
        "evidence": dict(candidate.evidence),
        "current_role": candidate.current_role,
        "current_logical_owner": candidate.current_logical_owner,
        "origin": candidate.origin,
    }


def pairing_to_record(pairing: CaptionPairing) -> dict[str, Any]:
    return {
        "caption": caption_to_record(pairing.caption),
        "paired_float_id": pairing.float_candidate.float_id if pairing.float_candidate else None,
        "paired_float_source_v8_ids": list(pairing.float_candidate.source_v8_ids) if pairing.float_candidate else [],
        "paired_float_type": pairing.float_candidate.float_type if pairing.float_candidate else None,
        "pairing_confidence": pairing.confidence,
        "pairing_reason": pairing.reason,
        "evidence": dict(pairing.evidence),
    }


def _candidate_from_match(
    node: DocumentNode,
    match: CaptionMatch,
    *,
    origin: str,
    text: str,
    metadata_key: str | None = None,
) -> CaptionCandidate:
    normalized = normalize_caption_text(text, match.caption_type)
    evidence = dict(match.evidence)
    if metadata_key:
        evidence["metadata_key"] = metadata_key
    confidence = match.confidence
    if origin != "text_block":
        confidence = max(confidence, 0.9)
    return CaptionCandidate(
        caption_id=f"cap_{_safe_id(node.node_id)}_{origin}_{metadata_key or 'text'}",
        caption_type=match.caption_type,
        caption_number=match.caption_number,
        text=" ".join(str(text or "").split()),
        normalized_text=normalized,
        source_v8_ids=[node.node_id],
        page_idx=node.page_idx,
        bbox=_node_bbox(node),
        confidence=min(confidence, 0.99),
        evidence=evidence,
        current_role=node.node_type.value,
        current_logical_owner=_logical_owner(node),
        origin=origin,
    )


def caption_float_types_compatible(caption_type: str, float_type: str) -> bool:
    """Return whether a caption type may be attached to a rendered float type."""

    caption_value = str(caption_type or "").casefold()
    float_value = str(float_type or "").casefold()
    if caption_value in {"figure", "image", "chart"} and float_value == "figure":
        return True
    if caption_value == "table" and float_value == "table":
        return True
    if caption_value == "algorithm" and float_value == "algorithm":
        return True
    return False


def _pairing_score(caption: CaptionCandidate, float_candidate: FloatCandidate) -> tuple[float, dict[str, Any]]:
    score = 0.0
    evidence: dict[str, Any] = {}
    if caption.caption_type != "unknown" and caption_float_types_compatible(caption.caption_type, float_candidate.float_type):
        score += 0.42
        evidence["type_match"] = True
    elif caption.caption_type != "unknown":
        score -= 0.55
        evidence["type_mismatch"] = True
    if caption.caption_number and float_candidate.caption_number:
        if caption.caption_number.casefold() == float_candidate.caption_number.casefold():
            score += 0.28
            evidence["number_match"] = True
        else:
            score -= 0.08
            evidence["number_mismatch"] = True
    if caption.page_idx is not None and float_candidate.page_idx is not None:
        page_delta = abs(caption.page_idx - float_candidate.page_idx)
        evidence["page_delta"] = page_delta
        if page_delta == 0:
            score += 0.18
        elif page_delta == 1:
            score += 0.04
        else:
            score -= min(0.2, page_delta * 0.04)
    cbox = _bbox_from_list(caption.bbox)
    fbox = _bbox_from_list(float_candidate.bbox)
    if cbox and fbox:
        x_overlap = _x_overlap_ratio(cbox, fbox)
        vertical_gap = _vertical_gap(cbox, fbox)
        evidence["x_overlap"] = round(x_overlap, 4)
        evidence["vertical_gap"] = round(vertical_gap, 4)
        evidence["caption_below_float"] = cbox.y0 >= fbox.y1
        evidence["caption_above_float"] = cbox.y1 <= fbox.y0
        score += min(x_overlap, 1.0) * 0.16
        if vertical_gap <= 80:
            score += 0.14
        elif vertical_gap <= 180:
            score += 0.07
        else:
            score -= min(0.16, (vertical_gap - 180) / 1000)
    if caption.current_logical_owner and caption.current_logical_owner == float_candidate.logical_owner:
        score += 0.12
        evidence["logical_owner_match"] = True
    if score >= 0.72:
        evidence["reason"] = "paired_by_type_number_layout"
    elif score >= 0.45:
        evidence["reason"] = "paired_by_type_layout"
    else:
        evidence["reason"] = "low_pairing_score"
    return max(0.0, min(1.0, score)), evidence


def _caption_type_from_label(label: str) -> str:
    value = str(label or "").casefold().rstrip(".")
    if value in {"figure", "fig"}:
        return "figure"
    if value in {"table", "tab"}:
        return "table"
    if value in {"algorithm", "alg"}:
        return "algorithm"
    return "unknown"


def _caption_type_from_node(node: DocumentNode) -> str | None:
    if node.node_type == BlockType.FIGURE:
        return "figure"
    if node.node_type == BlockType.TABLE:
        return "table"
    if node.node_type == BlockType.ALGORITHM:
        return "algorithm"
    role_values = [
        node.raw_type,
        node.metadata.get("layout_role"),
        node.metadata.get("canonical_type"),
        node.metadata.get("role"),
        node.metadata.get("type"),
        node.metadata.get("float_type"),
    ]
    joined = " ".join(str(value or "").casefold() for value in role_values)
    if "figure" in joined or "image" in joined or "chart" in joined:
        return "figure"
    if "table" in joined:
        return "table"
    if "algorithm" in joined or "alg" == joined.strip():
        return "algorithm"
    return None


def _caption_label_pattern(caption_type: str) -> str:
    if caption_type == "figure":
        return r"figure|fig\.?"
    if caption_type == "table":
        return r"table|tab\.?"
    if caption_type == "algorithm":
        return r"algorithm|alg\.?"
    return r"figure|fig\.?|table|tab\.?|algorithm|alg\.?"


def _looks_like_reference_sentence_after_caption_number(body: str) -> bool:
    return bool(
        re.match(
            r"^(?:shows?|reports?|illustrates?|depicts?|is\s+used|can\s+be\s+seen|demonstrates?)\b",
            str(body or "").strip(),
            flags=re.IGNORECASE,
        )
    )


def _normalize_caption_number(number: str) -> str:
    return str(number or "").strip().rstrip(".")


def _subfigure_suffix(number: str) -> str | None:
    match = re.search(r"\(([a-zA-Z0-9]+)\)\s*$", str(number or ""))
    return match.group(1).casefold() if match else None


def _iter_caption_metadata(node: DocumentNode) -> Iterable[tuple[str, Any]]:
    for key in FLOAT_METADATA_CAPTION_KEYS:
        value = node.metadata.get(key)
        if isinstance(value, str) and value.strip():
            yield key, value
    nested = node.metadata.get("metadata")
    if isinstance(nested, dict):
        for key in FLOAT_METADATA_CAPTION_KEYS:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                yield f"metadata.{key}", value


def _metadata_origin(key: str) -> str:
    value = key.casefold()
    if "crop" in value:
        return "crop_metadata"
    if "caption" in value:
        return "caption_metadata"
    return "float_metadata"


def _caption_evidence_source(metadata: dict[str, Any]) -> str:
    layer = str(metadata.get("caption_source_layer") or "").casefold()
    role = str(metadata.get("mineru_caption_role") or metadata.get("raw_caption_type") or "")
    if role and layer == "middle":
        return "mineru_middle_child"
    if role and layer == "content_list":
        return "content_list_field"
    if role and layer == "content_list_v2":
        return "content_list_v2_field"
    if role:
        return "document_ir_caption_metadata"
    return "regex_only"


def _footnote_evidence_source(metadata: dict[str, Any]) -> str:
    layer = str(metadata.get("footnote_source_layer") or "").casefold()
    role = str(metadata.get("mineru_footnote_role") or metadata.get("raw_footnote_type") or "")
    if role and layer == "middle":
        return "mineru_middle_child"
    if role and layer == "content_list":
        return "content_list_field"
    if role and layer == "content_list_v2":
        return "content_list_v2_field"
    if role:
        return "document_ir_caption_metadata"
    return "regex_only"


def _metadata_parent_float_id(metadata: dict[str, Any]) -> str | None:
    for key in ("caption_parent_float_id", "footnote_parent_float_id", "parent_float_source_id", "parent_block_id"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _metadata_source_layers(metadata: dict[str, Any], *, kind: str) -> list[str]:
    keys = ("caption_source_layer", "source_layer_hierarchy") if kind == "caption" else ("footnote_source_layer", "source_layer_hierarchy")
    layers: list[str] = []
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, list):
            layers.extend(str(part) for part in value if str(part))
        elif value:
            layers.extend(str(value).split(","))
    return list(dict.fromkeys(layer.strip() for layer in layers if layer.strip()))


def _first_metadata_caption(node: DocumentNode) -> str | None:
    for _key, value in _iter_caption_metadata(node):
        text = " ".join(str(value).split()).strip()
        if text:
            return text
    return None


def _caption_number_from_metadata_or_text(node: DocumentNode, text: str) -> str | None:
    for key in ("caption_number", "figure_number", "table_number", "algorithm_number", "number"):
        value = node.metadata.get(key)
        if value not in (None, ""):
            return str(value).strip()
    match = parse_caption_prefix(text)
    return match.caption_number if match else None


def _logical_owner(node: DocumentNode) -> str | None:
    for key in ("logical_owner", "logical_owner_id", "owner_id", "v8_logical_owner", "contentlist_owner_id"):
        value = node.metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _node_bbox(node: DocumentNode) -> list[float] | None:
    if node.bboxes:
        x0 = min(box.x0 for box in node.bboxes)
        y0 = min(box.y0 for box in node.bboxes)
        x1 = max(box.x1 for box in node.bboxes)
        y1 = max(box.y1 for box in node.bboxes)
        return [x0, y0, x1, y1]
    raw = node.metadata.get("bbox")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
        except (TypeError, ValueError):
            return None
    return None


def _bbox_from_list(values: list[float] | None) -> BBox | None:
    if values is None:
        return None
    try:
        return BBox.from_list(values)
    except Exception:
        return None


def _x_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    denom = max(1e-6, min(a.x1 - a.x0, b.x1 - b.x0))
    return overlap / denom


def _vertical_gap(a: BBox, b: BBox) -> float:
    if a.y0 >= b.y1:
        return a.y0 - b.y1
    if b.y0 >= a.y1:
        return b.y0 - a.y1
    return 0.0


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "node")).strip("_") or "node"


def _safe_number(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)
