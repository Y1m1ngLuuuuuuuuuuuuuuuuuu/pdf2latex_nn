"""Document-local heading style registry for the v8 decoder path.

The v8 path starts from MinerU middle.json geometry.  Individual title blocks
can be noisy, but scientific papers usually reuse a small set of heading
styles.  This module clusters heading candidates by document-local visual style
and resolves each style to a stable heading level before the render-tree stack
attaches body content.

This is decoder-side only: it does not mutate v7/v8 facts, does not enter the
GNN graph, and does not use TeX source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from src.ir import BBox, BlockType, DocumentIR, DocumentNode


HEADING_NUMBER_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+){0,3})\.?\s+(?P<title>.+)$")
REFERENCE_HEADING_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)
ABSTRACT_HEADING_RE = re.compile(r"^\s*(abstract|摘要)\s*$", re.IGNORECASE)
TOP_LEVEL_UNNUMBERED_HEADINGS = {
    "introduction",
    "related work",
    "background",
    "methodology",
    "method",
    "methods",
    "experiments",
    "experimental setup",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
}


@dataclass(frozen=True)
class HeadingStyleFeature:
    node_id: str
    text: str
    normalized_text: str
    page_idx: int
    reading_index: int
    style_id: str
    signature: tuple[Any, ...]
    font_size: float | None
    font_rank: int
    bold_ratio: float
    alignment: str
    column_span: str
    column_kind: str
    width_ratio: float | None
    center_distance_ratio: float | None
    x0_norm: float | None
    numbering_depth: int | None
    known_top_level: bool
    is_reference_heading: bool
    is_abstract_heading: bool
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadingStyleSummary:
    style_id: str
    signature: tuple[Any, ...]
    candidates: list[HeadingStyleFeature] = field(default_factory=list)
    resolved_level: int | None = None
    resolution_reason: str | None = None

    @property
    def median_font_size(self) -> float | None:
        sizes = [item.font_size for item in self.candidates if item.font_size is not None]
        return float(median(sizes)) if sizes else None

    @property
    def median_font_rank(self) -> float:
        ranks = [item.font_rank for item in self.candidates]
        return float(median(ranks)) if ranks else 99.0

    @property
    def dominant_alignment(self) -> str:
        return _mode([item.alignment for item in self.candidates], default="unknown")

    @property
    def dominant_column_span(self) -> str:
        return _mode([item.column_span for item in self.candidates], default="unknown")

    @property
    def has_known_top_level_seed(self) -> bool:
        return any(item.known_top_level or item.is_reference_heading for item in self.candidates)

    @property
    def numbering_depths(self) -> list[int]:
        return [item.numbering_depth for item in self.candidates if item.numbering_depth is not None]

    @property
    def visual_prominence(self) -> float:
        score = 0.0
        score += max(0.0, 6.0 - self.median_font_rank) * 2.0
        if self.dominant_alignment == "center":
            score += 1.5
        if self.dominant_column_span == "full_width":
            score += 1.0
        if self.has_known_top_level_seed:
            score += 4.0
        return score


class V8HeadingStyleResolver:
    """Resolve v8 title nodes through a document-local style registry."""

    def __init__(
        self,
        document: DocumentIR,
        *,
        excluded_source_ids: set[str] | None = None,
    ) -> None:
        self.document = document
        self.excluded_source_ids = set(excluded_source_ids or set())
        self.features_by_node_id: dict[str, HeadingStyleFeature] = {}
        self.styles: dict[str, HeadingStyleSummary] = {}
        self._build()

    def resolve(self, node: DocumentNode) -> tuple[int, dict[str, Any]]:
        feature = self.features_by_node_id.get(node.node_id)
        if feature is None:
            level, evidence = fallback_heading_level_from_node(node, document=self.document)
            evidence["style_registry"] = "fallback_no_candidate"
            return level, evidence
        summary = self.styles.get(feature.style_id)
        level = summary.resolved_level if summary and summary.resolved_level is not None else 2
        return min(max(int(level), 1), 3), {
            "rule": "document_local_heading_style_registry",
            "style_id": feature.style_id,
            "resolved_level": min(max(int(level), 1), 3),
            "style_resolution_reason": summary.resolution_reason if summary else None,
            "style_signature": list(feature.signature),
            "style_candidate_count": len(summary.candidates) if summary else 1,
            **feature.evidence,
        }

    def to_diagnostic(self) -> dict[str, Any]:
        return {
            "schema_version": "v8_heading_style_registry_v1",
            "style_count": len(self.styles),
            "candidate_count": len(self.features_by_node_id),
            "styles": [
                {
                    "style_id": summary.style_id,
                    "signature": list(summary.signature),
                    "resolved_level": summary.resolved_level,
                    "resolution_reason": summary.resolution_reason,
                    "candidate_count": len(summary.candidates),
                    "median_font_size": summary.median_font_size,
                    "median_font_rank": summary.median_font_rank,
                    "dominant_alignment": summary.dominant_alignment,
                    "dominant_column_span": summary.dominant_column_span,
                    "visual_prominence": round(summary.visual_prominence, 4),
                    "examples": [
                        {
                            "node_id": item.node_id,
                            "text": item.text,
                            "page_idx": item.page_idx,
                            "reading_index": item.reading_index,
                            "font_size": item.font_size,
                            "font_rank": item.font_rank,
                            "alignment": item.alignment,
                            "column_span": item.column_span,
                            "numbering_depth": item.numbering_depth,
                            "known_top_level": item.known_top_level,
                        }
                        for item in summary.candidates[:8]
                    ],
                }
                for summary in sorted(self.styles.values(), key=lambda item: item.style_id)
            ],
        }

    def _build(self) -> None:
        candidates = [
            node
            for node in sorted(self.document.nodes, key=lambda item: (item.reading_index, item.page_idx, item.node_id))
            if self._is_candidate_node(node)
        ]
        font_ranks = _font_ranks(candidates)
        for node in candidates:
            feature = _feature_for_node(node, self.document, font_rank=font_ranks.get(node.node_id, 99))
            self.features_by_node_id[node.node_id] = feature
            summary = self.styles.setdefault(
                feature.style_id,
                HeadingStyleSummary(style_id=feature.style_id, signature=feature.signature),
            )
            summary.candidates.append(feature)
        self._resolve_style_levels()

    def _is_candidate_node(self, node: DocumentNode) -> bool:
        if node.node_type != BlockType.TITLE:
            return False
        if node.node_id in self.excluded_source_ids:
            return False
        text = clean_text(node.text)
        if not text:
            return False
        if ABSTRACT_HEADING_RE.match(text):
            return False
        return True

    def _resolve_style_levels(self) -> None:
        if not self.styles:
            return

        for summary in self.styles.values():
            depths = summary.numbering_depths
            if depths:
                depth = round(float(median(depths)))
                summary.resolved_level = min(max(int(depth), 1), 3)
                summary.resolution_reason = "numbering_depth_median"
            elif summary.has_known_top_level_seed:
                summary.resolved_level = 1
                summary.resolution_reason = "known_top_level_seed"

        level1_ranks = [
            summary.median_font_rank
            for summary in self.styles.values()
            if summary.resolved_level == 1 and summary.median_font_rank < 99
        ]
        best_level1_rank = min(level1_ranks) if level1_ranks else None

        unresolved = [summary for summary in self.styles.values() if summary.resolved_level is None]
        unresolved.sort(key=lambda item: (-item.visual_prominence, item.median_font_rank, item.style_id))
        if best_level1_rank is None and unresolved:
            top = unresolved[0]
            top.resolved_level = 1
            top.resolution_reason = "most_prominent_unseeded_style"
            best_level1_rank = top.median_font_rank

        for summary in unresolved:
            if summary.resolved_level is not None:
                continue
            if best_level1_rank is not None and summary.median_font_rank >= best_level1_rank + 1:
                summary.resolved_level = 2
                summary.resolution_reason = "smaller_than_level1_style"
            elif summary.dominant_alignment == "left":
                summary.resolved_level = 2
                summary.resolution_reason = "left_aligned_body_heading_style"
            else:
                summary.resolved_level = 1
                summary.resolution_reason = "prominent_centered_style"


def build_v8_heading_style_resolver(
    document: DocumentIR,
    *,
    excluded_source_ids: set[str] | None = None,
) -> V8HeadingStyleResolver:
    return V8HeadingStyleResolver(document, excluded_source_ids=excluded_source_ids)


def fallback_heading_level_from_node(node: DocumentNode, *, document: DocumentIR | None = None) -> tuple[int, dict[str, Any]]:
    text = clean_text(node.text)
    match = HEADING_NUMBER_RE.match(text)
    if match:
        depth = len([part for part in match.group("num").split(".") if part])
        level = min(max(depth, 1), 3)
        return level, {"rule": "numbering_depth", "numbering_depth": depth, "level": level}
    normalized = normalize_heading_text(text)
    centered, centered_evidence = _alignment_for_node(node, document=document)
    if normalized in TOP_LEVEL_UNNUMBERED_HEADINGS:
        return 1, {"rule": "known_top_level_heading", "normalized_text": normalized, **centered_evidence}
    if centered == "center":
        return 1, {"rule": "document_local_centered_heading", "normalized_text": normalized, **centered_evidence}
    return 2, {"rule": "left_aligned_unnumbered_heading", "normalized_text": normalized, **centered_evidence}


def _feature_for_node(node: DocumentNode, document: DocumentIR, *, font_rank: int) -> HeadingStyleFeature:
    text = clean_text(node.text)
    normalized = normalize_heading_text(text)
    number_match = HEADING_NUMBER_RE.match(text)
    numbering_depth = None
    if number_match:
        numbering_depth = len([part for part in number_match.group("num").split(".") if part])
    alignment, evidence = _alignment_for_node(node, document=document)
    column_span = str(evidence.get("column_span") or "unknown")
    font_size = _node_font_size(node)
    bold_ratio = _node_bold_ratio(node)
    if numbering_depth is not None:
        signature: tuple[Any, ...] = ("numbered", min(max(numbering_depth, 1), 3))
    else:
        signature = (
            "visual",
            alignment,
            column_span,
            font_rank,
            "bold" if bold_ratio >= 0.50 else "regular",
        )
    style_id = "hs_" + "_".join(str(part).replace(" ", "_") for part in signature)
    return HeadingStyleFeature(
        node_id=node.node_id,
        text=text,
        normalized_text=normalized,
        page_idx=node.page_idx,
        reading_index=node.reading_index,
        style_id=style_id,
        signature=signature,
        font_size=font_size,
        font_rank=font_rank,
        bold_ratio=bold_ratio,
        alignment=alignment,
        column_span=column_span,
        column_kind=str(evidence.get("column_kind") or "unknown"),
        width_ratio=_float_or_none(evidence.get("width_ratio")),
        center_distance_ratio=_float_or_none(evidence.get("center_distance_ratio")),
        x0_norm=_float_or_none(evidence.get("x0_norm")),
        numbering_depth=numbering_depth,
        known_top_level=normalized in TOP_LEVEL_UNNUMBERED_HEADINGS,
        is_reference_heading=bool(REFERENCE_HEADING_RE.match(text)),
        is_abstract_heading=bool(ABSTRACT_HEADING_RE.match(text)),
        evidence={
            "normalized_text": normalized,
            "font_size": font_size,
            "font_rank": font_rank,
            "bold_ratio": round(bold_ratio, 4),
            "numbering_depth": numbering_depth,
            "known_top_level": normalized in TOP_LEVEL_UNNUMBERED_HEADINGS,
            **evidence,
        },
    )


def _font_ranks(nodes: list[DocumentNode]) -> dict[str, int]:
    sizes = sorted({_rounded_font_size(_node_font_size(node)) for node in nodes if _node_font_size(node)}, reverse=True)
    sizes = [size for size in sizes if size is not None]
    rank_by_size = {size: index + 1 for index, size in enumerate(sizes)}
    result: dict[str, int] = {}
    for node in nodes:
        size = _rounded_font_size(_node_font_size(node))
        result[node.node_id] = rank_by_size.get(size, 99)
    return result


def _rounded_font_size(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 2.0) / 2.0


def _node_font_size(node: DocumentNode) -> float | None:
    for key in ("font_size", "style_baseline_size", "baseline_font_size"):
        value = _float_or_none(node.features.get(key))
        if value:
            return value
    sizes = [float(span.font_size) for span in node.spans if span.font_size is not None and (span.text or "").strip()]
    return float(median(sizes)) if sizes else None


def _node_bold_ratio(node: DocumentNode) -> float:
    chars = [str(span.text or "") for span in node.spans if str(span.text or "").strip()]
    total = sum(len(text.strip()) for text in chars)
    if not total:
        return 1.0 if node.node_type == BlockType.TITLE else 0.0
    bold = sum(len(str(span.text or "").strip()) for span in node.spans if span.is_bold)
    return bold / max(total, 1)


def _alignment_for_node(node: DocumentNode, *, document: DocumentIR | None = None) -> tuple[str, dict[str, Any]]:
    bbox = node.bboxes[0] if node.bboxes else None
    page_width = page_width_for_node(node, document=document)
    if bbox is None or page_width <= 0:
        return "unknown", {"alignment_rule": "missing_bbox_or_page_width"}
    center = (bbox.x0 + bbox.x1) / 2.0
    width = max(0.0, bbox.x1 - bbox.x0)
    width_ratio = width / page_width
    column_id = node.features.get("column_id")
    if column_id in (-1, "-1") or width_ratio >= 0.50:
        target_center = page_width / 2.0
        column_kind = "full_width"
        column_span = "full_width"
    elif center < page_width * 0.52:
        target_center = page_width * 0.25
        column_kind = "left_column"
        column_span = "single_column"
    else:
        target_center = page_width * 0.75
        column_kind = "right_column"
        column_span = "single_column"
    distance = abs(center - target_center)
    threshold = max(24.0, page_width * 0.050)
    x0_norm = float(bbox.x0) / page_width
    if column_kind == "left_column":
        left_aligned_margin = x0_norm <= 0.12
    elif column_kind == "right_column":
        left_aligned_margin = abs(x0_norm - 0.515) <= 0.055
    else:
        left_aligned_margin = False
    alignment = "center" if distance <= threshold and not left_aligned_margin else "left"
    return alignment, {
        "alignment_rule": "document_local_column_center_distance",
        "column_kind": column_kind,
        "column_span": column_span,
        "bbox_center_x": round(center, 3),
        "target_center_x": round(target_center, 3),
        "center_distance": round(distance, 3),
        "center_distance_ratio": round(distance / page_width, 5),
        "center_threshold": round(threshold, 3),
        "left_aligned_margin": left_aligned_margin,
        "width_ratio": round(width_ratio, 5),
        "x0_norm": round(x0_norm, 5),
    }


def page_width_for_node(node: DocumentNode, *, document: DocumentIR | None = None) -> float:
    value = node.features.get("page_width") or node.metadata.get("page_width")
    try:
        width = float(value)
        if width > 0:
            return width
    except (TypeError, ValueError):
        pass
    if document is not None:
        for page in document.pages:
            if page.page_idx == node.page_idx and page.width > 0:
                return float(page.width)
    if node.bboxes:
        return max(1.0, float(node.bboxes[0].x1))
    return 612.0


def normalize_heading_text(text: str) -> str:
    return clean_text(text).casefold().strip(" .:")


def clean_text(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip()


def _mode(values: list[str], *, default: str) -> str:
    if not values:
        return default
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
