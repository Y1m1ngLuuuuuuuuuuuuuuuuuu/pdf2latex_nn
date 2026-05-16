"""Heading evidence and document-local heading style profiling.

This module is deliberately independent from the LaTeX generator.  It only
summarizes layout signals that the decoder can use to build a deterministic
heading stack before local GNN relations are applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.reasoning.layout_state_machine import (
    canonical_type,
    heading_prefix,
    infer_body_font_size,
    is_noise_record,
    layout_role,
    node_font_size,
    record_text,
)


@dataclass(frozen=True)
class HeadingEvidence:
    node_id: int
    text: str
    font_size: float
    relative_font_size: float
    is_bold: bool
    mineru_type: str
    layout_role: str
    numbering_style: str
    numbering_level: int | None
    is_line_isolated: bool
    vertical_gap_before: float
    vertical_gap_after: float
    score: float
    inferred_level: int | None


@dataclass(frozen=True)
class HeadingStyleProfile:
    body_font_size: float
    heading_font_clusters: tuple[float, ...]
    numbering_styles: tuple[str, ...]
    level_by_font_cluster: dict[float, int] = field(default_factory=dict)
    level_by_numbering_pattern: dict[str, int] = field(default_factory=dict)


def collect_heading_evidence(
    records_by_id: dict[int, dict[str, Any]],
    *,
    text_by_id: dict[int, str] | None = None,
) -> dict[int, HeadingEvidence]:
    """Collect per-node heading evidence without mutating records."""

    text_by_id = text_by_id or {}
    body_font_size = infer_body_font_size(records_by_id.values())
    evidence: dict[int, HeadingEvidence] = {}
    for node_id, record in records_by_id.items():
        text = text_by_id.get(node_id) or record_text(record)
        font_size = node_font_size(record)
        relative = font_size / body_font_size if body_font_size > 0 and font_size > 0 else 0.0
        raw_type = str(record.get("type") or record.get("raw_type") or record.get("block_type") or "").casefold()
        role = layout_role(record)
        numbering_style, numbering_level = heading_prefix(text)
        isolated = is_line_isolated(record)
        gap_before = numeric_record_value(record, "vertical_gap_before", "gap_before", "space_before")
        gap_after = numeric_record_value(record, "vertical_gap_after", "gap_after", "space_after")
        bold = record_is_bold(record)
        score = heading_score(
            record,
            text=text,
            relative_font_size=relative,
            is_bold=bold,
            numbering_level=numbering_level,
            is_line_isolated=isolated,
            vertical_gap_before=gap_before,
            vertical_gap_after=gap_after,
        )
        evidence[node_id] = HeadingEvidence(
            node_id=node_id,
            text=text,
            font_size=font_size,
            relative_font_size=relative,
            is_bold=bold,
            mineru_type=raw_type or canonical_type(record),
            layout_role=role,
            numbering_style=numbering_style,
            numbering_level=numbering_level,
            is_line_isolated=isolated,
            vertical_gap_before=gap_before,
            vertical_gap_after=gap_after,
            score=score,
            inferred_level=numbering_level,
        )
    return evidence


def learn_heading_style_profile(
    evidence_by_id: dict[int, HeadingEvidence],
    *,
    body_font_size: float | None = None,
) -> HeadingStyleProfile:
    """Learn document-local heading clusters from high-confidence evidence."""

    if body_font_size is None:
        font_sizes = [item.font_size for item in evidence_by_id.values() if item.font_size > 0]
        body_font_size = min(font_sizes) if font_sizes else 0.0
    cluster_scores: dict[float, float] = {}
    numbering_styles: dict[str, int] = {}
    level_by_numbering: dict[str, int] = {}
    for evidence in evidence_by_id.values():
        if evidence.score < 1.25:
            continue
        if evidence.relative_font_size >= 1.03 or evidence.mineru_type in {"title", "section", "subsection"}:
            cluster = round(evidence.relative_font_size * 20.0) / 20.0
            cluster_scores[cluster] = max(cluster_scores.get(cluster, 0.0), evidence.score)
        if evidence.numbering_style not in {"empty", "freeform", "custom_colon"}:
            numbering_styles[evidence.numbering_style] = numbering_styles.get(evidence.numbering_style, 0) + 1
            if evidence.numbering_level is not None:
                level_by_numbering.setdefault(evidence.numbering_style, evidence.numbering_level)

    clusters = tuple(
        sorted(cluster_scores, key=lambda cluster: (-cluster_scores[cluster], -cluster))
    )
    level_by_cluster = {
        cluster: index + 1
        for index, cluster in enumerate(sorted(clusters, reverse=True)[:5])
    }
    return HeadingStyleProfile(
        body_font_size=float(body_font_size or 0.0),
        heading_font_clusters=clusters,
        numbering_styles=tuple(sorted(numbering_styles, key=lambda key: (-numbering_styles[key], key))),
        level_by_font_cluster=level_by_cluster,
        level_by_numbering_pattern=level_by_numbering,
    )


def heading_score(
    record: dict[str, Any],
    *,
    text: str,
    relative_font_size: float,
    is_bold: bool,
    numbering_level: int | None,
    is_line_isolated: bool,
    vertical_gap_before: float,
    vertical_gap_after: float,
) -> float:
    if is_noise_record(record, text):
        return -5.0
    score = 0.0
    if canonical_type(record) == "title":
        score += 1.5
    if layout_role(record) == "heading":
        score += 1.0
    if relative_font_size >= 1.15:
        score += 1.2
    elif relative_font_size >= 1.05:
        score += 0.5
    if is_bold:
        score += 0.4
    if numbering_level is not None:
        score += 0.9
    if is_line_isolated:
        score += 0.4
    if vertical_gap_before > 0 or vertical_gap_after > 0:
        score += min(0.5, (vertical_gap_before + vertical_gap_after) / 48.0)
    if len(" ".join(text.split())) > 180:
        score -= 1.0
    return score


def is_line_isolated(record: dict[str, Any]) -> bool:
    if bool(record.get("layout_is_band_boundary")):
        return True
    if str(record.get("layout_band_type") or "").casefold() in {"full_span", "single_column"}:
        return True
    line_count = record.get("line_count")
    if isinstance(line_count, (int, float)):
        return int(line_count) <= 1
    return False


def numeric_record_value(record: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def record_is_bold(record: dict[str, Any]) -> bool:
    if isinstance(record.get("is_bold"), bool):
        return bool(record["is_bold"])
    spans = record.get("style_spans")
    if not isinstance(spans, list):
        return False
    total = 0
    bold = 0
    for span in spans:
        if not isinstance(span, dict):
            continue
        text = str(span.get("text") or "")
        count = int(span.get("char_count") or len(text) or 1)
        total += max(1, count)
        if bool(span.get("is_bold")):
            bold += max(1, count)
    return total > 0 and bold / total >= 0.5
