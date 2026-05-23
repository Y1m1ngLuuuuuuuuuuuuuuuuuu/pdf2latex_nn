"""V8-specific document style detector.

The generic :mod:`style_profile` extractor works from stable DocumentIR.  V8
adds one more source of evidence: raw MinerU middle lines preserved as
``source_lines`` metadata.  This module keeps the public renderer contract the
same by producing a normal ``StyleProfile`` plus a diagnostics JSON.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from statistics import median
from typing import Any, Iterable

from src.generation.style_profile import StyleProfileExtractor
from src.ir import BBox, BlockType, DocumentIR, DocumentNode, RenderTreeIR, StyleProfile


@dataclass(frozen=True)
class V8StyleDetectorConfig:
    """Tunable thresholds for conservative v8 style detection."""

    min_line_pitch_samples: int = 8
    min_paragraph_gap_samples: int = 5
    min_indent_samples: int = 5
    max_line_pitch_multiplier: float = 1.8
    min_line_pitch_multiplier: float = 1.05
    max_paragraph_spacing_multiplier: float = 1.5
    max_indent_pt: float = 36.0


class V8StyleDetector:
    """Estimate page typography/rhythm from v8 DocumentIR line evidence."""

    def __init__(self, config: V8StyleDetectorConfig | None = None) -> None:
        self.config = config or V8StyleDetectorConfig()
        self.base_extractor = StyleProfileExtractor()

    def detect(self, document: DocumentIR, *, tree: RenderTreeIR | None = None) -> tuple[StyleProfile, dict[str, Any]]:
        base = self.base_extractor.extract(document)
        body_nodes = _body_text_nodes(document)
        body_font_size = _float_or_none((base.renderer_options or {}).get("body_font_size"))
        if body_font_size is None:
            body_font_size = _estimate_body_font_size_from_nodes(body_nodes)

        line_pitch = self._estimate_line_pitch(body_nodes, body_font_size)
        paragraph_spacing = self._estimate_paragraph_spacing(body_nodes, body_font_size)
        paragraph_indent = self._estimate_paragraph_indent(body_nodes)
        heading_role_styles, heading_diag = self._heading_role_styles_from_tree(tree)

        renderer_options = dict(base.renderer_options or {})
        if body_font_size is not None:
            renderer_options["body_font_size"] = body_font_size
        if line_pitch["value"] is not None:
            renderer_options["body_line_height"] = line_pitch["value"]
        if paragraph_spacing["value"] is not None:
            renderer_options["paragraph_spacing"] = paragraph_spacing["value"]
        if paragraph_indent["value"] is not None:
            renderer_options["paragraph_indent"] = paragraph_indent["value"]
        renderer_options["v8_style_detector"] = {
            "schema_version": "v8_style_detector_v1",
            "line_pitch_source": line_pitch["source"],
            "paragraph_spacing_source": paragraph_spacing["source"],
            "paragraph_indent_source": paragraph_indent["source"],
        }
        renderer_options["source"] = "v8_style_detector"

        role_styles = dict(base.role_styles or {})
        for role, style in heading_role_styles.items():
            merged = dict(role_styles.get(role) or {})
            merged.update(style)
            role_styles[role] = merged

        metadata = dict(base.metadata or {})
        metadata["v8_style_detector"] = "v1"
        style = replace(
            base,
            profile_id="v8_original_like",
            role_styles=role_styles,
            renderer_options=renderer_options,
            metadata=metadata,
        )
        diagnostics = {
            "schema_version": "v8_style_detector_diagnostics_v1",
            "doc_id": document.doc_id,
            "body_node_count": len(body_nodes),
            "body_font_size": body_font_size,
            "body_line_height": line_pitch,
            "paragraph_spacing": paragraph_spacing,
            "paragraph_indent": paragraph_indent,
            "heading_styles": heading_diag,
            "column_mode": (base.renderer_options or {}).get("column_mode"),
            "column_gap_pt": (base.renderer_options or {}).get("column_gap_pt"),
            "page_layout": base.page_layout,
        }
        return style, diagnostics

    def _estimate_line_pitch(self, nodes: list[DocumentNode], body_font_size: float | None) -> dict[str, Any]:
        y_deltas: list[float] = []
        height_estimates: list[float] = []
        for node in nodes:
            lines = _line_records(node)
            by_page: dict[int, list[BBox]] = defaultdict(list)
            for line in lines:
                bbox = _bbox_from_line_record(line)
                page_idx = _int_or_none(line.get("page_idx")) if isinstance(line, dict) else None
                if bbox is None:
                    continue
                by_page[page_idx if page_idx is not None else node.page_idx].append(bbox)
                height_estimates.append(max(bbox.y1 - bbox.y0, 0.0))
            for boxes in by_page.values():
                boxes = sorted(boxes, key=lambda box: (box.y0, box.x0))
                for prev, curr in zip(boxes, boxes[1:], strict=False):
                    delta = curr.y0 - prev.y0
                    if _reasonable_line_pitch(delta, body_font_size):
                        y_deltas.append(delta)
            if len(lines) >= 2 and node.bboxes:
                box = node.bboxes[0]
                approx = (box.y1 - box.y0) / max(len(lines), 1)
                if _reasonable_line_pitch(approx, body_font_size):
                    y_deltas.append(approx)
        value = _safe_median(y_deltas)
        source = "middle_line_y_delta" if len(y_deltas) >= self.config.min_line_pitch_samples else "fallback"
        if value is None and body_font_size is not None:
            value = body_font_size * 1.2
            source = "font_size_multiplier"
        if value is not None and body_font_size is not None:
            value = min(
                max(value, body_font_size * self.config.min_line_pitch_multiplier),
                body_font_size * self.config.max_line_pitch_multiplier,
            )
        return {
            "value": value,
            "source": source,
            "sample_count": len(y_deltas),
            "line_height_sample_count": len(height_estimates),
        }

    def _estimate_paragraph_spacing(self, nodes: list[DocumentNode], body_font_size: float | None) -> dict[str, Any]:
        gaps: list[float] = []
        sorted_nodes = sorted(nodes, key=lambda node: (node.page_idx, _column_id(node), node.reading_index))
        for prev, curr in zip(sorted_nodes, sorted_nodes[1:], strict=False):
            if prev.page_idx != curr.page_idx or _column_id(prev) != _column_id(curr):
                continue
            if not prev.bboxes or not curr.bboxes:
                continue
            gap = curr.bboxes[0].y0 - prev.bboxes[-1].y1
            max_gap = (body_font_size or 10.0) * 4.0
            if 0.0 <= gap <= max_gap:
                gaps.append(gap)
        raw_value = _safe_median(gaps)
        value = raw_value
        # Text bboxes do not include the full TeX line box, so raw inter-block
        # white space contains some normal line leading.  Use only the excess as
        # global parskip; otherwise two-column papers become too loose.
        if value is not None and body_font_size is not None:
            value = max(0.0, value - body_font_size * 0.4)
        source = "same_column_block_gap" if len(gaps) >= self.config.min_paragraph_gap_samples else "fallback"
        if value is not None and body_font_size is not None:
            value = min(value, body_font_size * min(self.config.max_paragraph_spacing_multiplier, 0.65))
        return {
            "value": value,
            "raw_value": raw_value,
            "source": source,
            "sample_count": len(gaps),
        }

    def _estimate_paragraph_indent(self, nodes: list[DocumentNode]) -> dict[str, Any]:
        by_scope: dict[tuple[int, int | None], list[float]] = defaultdict(list)
        for node in nodes:
            if not node.bboxes:
                continue
            by_scope[(node.page_idx, _column_id(node))].append(node.bboxes[0].x0)
        baselines = {
            scope: _quantile(values, 0.1)
            for scope, values in by_scope.items()
            if len(values) >= 3
        }
        indents: list[float] = []
        for node in nodes:
            if not node.bboxes:
                continue
            baseline = baselines.get((node.page_idx, _column_id(node)))
            if baseline is None:
                continue
            indent = node.bboxes[0].x0 - baseline
            if 1.0 <= indent <= self.config.max_indent_pt:
                indents.append(indent)
        value = _safe_median(indents)
        source = "column_left_delta" if len(indents) >= self.config.min_indent_samples else "fallback"
        if value is None:
            value = 0.0
        return {
            "value": value,
            "source": source,
            "sample_count": len(indents),
        }

    def _heading_role_styles_from_tree(self, tree: RenderTreeIR | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        if tree is None:
            return {}, {"source": "missing_tree", "styles": []}
        registry = tree.metadata.get("heading_style_registry")
        if not isinstance(registry, dict):
            return {}, {"source": "missing_heading_style_registry", "styles": []}
        role_by_level = {1: "section", 2: "subsection", 3: "subsubsection"}
        styles_by_role: dict[str, dict[str, Any]] = {}
        diagnostics: list[dict[str, Any]] = []
        for style in registry.get("styles") or []:
            if not isinstance(style, dict):
                continue
            level = _int_or_none(style.get("resolved_level"))
            role = role_by_level.get(level)
            if role is None:
                continue
            font_size = _float_or_none(style.get("median_font_size"))
            if font_size is None:
                continue
            styles_by_role[role] = {
                "font_size": font_size,
                "relative_font_size": style.get("relative_to_body"),
                "bold": bool(style.get("dominant_bold", True)),
                "alignment": style.get("dominant_alignment"),
                "v8_heading_style_id": style.get("style_id"),
            }
            diagnostics.append(
                {
                    "role": role,
                    "style_id": style.get("style_id"),
                    "font_size": font_size,
                    "alignment": style.get("dominant_alignment"),
                    "candidate_count": style.get("candidate_count"),
                }
            )
        return styles_by_role, {"source": "heading_style_registry", "styles": diagnostics}


def detect_v8_style(document: DocumentIR, *, tree: RenderTreeIR | None = None) -> tuple[StyleProfile, dict[str, Any]]:
    """Convenience wrapper used by v8 scripts."""

    return V8StyleDetector().detect(document, tree=tree)


def _body_text_nodes(document: DocumentIR) -> list[DocumentNode]:
    body_types = {BlockType.TEXT, BlockType.LIST}
    nodes: list[DocumentNode] = []
    for node in document.nodes:
        if node.node_type not in body_types:
            continue
        if _metadata_role(node) in {
            "document_title",
            "paper_title",
            "author",
            "authors",
            "author_block",
            "affiliation",
            "email",
            "abstract_title",
            "page_header",
            "page_footer",
            "page_number",
            "header",
            "footer",
        }:
            continue
        if len(node.text.strip()) < 12:
            continue
        nodes.append(node)
    return nodes


def _line_records(node: DocumentNode) -> list[dict[str, Any]]:
    value = node.metadata.get("source_lines")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    records: list[dict[str, Any]] = []
    for index, span in enumerate(node.spans):
        if span.bbox is None:
            continue
        records.append(
            {
                "line_id": f"{node.node_id}:span:{index}",
                "text": span.text,
                "page_idx": node.page_idx,
                "bbox": span.bbox.to_list(),
            }
        )
    return records


def _bbox_from_line_record(record: dict[str, Any]) -> BBox | None:
    bbox = record.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        return BBox.from_list([float(value) for value in bbox])
    except (TypeError, ValueError):
        return None


def _estimate_body_font_size_from_nodes(nodes: Iterable[DocumentNode]) -> float | None:
    weighted: Counter[float] = Counter()
    for node in nodes:
        for span in node.spans:
            if span.font_size is None:
                continue
            weighted[round(float(span.font_size) * 4.0) / 4.0] += max(len(span.text.strip()), 1)
        fallback = _float_or_none(node.features.get("font_size")) or _float_or_none(node.features.get("style_baseline_size"))
        if fallback is not None:
            weighted[round(fallback * 4.0) / 4.0] += max(len(node.text.strip()), 1)
    if not weighted:
        return None
    return float(weighted.most_common(1)[0][0])


def _reasonable_line_pitch(value: float, body_font_size: float | None) -> bool:
    if value <= 0.0:
        return False
    if body_font_size is None:
        return 4.0 <= value <= 36.0
    return body_font_size * 0.6 <= value <= body_font_size * 2.4


def _column_id(node: DocumentNode) -> int | None:
    value = node.features.get("column_id", node.metadata.get("column_id"))
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _metadata_role(node: DocumentNode) -> str:
    for key in ("layout_role", "canonical_type", "raw_type", "type"):
        value = node.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    return ""


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _safe_median(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values if value is not None]
    if not data:
        return None
    return float(median(data))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return float(ordered[index])
