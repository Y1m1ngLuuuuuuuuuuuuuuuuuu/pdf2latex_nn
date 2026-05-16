"""Extract document-level style profiles from frontend IR.

This module is intentionally statistical and conservative.  It estimates the
global layout choices that make one paper look different from another, while
leaving local span styling to the renderer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from statistics import median
from typing import Iterable

from src.generation.font_resolver import build_latex_font_setup, canonicalize_pdf_font_name, resolve_pdf_font
from src.ir import BlockType, BBox, DocumentIR, DocumentNode, RendererMode, StyleProfile


@dataclass(frozen=True)
class StyleProfileExtractorConfig:
    profile_id: str = "original_like"
    renderer_mode: RendererMode = RendererMode.ORIGINAL_LIKE
    min_body_span_chars: int = 3
    two_column_min_pages_ratio: float = 0.35
    stable_two_column_min_pages_ratio: float = 0.80
    full_width_threshold: float = 0.65
    header_footer_zone_ratio: float = 0.18
    header_footer_repeat_min_pages: int = 2
    header_footer_repeat_min_page_ratio: float = 0.35
    page_number_min_page_ratio: float = 0.50
    output_paper: str = "a4"
    output_paper_width_pt: float = 595.28
    output_paper_height_pt: float = 841.89
    body_font_bucket_pt: float = 0.25


class StyleProfileExtractor:
    """Build a StyleProfile sidecar from DocumentIR."""

    def __init__(self, config: StyleProfileExtractorConfig | None = None) -> None:
        self.config = config or StyleProfileExtractorConfig()

    def extract(self, document: DocumentIR) -> StyleProfile:
        page_layout = self._extract_page_layout(document)
        body_font_size = self._estimate_body_font_size(document)
        body_font_family = self._estimate_body_font_family(document)
        role_styles = self._extract_role_styles(document, body_font_size, body_font_family)
        role_fonts = {
            role: str(style.get("font_family"))
            for role, style in role_styles.items()
            if isinstance(style, dict) and style.get("font_family")
        }
        body_font_info = resolve_pdf_font(body_font_family)
        font_clusters = self._extract_font_clusters(document, body_font_size)
        paragraph_spacing = self._normalize_paragraph_spacing(
            self._estimate_paragraph_spacing(document),
            page_layout,
            body_font_size,
        )
        paragraph_indent = self._scaled_horizontal_length(
            self._estimate_paragraph_indent(document, page_layout),
            page_layout,
        )
        display_spacing = self._normalize_vertical_spacing_dict(
            self._estimate_display_spacing(document, body_font_size),
            page_layout,
            max_multiplier=2.0,
            body_font_size=body_font_size,
        )
        list_spacing = self._normalize_vertical_spacing_dict(
            self._estimate_list_spacing(document, body_font_size),
            page_layout,
            max_multiplier=1.2,
            body_font_size=body_font_size,
        )
        renderer_options = {
            "body_font_size": body_font_size,
            "body_font_family": body_font_family,
            "body_font_class": body_font_info.font_class if body_font_info else None,
            "body_text_width": page_layout.get("body_text_width"),
            "body_column_width": page_layout.get("body_column_width"),
            "body_text_region_width": page_layout.get("body_text_region_width"),
            "font_clusters": font_clusters["global"],
            "role_font_clusters": font_clusters["by_role"],
            "paragraph_indent": paragraph_indent,
            "paragraph_spacing": paragraph_spacing,
            "display_math_spacing": display_spacing,
            "list_spacing": list_spacing,
            "bibliography": self._extract_bibliography_style(document, body_font_size, page_layout),
            "header_footer": self._extract_header_footer_style(document),
            "column_mode": page_layout.get("column_mode"),
            "column_gap": page_layout.get("column_gap"),
            "column_gap_pt": self._scaled_horizontal_length(page_layout.get("column_gap"), page_layout),
            "mixed_column_strategy": "multicols_by_layout_band" if page_layout.get("column_mode") == "mixed" else None,
            "geometry_options": self._geometry_options_from_layout(page_layout),
            "font_setup": build_latex_font_setup(body_font_family, role_fonts),
            "source": "statistical_document_ir",
        }
        packages = ["amsmath", "amssymb", "graphicx", "float", "booktabs", "hyperref", "geometry", "setspace", "enumitem", "titlesec"]
        if page_layout.get("column_mode") == "mixed":
            packages.append("multicol")
        documentclass_options: list[str] = []
        if page_layout.get("column_mode") == "two_column":
            documentclass_options.append("twocolumn")

        return StyleProfile(
            profile_id=self.config.profile_id,
            mode=self.config.renderer_mode,
            documentclass="article",
            documentclass_options=documentclass_options,
            packages=packages,
            page_layout=page_layout,
            role_styles=role_styles,
            renderer_options=renderer_options,
            learned_from=[document.doc_id],
            metadata={
                "doc_id": document.doc_id,
                "extractor": "StyleProfileExtractor",
                "extractor_version": "v1",
            },
        )

    def _extract_page_layout(self, document: DocumentIR) -> dict[str, object]:
        page_sizes = [(page.width, page.height) for page in document.pages]
        widths = [width for width, _height in page_sizes if width > 0]
        heights = [height for _width, height in page_sizes if height > 0]
        page_width = _safe_median(widths, 1000.0)
        page_height = _safe_median(heights, 1000.0)

        page_nodes = _nodes_by_page(document.nodes)
        margins_by_page: list[dict[str, float]] = []
        column_counts: list[int] = []
        column_gaps: list[float] = []
        mixed_band_pages = 0
        text_widths: list[float] = []
        text_heights: list[float] = []
        body_text_widths: list[float] = []
        body_text_region_widths: list[float] = []
        body_column_widths: list[float] = []
        body_margins_by_page: list[dict[str, float]] = []
        for page in document.pages:
            nodes = page_nodes.get(page.page_idx, [])
            boxes = [box for node in nodes if _is_layout_node(node) for box in node.bboxes]
            if not boxes:
                continue
            min_x = min(box.x0 for box in boxes)
            max_x = max(box.x1 for box in boxes)
            min_y = min(box.y0 for box in boxes)
            max_y = max(box.y1 for box in boxes)
            text_widths.append(max(max_x - min_x, 0.0))
            text_heights.append(max(max_y - min_y, 0.0))
            margins_by_page.append(
                {
                    "left": max(min_x, 0.0),
                    "right": max(page.width - max_x, 0.0),
                    "top": max(min_y, 0.0),
                    "bottom": max(page.height - max_y, 0.0),
                }
            )
            count, gap = _estimate_page_columns(nodes, page.width, self.config.full_width_threshold)
            column_counts.append(count)
            if gap is not None:
                column_gaps.append(gap)
            if count >= 2 and _page_has_full_width_blocker(nodes, page.width, self.config.full_width_threshold):
                mixed_band_pages += 1
            body_geometry = _estimate_page_body_text_geometry(nodes, page.width, self.config.full_width_threshold)
            if body_geometry:
                body_text_width = _float_or_none(body_geometry.get("text_width"))
                body_region_width = _float_or_none(body_geometry.get("text_region_width"))
                body_margins = body_geometry.get("margins")
                if body_text_width:
                    body_text_widths.append(body_text_width)
                if body_region_width:
                    body_text_region_widths.append(body_region_width)
                if isinstance(body_margins, dict):
                    body_margins_by_page.append(body_margins)
                for value in body_geometry.get("column_widths", []):
                    width = _float_or_none(value)
                    if width:
                        body_column_widths.append(width)

        column_count = _dominant_int(column_counts, default=1)
        two_column_ratio = 0.0
        if column_counts:
            two_column_ratio = sum(1 for count in column_counts if count >= 2) / len(column_counts)
            if two_column_ratio >= self.config.two_column_min_pages_ratio:
                column_count = max(column_count, 2)
        column_mode = "single"
        mixed_band_ratio = (mixed_band_pages / len(column_counts)) if column_counts else 0.0
        if two_column_ratio >= self.config.stable_two_column_min_pages_ratio:
            column_mode = "mixed" if mixed_band_pages else "two_column"
        elif two_column_ratio >= self.config.two_column_min_pages_ratio:
            column_mode = "mixed"
        elif mixed_band_ratio >= self.config.two_column_min_pages_ratio:
            column_mode = "mixed"

        margins = _median_margins(margins_by_page)
        body_margins = _median_margins(body_margins_by_page) if body_margins_by_page else {}
        column_gap = _safe_median(column_gaps, 0.0)
        return {
            "page_width": page_width,
            "page_height": page_height,
            "aspect_ratio": (page_width / page_height) if page_height else None,
            "margins": margins,
            "margin_ratios": _margin_ratios(margins, page_width, page_height),
            "body_margins": body_margins,
            "body_margin_ratios": _margin_ratios(body_margins, page_width, page_height) if body_margins else {},
            "text_width": _safe_median(text_widths, None),
            "text_height": _safe_median(text_heights, None),
            "body_text_width": _safe_median(body_text_widths, None),
            "body_text_region_width": _safe_median(body_text_region_widths, None),
            "body_column_width": _safe_median(body_column_widths, None),
            "body_column_width_ratio": (_safe_median(body_column_widths, None) / page_width) if page_width and body_column_widths else None,
            "column_count": column_count,
            "column_gap": column_gap,
            "column_gap_ratio": (column_gap / page_width) if page_width and column_gap is not None else None,
            "column_mode": column_mode,
            "two_column_page_ratio": two_column_ratio,
            "mixed_band_page_ratio": mixed_band_ratio,
            "coordinate_space": str(document.coordinate_space.value if hasattr(document.coordinate_space, "value") else document.coordinate_space),
            "mixed_columns": bool(mixed_band_pages) or (len(set(column_counts)) > 1 if column_counts else False),
        }

    def _estimate_body_font_size(self, document: DocumentIR) -> float | None:
        weighted_sizes: Counter[float] = Counter()
        for node in document.nodes:
            if not _is_body_style_node(node):
                continue
            for size, weight in _iter_span_sizes(node, self.config.min_body_span_chars):
                weighted_sizes[_bucket_font_size(size, self.config.body_font_bucket_pt)] += weight
            fallback = _node_feature_float(node, "style_baseline_size") or _node_feature_float(node, "font_size")
            if fallback:
                weighted_sizes[_bucket_font_size(fallback, self.config.body_font_bucket_pt)] += max(len(node.text), 1)
        if not weighted_sizes:
            return None
        return float(weighted_sizes.most_common(1)[0][0])

    def _estimate_body_font_family(self, document: DocumentIR) -> str | None:
        weighted_fonts: Counter[str] = Counter()
        for node in document.nodes:
            if not _is_body_style_node(node):
                continue
            for span in node.spans:
                font_name = (span.font_name or "").strip()
                if not font_name or len(span.text.strip()) < self.config.min_body_span_chars:
                    continue
                weighted_fonts[_normalize_font_name(font_name)] += max(len(span.text), 1)
        if not weighted_fonts:
            return None
        return weighted_fonts.most_common(1)[0][0]

    def _extract_role_styles(
        self,
        document: DocumentIR,
        body_font_size: float | None,
        body_font_family: str | None,
    ) -> dict[str, dict[str, object]]:
        role_to_sizes: dict[str, list[float]] = defaultdict(list)
        role_to_bold: dict[str, list[bool]] = defaultdict(list)
        role_to_italic: dict[str, list[bool]] = defaultdict(list)
        role_to_fonts: dict[str, Counter[str]] = defaultdict(Counter)

        for node in document.nodes:
            role = _role_for_node(node)
            sizes = [size for size, _weight in _iter_span_sizes(node, 1)]
            if not sizes:
                fallback = _node_feature_float(node, "font_size")
                if fallback:
                    sizes = [fallback]
            role_to_sizes[role].extend(sizes)
            bold_ratio = _style_flag_ratio(node, "is_bold")
            italic_ratio = _style_flag_ratio(node, "is_italic")
            if bold_ratio is not None:
                role_to_bold[role].append(bold_ratio >= 0.5)
            if italic_ratio is not None:
                role_to_italic[role].append(italic_ratio >= 0.5)
            for span in node.spans:
                if span.font_name:
                    role_to_fonts[role][_normalize_font_name(span.font_name)] += max(len(span.text), 1)

        styles: dict[str, dict[str, object]] = {}
        for role, sizes in role_to_sizes.items():
            size = _safe_median(sizes, body_font_size)
            styles[role] = {
                "font_size": size,
                "relative_font_size": (size / body_font_size) if size and body_font_size else None,
                "font_family": _counter_top(role_to_fonts[role]) or body_font_family,
                "font_class": _font_class(_counter_top(role_to_fonts[role]) or body_font_family),
                "bold": _majority_bool(role_to_bold[role]),
                "italic": _majority_bool(role_to_italic[role]),
            }

        styles.setdefault(
            "body",
            {
                "font_size": body_font_size,
                "relative_font_size": 1.0 if body_font_size else None,
                "font_family": body_font_family,
                "font_class": _font_class(body_font_family),
                "bold": False,
                "italic": False,
            },
        )
        return styles

    def _extract_font_clusters(self, document: DocumentIR, body_font_size: float | None) -> dict[str, object]:
        global_clusters: dict[float, dict[str, object]] = {}
        role_clusters: dict[str, dict[float, dict[str, object]]] = defaultdict(dict)
        for node in document.nodes:
            if node.node_type == BlockType.HEADER_FOOTER:
                continue
            role = _role_for_node(node)
            spans = list(node.spans)
            if not spans:
                fallback_size = _node_feature_float(node, "style_baseline_size") or _node_feature_float(node, "font_size")
                if fallback_size:
                    _add_font_cluster_record(
                        global_clusters,
                        size=fallback_size,
                        weight=max(len(node.text.strip()), 1),
                        role=role,
                        font_family=None,
                        is_bold=False,
                        is_italic=False,
                    )
                    _add_font_cluster_record(
                        role_clusters[role],
                        size=fallback_size,
                        weight=max(len(node.text.strip()), 1),
                        role=role,
                        font_family=None,
                        is_bold=False,
                        is_italic=False,
                    )
                continue
            for span in spans:
                if span.font_size is None:
                    continue
                text = span.text or ""
                weight = max(len(text.strip()), 1)
                font_family = _normalize_font_name(span.font_name or "") if span.font_name else None
                _add_font_cluster_record(
                    global_clusters,
                    size=span.font_size,
                    weight=weight,
                    role=role,
                    font_family=font_family,
                    is_bold=span.is_bold,
                    is_italic=span.is_italic,
                )
                _add_font_cluster_record(
                    role_clusters[role],
                    size=span.font_size,
                    weight=weight,
                    role=role,
                    font_family=font_family,
                    is_bold=span.is_bold,
                    is_italic=span.is_italic,
                )
        return {
            "global": _finalize_font_clusters(global_clusters, body_font_size),
            "by_role": {
                role: _finalize_font_clusters(clusters, body_font_size)
                for role, clusters in sorted(role_clusters.items())
            },
        }

    def _estimate_paragraph_indent(self, document: DocumentIR, page_layout: dict[str, object]) -> float | None:
        text_nodes = [node for node in document.nodes if node.node_type == BlockType.TEXT and node.bboxes]
        if len(text_nodes) < 3:
            return None
        left_margin = 0.0
        margins = page_layout.get("margins")
        if isinstance(margins, dict):
            left_margin = float(margins.get("left") or 0.0)
        starts = [node.bboxes[0].x0 - left_margin for node in text_nodes]
        starts = [value for value in starts if value >= 0]
        if not starts:
            return None
        baseline = min(starts)
        page_width = _float_or_none(page_layout.get("page_width")) or 0.0
        max_reasonable_indent = 0.25 * page_width if page_width else float("inf")
        indents = [
            value - baseline
            for value in starts
            if 1.0 < value - baseline <= max_reasonable_indent
        ]
        return _safe_median(indents, 0.0)

    def _estimate_paragraph_spacing(self, document: DocumentIR) -> float | None:
        gaps: list[float] = []
        nodes = sorted(
            [node for node in document.nodes if node.node_type == BlockType.TEXT and node.bboxes],
            key=lambda item: (item.page_idx, item.reading_index),
        )
        for prev, curr in zip(nodes, nodes[1:]):
            if prev.page_idx != curr.page_idx:
                continue
            prev_box = prev.bboxes[-1]
            curr_box = curr.bboxes[0]
            gap = curr_box.y0 - prev_box.y1
            if 0 <= gap < 100:
                gaps.append(gap)
        return _safe_median(gaps, None)

    def _estimate_display_spacing(self, document: DocumentIR, body_font_size: float | None) -> dict[str, float | None]:
        before: list[float] = []
        after: list[float] = []
        nodes = sorted(
            [node for node in document.nodes if node.bboxes and node.node_type != BlockType.HEADER_FOOTER],
            key=lambda item: (item.page_idx, item.reading_index),
        )
        for index, node in enumerate(nodes):
            if node.node_type != BlockType.EQUATION:
                continue
            box = node.bboxes[0]
            if index > 0 and nodes[index - 1].page_idx == node.page_idx and nodes[index - 1].bboxes:
                gap = box.y0 - nodes[index - 1].bboxes[-1].y1
                if 0 <= gap < 150:
                    before.append(gap)
            if index + 1 < len(nodes) and nodes[index + 1].page_idx == node.page_idx and nodes[index + 1].bboxes:
                gap = nodes[index + 1].bboxes[0].y0 - node.bboxes[-1].y1
                if 0 <= gap < 150:
                    after.append(gap)
        fallback = body_font_size * 0.8 if body_font_size else None
        return {
            "above": _safe_median(before, fallback),
            "below": _safe_median(after, fallback),
        }

    def _estimate_list_spacing(self, document: DocumentIR, body_font_size: float | None) -> dict[str, float | None]:
        list_nodes = sorted(
            [node for node in document.nodes if node.node_type == BlockType.LIST and node.bboxes],
            key=lambda item: (item.page_idx, item.reading_index),
        )
        item_gaps: list[float] = []
        for prev, curr in zip(list_nodes, list_nodes[1:]):
            if prev.page_idx != curr.page_idx:
                continue
            gap = curr.bboxes[0].y0 - prev.bboxes[-1].y1
            if 0 <= gap < 100:
                item_gaps.append(gap)
        fallback = body_font_size * 0.25 if body_font_size else None
        return {
            "itemsep": _safe_median(item_gaps, fallback),
            "topsep": body_font_size * 0.5 if body_font_size else None,
        }

    def _extract_bibliography_style(
        self,
        document: DocumentIR,
        body_font_size: float | None,
        page_layout: dict[str, object],
    ) -> dict[str, object]:
        reference_nodes = [node for node in document.nodes if node.node_type == BlockType.REFERENCE]
        sizes = [size for node in reference_nodes for size, _weight in _iter_span_sizes(node, 1)]
        column_layout = _estimate_reference_column_layout(reference_nodes, page_layout)
        return {
            "font_size": _safe_median(sizes, body_font_size),
            "label_style": "numeric",
            "strip_source_labels": True,
            "citation_key_strategy": "source_key_or_ref_number",
            **column_layout,
        }

    def _extract_header_footer_style(self, document: DocumentIR) -> dict[str, object]:
        nodes = [node for node in document.nodes if node.node_type == BlockType.HEADER_FOOTER]
        sizes = [size for node in nodes for size, _weight in _iter_span_sizes(node, 1)]
        examples = [node.text.strip() for node in nodes if node.text.strip()][:5]
        page_count = max(len(document.pages), 1)
        page_by_idx = {page.page_idx: page for page in document.pages}
        classified: list[dict[str, object]] = []
        page_number_candidates: list[dict[str, object]] = []
        repeated_text_candidates: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for node in nodes:
            text = " ".join(str(node.text or "").split())
            if not text or not node.bboxes:
                continue
            page = page_by_idx.get(node.page_idx)
            page_width = float(page.width) if page and page.width else 1000.0
            page_height = float(page.height) if page and page.height else 1000.0
            box = node.bboxes[0]
            zone = _header_footer_zone(box, page_height, self.config.header_footer_zone_ratio)
            if zone is None:
                continue
            slot = _horizontal_slot(box, page_width)
            entry = {
                "node_id": node.node_id,
                "page_idx": node.page_idx,
                "text": text,
                "normalized_text": _normalize_header_footer_text(text),
                "zone": zone,
                "slot": slot,
            }
            classified.append(entry)
            if _is_page_number_text(text):
                page_number_candidates.append(entry)
            else:
                repeated_text_candidates[(zone, str(entry["normalized_text"]))].append(entry)

        page_number = self._infer_page_number_profile(page_number_candidates, page_count)
        header = self._infer_repeated_header_footer_text(repeated_text_candidates, zone="header", page_count=page_count)
        footer = self._infer_repeated_header_footer_text(repeated_text_candidates, zone="footer", page_count=page_count)
        return {
            "font_size": _safe_median(sizes, None),
            "example_count": len(examples),
            "examples": examples,
            "page_number": page_number,
            "header": header,
            "footer": footer,
            "classified_count": len(classified),
            "render_by_default": bool(page_number.get("enabled") or header.get("enabled") or footer.get("enabled")),
        }

    def _infer_page_number_profile(self, candidates: list[dict[str, object]], page_count: int) -> dict[str, object]:
        pages = {int(item["page_idx"]) for item in candidates if isinstance(item.get("page_idx"), int)}
        slot_counter = Counter(str(item["slot"]) for item in candidates)
        zone_counter = Counter(str(item["zone"]) for item in candidates)
        min_pages = max(2, math.ceil(page_count * self.config.page_number_min_page_ratio))
        enabled = len(pages) >= min_pages
        zone = zone_counter.most_common(1)[0][0] if zone_counter else "footer"
        slot = slot_counter.most_common(1)[0][0] if slot_counter else "center"
        return {
            "enabled": enabled,
            "confidence": len(pages) / max(page_count, 1),
            "pages_seen": len(pages),
            "position": f"{zone}_{slot}",
            "zone": zone,
            "slot": slot,
            "source": "numeric_edge_nodes",
        }

    def _infer_repeated_header_footer_text(
        self,
        candidates: dict[tuple[str, str], list[dict[str, object]]],
        *,
        zone: str,
        page_count: int,
    ) -> dict[str, object]:
        best_entries: list[dict[str, object]] = []
        best_key = ""
        for (candidate_zone, normalized_text), entries in candidates.items():
            if candidate_zone != zone or not normalized_text:
                continue
            pages = {int(item["page_idx"]) for item in entries if isinstance(item.get("page_idx"), int)}
            best_pages = {int(item["page_idx"]) for item in best_entries if isinstance(item.get("page_idx"), int)}
            if len(pages) > len(best_pages):
                best_key = normalized_text
                best_entries = entries
        pages = {int(item["page_idx"]) for item in best_entries if isinstance(item.get("page_idx"), int)}
        min_pages = max(self.config.header_footer_repeat_min_pages, math.ceil(page_count * self.config.header_footer_repeat_min_page_ratio))
        enabled = len(pages) >= min_pages
        slot_counter = Counter(str(item["slot"]) for item in best_entries)
        text_counter = Counter(str(item["text"]) for item in best_entries)
        text = text_counter.most_common(1)[0][0] if text_counter else ""
        slot = slot_counter.most_common(1)[0][0] if slot_counter else "center"
        return {
            "enabled": enabled,
            "confidence": len(pages) / max(page_count, 1),
            "pages_seen": len(pages),
            "position": f"{zone}_{slot}",
            "zone": zone,
            "slot": slot,
            "text": text if enabled else "",
            "normalized_text": best_key,
            "source": "repeated_edge_text",
        }

    def _geometry_options_from_layout(self, page_layout: dict[str, object]) -> dict[str, str]:
        page_width = _float_or_none(page_layout.get("page_width"))
        page_height = _float_or_none(page_layout.get("page_height"))
        margins = page_layout.get("margins") if isinstance(page_layout.get("margins"), dict) else {}
        body_margins = page_layout.get("body_margins") if isinstance(page_layout.get("body_margins"), dict) else {}
        if not page_width or not page_height or not isinstance(margins, dict):
            return {}
        horizontal_margins = body_margins if _valid_horizontal_margins(body_margins) else margins
        if self.config.output_paper.casefold() == "a4":
            return {
                "paperwidth": _pt(self.config.output_paper_width_pt),
                "paperheight": _pt(self.config.output_paper_height_pt),
                "left": _scaled_margin_pt(horizontal_margins.get("left"), page_width, self.config.output_paper_width_pt),
                "right": _scaled_margin_pt(horizontal_margins.get("right"), page_width, self.config.output_paper_width_pt),
                "top": _scaled_margin_pt(margins.get("top"), page_height, self.config.output_paper_height_pt),
                "bottom": _scaled_margin_pt(margins.get("bottom"), page_height, self.config.output_paper_height_pt),
            }
        return {
            "paperwidth": _pt_from_normalized(page_width),
            "paperheight": _pt_from_normalized(page_height),
            "left": _pt_from_normalized(_float_or_none(horizontal_margins.get("left"))),
            "right": _pt_from_normalized(_float_or_none(horizontal_margins.get("right"))),
            "top": _pt_from_normalized(_float_or_none(margins.get("top"))),
            "bottom": _pt_from_normalized(_float_or_none(margins.get("bottom"))),
        }

    def _scaled_horizontal_length(self, value: object, page_layout: dict[str, object]) -> float | None:
        source = _float_or_none(value)
        page_width = _float_or_none(page_layout.get("page_width"))
        if source is None or not page_width:
            return None
        if self.config.output_paper.casefold() == "a4":
            return source / page_width * self.config.output_paper_width_pt
        return source * 0.792

    def _scaled_vertical_length(self, value: object, page_layout: dict[str, object]) -> float | None:
        source = _float_or_none(value)
        page_height = _float_or_none(page_layout.get("page_height"))
        if source is None or not page_height:
            return None
        if self.config.output_paper.casefold() == "a4":
            return source / page_height * self.config.output_paper_height_pt
        return source * 0.792

    def _normalize_paragraph_spacing(
        self,
        value: object,
        page_layout: dict[str, object],
        body_font_size: float | None,
    ) -> float | None:
        scaled = self._scaled_vertical_length(value, page_layout)
        if scaled is None:
            return None
        if body_font_size:
            return min(max(scaled, 0.0), body_font_size * 0.45)
        return min(max(scaled, 0.0), 4.0)

    def _normalize_vertical_spacing_dict(
        self,
        values: dict[str, float | None],
        page_layout: dict[str, object],
        *,
        max_multiplier: float,
        body_font_size: float | None,
    ) -> dict[str, float | None]:
        limit = body_font_size * max_multiplier if body_font_size else None
        normalized: dict[str, float | None] = {}
        for key, value in values.items():
            scaled = self._scaled_vertical_length(value, page_layout)
            if scaled is not None and limit is not None:
                scaled = min(max(scaled, 0.0), limit)
            normalized[key] = scaled
        return normalized


def _nodes_by_page(nodes: Iterable[DocumentNode]) -> dict[int, list[DocumentNode]]:
    by_page: dict[int, list[DocumentNode]] = defaultdict(list)
    for node in nodes:
        by_page[node.page_idx].append(node)
    return by_page


def _is_layout_node(node: DocumentNode) -> bool:
    return node.node_type not in {BlockType.HEADER_FOOTER, BlockType.FOOTNOTE, BlockType.MARGIN_NOTE, BlockType.TOC, BlockType.OTHER}


def _is_body_style_node(node: DocumentNode) -> bool:
    if node.node_type not in {BlockType.TEXT, BlockType.LIST}:
        return False
    layer = str(node.metadata.get("layout_layer") or "").casefold()
    role = str(node.metadata.get("layout_role") or "").casefold()
    if layer in {"metadata_layer", "noise_layer", "float_layer"}:
        return False
    if role in {
        "abstract",
        "abstract_body",
        "abstract_title",
        "author",
        "authors",
        "affiliation",
        "caption",
        "correspondence",
        "date",
        "email",
        "figure_caption",
        "front_matter",
        "index_terms",
        "keywords",
        "metadata",
        "table_caption",
    }:
        return False
    if node.flags.get("is_noise") or node.node_type in {BlockType.HEADER_FOOTER, BlockType.FOOTNOTE, BlockType.MARGIN_NOTE}:
        return False
    return bool((node.text or "").strip())


def _estimate_page_body_text_geometry(
    nodes: list[DocumentNode],
    page_width: float,
    full_width_threshold: float,
) -> dict[str, object]:
    boxes = [
        node.bboxes[0]
        for node in nodes
        if _is_body_style_node(node)
        and node.bboxes
        and 4.0 <= max(node.bboxes[0].x1 - node.bboxes[0].x0, 0.0)
    ]
    if not boxes:
        return {}
    min_x = min(box.x0 for box in boxes)
    max_x = max(box.x1 for box in boxes)
    margins = {"left": max(min_x, 0.0), "right": max(page_width - max_x, 0.0)}
    columns = _split_body_text_columns(boxes, page_width)
    column_widths = [
        _safe_median([max(box.x1 - box.x0, 0.0) for box in column], None)
        for column in columns
        if column
    ]
    column_widths = [float(width) for width in column_widths if width]
    text_region_width = max(max_x - min_x, 0.0)
    text_width = _safe_median([max(box.x1 - box.x0, 0.0) for box in boxes], None)
    return {
        "margins": margins,
        "text_width": text_width,
        "text_region_width": text_region_width,
        "column_widths": column_widths,
    }


def _split_body_text_columns(boxes: list[BBox], page_width: float) -> list[list[BBox]]:
    if len(boxes) < 2 or page_width <= 0:
        return [boxes]
    centers = sorted((((box.x0 + box.x1) / 2.0, box.x0, box.y0, box) for box in boxes), key=lambda item: (item[0], item[1], item[2]))
    gaps = [(centers[index + 1][0] - centers[index][0], index) for index in range(len(centers) - 1)]
    if not gaps:
        return [boxes]
    largest_gap, split_index = max(gaps)
    if largest_gap < 0.12 * page_width:
        return [boxes]
    left = [box for _center, _x0, _y0, box in centers[: split_index + 1]]
    right = [box for _center, _x0, _y0, box in centers[split_index + 1 :]]
    if not left or not right:
        return [boxes]
    return [left, right]


def _estimate_reference_column_layout(
    reference_nodes: list[DocumentNode],
    page_layout: dict[str, object],
) -> dict[str, object]:
    """Estimate bibliography columns from individual reference boxes.

    A full reference section often has a union bbox that spans the page even
    when the actual items live in two columns.  This helper therefore clusters
    per-node/per-line boxes by x center instead of using the section envelope.
    """

    page_width = _float_or_none(page_layout.get("page_width")) or 1000.0
    boxes_by_page: dict[int, list[BBox]] = defaultdict(list)
    for node in reference_nodes:
        for box in node.bboxes:
            width = max(box.x1 - box.x0, 0.0)
            height = max(box.y1 - box.y0, 0.0)
            if width >= 4.0 and height >= 1.0:
                boxes_by_page[node.page_idx].append(box)

    two_column_pages = 0
    single_column_pages = 0
    column_widths: list[float] = []
    column_gaps: list[float] = []
    narrow_box_count = 0
    full_span_box_count = 0
    reference_item_count = sum(_reference_item_count(node) for node in reference_nodes)
    compact_reference_chunks = bool(reference_nodes) and reference_item_count >= max(len(reference_nodes) * 3, 8)

    for boxes in boxes_by_page.values():
        if not boxes:
            continue
        narrow_boxes = [box for box in boxes if max(box.x1 - box.x0, 0.0) < 0.72 * page_width]
        full_span_box_count += len(boxes) - len(narrow_boxes)
        narrow_box_count += len(narrow_boxes)
        columns = _split_body_text_columns(narrow_boxes, page_width)
        if len(columns) == 2 and all(columns):
            left, right = columns
            left_centers = [(box.x0 + box.x1) / 2.0 for box in left]
            right_centers = [(box.x0 + box.x1) / 2.0 for box in right]
            center_gap = min(right_centers) - max(left_centers)
            left_max = max(box.x1 for box in left)
            right_min = min(box.x0 for box in right)
            gutter = max(right_min - left_max, 0.0)
            if center_gap >= 0.12 * page_width and gutter >= 0.02 * page_width:
                two_column_pages += 1
                column_gaps.append(gutter)
                column_widths.extend(
                    [
                        _safe_median([max(box.x1 - box.x0, 0.0) for box in left], 0.0),
                        _safe_median([max(box.x1 - box.x0, 0.0) for box in right], 0.0),
                    ]
                )
                continue
        if boxes:
            single_column_pages += 1

    considered_pages = two_column_pages + single_column_pages
    if considered_pages == 0:
        column_mode = "unknown"
        confidence = 0.0
    else:
        two_ratio = two_column_pages / considered_pages
        if two_column_pages > 0 and two_ratio >= 0.35:
            column_mode = "two_column"
            confidence = two_ratio
        elif full_span_box_count > narrow_box_count:
            column_mode = "single"
            confidence = single_column_pages / considered_pages
        else:
            column_mode = "single"
            confidence = max(single_column_pages / considered_pages, 1.0 - two_ratio)

    body_column_mode = str(page_layout.get("column_mode") or "").casefold()
    page_width = max(page_width, 1.0)
    median_ref_width = _safe_median(
        [max(box.x1 - box.x0, 0.0) for boxes in boxes_by_page.values() for box in boxes],
        None,
    )
    narrow_reference_chunks = median_ref_width is not None and median_ref_width <= 0.58 * page_width
    if (
        column_mode == "single"
        and compact_reference_chunks
        and narrow_reference_chunks
        and body_column_mode in {"two_column", "mixed"}
    ):
        # MinerU sometimes compresses an entire two-column bibliography page
        # into one or two reference_list nodes whose bbox only covers a single
        # column.  In that case the per-node cluster is under-observed, so
        # inherit the body column prior instead of forcing single-column refs.
        column_mode = "two_column"
        confidence = max(float(confidence), 0.55)

    return {
        "column_mode": column_mode,
        "column_count": 2 if column_mode == "two_column" else (1 if column_mode == "single" else None),
        "column_confidence": round(float(confidence), 4),
        "column_pages": {"two_column": two_column_pages, "single": single_column_pages},
        "column_width": _safe_median(column_widths, None),
        "column_gap": _safe_median(column_gaps, None),
        "reference_item_count": reference_item_count,
        "compact_reference_chunks": compact_reference_chunks,
        "column_detection_reason": "compact_reference_body_column_fallback"
        if compact_reference_chunks and column_mode == "two_column" and two_column_pages == 0
        else "reference_bbox_clusters",
    }


def _reference_item_count(node: DocumentNode) -> int:
    items = node.metadata.get("reference_items")
    if isinstance(items, list):
        return len(items)
    return 1 if str(node.text or "").strip() else 0


def _estimate_page_columns(
    nodes: list[DocumentNode],
    page_width: float,
    full_width_threshold: float,
) -> tuple[int, float | None]:
    text_boxes = [
        node.bboxes[0]
        for node in nodes
        if _is_body_style_node(node)
        and node.bboxes
        and (node.bboxes[0].x1 - node.bboxes[0].x0) < full_width_threshold * page_width
    ]
    if len(text_boxes) < 2:
        return 1, None
    centers = sorted((box.x0 + box.x1) / 2.0 for box in text_boxes)
    gaps = [(centers[index + 1] - centers[index], index) for index in range(len(centers) - 1)]
    if not gaps:
        return 1, None
    largest_gap, split_index = max(gaps)
    if largest_gap < 0.12 * page_width:
        return 1, None
    left = centers[: split_index + 1]
    right = centers[split_index + 1 :]
    if len(left) < 1 or len(right) < 1:
        return 1, None
    left_max = max(box.x1 for box in text_boxes if (box.x0 + box.x1) / 2.0 <= max(left))
    right_min = min(box.x0 for box in text_boxes if (box.x0 + box.x1) / 2.0 >= min(right))
    return 2, max(right_min - left_max, 0.0)


def _page_has_full_width_blocker(nodes: list[DocumentNode], page_width: float, full_width_threshold: float) -> bool:
    if page_width <= 0:
        return False
    center = page_width / 2.0
    margin = 0.05 * page_width
    for node in nodes:
        if not _is_layout_node(node) or not node.bboxes:
            continue
        box = node.bboxes[0]
        width = max(box.x1 - box.x0, 0.0)
        crosses_center = box.x0 < center - margin and box.x1 > center + margin
        if width >= full_width_threshold * page_width or crosses_center:
            return True
    return False


def _iter_span_sizes(node: DocumentNode, min_chars: int) -> Iterable[tuple[float, int]]:
    for span in node.spans:
        if span.font_size is None:
            continue
        text = span.text or ""
        char_count = len(text.strip())
        if char_count < min_chars:
            continue
        yield float(span.font_size), max(char_count, 1)


def _bucket_font_size(size: float, bucket: float) -> float:
    if bucket <= 0:
        return round(float(size), 2)
    return round(round(float(size) / bucket) * bucket, 2)


def _style_flag_ratio(node: DocumentNode, flag_name: str) -> float | None:
    total = 0
    matched = 0
    for span in node.spans:
        weight = max(len((span.text or "").strip()), 1)
        total += weight
        if bool(getattr(span, flag_name)):
            matched += weight
    if total == 0:
        return None
    return matched / total


def _role_for_node(node: DocumentNode) -> str:
    if node.node_type == BlockType.HEADER_FOOTER:
        return "header_footer"
    if node.node_type == BlockType.FOOTNOTE:
        return "footnote"
    if node.node_type == BlockType.MARGIN_NOTE:
        return "margin_note"
    if node.node_type == BlockType.TITLE:
        level = node.features.get("heading_level") or node.metadata.get("heading_level")
        if level in {1, "1", "section"}:
            return "section"
        if level in {2, "2", "subsection"}:
            return "subsection"
        if level in {3, "3", "subsubsection"}:
            return "subsubsection"
        return "heading"
    if node.node_type == BlockType.REFERENCE:
        return "bibliography"
    if node.node_type == BlockType.TABLE:
        return "table"
    if node.node_type == BlockType.FIGURE:
        return "figure"
    if node.node_type in {BlockType.EQUATION, BlockType.INLINE_MATH}:
        return "math"
    if node.node_type == BlockType.LIST:
        return "list"
    if node.node_type == BlockType.ALGORITHM:
        return "algorithm"
    if node.node_type == BlockType.CODE:
        return "code"
    return "body"


def _node_feature_float(node: DocumentNode, key: str) -> float | None:
    value = node.features.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_font_name(font_name: str) -> str:
    return canonicalize_pdf_font_name(str(font_name or ""))


def _font_class(font_name: str | None) -> str | None:
    info = resolve_pdf_font(font_name)
    return info.font_class if info else None


def _add_font_cluster_record(
    clusters: dict[float, dict[str, object]],
    *,
    size: float,
    weight: int,
    role: str,
    font_family: str | None,
    is_bold: bool,
    is_italic: bool,
) -> None:
    bucket = round(float(size) * 2.0) / 2.0
    record = clusters.setdefault(
        bucket,
        {
            "font_size": bucket,
            "char_weight": 0,
            "roles": Counter(),
            "font_families": Counter(),
            "bold_weight": 0,
            "italic_weight": 0,
        },
    )
    record["char_weight"] = int(record["char_weight"]) + weight
    roles = record["roles"]
    if isinstance(roles, Counter):
        roles[role] += weight
    font_families = record["font_families"]
    if isinstance(font_families, Counter) and font_family:
        font_families[font_family] += weight
    if is_bold:
        record["bold_weight"] = int(record["bold_weight"]) + weight
    if is_italic:
        record["italic_weight"] = int(record["italic_weight"]) + weight


def _finalize_font_clusters(clusters: dict[float, dict[str, object]], body_font_size: float | None) -> list[dict[str, object]]:
    finalized: list[dict[str, object]] = []
    for size, record in sorted(clusters.items(), key=lambda item: (-item[0], -int(item[1].get("char_weight", 0)))):
        weight = max(int(record.get("char_weight") or 0), 1)
        roles = record.get("roles")
        fonts = record.get("font_families")
        finalized.append(
            {
                "font_size": size,
                "relative_to_body": (size / body_font_size) if body_font_size else None,
                "char_weight": weight,
                "dominant_role": roles.most_common(1)[0][0] if isinstance(roles, Counter) and roles else None,
                "role_weights": dict(roles) if isinstance(roles, Counter) else {},
                "dominant_font_family": fonts.most_common(1)[0][0] if isinstance(fonts, Counter) and fonts else None,
                "font_family_weights": dict(fonts) if isinstance(fonts, Counter) else {},
                "bold_ratio": int(record.get("bold_weight") or 0) / weight,
                "italic_ratio": int(record.get("italic_weight") or 0) / weight,
            }
        )
    return finalized


def _safe_median(values: Iterable[float], default: float | None) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return default
    return float(median(cleaned))


def _median_margins(margins: list[dict[str, float]]) -> dict[str, float]:
    return {
        side: float(_safe_median([item[side] for item in margins if side in item], 0.0) or 0.0)
        for side in ("left", "right", "top", "bottom")
    }


def _margin_ratios(margins: dict[str, float], page_width: float, page_height: float) -> dict[str, float | None]:
    return {
        "left": (margins.get("left", 0.0) / page_width) if page_width else None,
        "right": (margins.get("right", 0.0) / page_width) if page_width else None,
        "top": (margins.get("top", 0.0) / page_height) if page_height else None,
        "bottom": (margins.get("bottom", 0.0) / page_height) if page_height else None,
    }


def _header_footer_zone(box: BBox, page_height: float, zone_ratio: float) -> str | None:
    if page_height <= 0:
        return None
    center_y = (box.y0 + box.y1) / 2.0
    ratio = center_y / page_height
    if ratio <= zone_ratio:
        return "header"
    if ratio >= 1.0 - zone_ratio:
        return "footer"
    return None


def _horizontal_slot(box: BBox, page_width: float) -> str:
    if page_width <= 0:
        return "center"
    center_x = (box.x0 + box.x1) / 2.0
    ratio = center_x / page_width
    if ratio < 1.0 / 3.0:
        return "left"
    if ratio > 2.0 / 3.0:
        return "right"
    return "center"


PAGE_NUMBER_TEXT_RE = re.compile(
    r"^\s*(?:page\s*)?(?:[-–—]\s*)?(?:\d+|[ivxlcdmIVXLCDM]+)(?:\s*/\s*\d+)?(?:\s*[-–—])?\s*$"
)


def _is_page_number_text(text: str) -> bool:
    value = " ".join(str(text or "").split())
    if not value:
        return False
    return bool(PAGE_NUMBER_TEXT_RE.match(value))


def _normalize_header_footer_text(text: str) -> str:
    value = " ".join(str(text or "").casefold().split())
    value = re.sub(r"\d+", "<num>", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _pt_from_normalized(value: float | None) -> str:
    # DocumentIR commonly uses a 0-1000 normalized page.  Map it to a
    # letter-height point scale so relative margins survive in LaTeX.
    if value is None:
        return "0pt"
    return f"{max(float(value), 0.0) * 0.792:.2f}pt"


def _pt(value: float | None) -> str:
    if value is None:
        return "0pt"
    return f"{max(float(value), 0.0):.2f}pt"


def _scaled_margin_pt(value: object, source_extent: float, target_extent_pt: float) -> str:
    margin = _float_or_none(value)
    if margin is None or source_extent <= 0:
        return "0pt"
    scaled = max(margin, 0.0) / source_extent * target_extent_pt
    # Avoid pathological full-page margins from noisy extraction, while keeping
    # the original paper's relative whitespace.
    scaled = min(max(scaled, 18.0), target_extent_pt * 0.28)
    return _pt(scaled)


def _valid_horizontal_margins(margins: dict[str, object]) -> bool:
    left = _float_or_none(margins.get("left"))
    right = _float_or_none(margins.get("right"))
    if left is None or right is None:
        return False
    return left >= 0 and right >= 0


def _dominant_int(values: Iterable[int], *, default: int) -> int:
    counter = Counter(values)
    if not counter:
        return default
    return int(counter.most_common(1)[0][0])


def _counter_top(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _majority_bool(values: Iterable[bool]) -> bool | None:
    items = list(values)
    if not items:
        return None
    return sum(1 for item in items if item) >= len(items) / 2.0
