"""Visual reading-order reconstruction for MinerU content_list_v2 output."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.perception.layout_probes import best_layout_probe, collect_layout_probes, has_strong_layout_probe
from src.perception.title_features import is_front_matter_date_text
from src.perception.xy_cut import rebuild_reading_order

AUXILIARY_TYPES = {
    "page_header",
    "page_footer",
    "page_number",
    "page_aside_text",
    "page_footnote",
}

TEXTUAL_TYPES = {
    "title",
    "paragraph",
    "text",
    "list",
    "index",
    "code",
    "algorithm",
}

MICRO_TEXT_TYPES = {
    "paragraph",
    "text",
}

MICRO_INLINE_MATH_TYPES = {
    "equation_inline",
    "inline_equation",
    "inline_formula",
    "inline_math",
    "math_inline",
}

MICRO_EQUATION_TYPES = {
    "display_formula",
    "equation",
    "equation_interline",
    "formula",
    "interline_equation",
}

MICRO_STRUCTURAL_TYPES = {
    "algorithm",
    "chart",
    "code",
    "figure",
    "image",
    "list",
    "page_aside_text",
    "page_footer",
    "page_footnote",
    "page_header",
    "page_number",
    "reference",
    "table",
    "title",
}

LIST_MARKER_PATTERNS = (
    ("arabic", re.compile(r"^\s*(\d+)[\.\)]\s+")),
    ("alpha", re.compile(r"^\s*([a-zA-Z])[\.\)]\s+")),
    ("roman", re.compile(r"^\s*([ivxlcdmIVXLCDM]+)[\.\)]\s+")),
    ("bullet", re.compile(r"^\s*[\u2022\-\*\>]\s+")),
    ("paren_arabic", re.compile(r"^\s*[\(\uff08](\d+)[\)\uff09]\s*")),
    ("paren_cjk", re.compile(r"^\s*[\(\uff08]([一二三四五六七八九十]+)[\)\uff09]\s*")),
    ("cjk_comma", re.compile(r"^\s*([一二三四五六七八九十]+)[、.．]\s*")),
    ("ordinal_cjk", re.compile(r"^\s*第[一二三四五六七八九十]+[，,、.．]\s*")),
)

RUN_IN_HEADING_PREFIX_RE = re.compile(r"^\s*(?P<number>\d+(?:\.\d+)+)\.?\s*$")
RUN_IN_HEADING_INLINE_PREFIX_RE = re.compile(r"^\s*(?P<number>\d+(?:\.\d+)+)\.?\s+(?P<tail>.+)$")

FULL_WIDTH_TYPES = {
    "title",
    "table",
    "chart",
    "image",
    "equation_interline",
    "algorithm",
    "code",
}

REFERENCE_LIST_TYPE = "reference_list"
LAYOUT_LAYER_MAIN_TEXT = "main_text_flow"
LAYOUT_LAYER_MATH = "math_layer"
LAYOUT_LAYER_FLOAT = "float_layer"
LAYOUT_LAYER_ANNOTATION = "annotation_layer"
LAYOUT_LAYER_METADATA = "metadata_layer"
LAYOUT_LAYER_NOISE = "noise_layer"
LAYOUT_LAYER_OTHER = "other_layer"
TOC_TITLE_ROLE = "toc_title"
TOC_ENTRY_ROLE = "toc_entry"
MATH_LAYOUT_TYPES = MICRO_INLINE_MATH_TYPES | MICRO_EQUATION_TYPES
FLOAT_LAYOUT_TYPES = {"figure", "image", "chart", "table", "algorithm"}
METADATA_LAYOUT_TYPES = {"title", "author", "affiliation", "abstract"}
NOISE_LAYOUT_TYPES = AUXILIARY_TYPES | {"header", "footer"}
TOC_ENTRY_TYPES = {"index", "toc", "table_of_contents"}
FOOTNOTE_LAYOUT_TYPES = {"page_footnote", "footnote", "foot_note"}
MARGIN_NOTE_LAYOUT_TYPES = {"margin_note", "marginnote", "side_note", "sidenote", "sidebar"}
CAPTION_LABEL_RE = re.compile(r"^\s*(?P<kind>fig\.?|figure|table|algorithm)\s*\.?\s*(?P<number>[A-Za-z]?\d+(?:\.\d+)*)\s*[:.\-]", re.IGNORECASE)
FOOTNOTE_PREFIX_RE = re.compile(r"^\s*(?:\d{1,3}|[*†‡§¶]|[¹²³⁴⁵⁶⁷⁸⁹⁰]+)\s+")
DUPLICATE_CONTINUATION_TYPES = {
    "paragraph",
    "text",
    "list",
    "reference",
    "references",
    "bibliography",
}
DUPLICATE_CONTINUATION_START_RE = re.compile(
    r"^\s*(?:[a-z,;:)\]}]|and\b|or\b|with\b|where\b|which\b|that\b|while\b|because\b|for\b|in\b|of\b|to\b|the\b|as\b|from\b|by\b|on\b|using\b|under\b|between\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SortConfig:
    """Thresholds for normalized MinerU v2 coordinates."""

    full_width_ratio: float = 0.60
    cross_column_left: float = 380.0
    cross_column_right: float = 620.0
    min_column_gap: float = 130.0
    min_blocks_per_column: int = 2
    y_tolerance: float = 8.0
    drop_empty_textual_blocks: bool = True


@dataclass(frozen=True)
class BlockView:
    block: dict[str, Any]
    original_index: int
    page_idx: int
    bbox: tuple[float, float, float, float]
    text: str
    is_textual: bool
    is_auxiliary: bool
    is_full_width: bool
    column_id: int | None = None

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)


@dataclass(frozen=True)
class ColumnarOrderView:
    node: dict[str, Any]
    original_index: int
    bbox: tuple[float, float, float, float] | None
    span_label: str
    column_label: str | None

    @property
    def x0(self) -> float:
        return self.bbox[0] if self.bbox is not None else 0.0

    @property
    def y0(self) -> float:
        return self.bbox[1] if self.bbox is not None else 0.0

    @property
    def x1(self) -> float:
        return self.bbox[2] if self.bbox is not None else 0.0

    @property
    def y1(self) -> float:
        return self.bbox[3] if self.bbox is not None else 0.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)


def load_content_list_v2(path: Path) -> list[list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {path}")
    pages: list[list[dict[str, Any]]] = []
    for page_idx, page in enumerate(data):
        if not isinstance(page, list):
            raise ValueError(f"Expected page {page_idx} to be a list")
        clean_page: list[dict[str, Any]] = []
        for block_idx, block in enumerate(page):
            if not isinstance(block, dict):
                raise ValueError(f"Expected page {page_idx} block {block_idx} to be an object")
            clean_page.append(block)
        pages.append(clean_page)
    return pages


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sort_content_list_v2(
    pages: list[list[dict[str, Any]]],
    *,
    config: SortConfig | None = None,
    keep_auxiliary: bool = False,
) -> dict[str, Any]:
    cfg = config or SortConfig()
    sorted_pages = []
    summaries = []

    for page_idx, page in enumerate(pages):
        views = [_make_block_view(block, i, page_idx, cfg) for i, block in enumerate(page)]
        empty_textual = [view for view in views if cfg.drop_empty_textual_blocks and view.is_textual and not view.text]
        sortable = [
            view
            for view in views
            if (keep_auxiliary or not view.is_auxiliary) and view not in empty_textual
        ]
        dropped = [view for view in views if view.is_auxiliary and not keep_auxiliary]
        ordered = _sort_page_blocks(sortable, cfg)
        enriched = _enrich_ordered_blocks(ordered)
        sorted_pages.append(enriched)
        summaries.append(
            {
                "page_idx": page_idx,
                "input_blocks": len(page),
                "output_blocks": len(enriched),
                "dropped_auxiliary_blocks": len(dropped),
                "dropped_empty_textual_blocks": len(empty_textual),
                "column_count": _count_columns(ordered),
                "full_width_blocks": sum(1 for view in ordered if view.is_full_width),
                "text_runs": len({item["text_run_id"] for item in enriched if item.get("text_run_id") is not None}),
            }
        )

    return {
        "schema_version": "visual_order_v1",
        "source_format": "mineru_content_list_v2",
        "config": {
            "full_width_ratio": cfg.full_width_ratio,
            "cross_column_left": cfg.cross_column_left,
            "cross_column_right": cfg.cross_column_right,
            "min_column_gap": cfg.min_column_gap,
            "min_blocks_per_column": cfg.min_blocks_per_column,
            "y_tolerance": cfg.y_tolerance,
            "drop_empty_textual_blocks": cfg.drop_empty_textual_blocks,
            "keep_auxiliary": keep_auxiliary,
        },
        "pages": sorted_pages,
        "page_summaries": summaries,
    }


def _build_native_listmarked_content(native_payload: Any) -> dict[str, Any]:
    """Flatten MinerU native JSON without merging or bbox rewriting."""

    items = _native_v2_items(native_payload)
    output: list[dict[str, Any]] = []
    list_item_counter = 0

    for global_order, source in enumerate(items):
        block = source["block"]
        text = extract_text(block)
        marker = detect_list_marker(text)
        content = block.get("content")
        list_type = content.get("list_type") if isinstance(content, dict) else None
        reference_items = _extract_reference_items(block)
        item = {
            "global_order": global_order,
            "page_idx": source["page_idx"],
            "original_index": source["block_idx"],
            "mineru_page_idx": source["page_idx"],
            "mineru_block_idx": source["block_idx"],
            "type": block.get("type"),
            "raw_type": block.get("type"),
            "list_type": list_type,
            "reference_items": reference_items,
            "bbox": list(block.get("bbox") or []),
            "text_for_embedding": text,
            "text": text,
            "list_marker": marker,
            "has_list_marker": marker is not None,
            "list_item_id": None,
            "block": block,
        }
        if marker is not None:
            item["list_item_id"] = f"li_{list_item_counter:05d}"
            list_item_counter += 1
        output.append(item)

    return {
        "schema_version": "content_native_listmarkers",
        "source_format": "mineru_content_list_v2",
        "config": {
            "preserve_mineru_page_block_order": True,
            "preserve_bbox": True,
            "sort_blocks": False,
            "merge_paragraphs": False,
            "merge_across_pages": False,
            "merge_across_columns": False,
            "infer_columns": False,
            "rewrite_layout_positions": False,
            "list_marker_detection": True,
            "style_enrichment": "external_pymupdf_step",
        },
        "items": output,
    }


def build_content_v7(native_payload: Any) -> dict[str, Any]:
    """Build list/style-ready content after safe block-level column repair.

    v7 preserves MinerU bbox coordinates and text atomics, adds list-marker
    metadata, then overwrites global order with `fix_columnar_reading_order`
    page by page. It does not merge text, append bbox chunks, infer paragraph
    continuations, or alter coordinates.
    """

    native = _build_native_listmarked_content(native_payload)
    page_groups: dict[int, list[dict[str, Any]]] = {}
    for item in native["items"]:
        page_idx = item.get("page_idx")
        page = page_idx if isinstance(page_idx, int) else 0
        page_groups.setdefault(page, []).append(dict(item))

    ordered: list[dict[str, Any]] = []
    for page_idx in sorted(page_groups):
        fixed_page = fix_columnar_reading_order(page_groups[page_idx])
        for page_order, item in enumerate(fixed_page):
            item["column_fix_page_order"] = page_order
            ordered.append(item)

    band_global_ids: dict[tuple[int, int], int] = {}
    for global_order, item in enumerate(ordered):
        item["global_order"] = global_order
        item["column_fix_global_order"] = global_order
        item["layout_flow_order"] = global_order
        page_idx = item.get("page_idx")
        band_id = item.get("layout_band_id")
        if isinstance(page_idx, int) and isinstance(band_id, int):
            key = (page_idx, band_id)
            if key not in band_global_ids:
                band_global_ids[key] = len(band_global_ids)
            item["layout_band_global_id"] = band_global_ids[key]
            item["layout_band_global_order"] = band_global_ids[key]
        else:
            item["layout_band_global_id"] = None
            item["layout_band_global_order"] = None
    annotate_repeated_header_footer_layers(ordered)
    annotate_footnote_layers(ordered)
    refine_front_matter_layers(ordered)
    mark_toc_layers(ordered)
    annotate_run_in_headings(ordered)
    ordered = annotate_duplicate_contained_continuations(ordered)

    return {
        "schema_version": "content_v7_columnfix_listmarkers",
        "source_format": "mineru_content_list_v2",
        "config": {
            **native["config"],
            "preserve_mineru_page_block_order": False,
            "safe_columnar_reading_order_fix": True,
            "sort_blocks": "block_level_full_span_isolated_left_column_then_right_column",
            "merge_paragraphs": False,
            "merge_across_pages": False,
            "merge_across_columns": False,
            "rewrite_layout_positions": False,
            "duplicate_contained_continuation_detection": True,
        },
        "items": ordered,
    }


def refresh_content_v7_layout_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute v7 layer/band/order metadata on an existing styled v7 JSON.

    Older ``*_content_list_v7_styles.json`` files predate the explicit
    layer/band contract. This function preserves all existing text, style and
    bbox fields, but reruns the safe page-level column order repair so graph
    building and relabeling see the same upgraded node order.
    """

    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Expected v7 payload with an items list")
    page_groups: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        page_idx = item.get("page_idx")
        page = page_idx if isinstance(page_idx, int) else 0
        hydrated = dict(item)
        hydrate_empty_text_fields(hydrated)
        page_groups.setdefault(page, []).append(hydrated)

    ordered: list[dict[str, Any]] = []
    for page_idx in sorted(page_groups):
        fixed_page = fix_columnar_reading_order(page_groups[page_idx])
        for page_order, item in enumerate(fixed_page):
            item["column_fix_page_order"] = page_order
            ordered.append(item)

    band_global_ids: dict[tuple[int, int], int] = {}
    for global_order, item in enumerate(ordered):
        item["global_order"] = global_order
        item["column_fix_global_order"] = global_order
        item["layout_flow_order"] = global_order
        page_idx = item.get("page_idx")
        band_id = item.get("layout_band_id")
        if isinstance(page_idx, int) and isinstance(band_id, int):
            key = (page_idx, band_id)
            if key not in band_global_ids:
                band_global_ids[key] = len(band_global_ids)
            item["layout_band_global_id"] = band_global_ids[key]
            item["layout_band_global_order"] = band_global_ids[key]
        else:
            item["layout_band_global_id"] = None
            item["layout_band_global_order"] = None
    annotate_repeated_header_footer_layers(ordered)
    annotate_footnote_layers(ordered)
    refine_front_matter_layers(ordered)
    mark_toc_layers(ordered)
    annotate_run_in_headings(ordered)
    ordered = annotate_duplicate_contained_continuations(ordered)

    refreshed = dict(payload)
    refreshed["schema_version"] = str(payload.get("schema_version") or "content_v7_columnfix_listmarkers")
    refreshed["items"] = ordered
    refreshed["layout_metadata_refreshed"] = True
    refreshed["layout_metadata_version"] = "layer_band_v1"
    config = dict(refreshed.get("config") or {})
    config["page_object_layers"] = True
    config["local_band_metadata"] = True
    config["duplicate_contained_continuation_detection"] = True
    refreshed["config"] = config
    return refreshed


def fix_columnar_reading_order(
    page_nodes: list[dict[str, Any]],
    *,
    full_span_width_ratio: float = 0.65,
    center_margin_ratio: float = 0.05,
) -> list[dict[str, Any]]:
    """Repair Z-shaped dual-column order for one page without touching bboxes.

    FULL_SPAN nodes become their own vertical blocks. Consecutive HALF_SPAN
    nodes between FULL_SPAN separators are sorted as left column top-to-bottom
    followed by right column top-to-bottom.
    """

    nodes = [dict(node) for node in page_nodes if isinstance(node, dict)]
    if len(nodes) <= 1:
        return _annotate_columnar_order(nodes, page_width=0.0, center_x=0.0, margin=0.0)

    page_width, center_x = _columnar_page_width_and_center(nodes)
    if page_width <= 0.0:
        return _annotate_columnar_order(nodes, page_width=0.0, center_x=0.0, margin=0.0)
    margin = center_margin_ratio * page_width

    views = [
        _columnar_order_view(
            node,
            original_index=index,
            page_width=page_width,
            center_x=center_x,
            margin=margin,
            full_span_width_ratio=full_span_width_ratio,
        )
        for index, node in enumerate(nodes)
    ]
    y_ordered = sorted(views, key=_columnar_top_key)

    blocks: list[tuple[str, list[ColumnarOrderView]]] = []
    current_half_block: list[ColumnarOrderView] = []
    for row in _cluster_columnar_rows(y_ordered):
        if _row_is_float_group(row, page_width=page_width):
            if current_half_block:
                blocks.append(("DOUBLE_COLUMN_BLOCK", current_half_block))
                current_half_block = []
            blocks.append(("FLOAT_GROUP_BLOCK", _sort_visual_row(row)))
            continue
        for view in row:
            if view.span_label == "FULL_SPAN":
                if current_half_block:
                    blocks.append(("DOUBLE_COLUMN_BLOCK", current_half_block))
                    current_half_block = []
                blocks.append(("FULL_SPAN", [view]))
                continue
            current_half_block.append(view)
    if current_half_block:
        blocks.append(("DOUBLE_COLUMN_BLOCK", current_half_block))

    ordered_entries: list[tuple[ColumnarOrderView, int, str, int, int]] = []
    for band_id, (block_type, block) in enumerate(blocks):
        if block_type == "FULL_SPAN":
            for local_order, view in enumerate(block):
                ordered_entries.append((view, band_id, "full_span", local_order, 2))
            continue
        if block_type == "FLOAT_GROUP_BLOCK":
            for local_order, view in enumerate(_sort_visual_row(block)):
                ordered_entries.append((view, band_id, "float_group", local_order, 2))
            continue
        combined = _sort_double_column_block(block, future_blocks=blocks[band_id + 1 :])
        for local_order, view in enumerate(combined):
            band_column_id = 0 if view.column_label == "LEFT_COL" else 1
            ordered_entries.append((view, band_id, "double_column", local_order, band_column_id))

    ordered_nodes = []
    for order, (view, band_id, band_type, band_local_order, band_column_id) in enumerate(ordered_entries):
        node = dict(view.node)
        layer = infer_layout_layer(node)
        node["column_fix_index"] = order
        node["column_fix_span"] = view.span_label
        node["column_fix_column"] = view.column_label
        node["column_fix_page_width"] = page_width
        node["column_fix_center_x"] = center_x
        node["column_fix_center_margin"] = margin
        node["layout_layer"] = layer
        node["layout_role"] = infer_layout_role(node, layer=layer)
        node["layout_probes"] = layout_probe_payloads(node)
        node["layout_band_id"] = band_id
        node["layout_band_type"] = band_type
        node["layout_band_local_order"] = band_local_order
        node["layout_band_column_id"] = band_column_id
        node["layout_band_column"] = {0: "left", 1: "right", 2: "full"}.get(band_column_id, "unknown")
        node["layout_is_band_boundary"] = is_layout_band_boundary_node(node, span_label=view.span_label, layer=layer)
        node["is_main_flow_candidate"] = layer == LAYOUT_LAYER_MAIN_TEXT
        ordered_nodes.append(node)
    _annotate_float_groups(ordered_nodes)
    _annotate_float_caption_groups(ordered_nodes)
    _annotate_adjacent_figure_fragments(ordered_nodes)
    return ordered_nodes


def fuse_micro_nodes(
    nodes: list[dict[str, Any]],
    *,
    y_center_tolerance_ratio: float = 0.30,
    small_equation_width_ratio: float = 0.35,
    small_equation_max_height: float = 48.0,
    max_inline_gap: float = 80.0,
) -> list[dict[str, Any]]:
    """Fuse same-line text/math shards into stable macro text nodes.

    This is intentionally a pre-reading-order pass. MinerU can emit inline
    formulas and short text fragments as separate physical nodes; feeding those
    shards directly to a GNN makes same-line order depend on noisy 2D topology.
    We first cluster mergeable shards by physical text line, sort them by x0,
    and replace each local run with one text node whose inline math is wrapped
    in ``$...$``.
    """

    if not nodes:
        return []

    normalized_nodes = [dict(node) for node in nodes if isinstance(node, dict)]
    candidates: list[tuple[int, dict[str, Any], tuple[float, float, float, float], int]] = []
    candidate_indexes: set[int] = set()
    for index, node in enumerate(normalized_nodes):
        bbox = _micro_single_bbox(node)
        page = _micro_node_page(node)
        if bbox is None or page is None:
            continue
        if not _is_micro_mergeable_node(
            node,
            bbox,
            page_width=_micro_page_width(normalized_nodes, page),
            small_equation_width_ratio=small_equation_width_ratio,
            small_equation_max_height=small_equation_max_height,
        ):
            continue
        candidates.append((index, node, bbox, page))
        candidate_indexes.add(index)

    if not candidates:
        return normalized_nodes

    lines_by_page: dict[int, list[list[tuple[int, dict[str, Any], tuple[float, float, float, float]]]]] = {}
    for index, node, bbox, page in sorted(candidates, key=lambda entry: (_micro_y_center(entry[2]), entry[2][0], entry[0])):
        page_lines = lines_by_page.setdefault(page, [])
        placed = False
        for line in page_lines:
            if _fits_micro_line(bbox, [entry[2] for entry in line], y_center_tolerance_ratio):
                line.append((index, node, bbox))
                placed = True
                break
        if not placed:
            page_lines.append([(index, node, bbox)])

    fused_nodes: list[dict[str, Any]] = []
    for page, lines in lines_by_page.items():
        page_width = _micro_page_width(normalized_nodes, page)
        for line in lines:
            sorted_line = sorted(line, key=lambda entry: (entry[2][0], entry[2][1], entry[0]))
            for run in _split_micro_line_by_gap(sorted_line, max_inline_gap=max_inline_gap, page_width=page_width):
                if len(run) == 1 and not _should_rewrite_single_micro_node(run[0][1]):
                    continue
                fused_nodes.append(_make_micro_fused_node(run, page=page))

    if not fused_nodes:
        return normalized_nodes

    fused_source_indexes = {
        index
        for fused in fused_nodes
        for index in fused.get("source_node_indexes", [])
        if isinstance(index, int)
    }
    output_with_position: list[tuple[int, dict[str, Any]]] = [
        (index, node)
        for index, node in enumerate(normalized_nodes)
        if index not in fused_source_indexes
    ]
    for fused in fused_nodes:
        source_indexes = [value for value in fused.get("source_node_indexes", []) if isinstance(value, int)]
        insertion_index = min(source_indexes) if source_indexes else len(normalized_nodes)
        output_with_position.append((insertion_index, fused))
    output = [node for _, node in sorted(output_with_position, key=lambda entry: entry[0])]
    for order, node in enumerate(output):
        node.setdefault("micro_fused_order", order)
    return output


def extract_text(block: dict[str, Any]) -> str:
    """Extract readable text from MinerU v2 nested content structures."""

    return _join_text_parts(_extract_text_parts(block.get("content")))


def _extract_text_parts(value: Any) -> list[str]:
    parts: list[str] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                parts.append(stripped)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if isinstance(value, dict):
            if "content" in value and isinstance(value.get("content"), str):
                visit(value["content"])
            for key in (
                "title_content",
                "paragraph_content",
                "math_content",
                "code_content",
                "algorithm_content",
                "item_content",
                "list_items",
                "table_caption",
                "chart_caption",
                "image_caption",
                "page_footnote_content",
            ):
                if key in value:
                    visit(value[key])

    visit(value)
    return parts


def _join_text_parts(parts: list[str]) -> str:
    return " ".join(parts)


def _extract_reference_items(block: dict[str, Any]) -> list[str]:
    content = block.get("content")
    if not isinstance(content, dict) or content.get("list_type") != REFERENCE_LIST_TYPE:
        return []
    items = content.get("list_items")
    if not isinstance(items, list):
        return []

    reference_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = _join_text_parts(_extract_text_parts(item.get("item_content")))
        if text:
            reference_items.append(text)
    return reference_items


def _native_v2_items(payload: Any) -> list[dict[str, Any]]:
    """Return MinerU native blocks in exact source order."""

    if isinstance(payload, list) and (not payload or all(isinstance(page, list) for page in payload)):
        items: list[dict[str, Any]] = []
        for page_idx, page in enumerate(payload):
            if not isinstance(page, list):
                continue
            for block_idx, block in enumerate(page):
                if isinstance(block, dict):
                    items.append({"page_idx": page_idx, "block_idx": block_idx, "block": dict(block)})
        return items

    if isinstance(payload, list) and all(isinstance(block, dict) for block in payload):
        items = []
        for block_idx, block in enumerate(payload):
            page_idx = block.get("page_idx")
            if not isinstance(page_idx, int):
                page_idx = 0
            items.append({"page_idx": page_idx, "block_idx": block_idx, "block": dict(block)})
        return items

    raise ValueError("Expected MinerU native content_list_v2 pages or a flat content_list")


def _annotate_columnar_order(
    nodes: list[dict[str, Any]],
    *,
    page_width: float,
    center_x: float,
    margin: float,
) -> list[dict[str, Any]]:
    annotated = []
    for order, node in enumerate(nodes):
        item = dict(node)
        layer = infer_layout_layer(item)
        item["column_fix_index"] = order
        item["column_fix_span"] = "FULL_SPAN"
        item["column_fix_column"] = None
        item["column_fix_page_width"] = page_width
        item["column_fix_center_x"] = center_x
        item["column_fix_center_margin"] = margin
        item["layout_layer"] = layer
        item["layout_role"] = infer_layout_role(item, layer=layer)
        item["layout_probes"] = layout_probe_payloads(item)
        item["layout_band_id"] = order
        item["layout_band_type"] = "full_span"
        item["layout_band_local_order"] = 0
        item["layout_band_column_id"] = 2
        item["layout_band_column"] = "full"
        item["layout_is_band_boundary"] = is_layout_band_boundary_node(item, span_label="FULL_SPAN", layer=layer)
        item["is_main_flow_candidate"] = layer == LAYOUT_LAYER_MAIN_TEXT
        annotated.append(item)
    return annotated


def _columnar_page_width_and_center(nodes: list[dict[str, Any]]) -> tuple[float, float]:
    text_boxes = [
        bbox
        for node in nodes
        for bbox in [_first_bbox(node.get("bbox"))]
        if bbox is not None and _is_columnar_textual_node(node)
    ]
    boxes = text_boxes or [
        bbox
        for node in nodes
        for bbox in [_first_bbox(node.get("bbox"))]
        if bbox is not None
    ]
    if not boxes:
        return 0.0, 0.0
    min_x = min(bbox[0] for bbox in boxes)
    max_x = max(bbox[2] for bbox in boxes)
    page_width = max(0.0, max_x - min_x)
    return page_width, min_x + page_width / 2.0


def _is_columnar_textual_node(node: dict[str, Any]) -> bool:
    node_type = str(node.get("type") or node.get("raw_type") or "").lower()
    if node_type in TEXTUAL_TYPES:
        return True
    if node.get("text_for_embedding") or node.get("text"):
        return True
    block = node.get("block")
    return isinstance(block, dict) and str(block.get("type") or "").lower() in TEXTUAL_TYPES


def _columnar_order_view(
    node: dict[str, Any],
    *,
    original_index: int,
    page_width: float,
    center_x: float,
    margin: float,
    full_span_width_ratio: float,
) -> ColumnarOrderView:
    bbox = _first_bbox(node.get("bbox"))
    if bbox is None:
        return ColumnarOrderView(node=node, original_index=original_index, bbox=None, span_label="FULL_SPAN", column_label=None)

    width = max(0.0, bbox[2] - bbox[0])
    crosses_center = bbox[0] < center_x - margin and bbox[2] > center_x + margin
    is_full_span = width > full_span_width_ratio * page_width or crosses_center
    if is_full_span:
        return ColumnarOrderView(node=node, original_index=original_index, bbox=bbox, span_label="FULL_SPAN", column_label=None)

    center = (bbox[0] + bbox[2]) / 2.0
    column_label = "LEFT_COL" if center <= center_x else "RIGHT_COL"
    return ColumnarOrderView(node=node, original_index=original_index, bbox=bbox, span_label="HALF_SPAN", column_label=column_label)


def _columnar_top_key(view: ColumnarOrderView) -> tuple[float, float, int]:
    return (view.y0, view.x0, view.original_index)


def _cluster_columnar_rows(views: list[ColumnarOrderView]) -> list[list[ColumnarOrderView]]:
    rows: list[list[ColumnarOrderView]] = []
    for view in sorted(views, key=_columnar_top_key):
        placed = False
        for row in rows:
            if _view_fits_row(view, row):
                row.append(view)
                placed = True
                break
        if not placed:
            rows.append([view])
    return [sorted(row, key=_columnar_top_key) for row in rows]


def _view_fits_row(view: ColumnarOrderView, row: list[ColumnarOrderView]) -> bool:
    if view.bbox is None or not row:
        return False
    row_y0 = min(member.y0 for member in row)
    row_y1 = max(member.y1 for member in row)
    height = max(view.y1 - view.y0, 1.0)
    row_height = max(row_y1 - row_y0, 1.0)
    tolerance = max(6.0, 0.35 * (height + row_height) / 2.0)
    center = (view.y0 + view.y1) / 2.0
    row_center = (row_y0 + row_y1) / 2.0
    if abs(center - row_center) <= tolerance:
        return True
    intersection = max(0.0, min(view.y1, row_y1) - max(view.y0, row_y0))
    return intersection / max(1.0, min(height, row_height)) >= 0.55


def _row_is_float_group(row: list[ColumnarOrderView], *, page_width: float) -> bool:
    float_views = [view for view in row if infer_layout_layer(view.node) == LAYOUT_LAYER_FLOAT]
    if len(float_views) < 2:
        return False
    if len(float_views) < len(row):
        return False
    caption_numbers = {
        match.group(2)
        for view in float_views
        for match in [re.search(r"\b(fig(?:ure)?|table)\s*\.?\s*(\d+)", _layout_text(view.node), re.IGNORECASE)]
        if match is not None
    }
    if len(caption_numbers) > 1:
        return False
    boxes = [view.bbox for view in float_views if view.bbox is not None]
    if len(boxes) < 2:
        return False
    union_width = max(box[2] for box in boxes) - min(box[0] for box in boxes)
    return union_width >= max(0.45 * page_width, max((box[2] - box[0] for box in boxes), default=0.0))


def _sort_columnar_column(views: list[ColumnarOrderView]) -> list[ColumnarOrderView]:
    ordered: list[ColumnarOrderView] = []
    for row in _cluster_columnar_rows(views):
        if any(infer_layout_layer(view.node) == LAYOUT_LAYER_FLOAT for view in row):
            ordered.extend(_sort_visual_row(row))
        else:
            ordered.extend(sorted(row, key=_columnar_top_key))
    return ordered


def _sort_double_column_block(
    block: list[ColumnarOrderView],
    *,
    future_blocks: list[tuple[str, list[ColumnarOrderView]]],
) -> list[ColumnarOrderView]:
    """Sort a local two-column block without letting floats leak past a single-column transition.

    The default contract remains left-column top-to-bottom, then right-column
    top-to-bottom.  A narrow exception handles mixed layouts where a lower
    left-column heading starts a full-width/single-column section while a right
    column table/figure physically above that heading still belongs to the
    previous section.  In that case, only the right-column float/caption nodes
    above the transition heading are flushed before the heading.
    """

    left_list = _sort_columnar_column([view for view in block if view.column_label == "LEFT_COL"])
    right_list = _sort_columnar_column([view for view in block if view.column_label == "RIGHT_COL"])
    transition_headings = [
        view
        for view in left_list
        if _starts_single_column_flow_after_double_block(view, future_blocks=future_blocks)
    ]
    if not transition_headings:
        return left_list + right_list

    remaining_right = list(right_list)
    combined: list[ColumnarOrderView] = []
    for left_view in left_list:
        if left_view in transition_headings:
            flush = [
                right_view
                for right_view in remaining_right
                if _right_float_precedes_transition_heading(right_view, left_view)
            ]
            if flush:
                combined.extend(flush)
                flushed_ids = {id(view) for view in flush}
                remaining_right = [view for view in remaining_right if id(view) not in flushed_ids]
        combined.append(left_view)
    combined.extend(remaining_right)
    return combined


def _starts_single_column_flow_after_double_block(
    view: ColumnarOrderView,
    *,
    future_blocks: list[tuple[str, list[ColumnarOrderView]]],
) -> bool:
    if view.column_label != "LEFT_COL":
        return False
    if infer_layout_layer(view.node) != LAYOUT_LAYER_MAIN_TEXT:
        return False
    if infer_layout_role(view.node, layer=LAYOUT_LAYER_MAIN_TEXT) != "heading":
        return False

    for block_type, block in future_blocks[:3]:
        if not block:
            continue
        if block_type == "FLOAT_GROUP_BLOCK":
            continue
        first = block[0]
        if first.y0 < view.y0:
            continue
        layer = infer_layout_layer(first.node)
        role = infer_layout_role(first.node, layer=layer)
        if block_type == "FULL_SPAN" and layer == LAYOUT_LAYER_MAIN_TEXT and role != "heading":
            return True
        if layer != LAYOUT_LAYER_NOISE:
            return False
    return False


def _right_float_precedes_transition_heading(
    right_view: ColumnarOrderView,
    heading_view: ColumnarOrderView,
) -> bool:
    if right_view.column_label != "RIGHT_COL":
        return False
    if right_view.y0 >= heading_view.y0:
        return False
    layer = infer_layout_layer(right_view.node)
    if layer == LAYOUT_LAYER_FLOAT:
        return True
    return bool(_float_caption_text(right_view.node))


def _sort_visual_row(row: list[ColumnarOrderView]) -> list[ColumnarOrderView]:
    return sorted(row, key=lambda view: (view.x0, view.y0, view.original_index))


def _annotate_float_groups(nodes: list[dict[str, Any]]) -> None:
    groups: dict[tuple[int, int], list[int]] = {}
    for index, node in enumerate(nodes):
        if node.get("layout_band_type") != "float_group":
            continue
        if infer_layout_layer(node) != LAYOUT_LAYER_FLOAT:
            continue
        page_idx = _numeric_or_none(node.get("page_idx"))
        if page_idx is None:
            page_idx = 0
        band_id = _numeric_or_none(node.get("layout_band_id"))
        if page_idx is None or band_id is None:
            continue
        groups.setdefault((page_idx, band_id), []).append(index)
    for group_number, ((page_idx, band_id), indexes) in enumerate(sorted(groups.items())):
        if len(indexes) < 2:
            continue
        boxes = [_parse_bbox(nodes[index].get("bbox")) for index in indexes]
        boxes = [box for box in boxes if box is not None]
        if not boxes:
            continue
        union = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
        ordered = sorted(indexes, key=lambda index: (_parse_bbox(nodes[index].get("bbox")) or (0, 0, 0, 0))[0])
        captioned = [index for index in ordered if _float_caption_text(nodes[index])]
        primary = captioned[-1] if captioned else ordered[0]
        group_id = f"figure_group_p{page_idx:04d}_{band_id:04d}_{group_number:04d}"
        member_ids = [str(nodes[index].get("global_order", nodes[index].get("original_index", index))) for index in ordered]
        caption = _float_caption_text(nodes[primary])
        for member_index, index in enumerate(ordered):
            nodes[index].update(
                {
                    "figure_group_id": group_id,
                    "image_group_id": group_id,
                    "figure_group_member_ids": member_ids,
                    "figure_group_member_index": member_index,
                    "figure_group_size": len(ordered),
                    "figure_group_primary": index == primary,
                    "figure_group_bbox": union,
                    "image_group_bbox": union,
                    "figure_group_caption": caption,
                    "figure_group_render_strategy": "union_pdf_crop",
                }
            )


def _float_caption_text(node: dict[str, Any]) -> str:
    text = _layout_text(node)
    match = re.search(r"\b((?:Fig\.?|Figure|Table|Algorithm)\s*\.?\s*[A-Za-z]?\d+(?:\.\d+)*[:.\-]?\s+.+)$", text, re.IGNORECASE | re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    return ""


def _annotate_float_caption_groups(nodes: list[dict[str, Any]]) -> None:
    for caption_index, caption_node in enumerate(nodes):
        role = str(caption_node.get("layout_role") or "").casefold()
        if not role.endswith("_caption"):
            continue
        caption_kind = role.removesuffix("_caption")
        if caption_kind == "fig":
            caption_kind = "figure"
        target_index = _nearest_float_for_caption(nodes, caption_index, caption_kind=caption_kind)
        if target_index is None:
            continue
        target = nodes[target_index]
        group_key = "table_group_id" if caption_kind == "table" else "figure_group_id"
        caption_key = "table_group_caption" if caption_kind == "table" else "figure_group_caption"
        bbox_key = "table_group_bbox" if caption_kind == "table" else "figure_group_bbox"
        strategy_key = "table_group_render_strategy" if caption_kind == "table" else "figure_group_render_strategy"
        group_id = str(target.get(group_key) or f"{caption_kind}_group_p{_numeric_or_none(target.get('page_idx')) or 0:04d}_{target_index:04d}")
        boxes = [_first_bbox(target.get("bbox")), _first_bbox(caption_node.get("bbox"))]
        boxes = [box for box in boxes if box is not None]
        union = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ] if boxes else None
        caption_text = _float_caption_text(caption_node)
        member_ids = [
            str(target.get("global_order", target_index)),
            str(caption_node.get("global_order", caption_index)),
        ]
        for member_index, index in enumerate((target_index, caption_index)):
            nodes[index].update(
                {
                    group_key: group_id,
                    caption_key: caption_text,
                    bbox_key: union,
                    strategy_key: "caption_attached_pdf_crop",
                    f"{caption_kind}_group_member_ids": member_ids,
                    f"{caption_kind}_group_member_index": member_index,
                    f"{caption_kind}_group_size": len(member_ids),
                    f"{caption_kind}_group_primary": index == target_index,
                }
            )
        if caption_kind == "figure":
            target["image_group_id"] = group_id
            target["image_group_caption"] = caption_text
            target["image_group_bbox"] = union
            caption_node["image_group_id"] = group_id
            caption_node["image_group_caption"] = caption_text
            caption_node["image_group_bbox"] = union


def _annotate_adjacent_figure_fragments(nodes: list[dict[str, Any]]) -> None:
    figure_indexes = [
        index
        for index, node in enumerate(nodes)
        if _is_groupable_figure_node(node)
    ]
    if len(figure_indexes) < 2:
        return
    parent = {index: index for index in figure_indexes}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for offset, left_index in enumerate(figure_indexes):
        for right_index in figure_indexes[offset + 1 :]:
            if _should_group_adjacent_figure_nodes(nodes[left_index], nodes[right_index], nodes):
                union(left_index, right_index)

    components: dict[int, list[int]] = {}
    for index in figure_indexes:
        components.setdefault(find(index), []).append(index)
    for component_number, indexes in enumerate(components.values()):
        if len(indexes) < 2:
            continue
        ordered = sorted(indexes, key=lambda index: _figure_fragment_sort_key(nodes[index], index))
        boxes = [_first_bbox(nodes[index].get("bbox")) for index in ordered]
        boxes = [box for box in boxes if box is not None]
        if not boxes:
            continue
        primary = _figure_fragment_primary(nodes, ordered)
        page_idx = _numeric_or_none(nodes[primary].get("page_idx")) or 0
        existing_group_id = next((str(nodes[index].get("figure_group_id")) for index in ordered if nodes[index].get("figure_group_id")), None)
        group_id = existing_group_id or f"figure_group_p{page_idx:04d}_adj_{component_number:04d}"
        union_box = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
        caption = _figure_fragment_caption(nodes[primary])
        member_ids = [str(nodes[index].get("global_order", nodes[index].get("original_index", index))) for index in ordered]
        for member_index, index in enumerate(ordered):
            nodes[index].update(
                {
                    "figure_group_id": group_id,
                    "image_group_id": group_id,
                    "figure_group_member_ids": member_ids,
                    "image_group_member_ids": member_ids,
                    "figure_group_member_index": member_index,
                    "image_group_member_index": member_index,
                    "figure_group_size": len(ordered),
                    "image_group_size": len(ordered),
                    "figure_group_primary": index == primary,
                    "image_group_primary": index == primary,
                    "figure_group_bbox": union_box,
                    "image_group_bbox": union_box,
                    "figure_group_caption": caption,
                    "image_group_caption": caption,
                    "figure_group_render_strategy": "union_pdf_crop",
                    "image_group_render_strategy": "union_pdf_crop",
                }
            )


def _is_groupable_figure_node(node: dict[str, Any]) -> bool:
    if str(node.get("layout_role") or "").casefold().endswith("_caption"):
        return False
    return infer_layout_layer(node) == LAYOUT_LAYER_FLOAT and _layout_raw_type(node) in {"figure", "image", "chart"}


def _should_group_adjacent_figure_nodes(
    left: dict[str, Any],
    right: dict[str, Any],
    page_nodes: list[dict[str, Any]],
) -> bool:
    left_page = _numeric_or_none(left.get("page_idx")) or 0
    right_page = _numeric_or_none(right.get("page_idx")) or 0
    if left_page != right_page:
        return False
    left_group = str(left.get("figure_group_id") or left.get("image_group_id") or "")
    right_group = str(right.get("figure_group_id") or right.get("image_group_id") or "")
    if left_group and right_group and left_group != right_group:
        return False
    left_box = _first_bbox(left.get("bbox"))
    right_box = _first_bbox(right.get("bbox"))
    if left_box is None or right_box is None:
        return False
    page_width, page_height = _page_extent_for_items(page_nodes, page_idx=left_page)
    y_overlap = _y_overlap_ratio(left_box, right_box)
    x_overlap = _x_overlap_ratio(left_box, right_box)
    x_gap = max(0.0, max(left_box[0], right_box[0]) - min(left_box[2], right_box[2]))
    y_gap = max(0.0, max(left_box[1], right_box[1]) - min(left_box[3], right_box[3]))
    same_row_close = y_overlap >= 0.12 and x_gap <= 0.18 * max(page_width, 1.0)
    stacked_close = x_overlap >= 0.18 and y_gap <= 0.10 * max(page_height, 1.0)
    if not (same_row_close or stacked_close):
        return False
    left_caption = _figure_fragment_caption_identity(left)
    right_caption = _figure_fragment_caption_identity(right)
    if left_caption and right_caption:
        return left_caption == right_caption
    if left_caption or right_caption:
        return True
    order_gap = abs((_numeric_or_none(left.get("global_order")) or 0) - (_numeric_or_none(right.get("global_order")) or 0))
    return same_row_close and order_gap <= 6


FIGURE_FRAGMENT_LABEL_RE = re.compile(r"\b(?:fig\.?|figure)\s*\.?\s*([A-Za-z]?\d+(?:\.\d+)*)", re.IGNORECASE)


def _figure_fragment_caption_identity(node: dict[str, Any]) -> tuple[str, str] | None:
    text = _figure_fragment_caption(node)
    if not text:
        return None
    label_match = FIGURE_FRAGMENT_LABEL_RE.search(text)
    if label_match:
        return ("label", label_match.group(1).casefold())
    cleaned = re.sub(r"[^a-z0-9]+", "", text.casefold())
    if len(cleaned) >= 10 and cleaned not in {"figure", "image", "fig"}:
        return ("text", cleaned)
    return None


def _figure_fragment_caption(node: dict[str, Any]) -> str:
    for key in ("figure_group_caption", "image_group_caption", "figure_caption", "image_caption", "caption"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return _float_caption_text(node)


def _figure_fragment_primary(nodes: list[dict[str, Any]], indexes: list[int]) -> int:
    existing_primary = [
        index
        for index in indexes
        if bool(nodes[index].get("figure_group_primary") or nodes[index].get("image_group_primary"))
        and bool(_figure_fragment_caption_identity(nodes[index]) or _figure_fragment_caption(nodes[index]))
    ]
    if existing_primary:
        return min(existing_primary, key=lambda index: _figure_fragment_sort_key(nodes[index], index))
    captioned = [index for index in indexes if _figure_fragment_caption_identity(nodes[index])]
    if captioned:
        return min(captioned, key=lambda index: _figure_fragment_sort_key(nodes[index], index))
    return min(indexes, key=lambda index: _figure_fragment_sort_key(nodes[index], index))


def _figure_fragment_sort_key(node: dict[str, Any], fallback: int) -> tuple[int, float, float, int]:
    member_index = _numeric_or_none(node.get("figure_group_member_index"))
    if member_index is None:
        member_index = _numeric_or_none(node.get("image_group_member_index"))
    box = _first_bbox(node.get("bbox")) or (0.0, 0.0, 0.0, 0.0)
    if member_index is not None:
        return (0, float(member_index), box[0], fallback)
    return (1, box[1], box[0], fallback)


def _page_extent_for_items(nodes: list[dict[str, Any]], *, page_idx: int) -> tuple[float, float]:
    boxes = [
        box
        for node in nodes
        if (_numeric_or_none(node.get("page_idx")) or 0) == page_idx
        for box in [_first_bbox(node.get("bbox"))]
        if box is not None
    ]
    if not boxes:
        return 1000.0, 1000.0
    return max(max(box[2] for box in boxes), 1000.0), max(max(box[3] for box in boxes), 1000.0)


def _y_overlap_ratio(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    intersection = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    min_height = max(1.0, min(left[3] - left[1], right[3] - right[1]))
    return intersection / min_height


def _nearest_float_for_caption(nodes: list[dict[str, Any]], caption_index: int, *, caption_kind: str) -> int | None:
    caption = nodes[caption_index]
    caption_box = _first_bbox(caption.get("bbox"))
    if caption_box is None:
        return None
    page_idx = _numeric_or_none(caption.get("page_idx")) or 0
    best: tuple[float, int] | None = None
    for index, node in enumerate(nodes):
        if index == caption_index:
            continue
        if (_numeric_or_none(node.get("page_idx")) or 0) != page_idx:
            continue
        if infer_layout_layer(node) != LAYOUT_LAYER_FLOAT:
            continue
        if str(node.get("layout_role") or "").casefold().endswith("_caption"):
            continue
        if not _float_kind_matches_caption(node, caption_kind):
            continue
        box = _first_bbox(node.get("bbox"))
        if box is None:
            continue
        overlap = _x_overlap_ratio(caption_box, box)
        if overlap < 0.20:
            continue
        gap = max(0.0, max(caption_box[1], box[1]) - min(caption_box[3], box[3]))
        if gap > 180.0:
            continue
        order_distance = abs(int(_numeric_or_none(caption.get("global_order")) or caption_index) - int(_numeric_or_none(node.get("global_order")) or index))
        score = gap + 12.0 * order_distance - 25.0 * overlap
        if best is None or score < best[0]:
            best = (score, index)
    return best[1] if best is not None else None


def _float_kind_matches_caption(node: dict[str, Any], caption_kind: str) -> bool:
    raw = _layout_raw_type(node)
    if caption_kind == "figure":
        return raw in {"figure", "image", "chart"}
    if caption_kind == "table":
        return raw == "table"
    if caption_kind == "algorithm":
        return raw == "algorithm"
    return False


def _x_overlap_ratio(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    min_width = max(1.0, min(left[2] - left[0], right[2] - right[0]))
    return intersection / min_width


def infer_layout_layer(node: dict[str, Any]) -> str:
    """Assign a coarse page-object layer before graph construction.

    This keeps main text flow, floats, display math, metadata, and visual noise
    from competing inside one undifferentiated reading-order sequence.
    """

    node_type = _layout_raw_type(node)
    text = _layout_text(node)
    if node_type in TOC_ENTRY_TYPES or is_toc_title_text(text):
        return LAYOUT_LAYER_METADATA
    if has_strong_layout_probe(node, text=text, roles={"footer"}):
        return LAYOUT_LAYER_NOISE
    if node_type in FOOTNOTE_LAYOUT_TYPES or str(node.get("layout_role") or "").casefold() in {"footnote", "page_footnote"}:
        return LAYOUT_LAYER_ANNOTATION
    if node_type in MARGIN_NOTE_LAYOUT_TYPES or str(node.get("layout_role") or "").casefold() in {"margin_note", "marginnote", "side_note", "sidenote"}:
        return LAYOUT_LAYER_ANNOTATION
    if _caption_kind(text) is not None:
        return LAYOUT_LAYER_FLOAT
    if str(node.get("list_type") or "").lower() == REFERENCE_LIST_TYPE:
        return LAYOUT_LAYER_MAIN_TEXT
    if node_type in NOISE_LAYOUT_TYPES:
        return LAYOUT_LAYER_NOISE
    if node_type in FLOAT_LAYOUT_TYPES:
        return LAYOUT_LAYER_FLOAT
    if node_type in MATH_LAYOUT_TYPES:
        return LAYOUT_LAYER_MATH
    if node_type in {"reference", "references", "bibliography", "list", "paragraph", "text", "code"}:
        return LAYOUT_LAYER_MAIN_TEXT
    if node_type in METADATA_LAYOUT_TYPES:
        return LAYOUT_LAYER_METADATA if _looks_like_front_matter(node, text) else LAYOUT_LAYER_MAIN_TEXT
    if text:
        return LAYOUT_LAYER_MAIN_TEXT
    return LAYOUT_LAYER_OTHER


def infer_layout_role(node: dict[str, Any], *, layer: str | None = None) -> str:
    node_type = _layout_raw_type(node)
    layer = layer or infer_layout_layer(node)
    text = _layout_text(node)
    if node_type in TOC_ENTRY_TYPES:
        return TOC_ENTRY_ROLE
    if is_toc_title_text(text):
        return TOC_TITLE_ROLE
    if layer == LAYOUT_LAYER_NOISE:
        return "noise"
    caption = _caption_kind(text)
    if layer == LAYOUT_LAYER_FLOAT and caption is not None:
        return f"{caption}_caption"
    if layer == LAYOUT_LAYER_FLOAT:
        return node_type or "float"
    if layer == LAYOUT_LAYER_MATH:
        return "inline_math" if node_type in MICRO_INLINE_MATH_TYPES else "display_math"
    if layer == LAYOUT_LAYER_ANNOTATION:
        if node_type in MARGIN_NOTE_LAYOUT_TYPES:
            return "margin_note"
        return "footnote"
    if str(node.get("list_type") or "").lower() == REFERENCE_LIST_TYPE:
        return "reference_list"
    if node_type in {"title", "section", "subsection", "subsubsection", "heading"}:
        return "heading"
    if detect_list_marker(text):
        return "list_item"
    return "body_text" if layer == LAYOUT_LAYER_MAIN_TEXT else layer


def is_layout_band_boundary_node(node: dict[str, Any], *, span_label: str, layer: str | None = None) -> bool:
    layer = layer or infer_layout_layer(node)
    if layer in {LAYOUT_LAYER_FLOAT, LAYOUT_LAYER_MATH, LAYOUT_LAYER_METADATA, LAYOUT_LAYER_ANNOTATION}:
        return True
    return span_label == "FULL_SPAN" and infer_layout_role(node, layer=layer) in {"heading", "reference_list"}


def _layout_raw_type(node: dict[str, Any]) -> str:
    block = node.get("block")
    block_type = block.get("type") if isinstance(block, dict) else None
    return str(node.get("type") or node.get("raw_type") or block_type or "").strip().lower()


def _layout_text(node: dict[str, Any]) -> str:
    for key in ("text_for_embedding", "text", "content", "latex"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value
    span_text = style_spans_text(node.get("style_spans"))
    if span_text:
        return span_text
    block = node.get("block")
    if isinstance(block, dict):
        text = extract_text(block)
        if text:
            return text
    return ""


def _caption_kind(text: str) -> str | None:
    match = CAPTION_LABEL_RE.match(str(text or ""))
    if not match:
        return None
    raw = match.group("kind").casefold().rstrip(".")
    if raw == "fig":
        return "figure"
    return raw


def layout_probe_payloads(node: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": probe.role,
            "confidence": probe.confidence,
            "reason": probe.reason,
            "scope": probe.scope,
            "strength": probe.strength,
        }
        for probe in collect_layout_probes(node, text=_layout_text(node))
    ]


def hydrate_empty_text_fields(node: dict[str, Any]) -> None:
    """Fill empty v7 text fields from PyMuPDF spans without overriding OCR text."""

    if str(node.get("text_for_embedding") or node.get("text") or "").strip():
        return
    fallback = style_spans_text(node.get("style_spans"))
    if not fallback:
        block = node.get("block")
        fallback = extract_text(block) if isinstance(block, dict) else ""
    if not fallback.strip():
        return
    node.setdefault("text", fallback)
    node.setdefault("text_for_embedding", fallback)
    if not str(node.get("text") or "").strip():
        node["text"] = fallback
    if not str(node.get("text_for_embedding") or "").strip():
        node["text_for_embedding"] = fallback
    node["text_fallback_source"] = "style_spans"


def style_spans_text(spans: Any) -> str:
    if not isinstance(spans, list):
        return ""
    parts: list[str] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        text = str(span.get("text") or span.get("content") or "").strip()
        if text:
            parts.append(text)
    return join_text_fragments(parts)


def join_text_fragments(parts: list[str]) -> str:
    text = ""
    no_space_before = set(",.;:!?%)]}”’")
    no_space_after = set("([{“‘")
    for raw in parts:
        part = raw.strip()
        if not part:
            continue
        if not text:
            text = part
            continue
        if text[-1] in no_space_after or part[0] in no_space_before or text[-1] in {"-", "/", "\u2010", "\u2011"}:
            text += part
        else:
            text += " " + part
    return text


def _looks_like_front_matter(node: dict[str, Any], text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return (
        normalized in {"abstract", "author", "authors"}
        or normalized.startswith("abstract")
        or is_toc_title_text(text)
        or is_front_matter_date_text(text)
        or has_strong_layout_probe(node, text=text, roles={"abstract", "front_matter", "affiliation"})
    )


def _looks_like_footer_noise(node: dict[str, Any], text: str) -> bool:
    return has_strong_layout_probe(node, text=text, roles={"footer"})


def annotate_repeated_header_footer_layers(
    items: list[dict[str, Any]],
    *,
    zone_ratio: float = 0.18,
    min_pages: int = 2,
    min_page_ratio: float = 0.35,
) -> None:
    """Mark repeated top/bottom page furniture as noise while retaining it in full IR."""

    page_heights = _page_heights(items)
    page_count = max(1, len(page_heights))
    protected_title_indexes = _first_page_document_title_indexes(items)
    candidates: dict[tuple[str, str], list[int]] = {}
    for index, item in enumerate(items):
        text = _layout_text(item)
        page_idx = _numeric_or_none(item.get("page_idx")) or 0
        bbox = _first_bbox(item.get("bbox"))
        if bbox is None:
            continue
        page_height = page_heights.get(page_idx) or max(bbox[3], 1.0)
        zone = _page_furniture_zone(bbox, page_height, zone_ratio)
        if zone is None:
            continue
        if _looks_like_page_number_text(text):
            _mark_noise_item(item, role="page_number", reason="page_number_zone")
            continue
        normalized = _normalize_repeated_furniture_text(text)
        if not normalized:
            continue
        if len(normalized) > 90:
            continue
        candidates.setdefault((zone, normalized), []).append(index)

    threshold = max(min_pages, int(page_count * min_page_ratio + 0.999))
    for (zone, normalized), indexes in candidates.items():
        pages = {_numeric_or_none(items[index].get("page_idx")) or 0 for index in indexes}
        if len(pages) < threshold:
            continue
        for index in indexes:
            if index in protected_title_indexes:
                continue
            _mark_noise_item(items[index], role=zone, reason="repeated_page_furniture", normalized=normalized)


def annotate_footnote_layers(items: list[dict[str, Any]], *, bottom_zone_ratio: float = 0.22) -> None:
    """Promote visually bottom-anchored note text into the annotation layer."""

    page_heights = _page_heights(items)
    body_font_by_page = _body_font_by_page(items)
    for item in items:
        raw = _layout_raw_type(item)
        role = str(item.get("layout_role") or "").casefold()
        text = _layout_text(item)
        if not text:
            continue
        if raw in FOOTNOTE_LAYOUT_TYPES or role in {"footnote", "page_footnote"}:
            _mark_annotation_item(item, role="footnote", reason="raw_footnote_type")
            continue
        if raw in MARGIN_NOTE_LAYOUT_TYPES or role in {"margin_note", "marginnote", "side_note", "sidenote"}:
            _mark_annotation_item(item, role="margin_note", reason="raw_margin_note_type")
            continue
        if str(item.get("layout_layer") or "").casefold() in {LAYOUT_LAYER_NOISE, LAYOUT_LAYER_METADATA}:
            continue
        if raw not in {"paragraph", "text", "paragraph_text", "list"}:
            continue
        bbox = _first_bbox(item.get("bbox"))
        if bbox is None:
            continue
        page_idx = _numeric_or_none(item.get("page_idx")) or 0
        page_height = page_heights.get(page_idx) or max(bbox[3], 1.0)
        if bbox[1] < page_height * (1.0 - bottom_zone_ratio):
            continue
        if not FOOTNOTE_PREFIX_RE.match(text):
            continue
        font_size = _node_font_size(item)
        body_font = body_font_by_page.get(page_idx, 0.0)
        if body_font > 0 and font_size > 0 and font_size > body_font * 0.92:
            continue
        _mark_annotation_item(item, role="footnote", reason="bottom_marker_small_text")


def _mark_noise_item(item: dict[str, Any], *, role: str, reason: str, normalized: str | None = None) -> None:
    item["layout_layer"] = LAYOUT_LAYER_NOISE
    item["layout_role"] = role if role in {"header", "footer", "page_number"} else "noise"
    item["is_main_flow_candidate"] = False
    item["layout_noise_reason"] = reason
    if normalized is not None:
        item["layout_noise_normalized_text"] = normalized
    item["layout_probes"] = layout_probe_payloads(item)


def _mark_metadata_item(item: dict[str, Any], *, role: str, reason: str | None = None) -> None:
    item["layout_layer"] = LAYOUT_LAYER_METADATA
    item["layout_role"] = role
    item["is_main_flow_candidate"] = False
    item.pop("layout_noise_reason", None)
    item.pop("layout_noise_normalized_text", None)
    if reason:
        item["front_matter_reason"] = reason
    item["layout_probes"] = layout_probe_payloads(item)


def _mark_annotation_item(item: dict[str, Any], *, role: str, reason: str) -> None:
    item["layout_layer"] = LAYOUT_LAYER_ANNOTATION
    item["layout_role"] = role
    item["is_main_flow_candidate"] = False
    item["annotation_reason"] = reason
    item["layout_probes"] = layout_probe_payloads(item)


def _page_heights(items: list[dict[str, Any]]) -> dict[int, float]:
    heights: dict[int, float] = {}
    for item in items:
        page_idx = _numeric_or_none(item.get("page_idx")) or 0
        explicit = item.get("page_height")
        if isinstance(explicit, (int, float)) and explicit > 0:
            heights[page_idx] = max(float(explicit), heights.get(page_idx, 0.0))
            continue
        bbox = _first_bbox(item.get("bbox"))
        if bbox is not None:
            heights[page_idx] = max(float(bbox[3]), heights.get(page_idx, 0.0))
    return heights


def _body_font_by_page(items: list[dict[str, Any]]) -> dict[int, float]:
    weighted: dict[int, dict[float, int]] = {}
    for item in items:
        raw = _layout_raw_type(item)
        if raw in {"title", "section", "subsection", "subsubsection", "heading"}:
            continue
        if FOOTNOTE_PREFIX_RE.match(_layout_text(item)):
            continue
        if str(item.get("layout_layer") or "").casefold() in {LAYOUT_LAYER_NOISE, LAYOUT_LAYER_METADATA, LAYOUT_LAYER_FLOAT, LAYOUT_LAYER_MATH}:
            continue
        size = _node_font_size(item)
        if size <= 0:
            continue
        page_idx = _numeric_or_none(item.get("page_idx")) or 0
        bucket = weighted.setdefault(page_idx, {})
        rounded = round(size, 1)
        bucket[rounded] = bucket.get(rounded, 0) + max(1, min(len(_layout_text(item)), 120))
    return {page: max(values.items(), key=lambda item: item[1])[0] for page, values in weighted.items() if values}


def _node_font_size(item: dict[str, Any]) -> float:
    for key in ("style_baseline_size", "baseline_font_size", "font_size", "font_size_px", "avg_font_size"):
        value = item.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    spans = item.get("style_spans")
    if isinstance(spans, list):
        return _weighted_micro_font_size([span for span in spans if isinstance(span, dict)])
    return 0.0


def _normalize_repeated_furniture_text(text: str) -> str:
    value = " ".join(str(text or "").casefold().split())
    if not value:
        return ""
    value = re.sub(r"\d+", "#", value)
    value = re.sub(r"[^a-z#©]+", " ", value)
    value = " ".join(value.split())
    if len(value) <= 1:
        return ""
    return value


def _page_furniture_zone(bbox: tuple[float, float, float, float], page_height: float, zone_ratio: float) -> str | None:
    if page_height <= 0:
        return None
    center_y = (bbox[1] + bbox[3]) / 2.0
    if center_y <= page_height * zone_ratio:
        return "header"
    if center_y >= page_height * (1.0 - zone_ratio):
        return "footer"
    return None


def _looks_like_page_number_text(text: str) -> bool:
    value = str(text or "").strip()
    return bool(re.fullmatch(r"(?:\d{1,4}|[ivxlcdm]{1,8})", value, flags=re.IGNORECASE))


def _first_page_document_title_indexes(items: list[dict[str, Any]]) -> set[int]:
    """Protect the real first-page paper title from repeated-header cleanup.

    Some conference templates repeat the paper title as a running header on
    later pages.  The repeated-header detector must remove those later running
    headers, but the original first-page title is still front matter and must
    survive as ``document_title``.
    """

    page_heights = _page_heights(items)
    candidates: list[tuple[float, float, int]] = []
    for index, item in enumerate(items):
        if item.get("page_idx") not in {0, None}:
            continue
        raw = _layout_raw_type(item)
        if raw not in {"title", "section", "heading"}:
            continue
        text = _layout_text(item).strip()
        if not text:
            continue
        normalized = " ".join(text.casefold().split())
        if _front_matter_text_marker(text) or is_toc_title_text(text) or _is_body_heading_candidate(item):
            continue
        if normalized.startswith(("fig", "figure", "table", "algorithm", "appendix", "references", "bibliography")):
            continue
        bbox = _parse_bbox(item.get("bbox"))
        if bbox is None:
            continue
        page_height = page_heights.get(0) or max(float(bbox[3]), 1.0)
        if bbox[1] > page_height * 0.38:
            continue
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        font_size = _node_font_size(item)
        score = width + height * 18.0 + font_size * 45.0 + min(len(text), 180)
        candidates.append((score, bbox[1], index))
    if not candidates:
        return set()
    candidates.sort(reverse=True)
    best_score, best_y0, best_index = candidates[0]
    protected = {best_index}
    for score, y0, index in candidates[1:]:
        if abs(y0 - best_y0) <= 80.0 and score >= best_score * 0.45:
            protected.add(index)
    return protected


def refine_front_matter_layers(items: list[dict[str, Any]]) -> None:
    """Mark title/author/abstract material before the first body heading.

    MinerU uses the same raw ``title`` type for the paper title and section
    headings. After v7 has produced a global order, this pass separates the
    front matter so it can stay out of main paragraph merge/flow decisions.
    """

    first_body_heading_order: int | None = None
    first_body_heading_y0: float | None = None
    for index, item in enumerate(items):
        if item.get("page_idx") not in {0, None}:
            continue
        if not _is_body_heading_candidate(item):
            continue
        first_body_heading_order = index
        bbox = _parse_bbox(item.get("bbox"))
        first_body_heading_y0 = bbox[1] if bbox is not None else None
        break
    if first_body_heading_order is None:
        return
    protected_title_indexes = _first_page_document_title_indexes(items)
    for index in protected_title_indexes:
        if 0 <= index < len(items):
            _mark_metadata_item(items[index], role="document_title", reason="first_page_document_title")
    for index, item in enumerate(items[:first_body_heading_order]):
        raw = _layout_raw_type(item)
        text = _layout_text(item)
        if raw in NOISE_LAYOUT_TYPES or not text:
            continue
        if index in protected_title_indexes:
            continue
        if str(item.get("layout_layer") or "").casefold() in {LAYOUT_LAYER_NOISE, LAYOUT_LAYER_ANNOTATION}:
            continue
        if raw in {"title", "paragraph", "text", "author", "affiliation"} or _front_matter_text_marker(text):
            _mark_metadata_item(item, role=_front_matter_role(item))
    for index, item in enumerate(items):
        if _looks_like_top_page_author_block(
            item,
            first_body_heading_y0=first_body_heading_y0,
            item_order=index,
            first_body_heading_order=first_body_heading_order,
        ):
            _mark_metadata_item(item, role=_front_matter_role(item))
    _mark_pre_heading_epigraph_sources(
        items,
        first_body_heading_y0=first_body_heading_y0,
        first_body_heading_order=first_body_heading_order,
    )


def _is_body_heading_candidate(item: dict[str, Any]) -> bool:
    raw = _layout_raw_type(item)
    if raw in TOC_ENTRY_TYPES:
        return False
    if raw not in {"title", "section", "subsection", "subsubsection", "heading"}:
        return False
    normalized = " ".join(_layout_text(item).lower().split())
    if not normalized:
        return False
    if is_toc_title_text(normalized):
        return False
    if _front_matter_text_marker(normalized):
        return False
    if normalized.startswith("introduction") or normalized.startswith("1 introduction"):
        return True
    if normalized.startswith(("references", "bibliography", "appendix")):
        return True
    return bool(re.match(r"^(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+\S+", normalized))


def _front_matter_text_marker(text: str) -> bool:
    normalized = " ".join(str(text).lower().split())
    return (
        normalized.startswith(("abstract", "keywords", "author", "authors", "affiliation"))
        or is_toc_title_text(text)
        or is_front_matter_date_text(text)
        or has_strong_layout_probe({}, text=text, roles={"abstract", "front_matter"})
    )


def _looks_like_top_page_author_block(
    item: dict[str, Any],
    *,
    first_body_heading_y0: float | None,
    item_order: int | None = None,
    first_body_heading_order: int | None = None,
) -> bool:
    """Catch two-column author/affiliation blocks that sort after the first heading."""

    if item.get("page_idx") not in {0, None}:
        return False
    text = _layout_text(item).strip()
    if not text:
        return False
    raw = _layout_raw_type(item)
    if raw in NOISE_LAYOUT_TYPES or raw in FLOAT_LAYOUT_TYPES or raw in MATH_LAYOUT_TYPES:
        return False
    if str(item.get("layout_layer") or "").casefold() in {LAYOUT_LAYER_NOISE, LAYOUT_LAYER_ANNOTATION}:
        return False
    bbox = _parse_bbox(item.get("bbox"))
    if bbox is None:
        return False
    y0 = bbox[1]
    if first_body_heading_y0 is not None and y0 >= first_body_heading_y0:
        return False
    normalized = " ".join(text.lower().split())
    after_body_order = (
        item_order is not None and first_body_heading_order is not None and item_order >= first_body_heading_order
    )
    strong_author_signal = (
        "@" in text
        or has_strong_layout_probe(item, text=text, roles={"affiliation"})
        or (y0 <= 320 and _looks_like_author_name(text))
    )
    if after_body_order and not strong_author_signal:
        # Keep right-column body text out of metadata, but still rescue author
        # and affiliation blocks that are physically above the first section.
        return False
    if _front_matter_text_marker(text):
        return True
    if has_strong_layout_probe(item, text=text, roles={"front_matter", "affiliation"}):
        return True
    # Author names in two-column IEEE-like papers are short top-page lines.
    if y0 <= 280 and _looks_like_author_name(text) and not normalized.startswith(("fig", "table", "algorithm")):
        return True
    return False


def _mark_pre_heading_epigraph_sources(
    items: list[dict[str, Any]],
    *,
    first_body_heading_y0: float | None,
    first_body_heading_order: int | None,
) -> None:
    """Keep pre-section epigraph attribution lines out of the body graph.

    Some templates render ``\\epigraph{quote}{source}`` between abstract and the
    first section.  The quote body is usually sorted before the first heading,
    but a short right-aligned source line can be pulled after the heading by
    noisy column/band sorting.  If the line is still physically above the first
    body heading and looks like a compact attribution, it belongs to front
    matter, not to the following section.
    """

    if first_body_heading_y0 is None or first_body_heading_order is None:
        return

    has_pre_heading_frontmatter_text = False
    for item in items[:first_body_heading_order]:
        if item.get("page_idx") not in {0, None}:
            continue
        if str(item.get("layout_layer") or "").casefold() != LAYOUT_LAYER_METADATA:
            continue
        text = _layout_text(item).strip()
        if len(text) >= 40 and not _front_matter_text_marker(text):
            has_pre_heading_frontmatter_text = True
            break
    if not has_pre_heading_frontmatter_text:
        return

    for index, item in enumerate(items):
        if index < first_body_heading_order:
            continue
        if item.get("page_idx") not in {0, None}:
            continue
        raw = _layout_raw_type(item)
        if raw not in {"paragraph", "text", "paragraph_text", "title"}:
            continue
        if str(item.get("layout_layer") or "").casefold() in {LAYOUT_LAYER_NOISE, LAYOUT_LAYER_ANNOTATION}:
            continue
        bbox = _parse_bbox(item.get("bbox"))
        if bbox is None:
            continue
        if bbox[1] >= first_body_heading_y0:
            continue
        text = _layout_text(item).strip()
        current_role = str(item.get("layout_role") or "").casefold()
        if "@" in text or current_role in {"author", "affiliation", "email", "correspondence"}:
            continue
        if not _looks_like_epigraph_source_line(text):
            continue
        item["layout_layer"] = LAYOUT_LAYER_METADATA
        item["layout_role"] = "front_matter"
        item["is_main_flow_candidate"] = False
        item["front_matter_reason"] = "pre_heading_epigraph_source"


def _looks_like_epigraph_source_line(text: str) -> bool:
    stripped = " ".join(str(text or "").replace(",", " ").split())
    if not stripped or len(stripped) > 90:
        return False
    lowered = stripped.casefold()
    if lowered.startswith(("fig", "figure", "table", "algorithm", "abstract", "keywords")):
        return False
    if any(char.isdigit() for char in stripped):
        return False
    if re.search(r"[!?;:]", stripped):
        return False
    if _looks_like_author_name(stripped):
        return True
    stripped = stripped.lstrip("-–— ").strip()
    words = [word for word in re.split(r"\s+", stripped) if word]
    if not 1 <= len(words) <= 8:
        return False
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    if not alpha_words:
        return False
    capitalized = sum(1 for word in alpha_words if word[0].isupper())
    return capitalized >= max(1, int(len(alpha_words) * 0.6))


def _front_matter_role(item: dict[str, Any]) -> str:
    text = _layout_text(item)
    probe = best_layout_probe(item, text=text)
    if is_toc_title_text(text):
        return TOC_TITLE_ROLE
    if is_front_matter_date_text(text):
        return "front_matter"
    if "@" in text:
        return "affiliation"
    if probe is not None and probe.role == "affiliation" and probe.confidence >= 0.80:
        return "affiliation"
    if _front_matter_text_marker(text):
        return "abstract" if text.strip().lower().startswith("abstract") else "front_matter"
    if _looks_like_author_name(text):
        return "author"
    return "front_matter"


def is_toc_title_text(text: str) -> bool:
    normalized = re.sub(r"[^a-z]+", "", str(text or "").casefold())
    return normalized in {"contents", "tableofcontents"}


def is_toc_record(item: dict[str, Any]) -> bool:
    role = str(item.get("layout_role") or "").casefold()
    canonical = str(item.get("canonical_type") or "").casefold()
    raw = _layout_raw_type(item)
    return (
        role in {TOC_TITLE_ROLE, TOC_ENTRY_ROLE}
        or canonical in {"toc", TOC_TITLE_ROLE, TOC_ENTRY_ROLE}
        or raw in TOC_ENTRY_TYPES
        or is_toc_title_text(_layout_text(item))
    )


def filter_graph_content_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop non-trainable OCR artifacts before GNN featurization."""

    annotated = annotate_duplicate_contained_continuations(items)
    return [
        item
        for item in annotated
        if not is_toc_record(item)
        and not is_duplicate_shadow_record(item)
        and str(item.get("layout_layer") or "").casefold() not in {LAYOUT_LAYER_NOISE, LAYOUT_LAYER_ANNOTATION}
    ]


def annotate_duplicate_contained_continuations(
    items: list[dict[str, Any]],
    *,
    lookahead: int = 12,
    min_duplicate_chars: int = 12,
) -> list[dict[str, Any]]:
    """Mark later OCR fragments already contained in an earlier text node.

    MinerU occasionally emits one logical paragraph node whose text already
    contains a later continuation bbox, while also keeping that continuation as
    an independent node.  Feeding both nodes to the graph gives contradictory
    supervision: the later node is visually real but textually duplicated.  We
    keep the raw record intact and mark the later node as ``duplicate_shadow`` /
    ``no_render`` so graph building, label generation and IR rendering can skip
    it with a single stable flag.
    """

    normalized = [dict(item) for item in items if isinstance(item, dict)]
    if len(normalized) <= 1:
        return normalized

    texts = [_layout_text(item) for item in normalized]
    compact_texts = [_duplicate_compact_text(text) for text in texts]
    for source_index, source in enumerate(normalized):
        source_compact = compact_texts[source_index]
        if len(source_compact) < min_duplicate_chars * 2:
            continue
        if not _duplicate_continuation_source_type(source):
            continue
        for target_index in range(source_index + 1, min(len(normalized), source_index + lookahead + 1)):
            target = normalized[target_index]
            if is_duplicate_shadow_record(target):
                continue
            if not _duplicate_continuation_target_type(target):
                continue
            target_compact = compact_texts[target_index]
            if len(target_compact) < min_duplicate_chars:
                continue
            if not _looks_like_duplicate_continuation_text(texts[target_index]):
                continue
            if not _contained_near_source_tail(source_compact, target_compact):
                continue
            if not _duplicate_continuation_scope_is_plausible(source, target):
                continue
            target["duplicate_shadow"] = True
            target["no_render"] = True
            target["duplicate_shadow_of_index"] = source_index
            target["duplicate_shadow_of_global_order"] = source.get("global_order")
            target["duplicate_shadow_reason"] = "contained_continuation_tail"
            target["duplicate_shadow_text_preview"] = texts[target_index][:200]
    return normalized


def is_duplicate_shadow_record(item: dict[str, Any]) -> bool:
    return bool(item.get("duplicate_shadow") or item.get("no_render") or item.get("render_skip"))


def _duplicate_continuation_source_type(item: dict[str, Any]) -> bool:
    raw = _layout_raw_type(item)
    canonical = str(item.get("canonical_type") or "").casefold()
    if raw in DUPLICATE_CONTINUATION_TYPES or canonical in DUPLICATE_CONTINUATION_TYPES:
        return True
    return bool(_layout_text(item)) and infer_layout_layer(item) == LAYOUT_LAYER_MAIN_TEXT


def _duplicate_continuation_target_type(item: dict[str, Any]) -> bool:
    raw = _layout_raw_type(item)
    canonical = str(item.get("canonical_type") or "").casefold()
    if raw in {"title", "section", "subsection", "subsubsection", "heading"} or canonical == "title":
        return False
    if infer_layout_layer(item) != LAYOUT_LAYER_MAIN_TEXT:
        return False
    return raw in DUPLICATE_CONTINUATION_TYPES or canonical in DUPLICATE_CONTINUATION_TYPES or bool(_layout_text(item))


def _duplicate_compact_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"[^0-9a-z]+", "", value)


def _looks_like_duplicate_continuation_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    return bool(DUPLICATE_CONTINUATION_START_RE.match(value))


def _contained_near_source_tail(source_compact: str, target_compact: str) -> bool:
    if not source_compact or not target_compact:
        return False
    position = source_compact.find(target_compact)
    if position < 0:
        return False
    if position == 0 and len(source_compact) == len(target_compact):
        return False
    tail_start = max(0, len(source_compact) - max(len(target_compact) * 3, 80))
    return position >= tail_start or source_compact.endswith(target_compact)


def _duplicate_continuation_scope_is_plausible(source: dict[str, Any], target: dict[str, Any]) -> bool:
    source_page = _numeric_or_none(source.get("page_idx"))
    target_page = _numeric_or_none(target.get("page_idx"))
    if source_page is not None and target_page is not None:
        if target_page < source_page:
            return False
        if target_page - source_page > 1:
            return False
    source_band = _numeric_or_none(source.get("layout_band_global_id"))
    target_band = _numeric_or_none(target.get("layout_band_global_id"))
    if source_band is not None and target_band is not None and target_band < source_band:
        return False
    return True


def document_toc_metadata(items: list[dict[str, Any]]) -> dict[str, Any]:
    toc_items = [item for item in items if is_toc_record(item)]
    if not toc_items:
        return {"has_toc": False}

    orders = [
        _numeric_or_none(item.get("global_order"))
        for item in toc_items
        if _numeric_or_none(item.get("global_order")) is not None
    ]
    pages = [
        _numeric_or_none(item.get("page_idx"))
        for item in toc_items
        if _numeric_or_none(item.get("page_idx")) is not None
    ]
    return {
        "has_toc": True,
        "toc_order": min(orders) if orders else None,
        "toc_page_idx": min(pages) if pages else None,
        "toc_item_count": len(toc_items),
    }


def mark_toc_layers(items: list[dict[str, Any]]) -> None:
    """Mark MinerU ``index`` blocks and their Contents heading as TOC metadata."""

    toc_entry_indexes = [index for index, item in enumerate(items) if _layout_raw_type(item) in TOC_ENTRY_TYPES]
    toc_title_indexes: set[int] = set()
    for index in toc_entry_indexes:
        items[index]["layout_layer"] = LAYOUT_LAYER_METADATA
        items[index]["layout_role"] = TOC_ENTRY_ROLE
        items[index]["canonical_type"] = "toc"
        items[index]["is_main_flow_candidate"] = False

        cursor = index - 1
        skipped_noise = 0
        while cursor >= 0 and skipped_noise <= 3:
            previous = items[cursor]
            if _layout_raw_type(previous) in NOISE_LAYOUT_TYPES:
                skipped_noise += 1
                cursor -= 1
                continue
            if is_toc_title_text(_layout_text(previous)):
                toc_title_indexes.add(cursor)
            break

    for index, item in enumerate(items):
        if is_toc_title_text(_layout_text(item)):
            next_non_noise = None
            for cursor in range(index + 1, min(len(items), index + 5)):
                candidate = items[cursor]
                if _layout_raw_type(candidate) in NOISE_LAYOUT_TYPES:
                    continue
                next_non_noise = candidate
                break
            if next_non_noise is not None and _layout_raw_type(next_non_noise) in TOC_ENTRY_TYPES:
                toc_title_indexes.add(index)

    for index in toc_title_indexes:
        items[index]["layout_layer"] = LAYOUT_LAYER_METADATA
        items[index]["layout_role"] = TOC_TITLE_ROLE
        items[index]["canonical_type"] = "toc"
        items[index]["is_main_flow_candidate"] = False


def annotate_run_in_headings(items: list[dict[str, Any]]) -> None:
    """Mark paragraph blocks whose first spans encode a run-in heading.

    Some journals render subsections as ``3.1. Bold Title. body text`` inside a
    single physical paragraph bbox. MinerU correctly keeps one bbox, but the
    structure parser still needs to see the heading boundary.  We only trust the
    signal when PyMuPDF spans show a dotted numeric prefix followed immediately
    by a bold title span; plain numbered list items remain untouched.
    """

    for item in items:
        info = detect_run_in_heading_from_spans(item)
        if info is None:
            continue
        item.update(info)
        item["layout_role"] = "heading"
        item["is_heading_candidate"] = True
        item["heading_level"] = info["run_in_heading_level"]


def detect_run_in_heading_from_spans(item: dict[str, Any]) -> dict[str, Any] | None:
    layer = str(item.get("layout_layer") or "").casefold()
    role = str(item.get("layout_role") or "").casefold()
    raw_type = _layout_raw_type(item)
    if layer in {LAYOUT_LAYER_NOISE, LAYOUT_LAYER_METADATA} or role in {TOC_TITLE_ROLE, TOC_ENTRY_ROLE}:
        return None
    if raw_type not in {"paragraph", "text", "paragraph_text", "body"}:
        return None
    if item.get("has_list_marker") or item.get("list_marker"):
        return None
    spans = item.get("style_spans")
    if not isinstance(spans, list) or len(spans) < 2:
        return None

    normalized_spans: list[tuple[int, str, bool]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            continue
        text = str(span.get("text") or span.get("content") or "").strip()
        if not text:
            continue
        normalized_spans.append((index, text, bool(span.get("is_bold"))))
    if len(normalized_spans) < 2:
        return None

    first_index, first_text, first_bold = normalized_spans[0]
    prefix_match = RUN_IN_HEADING_PREFIX_RE.match(first_text)
    inline_match = RUN_IN_HEADING_INLINE_PREFIX_RE.match(first_text)
    title_parts: list[str] = []
    body_start_position = 1
    number = ""

    if prefix_match:
        number = prefix_match.group("number")
        for position, (_span_index, text, is_bold) in enumerate(normalized_spans[1:], start=1):
            if not is_bold:
                if not title_parts:
                    return None
                body_start_position = position
                break
            title_parts.append(text)
            body_start_position = position + 1
            if text.rstrip().endswith((".", ":")):
                break
    elif inline_match and first_bold:
        number = inline_match.group("number")
        tail = inline_match.group("tail").strip()
        if not tail:
            return None
        title_parts.append(tail)
        body_start_position = 1
    else:
        return None

    title_raw = join_text_fragments(title_parts).strip()
    title_text = re.sub(r"[\s.:]+$", "", title_raw).strip()
    if not title_text or len(title_text) < 3 or len(title_text) > 120:
        return None
    if not any(char.isalpha() for char in title_text):
        return None

    body_parts = [text for _span_index, text, _is_bold in normalized_spans[body_start_position:]]
    body_text = join_text_fragments(body_parts).strip()
    full_text = _layout_text(item)
    if not body_text:
        stripped_prefix = RUN_IN_HEADING_INLINE_PREFIX_RE.sub("", full_text, count=1).strip()
        body_text = stripped_prefix[len(title_raw) :].lstrip(" .:") if stripped_prefix.startswith(title_raw) else ""

    level = min(max(2, number.count(".") + 1), 3)
    return {
        "run_in_heading": True,
        "run_in_heading_number": number,
        "run_in_heading_text": title_text,
        "run_in_heading_body": body_text,
        "run_in_heading_level": level,
        "run_in_heading_prefix_span_index": first_index,
    }


def _looks_like_author_name(text: str) -> bool:
    stripped = " ".join(text.replace(",", " ").split())
    if not stripped or len(stripped) > 80:
        return False
    if any(char.isdigit() for char in stripped):
        return False
    if re.search(r"[.!?;:]", stripped):
        return False
    words = [word for word in re.split(r"\s+", stripped) if word]
    if not 2 <= len(words) <= 6:
        return False
    long_words = [word for word in words if len(word.strip("-'")) >= 2]
    if len(long_words) < 2:
        return False
    capitalized = sum(1 for word in long_words if word[0].isupper())
    return capitalized >= max(2, int(len(long_words) * 0.6))


def detect_list_marker(text: str) -> dict[str, str] | None:
    stripped = text.strip()
    if not stripped:
        return None
    for marker_type, pattern in LIST_MARKER_PATTERNS:
        match = pattern.match(stripped)
        if match:
            return {
                "type": marker_type,
                "marker": match.group(0).strip(),
            }
    return None


def _make_block_view(block: dict[str, Any], index: int, page_idx: int, cfg: SortConfig) -> BlockView:
    bbox = _parse_bbox(block.get("bbox"))
    block_type = str(block.get("type", ""))
    width = max(0.0, bbox[2] - bbox[0])
    is_cross_column = bbox[0] <= cfg.cross_column_left and bbox[2] >= cfg.cross_column_right
    is_full_width = width >= 1000.0 * cfg.full_width_ratio or (block_type in FULL_WIDTH_TYPES and is_cross_column)
    return BlockView(
        block=block,
        original_index=index,
        page_idx=page_idx,
        bbox=bbox,
        text=extract_text(block),
        is_textual=block_type in TEXTUAL_TYPES,
        is_auxiliary=block_type in AUXILIARY_TYPES,
        is_full_width=is_full_width,
    )


def _parse_bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        return (0.0, 0.0, 1000.0, 1000.0)
    x0, y0, x1, y1 = (float(part) for part in value)
    return (x0, y0, x1, y1)


def _sort_page_blocks(blocks: list[BlockView], cfg: SortConfig) -> list[BlockView]:
    column_blocks, full_width_blocks = _assign_columns(blocks, cfg)
    assigned = column_blocks + full_width_blocks
    return rebuild_reading_order(
        assigned,
        full_span_ratio=cfg.full_width_ratio,
        x_tolerance=5.0,
        y_tolerance=cfg.y_tolerance,
        fallback_key=lambda view: (view.y0, view.x0, view.original_index),
    )


def _assign_columns(blocks: list[BlockView], cfg: SortConfig) -> tuple[list[BlockView], list[BlockView]]:
    full_width = [view for view in blocks if view.is_full_width]
    column_candidates = [view for view in blocks if not view.is_full_width]
    split = _infer_column_split(column_candidates, cfg)
    assigned = []
    for view in column_candidates:
        column_id = 0 if split is None or view.cx < split else 1
        assigned.append(
            BlockView(
                block=view.block,
                original_index=view.original_index,
                page_idx=view.page_idx,
                bbox=view.bbox,
                text=view.text,
                is_textual=view.is_textual,
                is_auxiliary=view.is_auxiliary,
                is_full_width=view.is_full_width,
                column_id=column_id,
            )
        )
    return assigned, full_width


def _infer_column_split(blocks: list[BlockView], cfg: SortConfig) -> float | None:
    centers = sorted(view.cx for view in blocks if view.width < 1000.0 * cfg.full_width_ratio)
    if len(centers) < cfg.min_blocks_per_column * 2:
        return None

    best_gap = 0.0
    best_index = -1
    for i, (left, right) in enumerate(zip(centers, centers[1:])):
        left_count = i + 1
        right_count = len(centers) - left_count
        if left_count < cfg.min_blocks_per_column or right_count < cfg.min_blocks_per_column:
            continue
        gap = right - left
        if gap > best_gap:
            best_gap = gap
            best_index = i

    if best_gap < cfg.min_column_gap or best_index < 0:
        return None
    return (centers[best_index] + centers[best_index + 1]) / 2.0


def _enrich_ordered_blocks(ordered: list[BlockView]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    current_run = -1
    previous_textual = False
    previous_column: int | None = None

    for visual_order, view in enumerate(ordered):
        logical_type = _logical_block_type(view.block)
        content = view.block.get("content")
        list_type = content.get("list_type") if isinstance(content, dict) else None
        reference_items = _extract_reference_items(view.block)
        run_id = None
        run_index = None
        if view.is_textual and view.text:
            if not previous_textual or previous_column != view.column_id:
                current_run += 1
                run_index = 0
            else:
                run_index = sum(1 for item in enriched if item.get("text_run_id") == f"p{view.page_idx:04d}_r{current_run:04d}")
            run_id = f"p{view.page_idx:04d}_r{current_run:04d}"
            previous_textual = True
            previous_column = view.column_id
        else:
            previous_textual = False
            previous_column = view.column_id

        enriched.append(
            {
                "page_idx": view.page_idx,
                "visual_order": visual_order,
                "original_index": view.original_index,
                "type": logical_type,
                "raw_type": view.block.get("type"),
                "list_type": list_type,
                "reference_items": reference_items,
                "bbox": list(view.bbox),
                "column_id": view.column_id,
                "is_full_width": view.is_full_width,
                "is_textual": view.is_textual,
                "text_for_embedding": view.text,
                "text_run_id": run_id,
                "text_run_index": run_index,
                "block": view.block,
            }
        )
    return enriched


def _count_columns(ordered: list[BlockView]) -> int:
    columns = {view.column_id for view in ordered if view.column_id is not None}
    return len(columns)


def _logical_block_type(block: dict[str, Any]) -> Any:
    content = block.get("content")
    if (
        block.get("type") == "list"
        and isinstance(content, dict)
        and content.get("list_type") == REFERENCE_LIST_TYPE
    ):
        return "reference"
    return block.get("type")


def _is_micro_mergeable_node(
    node: dict[str, Any],
    bbox: tuple[float, float, float, float],
    *,
    page_width: float,
    small_equation_width_ratio: float,
    small_equation_max_height: float,
) -> bool:
    node_type = _micro_node_type(node)
    if node_type in MICRO_STRUCTURAL_TYPES:
        return False
    if not _micro_node_text(node).strip():
        return False
    if node_type in MICRO_INLINE_MATH_TYPES:
        return True
    if node_type in MICRO_TEXT_TYPES:
        return True
    if node_type in MICRO_EQUATION_TYPES:
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        return height <= small_equation_max_height and width <= max(1.0, page_width) * small_equation_width_ratio
    return False


def _micro_node_type(node: dict[str, Any]) -> str:
    return str(node.get("type") or node.get("raw_type") or node.get("block_type") or "").casefold()


def _micro_node_text(node: dict[str, Any]) -> str:
    return _layout_text(node)


def _micro_single_bbox(node: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = node.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(part) for part in bbox)
    except (TypeError, ValueError):
        return None
    if x1 < x0 or y1 < y0:
        return None
    return (x0, y0, x1, y1)


def _micro_node_page(node: dict[str, Any]) -> int | None:
    pages = node.get("source_page_idxs")
    if isinstance(pages, list) and len(pages) == 1 and isinstance(pages[0], int):
        return pages[0]
    page = node.get("page_idx")
    return page if isinstance(page, int) else None


def _micro_page_width(nodes: list[dict[str, Any]], page: int) -> float:
    boxes = [
        bbox
        for node in nodes
        if _micro_node_page(node) == page
        for bbox in [_micro_single_bbox(node)]
        if bbox is not None
    ]
    if not boxes:
        return 1000.0
    return max(1.0, max(bbox[2] for bbox in boxes) - min(bbox[0] for bbox in boxes))


def _micro_y_center(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def _micro_height(bbox: tuple[float, float, float, float]) -> float:
    return max(1.0, bbox[3] - bbox[1])


def _fits_micro_line(
    bbox: tuple[float, float, float, float],
    line_bboxes: list[tuple[float, float, float, float]],
    y_center_tolerance_ratio: float,
) -> bool:
    if not line_bboxes:
        return True
    center = _micro_y_center(bbox)
    height = _micro_height(bbox)
    for line_bbox in line_bboxes:
        line_center = _micro_y_center(line_bbox)
        line_height = _micro_height(line_bbox)
        tolerance = y_center_tolerance_ratio * ((height + line_height) / 2.0)
        if abs(center - line_center) <= tolerance:
            return True
    return False


def _split_micro_line_by_gap(
    line: list[tuple[int, dict[str, Any], tuple[float, float, float, float]]],
    *,
    max_inline_gap: float,
    page_width: float,
) -> list[list[tuple[int, dict[str, Any], tuple[float, float, float, float]]]]:
    if not line:
        return []
    runs: list[list[tuple[int, dict[str, Any], tuple[float, float, float, float]]]] = [[line[0]]]
    for entry in line[1:]:
        previous_bbox = runs[-1][-1][2]
        bbox = entry[2]
        gap = bbox[0] - previous_bbox[2]
        avg_height = (_micro_height(previous_bbox) + _micro_height(bbox)) / 2.0
        safe_gap = min(max_inline_gap, max(20.0, page_width * 0.06, avg_height * 4.0))
        if gap > safe_gap:
            runs.append([entry])
        else:
            runs[-1].append(entry)
    return runs


def _should_rewrite_single_micro_node(node: dict[str, Any]) -> bool:
    return _micro_node_type(node) in MICRO_INLINE_MATH_TYPES


def _make_micro_fused_node(
    run: list[tuple[int, dict[str, Any], tuple[float, float, float, float]]],
    *,
    page: int,
) -> dict[str, Any]:
    ordered = sorted(run, key=lambda entry: (entry[2][0], entry[2][1], entry[0]))
    bboxes = [entry[2] for entry in ordered]
    source_nodes = [entry[1] for entry in ordered]
    source_indexes = [entry[0] for entry in ordered]
    text = _fuse_micro_text(source_nodes, bboxes)
    global_orders = [_numeric_or_none(node.get("global_order")) for node in source_nodes]
    visual_orders = [_numeric_or_none(node.get("visual_order")) for node in source_nodes]
    original_indexes = [_numeric_or_none(node.get("original_index")) for node in source_nodes]
    style_spans = [
        dict(span)
        for node in source_nodes
        for span in (node.get("style_spans") if isinstance(node.get("style_spans"), list) else [])
        if isinstance(span, dict)
    ]

    fused: dict[str, Any] = {
        "type": "text",
        "raw_type": "micro_fused_line",
        "page_idx": page,
        "bbox": [
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        ],
        "text_for_embedding": text,
        "text": text,
        "is_textual": True,
        "is_full_width": False,
        "micro_fused": True,
        "micro_fusion_count": len(source_nodes),
        "source_node_indexes": source_indexes,
        "source_global_orders": [int(value) for value in global_orders if value is not None],
        "source_visual_orders": [int(value) for value in visual_orders if value is not None],
        "source_original_indexes": [int(value) for value in original_indexes if value is not None],
        "source_page_idxs": [page],
        "micro_source_types": [_micro_node_type(node) for node in source_nodes],
        "block": {
            "type": "paragraph",
            "content": {"paragraph_content": [{"type": "text", "content": text}]},
        },
    }
    if style_spans:
        fused["style_spans"] = style_spans
        fused["style_baseline_size"] = _weighted_micro_font_size(style_spans)
    if any(node.get("column_id") is not None for node in source_nodes):
        fused["column_id"] = source_nodes[0].get("column_id")
    if any(value is not None for value in global_orders):
        fused["global_order"] = min(value for value in global_orders if value is not None)
    if any(value is not None for value in visual_orders):
        fused["visual_order"] = min(value for value in visual_orders if value is not None)
    if any(value is not None for value in original_indexes):
        fused["original_index"] = min(value for value in original_indexes if value is not None)
    return fused


def _fuse_micro_text(nodes: list[dict[str, Any]], bboxes: list[tuple[float, float, float, float]]) -> str:
    text = ""
    previous_bbox: tuple[float, float, float, float] | None = None
    for node, bbox in zip(nodes, bboxes):
        piece = _micro_text_piece(node)
        if not piece:
            continue
        if not text:
            text = piece
        elif _micro_should_join_without_space(text, piece, previous_bbox, bbox):
            text += piece
        else:
            text += " " + piece
        previous_bbox = bbox
    return " ".join(text.split())


def _micro_text_piece(node: dict[str, Any]) -> str:
    text = " ".join(_micro_node_text(node).split())
    if not text:
        return ""
    if _micro_node_type(node) in MICRO_INLINE_MATH_TYPES | MICRO_EQUATION_TYPES:
        return _ensure_inline_math(text)
    return text


def _ensure_inline_math(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if (stripped.startswith("$") and stripped.endswith("$")) or (
        stripped.startswith(r"\(") and stripped.endswith(r"\)")
    ):
        return stripped
    if stripped.startswith(r"\[") and stripped.endswith(r"\]"):
        stripped = stripped[2:-2].strip()
    return f"${stripped}$"


def _micro_should_join_without_space(
    left: str,
    right: str,
    previous_bbox: tuple[float, float, float, float] | None,
    bbox: tuple[float, float, float, float],
) -> bool:
    if not left or not right:
        return False
    if right[0] in ",.;:!?%)]}，。；：！？、》）】":
        return True
    if left[-1] in "([{（《【":
        return True
    if previous_bbox is not None:
        gap = bbox[0] - previous_bbox[2]
        avg_height = (_micro_height(previous_bbox) + _micro_height(bbox)) / 2.0
        if gap <= max(1.0, avg_height * 0.12):
            return True
    return False


def _weighted_micro_font_size(style_spans: list[dict[str, Any]]) -> float:
    weighted: dict[float, int] = {}
    for span in style_spans:
        size = span.get("font_size")
        if not isinstance(size, (int, float)):
            continue
        weight = int(span.get("char_count") or len(str(span.get("text") or "")) or 1)
        weighted[float(size)] = weighted.get(float(size), 0) + max(1, weight)
    if not weighted:
        return 0.0
    return max(weighted.items(), key=lambda item: item[1])[0]


def _numeric_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _micro_output_sort_key(node: dict[str, Any]) -> tuple[int, float, float, int]:
    bbox = _micro_single_bbox(node) or _first_bbox(node.get("bbox")) or (0.0, 0.0, 0.0, 0.0)
    page = _micro_node_page(node)
    if page is None:
        page = 0
    source_indexes = node.get("source_node_indexes")
    int_source_indexes = [int(value) for value in source_indexes if isinstance(value, int)] if isinstance(source_indexes, list) else []
    if int_source_indexes:
        source_index = min(int_source_indexes)
    else:
        source_index = _numeric_or_none(node.get("global_order"))
        if source_index is None:
            source_index = _numeric_or_none(node.get("original_index")) or 0
    return (page, bbox[1], bbox[0], source_index)


def _first_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    vals = value[:4]
    return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
