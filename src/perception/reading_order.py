"""Visual reading-order reconstruction for MinerU content_list_v2 output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
LAYOUT_LAYER_METADATA = "metadata_layer"
LAYOUT_LAYER_NOISE = "noise_layer"
LAYOUT_LAYER_OTHER = "other_layer"
MATH_LAYOUT_TYPES = MICRO_INLINE_MATH_TYPES | MICRO_EQUATION_TYPES
FLOAT_LAYOUT_TYPES = {"figure", "image", "chart", "table", "algorithm"}
METADATA_LAYOUT_TYPES = {"title", "author", "affiliation", "abstract"}
NOISE_LAYOUT_TYPES = AUXILIARY_TYPES | {"header", "footer"}
FRONT_MATTER_AFFILIATION_RE = re.compile(
    r"\b("
    r"affiliation|department|university|institute|school|college|faculty|"
    r"laborator(?:y|ies)|\blab\b|centre|center|research|academy|"
    r"cnrs|inria|google|microsoft|norway|china|usa|uk|germany|france|italy"
    r")\b",
    re.IGNORECASE,
)
FRONT_MATTER_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")


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
    refine_front_matter_layers(ordered)

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
    refine_front_matter_layers(ordered)

    refreshed = dict(payload)
    refreshed["schema_version"] = str(payload.get("schema_version") or "content_v7_columnfix_listmarkers")
    refreshed["items"] = ordered
    refreshed["layout_metadata_refreshed"] = True
    refreshed["layout_metadata_version"] = "layer_band_v1"
    config = dict(refreshed.get("config") or {})
    config["page_object_layers"] = True
    config["local_band_metadata"] = True
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
    for view in y_ordered:
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
        left_list = [view for view in block if view.column_label == "LEFT_COL"]
        right_list = [view for view in block if view.column_label == "RIGHT_COL"]
        left_list.sort(key=_columnar_top_key)
        right_list.sort(key=_columnar_top_key)
        combined = left_list + right_list
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
        node["layout_band_id"] = band_id
        node["layout_band_type"] = band_type
        node["layout_band_local_order"] = band_local_order
        node["layout_band_column_id"] = band_column_id
        node["layout_band_column"] = {0: "left", 1: "right", 2: "full"}.get(band_column_id, "unknown")
        node["layout_is_band_boundary"] = is_layout_band_boundary_node(node, span_label=view.span_label, layer=layer)
        node["is_main_flow_candidate"] = layer == LAYOUT_LAYER_MAIN_TEXT
        ordered_nodes.append(node)
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


def infer_layout_layer(node: dict[str, Any]) -> str:
    """Assign a coarse page-object layer before graph construction.

    This keeps main text flow, floats, display math, metadata, and visual noise
    from competing inside one undifferentiated reading-order sequence.
    """

    node_type = _layout_raw_type(node)
    text = _layout_text(node)
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
    if layer == LAYOUT_LAYER_NOISE:
        return "noise"
    if layer == LAYOUT_LAYER_FLOAT:
        return node_type or "float"
    if layer == LAYOUT_LAYER_MATH:
        return "inline_math" if node_type in MICRO_INLINE_MATH_TYPES else "display_math"
    if str(node.get("list_type") or "").lower() == REFERENCE_LIST_TYPE:
        return "reference_list"
    if detect_list_marker(_layout_text(node)):
        return "list_item"
    if node_type in {"title", "section", "subsection", "subsubsection", "heading"}:
        return "heading"
    return "body_text" if layer == LAYOUT_LAYER_MAIN_TEXT else layer


def is_layout_band_boundary_node(node: dict[str, Any], *, span_label: str, layer: str | None = None) -> bool:
    layer = layer or infer_layout_layer(node)
    if layer in {LAYOUT_LAYER_FLOAT, LAYOUT_LAYER_MATH, LAYOUT_LAYER_METADATA}:
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
        or FRONT_MATTER_EMAIL_RE.search(text) is not None
        or FRONT_MATTER_AFFILIATION_RE.search(text) is not None
    )


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
    for index, item in enumerate(items[:first_body_heading_order]):
        raw = _layout_raw_type(item)
        text = _layout_text(item)
        if raw in NOISE_LAYOUT_TYPES or not text:
            continue
        if raw in {"title", "paragraph", "text", "author", "affiliation"} or _front_matter_text_marker(text):
            item["layout_layer"] = LAYOUT_LAYER_METADATA
            item["layout_role"] = "abstract" if _front_matter_text_marker(text) else "front_matter"
            item["is_main_flow_candidate"] = False
    for item in items:
        if _looks_like_top_page_author_block(item, first_body_heading_y0=first_body_heading_y0):
            item["layout_layer"] = LAYOUT_LAYER_METADATA
            item["layout_role"] = _front_matter_role(item)
            item["is_main_flow_candidate"] = False


def _is_body_heading_candidate(item: dict[str, Any]) -> bool:
    raw = _layout_raw_type(item)
    if raw not in {"title", "section", "subsection", "subsubsection", "heading"}:
        return False
    normalized = " ".join(_layout_text(item).lower().split())
    if not normalized:
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
    return normalized.startswith(("abstract", "keywords", "author", "authors", "affiliation"))


def _looks_like_top_page_author_block(item: dict[str, Any], *, first_body_heading_y0: float | None) -> bool:
    """Catch two-column author/affiliation blocks that sort after the first heading."""

    if item.get("page_idx") not in {0, None}:
        return False
    text = _layout_text(item).strip()
    if not text:
        return False
    raw = _layout_raw_type(item)
    if raw in NOISE_LAYOUT_TYPES or raw in FLOAT_LAYOUT_TYPES or raw in MATH_LAYOUT_TYPES:
        return False
    bbox = _parse_bbox(item.get("bbox"))
    if bbox is None:
        return False
    y0 = bbox[1]
    if first_body_heading_y0 is not None and y0 >= first_body_heading_y0:
        return False
    normalized = " ".join(text.lower().split())
    if _front_matter_text_marker(text):
        return True
    if FRONT_MATTER_EMAIL_RE.search(text) or FRONT_MATTER_AFFILIATION_RE.search(text):
        return True
    # Author names in two-column IEEE-like papers are short top-page lines.
    if y0 <= 280 and _looks_like_author_name(text) and not normalized.startswith(("fig", "table", "algorithm")):
        return True
    return False


def _front_matter_role(item: dict[str, Any]) -> str:
    text = _layout_text(item)
    if _front_matter_text_marker(text):
        return "abstract" if text.strip().lower().startswith("abstract") else "front_matter"
    if FRONT_MATTER_EMAIL_RE.search(text) or FRONT_MATTER_AFFILIATION_RE.search(text):
        return "affiliation"
    if _looks_like_author_name(text):
        return "author"
    return "front_matter"


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
