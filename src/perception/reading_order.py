"""Visual reading-order reconstruction for MinerU content_list_v2 output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    "list",
    "index",
    "code",
    "algorithm",
}

MERGEABLE_TYPES = {
    "paragraph",
    "list",
}

FLOAT_TYPES = {
    "algorithm",
    "chart",
    "code",
    "equation",
    "equation_interline",
    "figure",
    "image",
    "interline_equation",
    "table",
}

TERMINAL_PUNCTUATION = {
    ".",
    "?",
    "!",
    "。",
    "？",
    "！",
    ";",
    "；",
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


@dataclass(frozen=True)
class SortConfig:
    """Thresholds for normalized MinerU v2 coordinates."""

    full_width_ratio: float = 0.62
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


def build_content_v3(
    visual_order_payload: dict[str, Any],
    *,
    same_page_cross_column_y_tolerance: float = 80.0,
    cross_page_bottom_threshold: float = 780.0,
    cross_page_top_threshold: float = 260.0,
) -> dict[str, Any]:
    """Merge visually ordered text blocks into cross-page logical content."""

    pages = visual_order_payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Expected visual order payload with a pages list")

    flattened: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, list):
            continue
        for item in page:
            if isinstance(item, dict):
                flattened.append(item)

    merged: list[dict[str, Any]] = []
    for item in flattened:
        candidate = _new_v3_item(item, len(merged))
        if merged and _should_merge_v3(
            merged[-1],
            candidate,
            same_page_cross_column_y_tolerance=same_page_cross_column_y_tolerance,
            cross_page_bottom_threshold=cross_page_bottom_threshold,
            cross_page_top_threshold=cross_page_top_threshold,
        ):
            _merge_v3_item(merged[-1], candidate)
        else:
            merged.append(candidate)

    for idx, item in enumerate(merged):
        item["global_order"] = idx

    return {
        "schema_version": "content_v3_paragraph_merge",
        "source_format": "content_list_v2_visual_order",
        "config": {
            "same_page_cross_column_y_tolerance": same_page_cross_column_y_tolerance,
            "cross_page_bottom_threshold": cross_page_bottom_threshold,
            "cross_page_top_threshold": cross_page_top_threshold,
            "mergeable_types": sorted(MERGEABLE_TYPES),
            "terminal_punctuation": sorted(TERMINAL_PUNCTUATION),
        },
        "items": merged,
    }


def build_content_v4(
    content_v3_payload: dict[str, Any],
    *,
    x_alignment_tolerance: float = 28.0,
    parent_indent_threshold: float = 20.0,
    max_skipped_float_blocks: int = 3,
    float_cross_page_top_threshold: float = 760.0,
) -> dict[str, Any]:
    """Add list annotations and merge clear paragraph continuations across floats."""

    raw_items = content_v3_payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("Expected content v3 payload with an items list")

    items = [dict(item) for item in raw_items if isinstance(item, dict)]
    output: list[dict[str, Any]] = []
    active_parent_idx: int | None = None
    list_item_counter = 0
    consumed_indexes: set[int] = set()

    for item_index, raw_item in enumerate(items):
        if item_index in consumed_indexes:
            continue

        item = raw_item
        text = str(item.get("text_for_embedding") or "")
        marker = detect_list_marker(text)
        item["list_marker"] = marker
        item["list_level"] = None
        item["list_item_id"] = None
        item["list_parent_global_order"] = None

        if marker is not None:
            parent_idx = _find_list_parent(output, item, parent_indent_threshold)
            if parent_idx is not None:
                active_parent_idx = parent_idx
            list_item_id = f"li_{list_item_counter:05d}"
            list_item_counter += 1
            item["list_level"] = 1 if active_parent_idx is not None else 0
            item["list_item_id"] = list_item_id
            item["list_parent_global_order"] = (
                output[active_parent_idx].get("global_order", active_parent_idx) if active_parent_idx is not None else None
            )
            output.append(item)
            continue

        continuation = _find_float_separated_paragraph_continuation(
            items,
            item_index,
            max_skipped_float_blocks=max_skipped_float_blocks,
            float_cross_page_top_threshold=float_cross_page_top_threshold,
        )
        if continuation is not None:
            continuation_index, skipped = continuation
            _merge_v3_item(item, _new_v3_item_like(items[continuation_index]))
            item["float_continuation_merge_count"] = int(item.get("float_continuation_merge_count", 0)) + 1
            item["skipped_float_global_orders"] = [
                skipped_item.get("global_order", items.index(skipped_item))
                for skipped_item in skipped
            ]
            item["skipped_float_types"] = [skipped_item.get("type") for skipped_item in skipped]
            consumed_indexes.add(continuation_index)

        output.append(item)
        if item.get("type") not in MERGEABLE_TYPES:
            active_parent_idx = None

    for idx, item in enumerate(output):
        item["global_order"] = idx

    return {
        "schema_version": "content_v4_listaware",
        "source_format": content_v3_payload.get("schema_version", "content_v3"),
        "config": {
            "x_alignment_tolerance": x_alignment_tolerance,
            "parent_indent_threshold": parent_indent_threshold,
            "merge_marked_items": False,
            "max_skipped_float_blocks": max_skipped_float_blocks,
            "float_cross_page_top_threshold": float_cross_page_top_threshold,
            "float_types": sorted(FLOAT_TYPES),
        },
        "items": output,
    }


def extract_text(block: dict[str, Any]) -> str:
    """Extract readable text from MinerU v2 nested content structures."""

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
                "list_items",
                "table_caption",
                "chart_caption",
                "image_caption",
                "page_footnote_content",
            ):
                if key in value:
                    visit(value[key])

    visit(block.get("content"))
    return " ".join(parts)


def has_terminal_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    return stripped[-1] in TERMINAL_PUNCTUATION


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
    full_width_blocks = sorted(full_width_blocks, key=lambda view: (view.y0, view.x0, view.original_index))

    ordered: list[BlockView] = []
    emitted: set[int] = set()
    for full in full_width_blocks:
        before = [
            view
            for view in column_blocks
            if view.original_index not in emitted and view.cy < full.y0 - cfg.y_tolerance
        ]
        ordered.extend(_sort_column_region(before))
        emitted.update(view.original_index for view in before)
        ordered.append(full)

    remaining = [view for view in column_blocks if view.original_index not in emitted]
    ordered.extend(_sort_column_region(remaining))
    return ordered


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


def _sort_column_region(blocks: list[BlockView]) -> list[BlockView]:
    """Sort one region between full-width blocks: left column, then right."""

    return sorted(
        blocks,
        key=lambda view: (view.column_id or 0, view.y0, view.x0, view.original_index),
    )


def _enrich_ordered_blocks(ordered: list[BlockView]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    current_run = -1
    previous_textual = False
    previous_column: int | None = None

    for visual_order, view in enumerate(ordered):
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
                "type": view.block.get("type"),
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


def _new_v3_item(item: dict[str, Any], global_order: int) -> dict[str, Any]:
    bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else []
    text = item.get("text_for_embedding") or ""
    return {
        "global_order": global_order,
        "type": item.get("type"),
        "page_idx": item.get("page_idx"),
        "visual_order": item.get("visual_order"),
        "original_index": item.get("original_index"),
        "bbox": list(bbox),
        "column_id": item.get("column_id"),
        "is_full_width": item.get("is_full_width"),
        "is_textual": item.get("is_textual"),
        "text_for_embedding": text,
        "merge_count": 1,
        "source_page_idxs": [item.get("page_idx")],
        "source_visual_orders": [item.get("visual_order")],
        "source_original_indexes": [item.get("original_index")],
        "block": item.get("block"),
    }


def _should_merge_v3(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    same_page_cross_column_y_tolerance: float,
    cross_page_bottom_threshold: float,
    cross_page_top_threshold: float,
) -> bool:
    if previous.get("type") != current.get("type"):
        return False
    if previous.get("type") not in MERGEABLE_TYPES:
        return False
    if not previous.get("text_for_embedding") or not current.get("text_for_embedding"):
        return False
    if has_terminal_punctuation(str(previous.get("text_for_embedding", ""))):
        return False

    prev_page = _last_value(previous.get("source_page_idxs"))
    cur_page = current.get("page_idx")
    if not isinstance(prev_page, int) or not isinstance(cur_page, int):
        return False

    prev_bbox = _last_bbox(previous.get("bbox"))
    cur_bbox = _first_bbox(current.get("bbox"))
    if prev_bbox is None or cur_bbox is None:
        return False

    if prev_page == cur_page:
        return _is_same_page_cross_column_continuation(previous, current, prev_bbox, cur_bbox, same_page_cross_column_y_tolerance)
    if cur_page == prev_page + 1:
        return prev_bbox[3] >= cross_page_bottom_threshold and cur_bbox[1] <= cross_page_top_threshold
    return False


def _merge_v3_item(previous: dict[str, Any], current: dict[str, Any]) -> None:
    previous["bbox"].extend(current.get("bbox", []))
    previous["text_for_embedding"] = _join_continuation_text(
        str(previous.get("text_for_embedding", "")),
        str(current.get("text_for_embedding", "")),
    )
    previous["merge_count"] = int(previous.get("merge_count", 1)) + int(current.get("merge_count", 1))
    for key in ("source_page_idxs", "source_visual_orders", "source_original_indexes"):
        previous.setdefault(key, [])
        previous[key].extend(current.get(key, []))


def _join_continuation_text(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-"):
        return left[:-1] + right
    return left + " " + right


def _is_same_page_cross_column_continuation(
    previous: dict[str, Any],
    current: dict[str, Any],
    prev_bbox: tuple[float, float, float, float],
    cur_bbox: tuple[float, float, float, float],
    y_tolerance: float,
) -> bool:
    prev_col = previous.get("column_id")
    cur_col = current.get("column_id")
    if prev_col == 0 and cur_col == 1:
        return cur_bbox[1] <= prev_bbox[1] + y_tolerance
    return False


def _last_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) < 4 or len(value) % 4 != 0:
        return None
    vals = value[-4:]
    return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def _first_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    vals = value[:4]
    return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def _last_value(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[-1]
    return None


def _new_v3_item_like(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox": list(item.get("bbox") or []),
        "text_for_embedding": item.get("text_for_embedding") or "",
        "merge_count": item.get("merge_count", 1),
        "source_page_idxs": list(item.get("source_page_idxs") or [item.get("page_idx")]),
        "source_visual_orders": list(item.get("source_visual_orders") or [item.get("visual_order")]),
        "source_original_indexes": list(item.get("source_original_indexes") or [item.get("original_index")]),
    }


def _find_float_separated_paragraph_continuation(
    items: list[dict[str, Any]],
    item_index: int,
    *,
    max_skipped_float_blocks: int,
    float_cross_page_top_threshold: float,
) -> tuple[int, list[dict[str, Any]]] | None:
    current = items[item_index]
    if current.get("type") != "paragraph":
        return None
    if detect_list_marker(str(current.get("text_for_embedding") or "")) is not None:
        return None
    if not str(current.get("text_for_embedding") or "").rstrip().endswith("-"):
        return None

    skipped: list[dict[str, Any]] = []
    cursor = item_index + 1
    while cursor < len(items) and _is_float_block(items[cursor]) and len(skipped) < max_skipped_float_blocks:
        skipped.append(items[cursor])
        cursor += 1

    if not skipped or cursor >= len(items):
        return None

    candidate = items[cursor]
    if candidate.get("type") != "paragraph":
        return None
    if detect_list_marker(str(candidate.get("text_for_embedding") or "")) is not None:
        return None
    if not candidate.get("text_for_embedding"):
        return None

    prev_page = _last_value(current.get("source_page_idxs"))
    cur_page = candidate.get("page_idx")
    prev_bbox = _last_bbox(current.get("bbox"))
    cur_bbox = _first_bbox(candidate.get("bbox"))
    if not isinstance(prev_page, int) or not isinstance(cur_page, int) or prev_bbox is None or cur_bbox is None:
        return None

    if cur_page == prev_page + 1 and cur_bbox[1] <= float_cross_page_top_threshold:
        return cursor, skipped
    if cur_page == prev_page:
        return cursor, skipped
    return None


def _is_float_block(item: dict[str, Any]) -> bool:
    return str(item.get("type") or "").lower() in FLOAT_TYPES


def _find_list_parent(output: list[dict[str, Any]], item: dict[str, Any], indent_threshold: float) -> int | None:
    item_bbox = _first_bbox(item.get("bbox"))
    if item_bbox is None:
        return None
    for idx in range(len(output) - 1, -1, -1):
        candidate = output[idx]
        text = str(candidate.get("text_for_embedding") or "").rstrip()
        if not text.endswith((':', '：')):
            continue
        candidate_bbox = _first_bbox(candidate.get("bbox"))
        if candidate_bbox is None:
            continue
        if item_bbox[0] > candidate_bbox[0] + indent_threshold:
            return idx
        return None
    return None
