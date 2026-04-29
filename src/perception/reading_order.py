"""Visual reading-order reconstruction for MinerU content_list_v2 output."""

from __future__ import annotations

import json
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
