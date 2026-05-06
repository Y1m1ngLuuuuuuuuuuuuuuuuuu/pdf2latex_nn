"""State-machine reading order utilities for noisy MinerU bboxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


FallbackKey = Callable[[Any], Any]


@dataclass(frozen=True)
class _NodeView:
    node: Any
    original_index: int
    page_idx: float
    bbox: tuple[float, float, float, float] | None
    fallback_key: Any

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


def is_before(
    node_a: Any,
    node_b: Any,
    *,
    y_overlap_threshold: float = 0.30,
    x_tolerance: float = 5.0,
    y_tolerance: float = 5.0,
) -> bool:
    """Return True when `node_a` should be read before `node_b`.

    Rule A: if vertical overlap is small, the block with the higher bottom edge
    comes first. Rule B: if vertical overlap is significant, left block comes
    before right block with a small horizontal-overlap tolerance.
    """

    bbox_a = _node_bbox_from_node(node_a)
    bbox_b = _node_bbox_from_node(node_b)
    if bbox_a is None or bbox_b is None:
        return False

    overlap = y_overlap_score(bbox_a, bbox_b)
    if overlap < y_overlap_threshold:
        return bbox_a[3] <= bbox_b[3] - y_tolerance
    return bbox_a[2] <= bbox_b[0] + x_tolerance


def rebuild_reading_order(
    page_nodes: Iterable[Any],
    *,
    full_span_ratio: float = 0.60,
    x_tolerance: float = 5.0,
    y_tolerance: float = 5.0,
    fallback_key: FallbackKey | None = None,
    write_index: bool = True,
    index_key: str = "index",
) -> list[Any]:
    """Rebuild one page's 1D reading order via a single/double-column state machine.

    The algorithm first scans blocks from top to bottom and segments the page
    into alternating layout regimes:

    - SINGLE: full-span blocks wider than `full_span_ratio * page_width`.
    - DOUBLE: narrower blocks treated as left/right column material.

    SINGLE blocks keep top-to-bottom order. DOUBLE blocks are split around the
    page center; all left-column blocks are emitted top-to-bottom before all
    right-column blocks. By default this overwrites dict-like nodes with a
    page-local monotonically increasing `index`.
    """

    nodes = list(page_nodes)
    views = _build_views(nodes, fallback_key=fallback_key)
    if len(views) <= 1:
        if write_index:
            _write_indices([view.node for view in views], key=index_key)
        return [view.node for view in views]

    ordered = _state_machine_sort_views(
        views,
        full_span_ratio=full_span_ratio,
        y_tolerance=y_tolerance,
    )
    ordered_nodes = [view.node for view in ordered]
    if write_index:
        _write_indices(ordered_nodes, key=index_key)
    return ordered_nodes


def sort_nodes_by_reading_order(
    nodes: Iterable[Any],
    *,
    y_tolerance: float = 5.0,
    x_tolerance: float = 5.0,
    fallback_key: FallbackKey | None = None,
) -> list[Any]:
    """Sort nodes page-by-page using state-machine reading-order reconstruction."""

    views = _build_views(list(nodes), fallback_key=fallback_key)
    if not views:
        return []

    with_bbox = [view for view in views if view.bbox is not None]
    without_bbox = [view for view in views if view.bbox is None]
    ordered_nodes: list[Any] = []

    for page_idx in sorted({view.page_idx for view in with_bbox}):
        page_views = [view for view in with_bbox if view.page_idx == page_idx]
        ordered_nodes.extend(
            rebuild_reading_order(
                [view.node for view in page_views],
                y_tolerance=y_tolerance,
                x_tolerance=x_tolerance,
                fallback_key=fallback_key,
                write_index=False,
            )
        )

    if without_bbox:
        ordered_nodes.extend(
            view.node for view in sorted(without_bbox, key=lambda view: (view.fallback_key, view.original_index))
        )
    return ordered_nodes


def sort_node_indices_by_reading_order(
    nodes: Iterable[Any],
    *,
    y_tolerance: float = 5.0,
    x_tolerance: float = 5.0,
    fallback_key: FallbackKey | None = None,
) -> list[int]:
    """Return original list indices in state-machine reading order."""

    node_list = list(nodes)
    views = _build_views(node_list, fallback_key=fallback_key)
    if not views:
        return []
    ordered_nodes = sort_nodes_by_reading_order(
        node_list,
        y_tolerance=y_tolerance,
        x_tolerance=x_tolerance,
        fallback_key=fallback_key,
    )
    index_by_identity = {id(view.node): view.original_index for view in views}
    return [index_by_identity[id(node)] for node in ordered_nodes]


def reading_order_ranks(
    nodes: Iterable[Any],
    *,
    y_tolerance: float = 5.0,
    x_tolerance: float = 5.0,
    fallback_key: FallbackKey | None = None,
) -> list[int]:
    """Return rank per original node index."""

    order = sort_node_indices_by_reading_order(
        nodes,
        y_tolerance=y_tolerance,
        x_tolerance=x_tolerance,
        fallback_key=fallback_key,
    )
    ranks = [0] * len(order)
    for rank, original_idx in enumerate(order):
        if 0 <= original_idx < len(ranks):
            ranks[original_idx] = rank
    return ranks


def y_overlap_score(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1]))
    height_a = max(1.0, bbox_a[3] - bbox_a[1])
    height_b = max(1.0, bbox_b[3] - bbox_b[1])
    return max(intersection / height_a, intersection / height_b)


def _state_machine_sort_views(
    views: list[_NodeView],
    *,
    full_span_ratio: float,
    y_tolerance: float,
) -> list[_NodeView]:
    page_width, center_x = _page_width_and_center(views)
    if page_width <= 0.0:
        return sorted(views, key=lambda view: _top_to_bottom_key(view, y_tolerance))

    ordered_by_y = sorted(views, key=lambda view: _top_to_bottom_key(view, y_tolerance))
    mode_blocks: list[tuple[str, list[_NodeView]]] = []
    current_mode: str | None = None
    current_block: list[_NodeView] = []

    for view in ordered_by_y:
        node_mode = "SINGLE" if _is_full_span(view, page_width, full_span_ratio) else "DOUBLE"
        if current_mode is not None and node_mode != current_mode and current_block:
            mode_blocks.append((current_mode, current_block))
            current_block = []
        current_mode = node_mode
        current_block.append(view)

    if current_mode is not None and current_block:
        mode_blocks.append((current_mode, current_block))

    ordered: list[_NodeView] = []
    for mode, block in mode_blocks:
        if mode == "SINGLE":
            ordered.extend(sorted(block, key=lambda view: _top_to_bottom_key(view, y_tolerance)))
        else:
            ordered.extend(_sort_double_column_block(block, center_x, y_tolerance))
    return ordered


def _page_width_and_center(views: list[_NodeView]) -> tuple[float, float]:
    min_x = min(view.x0 for view in views)
    max_x = max(view.x1 for view in views)
    page_width = max(0.0, max_x - min_x)
    return page_width, min_x + page_width / 2.0


def _is_full_span(view: _NodeView, page_width: float, full_span_ratio: float) -> bool:
    width = max(0.0, view.x1 - view.x0)
    return width > max(0.0, full_span_ratio) * page_width


def _sort_double_column_block(
    block: list[_NodeView],
    center_x: float,
    y_tolerance: float,
) -> list[_NodeView]:
    left_col: list[_NodeView] = []
    right_col: list[_NodeView] = []
    for view in block:
        if (view.x0 + view.x1) / 2.0 <= center_x:
            left_col.append(view)
        else:
            right_col.append(view)
    left_col.sort(key=lambda view: _top_to_bottom_key(view, y_tolerance))
    right_col.sort(key=lambda view: _top_to_bottom_key(view, y_tolerance))
    return left_col + right_col


def _top_to_bottom_key(view: _NodeView, y_tolerance: float) -> tuple[int, float, float, Any, int]:
    bucket_size = max(1.0, y_tolerance)
    y_bucket = int(view.y0 // bucket_size)
    return (y_bucket, view.x0, view.y0, view.fallback_key, view.original_index)


def _build_views(nodes: list[Any], *, fallback_key: FallbackKey | None) -> list[_NodeView]:
    views: list[_NodeView] = []
    for idx, node in enumerate(nodes):
        views.append(
            _NodeView(
                node=node,
                original_index=idx,
                page_idx=_node_page_from_node(node),
                bbox=_node_bbox_from_node(node),
                fallback_key=fallback_key(node) if fallback_key is not None else idx,
            )
        )
    return views


def _node_record(node: Any) -> dict[str, Any]:
    record = getattr(node, "record", None)
    if isinstance(record, dict):
        return record
    return node if isinstance(node, dict) else {}


def _node_page_from_node(node: Any) -> float:
    record = _node_record(node)
    pages = record.get("source_page_idxs")
    if isinstance(pages, list) and pages:
        value = _numeric(pages[0])
        if value is not None:
            return value
    for key in ("page_idx", "page", "page_id"):
        value = _numeric(record.get(key))
        if value is not None:
            return value
    value = _numeric(getattr(node, "page_idx", None))
    return value if value is not None else 0.0


def _node_bbox_from_node(node: Any) -> tuple[float, float, float, float] | None:
    record = _node_record(node)
    value = record.get("bbox")
    if value is None:
        value = getattr(node, "bbox", None)
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if x1 < x0 or y1 < y0:
        return None
    return (x0, y0, x1, y1)


def _write_indices(nodes: list[Any], *, key: str) -> None:
    for idx, node in enumerate(nodes):
        if isinstance(node, dict):
            node[key] = idx
        elif hasattr(node, key):
            setattr(node, key, idx)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
