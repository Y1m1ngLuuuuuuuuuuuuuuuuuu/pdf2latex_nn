#!/usr/bin/env python3
"""Visualize supervised graph labels directly on the source PDF pages.

This script is intentionally independent from the inference/decoder pipeline:
it draws the labeled PyG graph produced by ``label_generator.py`` so the
alignment labels can be inspected before GNN training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


MERGE_LABEL = 0
PARENT_CHILD_LABEL = 1
NONE_LABEL = 2

NOISE_TYPES = {
    "page_header",
    "header",
    "page_footer",
    "footer",
    "page_number",
}
MERGE_COLOR = (0.00, 0.62, 0.18)
PARENT_COLOR = (0.88, 0.05, 0.05)
BBOX_COLOR = (0.25, 0.55, 0.95)
INDEX_COLOR = (0.95, 0.00, 0.00)
CROSS_PAGE_COLOR = (0.35, 0.20, 0.75)


@dataclass(frozen=True)
class BBoxRef:
    node_index: int
    chunk_index: int
    page_idx: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class PointRef:
    page_idx: int
    x: float
    y: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, help="Original PDF file")
    parser.add_argument("--content-json", type=Path, required=True, help="MinerU content_v7_styles.json, or compatible items JSON")
    parser.add_argument("--graph", type=Path, required=True, help="Labeled PyG graph .pt produced by label_generator.py")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_output") / "graph_visuals",
        help="Directory for output page PNGs",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix; defaults to PDF stem")
    parser.add_argument("--zoom", type=float, default=2.0, help="Render zoom for high-resolution PNG output")
    parser.add_argument("--max-pages", type=int, default=0, help="0 means all pages")
    parser.add_argument("--draw-noise", action="store_true", help="Also draw page header/footer/page-number nodes")
    parser.add_argument("--draw-cross-page", action="store_true", help="Mark cross-page labeled edges with endpoint tags")
    parser.add_argument("--bbox-opacity", type=float, default=0.28)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.pdf.stem

    graph = load_graph(args.graph)
    records = load_records_for_graph(args.content_json, graph)
    edge_index = graph.edge_index.detach().cpu()
    labels = load_edge_labels(graph)
    if int(edge_index.shape[1]) != len(labels):
        raise ValueError(f"edge count ({int(edge_index.shape[1])}) does not match label count ({len(labels)})")

    written = draw_graph_labels(
        pdf_path=args.pdf,
        records=records,
        edge_index=edge_index,
        labels=labels,
        output_dir=args.output_dir,
        prefix=prefix,
        zoom=args.zoom,
        max_pages=args.max_pages,
        draw_noise=args.draw_noise,
        draw_cross_page=args.draw_cross_page,
        bbox_opacity=args.bbox_opacity,
    )
    print(f"wrote_dir={args.output_dir}")
    print(f"written_pages={len(written)}")
    for path in written:
        print(path)
    return 0


def load_graph(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def load_edge_labels(graph: Any) -> list[int]:
    labels = getattr(graph, "y", None)
    if labels is None:
        labels = getattr(graph, "edge_label", None)
    if labels is None:
        raise ValueError("Graph does not contain y or edge_label labels")
    return [int(value) for value in labels.detach().cpu().long().tolist()]


def load_records_for_graph(content_json: Path, graph: Any) -> list[dict[str, Any]]:
    graph_records = getattr(graph, "node_records", None)
    if isinstance(graph_records, list) and len(graph_records) == int(graph.num_nodes):
        return [dict(record) if isinstance(record, dict) else {} for record in graph_records]

    content_records = load_content_records(content_json)
    if len(content_records) != int(graph.num_nodes):
        raise ValueError(
            f"content record count ({len(content_records)}) does not match graph.num_nodes ({int(graph.num_nodes)}), "
            "and graph.node_records is unavailable"
        )
    return content_records


def load_content_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", payload if isinstance(payload, list) else [])
    if not isinstance(items, list):
        raise ValueError(f"Expected {path} to contain an items list")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        record = dict(item)
        record.setdefault("global_order", index)
        record.setdefault("original_index", index)
        records.append(record)
    return records


def draw_graph_labels(
    *,
    pdf_path: Path,
    records: list[dict[str, Any]],
    edge_index: Any,
    labels: list[int],
    output_dir: Path,
    prefix: str,
    zoom: float,
    max_pages: int,
    draw_noise: bool,
    draw_cross_page: bool,
    bbox_opacity: float,
) -> list[Path]:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        page_count = len(doc) if max_pages <= 0 else min(len(doc), max_pages)
        node_boxes = build_node_bbox_index(records)
        draw_bboxes(doc, records, node_boxes, page_count=page_count, draw_noise=draw_noise, bbox_opacity=bbox_opacity)
        draw_labeled_edges(
            doc,
            records=records,
            node_boxes=node_boxes,
            edge_index=edge_index,
            labels=labels,
            page_count=page_count,
            draw_cross_page=draw_cross_page,
        )
        return render_pages(doc, output_dir=output_dir, prefix=prefix, zoom=zoom, page_count=page_count)
    finally:
        doc.close()


def build_node_bbox_index(records: list[dict[str, Any]]) -> dict[int, list[BBoxRef]]:
    output: dict[int, list[BBoxRef]] = {}
    for node_index, record in enumerate(records):
        chunks = bbox_chunks(record.get("bbox"))
        if not chunks:
            continue
        page_idxs = page_indices_for_record(record, len(chunks))
        refs = []
        for chunk_index, (bbox, page_idx) in enumerate(zip(chunks, page_idxs)):
            if page_idx is None:
                continue
            refs.append(BBoxRef(node_index=node_index, chunk_index=chunk_index, page_idx=page_idx, bbox=bbox))
        if refs:
            output[node_index] = refs
    return output


def page_indices_for_record(record: dict[str, Any], chunk_count: int) -> list[int | None]:
    source_pages = record.get("source_page_idxs")
    if isinstance(source_pages, list) and len(source_pages) == chunk_count:
        return [parse_int(value) for value in source_pages]
    page_idx = parse_int(record.get("page_idx"))
    return [page_idx for _ in range(chunk_count)]


def draw_bboxes(
    doc: Any,
    records: list[dict[str, Any]],
    node_boxes: dict[int, list[BBoxRef]],
    *,
    page_count: int,
    draw_noise: bool,
    bbox_opacity: float,
) -> None:
    for node_index, refs in node_boxes.items():
        record = records[node_index] if node_index < len(records) else {}
        if not draw_noise and is_noise_record(record):
            continue
        for ref in refs:
            if ref.page_idx < 0 or ref.page_idx >= page_count:
                continue
            page = doc[ref.page_idx]
            rect = bbox_to_page_rect(page, ref.bbox)
            draw_rect(page, rect, color=BBOX_COLOR, width=0.55, opacity=bbox_opacity)
            draw_index_label(page, rect, node_index, chunk_index=ref.chunk_index if len(refs) > 1 else None)


def draw_labeled_edges(
    doc: Any,
    *,
    records: list[dict[str, Any]],
    node_boxes: dict[int, list[BBoxRef]],
    edge_index: Any,
    labels: list[int],
    page_count: int,
    draw_cross_page: bool,
) -> None:
    for edge_pos, label in enumerate(labels):
        if label not in (MERGE_LABEL, PARENT_CHILD_LABEL):
            continue
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        if is_noise_index(records, source) or is_noise_index(records, target):
            continue
        source_point = node_point(doc, node_boxes, source, prefer_last=True, page_count=page_count)
        target_point = node_point(doc, node_boxes, target, prefer_last=False, page_count=page_count)
        if source_point is None or target_point is None:
            continue
        if source_point.page_idx != target_point.page_idx:
            if draw_cross_page:
                draw_cross_page_edge_tags(doc, source_point, target_point, source=source, target=target, label=label, page_count=page_count)
            continue
        page = doc[source_point.page_idx]
        start = (source_point.x, source_point.y)
        end = (target_point.x, target_point.y)
        if label == MERGE_LABEL:
            draw_merge_line(page, start, end)
        elif label == PARENT_CHILD_LABEL:
            draw_parent_arrow(page, start, end)


def node_point(
    doc: Any,
    node_boxes: dict[int, list[BBoxRef]],
    node_index: int,
    *,
    prefer_last: bool,
    page_count: int,
) -> PointRef | None:
    refs = node_boxes.get(node_index)
    if not refs:
        return None
    ref = refs[-1] if prefer_last else refs[0]
    if ref.page_idx < 0 or ref.page_idx >= page_count:
        return None
    rect = bbox_to_page_rect(doc[ref.page_idx], ref.bbox)
    return PointRef(page_idx=ref.page_idx, x=(rect.x0 + rect.x1) / 2.0, y=(rect.y0 + rect.y1) / 2.0)


def draw_merge_line(page: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    page.draw_line(start, end, color=MERGE_COLOR, width=2.4, overlay=True)
    draw_edge_tag(page, midpoint(start, end), "M", MERGE_COLOR)


def draw_parent_arrow(page: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    page.draw_line(start, end, color=PARENT_COLOR, width=1.7, overlay=True)
    draw_arrow_head(page, start, end, color=PARENT_COLOR)
    draw_edge_tag(page, midpoint(start, end), "P", PARENT_COLOR)


def draw_cross_page_edge_tags(
    doc: Any,
    source_point: PointRef,
    target_point: PointRef,
    *,
    source: int,
    target: int,
    label: int,
    page_count: int,
) -> None:
    tag = "M" if label == MERGE_LABEL else "P"
    if 0 <= source_point.page_idx < page_count:
        draw_edge_tag(doc[source_point.page_idx], (source_point.x, source_point.y), f"{tag}->{target}", CROSS_PAGE_COLOR)
    if 0 <= target_point.page_idx < page_count:
        draw_edge_tag(doc[target_point.page_idx], (target_point.x, target_point.y), f"{source}->{tag}", CROSS_PAGE_COLOR)


def draw_arrow_head(page: Any, start: tuple[float, float], end: tuple[float, float], *, color: tuple[float, float, float]) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1.0:
        return
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    arrow_len = 8.0
    arrow_width = 4.5
    left = (end[0] - ux * arrow_len + px * arrow_width, end[1] - uy * arrow_len + py * arrow_width)
    right = (end[0] - ux * arrow_len - px * arrow_width, end[1] - uy * arrow_len - py * arrow_width)
    page.draw_line(end, left, color=color, width=1.4, overlay=True)
    page.draw_line(end, right, color=color, width=1.4, overlay=True)


def draw_edge_tag(page: Any, point: tuple[float, float], text: str, color: tuple[float, float, float]) -> None:
    import fitz

    width = max(9.0, 4.0 * len(text) + 4.0)
    height = 8.0
    rect = fitz.Rect(point[0] - width / 2.0, point[1] - height / 2.0, point[0] + width / 2.0, point[1] + height / 2.0)
    rect = keep_rect_on_page(rect, page)
    try:
        page.draw_rect(rect, color=color, fill=(1, 1, 1), width=0.3, fill_opacity=0.82, overlay=True)
    except TypeError:
        page.draw_rect(rect, color=color, fill=(1, 1, 1), width=0.3, overlay=True)
    page.insert_text((rect.x0 + 1.3, rect.y1 - 1.7), text, fontsize=5.5, color=color, overlay=True)


def draw_index_label(page: Any, rect: Any, node_index: int, *, chunk_index: int | None) -> None:
    label = str(node_index) if chunk_index is None else f"{node_index}.{chunk_index}"
    draw_small_label(page, rect, label, color=INDEX_COLOR)


def draw_small_label(page: Any, rect: Any, label: str, *, color: tuple[float, float, float]) -> None:
    import fitz

    width = max(10.0, 3.8 * len(label) + 4.0)
    height = 7.0
    label_rect = fitz.Rect(rect.x0, max(page.rect.y0, rect.y0 - height), rect.x0 + width, max(page.rect.y0, rect.y0 - height) + height)
    label_rect = keep_rect_on_page(label_rect, page)
    try:
        page.draw_rect(label_rect, color=color, fill=(1, 1, 1), width=0.25, fill_opacity=0.72, overlay=True)
    except TypeError:
        page.draw_rect(label_rect, color=color, fill=(1, 1, 1), width=0.25, overlay=True)
    page.insert_text((label_rect.x0 + 1.1, label_rect.y1 - 1.4), label, fontsize=5.0, color=color, overlay=True)


def draw_rect(page: Any, rect: Any, *, color: tuple[float, float, float], width: float, opacity: float) -> None:
    try:
        page.draw_rect(rect, color=color, width=width, stroke_opacity=opacity, overlay=True)
    except TypeError:
        page.draw_rect(rect, color=color, width=width, overlay=True)


def render_pages(doc: Any, *, output_dir: Path, prefix: str, zoom: float, page_count: int) -> list[Path]:
    import fitz

    matrix = fitz.Matrix(zoom, zoom)
    written: list[Path] = []
    for page_idx in range(page_count):
        pix = doc[page_idx].get_pixmap(matrix=matrix, alpha=False)
        output_path = output_dir / f"{prefix}_page_{page_idx + 1:03d}.png"
        pix.save(output_path)
        written.append(output_path)
    return written


def bbox_to_page_rect(page: Any, bbox: tuple[float, float, float, float]) -> Any:
    import fitz

    x0, y0, x1, y1 = bbox
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    max_coord = max(abs(x0), abs(y0), abs(x1), abs(y1))
    if max_coord <= 1.5:
        rect = fitz.Rect(x0 * page_width, y0 * page_height, x1 * page_width, y1 * page_height)
    elif max_coord <= 1005.0:
        rect = fitz.Rect(x0 / 1000.0 * page_width, y0 / 1000.0 * page_height, x1 / 1000.0 * page_width, y1 / 1000.0 * page_height)
    elif x1 <= page_width + 5.0 and y1 <= page_height + 5.0:
        rect = fitz.Rect(x0, y0, x1, y1)
    else:
        scale_x = page_width / max(1.0, max(x1, 1000.0))
        scale_y = page_height / max(1.0, max(y1, 1000.0))
        rect = fitz.Rect(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
    return normalize_rect(rect) & page.rect


def normalize_rect(rect: Any) -> Any:
    import fitz

    return fitz.Rect(min(rect.x0, rect.x1), min(rect.y0, rect.y1), max(rect.x0, rect.x1), max(rect.y0, rect.y1))


def keep_rect_on_page(rect: Any, page: Any) -> Any:
    import fitz

    width = rect.width
    height = rect.height
    x0 = min(max(float(page.rect.x0), float(rect.x0)), float(page.rect.x1) - width)
    y0 = min(max(float(page.rect.y0), float(rect.y0)), float(page.rect.y1) - height)
    return fitz.Rect(x0, y0, x0 + width, y0 + height)


def bbox_chunks(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    usable_len = len(value) - (len(value) % 4)
    chunks: list[tuple[float, float, float, float]] = []
    for index in range(0, usable_len, 4):
        try:
            x0, y0, x1, y1 = (float(item) for item in value[index : index + 4])
        except (TypeError, ValueError):
            continue
        chunks.append((x0, y0, x1, y1))
    return chunks


def is_noise_index(records: list[dict[str, Any]], node_index: int) -> bool:
    if node_index < 0 or node_index >= len(records):
        return False
    return is_noise_record(records[node_index])


def is_noise_record(record: dict[str, Any]) -> bool:
    raw_type = str(record.get("type") or record.get("raw_type") or record.get("block_type") or "").lower()
    return raw_type in NOISE_TYPES


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def midpoint(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    return ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)


if __name__ == "__main__":
    raise SystemExit(main())
