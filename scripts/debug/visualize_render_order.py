#!/usr/bin/env python3
"""Overlay decoded/rendered node order on the original PDF for visual QA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


TYPE_COLORS: dict[str, tuple[float, float, float]] = {
    "title": (0.10, 0.25, 0.95),
    "text": (0.90, 0.10, 0.45),
    "paragraph": (0.90, 0.10, 0.45),
    "equation": (0.85, 0.10, 0.10),
    "inline_math": (0.65, 0.05, 0.85),
    "list": (0.00, 0.55, 0.55),
    "table": (0.00, 0.55, 0.15),
    "figure": (0.95, 0.50, 0.05),
    "algorithm": (0.45, 0.20, 0.85),
    "code": (0.20, 0.20, 0.20),
    "reference": (0.55, 0.32, 0.12),
}
DEFAULT_COLOR = (0.15, 0.15, 0.15)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, help="Original source PDF")
    parser.add_argument("--content-json", type=Path, required=True, help="content_v*_styles.json used by graph generation")
    parser.add_argument("--graph", type=Path, help="PyG graph .pt; needed to match micro-fused records")
    parser.add_argument("--logits", type=Path, help="Saved edge logits/probabilities from step5_generate_tex.py")
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint; used only when --logits is absent")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for annotated PDF, PNG pages, and JSON")
    parser.add_argument("--prefix", default=None, help="Output file prefix; defaults to PDF stem")
    parser.add_argument("--mode", choices=("render", "content"), default="render")
    parser.add_argument("--title", default=None, help="Optional document title override")
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--parent-threshold", type=float, default=0.0)
    parser.add_argument("--sibling-threshold", type=float, default=0.5)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means render PNG previews for all pages")
    parser.add_argument("--zoom", type=float, default=2.0, help="PNG preview render zoom")
    parser.add_argument("--no-lines", action="store_true", help="Do not draw same-page center-to-center order lines")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.pdf.stem

    entries = build_entries(args)
    annotated_pdf = args.output_dir / f"{prefix}_{args.mode}_order_annotated.pdf"
    order_json = args.output_dir / f"{prefix}_{args.mode}_order.json"
    png_dir = args.output_dir / f"{prefix}_{args.mode}_order_pages"

    write_order_json(entries, order_json)
    draw_order_overlay(
        pdf_path=args.pdf,
        entries=entries,
        output_pdf=annotated_pdf,
        png_dir=png_dir,
        zoom=args.zoom,
        max_pages=args.max_pages,
        draw_lines=not args.no_lines,
    )
    print(f"wrote {annotated_pdf}")
    print(f"wrote {order_json}")
    print(f"wrote {png_dir}")
    print(f"entries={len(entries)}")
    return 0


def build_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    step5 = load_step5_module()
    data = torch.load(args.graph, map_location="cpu", weights_only=False) if args.graph else None
    node_records = step5.load_node_records(args.content_json, data) if data is not None else load_content_records(args.content_json)

    if args.mode == "content":
        return entries_from_content_records(node_records)

    if data is None:
        raise ValueError("--graph is required in render mode")
    logits = load_or_infer_logits(args, data)

    from src.reasoning.postprocess import TreeDecoder, TreeDecoderConfig

    decoder = TreeDecoder(
        TreeDecoderConfig(
            merge_threshold=args.merge_threshold,
            parent_threshold=args.parent_threshold,
            sibling_threshold=args.sibling_threshold,
        )
    )
    root = decoder.decode(node_records, data.edge_index.detach().cpu(), logits)
    title = args.title or step5.infer_document_title(node_records)
    return entries_from_tree(root, document_title=title)


def load_step5_module() -> Any:
    module_path = PROJECT_ROOT / "scripts" / "pipeline" / "step5_generate_tex.py"
    spec = importlib.util.spec_from_file_location("step5_generate_tex_debug_import", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_or_infer_logits(args: argparse.Namespace, data: Any) -> Any:
    import torch

    if args.logits:
        return torch.load(args.logits, map_location="cpu", weights_only=False)
    if not args.checkpoint:
        raise ValueError("render mode requires either --logits or --checkpoint")

    from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT

    step5 = load_step5_module()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    config = step5.checkpoint_compatible_config(config if isinstance(config, EdgeGATConfig) else EdgeGATConfig(), state_dict)
    model = EdgeRelationGAT(config)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        return model(data).detach().cpu()


def load_content_records(content_json: Path) -> list[dict[str, Any]]:
    step5 = load_step5_module()
    payload = json.loads(content_json.read_text(encoding="utf-8"))
    if is_native_v2_pages(payload):
        return flatten_native_v2_pages(payload)
    items = payload.get("items", payload if isinstance(payload, list) else [])
    if not isinstance(items, list):
        raise ValueError(f"Expected an item list in {content_json}")
    records = []
    for order, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        record = step5.record_from_content_item(item)
        record.setdefault("global_order", order)
        record.setdefault("original_index", order)
        records.append(record)
    return records


def is_native_v2_pages(payload: Any) -> bool:
    return isinstance(payload, list) and (not payload or all(isinstance(page, list) for page in payload))


def flatten_native_v2_pages(pages: list[Any]) -> list[dict[str, Any]]:
    """Flatten MinerU native content_list_v2 pages without sorting or merging."""

    from src.perception.reading_order import extract_text

    records: list[dict[str, Any]] = []
    for page_idx, page in enumerate(pages):
        if not isinstance(page, list):
            continue
        for block_idx, block in enumerate(page):
            if not isinstance(block, dict):
                continue
            record = dict(block)
            text = extract_text(block)
            global_order = len(records)
            record.update(
                {
                    "page_idx": page_idx,
                    "global_order": global_order,
                    "original_index": block_idx,
                    "visual_order": block_idx,
                    "text_for_embedding": text,
                    "text": text,
                    "raw_type": block.get("type"),
                }
            )
            records.append(record)
    return records


def entries_from_content_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from src.reasoning.postprocess import ResolvedNode, node_reading_order_key, node_record_text

    nodes = [ResolvedNode(node_id=index, record=dict(record), merged_node_ids=[index]) for index, record in enumerate(records)]
    ordered = sorted(nodes, key=node_reading_order_key)
    entries = []
    for order, node in enumerate(ordered, start=1):
        entries.append(entry_from_node(order=order, depth=0, node=node, parent_order=None, text=node_record_text(node.record)))
    return entries


def entries_from_tree(root: Any, *, document_title: str | None) -> list[dict[str, Any]]:
    from src.reasoning.postprocess import sorted_render_children

    entries: list[dict[str, Any]] = []

    def walk(node: Any, *, depth: int, parent_order: int | None) -> None:
        children = sorted_render_children(getattr(node, "children", []))
        for child in children:
            order = len(entries) + 1
            entries.append(entry_from_node(order=order, depth=depth, node=child, parent_order=parent_order, text=getattr(child, "text", "")))
            walk(child, depth=depth + 1, parent_order=order)

    walk(root, depth=0, parent_order=None)
    if document_title:
        for entry in entries:
            entry["document_title"] = document_title
    return entries


def entry_from_node(*, order: int, depth: int, node: Any, parent_order: int | None, text: str) -> dict[str, Any]:
    from src.reasoning.postprocess import canonical_render_type

    record = dict(getattr(node, "record", {}))
    node_id = int(getattr(node, "node_id", order - 1))
    merged_node_ids = list(getattr(node, "merged_node_ids", []) or [node_id])
    bboxes = list(iter_node_bboxes(node))
    block_type = canonical_render_type(record)
    return {
        "order": order,
        "depth": depth,
        "parent_order": parent_order,
        "node_id": node_id,
        "merged_node_ids": merged_node_ids,
        "type": block_type,
        "page_idxs": sorted({bbox["page_idx"] for bbox in bboxes if bbox["page_idx"] is not None}),
        "bboxes": bboxes,
        "reading_keys": {
            key: record.get(key)
            for key in (
                "regime_reading_order",
                "dag_reading_order",
                "xycut_reading_order",
                "global_order",
                "visual_order",
                "original_index",
                "index",
            )
            if record.get(key) is not None
        },
        "text_preview": preview_text(text),
    }


def iter_node_bboxes(node: Any) -> list[dict[str, Any]]:
    records = []
    primary = getattr(node, "record", None)
    if isinstance(primary, dict):
        records.append(primary)
        records.extend(record for record in primary.get("merged_records", []) if isinstance(record, dict))

    output: list[dict[str, Any]] = []
    for record_idx, record in enumerate(records):
        chunks = bbox_chunks(record.get("bbox"))
        if not chunks:
            continue
        page_idxs = record.get("source_page_idxs")
        if not isinstance(page_idxs, list) or len(page_idxs) != len(chunks):
            page_idxs = [record.get("page_idx")] * len(chunks)
        for chunk_idx, (bbox, page_idx) in enumerate(zip(chunks, page_idxs)):
            try:
                page = int(page_idx)
            except (TypeError, ValueError):
                page = None
            output.append(
                {
                    "record_idx": record_idx,
                    "chunk_idx": chunk_idx,
                    "page_idx": page,
                    "bbox": [float(value) for value in bbox],
                }
            )
    return output


def bbox_chunks(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    usable_len = len(value) - (len(value) % 4)
    chunks = []
    for index in range(0, usable_len, 4):
        x0, y0, x1, y1 = value[index : index + 4]
        chunks.append((float(x0), float(y0), float(x1), float(y1)))
    return chunks


def draw_order_overlay(
    *,
    pdf_path: Path,
    entries: list[dict[str, Any]],
    output_pdf: Path,
    png_dir: Path,
    zoom: float,
    max_pages: int,
    draw_lines: bool,
) -> None:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        previous_center_by_page: dict[int, tuple[float, float]] = {}
        for entry in entries:
            color = TYPE_COLORS.get(str(entry.get("type")), DEFAULT_COLOR)
            label = f"{int(entry['order']):03d}"
            for bbox_info in entry.get("bboxes", []):
                page_idx = bbox_info.get("page_idx")
                if not isinstance(page_idx, int) or page_idx < 0 or page_idx >= len(doc):
                    continue
                page = doc[page_idx]
                rect = normalized_bbox_to_page_rect(page, bbox_info["bbox"])
                page.draw_rect(rect, color=color, width=1.2, overlay=True)
                center = ((rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0)
                if draw_lines and page_idx in previous_center_by_page:
                    page.draw_line(previous_center_by_page[page_idx], center, color=(0.35, 0.35, 0.35), width=0.4, overlay=True)
                previous_center_by_page[page_idx] = center
                draw_label(page, rect, label, color=color)
                add_pdf_annotation(page, rect, entry)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_pdf, garbage=4, deflate=True)
        render_png_pages(doc, png_dir=png_dir, zoom=zoom, max_pages=max_pages)
    finally:
        doc.close()


def normalized_bbox_to_page_rect(page: Any, bbox: list[float]) -> Any:
    import fitz

    x0, y0, x1, y1 = bbox[:4]
    width = float(page.rect.width)
    height = float(page.rect.height)
    rect = fitz.Rect(x0 / 1000.0 * width, y0 / 1000.0 * height, x1 / 1000.0 * width, y1 / 1000.0 * height)
    return rect & page.rect


def draw_label(page: Any, rect: Any, label: str, *, color: tuple[float, float, float]) -> None:
    import fitz

    font_size = 6.0
    label_width = max(14.0, 4.2 * len(label) + 3.0)
    label_height = 8.0
    x0 = max(float(page.rect.x0), min(float(rect.x0), float(page.rect.x1) - label_width))
    y0 = max(float(page.rect.y0), float(rect.y0) - label_height)
    if y0 <= page.rect.y0 + 1:
        y0 = min(float(rect.y0) + 1.0, float(page.rect.y1) - label_height)
    label_rect = fitz.Rect(x0, y0, x0 + label_width, y0 + label_height)
    try:
        page.draw_rect(label_rect, color=color, fill=(1, 1, 1), width=0.4, fill_opacity=0.78, overlay=True)
    except TypeError:
        page.draw_rect(label_rect, color=color, fill=(1, 1, 1), width=0.4, overlay=True)
    page.insert_text((label_rect.x0 + 1.4, label_rect.y1 - 1.8), label, fontsize=font_size, color=color, overlay=True)


def add_pdf_annotation(page: Any, rect: Any, entry: dict[str, Any]) -> None:
    text = (
        f"order={entry.get('order')} depth={entry.get('depth')} parent={entry.get('parent_order')}\n"
        f"type={entry.get('type')} node_id={entry.get('node_id')} merged={entry.get('merged_node_ids')}\n"
        f"text={entry.get('text_preview')}"
    )
    annot = page.add_text_annot((rect.x1, rect.y0), text)
    annot.set_info(title=f"render order {entry.get('order')}", content=text)
    annot.update()


def render_png_pages(doc: Any, *, png_dir: Path, zoom: float, max_pages: int) -> None:
    import fitz

    png_dir.mkdir(parents=True, exist_ok=True)
    matrix = fitz.Matrix(zoom, zoom)
    page_count = len(doc) if max_pages <= 0 else min(len(doc), max_pages)
    for page_idx in range(page_count):
        pix = doc[page_idx].get_pixmap(matrix=matrix, alpha=False)
        pix.save(png_dir / f"page_{page_idx + 1:03d}.png")


def write_order_json(entries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")


def preview_text(text: Any, limit: int = 160) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


if __name__ == "__main__":
    raise SystemExit(main())
