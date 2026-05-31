#!/usr/bin/env python3
"""Convert MinerU content-list output directly to ComparisonStructureV1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comparison_structure import write_comparison_json  # noqa: E402
from tools.baselines.direct_comparison_utils import (  # noqa: E402
    DirectComparisonBuilder,
    clean_text,
    doc_id_from_path,
    flatten_content,
    looks_like_caption,
    source_metadata,
)


def contentlist_to_comparison(path: Path, *, doc_id: str | None = None) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    builder = DirectComparisonBuilder(
        doc_id=doc_id or doc_id_from_path(path),
        source_format="contentlist_direct",
        source_path=path,
    )
    if is_v2_pages(payload):
        convert_v2_pages(payload, builder)
    elif isinstance(payload, list):
        convert_v1_items(payload, builder)
    else:
        raise ValueError(f"Unsupported content-list payload: {type(payload).__name__}")
    return builder.document(metadata={"adapter": "contentlist_direct", "input_kind": "content_list_v2" if is_v2_pages(payload) else "content_list"})


def is_v2_pages(payload: Any) -> bool:
    return isinstance(payload, list) and bool(payload) and isinstance(payload[0], list)


def convert_v2_pages(pages: list[Any], builder: DirectComparisonBuilder) -> None:
    for page_idx, page in enumerate(pages):
        if not isinstance(page, list):
            continue
        for block in page:
            if not isinstance(block, dict):
                continue
            add_v2_block(block, builder, page_idx=page_idx)


def add_v2_block(block: dict[str, Any], builder: DirectComparisonBuilder, *, page_idx: int) -> None:
    raw_type = str(block.get("type") or "").casefold()
    content = block.get("content", block)
    text = flatten_content(content)
    metadata = source_metadata(raw_type, page_idx, block.get("bbox"))
    level = normalized_level(content.get("level") if isinstance(content, dict) else block.get("level"))

    if raw_type in {"title", "paragraph_title"}:
        if not builder.maybe_document_title(text, level=level, metadata=metadata):
            builder.add_heading(text, level or 1, metadata=metadata)
        return
    if raw_type in {"abstract"}:
        builder.add_block("abstract", text or "Abstract", parent_id=builder.current_parent_id(), marker="contentlist_abstract", metadata=metadata)
        return
    if raw_type in {"index", "list", "list_item"}:
        add_list_items(content, builder, metadata=metadata)
        return
    if raw_type in {"image", "figure"}:
        builder.add_float("figure", text, page_idx=page_idx, marker=raw_type, metadata=metadata)
        add_embedded_captions(content, builder, page_idx=page_idx, metadata=metadata)
        return
    if raw_type in {"table"}:
        builder.add_float("table", text, page_idx=page_idx, marker=raw_type, metadata=metadata)
        add_embedded_captions(content, builder, page_idx=page_idx, metadata=metadata)
        return
    if raw_type in {"algorithm", "code"}:
        builder.add_float("algorithm", text, page_idx=page_idx, marker=raw_type, metadata=metadata)
        add_embedded_captions(content, builder, page_idx=page_idx, metadata=metadata)
        return
    if "caption" in raw_type or looks_like_caption(text):
        builder.add_caption(text, page_idx=page_idx, metadata=metadata)
        return
    if "equation" in raw_type or "formula" in raw_type:
        builder.add_block("display_math", text or "[DISPLAY_MATH]", display_math_count=1, metadata=metadata)
        return
    if raw_type in {"reference", "ref_text", "reference_item"}:
        builder.add_block("reference_item", text, metadata=metadata)
        return
    builder.add_textual(text, metadata=metadata)


def convert_v1_items(items: list[Any], builder: DirectComparisonBuilder) -> None:
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get("type") or "").casefold()
        page_idx = item.get("page_idx", item.get("page"))
        text = flatten_content(item.get("text", item.get("content", item)))
        metadata = source_metadata(raw_type, page_idx, item.get("bbox"), {"content_index": index})
        level = normalized_level(item.get("text_level") or item.get("level"))
        if raw_type in {"title"} or (raw_type == "text" and level):
            if not builder.maybe_document_title(text, level=level, metadata=metadata):
                builder.add_heading(text, level or 1, metadata=metadata)
        elif raw_type in {"image", "figure"}:
            builder.add_float("figure", text, page_idx=int(page_idx) if isinstance(page_idx, int) else -1, marker=raw_type, metadata=metadata)
        elif raw_type == "table":
            builder.add_float("table", text, page_idx=int(page_idx) if isinstance(page_idx, int) else -1, marker=raw_type, metadata=metadata)
        elif raw_type in {"algorithm", "code"}:
            builder.add_float("algorithm", text, page_idx=int(page_idx) if isinstance(page_idx, int) else -1, marker=raw_type, metadata=metadata)
        elif "caption" in raw_type or looks_like_caption(text):
            builder.add_caption(text, page_idx=int(page_idx) if isinstance(page_idx, int) else -1, metadata=metadata)
        elif raw_type in {"equation", "formula", "interline_equation"}:
            builder.add_block("display_math", text or "[DISPLAY_MATH]", display_math_count=1, metadata=metadata)
        elif raw_type in {"reference", "ref_text", "reference_item"}:
            builder.add_block("reference_item", text, metadata=metadata)
        else:
            builder.add_textual(text, metadata=metadata)


def add_embedded_captions(content: Any, builder: DirectComparisonBuilder, *, page_idx: int, metadata: dict[str, Any]) -> None:
    if not isinstance(content, dict):
        return
    for key, value in content.items():
        if "caption" not in str(key).casefold():
            continue
        caption = flatten_content(value)
        if caption:
            builder.add_caption(caption, page_idx=page_idx, metadata=metadata | {"caption_source_key": key})


def add_list_items(content: Any, builder: DirectComparisonBuilder, *, metadata: dict[str, Any]) -> None:
    items: list[Any] = []
    if isinstance(content, dict):
        for key in ("list_items", "index_content", "items", "content"):
            if isinstance(content.get(key), list):
                items = content[key]
                break
    elif isinstance(content, list):
        items = content
    if not items:
        text = flatten_content(content)
        if text:
            builder.add_block("list_item", text, metadata=metadata)
        return
    for offset, item in enumerate(items):
        text = flatten_content(item)
        if text:
            builder.add_block("list_item", clean_text(text), marker=str(offset + 1), metadata=metadata | {"list_item_index": offset})


def normalized_level(value: Any) -> int | None:
    try:
        level = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, min(level, 5))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--doc-id")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    document = contentlist_to_comparison(args.input, doc_id=args.doc_id)
    write_comparison_json(document, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

