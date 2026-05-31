#!/usr/bin/env python3
"""Convert MinerU middle/model output directly to ComparisonStructureV1."""

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
    doc_id_from_path,
    flatten_content,
    flatten_mineru_lines,
    looks_like_caption,
    source_metadata,
)


def mineru_middle_to_comparison(path: Path, *, doc_id: str | None = None) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    builder = DirectComparisonBuilder(
        doc_id=doc_id or doc_id_from_path(path),
        source_format="mineru_direct",
        source_path=path,
    )
    pdf_info = payload.get("pdf_info") if isinstance(payload, dict) else None
    if isinstance(pdf_info, list):
        for page_idx, page in enumerate(pdf_info):
            add_page(page, builder, page_idx=page_idx)
    elif isinstance(payload, list):
        for page_idx, page in enumerate(payload):
            add_page(page, builder, page_idx=page_idx)
    else:
        raise ValueError(f"Unsupported MinerU direct payload: {type(payload).__name__}")
    return builder.document(metadata={"adapter": "mineru_direct", "input_kind": "middle_json"})


def add_page(page: Any, builder: DirectComparisonBuilder, *, page_idx: int) -> None:
    if not isinstance(page, dict):
        return
    blocks = page.get("preproc_blocks") or page.get("para_blocks") or page.get("blocks") or []
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if isinstance(block, dict):
            add_middle_block(block, builder, page_idx=page_idx)


def add_middle_block(block: dict[str, Any], builder: DirectComparisonBuilder, *, page_idx: int) -> None:
    raw_type = str(block.get("type") or block.get("block_type") or "").casefold()
    text = flatten_mineru_lines(block.get("lines")) or flatten_content(block)
    level = normalized_level(block.get("level"))
    metadata = source_metadata(raw_type, page_idx, block.get("bbox"), {"block_index": block.get("index")})

    if raw_type in {"title", "paragraph_title"}:
        if not builder.maybe_document_title(text, level=level, metadata=metadata):
            builder.add_heading(text, level or 1, metadata=metadata)
        return
    if raw_type == "abstract":
        builder.add_block("abstract", text or "Abstract", parent_id=builder.current_parent_id(), marker="mineru_abstract", metadata=metadata)
        return
    if raw_type in {"image", "figure"}:
        builder.add_float("figure", text, page_idx=page_idx, marker=raw_type, metadata=metadata)
        return
    if raw_type == "table":
        builder.add_float("table", text, page_idx=page_idx, marker=raw_type, metadata=metadata)
        return
    if raw_type in {"algorithm", "code"} or str(block.get("sub_type") or "").casefold() in {"algorithm", "code"}:
        builder.add_float("algorithm", text, page_idx=page_idx, marker=raw_type or "algorithm", metadata=metadata)
        return
    if "caption" in raw_type or looks_like_caption(text):
        builder.add_caption(text, page_idx=page_idx, metadata=metadata)
        return
    if "equation" in raw_type or "formula" in raw_type:
        builder.add_block("display_math", text or "[DISPLAY_MATH]", display_math_count=1, metadata=metadata)
        return
    if raw_type in {"index", "list", "list_item"}:
        for offset, item in enumerate(extract_list_items(block)):
            builder.add_block("list_item", item, marker=str(offset + 1), metadata=metadata | {"list_item_index": offset})
        return
    if raw_type in {"reference", "ref_text", "reference_item"} or str(block.get("sub_type") or "").casefold() in {"reference", "ref_text"}:
        builder.add_block("reference_item", text, metadata=metadata)
        return
    builder.add_textual(text, metadata=metadata)


def extract_list_items(block: dict[str, Any]) -> list[str]:
    for key in ("items", "list_items", "index_content"):
        value = block.get(key)
        if isinstance(value, list):
            return [text for text in (flatten_content(item) for item in value) if text]
    text = flatten_mineru_lines(block.get("lines")) or flatten_content(block)
    return [text] if text else []


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
    document = mineru_middle_to_comparison(args.input, doc_id=args.doc_id)
    write_comparison_json(document, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

