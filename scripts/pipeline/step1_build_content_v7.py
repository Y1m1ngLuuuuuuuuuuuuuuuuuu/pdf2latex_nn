#!/usr/bin/env python3
"""Build safe column-fixed content v7 from MinerU content_list_v2 JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.reading_order import build_content_v7, write_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="MinerU native content_list_v2.json")
    parser.add_argument("--output", type=Path, required=True, help="Output content v7 JSON path")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    v7 = build_content_v7(payload)
    v7["source_path"] = str(args.input)
    write_json(args.output, v7)
    items = v7["items"]
    marker_items = [item for item in items if item.get("list_marker")]
    full_span = sum(1 for item in items if item.get("column_fix_span") == "FULL_SPAN")
    left = sum(1 for item in items if item.get("column_fix_column") == "LEFT_COL")
    right = sum(1 for item in items if item.get("column_fix_column") == "RIGHT_COL")
    print(f"wrote {args.output}")
    print(
        f"items={len(items)} list_marker_items={len(marker_items)} "
        f"full_span={full_span} left={left} right={right} schema={v7.get('schema_version')}"
    )
    for item in items[:20]:
        print(
            "order={global_order} page={page_idx} block={block_idx} span={span} col={col} text={text}".format(
                global_order=item.get("global_order"),
                page_idx=item.get("page_idx"),
                block_idx=item.get("mineru_block_idx"),
                span=item.get("column_fix_span"),
                col=item.get("column_fix_column"),
                text=str(item.get("text_for_embedding") or "")[:100].replace("\n", " "),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
