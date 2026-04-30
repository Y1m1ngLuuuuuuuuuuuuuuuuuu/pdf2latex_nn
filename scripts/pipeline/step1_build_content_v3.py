#!/usr/bin/env python3
"""Merge visual-order MinerU blocks into content v3 logical items."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.reading_order import build_content_v3, write_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="*_content_list_v2_visual_order.json file")
    parser.add_argument("--output", type=Path, required=True, help="Output content v3 JSON path")
    parser.add_argument("--same-page-y-tolerance", type=float, default=80.0)
    parser.add_argument("--cross-page-bottom-threshold", type=float, default=780.0)
    parser.add_argument("--cross-page-top-threshold", type=float, default=260.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    v3 = build_content_v3(
        payload,
        same_page_cross_column_y_tolerance=args.same_page_y_tolerance,
        cross_page_bottom_threshold=args.cross_page_bottom_threshold,
        cross_page_top_threshold=args.cross_page_top_threshold,
    )
    v3["source_path"] = str(args.input)
    write_json(args.output, v3)
    merged = [item for item in v3["items"] if item.get("merge_count", 1) > 1]
    print(f"wrote {args.output}")
    print(f"items={len(v3['items'])} merged_items={len(merged)} merged_source_blocks={sum(item['merge_count'] for item in merged)}")
    for item in merged[:20]:
        print(
            "merged order={global_order} type={type} pages={source_page_idxs} "
            "orders={source_visual_orders} count={merge_count} text={text}".format(
                global_order=item["global_order"],
                type=item["type"],
                source_page_idxs=item["source_page_idxs"],
                source_visual_orders=item["source_visual_orders"],
                merge_count=item["merge_count"],
                text=item["text_for_embedding"][:120].replace("\n", " "),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
