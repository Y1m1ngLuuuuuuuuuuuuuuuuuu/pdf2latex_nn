#!/usr/bin/env python3
"""Build list-aware content v4 from content v3 JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.reading_order import build_content_v4, write_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="*_content_list_v3.json")
    parser.add_argument("--output", type=Path, required=True, help="Output content v4 JSON")
    parser.add_argument("--x-alignment-tolerance", type=float, default=28.0)
    parser.add_argument("--parent-indent-threshold", type=float, default=20.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    v4 = build_content_v4(
        payload,
        x_alignment_tolerance=args.x_alignment_tolerance,
        parent_indent_threshold=args.parent_indent_threshold,
    )
    v4["source_path"] = str(args.input)
    write_json(args.output, v4)
    items = v4["items"]
    list_items = [item for item in items if item.get("list_item_id")]
    print(f"wrote {args.output}")
    print(f"items={len(items)} list_items={len(list_items)} marked_item_merge=False")
    for item in list_items[:20]:
        print(
            "order={global_order} id={list_item_id} marker={marker} text={text}".format(
                global_order=item.get("global_order"),
                list_item_id=item.get("list_item_id"),
                marker=(item.get("list_marker") or {}).get("marker"),
                text=str(item.get("text_for_embedding") or "")[:120].replace("\n", " "),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
