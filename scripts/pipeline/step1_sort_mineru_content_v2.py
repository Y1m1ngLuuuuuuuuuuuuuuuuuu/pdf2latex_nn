#!/usr/bin/env python3
"""Build page-local visual reading order for MinerU content_list_v2 JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.reading_order import load_content_list_v2, sort_content_list_v2, write_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="MinerU *_content_list_v2.json file")
    parser.add_argument("--output", type=Path, required=True, help="Output visual-order JSON path")
    parser.add_argument(
        "--keep-auxiliary",
        action="store_true",
        help="Keep page headers, footers, page numbers, and other auxiliary blocks",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    pages = load_content_list_v2(args.input)
    payload = sort_content_list_v2(pages, keep_auxiliary=args.keep_auxiliary)
    payload["source_path"] = str(args.input)
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"pages={len(payload['pages'])}")
    for summary in payload["page_summaries"]:
        print(
            "page={page_idx} input={input_blocks} output={output_blocks} "
            "columns={column_count} full_width={full_width_blocks} text_runs={text_runs}".format(**summary)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
