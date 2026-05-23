#!/usr/bin/env python3
"""Build standalone v8 content from MinerU raw middle.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.perception.mineru_v8_reflow import build_v8_from_middle, dump_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--middle-json", required=True, type=Path)
    parser.add_argument(
        "--content-list-json",
        type=Path,
        help="Optional MinerU content_list.json used only to attach float/table asset metadata.",
    )
    parser.add_argument(
        "--style-content-list-json",
        type=Path,
        help="Optional v7-style content list used only to attach style spans by bbox.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--middle-block-source",
        choices=("preproc_blocks", "para_blocks"),
        default="preproc_blocks",
        help="Use preproc_blocks for v8 raw-order reconstruction; para_blocks is only for comparison.",
    )
    parser.add_argument("--debug-page", type=int, help="Zero-based page index to include detailed page order for.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = build_v8_from_middle(
        doc_id=args.doc_id,
        middle_json_path=args.middle_json,
        content_list_json_path=args.content_list_json,
        style_content_list_json_path=args.style_content_list_json,
        middle_block_source=args.middle_block_source,
        debug_page=args.debug_page,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    content_path = args.output_dir / f"{args.doc_id}_content_list_v8.json"
    diag_path = args.output_dir / f"{args.doc_id}_v8_diagnostics.json"
    dump_json(content_path, {k: v for k, v in payload.items() if k != "diagnostics"})
    dump_json(diag_path, payload["diagnostics"])
    print(f"wrote {content_path}")
    print(f"wrote {diag_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
