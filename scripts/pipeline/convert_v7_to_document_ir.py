#!/usr/bin/env python3
"""Convert *_content_list_v7_styles.json into the stable DocumentIR JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters import MinerUV7DocumentIRAdapterConfig, write_v7_document_ir  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="*_content_list_v7_styles.json")
    parser.add_argument("--output", type=Path, required=True, help="Output DocumentIR JSON")
    parser.add_argument("--pdf", type=Path, help="Optional source PDF path for provenance")
    parser.add_argument("--doc-id", help="Optional stable document id")
    parser.add_argument("--allow-unstyled", action="store_true", help="Allow *_content_list_v7.json without style spans")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = MinerUV7DocumentIRAdapterConfig(require_styles=not args.allow_unstyled)
    document = write_v7_document_ir(
        args.input,
        args.output,
        pdf_path=args.pdf,
        doc_id=args.doc_id,
        config=config,
    )
    print(f"wrote {args.output}")
    print(f"doc_id={document.doc_id} pages={len(document.pages)} nodes={len(document.nodes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
