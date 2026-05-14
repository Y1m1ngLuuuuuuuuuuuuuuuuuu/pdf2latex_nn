#!/usr/bin/env python3
"""Convert Markdown/Nougat MMD into the shared structure-comparison JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comparison_structure import markdown_file_to_comparison, write_comparison_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--doc-id")
    args = parser.parse_args()
    document = markdown_file_to_comparison(args.input, doc_id=args.doc_id)
    write_comparison_json(document, args.output)
    print(f"wrote {args.output} blocks={len(document.blocks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
