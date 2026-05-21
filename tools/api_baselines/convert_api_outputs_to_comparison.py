#!/usr/bin/env python3
"""Convert API document outputs into comparison_structure_v1 JSON."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from common import infer_source_format, load_manifest_items, parse_doc_ids, safe_name, slice_items, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation.comparison_structure import (  # noqa: E402
    latex_file_to_comparison,
    markdown_file_to_comparison,
    write_comparison_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--doc-output-dir", type=Path, required=True)
    parser.add_argument("--source-format", choices=["latex", "markdown", "auto"], default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--doc-ids")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def find_doc_output(root: Path, doc_id: str) -> Path | None:
    for suffix in (".tex", ".md", ".mmd", ".markdown"):
        path = root / f"{safe_name(doc_id)}{suffix}"
        if path.exists():
            return path
    return None


def main() -> int:
    args = build_arg_parser().parse_args()
    items = slice_items(load_manifest_items(args.manifest), offset=args.offset, limit=args.limit, doc_ids=parse_doc_ids(args.doc_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    failed = 0
    for item in items:
        doc_id = str(item["doc_id"])
        output_path = args.output_dir / f"{safe_name(doc_id)}.json"
        if args.skip_existing and output_path.exists():
            ok += 1
            continue
        source = find_doc_output(args.doc_output_dir, doc_id)
        if not source:
            failed += 1
            write_json(args.output_dir / f"{safe_name(doc_id)}.error.json", {"error": "missing doc output", "doc_id": doc_id})
            continue
        try:
            fmt = infer_source_format(source, args.source_format)
            doc = latex_file_to_comparison(source, doc_id=doc_id) if fmt == "latex" else markdown_file_to_comparison(source, doc_id=doc_id)
            write_comparison_json(doc, output_path)
            ok += 1
        except Exception as exc:
            failed += 1
            write_json(output_path.with_suffix(".error.json"), {"error": str(exc), "traceback": traceback.format_exc()})
    print(f"converted ok={ok} failed={failed} output_dir={args.output_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

