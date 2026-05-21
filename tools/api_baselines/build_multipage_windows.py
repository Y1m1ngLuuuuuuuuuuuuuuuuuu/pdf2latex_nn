#!/usr/bin/env python3
"""Build consecutive multi-page windows for API/VLM document baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import load_manifest_items, parse_doc_ids, read_json, safe_name, slice_items, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--page-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--doc-ids")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.window_size <= 0:
        raise SystemExit("--window-size must be positive")
    if args.overlap < 0 or args.overlap >= args.window_size:
        raise SystemExit("--overlap must be >=0 and < window-size")
    items = slice_items(load_manifest_items(args.manifest), offset=args.offset, limit=args.limit, doc_ids=parse_doc_ids(args.doc_ids))
    windows: list[dict[str, object]] = []
    stride = args.window_size - args.overlap
    for item in items:
        doc_id = str(item["doc_id"])
        sidecar_path = args.page_image_root / safe_name(doc_id) / "pages.json"
        if not sidecar_path.exists():
            print(f"skip missing pages sidecar doc_id={doc_id} path={sidecar_path}")
            continue
        sidecar = read_json(sidecar_path)
        pages = sidecar.get("pages") or []
        num_pages = len(pages)
        start = 0
        while start < num_pages:
            end = min(num_pages, start + args.window_size)
            page_slice = pages[start:end]
            page_numbers = [int(page["page_index"]) for page in page_slice]
            window_id = f"{safe_name(doc_id)}_p{page_numbers[0]:04d}_p{page_numbers[-1]:04d}"
            windows.append(
                {
                    "doc_id": doc_id,
                    "window_id": window_id,
                    "pages": page_numbers,
                    "image_paths": [str(page["image_path"]) for page in page_slice],
                    "pdf_path": item.get("pdf_path"),
                    "is_first_window": start == 0,
                    "is_last_window": end >= num_pages,
                }
            )
            if end >= num_pages:
                break
            start += stride
    write_json(
        args.output,
        {
            "schema_version": "api_window_manifest_v1",
            "window_size": args.window_size,
            "overlap": args.overlap,
            "source_manifest": str(args.manifest),
            "items": windows,
        },
    )
    print(f"wrote {args.output} windows={len(windows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

