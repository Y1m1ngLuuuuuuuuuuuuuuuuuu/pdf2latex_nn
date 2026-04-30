#!/usr/bin/env python3
"""Add PyMuPDF style span sequences to content v3 JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.style_spans import StyleConfig, enrich_content_v3_with_styles  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="*_content_list_v3.json")
    parser.add_argument("--pdf", type=Path, required=True, help="Original PDF")
    parser.add_argument("--output", type=Path, required=True, help="Output style-enriched v3 JSON")
    parser.add_argument("--clip-margin", type=float, default=2.0)
    parser.add_argument("--size-bucket", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = enrich_content_v3_with_styles(
        args.input,
        args.pdf,
        args.output,
        StyleConfig(clip_margin=args.clip_margin, size_bucket=args.size_bucket),
    )
    items = payload.get("items", [])
    nonempty = [item for item in items if item.get("style_spans")]
    print(f"wrote {args.output}")
    print(f"items={len(items)} items_with_styles={len(nonempty)}")
    for item in nonempty[:10]:
        preview = item["style_spans"][0]["text"][:80].replace("\n", " ")
        print(
            "order={global_order} type={type} spans={count} baseline={baseline} first={preview}".format(
                global_order=item.get("global_order"),
                type=item.get("type"),
                count=item.get("style_span_count"),
                baseline=item.get("style_baseline_size"),
                preview=preview,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
