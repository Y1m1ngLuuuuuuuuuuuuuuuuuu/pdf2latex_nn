#!/usr/bin/env python3
"""Build a stable manifest for API/VLM baseline evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import existing_path_from, load_manifest_items, parse_doc_ids, slice_items, stable_doc_id, utc_now, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--doc-ids")
    parser.add_argument("--require-pdf", action="store_true")
    parser.add_argument("--prefer-existing-gold", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--sort-by-doc-id", action="store_true", default=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    raw_items = load_manifest_items(args.source_manifest)
    selected = slice_items(
        raw_items,
        offset=args.offset,
        limit=args.limit,
        doc_ids=parse_doc_ids(args.doc_ids),
        sort_by_doc_id=args.sort_by_doc_id,
    )
    items: list[dict[str, object]] = []
    missing_pdf = 0
    missing_gold = 0
    for record in selected:
        doc_id = stable_doc_id(record)
        pdf = existing_path_from(record, ("pdf_path", "source_pdf", "original_pdf", "paired_original_pdf", "gold_pdf"))
        gold = existing_path_from(record, ("gold_comparison_path", "gold_structure", "gold_json", "gold_comparison"))
        ours = existing_path_from(record, ("ours_comparison_path", "generated_structure", "pred_comparison", "prediction_structure"))
        nougat = existing_path_from(record, ("nougat_comparison_path", "nougat_structure", "nougat_comparison"))
        gold_tex = existing_path_from(record, ("gold_tex_path", "source_tex", "tex_path"))
        if pdf is None:
            missing_pdf += 1
            if args.require_pdf or args.skip_missing:
                continue
        if gold is None:
            missing_gold += 1
            if args.prefer_existing_gold and args.skip_missing:
                continue
        items.append(
            {
                "doc_id": doc_id,
                "pdf_path": str(pdf) if pdf else None,
                "gold_comparison_path": str(gold) if gold else None,
                "ours_comparison_path": str(ours) if ours else None,
                "nougat_comparison_path": str(nougat) if nougat else None,
                "gold_tex_path": str(gold_tex) if gold_tex else None,
                "metadata": {"source_record": record},
            }
        )
    payload = {
        "schema_version": "api_eval_manifest_v1",
        "created_at": utc_now(),
        "source_manifest": str(args.source_manifest),
        "summary": {
            "source_items": len(raw_items),
            "selected_items": len(selected),
            "written_items": len(items),
            "missing_pdf_in_selection": missing_pdf,
            "missing_gold_comparison_in_selection": missing_gold,
        },
        "items": items,
    }
    write_json(args.output, payload)
    print(f"wrote {args.output} docs={len(items)} missing_pdf={missing_pdf} missing_gold={missing_gold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

