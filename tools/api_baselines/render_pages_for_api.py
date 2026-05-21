#!/usr/bin/env python3
"""Render manifest PDFs into page PNGs for image-window API baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import load_manifest_items, parse_doc_ids, resolve_path, safe_name, slice_items, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--doc-ids")
    return parser


def render_pdf(pdf_path: Path, output_dir: Path, *, doc_id: str, dpi: int, skip_existing: bool) -> dict[str, object]:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("PyMuPDF/fitz is required for PDF page rendering.") from exc
    doc_dir = output_dir / safe_name(doc_id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[dict[str, object]] = []
    for page_idx in range(document.page_count):
        out = doc_dir / f"page_{page_idx + 1:04d}.png"
        if not (skip_existing and out.exists()):
            pix = document[page_idx].get_pixmap(matrix=matrix, alpha=False)
            pix.save(out)
            width, height = pix.width, pix.height
        else:
            # Avoid loading image dependencies just to inspect dimensions.
            pix = document[page_idx].get_pixmap(matrix=matrix, alpha=False)
            width, height = pix.width, pix.height
        pages.append({"page_index": page_idx + 1, "image_path": str(out), "width": width, "height": height})
    sidecar = {"doc_id": doc_id, "pdf_path": str(pdf_path), "dpi": dpi, "num_pages": document.page_count, "pages": pages}
    write_json(doc_dir / "pages.json", sidecar)
    document.close()
    return sidecar


def main() -> int:
    args = build_arg_parser().parse_args()
    items = slice_items(load_manifest_items(args.manifest), offset=args.offset, limit=args.limit, doc_ids=parse_doc_ids(args.doc_ids))
    rendered = 0
    for item in items:
        doc_id = str(item["doc_id"])
        pdf = resolve_path(item.get("pdf_path")) if item.get("pdf_path") else None
        if not pdf or not pdf.exists():
            print(f"skip missing pdf doc_id={doc_id} path={item.get('pdf_path')}")
            continue
        render_pdf(pdf, args.output_dir, doc_id=doc_id, dpi=args.dpi, skip_existing=args.skip_existing)
        rendered += 1
    print(f"rendered_docs={rendered} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

