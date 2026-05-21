#!/usr/bin/env python3
"""Convert HRDH per-page images into per-document PDFs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.comphrdoc.common import config_path, load_config, natural_page_key, read_json, safe_doc_id, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/external_eval/comphrdoc_test500.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--sidecar-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    manifest_path = args.manifest or config_path(cfg, "outputs", "manifest")
    payload = read_json(manifest_path)
    docs = payload.get("documents", payload if isinstance(payload, list) else [])
    if args.limit:
        docs = docs[: args.limit]
    pdf_dir = args.pdf_dir or config_path(cfg, "outputs", "pdf_dir")
    sidecar_dir = args.sidecar_dir or config_path(cfg, "outputs", "pdf_sidecar_dir")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, Any]] = []
    for doc in docs:
        doc_id = str(doc["document_id"])
        pdf_path = pdf_dir / f"{safe_doc_id(doc_id)}.pdf"
        sidecar_path = sidecar_dir / f"{safe_doc_id(doc_id)}.json"
        if pdf_path.exists() and sidecar_path.exists() and not args.overwrite:
            outputs.append({"document_id": doc_id, "pdf_path": str(pdf_path), "sidecar": str(sidecar_path), "reused": True})
            continue
        page_paths = [Path(path) for path in doc.get("page_images", [])]
        if not page_paths:
            image_dir = Path(str(doc["image_dir"]))
            page_paths = sorted(
                [path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}],
                key=natural_page_key,
            )
        pdf_info = images_to_pdf(page_paths, pdf_path)
        write_json(sidecar_path, {"document_id": doc_id, "pdf_path": str(pdf_path), "pages": pdf_info})
        outputs.append({"document_id": doc_id, "pdf_path": str(pdf_path), "sidecar": str(sidecar_path), "pages": len(pdf_info)})
        print(f"[comphrdoc] pdf {doc_id} pages={len(pdf_info)} -> {pdf_path}", flush=True)
    updated = dict(payload) if isinstance(payload, dict) else {"documents": docs}
    by_id = {item["document_id"]: item for item in outputs}
    for doc in updated.get("documents", []):
        if doc.get("document_id") in by_id:
            doc["pdf_path"] = by_id[doc["document_id"]]["pdf_path"]
            doc["pdf_sidecar"] = by_id[doc["document_id"]]["sidecar"]
    write_json(manifest_path, updated)
    print(f"[comphrdoc] updated manifest pdf paths -> {manifest_path}")
    return 0


def images_to_pdf(page_paths: list[Path], pdf_path: Path) -> list[dict[str, Any]]:
    from PIL import Image

    if not page_paths:
        raise ValueError("No page images provided")
    images = []
    info = []
    for index, path in enumerate(page_paths):
        image = Image.open(path)
        dpi = image.info.get("dpi", (72, 72))
        rgb = image.convert("RGB")
        images.append(rgb)
        info.append(
            {
                "page": index,
                "image_path": str(path),
                "width": image.width,
                "height": image.height,
                "dpi": list(dpi) if isinstance(dpi, tuple) else dpi,
            }
        )
    first, rest = images[0], images[1:]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    first.save(pdf_path, save_all=True, append_images=rest, resolution=72.0)
    for image in images:
        image.close()
    return info


if __name__ == "__main__":
    raise SystemExit(main())
