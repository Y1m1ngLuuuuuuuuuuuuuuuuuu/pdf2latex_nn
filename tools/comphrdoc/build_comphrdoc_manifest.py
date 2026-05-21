#!/usr/bin/env python3
"""Build a manifest for the prepared CompHRDoc/HRDH test500 split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.comphrdoc.common import config_path, doc_id_from_json, load_config, natural_page_key, read_json, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/external_eval/comphrdoc_test500.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    gold_dir = config_path(cfg, "paths", "gold_test_eval_dir")
    images_dir = config_path(cfg, "paths", "hrdh_images_dir")
    test_json_dir = config_path(cfg, "paths", "hrdh_test_json_dir")
    output = args.output or config_path(cfg, "outputs", "manifest")

    docs: list[dict[str, Any]] = []
    for gold_path in sorted(gold_dir.glob("*.json")):
        doc_id = doc_id_from_json(gold_path)
        image_doc_dir = images_dir / doc_id
        image_paths = sorted(image_doc_dir.glob("*"), key=natural_page_key)
        image_paths = [path for path in image_paths if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        hrdh_test_json = test_json_dir / f"{doc_id}.json"
        if not image_paths or not hrdh_test_json.exists():
            continue
        gold_units = read_json(gold_path)
        test_units = read_json(hrdh_test_json)
        docs.append(
            {
                "document_id": doc_id,
                "gold_json": str(gold_path),
                "hrdh_test_json": str(hrdh_test_json),
                "image_dir": str(image_doc_dir),
                "page_images": [str(path) for path in image_paths],
                "num_pages": len(image_paths),
                "gold_units": len(gold_units) if isinstance(gold_units, list) else None,
                "test_units": len(test_units) if isinstance(test_units, list) else None,
            }
        )
        if args.limit and len(docs) >= args.limit:
            break
    payload = {
        "schema_version": "comphrdoc_test500_manifest_v1",
        "config": str(args.config),
        "gold_dir": str(gold_dir),
        "images_dir": str(images_dir),
        "hrdh_test_json_dir": str(test_json_dir),
        "documents": docs,
        "count": len(docs),
    }
    write_json(output, payload)
    print(f"[comphrdoc] wrote manifest docs={len(docs)} -> {output}")
    return 0 if docs else 1


if __name__ == "__main__":
    raise SystemExit(main())
