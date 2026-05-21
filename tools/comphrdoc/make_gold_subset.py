#!/usr/bin/env python3
"""Create a CompHRDoc gold subset folder matching a manifest slice."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.comphrdoc.common import config_path, load_config, read_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/external_eval/comphrdoc_test500.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--clean", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    manifest = read_json(args.manifest or config_path(cfg, "outputs", "manifest"))
    docs = manifest.get("documents", manifest if isinstance(manifest, list) else [])
    if args.offset:
        docs = docs[args.offset :]
    if args.limit:
        docs = docs[: args.limit]
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        source = Path(str(doc["gold_json"]))
        shutil.copy2(source, args.out_dir / source.name)
    print(f"[comphrdoc] gold subset docs={len(docs)} -> {args.out_dir}")
    return 0 if docs else 1


if __name__ == "__main__":
    raise SystemExit(main())
