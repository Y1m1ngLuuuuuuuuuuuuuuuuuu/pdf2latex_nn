#!/usr/bin/env python3
"""Copy/normalize CompHRDoc gold JSON into prediction JSON for oracle smoke."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.comphrdoc.common import config_path, load_config, normalize_class, read_json, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/external_eval/comphrdoc_test500.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-output-dir", type=Path)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    manifest = read_json(args.manifest or config_path(cfg, "outputs", "manifest"))
    docs = manifest.get("documents", manifest if isinstance(manifest, list) else [])
    if args.limit:
        docs = docs[: args.limit]
    out_dir = args.out_dir or (config_path(cfg, "outputs", "prediction_root") / f"oracle_gold_smoke{len(docs)}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc in docs:
        gold_path = Path(str(doc["gold_json"]))
        pred_units = normalize_units(read_json(gold_path))
        write_json(out_dir / gold_path.name, pred_units)
    print(f"[comphrdoc] oracle copied docs={len(docs)} -> {out_dir}")

    if args.run_eval:
        eval_dir = args.eval_output_dir or (config_path(cfg, "outputs", "report_root") / f"oracle_gold_smoke{len(docs)}")
        command = [
            args.python_bin,
            str(Path("tools/comphrdoc/run_comphrdoc_eval.py")),
            "--gt-folder",
            str(config_path(cfg, "paths", "gold_test_eval_dir")),
            "--pred-folder",
            str(out_dir),
            "--output-dir",
            str(eval_dir),
            "--comphrdoc-repo",
            str(config_path(cfg, "paths", "comphrdoc_repo")),
            "--tasks",
            "teds",
            "reading_order",
            "classify",
            "--reading-workers",
            "1",
        ]
        print("[comphrdoc] $ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
    return 0


def normalize_units(units: Any) -> list[dict[str, Any]]:
    if not isinstance(units, list):
        raise ValueError("CompHRDoc gold JSON must be a list")
    normalized: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            continue
        record = dict(unit)
        record["class"] = normalize_class(str(record.get("class") or "paraline"))
        record["page"] = int(record.get("page", 0) or 0)
        record["box"] = [int(round(float(v))) for v in record.get("box", [0, 0, 0, 0])[:4]]
        record["is_meta"] = bool(record.get("is_meta", record["class"] in {"title", "author", "mail", "affili", "header", "footer", "footnote"}))
        record["parent_id"] = int(record.get("parent_id", -1))
        record["relation"] = str(record.get("relation") or ("meta" if record["is_meta"] else "contain"))
        normalized.append(record)
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
