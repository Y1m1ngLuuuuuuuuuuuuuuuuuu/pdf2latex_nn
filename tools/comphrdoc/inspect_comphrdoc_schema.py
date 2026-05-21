#!/usr/bin/env python3
"""Inspect CompHRDoc gold JSON schema and label vocabulary."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.comphrdoc.common import config_path, load_config, normalize_class, read_json, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/external_eval/comphrdoc_test500.yaml"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    gold_dir = config_path(cfg, "paths", "gold_test_eval_dir")
    files = sorted(gold_dir.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    top_types = collections.Counter()
    key_counts = collections.Counter()
    value_vocab: dict[str, collections.Counter[str]] = {
        "class": collections.Counter(),
        "type": collections.Counter(),
        "label": collections.Counter(),
        "category": collections.Counter(),
        "relation": collections.Counter(),
    }
    required_keys = None
    examples: dict[str, Any] = {}
    unit_count = 0
    for path in files:
        obj = read_json(path)
        top_types[type(obj).__name__] += 1
        if not isinstance(obj, list):
            continue
        if path.name not in examples:
            examples[path.name] = obj[:3]
        for unit in obj:
            if not isinstance(unit, dict):
                continue
            unit_count += 1
            keys = set(unit)
            required_keys = keys if required_keys is None else required_keys & keys
            key_counts.update(keys)
            for key in value_vocab:
                if key in unit:
                    value_vocab[key][str(unit[key])] += 1
    official_class_vocab = sorted({normalize_class(label) for label in value_vocab["class"]})
    payload = {
        "schema_version": "comphrdoc_schema_inspection_v1",
        "files_inspected": len(files),
        "unit_count": unit_count,
        "top_level_types": dict(top_types),
        "common_required_keys": sorted(required_keys or []),
        "key_counts": dict(key_counts),
        "vocab": {key: dict(counter.most_common()) for key, counter in value_vocab.items()},
        "official_class_vocab_normalized": official_class_vocab,
        "prediction_required_fields_inferred": [
            "text",
            "box",
            "class",
            "page",
            "is_meta",
            "parent_id",
            "relation",
        ],
        "notes": [
            "Official classify_eval requires prediction JSON files to have the same file set and unit count as gold.",
            "Official teds/reading_order consume text/class/parent_id/relation; reading_order separately evaluates floating groups.",
        ],
        "examples": examples,
    }
    print("Top-level:", dict(top_types))
    print("Required keys:", payload["common_required_keys"])
    print("Class vocab:", official_class_vocab)
    print("Relation vocab:", dict(value_vocab["relation"].most_common()))
    if args.output:
        write_json(args.output, payload)
        print(f"[comphrdoc] wrote schema report -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
