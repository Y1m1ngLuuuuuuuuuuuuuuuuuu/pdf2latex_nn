#!/usr/bin/env python3
"""Compare structural statistics between CompHRDoc gold and prediction folders."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.comphrdoc.common import read_json, write_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    gold_files = {path.name: path for path in sorted(args.gold_dir.glob("*.json"))}
    pred_files = {path.name: path for path in sorted(args.pred_dir.glob("*.json"))}
    rows = []
    global_gold = empty_profile()
    global_pred = empty_profile()
    for name in sorted(set(gold_files) & set(pred_files)):
        gold = read_json(gold_files[name])
        pred = read_json(pred_files[name])
        gold_profile = profile(gold)
        pred_profile = profile(pred)
        rows.append(
            {
                "file": name,
                "node_count_ratio": safe_ratio(pred_profile["node_count"], gold_profile["node_count"]),
                "gold": compact_profile(gold_profile),
                "pred": compact_profile(pred_profile),
                "class_vocab_missing_in_pred": sorted(set(gold_profile["class_counts"]) - set(pred_profile["class_counts"])),
                "class_vocab_extra_in_pred": sorted(set(pred_profile["class_counts"]) - set(gold_profile["class_counts"])),
            }
        )
        merge_profile(global_gold, gold_profile)
        merge_profile(global_pred, pred_profile)
    payload = {
        "schema_version": "comphrdoc_tree_stats_compare_v1",
        "gold_dir": str(args.gold_dir),
        "pred_dir": str(args.pred_dir),
        "documents": len(rows),
        "gold_global": compact_profile(global_gold, top_n=100),
        "pred_global": compact_profile(global_pred, top_n=100),
        "global_class_vocab_missing_in_pred": sorted(set(global_gold["class_counts"]) - set(global_pred["class_counts"])),
        "global_class_vocab_extra_in_pred": sorted(set(global_pred["class_counts"]) - set(global_gold["class_counts"])),
        "rows": rows,
    }
    write_json(args.output, payload)
    print(f"[comphrdoc] compared tree stats docs={len(rows)} -> {args.output}")
    return 0


def empty_profile() -> dict[str, Any]:
    return {
        "node_count": 0,
        "root_class_counts": collections.Counter(),
        "depth_counts": collections.Counter(),
        "parent_child_counts": collections.Counter(),
        "class_counts": collections.Counter(),
    }


def profile(units: Any) -> dict[str, Any]:
    prof = empty_profile()
    if not isinstance(units, list):
        return prof
    prof["node_count"] = len(units)
    parents = [parse_parent(unit.get("parent_id", -1)) if isinstance(unit, dict) else -1 for unit in units]
    classes = [str(unit.get("class", "")) if isinstance(unit, dict) else "" for unit in units]
    depths = compute_depths(parents)
    for idx, cls in enumerate(classes):
        prof["class_counts"][cls] += 1
        if parents[idx] < 0:
            prof["root_class_counts"][cls] += 1
        if 0 <= parents[idx] < len(classes) and parents[idx] != idx:
            prof["parent_child_counts"][f"{classes[parents[idx]]}->{cls}"] += 1
        prof["depth_counts"][str(depths[idx])] += 1
    return prof


def compact_profile(prof: dict[str, Any], top_n: int = 30) -> dict[str, Any]:
    return {
        "node_count": prof["node_count"],
        "root_class_distribution": dict(prof["root_class_counts"].most_common(top_n)),
        "depth_distribution": dict(prof["depth_counts"].most_common(top_n)),
        "parent_class_child_class_top_pairs": dict(prof["parent_child_counts"].most_common(top_n)),
        "class_distribution": dict(prof["class_counts"].most_common(top_n)),
    }


def merge_profile(dst: dict[str, Any], src: dict[str, Any]) -> None:
    dst["node_count"] += int(src["node_count"])
    for key in ("root_class_counts", "depth_counts", "parent_child_counts", "class_counts"):
        dst[key].update(src[key])


def parse_parent(value: Any) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return -1


def compute_depths(parents: list[int]) -> list[int]:
    depths = []
    n = len(parents)
    for i in range(n):
        cur = i
        depth = 0
        seen = set()
        while 0 <= parents[cur] < n and parents[cur] != cur and cur not in seen:
            seen.add(cur)
            cur = parents[cur]
            depth += 1
        depths.append(depth)
    return depths


def safe_ratio(a: int | float, b: int | float) -> float | None:
    return float(a) / float(b) if b else None


if __name__ == "__main__":
    raise SystemExit(main())
