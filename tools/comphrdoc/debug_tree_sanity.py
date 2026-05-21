#!/usr/bin/env python3
"""Diagnose CompHRDoc parent_id/relation tree health for gold/pred JSON folders."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
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
    names = sorted(set(gold_files) | set(pred_files))
    rows = []
    aggregate = new_counter_payload()
    for name in names:
        gold = read_json(gold_files[name]) if name in gold_files else None
        pred = read_json(pred_files[name]) if name in pred_files else None
        gold_stats = tree_stats(gold) if isinstance(gold, list) else missing_stats()
        pred_stats = tree_stats(pred) if isinstance(pred, list) else missing_stats()
        row = {
            "file": name,
            "gold_exists": name in gold_files,
            "pred_exists": name in pred_files,
            "gold": gold_stats,
            "pred": pred_stats,
            "node_count_ratio": safe_ratio(pred_stats["node_count"], gold_stats["node_count"]),
            "page_count_match": pred_stats["page_count"] == gold_stats["page_count"],
        }
        rows.append(row)
        update_aggregate(aggregate, row)
    payload = {
        "schema_version": "comphrdoc_tree_sanity_v1",
        "gold_dir": str(args.gold_dir),
        "pred_dir": str(args.pred_dir),
        "documents": len(rows),
        "summary": finalize_aggregate(aggregate),
        "rows": rows,
    }
    write_json(args.output, payload)
    print(f"[comphrdoc] tree sanity docs={len(rows)} -> {args.output}")
    return 0


def missing_stats() -> dict[str, Any]:
    return {
        "node_count": 0,
        "page_count": 0,
        "node_id_unique": False,
        "missing_parent_ids": [],
        "self_parent_ids": [],
        "cycle_nodes": [],
        "root_count": 0,
        "unreachable_count": 0,
        "max_depth": 0,
        "mean_depth": 0.0,
        "parent_child_counts": {},
        "children_parent_consistency_errors": [],
    }


def tree_stats(units: list[Any]) -> dict[str, Any]:
    records = [unit if isinstance(unit, dict) else {} for unit in units]
    n = len(records)
    pages = {int(unit.get("page", 0) or 0) for unit in records}
    explicit_ids = [unit.get("id", unit.get("node_id")) for unit in records if unit.get("id", unit.get("node_id")) is not None]
    node_id_unique = len(explicit_ids) == len(set(map(str, explicit_ids))) if explicit_ids else True
    parents = [parse_parent(unit.get("parent_id", -1)) for unit in records]
    missing = [i for i, p in enumerate(parents) if p >= n or p < -1]
    self_parent = [i for i, p in enumerate(parents) if p == i]
    children: dict[int, list[int]] = collections.defaultdict(list)
    roots = []
    pair_counts: collections.Counter[str] = collections.Counter()
    for i, p in enumerate(parents):
        if 0 <= p < n and p != i:
            children[p].append(i)
            pair = f"{records[p].get('class', '')}->{records[i].get('class', '')}"
            pair_counts[pair] += 1
        elif p < 0:
            roots.append(i)
    cycle_nodes = detect_cycles(parents)
    depths = compute_depths(parents)
    reachable = set()
    stack = list(roots)
    while stack:
        cur = stack.pop()
        if cur in reachable or cur < 0 or cur >= n:
            continue
        reachable.add(cur)
        stack.extend(children.get(cur, []))
    consistency_errors = children_consistency(records, parents)
    return {
        "node_count": n,
        "page_count": len(pages),
        "node_id_unique": node_id_unique,
        "missing_parent_ids": missing[:50],
        "missing_parent_count": len(missing),
        "self_parent_ids": self_parent[:50],
        "self_parent_count": len(self_parent),
        "cycle_nodes": sorted(cycle_nodes)[:50],
        "cycle_node_count": len(cycle_nodes),
        "root_count": len(roots),
        "unreachable_count": n - len(reachable),
        "max_depth": max(depths) if depths else 0,
        "mean_depth": float(statistics.mean(depths)) if depths else 0.0,
        "parent_child_counts": dict(pair_counts.most_common(50)),
        "children_parent_consistency_errors": consistency_errors[:50],
        "children_parent_consistency_error_count": len(consistency_errors),
    }


def parse_parent(value: Any) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return -1


def detect_cycles(parents: list[int]) -> set[int]:
    n = len(parents)
    cycle_nodes: set[int] = set()
    for start in range(n):
        seen: dict[int, int] = {}
        cur = start
        step = 0
        while 0 <= cur < n:
            if cur in seen:
                cycle_nodes.update(node for node, pos in seen.items() if pos >= seen[cur])
                break
            seen[cur] = step
            step += 1
            nxt = parents[cur]
            if nxt == cur:
                cycle_nodes.add(cur)
                break
            cur = nxt
    return cycle_nodes


def compute_depths(parents: list[int]) -> list[int]:
    depths = []
    n = len(parents)
    for i in range(n):
        depth = 0
        cur = i
        seen = set()
        while 0 <= parents[cur] < n and parents[cur] != cur and cur not in seen:
            seen.add(cur)
            cur = parents[cur]
            depth += 1
        depths.append(depth)
    return depths


def children_consistency(records: list[dict[str, Any]], parents: list[int]) -> list[dict[str, int]]:
    errors = []
    for parent_index, unit in enumerate(records):
        children = unit.get("children")
        if not isinstance(children, list):
            continue
        for child in children:
            child_index = parse_parent(child)
            if not (0 <= child_index < len(records)) or parents[child_index] != parent_index:
                errors.append({"parent": parent_index, "child": child_index})
    return errors


def safe_ratio(a: int | float, b: int | float) -> float | None:
    return float(a) / float(b) if b else None


def new_counter_payload() -> dict[str, Any]:
    return {"pred_missing": 0, "gold_missing": 0, "pred_errors": collections.Counter(), "gold_errors": collections.Counter()}


def update_aggregate(aggregate: dict[str, Any], row: dict[str, Any]) -> None:
    if not row["pred_exists"]:
        aggregate["pred_missing"] += 1
    if not row["gold_exists"]:
        aggregate["gold_missing"] += 1
    for side in ("gold", "pred"):
        stats = row[side]
        counter = aggregate[f"{side}_errors"]
        for key in ("missing_parent_count", "self_parent_count", "cycle_node_count", "unreachable_count", "children_parent_consistency_error_count"):
            counter[key] += int(stats.get(key, 0) or 0)
        if not stats.get("node_id_unique", True):
            counter["duplicate_node_id_docs"] += 1


def finalize_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "pred_missing": aggregate["pred_missing"],
        "gold_missing": aggregate["gold_missing"],
        "pred_errors": dict(aggregate["pred_errors"]),
        "gold_errors": dict(aggregate["gold_errors"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
