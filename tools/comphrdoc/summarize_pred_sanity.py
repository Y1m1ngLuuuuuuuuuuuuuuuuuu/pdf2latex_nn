#!/usr/bin/env python3
"""Sanity-check CompHRDoc prediction folders before official evaluation."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.comphrdoc.common import OFFICIAL_CLASSES, config_path, load_config, read_json, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/external_eval/comphrdoc_test500.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Only check the first N manifest documents.")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    manifest = read_json(args.manifest or config_path(cfg, "outputs", "manifest"))
    docs = manifest.get("documents", manifest if isinstance(manifest, list) else [])
    if args.offset:
        docs = docs[args.offset :]
    if args.limit is not None:
        docs = docs[: args.limit]
    expected = {f"{doc['document_id']}.json": doc for doc in docs}
    rows: list[dict[str, Any]] = []
    unknown_labels = collections.Counter()
    parse_fail = []
    missing = []
    for filename, doc in expected.items():
        pred_path = args.pred_dir / filename
        if not pred_path.exists():
            missing.append(filename)
            continue
        try:
            pred = read_json(pred_path)
            gold = read_json(Path(str(doc["gold_json"]))) if doc.get("gold_json") else []
        except Exception as exc:  # noqa: BLE001
            parse_fail.append({"file": filename, "error": repr(exc)})
            continue
        labels = [str(unit.get("class")) for unit in pred if isinstance(unit, dict)]
        for label in labels:
            if label not in OFFICIAL_CLASSES:
                unknown_labels[label] += 1
        rows.append(
            {
                "document_id": doc["document_id"],
                "pred_nodes": len(pred) if isinstance(pred, list) else None,
                "gold_nodes": len(gold) if isinstance(gold, list) else None,
                "node_count_ratio": (len(pred) / len(gold)) if isinstance(pred, list) and isinstance(gold, list) and gold else None,
                "empty_tree": not any(isinstance(unit, dict) and int(unit.get("parent_id", -1)) >= 0 for unit in pred),
                "empty_reading_order": len(pred) == 0 if isinstance(pred, list) else True,
                "page_count_match": page_count(pred) == int(doc.get("num_pages", -1)),
            }
        )
    payload = {
        "schema_version": "comphrdoc_pred_sanity_v1",
        "pred_dir": str(args.pred_dir),
        "expected_docs": len(expected),
        "checked_docs": len(rows),
        "missing_prediction": missing,
        "json_parse_fail": parse_fail,
        "unknown_class_label": dict(unknown_labels),
        "rows": rows,
        "summary": {
            "missing_count": len(missing),
            "parse_fail_count": len(parse_fail),
            "unknown_label_count": sum(unknown_labels.values()),
            "empty_tree_count": sum(1 for row in rows if row["empty_tree"]),
            "empty_reading_order_count": sum(1 for row in rows if row["empty_reading_order"]),
            "page_count_match_count": sum(1 for row in rows if row["page_count_match"]),
        },
    }
    output = args.output or (args.pred_dir.parent / f"{args.pred_dir.name}_sanity.json")
    write_json(output, payload)
    print(f"[comphrdoc] sanity checked={len(rows)} missing={len(missing)} parse_fail={len(parse_fail)} -> {output}")
    return 0 if not missing and not parse_fail and not unknown_labels else 1


def page_count(units: Any) -> int:
    if not isinstance(units, list):
        return 0
    pages = {int(unit.get("page", 0)) for unit in units if isinstance(unit, dict)}
    return len(pages)


if __name__ == "__main__":
    raise SystemExit(main())
