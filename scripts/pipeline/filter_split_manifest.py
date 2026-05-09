#!/usr/bin/env python3
"""Filter a v7 graph manifest and emit deterministic document-level splits."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input v7 manifest")
    parser.add_argument("--output", type=Path, required=True, help="Filtered manifest output")
    parser.add_argument("--max-orphan-ratio", type=float, help="Keep documents with orphan_ratio <= this value")
    parser.add_argument("--min-candidate-recall", type=float, default=1.0)
    parser.add_argument("--min-non-none-edges", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Optional cap after filtering and shuffling")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--split-dir", type=Path, help="Directory for train/val/test manifests")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Preserve input order instead of deterministic shuffling.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = read_manifest(args.input)
    documents = list(payload["documents"])
    filtered = [
        record
        for record in documents
        if passes_filters(
            record,
            max_orphan_ratio=args.max_orphan_ratio,
            min_candidate_recall=args.min_candidate_recall,
            min_non_none_edges=args.min_non_none_edges,
        )
    ]
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(filtered)
    if args.limit is not None:
        filtered = filtered[: args.limit]

    metadata = {
        "source_manifest": str(args.input),
        "source_count": len(documents),
        "filtered_count": len(filtered),
        "filters": {
            "max_orphan_ratio": args.max_orphan_ratio,
            "min_candidate_recall": args.min_candidate_recall,
            "min_non_none_edges": args.min_non_none_edges,
            "limit": args.limit,
        },
        "split_method": "document_level",
        "seed": args.seed,
    }
    write_manifest(args.output, filtered, metadata=metadata)

    if args.split_dir:
        splits = split_documents(
            filtered,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        args.split_dir.mkdir(parents=True, exist_ok=True)
        for split_name, records in splits.items():
            write_manifest(
                args.split_dir / f"{args.output.stem}_{split_name}.json",
                records,
                metadata={
                    **metadata,
                    "split": split_name,
                    "split_count": len(records),
                    "split_ratios": {
                        "train": args.train_ratio,
                        "val": args.val_ratio,
                        "test": args.test_ratio,
                    },
                },
            )
        write_json(
            args.split_dir / f"{args.output.stem}_summary.json",
            {
                **metadata,
                "splits": {
                    name: {
                        "count": len(records),
                        "document_ids": [str(record.get("document_id", "")) for record in records],
                    }
                    for name, records in splits.items()
                },
                "label_counts": label_counts(filtered),
            },
        )

    print(
        f"filtered {len(filtered)}/{len(documents)} docs -> {args.output} "
        f"labels={label_counts(filtered)}"
    )
    return 0


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"documents": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise ValueError(f"Expected manifest with a documents list: {path}")
    return payload


def passes_filters(
    record: dict[str, Any],
    *,
    max_orphan_ratio: float | None,
    min_candidate_recall: float,
    min_non_none_edges: int,
) -> bool:
    if max_orphan_ratio is not None and float(record.get("orphan_ratio", 1.0)) > max_orphan_ratio:
        return False
    recall = record.get("candidate_edge_recall")
    if recall is not None and float(recall) < min_candidate_recall:
        return False
    labels = normalized_label_counts(record.get("label_counts", {}))
    if labels[0] + labels[1] < min_non_none_edges:
        return False
    graph_path = record.get("graph_path")
    if not graph_path:
        return False
    return True


def normalized_label_counts(value: Any) -> dict[int, int]:
    labels = {0: 0, 1: 0, 2: 0}
    if not isinstance(value, dict):
        return labels
    name_to_id = {"merge": 0, "parent_child": 1, "none": 2}
    for raw_key, raw_count in value.items():
        try:
            raw_key_text = str(raw_key)
            key = name_to_id[raw_key_text] if raw_key_text in name_to_id else int(raw_key)
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        labels[2 if key >= 2 else key] += count
    return labels


def label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {0: 0, 1: 0, 2: 0}
    for record in records:
        counts = normalized_label_counts(record.get("label_counts", {}))
        for label, value in counts.items():
            totals[label] += value
    return {"merge": totals[0], "parent_child": totals[1], "none": totals[2]}


def split_documents(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one split ratio must be positive")
    train_ratio = train_ratio / total
    val_ratio = val_ratio / total
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    size = len(shuffled)
    train_count = max(1, int(round(size * train_ratio))) if size else 0
    val_count = int(round(size * val_ratio))
    if size >= 3 and val_ratio > 0:
        val_count = max(1, val_count)
    if train_count + val_count >= size and size > 1:
        train_count = max(1, size - val_count - 1)
    test_count = max(0, size - train_count - val_count)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count : train_count + val_count + test_count],
    }


def write_manifest(path: Path, documents: list[dict[str, Any]], *, metadata: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "schema_version": "v7_filtered_manifest_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "success_count": len(documents),
            "documents": documents,
            "metadata": metadata,
        },
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
