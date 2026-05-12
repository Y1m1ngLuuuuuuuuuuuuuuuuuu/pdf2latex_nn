#!/usr/bin/env python3
"""Remove author-biography/backmatter MERGE labels from an existing manifest.

This is a deterministic hard-case cleanup pass.  It does not rebuild graph
features or rerun TeX alignment; it copies each graph, flips MERGE labels whose
source or target node is author biography/backmatter to NONE, and writes a new
manifest with updated label counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.label_generator import item_is_author_biography_or_backmatter  # noqa: E402


LABEL_NAMES = {0: "merge", 1: "parent_child", 2: "none"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--graph-output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
    payload = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    records = payload.get("documents", payload if isinstance(payload, list) else [])
    if not isinstance(records, list):
        raise ValueError(f"Expected manifest documents list: {args.input_manifest}")

    args.graph_output_dir.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []
    total_flipped = 0
    changed_docs = 0

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or not record.get("graph_path"):
            continue
        graph_path = Path(str(record["graph_path"]))
        doc_id = str(record.get("document_id") or record.get("id") or graph_path.stem)
        output_graph = args.graph_output_dir / f"{safe_filename(doc_id)}_bio_merge_clean_graph.pt"
        if output_graph.exists() and not args.force:
            graph = torch.load(output_graph, map_location="cpu", weights_only=False)
            flipped = int(getattr(graph, "author_bio_merge_flipped", 0))
        else:
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            flipped = flip_author_bio_merges(graph)
            graph.author_bio_merge_flipped = int(flipped)
            graph.source_graph_path = str(graph_path)
            torch.save(graph, output_graph)
        new_record = {
            **record,
            "graph_path": str(output_graph.resolve()),
            "source_graph_path": str(graph_path),
            "pre_author_bio_merge_clean_label_counts": record.get("label_counts"),
            "label_counts": graph_label_counts(graph.y),
            "author_bio_merge_flipped": int(flipped),
        }
        output_records.append(new_record)
        total_flipped += int(flipped)
        if flipped:
            changed_docs += 1
        if index == 1 or index % 100 == 0 or index == len(records):
            print(
                f"[author-bio-merge-clean] {index}/{len(records)} "
                f"changed_docs={changed_docs} flipped={total_flipped}",
                flush=True,
            )

    output_payload = {
        "schema_version": "v7_author_bio_merge_clean_manifest_v1",
        "source_manifest": str(args.input_manifest),
        "num_documents": len(output_records),
        "changed_documents": changed_docs,
        "author_bio_merge_flipped": total_flipped,
        "label_totals": aggregate_label_counts(output_records),
        "documents": output_records,
    }
    write_json(args.output_manifest, output_payload)
    print(
        f"wrote {args.output_manifest} docs={len(output_records)} "
        f"changed_docs={changed_docs} flipped={total_flipped} labels={output_payload['label_totals']}"
    )
    return 0


def flip_author_bio_merges(graph: Any) -> int:
    import torch

    if not hasattr(graph, "y") or not hasattr(graph, "edge_index"):
        return 0
    records = getattr(graph, "node_records", None)
    if not isinstance(records, list) or not records:
        return 0
    y = graph.y.detach().clone().long()
    edge_index = graph.edge_index.detach().cpu().long()
    flipped = 0
    for edge_pos in torch.nonzero(y.cpu() == 0, as_tuple=False).flatten().tolist():
        source = int(edge_index[0, edge_pos])
        target = int(edge_index[1, edge_pos])
        source_record = records[source] if 0 <= source < len(records) and isinstance(records[source], dict) else {}
        target_record = records[target] if 0 <= target < len(records) and isinstance(records[target], dict) else {}
        if item_is_author_biography_or_backmatter(source_record) or item_is_author_biography_or_backmatter(target_record):
            y[edge_pos] = 2
            flipped += 1
    graph.y = y.to(graph.y.device)
    graph.label_counts = graph_label_counts(graph.y)
    return flipped


def graph_label_counts(labels: Any) -> dict[str, int]:
    import torch

    y = labels.detach().cpu().long()
    y = torch.where(y >= 2, torch.full_like(y, 2), y)
    counts = torch.bincount(y, minlength=3).tolist()
    return {LABEL_NAMES[idx]: int(counts[idx]) for idx in range(3)}


def aggregate_label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {name: 0 for name in LABEL_NAMES.values()}
    for record in records:
        counts = record.get("label_counts", {})
        if not isinstance(counts, dict):
            continue
        for name in totals:
            totals[name] += int(counts.get(name, 0))
    return totals


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
