#!/usr/bin/env python3
"""Attach middle-fragment MERGE labels to rebuilt fragment graphs.

This branch labeler does not read TeX.  It uses the fragment_id bridge emitted
by the middle-fragment pseudo-v7 payload and labels only positive MERGE edges;
all other graph edges are NONE.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MERGE = 0
PARENT_CHILD = 1
NONE = 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True, help="Manifest after graph rebuild.")
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--graph-output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    records = load_manifest_records(resolve(args.input_manifest))
    output_graph_dir = resolve(args.graph_output_dir)
    output_graph_dir.mkdir(parents=True, exist_ok=True)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start = time.time()
    for record in records:
        try:
            successes.append(label_one(record, output_graph_dir=output_graph_dir, force=bool(args.force)))
        except Exception as exc:  # pragma: no cover - batch safety
            failures.append(
                {
                    "document_id": record.get("document_id") or record.get("id"),
                    "graph_path": record.get("graph_path"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    summary = build_summary(successes, failures=failures, elapsed_seconds=time.time() - start)
    output_manifest = resolve(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(
            {
                "schema_version": "middle_fragment_labeled_graph_manifest_v1",
                "source_manifest": str(args.input_manifest),
                "num_documents": len(successes),
                "num_failed": len(failures),
                "label_totals": summary["label_totals"],
                "documents": successes,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_output = resolve(args.summary_output) if args.summary_output else output_manifest.with_suffix(".summary.json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if successes else 2


def label_one(record: dict[str, Any], *, output_graph_dir: Path, force: bool) -> dict[str, Any]:
    import torch

    doc_id = str(record.get("document_id") or record.get("id") or Path(str(record["graph_path"])).stem)
    graph_path = Path(str(record["graph_path"]))
    label_path = Path(str(record["middle_fragment_merge_labels"]))
    output_graph = output_graph_dir / f"{safe_filename(doc_id)}_middlefrag_labeled_graph.pt"
    if output_graph.exists() and not force:
        graph = torch.load(output_graph, map_location="cpu", weights_only=False)
        label_counts = tensor_label_counts(graph.y)
        return {
            **record,
            "graph_path": str(output_graph.resolve()),
            "source_graph_path": str(graph_path),
            "label_counts": label_counts,
            "middle_fragment_labeled_reused": True,
            "middle_fragment_label_report": getattr(graph, "middle_fragment_label_report", {}),
        }

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    labels_payload = load_json(label_path)
    edge_index = graph.edge_index.detach().cpu().long()
    edge_count = int(edge_index.shape[1])
    y = torch.full((edge_count,), NONE, dtype=torch.long)
    edge_lookup: dict[tuple[int, int], list[int]] = {}
    for edge_pos in range(edge_count):
        edge_lookup.setdefault((int(edge_index[0, edge_pos]), int(edge_index[1, edge_pos])), []).append(edge_pos)

    fragment_to_gnn = fragment_id_to_gnn_index(graph)
    missing_node_edges = 0
    missing_candidate_edges = 0
    labeled_edge_positions: set[int] = set()
    positives = labels_payload.get("positive_merge_edges")
    if not isinstance(positives, list):
        positives = []
    for edge in positives:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("src_fragment_id") or "")
        dst = str(edge.get("dst_fragment_id") or "")
        if src not in fragment_to_gnn or dst not in fragment_to_gnn:
            missing_node_edges += 1
            continue
        src_idx = fragment_to_gnn[src]
        dst_idx = fragment_to_gnn[dst]
        positions = edge_lookup.get((src_idx, dst_idx)) or edge_lookup.get((dst_idx, src_idx)) or []
        if not positions:
            missing_candidate_edges += 1
            continue
        for pos in positions:
            y[pos] = MERGE
            labeled_edge_positions.add(pos)

    graph.y = y
    graph.edge_label = y
    graph.middle_fragment_label_schema = {
        "schema_version": "middle_fragment_graph_labels_v1",
        "label_space": {"MERGE": MERGE, "PARENT_CHILD": PARENT_CHILD, "NONE": NONE},
        "source": "middle_fragment_merge_labels",
    }
    report = {
        "doc_id": doc_id,
        "graph_node_count": int(getattr(graph, "num_nodes", 0) or 0),
        "graph_edge_count": edge_count,
        "positive_merge_edges_requested": len(positives),
        "positive_merge_edge_positions_labeled": len(labeled_edge_positions),
        "missing_node_edges": missing_node_edges,
        "missing_candidate_edges": missing_candidate_edges,
        "label_counts": tensor_label_counts(y),
        "candidate_edge_recall": safe_div(len(positives) - missing_node_edges - missing_candidate_edges, len(positives)),
    }
    graph.middle_fragment_label_report = report
    graph.middle_fragment_merge_labels = str(label_path)
    output_graph.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output_graph)
    return {
        **record,
        "graph_path": str(output_graph.resolve()),
        "source_graph_path": str(graph_path),
        "label_counts": report["label_counts"],
        "middle_fragment_label_report": report,
        "middle_fragment_labeled_reused": False,
    }


def fragment_id_to_gnn_index(graph: Any) -> dict[str, int]:
    mapping: dict[str, int] = {}
    existing = getattr(graph, "v7_id_to_gnn_idx", None)
    if isinstance(existing, dict):
        for key, value in existing.items():
            mapping[str(key)] = int(value)
    records = getattr(graph, "node_records", None)
    if isinstance(records, list):
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            for key in ("_v7_node_id", "fragment_id", "node_id"):
                value = record.get(key)
                if isinstance(value, str) and value:
                    mapping[value] = idx
            values = record.get("_v7_source_node_ids")
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and value:
                        mapping[value] = idx
    return mapping


def build_summary(successes: list[dict[str, Any]], *, failures: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    label_totals = {"merge": 0, "parent_child": 0, "none": 0}
    requested = 0
    labeled = 0
    missing_nodes = 0
    missing_candidates = 0
    for record in successes:
        counts = normalize_counts(record.get("label_counts"))
        for key in label_totals:
            label_totals[key] += counts.get(key, 0)
        report = record.get("middle_fragment_label_report") or {}
        requested += int(report.get("positive_merge_edges_requested") or 0)
        labeled += int(report.get("positive_merge_edge_positions_labeled") or 0)
        missing_nodes += int(report.get("missing_node_edges") or 0)
        missing_candidates += int(report.get("missing_candidate_edges") or 0)
    return {
        "schema_version": "middle_fragment_graph_label_summary_v1",
        "doc_count": len(successes),
        "failed_count": len(failures),
        "elapsed_seconds": elapsed_seconds,
        "label_totals": label_totals,
        "positive_merge_edges_requested": requested,
        "positive_merge_edge_positions_labeled": labeled,
        "missing_node_edges": missing_nodes,
        "missing_candidate_edges": missing_candidates,
        "candidate_edge_recall": safe_div(requested - missing_nodes - missing_candidates, requested),
        "failures": failures[:20],
    }


def tensor_label_counts(labels: Any) -> dict[str, int]:
    import torch

    y = labels.detach().cpu().long() if hasattr(labels, "detach") else torch.tensor(labels, dtype=torch.long)
    y = torch.where(y >= NONE, torch.full_like(y, NONE), y)
    counts = torch.bincount(y, minlength=3).tolist()
    return {"merge": int(counts[MERGE]), "parent_child": int(counts[PARENT_CHILD]), "none": int(counts[NONE])}


def normalize_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {"merge": 0, "parent_child": 0, "none": 0}
    return {
        "merge": int(value.get("merge", value.get("0", value.get(0, 0))) or 0),
        "parent_child": int(value.get("parent_child", value.get("1", value.get(1, 0))) or 0),
        "none": int(value.get("none", value.get("2", value.get(2, 0))) or 0),
    }


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    records = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Expected manifest list or documents list: {path}")
    return [record for record in records if isinstance(record, dict) and record.get("graph_path")]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
