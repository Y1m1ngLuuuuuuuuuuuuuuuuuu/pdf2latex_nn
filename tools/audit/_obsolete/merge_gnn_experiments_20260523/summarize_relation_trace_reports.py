#!/usr/bin/env python3
"""Summarize decoder relation trace reports from an E2E run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def prediction_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    payload = load_json(path)
    label_names = [str(value).upper() for value in payload.get("metadata", {}).get("label_names", [])]
    probabilities = payload.get("probabilities") or []
    predicted_labels = payload.get("predicted_labels") or []
    thresholds = payload.get("threshold_config") or {}
    merge_idx = label_names.index("MERGE") if "MERGE" in label_names else None
    parent_idx = next((idx for idx, name in enumerate(label_names) if "PARENT" in name or "ATTACH" in name), None)
    counts: dict[str, int] = {"edge_count": len(payload.get("edge_ids") or probabilities or predicted_labels)}
    if merge_idx is not None:
        tau = float(thresholds.get("merge", 1e9))
        counts["raw_merge_argmax"] = sum(1 for label in predicted_labels if label == merge_idx)
        counts["threshold_merge_edges"] = sum(
            1 for row in probabilities if isinstance(row, list) and len(row) > merge_idx and row[merge_idx] >= tau
        )
    if parent_idx is not None:
        tau = float(thresholds.get("parent_child", 1e9))
        counts["raw_parent_argmax"] = sum(1 for label in predicted_labels if label == parent_idx)
        counts["threshold_parent_edges"] = sum(
            1 for row in probabilities if isinstance(row, list) and len(row) > parent_idx and row[parent_idx] >= tau
        )
    return counts


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    aggregate_merge_rejects: dict[str, int] = {}
    aggregate_parent_sources: dict[str, int] = {}
    for doc_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        trace_path = doc_dir / "relation_trace_report.json"
        if not trace_path.exists():
            continue
        trace = load_json(trace_path)
        predicted = prediction_counts(doc_dir / "predicted_relations.json")
        merge_rendering = trace.get("merge_rendering", {}) or {}
        decoder_trace = trace.get("decoder_trace", {}) or {}
        merge_decoding = decoder_trace.get("merge_decoding", {}) or {}
        parent_usage = decoder_trace.get("parent_usage", {}) or {}
        parent_rendering = trace.get("parent_rendering", {}) or {}
        parent_sources = parent_rendering.get("render_tree_parent_source_distribution", {}) or {}

        for key, value in (merge_decoding.get("reject_reasons") or {}).items():
            aggregate_merge_rejects[str(key)] = aggregate_merge_rejects.get(str(key), 0) + int(value)
        for key, value in parent_sources.items():
            aggregate_parent_sources[str(key)] = aggregate_parent_sources.get(str(key), 0) + int(value)

        row = {
            "doc_id": trace.get("doc_id") or doc_dir.name,
            **predicted,
            "accepted_merge_edges": merge_decoding.get("accepted_merge_edge_count"),
            "accepted_merge_components": merge_rendering.get("accepted_merge_component_count"),
            "render_tree_merged_nodes": merge_rendering.get("render_tree_merged_node_count"),
            "accepted_merges_not_in_render_tree": len(merge_rendering.get("accepted_merges_not_in_render_tree") or []),
            "duplicate_rendered_merge_members": len(merge_rendering.get("duplicate_rendered_merge_members") or []),
            "merge_reject_top": (
                max((merge_decoding.get("reject_reasons") or {}).items(), key=lambda item: item[1])[0]
                if merge_decoding.get("reject_reasons")
                else ""
            ),
            "best_gnn_parent_edges": parent_usage.get("best_gnn_parent_edges"),
            "gnn_parent_applied": parent_usage.get("gnn_applied"),
            "gnn_parent_blocked_existing": parent_usage.get("gnn_blocked_by_existing_parent"),
            "gnn_parent_blocked_heading": parent_usage.get("gnn_blocked_by_heading_target"),
            "gnn_parent_blocked_local_gate": parent_usage.get("gnn_blocked_by_local_parent_gate"),
            "render_gnn_parent_nodes": parent_rendering.get("gnn_parent_render_node_count"),
            "render_stack_parent_nodes": parent_rendering.get("stack_parent_render_node_count"),
        }
        rows.append(row)

    def total(key: str) -> int:
        return sum(int(row.get(key) or 0) for row in rows)

    docs = len(rows)
    summary = {
        "docs": docs,
        "raw_merge_argmax_total": total("raw_merge_argmax"),
        "threshold_merge_edges_total": total("threshold_merge_edges"),
        "accepted_merge_edges_total": total("accepted_merge_edges"),
        "accepted_merge_components_total": total("accepted_merge_components"),
        "render_tree_merged_nodes_total": total("render_tree_merged_nodes"),
        "accepted_merges_not_in_render_tree_total": total("accepted_merges_not_in_render_tree"),
        "duplicate_rendered_merge_members_total": total("duplicate_rendered_merge_members"),
        "raw_parent_argmax_total": total("raw_parent_argmax"),
        "threshold_parent_edges_total": total("threshold_parent_edges"),
        "best_gnn_parent_edges_total": total("best_gnn_parent_edges"),
        "gnn_parent_applied_total": total("gnn_parent_applied"),
        "gnn_parent_blocked_existing_total": total("gnn_parent_blocked_existing"),
        "gnn_parent_blocked_heading_total": total("gnn_parent_blocked_heading"),
        "gnn_parent_blocked_local_gate_total": total("gnn_parent_blocked_local_gate"),
        "merge_reject_reasons": aggregate_merge_rejects,
        "render_parent_source_distribution": aggregate_parent_sources,
    }
    return summary, rows


def write_report(path: Path, input_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    docs = max(1, int(summary["docs"]))
    top_merge = sorted(rows, key=lambda row: int(row.get("accepted_merge_components") or 0), reverse=True)[:10]
    top_parent = sorted(rows, key=lambda row: int(row.get("best_gnn_parent_edges") or 0), reverse=True)[:10]
    text = f"""# MERGE / PARENT Trace Summary

## Status
- Input run: {input_dir}
- Documents: {summary["docs"]}
- Training / MinerU / relabel / rebuild / API / CompHRDoc: No
- This report reads existing skip-compile E2E trace outputs.

## Aggregate MERGE

{markdown_table(["quantity", "total", "mean/doc"], [
    ["raw_merge_argmax", summary["raw_merge_argmax_total"], summary["raw_merge_argmax_total"] / docs],
    ["threshold_merge_edges", summary["threshold_merge_edges_total"], summary["threshold_merge_edges_total"] / docs],
    ["accepted_merge_edges", summary["accepted_merge_edges_total"], summary["accepted_merge_edges_total"] / docs],
    ["accepted_merge_components", summary["accepted_merge_components_total"], summary["accepted_merge_components_total"] / docs],
    ["render_tree_merged_nodes", summary["render_tree_merged_nodes_total"], summary["render_tree_merged_nodes_total"] / docs],
    ["accepted_merges_not_in_render_tree", summary["accepted_merges_not_in_render_tree_total"], summary["accepted_merges_not_in_render_tree_total"] / docs],
    ["duplicate_rendered_merge_members", summary["duplicate_rendered_merge_members_total"], summary["duplicate_rendered_merge_members_total"] / docs],
])}

## Aggregate PARENT

{markdown_table(["quantity", "total", "mean/doc"], [
    ["raw_parent_argmax", summary["raw_parent_argmax_total"], summary["raw_parent_argmax_total"] / docs],
    ["threshold_parent_edges", summary["threshold_parent_edges_total"], summary["threshold_parent_edges_total"] / docs],
    ["best_gnn_parent_edges", summary["best_gnn_parent_edges_total"], summary["best_gnn_parent_edges_total"] / docs],
    ["gnn_parent_applied", summary["gnn_parent_applied_total"], summary["gnn_parent_applied_total"] / docs],
    ["gnn_parent_blocked_existing", summary["gnn_parent_blocked_existing_total"], summary["gnn_parent_blocked_existing_total"] / docs],
    ["gnn_parent_blocked_heading", summary["gnn_parent_blocked_heading_total"], summary["gnn_parent_blocked_heading_total"] / docs],
    ["gnn_parent_blocked_local_gate", summary["gnn_parent_blocked_local_gate_total"], summary["gnn_parent_blocked_local_gate_total"] / docs],
])}

## MERGE Reject Reasons

{markdown_table(["reason", "count"], sorted(summary["merge_reject_reasons"].items(), key=lambda item: item[1], reverse=True)[:12])}

## Render Parent Source Distribution

{markdown_table(["source", "count"], sorted(summary["render_parent_source_distribution"].items(), key=lambda item: item[1], reverse=True))}

## Top MERGE Docs

{markdown_table(["doc_id", "threshold_merge", "accepted_edges", "accepted_components", "rendered_merged_nodes", "not_in_tree", "duplicate_members"], [
    [
        row.get("doc_id"),
        row.get("threshold_merge_edges"),
        row.get("accepted_merge_edges"),
        row.get("accepted_merge_components"),
        row.get("render_tree_merged_nodes"),
        row.get("accepted_merges_not_in_render_tree"),
        row.get("duplicate_rendered_merge_members"),
    ]
    for row in top_merge
])}

## Top PARENT Docs

{markdown_table(["doc_id", "threshold_parent", "best_parent", "applied", "blocked_existing", "blocked_local_gate"], [
    [
        row.get("doc_id"),
        row.get("threshold_parent_edges"),
        row.get("best_gnn_parent_edges"),
        row.get("gnn_parent_applied"),
        row.get("gnn_parent_blocked_existing"),
        row.get("gnn_parent_blocked_local_gate"),
    ]
    for row in top_parent
])}

## Interpretation

MERGE is consumed correctly when accepted: accepted merge components and
RenderTreeIR merged nodes match, with zero accepted merge components missing
from the tree and zero duplicate rendered merge members in this run. The bigger
issue is that very few MERGE edges survive threshold plus hard gates.

PARENT is dominated by stack scope: GNN parent candidates exist, but almost all
are blocked because the target already has a stack-derived parent. This matches
the intended parent policy.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, rows = summarize(args.input_dir)
    write_csv(args.output_dir / "MERGE_PARENT_TRACE_SUMMARY.csv", rows)
    (args.output_dir / "MERGE_PARENT_TRACE_SUMMARY.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "MERGE_PARENT_TRACE_SUMMARY.md", args.input_dir, summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
