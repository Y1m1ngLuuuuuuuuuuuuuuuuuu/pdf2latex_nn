#!/usr/bin/env python3
"""Refresh graph candidate edges without recomputing node embeddings.

Use this after changing PDF-only graph topology rules.  The existing graph
already contains the expensive node tensor ``x`` with SciBERT features, so this
entrypoint only rebuilds ``edge_index`` / ``edge_attr`` from the current v7 JSON
and current ``graph_builder`` candidate-edge logic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.reading_order import filter_graph_content_items, fuse_micro_nodes  # noqa: E402
from src.perception.schema import EDGE_ATTR_FIELDS, SCIBERT_DIM  # noqa: E402
from src.reasoning.graph_builder import (  # noqa: E402
    build_candidate_edge_pairs,
    build_edge_attr_matrix,
    build_edge_index_from_pairs,
    build_scroll_layout,
    infer_column_ids,
    v7_reading_order_indices,
    _ranks_from_order,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--graph-output-dir", type=Path, required=True)
    parser.add_argument("--error-log", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--sequential-window", type=int, default=15)
    parser.add_argument("--spatial-k", type=int, default=3)
    parser.add_argument("--long-sight-window", type=int, default=40)
    parser.add_argument("--scope-anchor-window", type=int, default=160)
    parser.add_argument("--float-skip-window", type=int, default=40)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    records = load_manifest_records(args.input_manifest)
    if args.max_docs > 0:
        records = records[: args.max_docs]
    if not records:
        raise ValueError(f"No usable records found in {args.input_manifest}")

    args.graph_output_dir.mkdir(parents=True, exist_ok=True)
    if args.error_log:
        args.error_log.parent.mkdir(parents=True, exist_ok=True)
        args.error_log.write_text("", encoding="utf-8")

    jobs = [
        {
            "record": record,
            "graph_output_dir": str(args.graph_output_dir),
            "force": bool(args.force),
            "config": {
                "bidirectional": not args.directed,
                "sequential_window": int(args.sequential_window),
                "spatial_k": int(args.spatial_k),
                "long_sight_window": int(args.long_sight_window),
                "scope_anchor_window": int(args.scope_anchor_window),
                "float_skip_window": int(args.float_skip_window),
            },
        }
        for record in records
    ]

    start = time.time()
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    print(f"refresh_edges setup docs={len(jobs)} workers={args.workers} input={args.input_manifest}", flush=True)

    if args.workers <= 1:
        for idx, job in enumerate(jobs, start=1):
            result = refresh_one(job)
            handle_result(result, successes, failures, args.error_log)
            print_progress(idx, len(jobs), successes, failures, start)
            write_manifest(args.output_manifest, args.input_manifest, successes, failures, start)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(refresh_one, job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                handle_result(result, successes, failures, args.error_log)
                print_progress(idx, len(jobs), successes, failures, start)
                write_manifest(args.output_manifest, args.input_manifest, successes, failures, start)

    write_manifest(args.output_manifest, args.input_manifest, successes, failures, start)
    print(f"wrote manifest={args.output_manifest} documents={len(successes)} failed={len(failures)}", flush=True)
    return 0 if successes else 2


def refresh_one(job: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch

        record = dict(job["record"])
        doc_id = str(record.get("document_id") or record.get("id") or Path(str(record["graph_path"])).stem)
        output_graph = (Path(job["graph_output_dir"]) / f"{safe_filename(doc_id)}_v7_edge_refresh_graph.pt").resolve()
        if output_graph.exists() and not bool(job["force"]):
            return {"ok": True, "record": build_output_record(record, output_graph, reused=True)}

        graph = torch.load(Path(str(record["graph_path"])), map_location="cpu", weights_only=False)
        items = load_graph_items(Path(str(record["content_json"])), graph)
        if len(items) != int(graph.num_nodes):
            raise ValueError(f"content node count ({len(items)}) != graph.num_nodes ({int(graph.num_nodes)})")
        if not hasattr(graph, "x") or graph.x is None or int(graph.x.shape[1]) < SCIBERT_DIM:
            raise ValueError("existing graph.x does not contain SciBERT semantic features")

        regime_order = v7_reading_order_indices(items)
        regime_ranks = _ranks_from_order(regime_order, len(items))
        column_ids = infer_column_ids(items)
        scroll_layout = build_scroll_layout(items, reading_order_indices=regime_order, column_ids=column_ids)
        edge_pairs = build_candidate_edge_pairs(
            items,
            reading_order_indices=regime_order,
            column_ids=column_ids,
            **dict(job["config"]),
        )
        semantic = graph.x[:, :SCIBERT_DIM].detach().cpu().float()
        graph.edge_index = build_edge_index_from_pairs(edge_pairs)
        graph.edge_attr = build_edge_attr_matrix(
            items,
            semantic,
            edge_pairs=edge_pairs,
            reading_order_ranks=regime_ranks,
            scroll_layout=scroll_layout,
        )
        graph.edge_source_types = [source_type for _, _, source_type in edge_pairs]
        graph.edge_attr_schema = {
            "dim": len(EDGE_ATTR_FIELDS),
            "fields": EDGE_ATTR_FIELDS,
            "topology": {
                "strategy": "edge_refresh_current_graph_builder",
                **dict(job["config"]),
            },
        }
        clear_stale_labels(graph)
        graph.edge_refresh_source_graph = str(record["graph_path"])
        output_graph.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, output_graph)
        return {
            "ok": True,
            "record": build_output_record(
                record,
                output_graph,
                reused=False,
                shape={
                    "x": list(graph.x.shape),
                    "edge_index": list(graph.edge_index.shape),
                    "edge_attr": list(graph.edge_attr.shape),
                },
            ),
        }
    except Exception as exc:  # pragma: no cover - batch data path.
        return {
            "ok": False,
            "document_id": job.get("record", {}).get("document_id"),
            "graph_path": job.get("record", {}).get("graph_path"),
            "content_json": job.get("record", {}).get("content_json"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }


def load_graph_items(content_json: Path, graph: Any) -> list[dict[str, Any]]:
    payload = json.loads(content_json.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Expected items list in {content_json}")
    graph_items = filter_graph_content_items([item for item in items if isinstance(item, dict)])
    if bool(getattr(graph, "micro_fusion_applied", False)):
        graph_items = fuse_micro_nodes(graph_items)
    return graph_items


def clear_stale_labels(graph: Any) -> None:
    for key in (
        "y",
        "edge_label",
        "label_counts",
        "pdf_to_tex",
        "pdf_to_tex_scores",
        "alignment_quality",
        "alignment_schema",
        "candidate_edge_recall",
        "candidate_edge_missing",
        "candidate_edge_recall_report",
    ):
        try:
            del graph[key]
        except Exception:
            try:
                delattr(graph, key)
            except Exception:
                pass


def build_output_record(
    record: dict[str, Any],
    graph_path: Path,
    *,
    reused: bool,
    shape: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = dict(record)
    output["source_graph_path"] = record.get("graph_path")
    output["graph_path"] = str(graph_path)
    output["edge_refreshed_graph_reused"] = bool(reused)
    output["edge_attr_dim"] = len(EDGE_ATTR_FIELDS)
    if shape:
        output["graph_shape"] = shape
    return output


def handle_result(result: dict[str, Any], successes: list[dict[str, Any]], failures: list[dict[str, Any]], error_log: Path | None) -> None:
    if result.get("ok"):
        successes.append(result["record"])
        return
    failures.append(result)
    if error_log is not None:
        with error_log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


def print_progress(done: int, total: int, successes: list[dict[str, Any]], failures: list[dict[str, Any]], start: float) -> None:
    if done == 1 or done == total or done % 50 == 0:
        elapsed = max(1e-6, time.time() - start)
        rate = done / elapsed
        eta = (total - done) / max(rate, 1e-6)
        print(
            f"[refresh-edges] progress={done}/{total} ok={len(successes)} failed={len(failures)} "
            f"rate={rate:.2f}/s eta={eta/60:.1f}m",
            flush=True,
        )


def write_manifest(output_path: Path, source_manifest: Path, successes: list[dict[str, Any]], failures: list[dict[str, Any]], start: float) -> None:
    payload = {
        "schema_version": "v7_graph_edge_refresh_manifest_v1",
        "source_manifest": str(source_manifest),
        "num_documents": len(successes),
        "num_failed": len(failures),
        "elapsed_seconds": time.time() - start,
        "edge_attr_dim": len(EDGE_ATTR_FIELDS),
        "documents": sorted(successes, key=lambda record: str(record.get("document_id", ""))),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Expected manifest list or documents list: {path}")
    return [
        record
        for record in records
        if isinstance(record, dict) and record.get("graph_path") and record.get("content_json") and record.get("tex_path")
    ]


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
