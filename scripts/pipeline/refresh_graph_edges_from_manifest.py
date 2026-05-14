#!/usr/bin/env python3
"""Refresh graph order-dependent features and candidate edges.

Use this after changing PDF-only graph topology rules.  The existing graph
already contains the expensive node tensor ``x`` with SciBERT features.  This
entrypoint therefore preserves semantic embeddings and rebuilds only
order-dependent layout features, ``edge_index`` and ``edge_attr`` from the
current v7 JSON and current ``graph_builder`` candidate-edge logic.
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
    build_derived_stats_matrix,
    build_candidate_edge_pairs,
    build_edge_attr_matrix,
    build_edge_index_from_pairs,
    build_message_edge_mask,
    build_flow_context_matrix,
    build_scroll_geometry_matrix,
    build_scroll_layout,
    build_sequence_position_matrix,
    infer_column_ids,
    layout_layer_name,
    make_node_records,
    original_index_reading_order_indices,
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
    parser.add_argument(
        "--reading-order-source",
        choices=("v7", "original_index"),
        default="v7",
        help=(
            "Which one-dimensional order should drive scroll/index features and candidate edges. "
            "'original_index' ignores repaired layout_flow_order for Raw-MinerU-Flow ablations."
        ),
    )
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
                "reading_order_source": str(args.reading_order_source),
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

        reading_order_source = str(job["config"].pop("reading_order_source", "v7"))
        regime_order = select_reading_order(items, source=reading_order_source)
        regime_ranks = _ranks_from_order(regime_order, len(items))
        column_ids = infer_column_ids(items)
        feature_items = (
            with_original_index_flow_metadata(items, regime_order, column_ids)
            if reading_order_source == "original_index"
            else items
        )
        scroll_layout = build_scroll_layout(items, reading_order_indices=regime_order, column_ids=column_ids)
        edge_pairs = build_candidate_edge_pairs(
            feature_items,
            reading_order_indices=regime_order,
            column_ids=column_ids,
            **dict(job["config"]),
        )
        semantic = graph.x[:, :SCIBERT_DIM].detach().cpu().float()
        graph.x = refresh_order_dependent_node_features(
            graph,
            feature_items,
            regime_order=regime_order,
            regime_ranks=regime_ranks,
            column_ids=column_ids,
            scroll_layout=scroll_layout,
            reading_order_source=reading_order_source,
        )
        graph.edge_index = build_edge_index_from_pairs(edge_pairs)
        graph.edge_attr = build_edge_attr_matrix(
            feature_items,
            semantic,
            edge_pairs=edge_pairs,
            reading_order_ranks=regime_ranks,
            scroll_layout=scroll_layout,
        )
        graph.message_edge_mask = build_message_edge_mask(feature_items, edge_pairs=edge_pairs)
        graph.edge_source_types = [source_type for _, _, source_type in edge_pairs]
        graph.edge_attr_schema = {
            "dim": len(EDGE_ATTR_FIELDS),
            "fields": EDGE_ATTR_FIELDS,
            "topology": {
                "strategy": "edge_refresh_current_graph_builder",
                "reading_order_source": reading_order_source,
                **dict(job["config"]),
            },
        }
        graph.node_records = make_node_records(
            feature_items,
            column_ids=column_ids,
            reading_order_ranks=regime_ranks,
            scroll_layout=scroll_layout,
        )
        graph.reading_order_source = reading_order_source
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
                reading_order_source=reading_order_source,
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


def select_reading_order(items: list[dict[str, Any]], *, source: str) -> list[int]:
    if source == "v7":
        return v7_reading_order_indices(items)
    if source == "original_index":
        return original_index_reading_order_indices(items)
    raise ValueError(f"Unsupported reading order source: {source}")


def with_original_index_flow_metadata(
    items: list[dict[str, Any]],
    order: list[int],
    column_ids: list[int],
) -> list[dict[str, Any]]:
    """Return item copies whose flow metadata is derived from raw MinerU order.

    The Raw-MinerU-Flow ablation should not leak repaired v7 band/order values
    through ``flow_context`` or edge layout-flow attributes.  We keep coarse
    layout-layer/type probes, but replace band-local fields with page-local raw
    order fields.
    """

    output = [dict(item) for item in items]
    page_counts: dict[int, int] = {}
    page_seen: dict[int, int] = {}
    page_order: dict[int, int] = {}
    for item_idx in order:
        page_idx = int_or_default(output[item_idx].get("page_idx"), 0)
        if page_idx not in page_order:
            page_order[page_idx] = len(page_order)
        page_counts[page_idx] = page_counts.get(page_idx, 0) + 1
    for rank, item_idx in enumerate(order):
        item = output[item_idx]
        page_idx = int_or_default(item.get("page_idx"), 0)
        local = page_seen.get(page_idx, 0)
        page_seen[page_idx] = local + 1
        column_id = column_ids[item_idx] if item_idx < len(column_ids) else 2
        item["layout_flow_order"] = rank
        item["layout_band_id"] = f"raw_page_{page_idx}"
        item["layout_band_global_id"] = page_idx
        item["layout_band_global_order"] = page_order[page_idx]
        item["layout_band_type"] = "raw_mineru_page"
        item["layout_band_column_id"] = column_id
        item["layout_band_column"] = {0: "left", 1: "right"}.get(column_id, "full")
        item["layout_band_local_order"] = local
        item["layout_is_band_boundary"] = local == 0
        item["is_main_flow_candidate"] = bool(item.get("is_main_flow_candidate", layout_layer_name(item) == "main_text_flow"))
    return output


def refresh_order_dependent_node_features(
    graph: Any,
    items: list[dict[str, Any]],
    *,
    regime_order: list[int],
    regime_ranks: list[int],
    column_ids: list[int],
    scroll_layout: Any,
    reading_order_source: str,
) -> Any:
    """Rewrite node feature slices that encode reading flow."""

    import torch

    x = graph.x.detach().cpu().float().clone()
    schema = getattr(graph, "feature_schema", None)
    if not isinstance(schema, dict):
        raise ValueError("graph.feature_schema is required to refresh order-dependent features")

    replace_feature_slice(
        x,
        schema,
        "scroll_geometry",
        build_scroll_geometry_matrix(items, scroll_layout=scroll_layout),
    )
    replace_feature_slice(
        x,
        schema,
        "derived_stats",
        build_derived_stats_matrix(items, reading_order_ranks=regime_ranks),
    )
    replace_feature_slice(
        x,
        schema,
        "sequence_position",
        build_sequence_position_matrix(items, reading_order_ranks=regime_ranks),
    )
    replace_feature_slice(
        x,
        schema,
        "flow_context",
        build_flow_context_matrix(items),
    )
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def replace_feature_slice(x: Any, schema: dict[str, Any], group_name: str, values: Any) -> None:
    entry = schema.get(group_name)
    if not isinstance(entry, dict):
        raise ValueError(f"feature_schema lacks group {group_name!r}")
    start = int(entry["start"])
    end = int(entry["end"])
    if int(values.shape[0]) != int(x.shape[0]) or int(values.shape[1]) != end - start:
        raise ValueError(
            f"bad refreshed feature shape for {group_name}: {tuple(values.shape)} expected {(int(x.shape[0]), end - start)}"
        )
    x[:, start:end] = values.to(dtype=x.dtype)


def int_or_default(value: Any, default: int) -> int:
    return value if isinstance(value, int) else default


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
    reading_order_source: str | None = None,
) -> dict[str, Any]:
    output = dict(record)
    output["source_graph_path"] = record.get("graph_path")
    output["graph_path"] = str(graph_path)
    output["edge_refreshed_graph_reused"] = bool(reused)
    output["edge_attr_dim"] = len(EDGE_ATTR_FIELDS)
    if reading_order_source is not None:
        output["reading_order_source"] = reading_order_source
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
