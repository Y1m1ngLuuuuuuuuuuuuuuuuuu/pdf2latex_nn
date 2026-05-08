#!/usr/bin/env python3
"""Rebuild v7 graph feature tensors for every document in a manifest.

This entrypoint is intentionally narrower than the staged dataset builder:
it does not run MinerU, does not rewrite v7 JSON, and does not label edges.
It only rebuilds ``content_v7_styles.json -> graph.pt`` with the current
feature schema, then writes a new manifest that can be passed to
``relabel_manifest.py``.
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

from src.perception.schema import FeatureTensorSchema  # noqa: E402
from src.perception.reading_order import refresh_content_v7_layout_metadata  # noqa: E402
from src.pipeline.v7_contract import assert_v7_content_json  # noqa: E402
from src.reasoning.graph_builder import GraphBuildConfig, build_graph_from_content_v7  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--graph-output-dir", type=Path, required=True)
    parser.add_argument("--content-output-dir", type=Path, help="Optional directory for refreshed v7_styles JSON copies")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models/huggingface/allenai/scibert_scivocab_uncased",
    )
    parser.add_argument("--error-log", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--embedding-device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequential-window", type=int, default=15)
    parser.add_argument("--spatial-k", type=int, default=3)
    parser.add_argument("--long-sight-window", type=int, default=40)
    parser.add_argument("--scope-anchor-window", type=int, default=160)
    parser.add_argument("--float-skip-window", type=int, default=40)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--fuse-micro-nodes", action="store_true")
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
            "content_output_dir": str(args.content_output_dir) if args.content_output_dir else None,
            "model_path": str(args.model_path),
            "force": bool(args.force),
            "config": {
                "max_length": args.max_length,
                "stride": args.stride,
                "batch_size": args.batch_size,
                "embedding_device": args.embedding_device,
                "bidirectional_edges": not args.directed,
                "sequential_window": args.sequential_window,
                "spatial_k": args.spatial_k,
                "long_sight_window": args.long_sight_window,
                "scope_anchor_window": args.scope_anchor_window,
                "float_skip_window": args.float_skip_window,
                "fuse_micro_nodes": bool(args.fuse_micro_nodes),
            },
        }
        for record in records
    ]

    start = time.time()
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    print(
        f"rebuild_graphs setup docs={len(jobs)} workers={args.workers} "
        f"node_dim={FeatureTensorSchema().node_feature_dim} edge_dim={FeatureTensorSchema().edge_attr_dim}",
        flush=True,
    )

    if args.workers <= 1:
        for idx, job in enumerate(jobs, start=1):
            result = rebuild_one(job)
            handle_result(result, successes, failures, args.error_log)
            print_progress(idx, len(jobs), successes, failures, start, args.output_manifest)
            write_manifest(args.output_manifest, args.input_manifest, successes, failures, start)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(rebuild_one, job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                handle_result(result, successes, failures, args.error_log)
                print_progress(idx, len(jobs), successes, failures, start, args.output_manifest)
                write_manifest(args.output_manifest, args.input_manifest, successes, failures, start)

    write_manifest(args.output_manifest, args.input_manifest, successes, failures, start)
    print(f"wrote manifest={args.output_manifest} documents={len(successes)} failed={len(failures)}", flush=True)
    return 0 if successes else 2


def rebuild_one(job: dict[str, Any]) -> dict[str, Any]:
    try:
        record = dict(job["record"])
        doc_id = str(record.get("document_id") or record.get("id") or Path(str(record["content_json"])).stem)
        content_json = refresh_content_json(record, doc_id=doc_id, job=job)
        output_graph = (Path(job["graph_output_dir"]) / f"{safe_filename(doc_id)}_v7_graph.pt").resolve()

        if output_graph.exists() and not bool(job["force"]):
            return {"ok": True, "record": build_output_record(record, output_graph, content_json=content_json, reused=True)}

        assert_v7_content_json(content_json, require_styles=True)
        cfg = GraphBuildConfig(model_path=Path(job["model_path"]), **dict(job["config"]))
        graph = build_graph_from_content_v7(content_json, output_graph, cfg)
        shape = {
            "x": list(graph.x.shape),
            "edge_index": list(graph.edge_index.shape),
            "edge_attr": list(graph.edge_attr.shape),
        }
        return {"ok": True, "record": build_output_record(record, output_graph, content_json=content_json, reused=False, shape=shape)}
    except Exception as exc:  # pragma: no cover - batch data path.
        return {
            "ok": False,
            "document_id": job.get("record", {}).get("document_id"),
            "content_json": job.get("record", {}).get("content_json"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }


def refresh_content_json(record: dict[str, Any], *, doc_id: str, job: dict[str, Any]) -> Path:
    source = Path(str(record["content_json"]))
    output_dir = job.get("content_output_dir")
    if not output_dir:
        return source
    output_path = Path(str(output_dir)) / doc_id / "auto" / f"{doc_id}_content_list_v7_styles.json"
    if output_path.exists() and not bool(job["force"]):
        assert_v7_content_json(output_path, require_styles=True)
        return output_path
    payload = json.loads(source.read_text(encoding="utf-8"))
    refreshed = refresh_content_v7_layout_metadata(payload)
    refreshed["source_content_json"] = str(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(refreshed, ensure_ascii=False, indent=2), encoding="utf-8")
    assert_v7_content_json(output_path, require_styles=True)
    return output_path


def build_output_record(
    record: dict[str, Any],
    graph_path: Path,
    *,
    content_json: Path,
    reused: bool,
    shape: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = dict(record)
    output["source_graph_path"] = record.get("graph_path")
    output["source_content_json"] = record.get("content_json")
    output["content_json"] = str(content_json)
    output["graph_path"] = str(graph_path)
    output["rebuilt_graph_reused"] = bool(reused)
    output["feature_schema_node_dim"] = FeatureTensorSchema().node_feature_dim
    output["edge_attr_dim"] = FeatureTensorSchema().edge_attr_dim
    if shape is not None:
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


def print_progress(
    done: int,
    total: int,
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    start: float,
    output_manifest: Path,
) -> None:
    if done == 1 or done == total or done % 10 == 0:
        elapsed = max(1e-6, time.time() - start)
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0.0
        print(
            f"[rebuild-graphs] progress={done}/{total} ok={len(successes)} failed={len(failures)} "
            f"rate={rate:.2f}/s eta={eta/60:.1f}m manifest={output_manifest}",
            flush=True,
        )


def write_manifest(
    output_path: Path,
    source_manifest: Path,
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    start: float,
) -> None:
    payload = {
        "schema_version": "v7_graph_rebuild_manifest_v1",
        "source_manifest": str(source_manifest),
        "num_documents": len(successes),
        "num_failed": len(failures),
        "elapsed_seconds": time.time() - start,
        "feature_schema_node_dim": FeatureTensorSchema().node_feature_dim,
        "edge_attr_dim": FeatureTensorSchema().edge_attr_dim,
        "documents": sorted(successes, key=lambda record: str(record.get("document_id", ""))),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Expected manifest list or documents list: {path}")
    cleaned = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if not record.get("content_json") or not record.get("tex_path"):
            continue
        cleaned.append(record)
    return cleaned


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
