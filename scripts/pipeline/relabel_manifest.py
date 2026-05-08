#!/usr/bin/env python3
"""Relabel an existing graph manifest with the current v7 AlignmentLabeler.

This is the Data-Delta test entrypoint: it deliberately does not rebuild
features or overwrite old baseline graphs.  It reads an existing strict
manifest, injects fresh labels into copies of those graphs, and writes a new
manifest plus a delta report so training results can be compared cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.label_generator import AlignmentLabeler, AlignmentLabelerConfig, AlignmentQualityError  # noqa: E402


LABEL_NAMES = {0: "merge", 1: "parent_child", 2: "none"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True, help="Existing strict manifest JSON")
    parser.add_argument("--output-manifest", type=Path, required=True, help="New manifest JSON to write")
    parser.add_argument("--graph-output-dir", type=Path, required=True, help="Directory for relabeled .pt graphs")
    parser.add_argument("--mapping-output-dir", type=Path, required=True, help="Directory for alignment mapping JSON files")
    parser.add_argument("--delta-report", type=Path, help="Optional aggregate label-delta report JSON")
    parser.add_argument("--error-log", type=Path, help="Optional JSONL skip/error log")
    parser.add_argument("--max-docs", type=int, default=0, help="Optional cap for smoke tests; 0 means all")
    parser.add_argument("--workers", type=int, default=1, help="Parallel relabel workers")
    parser.add_argument("--force", action="store_true", help="Regenerate even if output graph exists")
    parser.add_argument("--similarity-threshold", type=float, default=65.0)
    parser.add_argument("--max-orphan-ratio", type=float, default=0.30)
    parser.add_argument("--max-unmapped-tex-ratio", type=float, default=0.60)
    parser.add_argument("--max-isolated-node-ratio", type=float, default=0.90)
    parser.add_argument("--min-section-nodes", type=int, default=1)
    parser.add_argument(
        "--allow-bad-alignment",
        action="store_true",
        help="Keep graphs even when the current quality gates fail. Default skips them.",
    )
    parser.add_argument(
        "--profile-candidate-recall",
        action="store_true",
        help="Also run the oracle candidate-edge recall probe for each relabeled document.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_records = load_manifest_records(args.input_manifest)
    if args.max_docs and args.max_docs > 0:
        input_records = input_records[: args.max_docs]
    if not input_records:
        raise ValueError(f"No records found in {args.input_manifest}")

    args.graph_output_dir.mkdir(parents=True, exist_ok=True)
    args.mapping_output_dir.mkdir(parents=True, exist_ok=True)
    if args.error_log:
        args.error_log.parent.mkdir(parents=True, exist_ok=True)
        if args.error_log.exists():
            args.error_log.unlink()

    jobs = [
        {
            "record": record,
            "graph_output_dir": str(args.graph_output_dir),
            "mapping_output_dir": str(args.mapping_output_dir),
            "force": bool(args.force),
            "profile_candidate_recall": bool(args.profile_candidate_recall),
            "config": {
                "similarity_threshold": args.similarity_threshold,
                "max_orphan_ratio": args.max_orphan_ratio,
                "max_unmapped_tex_ratio": args.max_unmapped_tex_ratio,
                "max_isolated_node_ratio": args.max_isolated_node_ratio,
                "min_section_nodes": args.min_section_nodes,
                "abort_on_bad_alignment": not args.allow_bad_alignment,
            },
        }
        for record in input_records
    ]

    start = time.time()
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    print(
        f"relabel setup docs={len(jobs)} workers={args.workers} "
        f"input={args.input_manifest} output={args.output_manifest}"
    )

    if args.workers <= 1:
        for idx, job in enumerate(jobs, start=1):
            result = relabel_one(job)
            handle_result(result, successes, failures, args.error_log)
            print_progress(idx, len(jobs), successes, failures, start)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(relabel_one, job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                handle_result(result, successes, failures, args.error_log)
                print_progress(idx, len(jobs), successes, failures, start)

    successes.sort(key=lambda record: str(record.get("document_id", "")))
    manifest_payload = {
        "schema_version": "v7_relabel_manifest_v1",
        "source_manifest": str(args.input_manifest),
        "num_documents": len(successes),
        "num_failed": len(failures),
        "label_totals": aggregate_label_counts(successes, key="label_counts"),
        "old_label_totals": aggregate_label_counts(successes, key="old_label_counts"),
        "documents": successes,
    }
    write_json(args.output_manifest, manifest_payload)

    report_path = args.delta_report or args.output_manifest.with_suffix(".delta_report.json")
    report = build_delta_report(
        source_manifest=str(args.input_manifest),
        output_manifest=str(args.output_manifest),
        successes=successes,
        failures=failures,
        elapsed_seconds=time.time() - start,
    )
    write_json(report_path, report)
    print(f"wrote manifest={args.output_manifest}")
    print(f"wrote delta_report={report_path}")
    print_delta_summary(report)
    return 0 if successes else 2


def relabel_one(job: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch

        from src.datasets.document_dataset import DocumentDatasetConfig, sanitize_graph_data
        from tools.profile_candidate_edge_recall import profile_candidate_recall

        record = dict(job["record"])
        doc_id = str(record.get("document_id") or record.get("id") or Path(str(record.get("graph_path"))).stem)
        graph_path = Path(str(record["graph_path"]))
        content_json = Path(str(record["content_json"]))
        tex_path = Path(str(record["tex_path"]))
        output_graph = (Path(job["graph_output_dir"]) / f"{safe_filename(doc_id)}_v7_relabel_labeled_graph.pt").resolve()
        output_mapping = (Path(job["mapping_output_dir"]) / f"{safe_filename(doc_id)}_v7_relabel_mapping.json").resolve()

        if output_graph.exists() and not bool(job["force"]):
            graph = torch.load(output_graph, map_location="cpu", weights_only=False)
        else:
            config = AlignmentLabelerConfig(
                **dict(job["config"]),
                output_mapping_json=output_mapping,
            )
            labeler = AlignmentLabeler(
                content_json_path=content_json,
                tex_path=tex_path,
                graph_path=graph_path,
                config=config,
            )
            graph = labeler.run(output_graph_path=output_graph, overwrite=False)
            if bool(job.get("profile_candidate_recall")):
                recall_report = profile_candidate_recall(graph, labeler, max_examples=10)
                graph.candidate_edge_recall_report = recall_report
                graph.candidate_edge_recall = float(recall_report["overall"]["recall"])
                graph.candidate_edge_missing = int(recall_report["overall"]["missing_edges"])
                torch.save(graph, output_graph)

        graph = sanitize_graph_data(
            graph,
            config=DocumentDatasetConfig(root=Path(job["graph_output_dir"])),
            require_labels=True,
        )
        label_counts = graph_label_counts(graph.y)
        old_label_counts = normalize_manifest_label_counts(record.get("label_counts", {}))
        quality = getattr(graph, "alignment_quality", {}) if hasattr(graph, "alignment_quality") else {}

        out_record = {
            **record,
            "graph_path": str(output_graph),
            "source_graph_path": str(graph_path),
            "alignment_mapping": str(output_mapping) if output_mapping.exists() else record.get("alignment_mapping"),
            "source_alignment_mapping": record.get("alignment_mapping"),
            "label_counts": label_counts,
            "old_label_counts": old_label_counts,
            "orphan_ratio": float(quality.get("orphan_ratio", record.get("orphan_ratio", 0.0))),
            "old_orphan_ratio": record.get("orphan_ratio"),
            "alignment_quality": quality,
            "candidate_edge_recall": getattr(graph, "candidate_edge_recall", record.get("candidate_edge_recall", None)),
            "candidate_edge_missing": getattr(graph, "candidate_edge_missing", record.get("candidate_edge_missing", None)),
        }
        return {"ok": True, "record": out_record}
    except AlignmentQualityError as exc:
        return failure_payload(job, exc, kind="alignment_quality")
    except Exception as exc:  # pragma: no cover - exercised by batch data.
        return failure_payload(job, exc, kind=type(exc).__name__)


def handle_result(
    result: dict[str, Any],
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    error_log: Path | None,
) -> None:
    if result.get("ok"):
        successes.append(result["record"])
        return
    failures.append(result)
    if error_log is not None:
        with error_log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


def failure_payload(job: dict[str, Any], exc: Exception, *, kind: str) -> dict[str, Any]:
    record = dict(job.get("record", {}))
    return {
        "ok": False,
        "kind": kind,
        "document_id": record.get("document_id") or record.get("id"),
        "graph_path": record.get("graph_path"),
        "content_json": record.get("content_json"),
        "tex_path": record.get("tex_path"),
        "error": str(exc),
    }


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Expected {path} to contain a list or documents list")
    required = {"graph_path", "content_json", "tex_path"}
    cleaned = []
    for record in records:
        if not isinstance(record, dict):
            continue
        missing = [key for key in required if not record.get(key)]
        if missing:
            continue
        cleaned.append(record)
    return cleaned


def graph_label_counts(labels: Any) -> dict[str, int]:
    import torch

    y = labels.detach().cpu().long()
    y = torch.where(y >= 2, torch.full_like(y, 2), y)
    counts = torch.bincount(y, minlength=3).tolist()
    return {LABEL_NAMES[idx]: int(counts[idx]) for idx in range(3)}


def normalize_manifest_label_counts(value: Any) -> dict[str, int]:
    result = {name: 0 for name in LABEL_NAMES.values()}
    if not isinstance(value, dict):
        return result
    for key, count in value.items():
        try:
            label = int(key)
        except (TypeError, ValueError):
            label = next((idx for idx, name in LABEL_NAMES.items() if str(key) == name), 2)
        label = 2 if label >= 2 else label
        result[LABEL_NAMES[label]] += int(count)
    return result


def aggregate_label_counts(records: list[dict[str, Any]], *, key: str) -> dict[str, int]:
    totals = {name: 0 for name in LABEL_NAMES.values()}
    for record in records:
        counts = normalize_manifest_label_counts(record.get(key, {}))
        for name, count in counts.items():
            totals[name] += int(count)
    return totals


def build_delta_report(
    *,
    source_manifest: str,
    output_manifest: str,
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    old_totals = aggregate_label_counts(successes, key="old_label_counts")
    new_totals = aggregate_label_counts(successes, key="label_counts")
    deltas = {
        name: {
            "old": old_totals.get(name, 0),
            "new": new_totals.get(name, 0),
            "delta": new_totals.get(name, 0) - old_totals.get(name, 0),
            "ratio_new_over_old": safe_ratio(new_totals.get(name, 0), old_totals.get(name, 0)),
        }
        for name in LABEL_NAMES.values()
    }
    return {
        "schema_version": "v7_relabel_delta_report_v1",
        "source_manifest": source_manifest,
        "output_manifest": output_manifest,
        "num_success": len(successes),
        "num_failed": len(failures),
        "elapsed_seconds": elapsed_seconds,
        "old_label_totals": old_totals,
        "new_label_totals": new_totals,
        "delta": deltas,
        "failures": failures[:200],
    }


def print_progress(done: int, total: int, successes: list[dict[str, Any]], failures: list[dict[str, Any]], start: float) -> None:
    if done == 1 or done == total or done % 10 == 0:
        elapsed = max(1e-6, time.time() - start)
        rate = done / elapsed
        eta = (total - done) / max(rate, 1e-6)
        print(
            f"[{done}/{total}] ok={len(successes)} failed={len(failures)} "
            f"rate={rate:.2f}/s eta={eta/60:.1f}m",
            flush=True,
        )


def print_delta_summary(report: dict[str, Any]) -> None:
    print("label delta summary")
    for name, row in report["delta"].items():
        print(
            f"  {name}: old={row['old']} new={row['new']} "
            f"delta={row['delta']} ratio={row['ratio_new_over_old']:.4f}"
        )
    print(f"success={report['num_success']} failed={report['num_failed']} elapsed={report['elapsed_seconds'] / 60:.1f}m")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator == 0:
        return 0.0
    return float(numerator) / denominator


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
