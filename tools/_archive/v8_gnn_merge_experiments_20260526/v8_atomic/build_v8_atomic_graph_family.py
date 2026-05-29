#!/usr/bin/env python3
"""Build a v8 atomic MERGE PyG graph family from selected MinerU outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.v8_atomic_merge import build_v8_atomic_merge_payload
from src.reasoning.v8_atomic_graph_builder import (
    EDGE_ATTR_SCHEMA,
    NODE_FEATURE_SCHEMA,
    build_v8_atomic_pyg_data,
    save_v8_atomic_pyg_data,
    summarize_v8_atomic_graph_payload,
)
from src.reasoning.v8_atomic_labeler import build_v8_atomic_merge_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-manifest", required=True, help="selected200 JSON with content/TeX paths")
    parser.add_argument("--json-output-root", required=True, help="directory for per-doc v8 atomic JSON artifacts")
    parser.add_argument("--graph-output-root", required=True, help="directory for per-doc PyG graph .pt files")
    parser.add_argument("--report-dir", required=True, help="directory for audit summaries")
    parser.add_argument("--output-manifest", required=True, help="output graph-family manifest JSON")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--doc-ids", nargs="*")
    parser.add_argument("--candidate-window", type=int, default=4)
    parser.add_argument("--min-tex-alignment-confidence", type=float, default=0.55)
    parser.add_argument("--middle-block-source", choices=["preproc_blocks", "para_blocks"], default="preproc_blocks")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    selected_manifest = Path(args.selected_manifest)
    json_root = Path(args.json_output_root)
    graph_root = Path(args.graph_output_root)
    report_dir = Path(args.report_dir)
    output_manifest = Path(args.output_manifest)
    json_root.mkdir(parents=True, exist_ok=True)
    graph_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    items = _load_items(selected_manifest)
    if args.doc_ids:
        wanted = set(args.doc_ids)
        items = [item for item in items if str(item.get("doc_id")) in wanted]
    items = sorted(items, key=lambda item: str(item.get("doc_id") or ""))
    end = args.offset + args.limit if args.limit is not None else None
    items = items[args.offset:end]

    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, item in enumerate(items, 1):
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            errors.append({"doc_id": "", "error": "missing_doc_id"})
            continue
        doc_json_dir = json_root / doc_id
        graph_path = graph_root / f"{doc_id}_v8_atomic_merge_graph.pt"
        record_path = doc_json_dir / f"{doc_id}_v8_atomic_merge_record.json"
        if args.skip_existing and graph_path.exists() and record_path.exists():
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                records.append(record)
                if record.get("summary"):
                    summaries.append(record["summary"])
                print(f"[{idx}/{len(items)}] skip existing {doc_id}")
                continue
            except Exception:
                pass

        try:
            paths = _resolve_paths(item)
            doc_json_dir.mkdir(parents=True, exist_ok=True)
            graph_payload = build_v8_atomic_merge_payload(
                doc_id=doc_id,
                middle_json_path=paths["middle_json"],
                content_list_json_path=paths.get("content_list_json"),
                style_content_list_json_path=paths.get("style_content_list_json"),
                middle_block_source=args.middle_block_source,
                candidate_window=args.candidate_window,
            )
            label_payload = build_v8_atomic_merge_labels(
                graph_payload,
                source_tex_path=paths.get("source_tex"),
                min_tex_alignment_confidence=args.min_tex_alignment_confidence,
            )
            _write_json(doc_json_dir / f"{doc_id}_v8_atomic_merge_graph_view.json", graph_payload)
            _write_json(doc_json_dir / f"{doc_id}_v8_atomic_nodes.json", {"doc_id": doc_id, "nodes": graph_payload["nodes"]})
            _write_json(
                doc_json_dir / f"{doc_id}_v8_atomic_candidate_edges.json",
                {"doc_id": doc_id, "candidate_edges": graph_payload["candidate_edges"]},
            )
            _write_json(doc_json_dir / f"{doc_id}_v8_atomic_merge_labels.json", label_payload)

            data = build_v8_atomic_pyg_data(
                graph_payload,
                label_payload,
                source_graph_path=str(doc_json_dir / f"{doc_id}_v8_atomic_merge_graph_view.json"),
                source_label_path=str(doc_json_dir / f"{doc_id}_v8_atomic_merge_labels.json"),
            )
            save_v8_atomic_pyg_data(data, graph_path)
            summary = summarize_v8_atomic_graph_payload(graph_payload, label_payload)
            summary.update(
                {
                    "graph_path": str(graph_path),
                    "json_dir": str(doc_json_dir),
                    "middle_json": str(paths["middle_json"]),
                    "content_json": str(paths.get("style_content_list_json") or paths.get("content_list_json") or ""),
                    "source_tex": str(paths.get("source_tex") or ""),
                }
            )
            record = {
                "schema_version": "v8_atomic_merge_graph_record_v1",
                "doc_id": doc_id,
                "graph_path": str(graph_path),
                "json_dir": str(doc_json_dir),
                "summary": summary,
                "paths": {key: str(value) for key, value in paths.items() if value},
            }
            _write_json(record_path, record)
            records.append(record)
            summaries.append(summary)
            print(
                f"[{idx}/{len(items)}] built {doc_id} nodes={summary['node_count']} "
                f"edges={summary['candidate_edge_count']} train_merge={summary['trainable_merge_positive_count']}"
            )
        except Exception as exc:  # keep batch moving
            error = {"doc_id": doc_id, "error": type(exc).__name__, "message": str(exc)}
            errors.append(error)
            print(f"[{idx}/{len(items)}] ERROR {doc_id}: {type(exc).__name__}: {exc}")

    manifest = {
        "schema_version": "v8_atomic_merge_graph_family_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_manifest": str(selected_manifest),
        "json_output_root": str(json_root),
        "graph_output_root": str(graph_root),
        "node_feature_dim": len(NODE_FEATURE_SCHEMA),
        "edge_attr_dim": len(EDGE_ATTR_SCHEMA),
        "label_schema": {"0": "MERGE", "1": "PARENT_CHILD_UNUSED", "2": "NONE_OR_MASKED_UNKNOWN"},
        "items": records,
        "errors": errors,
        "summary": _aggregate_summaries(summaries, errors),
    }
    _write_json(output_manifest, manifest)
    _write_json(report_dir / "summary.json", manifest["summary"])
    _write_json(report_dir / "errors.json", {"errors": errors})
    _write_csv(report_dir / "summary.csv", summaries)
    _write_report(report_dir / "V8_ATOMIC_MERGE_SELECTED200_AUDIT_REPORT.md", manifest, summaries)


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("items") or payload.get("docs") or payload.get("records") or []
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError(f"{path} does not contain a list of items")
    return [item for item in items if isinstance(item, dict)]


def _resolve_paths(item: dict[str, Any]) -> dict[str, Path]:
    doc_id = str(item.get("doc_id") or "")
    style_path = _first_existing(
        item.get("content_json_path"),
        item.get("content_json"),
        item.get("style_content_list_json"),
        *[p for p in item.get("content_list_candidates", []) if str(p).endswith("_content_list_v7_styles.json")],
    )
    content_path = _first_existing(
        *[p for p in item.get("content_list_candidates", []) if str(p).endswith("_content_list.json")],
        item.get("content_list_json"),
        style_path,
    )
    middle_path = _infer_middle_path(style_path or content_path, doc_id)
    source_tex = _first_existing(item.get("tex_path"), item.get("main_tex"))
    if not middle_path or not middle_path.exists():
        raise FileNotFoundError(f"middle json not found for {doc_id}")
    return {
        "middle_json": middle_path,
        "content_list_json": content_path,
        "style_content_list_json": style_path,
        "source_tex": source_tex,
    }


def _first_existing(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(str(value))
        if path.exists():
            return path
    return None


def _infer_middle_path(anchor: Path | None, doc_id: str) -> Path | None:
    if anchor is None:
        return None
    parent = anchor.parent
    candidates = [
        parent / f"{doc_id}_middle.json",
        parent / anchor.name.replace("_content_list_v7_styles.json", "_middle.json"),
        parent / anchor.name.replace("_content_list_v7.json", "_middle.json"),
        parent / anchor.name.replace("_content_list.json", "_middle.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(parent.glob("*_middle.json"))
    return matches[0] if matches else None


def _aggregate_summaries(summaries: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    channel_counts = Counter()
    family_counts = Counter()
    label_counts = Counter()
    strength_counts = Counter()
    for summary in summaries:
        for key in [
            "node_count",
            "candidate_edge_count",
            "edge_label_count",
            "trainable_edge_count",
            "trainable_merge_positive_count",
            "trainable_none_negative_count",
        ]:
            totals[key] += int(summary.get(key) or 0)
        channel_counts.update(summary.get("channel_counts") or {})
        family_counts.update(summary.get("candidate_family_counts") or {})
        label_counts.update(summary.get("label_counts") or {})
        strength_counts.update(summary.get("label_strength_counts") or {})
    return {
        "processed_docs": len(summaries),
        "error_docs": len(errors),
        "node_feature_dim": len(NODE_FEATURE_SCHEMA),
        "edge_attr_dim": len(EDGE_ATTR_SCHEMA),
        **dict(totals),
        "channel_counts": dict(sorted(channel_counts.items())),
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "label_strength_counts": dict(sorted(strength_counts.items())),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "doc_id",
        "node_count",
        "candidate_edge_count",
        "trainable_edge_count",
        "trainable_merge_positive_count",
        "trainable_none_negative_count",
        "graph_path",
        "json_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in fields})


def _write_report(path: Path, manifest: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    summary = manifest["summary"]
    lines = [
        "# V8 Atomic MERGE Selected200 Audit",
        "",
        "## Status",
        f"- processed_docs: {summary['processed_docs']}",
        f"- error_docs: {summary['error_docs']}",
        f"- node_feature_dim: {summary['node_feature_dim']}",
        f"- edge_attr_dim: {summary['edge_attr_dim']}",
        "- training_started: No",
        "- mineru_rerun: No",
        "",
        "## Totals",
        f"- node_count: {summary.get('node_count', 0)}",
        f"- candidate_edge_count: {summary.get('candidate_edge_count', 0)}",
        f"- trainable_edge_count: {summary.get('trainable_edge_count', 0)}",
        f"- trainable_merge_positive_count: {summary.get('trainable_merge_positive_count', 0)}",
        f"- trainable_none_negative_count: {summary.get('trainable_none_negative_count', 0)}",
        "",
        "## Channel Counts",
    ]
    for key, value in (summary.get("channel_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Candidate Family Counts"])
    for key, value in (summary.get("candidate_family_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Label Counts"])
    for key, value in (summary.get("label_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "This is a graph-family audit only. It confirms that middle-derived atomic fragments can be converted into PyG graphs with MERGE/NONE supervision sidecars. PARENT_CHILD is intentionally unused in this graph family.",
            "",
            "## Artifacts",
            f"- graph_manifest: {manifest.get('output_manifest', '')}",
            f"- graph_output_root: {manifest.get('graph_output_root', '')}",
            f"- json_output_root: {manifest.get('json_output_root', '')}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
