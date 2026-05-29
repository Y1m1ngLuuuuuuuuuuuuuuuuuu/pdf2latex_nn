#!/usr/bin/env python3
"""Build v8 atomic merge JSON artifacts from MinerU middle output.

Outputs are intentionally split:

* nodes/edges JSON: model inputs, no merge truth leakage.
* labels JSON: optional truth sidecar from v8 deterministic merges and TeX.
* record JSON: reproducibility paths and summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.v8_atomic_merge import build_v8_atomic_merge_payload  # noqa: E402
from src.reasoning.v8_atomic_labeler import build_v8_atomic_merge_labels  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--middle-json", required=True, type=Path)
    parser.add_argument("--content-list-json", type=Path)
    parser.add_argument("--style-content-list-json", type=Path)
    parser.add_argument("--source-tex", type=Path, help="Optional TeX source for label sidecar only.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--middle-block-source", default="preproc_blocks", choices=("preproc_blocks", "para_blocks"))
    parser.add_argument("--candidate-window", type=int, default=4)
    parser.add_argument("--min-tex-alignment-confidence", type=float, default=0.55)
    parser.add_argument("--skip-labels", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_v8_atomic_merge_payload(
        doc_id=args.doc_id,
        middle_json_path=args.middle_json,
        content_list_json_path=args.content_list_json,
        style_content_list_json_path=args.style_content_list_json,
        middle_block_source=args.middle_block_source,
        candidate_window=args.candidate_window,
    )
    graph_view_path = args.output_dir / f"{args.doc_id}_v8_atomic_merge_graph_view.json"
    nodes_path = args.output_dir / f"{args.doc_id}_v8_atomic_nodes.json"
    edges_path = args.output_dir / f"{args.doc_id}_v8_atomic_candidate_edges.json"
    write_json(graph_view_path, payload)
    write_json(nodes_path, pick_graph_part(payload, "nodes"))
    write_json(edges_path, pick_graph_part(payload, "candidate_edges"))

    labels_path = None
    labels_payload: dict[str, Any] | None = None
    if not args.skip_labels:
        labels_payload = build_v8_atomic_merge_labels(
            payload,
            source_tex_path=args.source_tex,
            min_tex_alignment_confidence=args.min_tex_alignment_confidence,
        )
        labels_path = args.output_dir / f"{args.doc_id}_v8_atomic_merge_labels.json"
        write_json(labels_path, labels_payload)

    record = {
        "schema_version": "v8_atomic_merge_build_record_v1",
        "doc_id": args.doc_id,
        "inputs": {
            "middle_json": str(args.middle_json),
            "content_list_json": str(args.content_list_json) if args.content_list_json else None,
            "style_content_list_json": str(args.style_content_list_json) if args.style_content_list_json else None,
            "source_tex": str(args.source_tex) if args.source_tex else None,
        },
        "outputs": {
            "graph_view": str(graph_view_path),
            "nodes": str(nodes_path),
            "candidate_edges": str(edges_path),
            "labels": str(labels_path) if labels_path else None,
        },
        "graph_summary": payload.get("summary"),
        "label_summary": labels_payload.get("summary") if labels_payload else None,
    }
    record_path = args.output_dir / f"{args.doc_id}_v8_atomic_merge_record.json"
    write_json(record_path, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


def pick_graph_part(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "schema_version": f"v8_atomic_{key}_v1",
        "doc_id": payload.get("doc_id"),
        "source": payload.get("source"),
        "config": payload.get("config"),
        key: payload.get(key) or [],
        "summary": payload.get("summary"),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

