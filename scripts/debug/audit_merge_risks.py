#!/usr/bin/env python3
"""Audit decoded MERGE edges for long-distance and float-contamination risks.

This is an E2E safety probe, not a training script.  It loads a checkpoint and a
manifest, runs the same TreeDecoder MERGE contraction path used by generation,
and reports edges that are risky in production:

- cross-page or long-index-distance MERGE contractions
- MERGE contractions crossing figure/table/equation/caption/algorithm blocks
- endpoints that look like float/caption/table/noise layers rather than body text

The goal is to detect cases where a model with good F1 may still absorb table or
figure text into ordinary paragraphs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.batch_visual_qa_inference import load_model, resolve_device, select_documents  # noqa: E402
from scripts.pipeline.step5_generate_tex import load_node_records  # noqa: E402
from scripts.pipeline.train_edge_gnn_full import split_indices  # noqa: E402
from src.pipeline.v7_contract import assert_v7_content_json, assert_v7_graph_data  # noqa: E402
from src.reasoning.postprocess import ResolvedNode, TreeDecoder, TreeDecoderConfig, build_heading_skeleton  # noqa: E402


FLOATISH_TYPES = {
    "figure",
    "table",
    "caption",
    "figure_caption",
    "table_caption",
    "algorithm",
    "equation",
    "inline_math",
    "code",
}
FLOATISH_LAYER_TOKENS = {
    "float",
    "caption",
    "figure",
    "table",
    "algorithm",
    "equation",
    "math",
    "header_footer",
    "noise",
    "footnote",
    "margin_note",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--document-id", action="append", default=[])
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--parent-threshold", type=float, default=0.0)
    parser.add_argument("--require-merge-argmax", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-parent-argmax", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--long-index-delta", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
    docs = select_manifest_documents(args)
    device = resolve_device(args.device, torch=torch)
    model = load_model(args.checkpoint, device=device, torch=torch)
    decoder = TreeDecoder(
        TreeDecoderConfig(
            merge_threshold=args.merge_threshold,
            parent_threshold=args.parent_threshold,
            require_merge_argmax=args.require_merge_argmax,
            require_parent_argmax=args.require_parent_argmax,
        )
    )

    doc_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for doc in docs:
        row, risks = audit_one_document(doc, model=model, decoder=decoder, args=args, torch=torch, device=device)
        doc_rows.append(row)
        risk_rows.extend(risks)
        print(
            f"{row['document_id']} merges={row['accepted_merges']} "
            f"risks={row['risk_edges']} long={row['long_distance_merges']} "
            f"float_cross={row['crosses_float_merges']} non_text={row['non_text_endpoint_merges']}",
            flush=True,
        )

    payload = {
        "schema_version": "merge_risk_audit_v1",
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "thresholds": {
            "merge": args.merge_threshold,
            "parent": args.parent_threshold,
            "require_merge_argmax": args.require_merge_argmax,
            "require_parent_argmax": args.require_parent_argmax,
            "long_index_delta": args.long_index_delta,
        },
        "summary": summarize(doc_rows),
        "documents": doc_rows,
        "risk_edges": risk_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        write_csv(args.output_csv, risk_rows)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def select_manifest_documents(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    docs = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(docs, list):
        raise ValueError(f"Expected manifest list or documents list: {args.manifest}")
    docs = [doc for doc in docs if isinstance(doc, dict)]
    explicit = set(args.document_id or [])
    if explicit:
        selected = [doc for doc in docs if str(doc.get("document_id")) in explicit]
        found = {str(doc.get("document_id")) for doc in selected}
        missing = sorted(explicit - found)
        if missing:
            raise ValueError(f"Requested document ids not found: {missing}")
        return selected[: args.limit]
    if args.split == "all":
        split_docs = docs
    else:
        splits = split_indices(len(docs), args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)
        split_docs = [docs[index] for index in splits[args.split]]
    selected = []
    for doc in split_docs:
        if all(doc.get(key) and Path(str(doc[key])).exists() for key in ("pdf_path", "content_json", "graph_path")):
            selected.append(doc)
        if len(selected) >= args.limit:
            break
    return selected


def audit_one_document(
    doc: dict[str, Any],
    *,
    model: Any,
    decoder: TreeDecoder,
    args: argparse.Namespace,
    torch: Any,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    content_json = Path(str(doc["content_json"]))
    graph_path = Path(str(doc["graph_path"]))
    assert_v7_content_json(content_json, require_styles=True)
    data = torch.load(graph_path, map_location=device, weights_only=False)
    assert_v7_graph_data(data, graph_path)
    with torch.no_grad():
        logits = model(data.to(device)).detach().cpu()
    node_records = load_node_records(content_json, data)
    probs = decoder.edge_probabilities(logits)
    raw_skeleton = build_heading_skeleton(
        {
            index: ResolvedNode(node_id=index, record=dict(record), merged_node_ids=[index])
            for index, record in enumerate(node_records)
        }
    )
    contracted = decoder.contract_merge_nodes(node_records, data.edge_index.detach().cpu(), probs, raw_skeleton=raw_skeleton)

    risk_rows = []
    for edge in contracted.merge_edges:
        source_record = node_records[edge.source]
        target_record = node_records[edge.target]
        risk = classify_merge_risk(edge, node_records, source_record, target_record, args.long_index_delta)
        if risk["risk_flags"]:
            risk_rows.append(
                {
                    "document_id": str(doc.get("document_id", "")),
                    "source": edge.source,
                    "target": edge.target,
                    "score": edge.score,
                    **risk,
                    "source_text": snippet(text_of(source_record)),
                    "target_text": snippet(text_of(target_record)),
                }
            )

    row = {
        "document_id": str(doc.get("document_id", "")),
        "graph_path": str(graph_path),
        "content_json": str(content_json),
        "accepted_merges": len(contracted.merge_edges),
        "risk_edges": len(risk_rows),
        "long_distance_merges": sum("long_index_delta" in r["risk_flags"] or "cross_page" in r["risk_flags"] for r in risk_rows),
        "crosses_float_merges": sum("crosses_floatish_intermediate" in r["risk_flags"] for r in risk_rows),
        "non_text_endpoint_merges": sum("floatish_endpoint" in r["risk_flags"] for r in risk_rows),
    }
    return row, risk_rows


def classify_merge_risk(
    edge: Any,
    node_records: list[dict[str, Any]],
    source_record: dict[str, Any],
    target_record: dict[str, Any],
    long_index_delta: int,
) -> dict[str, Any]:
    flags: list[str] = []
    source_page = page_of(source_record)
    target_page = page_of(target_record)
    source_order = order_of(source_record, edge.source)
    target_order = order_of(target_record, edge.target)
    index_delta = abs(int(target_order) - int(source_order))
    page_delta = abs(int(target_page) - int(source_page))
    if page_delta > 0:
        flags.append("cross_page")
    if index_delta > int(long_index_delta):
        flags.append("long_index_delta")
    if is_floatish(source_record) or is_floatish(target_record):
        flags.append("floatish_endpoint")

    lo = min(int(source_order), int(target_order))
    hi = max(int(source_order), int(target_order))
    intermediate = [
        record
        for record in node_records
        if lo < int(order_of(record, -1)) < hi and is_floatish(record)
    ]
    if intermediate:
        flags.append("crosses_floatish_intermediate")
    return {
        "risk_flags": "|".join(sorted(set(flags))),
        "source_type": record_type(source_record),
        "target_type": record_type(target_record),
        "source_layer": str(source_record.get("layout_layer") or source_record.get("layout_role") or ""),
        "target_layer": str(target_record.get("layout_layer") or target_record.get("layout_role") or ""),
        "source_page": source_page,
        "target_page": target_page,
        "source_order": source_order,
        "target_order": target_order,
        "index_delta": index_delta,
        "page_delta": page_delta,
        "intermediate_floatish_count": len(intermediate),
        "intermediate_floatish_types": "|".join(sorted({record_type(record) for record in intermediate})),
    }


def record_type(record: dict[str, Any]) -> str:
    return str(record.get("canonical_type") or record.get("type") or record.get("category") or "").lower()


def page_of(record: dict[str, Any]) -> int:
    for key in ("page_idx", "page", "page_id"):
        if key in record:
            try:
                return int(record[key])
            except Exception:
                pass
    return 0


def order_of(record: dict[str, Any], fallback: int) -> int:
    for key in ("global_order", "index", "order", "original_index"):
        if key in record:
            try:
                return int(record[key])
            except Exception:
                pass
    return int(fallback)


def is_floatish(record: dict[str, Any]) -> bool:
    rtype = record_type(record)
    if rtype in FLOATISH_TYPES:
        return True
    layer = " ".join(str(record.get(key) or "").lower() for key in ("layout_layer", "layout_role", "role"))
    return any(token in layer for token in FLOATISH_LAYER_TOKENS)


def text_of(record: dict[str, Any]) -> str:
    return str(record.get("text_for_embedding") or record.get("text") or record.get("content") or record.get("latex") or "")


def snippet(text: str, limit: int = 180) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    docs = len(rows)
    merges = sum(int(row["accepted_merges"]) for row in rows)
    risks = sum(int(row["risk_edges"]) for row in rows)
    return {
        "documents": docs,
        "accepted_merges": merges,
        "risk_edges": risks,
        "risk_per_merge": safe_div(risks, merges),
        "docs_with_risk": sum(1 for row in rows if int(row["risk_edges"]) > 0),
        "long_distance_merges": sum(int(row["long_distance_merges"]) for row in rows),
        "crosses_float_merges": sum(int(row["crosses_float_merges"]) for row in rows),
        "non_text_endpoint_merges": sum(int(row["non_text_endpoint_merges"]) for row in rows),
    }


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "document_id",
        "source",
        "target",
        "score",
        "risk_flags",
        "source_type",
        "target_type",
        "source_layer",
        "target_layer",
        "source_page",
        "target_page",
        "source_order",
        "target_order",
        "index_delta",
        "page_delta",
        "intermediate_floatish_count",
        "intermediate_floatish_types",
        "source_text",
        "target_text",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    raise SystemExit(main())
