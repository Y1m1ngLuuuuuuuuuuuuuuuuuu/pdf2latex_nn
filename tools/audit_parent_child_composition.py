#!/usr/bin/env python3
"""Audit what the 3-class PARENT_CHILD label is made of.

This is intentionally read-only.  It does not relabel graphs, mutate tensors, or
change the current MERGE / PARENT_CHILD / NONE training target.  The goal is to
understand whether PARENT_CHILD is dominated by section-style attachments,
float/caption relations, formula/list/table local relations, or unknown pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


LABEL_NAMES = {0: "MERGE", 1: "PARENT_CHILD", 2: "NONE"}
TEXT_TYPES = {"text", "paragraph", "reference"}
HEADING_TYPES = {"title", "heading", "section", "subsection", "subsubsection"}
FLOAT_TYPES = {"figure", "table", "algorithm"}
CAPTION_ROLES = {"caption", "figure_caption", "table_caption", "algorithm_caption"}
FORMULA_TYPES = {"equation", "formula", "display_math", "inline_math"}
LIST_TYPES = {"list", "list_item"}
PAGE_FURNITURE_TYPES = {"header", "footer", "page_header", "page_footer", "page_number"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest containing graph_path records.")
    parser.add_argument("--output", type=Path, required=True, help="JSON audit report path.")
    parser.add_argument("--csv-output", type=Path, help="Optional CSV summary path.")
    parser.add_argument("--max-docs", type=int, default=0, help="Optional document cap for smoke tests.")
    parser.add_argument("--max-examples", type=int, default=20, help="Examples retained per relation family.")
    parser.add_argument("--label", choices=["parent_child", "merge", "none", "all"], default="parent_child")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    records = load_manifest_records(args.manifest)
    if args.max_docs and args.max_docs > 0:
        records = records[: args.max_docs]
    if not records:
        raise ValueError(f"No graph records found in {args.manifest}")

    report = audit_records(records, args)
    write_json(args.output, report)
    if args.csv_output:
        write_csv(args.csv_output, report)
    print_summary(report)
    return 0


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Expected {path} to contain a list or documents list")
    cleaned = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("graph_path"):
            cleaned.append(record)
    return cleaned


def audit_records(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    label_filter = {"parent_child": {1}, "merge": {0}, "none": {2}, "all": {0, 1, 2}}[args.label]
    totals_by_label: Counter[str] = Counter()
    type_pair_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    source_family_counts: Counter[str] = Counter()
    layout_pair_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failed: list[dict[str, Any]] = []

    total_edges = 0
    docs_ok = 0
    docs_with_label = 0
    for idx, record in enumerate(records, start=1):
        graph_path = Path(str(record["graph_path"]))
        try:
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            y = normalized_labels(getattr(graph, "y", None))
            edge_index = getattr(graph, "edge_index", None)
            node_records = normalize_node_records(getattr(graph, "node_records", []))
            edge_sources = list(getattr(graph, "edge_source_types", []) or [])
            if y is None or edge_index is None or not node_records:
                raise ValueError("graph missing y, edge_index, or node_records")
            edge_count = int(edge_index.shape[1])
            total_edges += edge_count
            docs_ok += 1
            doc_has_label = False
            for edge_pos in range(edge_count):
                label = int(y[edge_pos])
                label_name = LABEL_NAMES.get(label, str(label))
                totals_by_label[label_name] += 1
                if label not in label_filter:
                    continue
                src_idx = int(edge_index[0, edge_pos])
                dst_idx = int(edge_index[1, edge_pos])
                if src_idx >= len(node_records) or dst_idx >= len(node_records):
                    continue
                src = node_records[src_idx]
                dst = node_records[dst_idx]
                src_type = canonical_record_type(src)
                dst_type = canonical_record_type(dst)
                type_pair = f"{src_type}->{dst_type}"
                family = classify_relation_family(src, dst)
                source_type = edge_sources[edge_pos] if edge_pos < len(edge_sources) else ""
                source_family = f"{source_type or 'unknown'}::{family}"
                layout_pair = f"{record_layout(src)}->{record_layout(dst)}"
                type_pair_counts[type_pair] += 1
                family_counts[family] += 1
                source_type_counts[source_type or "unknown"] += 1
                source_family_counts[source_family] += 1
                layout_pair_counts[layout_pair] += 1
                doc_has_label = True
                if len(examples[family]) < args.max_examples:
                    examples[family].append(
                        {
                            "document_id": record.get("document_id") or record.get("id") or graph_path.stem,
                            "graph_path": str(graph_path),
                            "edge_pos": edge_pos,
                            "label": label_name,
                            "edge_source_type": source_type,
                            "src_idx": src_idx,
                            "dst_idx": dst_idx,
                            "src_type": src_type,
                            "dst_type": dst_type,
                            "src_layout": record_layout(src),
                            "dst_layout": record_layout(dst),
                            "src_text": text_preview(src),
                            "dst_text": text_preview(dst),
                            "src_v7": src.get("_v7_node_id") or src.get("_v7_source_node_ids"),
                            "dst_v7": dst.get("_v7_node_id") or dst.get("_v7_source_node_ids"),
                        }
                    )
            if doc_has_label:
                docs_with_label += 1
        except Exception as exc:  # pragma: no cover - batch robustness.
            failed.append(
                {
                    "document_id": record.get("document_id") or record.get("id"),
                    "graph_path": str(graph_path),
                    "error": str(exc),
                }
            )
        if idx == 1 or idx == len(records) or idx % 250 == 0:
            print(f"[{idx}/{len(records)}] ok={docs_ok} failed={len(failed)}", flush=True)

    selected_edges = sum(type_pair_counts.values())
    return {
        "schema_version": "parent_child_composition_audit_v1",
        "manifest": str(args.manifest),
        "label_filter": args.label,
        "num_records": len(records),
        "docs_ok": docs_ok,
        "docs_failed": len(failed),
        "docs_with_selected_label": docs_with_label,
        "total_edges": total_edges,
        "selected_edges": selected_edges,
        "label_totals": dict(totals_by_label),
        "selected_type_pair_counts": counter_payload(type_pair_counts, selected_edges),
        "selected_family_counts": counter_payload(family_counts, selected_edges),
        "selected_edge_source_counts": counter_payload(source_type_counts, selected_edges),
        "selected_source_family_counts": counter_payload(source_family_counts, selected_edges),
        "selected_layout_pair_counts": counter_payload(layout_pair_counts, selected_edges),
        "examples": dict(examples),
        "failed": failed[:200],
    }


def normalized_labels(y: Any) -> Any:
    if y is None:
        return None
    import torch

    labels = y.detach().cpu().long()
    return torch.where(labels >= 2, torch.full_like(labels, 2), labels)


def normalize_node_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [record if isinstance(record, dict) else {} for record in value]
    return []


def canonical_record_type(record: dict[str, Any]) -> str:
    raw = str(
        record.get("canonical_type")
        or record.get("type")
        or record.get("raw_type")
        or record.get("block_type")
        or ""
    ).casefold()
    list_type = str(record.get("list_type") or "").casefold()
    role = str(record.get("layout_role") or "").casefold()
    layer = str(record.get("layout_layer") or "").casefold()
    if list_type == "reference_list" or raw in {"reference", "references", "bibliography"}:
        return "reference"
    if raw in {"paragraph", "text"}:
        if role in CAPTION_ROLES:
            return "caption"
        return "text"
    if raw in {"title", "heading"}:
        return "title"
    if raw in {"equation", "equation_interline", "interline_equation", "display_formula", "formula"}:
        return "equation"
    if raw in {"image", "chart"}:
        return "figure"
    if raw in {"table", "figure", "algorithm", "list", "code", "caption"}:
        return raw
    if layer in PAGE_FURNITURE_TYPES:
        return layer
    return raw or "unknown"


def classify_relation_family(src: dict[str, Any], dst: dict[str, Any]) -> str:
    src_type = canonical_record_type(src)
    dst_type = canonical_record_type(dst)
    src_role = record_role(src)
    dst_role = record_role(dst)
    if is_heading_like(src) and is_heading_like(dst):
        return "heading_to_heading"
    if is_heading_like(src) and is_body_like(dst):
        return "heading_to_body"
    if is_body_like(src) and is_heading_like(dst):
        return "body_to_heading"
    if is_caption_like(src) and dst_type in FLOAT_TYPES:
        return "caption_to_float"
    if src_type in FLOAT_TYPES and is_caption_like(dst):
        return "float_to_caption"
    if src_type in FLOAT_TYPES and dst_type in FLOAT_TYPES:
        return "float_to_float"
    if is_formula_like(src) and is_body_like(dst):
        return "formula_to_body"
    if is_body_like(src) and is_formula_like(dst):
        return "body_to_formula"
    if src_type in LIST_TYPES and dst_type in LIST_TYPES:
        return "list_to_list"
    if src_type == "table" and dst_type == "table":
        return "table_to_table"
    if src_type == "reference" and dst_type == "reference":
        return "reference_to_reference"
    if src_type in PAGE_FURNITURE_TYPES or dst_type in PAGE_FURNITURE_TYPES:
        return "page_furniture_relation"
    if src_type == dst_type:
        return f"same_{src_type}"
    if src_role or dst_role:
        return f"other_role:{src_type}/{src_role or '-'}->{dst_type}/{dst_role or '-'}"
    return f"other:{src_type}->{dst_type}"


def is_heading_like(record: dict[str, Any]) -> bool:
    typ = canonical_record_type(record)
    role = record_role(record)
    return typ in HEADING_TYPES or role in {"heading", "section_heading", "title"}


def is_caption_like(record: dict[str, Any]) -> bool:
    typ = canonical_record_type(record)
    role = record_role(record)
    text = text_preview(record).casefold()
    return typ == "caption" or role in CAPTION_ROLES or text.startswith(("figure ", "fig. ", "table ", "algorithm "))


def is_formula_like(record: dict[str, Any]) -> bool:
    return canonical_record_type(record) in FORMULA_TYPES


def is_body_like(record: dict[str, Any]) -> bool:
    typ = canonical_record_type(record)
    role = record_role(record)
    return typ in TEXT_TYPES or role in {"paragraph", "body", "main_text"}


def record_role(record: dict[str, Any]) -> str:
    return str(record.get("layout_role") or record.get("role") or "").casefold()


def record_layout(record: dict[str, Any]) -> str:
    layer = str(record.get("layout_layer") or "").casefold() or "-"
    role = record_role(record) or "-"
    return f"{layer}/{role}"


def text_preview(record: dict[str, Any], max_len: int = 160) -> str:
    text = str(record.get("text_preview") or record.get("text") or "")
    return " ".join(text.split())[:max_len]


def counter_payload(counter: Counter[str], total: int, *, top_n: int = 200) -> list[dict[str, Any]]:
    rows = []
    for key, count in counter.most_common(top_n):
        rows.append({"key": key, "count": int(count), "ratio": safe_ratio(count, total)})
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["group", "key", "count", "ratio"])
        writer.writeheader()
        for group in (
            "selected_family_counts",
            "selected_type_pair_counts",
            "selected_edge_source_counts",
            "selected_source_family_counts",
            "selected_layout_pair_counts",
        ):
            for row in report.get(group, []):
                writer.writerow({"group": group, **row})


def print_summary(report: dict[str, Any]) -> None:
    print(json.dumps(
        {
            "num_records": report["num_records"],
            "docs_ok": report["docs_ok"],
            "docs_failed": report["docs_failed"],
            "selected_edges": report["selected_edges"],
            "label_totals": report["label_totals"],
            "top_families": report["selected_family_counts"][:12],
            "top_type_pairs": report["selected_type_pair_counts"][:12],
        },
        ensure_ascii=False,
        indent=2,
    ))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


if __name__ == "__main__":
    raise SystemExit(main())
