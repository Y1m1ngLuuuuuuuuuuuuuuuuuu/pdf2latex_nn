#!/usr/bin/env python3
"""Audit a v7 labeled manifest and select high-risk documents for QA.

The script is intentionally lightweight: it only reads manifest JSON records
and optional error JSONL files.  It does not load graph tensors, so it is safe
to run immediately after relabeling before expensive visual inspection.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


LABEL_KEYS = ("merge", "parent_child", "none")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rebuild-errors", type=Path)
    parser.add_argument("--label-errors", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dangerous-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=30)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    records = load_records(args.manifest)
    if not records:
        raise ValueError(f"No document records found in {args.manifest}")

    summary = summarize(records)
    summary["manifest"] = str(args.manifest)
    summary["rebuild_error_count"] = count_jsonl(args.rebuild_errors)
    summary["label_error_count"] = count_jsonl(args.label_errors)

    dangerous = sorted(
        (risk_record(record) for record in records),
        key=lambda item: item["risk_score"],
        reverse=True,
    )[: max(1, args.top_k)]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.dangerous_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.dangerous_json.write_text(json.dumps(dangerous, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(printable_summary(summary), ensure_ascii=False, indent=2))
    print(f"wrote_summary={args.output_json}")
    print(f"wrote_dangerous={args.dangerous_json}")
    print("top_dangerous=")
    for item in dangerous[:10]:
        print(
            f"  {item['document_id']} score={item['risk_score']:.3f} "
            f"orphan={item['orphan_ratio']:.3f} unmapped={item['unmapped_tex_ratio']:.3f} "
            f"recall={fmt_float(item['candidate_edge_recall'])} "
            f"missing={item['candidate_edge_missing']} labels={item['label_counts']}"
        )
    return 0


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list or a documents list")
    return [dict(record) for record in records if isinstance(record, dict)]


def count_jsonl(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    label_totals = Counter()
    orphan_values: list[float] = []
    raw_orphan_values: list[float] = []
    unmapped_values: list[float] = []
    raw_unmapped_values: list[float] = []
    recall_values: list[float] = []
    missing_values: list[int] = []
    isolated_values: list[float] = []
    metadata_orphans = 0
    float_unmapped = 0
    bib_unmapped = 0
    visual_exempt = 0

    for record in records:
        counts = label_counts(record)
        label_totals.update(counts)
        quality = quality_dict(record)
        orphan_values.append(float(record.get("orphan_ratio", quality.get("orphan_ratio", 0.0)) or 0.0))
        raw_orphan_values.append(float(quality.get("raw_orphan_ratio", orphan_values[-1]) or 0.0))
        unmapped_values.append(float(quality.get("unmapped_tex_ratio", 0.0) or 0.0))
        raw_unmapped_values.append(float(quality.get("raw_unmapped_tex_ratio", quality.get("unmapped_tex_ratio", 0.0)) or 0.0))
        if record.get("candidate_edge_recall") is not None:
            recall_values.append(float(record["candidate_edge_recall"]))
        if record.get("candidate_edge_missing") is not None:
            missing_values.append(int(record["candidate_edge_missing"]))
        if quality.get("isolated_node_ratio") is not None:
            isolated_values.append(float(quality["isolated_node_ratio"]))
        metadata_orphans += int(quality.get("metadata_orphan_count", 0) or 0)
        float_unmapped += int(quality.get("unmapped_float_tex_count", 0) or 0)
        bib_unmapped += int(quality.get("unmapped_bibliography_tex_count", 0) or 0)
        visual_exempt += int(quality.get("expected_visual_orphan_exempt_count", 0) or 0)

    positives = label_totals["merge"] + label_totals["parent_child"]
    total_edges = positives + label_totals["none"]
    return {
        "schema_version": "labeled_manifest_audit_v1",
        "num_documents": len(records),
        "label_totals": dict(label_totals),
        "label_ratios": {key: safe_div(label_totals[key], total_edges) for key in LABEL_KEYS},
        "positive_edge_ratio": safe_div(positives, total_edges),
        "orphan_ratio": describe(orphan_values),
        "raw_orphan_ratio": describe(raw_orphan_values),
        "unmapped_tex_ratio": describe(unmapped_values),
        "raw_unmapped_tex_ratio": describe(raw_unmapped_values),
        "candidate_edge_recall": describe(recall_values),
        "candidate_edge_missing": describe(missing_values),
        "isolated_node_ratio": describe(isolated_values),
        "metadata_orphan_count": metadata_orphans,
        "unmapped_float_tex_count": float_unmapped,
        "unmapped_bibliography_tex_count": bib_unmapped,
        "expected_visual_orphan_exempt_count": visual_exempt,
    }


def risk_record(record: dict[str, Any]) -> dict[str, Any]:
    quality = quality_dict(record)
    counts = label_counts(record)
    recall = nullable_float(record.get("candidate_edge_recall"))
    missing = int(record.get("candidate_edge_missing") or 0)
    orphan = float(record.get("orphan_ratio", quality.get("orphan_ratio", 0.0)) or 0.0)
    raw_orphan = float(quality.get("raw_orphan_ratio", orphan) or 0.0)
    unmapped = float(quality.get("unmapped_tex_ratio", 0.0) or 0.0)
    raw_unmapped = float(quality.get("raw_unmapped_tex_ratio", unmapped) or 0.0)
    isolated = float(quality.get("isolated_node_ratio", 0.0) or 0.0)
    positives = counts["merge"] + counts["parent_child"]
    total = positives + counts["none"]
    positive_ratio = safe_div(positives, total)
    no_merge_penalty = 0.12 if counts["merge"] == 0 else 0.0
    low_parent_penalty = 0.12 if counts["parent_child"] < 2 else 0.0
    recall_penalty = 0.0 if recall is None else max(0.0, 1.0 - recall)
    score = (
        2.5 * recall_penalty
        + 0.35 * min(missing, 20) / 20.0
        + 1.2 * orphan
        + 0.45 * raw_orphan
        + 1.2 * unmapped
        + 0.35 * raw_unmapped
        + 0.4 * isolated
        + no_merge_penalty
        + low_parent_penalty
        + (0.1 if positive_ratio < 0.005 else 0.0)
    )
    return {
        "document_id": str(record.get("document_id") or record.get("id") or Path(str(record.get("graph_path", ""))).stem),
        "risk_score": round(float(score), 8),
        "label_counts": counts,
        "orphan_ratio": orphan,
        "raw_orphan_ratio": raw_orphan,
        "unmapped_tex_ratio": unmapped,
        "raw_unmapped_tex_ratio": raw_unmapped,
        "isolated_node_ratio": isolated,
        "candidate_edge_recall": recall,
        "candidate_edge_missing": missing,
        "pdf_path": record.get("pdf_path"),
        "content_json": record.get("content_json"),
        "graph_path": record.get("graph_path"),
        "tex_path": record.get("tex_path"),
        "alignment_mapping": record.get("alignment_mapping"),
    }


def label_counts(record: dict[str, Any]) -> dict[str, int]:
    raw = record.get("label_counts", {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "merge": int(raw.get("merge", raw.get("0", 0)) or 0),
        "parent_child": int(raw.get("parent_child", raw.get("1", 0)) or 0),
        "none": int(raw.get("none", raw.get("2", 0)) or 0),
    }


def quality_dict(record: dict[str, Any]) -> dict[str, Any]:
    quality = record.get("alignment_quality", {})
    return quality if isinstance(quality, dict) else {}


def nullable_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except Exception:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def describe(values: list[float] | list[int]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values if value is not None and not math.isnan(float(value)))
    if not clean:
        return {"count": 0, "min": None, "p25": None, "mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(clean),
        "min": clean[0],
        "p25": percentile(clean, 0.25),
        "mean": mean(clean),
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "max": clean[-1],
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def printable_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_documents": summary["num_documents"],
        "label_totals": summary["label_totals"],
        "label_ratios": summary["label_ratios"],
        "orphan_ratio_p95": summary["orphan_ratio"]["p95"],
        "unmapped_tex_ratio_p95": summary["unmapped_tex_ratio"]["p95"],
        "candidate_edge_recall_min": summary["candidate_edge_recall"]["min"],
        "candidate_edge_recall_p50": summary["candidate_edge_recall"]["p50"],
        "rebuild_error_count": summary.get("rebuild_error_count"),
        "label_error_count": summary.get("label_error_count"),
    }


def fmt_float(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
