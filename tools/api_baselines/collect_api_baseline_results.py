#!/usr/bin/env python3
"""Evaluate API comparison outputs and collect aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import traceback
from pathlib import Path
from typing import Any

from common import load_manifest_items, parse_doc_ids, resolve_path, safe_name, slice_items, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation.structure_metrics import evaluate_comparison_structures, load_comparison_json  # noqa: E402


METRIC_PATHS = {
    "heading_tree_accuracy": ("heading_tree_accuracy", "score"),
    "reading_order_accuracy": ("reading_order_accuracy", "score"),
    "paragraph_boundary_f1": ("paragraph_boundary_f1", "f1"),
    "paragraph_text_coverage_f1": ("paragraph_text_coverage_f1", "f1"),
    "section_attachment_f1": ("section_attachment_f1", "f1"),
    "section_attachment_body_no_float_f1": ("section_attachment_body_no_float_f1", "f1"),
    "reference_section_completeness": ("reference_section_completeness", "score"),
    "float_caption_attachment_accuracy": ("float_caption_attachment_accuracy", "score"),
    "generated_structure_validity": ("generated_structure_validity", "score"),
    "macro_structure_score": ("macro_structure_score",),
}


def get_metric(metrics: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = metrics
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return float(current) if isinstance(current, (int, float)) else None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold-comparison-dir", type=Path)
    parser.add_argument("--pred-comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-name", required=True)
    parser.add_argument("--source-format", default="latex")
    parser.add_argument("--try-compile", action="store_true")
    parser.add_argument("--pred-tex-dir", type=Path)
    parser.add_argument("--original-pdf-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--doc-ids")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    items = slice_items(load_manifest_items(args.manifest), offset=args.offset, limit=args.limit, doc_ids=parse_doc_ids(args.doc_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_doc_path = args.output_dir / "per_doc_metrics.jsonl"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with per_doc_path.open("w", encoding="utf-8") as fp:
        for item in items:
            doc_id = str(item["doc_id"])
            gold_path = None
            if args.gold_comparison_dir:
                gold_path = args.gold_comparison_dir / f"{safe_name(doc_id)}.json"
            if not gold_path or not gold_path.exists():
                gold_path = resolve_path(item.get("gold_comparison_path")) if item.get("gold_comparison_path") else None
            pred_path = args.pred_comparison_dir / f"{safe_name(doc_id)}.json"
            if not gold_path or not gold_path.exists() or not pred_path.exists():
                failures.append({"doc_id": doc_id, "error": "missing gold or pred comparison"})
                continue
            try:
                metrics = evaluate_comparison_structures(load_comparison_json(gold_path), load_comparison_json(pred_path))
                flat = {"doc_id": doc_id}
                for name, path in METRIC_PATHS.items():
                    flat[name] = get_metric(metrics, path)
                rows.append(flat)
                fp.write(json.dumps({"doc_id": doc_id, "metrics": metrics}, ensure_ascii=False) + "\n")
            except Exception as exc:
                failures.append({"doc_id": doc_id, "error": str(exc), "traceback": traceback.format_exc()})
    summary = {"system_name": args.system_name, "docs_requested": len(items), "docs_evaluated": len(rows), "failures": failures}
    for name in METRIC_PATHS:
        values = [row[name] for row in rows if isinstance(row.get(name), (int, float))]
        summary[name] = statistics.fmean(values) if values else None
    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["doc_id", *METRIC_PATHS.keys()])
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# API Baseline Report",
        "",
        f"- system: `{args.system_name}`",
        f"- source_format: `{args.source_format}`",
        f"- docs requested: {len(items)}",
        f"- docs evaluated: {len(rows)}",
        f"- failures: {len(failures)}",
        "",
        "## Mean Metrics",
        "",
        "| metric | mean |",
        "| --- | ---: |",
    ]
    for name in METRIC_PATHS:
        value = summary.get(name)
        report.append(f"| {name} | {value:.4f} |" if isinstance(value, float) else f"| {name} | n/a |")
    report += [
        "",
        "## Comparison Template",
        "",
        "| system | heading_tree | reading_order | paragraph_text_coverage | section_body_no_float | references | float_caption | validity | macro |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {args.system_name} | | | | | | | | |",
        "| ours | | | | | | | | |",
        "| nougat | | | | | | | | |",
    ]
    (args.output_dir / "API_BASELINE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir} evaluated={len(rows)} failures={len(failures)}")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())

