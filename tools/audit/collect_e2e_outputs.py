#!/usr/bin/env python3
"""Collect canonical PDF2LaTeX E2E case summaries into a rollup."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summaries = []
    for path in sorted(args.input_root.glob("*/case_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    rollup = build_rollup(summaries)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rollup, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [flatten_summary(summary) for summary in summaries]
        if rows:
            with args.output_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    print(f"collected {len(summaries)} cases")
    return 0


def build_rollup(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    failure_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for summary in summaries:
        status_counts[str(summary.get("status"))] += 1
        for failure in summary.get("failures") or []:
            failure_counts[str(failure.get("failure_type") or "unknown")] += 1
    return {
        "schema_version": "pdf2latex_e2e_collected_rollup_v1",
        "doc_count": len(summaries),
        "status_counts": dict(status_counts),
        "failure_type_counts": dict(failure_counts),
        "summaries": summaries,
    }


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": summary.get("doc_id"),
        "stratum": summary.get("stratum"),
        "status": summary.get("status"),
        "generated_tex": bool((summary.get("outputs") or {}).get("generated_tex")),
        "generated_pdf": bool((summary.get("outputs") or {}).get("generated_pdf")),
        "compile_success": summary.get("compile_success"),
        "comparison_metrics": summary.get("comparison_metrics"),
        "visual_qa_status": summary.get("visual_qa_status"),
        "main_failure_type": summary.get("main_failure_type"),
        "failure_count": len(summary.get("failures") or []),
    }


if __name__ == "__main__":
    raise SystemExit(main())

