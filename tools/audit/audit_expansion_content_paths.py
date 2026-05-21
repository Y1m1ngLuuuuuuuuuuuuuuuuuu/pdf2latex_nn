#!/usr/bin/env python3
"""Audit which v7 content JSON files an expansion run actually consumed."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.content_resolver import V7SchemaThresholds, validate_content_v7_schema  # noqa: E402


SUCCESS_RE = re.compile(r"\[staged\]\s+success\s+id=([^\s]+)")
SKIP_RE = re.compile(r"\[staged\]\s+skip\s+id=([^\s]+)\s+error=(.*)$")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--success-manifest", type=Path)
    parser.add_argument("--skip-log", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--mineru-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-pdf-dir", type=Path, default=PROJECT_ROOT / "data/01_raw_pdfs")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = audit_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "current_run_content_path_audit.json"
    csv_path = args.output_dir / "current_run_content_path_audit.csv"
    md_path = args.output_dir / "current_run_content_path_audit.md"
    json_path.write_text(json.dumps({"documents": rows, "summary": summarize(rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    md_path.write_text(markdown_report(rows), encoding="utf-8")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    return 0


def audit_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for doc_id, record in load_success_records(args.success_manifest).items():
        records[doc_id] = {**record, "document_id": doc_id, "status": "success", "skip_reason": None}
    for record in load_skip_records(args.skip_log):
        doc_id = str(record.get("document_id") or "")
        if not doc_id:
            continue
        records.setdefault(doc_id, {"document_id": doc_id})
        records[doc_id].update({"status": "skip", "skip_reason": record.get("error") or record.get("error_type"), **record})
    for doc_id, status, reason in load_log_status(args.run_log):
        records.setdefault(doc_id, {"document_id": doc_id})
        if records[doc_id].get("status") != "success":
            records[doc_id].update({"status": status, "skip_reason": reason or records[doc_id].get("skip_reason")})

    rows = []
    for doc_id, record in sorted(records.items()):
        raw_pdf = path_value(record.get("pdf_path")) or args.raw_pdf_dir / f"{doc_id}.pdf"
        content = path_value(record.get("content_json")) or default_content_path(doc_id, args.mineru_roots[0])
        metrics = validate_content_v7_schema(
            content,
            raw_pdf_path=raw_pdf,
            thresholds=V7SchemaThresholds(require_v7_schema_fields=True),
        )
        rows.append({
            "doc_id": doc_id,
            "status": record.get("status", "unknown"),
            "actual_content_json_path": str(content),
            "content_mtime": metrics.mtime,
            "content_parent_dir": content_root_label(content),
            "raw_pdf_path": str(raw_pdf),
            "content_page_count": metrics.content_page_count,
            "raw_pdf_page_count": metrics.raw_pdf_page_count,
            "page_count_match": metrics.page_count_match,
            "layout_layer_coverage": metrics.layout_layer_coverage,
            "layout_role_coverage": metrics.layout_role_coverage,
            "canonical_type_coverage": metrics.canonical_type_coverage,
            "style_spans_coverage": metrics.style_spans_coverage,
            "has_v7_node_ids": metrics.has_v7_node_ids,
            "stale_schema_flag": metrics.stale_schema_flag,
            "failed_reasons": "|".join(metrics.failed_reasons),
            "skip_reason": record.get("skip_reason"),
        })
    return rows


def load_success_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("documents", payload) if isinstance(payload, dict) else payload
    result = {}
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        source = record.get("source_record") if isinstance(record.get("source_record"), dict) else {}
        merged = {**source, **record}
        doc_id = str(merged.get("document_id") or "")
        if doc_id:
            result[doc_id] = merged
    return result


def load_skip_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def load_log_status(path: Path | None) -> list[tuple[str, str, str | None]]:
    if path is None or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if match := SUCCESS_RE.search(line):
            rows.append((match.group(1), "success", None))
        elif match := SKIP_RE.search(line):
            rows.append((match.group(1), "skip", match.group(2)))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dir[str(row["content_parent_dir"])].append(row)
    return {
        "processed_docs_count": len(rows),
        "old_mineru_output_count": sum(1 for row in rows if row["content_parent_dir"] == "mineru_output"),
        "newer_v7_dir_count": sum(1 for row in rows if row["content_parent_dir"] != "mineru_output"),
        "stale_schema_count": sum(1 for row in rows if row["stale_schema_flag"]),
        "by_content_parent_dir": {name: summarize_group(group) for name, group in sorted(by_dir.items())},
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = sum(1 for row in rows if row["status"] == "success")
    skip = sum(1 for row in rows if row["status"] != "success")
    return {
        "docs": len(rows),
        "success": success,
        "skip": skip,
        "pass_rate": success / max(1, len(rows)),
        "layout_layer_coverage": median(row["layout_layer_coverage"] for row in rows),
        "layout_role_coverage": median(row["layout_role_coverage"] for row in rows),
        "canonical_type_coverage": median(row["canonical_type_coverage"] for row in rows),
        "page_count_match_rate": sum(1 for row in rows if row["page_count_match"] is True) / max(1, len(rows)),
    }


def markdown_report(rows: list[dict[str, Any]]) -> str:
    summary = summarize(rows)
    lines = ["# Current Run Content Path Audit", ""]
    lines.append(f"processed docs count: {summary['processed_docs_count']}")
    lines.append(f"old mineru_output count: {summary['old_mineru_output_count']}")
    lines.append(f"newer v7 dir count: {summary['newer_v7_dir_count']}")
    lines.append(f"stale schema count: {summary['stale_schema_count']}")
    lines.append("")
    lines.append("| content_parent_dir | docs | success | skip | pass_rate | layout_layer | layout_role | canonical_type | page_count_match |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, data in summary["by_content_parent_dir"].items():
        lines.append(
            f"| {name} | {data['docs']} | {data['success']} | {data['skip']} | "
            f"{data['pass_rate']:.3f} | {data['layout_layer_coverage']:.3f} | "
            f"{data['layout_role_coverage']:.3f} | {data['canonical_type_coverage']:.3f} | "
            f"{data['page_count_match_rate']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def median(values: Any) -> float:
    vals = [float(value) for value in values if value is not None]
    return statistics.median(vals) if vals else 0.0


def default_content_path(doc_id: str, root: Path) -> Path:
    return root / doc_id / "auto" / f"{doc_id}_content_list_v7_styles.json"


def content_root_label(path: Path) -> str:
    parts = list(path.parts)
    if "02_mineru_outputs" in parts:
        idx = parts.index("02_mineru_outputs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def path_value(value: Any) -> Path | None:
    return Path(str(value)) if value else None


if __name__ == "__main__":
    raise SystemExit(main())
