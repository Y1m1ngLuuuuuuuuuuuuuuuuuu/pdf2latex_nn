#!/usr/bin/env python3
"""Find alternative v7 content JSON candidates for processed expansion docs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.content_resolver import (  # noqa: E402
    V7SchemaThresholds,
    enumerate_v7_content_candidates,
    score_candidate,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-audit-json", type=Path, required=True)
    parser.add_argument("--mineru-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = load_audit_rows(args.content_audit_json)
    payload = []
    stale_with_valid_alt = 0
    top_dirs = Counter()
    for row in rows:
        doc_id = str(row["doc_id"])
        raw_pdf = Path(str(row["raw_pdf_path"])) if row.get("raw_pdf_path") else None
        actual = str(row.get("actual_content_json_path") or "")
        candidates = [
            score_candidate(path, raw_pdf_path=raw_pdf, thresholds=V7SchemaThresholds())
            for path in enumerate_v7_content_candidates(doc_id, tuple(args.mineru_roots))
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        valid = [candidate for candidate in candidates if not candidate.failed_reasons]
        has_valid_alt = any(candidate.path != actual for candidate in valid)
        if row.get("stale_schema_flag") and has_valid_alt:
            stale_with_valid_alt += 1
        if valid:
            top_dirs[content_root_label(Path(valid[0].path))] += 1
        payload.append(
            {
                "doc_id": doc_id,
                "status": row.get("status"),
                "actual_content_json_path": actual,
                "actual_stale_schema_flag": row.get("stale_schema_flag"),
                "has_valid_alternative": has_valid_alt,
                "best_candidate_path": valid[0].path if valid else None,
                "candidates": [candidate.to_dict() for candidate in candidates],
            }
        )
    summary = {
        "processed_docs": len(rows),
        "stale_docs_with_newer_valid_content": stale_with_valid_alt,
        "docs_with_any_valid_candidate": sum(1 for item in payload if item["best_candidate_path"]),
        "top_alternative_dirs": dict(top_dirs.most_common(20)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "alternative_v7_content_candidates.json").write_text(
        json.dumps({"summary": summary, "documents": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "alternative_v7_content_summary.md").write_text(markdown(summary, payload), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_audit_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("documents", payload) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)]


def markdown(summary: dict[str, Any], payload: list[dict[str, Any]]) -> str:
    lines = ["# Alternative V7 Content Summary", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    examples = [item for item in payload if item.get("has_valid_alternative")][:10]
    lines.extend(["", "## Examples", "", "| doc_id | actual | best |", "|---|---|---|"])
    for item in examples:
        lines.append(f"| {item['doc_id']} | {item.get('actual_content_json_path')} | {item.get('best_candidate_path')} |")
    return "\n".join(lines) + "\n"


def content_root_label(path: Path) -> str:
    parts = list(path.parts)
    if "02_mineru_outputs" in parts:
        idx = parts.index("02_mineru_outputs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


if __name__ == "__main__":
    raise SystemExit(main())
