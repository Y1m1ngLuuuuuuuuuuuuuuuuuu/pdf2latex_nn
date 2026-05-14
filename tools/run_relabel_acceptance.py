#!/usr/bin/env python3
"""Run post-relabel acceptance checks and generate dangerous-case visuals."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--rebuild-errors", type=Path)
    parser.add_argument("--label-errors", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("debug_output") / "relabel_acceptance")
    parser.add_argument("--top-covers", type=int, default=5)
    parser.add_argument("--top-full", type=int, default=2)
    parser.add_argument("--zoom", type=float, default=1.6)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_json = args.output_dir / f"{args.tag}_audit.json"
    dangerous_json = args.output_dir / f"{args.tag}_dangerous.json"

    run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "audit_labeled_manifest.py"),
            "--manifest",
            str(args.manifest),
            "--output-json",
            str(audit_json),
            "--dangerous-json",
            str(dangerous_json),
            "--top-k",
            str(max(args.top_covers, args.top_full, 30)),
        ]
        + optional_path_args("--rebuild-errors", args.rebuild_errors)
        + optional_path_args("--label-errors", args.label_errors)
    )

    dangerous = json.loads(dangerous_json.read_text(encoding="utf-8"))
    visual_records: list[dict[str, Any]] = []
    for rank, record in enumerate(dangerous[: args.top_covers], start=1):
        visual_records.append(render_visual(record, args, rank=rank, max_pages=1, suffix="cover"))
    for rank, record in enumerate(dangerous[: args.top_full], start=1):
        visual_records.append(render_visual(record, args, rank=rank, max_pages=0, suffix="full"))

    payload = {
        "schema_version": "relabel_acceptance_v1",
        "manifest": str(args.manifest),
        "audit_json": str(audit_json),
        "dangerous_json": str(dangerous_json),
        "visuals": visual_records,
    }
    report_path = args.output_dir / f"{args.tag}_acceptance.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote_acceptance={report_path}")
    for item in visual_records:
        print(f"visual {item['document_id']} {item['mode']} -> {item['output_dir']}")
    return 0


def render_visual(record: dict[str, Any], args: argparse.Namespace, *, rank: int, max_pages: int, suffix: str) -> dict[str, Any]:
    doc_id = str(record["document_id"])
    output_dir = args.output_dir / f"{args.tag}_{suffix}_{rank:02d}_{safe_name(doc_id)}"
    pdf_path = require_path(record, "pdf_path")
    content_json = require_path(record, "content_json")
    graph_path = require_path(record, "graph_path")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "visualize_graph_labels.py"),
        "--pdf",
        str(pdf_path),
        "--content-json",
        str(content_json),
        "--graph",
        str(graph_path),
        "--output-dir",
        str(output_dir),
        "--prefix",
        safe_name(doc_id),
        "--zoom",
        str(args.zoom),
        "--draw-cross-page",
    ]
    if max_pages > 0:
        command.extend(["--max-pages", str(max_pages)])
    run(command)
    return {
        "document_id": doc_id,
        "mode": suffix,
        "rank": rank,
        "risk_score": record.get("risk_score"),
        "output_dir": str(output_dir),
        "pdf_path": str(pdf_path),
        "content_json": str(content_json),
        "graph_path": str(graph_path),
        "tex_path": record.get("tex_path"),
        "alignment_mapping": record.get("alignment_mapping"),
    }


def require_path(record: dict[str, Any], key: str) -> Path:
    value = record.get(key)
    if not value:
        raise ValueError(f"Dangerous record {record.get('document_id')} lacks {key}")
    return Path(str(value))


def optional_path_args(flag: str, path: Path | None) -> list[str]:
    if path is None:
        return []
    return [flag, str(path)]


def run(command: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def safe_name(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "doc"


if __name__ == "__main__":
    raise SystemExit(main())
