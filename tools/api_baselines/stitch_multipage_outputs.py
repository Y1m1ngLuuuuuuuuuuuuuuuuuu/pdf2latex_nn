#!/usr/bin/env python3
"""Stitch multi-page API window outputs into document-level files."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from common import load_manifest_items, parse_doc_ids, read_json, safe_name, slice_items, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--window-output-dir", type=Path)
    parser.add_argument("--full-output-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deduplicate-overlap", action="store_true")
    parser.add_argument("--preserve-page-order", action="store_true")
    parser.add_argument("--normalize-labels", action="store_true")
    parser.add_argument("--strip-state-comments", action="store_true", default=False)
    parser.add_argument("--source-format", choices=["latex", "markdown", "auto"], default="auto")
    return parser


def normalize_labels(text: str) -> tuple[str, list[str]]:
    rewrites: list[str] = []

    def repl(match: re.Match[str]) -> str:
        kind = match.group(1).lower()
        num = match.group(2)
        prefix = {"figure": "fig", "fig": "fig", "table": "tab", "equation": "eq", "eq": "eq"}.get(kind, kind)
        new = f"\\label{{{prefix}:{kind}_{num}}}"
        rewrites.append(match.group(0) + " -> " + new)
        return new

    return re.sub(r"\\label\{(figure|fig|table|equation|eq)[_:-]?(\d+)\}", repl, text, flags=re.I), rewrites


def strip_state(text: str) -> str:
    return re.sub(r"% STATE_BEGIN.*?% STATE_END", "", text, flags=re.S)


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest_items = {str(item["doc_id"]): item for item in load_manifest_items(args.manifest)}
    windows = read_json(args.windows).get("items") or []
    by_doc: dict[str, list[dict[str, object]]] = {}
    for window in windows:
        by_doc.setdefault(str(window["doc_id"]), []).append(window)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for doc_id in sorted(manifest_items):
        out_path = args.output_dir / f"{safe_name(doc_id)}.tex"
        report = {"doc_id": doc_id, "num_windows": 0, "duplicate_blocks_removed": 0, "label_rewrites": [], "warnings": []}
        if args.full_output_dir:
            full = args.full_output_dir / f"{safe_name(doc_id)}.tex"
            if full.exists():
                shutil.copy2(full, out_path)
                write_json(out_path.with_suffix(".stitch_report.json"), report | {"mode": "full_copy"})
                continue
        parts: list[str] = []
        seen: set[str] = set()
        for window in sorted(by_doc.get(doc_id, []), key=lambda w: (w.get("pages") or [0])[0]):
            window_id = str(window["window_id"])
            path = (args.window_output_dir or Path(".")) / safe_name(doc_id) / f"{safe_name(window_id)}.tex"
            if not path.exists():
                report["warnings"].append(f"missing window output {path}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if args.strip_state_comments:
                text = strip_state(text)
            if args.normalize_labels:
                text, rewrites = normalize_labels(text)
                report["label_rewrites"].extend(rewrites)
            if args.deduplicate_overlap:
                normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("%"))
                if normalized in seen:
                    report["duplicate_blocks_removed"] += 1
                    continue
                seen.add(normalized)
            parts.append(text.strip())
        out_path.write_text("\n\n".join(part for part in parts if part) + "\n", encoding="utf-8")
        report["num_windows"] = len(parts)
        report["overlap_policy"] = "exact_window_text_dedup" if args.deduplicate_overlap else "concat"
        write_json(out_path.with_suffix(".stitch_report.json"), report)
    print(f"stitched docs={len(manifest_items)} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

