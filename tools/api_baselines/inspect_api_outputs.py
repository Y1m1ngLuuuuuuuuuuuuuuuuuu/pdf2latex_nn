#!/usr/bin/env python3
"""Inspect API outputs for obvious generation problems."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from common import infer_source_format, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-format", choices=["latex", "markdown", "auto"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def repeated_ngram(text: str, n: int = 5) -> bool:
    tokens = re.findall(r"\w+", text.lower())
    grams = [" ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    if not grams:
        return False
    counts = Counter(grams)
    return counts.most_common(1)[0][1] >= 5


def inspect_file(path: Path, source_format: str) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    fmt = infer_source_format(path, source_format)
    begin_count = len(re.findall(r"\\begin\{([^}]+)\}", text))
    end_count = len(re.findall(r"\\end\{([^}]+)\}", text))
    return {
        "path": str(path),
        "source_format": fmt,
        "chars": len(text),
        "empty_output": not bool(text.strip()),
        "too_short_output": len(text.strip()) < 80,
        "repeated_ngram_degeneration": repeated_ngram(text),
        "contains_explanation_text": bool(re.search(r"\b(here is|i will|below is|sure,?)\b", text[:500], re.I)),
        "contains_page_noise": bool(re.search(r"\b(page\s+\d+|arxiv:|copyright)\b", text, re.I)),
        "invalid_encoding_replacement": "\ufffd" in text,
        "latex_command_count": len(re.findall(r"\\[A-Za-z]+", text)),
        "suspiciously_low_latex_command_count": fmt == "latex" and len(re.findall(r"\\[A-Za-z]+", text)) < 2,
        "rough_unbalanced_braces": text.count("{") != text.count("}"),
        "rough_unmatched_begin_end": begin_count != end_count,
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    files = sorted([p for p in args.output_dir.iterdir() if p.suffix.lower() in {".tex", ".md", ".mmd", ".markdown"}])
    results = [inspect_file(path, args.source_format) for path in files]
    summary = {"files": len(results)}
    for key in (
        "empty_output",
        "too_short_output",
        "repeated_ngram_degeneration",
        "contains_explanation_text",
        "contains_page_noise",
        "invalid_encoding_replacement",
        "suspiciously_low_latex_command_count",
        "rough_unbalanced_braces",
        "rough_unmatched_begin_end",
    ):
        summary[key] = sum(1 for row in results if row.get(key))
    write_json(args.output, {"schema_version": "api_output_inspection_v1", "summary": summary, "files": results})
    print(f"wrote {args.output} files={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

