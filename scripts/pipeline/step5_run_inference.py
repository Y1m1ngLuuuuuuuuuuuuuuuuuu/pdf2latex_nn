#!/usr/bin/env python3
"""Compatibility wrapper for the canonical step5 generator.

Historically this script duplicated inference and rendered through
``TreeDecoder.render_document``.  Production generation now lives in
``step5_generate_tex.py`` and defaults to the full-v7 IR renderer.  This wrapper
keeps the old command name alive while forwarding all work to the canonical
entrypoint.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_STEP5 = PROJECT_ROOT / "scripts" / "pipeline" / "step5_generate_tex.py"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="Input PyG graph .pt")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--content-json", type=Path, help="Full styled content_v7 JSON. Required for --renderer ir.")
    parser.add_argument("--source-pdf", type=Path, help="Optional source PDF used for figure/table crops")
    parser.add_argument("--source-tex", type=Path, help="Optional source TeX for citation/float sidecars")
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--asset-latex-prefix", default="assets")
    parser.add_argument("--merge-threshold", type=float)
    parser.add_argument("--parent-threshold", type=float)
    parser.add_argument("--threshold", type=float, help="Compatibility alias applied to both merge and parent thresholds")
    parser.add_argument("--renderer", choices=["ir"], default="ir")
    parser.add_argument(
        "--heading-skeleton-mode",
        choices=["stack"],
        default="stack",
        help="Canonical decoder mode. Only stack is supported.",
    )
    parser.add_argument("--render-table-crops", action="store_true")
    parser.add_argument("--logits-output", type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    threshold = 0.5 if args.threshold is None else float(args.threshold)
    merge_threshold = threshold if args.merge_threshold is None else float(args.merge_threshold)
    parent_threshold = threshold if args.parent_threshold is None else float(args.parent_threshold)

    command = [
        sys.executable,
        str(CANONICAL_STEP5),
        "--graph",
        str(args.graph),
        "--checkpoint",
        str(args.checkpoint),
        "--output-tex",
        str(args.output_tex),
        "--merge-threshold",
        str(merge_threshold),
        "--parent-threshold",
        str(parent_threshold),
        "--renderer",
        args.renderer,
        "--heading-skeleton-mode",
        args.heading_skeleton_mode,
        "--asset-latex-prefix",
        args.asset_latex_prefix,
    ]
    optional_paths = {
        "--content-json": args.content_json,
        "--source-pdf": args.source_pdf,
        "--source-tex": args.source_tex,
        "--asset-dir": args.asset_dir,
        "--logits-output": args.logits_output,
    }
    for flag, value in optional_paths.items():
        if value is not None:
            command.extend([flag, str(value)])
    if args.render_table_crops:
        command.append("--render-table-crops")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
