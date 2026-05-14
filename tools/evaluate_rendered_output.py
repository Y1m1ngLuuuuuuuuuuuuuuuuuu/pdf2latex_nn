#!/usr/bin/env python3
"""Compile generated LaTeX if needed and compare rendered PDF layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.compile_eval import compile_latex  # noqa: E402
from src.evaluation.visual_qa import compare_pdf_layouts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-pdf", type=Path, required=True)
    parser.add_argument("--pred-tex", type=Path, help="Generated LaTeX to compile before layout comparison.")
    parser.add_argument("--pred-pdf", type=Path, help="Already-rendered prediction PDF.")
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/eval_rendered"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dpi", type=int, default=72)
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()

    if not args.pred_tex and not args.pred_pdf:
        parser.error("Provide either --pred-tex or --pred-pdf.")

    compile_report = None
    pred_pdf = args.pred_pdf
    if args.pred_tex:
        compile_report = compile_latex(args.pred_tex, output_dir=args.work_dir / "compile", engine=args.engine, timeout=args.timeout)
        pred_pdf = Path(compile_report["output_pdf"]) if compile_report.get("output_pdf") else None

    layout_report = None
    if pred_pdf and Path(pred_pdf).exists():
        layout_report = compare_pdf_layouts(args.gold_pdf, Path(pred_pdf), dpi=args.dpi, max_pages=args.max_pages)

    report = {
        "schema_version": "rendered_output_eval_v1",
        "gold_pdf": str(args.gold_pdf),
        "pred_tex": str(args.pred_tex) if args.pred_tex else None,
        "pred_pdf": str(pred_pdf) if pred_pdf else None,
        "latex_compile_success": compile_report,
        "page_layout_similarity": layout_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
