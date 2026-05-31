"""Small generation-side wrapper around the shared LaTeX compile evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.compile_eval import compile_latex, write_compile_report


def compile_tex_with_report(
    tex_path: Path,
    *,
    output_dir: Path,
    engine: str = "auto",
    timeout: int = 120,
    passes: int = 2,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Compile ``tex_path`` and optionally write the JSON compile report."""

    report = compile_latex(
        tex_path,
        output_dir=output_dir,
        engine=engine,
        timeout=timeout,
        passes=passes,
    )
    if report_path is not None:
        write_compile_report(report, report_path)
    return report
