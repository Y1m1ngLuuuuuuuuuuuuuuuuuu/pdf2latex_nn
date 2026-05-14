"""LaTeX compilation checks for generated outputs."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ENGINE_ORDER = ("latexmk", "xelatex", "pdflatex")


def compile_latex(
    tex_path: Path,
    *,
    output_dir: Path,
    engine: str = "auto",
    timeout: int = 120,
    passes: int = 2,
) -> dict[str, Any]:
    """Compile a LaTeX file and return a JSON-serializable report."""

    tex_path = tex_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_engine = resolve_engine(engine)
    if selected_engine is None:
        return {
            "success": False,
            "engine": engine,
            "output_pdf": None,
            "elapsed_sec": 0.0,
            "returncode": None,
            "error_summary": "No LaTeX engine found.",
            "log_tail": "",
        }

    start = time.time()
    command = build_compile_command(selected_engine, tex_path, output_dir)
    try:
        result = run_compile_commands(command, tex_path.parent, timeout=timeout, passes=passes if selected_engine != "latexmk" else 1)
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "engine": selected_engine,
            "command": command,
            "output_pdf": None,
            "elapsed_sec": round(time.time() - start, 3),
            "returncode": None,
            "error_summary": f"Compilation timed out after {timeout}s.",
            "log_tail": tail_text(exc.stdout or "", 120),
        }

    output_pdf = output_dir / f"{tex_path.stem}.pdf"
    success = result.returncode == 0 and output_pdf.exists()
    log_text = result.stdout or ""
    return {
        "success": success,
        "engine": selected_engine,
        "command": command,
        "output_pdf": str(output_pdf) if output_pdf.exists() else None,
        "elapsed_sec": round(time.time() - start, 3),
        "returncode": result.returncode,
        "error_summary": "" if success else summarize_latex_error(log_text),
        "log_tail": tail_text(log_text, 160),
    }


def resolve_engine(engine: str) -> str | None:
    if engine != "auto":
        return engine if shutil.which(engine) else None
    for candidate in DEFAULT_ENGINE_ORDER:
        if shutil.which(candidate):
            return candidate
    return None


def run_compile_commands(command: list[str], cwd: Path, *, timeout: int, passes: int) -> subprocess.CompletedProcess[str]:
    combined = ""
    last: subprocess.CompletedProcess[str] | None = None
    for _ in range(max(1, passes)):
        last = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        combined += last.stdout or ""
        if last.returncode != 0:
            break
    assert last is not None
    return subprocess.CompletedProcess(last.args, last.returncode, stdout=combined, stderr=None)


def build_compile_command(engine: str, tex_path: Path, output_dir: Path) -> list[str]:
    if engine == "latexmk":
        return [
            engine,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-outdir={output_dir}",
            str(tex_path),
        ]
    command = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={output_dir}",
        str(tex_path),
    ]
    return command


def summarize_latex_error(log_text: str) -> str:
    lines = str(log_text or "").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("!") or re.search(r":\d+:\s", line):
            return "\n".join(lines[index : index + 6])
    for pattern in ["Emergency stop", "Fatal error", "Undefined control sequence", "Missing $ inserted"]:
        for index, line in enumerate(lines):
            if pattern in line:
                return "\n".join(lines[index : index + 6])
    return tail_text(log_text, 30)


def tail_text(text: str, lines: int) -> str:
    values = str(text or "").splitlines()
    return "\n".join(values[-lines:])


def write_compile_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
