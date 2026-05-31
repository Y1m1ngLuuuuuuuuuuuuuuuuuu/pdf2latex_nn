"""Failure taxonomy helpers for canonical PDF2LaTeX E2E runs."""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


Stage = Literal[
    "input_discovery",
    "fact_layer",
    "document_ir",
    "render_tree_ir",
    "generation",
    "compile",
    "comparison_conversion",
    "structure_metrics",
    "visual_qa",
    "artifact_missing",
]

FailureType = Literal[
    "artifact_missing",
    "missing_original_pdf",
    "missing_mineru_output",
    "missing_observable_facts",
    "mineru_raw_missing",
    "observable_fact_build_error",
    "document_ir_build_error",
    "render_tree_build_error",
    "generated_tex_missing",
    "generation_error",
    "latex_compile_error",
    "unicode_or_special_char",
    "body_math_syntax_error",
    "missing_graphics_asset",
    "formula_renderer_gap",
    "float_caption_renderer_gap",
    "reference_renderer_gap",
    "table_renderer_gap",
    "algorithm_renderer_gap",
    "comparison_conversion_error",
    "gold_comparison_missing",
    "visual_render_unavailable",
    "historical_artifact_only",
    "unknown",
]

Severity = Literal["info", "warning", "recoverable", "blocking"]


@dataclass(frozen=True)
class E2EFailure:
    stage: Stage
    failure_type: FailureType
    severity: Severity
    message: str
    traceback_path: str | None = None
    recommended_next_action: str = "review_stage_output"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_failure(
    *,
    stage: Stage,
    failure_type: FailureType,
    severity: Severity,
    message: str,
    traceback_path: str | Path | None = None,
    recommended_next_action: str = "review_stage_output",
) -> E2EFailure:
    return E2EFailure(
        stage=stage,
        failure_type=failure_type,
        severity=severity,
        message=message,
        traceback_path=str(traceback_path) if traceback_path is not None else None,
        recommended_next_action=recommended_next_action,
    )


def exception_failure(
    exc: BaseException,
    *,
    stage: Stage,
    failure_type: FailureType = "unknown",
    severity: Severity = "blocking",
    traceback_path: str | Path | None = None,
    recommended_next_action: str = "inspect_traceback",
) -> E2EFailure:
    if traceback_path is not None:
        path = Path(traceback_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
    return make_failure(
        stage=stage,
        failure_type=failure_type,
        severity=severity,
        message=f"{type(exc).__name__}: {exc}",
        traceback_path=traceback_path,
        recommended_next_action=recommended_next_action,
    )


def classify_compile_failure(report: dict[str, Any]) -> FailureType:
    """Map a compile report into a stable failure type."""

    if bool(report.get("success")):
        return "unknown"
    raw_text = "\n".join(
        str(report.get(key) or "")
        for key in ("error_summary", "log_tail", "stdout", "stderr")
    ).casefold()
    text = re.sub(r"\s+", " ", raw_text)
    compact_text = re.sub(r"\s+", "", raw_text)
    if not text.strip():
        return "latex_compile_error"
    if any(marker in text for marker in ("unicode", "inputenc", "not set up for use with latex")):
        return "unicode_or_special_char"
    if any(marker in text for marker in ("missing $ inserted", "extra }, or forgotten $", "display math", "math mode")):
        return "body_math_syntax_error"
    if "undefinedcontrolsequence" in compact_text or "undefined control sequence" in text:
        return "latex_compile_error"
    if (
        ("file `" in text and any(marker in text for marker in ("not found", "draft setting")))
        or "cannot determine size of graphic" in text
        or ("pdftex.deferror" in compact_text and any(marker in text for marker in ("not found", "draft setting")))
    ):
        return "missing_graphics_asset"
    if any(marker in text for marker in ("tabular", "misplaced \\noalign", "extra alignment tab")):
        return "table_renderer_gap"
    if any(marker in text for marker in ("verbatim", "algorithmic", "\\begin{algorithm", "\\end{algorithm", "algorithm2e")):
        return "algorithm_renderer_gap"
    if any(marker in text for marker in ("citation", "bibitem", "bibliography", "natbib")):
        return "reference_renderer_gap"
    if re.search(r"undefined control sequence|emergency stop|fatal error", text):
        return "latex_compile_error"
    return "latex_compile_error"


def write_failure_taxonomy(
    output_path: str | Path,
    failures: list[E2EFailure],
    *,
    doc_id: str,
    status: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "pdf2latex_e2e_failure_taxonomy_v1",
        "doc_id": doc_id,
        "status": status,
        "failures": [failure.to_dict() for failure in failures],
        "failure_count": len(failures),
        "blocking_count": sum(1 for failure in failures if failure.severity == "blocking"),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
