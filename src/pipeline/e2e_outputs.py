"""Canonical output layout helpers for PDF2LaTeX E2E runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STAGE_DIRS = {
    "input": "00_input",
    "facts": "01_facts",
    "ir": "02_ir",
    "generation": "03_generation",
    "compile": "04_compile",
    "comparison": "05_comparison",
    "visual_qa": "06_visual_qa",
    "failure": "07_failure",
}


@dataclass(frozen=True)
class E2EOutputLayout:
    root: Path

    def stage_dir(self, stage: str) -> Path:
        if stage not in STAGE_DIRS:
            raise KeyError(f"Unknown E2E stage {stage!r}")
        path = self.root / STAGE_DIRS[stage]
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def case_summary(self) -> Path:
        return self.root / "CASE_SUMMARY.md"


def ensure_e2e_layout(root: str | Path) -> E2EOutputLayout:
    layout = E2EOutputLayout(Path(root))
    layout.root.mkdir(parents=True, exist_ok=True)
    for stage in STAGE_DIRS:
        layout.stage_dir(stage)
    return layout


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_stage_skipped(stage_dir: str | Path, *, stage: str, reason: str, details: dict[str, Any] | None = None) -> Path:
    payload = {
        "schema_version": "pdf2latex_e2e_stage_skipped_v1",
        "stage": stage,
        "status": "skipped",
        "reason": reason,
        "details": details or {},
    }
    return write_json(Path(stage_dir) / "STAGE_SKIPPED.json", payload)


def copy_if_exists(source: str | Path | None, target: str | Path) -> bool:
    if source is None:
        return False
    src = Path(source)
    if not src.exists() or not src.is_file():
        return False
    dst = Path(target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return True
    shutil.copy2(src, dst)
    return True


def write_case_summary(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Case Summary: {summary.get('doc_id', 'unknown')}",
        "",
        f"- status: {summary.get('status', 'unknown')}",
        f"- stratum: {summary.get('stratum', 'unknown')}",
        f"- artifact_root: `{summary.get('artifact_root') or ''}`",
        "",
        "## Stages",
    ]
    for stage in summary.get("stages", []):
        lines.append(
            f"- {stage.get('stage')}: {stage.get('status')} "
            f"({stage.get('message') or stage.get('reason') or 'ok'})"
        )
    failures = summary.get("failures") or []
    lines.extend(["", "## Failures"])
    if failures:
        for failure in failures:
            lines.append(
                f"- {failure.get('severity')} / {failure.get('stage')} / "
                f"{failure.get('failure_type')}: {failure.get('message')}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Outputs"])
    for key, value in (summary.get("outputs") or {}).items():
        lines.append(f"- {key}: `{value}`")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataclass_to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value

