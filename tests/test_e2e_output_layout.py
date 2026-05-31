from __future__ import annotations

import json

from src.pipeline.e2e_outputs import STAGE_DIRS, ensure_e2e_layout, write_stage_skipped


def test_output_directory_layout_is_stable(tmp_path):
    layout = ensure_e2e_layout(tmp_path / "case")

    for stage_dir in STAGE_DIRS.values():
        assert (layout.root / stage_dir).is_dir()
    assert layout.case_summary.name == "CASE_SUMMARY.md"


def test_stage_skipped_writes_json(tmp_path):
    path = write_stage_skipped(tmp_path, stage="visual_qa", reason="tool_unavailable")
    payload = json.loads(path.read_text())

    assert payload["schema_version"] == "pdf2latex_e2e_stage_skipped_v1"
    assert payload["stage"] == "visual_qa"
    assert payload["status"] == "skipped"
    assert payload["reason"] == "tool_unavailable"

