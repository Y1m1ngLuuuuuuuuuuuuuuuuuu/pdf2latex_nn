from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_TOOLS = PROJECT_ROOT / "tools" / "api_baselines"
sys.path.insert(0, str(API_TOOLS))

from stitch_multipage_outputs import main as stitch_main  # noqa: E402


def test_stitch_creates_document_output(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    windows = tmp_path / "windows.json"
    window_root = tmp_path / "window_outputs"
    doc_dir = window_root / "doc_a"
    doc_dir.mkdir(parents=True)
    (doc_dir / "doc_a_p0001_p0002.tex").write_text("\\section{A}\nText A\n", encoding="utf-8")
    (doc_dir / "doc_a_p0002_p0003.tex").write_text("\\section{B}\nText B\n", encoding="utf-8")
    manifest.write_text(json.dumps({"items": [{"doc_id": "doc_a"}]}), encoding="utf-8")
    windows.write_text(
        json.dumps(
            {
                "items": [
                    {"doc_id": "doc_a", "window_id": "doc_a_p0001_p0002", "pages": [1, 2]},
                    {"doc_id": "doc_a", "window_id": "doc_a_p0002_p0003", "pages": [2, 3]},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "doc_outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stitch_multipage_outputs.py",
            "--manifest",
            str(manifest),
            "--windows",
            str(windows),
            "--window-output-dir",
            str(window_root),
            "--output-dir",
            str(output),
            "--deduplicate-overlap",
            "--preserve-page-order",
        ],
    )
    assert stitch_main() == 0
    text = (output / "doc_a.tex").read_text(encoding="utf-8")
    assert "\\section{A}" in text
    assert "\\section{B}" in text
