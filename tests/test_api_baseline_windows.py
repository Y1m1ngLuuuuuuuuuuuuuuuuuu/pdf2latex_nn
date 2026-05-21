from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_TOOLS = PROJECT_ROOT / "tools" / "api_baselines"
sys.path.insert(0, str(API_TOOLS))

from build_multipage_windows import main as build_windows_main  # noqa: E402


def test_window_generation_with_overlap(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    page_root = tmp_path / "pages"
    page_dir = page_root / "doc_a"
    page_dir.mkdir(parents=True)
    pages = []
    for page in range(1, 8):
        image = page_dir / f"page_{page:04d}.png"
        image.write_bytes(b"png")
        pages.append({"page_index": page, "image_path": str(image), "width": 10, "height": 10})
    (page_dir / "pages.json").write_text(json.dumps({"doc_id": "doc_a", "pages": pages}), encoding="utf-8")
    manifest.write_text(json.dumps({"items": [{"doc_id": "doc_a", "pdf_path": "/tmp/doc_a.pdf"}]}), encoding="utf-8")
    output = tmp_path / "windows.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_multipage_windows.py",
            "--manifest",
            str(manifest),
            "--page-image-root",
            str(page_root),
            "--output",
            str(output),
            "--window-size",
            "4",
            "--overlap",
            "1",
        ],
    )
    assert build_windows_main() == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert [item["pages"] for item in data["items"]] == [[1, 2, 3, 4], [4, 5, 6, 7]]

