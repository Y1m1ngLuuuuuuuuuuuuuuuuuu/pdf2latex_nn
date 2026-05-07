from __future__ import annotations

from pathlib import Path

from tools.sync_compiled_sources_to_pool import build_direct_dir_index, build_pdf_index, find_missing_candidates, sync_one


def test_finds_only_compiled_sources_with_matching_pdf_and_missing_pool(tmp_path: Path) -> None:
    raw_pdf_dir = tmp_path / "data/01_raw_pdfs"
    compiled_dir = tmp_path / "data/03_tex_sources"
    pool_dir = tmp_path / "data/03_tex_source_pool"
    for path in [raw_pdf_dir, compiled_dir, pool_dir]:
        path.mkdir(parents=True)

    (raw_pdf_dir / "2501.00001.pdf").write_bytes(b"%PDF")
    (raw_pdf_dir / "2501.00002.pdf").write_bytes(b"%PDF")
    (compiled_dir / "2501.00001").mkdir()
    (compiled_dir / "2501.00001" / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    (compiled_dir / "2501.00002").mkdir()
    (compiled_dir / "2501.00002" / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    (compiled_dir / "2501.99999").mkdir()
    (compiled_dir / "2501.99999" / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    (pool_dir / "2501.00002").mkdir()

    candidates = find_missing_candidates(
        build_pdf_index(raw_pdf_dir),
        build_direct_dir_index(compiled_dir),
        build_direct_dir_index(pool_dir),
        pool_dir,
    )

    assert [candidate.document_id for candidate in candidates] == ["2501.00001"]


def test_sync_one_copies_tree_without_overwriting_existing_pool_entry(tmp_path: Path) -> None:
    raw_pdf_dir = tmp_path / "data/01_raw_pdfs"
    compiled_dir = tmp_path / "data/03_tex_sources"
    pool_dir = tmp_path / "data/03_tex_source_pool"
    for path in [raw_pdf_dir, compiled_dir, pool_dir]:
        path.mkdir(parents=True)
    (raw_pdf_dir / "2501.00001.pdf").write_bytes(b"%PDF")
    source_dir = compiled_dir / "2501.00001"
    source_dir.mkdir()
    (source_dir / "main.tex").write_text("hello", encoding="utf-8")

    candidate = find_missing_candidates(
        build_pdf_index(raw_pdf_dir),
        build_direct_dir_index(compiled_dir),
        build_direct_dir_index(pool_dir),
        pool_dir,
    )[0]
    stats = sync_one(candidate, "copy")

    assert (pool_dir / "2501.00001" / "main.tex").read_text(encoding="utf-8") == "hello"
    assert stats.files == 1

    (pool_dir / "2501.00001" / "main.tex").write_text("existing", encoding="utf-8")
    second_stats = sync_one(candidate, "copy")
    assert (pool_dir / "2501.00001" / "main.tex").read_text(encoding="utf-8") == "existing"
    assert second_stats.files == 0
