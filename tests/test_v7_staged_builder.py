from __future__ import annotations

import json
from pathlib import Path

from scripts.pipeline.build_mini_dataset import CandidateSample
from scripts.pipeline.build_v7_dataset_staged import (
    chunked,
    config_from_args,
    load_excluded_document_ids,
    prepare_pdf_batch_dir,
)


def test_chunked_keeps_order_and_last_partial() -> None:
    items = [
        CandidateSample(str(index), Path(f"{index}.pdf"), Path(f"tex/{index}"), Path(f"tex/{index}/main.tex"))
        for index in range(5)
    ]

    assert [[candidate.document_id for candidate in batch] for batch in chunked(items, 2)] == [
        ["0", "1"],
        ["2", "3"],
        ["4"],
    ]


def test_load_excluded_document_ids_reads_manifest_documents(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"documents": [{"document_id": "2501.00001"}, {"document_id": "2501.00002"}]}),
        encoding="utf-8",
    )

    assert load_excluded_document_ids((manifest, tmp_path / "missing.json")) == {"2501.00001", "2501.00002"}


def test_prepare_pdf_batch_dir_links_or_copies_named_pdfs(tmp_path: Path) -> None:
    source_pdf = tmp_path / "raw" / "2501.00001.pdf"
    source_pdf.parent.mkdir()
    source_pdf.write_bytes(b"%PDF")
    candidate = CandidateSample(
        document_id="2501.00001",
        pdf_path=source_pdf,
        tex_dir=tmp_path / "tex/2501.00001",
        main_tex_path=tmp_path / "tex/2501.00001/main.tex",
    )

    batch_dir = tmp_path / "batch"
    prepare_pdf_batch_dir([candidate], batch_dir)

    assert (batch_dir / "2501.00001.pdf").read_bytes() == b"%PDF"


def test_config_exposes_page_aware_mineru_batch_limit() -> None:
    parser_args = [
        "--target",
        "10",
        "--mineru-batch-size",
        "8",
        "--mineru-batch-max-pages",
        "128",
        "--dry-run",
    ]
    from scripts.pipeline.build_v7_dataset_staged import build_arg_parser

    config = config_from_args(build_arg_parser().parse_args(parser_args))

    assert config.mineru_batch_size == 8
    assert config.mineru_batch_max_pages == 128
