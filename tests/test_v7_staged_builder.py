from __future__ import annotations

import json
from pathlib import Path

from scripts.pipeline.build_mini_dataset import CandidateSample, config_from_args as mini_config_from_args, build_arg_parser as build_mini_arg_parser, scan_candidates
from scripts.pipeline.build_v7_dataset_staged import (
    build_arg_parser,
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


def test_scan_candidates_uses_compile_manifest_pairs(tmp_path: Path) -> None:
    raw_pdf_dir = tmp_path / "data/01_raw_pdfs"
    source_dir = tmp_path / "data/03_tex_sources/2501.00001"
    wrong_pool = tmp_path / "data/03_tex_source_pool/2501.00001"
    report_dir = tmp_path / "data/09_eval_reports/arxiv_2025_source_pool_compile"
    raw_pdf_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    wrong_pool.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    pdf_path = raw_pdf_dir / "2501.00001.pdf"
    pdf_path.write_bytes(b"%PDF")
    (source_dir / "paper.tex").write_text("\\documentclass{article}\\begin{document}ok\\end{document}", encoding="utf-8")
    (wrong_pool / "main.tex").write_text("\\documentclass{article}\\begin{document}wrong\\end{document}", encoding="utf-8")
    accepted = report_dir / "accepted.jsonl"
    accepted.write_text(
        json.dumps(
            {
                "arxiv_id": "2501.00001",
                "status": "accepted",
                "pdf": str(pdf_path),
                "source_dir": str(source_dir),
                "main_tex": "paper.tex",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = mini_config_from_args(
        build_mini_arg_parser().parse_args(
            [
                "--project-root",
                str(tmp_path),
                "--raw-pdf-dir",
                str(raw_pdf_dir),
                "--tex-source-dir",
                str(wrong_pool.parent),
                "--compiled-accepted-manifest",
                str(accepted),
                "--no-auto-compiled-accepted-manifests",
                "--require-compiled-accepted",
            ]
        )
    )

    candidates = scan_candidates(config)

    assert len(candidates) == 1
    assert candidates[0].pdf_path == pdf_path.resolve()
    assert candidates[0].main_tex_path == (source_dir / "paper.tex").resolve()
    assert candidates[0].pdf_origin == "compiled_from_tex"


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
    config = config_from_args(build_arg_parser().parse_args(parser_args))

    assert config.mineru_batch_size == 8
    assert config.mineru_batch_max_pages == 128


def test_target_total_only_requests_missing_successes(tmp_path: Path) -> None:
    manifest = tmp_path / "done.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {"document_id": "2501.00001"},
                    {"document_id": "2501.00002"},
                    {"document_id": "2501.00003"},
                ]
            }
        ),
        encoding="utf-8",
    )

    config = config_from_args(
        build_arg_parser().parse_args(
            [
                "--target",
                "999",
                "--target-total",
                "5",
                "--exclude-manifest",
                str(manifest),
                "--dry-run",
            ]
        )
    )

    assert config.target_total == 5
    assert config.excluded_success_count == 3
    assert config.mini.target == 2


def test_target_total_can_be_already_satisfied(tmp_path: Path) -> None:
    manifest = tmp_path / "done.json"
    manifest.write_text(
        json.dumps({"documents": [{"document_id": "a"}, {"document_id": "b"}]}),
        encoding="utf-8",
    )

    config = config_from_args(
        build_arg_parser().parse_args(
            [
                "--target-total",
                "2",
                "--exclude-manifest",
                str(manifest),
                "--dry-run",
            ]
        )
    )

    assert config.mini.target == 0
