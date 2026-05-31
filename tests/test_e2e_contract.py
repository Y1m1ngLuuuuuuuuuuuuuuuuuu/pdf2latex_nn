from __future__ import annotations

import json

from scripts.pipeline.run_pdf2latex_batch_e2e import write_rollups
from scripts.pipeline.run_pdf2latex_e2e import run_case
from src.pipeline.e2e_contract import E2ECaseConfig


def test_missing_pdf_is_recoverable_artifact_failure(tmp_path):
    config = E2ECaseConfig(
        doc_id="missing",
        output_dir=tmp_path / "missing",
        no_tex_source_inference=True,
    )
    summary = run_case(config)

    assert (tmp_path / "missing" / "CASE_SUMMARY.md").exists()
    assert (tmp_path / "missing" / "07_failure" / "failure_taxonomy.json").exists()
    assert any(failure["failure_type"] == "missing_original_pdf" for failure in summary["failures"])
    assert any(failure["failure_type"] == "generated_tex_missing" for failure in summary["failures"])


def test_missing_gold_skips_metrics_but_not_generation(tmp_path):
    tex = tmp_path / "generated.tex"
    tex.write_text("\\\\documentclass{article}\\n\\\\begin{document}\\nHello world.\\n\\\\end{document}\\n")
    config = E2ECaseConfig(
        doc_id="doc",
        output_dir=tmp_path / "doc",
        generated_tex_path=tex,
        evaluate=True,
        no_tex_source_inference=True,
    )
    summary = run_case(config)

    assert summary["outputs"]["generated_tex"].endswith("generated.tex")
    assert summary["comparison_metrics"] is False
    assert any(failure["failure_type"] == "gold_comparison_missing" for failure in summary["failures"])


def test_batch_rollup_handles_success_and_failure(tmp_path):
    results = [
        {
            "doc_id": "ok",
            "stratum": "ordinary",
            "status": "completed",
            "stages": [{"stage": "generation", "status": "ok"}],
            "failures": [],
            "outputs": {"generated_tex": "x"},
            "compile_success": False,
            "comparison_metrics": False,
            "visual_qa_status": "skipped",
            "main_failure_type": None,
        },
        {
            "doc_id": "bad",
            "stratum": "hard",
            "status": "completed_with_blocking_failures",
            "stages": [{"stage": "generation", "status": "failed"}],
            "failures": [{"failure_type": "generated_tex_missing", "severity": "blocking"}],
            "outputs": {},
            "compile_success": None,
            "comparison_metrics": False,
            "visual_qa_status": "skipped",
            "main_failure_type": "generated_tex_missing",
        },
    ]
    write_rollups(tmp_path, results)

    rollup = json.loads((tmp_path / "batch_rollup.json").read_text())
    assert rollup["doc_count"] == 2
    assert rollup["failure_type_counts"]["generated_tex_missing"] == 1
    assert (tmp_path / "batch_rollup.csv").exists()
    assert (tmp_path / "batch_rollup.md").exists()

