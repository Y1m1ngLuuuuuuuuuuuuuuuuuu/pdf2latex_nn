from __future__ import annotations

import json

from src.pipeline.failure_taxonomy import classify_compile_failure, make_failure, write_failure_taxonomy


def test_failure_taxonomy_schema_validates(tmp_path):
    failure = make_failure(
        stage="input_discovery",
        failure_type="missing_original_pdf",
        severity="recoverable",
        message="PDF missing",
        recommended_next_action="provide_pdf",
    )
    path = tmp_path / "failure_taxonomy.json"
    payload = write_failure_taxonomy(path, [failure], doc_id="doc", status="completed")

    loaded = json.loads(path.read_text())
    assert loaded == payload
    assert loaded["schema_version"] == "pdf2latex_e2e_failure_taxonomy_v1"
    assert loaded["failures"][0]["failure_type"] == "missing_original_pdf"
    assert loaded["blocking_count"] == 0


def test_compile_failure_classifier_unicode_and_math():
    unicode_report = {"success": False, "error_summary": "Unicode character ∆ not set up for use with LaTeX"}
    math_report = {"success": False, "error_summary": "Missing $ inserted"}

    assert classify_compile_failure(unicode_report) == "unicode_or_special_char"
    assert classify_compile_failure(math_report) == "body_math_syntax_error"

