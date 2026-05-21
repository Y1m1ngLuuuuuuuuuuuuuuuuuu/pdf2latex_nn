import json
import os
import time
from pathlib import Path

import src.perception.content_resolver as resolver
from src.perception.content_resolver import (
    ContentResolverConfig,
    V7SchemaThresholds,
    resolve_v7_content_for_doc,
    validate_content_v7_schema,
)


def write_content(path: Path, *, items: list[dict], page_count: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": "content_v7_with_styles", "page_count": page_count, "items": items}),
        encoding="utf-8",
    )
    return path


def item(**extra):
    base = {
        "text": "Body",
        "page_idx": 0,
        "layout_layer": "main_text_flow",
        "layout_role": "body_text",
        "canonical_type": "paragraph",
        "style_spans": [{"text": "Body"}],
        "_v7_node_id": "v7_p0000_b000001",
    }
    base.update(extra)
    return base


def test_old_mineru_output_missing_layout_fields_is_rejected(tmp_path):
    path = write_content(tmp_path / "mineru_output" / "doc1" / "auto" / "doc1_content_list_v7_styles.json", items=[{"text": "old"}])

    metrics = validate_content_v7_schema(path, thresholds=V7SchemaThresholds())

    assert metrics.stale_schema_flag
    assert "stale_v7_schema_missing_layout_layer" in metrics.failed_reasons
    assert "stale_v7_schema_missing_layout_role" in metrics.failed_reasons


def test_same_doc_old_and_new_content_selects_new_valid_content(tmp_path):
    old = write_content(tmp_path / "mineru_output" / "doc1" / "auto" / "doc1_content_list_v7_styles.json", items=[{"text": "old"}])
    new = write_content(
        tmp_path / "mineru_output_v7_tocfilter_latest" / "doc1" / "auto" / "doc1_content_list_v7_styles.json",
        items=[item()],
    )
    os.utime(old, (time.time() + 100, time.time() + 100))

    result = resolve_v7_content_for_doc(
        "doc1",
        raw_pdf_path=None,
        config=ContentResolverConfig(mineru_output_dirs=(old.parents[2], new.parents[2])),
    )

    assert result.selected_path == new


def test_page_count_mismatch_is_rejected(tmp_path, monkeypatch):
    path = write_content(tmp_path / "root" / "doc1" / "auto" / "doc1_content_list_v7_styles.json", items=[item()], page_count=2)
    monkeypatch.setattr(resolver, "pdf_page_count", lambda raw_pdf_path: 1)

    metrics = validate_content_v7_schema(path, raw_pdf_path=tmp_path / "doc1.pdf", thresholds=V7SchemaThresholds())

    assert "content_page_count_mismatch" in metrics.failed_reasons
    assert metrics.page_count_match is False


def test_mtime_new_but_low_coverage_is_rejected(tmp_path):
    old_valid = write_content(tmp_path / "old_v7" / "doc1" / "auto" / "doc1_content_list_v7_styles.json", items=[item()])
    new_bad = write_content(tmp_path / "new_v7" / "doc1" / "auto" / "doc1_content_list_v7_styles.json", items=[{"text": "bad"}])
    os.utime(new_bad, (time.time() + 200, time.time() + 200))

    result = resolve_v7_content_for_doc(
        "doc1",
        raw_pdf_path=None,
        config=ContentResolverConfig(mineru_output_dirs=(old_valid.parents[2], new_bad.parents[2])),
    )

    assert result.selected_path == old_valid


def test_high_coverage_old_content_is_accepted_if_valid(tmp_path):
    path = write_content(tmp_path / "mineru_output_v7_layerband" / "doc1" / "auto" / "doc1_content_list_v7_styles.json", items=[item()])

    result = resolve_v7_content_for_doc("doc1", raw_pdf_path=None, config=ContentResolverConfig(mineru_output_dirs=(path.parents[2],)))

    assert result.selected_path == path


def test_no_valid_content_returns_stale_skip_reason(tmp_path):
    path = write_content(tmp_path / "mineru_output" / "doc1" / "auto" / "doc1_content_list_v7_styles.json", items=[{"text": "old"}])

    result = resolve_v7_content_for_doc("doc1", raw_pdf_path=None, config=ContentResolverConfig(mineru_output_dirs=(path.parents[2],)))

    assert result.selected_path is None
    assert result.skip_reason == "stale_or_missing_v7_schema"


def test_force_refresh_flag_does_not_default_to_mineru_rerun(tmp_path):
    path = write_content(tmp_path / "mineru_output" / "doc1" / "auto" / "doc1_content_list_v7_styles.json", items=[{"text": "old"}])

    default = resolve_v7_content_for_doc("doc1", raw_pdf_path=None, config=ContentResolverConfig(mineru_output_dirs=(path.parents[2],)))
    forced = resolve_v7_content_for_doc(
        "doc1",
        raw_pdf_path=None,
        config=ContentResolverConfig(mineru_output_dirs=(path.parents[2],), force_refresh_v7_conversion=True),
    )

    assert default.skip_reason == "stale_or_missing_v7_schema"
    assert forced.skip_reason == "force_refresh_v7_conversion_required"
