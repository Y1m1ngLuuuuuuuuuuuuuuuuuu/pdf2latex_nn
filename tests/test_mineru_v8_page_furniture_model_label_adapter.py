from __future__ import annotations

import json
from pathlib import Path

from src.adapters.mineru_v8_document_ir import convert_v8_payload_to_document_ir


def _payload(items, *, source=None):
    return {
        "schema_version": "content_list_v8_reflow_v1_contentlist_merge_hint_experiment",
        "doc_id": "page_furniture_doc",
        "items": items,
        "source": source or {},
    }


def _first_node(payload):
    document = convert_v8_payload_to_document_ir(payload, doc_id="page_furniture_doc")
    assert document.nodes
    return document.nodes[0]


def _content_node(tmp_path: Path, raw_item: dict, *, v8_type: str = "text"):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([raw_item]), encoding="utf-8")
    return _first_node(
        _payload(
            [
                {
                    "id": "v8_item",
                    "type": v8_type,
                    "text": raw_item.get("text", ""),
                    "page_idx": 0,
                    "bbox": raw_item.get("bbox", [0, 0, 20, 20]),
                    "source_content_list_index": 0,
                }
            ],
            source={"content_list_json": str(content_path)},
        )
    )


def test_content_list_header_preserved(tmp_path: Path):
    node = _content_node(tmp_path, {"type": "header", "text": "Conference header", "bbox": [0, 0, 20, 20]})
    assert node.metadata["mineru_page_furniture_role"] == "page_header"
    assert node.metadata["is_page_header"] is True
    assert node.metadata["should_exclude_from_body_order"] is True


def test_content_list_footer_preserved(tmp_path: Path):
    node = _content_node(tmp_path, {"type": "footer", "text": "Footer note", "bbox": [0, 760, 20, 780]})
    assert node.metadata["mineru_page_furniture_role"] == "page_footer"
    assert node.metadata["is_page_footer"] is True
    assert node.metadata["should_exclude_from_visible_prose_metric"] is True


def test_content_list_page_number_preserved(tmp_path: Path):
    node = _content_node(tmp_path, {"type": "page_number", "text": "12", "bbox": [300, 770, 320, 790]})
    assert node.metadata["mineru_page_furniture_role"] == "page_number"
    assert node.metadata["is_page_number"] is True


def test_page_footnote_preserved_as_note_not_body(tmp_path: Path):
    node = _content_node(tmp_path, {"type": "page_footnote", "text": "* Equal contribution", "bbox": [70, 720, 300, 745]})
    assert node.metadata["mineru_page_furniture_role"] == "page_footnote"
    assert node.metadata["is_page_footnote"] is True
    assert node.metadata["should_exclude_from_body_order"] is True


def test_middle_discarded_block_preserved(tmp_path: Path):
    middle_path = tmp_path / "middle.json"
    middle_path.write_text(
        json.dumps(
            {
                "pdf_info": [
                    {
                        "page_idx": 0,
                        "discarded_blocks": [
                            {"index": 3, "type": "discarded", "text": "printer mark", "bbox": [1, 1, 10, 10]}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    node = _first_node(
        _payload(
            [
                {
                    "id": "v8_discarded",
                    "type": "text",
                    "text": "printer mark",
                    "page_idx": 0,
                    "bbox": [1, 1, 10, 10],
                    "source_block_ids": ["page_furniture_doc:p0000:m000003"],
                }
            ],
            source={"middle_json": str(middle_path)},
        )
    )
    assert node.metadata["mineru_page_furniture_role"] == "discarded_block"
    assert node.metadata["is_discarded_block"] is True
    assert node.metadata["page_furniture_confidence"] == "strong_middle_discarded"


def test_model_doc_title_metadata_preserved(tmp_path: Path):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            [
                {
                    "page_info": {"page_no": 0, "width": 612, "height": 792},
                    "layout_dets": [
                        {"cls_id": 6, "label": "doc_title", "score": 0.99, "bbox": [100, 50, 500, 90], "index": 1}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    node = _first_node(
        _payload(
            [{"id": "v8_title", "type": "text", "text": "A Paper Title", "page_idx": 0, "bbox": [100, 50, 500, 90]}],
            source={"model_json": str(model_path)},
        )
    )
    assert node.metadata["model_label"] == "doc_title"
    assert node.metadata["model_role_vote"] == "doc_title"
    assert node.metadata["is_document_title_candidate"] is True
    assert node.metadata["title_negative_for_body_heading"] is True


def test_model_header_footer_negative_mask_preserved(tmp_path: Path):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            [
                {
                    "page_info": {"page_no": 0, "width": 612, "height": 792},
                    "layout_dets": [{"cls_id": 1, "label": "header", "score": 0.91, "bbox": [0, 0, 612, 30], "index": 1}],
                }
            ]
        ),
        encoding="utf-8",
    )
    node = _first_node(
        _payload(
            [{"id": "v8_header", "type": "text", "text": "Running title", "page_idx": 0, "bbox": [0, 0, 612, 30]}],
            source={"model_json": str(model_path)},
        )
    )
    assert node.metadata["model_label"] == "header"
    assert node.metadata["mineru_page_furniture_role"] == "page_header"
    assert node.metadata["should_exclude_from_heading_detection"] is True


def test_ordinary_section_heading_near_top_is_not_masked():
    node = _first_node(
        _payload(
            [{"id": "v8_heading", "type": "title", "text": "1 Introduction", "page_idx": 1, "bbox": [80, 60, 260, 80]}]
        )
    )
    assert "mineru_page_furniture_role" not in node.metadata
    assert "should_exclude_from_body_order" not in node.metadata


def test_ordinary_short_body_text_is_not_page_furniture():
    node = _first_node(
        _payload([{"id": "v8_text", "type": "text", "text": "Short result.", "page_idx": 2, "bbox": [80, 400, 180, 420]}])
    )
    assert "mineru_page_furniture_role" not in node.metadata
    assert "model_label" not in node.metadata


def test_no_renderer_or_graph_mutation_required(tmp_path: Path):
    node = _content_node(tmp_path, {"type": "footer", "text": "Proceedings footer", "bbox": [0, 760, 300, 780]})
    assert node.metadata["is_page_footer"] is True
    assert "graph" not in node.metadata
    assert "renderer" not in node.metadata
