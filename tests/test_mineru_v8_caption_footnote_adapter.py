from __future__ import annotations

import json
from pathlib import Path

from src.adapters.mineru_v8_document_ir import convert_v8_payload_to_document_ir


def _payload(items, *, source=None):
    return {
        "schema_version": "content_list_v8_reflow_v1_contentlist_merge_hint_experiment",
        "doc_id": "caption_doc",
        "items": items,
        "source": source or {},
    }


def _first_node_metadata(payload):
    document = convert_v8_payload_to_document_ir(payload, doc_id="caption_doc")
    assert document.nodes
    return document.nodes[0].metadata


def test_middle_image_caption_preserved(tmp_path: Path):
    middle_path = tmp_path / "middle.json"
    middle_path.write_text(
        json.dumps(
            {
                "pdf_info": [
                    {
                        "page_idx": 0,
                        "preproc_blocks": [
                            {
                                "index": 7,
                                "type": "image",
                                "bbox": [10, 20, 200, 220],
                                "image_caption": ["Figure 1: A preserved image caption."],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    metadata = _first_node_metadata(
        _payload(
            [
                {
                    "id": "v8_img",
                    "type": "image",
                    "text": "",
                    "page_idx": 0,
                    "bbox": [10, 20, 200, 220],
                    "source_block_ids": ["caption_doc:p0000:m000007"],
                }
            ],
            source={"middle_json": str(middle_path)},
        )
    )

    assert metadata["mineru_caption_role"] == "image_caption"
    assert metadata["caption_type"] == "figure"
    assert metadata["caption_confidence"] == "strong_middle_child"
    assert "A preserved image caption" in metadata["caption_text"]


def test_middle_table_caption_preserved(tmp_path: Path):
    middle_path = tmp_path / "middle.json"
    middle_path.write_text(
        json.dumps(
            {
                "pdf_info": [
                    {
                        "page_idx": 0,
                        "preproc_blocks": [
                            {"index": 1, "type": "table", "bbox": [0, 0, 10, 10], "table_caption": ["Table 1: Results."]}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    metadata = _first_node_metadata(
        _payload(
            [{"id": "v8_tbl", "type": "table", "text": "", "page_idx": 0, "bbox": [0, 0, 10, 10], "source_block_ids": ["caption_doc:p0000:m000001"]}],
            source={"middle_json": str(middle_path)},
        )
    )

    assert metadata["mineru_caption_role"] == "table_caption"
    assert metadata["caption_type"] == "table"


def test_middle_chart_caption_preserved(tmp_path: Path):
    middle_path = tmp_path / "middle.json"
    middle_path.write_text(
        json.dumps({"pdf_info": [{"page_idx": 0, "preproc_blocks": [{"index": 2, "type": "chart", "chart_caption": ["Chart 1: Trend."]}]}]}),
        encoding="utf-8",
    )
    metadata = _first_node_metadata(
        _payload(
            [{"id": "v8_chart", "type": "chart", "text": "", "page_idx": 0, "bbox": [0, 0, 10, 10], "source_block_ids": ["caption_doc:p0000:m000002"]}],
            source={"middle_json": str(middle_path)},
        )
    )

    assert metadata["mineru_caption_role"] == "chart_caption"
    assert metadata["caption_type"] == "chart"


def test_content_list_image_caption_preserved(tmp_path: Path):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([{"type": "image", "image_caption": ["Figure 2: Content-list caption."]}]), encoding="utf-8")
    metadata = _first_node_metadata(
        _payload(
            [{"id": "v8_img", "type": "image", "text": "", "page_idx": 0, "bbox": [0, 0, 10, 10], "source_content_list_index": 0}],
            source={"content_list_json": str(content_path)},
        )
    )

    assert metadata["caption_source_layer"] == "content_list"
    assert metadata["caption_type"] == "figure"
    assert "Content-list caption" in metadata["caption_text"]


def test_content_list_table_caption_preserved(tmp_path: Path):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([{"type": "table", "table_caption": ["Table 2: Content-list table."]}]), encoding="utf-8")
    metadata = _first_node_metadata(
        _payload(
            [{"id": "v8_table", "type": "table", "text": "", "page_idx": 0, "bbox": [0, 0, 10, 10], "source_content_list_index": 0}],
            source={"content_list_json": str(content_path)},
        )
    )

    assert metadata["caption_source_layer"] == "content_list"
    assert metadata["caption_type"] == "table"


def test_code_caption_algorithm_subtype_preserved_as_algorithm(tmp_path: Path):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(
        json.dumps([{"type": "code", "sub_type": "algorithm", "code_caption": ["Algorithm 1: Procedure."], "code_body": ["return x"]}]),
        encoding="utf-8",
    )
    metadata = _first_node_metadata(
        _payload(
            [{"id": "v8_alg", "type": "code", "text": "", "page_idx": 0, "bbox": [0, 0, 10, 10], "source_content_list_index": 0}],
            source={"content_list_json": str(content_path)},
        )
    )

    assert metadata["caption_type"] == "algorithm"
    assert metadata["mineru_caption_role"] == "algorithm_caption"
    assert metadata["is_algorithm_subtype"] is True


def test_float_footnotes_preserved(tmp_path: Path):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(
        json.dumps([{"type": "image", "image_footnote": ["Image note."]}]),
        encoding="utf-8",
    )
    metadata = _first_node_metadata(
        _payload(
            [{"id": "v8_note", "type": "image", "text": "", "page_idx": 0, "bbox": [0, 0, 10, 10], "source_content_list_index": 0}],
            source={"content_list_json": str(content_path)},
        )
    )

    assert metadata["mineru_footnote_role"] == "image_footnote"
    assert metadata["footnote_type"] == "image_note"
    assert "Image note" in metadata["footnote_text"]


def test_body_reference_is_not_caption_metadata():
    metadata = _first_node_metadata(
        _payload(
            [
                {
                    "id": "v8_text",
                    "type": "text",
                    "text": "Figure 3 shows the overview and Table 1 reports the score.",
                    "page_idx": 0,
                    "bbox": [0, 0, 10, 10],
                }
            ]
        )
    )

    assert "mineru_caption_role" not in metadata
    assert "caption_text" not in metadata


def test_caption_body_footnote_child_ids_preserved(tmp_path: Path):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([{"type": "table", "table_caption": ["Table 3: IDs."], "table_footnote": ["note"]}]), encoding="utf-8")
    metadata = _first_node_metadata(
        _payload(
            [
                {
                    "id": "v8_table",
                    "type": "table",
                    "text": "",
                    "page_idx": 0,
                    "bbox": [0, 0, 10, 10],
                    "source_content_list_index": 0,
                    "source_block_ids": ["caption_doc:p0000:m000003"],
                    "source_line_ids": ["caption_doc:p0000:m000003:l0000:s0000"],
                }
            ],
            source={"content_list_json": str(content_path)},
        )
    )

    assert metadata["body_node_ids"]
    assert metadata["caption_node_ids"]
    assert metadata["footnote_node_ids"]
    assert metadata["parent_float_source_id"] == "caption_doc:p0000:m000003"
