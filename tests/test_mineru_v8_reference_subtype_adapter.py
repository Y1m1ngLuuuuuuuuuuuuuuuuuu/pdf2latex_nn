from __future__ import annotations

import json
from pathlib import Path

from src.adapters.mineru_v8_document_ir import convert_v8_payload_to_document_ir
from src.ir import BlockType


def _payload(items, *, source=None):
    return {
        "schema_version": "content_list_v8_reflow_v1_contentlist_merge_hint_experiment",
        "doc_id": "reference_doc",
        "items": items,
        "source": source or {},
    }


def _first_node(payload):
    document = convert_v8_payload_to_document_ir(payload, doc_id="reference_doc")
    assert document.nodes
    return document.nodes[0]


def test_content_list_ref_text_subtype_preserved(tmp_path: Path):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(
        json.dumps([{"type": "list", "sub_type": "ref_text", "text": "[1] A. Author. Title."}]),
        encoding="utf-8",
    )
    node = _first_node(
        _payload(
            [{"id": "v8_ref", "type": "list", "text": "", "page_idx": 0, "bbox": [0, 0, 10, 10], "source_content_list_index": 0}],
            source={"content_list_json": str(content_path)},
        )
    )

    assert node.node_type == BlockType.REFERENCE
    assert node.metadata["mineru_reference_role"] == "ref_text"
    assert node.metadata["reference_confidence"] == "strong_ref_text_subtype"
    assert node.metadata["reference_context_role"] == "reference_item"
    assert node.metadata["is_reference_item"] is True


def test_ref_text_list_items_and_order_preserved(tmp_path: Path):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(
        json.dumps([{"type": "list", "sub_type": "ref_text", "list_items": ["[2] B. Author. Paper."]}]),
        encoding="utf-8",
    )
    node = _first_node(
        _payload(
            [{"id": "v8_ref", "type": "list", "text": "", "page_idx": 0, "bbox": [0, 0, 10, 10], "source_content_list_index": 0}],
            source={"content_list_json": str(content_path)},
        )
    )

    assert "[2] B. Author" in node.metadata["reference_text"]
    assert node.metadata["reference_item_ids"]
    assert node.metadata["list_item_order"] == 0


def test_references_heading_preserved(tmp_path: Path):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([{"type": "text", "text": "References"}]), encoding="utf-8")
    node = _first_node(
        _payload(
            [{"id": "v8_heading", "type": "text", "text": "References", "page_idx": 0, "bbox": [0, 0, 10, 10], "source_content_list_index": 0}],
            source={"content_list_json": str(content_path)},
        )
    )

    assert node.metadata["mineru_reference_role"] == "reference_heading"
    assert node.metadata["reference_context_role"] == "reference_heading"
    assert node.metadata["reference_heading_ids"]


def test_ordinary_body_list_remains_list(tmp_path: Path):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([{"type": "list", "sub_type": "ordinary", "text": "1. Train the model."}]), encoding="utf-8")
    node = _first_node(
        _payload(
            [{"id": "v8_list", "type": "list", "text": "1. Train the model.", "page_idx": 0, "bbox": [0, 0, 10, 10], "source_content_list_index": 0}],
            source={"content_list_json": str(content_path)},
        )
    )

    assert node.node_type == BlockType.LIST
    assert "mineru_reference_role" not in node.metadata


def test_body_citation_is_not_reference_item():
    node = _first_node(
        _payload(
            [{"id": "v8_body", "type": "text", "text": "see [1] for implementation details.", "page_idx": 0, "bbox": [0, 0, 10, 10]}]
        )
    )

    assert node.node_type != BlockType.REFERENCE
    assert "mineru_reference_role" not in node.metadata


def test_numbered_body_item_is_not_reference_item():
    node = _first_node(
        _payload(
            [{"id": "v8_body_list", "type": "list", "text": "1. Initialize parameters.", "page_idx": 0, "bbox": [0, 0, 10, 10]}]
        )
    )

    assert node.node_type == BlockType.LIST
    assert "mineru_reference_role" not in node.metadata


def test_algorithm_step_numbers_are_not_reference_items():
    node = _first_node(
        _payload(
            [{"id": "v8_code", "type": "code", "text": "1: for each node do", "page_idx": 0, "bbox": [0, 0, 10, 10]}]
        )
    )

    assert node.node_type == BlockType.CODE
    assert "mineru_reference_role" not in node.metadata


def test_reference_source_ids_preserved(tmp_path: Path):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([{"type": "list", "sub_type": "ref_text", "text": "[3] C. Author."}]), encoding="utf-8")
    node = _first_node(
        _payload(
            [
                {
                    "id": "v8_ref",
                    "type": "list",
                    "text": "",
                    "page_idx": 0,
                    "bbox": [0, 0, 10, 10],
                    "source_content_list_index": 0,
                    "source_block_ids": ["reference_doc:p0000:m000003"],
                }
            ],
            source={"content_list_json": str(content_path)},
        )
    )

    assert "reference_doc:p0000:m000003" in node.metadata["reference_source_ids"]
    assert node.metadata["parent_reference_block_id"] == "reference_doc:p0000:m000003"


def test_v8_ref_text_type_preserved_without_content_list_index():
    node = _first_node(
        _payload(
            [
                {
                    "id": "v8_ref",
                    "type": "ref_text",
                    "text": "1. A. Author. Title.",
                    "page_idx": 0,
                    "bbox": [0, 0, 10, 10],
                    "source_block_ids": ["reference_doc:p0000:m000004"],
                }
            ]
        )
    )

    assert node.node_type == BlockType.REFERENCE
    assert node.metadata["mineru_reference_role"] == "ref_text"
    assert node.metadata["reference_confidence"] == "strong_ref_text_subtype"


def test_no_renderer_or_graph_mutation_required(tmp_path: Path):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([{"type": "list", "sub_type": "ref_text", "text": "[4] D. Author."}]), encoding="utf-8")
    node = _first_node(
        _payload(
            [{"id": "v8_ref", "type": "list", "text": "", "page_idx": 0, "bbox": [0, 0, 10, 10], "source_content_list_index": 0}],
            source={"content_list_json": str(content_path)},
        )
    )

    assert node.metadata["reference_context_role"] == "reference_item"
    assert "graph" not in node.metadata
