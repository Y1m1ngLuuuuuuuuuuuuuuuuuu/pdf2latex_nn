from __future__ import annotations

from pathlib import Path

from src.adapters.mineru_v8_document_ir import convert_v8_payload_to_document_ir, normalize_v8_items_for_adapter
from src.ir import BlockType


def _payload(item: dict) -> dict:
    base = {
        "id": "v8_000001",
        "type": "code",
        "text": "",
        "page_idx": 0,
        "bbox": [10, 20, 200, 160],
        "reading_order": 0,
    }
    base.update(item)
    return {"schema_version": "test", "doc_id": "doc", "items": [base], "atomic_blocks": []}


def test_raw_code_subtype_algorithm_preserves_document_ir_identity() -> None:
    document = convert_v8_payload_to_document_ir(
        _payload({"sub_type": "algorithm", "code_body": "Algorithm 1: Train\nRequire: data\nreturn model"})
    )
    node = document.nodes[0]
    assert node.node_type == BlockType.ALGORITHM
    assert node.metadata["is_algorithm_subtype"] is True
    assert node.metadata["mineru_subtype"] == "algorithm"
    assert node.metadata["algorithm_confidence"] == "strong_subtype"
    assert "Require: data" in node.text


def test_content_list_v2_algorithm_content_preserves_identity() -> None:
    document = convert_v8_payload_to_document_ir(
        _payload({"type": "algorithm", "algorithm_content": "Algorithm 2: Decode\nInput: x\nOutput: y"})
    )
    node = document.nodes[0]
    assert node.node_type == BlockType.ALGORITHM
    assert node.metadata["algorithm_content"].startswith("Algorithm 2")


def test_content_list_pointer_recovers_algorithm_subtype(tmp_path: Path) -> None:
    content_list = tmp_path / "doc_content_list.json"
    content_list.write_text(
        '[{"type":"code","sub_type":"algorithm","code_body":"Algorithm 3: Update","code_caption":["Algorithm 3: Update"]}]',
        encoding="utf-8",
    )
    payload = _payload(
        {
            "content_list_json": str(content_list),
            "source_content_list_index": 0,
            "content_list_type": "code",
        }
    )
    normalized = normalize_v8_items_for_adapter(payload)
    assert normalized[0]["canonical_type"] == "algorithm"
    assert normalized[0]["raw_sub_type"] == "algorithm"
    assert normalized[0]["code_body"] == "Algorithm 3: Update"


def test_code_caption_alone_preserves_algorithm_caption_metadata() -> None:
    document = convert_v8_payload_to_document_ir(_payload({"code_caption": ["Algorithm 4: Score"], "code_body": "return score"}))
    node = document.nodes[0]
    assert node.node_type == BlockType.ALGORITHM
    assert node.metadata["code_caption"] == ["Algorithm 4: Score"]
    assert node.metadata["algorithm_confidence"] == "medium_caption"


def test_algorithm_reference_paragraph_is_not_algorithm() -> None:
    document = convert_v8_payload_to_document_ir(
        _payload({"type": "text", "text": "Algorithm 1 shows the training procedure.", "page_idx": 1})
    )
    assert document.nodes[0].node_type == BlockType.TEXT


def test_plain_code_without_algorithm_subtype_remains_code() -> None:
    document = convert_v8_payload_to_document_ir(_payload({"code_body": "print('hello')"}))
    assert document.nodes[0].node_type == BlockType.CODE


def test_algorithm_body_and_caption_source_ids_preserved() -> None:
    document = convert_v8_payload_to_document_ir(
        _payload(
            {
                "sub_type": "algorithm",
                "source_block_ids": ["m1"],
                "source_line_ids": ["l1"],
                "code_caption": ["Algorithm 5: Loop"],
                "code_body": "Algorithm 5: Loop\nfor i do update",
            }
        )
    )
    node = document.nodes[0]
    assert node.metadata["algorithm_body_ids"] == ["m1"]
    assert node.metadata["algorithm_caption_ids"] == ["l1"]


def test_normalize_does_not_mutate_v8_payload() -> None:
    payload = _payload({"sub_type": "algorithm", "code_body": "Algorithm 6: Stable"})
    original = {**payload["items"][0]}
    normalize_v8_items_for_adapter(payload)
    assert payload["items"][0] == original


def test_no_graph_or_renderer_mutation_contract() -> None:
    payload = _payload({"sub_type": "algorithm", "code_body": "Algorithm 7: Contract"})
    normalized = normalize_v8_items_for_adapter(payload)
    assert "graph" not in normalized[0]
    assert "render_tree" not in normalized[0]
