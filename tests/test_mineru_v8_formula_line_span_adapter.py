from __future__ import annotations

from copy import deepcopy

from src.adapters.mineru_v8_document_ir import convert_v8_payload_to_document_ir, normalize_v8_items_for_adapter
from src.ir import BlockType


def _payload(item: dict) -> dict:
    base = {
        "id": "v8_000001",
        "type": "text",
        "text": "ordinary text",
        "page_idx": 0,
        "bbox": [10, 20, 200, 80],
        "reading_order": 0,
    }
    base.update(item)
    return {"schema_version": "test", "doc_id": "doc", "items": [base], "atomic_blocks": []}


def test_middle_inline_equation_span_preserved_as_inline_formula_metadata() -> None:
    document = convert_v8_payload_to_document_ir(
        _payload(
            {
                "type": "inline_equation",
                "text": "x_i",
                "source_line_ids": ["p0:b1:l0:s0"],
                "source_block_ids": ["p0:b1"],
                "source_lines": [
                    {"line_id": "p0:b1:l0:s0", "bbox": [10, 20, 30, 32], "text": "x_i"},
                ],
            }
        )
    )
    node = document.nodes[0]
    assert node.node_type == BlockType.INLINE_MATH
    assert node.metadata["is_inline_math"] is True
    assert node.metadata["is_display_math"] is False
    assert node.metadata["formula_context_role"] == "inline_attachment"
    assert node.metadata["parent_line_id"] == "p0:b1:l0:s0"
    assert node.metadata["parent_block_id"] == "p0:b1"


def test_middle_interline_equation_span_preserved_as_display_formula_metadata() -> None:
    document = convert_v8_payload_to_document_ir(
        _payload(
            {
                "type": "interline_equation",
                "text": "E = mc^2",
                "source_line_ids": ["p0:b2:l0:s0"],
                "source_block_ids": ["p0:b2"],
            }
        )
    )
    node = document.nodes[0]
    assert node.node_type == BlockType.EQUATION
    assert node.metadata["is_display_math"] is True
    assert node.metadata["formula_context_role"] == "display_math"
    assert node.metadata["formula_latex"] == "E = mc^2"


def test_content_list_equation_latex_preserved() -> None:
    document = convert_v8_payload_to_document_ir(
        _payload(
            {
                "type": "equation",
                "text": "\\sum_i x_i",
                "text_format": "latex",
                "content_list_type": "equation",
            }
        )
    )
    node = document.nodes[0]
    assert node.node_type == BlockType.EQUATION
    assert node.metadata["formula_latex"] == "\\sum_i x_i"
    assert node.metadata["text_format"] == "latex"
    assert node.metadata["formula_confidence"] == "strong_span_interline"


def test_line_span_bbox_and_parent_ids_preserved() -> None:
    document = convert_v8_payload_to_document_ir(
        _payload(
            {
                "type": "inline_equation",
                "text": "\\alpha",
                "source_line_ids": ["p0:b3:l2:s4"],
                "source_block_ids": ["p0:b3"],
                "source_lines": [
                    {"line_id": "p0:b3:l2:s4", "bbox": [11, 22, 33, 44], "text": "\\alpha"},
                ],
            }
        )
    )
    node = document.nodes[0]
    assert node.metadata["line_span_ids"] == ["p0:b3:l2:s4"]
    assert node.metadata["line_bbox"] == [11, 22, 33, 44]
    assert node.metadata["span_bbox"] == [11, 22, 33, 44]


def test_ordinary_variable_like_text_is_not_formula() -> None:
    document = convert_v8_payload_to_document_ir(
        _payload({"type": "text", "text": "The variable x is updated in Eq. 2.", "page_idx": 1})
    )
    node = document.nodes[0]
    assert node.node_type == BlockType.TEXT
    assert "formula_context_role" not in node.metadata


def test_weak_formula_artifact_marked_uncertain_when_evidence_is_weak() -> None:
    document = convert_v8_payload_to_document_ir(
        _payload({"type": "text", "text": "x", "text_format": "latex", "bbox": [1, 2, 3, 4]})
    )
    node = document.nodes[0]
    assert node.metadata["formula_context_role"] == "formula_ocr_artifact"
    assert node.metadata["formula_confidence"] == "medium_equation_text"


def test_normalize_does_not_mutate_raw_v8_payload() -> None:
    payload = _payload({"type": "interline_equation", "text": "a=b"})
    original = deepcopy(payload)
    normalize_v8_items_for_adapter(payload)
    assert payload == original


def test_no_graph_or_renderer_mutation_contract() -> None:
    normalized = normalize_v8_items_for_adapter(_payload({"type": "interline_equation", "text": "a=b"}))
    assert "graph" not in normalized[0]
    assert "render_tree" not in normalized[0]


def test_document_ir_metadata_contains_formula_context_role() -> None:
    document = convert_v8_payload_to_document_ir(_payload({"type": "interline_equation", "text": "a=b"}))
    assert document.nodes[0].metadata["formula_context_role"] == "display_math"
