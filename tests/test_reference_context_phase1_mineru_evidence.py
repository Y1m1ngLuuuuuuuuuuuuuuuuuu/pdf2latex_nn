from __future__ import annotations

from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR
from src.reasoning.reference_context_group import (
    canonical_mineru_reference_id,
    reference_evidence_contexts_from_document,
)


def _node(node_id: str, text: str = "", *, node_type: BlockType = BlockType.REFERENCE, metadata=None) -> DocumentNode:
    return DocumentNode(
        node_id=node_id,
        node_type=node_type,
        text=text,
        page_idx=0,
        bboxes=[BBox(0, 0, 100, 100)],
        reading_index=0,
        metadata=metadata or {},
    )


def _document(*nodes: DocumentNode) -> DocumentIR:
    return DocumentIR(
        doc_id="reference_phase1_doc",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=list(nodes),
        reading_order=[node.node_id for node in nodes],
    )


def _reference_metadata(text="[1] A. Author. Title.", *, role="ref_text", context_role="reference_item") -> dict:
    return {
        "mineru_reference_role": role,
        "raw_reference_sub_type": "ref_text",
        "reference_text": text,
        "reference_label": "[1]",
        "reference_source_layer": "content_list",
        "reference_confidence": "strong_ref_text_subtype",
        "reference_list_item_index": 0,
        "reference_parent_block_id": "ref-block",
        "reference_source_ids": ["ref-source-1"],
        "is_reference_item": context_role == "reference_item",
        "is_reference_section_candidate": True,
        "reference_context_role": context_role,
    }


def test_ref_text_list_item_becomes_high_confidence_reference_item():
    contexts = reference_evidence_contexts_from_document(_document(_node("ref1", metadata=_reference_metadata())))

    assert contexts[0].context_kind == "reference_item"
    assert contexts[0].confidence_tier == "high"
    assert contexts[0].evidence_source == "mineru_ref_text_subtype"


def test_references_heading_near_ref_text_becomes_heading_evidence():
    contexts = reference_evidence_contexts_from_document(
        _document(
            _node(
                "heading",
                "References",
                node_type=BlockType.TITLE,
                metadata={
                    **_reference_metadata("References", role="reference_heading", context_role="reference_heading"),
                    "reference_confidence": "strong_reference_region",
                },
            )
        )
    )

    assert contexts[0].context_kind == "reference_heading"
    assert contexts[0].confidence_tier == "high"


def test_contiguous_ref_text_list_can_be_bibliography_block():
    first = _node("ref1", metadata=_reference_metadata("[1] A. Author."))
    second = _node("ref2", metadata={**_reference_metadata("[2] B. Author."), "reference_list_item_index": 1})
    contexts = reference_evidence_contexts_from_document(_document(first, second))

    assert [context.context_kind for context in contexts] == ["reference_item", "reference_item"]
    assert {context.parent_reference_block_id for context in contexts} == {"ref-block"}


def test_body_citation_remains_body_citation():
    contexts = reference_evidence_contexts_from_document(
        _document(_node("body", "see [1] for details.", node_type=BlockType.TEXT))
    )

    assert contexts[0].context_kind == "body_citation_guard"
    assert contexts[0].confidence_tier == "diagnostic_only"


def test_body_numbered_list_remains_ordinary_list():
    contexts = reference_evidence_contexts_from_document(
        _document(_node("list", "1. Initialize parameters.", node_type=BlockType.LIST))
    )

    assert contexts[0].context_kind == "ordinary_list"


def test_algorithm_step_number_not_reference():
    contexts = reference_evidence_contexts_from_document(
        _document(_node("alg", "1: for each node do", node_type=BlockType.CODE))
    )

    assert contexts == []


def test_equation_number_not_reference():
    contexts = reference_evidence_contexts_from_document(
        _document(_node("eq", "(1)", node_type=BlockType.EQUATION))
    )

    assert contexts == []


def test_middle_and_content_list_same_reference_share_canonical_id():
    first = _node("ref1", metadata=_reference_metadata("[1] A. Author."))
    second = _node(
        "ref2",
        metadata={**_reference_metadata("[1] A. Author."), "reference_source_layer": "middle", "reference_source_ids": ["ref-source-1"]},
    )

    assert canonical_mineru_reference_id(first) == canonical_mineru_reference_id(second)


def test_regex_only_reference_like_text_remains_diagnostic():
    contexts = reference_evidence_contexts_from_document(
        _document(_node("regex", "[9] Someone. A paper from 2020.", node_type=BlockType.TEXT))
    )

    assert contexts[0].context_kind == "reference_like_diagnostic"
    assert contexts[0].evidence_source == "regex_only"


def test_no_renderer_or_graph_objects_required():
    contexts = reference_evidence_contexts_from_document(_document(_node("ref1", metadata=_reference_metadata())))

    assert len(contexts) == 1
    assert contexts[0].context_kind == "reference_item"
