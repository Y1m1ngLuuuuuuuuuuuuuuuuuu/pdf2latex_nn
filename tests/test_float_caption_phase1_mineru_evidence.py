from __future__ import annotations

from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR
from src.reasoning.float_caption_matcher import (
    canonical_mineru_caption_id,
    caption_evidence_contexts_from_document,
)


def _node(node_id: str, text: str = "", *, node_type: BlockType = BlockType.FIGURE, metadata=None) -> DocumentNode:
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
        doc_id="float_phase1_doc",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=list(nodes),
        reading_order=[node.node_id for node in nodes],
    )


def _caption_metadata(role: str, text: str, *, caption_type: str, source_layer: str = "middle") -> dict:
    return {
        "mineru_caption_role": role,
        "raw_caption_type": role,
        "caption_text": text,
        "caption_type": caption_type,
        "caption_source_layer": source_layer,
        "caption_confidence": "strong_middle_child" if source_layer == "middle" else "strong_content_list_field",
        "caption_parent_float_id": "float-1",
        "caption_source_ids": ["cap-1"],
    }


def test_image_caption_becomes_high_confidence_figure_evidence():
    contexts = caption_evidence_contexts_from_document(
        _document(_node("fig1", metadata=_caption_metadata("image_caption", "Figure 1: A sample.", caption_type="figure")))
    )

    assert contexts[0].context_kind == "caption"
    assert contexts[0].caption_type == "figure"
    assert contexts[0].confidence_tier == "high"
    assert contexts[0].evidence_source == "mineru_middle_child"


def test_table_caption_becomes_high_confidence_table_evidence():
    contexts = caption_evidence_contexts_from_document(
        _document(_node("tbl1", node_type=BlockType.TABLE, metadata=_caption_metadata("table_caption", "Table 1: Results.", caption_type="table")))
    )

    assert contexts[0].caption_type == "table"
    assert contexts[0].confidence_tier == "high"


def test_chart_caption_becomes_chart_evidence():
    contexts = caption_evidence_contexts_from_document(
        _document(_node("chart1", metadata=_caption_metadata("chart_caption", "Chart 1: Trend.", caption_type="chart")))
    )

    assert contexts[0].caption_type == "chart"
    assert contexts[0].context_kind == "caption"


def test_algorithm_code_caption_becomes_algorithm_caption_evidence():
    contexts = caption_evidence_contexts_from_document(
        _document(
            _node(
                "alg1",
                node_type=BlockType.ALGORITHM,
                metadata=_caption_metadata("algorithm_caption", "Algorithm 1: Procedure.", caption_type="algorithm"),
            )
        )
    )

    assert contexts[0].caption_type == "algorithm"
    assert contexts[0].evidence["mineru_caption_role"] == "algorithm_caption"


def test_footnote_becomes_note_evidence_not_body():
    node = _node(
        "fig-note",
        metadata={
            "mineru_footnote_role": "image_footnote",
            "raw_footnote_type": "image_footnote",
            "footnote_text": "Image note.",
            "footnote_type": "image_note",
            "footnote_source_layer": "content_list",
            "footnote_confidence": "strong_content_list_field",
            "footnote_parent_float_id": "float-1",
        },
    )
    contexts = caption_evidence_contexts_from_document(_document(node))

    assert contexts[0].context_kind == "footnote"
    assert contexts[0].footnote_type == "image_note"
    assert contexts[0].evidence_source == "content_list_field"


def test_figure_shows_remains_body_reference_without_mineru_evidence():
    contexts = caption_evidence_contexts_from_document(
        _document(_node("body1", "Figure 3 shows the overview.", node_type=BlockType.TEXT))
    )

    assert contexts[0].context_kind == "body_reference_guard"
    assert contexts[0].confidence_tier == "diagnostic_only"


def test_as_shown_in_fig_remains_body_reference():
    contexts = caption_evidence_contexts_from_document(
        _document(_node("body2", "As shown in Fig. 2, the model improves.", node_type=BlockType.TEXT))
    )

    assert contexts[0].context_kind == "body_reference_guard"


def test_middle_and_content_list_same_caption_share_canonical_id():
    middle = _node(
        "fig-middle",
        metadata=_caption_metadata("image_caption", "Figure 1: Shared caption.", caption_type="figure", source_layer="middle"),
    )
    content = _node(
        "fig-content",
        metadata={
            **_caption_metadata("image_caption", "Figure 1: Shared caption.", caption_type="figure", source_layer="content_list"),
            "caption_source_ids": ["cap-1"],
        },
    )

    assert canonical_mineru_caption_id(middle) == canonical_mineru_caption_id(content)


def test_regex_only_caption_like_text_remains_diagnostic():
    contexts = caption_evidence_contexts_from_document(
        _document(_node("text-cap", "Figure 4: Regex-only caption.", node_type=BlockType.TEXT))
    )

    assert contexts[0].context_kind == "caption_like_diagnostic"
    assert contexts[0].evidence_source == "regex_only"
    assert contexts[0].confidence_tier == "diagnostic_only"


def test_no_renderer_or_graph_objects_required():
    contexts = caption_evidence_contexts_from_document(
        _document(_node("fig1", metadata=_caption_metadata("image_caption", "Figure 1: Safe.", caption_type="figure")))
    )

    assert len(contexts) == 1
    assert contexts[0].context_kind == "caption"
