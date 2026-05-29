from __future__ import annotations

from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR, RenderRole
from src.reasoning.float_caption_layout import apply_float_caption_layout
from src.reasoning.v8_render_tree import build_v8_render_tree


def _document(nodes: list[DocumentNode]) -> DocumentIR:
    return DocumentIR(
        doc_id="doc",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
        provenance={"source": "v8_full_observable_facts"},
    )


def test_metadata_caption_materializes_on_existing_float_without_mutating_v8() -> None:
    figure = DocumentNode(
        node_id="fig1",
        node_type=BlockType.FIGURE,
        text="",
        page_idx=0,
        bboxes=[BBox(100, 100, 800, 400)],
        reading_index=0,
        metadata={"figure_caption": "Figure 1: Model overview"},
    )
    document = _document([figure])
    before_metadata = dict(figure.metadata)
    tree = build_v8_render_tree(document, document_ir_path="document_ir.json")

    updated, result = apply_float_caption_layout(document, tree, enabled=True)

    assert figure.metadata == before_metadata
    fig_node = next(node for node in updated.nodes if node.role == RenderRole.FIGURE)
    assert fig_node.text == "Figure 1: Model overview"
    assert fig_node.attributes["float_caption_layout"] is True
    assert result.promoted_captions


def test_high_confidence_caption_without_float_creates_placeholder() -> None:
    caption = DocumentNode(
        node_id="cap1",
        node_type=BlockType.TEXT,
        text="Figure 2: Missing visual asset",
        page_idx=0,
        bboxes=[BBox(100, 420, 800, 460)],
        reading_index=0,
    )
    document = _document([caption])
    tree = build_v8_render_tree(document, document_ir_path="document_ir.json")

    updated, result = apply_float_caption_layout(document, tree, enabled=True)

    placeholders = [node for node in updated.nodes if node.attributes.get("placeholder_float")]
    assert placeholders
    assert placeholders[0].role == RenderRole.FIGURE
    assert result.placeholder_floats


def test_duplicate_captions_are_suppressed_but_subfigures_are_kept() -> None:
    nodes = [
        DocumentNode(
            node_id="cap_a",
            node_type=BlockType.TEXT,
            text="Fig. 2(a): Left panel.",
            page_idx=0,
            bboxes=[BBox(100, 100, 400, 120)],
            reading_index=0,
        ),
        DocumentNode(
            node_id="cap_b",
            node_type=BlockType.TEXT,
            text="Fig. 2(b): Right panel.",
            page_idx=0,
            bboxes=[BBox(500, 100, 800, 120)],
            reading_index=1,
        ),
        DocumentNode(
            node_id="cap_dup",
            node_type=BlockType.TEXT,
            text="Fig. 2(a): Left panel.",
            page_idx=0,
            bboxes=[BBox(100, 130, 400, 150)],
            reading_index=2,
        ),
    ]
    document = _document(nodes)

    _updated, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    assert len(result.placeholder_floats) == 2
    assert len(result.duplicate_caption_suppression) == 1


def test_caption_paragraph_is_marked_consumed_when_promoted() -> None:
    caption = DocumentNode(
        node_id="cap1",
        node_type=BlockType.TEXT,
        text="Table 1: Dataset statistics",
        page_idx=0,
        bboxes=[BBox(100, 420, 800, 460)],
        reading_index=0,
    )
    document = _document([caption])
    tree = build_v8_render_tree(document, document_ir_path="document_ir.json")

    updated, _result = apply_float_caption_layout(document, tree, enabled=True)

    original = next(node for node in updated.nodes if node.render_id == "r_cap1")
    assert original.attributes["float_caption_consumed"] is True

