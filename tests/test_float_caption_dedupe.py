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


def test_text_and_metadata_caption_render_once_with_text_canonical() -> None:
    nodes = [
        DocumentNode(
            node_id="fig1",
            node_type=BlockType.FIGURE,
            text="",
            page_idx=0,
            bboxes=[BBox(100, 100, 800, 400)],
            reading_index=0,
            metadata={"figure_caption": "Figure 1: Model overview"},
        ),
        DocumentNode(
            node_id="cap1",
            node_type=BlockType.TEXT,
            text="Figure 1: Model overview",
            page_idx=0,
            bboxes=[BBox(100, 410, 800, 440)],
            reading_index=1,
        ),
    ]
    document = _document(nodes)
    tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    figure = next(node for node in tree.nodes if node.role == RenderRole.FIGURE)
    assert figure.attributes["float_caption_layout_caption"]["origin"] == "text_block"
    assert len(result.noncanonical_suppressed_candidates) == 1
    assert result.noncanonical_suppressed_candidates[0]["reason"] == "duplicate_suppressed_by_canonical"


def test_crop_and_float_metadata_caption_render_once() -> None:
    figure = DocumentNode(
        node_id="fig1",
        node_type=BlockType.FIGURE,
        text="",
        page_idx=0,
        bboxes=[BBox(100, 100, 800, 400)],
        reading_index=0,
        metadata={
            "figure_caption": "Figure 2: Qualitative examples",
            "crop_caption": "Figure 2: Qualitative examples",
        },
    )
    document = _document([figure])
    _tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    assert len(result.promoted_captions) == 1
    assert len(result.noncanonical_suppressed_candidates) == 1


def test_subfigure_markers_are_distinct_canonical_identities() -> None:
    nodes = [
        DocumentNode("cap_a", BlockType.TEXT, "Fig. 2(a): Left panel.", 0, [BBox(100, 100, 400, 120)], 0),
        DocumentNode("cap_b", BlockType.TEXT, "Fig. 2(b): Right panel.", 0, [BBox(500, 100, 800, 120)], 1),
    ]
    document = _document(nodes)
    _tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    numbers = {item["caption_number"] for item in result.promoted_captions}
    assert {"2(a)", "2(b)"} <= numbers
    assert result.noncanonical_suppressed_candidates == []


def test_panel_label_and_synthetic_fallback_do_not_materialize() -> None:
    nodes = [
        DocumentNode("panel", BlockType.TEXT, "Fig. 3: (a)", 0, [BBox(100, 100, 400, 120)], 0),
        DocumentNode("generic", BlockType.TEXT, "Figure 4: Figure", 0, [BBox(100, 150, 400, 170)], 1),
    ]
    document = _document(nodes)
    _tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    classes = {item["caption_candidate_class"] for item in result.noncanonical_suppressed_candidates}
    assert "PANEL_LABEL" in classes
    assert "SYNTHETIC_FALLBACK_CAPTION" in classes
    assert result.promoted_captions == []
