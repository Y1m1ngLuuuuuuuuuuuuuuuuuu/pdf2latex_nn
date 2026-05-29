from __future__ import annotations

from src.generation.ir_renderer import OriginalLikeIRLatexRenderer
from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR, RendererMode, RenderRole, StyleProfile
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


def _style() -> StyleProfile:
    return StyleProfile(profile_id="test", mode=RendererMode.ORIGINAL_LIKE)


def test_crop_only_caption_materializes_when_canonical_and_confident() -> None:
    figure = DocumentNode(
        node_id="fig1",
        node_type=BlockType.FIGURE,
        text="",
        page_idx=0,
        bboxes=[BBox(100, 100, 800, 400)],
        reading_index=0,
        metadata={"crop_caption": "Figure 5: Crop metadata caption"},
    )
    document = _document([figure])
    tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)
    latex = OriginalLikeIRLatexRenderer().render(document, tree, _style())

    assert result.promoted_captions[0]["origin"] == "crop_metadata"
    assert r"\caption{Crop metadata caption}" in latex


def test_promoted_caption_appears_in_render_tree_and_generated_tex() -> None:
    figure = DocumentNode(
        node_id="fig1",
        node_type=BlockType.FIGURE,
        text="",
        page_idx=0,
        bboxes=[BBox(100, 100, 800, 400)],
        reading_index=0,
        metadata={"figure_caption": "Figure 6: Render tree wiring"},
    )
    document = _document([figure])
    tree, _result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    figure_node = next(node for node in tree.nodes if node.role == RenderRole.FIGURE)
    assert figure_node.attributes["float_caption_layout_caption"]["text"] == "Figure 6: Render tree wiring"
    latex = OriginalLikeIRLatexRenderer().render(document, tree, _style())
    assert r"\caption{Render tree wiring}" in latex


def test_caption_consumption_does_not_suppress_unrelated_body_text() -> None:
    nodes = [
        DocumentNode("fig1", BlockType.FIGURE, "", 0, [BBox(100, 100, 800, 400)], 0),
        DocumentNode("cap1", BlockType.TEXT, "Figure 7: Caption text", 0, [BBox(100, 410, 800, 440)], 1),
        DocumentNode("body1", BlockType.TEXT, "This ordinary body paragraph remains visible.", 0, [BBox(100, 460, 800, 500)], 2),
    ]
    document = _document(nodes)
    tree, _result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    body = next(node for node in tree.nodes if node.render_id == "r_body1")
    caption = next(node for node in tree.nodes if node.render_id == "r_cap1")
    assert not body.attributes.get("float_caption_consumed")
    assert caption.attributes.get("float_caption_consumed") is True


def test_algorithm_body_is_not_changed_by_caption_materialization() -> None:
    algorithm = DocumentNode(
        node_id="alg1",
        node_type=BlockType.ALGORITHM,
        text="for i = 1..n do update",
        page_idx=0,
        bboxes=[BBox(100, 100, 800, 400)],
        reading_index=0,
        metadata={"algorithm_caption": "Algorithm 2: Update rule"},
    )
    document = _document([algorithm])
    tree, _result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    alg_node = next(node for node in tree.nodes if node.role == RenderRole.ALGORITHM)
    assert alg_node.latex is None
    assert alg_node.attributes["float_caption_layout_caption"]["text"] == "Algorithm 2: Update rule"
