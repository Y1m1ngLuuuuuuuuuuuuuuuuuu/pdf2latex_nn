from __future__ import annotations

from dataclasses import replace

from src.generation.ir_renderer import OriginalLikeIRLatexRenderer
from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR, RendererMode, RenderRole, StyleProfile
from src.reasoning.float_caption_layout import apply_float_caption_layout
from src.reasoning.v8_render_tree import build_v8_render_tree


def _style() -> StyleProfile:
    return StyleProfile(profile_id="test", mode=RendererMode.ORIGINAL_LIKE)


def _document(nodes: list[DocumentNode]) -> DocumentIR:
    return DocumentIR(
        doc_id="doc",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
        provenance={"source": "v8_full_observable_facts"},
    )


def test_regex_only_caption_does_not_create_placeholder() -> None:
    caption = DocumentNode(
        node_id="cap1",
        node_type=BlockType.TEXT,
        text="Figure 3: Architecture overview",
        page_idx=0,
        bboxes=[BBox(100, 100, 800, 140)],
        reading_index=0,
    )
    document = _document([caption])
    tree = build_v8_render_tree(document, document_ir_path="document_ir.json")
    tree, _result = apply_float_caption_layout(document, tree, enabled=True)

    latex = OriginalLikeIRLatexRenderer().render(document, tree, _style())

    assert r"\caption{Architecture overview}" not in latex
    assert "Architecture overview" in latex


def test_float_without_caption_keeps_crop_fallback_without_invented_caption() -> None:
    figure = DocumentNode(
        node_id="fig1",
        node_type=BlockType.FIGURE,
        text="",
        page_idx=0,
        bboxes=[BBox(100, 100, 800, 400)],
        reading_index=0,
    )
    document = _document([figure])
    tree = build_v8_render_tree(document, document_ir_path="document_ir.json", enable_float_caption_layout=True)

    latex = OriginalLikeIRLatexRenderer().render(document, tree, _style())

    assert r"\begin{figure" in latex
    assert r"\caption{Figure}" not in latex


def test_algorithm_caption_is_deferred_by_floatcaption_sprint() -> None:
    algorithm = DocumentNode(
        node_id="alg1",
        node_type=BlockType.ALGORITHM,
        text="Algorithm 1: Training procedure",
        page_idx=0,
        bboxes=[BBox(100, 100, 800, 400)],
        reading_index=0,
    )
    document = _document([algorithm])
    tree = build_v8_render_tree(document, document_ir_path="document_ir.json", enable_float_caption_layout=True)
    alg_node = next(node for node in tree.nodes if node.role == RenderRole.ALGORITHM)
    assert not alg_node.attributes.get("float_caption_layout_caption")

    # The Algorithm renderer may still decide how to handle algorithm text; this
    # FloatCaption sprint must not attach algorithm captions through the
    # caption-materialization pass.


def test_float_caption_layout_caption_overrides_source_metadata() -> None:
    figure = DocumentNode(
        node_id="fig1",
        node_type=BlockType.FIGURE,
        text="",
        page_idx=0,
        bboxes=[BBox(100, 100, 800, 400)],
        reading_index=0,
        metadata={"figure_caption": "Figure 1: Old metadata caption"},
    )
    document = _document([figure])
    tree = build_v8_render_tree(document, document_ir_path="document_ir.json")
    nodes = []
    for node in tree.nodes:
        if node.role == RenderRole.FIGURE:
            attrs = dict(node.attributes)
            attrs["float_caption_layout_caption"] = {"text": "Figure 1: Layout caption wins"}
            nodes.append(replace(node, text="Figure 1: Layout caption wins", attributes=attrs))
        else:
            nodes.append(node)
    tree = replace(tree, nodes=nodes)

    latex = OriginalLikeIRLatexRenderer().render(document, tree, _style())

    assert r"\caption{Layout caption wins}" in latex
    assert "Old metadata caption" not in latex
