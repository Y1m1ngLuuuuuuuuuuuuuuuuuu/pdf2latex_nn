from __future__ import annotations

from dataclasses import replace

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


def _figure(node_id: str = "fig1", *, caption: str | None = None) -> DocumentNode:
    metadata = {"figure_caption": caption} if caption else {}
    return DocumentNode(node_id, BlockType.FIGURE, "", 0, [BBox(100, 100, 800, 400)], 0, metadata=metadata)


def _table(node_id: str = "tbl1", *, caption: str | None = None) -> DocumentNode:
    metadata = {"table_caption": caption} if caption else {}
    return DocumentNode(node_id, BlockType.TABLE, "", 0, [BBox(100, 100, 800, 400)], 0, metadata=metadata)


def test_mineru_backed_figure_caption_materializes_into_figure_caption() -> None:
    document = _document([_figure(caption="Figure 1: Model overview")])
    tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)
    latex = OriginalLikeIRLatexRenderer().render(document, tree, _style())

    assert result.promoted_captions
    assert r"\caption{Model overview}" in latex


def test_mineru_backed_table_caption_materializes_into_table_caption() -> None:
    document = _document([_table(caption="Table 1: Dataset statistics")])
    tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)
    latex = OriginalLikeIRLatexRenderer().render(document, tree, _style())

    assert result.promoted_captions
    assert r"\caption{Dataset statistics}" in latex


def test_regex_only_caption_does_not_materialize_or_create_placeholder() -> None:
    caption = DocumentNode("cap1", BlockType.TEXT, "Figure 2: Regex only", 0, [BBox(100, 410, 800, 440)], 0)
    document = _document([caption])
    tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    assert result.promoted_captions == []
    assert result.placeholder_floats == []
    assert not any(node.attributes.get("placeholder_float") for node in tree.nodes)


def test_diagnostic_only_caption_does_not_materialize() -> None:
    caption = DocumentNode(
        "cap1",
        BlockType.TEXT,
        "Figure 3: Diagnostic only",
        0,
        [BBox(100, 410, 800, 440)],
        0,
        metadata={"caption_diagnostic_only": True},
    )
    document = _document([caption])
    _tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    assert result.promoted_captions == []


def test_caption_paragraph_suppression_requires_exact_consumed_source() -> None:
    nodes = [
        _figure(caption=None),
        DocumentNode(
            "cap1",
            BlockType.TEXT,
            "Figure 1: Exact caption",
            0,
            [BBox(100, 410, 800, 440)],
            1,
            metadata={
                "caption_text": "Figure 1: Exact caption",
                "mineru_caption_role": "image_caption",
                "caption_confidence": "strong_middle_child",
                "caption_source_layer": "middle",
            },
        ),
    ]
    document = _document(nodes)
    tree, _result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    caption_node = next(node for node in tree.nodes if node.render_id == "r_cap1")
    assert caption_node.attributes.get("float_caption_consumed") is True


def test_mixed_body_paragraph_with_caption_like_prefix_is_not_suppressed() -> None:
    nodes = [
        _figure(caption=None),
        DocumentNode(
            "body1",
            BlockType.TEXT,
            "Figure 1: Exact caption shows the model output in detail.",
            0,
            [BBox(100, 410, 800, 460)],
            1,
            metadata={
                "caption_text": "Figure 1: Exact caption",
                "mineru_caption_role": "image_caption",
                "caption_confidence": "strong_middle_child",
                "caption_source_layer": "middle",
            },
        ),
    ]
    document = _document(nodes)
    tree, _result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    body_node = next(node for node in tree.nodes if node.render_id == "r_body1")
    assert not body_node.attributes.get("float_caption_consumed")


def test_duplicate_caption_key_suppresses_second_render() -> None:
    document = _document(
        [
            _figure(caption="Figure 4: Same caption"),
            replace(_figure("fig2", caption="Figure 4: Same caption"), reading_index=1, bboxes=[BBox(100, 500, 800, 800)]),
        ]
    )
    _tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    assert len(result.promoted_captions) == 1
    assert result.duplicate_caption_suppression


def test_subfigure_identities_are_not_deduped_together() -> None:
    document = _document(
        [
            _figure(caption="Fig. 2(a): Left panel."),
            replace(_figure("fig2", caption="Fig. 2(b): Right panel."), reading_index=1, bboxes=[BBox(100, 500, 800, 800)]),
        ]
    )
    _tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    numbers = {item["caption_number"] for item in result.promoted_captions}
    assert {"2(a)", "2(b)"} <= numbers


def test_wrong_type_pairing_is_blocked() -> None:
    document = _document([_figure(caption="Table 2: Wrong type")])
    _tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    assert result.promoted_captions == []


def test_caption_text_is_compile_safe_escaped() -> None:
    document = _document([_figure(caption="Figure 5: 50% A&B_x # {raw} $bad$")])
    tree, _result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)
    latex = OriginalLikeIRLatexRenderer().render(document, tree, _style())

    assert r"50\%" in latex
    assert r"A\&B" in latex
    assert r"\_x" in latex
    assert r"\# \{raw\}" in latex
    assert "$bad$" not in latex


def test_no_caption_only_placeholder_created() -> None:
    caption = DocumentNode(
        "cap1",
        BlockType.TEXT,
        "Figure 6: Missing float",
        0,
        [BBox(100, 410, 800, 440)],
        0,
        metadata={
            "caption_text": "Figure 6: Missing float",
            "mineru_caption_role": "image_caption",
            "caption_confidence": "strong_middle_child",
            "caption_source_layer": "middle",
        },
    )
    document = _document([caption])
    tree, result = apply_float_caption_layout(document, build_v8_render_tree(document, document_ir_path="document_ir.json"), enabled=True)

    assert result.promoted_captions == []
    assert result.placeholder_floats == []
    assert not any(node.attributes.get("placeholder_float") for node in tree.nodes)
