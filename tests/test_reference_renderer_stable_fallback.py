from __future__ import annotations

from src.generation.ir_renderer import OriginalLikeIRLatexRenderer
from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR, RendererMode, RenderRole, RenderTreeIR, RenderTreeNode, StyleProfile


def _document(nodes: list[DocumentNode]) -> DocumentIR:
    return DocumentIR(
        doc_id="reference_fallback_doc",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )


def _style() -> StyleProfile:
    return StyleProfile(profile_id="test", mode=RendererMode.ORIGINAL_LIKE)


def _tree(role: RenderRole, source_ids: list[str]) -> RenderTreeIR:
    return RenderTreeIR(
        doc_id="reference_fallback_doc",
        document_ir_path="document_ir.json",
        root_id="root",
        nodes=[
            RenderTreeNode(render_id="root", role=RenderRole.ROOT, children=["target"]),
            RenderTreeNode(render_id="target", role=role, source_node_ids=source_ids),
        ],
    )


def _reference_node(node_id: str, text: str = "", *, metadata: dict | None = None) -> DocumentNode:
    return DocumentNode(
        node_id=node_id,
        node_type=BlockType.REFERENCE,
        text=text,
        page_idx=0,
        bboxes=[BBox(100, 700, 900, 730)],
        reading_index=0,
        metadata=metadata or {},
    )


def test_mineru_reference_text_renders_stable_thebibliography() -> None:
    node = _reference_node(
        "ref1",
        metadata={
            "mineru_reference_role": "ref_text",
            "reference_context_role": "reference_item",
            "reference_text": "[1] A. Author. Stable paper.",
            "is_reference_item": True,
        },
    )

    tex = OriginalLikeIRLatexRenderer().render(_document([node]), _tree(RenderRole.REFERENCES, ["ref1"]), _style())

    assert r"\begin{thebibliography}{99}" in tex
    assert r"\bibitem{ref_1} A. Author. Stable paper." in tex
    assert "[1] A. Author" not in tex


def test_reference_items_list_renders_multiple_stable_items_and_skips_empty() -> None:
    node = _reference_node(
        "refblock",
        metadata={
            "reference_items": [
                {"text": "[1] First paper."},
                {"text": ""},
                {"raw_text": "[2] Second paper."},
            ]
        },
    )

    tex = OriginalLikeIRLatexRenderer().render(_document([node]), _tree(RenderRole.REFERENCES, ["refblock"]), _style())

    assert tex.count(r"\bibitem") == 2
    assert r"\bibitem{ref_1} First paper." in tex
    assert r"\bibitem{ref_2} Second paper." in tex


def test_unsafe_reference_text_is_escaped_compile_safely() -> None:
    node = _reference_node(
        "ref1",
        "[1] Smith & Co. 50% result_with_hash #1 {raw} $bad$ https://example.com/a_b",
    )

    tex = OriginalLikeIRLatexRenderer().render(_document([node]), _tree(RenderRole.REFERENCES, ["ref1"]), _style())

    assert r"Smith \& Co. 50\%" in tex
    assert r"result\_with\_hash \#1 \{raw\}" in tex
    assert "$bad$" not in tex
    assert r"a\_b" in tex


def test_reference_heading_is_not_rendered_as_bibitem() -> None:
    heading = _reference_node(
        "ref_heading",
        "References",
        metadata={"mineru_reference_role": "reference_heading", "reference_context_role": "reference_heading"},
    )

    tex = OriginalLikeIRLatexRenderer().render(_document([heading]), _tree(RenderRole.REFERENCES, ["ref_heading"]), _style())

    assert r"\bibitem" not in tex


def test_ordinary_body_citation_is_not_promoted_to_reference_list() -> None:
    node = DocumentNode(
        "body1",
        BlockType.TEXT,
        "See [1] for implementation details.",
        0,
        [BBox(100, 100, 900, 130)],
        0,
    )

    tex = OriginalLikeIRLatexRenderer().render(_document([node]), _tree(RenderRole.PARAGRAPH, ["body1"]), _style())

    assert r"\begin{thebibliography}" not in tex
    assert "See [1]" in tex


def test_ordinary_list_is_not_promoted_to_reference_list() -> None:
    node = DocumentNode(
        "list1",
        BlockType.LIST,
        "1. Initialize parameters.",
        0,
        [BBox(100, 100, 900, 130)],
        0,
    )

    tex = OriginalLikeIRLatexRenderer().render(_document([node]), _tree(RenderRole.LIST, ["list1"]), _style())

    assert r"\begin{thebibliography}" not in tex


def test_duplicate_reference_text_is_rendered_once() -> None:
    first = _reference_node("ref1", "[1] A. Author. Same paper.")
    second = _reference_node("ref2", "[1] A. Author. Same paper.")

    tex = OriginalLikeIRLatexRenderer().render(_document([first, second]), _tree(RenderRole.REFERENCES, ["ref1", "ref2"]), _style())

    assert tex.count(r"\bibitem") == 1
