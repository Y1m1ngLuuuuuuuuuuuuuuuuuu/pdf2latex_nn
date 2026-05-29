from __future__ import annotations

from src.generation.ir_renderer import IRLatexRenderConfig
from src.generation.render_surface import render_original_like_document
from src.ir import BlockType, DocumentIR, DocumentNode, PageIR, RenderRole, RenderTreeIR, RenderTreeNode, RendererMode, StyleProfile
from src.reasoning.front_matter_extractor import FrontMatterAbstract, FrontMatterIR, FrontMatterSpan


def _node(node_id: str, text: str, reading_index: int, *, node_type: BlockType = BlockType.TEXT) -> DocumentNode:
    return DocumentNode(
        node_id=node_id,
        node_type=node_type,
        text=text,
        page_idx=0,
        bboxes=[],
        reading_index=reading_index,
        metadata={},
    )


def _render(nodes: list[DocumentNode], tree_nodes: list[RenderTreeNode], children: list[str]) -> str:
    document = DocumentIR(
        doc_id="escape_routing",
        pages=[PageIR(page_idx=0, width=600, height=800, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    tree = RenderTreeIR(
        doc_id="escape_routing",
        root_id="root",
        nodes=[RenderTreeNode("root", RenderRole.ROOT, children=children), *tree_nodes],
        document_ir_path="document_ir.json",
    )
    style = StyleProfile(profile_id="style", mode=RendererMode.ORIGINAL_LIKE)
    return render_original_like_document(
        document,
        tree,
        style=style,
        config=IRLatexRenderConfig(include_maketitle=False, front_matter_mode="none"),
        resolve_citations=False,
    )


def test_ir_paragraph_text_uses_canonical_helper() -> None:
    tex = _render(
        [_node("p", "A&B_1 uses ϕ, ∆t, and x−y.", 0)],
        [RenderTreeNode("p", RenderRole.PARAGRAPH, source_node_ids=["p"])],
        ["p"],
    )

    assert r"A\&B\_1" in tex
    assert r"\ensuremath{\phi}" in tex
    assert r"\ensuremath{\Delta}t" in tex
    assert "x-y" in tex
    assert "ϕ" not in tex
    assert "∆" not in tex
    assert "−" not in tex


def test_heading_caption_list_and_note_text_use_canonical_helper() -> None:
    nodes = [
        _node("h", "Intro_ϕ", 0, node_type=BlockType.TITLE),
        _node("c", "Figure 1: ∆ and ϕ", 1),
        _node("li", "1. Item_ϕ & ∆", 2, node_type=BlockType.LIST),
        _node("n", "note_ϕ & ∆", 3, node_type=BlockType.FOOTNOTE),
    ]
    tex = _render(
        nodes,
        [
            RenderTreeNode("h", RenderRole.SECTION, source_node_ids=["h"]),
            RenderTreeNode("c", RenderRole.CAPTION, source_node_ids=["c"]),
            RenderTreeNode("li", RenderRole.LIST_ITEM, source_node_ids=["li"]),
            RenderTreeNode("n", RenderRole.FOOTNOTE, source_node_ids=["n"]),
        ],
        ["h", "c", "li", "n"],
    )

    assert r"\section*{Intro\_\ensuremath{\phi}}" in tex
    assert r"Figure 1: \ensuremath{\Delta} and \ensuremath{\phi}" in tex
    assert r"Item\_\ensuremath{\phi} \& \ensuremath{\Delta}" in tex
    assert r"\footnote{note\_\ensuremath{\phi} \& \ensuremath{\Delta}}" in tex
    assert "∆" not in tex
    assert "ϕ" not in tex


def test_frontmatter_title_author_and_abstract_use_canonical_helper() -> None:
    nodes = [
        _node("t", "Title_ϕ & ∆", 0, node_type=BlockType.TITLE),
        _node("a", "Alice_ϕ & Bob", 1),
        _node("ab", "Abstract body ∆ and ϕ.", 2),
    ]
    document = DocumentIR(
        doc_id="escape_routing_frontmatter",
        pages=[PageIR(page_idx=0, width=600, height=800, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    tree = RenderTreeIR(
        doc_id="escape_routing_frontmatter",
        root_id="root",
        nodes=[RenderTreeNode("root", RenderRole.ROOT, children=[])],
        document_ir_path="document_ir.json",
    )
    front_matter = FrontMatterIR(
        title=FrontMatterSpan("TITLE", "Title_ϕ & ∆", ["t"], [], 0.95),
        authors=[FrontMatterSpan("AUTHOR", "Alice_ϕ & Bob", ["a"], [], 0.95)],
        affiliations=[],
        emails=[],
        notes=[],
        abstract=FrontMatterAbstract(
            title=None,
            body=FrontMatterSpan("ABSTRACT_BODY", "Abstract body ∆ and ϕ.", ["ab"], [], 0.95),
        ),
        misc=[],
        region=None,
        lines=[],
    )
    tex = render_original_like_document(
        document,
        tree,
        style=StyleProfile(profile_id="style", mode=RendererMode.ORIGINAL_LIKE),
        config=IRLatexRenderConfig(
            include_maketitle=False,
            front_matter_ir=front_matter,
            front_matter_renderer_experimental=True,
        ),
        resolve_citations=False,
    )

    assert r"\title{Title\_\ensuremath{\phi} \& \ensuremath{\Delta}}" in tex
    assert r"\author{Alice\_\ensuremath{\phi} \& Bob}" in tex
    assert r"Abstract body \ensuremath{\Delta} and \ensuremath{\phi}." in tex
    assert "∆" not in tex
    assert "ϕ" not in tex


def test_math_and_raw_latex_paths_are_protected_from_text_escape() -> None:
    tex = _render(
        [_node("m", "∆t − C", 0, node_type=BlockType.INLINE_MATH)],
        [
            RenderTreeNode("m", RenderRole.INLINE_MATH, source_node_ids=["m"]),
            RenderTreeNode("raw", RenderRole.RAW_LATEX, text=r"\customraw{ϕ_∆}"),
        ],
        ["m", "raw"],
    )

    assert r"$\Delta{}t - C$" in tex
    assert r"\customraw{ϕ_∆}" in tex


def test_verbatim_code_fallback_is_not_escaped_as_normal_text() -> None:
    tex = _render(
        [_node("code", "value_ϕ = ∆t & 1", 0, node_type=BlockType.CODE)],
        [RenderTreeNode("code", RenderRole.CODE, source_node_ids=["code"])],
        ["code"],
    )

    assert r"\begin{verbatim}" in tex
    assert "value_" in tex
    assert "& 1" in tex
    assert r"value\_" not in tex
    assert r"\& 1" not in tex
