from __future__ import annotations

from src.generation.ir_renderer import IRLatexRenderConfig
from src.generation.render_surface import render_original_like_document
from src.ir import BlockType, DocumentIR, DocumentNode, PageIR, RenderRole, RenderTreeIR, RenderTreeNode, RendererMode, StyleProfile
from src.ir.serialization import write_json
from src.reasoning.front_matter_ir_loader import load_front_matter_ir_sidecar


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


def _document() -> DocumentIR:
    nodes = [
        _node("title", "A&B_Title", 0, node_type=BlockType.TITLE),
        _node("author", "Alice & Bob", 1),
        _node("affiliation", "Example University", 2),
        _node("email", "alice_bob@example.edu", 3),
        _node("abstract", "Abstract This is the abstract body.", 4),
        _node("intro", "1 Introduction", 5, node_type=BlockType.TITLE),
        _node("body", "The body remains visible.", 6),
    ]
    return DocumentIR(
        doc_id="frontmatter_renderer_phase0",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )


def _tree() -> RenderTreeIR:
    nodes = [
        RenderTreeNode("root", RenderRole.ROOT, children=["rt", "ra", "rf", "re", "ri", "rb"]),
        RenderTreeNode("rt", RenderRole.DOCUMENT_TITLE, source_node_ids=["title"]),
        RenderTreeNode("ra", RenderRole.AUTHOR_BLOCK, source_node_ids=["author", "affiliation", "email"]),
        RenderTreeNode("rf", RenderRole.PARAGRAPH, source_node_ids=["front_note"]),
        RenderTreeNode("re", RenderRole.ABSTRACT, source_node_ids=["abstract"]),
        RenderTreeNode("ri", RenderRole.SECTION, source_node_ids=["intro"]),
        RenderTreeNode("rb", RenderRole.PARAGRAPH, source_node_ids=["body"]),
    ]
    return RenderTreeIR(
        doc_id="frontmatter_renderer_phase0",
        root_id="root",
        nodes=nodes,
        document_ir_path="document_ir.json",
    )


def _style() -> StyleProfile:
    return StyleProfile(profile_id="style", mode=RendererMode.ORIGINAL_LIKE)


def _sidecar() -> dict:
    return {
        "doc_id": "frontmatter_renderer_phase0",
        "schema_version": "frontmatter_ir_phase0_v1",
        "title": {"text": "A&B_Title", "source_v8_ids": ["title"], "confidence": "high", "evidence": []},
        "authors": [{"text": "Alice & Bob", "source_v8_ids": ["author"], "confidence": "high", "evidence": []}],
        "affiliations": [
            {"text": "Example University", "source_v8_ids": ["affiliation"], "confidence": "high", "evidence": []}
        ],
        "emails": [{"text": "alice_bob@example.edu", "source_v8_ids": ["email"], "confidence": "high", "evidence": []}],
        "orcids": [],
        "abstract": {
            "title": None,
            "body": "This is the abstract body.",
            "source_v8_ids": ["abstract"],
            "confidence": "high",
            "evidence": [],
        },
        "front_notes": [{"text": "Keywords: safe", "source_v8_ids": ["front_note"], "confidence": "high", "evidence": []}],
        "first_body_boundary": {"page_idx": 0, "source_v8_id": "intro", "reason": "first_body_heading"},
        "unassigned_frontmatter_lines": [],
    }


def test_flag_off_preserves_current_output_path() -> None:
    tex = render_original_like_document(
        _document(),
        _tree(),
        style=_style(),
        config=IRLatexRenderConfig(include_maketitle=False, front_matter_mode="none"),
        resolve_citations=False,
    )

    assert r"\maketitle" not in tex
    assert "The body remains visible." in tex


def test_frontmatter_renderer_emits_compile_safe_maketitle_and_abstract(tmp_path) -> None:
    sidecar_path = tmp_path / "frontmatter_ir.json"
    write_json(sidecar_path, _sidecar())
    front_matter = load_front_matter_ir_sidecar(sidecar_path)

    tex = render_original_like_document(
        _document(),
        _tree(),
        style=_style(),
        config=IRLatexRenderConfig(
            include_maketitle=False,
            front_matter_mode="original_like",
            front_matter_ir=front_matter,
            front_matter_renderer_experimental=True,
        ),
        resolve_citations=False,
    )

    assert r"\title{A\&B\_Title}" in tex
    assert r"\author{Alice \& Bob \\ Example University \\ \texttt{alice\_bob@example.edu}}" in tex
    assert r"\maketitle" in tex
    assert r"\begin{abstract}" in tex
    assert "This is the abstract body." in tex


def test_frontmatter_renderer_suppresses_only_consumed_source_nodes(tmp_path) -> None:
    sidecar_path = tmp_path / "frontmatter_ir.json"
    write_json(sidecar_path, _sidecar())
    front_matter = load_front_matter_ir_sidecar(sidecar_path)

    tex = render_original_like_document(
        _document(),
        _tree(),
        style=_style(),
        config=IRLatexRenderConfig(
            include_maketitle=False,
            front_matter_mode="original_like",
            front_matter_ir=front_matter,
            front_matter_renderer_experimental=True,
        ),
        resolve_citations=False,
    )

    assert tex.count("A\\&B\\_Title") == 1
    assert tex.count("Alice \\& Bob") == 1
    assert "Keywords: safe" not in tex
    assert "Introduction" in tex
    assert "The body remains visible." in tex
