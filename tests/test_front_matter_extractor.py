from __future__ import annotations

from src.generation.ir_renderer import IRLatexRenderConfig, OriginalLikeIRLatexRenderer
from src.generation.style_profile import StyleProfileExtractor
from src.ir import (
    BBox,
    BlockType,
    DocumentIR,
    DocumentNode,
    PageIR,
    RenderRole,
    RenderTreeIR,
    RenderTreeNode,
    StyleSpan,
)
from src.reasoning.front_matter_extractor import extract_front_matter


def _node(
    node_id: str,
    text: str,
    reading_index: int,
    *,
    node_type: BlockType = BlockType.TEXT,
    bbox: BBox | None = None,
    font_size: float = 10.0,
    bold: bool = False,
    metadata: dict | None = None,
) -> DocumentNode:
    return DocumentNode(
        node_id=node_id,
        node_type=node_type,
        text=text,
        page_idx=0,
        bboxes=[bbox or BBox(100, 40 + reading_index * 35, 900, 60 + reading_index * 35)],
        reading_index=reading_index,
        spans=[StyleSpan(text=text, font_name="Times-Bold" if bold else "Times-Roman", font_size=font_size, is_bold=bold)],
        metadata=metadata or {},
    )


def _front_matter_document() -> DocumentIR:
    nodes = [
        _node(
            "title",
            "A Tiny Reconstruction Study",
            0,
            node_type=BlockType.TITLE,
            bbox=BBox(80, 40, 920, 70),
            font_size=18,
            bold=True,
            metadata={"layout_role": "document_title"},
        ),
        _node(
            "authors",
            "Alice Smith, Bob Jones",
            1,
            bbox=BBox(220, 86, 780, 106),
            font_size=11,
            metadata={"layout_role": "author_block"},
        ),
        _node(
            "aff",
            "Department of Computer Science, Example University",
            2,
            bbox=BBox(180, 116, 820, 136),
            font_size=10,
            metadata={"layout_role": "affiliation"},
        ),
        _node(
            "email",
            "{alice,bob}@example.edu",
            3,
            bbox=BBox(280, 146, 720, 166),
            font_size=10,
            metadata={"layout_role": "email"},
        ),
        _node(
            "abstract",
            "Abstract This paper reconstructs visible front matter.",
            4,
            bbox=BBox(100, 190, 900, 225),
            font_size=10,
            metadata={"layout_role": "abstract"},
        ),
        _node("intro", "1 Introduction", 5, node_type=BlockType.TITLE, bbox=BBox(100, 260, 900, 285), font_size=13, bold=True),
        _node("body", "Example University appears here as ordinary body text.", 6, bbox=BBox(100, 300, 900, 330)),
    ]
    return DocumentIR(
        doc_id="front_matter_demo",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )


def test_rule_based_front_matter_extractor_groups_core_roles() -> None:
    front = extract_front_matter(_front_matter_document())

    assert front.title is not None
    assert front.title.text == "A Tiny Reconstruction Study"
    assert [span.text for span in front.authors] == ["Alice Smith, Bob Jones"]
    assert [span.text for span in front.affiliations] == ["Department of Computer Science, Example University"]
    assert [span.text for span in front.emails] == ["{alice,bob}@example.edu"]
    assert front.abstract is not None
    assert front.abstract.body is not None
    assert front.abstract.body.text == "This paper reconstructs visible front matter."
    assert "body" not in front.consumed_source_node_ids


def test_body_institution_text_does_not_reenter_front_matter_after_body_start() -> None:
    front = extract_front_matter(_front_matter_document())

    all_front_text = "\n".join(span.text for span in front.all_spans())
    assert "Example University appears here as ordinary body text" not in all_front_text


def test_two_line_title_merges_before_author_block() -> None:
    document = _front_matter_document()
    nodes = [
        _node(
            "title_0",
            "TEXOCR: Advancing Document OCR Models",
            0,
            node_type=BlockType.TITLE,
            bbox=BBox(70, 40, 930, 70),
            font_size=18,
            bold=True,
            metadata={"layout_role": "document_title"},
        ),
        _node(
            "title_1",
            "for Compilable Page-to-LaTeX Reconstruction",
            1,
            node_type=BlockType.TITLE,
            bbox=BBox(80, 72, 920, 100),
            font_size=18,
            bold=True,
            metadata={"layout_role": "document_title"},
        ),
        *[
            DocumentNode(
                **{
                    **node.__dict__,
                    "reading_index": node.reading_index + 2,
                }
            )
            for node in document.nodes
            if node.node_id != "title"
        ],
    ]
    merged_doc = DocumentIR(
        doc_id="two_line_title",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )

    front = extract_front_matter(merged_doc)

    assert front.title is not None
    assert front.title.text == (
        "TEXOCR: Advancing Document OCR Models\n"
        "for Compilable Page-to-LaTeX Reconstruction"
    )
    assert front.authors
    assert front.authors[0].text == "Alice Smith, Bob Jones"


def test_renderer_consumes_extracted_front_matter_once() -> None:
    document = _front_matter_document()
    tree = RenderTreeIR(
        doc_id=document.doc_id,
        document_ir_path="document_ir.json",
        root_id="root",
        nodes=[
            RenderTreeNode(
                render_id="root",
                role=RenderRole.ROOT,
                children=["title", "authors", "aff", "email", "abstract", "intro", "body"],
            ),
            *[
                RenderTreeNode(
                    render_id=node.node_id,
                    role=RenderRole.SECTION if node.node_id == "intro" else RenderRole.PARAGRAPH,
                    source_node_ids=[node.node_id],
                )
                for node in document.nodes
            ],
        ],
    )
    style = StyleProfileExtractor().extract(document)
    latex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(front_matter_mode="original_like")).render(document, tree, style)

    assert latex.count("A Tiny Reconstruction Study") == 1
    assert latex.count("Alice Smith") == 1
    assert latex.count("Department of Computer Science") == 1
    assert latex.count(r"\begin{abstract}") == 1
    assert latex.count("This paper reconstructs visible front matter.") == 1
    assert "Example University appears here as ordinary body text." in latex
