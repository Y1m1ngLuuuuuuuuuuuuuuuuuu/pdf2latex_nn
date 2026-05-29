from __future__ import annotations

from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR, StyleSpan
from src.reasoning.front_matter_extractor import extract_front_matter_phase0, front_matter_ir_to_phase0_sidecar


def _node(
    node_id: str,
    text: str,
    reading_index: int,
    *,
    node_type: BlockType = BlockType.TEXT,
    bbox: BBox | None = None,
    font_size: float = 10.0,
    metadata: dict | None = None,
) -> DocumentNode:
    return DocumentNode(
        node_id=node_id,
        node_type=node_type,
        text=text,
        page_idx=0,
        bboxes=[bbox or BBox(100, 50 + reading_index * 35, 900, 70 + reading_index * 35)],
        reading_index=reading_index,
        spans=[StyleSpan(text=text, font_name="Times-Roman", font_size=font_size, bbox=bbox)],
        metadata=metadata or {},
    )


def _doc(nodes: list[DocumentNode]) -> DocumentIR:
    return DocumentIR(
        doc_id="phase0_frontmatter",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )


def _base_doc() -> DocumentIR:
    return _doc(
        [
            _node(
                "title",
                "Evidence-first PDF Reconstruction",
                0,
                node_type=BlockType.TITLE,
                font_size=18,
                metadata={
                    "model_label": "doc_title",
                    "model_label_confidence": "strong_model_label",
                    "is_document_title_candidate": True,
                    "title_negative_for_body_heading": True,
                    "front_matter_negative_for_body_heading": True,
                },
            ),
            _node(
                "authors",
                "Alice Smith, Bob Jones",
                1,
                metadata={"is_author_affiliation_candidate": True, "model_label": "text", "model_label_confidence": "strong_model_label"},
            ),
            _node(
                "aff",
                "Department of Computer Science, Example University",
                2,
                metadata={"is_author_affiliation_candidate": True, "model_label": "text", "model_label_confidence": "strong_model_label"},
            ),
            _node("email", "alice@example.edu", 3, metadata={"is_author_affiliation_candidate": True}),
            _node("abstract", "Abstract This paper studies deterministic front matter.", 4),
            _node("intro", "1 Introduction", 5, node_type=BlockType.TITLE, font_size=13),
            _node("body_heading", "2 Method", 6, node_type=BlockType.TITLE, font_size=13),
        ]
    )


def test_model_doc_title_first_page_line_becomes_title() -> None:
    front = extract_front_matter_phase0(_base_doc())

    assert front.title is not None
    assert front.title.text == "Evidence-first PDF Reconstruction"


def test_running_header_title_is_not_title() -> None:
    document = _doc(
        [
            _node(
                "header",
                "Evidence-first PDF Reconstruction",
                0,
                node_type=BlockType.TITLE,
                metadata={
                    "mineru_page_furniture_role": "page_header",
                    "is_page_header": True,
                    "should_exclude_from_heading_detection": True,
                    "should_exclude_from_visible_prose_metric": True,
                },
            ),
            _node("intro", "1 Introduction", 1, node_type=BlockType.TITLE, font_size=13),
        ]
    )

    assert extract_front_matter_phase0(document).title is None


def test_author_line_between_title_and_abstract_becomes_author() -> None:
    front = extract_front_matter_phase0(_base_doc())

    assert [span.text for span in front.authors] == ["Alice Smith, Bob Jones"]


def test_affiliation_keyword_line_becomes_affiliation() -> None:
    front = extract_front_matter_phase0(_base_doc())

    assert [span.text for span in front.affiliations] == ["Department of Computer Science, Example University"]


def test_email_line_becomes_email() -> None:
    front = extract_front_matter_phase0(_base_doc())

    assert [span.text for span in front.emails] == ["alice@example.edu"]


def test_abstract_heading_becomes_abstract_title() -> None:
    front = extract_front_matter_phase0(_base_doc())

    assert front.abstract is not None
    assert front.abstract.title is not None
    assert front.abstract.title.text == "Abstract"


def test_abstract_body_stops_at_first_body_heading() -> None:
    front = extract_front_matter_phase0(_base_doc())

    assert front.abstract is not None
    assert front.abstract.body is not None
    assert "1 Introduction" not in front.abstract.body.text
    assert "2 Method" not in front.abstract.body.text


def test_ordinary_body_heading_after_boundary_remains_body_heading() -> None:
    front = extract_front_matter_phase0(_base_doc())

    assert "body_heading" not in front.consumed_source_node_ids


def test_figure_caption_before_body_boundary_is_not_front_matter() -> None:
    document = _doc(
        [
            _node("title", "A Paper Title", 0, node_type=BlockType.TITLE, font_size=18, metadata={"model_label": "doc_title"}),
            _node("cap", "Figure 1: Overview of the method", 1),
            _node("intro", "1 Introduction", 2, node_type=BlockType.TITLE),
        ]
    )

    assert "cap" not in extract_front_matter_phase0(document).consumed_source_node_ids


def test_reference_item_is_not_front_matter() -> None:
    document = _doc(
        [
            _node("title", "A Paper Title", 0, node_type=BlockType.TITLE, font_size=18, metadata={"model_label": "doc_title"}),
            _node("ref", "[1] A. Smith, An article, 2024", 1),
            _node("intro", "1 Introduction", 2, node_type=BlockType.TITLE),
        ]
    )

    assert "ref" not in extract_front_matter_phase0(document).consumed_source_node_ids


def test_sidecar_does_not_carry_renderer_or_graph_mutation_fields() -> None:
    front = extract_front_matter_phase0(_base_doc())
    sidecar = front_matter_ir_to_phase0_sidecar("doc", front)

    payload = str(sidecar).casefold()
    assert "renderer" not in payload
    assert "graph" not in payload
