from __future__ import annotations

from src.generation.citations import CitationResolver, expand_citation_labels, strip_reference_label
from src.generation.ir_renderer import IRLatexRenderConfig, OriginalLikeIRLatexRenderer
from src.generation.latex_renderer import render_inline_math
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


def build_document() -> DocumentIR:
    nodes = [
        DocumentNode(
            node_id="n0",
            node_type=BlockType.TITLE,
            text="Demo Paper",
            page_idx=0,
            bboxes=[BBox(100, 20, 900, 60)],
            reading_index=0,
            spans=[StyleSpan(text="Demo Paper", font_name="Times-Bold", font_size=16, is_bold=True)],
        ),
        DocumentNode(
            node_id="n1",
            node_type=BlockType.TEXT,
            text="Prior work [1-2] shows this.",
            page_idx=0,
            bboxes=[BBox(100, 100, 500, 120)],
            reading_index=1,
            spans=[StyleSpan(text="Prior work [1-2] shows this.", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="n2",
            node_type=BlockType.REFERENCE,
            text="[1] First paper.\n[2] Second paper.",
            page_idx=0,
            bboxes=[BBox(100, 800, 900, 860)],
            reading_index=2,
            spans=[StyleSpan(text="[1] First paper.", font_name="Times-Roman", font_size=9)],
        ),
    ]
    return DocumentIR(
        doc_id="demo",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )


def build_two_column_styled_document() -> DocumentIR:
    nodes = [
        DocumentNode(
            node_id="h1",
            node_type=BlockType.TITLE,
            text="1 Introduction",
            page_idx=0,
            bboxes=[BBox(70, 60, 930, 90)],
            reading_index=0,
            spans=[StyleSpan(text="1 Introduction", font_name="Times-Bold", font_size=14, is_bold=True)],
            features={"heading_level": 1},
        ),
        DocumentNode(
            node_id="l0",
            node_type=BlockType.TEXT,
            text="Left body.",
            page_idx=0,
            bboxes=[BBox(70, 120, 450, 140)],
            reading_index=1,
            spans=[StyleSpan(text="Left body.", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="r0",
            node_type=BlockType.TEXT,
            text="Right body.",
            page_idx=0,
            bboxes=[BBox(550, 120, 930, 140)],
            reading_index=2,
            spans=[StyleSpan(text="Right body.", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="l1",
            node_type=BlockType.TEXT,
            text="Styled [1] and code.",
            page_idx=1,
            bboxes=[BBox(70, 120, 450, 140)],
            reading_index=3,
            spans=[
                StyleSpan(text="Styled ", font_name="Times-Bold", font_size=10, is_bold=True),
                StyleSpan(text="[1]", font_name="Times-Roman", font_size=10),
                StyleSpan(text=" and ", font_name="Times-Roman", font_size=10),
                StyleSpan(text="code", font_name="Courier", font_size=10, is_inline_code=True),
                StyleSpan(text=" plus ", font_name="Times-Roman", font_size=10),
                StyleSpan(text="x_i", font_name="Math", font_size=10, is_inline_math=True),
                StyleSpan(text=".", font_name="Times-Roman", font_size=10),
            ],
        ),
        DocumentNode(
            node_id="r1",
            node_type=BlockType.TEXT,
            text="Right body 2.",
            page_idx=1,
            bboxes=[BBox(550, 120, 930, 140)],
            reading_index=4,
            spans=[StyleSpan(text="Right body 2.", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="ref",
            node_type=BlockType.REFERENCE,
            text="[1] Keyed paper.",
            page_idx=1,
            bboxes=[BBox(70, 850, 930, 880)],
            reading_index=5,
            metadata={"reference_items": [{"label": "1", "citation_key": "smith2024", "text": "[1] Keyed paper."}]},
            spans=[StyleSpan(text="[1] Keyed paper.", font_name="Times-Roman", font_size=9)],
        ),
    ]
    return DocumentIR(
        doc_id="styled",
        pages=[
            PageIR(page_idx=0, width=1000, height=1000, node_ids=["h1", "l0", "r0"]),
            PageIR(page_idx=1, width=1000, height=1000, node_ids=["l1", "r1", "ref"]),
        ],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )


def test_style_profile_extractor_estimates_global_profile():
    profile = StyleProfileExtractor().extract(build_document())

    assert profile.page_layout["page_width"] == 1000.0
    assert profile.renderer_options["body_font_size"] == 10.0
    assert profile.renderer_options["font_setup"]["main_font"] == "TeX Gyre Termes"
    assert profile.renderer_options["font_setup"]["requires_engine"] == "xelatex_or_lualatex"
    assert profile.role_styles["heading"]["relative_font_size"] == 1.6
    assert profile.role_styles["body"]["font_class"] == "serif"
    assert profile.renderer_options["bibliography"]["strip_source_labels"] is True
    assert profile.renderer_options["geometry_options"]["paperwidth"].endswith("pt")
    assert profile.renderer_options["font_clusters"][0]["font_size"] == 16.0
    assert profile.renderer_options["role_font_clusters"]["bibliography"][0]["font_size"] == 9.0


def test_style_profile_extractor_records_columns_and_spacing():
    profile = StyleProfileExtractor().extract(build_two_column_styled_document())

    assert profile.page_layout["column_count"] == 2
    assert profile.page_layout["column_mode"] == "two_column"
    assert "twocolumn" in profile.documentclass_options
    assert profile.renderer_options["display_math_spacing"]["above"] == 8.0
    assert profile.renderer_options["list_spacing"]["itemsep"] == 2.5


def test_style_profile_keeps_header_footer_out_of_body_style():
    document = build_document()
    document = DocumentIR(
        doc_id=document.doc_id,
        pages=document.pages,
        nodes=[
            *document.nodes,
            DocumentNode(
                node_id="hf",
                node_type=BlockType.HEADER_FOOTER,
                text="5",
                page_idx=0,
                bboxes=[BBox(490, 960, 510, 975)],
                reading_index=3,
                spans=[StyleSpan(text="5", font_name="Times-Roman", font_size=6)],
            ),
        ],
        reading_order=[*document.reading_order, "hf"],
    )

    profile = StyleProfileExtractor().extract(document)

    assert profile.renderer_options["body_font_size"] == 10.0
    assert profile.role_styles["header_footer"]["font_size"] == 6.0
    assert profile.renderer_options["header_footer"]["render_by_default"] is False
    assert profile.renderer_options["header_footer"]["page_number"]["enabled"] is False


def test_style_profile_infers_global_header_footer_and_renderer_emits_fancyhdr():
    nodes = []
    pages = []
    reading_order = []
    for page_idx in range(3):
        header_id = f"h{page_idx}"
        body_id = f"b{page_idx}"
        page_id = f"p{page_idx}"
        pages.append(PageIR(page_idx=page_idx, width=1000, height=1000, node_ids=[header_id, body_id, page_id]))
        nodes.extend(
            [
                DocumentNode(
                    node_id=header_id,
                    node_type=BlockType.HEADER_FOOTER,
                    text="Journal of Useful Results",
                    page_idx=page_idx,
                    bboxes=[BBox(350, 20, 650, 35)],
                    reading_index=page_idx * 3,
                    spans=[StyleSpan(text="Journal of Useful Results", font_name="Times-Roman", font_size=7)],
                ),
                DocumentNode(
                    node_id=body_id,
                    node_type=BlockType.TEXT,
                    text=f"Body page {page_idx + 1}.",
                    page_idx=page_idx,
                    bboxes=[BBox(100, 120, 900, 150)],
                    reading_index=page_idx * 3 + 1,
                    spans=[StyleSpan(text=f"Body page {page_idx + 1}.", font_name="Times-Roman", font_size=10)],
                ),
                DocumentNode(
                    node_id=page_id,
                    node_type=BlockType.HEADER_FOOTER,
                    text=str(page_idx + 1),
                    page_idx=page_idx,
                    bboxes=[BBox(490, 960, 510, 975)],
                    reading_index=page_idx * 3 + 2,
                    spans=[StyleSpan(text=str(page_idx + 1), font_name="Times-Roman", font_size=7)],
                ),
            ]
        )
        reading_order.extend([header_id, body_id, page_id])
    document = DocumentIR(doc_id="hf", pages=pages, nodes=nodes, reading_order=reading_order)
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="hf",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["body"]),
            RenderTreeNode(render_id="body", role=RenderRole.PARAGRAPH, source_node_ids=["b0"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile)

    assert profile.renderer_options["header_footer"]["render_by_default"] is True
    assert profile.renderer_options["header_footer"]["header"]["enabled"] is True
    assert profile.renderer_options["header_footer"]["page_number"]["enabled"] is True
    assert r"\usepackage{fancyhdr}" in tex
    assert r"\fancyhead[C]{Journal of Useful Results}" in tex
    assert r"\fancyfoot[C]{\thepage}" in tex
    assert "Journal of Useful Results" not in tex.split(r"\begin{document}", 1)[1]


def test_citation_resolver_strips_reference_labels_and_rewrites_body_markers():
    resolution = CitationResolver().resolve_document(build_document())

    assert [entry.text for entry in resolution.entries] == ["First paper.", "Second paper."]
    assert resolution.text_by_node_id["n1"] == r"Prior work \cite{ref_1,ref_2} shows this."
    assert strip_reference_label("【3】 Third paper.") == "Third paper."
    assert expand_citation_labels("1, 3-5") == ["1", "3", "4", "5"]


def test_citation_resolver_prefers_reference_item_keys():
    resolution = CitationResolver().resolve_document(build_two_column_styled_document())

    assert resolution.entries[0].key == "smith2024"
    assert resolution.text_by_node_id["l1"] == r"Styled \cite{smith2024} and code."


def test_inline_math_renderer_rejects_lone_symbol_font_braces():
    assert render_inline_math("{") == r"\{"
    assert render_inline_math("}") == r"\}"
    assert render_inline_math(r"x_i") == r"$x_i$"
    assert render_inline_math("x∈X⊆R") == r"$x\inX\subseteqR$"


def test_original_like_ir_renderer_uses_style_and_citation_resolution():
    document = build_document()
    profile = StyleProfileExtractor().extract(document)
    citations = CitationResolver().resolve_document(document)
    tree = RenderTreeIR(
        doc_id="demo",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["r1", "r2"]),
            RenderTreeNode(render_id="r1", role=RenderRole.PARAGRAPH, source_node_ids=["n1"]),
            RenderTreeNode(render_id="r2", role=RenderRole.REFERENCES, source_node_ids=["n2"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile, citations)

    assert r"\cite{ref_1,ref_2}" in tex
    assert r"\bibitem{ref_1} First paper." in tex
    assert "[1] First paper" not in tex


def test_original_like_ir_renderer_groups_repeated_reference_nodes_once():
    document = build_document()
    profile = StyleProfileExtractor().extract(document)
    citations = CitationResolver().resolve_document(document)
    tree = RenderTreeIR(
        doc_id="demo",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["refs_title", "refs_item"]),
            RenderTreeNode(render_id="refs_title", role=RenderRole.REFERENCES, text="References"),
            RenderTreeNode(render_id="refs_item", role=RenderRole.REFERENCE_ITEM, source_node_ids=["n2"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile, citations)

    assert tex.count(r"\begin{thebibliography}") == 1
    assert tex.count(r"\bibitem{ref_1}") == 1
    assert tex.count(r"\bibitem{ref_2}") == 1


def test_original_like_ir_renderer_consumes_style_profile_and_spans():
    document = build_two_column_styled_document()
    profile = StyleProfileExtractor().extract(document)
    citations = CitationResolver().resolve_document(document)
    tree = RenderTreeIR(
        doc_id="styled",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["p1", "refs"]),
            RenderTreeNode(render_id="p1", role=RenderRole.PARAGRAPH, source_node_ids=["l1"]),
            RenderTreeNode(render_id="refs", role=RenderRole.REFERENCES, source_node_ids=["ref"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile, citations)

    assert r"\documentclass[twocolumn]{article}" in tex
    assert r"\geometry{" in tex
    assert r"\setlength{\parindent}" in tex
    assert r"\textbf{Styled }" in tex
    assert r"\cite{smith2024}" in tex
    assert r"\texttt{code}" in tex
    assert r"$x_i$" in tex
    assert r"\bibitem{smith2024} Keyed paper." in tex


def test_original_like_ir_renderer_can_emit_optional_fontspec_setup():
    document = build_two_column_styled_document()
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="styled",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["p1"]),
            RenderTreeNode(render_id="p1", role=RenderRole.PARAGRAPH, source_node_ids=["l1"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(enable_fontspec=True)).render(document, tree, profile)

    assert r"\usepackage{fontspec}" in tex
    assert r"\setmainfont{TeX Gyre Termes}" in tex
    assert r"\setmonofont{TeX Gyre Cursor}" in tex


def test_original_like_ir_renderer_preserves_span_font_family_and_scripts_from_features():
    nodes = [
        DocumentNode(
            node_id="body",
            node_type=BlockType.TEXT,
            text="Body baseline.",
            page_idx=0,
            bboxes=[BBox(100, 40, 400, 60)],
            reading_index=0,
            spans=[StyleSpan(text="Body baseline.", font_name="Times-Roman", font_size=10, bbox=BBox(100, 40, 400, 60))],
            features={"style_baseline_size": 10.0},
        ),
        DocumentNode(
            node_id="styled",
            node_type=BlockType.TEXT,
            text="x2 Sans",
            page_idx=0,
            bboxes=[BBox(100, 100, 260, 120)],
            reading_index=1,
            spans=[
                StyleSpan(text="x", font_name="Times-Roman", font_size=10, bbox=BBox(100, 100, 112, 120)),
                StyleSpan(text="2", font_name="Times-Roman", font_size=7, bbox=BBox(113, 96, 120, 104)),
                StyleSpan(text=" Sans", font_name="Helvetica", font_size=10, bbox=BBox(125, 100, 260, 120)),
            ],
            features={"style_baseline_size": 10.0},
        ),
    ]
    document = DocumentIR(
        doc_id="span_style",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["body", "styled"])],
        nodes=nodes,
        reading_order=["body", "styled"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="span_style",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["p"]),
            RenderTreeNode(render_id="p", role=RenderRole.PARAGRAPH, source_node_ids=["styled"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile)

    assert r"x\raisebox{0.55ex}{\scriptsize 2}" in tex
    assert r"\textsf{ Sans}" in tex


def test_original_like_ir_renderer_reconstructs_author_block_from_span_lines():
    nodes = [
        DocumentNode(
            node_id="title",
            node_type=BlockType.TITLE,
            text="Learning in Multiple Spaces",
            page_idx=0,
            bboxes=[BBox(120, 40, 880, 80)],
            reading_index=0,
            spans=[StyleSpan(text="Learning in Multiple Spaces", font_name="Times-Bold", font_size=18, is_bold=True, bbox=BBox(120, 40, 880, 80))],
        ),
        DocumentNode(
            node_id="authors",
            node_type=BlockType.TEXT,
            text=(
                "Alice Doe1 Bob Roe2 1 Department of Computer Science, Example University "
                "2 Institute of Intelligent Systems alice@example.edu bob@example.edu"
            ),
            page_idx=0,
            bboxes=[BBox(170, 90, 830, 170)],
            reading_index=1,
            spans=[
                StyleSpan(text="Alice Doe1", font_name="Times-Roman", font_size=10, bbox=BBox(250, 92, 360, 104)),
                StyleSpan(text="Bob Roe2", font_name="Times-Roman", font_size=10, bbox=BBox(380, 92, 480, 104)),
                StyleSpan(text="1 Department of Computer Science, Example University", font_name="Times-Roman", font_size=8, bbox=BBox(205, 114, 650, 124)),
                StyleSpan(text="2 Institute of Intelligent Systems", font_name="Times-Roman", font_size=8, bbox=BBox(265, 132, 570, 142)),
                StyleSpan(text="alice@example.edu; bob@example.edu", font_name="Courier", font_size=8, bbox=BBox(300, 150, 560, 160)),
            ],
        ),
    ]
    document = DocumentIR(
        doc_id="front_matter",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["title", "authors"])],
        nodes=nodes,
        reading_order=["title", "authors"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="front_matter",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["title", "authors"]),
            RenderTreeNode(render_id="title", role=RenderRole.DOCUMENT_TITLE, source_node_ids=["title"]),
            RenderTreeNode(render_id="authors", role=RenderRole.AUTHOR_BLOCK, source_node_ids=["authors"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(front_matter_mode="original_like")).render(document, tree, profile)

    assert r"\maketitle" not in tex
    assert r"\begin{minipage}{0.94\textwidth}" in tex
    assert "Alice Doe1 Bob Roe2" in tex
    assert r"{\small 1 Department of Computer Science, Example University}" in tex
    assert r"\texttt{alice@example.edu; bob@example.edu}" in tex


def test_original_like_renderer_outputs_one_table_for_grouped_fragments():
    nodes = [
        DocumentNode(
            node_id="t0",
            node_type=BlockType.TABLE,
            text="",
            page_idx=0,
            bboxes=[BBox(100, 100, 240, 500)],
            reading_index=0,
            metadata={
                "table_group_id": "table_group_p0000_0000",
                "table_group_size": 2,
                "table_group_primary": False,
                "table_group_bbox": [100, 100, 390, 500],
                "table_group_caption": "Table 1: Wide result table.",
            },
        ),
        DocumentNode(
            node_id="t1",
            node_type=BlockType.TABLE,
            text="",
            page_idx=0,
            bboxes=[BBox(245, 102, 390, 498)],
            reading_index=1,
            metadata={
                "table_group_id": "table_group_p0000_0000",
                "table_group_size": 2,
                "table_group_primary": True,
                "table_group_bbox": [100, 100, 390, 500],
                "table_group_caption": "Table 1: Wide result table.",
            },
        ),
    ]
    document = DocumentIR(
        doc_id="tables",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["t0", "t1"])],
        nodes=nodes,
        reading_order=["t0", "t1"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="tables",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["rt0", "rt1"]),
            RenderTreeNode(render_id="rt0", role=RenderRole.TABLE, source_node_ids=["t0"]),
            RenderTreeNode(render_id="rt1", role=RenderRole.TABLE, source_node_ids=["t1"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile)

    assert tex.count(r"\begin{table}[H]") == 1
    assert "TODO_TABLE_RECONSTRUCT: BBOX=(100, 100, 390, 500), ID=table_group_p0000_0000" in tex
    assert r"\caption{Table 1: Wide result table.}" in tex


def test_original_like_ir_renderer_groups_list_siblings_and_keeps_equation_inside_item():
    nodes = [
        DocumentNode(
            node_id="i1",
            node_type=BlockType.TEXT,
            text="1. Euclidean distance captures geometry.",
            page_idx=0,
            bboxes=[BBox(100, 100, 900, 120)],
            reading_index=1,
            spans=[StyleSpan(text="1. Euclidean distance captures geometry.", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="eq",
            node_type=BlockType.EQUATION,
            text=r"d_E(x,c_k)=\|f_\theta(x)-c_k\|_2",
            page_idx=0,
            bboxes=[BBox(250, 140, 750, 180)],
            reading_index=2,
        ),
        DocumentNode(
            node_id="i2",
            node_type=BlockType.TEXT,
            text="2. Cosine distance captures direction.",
            page_idx=0,
            bboxes=[BBox(100, 210, 900, 230)],
            reading_index=3,
            spans=[StyleSpan(text="2. Cosine distance captures direction.", font_name="Times-Roman", font_size=10)],
        ),
    ]
    document = DocumentIR(
        doc_id="list_eq",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["i1", "eq", "i2"])],
        nodes=nodes,
        reading_order=["i1", "eq", "i2"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="list_eq",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["ri2", "req", "ri1"]),
            RenderTreeNode(render_id="ri1", role=RenderRole.PARAGRAPH, source_node_ids=["i1"]),
            RenderTreeNode(render_id="req", role=RenderRole.DISPLAY_EQUATION, source_node_ids=["eq"]),
            RenderTreeNode(render_id="ri2", role=RenderRole.PARAGRAPH, source_node_ids=["i2"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile)

    assert tex.count(r"\begin{enumerate}") == 1
    assert tex.count(r"\item") == 2
    assert "1. Euclidean" not in tex
    assert "2. Cosine" not in tex
    assert tex.index("Euclidean distance") < tex.index(r"\[") < tex.index("Cosine distance")
