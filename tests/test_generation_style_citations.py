from __future__ import annotations

from dataclasses import replace

import pytest

from src.generation.citations import CitationResolver, expand_citation_labels, sanitize_bbl_bibitem_keys, strip_reference_label
from src.generation.front_matter import render_author_block_original_like
from src.generation.ir_renderer import IRLatexRenderConfig, OriginalLikeIRLatexRenderer
from src.generation.latex_renderer import list_environment_for_text, render_equation, render_inline_math, render_text_with_inline_latex
from src.generation.render_surface import render_original_like_document
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
    assert profile.page_layout["column_mode"] == "mixed"
    assert "twocolumn" not in profile.documentclass_options
    assert profile.renderer_options["display_math_spacing"]["above"] > 0
    assert profile.renderer_options["list_spacing"]["itemsep"] > 0


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


def test_citation_resolver_infers_author_year_keys_and_labels():
    nodes = [
        DocumentNode(
            node_id="body",
            node_type=BlockType.TEXT,
            text="Smith et al. (2020) introduced it; later work (Doe, 2021; Roe and Poe, 2022) extended it.",
            page_idx=0,
            bboxes=[BBox(100, 100, 900, 130)],
            reading_index=0,
        ),
        DocumentNode(
            node_id="refs",
            node_type=BlockType.REFERENCE,
            text="\n".join(
                [
                    "Smith, J., Doe, A. (2020). First author-year paper.",
                    "Doe, B. (2021). Second paper.",
                    "Roe, C. and Poe, D. (2022). Third paper.",
                ]
            ),
            page_idx=0,
            bboxes=[BBox(100, 800, 900, 900)],
            reading_index=1,
        ),
    ]
    document = DocumentIR(
        doc_id="author_year",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["body", "refs"])],
        nodes=nodes,
        reading_order=["body", "refs"],
    )

    resolution = CitationResolver().resolve_document(document)

    assert resolution.citation_style == "author_year"
    assert [entry.key for entry in resolution.entries] == ["Smith2020", "Doe2021", "Roe2022"]
    assert resolution.entries[0].display_label == "Smith, 2020"
    assert resolution.text_by_node_id["body"] == (
        r"\cite{Smith2020} introduced it; later work \cite{Doe2021,Roe2022} extended it."
    )


def test_inline_math_renderer_rejects_lone_symbol_font_braces():
    assert render_inline_math("{") == r"\{"
    assert render_inline_math("}") == r"\}"
    assert render_inline_math(r"x_i") == r"$x_i$"
    assert render_inline_math("x∈X⊆R") == r"$x\in X\subseteqR$"


def test_text_renderer_protects_bare_latex_math_fragments():
    tex = render_text_with_inline_latex(r'Caption says \\mathrm { p } ^ { \\mathrm { , } } is the number.')

    assert r"$\mathrm{p}^{\mathrm{,}}$" in tex
    assert r"\textbackslash{}" not in tex


def test_display_equation_preserves_numbering_and_align_semantics():
    numbered = render_equation(r"E = mc^2 (1)")
    aligned = render_equation("a &= b\\\\\nc &= d")

    assert numbered == "\\begin{equation}\n" + r"E = mc^2 \tag{1}" + "\n\\end{equation}"
    assert aligned.startswith(r"\begin{align}")
    assert aligned.endswith(r"\end{align}")


def test_display_equation_scales_oversized_array_without_losing_tag():
    body = (
        r"\begin{array}{rlr}"
        r"{-\frac{\partial}{\partial y_i}\mathcal{L}=\sum_{j\ne i}w_{ij}\frac{\partial}{\partial y_i}\log v_{ij}}\\"
        r"&=&{\sum_{j\ne i}w_{ij}\cdot\frac{1}{v_{ij}}\cdot(-2abv_{ij}^{2}\|y_i-y_j\|_2^{2(b-1)})(y_i-y_j)}\\"
        r"&=&{-\sum_{j\ne i}2ab\|y_i-y_j\|_2^{2(b-1)}v_{ij}w_{ij}(y_i-y_j)+\sum_{j\ne i}\frac{2b}{\|y_i-y_j\|_2^2}v_{ij}(1-w_{ij})(y_i-y_j)}"
        r"\end{array}\tag{A.4}"
    )

    rendered = render_equation(body, label="eq:A.4")

    assert r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{$\displaystyle \begin{array}" in rendered
    assert r"\tag{A.4}" in rendered
    assert r"\label{eq:A.4}" in rendered


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


def test_canonical_render_surface_builds_style_and_citations():
    document = build_document()
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

    tex = render_original_like_document(document, tree)

    assert r"\cite{ref_1,ref_2}" in tex
    assert r"\bibitem{ref_1} First paper." in tex


def test_cross_reference_replacement_preserves_space_before_following_word():
    figure = DocumentNode(
        node_id="fig5",
        node_type=BlockType.FIGURE,
        text="Figure 5: Results.",
        page_idx=0,
        bboxes=[BBox(100, 100, 900, 500)],
        reading_index=0,
        metadata={"figure_caption": "Figure 5: Results."},
    )
    document = DocumentIR(
        doc_id="xref",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["fig5"])],
        nodes=[figure],
        reading_order=["fig5"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="xref",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["p"]),
            RenderTreeNode(render_id="p", role=RenderRole.PARAGRAPH, text="Figure 5presents the result."),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert r"Figure \ref{fig:5} presents the result." in tex


def test_decimal_section_prefix_is_not_treated_as_ordered_list():
    assert list_environment_for_text("0.1 Playing Go -- possible representations.") is None
    assert list_environment_for_text("3.2. Logic Variable Considerations.") is None
    assert list_environment_for_text("1. A real list item.") == "enumerate"


def test_title_like_numbered_node_does_not_open_enumerate():
    title = DocumentNode(
        node_id="title",
        node_type=BlockType.TITLE,
        text="1. Introduction",
        page_idx=0,
        bboxes=[BBox(100, 100, 400, 130)],
        reading_index=0,
        spans=[StyleSpan(text="1. Introduction", font_name="Times-Bold", font_size=14, is_bold=True)],
    )
    document = DocumentIR(
        doc_id="numbered_title",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["title"])],
        nodes=[title],
        reading_order=["title"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="numbered_title",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["p"]),
            # Deliberately route the title through a paragraph role to exercise
            # the renderer safety net rather than the happy-path heading role.
            RenderTreeNode(render_id="p", role=RenderRole.PARAGRAPH, source_node_ids=["title"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert r"\begin{enumerate}" not in tex
    assert "Introduction" in tex


def test_run_in_heading_splits_bold_heading_from_body():
    node = DocumentNode(
        node_id="runin",
        node_type=BlockType.TEXT,
        text=(
            "1.1. Playing Go -- possible representations. "
            "As an illustration, let's consider a game engine."
        ),
        page_idx=0,
        bboxes=[BBox(100, 100, 900, 150)],
        reading_index=0,
        spans=[
            StyleSpan(text="1.1.", font_name="Times-Roman", font_size=10, is_bold=False),
            StyleSpan(text="Playing Go -- possible representations.", font_name="Times-Bold", font_size=10, is_bold=True),
            StyleSpan(
                text="As an illustration, let's consider a game engine.",
                font_name="Times-Roman",
                font_size=10,
                is_bold=False,
            ),
        ],
        features={"run_in_heading_level": 2},
        metadata={"layout_role": "heading", "run_in_heading_level": 2},
    )
    document = DocumentIR(
        doc_id="runin",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["runin"])],
        nodes=[node],
        reading_order=["runin"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="runin",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["s"]),
            RenderTreeNode(render_id="s", role=RenderRole.SUBSECTION, source_node_ids=["runin"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert r"\subsection{Playing Go -- possible representations.}" in tex
    assert "As an illustration, let's consider a game engine." in tex
    assert r"\begin{enumerate}" not in tex
    assert "As an illustration" not in tex.split(r"\subsection{", 1)[1].split("}", 1)[0]


def test_run_in_heading_uses_explicit_v7_split_when_all_spans_are_bold():
    node = DocumentNode(
        node_id="runin",
        node_type=BlockType.TEXT,
        text=(
            "0.1 Playing Go -- possible representations. "
            "As an illustration, let's consider a game engine."
        ),
        page_idx=0,
        bboxes=[BBox(100, 100, 760, 170)],
        reading_index=0,
        spans=[
            StyleSpan(
                text=(
                    "0.1 Playing Go -- possible representations. "
                    "As an illustration, let's consider a game engine."
                ),
                font_name="Times-Bold",
                font_size=10,
                is_bold=True,
            )
        ],
        features={
            "run_in_heading_level": 2,
            "run_in_heading_text": "Playing Go -- possible representations.",
            "run_in_heading_body": "As an illustration, let's consider a game engine.",
        },
        metadata={
            "layout_role": "heading",
            "run_in_heading_level": 2,
            "run_in_heading_text": "Playing Go -- possible representations.",
            "run_in_heading_body": "As an illustration, let's consider a game engine.",
        },
    )
    document = DocumentIR(
        doc_id="runin_explicit",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["runin"])],
        nodes=[node],
        reading_order=["runin"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="runin_explicit",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["s"]),
            RenderTreeNode(render_id="s", role=RenderRole.SUBSECTION, source_node_ids=["runin"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert r"\subsection{Playing Go -- possible representations.}" in tex
    assert "As an illustration, let's consider a game engine." in tex
    assert "As an illustration" not in tex.split(r"\subsection{", 1)[1].split("}", 1)[0]


def test_style_profile_body_geometry_ignores_front_matter_abstract_width():
    nodes = [
        DocumentNode(
            node_id="abs",
            node_type=BlockType.TEXT,
            text="Abstract narrow block.",
            page_idx=0,
            bboxes=[BBox(260, 120, 740, 220)],
            reading_index=0,
            spans=[StyleSpan(text="Abstract narrow block.", font_name="Times-Roman", font_size=9)],
            metadata={"layout_role": "abstract"},
        ),
        DocumentNode(
            node_id="body",
            node_type=BlockType.TEXT,
            text="Main body paragraph has the document text width.",
            page_idx=1,
            bboxes=[BBox(120, 120, 880, 170)],
            reading_index=1,
            spans=[StyleSpan(text="Main body paragraph has the document text width.", font_name="Times-Roman", font_size=10)],
        ),
    ]
    document = DocumentIR(
        doc_id="body_geometry",
        pages=[
            PageIR(page_idx=0, width=1000, height=1000, node_ids=["abs"]),
            PageIR(page_idx=1, width=1000, height=1000, node_ids=["body"]),
        ],
        nodes=nodes,
        reading_order=["abs", "body"],
    )

    profile = StyleProfileExtractor().extract(document)

    assert profile.page_layout["body_margins"]["left"] == 120
    assert profile.page_layout["body_margins"]["right"] == 120


def test_author_block_does_not_render_metadata_title_as_author_grid_cell():
    title = DocumentNode(
        node_id="title",
        node_type=BlockType.TITLE,
        text="ON CONSTRUCTING FINITE AUTOMATA BY RELATIONAL PROGRAMMING",
        page_idx=0,
        bboxes=[BBox(220, 186, 777, 218)],
        reading_index=0,
        spans=[StyleSpan(text="ON CONSTRUCTING FINITE AUTOMATA BY RELATIONAL PROGRAMMING", font_size=16, is_bold=True)],
        metadata={"layout_layer": "metadata_layer", "layout_role": "author"},
    )
    author = DocumentNode(
        node_id="author",
        node_type=BlockType.TEXT,
        text="ATTILA EGRI-NAGY AND CHRYSTOPHER L. NEHANIV",
        page_idx=0,
        bboxes=[BBox(300, 243, 694, 257)],
        reading_index=1,
        spans=[StyleSpan(text="ATTILA EGRI-NAGY AND CHRYSTOPHER L. NEHANIV", font_size=10)],
        metadata={"layout_layer": "metadata_layer", "layout_role": "author"},
    )

    rendered = render_author_block_original_like("", [title, author])

    assert "ATTILA EGRI-NAGY" in rendered
    assert "ON CONSTRUCTING FINITE AUTOMATA" not in rendered


def test_original_like_ir_renderer_emits_author_year_bibitem_optional_labels():
    nodes = [
        DocumentNode(
            node_id="body",
            node_type=BlockType.TEXT,
            text="Smith et al. (2020) introduced it.",
            page_idx=0,
            bboxes=[BBox(100, 100, 900, 130)],
            reading_index=0,
        ),
        DocumentNode(
            node_id="refs",
            node_type=BlockType.REFERENCE,
            text="Smith, J. (2020). First author-year paper.",
            page_idx=0,
            bboxes=[BBox(100, 800, 900, 850)],
            reading_index=1,
        ),
    ]
    document = DocumentIR(
        doc_id="author_year_render",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["body", "refs"])],
        nodes=nodes,
        reading_order=["body", "refs"],
    )
    profile = StyleProfileExtractor().extract(document)
    citations = CitationResolver().resolve_document(document)
    tree = RenderTreeIR(
        doc_id="author_year_render",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["p", "refs"]),
            RenderTreeNode(render_id="p", role=RenderRole.PARAGRAPH, source_node_ids=["body"]),
            RenderTreeNode(render_id="refs", role=RenderRole.REFERENCES, source_node_ids=["refs"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile, citations)

    assert r"\cite{Smith2020}" in tex
    assert r"\bibitem[Smith, 2020]{Smith2020} Smith, J. (2020). First author-year paper." in tex
    assert r"\usepackage{cite}" not in tex


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


def test_reference_columns_are_inferred_from_item_clusters_not_union_bbox():
    nodes = [
        DocumentNode(
            node_id="body",
            node_type=BlockType.TEXT,
            text="Body.",
            page_idx=0,
            bboxes=[BBox(90, 100, 910, 130)],
            reading_index=0,
            spans=[StyleSpan(text="Body.", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="ref_l0",
            node_type=BlockType.REFERENCE,
            text="[1] Left reference.",
            page_idx=1,
            bboxes=[BBox(70, 100, 450, 120)],
            reading_index=1,
            metadata={"reference_items": [{"label": "1", "text": "[1] Left reference."}]},
            spans=[StyleSpan(text="[1] Left reference.", font_name="Times-Roman", font_size=8)],
        ),
        DocumentNode(
            node_id="ref_l1",
            node_type=BlockType.REFERENCE,
            text="[2] Left reference two.",
            page_idx=1,
            bboxes=[BBox(70, 130, 450, 150)],
            reading_index=2,
            metadata={"reference_items": [{"label": "2", "text": "[2] Left reference two."}]},
            spans=[StyleSpan(text="[2] Left reference two.", font_name="Times-Roman", font_size=8)],
        ),
        DocumentNode(
            node_id="ref_r0",
            node_type=BlockType.REFERENCE,
            text="[3] Right reference.",
            page_idx=1,
            bboxes=[BBox(550, 100, 930, 120)],
            reading_index=3,
            metadata={"reference_items": [{"label": "3", "text": "[3] Right reference."}]},
            spans=[StyleSpan(text="[3] Right reference.", font_name="Times-Roman", font_size=8)],
        ),
        DocumentNode(
            node_id="ref_r1",
            node_type=BlockType.REFERENCE,
            text="[4] Right reference two.",
            page_idx=1,
            bboxes=[BBox(550, 130, 930, 150)],
            reading_index=4,
            metadata={"reference_items": [{"label": "4", "text": "[4] Right reference two."}]},
            spans=[StyleSpan(text="[4] Right reference two.", font_name="Times-Roman", font_size=8)],
        ),
    ]
    document = DocumentIR(
        doc_id="reference-columns",
        pages=[
            PageIR(page_idx=0, width=1000, height=1000, node_ids=["body"]),
            PageIR(page_idx=1, width=1000, height=1000, node_ids=["ref_l0", "ref_l1", "ref_r0", "ref_r1"]),
        ],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="reference-columns",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["body", "refs"]),
            RenderTreeNode(render_id="body", role=RenderRole.PARAGRAPH, source_node_ids=["body"]),
            RenderTreeNode(render_id="refs", role=RenderRole.REFERENCES, source_node_ids=["ref_l0", "ref_l1", "ref_r0", "ref_r1"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(
        document,
        tree,
        profile,
        CitationResolver().resolve_document(document),
    )

    assert profile.renderer_options["bibliography"]["column_mode"] == "two_column"
    assert r"\begin{multicols*}{2}" in tex
    assert tex.index(r"\begin{multicols*}{2}") < tex.index(r"\begin{thebibliography}")


def test_compact_reference_list_uses_body_column_prior_when_bbox_is_under_observed():
    reference_items = [
        {"label": str(index), "text": f"[{index}] Reference item {index}."}
        for index in range(1, 18)
    ]
    nodes = [
        DocumentNode(
            node_id="left",
            node_type=BlockType.TEXT,
            text="Left body.",
            page_idx=0,
            bboxes=[BBox(70, 100, 450, 130)],
            reading_index=0,
            spans=[StyleSpan(text="Left body.", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="right",
            node_type=BlockType.TEXT,
            text="Right body.",
            page_idx=0,
            bboxes=[BBox(550, 100, 930, 130)],
            reading_index=1,
            spans=[StyleSpan(text="Right body.", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="ref_compact",
            node_type=BlockType.REFERENCE,
            text=" ".join(item["text"] for item in reference_items),
            page_idx=1,
            bboxes=[BBox(70, 100, 455, 145)],
            reading_index=2,
            metadata={"reference_items": reference_items, "layout_band_type": "full_span"},
            spans=[StyleSpan(text="[1] Reference item 1.", font_name="Times-Roman", font_size=8)],
        ),
    ]
    document = DocumentIR(
        doc_id="compact-reference",
        pages=[
            PageIR(page_idx=0, width=1000, height=1000, node_ids=["left", "right"]),
            PageIR(page_idx=1, width=1000, height=1000, node_ids=["ref_compact"]),
        ],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="compact-reference",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["left", "right", "refs"]),
            RenderTreeNode(render_id="left", role=RenderRole.PARAGRAPH, source_node_ids=["left"]),
            RenderTreeNode(render_id="right", role=RenderRole.PARAGRAPH, source_node_ids=["right"]),
            RenderTreeNode(render_id="refs", role=RenderRole.REFERENCES, source_node_ids=["ref_compact"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(
        document,
        tree,
        profile,
        CitationResolver().resolve_document(document),
    )

    assert profile.renderer_options["bibliography"]["column_mode"] == "two_column"
    assert profile.renderer_options["bibliography"]["column_detection_reason"] == "compact_reference_body_column_fallback"
    assert r"\begin{multicols*}{2}" in tex


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

    assert r"\documentclass{article}" in tex
    assert r"\begin{multicols}{2}" in tex
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


def test_original_like_ir_renderer_anchors_footnotes_and_margin_notes():
    nodes = [
        DocumentNode(
            node_id="body",
            node_type=BlockType.TEXT,
            text="Main statement.",
            page_idx=0,
            bboxes=[BBox(100, 100, 700, 120)],
            reading_index=0,
            spans=[StyleSpan(text="Main statement.", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="fn",
            node_type=BlockType.FOOTNOTE,
            text=r"[1] Footnote with \\mathrm { p }.",
            page_idx=0,
            bboxes=[BBox(100, 900, 700, 925)],
            reading_index=1,
            spans=[StyleSpan(text=r"[1] Footnote with \\mathrm { p }.", font_name="Times-Roman", font_size=8)],
        ),
        DocumentNode(
            node_id="mn",
            node_type=BlockType.MARGIN_NOTE,
            text="Side observation.",
            page_idx=0,
            bboxes=[BBox(760, 120, 940, 170)],
            reading_index=2,
            spans=[StyleSpan(text="Side observation.", font_name="Times-Roman", font_size=8)],
        ),
    ]
    document = DocumentIR(
        doc_id="notes",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["body", "fn", "mn"])],
        nodes=nodes,
        reading_order=["body", "fn", "mn"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="notes",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["p", "fn", "mn"]),
            RenderTreeNode(render_id="p", role=RenderRole.PARAGRAPH, source_node_ids=["body"]),
            RenderTreeNode(render_id="fn", role=RenderRole.PARAGRAPH, source_node_ids=["fn"]),
            RenderTreeNode(render_id="mn", role=RenderRole.PARAGRAPH, source_node_ids=["mn"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile)

    assert r"Main statement.\footnote{Footnote with $\mathrm{p}$.}\marginpar{\footnotesize Side observation.}" in tex
    assert "[1] Footnote" not in tex
    assert tex.count(r"\footnote{") == 1


def test_original_like_ir_renderer_replaces_superscript_marker_with_matching_footnote():
    nodes = [
        DocumentNode(
            node_id="body",
            node_type=BlockType.TEXT,
            text="Claim1",
            page_idx=0,
            bboxes=[BBox(100, 100, 700, 120)],
            reading_index=0,
            spans=[
                StyleSpan(text="Claim", font_name="Times-Roman", font_size=10, bbox=BBox(100, 100, 150, 120)),
                StyleSpan(text="1", font_name="Times-Roman", font_size=6, bbox=BBox(151, 95, 157, 103)),
            ],
            features={"style_baseline_size": 10.0},
        ),
        DocumentNode(
            node_id="fn",
            node_type=BlockType.FOOTNOTE,
            text="1 Footnote body.",
            page_idx=0,
            bboxes=[BBox(100, 910, 700, 930)],
            reading_index=1,
            metadata={"footnote_marker": "1"},
        ),
    ]
    document = DocumentIR(
        doc_id="marker_note",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["body", "fn"])],
        nodes=nodes,
        reading_order=["body", "fn"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="marker_note",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["p", "fn"]),
            RenderTreeNode(render_id="p", role=RenderRole.PARAGRAPH, source_node_ids=["body"]),
            RenderTreeNode(render_id="fn", role=RenderRole.FOOTNOTE, source_node_ids=["fn"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile)

    assert r"Claim\footnote{Footnote body.}" in tex
    assert r"\raisebox{0.55ex}{\scriptsize 1}" not in tex
    assert r"\footnotetext" not in tex


def test_original_like_ir_renderer_keeps_unanchored_footnote_as_footnotetext():
    nodes = [
        DocumentNode(
            node_id="fn",
            node_type=BlockType.FOOTNOTE,
            text="* Orphan note.",
            page_idx=0,
            bboxes=[BBox(100, 900, 700, 920)],
            reading_index=0,
        ),
    ]
    document = DocumentIR(
        doc_id="orphan_note",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["fn"])],
        nodes=nodes,
        reading_order=["fn"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="orphan_note",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["fn"]),
            RenderTreeNode(render_id="fn", role=RenderRole.FOOTNOTE, source_node_ids=["fn"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile)

    assert r"\footnotetext{Orphan note.}" in tex


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
    assert r"\caption{Wide result table}" in tex


def test_original_like_ir_renderer_renders_figure_bbox_placeholder():
    node = DocumentNode(
        node_id="fig0",
        node_type=BlockType.FIGURE,
        text="Figure 1: Qualitative examples.",
        page_idx=0,
        bboxes=[BBox(100, 120, 900, 520)],
        reading_index=0,
        metadata={"figure_caption": "Figure 1: Qualitative examples."},
    )
    document = DocumentIR(
        doc_id="figures",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["fig0"])],
        nodes=[node],
        reading_order=["fig0"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="figures",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["rf0"]),
            RenderTreeNode(render_id="rf0", role=RenderRole.FIGURE, source_node_ids=["fig0"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile)

    assert r"\begin{figure}[H]" in tex
    assert "TODO_FIGURE_RECONSTRUCT: BBOX=(100, 120, 900, 520), ID=fig0" in tex
    assert r"\caption{" in tex
    assert "Qualitative examples" in tex
    assert r"\label{fig:1}" in tex


def test_original_like_ir_renderer_deduplicates_repeated_figure_group_nodes():
    node = DocumentNode(
        node_id="fig0",
        node_type=BlockType.FIGURE,
        text="Figure 1: Qualitative examples.",
        page_idx=0,
        bboxes=[BBox(100, 120, 900, 520)],
        reading_index=0,
        metadata={
            "figure_group_id": "figure_group_p0000_0000",
            "figure_group_primary": True,
            "figure_group_size": 1,
            "figure_caption": "Figure 1: Qualitative examples.",
        },
    )
    document = DocumentIR(
        doc_id="figures",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["fig0"])],
        nodes=[node],
        reading_order=["fig0"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="figures",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["rf0", "rf0_dup"]),
            RenderTreeNode(render_id="rf0", role=RenderRole.FIGURE, source_node_ids=["fig0"]),
            RenderTreeNode(render_id="rf0_dup", role=RenderRole.FIGURE, source_node_ids=["fig0"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile)

    assert tex.count(r"\begin{figure}[H]") == 1
    assert tex.count("TODO_FIGURE_RECONSTRUCT") == 1
    assert tex.count(r"\caption{Qualitative examples}") == 1


def test_original_like_ir_renderer_merges_adjacent_figures_when_only_one_has_caption():
    left = DocumentNode(
        node_id="fig_left",
        node_type=BlockType.FIGURE,
        text="",
        page_idx=0,
        bboxes=[BBox(100, 140, 430, 350)],
        reading_index=0,
    )
    right = DocumentNode(
        node_id="fig_right",
        node_type=BlockType.FIGURE,
        text="Figure 4: Mask iteration examples.",
        page_idx=0,
        bboxes=[BBox(455, 150, 820, 345)],
        reading_index=1,
        metadata={"figure_caption": "Figure 4: Mask iteration examples."},
    )
    document = DocumentIR(
        doc_id="adjacent_figures",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["fig_left", "fig_right"])],
        nodes=[left, right],
        reading_order=["fig_left", "fig_right"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="adjacent_figures",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["left", "right"]),
            RenderTreeNode(render_id="left", role=RenderRole.FIGURE, source_node_ids=["fig_left"]),
            RenderTreeNode(render_id="right", role=RenderRole.FIGURE, source_node_ids=["fig_right"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile)

    assert tex.count(r"\begin{figure}[H]") == 1
    assert tex.count("TODO_FIGURE_RECONSTRUCT") == 2
    assert tex.count(r"\caption{Mask iteration examples}") == 1


def test_original_like_ir_renderer_skips_caption_node_consumed_by_figure():
    figure = DocumentNode(
        node_id="fig0",
        node_type=BlockType.FIGURE,
        text="Figure 5: Visualization results on six datasets.",
        page_idx=0,
        bboxes=[BBox(100, 100, 900, 420)],
        reading_index=0,
        metadata={"figure_caption": "Figure 5: Visualization results on six datasets."},
    )
    caption = DocumentNode(
        node_id="cap0",
        node_type=BlockType.TEXT,
        text="Figure 5: Visualization results on six datasets.",
        page_idx=0,
        bboxes=[BBox(100, 430, 900, 470)],
        reading_index=1,
        metadata={"raw_type": "figure_caption", "layout_role": "figure_caption"},
    )
    document = DocumentIR(
        doc_id="caption_dedupe",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["fig0", "cap0"])],
        nodes=[figure, caption],
        reading_order=["fig0", "cap0"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="caption_dedupe",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["fig", "cap"]),
            RenderTreeNode(render_id="fig", role=RenderRole.FIGURE, source_node_ids=["fig0"]),
            RenderTreeNode(render_id="cap", role=RenderRole.CAPTION, source_node_ids=["cap0"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer().render(document, tree, profile)

    assert tex.count("Visualization results on six datasets") == 1
    assert tex.count(r"\caption{Visualization results on six datasets}") == 1


def test_original_like_ir_renderer_can_crop_figure_from_source_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "source.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.draw_rect(fitz.Rect(20, 20, 180, 120), color=(1, 0, 0), fill=(1, 0.9, 0.9))
    doc.save(pdf_path)
    doc.close()

    node = DocumentNode(
        node_id="fig1",
        node_type=BlockType.FIGURE,
        text="Figure 2: Cropped panel.",
        page_idx=0,
        bboxes=[BBox(100, 100, 900, 600)],
        reading_index=0,
        metadata={"figure_caption": "Figure 2: Cropped panel.", "source_pdf": str(pdf_path)},
    )
    document = DocumentIR(
        doc_id="figure_crop",
        source_pdf=str(pdf_path),
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["fig1"])],
        nodes=[node],
        reading_order=["fig1"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="figure_crop",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["rf1"]),
            RenderTreeNode(render_id="rf1", role=RenderRole.FIGURE, source_node_ids=["fig1"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(
        IRLatexRenderConfig(figure_asset_output_dir=tmp_path / "assets", figure_asset_latex_prefix="assets")
    ).render(document, tree, profile)

    assert r"\includegraphics[width=" in tex
    assert r"{assets/figure_fig1.png}" in tex
    assert (tmp_path / "assets" / "figure_fig1.png").exists()


def test_original_like_ir_renderer_prefers_existing_mineru_figure_asset(tmp_path):
    image_path = tmp_path / "mineru_image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    node = DocumentNode(
        node_id="fig_asset",
        node_type=BlockType.FIGURE,
        text="Figure 3: Extracted image.",
        page_idx=0,
        bboxes=[BBox(100, 100, 700, 500)],
        reading_index=0,
        metadata={"img_path": str(image_path), "figure_caption": "Figure 3: Extracted image."},
    )
    document = DocumentIR(
        doc_id="figure_asset",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["fig_asset"])],
        nodes=[node],
        reading_order=["fig_asset"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="figure_asset",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["rf"]),
            RenderTreeNode(render_id="rf", role=RenderRole.FIGURE, source_node_ids=["fig_asset"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(
        IRLatexRenderConfig(figure_asset_output_dir=tmp_path / "assets", figure_asset_latex_prefix="assets")
    ).render(document, tree, profile)

    assert r"\includegraphics[width=0.600\linewidth]{assets/figure_fig_asset.png}" in tex
    assert (tmp_path / "assets" / "figure_fig_asset.png").exists()
    assert "TODO_FIGURE_RECONSTRUCT" not in tex


def test_original_like_ir_renderer_wraps_mixed_double_column_bands_in_multicols():
    nodes = [
        DocumentNode(
            node_id="title",
            node_type=BlockType.TITLE,
            text="Paper Title",
            page_idx=0,
            bboxes=[BBox(100, 20, 900, 60)],
            reading_index=0,
            spans=[StyleSpan(text="Paper Title", font_name="Times-Bold", font_size=16, is_bold=True)],
            metadata={"layout_band_type": "full_span", "layout_band_global_id": 0},
        ),
        DocumentNode(
            node_id="left",
            node_type=BlockType.TEXT,
            text="Left column text.",
            page_idx=0,
            bboxes=[BBox(70, 120, 450, 145)],
            reading_index=1,
            spans=[StyleSpan(text="Left column text.", font_name="Times-Roman", font_size=10)],
            metadata={"layout_band_type": "double_column", "layout_band_global_id": 1, "layout_band_column": "left"},
        ),
        DocumentNode(
            node_id="right",
            node_type=BlockType.TEXT,
            text="Right column text.",
            page_idx=0,
            bboxes=[BBox(550, 120, 930, 145)],
            reading_index=2,
            spans=[StyleSpan(text="Right column text.", font_name="Times-Roman", font_size=10)],
            metadata={"layout_band_type": "double_column", "layout_band_global_id": 1, "layout_band_column": "right"},
        ),
        DocumentNode(
            node_id="wideeq",
            node_type=BlockType.EQUATION,
            text=r"E = mc^2",
            page_idx=0,
            bboxes=[BBox(250, 220, 750, 250)],
            reading_index=3,
            metadata={"layout_band_type": "full_span", "layout_band_global_id": 2},
        ),
    ]
    document = DocumentIR(
        doc_id="mixed",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    profile = StyleProfileExtractor().extract(document)
    profile = replace(
        profile,
        page_layout={**profile.page_layout, "column_mode": "mixed"},
        renderer_options={**profile.renderer_options, "column_mode": "mixed"},
    )
    tree = RenderTreeIR(
        doc_id="mixed",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["title", "left", "right", "eq"]),
            RenderTreeNode(render_id="title", role=RenderRole.DOCUMENT_TITLE, source_node_ids=["title"]),
            RenderTreeNode(render_id="left", role=RenderRole.PARAGRAPH, source_node_ids=["left"]),
            RenderTreeNode(render_id="right", role=RenderRole.PARAGRAPH, source_node_ids=["right"]),
            RenderTreeNode(render_id="eq", role=RenderRole.DISPLAY_EQUATION, source_node_ids=["wideeq"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert r"\usepackage{multicol}" in tex
    assert r"\begin{multicols*}{2}" not in tex
    assert tex.index("Paper Title") < tex.index(r"\begin{multicols}{2}")
    assert tex.index(r"\begin{multicols}{2}") < tex.index("Left column text.") < tex.index("Right column text.")
    assert tex.index(r"\end{multicols}") < tex.index(r"\[")


def test_original_like_ir_renderer_infers_mixed_columns_from_bbox_when_band_metadata_missing():
    nodes = [
        DocumentNode(
            node_id="title",
            node_type=BlockType.TITLE,
            text="Wide Title",
            page_idx=0,
            bboxes=[BBox(100, 20, 900, 60)],
            reading_index=0,
            spans=[StyleSpan(text="Wide Title", font_name="Times-Bold", font_size=16, is_bold=True)],
        ),
        DocumentNode(
            node_id="left",
            node_type=BlockType.TEXT,
            text="Left column text.",
            page_idx=0,
            bboxes=[BBox(70, 120, 450, 145)],
            reading_index=1,
            spans=[StyleSpan(text="Left column text.", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="right",
            node_type=BlockType.TEXT,
            text="Right column text.",
            page_idx=0,
            bboxes=[BBox(550, 120, 930, 145)],
            reading_index=2,
            spans=[StyleSpan(text="Right column text.", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="wideeq",
            node_type=BlockType.EQUATION,
            text=r"E = mc^2",
            page_idx=0,
            bboxes=[BBox(250, 220, 750, 250)],
            reading_index=3,
        ),
    ]
    document = DocumentIR(
        doc_id="mixed_inferred",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    profile = StyleProfileExtractor().extract(document)
    assert profile.page_layout["column_mode"] == "mixed"
    tree = RenderTreeIR(
        doc_id="mixed_inferred",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["title", "left", "right", "eq"]),
            RenderTreeNode(render_id="title", role=RenderRole.DOCUMENT_TITLE, source_node_ids=["title"]),
            RenderTreeNode(render_id="left", role=RenderRole.PARAGRAPH, source_node_ids=["left"]),
            RenderTreeNode(render_id="right", role=RenderRole.PARAGRAPH, source_node_ids=["right"]),
            RenderTreeNode(render_id="eq", role=RenderRole.DISPLAY_EQUATION, source_node_ids=["wideeq"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert r"\begin{multicols*}{2}" not in tex
    assert r"\begin{multicols}{2}" in tex
    assert tex.index("Wide Title") < tex.index(r"\begin{multicols}{2}") < tex.index("Left column text.")
    assert tex.index(r"\end{multicols}") < tex.index(r"\[")


def test_original_like_ir_renderer_defers_float_that_interrupts_open_sentence():
    nodes = [
        DocumentNode(
            node_id="p1",
            node_type=BlockType.TEXT,
            text="All experiments were conducted on a single NVIDIA RTX 8000 GPU",
            page_idx=0,
            bboxes=[BBox(80, 100, 450, 150)],
            reading_index=0,
            spans=[StyleSpan(text="All experiments were conducted on a single NVIDIA RTX 8000 GPU", font_name="Times-Roman", font_size=10)],
        ),
        DocumentNode(
            node_id="tbl",
            node_type=BlockType.TABLE,
            text="Table 1: Results.",
            page_idx=0,
            bboxes=[BBox(80, 160, 920, 360)],
            reading_index=1,
            metadata={"table_caption": "Table 1: Results."},
        ),
        DocumentNode(
            node_id="p2",
            node_type=BlockType.TEXT,
            text="with 48GB memory.",
            page_idx=1,
            bboxes=[BBox(80, 100, 450, 140)],
            reading_index=2,
            spans=[StyleSpan(text="with 48GB memory.", font_name="Times-Roman", font_size=10)],
        ),
    ]
    document = DocumentIR(
        doc_id="float_interrupt",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["p1", "tbl"]), PageIR(page_idx=1, width=1000, height=1000, node_ids=["p2"])],
        nodes=nodes,
        reading_order=["p1", "tbl", "p2"],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="float_interrupt",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["p1", "tbl", "p2"]),
            RenderTreeNode(render_id="p1", role=RenderRole.PARAGRAPH, source_node_ids=["p1"]),
            RenderTreeNode(render_id="tbl", role=RenderRole.TABLE, source_node_ids=["tbl"]),
            RenderTreeNode(render_id="p2", role=RenderRole.PARAGRAPH, source_node_ids=["p2"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert tex.index("NVIDIA RTX 8000 GPU") < tex.index("with 48GB memory.") < tex.index(r"\begin{table}")


def test_original_like_ir_renderer_uses_mixed_bands_for_appendix_tail_after_references():
    nodes = [
        DocumentNode(
            node_id="refs",
            node_type=BlockType.REFERENCE,
            text="[1] Ref.",
            page_idx=0,
            bboxes=[BBox(80, 100, 900, 130)],
            reading_index=0,
            metadata={"reference_items": ["[1] Ref."]},
        ),
        DocumentNode(
            node_id="app_title",
            node_type=BlockType.TITLE,
            text="Appendix A Extra Proofs",
            page_idx=1,
            bboxes=[BBox(80, 50, 920, 80)],
            reading_index=1,
            metadata={"layout_band_type": "full_span", "layout_band_global_id": 10, "_appendix_heading": True},
        ),
        DocumentNode(
            node_id="app_left",
            node_type=BlockType.TEXT,
            text="Left appendix column.",
            page_idx=1,
            bboxes=[BBox(80, 120, 450, 145)],
            reading_index=2,
            metadata={"layout_band_type": "double_column", "layout_band_global_id": 11, "layout_band_column": "left"},
        ),
        DocumentNode(
            node_id="app_right",
            node_type=BlockType.TEXT,
            text="Right appendix column.",
            page_idx=1,
            bboxes=[BBox(550, 120, 920, 145)],
            reading_index=3,
            metadata={"layout_band_type": "double_column", "layout_band_global_id": 11, "layout_band_column": "right"},
        ),
        DocumentNode(
            node_id="app_eq",
            node_type=BlockType.EQUATION,
            text=r"a=b",
            page_idx=1,
            bboxes=[BBox(250, 220, 750, 250)],
            reading_index=4,
            metadata={"layout_band_type": "full_span", "layout_band_global_id": 12},
        ),
    ]
    document = DocumentIR(
        doc_id="appendix_tail",
        pages=[
            PageIR(page_idx=0, width=1000, height=1000, node_ids=["refs"]),
            PageIR(page_idx=1, width=1000, height=1000, node_ids=["app_title", "app_left", "app_right", "app_eq"]),
        ],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    profile = StyleProfileExtractor().extract(document)
    profile = replace(
        profile,
        page_layout={**profile.page_layout, "column_mode": "mixed"},
        renderer_options={**profile.renderer_options, "column_mode": "mixed"},
    )
    tree = RenderTreeIR(
        doc_id="appendix_tail",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["refs"]),
            RenderTreeNode(render_id="refs", role=RenderRole.REFERENCES, source_node_ids=["refs"], children=["app_title", "app_left", "app_right", "app_eq"]),
            RenderTreeNode(render_id="app_title", role=RenderRole.SECTION, source_node_ids=["app_title"], attributes={"appendix_heading": True}),
            RenderTreeNode(render_id="app_left", role=RenderRole.PARAGRAPH, source_node_ids=["app_left"]),
            RenderTreeNode(render_id="app_right", role=RenderRole.PARAGRAPH, source_node_ids=["app_right"]),
            RenderTreeNode(render_id="app_eq", role=RenderRole.DISPLAY_EQUATION, source_node_ids=["app_eq"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert r"\appendix" in tex
    appendix_tail = tex[tex.index(r"\appendix") :]
    assert r"\begin{multicols*}{2}" not in appendix_tail
    assert tex.index(r"\appendix") < tex.index(r"\section{Extra Proofs}") < tex.index(r"\begin{multicols}{2}")
    assert tex.index(r"\begin{multicols}{2}") < tex.index("Left appendix column.") < tex.index("Right appendix column.")
    assert tex.index(r"\end{multicols}") < tex.index(r"\[")


def test_original_like_ir_renderer_keeps_full_width_appendix_tail_single_column_after_references():
    nodes = [
        DocumentNode(
            node_id="refs",
            node_type=BlockType.REFERENCE,
            text="[1] Ref.",
            page_idx=0,
            bboxes=[BBox(80, 100, 900, 130)],
            reading_index=0,
            metadata={"reference_items": ["[1] Ref."]},
        ),
        DocumentNode(
            node_id="app_title",
            node_type=BlockType.TITLE,
            text="Appendix A Extra Proofs",
            page_idx=1,
            bboxes=[BBox(80, 50, 920, 80)],
            reading_index=1,
            metadata={"layout_band_type": "full_span", "layout_band_global_id": 10, "_appendix_heading": True},
        ),
        DocumentNode(
            node_id="app_body",
            node_type=BlockType.TEXT,
            text="This appendix is reset to a full-width single-column layout.",
            page_idx=1,
            bboxes=[BBox(80, 120, 920, 170)],
            reading_index=2,
            metadata={"layout_band_type": "full_span", "layout_band_global_id": 11},
        ),
        DocumentNode(
            node_id="app_eq",
            node_type=BlockType.EQUATION,
            text=r"a=b",
            page_idx=1,
            bboxes=[BBox(250, 220, 750, 250)],
            reading_index=3,
            metadata={"layout_band_type": "full_span", "layout_band_global_id": 12},
        ),
    ]
    document = DocumentIR(
        doc_id="appendix_tail_single",
        pages=[
            PageIR(page_idx=0, width=1000, height=1000, node_ids=["refs"]),
            PageIR(page_idx=1, width=1000, height=1000, node_ids=["app_title", "app_body", "app_eq"]),
        ],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    profile = StyleProfileExtractor().extract(document)
    profile = replace(
        profile,
        page_layout={**profile.page_layout, "column_mode": "mixed"},
        renderer_options={**profile.renderer_options, "column_mode": "mixed"},
    )
    tree = RenderTreeIR(
        doc_id="appendix_tail_single",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["refs"]),
            RenderTreeNode(render_id="refs", role=RenderRole.REFERENCES, source_node_ids=["refs"], children=["app_title", "app_body", "app_eq"]),
            RenderTreeNode(render_id="app_title", role=RenderRole.SECTION, source_node_ids=["app_title"], attributes={"appendix_heading": True}),
            RenderTreeNode(render_id="app_body", role=RenderRole.PARAGRAPH, source_node_ids=["app_body"]),
            RenderTreeNode(render_id="app_eq", role=RenderRole.DISPLAY_EQUATION, source_node_ids=["app_eq"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    appendix_tail = tex[tex.index(r"\appendix") :]
    assert r"\appendix" in tex
    assert "This appendix is reset to a full-width single-column layout." in appendix_tail
    assert r"\begin{multicols*}{2}" not in appendix_tail


def test_original_like_ir_renderer_emits_float_equation_algorithm_labels_and_rewrites_cross_refs():
    nodes = [
        DocumentNode(
            node_id="body",
            node_type=BlockType.TEXT,
            text="See Figure 3, Fig. 3, Table 2, Equation (4), and Algorithm 1 for details.",
            page_idx=0,
            bboxes=[BBox(80, 100, 900, 130)],
            reading_index=0,
        ),
        DocumentNode(
            node_id="fig",
            node_type=BlockType.FIGURE,
            text="Figure 3: Qualitative examples.",
            page_idx=0,
            bboxes=[BBox(100, 200, 500, 400)],
            reading_index=1,
            metadata={"figure_caption": "Figure 3: Qualitative examples."},
        ),
        DocumentNode(
            node_id="tab",
            node_type=BlockType.TABLE,
            text="Table 2: Results.",
            page_idx=0,
            bboxes=[BBox(100, 430, 900, 600)],
            reading_index=2,
            metadata={"table_caption": "Table 2: Results."},
        ),
        DocumentNode(
            node_id="eq",
            node_type=BlockType.EQUATION,
            text=r"y=x \tag{4}",
            page_idx=0,
            bboxes=[BBox(250, 630, 750, 660)],
            reading_index=3,
        ),
        DocumentNode(
            node_id="alg",
            node_type=BlockType.ALGORITHM,
            text="Algorithm 1: Demo\nInput: x\nreturn x",
            page_idx=0,
            bboxes=[BBox(100, 700, 900, 850)],
            reading_index=4,
        ),
    ]
    document = DocumentIR(
        doc_id="cross_refs",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="cross_refs",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["body", "fig", "tab", "eq", "alg"]),
            RenderTreeNode(render_id="body", role=RenderRole.PARAGRAPH, source_node_ids=["body"]),
            RenderTreeNode(render_id="fig", role=RenderRole.FIGURE, source_node_ids=["fig"]),
            RenderTreeNode(render_id="tab", role=RenderRole.TABLE, source_node_ids=["tab"]),
            RenderTreeNode(render_id="eq", role=RenderRole.DISPLAY_EQUATION, source_node_ids=["eq"]),
            RenderTreeNode(render_id="alg", role=RenderRole.ALGORITHM, source_node_ids=["alg"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert r"Figure \ref{fig:3}" in tex
    assert r"Fig. \ref{fig:3}" in tex
    assert r"Table \ref{tab:2}" in tex
    assert r"Equation \ref{eq:4}" in tex
    assert r"Algorithm \ref{alg:1}" in tex
    assert r"\label{fig:3}" in tex
    assert r"\label{tab:2}" in tex
    assert r"\label{eq:4}" in tex
    assert r"\label{alg:1}" in tex


def test_source_bbl_keys_are_sanitized_to_match_repaired_cites():
    raw = r"""
\begin{thebibliography}{1}
\bibitem[Demo, 2024]{10.5555/3618408.3619982}
Demo reference.
\end{thebibliography}
"""

    repaired = sanitize_bbl_bibitem_keys(raw)

    assert r"\bibitem[Demo, 2024]{ref_10.5555_3618408.3619982}" in repaired
    assert r"\bibitem[Demo, 2024]{10.5555/3618408.3619982}" not in repaired


def test_original_like_ir_renderer_injects_referenced_float_missing_from_tree():
    nodes = [
        DocumentNode(
            node_id="body",
            node_type=BlockType.TEXT,
            text="See Figure 1 for details.",
            page_idx=0,
            bboxes=[BBox(80, 100, 900, 130)],
            reading_index=0,
        ),
        DocumentNode(
            node_id="fig",
            node_type=BlockType.FIGURE,
            text="Figure 1: Missing from decoder tree.",
            page_idx=0,
            bboxes=[BBox(100, 200, 500, 400)],
            reading_index=1,
            metadata={"figure_caption": "Figure 1: Missing from decoder tree."},
        ),
    ]
    document = DocumentIR(
        doc_id="missing_ref_float",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="missing_ref_float",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["body"]),
            RenderTreeNode(render_id="body", role=RenderRole.PARAGRAPH, source_node_ids=["body"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert r"Figure \ref{fig:1}" in tex
    assert r"\caption{Missing from decoder tree}" in tex
    assert r"\label{fig:1}" in tex


def test_original_like_ir_renderer_places_referenced_missing_table_at_physical_position():
    nodes = [
        DocumentNode(
            node_id="p1",
            node_type=BlockType.TEXT,
            text="The first result is shown in Table 1.",
            page_idx=0,
            bboxes=[BBox(80, 100, 900, 130)],
            reading_index=0,
        ),
        DocumentNode(
            node_id="p2",
            node_type=BlockType.TEXT,
            text="The second paragraph should stay after the table.",
            page_idx=0,
            bboxes=[BBox(80, 150, 900, 180)],
            reading_index=1,
        ),
        DocumentNode(
            node_id="tab",
            node_type=BlockType.TABLE,
            text="Table 1: Deferred table.",
            page_idx=0,
            bboxes=[BBox(100, 500, 700, 700)],
            reading_index=99,
            metadata={"table_caption": "Table 1: Deferred table."},
        ),
    ]
    document = DocumentIR(
        doc_id="referenced_table_anchor",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="referenced_table_anchor",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["p1", "p2"]),
            RenderTreeNode(render_id="p1", role=RenderRole.PARAGRAPH, source_node_ids=["p1"]),
            RenderTreeNode(render_id="p2", role=RenderRole.PARAGRAPH, source_node_ids=["p2"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert tex.index(r"Table \ref{tab:1}") < tex.index(r"\caption{Deferred table}")
    assert tex.index("The second paragraph should stay after the table") < tex.index(r"\caption{Deferred table}")


def test_original_like_ir_renderer_defers_float_after_interrupted_sentence():
    nodes = [
        DocumentNode(
            node_id="left",
            node_type=BlockType.TEXT,
            text="The long statement remains",
            page_idx=0,
            bboxes=[BBox(80, 100, 900, 130)],
            reading_index=0,
        ),
        DocumentNode(
            node_id="fig",
            node_type=BlockType.FIGURE,
            text="Figure 3: Interrupting figure.",
            page_idx=0,
            bboxes=[BBox(100, 150, 500, 350)],
            reading_index=1,
            metadata={"figure_caption": "Figure 3: Interrupting figure."},
        ),
        DocumentNode(
            node_id="right",
            node_type=BlockType.TEXT,
            text="with the continuation after the inserted visual.",
            page_idx=0,
            bboxes=[BBox(80, 370, 900, 410)],
            reading_index=2,
        ),
    ]
    document = DocumentIR(
        doc_id="missing_float_sentence_defer",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="missing_float_sentence_defer",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["left", "right"]),
            RenderTreeNode(render_id="left", role=RenderRole.PARAGRAPH, source_node_ids=["left"]),
            RenderTreeNode(render_id="right", role=RenderRole.PARAGRAPH, source_node_ids=["right"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert tex.index("The long statement remains") < tex.index("with the continuation after the inserted visual.")
    assert tex.index("with the continuation after the inserted visual.") < tex.index(r"\begin{figure}[H]")
    assert r"\caption{Interrupting figure}" in tex


def test_original_like_ir_renderer_injects_missing_table_even_without_reference():
    nodes = [
        DocumentNode(
            node_id="body",
            node_type=BlockType.TEXT,
            text="A paragraph before the table.",
            page_idx=0,
            bboxes=[BBox(80, 100, 900, 130)],
            reading_index=0,
        ),
        DocumentNode(
            node_id="tab",
            node_type=BlockType.TABLE,
            text="Table 2: Unreferenced but present.",
            page_idx=0,
            bboxes=[BBox(100, 200, 700, 400)],
            reading_index=1,
            metadata={"table_caption": "Table 2: Unreferenced but present."},
        ),
    ]
    document = DocumentIR(
        doc_id="missing_table",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )
    profile = StyleProfileExtractor().extract(document)
    tree = RenderTreeIR(
        doc_id="missing_table",
        document_ir_path="document_ir.json",
        root_id="r0",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["body"]),
            RenderTreeNode(render_id="body", role=RenderRole.PARAGRAPH, source_node_ids=["body"]),
        ],
    )

    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(include_maketitle=False)).render(document, tree, profile)

    assert r"\caption{Unreferenced but present}" in tex
    assert r"\label{tab:2}" in tex


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
