import json

from src.reasoning.label_generator import (
    AlignmentLabeler,
    AlignmentLabelerConfig,
    LayoutBreakerException,
    PdfAlignmentNode,
    build_visual_hierarchy,
    clean_text,
    visual_parent_pair_is_quality_gate_required,
)
from src.reasoning.tex_relation_labeler import TexRelationLabel


def has_alignment_deps():
    try:
        import torch  # noqa: F401
        from torch_geometric.data import Data  # noqa: F401
        import TexSoup  # noqa: F401
        import rapidfuzz  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def test_clean_text_collapses_latex_math_and_noise():
    assert clean_text("A $x+y$ formula, with punctuation!") == "axyformulawithpunctuation"
    assert clean_text(r"\begin{equation}x=y\end{equation}") == "xy"


def test_tex_parser_standardizes_basic_nodes_and_unwraps_unknown_macros(tmp_path):
    if not has_alignment_deps():
        return
    content_path = tmp_path / "content.json"
    graph_path = tmp_path / "graph.pt"
    tex_path = tmp_path / "main.tex"
    content_path.write_text('{"items":[]}', encoding="utf-8")
    tex_path.write_text(
        r"""
        \section{Method}
        \mybold{Wrapped paragraph text.}
        \[
        x = y
        \]
        \begin{itemize}
        \item Apple item.
        \end{itemize}
        \begin{figure}
        \caption{A useful figure.}
        \end{figure}
        """,
        encoding="utf-8",
    )

    labeler = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path)
    nodes = labeler.parse_tex_nodes()
    node_types = [node.node_type for node in nodes]

    assert "section" in node_types
    assert "paragraph" in node_types
    assert "equation_display" in node_types
    assert "list_container" in node_types
    assert "list_item" in node_types
    assert "figure_caption" in node_types
    assert any("Wrapped paragraph text" in node.text for node in nodes)


def test_tex_parser_silences_visual_macros_and_layout_arguments(tmp_path):
    if not has_alignment_deps():
        return
    content_path = tmp_path / "content.json"
    graph_path = tmp_path / "graph.pt"
    tex_path = tmp_path / "main.tex"
    content_path.write_text('{"items":[]}', encoding="utf-8")
    tex_path.write_text(
        r"""
        \begin{document}
        \section{Method}
        Text before.
        \includegraphics[width=.75\textwidth]{figures/vit.png}
        \resizebox{\textwidth}{!}{\includegraphics{plots/demo.pdf}}
        Text after.
        \end{document}
        """,
        encoding="utf-8",
    )

    labeler = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path)
    nodes = labeler.parse_tex_nodes()
    combined_clean = " ".join(node.clean_text for node in nodes)
    combined_text = " ".join(node.text for node in nodes)

    assert "width" not in combined_clean
    assert "textwidth" not in combined_clean
    assert "figures" not in combined_clean
    assert "png" not in combined_clean
    assert "demo" not in combined_clean
    assert "Text before" in combined_text
    assert "Text after" in combined_text


def test_tex_parser_rejects_layout_only_float_residue(tmp_path):
    if not has_alignment_deps():
        return
    content_path = tmp_path / "content.json"
    graph_path = tmp_path / "graph.pt"
    tex_path = tmp_path / "main.tex"
    content_path.write_text('{"items":[]}', encoding="utf-8")
    tex_path.write_text(
        r"""
        \begin{document}
        \maketitle
        \section{Introduction}
        Body paragraph before the visual material.
        \begin{figure}[htbp]
        \centering
        \color{purple}
        \begin{subfigure}[t]{0.7\textwidth}
        \includegraphics[width=\linewidth]{figs/demo.png}
        \end{subfigure}
        \caption{Useful visual caption.}
        \end{figure}
        Body paragraph after the visual material.
        \end{document}
        """,
        encoding="utf-8",
    )

    labeler = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path)
    nodes = labeler.parse_tex_nodes()
    combined_clean = " ".join(node.clean_text for node in nodes)
    combined_text = " ".join(node.text for node in nodes)

    assert "htbp" not in combined_clean
    assert "t07" not in combined_clean
    assert "purple" not in combined_clean
    assert "maketitle" not in combined_clean
    assert "Introduction" in combined_text
    assert "Useful visual caption" in combined_text
    assert "Body paragraph before" in combined_text
    assert "Body paragraph after" in combined_text


def test_tex_parser_rejects_poison_drawing_environment(tmp_path):
    if not has_alignment_deps():
        return
    content_path = tmp_path / "content.json"
    graph_path = tmp_path / "graph.pt"
    tex_path = tmp_path / "main.tex"
    content_path.write_text('{"items":[]}', encoding="utf-8")
    tex_path.write_text(
        r"""
        \section{Bad}
        \begin{tikzpicture}
        \node {visual text};
        \end{tikzpicture}
        """,
        encoding="utf-8",
    )

    labeler = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path)

    try:
        labeler.parse_tex_nodes()
    except LayoutBreakerException:
        return
    raise AssertionError("expected LayoutBreakerException")


def test_tex_parser_ignores_preamble_when_document_environment_exists(tmp_path):
    if not has_alignment_deps():
        return
    content_path = tmp_path / "content.json"
    graph_path = tmp_path / "graph.pt"
    tex_path = tmp_path / "main.tex"
    content_path.write_text('{"items":[]}', encoding="utf-8")
    tex_path.write_text(
        r"""
        \newcommand{\projectname}{Preamble Only Noise}
        \title{Should Not Become Paragraph}
        \begin{document}
        \section{Inside}
        Body text.
        \end{document}
        """,
        encoding="utf-8",
    )

    labeler = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path)
    nodes = labeler.parse_tex_nodes()

    assert [node.node_type for node in nodes] == ["section", "paragraph"]
    assert all("Preamble Only Noise" not in node.text for node in nodes)


def test_tex_parser_accepts_starred_section_commands(tmp_path):
    if not has_alignment_deps():
        return
    content_path = tmp_path / "content.json"
    graph_path = tmp_path / "graph.pt"
    tex_path = tmp_path / "main.tex"
    content_path.write_text('{"items":[]}', encoding="utf-8")
    tex_path.write_text(
        r"""
        \begin{document}
        \section{Main}
        \section*{Acknowledgment}
        \end{document}
        """,
        encoding="utf-8",
    )

    labeler = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path)
    sections = [node for node in labeler.parse_tex_nodes() if node.node_type == "section"]

    assert [section.text for section in sections] == ["Main", "Acknowledgment"]
    assert [section.source_name for section in sections] == ["section", "section"]


def test_tex_parser_keeps_inline_math_inside_list_item_but_splits_display_math(tmp_path):
    if not has_alignment_deps():
        return
    content_path = tmp_path / "content.json"
    graph_path = tmp_path / "graph.pt"
    tex_path = tmp_path / "main.tex"
    content_path.write_text('{"items":[]}', encoding="utf-8")
    tex_path.write_text(
        r"""
        \begin{document}
        \begin{enumerate}
        \item Euclidean distance (\(d_E\)): Captures global relationships
        \[
        d_E(x,c_k)=\|f_\theta(x)-c_k\|_2
        \]
        \end{enumerate}
        \end{document}
        """,
        encoding="utf-8",
    )

    labeler = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path)
    nodes = labeler.parse_tex_nodes()
    list_containers = [node for node in nodes if node.node_type == "list_container"]
    list_items = [node for node in nodes if node.node_type == "list_item"]
    equations = [node for node in nodes if node.node_type == "equation_display"]

    assert len(list_containers) == 1
    assert len(list_items) == 1
    assert len(equations) == 1
    assert "Euclidean distance" in list_items[0].text
    assert "d_E" in list_items[0].text
    assert "f_\\theta" not in list_items[0].text
    assert list_items[0].parent_id == list_containers[0].tex_id
    assert equations[0].parent_id == list_containers[0].tex_id


def test_alignment_labeler_injects_merge_parent_and_none_labels(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"
    mapping_path = tmp_path / "mapping.json"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"text_for_embedding": "Introduction"},
                    {
                        "text_for_embedding": (
                            "This is the same long paragraph about cyber threats and few-shot learning."
                        )
                    },
                    {"text_for_embedding": "cyber threats and few-shot learning."},
                    {"text_for_embedding": "Apple item."},
                    {"text_for_embedding": "Banana item."},
                    {"text_for_embedding": "zzzz qqqq xxxx"},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \section{Introduction}
        This is the same long paragraph about cyber threats and few-shot learning.

        \begin{itemize}
        \item Apple item.
        \item Banana item.
        \end{itemize}
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((6, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 1, 3, 5], [1, 2, 0, 4, 0]], dtype=torch.long),
        edge_attr=torch.zeros((5, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(output_mapping_json=mapping_path),
    ).run()

    assert graph.y.dtype == torch.long
    assert graph.y.tolist() == [
        int(TexRelationLabel.PARENT_CHILD),
        int(TexRelationLabel.MERGE),
        int(TexRelationLabel.NONE),
        int(TexRelationLabel.NONE),
        int(TexRelationLabel.NONE),
    ]
    assert graph.label_counts == {0: 1, 1: 1, 2: 3}
    assert mapping_path.exists()
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert all({"node_type", "clean_text", "parent_id"} <= set(node) for node in mapping["tex_nodes"])
    saved = torch.load(graph_path, map_location="cpu", weights_only=False)
    assert saved.y.tolist() == graph.y.tolist()


def test_visual_hierarchy_uses_colon_paragraph_as_list_proxy_parent():
    nodes = [
        PdfAlignmentNode(
            node_index=0,
            text="1 Introduction",
            clean=clean_text("1 Introduction"),
            item={"type": "title", "text_for_embedding": "1 Introduction", "bbox": [80, 80, 360, 110]},
        ),
        PdfAlignmentNode(
            node_index=1,
            text="However, ADD faces several shortcomings related to:",
            clean=clean_text("However, ADD faces several shortcomings related to:"),
            item={
                "type": "paragraph",
                "text_for_embedding": "However, ADD faces several shortcomings related to:",
                "bbox": [80, 130, 800, 160],
            },
        ),
        PdfAlignmentNode(
            node_index=2,
            text="• Data: Creating labeled datasets is costly.",
            clean=clean_text("• Data: Creating labeled datasets is costly."),
            item={
                "type": "paragraph",
                "text_for_embedding": "• Data: Creating labeled datasets is costly.",
                "bbox": [120, 180, 800, 210],
            },
        ),
        PdfAlignmentNode(
            node_index=3,
            text="• Machine Learning: ML methods struggle with rare cases.",
            clean=clean_text("• Machine Learning: ML methods struggle with rare cases."),
            item={
                "type": "paragraph",
                "text_for_embedding": "• Machine Learning: ML methods struggle with rare cases.",
                "bbox": [120, 220, 800, 250],
            },
        ),
    ]

    hierarchy = build_visual_hierarchy(nodes, config=AlignmentLabelerConfig())

    assert hierarchy.parent_by_node[1] == 0
    assert hierarchy.parent_by_node[2] == 1
    assert hierarchy.parent_by_node[3] == 1


def test_visual_hierarchy_closes_references_before_appendix_headings():
    nodes = [
        PdfAlignmentNode(
            node_index=0,
            text="References",
            clean=clean_text("References"),
            item={"type": "title", "text_for_embedding": "References", "bbox": [220, 120, 330, 145]},
        ),
        PdfAlignmentNode(
            node_index=1,
            text="[1] A. Author. A paper.",
            clean=clean_text("[1] A. Author. A paper."),
            item={
                "type": "reference",
                "list_type": "reference_list",
                "text_for_embedding": "[1] A. Author. A paper.",
                "bbox": [80, 170, 520, 190],
            },
        ),
        PdfAlignmentNode(
            node_index=2,
            text="A.2. Markov Decision Process (MDP)",
            clean=clean_text("A.2. Markov Decision Process (MDP)"),
            item={
                "type": "title",
                "text_for_embedding": "A.2. Markov Decision Process (MDP)",
                "bbox": [80, 300, 420, 320],
            },
        ),
        PdfAlignmentNode(
            node_index=3,
            text="The appendix continues with technical definitions.",
            clean=clean_text("The appendix continues with technical definitions."),
            item={
                "type": "paragraph",
                "text_for_embedding": "The appendix continues with technical definitions.",
                "bbox": [80, 340, 760, 365],
            },
        ),
    ]

    hierarchy = build_visual_hierarchy(nodes, config=AlignmentLabelerConfig())

    assert hierarchy.parent_by_node[1] == 0
    assert hierarchy.parent_by_node[2] is None
    assert hierarchy.parent_by_node[3] == 2


def test_visual_hierarchy_does_not_reuse_stale_list_intro_after_body_text():
    nodes = [
        PdfAlignmentNode(
            node_index=0,
            text="1 Introduction",
            clean=clean_text("1 Introduction"),
            item={"type": "title", "text_for_embedding": "1 Introduction", "bbox": [80, 80, 360, 110]},
        ),
        PdfAlignmentNode(
            node_index=1,
            text="Guidelines:",
            clean=clean_text("Guidelines:"),
            item={"type": "paragraph", "text_for_embedding": "Guidelines:", "bbox": [80, 130, 220, 150]},
        ),
        PdfAlignmentNode(
            node_index=2,
            text="• First local item.",
            clean=clean_text("• First local item."),
            item={"type": "paragraph", "text_for_embedding": "• First local item.", "bbox": [120, 170, 520, 190]},
        ),
        PdfAlignmentNode(
            node_index=3,
            text="Question: A later checklist question starts a new block.",
            clean=clean_text("Question: A later checklist question starts a new block."),
            item={
                "type": "paragraph",
                "text_for_embedding": "Question: A later checklist question starts a new block.",
                "bbox": [80, 230, 760, 250],
            },
        ),
        PdfAlignmentNode(
            node_index=4,
            text="• This item belongs to the current section, not the stale Guidelines line.",
            clean=clean_text("• This item belongs to the current section, not the stale Guidelines line."),
            item={
                "type": "paragraph",
                "text_for_embedding": "• This item belongs to the current section, not the stale Guidelines line.",
                "bbox": [120, 280, 760, 300],
            },
        ),
    ]

    hierarchy = build_visual_hierarchy(nodes, config=AlignmentLabelerConfig())

    assert hierarchy.parent_by_node[1] == 0
    assert hierarchy.parent_by_node[2] == 1
    assert hierarchy.parent_by_node[3] == 0
    assert hierarchy.parent_by_node[4] == 0


def test_visual_hierarchy_closes_references_before_non_reference_float():
    nodes = [
        PdfAlignmentNode(
            node_index=0,
            text="References",
            clean=clean_text("References"),
            item={"type": "title", "text_for_embedding": "References", "bbox": [220, 120, 330, 145]},
        ),
        PdfAlignmentNode(
            node_index=1,
            text="[1] A. Author. A paper.",
            clean=clean_text("[1] A. Author. A paper."),
            item={
                "type": "reference",
                "list_type": "reference_list",
                "text_for_embedding": "[1] A. Author. A paper.",
                "bbox": [80, 170, 520, 190],
            },
        ),
        PdfAlignmentNode(
            node_index=2,
            text="Fig. S5. Supplementary segmentation result.",
            clean=clean_text("Fig. S5. Supplementary segmentation result."),
            item={
                "type": "image",
                "text_for_embedding": "Fig. S5. Supplementary segmentation result.",
                "bbox": [200, 300, 760, 520],
            },
        ),
    ]

    hierarchy = build_visual_hierarchy(nodes, config=AlignmentLabelerConfig())

    assert hierarchy.parent_by_node[1] == 0
    assert hierarchy.parent_by_node[2] is None


def test_visual_parent_quality_gate_ignores_float_and_misplaced_reference_children():
    section = PdfAlignmentNode(
        node_index=0,
        text="IV. Conclusions",
        clean=clean_text("IV. Conclusions"),
        item={"type": "title", "text_for_embedding": "IV. Conclusions"},
    )
    paragraph = PdfAlignmentNode(
        node_index=1,
        text="The conclusion text continues.",
        clean=clean_text("The conclusion text continues."),
        item={"type": "paragraph", "text_for_embedding": "The conclusion text continues."},
    )
    figure = PdfAlignmentNode(
        node_index=2,
        text="Fig. S5. Supplementary result.",
        clean=clean_text("Fig. S5. Supplementary result."),
        item={"type": "chart", "text_for_embedding": "Fig. S5. Supplementary result."},
    )
    reference_item = PdfAlignmentNode(
        node_index=3,
        text="[3] A. Author. Paper.",
        clean=clean_text("[3] A. Author. Paper."),
        item={"type": "list", "list_type": "reference_list", "text_for_embedding": "[3] A. Author. Paper."},
    )
    references = PdfAlignmentNode(
        node_index=4,
        text="References",
        clean=clean_text("References"),
        item={"type": "title", "text_for_embedding": "References"},
    )

    assert visual_parent_pair_is_quality_gate_required(section, paragraph)
    assert not visual_parent_pair_is_quality_gate_required(section, figure)
    assert not visual_parent_pair_is_quality_gate_required(section, reference_item)
    assert visual_parent_pair_is_quality_gate_required(references, reference_item)


def test_alignment_labeler_accumulates_pdf_fragments_for_one_tex_node(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"text_for_embedding": "Introduction"},
                    {"text_for_embedding": "A long paragraph starts in the left column"},
                    {"text_for_embedding": "and continues in the right column after a break."},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \section{Introduction}
        A long paragraph starts in the left column and continues in the right column after a break.
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.zeros((2, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(),
    ).run()

    assert graph.y.tolist() == [
        int(TexRelationLabel.PARENT_CHILD),
        int(TexRelationLabel.MERGE),
    ]
    assert graph.pdf_to_tex[1] == graph.pdf_to_tex[2]


def test_alignment_labeler_blind_aligns_display_equation_by_pdf_type(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "title", "text_for_embedding": "Method"},
                    {"type": "paragraph", "text_for_embedding": "Before the equation."},
                    {"type": "equation_interline", "text_for_embedding": "visually rendered math"},
                    {"type": "paragraph", "text_for_embedding": "After the equation."},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \section{Method}
        Before the equation.
        \[
        \frac{\alpha}{\beta}
        \]
        After the equation.
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((4, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_attr=torch.zeros((3, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path).run()
    eq_tex_id = graph.pdf_to_tex[2]

    assert eq_tex_id is not None
    assert graph.pdf_to_tex_scores[2] == 100.0


def test_alignment_labeler_refuses_cross_type_merge_inside_same_tex_item(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "type": "paragraph",
                        "text_for_embedding": "1. Euclidean distance ( d _ { E } ) : Captures global relationships",
                    },
                    {
                        "type": "equation_interline",
                        "text_for_embedding": "d _ { E } ( x , c _ { k } ) = || f _ theta ( x ) - c _ k || _ 2",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \begin{enumerate}
        \item Euclidean distance (\(d_E\)): Captures global relationships
        \[
        d_E(x,c_k)=\|f_\theta(x)-c_k\|_2
        \]
        \end{enumerate}
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((2, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.zeros((1, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path).run()

    assert graph.y.tolist() == [int(TexRelationLabel.NONE)]


def test_alignment_labeler_allows_same_type_text_continuation_after_list_marker(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "paragraph", "text_for_embedding": "1. First item starts here"},
                    {"type": "paragraph", "text_for_embedding": "and continues on the next physical line."},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \begin{enumerate}
        \item First item starts here and continues on the next physical line.
        \end{enumerate}
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((2, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path).run()

    assert graph.y.tolist() == [int(TexRelationLabel.MERGE), int(TexRelationLabel.NONE)]


def test_alignment_labeler_refuses_same_tex_text_list_merge(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "paragraph", "text_for_embedding": "We collected diary entries,"},
                    {"type": "list", "text_for_embedding": "post-study interviews, and compliance data."},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        "We collected diary entries, post-study interviews, and compliance data.",
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((2, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path).run()

    assert graph.y.tolist() == [int(TexRelationLabel.NONE), int(TexRelationLabel.NONE)]


def test_alignment_labeler_refuses_same_tex_run_in_heading_merge(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "paragraph", "text_for_embedding": "Buckets contain lists of key value pairs."},
                    {"type": "paragraph", "text_for_embedding": "Put operation. To add a pair, compute a hash value."},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        "Buckets contain lists of key value pairs. Put operation. To add a pair, compute a hash value.",
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((2, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path).run()

    assert graph.y.tolist() == [int(TexRelationLabel.NONE), int(TexRelationLabel.NONE)]


def test_alignment_labeler_refuses_same_tex_terminal_to_new_paragraph_merge(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "title", "text_for_embedding": "Introduction"},
                    {"type": "paragraph", "text_for_embedding": "The first visual paragraph is complete."},
                    {"type": "paragraph", "text_for_embedding": "The second visual paragraph starts here."},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \section{Introduction}
        The first visual paragraph is complete. The second visual paragraph starts here.
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        edge_index=torch.tensor([[1, 2], [2, 1]], dtype=torch.long),
        edge_attr=torch.zeros((2, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path).run()

    assert graph.y.tolist() == [int(TexRelationLabel.NONE), int(TexRelationLabel.NONE)]


def test_alignment_labeler_allows_same_tex_hyphenated_text_continuation(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "title", "text_for_embedding": "Introduction"},
                    {"type": "paragraph", "text_for_embedding": "The transportation system uses trans-"},
                    {"type": "paragraph", "text_for_embedding": "portation-aware routing to continue."},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \section{Introduction}
        The transportation system uses transportation-aware routing to continue.
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        edge_index=torch.tensor([[1], [2]], dtype=torch.long),
        edge_attr=torch.zeros((1, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path).run()

    assert graph.y.tolist() == [int(TexRelationLabel.MERGE)]


def test_alignment_labeler_refuses_same_tex_merge_across_edge_gutter(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "paragraph", "text_for_embedding": "The first physical fragment"},
                    {"type": "paragraph", "text_for_embedding": "continues after a large column gap."},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text("The first physical fragment continues after a large column gap.", encoding="utf-8")
    data = Data(
        x=torch.zeros((2, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
    )
    data.edge_attr_schema = {"fields": ["has_x_gutter", "y_overlap_ratio"]}
    torch.save(data, graph_path)

    graph = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path).run()

    assert graph.y.tolist() == [int(TexRelationLabel.NONE)]


def test_alignment_labeler_refuses_merge_across_intermediate_list_marker(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "equation_interline", "text_for_embedding": "d ch equation"},
                    {"type": "paragraph", "text_for_embedding": "4. Wasserstein distance"},
                    {"type": "equation_interline", "text_for_embedding": "d w equation"},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \begin{enumerate}
        \item Wasserstein distance
        \[
        d ch equation
        \]
        \[
        d w equation
        \]
        \end{enumerate}
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0], [2]], dtype=torch.long),
        edge_attr=torch.zeros((1, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(similarity_threshold=40.0),
    ).run()

    assert graph.y.tolist() == [int(TexRelationLabel.NONE)]


def test_alignment_labeler_exempts_expected_page_noise_from_orphan_ratio(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "title", "text_for_embedding": "Introduction", "bbox": [100, 100, 300, 120]},
                    {"type": "paragraph", "text_for_embedding": "Body paragraph.", "bbox": [100, 140, 500, 180]},
                    {"type": "page_number", "text_for_embedding": "2", "bbox": [490, 982, 510, 995]},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \section{Introduction}
        Body paragraph.
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.zeros((1, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(
            abort_on_bad_alignment=True,
            max_orphan_ratio=0.0,
            max_isolated_node_ratio=0.0,
        ),
    ).run()

    assert graph.y.tolist() == [int(TexRelationLabel.PARENT_CHILD)]
    assert graph.alignment_quality["raw_orphan_count"] == 1
    assert graph.alignment_quality["orphan_count"] == 0
    assert graph.alignment_quality["expected_visual_orphan_exempt_count"] == 1


def test_alignment_labeler_treats_matched_pre_section_text_as_document_root_scoped(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "type": "paragraph",
                        "text_for_embedding": "Abstract. This paper introduces a small method.",
                        "bbox": [100, 120, 500, 170],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text("Abstract. This paper introduces a small method.", encoding="utf-8")
    data = Data(
        x=torch.zeros((1, 4), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(
            abort_on_bad_alignment=True,
            max_orphan_ratio=0.0,
            max_isolated_node_ratio=0.0,
        ),
    ).run()

    assert graph.y.tolist() == []
    assert graph.pdf_to_tex[0] is not None
    assert graph.alignment_quality["document_root_scoped_count"] == 1
    assert graph.alignment_quality["isolated_node_count"] == 0


def test_alignment_labeler_global_caption_fallback_recovers_missed_float_caption(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "title", "text_for_embedding": "Introduction", "bbox": [100, 100, 300, 120]},
                    {"type": "paragraph", "text_for_embedding": "Body paragraph.", "bbox": [100, 140, 500, 180]},
                    {"type": "paragraph", "text_for_embedding": "Figure 1: GPU kernel timeline.", "bbox": [120, 500, 480, 525]},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \section{Introduction}
        Body paragraph.
        \begin{figure}
        \caption{Figure 1: GPU kernel timeline.}
        \end{figure}
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        edge_attr=torch.zeros((2, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(
            similarity_threshold=101.0,
            max_orphan_ratio=1.0,
            max_window_nodes=1,
            caption_fallback_threshold=80.0,
        ),
    ).run()

    caption_tex_id = graph.pdf_to_tex[2]
    assert caption_tex_id is not None
    assert graph.pdf_to_tex_scores[2] >= 80.0


def test_alignment_quality_exempts_metadata_orphans_from_main_orphan_gate(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"
    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "type": "title",
                        "layout_layer": "metadata_layer",
                        "text_for_embedding": "Template Assembled Paper Title",
                    },
                    {
                        "type": "paragraph",
                        "layout_layer": "main_text_flow",
                        "text_for_embedding": "Body paragraph.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text("Body paragraph.", encoding="utf-8")
    data = Data(
        x=torch.zeros((2, 4), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(
            abort_on_bad_alignment=True,
            max_orphan_ratio=0.0,
            max_unmapped_tex_ratio=1.0,
            max_isolated_node_ratio=1.0,
        ),
    ).run()

    assert graph.alignment_quality["orphan_ratio"] == 0.0
    assert graph.alignment_quality["metadata_orphan_count"] == 1
    assert graph.alignment_quality["metadata_orphan_ratio"] == 1.0


def test_alignment_quality_excludes_float_captions_from_main_unmapped_gate(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"
    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "title", "text_for_embedding": "Introduction"},
                    {"type": "paragraph", "text_for_embedding": "Body paragraph."},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \section{Introduction}
        Body paragraph.
        \begin{figure}
        \caption{An intentionally absent float caption.}
        \end{figure}
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((2, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.zeros((1, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(
            abort_on_bad_alignment=True,
            max_orphan_ratio=1.0,
            max_unmapped_tex_ratio=0.0,
            max_isolated_node_ratio=1.0,
            caption_fallback_threshold=101.0,
        ),
    ).run()

    assert graph.alignment_quality["unmapped_tex_ratio"] == 0.0
    assert graph.alignment_quality["raw_unmapped_tex_ratio"] > 0.0
    assert graph.alignment_quality["unmapped_float_tex_count"] == 1


def test_alignment_quality_excludes_bibliography_from_main_unmapped_gate(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"
    content_path.write_text(
        json.dumps({"items": [{"type": "paragraph", "text_for_embedding": "Body paragraph."}]}),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        Body paragraph.
        \begin{thebibliography}{10}
        \bibitem{a} Author. Missing rendered reference.
        \end{thebibliography}
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((1, 4), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(
            abort_on_bad_alignment=True,
            max_orphan_ratio=1.0,
            max_unmapped_tex_ratio=0.0,
            max_isolated_node_ratio=1.0,
        ),
    ).run(overwrite=False)

    assert graph.alignment_quality["unmapped_tex_ratio"] == 0.0
    assert graph.alignment_quality["weak_tex_count"] >= 1
    assert graph.alignment_quality["raw_unmapped_tex_ratio"] > 0.0


def test_alignment_keeps_pending_section_anchor_across_unmatched_keywords(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"
    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "paragraph", "text_for_embedding": "Abstract. This paper studies playlists."},
                    {"type": "paragraph", "text_for_embedding": "Keywords: music retrieval deployment"},
                    {"type": "title", "text_for_embedding": "1 Introduction"},
                    {"type": "paragraph", "text_for_embedding": "Search engines explore large catalogs."},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        Abstract. This paper studies playlists.
        \section{Introduction}
        Search engines explore large catalogs.
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((4, 4), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(
            abort_on_bad_alignment=False,
            max_orphan_ratio=1.0,
            max_unmapped_tex_ratio=1.0,
            max_isolated_node_ratio=1.0,
        ),
    ).run(overwrite=False)

    assert graph.pdf_to_tex[1] is None
    assert graph.pdf_to_tex[2] is not None
    assert graph.pdf_to_tex[3] is not None


def test_alignment_labeler_does_not_merge_independent_reference_list_blocks_when_alignment_misses(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "type": "list",
                        "list_type": "reference_list",
                        "text_for_embedding": "Author A. First paper. Author B. Second paper.",
                        "reference_items": ["Author A. First paper.", "Author B. Second paper."],
                    },
                    {
                        "type": "list",
                        "list_type": "reference_list",
                        "text_for_embedding": "Author C. Third paper.",
                        "reference_items": ["Author C. Third paper."],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(r"\section{References}", encoding="utf-8")
    data = Data(
        x=torch.zeros((2, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(content_json_path=content_path, tex_path=tex_path, graph_path=graph_path).run()

    assert graph.y.tolist() == [int(TexRelationLabel.NONE), int(TexRelationLabel.NONE)]


def test_alignment_labeler_refuses_same_tex_merge_for_titles_and_non_adjacent_fragments(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "title", "text_for_embedding": "Paper Title", "bbox": [100, 100, 500, 130]},
                    {"type": "paragraph", "text_for_embedding": "Author Name", "bbox": [150, 150, 450, 170]},
                    {"type": "paragraph", "text_for_embedding": "Affiliation", "bbox": [150, 175, 450, 195]},
                    {"type": "title", "text_for_embedding": "Abstract", "bbox": [100, 230, 200, 250]},
                    {"type": "paragraph", "text_for_embedding": "Abstract body.", "bbox": [100, 260, 500, 300]},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        "Paper Title Author Name Affiliation Abstract Abstract body.",
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((5, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 0, 1], [1, 2, 2, 4]], dtype=torch.long),
        edge_attr=torch.zeros((4, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(max_window_nodes=5, similarity_threshold=60.0),
    ).run()

    assert graph.y.tolist() == [
        int(TexRelationLabel.NONE),
        int(TexRelationLabel.NONE),
        int(TexRelationLabel.NONE),
        int(TexRelationLabel.NONE),
    ]


def test_alignment_labeler_uses_visual_heading_stack_when_tex_paths_are_flat(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "title", "text_for_embedding": "Introduction", "bbox": [100, 100, 300, 120]},
                    {"type": "paragraph", "text_for_embedding": "Body paragraph.", "bbox": [100, 130, 500, 180]},
                    {"type": "title", "text_for_embedding": "1.1 Details", "bbox": [100, 190, 300, 210]},
                    {"type": "paragraph", "text_for_embedding": "Detail paragraph.", "bbox": [100, 220, 500, 270]},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        """
        Introduction

        Body paragraph.

        1.1 Details

        Detail paragraph.
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((4, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 0, 2, 1], [1, 2, 3, 3]], dtype=torch.long),
        edge_attr=torch.zeros((4, 15), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(),
    ).run()

    assert graph.y.tolist() == [
        int(TexRelationLabel.PARENT_CHILD),
        int(TexRelationLabel.PARENT_CHILD),
        int(TexRelationLabel.PARENT_CHILD),
        int(TexRelationLabel.NONE),
    ]


def test_alignment_labeler_rejects_incompatible_numbered_heading_parent(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "title", "text_for_embedding": "1 Introduction", "bbox": [80, 80, 360, 110]},
                    {"type": "paragraph", "text_for_embedding": "Intro body.", "bbox": [80, 125, 500, 170]},
                    {"type": "title", "text_for_embedding": "2 Related Research", "bbox": [80, 210, 420, 240]},
                    {"type": "title", "text_for_embedding": "2.3 Continual Assurance", "bbox": [80, 260, 470, 290]},
                    {"type": "paragraph", "text_for_embedding": "Continual assurance body.", "bbox": [80, 305, 500, 350]},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \section{1 Introduction}
        Intro body.

        \section{2 Related Research}
        \subsection{2.3 Continual Assurance}
        Continual assurance body.
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((5, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 2], [3, 3]], dtype=torch.long),
        edge_attr=torch.zeros((2, 22), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(),
    ).run()

    assert graph.y.tolist() == [
        int(TexRelationLabel.NONE),
        int(TexRelationLabel.PARENT_CHILD),
    ]


def test_alignment_labeler_keeps_cross_column_numbered_list_under_original_scope(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "title", "text_for_embedding": "Methodology", "bbox": [80, 80, 360, 110]},
                    {"type": "paragraph", "text_for_embedding": "1) First step.", "bbox": [80, 125, 500, 150]},
                    {"type": "title", "text_for_embedding": "A. Evaluation", "bbox": [80, 170, 360, 200]},
                    {"type": "paragraph", "text_for_embedding": "2) Second step.", "bbox": [560, 125, 930, 150]},
                    {"type": "paragraph", "text_for_embedding": "3) Third step.", "bbox": [560, 165, 930, 190]},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        """
        Methodology

        1) First step.

        A. Evaluation

        2) Second step.

        3) Third step.
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((5, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 2, 0], [3, 3, 4]], dtype=torch.long),
        edge_attr=torch.zeros((3, 22), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(),
    ).run()

    assert graph.y.tolist() == [
        int(TexRelationLabel.PARENT_CHILD),
        int(TexRelationLabel.NONE),
        int(TexRelationLabel.PARENT_CHILD),
    ]


def test_alignment_labeler_visual_parent_does_not_require_first_tex_anchor(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "title", "text_for_embedding": "Introduction", "bbox": [80, 80, 360, 110]},
                    {
                        "type": "paragraph",
                        "text_for_embedding": "This paragraph begins",
                        "bbox": [80, 125, 500, 150],
                    },
                    {
                        "type": "paragraph",
                        "text_for_embedding": "and continues later.",
                        "bbox": [80, 160, 500, 185],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text(
        r"""
        \section{Introduction}
        This paragraph begins and continues later.
        """,
        encoding="utf-8",
    )
    data = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0], [2]], dtype=torch.long),
        edge_attr=torch.zeros((1, 22), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(),
    ).run()

    assert graph.y.tolist() == [int(TexRelationLabel.PARENT_CHILD)]


def test_alignment_labeler_blocks_same_tex_merge_across_column_gutter_without_edge_attr(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v7_styles.json"
    tex_path = tmp_path / "main.tex"
    graph_path = tmp_path / "graph.pt"

    content_path.write_text(
        json.dumps(
            {
                "items": [
                    {"type": "paragraph", "text_for_embedding": "Calendar type definition", "bbox": [80, 100, 350, 130]},
                    {"type": "paragraph", "text_for_embedding": "Another calendar type definition", "bbox": [560, 102, 900, 132]},
                ]
            }
        ),
        encoding="utf-8",
    )
    tex_path.write_text("Calendar type definition Another calendar type definition", encoding="utf-8")
    data = Data(
        x=torch.zeros((2, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.zeros((1, 22), dtype=torch.float32),
    )
    torch.save(data, graph_path)

    graph = AlignmentLabeler(
        content_json_path=content_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(max_window_nodes=3, similarity_threshold=55.0),
    ).run()

    assert graph.y.tolist() == [int(TexRelationLabel.NONE)]
