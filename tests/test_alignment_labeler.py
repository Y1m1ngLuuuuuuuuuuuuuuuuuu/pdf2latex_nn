import json

from src.reasoning.label_generator import AlignmentLabeler, AlignmentLabelerConfig, LayoutBreakerException, clean_text
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


def test_alignment_labeler_merges_reference_list_blocks_even_when_fuzzy_alignment_misses(tmp_path):
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

    assert graph.y.tolist() == [int(TexRelationLabel.MERGE), int(TexRelationLabel.MERGE)]


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
