import json

from src.reasoning.label_generator import AlignmentLabeler, AlignmentLabelerConfig, clean_text
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
        edge_index=torch.tensor([[0, 1, 3, 5], [1, 2, 4, 0]], dtype=torch.long),
        edge_attr=torch.zeros((4, 15), dtype=torch.float32),
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
    ]
    assert graph.label_counts == {0: 1, 1: 1, 2: 2}
    assert mapping_path.exists()
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
