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
    assert clean_text("A $x+y$ formula, with punctuation!") == "amathplaceholderformulawithpunctuation"
    assert clean_text(r"\begin{equation}x=y\end{equation}") == "mathplaceholder"


def test_alignment_labeler_injects_merge_parent_sibling_and_none_labels(tmp_path):
    if not has_alignment_deps():
        return
    import torch
    from torch_geometric.data import Data

    content_path = tmp_path / "content_v4_styles.json"
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
        edge_attr=torch.zeros((4, 10), dtype=torch.float32),
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
        int(TexRelationLabel.SIBLING),
        int(TexRelationLabel.NONE),
    ]
    assert graph.label_counts == {0: 1, 1: 1, 2: 1, 3: 1}
    assert mapping_path.exists()
    saved = torch.load(graph_path, map_location="cpu", weights_only=False)
    assert saved.y.tolist() == graph.y.tolist()
