from src.generation.latex_renderer import RenderConfig, render_latex_document
from src.reasoning.postprocess import MERGE, PARENT_CHILD, DecodedEdge, build_resolved_tree, greedy_decode_relations


def has_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def test_greedy_decode_relations_drops_parent_cycles():
    if not has_torch():
        return
    import torch

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    scores = torch.tensor(
        [
            [0.01, 0.96, 0.01, 0.02],
            [0.01, 0.95, 0.01, 0.03],
            [0.01, 0.94, 0.01, 0.04],
        ]
    )

    decoded = greedy_decode_relations(edge_index, scores, threshold=0.5, num_nodes=3)

    assert [(edge.source, edge.target, edge.label) for edge in decoded] == [
        (0, 1, PARENT_CHILD),
        (1, 2, PARENT_CHILD),
    ]


def test_build_resolved_tree_merges_text_and_renderer_emits_tex_document():
    records = [
        {"type": "title", "text": "Introduction"},
        {"type": "paragraph", "text": "Cyber-"},
        {"type": "paragraph", "text": "security matters."},
    ]
    decoded = [
        DecodedEdge(source=1, target=2, label=MERGE, score=0.99),
        DecodedEdge(source=0, target=1, label=PARENT_CHILD, score=0.95),
    ]

    root = build_resolved_tree(records, decoded)
    tex = render_latex_document(root, RenderConfig(title="Demo"))

    assert r"\documentclass{article}" in tex
    assert r"\section{Introduction}" in tex
    assert "Cybersecurity matters." in tex
    assert r"\usepackage{amsmath}" in tex
