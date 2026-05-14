import json

import pytest

from src.reasoning.label_generator import AlignmentQualityError, LabelGeneratorConfig, label_graph_edges, pdf_item_text
from src.reasoning.tex_ast_builder import build_tex_ast, tex_nodes_by_id


def has_torch_and_pyg():
    try:
        import torch  # noqa: F401
        from torch_geometric.data import Data  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def test_pdf_item_text_uses_table_caption_not_cell_body():
    item = {
        "type": "table",
        "text_for_embedding": "Table 2: Results\nA B C 1 2 3",
        "table_caption": "Table 2: Results",
        "table_body": "A B C 1 2 3",
    }

    assert pdf_item_text(item) == "Table 2: Results"


def test_label_graph_edges_falls_back_to_none_for_low_similarity_orphans(tmp_path):
    if not has_torch_and_pyg():
        return
    import torch
    from torch_geometric.data import Data

    payload = build_tex_ast(
        r"""
        \section{Intro}
        A paragraph.
        Another paragraph.
        """
    )
    nodes = tex_nodes_by_id(payload)
    tex_ids = [node["tex_id"] for node in nodes.values()]
    data = Data(
        x=torch.zeros((3, 4)),
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.zeros((3, 15)),
    )
    data.node_records = [{"block_id": "P0"}, {"block_id": "P1"}, {"block_id": "P2"}]
    mapping = {
        "P0": {"tex_id": tex_ids[0], "score": 0.99},
        "P1": {"tex_id": tex_ids[1], "score": 0.99},
        "P2": {"tex_id": tex_ids[1], "score": 0.1},
    }
    orphan_log = tmp_path / "orphans.jsonl"

    result = label_graph_edges(
        data,
        tex_ast=payload,
        pdf_to_tex=mapping,
        config=LabelGeneratorConfig(similarity_threshold=0.55, max_orphan_ratio=1.0),
        orphan_log_path=orphan_log,
    )

    assert result.data.y.tolist() == [1, 2, 2]
    assert result.label_counts == {0: 0, 1: 1, 2: 2}
    assert len(result.orphan_alignments) == 1
    logged = json.loads(orphan_log.read_text(encoding="utf-8").strip())
    assert logged["reason"] == "low_similarity"


def test_label_graph_edges_aborts_when_default_orphan_ratio_exceeds_thirty_percent(tmp_path):
    if not has_torch_and_pyg():
        return
    import torch
    from torch_geometric.data import Data

    payload = build_tex_ast(
        r"""
        \section{Intro}
        A paragraph.
        """
    )
    nodes = tex_nodes_by_id(payload)
    tex_ids = [node["tex_id"] for node in nodes.values()]
    data = Data(
        x=torch.zeros((3, 4)),
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.zeros((3, 15)),
    )
    data.node_records = [{"block_id": "P0"}, {"block_id": "P1"}, {"block_id": "P2"}]
    mapping = {
        "P0": {"tex_id": tex_ids[0], "score": 0.99},
        "P1": {"tex_id": tex_ids[1], "score": 0.99},
        "P2": {"tex_id": tex_ids[1], "score": 0.1},
    }
    orphan_log = tmp_path / "orphans.jsonl"

    with pytest.raises(AlignmentQualityError, match="orphan_ratio=33.33%"):
        label_graph_edges(
            data,
            tex_ast=payload,
            pdf_to_tex=mapping,
            config=LabelGeneratorConfig(similarity_threshold=0.55),
            orphan_log_path=orphan_log,
        )

    logged = json.loads(orphan_log.read_text(encoding="utf-8").strip())
    assert logged["reason"] == "low_similarity"
