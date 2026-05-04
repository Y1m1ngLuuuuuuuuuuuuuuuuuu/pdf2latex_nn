import json


def has_torch_and_pyg():
    try:
        import torch  # noqa: F401
        from torch_geometric.data import Data  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def test_document_dataset_loads_graph_path_and_attaches_default_none_labels(tmp_path):
    if not has_torch_and_pyg():
        return
    import torch
    from torch_geometric.data import Data

    from src.datasets.document_dataset import DocumentDataset, DocumentDatasetConfig

    raw_graph = tmp_path / "raw.pt"
    data = Data(
        x=torch.zeros((2, 791)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 10)),
    )
    data.node_records = [{"block_id": "P0"}, {"block_id": "P1"}]
    torch.save(data, raw_graph)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"document_id": "doc1", "graph_path": str(raw_graph)}]), encoding="utf-8")

    dataset = DocumentDataset(DocumentDatasetConfig(root=tmp_path / "dataset", manifest_path=manifest))
    sample = dataset[0]

    assert sample.document_id == "doc1"
    assert sample.y.tolist() == [3, 3]
    assert sample.label_counts == {0: 0, 1: 0, 2: 0, 3: 2}
