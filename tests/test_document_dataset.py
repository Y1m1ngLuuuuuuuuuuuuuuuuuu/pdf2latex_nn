import json


def has_torch_and_pyg():
    try:
        import torch  # noqa: F401
        from torch_geometric.data import Data  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def make_graph(torch, Data, *, edge_index=None, x=None, edge_attr=None):
    if edge_index is None:
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_count = int(edge_index.shape[1])
    data = Data(
        x=torch.zeros((2, 818), dtype=torch.float64) if x is None else x,
        edge_index=edge_index,
        edge_attr=torch.zeros((edge_count, 15), dtype=torch.float64) if edge_attr is None else edge_attr,
    )
    data.node_records = [{"block_id": "P0"}, {"block_id": "P1"}]
    data.pipeline_version = "v7"
    data.graph_schema_version = "graph_v7"
    data.source_path = "synthetic_content_list_v7_styles.json"
    return data


def test_sanitize_graph_data_casts_float32_and_clamps_non_finite_values():
    if not has_torch_and_pyg():
        return
    import torch
    from torch_geometric.data import Data

    from src.datasets.document_dataset import sanitize_graph_data

    data = make_graph(
        torch,
        Data,
        x=torch.full((2, 818), float("nan"), dtype=torch.float64),
        edge_attr=torch.full((2, 15), float("inf"), dtype=torch.float64),
    )

    sanitized = sanitize_graph_data(data)

    assert sanitized.x.dtype == torch.float32
    assert sanitized.edge_attr.dtype == torch.float32
    assert not torch.isnan(sanitized.x).any()
    assert torch.isfinite(sanitized.edge_attr).all()


def test_document_dataset_filters_empty_edge_and_all_orphan_graphs(tmp_path):
    if not has_torch_and_pyg():
        return
    import torch
    from torch_geometric.data import Data

    from src.datasets.document_dataset import DocumentDataset, DocumentDatasetConfig

    valid_graph = tmp_path / "valid.pt"
    empty_graph = tmp_path / "empty.pt"
    orphan_graph = tmp_path / "orphan.pt"
    torch.save(make_graph(torch, Data), valid_graph)
    torch.save(make_graph(torch, Data, edge_index=torch.empty((2, 0), dtype=torch.long)), empty_graph)
    torch.save(make_graph(torch, Data), orphan_graph)

    tex_path = tmp_path / "doc.tex"
    tex_path.write_text(r"\section{Intro}" + "\n\n" + "A paragraph.", encoding="utf-8")
    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text(
        json.dumps(
            {
                "P0": {"tex_id": "T_1", "score": 0.99},
                "P1": {"tex_id": "T_2", "score": 0.99},
            }
        ),
        encoding="utf-8",
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "document_id": "valid",
                    "graph_path": str(valid_graph),
                    "tex_path": str(tex_path),
                    "pdf_to_tex_path": str(alignment_path),
                },
                {"document_id": "empty", "graph_path": str(empty_graph)},
                {"document_id": "orphan", "graph_path": str(orphan_graph)},
            ]
        ),
        encoding="utf-8",
    )

    dataset = DocumentDataset(DocumentDatasetConfig(root=tmp_path / "dataset", manifest_path=manifest))

    assert len(dataset) == 1
    sample = dataset[0]

    assert sample.document_id == "valid"
    assert sample.x.dtype == torch.float32
    assert sample.edge_attr.dtype == torch.float32
    assert sample.y.tolist() == [1, 2]
    skipped = (tmp_path / "dataset" / "processed" / "skipped_records.jsonl").read_text(encoding="utf-8")
    assert "empty edge graph" in skipped
    assert "all-orphan graph" in skipped


def test_document_dataset_preserves_existing_graph_labels(tmp_path):
    if not has_torch_and_pyg():
        return
    import torch
    from torch_geometric.data import Data

    from src.datasets.document_dataset import DocumentDataset, DocumentDatasetConfig

    graph_path = tmp_path / "labeled.pt"
    data = make_graph(torch, Data)
    data.y = torch.tensor([0, 2], dtype=torch.long)
    data.edge_label = data.y
    torch.save(data, graph_path)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"document_id": "labeled", "graph_path": str(graph_path)}]),
        encoding="utf-8",
    )

    dataset = DocumentDataset(DocumentDatasetConfig(root=tmp_path / "dataset", manifest_path=manifest))
    sample = dataset[0]

    assert sample.y.tolist() == [0, 2]
    assert sample.label_counts == {0: 1, 1: 0, 2: 1}


def test_document_dataset_skips_graph_when_alignment_quality_is_too_low(tmp_path):
    if not has_torch_and_pyg():
        return
    import torch
    from torch_geometric.data import Data

    from src.datasets.document_dataset import DocumentDataset, DocumentDatasetConfig

    bad_graph = tmp_path / "bad.pt"
    edge_index = torch.tensor([[0, 1, 1], [1, 2, 0]], dtype=torch.long)
    data = Data(x=torch.zeros((3, 818)), edge_index=edge_index, edge_attr=torch.zeros((3, 15)))
    data.node_records = [{"block_id": "P0"}, {"block_id": "P1"}, {"block_id": "P2"}]
    data.pipeline_version = "v7"
    data.graph_schema_version = "graph_v7"
    data.source_path = "synthetic_content_list_v7_styles.json"
    torch.save(data, bad_graph)

    tex_path = tmp_path / "doc.tex"
    tex_path.write_text(r"\section{Intro}" + "\n\n" + "A paragraph.", encoding="utf-8")
    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text(
        json.dumps(
            {
                "P0": {"tex_id": "T_1", "score": 0.99},
                "P1": {"tex_id": "T_2", "score": 0.99},
                "P2": {"tex_id": "T_2", "score": 0.1},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "document_id": "bad_alignment",
                    "graph_path": str(bad_graph),
                    "tex_path": str(tex_path),
                    "pdf_to_tex_path": str(alignment_path),
                }
            ]
        ),
        encoding="utf-8",
    )

    dataset = DocumentDataset(DocumentDatasetConfig(root=tmp_path / "dataset", manifest_path=manifest))

    assert len(dataset) == 0
    skipped = (tmp_path / "dataset" / "processed" / "skipped_records.jsonl").read_text(encoding="utf-8")
    assert "bad alignment quality" in skipped
    assert "orphan_ratio=33.33%" in skipped
