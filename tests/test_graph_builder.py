def test_feature_projector_balances_raw_node_features():
    try:
        import torch
    except ModuleNotFoundError:
        return

    from src.reasoning.gnn_model import FeatureProjector, FeatureProjectorConfig

    config = FeatureProjectorConfig(semantic_hidden_dim=32, layout_hidden_dim=16, dropout=0.0)
    projector = FeatureProjector(config)
    with torch.no_grad():
        projector.semantic_projection.weight.zero_()
        projector.semantic_projection.bias.fill_(1.0)
    x = torch.randn(4, 818)

    projected = projector(x)

    assert tuple(projected.shape) == (4, 48)
    assert torch.allclose(projected[:, :32].norm(p=2, dim=-1), torch.ones(4), atol=1e-5)


def test_feature_projector_l2_normalizes_raw_scibert_before_bottleneck():
    try:
        import torch
    except ModuleNotFoundError:
        return

    from src.reasoning.gnn_model import FeatureProjector, FeatureProjectorConfig

    config = FeatureProjectorConfig(semantic_hidden_dim=2, layout_hidden_dim=2, dropout=0.0)
    projector = FeatureProjector(config)
    with torch.no_grad():
        projector.semantic_projection.weight.zero_()
        projector.semantic_projection.weight[0, 0] = 1.0
        projector.semantic_projection.weight[1, 1] = 1.0
        projector.semantic_projection.bias[:] = torch.tensor([1.0, 0.0])
    x = torch.zeros((2, 818), dtype=torch.float32)
    x[0, :2] = torch.tensor([3.0, 4.0])
    x[1, :2] = torch.tensor([30.0, 40.0])

    projected = projector(x)

    assert torch.allclose(projected[0, :2], projected[1, :2], atol=1e-6)
    assert torch.allclose(projected[:, :2].norm(p=2, dim=-1), torch.ones(2), atol=1e-6)


def test_edge_relation_gat_outputs_one_logit_row_per_edge():
    try:
        import torch
        from torch_geometric.data import Data
    except ModuleNotFoundError:
        return

    from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT, FeatureProjectorConfig

    config = EdgeGATConfig(
        node_projector=FeatureProjectorConfig(semantic_hidden_dim=16, layout_hidden_dim=8, dropout=0.0),
        hidden_dim=8,
        heads=2,
        num_layers=1,
        dropout=0.0,
    )
    model = EdgeRelationGAT(config)
    data = Data(
        x=torch.randn(3, 818),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.randn(3, 15),
    )

    logits = model(data)

    assert tuple(logits.shape) == (3, 3)


def test_edge_relation_gat_edge_head_uses_symmetry_breaking_features():
    try:
        import torch
    except ModuleNotFoundError:
        return

    from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT, FeatureProjectorConfig

    config = EdgeGATConfig(
        node_projector=FeatureProjectorConfig(semantic_hidden_dim=16, layout_hidden_dim=8, dropout=0.0),
        hidden_dim=8,
        heads=2,
        num_layers=1,
        dropout=0.0,
    )
    model = EdgeRelationGAT(config)
    h = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float32)
    source = torch.tensor([0], dtype=torch.long)
    target = torch.tensor([1], dtype=torch.long)
    edge_attr = torch.tensor([[0.25, 0.5]], dtype=torch.float32)

    features = model._build_edge_features(h, source, target, edge_attr)

    assert model.edge_head[0].in_features == config.hidden_dim * config.heads * 4 + config.edge_dim
    assert features.tolist() == [[1.0, 2.0, 3.0, 5.0, 2.0, 3.0, 3.0, 10.0, 0.25, 0.5]]
