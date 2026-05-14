def test_feature_projector_balances_raw_node_features():
    try:
        import torch
    except ModuleNotFoundError:
        return

    from src.perception.schema import FeatureTensorSchema
    from src.reasoning.gnn_model import FeatureProjector, FeatureProjectorConfig

    config = FeatureProjectorConfig(semantic_hidden_dim=32, layout_hidden_dim=16, dropout=0.0)
    projector = FeatureProjector(config)
    with torch.no_grad():
        projector.semantic_projection.weight.zero_()
        projector.semantic_projection.bias.fill_(1.0)
    x = torch.randn(4, FeatureTensorSchema().node_feature_dim)

    projected = projector(x)

    assert tuple(projected.shape) == (4, 48)
    assert torch.allclose(projected[:, :32].norm(p=2, dim=-1), torch.ones(4), atol=1e-5)


def test_feature_projector_l2_normalizes_raw_scibert_before_bottleneck():
    try:
        import torch
    except ModuleNotFoundError:
        return

    from src.perception.schema import FeatureTensorSchema
    from src.reasoning.gnn_model import FeatureProjector, FeatureProjectorConfig

    config = FeatureProjectorConfig(semantic_hidden_dim=2, layout_hidden_dim=2, dropout=0.0)
    projector = FeatureProjector(config)
    with torch.no_grad():
        projector.semantic_projection.weight.zero_()
        projector.semantic_projection.weight[0, 0] = 1.0
        projector.semantic_projection.weight[1, 1] = 1.0
        projector.semantic_projection.bias[:] = torch.tensor([1.0, 0.0])
    x = torch.zeros((2, FeatureTensorSchema().node_feature_dim), dtype=torch.float32)
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

    from src.perception.schema import FeatureTensorSchema
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
        x=torch.randn(3, FeatureTensorSchema().node_feature_dim),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.randn(3, config.edge_dim),
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
    assert model.edge_head[0].out_features == 1024
    assert features.tolist() == [[1.0, 2.0, 3.0, 5.0, -2.0, -3.0, 3.0, 10.0, 0.25, 0.5]]


def test_edge_relation_gat_uses_message_edge_mask_only_for_propagation():
    try:
        import torch
        from torch_geometric.data import Data
    except ModuleNotFoundError:
        return

    from src.perception.schema import FeatureTensorSchema
    from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT, FeatureProjectorConfig

    config = EdgeGATConfig(
        node_projector=FeatureProjectorConfig(semantic_hidden_dim=16, layout_hidden_dim=8, dropout=0.0),
        hidden_dim=8,
        heads=2,
        num_layers=1,
        dropout=0.0,
        message_edge_mode="type_aware",
    )
    model = EdgeRelationGAT(config)
    data = Data(
        x=torch.randn(3, FeatureTensorSchema().node_feature_dim),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.randn(3, config.edge_dim),
        message_edge_mask=torch.tensor([True, False, True], dtype=torch.bool),
    )

    conv_edge_index, conv_edge_attr = model._message_passing_edges(
        data.edge_index,
        data.edge_attr,
        data.message_edge_mask,
    )
    logits = model(data)

    assert conv_edge_index.tolist() == [[0, 2], [1, 0]]
    assert tuple(conv_edge_attr.shape) == (2, config.edge_dim)
    assert tuple(logits.shape) == (3, 3)


def test_edge_relation_gat_y_network_bypasses_gnn_for_merge_and_keeps_shape():
    try:
        import torch
        from torch_geometric.data import Data
    except ModuleNotFoundError:
        return

    from src.perception.schema import FeatureTensorSchema
    from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT, FeatureProjectorConfig

    config = EdgeGATConfig(
        node_projector=FeatureProjectorConfig(semantic_hidden_dim=16, layout_hidden_dim=8, dropout=0.0),
        hidden_dim=8,
        heads=2,
        num_layers=1,
        dropout=0.0,
        prediction_architecture="y_network",
    )
    model = EdgeRelationGAT(config)
    data = Data(
        x=torch.randn(3, FeatureTensorSchema().node_feature_dim),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.randn(3, config.edge_dim),
    )

    logits = model(data)

    assert hasattr(model, "merge_head")
    assert hasattr(model, "parent_none_head")
    assert tuple(logits.shape) == (3, 3)


def test_edge_relation_gat_hard_merge_gate_only_suppresses_merge_logit():
    try:
        import torch
        from torch_geometric.data import Data
    except ModuleNotFoundError:
        return

    from src.perception.schema import FeatureTensorSchema
    from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT, FeatureProjectorConfig

    config = EdgeGATConfig(
        node_projector=FeatureProjectorConfig(semantic_hidden_dim=16, layout_hidden_dim=8, dropout=0.0),
        hidden_dim=8,
        heads=2,
        num_layers=0,
        dropout=0.0,
        prediction_architecture="y_network",
        merge_gate_mode="hard",
        merge_gate_logit=-123.0,
    )
    model = EdgeRelationGAT(config)
    data = Data(
        x=torch.randn(3, FeatureTensorSchema().node_feature_dim),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.randn(2, config.edge_dim),
        merge_candidate_mask=torch.tensor([True, False], dtype=torch.bool),
    )

    logits = model(data)

    assert float(logits[1, 0].detach()) == -123.0
    assert torch.isfinite(logits[:, 1:]).all()


def test_edge_relation_gat_appends_gaussian_edge_feature_without_rewriting_graph_attr():
    try:
        import torch
        from torch_geometric.data import Data
    except ModuleNotFoundError:
        return

    from src.perception.schema import EDGE_ATTR_FIELDS, FeatureTensorSchema
    from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT, FeatureProjectorConfig

    config = EdgeGATConfig(
        node_projector=FeatureProjectorConfig(semantic_hidden_dim=16, layout_hidden_dim=8, dropout=0.0),
        hidden_dim=8,
        heads=2,
        num_layers=1,
        dropout=0.0,
        prediction_architecture="y_network",
        gaussian_edge_feature_mode="center",
        gaussian_sigma=0.10,
    )
    model = EdgeRelationGAT(config)
    center_idx = EDGE_ATTR_FIELDS.index("center_distance")
    edge_attr = torch.zeros((2, config.edge_dim), dtype=torch.float32)
    edge_attr[0, center_idx] = 0.0
    edge_attr[1, center_idx] = 0.10
    data = Data(
        x=torch.randn(3, FeatureTensorSchema().node_feature_dim),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=edge_attr,
    )

    prepared = model._prepare_edge_attr(data.edge_attr)
    logits = model(data)

    assert model.raw_edge_dim == config.edge_dim
    assert model.effective_edge_dim == config.edge_dim + 1
    assert tuple(prepared.shape) == (2, config.edge_dim + 1)
    assert torch.allclose(prepared[:, -1], torch.tensor([1.0, 0.60653067]), atol=1e-6)
    assert tuple(logits.shape) == (2, 3)
