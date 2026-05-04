def test_feature_projector_balances_raw_node_features():
    try:
        import torch
    except ModuleNotFoundError:
        return

    from src.reasoning.gnn_model import FeatureProjector, FeatureProjectorConfig

    config = FeatureProjectorConfig(semantic_hidden_dim=32, layout_hidden_dim=16, dropout=0.0)
    projector = FeatureProjector(config)
    x = torch.randn(4, 791)

    projected = projector(x)

    assert tuple(projected.shape) == (4, 48)
