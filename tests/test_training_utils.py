from src.reasoning.training import DEFAULT_EDGE_CLASS_WEIGHTS


def has_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def test_focal_loss_and_macro_metrics_handle_imbalanced_edges():
    if not has_torch():
        return
    import torch

    from src.reasoning.training import FocalLoss, edge_precision_recall_f1

    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
        ]
    )
    target = torch.tensor([0, 1, 2, 1])
    loss = FocalLoss(weight=torch.tensor(DEFAULT_EDGE_CLASS_WEIGHTS))(logits, target)
    metrics = edge_precision_recall_f1(logits, target)

    assert float(loss.item()) > 0.0
    assert metrics.per_class[0]["recall"] == 1.0
    assert metrics.per_class[2]["recall"] == 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0


def test_inverse_frequency_weights_give_rare_classes_more_weight():
    if not has_torch():
        return
    import torch

    from src.reasoning.training import compute_inverse_frequency_weights

    weights = compute_inverse_frequency_weights(torch.tensor([2, 2, 2, 2, 0]))

    assert float(weights[0]) > float(weights[2])
