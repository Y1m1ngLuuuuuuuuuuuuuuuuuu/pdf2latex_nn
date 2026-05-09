import argparse

from src.perception.schema import FeatureTensorSchema
from scripts.pipeline import train_edge_gnn_full
from scripts.pipeline import filter_split_manifest
from scripts.pipeline import calibrate_edge_thresholds
from tools import watch_v7_batch


def test_full_training_split_is_deterministic_and_complete():
    splits = train_edge_gnn_full.split_indices(
        100,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=7,
    )

    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    assert sorted(splits["train"] + splits["val"] + splits["test"]) == list(range(100))
    assert splits == train_edge_gnn_full.split_indices(100, 0.8, 0.1, 0.1, seed=7)


def test_train_negative_edge_dropout_keeps_positives_and_drops_none():
    try:
        import torch
        from torch_geometric.data import Data
    except ModuleNotFoundError:
        return

    batch = Data(
        x=torch.zeros((3, FeatureTensorSchema().node_feature_dim), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long),
        edge_attr=torch.zeros((4, 15), dtype=torch.float32),
        y=torch.tensor([0, 1, 2, 2], dtype=torch.long),
    )

    filtered = train_edge_gnn_full.apply_train_negative_edge_dropout(batch, 0.999, torch=torch)

    assert filtered.edge_index.shape[1] == 2
    assert filtered.y.tolist() == [0, 1]


def test_ohem_loss_keeps_positive_and_hardest_none_edges():
    try:
        import torch
    except ModuleNotFoundError:
        return

    logits = torch.tensor(
        [
            [2.0, 0.0, -2.0],
            [0.0, 2.0, -2.0],
            [3.0, 0.0, -2.0],
            [0.0, 3.0, -2.0],
            [-2.0, 0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([0, 1, 2, 2, 2], dtype=torch.long)
    per_edge = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    expected = torch.cat([per_edge[:2], torch.topk(per_edge[2:], k=2).values]).mean()

    actual = train_edge_gnn_full.ohem_cross_entropy_loss(
        logits,
        target,
        negative_ratio=1.0,
        min_negatives=0,
        class_weights=None,
        torch=torch,
    )

    assert torch.allclose(actual, expected)


def test_parse_custom_class_weight_values():
    try:
        import torch
    except ModuleNotFoundError:
        return

    weights = train_edge_gnn_full.parse_class_weight_values("100,10,1", device=torch.device("cpu"), torch=torch)

    assert weights.dtype == torch.float32
    assert weights.tolist() == [100.0, 10.0, 1.0]


def test_parse_predictor_hidden_dims():
    assert train_edge_gnn_full.parse_int_tuple("1024,512,128") == (1024, 512, 128)


def test_calibrated_threshold_priority_predicts_merge_then_parent_then_none():
    try:
        import torch
    except ModuleNotFoundError:
        return

    prob = torch.tensor(
        [
            [0.60, 0.90, 0.01],
            [0.10, 0.80, 0.10],
            [0.20, 0.30, 0.50],
        ],
        dtype=torch.float32,
    )
    pred = calibrate_edge_thresholds.predict_with_thresholds(
        prob,
        tau_merge=0.50,
        tau_parent=0.50,
        mode="threshold_priority",
        torch=torch,
    )

    assert pred.tolist() == [0, 1, 2]


def test_focal_loss_uses_alpha_after_unweighted_pt():
    try:
        import torch
    except ModuleNotFoundError:
        return

    from src.reasoning.training import FocalLoss

    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], dtype=torch.float32)
    target = torch.tensor([0, 1], dtype=torch.long)
    alpha = torch.tensor([10.0, 1.0, 1.0], dtype=torch.float32)
    loss = FocalLoss(gamma=2.0, weight=alpha, reduction="none")(logits, target)

    assert loss[0] > loss[1]
    assert torch.isfinite(loss).all()


def test_filter_split_manifest_filters_by_orphan_and_positive_edges():
    records = [
        {"document_id": "a", "graph_path": "/tmp/a.pt", "orphan_ratio": 0.2, "candidate_edge_recall": 1.0, "label_counts": {"0": 1, "1": 0, "2": 10}},
        {"document_id": "b", "graph_path": "/tmp/b.pt", "orphan_ratio": 0.5, "candidate_edge_recall": 1.0, "label_counts": {"0": 1, "1": 0, "2": 10}},
        {"document_id": "c", "graph_path": "/tmp/c.pt", "orphan_ratio": 0.1, "candidate_edge_recall": 1.0, "label_counts": {"0": 0, "1": 0, "2": 10}},
        {
            "document_id": "d",
            "graph_path": "/tmp/d.pt",
            "orphan_ratio": 0.1,
            "candidate_edge_recall": 1.0,
            "label_counts": {"merge": 0, "parent_child": 2, "none": 10},
        },
    ]

    kept = [
        record["document_id"]
        for record in records
        if filter_split_manifest.passes_filters(
            record,
            max_orphan_ratio=0.3,
            min_candidate_recall=1.0,
            min_non_none_edges=1,
        )
    ]

    assert kept == ["a", "d"]


def test_v7_batch_watcher_parses_progress_and_eta(tmp_path):
    log_path = tmp_path / "batch.log"
    error_path = tmp_path / "errors.jsonl"
    log_path.write_text(
        "\n".join(
            [
                "candidate_count=20 target=10",
                "[mini-dataset] start id=doc1 success=0/10",
                "[mini-dataset] success id=doc1 success=1/10 labels={0: 1, 1: 2, 2: 3} orphan_ratio=10.00%",
                "mini-dataset:  10%|#         | 2/20 [00:10<01:30, 5.00s/doc]",
                "[mini-dataset] start id=doc2 success=1/10",
                "[mini-dataset] skip id=doc2 error=AlignmentQualityError: bad alignment quality",
            ]
        ),
        encoding="utf-8",
    )
    error_path.write_text('{"error_type": "AlignmentQualityError"}\n', encoding="utf-8")

    summary = watch_v7_batch.build_summary(
        argparse.Namespace(
            log=log_path,
            error_log=error_path,
            manifest=None,
            current_dir=tmp_path,
            tail=3,
            json_output=None,
            interval=0,
        )
    )

    assert summary["candidate_count"] == 20
    assert summary["target"] == 10
    assert summary["success_count"] == 1
    assert summary["skip_count"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["skip_types"] == {"AlignmentQualityError": 1}
    assert summary["error_stats"]["error_types"] == {"AlignmentQualityError": 1}
    assert summary["estimated"]["estimated_successes_if_rate_holds"] == 10
