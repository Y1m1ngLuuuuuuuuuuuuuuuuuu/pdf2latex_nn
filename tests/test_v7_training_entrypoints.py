import argparse

from scripts.pipeline import train_edge_gnn_full
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
