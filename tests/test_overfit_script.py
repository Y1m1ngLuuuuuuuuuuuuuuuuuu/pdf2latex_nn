import pytest

import test_overfit


class TinyDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return f"sample-{idx}"


def test_overfit_parser_defaults_match_single_batch_sanity_check():
    parser = test_overfit.build_arg_parser()
    args = parser.parse_args(["--root", "/tmp/dataset", "--manifest", "/tmp/manifest.json"])

    assert args.min_docs == 4
    assert args.max_docs == 8
    assert args.epochs == 200
    assert args.lr == 1e-3
    assert args.class_weights == "inverse"


def test_select_dataset_samples_uses_first_deterministic_single_batch_slice():
    samples = test_overfit.select_dataset_samples(TinyDataset(10), min_docs=4, max_docs=8)

    assert samples == [f"sample-{idx}" for idx in range(8)]


def test_select_dataset_samples_rejects_too_few_valid_documents():
    with pytest.raises(ValueError, match="Need at least 4 valid documents"):
        test_overfit.select_dataset_samples(TinyDataset(3), min_docs=4, max_docs=8)


def test_overfit_script_runs_one_epoch_on_tiny_valid_graph(tmp_path):
    try:
        import torch
        from torch_geometric.data import Data
    except ModuleNotFoundError:
        return

    graph_path = tmp_path / "graph.pt"
    data = Data(
        x=torch.zeros((2, 817), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 11), dtype=torch.float32),
    )
    data.node_records = [{"block_id": "P0"}, {"block_id": "P1"}]
    torch.save(data, graph_path)

    tex_path = tmp_path / "doc.tex"
    tex_path.write_text(r"\section{Intro}" + "\n\n" + "A paragraph.", encoding="utf-8")
    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text(
        '{"P0": {"tex_id": "T_1", "score": 0.99}, "P1": {"tex_id": "T_2", "score": 0.99}}',
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        (
            '[{"document_id": "tiny", '
            f'"graph_path": "{graph_path}", '
            f'"tex_path": "{tex_path}", '
            f'"pdf_to_tex_path": "{alignment_path}"'
            "}]"
        ),
        encoding="utf-8",
    )
    parser = test_overfit.build_arg_parser()
    args = parser.parse_args(
        [
            "--root",
            str(tmp_path / "dataset"),
            "--manifest",
            str(manifest_path),
            "--min-docs",
            "1",
            "--max-docs",
            "1",
            "--epochs",
            "1",
            "--hidden-dim",
            "4",
            "--heads",
            "1",
            "--num-layers",
            "1",
            "--semantic-hidden-dim",
            "4",
            "--layout-hidden-dim",
            "4",
            "--device",
            "cpu",
        ]
    )

    artifacts = test_overfit.run_overfit(args)
    result = artifacts.result

    assert result.num_documents == 1
    assert result.num_edges == 2
    assert result.class_counts["parent_child"] == 2
    assert artifacts.model_state_dict
