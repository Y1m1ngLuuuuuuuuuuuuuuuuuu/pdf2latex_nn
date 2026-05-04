#!/usr/bin/env python3
"""Train the edge-relation GAT on processed DocumentDataset graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.document_dataset import DocumentDataset, DocumentDatasetConfig, build_document_dataloader  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402
from src.reasoning.training import build_edge_loss, evaluate_edge_model, train_one_epoch  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root with processed graph files")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON/JSONL document manifest")
    parser.add_argument("--model-path", type=Path, help="Local SciBERT path if graphs must be built from JSON")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint output path")
    parser.add_argument("--loss", choices=["focal", "cross_entropy"], default="focal")
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
    dataset = DocumentDataset(
        DocumentDatasetConfig(root=args.root, manifest_path=args.manifest, model_path=args.model_path)
    )
    loader = build_document_dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeRelationGAT(EdgeGATConfig(hidden_dim=args.hidden_dim, heads=args.heads)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = build_edge_loss(loss_name=args.loss, device=device)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loader, optimizer, loss_fn, device=device)
        eval_metrics = evaluate_edge_model(model, loader, device=device)
        print(
            f"epoch={epoch} loss={train_metrics['loss']:.4f} "
            f"macro_f1={eval_metrics['macro_f1']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": model.config}, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
