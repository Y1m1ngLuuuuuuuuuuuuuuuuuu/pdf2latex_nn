#!/usr/bin/env python3
"""Train EdgeRelationGAT on a v7 manifest of labeled document graphs.

This is the full-scale training entrypoint. It consumes the `.pt` graph files
already produced by the v7 batch builder and does not mutate graph schemas,
manifest records, or PDF/TeX preprocessing outputs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.document_dataset import DocumentDataset, DocumentDatasetConfig, build_document_dataloader  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT, FeatureProjectorConfig  # noqa: E402
from src.reasoning.training import (  # noqa: E402
    FocalLoss,
    compute_inverse_frequency_weights,
    default_class_weight_tensor,
    edge_precision_recall_f1,
)


LABEL_NAMES = {
    0: "merge",
    1: "parent_child",
    2: "none",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="DocumentDataset root for processed cache")
    parser.add_argument("--manifest", type=Path, required=True, help="v7 manifest with graph_path records")
    parser.add_argument("--model-path", type=Path, help="SciBERT path only needed if a record lacks graph_path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for reports and checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--semantic-hidden-dim", type=int, default=96)
    parser.add_argument("--layout-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--class-weights", choices=["none", "default", "inverse"], default="none")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--selection-metric",
        choices=["val_macro_f1", "val_positive_macro_f1"],
        default="val_positive_macro_f1",
        help="Metric used for best checkpoint selection.",
    )
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
    set_seed(args.seed, torch=torch)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DocumentDataset(
        DocumentDatasetConfig(root=args.root, manifest_path=args.manifest, model_path=args.model_path)
    )
    if len(dataset) <= 0:
        raise ValueError(f"No trainable graphs found for manifest {args.manifest}")

    splits = split_indices(len(dataset), args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)
    split_samples = {
        name: [dataset[idx] for idx in indices]
        for name, indices in splits.items()
    }
    loaders = {
        name: build_document_dataloader(
            samples,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
        )
        for name, samples in split_samples.items()
        if samples
    }

    device = resolve_device(args.device, torch=torch)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_labels = collect_labels(split_samples["train"], torch=torch)
    loss_fn = build_loss(args, train_labels, device=device, torch=torch)

    split_summary = summarize_splits(split_samples)
    write_json(args.output_dir / "split_summary.json", split_summary)
    print("full-scale training setup")
    print(f"manifest={args.manifest}")
    print(f"dataset_size={len(dataset)} split_docs={ {key: len(value) for key, value in split_samples.items()} }")
    print(f"train_class_counts={count_labels(train_labels, torch=torch)}")
    print(f"device={device} epochs={args.epochs} batch_size={args.batch_size} loss={args.loss} weights={args.class_weights}")

    best_metric = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, loss_fn, device=device, torch=torch)
        row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
        for split_name, loader in loaders.items():
            metrics = evaluate(model, loader, loss_fn, device=device, torch=torch)
            row.update(flatten_metrics(split_name, metrics))
        history.append(row)

        selection_value = float(row.get(args.selection_metric, -1.0))
        if selection_value > best_metric:
            best_metric = selection_value
            best_epoch = epoch
            save_checkpoint(args.output_dir / "best_model.pth", model, args, epoch, row)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.log_every) == 0:
            print_epoch(row)
            write_json(args.output_dir / "history.json", history)

    save_checkpoint(args.output_dir / "last_model.pth", model, args, args.epochs, history[-1])
    report = {
        "schema_version": "edge_gnn_training_report_v1",
        "manifest": str(args.manifest),
        "dataset_size": len(dataset),
        "splits": split_summary,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "selection_metric": args.selection_metric,
        "final": history[-1],
        "args": serializable_args(args),
    }
    write_json(args.output_dir / "training_report.json", report)
    print(f"best_epoch={best_epoch} {args.selection_metric}={best_metric:.4f}")
    print(f"wrote {args.output_dir / 'best_model.pth'}")
    return 0


def build_model(args: argparse.Namespace) -> EdgeRelationGAT:
    return EdgeRelationGAT(
        EdgeGATConfig(
            node_projector=FeatureProjectorConfig(
                semantic_hidden_dim=args.semantic_hidden_dim,
                layout_hidden_dim=args.layout_hidden_dim,
                dropout=args.dropout,
            ),
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    )


def build_loss(args: argparse.Namespace, train_labels: Any, *, device: Any, torch: Any) -> Any:
    if args.class_weights == "inverse":
        weights = compute_inverse_frequency_weights(train_labels.detach().cpu()).to(device)
    elif args.class_weights == "default":
        weights = default_class_weight_tensor(device=device)
    else:
        weights = None
    if args.loss == "focal":
        return FocalLoss(gamma=args.gamma, weight=weights)
    return torch.nn.CrossEntropyLoss(weight=weights)


def train_one_epoch(model: Any, loader: Any, optimizer: Any, loss_fn: Any, *, device: Any, torch: Any) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, batch.y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu().item())}")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        batches += 1
    return total_loss / max(1, batches)


def evaluate(model: Any, loader: Any, loss_fn: Any, *, device: Any, torch: Any) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    batches = 0
    logits_list = []
    target_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = loss_fn(logits, batch.y)
            total_loss += float(loss.detach().cpu().item())
            batches += 1
            logits_list.append(logits.detach().cpu())
            target_list.append(batch.y.detach().cpu())
    if not logits_list:
        return empty_metrics()
    logits_all = torch.cat(logits_list, dim=0)
    target_all = torch.cat(target_list, dim=0)
    edge_metrics = edge_precision_recall_f1(logits_all, target_all, num_classes=3)
    positive_macro = (edge_metrics.per_class[0]["f1"] + edge_metrics.per_class[1]["f1"]) / 2.0
    return {
        "loss": total_loss / max(1, batches),
        "macro_f1": edge_metrics.macro_f1,
        "positive_macro_f1": positive_macro,
        "per_class": edge_metrics.per_class,
        "class_counts": count_labels(target_all, torch=torch),
    }


def empty_metrics() -> dict[str, Any]:
    return {
        "loss": 0.0,
        "macro_f1": 0.0,
        "positive_macro_f1": 0.0,
        "per_class": {idx: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for idx in range(3)},
        "class_counts": {name: 0 for name in LABEL_NAMES.values()},
    }


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_loss": metrics["loss"],
        f"{prefix}_macro_f1": metrics["macro_f1"],
        f"{prefix}_positive_macro_f1": metrics["positive_macro_f1"],
        f"{prefix}_per_class": metrics["per_class"],
        f"{prefix}_class_counts": metrics["class_counts"],
    }


def split_indices(
    size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    seed: int,
) -> dict[str, list[int]]:
    if size <= 0:
        return {"train": [], "val": [], "test": []}
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive")
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    train_count = max(1, int(round(size * train_ratio)))
    val_count = int(round(size * val_ratio))
    if size >= 3 and val_ratio > 0:
        val_count = max(1, val_count)
    if train_count + val_count >= size and size > 1:
        train_count = max(1, size - val_count - 1)
    test_count = max(0, size - train_count - val_count)
    return {
        "train": indices[:train_count],
        "val": indices[train_count : train_count + val_count],
        "test": indices[train_count + val_count : train_count + val_count + test_count],
    }


def collect_labels(samples: list[Any], *, torch: Any) -> Any:
    labels = [sample.y.detach().cpu().long() for sample in samples if hasattr(sample, "y") and sample.y.numel() > 0]
    if not labels:
        raise ValueError("Training split has no edge labels")
    merged = torch.cat(labels, dim=0)
    return torch.where(merged >= 2, torch.full_like(merged, 2), merged)


def count_labels(labels: Any, *, torch: Any) -> dict[str, int]:
    labels = torch.where(labels.detach().cpu().long() >= 2, torch.full_like(labels.detach().cpu().long(), 2), labels.detach().cpu().long())
    counts = torch.bincount(labels, minlength=3).tolist()
    return {LABEL_NAMES[idx]: int(counts[idx]) for idx in range(3)}


def summarize_splits(split_samples: dict[str, list[Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, samples in split_samples.items():
        summary[split_name] = {
            "num_documents": len(samples),
            "document_ids": [str(getattr(sample, "document_id", f"{split_name}_{idx}")) for idx, sample in enumerate(samples)],
            "num_nodes": sum(int(sample.num_nodes) for sample in samples),
            "num_edges": sum(int(sample.edge_index.shape[1]) for sample in samples),
        }
    return summary


def print_epoch(row: dict[str, Any]) -> None:
    parts = [
        f"epoch={int(row['epoch']):04d}",
        f"train_loss={row['train_loss']:.4f}",
    ]
    for split in ("train", "val", "test"):
        if f"{split}_macro_f1" in row:
            parts.append(f"{split}_f1={row[f'{split}_macro_f1']:.4f}")
            parts.append(f"{split}_pos_f1={row[f'{split}_positive_macro_f1']:.4f}")
    print(" ".join(parts))


def save_checkpoint(path: Path, model: Any, args: argparse.Namespace, epoch: int, metrics: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": model.config,
            "epoch": epoch,
            "metrics": metrics,
            "args": serializable_args(args),
            "checkpoint_type": "edge_relation_gat_full_training",
        },
        path,
    )


def resolve_device(value: str, *, torch: Any) -> Any:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, *, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
