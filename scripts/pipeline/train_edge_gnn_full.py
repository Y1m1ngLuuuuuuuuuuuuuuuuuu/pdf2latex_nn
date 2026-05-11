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
from src.perception.schema import EDGE_ATTR_FIELDS, FeatureTensorSchema  # noqa: E402
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
    parser.add_argument("--predictor-hidden-dims", default="1024,512,128")
    parser.add_argument("--predictor-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--edge-feature-mode",
        choices=["full", "simple_concat"],
        default="full",
        help="full uses concat([Hu,Hv,Hu-Hv,Hu*Hv,Euv]); simple_concat uses concat([Hu,Hv,Euv]).",
    )
    parser.add_argument("--semantic-hidden-dim", type=int, default=96)
    parser.add_argument("--layout-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--ablate-node-groups",
        default="",
        help=(
            "Comma-separated runtime node feature groups to zero. "
            "Supported: semantic,type,geometry,scroll,derived,style,sequence,column,title,layout_layer,flow_context,layout_all."
        ),
    )
    parser.add_argument(
        "--ablate-edge-groups",
        default="",
        help=(
            "Comma-separated runtime edge feature groups to zero. "
            "Supported: semantic,spatial,typography,overlap_gutter,index_bins,punctuation,layout_flow,all."
        ),
    )
    parser.add_argument(
        "--ablate-edge-fields",
        default="",
        help="Comma-separated exact edge_attr field names to zero at runtime.",
    )
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--class-weights", choices=["none", "default", "inverse", "custom"], default="none")
    parser.add_argument(
        "--class-weight-values",
        default="",
        help="Comma-separated custom class weights for MERGE,PARENT_CHILD,NONE, e.g. 100,10,1.",
    )
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument(
        "--positive-weight-multiplier",
        type=float,
        default=1.0,
        help="Extra multiplier applied to MERGE and PARENT_CHILD class weights.",
    )
    parser.add_argument(
        "--train-negative-dropout",
        type=float,
        default=0.0,
        help="Training-only probability of dropping NONE edges before model forward.",
    )
    parser.add_argument(
        "--ohem-negative-ratio",
        type=float,
        default=0.0,
        help="Training-only OHEM: keep all positive edges and top ratio*positive_count NONE losses. 0 disables.",
    )
    parser.add_argument(
        "--ohem-min-negatives",
        type=int,
        default=32,
        help="Minimum hard NONE edges kept per training batch when OHEM is enabled.",
    )
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
    args = build_arg_parser().parse_args()
    import torch

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
    if "train" in loaders:
        setattr(loaders["train"], "train_negative_dropout", args.train_negative_dropout)

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
    print(
        f"device={device} epochs={args.epochs} batch_size={args.batch_size} "
        f"loss={args.loss} weights={args.class_weights} "
        f"positive_weight_multiplier={args.positive_weight_multiplier} "
        f"train_negative_dropout={args.train_negative_dropout} "
        f"ohem_negative_ratio={args.ohem_negative_ratio}"
    )

    best_metric = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            loss_fn,
            device=device,
            torch=torch,
            ohem_negative_ratio=args.ohem_negative_ratio,
            ohem_min_negatives=args.ohem_min_negatives,
        )
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
    disabled_node_ranges = parse_node_feature_ablation_ranges(args.ablate_node_groups)
    disabled_edge_indices = parse_edge_feature_ablation_indices(args.ablate_edge_groups, args.ablate_edge_fields)
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
            predictor_hidden_dims=parse_int_tuple(args.predictor_hidden_dims),
            predictor_layer_norm=bool(args.predictor_layer_norm),
            edge_feature_mode=args.edge_feature_mode,
            disabled_node_feature_ranges=disabled_node_ranges,
            disabled_edge_attr_indices=disabled_edge_indices,
        )
    )


def parse_int_tuple(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not parts:
        return ()
    dims = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in dims):
        raise ValueError("predictor hidden dims must be positive integers")
    return dims


def parse_node_feature_ablation_ranges(value: str) -> tuple[tuple[int, int], ...]:
    groups = parse_name_list(value)
    if not groups:
        return ()
    ranges = node_feature_group_ranges()
    selected: list[tuple[int, int]] = []
    for group in groups:
        if group == "layout_all":
            selected.extend(span for name, span in ranges.items() if name != "semantic")
            continue
        if group not in ranges:
            raise ValueError(f"Unknown node ablation group '{group}'. Supported: {sorted(ranges)} plus layout_all")
        selected.append(ranges[group])
    return normalize_ranges(selected)


def parse_edge_feature_ablation_indices(groups_value: str, fields_value: str) -> tuple[int, ...]:
    field_to_idx = {field: idx for idx, field in enumerate(EDGE_ATTR_FIELDS)}
    groups = parse_name_list(groups_value)
    fields = parse_name_list(fields_value)
    selected: set[int] = set()
    edge_groups = edge_feature_group_fields()
    for group in groups:
        if group == "all":
            selected.update(range(len(EDGE_ATTR_FIELDS)))
            continue
        if group not in edge_groups:
            raise ValueError(f"Unknown edge ablation group '{group}'. Supported: {sorted(edge_groups)} plus all")
        selected.update(field_to_idx[field] for field in edge_groups[group] if field in field_to_idx)
    for field in fields:
        if field not in field_to_idx:
            raise ValueError(f"Unknown edge_attr field '{field}'. Supported: {EDGE_ATTR_FIELDS}")
        selected.add(field_to_idx[field])
    return tuple(sorted(selected))


def parse_name_list(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def normalize_ranges(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cleaned = sorted({(int(start), int(end)) for start, end in ranges if int(end) > int(start)})
    if not cleaned:
        return ()
    merged: list[tuple[int, int]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def node_feature_group_ranges() -> dict[str, tuple[int, int]]:
    schema = FeatureTensorSchema()
    cursor = 0
    ranges: dict[str, tuple[int, int]] = {}
    ranges["semantic"] = (cursor, cursor + schema.semantic_dim)
    cursor += schema.semantic_dim
    ranges["type"] = (cursor, cursor + schema.type_dim)
    cursor += schema.type_dim
    ordered_groups = [
        ("geometry", len(schema.geometry_fields)),
        ("scroll", len(schema.scroll_geometry_fields)),
        ("derived", len(schema.derived_stat_fields)),
        ("style", len(schema.style_stat_fields)),
        ("sequence", len(schema.sequence_position_fields)),
        ("column", len(schema.column_feature_fields)),
        ("title", len(schema.title_structure_fields)),
        ("layout_layer", len(schema.layout_layer_fields)),
        ("flow_context", len(schema.flow_context_fields)),
    ]
    for name, width in ordered_groups:
        ranges[name] = (cursor, cursor + width)
        cursor += width
    return ranges


def edge_feature_group_fields() -> dict[str, tuple[str, ...]]:
    return {
        "semantic": ("semantic_cosine",),
        "spatial": ("delta_y_gap", "delta_x_left", "left_alignment", "center_distance"),
        "typography": ("font_size_delta", "bold_to_regular", "line_height_ratio"),
        "overlap_gutter": ("y_overlap_ratio", "has_x_gutter"),
        "index_bins": (
            "index_delta_bin_adjacent",
            "index_delta_bin_skip_one",
            "index_delta_bin_near",
            "index_delta_bin_far",
            "index_delta_bin_reverse",
        ),
        "punctuation": ("source_ends_with_terminal_punctuation", "source_ends_with_hyphen"),
        "layout_flow": (
            "same_layout_layer",
            "same_layout_band",
            "same_band_column",
            "band_order_delta",
            "crosses_band_boundary",
        ),
    }


def build_loss(args: argparse.Namespace, train_labels: Any, *, device: Any, torch: Any) -> Any:
    weights = build_class_weights(args, train_labels, device=device, torch=torch)
    if args.loss == "focal":
        return FocalLoss(gamma=args.gamma, weight=weights)
    return torch.nn.CrossEntropyLoss(weight=weights)


def build_class_weights(args: argparse.Namespace, train_labels: Any, *, device: Any, torch: Any) -> Any | None:
    if args.class_weights == "custom":
        weights = parse_class_weight_values(args.class_weight_values, device=device, torch=torch)
    elif args.class_weights == "inverse":
        weights = compute_inverse_frequency_weights(train_labels.detach().cpu()).to(device)
    elif args.class_weights == "default":
        weights = default_class_weight_tensor(device=device)
    else:
        weights = None
    if weights is not None and args.positive_weight_multiplier != 1.0:
        weights = weights.clone()
        weights[:2] = weights[:2] * float(args.positive_weight_multiplier)
    return weights


def parse_class_weight_values(value: str, *, device: Any, torch: Any) -> Any:
    parts = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--class-weight-values must contain exactly 3 comma-separated values")
    weights = [float(part) for part in parts]
    if any(weight <= 0 for weight in weights):
        raise ValueError("--class-weight-values must be positive")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    *,
    device: Any,
    torch: Any,
    ohem_negative_ratio: float = 0.0,
    ohem_min_negatives: int = 32,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        batch = batch.to(device)
        negative_dropout = float(getattr(loader, "train_negative_dropout", 0.0))
        if negative_dropout > 0.0:
            batch = apply_train_negative_edge_dropout(batch, negative_dropout, torch=torch)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        if ohem_negative_ratio > 0.0:
            loss = ohem_cross_entropy_loss(
                logits,
                batch.y,
                negative_ratio=ohem_negative_ratio,
                min_negatives=ohem_min_negatives,
                class_weights=getattr(loss_fn, "weight", None),
                torch=torch,
            )
        else:
            loss = loss_fn(logits, batch.y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu().item())}")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        batches += 1
    return total_loss / max(1, batches)


def ohem_cross_entropy_loss(
    logits: Any,
    target: Any,
    *,
    negative_ratio: float,
    min_negatives: int,
    class_weights: Any | None,
    torch: Any,
) -> Any:
    """Online hard example mining for edge classification.

    All MERGE/PARENT_CHILD edges are kept.  NONE edges are sorted by their
    unreduced CE loss, and only the hardest K are kept, where
    K ~= negative_ratio * positive_count.
    """

    y = torch.where(target.long() >= 2, torch.full_like(target.long(), 2), target.long())
    weights = class_weights
    if weights is not None:
        weights = weights.to(device=logits.device, dtype=logits.dtype)
    per_edge_loss = torch.nn.functional.cross_entropy(logits, y, weight=weights, reduction="none")
    positive_mask = y != 2
    negative_mask = y == 2
    positive_loss = per_edge_loss[positive_mask]
    negative_loss = per_edge_loss[negative_mask]

    positive_count = int(positive_loss.numel())
    negative_count = int(negative_loss.numel())
    if negative_count > 0:
        if positive_count > 0:
            keep_negatives = max(int(round(positive_count * max(0.0, float(negative_ratio)))), int(min_negatives))
        else:
            keep_negatives = int(min_negatives)
        keep_negatives = max(1, min(negative_count, keep_negatives))
        negative_loss = torch.topk(negative_loss, k=keep_negatives, largest=True).values

    if positive_count == 0:
        selected = negative_loss
    elif int(negative_loss.numel()) == 0:
        selected = positive_loss
    else:
        selected = torch.cat([positive_loss, negative_loss], dim=0)
    if int(selected.numel()) == 0:
        return per_edge_loss.mean()
    return selected.mean()


def apply_train_negative_edge_dropout(batch: Any, dropout: float, *, torch: Any) -> Any:
    """Drop a random subset of NONE edges for training-time forward/loss only.

    The full graph distribution is still used by validation and test loaders.
    This keeps the training signal from being washed out by the dominant NONE
    class while preserving an honest evaluation distribution.
    """

    dropout = max(0.0, min(float(dropout), 0.999))
    if dropout <= 0.0 or not hasattr(batch, "y") or int(batch.y.numel()) == 0:
        return batch

    y = torch.where(batch.y.long() >= 2, torch.full_like(batch.y.long(), 2), batch.y.long())
    positive_mask = y != 2
    negative_indices = torch.nonzero(y == 2, as_tuple=False).flatten()
    keep_mask = positive_mask.clone()
    if int(negative_indices.numel()) > 0:
        random_keep = torch.rand(int(negative_indices.numel()), device=y.device) >= dropout
        keep_mask[negative_indices] = random_keep
    if int(keep_mask.sum().item()) == 0:
        keep_mask[torch.randint(0, int(y.numel()), (1,), device=y.device)] = True

    filtered = batch.clone()
    filtered.edge_index = filtered.edge_index[:, keep_mask]
    filtered.edge_attr = filtered.edge_attr[keep_mask]
    filtered.y = y[keep_mask]
    if hasattr(filtered, "edge_label") and filtered.edge_label is not None:
        filtered.edge_label = filtered.y
    return filtered


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
