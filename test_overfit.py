#!/usr/bin/env python3
"""Single-batch overfit sanity check for the edge-relation GAT.

This script intentionally trains on only 4-8 already-processed documents with
dropout disabled and no DataLoader shuffling. The goal is not generalization:
the model should be able to memorize this tiny batch. If it cannot drive loss
near zero, the data flow, labels, or model wiring need debugging before any
large-scale training run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.document_dataset import DocumentDataset, DocumentDatasetConfig, build_document_dataloader  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT, FeatureProjectorConfig  # noqa: E402
from src.reasoning.training import FocalLoss, compute_inverse_frequency_weights, edge_precision_recall_f1  # noqa: E402


LABEL_NAMES = {
    0: "merge",
    1: "parent_child",
    2: "sibling",
    3: "none",
}


@dataclass(frozen=True)
class OverfitResult:
    final_loss: float
    macro_f1_all_classes: float
    macro_f1_present_classes: float
    epochs: int
    num_documents: int
    num_nodes: int
    num_edges: int
    class_counts: dict[str, int]
    passed: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="DocumentDataset root")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON/JSONL document manifest")
    parser.add_argument("--model-path", type=Path, help="Local SciBERT path if graphs must be built from content JSON")
    parser.add_argument("--min-docs", type=int, default=4, help="Minimum valid documents required for the sanity batch")
    parser.add_argument("--max-docs", type=int, default=8, help="Maximum valid documents to use in the sanity batch")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--semantic-hidden-dim", type=int, default=64)
    parser.add_argument("--layout-hidden-dim", type=int, default=32)
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--class-weights", choices=["inverse", "default", "none"], default="inverse")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--loss-threshold", type=float, default=0.05)
    parser.add_argument("--f1-threshold", type=float, default=0.99)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--report-json", type=Path, help="Optional path to write final metrics")
    parser.add_argument("--no-fail", action="store_true", help="Return 0 even if the overfit criterion fails")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_overfit(args)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    if result.passed:
        print("PASS: single-batch overfit criterion reached.")
        return 0
    print("FAIL: model did not memorize the single batch under the requested thresholds.")
    return 0 if args.no_fail else 2


def run_overfit(args: argparse.Namespace) -> OverfitResult:
    import torch

    set_seed(args.seed, torch=torch)
    dataset = DocumentDataset(
        DocumentDatasetConfig(root=args.root, manifest_path=args.manifest, model_path=args.model_path)
    )
    samples = select_dataset_samples(dataset, min_docs=args.min_docs, max_docs=args.max_docs)
    document_ids = [str(getattr(sample, "document_id", f"doc_{idx}")) for idx, sample in enumerate(samples)]
    loader = build_document_dataloader(samples, batch_size=len(samples), shuffle=False)
    batch = next(iter(loader))

    device = resolve_device(args.device, torch=torch)
    batch = batch.to(device)
    if not hasattr(batch, "y") or batch.y.numel() == 0:
        raise ValueError("Single-batch overfit requires non-empty edge labels")

    model = build_overfit_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = build_overfit_loss(args, batch.y, device=device, torch=torch)

    class_counts = class_count_dict(batch.y.detach().cpu())
    print("single-batch overfit setup")
    print(f"documents={len(samples)} ids={document_ids}")
    print(f"nodes={int(batch.num_nodes)} edges={int(batch.edge_index.shape[1])}")
    print(f"class_counts={class_counts}")
    print(f"device={device} epochs={args.epochs} lr={args.lr} dropout=0.0 shuffle=False")

    final_loss = float("inf")
    final_all_macro = 0.0
    final_present_macro = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, batch.y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {float(loss.detach().cpu().item())}")
        loss.backward()
        grad_norm = total_grad_norm(model)
        if not torch.isfinite(torch.tensor(grad_norm)):
            raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}: {grad_norm}")
        optimizer.step()

        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.log_every) == 0:
            final_loss, final_all_macro, final_present_macro, per_class = evaluate_on_batch(model, batch, loss_fn)
            print_epoch(epoch, final_loss, final_all_macro, final_present_macro, per_class)

    final_loss, final_all_macro, final_present_macro, per_class = evaluate_on_batch(model, batch, loss_fn)
    print_epoch(args.epochs, final_loss, final_all_macro, final_present_macro, per_class, prefix="final")
    passed = final_loss <= args.loss_threshold and final_present_macro >= args.f1_threshold
    return OverfitResult(
        final_loss=final_loss,
        macro_f1_all_classes=final_all_macro,
        macro_f1_present_classes=final_present_macro,
        epochs=args.epochs,
        num_documents=len(samples),
        num_nodes=int(batch.num_nodes),
        num_edges=int(batch.edge_index.shape[1]),
        class_counts=class_counts,
        passed=passed,
    )


def build_overfit_model(args: argparse.Namespace) -> EdgeRelationGAT:
    return EdgeRelationGAT(
        EdgeGATConfig(
            node_projector=FeatureProjectorConfig(
                semantic_hidden_dim=args.semantic_hidden_dim,
                layout_hidden_dim=args.layout_hidden_dim,
                dropout=0.0,
            ),
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            num_layers=args.num_layers,
            dropout=0.0,
        )
    )


def build_overfit_loss(args: argparse.Namespace, labels: Any, *, device: Any, torch: Any) -> Any:
    if args.class_weights == "inverse":
        weights = compute_inverse_frequency_weights(labels.detach().cpu()).to(device)
    elif args.class_weights == "default":
        weights = torch.tensor([4.0, 5.0, 1.5, 1.0], dtype=torch.float32, device=device)
    else:
        weights = None
    if args.loss == "focal":
        return FocalLoss(gamma=args.gamma, weight=weights)
    return torch.nn.CrossEntropyLoss(weight=weights)


def select_dataset_samples(dataset: Any, *, min_docs: int, max_docs: int) -> list[Any]:
    if min_docs <= 0:
        raise ValueError("min_docs must be positive")
    if max_docs < min_docs:
        raise ValueError("max_docs must be >= min_docs")
    available = len(dataset)
    if available < min_docs:
        raise ValueError(f"Need at least {min_docs} valid documents, found {available}")
    return [dataset[idx] for idx in range(min(max_docs, available))]


def evaluate_on_batch(model: Any, batch: Any, loss_fn: Any) -> tuple[float, float, float, dict[int, dict[str, float]]]:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(batch)
        loss = loss_fn(logits, batch.y)
    metrics = edge_precision_recall_f1(logits.detach().cpu(), batch.y.detach().cpu())
    present_macro = present_class_macro_f1(metrics.per_class)
    return float(loss.detach().cpu().item()), metrics.macro_f1, present_macro, metrics.per_class


def present_class_macro_f1(per_class: dict[int, dict[str, float]]) -> float:
    present = [values["f1"] for values in per_class.values() if values.get("support", 0) > 0]
    return sum(present) / max(1, len(present))


def class_count_dict(labels: Any) -> dict[str, int]:
    import torch

    counts = torch.bincount(labels.long(), minlength=4).tolist()
    return {LABEL_NAMES[idx]: int(counts[idx]) for idx in range(4)}


def print_epoch(
    epoch: int,
    loss: float,
    macro_f1_all: float,
    macro_f1_present: float,
    per_class: dict[int, dict[str, float]],
    *,
    prefix: str = "epoch",
) -> None:
    class_bits = []
    for label, name in LABEL_NAMES.items():
        values = per_class[label]
        class_bits.append(
            f"{name}:P={values['precision']:.3f},R={values['recall']:.3f},F1={values['f1']:.3f},n={values['support']}"
        )
    print(
        f"{prefix}={epoch:04d} loss={loss:.6f} "
        f"macro_f1_all={macro_f1_all:.4f} macro_f1_present={macro_f1_present:.4f} "
        + " | ".join(class_bits)
    )


def total_grad_norm(model: Any) -> float:
    import math

    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().data.norm(2).item())
        squared += norm * norm
    return math.sqrt(squared)


def resolve_device(value: str, *, torch: Any) -> Any:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(value)


def set_seed(seed: int, *, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
