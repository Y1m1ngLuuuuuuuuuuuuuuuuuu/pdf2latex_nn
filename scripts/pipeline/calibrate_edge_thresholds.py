#!/usr/bin/env python3
"""Search class-specific edge decision thresholds on validation logits.

The model is not retrained.  The script loads a checkpoint, reproduces the
document-level split used by training, searches thresholds on validation data,
then locks the best threshold pair and reports metrics on test data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.document_dataset import DocumentDataset, DocumentDatasetConfig, build_document_dataloader  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402
from src.reasoning.training import edge_precision_recall_f1  # noqa: E402
from scripts.pipeline.train_edge_gnn_full import split_indices  # noqa: E402
from scripts.pipeline.step5_generate_tex import checkpoint_compatible_config  # noqa: E402
from src.reasoning.postprocess import can_contract_merge_records  # noqa: E402


LABEL_NAMES = {0: "merge", 1: "parent_child", 2: "none"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tau-min", type=float, default=0.05)
    parser.add_argument("--tau-max", type=float, default=0.95)
    parser.add_argument("--tau-step", type=float, default=0.01)
    parser.add_argument(
        "--min-merge-precision",
        type=float,
        default=0.0,
        help=(
            "Optional validation precision floor for MERGE while selecting the main calibrated threshold. "
            "Use this for deployment/visual QA settings where false MERGE edges are more harmful than missed merges."
        ),
    )
    parser.add_argument(
        "--precision-floors",
        default="0.70,0.75,0.80,0.85,0.90",
        help="Comma-separated MERGE precision floors to report as secondary operating points.",
    )
    parser.add_argument(
        "--apply-merge-gates",
        action="store_true",
        help=(
            "Apply TreeDecoder-compatible hard MERGE gates during calibration/evaluation. "
            "This measures the deployable constrained decoder rather than raw edge argmax."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["threshold_priority", "scaled_argmax"],
        default="threshold_priority",
        help="threshold_priority: MERGE then PARENT thresholds. scaled_argmax: choose max(p/tau) among classes passing thresholds.",
    )
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
    device = resolve_device(args.device, torch=torch)
    dataset = DocumentDataset(DocumentDatasetConfig(root=args.root, manifest_path=args.manifest))
    if len(dataset) <= 0:
        raise ValueError(f"No graphs found for {args.manifest}")
    splits = split_indices(len(dataset), args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)
    split_samples = {name: [dataset[idx] for idx in indices] for name, indices in splits.items()}
    loaders = {
        name: build_document_dataloader(samples, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        for name, samples in split_samples.items()
        if samples
    }

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    model_config = checkpoint_compatible_config(config if isinstance(config, EdgeGATConfig) else EdgeGATConfig(), state_dict)
    model = EdgeRelationGAT(model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    val_logits, val_target = collect_logits(model, loaders["val"], device=device, torch=torch)
    test_logits, test_target = collect_logits(model, loaders["test"], device=device, torch=torch)
    val_prob = torch.softmax(val_logits, dim=-1)
    test_prob = torch.softmax(test_logits, dim=-1)
    val_merge_allowed = collect_merge_gate_mask(split_samples["val"], torch=torch) if args.apply_merge_gates else None
    test_merge_allowed = collect_merge_gate_mask(split_samples["test"], torch=torch) if args.apply_merge_gates else None

    argmax_val = metric_payload(argmax_with_merge_gates(val_logits, val_merge_allowed, torch=torch), val_target)
    argmax_test = metric_payload(argmax_with_merge_gates(test_logits, test_merge_allowed, torch=torch), test_target)
    search = search_thresholds(
        val_prob,
        val_target,
        tau_values=tau_grid(args.tau_min, args.tau_max, args.tau_step),
        mode=args.mode,
        min_merge_precision=args.min_merge_precision,
        merge_allowed=val_merge_allowed,
        torch=torch,
    )
    test_pred = predict_with_thresholds(
        test_prob,
        tau_merge=search["tau_merge"],
        tau_parent=search["tau_parent"],
        mode=args.mode,
        merge_allowed=test_merge_allowed,
        torch=torch,
    )
    val_pred = predict_with_thresholds(
        val_prob,
        tau_merge=search["tau_merge"],
        tau_parent=search["tau_parent"],
        mode=args.mode,
        merge_allowed=val_merge_allowed,
        torch=torch,
    )
    payload = {
        "schema_version": "edge_threshold_calibration_v1",
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "split_docs": {name: len(samples) for name, samples in split_samples.items()},
        "search_grid": {"tau_min": args.tau_min, "tau_max": args.tau_max, "tau_step": args.tau_step},
        "constraints": {"min_merge_precision": args.min_merge_precision},
        "apply_merge_gates": bool(args.apply_merge_gates),
        "best_thresholds": {
            "tau_merge": search["tau_merge"],
            "tau_parent": search["tau_parent"],
            "val_positive_macro_f1": search["val_positive_macro_f1"],
            "val_macro_f1": search["val_macro_f1"],
            "val_merge_precision": search.get("merge_precision"),
            "val_merge_recall": search.get("merge_recall"),
            "val_merge_f1": search.get("merge_f1"),
        },
        "argmax": {"val": argmax_val, "test": argmax_test},
        "calibrated": {
            "val": metric_payload(val_pred, val_target),
            "test": metric_payload(test_pred, test_target),
        },
        "precision_constrained": precision_constrained_payload(
            val_prob,
            val_target,
            test_prob,
            test_target,
            tau_values=tau_grid(args.tau_min, args.tau_max, args.tau_step),
            floors=parse_float_list(args.precision_floors),
            mode=args.mode,
            val_merge_allowed=val_merge_allowed,
            test_merge_allowed=test_merge_allowed,
            torch=torch,
        ),
        "top_val_candidates": search["top_candidates"][:20],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(payload)
    print(f"wrote {args.output_json}")
    return 0


def collect_logits(model: Any, loader: Any, *, device: Any, torch: Any) -> tuple[Any, Any]:
    logits = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits.append(model(batch).detach().cpu())
            targets.append(batch.y.detach().cpu().long())
    if not logits:
        raise ValueError("Empty loader")
    merged_logits = torch.cat(logits, dim=0)
    merged_targets = torch.cat(targets, dim=0)
    merged_targets = torch.where(merged_targets >= 2, torch.full_like(merged_targets, 2), merged_targets)
    return merged_logits, merged_targets


def collect_merge_gate_mask(samples: list[Any], *, torch: Any) -> Any:
    """Return a CPU bool tensor indicating which edges may be MERGE-contracted.

    The raw edge classifier is deliberately over-complete: graph construction
    keeps recall high, then TreeDecoder refuses structurally impossible merges.
    This mask lets threshold calibration measure that deployable constrained
    decoder without changing training labels or logits.
    """

    masks = []
    for sample in samples:
        edge_count = int(sample.edge_index.shape[1])
        records = getattr(sample, "node_records", None)
        if not isinstance(records, list) or len(records) < int(sample.num_nodes):
            masks.append(torch.ones(edge_count, dtype=torch.bool))
            continue
        allowed = torch.zeros(edge_count, dtype=torch.bool)
        edge_index = sample.edge_index.detach().cpu().long()
        for edge_pos in range(edge_count):
            source = int(edge_index[0, edge_pos].item())
            target = int(edge_index[1, edge_pos].item())
            if 0 <= source < len(records) and 0 <= target < len(records):
                allowed[edge_pos] = bool(can_contract_merge_records(records[source], records[target]))
        masks.append(allowed)
    if not masks:
        return torch.zeros((0,), dtype=torch.bool)
    return torch.cat(masks, dim=0)


def tau_grid(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = float(start)
    while current <= float(stop) + 1e-9:
        values.append(round(current, 6))
        current += float(step)
    return values


def search_thresholds(
    prob: Any,
    target: Any,
    *,
    tau_values: list[float],
    mode: str,
    min_merge_precision: float = 0.0,
    merge_allowed: Any | None = None,
    torch: Any,
) -> dict[str, Any]:
    if mode == "threshold_priority":
        return search_thresholds_numpy_priority(
            prob,
            target,
            tau_values=tau_values,
            min_merge_precision=min_merge_precision,
            merge_allowed=merge_allowed,
        )

    best: dict[str, Any] | None = None
    top: list[dict[str, Any]] = []
    for tau_merge in tau_values:
        for tau_parent in tau_values:
            pred = predict_with_thresholds(
                prob,
                tau_merge=tau_merge,
                tau_parent=tau_parent,
                mode=mode,
                merge_allowed=merge_allowed,
                torch=torch,
            )
            metrics = edge_precision_recall_f1(pred, target, num_classes=3)
            positive_macro = (metrics.per_class[0]["f1"] + metrics.per_class[1]["f1"]) / 2.0
            row = {
                "tau_merge": float(tau_merge),
                "tau_parent": float(tau_parent),
                "val_positive_macro_f1": positive_macro,
                "val_macro_f1": metrics.macro_f1,
                "merge_precision": metrics.per_class[0]["precision"],
                "merge_recall": metrics.per_class[0]["recall"],
                "merge_f1": metrics.per_class[0]["f1"],
                "parent_f1": metrics.per_class[1]["f1"],
            }
            top.append(row)
            if float(row["merge_precision"]) < float(min_merge_precision):
                continue
            if best is None or sort_key(row) > sort_key(best):
                best = row
    top.sort(key=sort_key, reverse=True)
    if best is None:
        best = {**top[0], "constraint_satisfied": False}
    else:
        best = {**best, "constraint_satisfied": True}
    return {**best, "top_candidates": top}


def search_thresholds_numpy_priority(
    prob: Any,
    target: Any,
    *,
    tau_values: list[float],
    min_merge_precision: float = 0.0,
    merge_allowed: Any | None = None,
) -> dict[str, Any]:
    """Fast CPU grid search for MERGE-priority thresholding.

    The straightforward torch metric loop is correct but slow because it
    creates many small boolean reductions per grid point.  NumPy's vectorized
    reductions keep the same 0.01 grid search practical on ~200k validation
    edges.
    """

    import numpy as np

    prob_np = prob.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    p_merge = prob_np[:, 0]
    p_parent = prob_np[:, 1]
    if merge_allowed is None:
        merge_allowed_np = np.ones_like(p_merge, dtype=bool)
    else:
        merge_allowed_np = merge_allowed.detach().cpu().numpy().astype(bool)
    y_merge = target_np == 0
    y_parent = target_np == 1
    y_none = target_np == 2
    top: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for tau_merge in tau_values:
        merge_mask = (p_merge >= tau_merge) & merge_allowed_np
        not_merge = ~merge_mask
        merge_precision, merge_recall, f1_merge = precision_recall_f1_from_masks(merge_mask, y_merge, np=np)
        for tau_parent in tau_values:
            parent_mask = not_merge & (p_parent >= tau_parent)
            none_mask = not_merge & (p_parent < tau_parent)
            parent_precision, parent_recall, f1_parent = precision_recall_f1_from_masks(parent_mask, y_parent, np=np)
            _, _, f1_none = precision_recall_f1_from_masks(none_mask, y_none, np=np)
            positive_macro = (f1_merge + f1_parent) / 2.0
            macro = (f1_merge + f1_parent + f1_none) / 3.0
            row = {
                "tau_merge": float(tau_merge),
                "tau_parent": float(tau_parent),
                "val_positive_macro_f1": float(positive_macro),
                "val_macro_f1": float(macro),
                "merge_precision": float(merge_precision),
                "merge_recall": float(merge_recall),
                "merge_f1": float(f1_merge),
                "parent_precision": float(parent_precision),
                "parent_recall": float(parent_recall),
                "parent_f1": float(f1_parent),
            }
            top.append(row)
            if float(row["merge_precision"]) < float(min_merge_precision):
                continue
            if best is None or sort_key(row) > sort_key(best):
                best = row

    top.sort(key=sort_key, reverse=True)
    if best is None:
        best = {**top[0], "constraint_satisfied": False}
    else:
        best = {**best, "constraint_satisfied": True}
    return {**best, "top_candidates": top}


def f1_from_masks(pred_mask: Any, true_mask: Any, *, np: Any) -> float:
    return precision_recall_f1_from_masks(pred_mask, true_mask, np=np)[2]


def precision_recall_f1_from_masks(pred_mask: Any, true_mask: Any, *, np: Any) -> tuple[float, float, float]:
    true_positive = int(np.count_nonzero(pred_mask & true_mask))
    false_positive = int(np.count_nonzero(pred_mask & ~true_mask))
    false_negative = int(np.count_nonzero(~pred_mask & true_mask))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def precision_constrained_payload(
    val_prob: Any,
    val_target: Any,
    test_prob: Any,
    test_target: Any,
    *,
    tau_values: list[float],
    floors: list[float],
    mode: str,
    val_merge_allowed: Any | None = None,
    test_merge_allowed: Any | None = None,
    torch: Any,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for floor in floors:
        search = search_thresholds(
            val_prob,
            val_target,
            tau_values=tau_values,
            mode=mode,
            min_merge_precision=floor,
            merge_allowed=val_merge_allowed,
            torch=torch,
        )
        test_pred = predict_with_thresholds(
            test_prob,
            tau_merge=search["tau_merge"],
            tau_parent=search["tau_parent"],
            mode=mode,
            merge_allowed=test_merge_allowed,
            torch=torch,
        )
        val_pred = predict_with_thresholds(
            val_prob,
            tau_merge=search["tau_merge"],
            tau_parent=search["tau_parent"],
            mode=mode,
            merge_allowed=val_merge_allowed,
            torch=torch,
        )
        payload.append(
            {
                "val_precision_floor": float(floor),
                "constraint_satisfied": bool(search.get("constraint_satisfied", True)),
                "tau_merge": search["tau_merge"],
                "tau_parent": search["tau_parent"],
                "val": metric_payload(val_pred, val_target),
                "test": metric_payload(test_pred, test_target),
            }
        )
    return payload


def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["val_positive_macro_f1"]),
        float(row["merge_f1"]),
        float(row["parent_f1"]),
        float(row["val_macro_f1"]),
    )


def parse_float_list(value: str) -> list[float]:
    floors = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        floors.append(float(part))
    return floors


def predict_with_thresholds(
    prob: Any,
    *,
    tau_merge: float,
    tau_parent: float,
    mode: str,
    merge_allowed: Any | None = None,
    torch: Any,
) -> Any:
    if merge_allowed is None:
        allowed = torch.ones((prob.shape[0],), dtype=torch.bool, device=prob.device)
    else:
        allowed = merge_allowed.to(device=prob.device, dtype=torch.bool)
    if mode == "scaled_argmax":
        thresholds = torch.tensor([tau_merge, tau_parent, 1.0], dtype=prob.dtype, device=prob.device)
        passes = prob >= thresholds
        passes[:, 0] = passes[:, 0] & allowed
        scaled = prob / thresholds.clamp_min(1e-12)
        scaled = torch.where(passes, scaled, torch.full_like(scaled, -1.0))
        pred = scaled.argmax(dim=-1)
        return torch.where(scaled.max(dim=-1).values < 0.0, torch.full_like(pred, 2), pred)

    pred = torch.full((prob.shape[0],), 2, dtype=torch.long, device=prob.device)
    merge_mask = (prob[:, 0] >= tau_merge) & allowed
    parent_mask = (~merge_mask) & (prob[:, 1] >= tau_parent)
    pred[parent_mask] = 1
    pred[merge_mask] = 0
    return pred


def argmax_with_merge_gates(scores: Any, merge_allowed: Any | None, *, torch: Any) -> Any:
    if merge_allowed is None:
        return scores.argmax(dim=-1)
    gated = scores.clone()
    allowed = merge_allowed.to(device=gated.device, dtype=torch.bool)
    gated[~allowed, 0] = -torch.inf
    return gated.argmax(dim=-1)


def metric_payload(pred: Any, target: Any) -> dict[str, Any]:
    metrics = edge_precision_recall_f1(pred, target, num_classes=3)
    positive_macro = (metrics.per_class[0]["f1"] + metrics.per_class[1]["f1"]) / 2.0
    return {
        "macro_f1": metrics.macro_f1,
        "positive_macro_f1": positive_macro,
        "per_class": metrics.per_class,
        "class_counts": count_labels(target),
        "pred_counts": count_labels(pred),
    }


def count_labels(labels: Any) -> dict[str, int]:
    import torch

    y = labels.detach().cpu().long()
    y = torch.where(y >= 2, torch.full_like(y, 2), y)
    counts = torch.bincount(y, minlength=3).tolist()
    return {LABEL_NAMES[idx]: int(counts[idx]) for idx in range(3)}


def print_summary(payload: dict[str, Any]) -> None:
    print("threshold calibration")
    print(f"mode={payload['mode']}")
    print(f"thresholds={payload['best_thresholds']}")
    for section in ("argmax", "calibrated"):
        print(section)
        for split in ("val", "test"):
            metrics = payload[section][split]
            merge = metrics["per_class"]["0"] if "0" in metrics["per_class"] else metrics["per_class"][0]
            parent = metrics["per_class"]["1"] if "1" in metrics["per_class"] else metrics["per_class"][1]
            print(
                f"  {split}: pos_f1={metrics['positive_macro_f1']:.4f} macro={metrics['macro_f1']:.4f} "
                f"merge_f1={merge['f1']:.4f} parent_f1={parent['f1']:.4f} pred={metrics['pred_counts']}"
            )
    if payload.get("precision_constrained"):
        print("precision_constrained")
        for row in payload["precision_constrained"]:
            merge = row["test"]["per_class"]["0"] if "0" in row["test"]["per_class"] else row["test"]["per_class"][0]
            parent = row["test"]["per_class"]["1"] if "1" in row["test"]["per_class"] else row["test"]["per_class"][1]
            status = "ok" if row.get("constraint_satisfied", True) else "unmet"
            print(
                f"  floor={row['val_precision_floor']:.2f} status={status} tau=({row['tau_merge']:.2f},{row['tau_parent']:.2f}) "
                f"test_merge P/R/F1={merge['precision']:.4f}/{merge['recall']:.4f}/{merge['f1']:.4f} "
                f"test_parent_f1={parent['f1']:.4f} pos_f1={row['test']['positive_macro_f1']:.4f}"
            )


def resolve_device(value: str, *, torch: Any) -> Any:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    raise SystemExit(main())
