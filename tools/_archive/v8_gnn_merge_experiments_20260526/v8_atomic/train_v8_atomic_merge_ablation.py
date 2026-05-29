#!/usr/bin/env python3
"""Train selected200 v8 atomic MERGE ablations and compare hard-coded rules.

This is a branch-local trainer for the v8 atomic merge route.  It does not
modify v8 JSON, graph .pt files, labels, generator code, or E2E outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v8_atomic.diagnose_shortcut_feature_ablation import (  # noqa: E402
    ABLATIONS as SHORTCUT_ABLATIONS,
    apply_ablation,
    build_feature_plan,
    feature_plan_for_json,
)
from tools.v8_atomic.diagnose_v8_atomic_training_policy import (  # noqa: E402
    _load_graph_paths,
    select_policy_edges,
    threshold_grid,
    write_csv,
    write_json,
)


FEATURE_VARIANTS = ("all_features",) + tuple(ablation.name for ablation in SHORTCUT_ABLATIONS)


@dataclass
class GraphItem:
    path: Path
    data: Any


class AtomicMergeGNN:  # assigned after torch import
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policies", nargs="*", default=["body_list_focus", "body_list_float_skip_focus"])
    parser.add_argument("--variants", nargs="*", default=list(FEATURE_VARIANTS))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("This trainer requires torch and torch_geometric in the active environment") from exc

    define_model_class(torch=torch, nn=nn, F=F, SAGEConv=SAGEConv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_paths = _load_graph_paths(Path(args.manifest))
    if not graph_paths:
        raise SystemExit("manifest contains no graph paths")
    feature_plan = build_feature_plan(graph_paths[0], torch=torch)
    splits = split_graph_paths(graph_paths, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)
    device = resolve_device(args.device, torch=torch)
    graph_cache = load_graphs(graph_paths, torch=torch)

    rows: list[dict[str, Any]] = []
    hardcoded_rows: list[dict[str, Any]] = []
    threshold_tables: dict[str, list[dict[str, Any]]] = {}

    for policy in args.policies:
        hardcoded_rows.extend(evaluate_hardcoded_baselines(policy, splits, graph_cache, feature_plan=feature_plan, torch=torch))
        for variant in args.variants:
            if variant not in FEATURE_VARIANTS:
                raise SystemExit(f"unknown variant {variant}; allowed={FEATURE_VARIANTS}")
            run_dir = output_dir / "runs" / f"{policy}__{variant}"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = train_one_run(
                policy=policy,
                variant=variant,
                splits=splits,
                graph_cache=graph_cache,
                feature_plan=feature_plan,
                torch=torch,
                device=device,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                seed=args.seed,
                run_dir=run_dir,
                save_checkpoints=args.save_checkpoints,
            )
            rows.append(result["summary"])
            threshold_tables[f"{policy}::{variant}"] = result["test_threshold_grid"]
            write_csv(run_dir / "test_threshold_grid.csv", result["test_threshold_grid"])

    summary = {
        "schema_version": "v8_atomic_merge_training_ablation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "graph_count": len(graph_paths),
        "splits": {name: [str(path) for path in paths] for name, paths in splits.items()},
        "device": str(device),
        "epochs": args.epochs,
        "policies": args.policies,
        "variants": args.variants,
        "feature_plan": feature_plan_for_json(feature_plan),
        "training_rows": rows,
        "hardcoded_baselines": hardcoded_rows,
        "interpretation": interpret(rows, hardcoded_rows),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "threshold_tables.json", threshold_tables)
    write_csv(output_dir / "training_ablation_summary.csv", rows)
    write_csv(output_dir / "hardcoded_baseline_summary.csv", hardcoded_rows)
    write_report(output_dir / "V8_ATOMIC_MERGE_TRAINING_ABLATION_REPORT.md", summary)


def define_model_class(*, torch: Any, nn: Any, F: Any, SAGEConv: Any) -> None:
    global AtomicMergeGNN

    class _AtomicMergeGNN(nn.Module):
        def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int, dropout: float):
            super().__init__()
            self.dropout = float(dropout)
            self.node_encoder = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            self.convs = nn.ModuleList(SAGEConv(hidden_dim, hidden_dim) for _ in range(max(0, num_layers)))
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 4 + edge_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, data: Any, edge_attr: Any) -> Any:
            x = data.x
            edge_index = data.edge_index
            h = self.node_encoder(x)
            if self.convs:
                prop_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                for conv in self.convs:
                    h = conv(h, prop_edge_index)
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
            src, dst = edge_index
            hs = h[src]
            hd = h[dst]
            edge_features = torch.cat([hs, hd, hs - hd, hs * hd, edge_attr], dim=-1)
            return self.edge_mlp(edge_features).squeeze(-1)

    AtomicMergeGNN = _AtomicMergeGNN


def split_graph_paths(
    paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    seed: int,
) -> dict[str, list[Path]]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_n = max(1, int(round(n * train_ratio)))
    val_n = max(1, int(round(n * val_ratio)))
    if train_n + val_n >= n:
        val_n = max(1, n - train_n - 1)
    return {
        "train": shuffled[:train_n],
        "val": shuffled[train_n : train_n + val_n],
        "test": shuffled[train_n + val_n :],
    }


def resolve_device(value: str, *, torch: Any) -> Any:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_graphs(paths: list[Path], *, torch: Any) -> dict[Path, GraphItem]:
    cache: dict[Path, GraphItem] = {}
    for path in paths:
        data = torch.load(path, weights_only=False, map_location="cpu")
        cache[path] = GraphItem(path=path, data=data)
    return cache


def train_one_run(
    *,
    policy: str,
    variant: str,
    splits: dict[str, list[Path]],
    graph_cache: dict[Path, GraphItem],
    feature_plan: dict[str, Any],
    torch: Any,
    device: Any,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    run_dir: Path,
    save_checkpoints: bool,
) -> dict[str, Any]:
    sample = graph_cache[splits["train"][0]].data
    model = AtomicMergeGNN(
        node_dim=int(sample.x.shape[1]),
        edge_dim=int(sample.edge_attr.shape[1]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_stats = collect_split_stats(policy, splits["train"], graph_cache, torch=torch)
    pos_weight = torch.tensor(
        [max(0.1, min(20.0, train_stats["none_negative"] / max(1.0, train_stats["merge_positive"])))],
        device=device,
    )
    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_state = None
    best_threshold = 0.5
    rng = random.Random(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        train_paths = list(splits["train"])
        rng.shuffle(train_paths)
        total_loss = 0.0
        total_docs = 0
        for path in train_paths:
            cpu_data = graph_cache[path].data
            selection = select_policy_edges(cpu_data, policy, torch=torch)
            mask = selection.train_mask.to(device)
            if int(mask.sum().item()) == 0:
                continue
            data = cpu_data.clone().to(device)
            edge_attr = transformed_edge_attr(data, variant, feature_plan=feature_plan, torch=torch).to(device)
            target = selection.target.float().to(device)
            weights = selection.sample_weight.float().to(device).clamp_min(0.0)
            optimizer.zero_grad(set_to_none=True)
            logits = model(data, edge_attr)
            loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
            effective_weight = weights * mask.float()
            denom = effective_weight.sum().clamp_min(1.0)
            loss = (loss_raw * effective_weight).sum() / denom
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            total_docs += 1
        val_eval = evaluate_split(model, policy, variant, splits["val"], graph_cache, feature_plan=feature_plan, torch=torch, device=device)
        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, total_docs), 6),
            "val_best_f1": val_eval["best"]["merge_f1"],
            "val_best_threshold": val_eval["best"]["threshold"],
            "val_best_precision": val_eval["best"]["merge_precision"],
            "val_best_recall": val_eval["best"]["merge_recall"],
        }
        history.append(row)
        if row["val_best_f1"] > best_val_f1:
            best_val_f1 = float(row["val_best_f1"])
            best_threshold = float(row["val_best_threshold"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_eval = evaluate_split(model, policy, variant, splits["val"], graph_cache, feature_plan=feature_plan, torch=torch, device=device)
    test_eval = evaluate_split(model, policy, variant, splits["test"], graph_cache, feature_plan=feature_plan, torch=torch, device=device)
    test_at_val_threshold = metrics_at_threshold(test_eval["probs"], test_eval["target"], threshold=best_threshold, torch=torch)
    if save_checkpoints and best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "policy": policy,
                "variant": variant,
                "feature_plan": feature_plan_for_json(feature_plan),
                "best_threshold": best_threshold,
                "node_dim": int(sample.x.shape[1]),
                "edge_dim": int(sample.edge_attr.shape[1]),
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
            },
            run_dir / "best_model.pth",
        )
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "val_metrics.json", val_eval["summary"])
    write_json(run_dir / "test_metrics.json", {**test_eval["summary"], "test_at_val_threshold": test_at_val_threshold})
    summary = {
        "policy": policy,
        "variant": variant,
        "train_docs": len(splits["train"]),
        "val_docs": len(splits["val"]),
        "test_docs": len(splits["test"]),
        "train_edges": train_stats["trainable_total"],
        "train_merge_positive": train_stats["merge_positive"],
        "train_none_negative": train_stats["none_negative"],
        "best_val_threshold": round(best_threshold, 4),
        "best_val_f1": round(best_val_f1, 6),
        "test_best_threshold": test_eval["best"]["threshold"],
        "test_best_precision": test_eval["best"]["merge_precision"],
        "test_best_recall": test_eval["best"]["merge_recall"],
        "test_best_f1": test_eval["best"]["merge_f1"],
        "test_best_pred_merge_rate": test_eval["best"]["pred_merge_rate"],
        "test_precision_at_val_threshold": test_at_val_threshold["merge_precision"],
        "test_recall_at_val_threshold": test_at_val_threshold["merge_recall"],
        "test_f1_at_val_threshold": test_at_val_threshold["merge_f1"],
        "test_pred_merge_rate_at_val_threshold": test_at_val_threshold["pred_merge_rate"],
    }
    return {"summary": summary, "test_threshold_grid": test_eval["threshold_grid"]}


def transformed_edge_attr(data: Any, variant: str, *, feature_plan: dict[str, Any], torch: Any) -> Any:
    edge_attr = data.edge_attr.clone()
    if variant == "all_features":
        return edge_attr
    apply_ablation(edge_attr, ablation=variant, feature_plan=feature_plan)
    return edge_attr


def collect_split_stats(policy: str, paths: list[Path], graph_cache: dict[Path, GraphItem], *, torch: Any) -> dict[str, int]:
    stats = {"trainable_total": 0, "merge_positive": 0, "none_negative": 0}
    for path in paths:
        data = graph_cache[path].data
        selection = select_policy_edges(data, policy, torch=torch)
        mask = selection.train_mask
        target = selection.target
        stats["trainable_total"] += int(mask.sum().item())
        stats["merge_positive"] += int(((target == 1) & mask).sum().item())
        stats["none_negative"] += int(((target == 0) & mask).sum().item())
    return stats


def evaluate_split(
    model: Any,
    policy: str,
    variant: str,
    paths: list[Path],
    graph_cache: dict[Path, GraphItem],
    *,
    feature_plan: dict[str, Any],
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    model.eval()
    probs_list = []
    target_list = []
    with torch.no_grad():
        for path in paths:
            cpu_data = graph_cache[path].data
            selection = select_policy_edges(cpu_data, policy, torch=torch)
            mask_cpu = selection.train_mask
            if int(mask_cpu.sum().item()) == 0:
                continue
            data = cpu_data.clone().to(device)
            edge_attr = transformed_edge_attr(data, variant, feature_plan=feature_plan, torch=torch).to(device)
            mask = mask_cpu.to(device)
            logits = model(data, edge_attr)
            probs_list.append(torch.sigmoid(logits[mask]).detach().cpu())
            target_list.append(selection.target[mask_cpu].detach().cpu().long())
    if not probs_list:
        empty = {"merge_precision": 0.0, "merge_recall": 0.0, "merge_f1": 0.0, "pred_merge_rate": 0.0, "threshold": 0.5}
        return {"summary": {"status": "empty"}, "best": empty, "threshold_grid": [], "probs": torch.empty(0), "target": torch.empty(0)}
    probs = torch.cat(probs_list, dim=0)
    target = torch.cat(target_list, dim=0)
    grid = threshold_grid(probs, target, policy=f"{policy}_{variant}")
    best = max(grid, key=lambda row: row["merge_f1"])
    return {
        "summary": {
            "status": "ok",
            "edge_count": int(target.numel()),
            "merge_rate": round(float((target == 1).float().mean().item()), 6),
            "best": best,
        },
        "best": best,
        "threshold_grid": grid,
        "probs": probs,
        "target": target,
    }


def metrics_at_threshold(probs: Any, target: Any, *, threshold: float, torch: Any) -> dict[str, Any]:
    if int(target.numel()) == 0:
        return {"threshold": threshold, "merge_precision": 0.0, "merge_recall": 0.0, "merge_f1": 0.0, "pred_merge_rate": 0.0}
    pred = (probs >= float(threshold)).long()
    tp = int(((pred == 1) & (target == 1)).sum().item())
    fp = int(((pred == 1) & (target == 0)).sum().item())
    fn = int(((pred == 0) & (target == 1)).sum().item())
    tn = int(((pred == 0) & (target == 0)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "threshold": round(float(threshold), 4),
        "merge_precision": round(precision, 6),
        "merge_recall": round(recall, 6),
        "merge_f1": round(f1, 6),
        "pred_merge_rate": round(float((pred == 1).float().mean().item()), 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_hardcoded_baselines(
    policy: str,
    splits: dict[str, list[Path]],
    graph_cache: dict[Path, GraphItem],
    *,
    feature_plan: dict[str, Any],
    torch: Any,
) -> list[dict[str, Any]]:
    rows = []
    for split_name, paths in splits.items():
        rows.append(evaluate_same_middle_baseline(policy, split_name, paths, graph_cache, feature_plan=feature_plan, torch=torch))
        rows.append(evaluate_label_source_baseline(policy, split_name, paths, graph_cache, torch=torch))
    return rows


def evaluate_same_middle_baseline(
    policy: str,
    split_name: str,
    paths: list[Path],
    graph_cache: dict[Path, GraphItem],
    *,
    feature_plan: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    owner_index = feature_plan["schema"].index("same_middle_block")
    preds = []
    targets = []
    for path in paths:
        data = graph_cache[path].data
        selection = select_policy_edges(data, policy, torch=torch)
        mask = selection.train_mask
        if int(mask.sum().item()) == 0:
            continue
        preds.append((data.edge_attr[mask, owner_index] > 0.5).long())
        targets.append(selection.target[mask].long())
    return baseline_row(policy, split_name, "hardcoded_same_middle_block_feature", preds, targets, torch=torch)


def evaluate_label_source_baseline(
    policy: str,
    split_name: str,
    paths: list[Path],
    graph_cache: dict[Path, GraphItem],
    *,
    torch: Any,
) -> dict[str, Any]:
    preds = []
    targets = []
    hardcoded_sources = {"frontend_v8_deterministic_merge", "frontend_same_middle_block_lines"}
    for path in paths:
        data = graph_cache[path].data
        selection = select_policy_edges(data, policy, torch=torch)
        mask = selection.train_mask
        if int(mask.sum().item()) == 0:
            continue
        labels_by_edge = load_label_sources(data)
        pred_values = []
        for rec in data.edge_records:
            label_rec = labels_by_edge.get(str(rec.get("edge_id")), {})
            pred_values.append(1 if label_rec.get("label_source") in hardcoded_sources else 0)
        preds.append(torch.tensor(pred_values, dtype=torch.long)[mask])
        targets.append(selection.target[mask].long())
    row = baseline_row(policy, split_name, "hardcoded_label_source_frontend", preds, targets, torch=torch)
    row["note"] = "Uses label sidecar source; not an independent model baseline."
    return row


def load_label_sources(data: Any) -> dict[str, dict[str, Any]]:
    path_value = getattr(data, "source_labels", None)
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("edge_id")): row for row in payload.get("edge_labels", [])}


def baseline_row(policy: str, split_name: str, name: str, preds: list[Any], targets: list[Any], *, torch: Any) -> dict[str, Any]:
    if not preds:
        return {"policy": policy, "split": split_name, "baseline": name, "edge_count": 0, "merge_precision": 0.0, "merge_recall": 0.0, "merge_f1": 0.0}
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    tp = int(((pred == 1) & (target == 1)).sum().item())
    fp = int(((pred == 1) & (target == 0)).sum().item())
    fn = int(((pred == 0) & (target == 1)).sum().item())
    tn = int(((pred == 0) & (target == 0)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "policy": policy,
        "split": split_name,
        "baseline": name,
        "edge_count": int(target.numel()),
        "merge_rate": round(float((target == 1).float().mean().item()), 6),
        "pred_merge_rate": round(float((pred == 1).float().mean().item()), 6),
        "merge_precision": round(precision, 6),
        "merge_recall": round(recall, 6),
        "merge_f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def interpret(rows: list[dict[str, Any]], hardcoded_rows: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    for policy in sorted({row["policy"] for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        if not policy_rows:
            continue
        best = max(policy_rows, key=lambda row: row.get("test_f1_at_val_threshold", 0.0))
        notes.append(
            f"{policy}: best test-at-val-threshold variant is {best['variant']} "
            f"with F1={best['test_f1_at_val_threshold']}."
        )
        strict = next((row for row in policy_rows if row["variant"] == "C_strict_visual_text_style"), None)
        owner = next((row for row in policy_rows if row["variant"] == "all_features"), None)
        if strict and owner:
            delta = round(strict["test_f1_at_val_threshold"] - owner["test_f1_at_val_threshold"], 6)
            notes.append(f"{policy}: strict visual/text/style minus all_features F1={delta}.")
    test_hardcoded = [row for row in hardcoded_rows if row["split"] == "test"]
    for row in test_hardcoded:
        notes.append(
            f"{row['policy']} {row['baseline']} test F1={row['merge_f1']} "
            f"(pred_merge_rate={row.get('pred_merge_rate')})."
        )
    return notes


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V8 Atomic MERGE Training Ablation",
        "",
        "## Status",
        f"- graph_count: {summary['graph_count']}",
        f"- device: {summary['device']}",
        f"- epochs: {summary['epochs']}",
        "- full E2E: No",
        "- graph rebuild/relabel: No",
        "",
        "## Training Results",
        "| policy | variant | val threshold | test P | test R | test F1 | test pred merge rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["training_rows"]:
        lines.append(
            f"| {row['policy']} | {row['variant']} | {row['best_val_threshold']} | "
            f"{row['test_precision_at_val_threshold']} | {row['test_recall_at_val_threshold']} | "
            f"{row['test_f1_at_val_threshold']} | {row['test_pred_merge_rate_at_val_threshold']} |"
        )
    lines.extend(
        [
            "",
            "## Hard-coded Baselines",
            "| policy | split | baseline | P | R | F1 | pred merge rate | note |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["hardcoded_baselines"]:
        lines.append(
            f"| {row['policy']} | {row['split']} | {row['baseline']} | {row['merge_precision']} | "
            f"{row['merge_recall']} | {row['merge_f1']} | {row.get('pred_merge_rate', '')} | {row.get('note', '')} |"
        )
    lines.extend(["", "## Interpretation"])
    for note in summary["interpretation"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
