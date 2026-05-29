#!/usr/bin/env python3
"""Diagnose whether v8 atomic MERGE probes learn beyond owner features.

This script is intentionally a small edge-level diagnostic.  It does not train
the project GNN, rebuild graphs, relabel data, or run E2E.  It compares several
feature transforms on the same selected graph family:

* all_features: normal v1.1 edge attributes.
* owner_only: keep only middle/content/style owner indicators.
* no_owner: zero owner indicators during train and validation.
* train_all_eval_no_owner: train with all features, then remove owner features
  only at validation time to test whether the trained probe retained non-owner
  fallback signal.
* owner_dropout_p50: randomly zero owner indicators for half of train edges.
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

from tools.v8_atomic.diagnose_v8_atomic_training_policy import (  # noqa: E402
    POLICIES,
    _doc_split,
    _load_graph_paths,
    _resolve_device,
    select_policy_edges,
    threshold_grid,
    write_csv,
    write_json,
)


OWNER_FEATURES = (
    "same_middle_block",
    "same_content_owner",
    "same_style_content_owner",
)


@dataclass(frozen=True)
class Scenario:
    name: str
    train_transform: str
    eval_transform: str
    note: str


SCENARIOS = (
    Scenario("all_features", "all", "all", "Normal v1.1 feature set."),
    Scenario("owner_only", "owner_only", "owner_only", "Only owner indicator columns are visible."),
    Scenario("no_owner", "no_owner", "no_owner", "Owner indicator columns are zeroed in train and validation."),
    Scenario(
        "train_all_eval_no_owner",
        "all",
        "no_owner",
        "Train with owner, remove owner only at validation to expose reliance.",
    ),
    Scenario(
        "train_all_eval_owner_permuted",
        "all",
        "owner_permuted",
        "Train with owner, shuffle owner columns at validation.",
    ),
    Scenario(
        "owner_dropout_p50_eval_all",
        "owner_dropout_p50",
        "all",
        "Randomly zero owner columns for half of train edges, evaluate normally.",
    ),
    Scenario(
        "owner_dropout_p50_eval_no_owner",
        "owner_dropout_p50",
        "no_owner",
        "Randomly zero owner columns for half of train edges, evaluate with owner removed.",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policies", nargs="*", default=["body_list_focus", "body_list_float_skip_focus"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-edges", type=int, default=180000)
    parser.add_argument("--max-val-edges", type=int, default=80000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("This diagnostic requires torch in the active environment") from exc

    unknown = [policy for policy in args.policies if policy not in POLICIES]
    if unknown:
        raise SystemExit(f"unknown policies: {unknown}; allowed={POLICIES}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_paths = _load_graph_paths(Path(args.manifest))
    if not graph_paths:
        raise SystemExit("manifest contains no graph paths")
    train_paths, val_paths = _doc_split(graph_paths, seed=args.seed)
    device = _resolve_device(args.device, torch)
    owner_indices = resolve_feature_indices(graph_paths[0], torch=torch, feature_names=OWNER_FEATURES)

    rows: list[dict[str, Any]] = []
    threshold_results: dict[str, list[dict[str, Any]]] = {}
    for policy in args.policies:
        for scenario in SCENARIOS:
            result = run_scenario(
                policy=policy,
                scenario=scenario,
                train_paths=train_paths,
                val_paths=val_paths,
                torch=torch,
                device=device,
                owner_indices=[owner_indices[name] for name in OWNER_FEATURES],
                epochs=args.epochs,
                max_train_edges=args.max_train_edges,
                max_val_edges=args.max_val_edges,
                seed=args.seed,
            )
            rows.append(result["summary"])
            key = f"{policy}::{scenario.name}"
            threshold_results[key] = result["threshold_grid"]
            write_csv(out_dir / f"threshold_grid_{policy}_{scenario.name}.csv", result["threshold_grid"])

    comparisons = compute_comparisons(rows)
    summary = {
        "schema_version": "v8_atomic_owner_dependency_diagnostic_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "graph_count": len(graph_paths),
        "train_doc_count": len(train_paths),
        "val_doc_count": len(val_paths),
        "device": str(device),
        "epochs": args.epochs,
        "owner_features": list(OWNER_FEATURES),
        "owner_feature_indices": owner_indices,
        "policies": args.policies,
        "scenarios": [scenario.__dict__ for scenario in SCENARIOS],
        "rows": rows,
        "comparisons": comparisons,
        "interpretation": interpret(comparisons),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "threshold_results.json", threshold_results)
    write_csv(out_dir / "owner_dependency_rows.csv", rows)
    write_csv(out_dir / "owner_dependency_comparisons.csv", comparisons)
    write_report(out_dir / "OWNER_DEPENDENCY_DIAGNOSTIC_REPORT.md", summary)


def resolve_feature_indices(path: Path, *, torch: Any, feature_names: tuple[str, ...]) -> dict[str, int]:
    data = torch.load(path, weights_only=False, map_location="cpu")
    schema = list(getattr(data, "edge_attr_schema", []))
    missing = [name for name in feature_names if name not in schema]
    if missing:
        raise SystemExit(f"edge_attr_schema missing owner features: {missing}")
    return {name: schema.index(name) for name in feature_names}


def run_scenario(
    *,
    policy: str,
    scenario: Scenario,
    train_paths: list[Path],
    val_paths: list[Path],
    torch: Any,
    device: Any,
    owner_indices: list[int],
    epochs: int,
    max_train_edges: int,
    max_val_edges: int,
    seed: int,
) -> dict[str, Any]:
    train_x, train_y, train_w = load_edge_matrix(
        train_paths,
        policy,
        torch=torch,
        max_edges=max_train_edges,
        seed=seed,
        owner_indices=owner_indices,
        transform=scenario.train_transform,
    )
    val_x, val_y, _ = load_edge_matrix(
        val_paths,
        policy,
        torch=torch,
        max_edges=max_val_edges,
        seed=seed + 1,
        owner_indices=owner_indices,
        transform=scenario.eval_transform,
    )
    if train_x.numel() == 0 or val_x.numel() == 0:
        return {
            "summary": {
                "policy": policy,
                "scenario": scenario.name,
                "status": "skipped_empty_matrix",
            },
            "threshold_grid": [],
        }
    train_x, train_y, train_w = train_x.to(device), train_y.float().to(device), train_w.float().to(device)
    val_x, val_y = val_x.to(device), val_y.long().to(device)
    model = torch.nn.Linear(train_x.shape[1], 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    pos = float((train_y == 1).sum().item())
    neg = float((train_y == 0).sum().item())
    pos_weight = torch.tensor([max(0.1, min(20.0, neg / max(1.0, pos)))], device=device)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(train_x).squeeze(-1)
        loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight, reduction="none")
        loss = (loss_raw * train_w.clamp_min(0.05)).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.sigmoid(model(val_x).squeeze(-1)).detach().cpu()
    val_y_cpu = val_y.detach().cpu()
    grid = threshold_grid(probs, val_y_cpu, policy=f"{policy}_{scenario.name}")
    best = max(grid, key=lambda row: row["merge_f1"]) if grid else {}
    summary = {
        "policy": policy,
        "scenario": scenario.name,
        "status": "ok",
        "train_transform": scenario.train_transform,
        "eval_transform": scenario.eval_transform,
        "train_edges": int(train_y.numel()),
        "val_edges": int(val_y_cpu.numel()),
        "train_merge_rate": round(float((train_y == 1).float().mean().item()), 6),
        "val_merge_rate": round(float((val_y_cpu == 1).float().mean().item()), 6),
        "pos_weight": round(float(pos_weight.item()), 6),
        "best_threshold": best.get("threshold"),
        "best_precision": best.get("merge_precision"),
        "best_recall": best.get("merge_recall"),
        "best_merge_f1": best.get("merge_f1"),
        "best_pred_merge_rate": best.get("pred_merge_rate"),
    }
    for threshold in (0.45, 0.50, 0.55, 0.60):
        row = next((item for item in grid if item["threshold"] == threshold), None)
        if row:
            suffix = f"{threshold:.2f}".replace(".", "_")
            summary[f"f1_at_{suffix}"] = row["merge_f1"]
            summary[f"precision_at_{suffix}"] = row["merge_precision"]
            summary[f"recall_at_{suffix}"] = row["merge_recall"]
            summary[f"pred_merge_rate_at_{suffix}"] = row["pred_merge_rate"]
    return {"summary": summary, "threshold_grid": grid}


def load_edge_matrix(
    paths: list[Path],
    policy: str,
    *,
    torch: Any,
    max_edges: int,
    seed: int,
    owner_indices: list[int],
    transform: str,
) -> tuple[Any, Any, Any]:
    xs, ys, ws = [], [], []
    for path in paths:
        data = torch.load(path, weights_only=False, map_location="cpu")
        sel = select_policy_edges(data, policy, torch=torch)
        mask = sel.train_mask
        if int(mask.sum().item()) == 0:
            continue
        x = data.edge_attr[mask].clone()
        x = transform_features(x, transform=transform, owner_indices=owner_indices, seed=seed + len(xs), torch=torch)
        xs.append(x)
        ys.append(sel.target[mask])
        ws.append(sel.sample_weight[mask])
    if not xs:
        return torch.empty((0, 0)), torch.empty((0,), dtype=torch.long), torch.empty((0,))
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    w = torch.cat(ws, dim=0).float().clamp_min(0.0)
    if x.shape[0] > max_edges:
        generator = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=generator)[:max_edges]
        x, y, w = x[idx], y[idx], w[idx]
    return x, y, w


def transform_features(x: Any, *, transform: str, owner_indices: list[int], seed: int, torch: Any) -> Any:
    if transform == "all":
        return x
    if transform == "no_owner":
        x[:, owner_indices] = 0.0
        return x
    if transform == "owner_only":
        y = torch.zeros_like(x)
        y[:, owner_indices] = x[:, owner_indices]
        return y
    if transform == "owner_permuted":
        generator = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=generator)
        x[:, owner_indices] = x[idx][:, owner_indices]
        return x
    if transform == "owner_dropout_p50":
        generator = torch.Generator().manual_seed(seed)
        keep = torch.rand((x.shape[0],), generator=generator) >= 0.5
        x[~keep][:, owner_indices] = 0.0
        return x
    raise ValueError(f"unknown feature transform: {transform}")


def compute_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row.get("policy"), row.get("scenario")): row for row in rows if row.get("status") == "ok"}
    policies = sorted({row.get("policy") for row in rows if row.get("status") == "ok"})
    comparisons: list[dict[str, Any]] = []
    for policy in policies:
        all_row = by_key.get((policy, "all_features"))
        owner_only = by_key.get((policy, "owner_only"))
        no_owner = by_key.get((policy, "no_owner"))
        eval_no_owner = by_key.get((policy, "train_all_eval_no_owner"))
        eval_permuted = by_key.get((policy, "train_all_eval_owner_permuted"))
        dropout_all = by_key.get((policy, "owner_dropout_p50_eval_all"))
        dropout_no_owner = by_key.get((policy, "owner_dropout_p50_eval_no_owner"))
        if not all_row:
            continue
        comparisons.append(
            {
                "policy": policy,
                "all_f1": all_row.get("best_merge_f1"),
                "owner_only_f1": _get(owner_only, "best_merge_f1"),
                "no_owner_f1": _get(no_owner, "best_merge_f1"),
                "train_all_eval_no_owner_f1": _get(eval_no_owner, "best_merge_f1"),
                "train_all_eval_owner_permuted_f1": _get(eval_permuted, "best_merge_f1"),
                "owner_dropout_eval_all_f1": _get(dropout_all, "best_merge_f1"),
                "owner_dropout_eval_no_owner_f1": _get(dropout_no_owner, "best_merge_f1"),
                "all_minus_owner_only_f1": _delta(all_row, owner_only, "best_merge_f1"),
                "all_minus_no_owner_f1": _delta(all_row, no_owner, "best_merge_f1"),
                "all_minus_train_all_eval_no_owner_f1": _delta(all_row, eval_no_owner, "best_merge_f1"),
                "all_minus_owner_permuted_f1": _delta(all_row, eval_permuted, "best_merge_f1"),
                "dropout_all_minus_all_f1": _delta(dropout_all, all_row, "best_merge_f1"),
                "dropout_no_owner_minus_no_owner_f1": _delta(dropout_no_owner, no_owner, "best_merge_f1"),
            }
        )
    return comparisons


def _get(row: dict[str, Any] | None, key: str) -> Any:
    return row.get(key) if row else None


def _delta(left: dict[str, Any] | None, right: dict[str, Any] | None, key: str) -> float | None:
    if not left or not right:
        return None
    if left.get(key) is None or right.get(key) is None:
        return None
    return round(float(left[key]) - float(right[key]), 6)


def interpret(comparisons: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    for row in comparisons:
        policy = row["policy"]
        all_minus_owner = row.get("all_minus_owner_only_f1")
        all_minus_no_owner = row.get("all_minus_no_owner_f1")
        all_minus_eval_no_owner = row.get("all_minus_train_all_eval_no_owner_f1")
        if all_minus_owner is not None:
            if all_minus_owner >= 0.03:
                notes.append(f"{policy}: all_features beats owner_only by {all_minus_owner}; non-owner geometry/style adds real signal.")
            elif all_minus_owner >= 0.01:
                notes.append(f"{policy}: all_features only modestly beats owner_only; owner dominates but non-owner helps a little.")
            else:
                notes.append(f"{policy}: owner_only is close to all_features; model may mostly rely on owner indicators.")
        if all_minus_no_owner is not None and all_minus_no_owner >= 0.04:
            notes.append(f"{policy}: removing owner during train/validation costs {all_minus_no_owner} F1, so owner signal is material.")
        if all_minus_eval_no_owner is not None and all_minus_eval_no_owner >= 0.08:
            notes.append(f"{policy}: train-with-owner collapses when owner is removed at validation; add owner dropout or report no-owner branch.")
    return notes


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V8 Atomic Owner Dependency Diagnostic",
        "",
        "## Status",
        f"- graph_count: {summary['graph_count']}",
        f"- train_doc_count: {summary['train_doc_count']}",
        f"- val_doc_count: {summary['val_doc_count']}",
        f"- owner_features: {', '.join(summary['owner_features'])}",
        "- full_gnn_training: No",
        "- e2e: No",
        "",
        "## Scenario Results",
        "| policy | scenario | best_threshold | precision | recall | F1 | pred_merge_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row.get('policy')} | {row.get('scenario')} | {row.get('best_threshold')} | "
            f"{row.get('best_precision')} | {row.get('best_recall')} | {row.get('best_merge_f1')} | "
            f"{row.get('best_pred_merge_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "| policy | all | owner_only | no_owner | all-owner_only | all-no_owner | all-train_all_eval_no_owner |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['policy']} | {row.get('all_f1')} | {row.get('owner_only_f1')} | {row.get('no_owner_f1')} | "
            f"{row.get('all_minus_owner_only_f1')} | {row.get('all_minus_no_owner_f1')} | "
            f"{row.get('all_minus_train_all_eval_no_owner_f1')} |"
        )
    lines.extend(["", "## Interpretation"])
    for note in summary["interpretation"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
