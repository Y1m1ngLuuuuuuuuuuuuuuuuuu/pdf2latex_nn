#!/usr/bin/env python3
"""Ablate v8 atomic edge owner features on the selected graph family.

This is a lightweight edge-level diagnostic.  It does not rebuild graphs, change
labels, train the project GNN, or run E2E.  It compares the same linear probe
with and without owner indicator columns in ``edge_attr``.
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
class FeatureVariant:
    name: str
    zero_features: tuple[str, ...]


VARIANTS = (
    FeatureVariant("with_owner_features", ()),
    FeatureVariant("no_owner_features", OWNER_FEATURES),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policies", nargs="*", default=["current_weighted", "body_list_focus", "body_list_float_skip_focus"])
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
    for variant in VARIANTS:
        zero_indices = [owner_indices[name] for name in variant.zero_features]
        for policy in args.policies:
            result = train_probe_for_variant(
                policy=policy,
                variant=variant.name,
                train_paths=train_paths,
                val_paths=val_paths,
                torch=torch,
                device=device,
                epochs=args.epochs,
                max_train_edges=args.max_train_edges,
                max_val_edges=args.max_val_edges,
                seed=args.seed,
                zero_indices=zero_indices,
            )
            key = f"{policy}::{variant.name}"
            threshold_results[key] = result["threshold_grid"]
            write_csv(out_dir / f"threshold_grid_{policy}_{variant.name}.csv", result["threshold_grid"])
            rows.append(result["summary"])

    deltas = compute_owner_deltas(rows)
    summary = {
        "schema_version": "v8_atomic_owner_feature_ablation_v1",
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
        "rows": rows,
        "deltas": deltas,
        "interpretation": interpret_deltas(deltas),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "threshold_results.json", threshold_results)
    write_csv(out_dir / "owner_feature_ablation_rows.csv", rows)
    write_csv(out_dir / "owner_feature_ablation_deltas.csv", deltas)
    write_report(out_dir / "OWNER_FEATURE_ABLATION_REPORT.md", summary)


def resolve_feature_indices(path: Path, *, torch: Any, feature_names: tuple[str, ...]) -> dict[str, int]:
    data = torch.load(path, weights_only=False, map_location="cpu")
    schema = list(getattr(data, "edge_attr_schema", []))
    missing = [name for name in feature_names if name not in schema]
    if missing:
        raise SystemExit(f"edge_attr_schema missing owner features: {missing}")
    return {name: schema.index(name) for name in feature_names}


def train_probe_for_variant(
    *,
    policy: str,
    variant: str,
    train_paths: list[Path],
    val_paths: list[Path],
    torch: Any,
    device: Any,
    epochs: int,
    max_train_edges: int,
    max_val_edges: int,
    seed: int,
    zero_indices: list[int],
) -> dict[str, Any]:
    train_x, train_y, train_w = load_edge_matrix_variant(
        train_paths,
        policy,
        torch=torch,
        max_edges=max_train_edges,
        seed=seed,
        zero_indices=zero_indices,
    )
    val_x, val_y, _ = load_edge_matrix_variant(
        val_paths,
        policy,
        torch=torch,
        max_edges=max_val_edges,
        seed=seed + 1,
        zero_indices=zero_indices,
    )
    if train_x.numel() == 0 or val_x.numel() == 0:
        return {
            "summary": {
                "policy": policy,
                "variant": variant,
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
    grid = threshold_grid(probs, val_y_cpu, policy=f"{policy}_{variant}")
    best = max(grid, key=lambda row: row["merge_f1"]) if grid else {}
    summary = {
        "policy": policy,
        "variant": variant,
        "status": "ok",
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


def load_edge_matrix_variant(
    paths: list[Path],
    policy: str,
    *,
    torch: Any,
    max_edges: int,
    seed: int,
    zero_indices: list[int],
) -> tuple[Any, Any, Any]:
    xs, ys, ws = [], [], []
    for path in paths:
        data = torch.load(path, weights_only=False, map_location="cpu")
        sel = select_policy_edges(data, policy, torch=torch)
        mask = sel.train_mask
        if int(mask.sum().item()) == 0:
            continue
        x = data.edge_attr[mask].clone()
        if zero_indices:
            x[:, zero_indices] = 0.0
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


def compute_owner_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row.get("policy"), row.get("variant")): row for row in rows if row.get("status") == "ok"}
    policies = sorted({row.get("policy") for row in rows if row.get("status") == "ok"})
    deltas: list[dict[str, Any]] = []
    for policy in policies:
        with_owner = by_key.get((policy, "with_owner_features"))
        no_owner = by_key.get((policy, "no_owner_features"))
        if not with_owner or not no_owner:
            continue
        delta = {
            "policy": policy,
            "best_f1_delta_no_owner_minus_with_owner": _round_delta(no_owner.get("best_merge_f1"), with_owner.get("best_merge_f1")),
            "best_precision_delta_no_owner_minus_with_owner": _round_delta(no_owner.get("best_precision"), with_owner.get("best_precision")),
            "best_recall_delta_no_owner_minus_with_owner": _round_delta(no_owner.get("best_recall"), with_owner.get("best_recall")),
            "best_pred_merge_rate_delta_no_owner_minus_with_owner": _round_delta(no_owner.get("best_pred_merge_rate"), with_owner.get("best_pred_merge_rate")),
            "with_owner_best_f1": with_owner.get("best_merge_f1"),
            "no_owner_best_f1": no_owner.get("best_merge_f1"),
            "with_owner_best_threshold": with_owner.get("best_threshold"),
            "no_owner_best_threshold": no_owner.get("best_threshold"),
        }
        for threshold in ("0_45", "0_50", "0_55", "0_60"):
            delta[f"f1_at_{threshold}_delta_no_owner_minus_with_owner"] = _round_delta(
                no_owner.get(f"f1_at_{threshold}"),
                with_owner.get(f"f1_at_{threshold}"),
            )
        deltas.append(delta)
    return deltas


def _round_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def interpret_deltas(deltas: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    for row in deltas:
        policy = row["policy"]
        delta = row.get("best_f1_delta_no_owner_minus_with_owner")
        if delta is None:
            continue
        if delta <= -0.03:
            notes.append(f"{policy}: owner features are carrying substantial signal; run a no-owner real GNN sanity check before relying on them.")
        elif delta <= -0.01:
            notes.append(f"{policy}: owner features help, but geometry/style still retains most signal.")
        elif abs(delta) < 0.01:
            notes.append(f"{policy}: owner feature ablation barely changes F1; relative geometry/style is doing the useful work.")
        else:
            notes.append(f"{policy}: no-owner is better in this probe; owner features may add noise for this policy.")
    return notes


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V8 Atomic Owner Feature Ablation",
        "",
        "## Status",
        f"- graph_count: {summary['graph_count']}",
        f"- train_doc_count: {summary['train_doc_count']}",
        f"- val_doc_count: {summary['val_doc_count']}",
        f"- edge owner features zeroed in no-owner variant: {', '.join(summary['owner_features'])}",
        "- full_gnn_training: No",
        "- e2e: No",
        "",
        "## Results",
        "| policy | variant | best_threshold | precision | recall | F1 | pred_merge_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row.get('policy')} | {row.get('variant')} | {row.get('best_threshold')} | "
            f"{row.get('best_precision')} | {row.get('best_recall')} | {row.get('best_merge_f1')} | "
            f"{row.get('best_pred_merge_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Delta",
            "| policy | no-owner F1 - with-owner F1 | no-owner precision delta | no-owner recall delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["deltas"]:
        lines.append(
            f"| {row['policy']} | {row.get('best_f1_delta_no_owner_minus_with_owner')} | "
            f"{row.get('best_precision_delta_no_owner_minus_with_owner')} | "
            f"{row.get('best_recall_delta_no_owner_minus_with_owner')} |"
        )
    lines.extend(["", "## Interpretation"])
    for note in summary["interpretation"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
