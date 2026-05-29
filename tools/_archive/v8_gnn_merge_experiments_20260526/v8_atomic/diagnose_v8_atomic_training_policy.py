#!/usr/bin/env python3
"""Diagnose class weights, loss masks, and thresholds for v8 atomic MERGE graphs.

This is a lightweight diagnostic probe, not the project GNN training path.  It
uses edge attributes only so we can inspect whether a policy is likely to learn
"merge everywhere" before running a real graph model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


POLICIES = [
    "current_weighted",
    "strong_only",
    "body_list_focus",
    "body_list_float_skip_focus",
]


@dataclass(frozen=True)
class PolicySelection:
    train_mask: Any
    sample_weight: Any
    target: Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policies", nargs="*", default=POLICIES)
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

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_paths = _load_graph_paths(Path(args.manifest))
    if not graph_paths:
        raise SystemExit("manifest contains no graph paths")
    train_paths, val_paths = _doc_split(graph_paths, seed=args.seed)
    device = _resolve_device(args.device, torch)

    policy_stats: list[dict[str, Any]] = []
    threshold_results: dict[str, list[dict[str, Any]]] = {}
    probe_summaries: dict[str, Any] = {}

    for policy in args.policies:
        stats = compute_policy_stats(policy, graph_paths, torch=torch)
        policy_stats.append(stats)
        if stats["trainable_total"] == 0 or stats["merge_positive"] == 0 or stats["none_negative"] == 0:
            probe_summaries[policy] = {"status": "skipped_empty_or_single_class"}
            continue
        probe = train_linear_probe(
            policy,
            train_paths,
            val_paths,
            torch=torch,
            device=device,
            epochs=args.epochs,
            max_train_edges=args.max_train_edges,
            max_val_edges=args.max_val_edges,
            seed=args.seed,
        )
        threshold_results[policy] = probe["threshold_grid"]
        probe_summaries[policy] = probe["summary"]
        write_csv(out_dir / f"threshold_grid_{policy}.csv", probe["threshold_grid"])

    summary = {
        "schema_version": "v8_atomic_merge_training_policy_diagnostic_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "graph_count": len(graph_paths),
        "train_doc_count": len(train_paths),
        "val_doc_count": len(val_paths),
        "device": str(device),
        "epochs": args.epochs,
        "policies": args.policies,
        "policy_stats": policy_stats,
        "probe_summaries": probe_summaries,
        "recommendation": recommend_policy(policy_stats, probe_summaries),
    }
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "policy_stats.csv", policy_stats)
    write_json(out_dir / "threshold_results.json", threshold_results)
    write_report(out_dir / "V8_ATOMIC_MERGE_POLICY_DIAGNOSTIC_REPORT.md", summary)


def _load_graph_paths(manifest_path: Path) -> list[Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for item in payload.get("items", []):
        graph_path = item.get("graph_path")
        if graph_path and Path(graph_path).exists():
            paths.append(Path(graph_path))
    return sorted(paths)


def _doc_split(paths: list[Path], *, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * 0.2))
    return shuffled[val_count:], shuffled[:val_count]


def _resolve_device(value: str, torch: Any) -> Any:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def compute_policy_stats(policy: str, paths: list[Path], *, torch: Any) -> dict[str, Any]:
    counts = Counter()
    family_counts = Counter()
    source_counts = Counter()
    strength_counts = Counter()
    weighted = Counter()
    for path in paths:
        data = torch.load(path, weights_only=False, map_location="cpu")
        selection = select_policy_edges(data, policy, torch=torch)
        mask = selection.train_mask.cpu()
        target = selection.target.cpu()
        weight = selection.sample_weight.cpu()
        counts["total_edges"] += int(data.y.numel())
        counts["trainable_total"] += int(mask.sum().item())
        counts["merge_positive"] += int(((target == 1) & mask).sum().item())
        counts["none_negative"] += int(((target == 0) & mask).sum().item())
        weighted["merge_weight"] += float(weight[(target == 1) & mask].sum().item())
        weighted["none_weight"] += float(weight[(target == 0) & mask].sum().item())
        for rec, keep in zip(data.edge_records, mask.tolist()):
            if not keep:
                continue
            family_counts[str(rec.get("candidate_family") or "UNKNOWN")] += 1
            source_counts[str(rec.get("label_source") or "UNKNOWN")] += 1
            strength_counts[str(rec.get("label_strength") or "UNKNOWN")] += 1
    trainable_total = max(1, counts["trainable_total"])
    merge_count = counts["merge_positive"]
    none_count = counts["none_negative"]
    inv_weights = inverse_class_weights(merge_count, none_count)
    sqrt_weights = sqrt_inverse_class_weights(merge_count, none_count)
    return {
        "policy": policy,
        "total_edges": counts["total_edges"],
        "trainable_total": counts["trainable_total"],
        "merge_positive": merge_count,
        "none_negative": none_count,
        "merge_rate": round(merge_count / trainable_total, 6),
        "none_rate": round(none_count / trainable_total, 6),
        "current_merge_weight_sum": round(weighted["merge_weight"], 4),
        "current_none_weight_sum": round(weighted["none_weight"], 4),
        "current_weighted_merge_share": round(
            weighted["merge_weight"] / max(1e-9, weighted["merge_weight"] + weighted["none_weight"]),
            6,
        ),
        "inverse_class_weight_merge": round(inv_weights[1], 6),
        "inverse_class_weight_none": round(inv_weights[0], 6),
        "sqrt_class_weight_merge": round(sqrt_weights[1], 6),
        "sqrt_class_weight_none": round(sqrt_weights[0], 6),
        "family_counts": dict(sorted(family_counts.items())),
        "label_source_counts": dict(sorted(source_counts.items())),
        "label_strength_counts": dict(sorted(strength_counts.items())),
    }


def select_policy_edges(data: Any, policy: str, *, torch: Any) -> PolicySelection:
    y = data.y
    target = (y == 0).long()  # binary: MERGE=1, other trainable negative=0
    base_train = data.edge_train_mask.bool()
    base_weight = data.edge_loss_weight.float().clone()
    families = [str(rec.get("candidate_family") or "UNKNOWN") for rec in data.edge_records]
    strengths = [str(rec.get("label_strength") or "UNKNOWN") for rec in data.edge_records]
    labels = [str(rec.get("label") or "UNKNOWN") for rec in data.edge_records]

    keep = base_train.clone()
    if policy == "current_weighted":
        pass
    elif policy == "strong_only":
        keep = torch.tensor(
            [k and (s in {"strong", "hard_negative"}) for k, s in zip(base_train.tolist(), strengths)],
            dtype=torch.bool,
        )
        base_weight = torch.where(keep, torch.ones_like(base_weight), torch.zeros_like(base_weight))
    elif policy == "body_list_focus":
        allowed_pos = {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION"}
        allowed_neg = {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION", "LAYOUT_SCOPE_MISMATCH"}
        keep = torch.tensor(
            [
                bool(k)
                and (
                    (lab == "MERGE" and fam in allowed_pos and s == "strong")
                    or (lab == "NONE" and fam in allowed_neg)
                )
                for k, fam, lab, s in zip(base_train.tolist(), families, labels, strengths)
            ],
            dtype=torch.bool,
        )
        base_weight = torch.where(keep, torch.ones_like(base_weight), torch.zeros_like(base_weight))
    elif policy == "body_list_float_skip_focus":
        allowed_pos = {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION", "FLOAT_SKIP_CONTINUATION"}
        allowed_neg = {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION", "FLOAT_SKIP_CONTINUATION", "LAYOUT_SCOPE_MISMATCH"}
        keep = torch.tensor(
            [
                bool(k)
                and (
                    (lab == "MERGE" and fam in allowed_pos and s == "strong")
                    or (lab == "NONE" and fam in allowed_neg)
                )
                for k, fam, lab, s in zip(base_train.tolist(), families, labels, strengths)
            ],
            dtype=torch.bool,
        )
        base_weight = torch.where(keep, torch.ones_like(base_weight), torch.zeros_like(base_weight))
    else:
        raise ValueError(f"unknown policy: {policy}")
    return PolicySelection(train_mask=keep, sample_weight=base_weight, target=target)


def inverse_class_weights(merge_count: int, none_count: int) -> dict[int, float]:
    total = max(1, merge_count + none_count)
    return {
        1: total / max(1, 2 * merge_count),
        0: total / max(1, 2 * none_count),
    }


def sqrt_inverse_class_weights(merge_count: int, none_count: int) -> dict[int, float]:
    inv = inverse_class_weights(merge_count, none_count)
    return {key: math.sqrt(value) for key, value in inv.items()}


def train_linear_probe(
    policy: str,
    train_paths: list[Path],
    val_paths: list[Path],
    *,
    torch: Any,
    device: Any,
    epochs: int,
    max_train_edges: int,
    max_val_edges: int,
    seed: int,
) -> dict[str, Any]:
    train_x, train_y, train_w = load_edge_matrix(train_paths, policy, torch=torch, max_edges=max_train_edges, seed=seed)
    val_x, val_y, val_w = load_edge_matrix(val_paths, policy, torch=torch, max_edges=max_val_edges, seed=seed + 1)
    if train_x.numel() == 0 or val_x.numel() == 0:
        return {"summary": {"status": "skipped_empty_matrix"}, "threshold_grid": []}
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
    grid = threshold_grid(probs, val_y_cpu, policy=policy)
    best = max(grid, key=lambda row: row["merge_f1"]) if grid else {}
    return {
        "summary": {
            "status": "ok",
            "train_edges": int(train_y.numel()),
            "val_edges": int(val_y_cpu.numel()),
            "train_merge_rate": round(float((train_y == 1).float().mean().item()), 6),
            "val_merge_rate": round(float((val_y_cpu == 1).float().mean().item()), 6),
            "pos_weight": round(float(pos_weight.item()), 6),
            "best_threshold": best.get("threshold"),
            "best_merge_f1": best.get("merge_f1"),
            "best_precision": best.get("merge_precision"),
            "best_recall": best.get("merge_recall"),
            "best_pred_merge_rate": best.get("pred_merge_rate"),
        },
        "threshold_grid": grid,
    }


def load_edge_matrix(paths: list[Path], policy: str, *, torch: Any, max_edges: int, seed: int) -> tuple[Any, Any, Any]:
    xs, ys, ws = [], [], []
    for path in paths:
        data = torch.load(path, weights_only=False, map_location="cpu")
        sel = select_policy_edges(data, policy, torch=torch)
        mask = sel.train_mask
        if int(mask.sum().item()) == 0:
            continue
        xs.append(data.edge_attr[mask])
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


def threshold_grid(probs: Any, target: Any, *, policy: str) -> list[dict[str, Any]]:
    rows = []
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        pred = (probs >= threshold).long()
        tp = int(((pred == 1) & (target == 1)).sum().item())
        fp = int(((pred == 1) & (target == 0)).sum().item())
        fn = int(((pred == 0) & (target == 1)).sum().item())
        tn = int(((pred == 0) & (target == 0)).sum().item())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        rows.append(
            {
                "policy": policy,
                "threshold": threshold,
                "merge_precision": round(precision, 6),
                "merge_recall": round(recall, 6),
                "merge_f1": round(f1, 6),
                "pred_merge_rate": round(float((pred == 1).float().mean().item()), 6),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return rows


def recommend_policy(policy_stats: list[dict[str, Any]], probe_summaries: dict[str, Any]) -> dict[str, Any]:
    by_policy = {row["policy"]: row for row in policy_stats}
    recommendations = []
    current = by_policy.get("current_weighted")
    if current and current.get("merge_rate", 0) > 0.6:
        recommendations.append("current_weighted has a high MERGE base rate; avoid unweighted training.")
    for policy, summary in probe_summaries.items():
        if summary.get("status") != "ok":
            continue
        pred_rate = summary.get("best_pred_merge_rate") or 0.0
        precision = summary.get("best_precision") or 0.0
        if pred_rate > 0.75:
            recommendations.append(f"{policy}: best threshold still predicts too many MERGE edges; tighten mask or threshold.")
        elif precision < 0.70:
            recommendations.append(f"{policy}: precision is weak; use as diagnostic only.")
    preferred = "body_list_focus"
    if (probe_summaries.get("body_list_float_skip_focus") or {}).get("best_precision", 0) >= (
        probe_summaries.get("body_list_focus") or {}
    ).get("best_precision", 0):
        preferred = "body_list_float_skip_focus"
    return {
        "preferred_next_diagnostic_policy": preferred,
        "notes": recommendations,
        "do_not_full_train_yet": True,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (dict, list))})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V8 Atomic MERGE Policy Diagnostic",
        "",
        "## Status",
        f"- graph_count: {summary['graph_count']}",
        f"- train_doc_count: {summary['train_doc_count']}",
        f"- val_doc_count: {summary['val_doc_count']}",
        f"- device: {summary['device']}",
        f"- epochs: {summary['epochs']}",
        "- full_gnn_training: No",
        "- e2e: No",
        "",
        "## Policy Stats",
        "| policy | trainable | merge | none | merge_rate | weighted_merge_share |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["policy_stats"]:
        lines.append(
            f"| {row['policy']} | {row['trainable_total']} | {row['merge_positive']} | {row['none_negative']} | "
            f"{row['merge_rate']} | {row['current_weighted_merge_share']} |"
        )
    lines.extend(["", "## Probe Summary"])
    lines.append("| policy | best_threshold | precision | recall | f1 | pred_merge_rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for policy, row in summary["probe_summaries"].items():
        if row.get("status") != "ok":
            lines.append(f"| {policy} | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| {policy} | {row.get('best_threshold')} | {row.get('best_precision')} | {row.get('best_recall')} | "
            f"{row.get('best_merge_f1')} | {row.get('best_pred_merge_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            f"- preferred_next_diagnostic_policy: {summary['recommendation']['preferred_next_diagnostic_policy']}",
            f"- do_not_full_train_yet: {summary['recommendation']['do_not_full_train_yet']}",
        ]
    )
    for note in summary["recommendation"].get("notes", []):
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
