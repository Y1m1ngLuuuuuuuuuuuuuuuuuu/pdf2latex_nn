#!/usr/bin/env python3
"""Shortcut-feature ablations for v8 atomic MERGE edge probes.

This diagnostic answers a narrow question: after removing increasingly direct
rule/owner hints, how much MERGE signal remains in visual/text/style features?

It does not train the project GNN, rebuild graphs, relabel data, or run E2E.
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


OWNER_FEATURES = {
    "same_middle_block",
    "same_content_owner",
    "same_style_content_owner",
}

FAMILY_FEATURES_PREFIX = "family:"
LAYOUT_SCOPE_FEATURES_PREFIX = "layout_scope:"

STRICT_KEEP_FEATURES = {
    # Basic page/column geometry.
    "page_delta_norm",
    "same_page",
    "same_column",
    "column_transition:same_column_down",
    "column_transition:left_to_right_column",
    "column_transition:right_to_next_page_left",
    "column_transition:cross_page_same_column",
    "column_transition:cross_page_column_reset",
    "column_transition:other",
    # Visual spacing and overlap.
    "vertical_gap_norm",
    "x_overlap_ratio",
    "rel_dx0_norm",
    "rel_dx1_norm",
    "rel_dcx_norm",
    "rel_dy0_norm",
    "rel_dy1_norm",
    "rel_dcy_norm",
    "left_alignment_abs_delta_norm",
    "right_alignment_abs_delta_norm",
    "center_alignment_abs_delta_norm",
    "width_log_ratio",
    "height_log_ratio",
    "area_log_ratio",
    "y_overlap_ratio",
    "vertical_gap_by_line_height",
    "cross_page_bottom_to_top",
    "src_bottom_page_norm",
    "dst_top_page_norm",
    # Text boundary cues.
    "src_tail_open",
    "src_tail_hyphen",
    "dst_head_lowercase",
    "dst_head_parenthetical",
    "src_tail_hard_terminal",
    "src_tail_soft_punctuation",
    "src_tail_abbrev_like",
    "src_tail_citation_closed",
    "dst_tail_hard_terminal",
    "dst_tail_soft_punctuation",
    "src_unclosed_parenthesis",
    "src_unclosed_bracket",
    "src_unclosed_quote",
    "src_tail_after_math_symbol",
    "src_tail_last_token_stopword",
    "dst_head_first_token_stopword",
    "dst_starts_punctuation",
    "dst_starts_closing_bracket",
    "dst_head_conjunction",
    "dst_head_preposition",
    "dst_head_uppercase",
    "dst_head_continuation_word",
    # Column-flow cues that do not expose owner labels.
    "cross_page_bottom_to_top",
    "src_bottom_page_norm",
    "dst_top_page_norm",
    "src_near_column_bottom",
    "dst_near_column_top",
    "same_column_flow_lane",
    # Local paragraph rhythm.  Channel-count context and skipped-object counts
    # are intentionally excluded from this strict branch.
    "src_prev_gap_by_line_height",
    "src_next_gap_by_line_height",
    "dst_prev_gap_by_line_height",
    "dst_next_gap_by_line_height",
    "candidate_gap_vs_src_next_gap",
    "candidate_gap_vs_dst_prev_gap",
    # Style cues.
    "src_font_size_norm",
    "dst_font_size_norm",
    "font_size_delta_norm",
    "font_size_abs_delta_norm",
    "same_font_size_bucket",
    "src_bold_ratio",
    "dst_bold_ratio",
    "bold_ratio_delta",
    "bold_ratio_abs_delta",
    "same_bold_state",
}


@dataclass(frozen=True)
class Ablation:
    name: str
    description: str


ABLATIONS = (
    Ablation("A_no_owner", "Zero only same_middle/content/style owner indicators."),
    Ablation("B_no_owner_no_family_scope", "Zero owner indicators plus candidate family and layout scope one-hots."),
    Ablation(
        "C_strict_visual_text_style",
        "Keep only bbox/page-column geometry, text boundary cues, and font/bold style cues.",
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
    feature_plan = build_feature_plan(graph_paths[0], torch=torch)

    rows: list[dict[str, Any]] = []
    threshold_results: dict[str, list[dict[str, Any]]] = {}
    for policy in args.policies:
        for ablation in ABLATIONS:
            result = train_probe(
                policy=policy,
                ablation=ablation.name,
                train_paths=train_paths,
                val_paths=val_paths,
                torch=torch,
                device=device,
                feature_plan=feature_plan,
                epochs=args.epochs,
                max_train_edges=args.max_train_edges,
                max_val_edges=args.max_val_edges,
                seed=args.seed,
            )
            rows.append(result["summary"])
            key = f"{policy}::{ablation.name}"
            threshold_results[key] = result["threshold_grid"]
            write_csv(out_dir / f"threshold_grid_{policy}_{ablation.name}.csv", result["threshold_grid"])

    comparisons = compute_comparisons(rows)
    summary = {
        "schema_version": "v8_atomic_shortcut_feature_ablation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "graph_count": len(graph_paths),
        "train_doc_count": len(train_paths),
        "val_doc_count": len(val_paths),
        "device": str(device),
        "epochs": args.epochs,
        "policies": args.policies,
        "ablations": [ablation.__dict__ for ablation in ABLATIONS],
        "feature_plan": feature_plan_for_json(feature_plan),
        "rows": rows,
        "comparisons": comparisons,
        "interpretation": interpret(comparisons),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "threshold_results.json", threshold_results)
    write_csv(out_dir / "shortcut_feature_ablation_rows.csv", rows)
    write_csv(out_dir / "shortcut_feature_ablation_comparisons.csv", comparisons)
    write_report(out_dir / "SHORTCUT_FEATURE_ABLATION_REPORT.md", summary)


def build_feature_plan(path: Path, *, torch: Any) -> dict[str, Any]:
    data = torch.load(path, weights_only=False, map_location="cpu")
    schema = list(getattr(data, "edge_attr_schema", []))
    if not schema:
        raise SystemExit("graph is missing edge_attr_schema")
    schema_set = set(schema)
    missing_owner = sorted(OWNER_FEATURES - schema_set)
    missing_strict = sorted(STRICT_KEEP_FEATURES - schema_set)
    if missing_owner:
        raise SystemExit(f"edge_attr_schema missing owner features: {missing_owner}")
    if missing_strict:
        raise SystemExit(f"edge_attr_schema missing strict keep features: {missing_strict}")

    owner_indices = [schema.index(name) for name in sorted(OWNER_FEATURES)]
    family_indices = [idx for idx, name in enumerate(schema) if name.startswith(FAMILY_FEATURES_PREFIX)]
    layout_scope_indices = [idx for idx, name in enumerate(schema) if name.startswith(LAYOUT_SCOPE_FEATURES_PREFIX)]
    strict_keep_indices = [schema.index(name) for name in sorted(STRICT_KEEP_FEATURES)]
    all_indices = set(range(len(schema)))
    strict_zero_indices = sorted(all_indices - set(strict_keep_indices))
    return {
        "schema": schema,
        "owner_indices": owner_indices,
        "family_indices": family_indices,
        "layout_scope_indices": layout_scope_indices,
        "strict_keep_indices": strict_keep_indices,
        "strict_zero_indices": strict_zero_indices,
    }


def feature_plan_for_json(plan: dict[str, Any]) -> dict[str, Any]:
    schema = plan["schema"]
    return {
        "owner_features": [schema[idx] for idx in plan["owner_indices"]],
        "family_features": [schema[idx] for idx in plan["family_indices"]],
        "layout_scope_features": [schema[idx] for idx in plan["layout_scope_indices"]],
        "strict_keep_features": [schema[idx] for idx in plan["strict_keep_indices"]],
        "strict_zeroed_features": [schema[idx] for idx in plan["strict_zero_indices"]],
        "edge_attr_dim": len(schema),
    }


def train_probe(
    *,
    policy: str,
    ablation: str,
    train_paths: list[Path],
    val_paths: list[Path],
    torch: Any,
    device: Any,
    feature_plan: dict[str, Any],
    epochs: int,
    max_train_edges: int,
    max_val_edges: int,
    seed: int,
) -> dict[str, Any]:
    train_x, train_y, train_w = load_edge_matrix(
        train_paths,
        policy,
        ablation=ablation,
        torch=torch,
        max_edges=max_train_edges,
        seed=seed,
        feature_plan=feature_plan,
    )
    val_x, val_y, _ = load_edge_matrix(
        val_paths,
        policy,
        ablation=ablation,
        torch=torch,
        max_edges=max_val_edges,
        seed=seed + 1,
        feature_plan=feature_plan,
    )
    if train_x.numel() == 0 or val_x.numel() == 0:
        return {
            "summary": {
                "policy": policy,
                "ablation": ablation,
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
    grid = threshold_grid(probs, val_y_cpu, policy=f"{policy}_{ablation}")
    best = max(grid, key=lambda row: row["merge_f1"]) if grid else {}
    summary = {
        "policy": policy,
        "ablation": ablation,
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


def load_edge_matrix(
    paths: list[Path],
    policy: str,
    *,
    ablation: str,
    torch: Any,
    max_edges: int,
    seed: int,
    feature_plan: dict[str, Any],
) -> tuple[Any, Any, Any]:
    xs, ys, ws = [], [], []
    for path in paths:
        data = torch.load(path, weights_only=False, map_location="cpu")
        sel = select_policy_edges(data, policy, torch=torch)
        mask = sel.train_mask
        if int(mask.sum().item()) == 0:
            continue
        x = data.edge_attr[mask].clone()
        apply_ablation(x, ablation=ablation, feature_plan=feature_plan)
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


def apply_ablation(x: Any, *, ablation: str, feature_plan: dict[str, Any]) -> None:
    if ablation == "A_no_owner":
        x[:, feature_plan["owner_indices"]] = 0.0
        return
    if ablation == "B_no_owner_no_family_scope":
        zero = feature_plan["owner_indices"] + feature_plan["family_indices"] + feature_plan["layout_scope_indices"]
        x[:, zero] = 0.0
        return
    if ablation == "C_strict_visual_text_style":
        x[:, feature_plan["strict_zero_indices"]] = 0.0
        return
    raise ValueError(f"unknown ablation: {ablation}")


def compute_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row.get("policy"), row.get("ablation")): row for row in rows if row.get("status") == "ok"}
    policies = sorted({row.get("policy") for row in rows if row.get("status") == "ok"})
    comparisons: list[dict[str, Any]] = []
    for policy in policies:
        a = by_key.get((policy, "A_no_owner"))
        b = by_key.get((policy, "B_no_owner_no_family_scope"))
        c = by_key.get((policy, "C_strict_visual_text_style"))
        if not a:
            continue
        comparisons.append(
            {
                "policy": policy,
                "A_no_owner_f1": _get(a, "best_merge_f1"),
                "B_no_owner_no_family_scope_f1": _get(b, "best_merge_f1"),
                "C_strict_visual_text_style_f1": _get(c, "best_merge_f1"),
                "B_minus_A_f1": _delta(b, a, "best_merge_f1"),
                "C_minus_A_f1": _delta(c, a, "best_merge_f1"),
                "C_minus_B_f1": _delta(c, b, "best_merge_f1"),
                "A_precision": _get(a, "best_precision"),
                "B_precision": _get(b, "best_precision"),
                "C_precision": _get(c, "best_precision"),
                "A_recall": _get(a, "best_recall"),
                "B_recall": _get(b, "best_recall"),
                "C_recall": _get(c, "best_recall"),
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
        b_minus_a = row.get("B_minus_A_f1")
        c_minus_a = row.get("C_minus_A_f1")
        c_f1 = row.get("C_strict_visual_text_style_f1")
        if b_minus_a is not None:
            if b_minus_a <= -0.05:
                notes.append(f"{policy}: family/layout-scope features are a major shortcut beyond owner.")
            elif b_minus_a <= -0.02:
                notes.append(f"{policy}: family/layout-scope features provide meaningful prior signal.")
            else:
                notes.append(f"{policy}: removing family/layout scope changes little after owner removal.")
        if c_minus_a is not None:
            if c_minus_a <= -0.08:
                notes.append(f"{policy}: strict visual/text/style is much weaker; current no-owner was still using rule priors.")
            elif c_minus_a <= -0.03:
                notes.append(f"{policy}: strict visual/text/style loses some signal but remains useful.")
            else:
                notes.append(f"{policy}: strict visual/text/style is close to no-owner; shortcuts are not dominating.")
        if c_f1 is not None:
            if c_f1 >= 0.80:
                notes.append(f"{policy}: strict feature set still has strong learnable continuation signal.")
            elif c_f1 >= 0.70:
                notes.append(f"{policy}: strict feature set has moderate signal; useful but likely needs GNN context.")
            else:
                notes.append(f"{policy}: strict feature set is weak; owner/rules dominate this supervision.")
    return notes


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V8 Atomic Shortcut Feature Ablation",
        "",
        "## Status",
        f"- graph_count: {summary['graph_count']}",
        f"- train_doc_count: {summary['train_doc_count']}",
        f"- val_doc_count: {summary['val_doc_count']}",
        "- full_gnn_training: No",
        "- e2e: No",
        "",
        "## Ablations",
    ]
    for ablation in summary["ablations"]:
        lines.append(f"- {ablation['name']}: {ablation['description']}")
    lines.extend(
        [
            "",
            "## Results",
            "| policy | ablation | best_threshold | precision | recall | F1 | pred_merge_rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| {row.get('policy')} | {row.get('ablation')} | {row.get('best_threshold')} | "
            f"{row.get('best_precision')} | {row.get('best_recall')} | {row.get('best_merge_f1')} | "
            f"{row.get('best_pred_merge_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "| policy | A no-owner | B no-owner-no-family-scope | C strict | B-A | C-A | C-B |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['policy']} | {row.get('A_no_owner_f1')} | {row.get('B_no_owner_no_family_scope_f1')} | "
            f"{row.get('C_strict_visual_text_style_f1')} | {row.get('B_minus_A_f1')} | "
            f"{row.get('C_minus_A_f1')} | {row.get('C_minus_B_f1')} |"
        )
    lines.extend(["", "## Interpretation"])
    for note in summary["interpretation"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
