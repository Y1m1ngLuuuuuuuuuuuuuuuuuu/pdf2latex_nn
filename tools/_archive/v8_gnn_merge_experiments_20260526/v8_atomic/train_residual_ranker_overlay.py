#!/usr/bin/env python3
"""Train and evaluate a residual v8 atomic MERGE ranker.

This script is intentionally branch-local.  It does not modify v8 JSON,
graph tensors, labels, checkpoints, or the production v8 deterministic merge
path.  It trains a structured-feature tree ranker for residual BODY/LIST
continuation candidates, projects high-confidence predictions back to v8
logical owners, and evaluates whether the overlay is worth keeping.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.mineru_v8_reflow import build_v8_from_middle, dump_json  # noqa: E402
from tools.v8_atomic.diagnose_shortcut_feature_ablation import (  # noqa: E402
    apply_ablation,
    build_feature_plan,
    feature_plan_for_json,
)
from tools.v8_atomic.diagnose_v8_atomic_training_policy import (  # noqa: E402
    _load_graph_paths,
    select_policy_edges,
)
from tools.v8_atomic.project_predictions_to_v8 import (  # noqa: E402
    aggregate_rows,
    build_projected_payload,
    load_manifest,
    load_selected_lookup,
    normalize_tex_for_hash,
    render_payload,
    resolve_source_paths,
    run_paragraph_preservation_audit,
    select_predicted_owner_merges,
    strip_diagnostics,
    write_csv,
)


ALLOWED_FAMILIES = {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION"}


@dataclass(frozen=True)
class EvalConfig:
    name: str
    body_threshold: float
    list_threshold: float
    cap: int


DEFAULT_EVAL_CONFIGS = (
    EvalConfig("hp_cap1_b099_l097", 0.99, 0.97, 1),
    EvalConfig("hp_cap2_b098_l096", 0.98, 0.96, 2),
    EvalConfig("balanced_cap3_b097_l095", 0.97, 0.95, 3),
    EvalConfig("recall_cap5_b095_l093", 0.95, 0.93, 5),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--policy", default="body_list_focus")
    parser.add_argument("--variant", default="B_no_owner_no_family_scope")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--max-train-edges", type=int, default=250000)
    parser.add_argument("--model", choices=("auto", "lightgbm", "hist_gbdt"), default="auto")
    parser.add_argument("--max-iter", type=int, default=260)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--overlay-policy", default="strict_continuation_v2_conservative")
    parser.add_argument("--max-overlay-reading-gap", type=int, default=4)
    parser.add_argument("--configs", nargs="*", help="name:body_threshold:list_threshold:cap")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--doc-ids", nargs="*")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-paragraph-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-engine", default="auto")
    parser.add_argument("--compile-timeout", type=int, default=180)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("This script requires torch to load v8 atomic graph files") from exc

    graph_paths = _load_graph_paths(args.manifest)
    if not graph_paths:
        raise SystemExit("manifest contains no graph paths")
    graph_paths = sorted(graph_paths, key=lambda path: str(path))
    rng = random.Random(args.seed)
    shuffled = list(graph_paths)
    rng.shuffle(shuffled)
    train_n = max(1, int(round(len(shuffled) * args.train_ratio)))
    train_paths = shuffled[:train_n]
    val_paths = shuffled[train_n:] or shuffled[-1:]

    feature_plan = build_feature_plan(graph_paths[0], torch=torch)
    train_matrix = collect_edges(
        train_paths,
        policy=args.policy,
        variant=args.variant,
        feature_plan=feature_plan,
        torch=torch,
        max_edges=args.max_train_edges,
        seed=args.seed,
    )
    val_matrix = collect_edges(
        val_paths,
        policy=args.policy,
        variant=args.variant,
        feature_plan=feature_plan,
        torch=torch,
        max_edges=None,
        seed=args.seed,
    )
    if train_matrix["x"].shape[0] == 0 or len(set(train_matrix["y"].tolist())) < 2:
        raise SystemExit("selected training edges are empty or single-class")

    model_name, ranker = fit_ranker(
        train_matrix,
        requested=args.model,
        seed=args.seed,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
    )
    val_summary = evaluate_edge_thresholds(ranker, val_matrix)

    manifest = load_manifest(args.manifest)
    selected_lookup = load_selected_lookup(manifest)
    items = selected_items(manifest, offset=args.offset, limit=args.limit, doc_ids=args.doc_ids)
    configs = parse_eval_configs(args.configs) if args.configs else list(DEFAULT_EVAL_CONFIGS)

    run_summaries: list[dict[str, Any]] = []
    for config in configs:
        config_dir = args.output_dir / config.name
        rows, errors = project_config(
            config=config,
            items=items,
            selected_lookup=selected_lookup,
            ranker=ranker,
            torch=torch,
            feature_plan=feature_plan,
            variant=args.variant,
            overlay_policy=args.overlay_policy,
            max_overlay_reading_gap=args.max_overlay_reading_gap,
            output_dir=config_dir,
            skip_existing=args.skip_existing,
            render=args.render,
            run_paragraph_audit=args.run_paragraph_audit,
            compile_pdf=args.compile,
            compile_engine=args.compile_engine,
            compile_timeout=args.compile_timeout,
        )
        summary = {
            "name": config.name,
            "body_threshold": config.body_threshold,
            "list_threshold": config.list_threshold,
            "cap": config.cap,
            "doc_count": len(rows),
            "error_count": len(errors),
            "aggregate": aggregate_rows(rows),
            "rows": rows,
            "errors": errors,
        }
        add_utility_fields(summary)
        dump_json(config_dir / "summary.json", summary)
        write_csv(config_dir / "summary.csv", rows)
        run_summaries.append(config_summary_row(summary))

    final = {
        "schema_version": "v8_residual_ranker_overlay_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "policy": args.policy,
        "variant": args.variant,
        "model_backend": model_name,
        "graph_count": len(graph_paths),
        "train_doc_count": len(train_paths),
        "val_doc_count": len(val_paths),
        "train_edges": int(train_matrix["x"].shape[0]),
        "train_positive_rate": round(float(train_matrix["y"].mean()), 6),
        "val_edges": int(val_matrix["x"].shape[0]),
        "val_positive_rate": round(float(val_matrix["y"].mean()), 6) if val_matrix["x"].shape[0] else None,
        "feature_plan": feature_plan_for_json(feature_plan),
        "edge_threshold_summary": val_summary,
        "runs": run_summaries,
        "decision": decide(run_summaries),
    }
    dump_json(args.output_dir / "summary.json", final)
    write_csv(args.output_dir / "residual_ranker_overlay_runs.csv", run_summaries)
    write_report(args.output_dir / "RESIDUAL_RANKER_OVERLAY_REPORT.md", final)
    return 0


def collect_edges(
    paths: list[Path],
    *,
    policy: str,
    variant: str,
    feature_plan: dict[str, Any],
    torch: Any,
    max_edges: int | None,
    seed: int,
) -> dict[str, Any]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    families: list[str] = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        selection = select_policy_edges(data, policy, torch=torch)
        mask = selection.train_mask.bool()
        if int(mask.sum().item()) == 0:
            continue
        edge_attr = data.edge_attr.clone()
        if variant != "all_features":
            apply_ablation(edge_attr, ablation=variant, feature_plan=feature_plan)
        xs.append(edge_attr[mask].cpu().numpy())
        ys.append(selection.target[mask].cpu().numpy().astype(np.int64))
        for rec, keep in zip(data.edge_records, mask.tolist(), strict=False):
            if keep:
                families.append(str(rec.get("candidate_family") or "UNKNOWN"))
    if not xs:
        return {"x": np.empty((0, 0), dtype=np.float32), "y": np.empty((0,), dtype=np.int64), "families": []}
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_edges and x.shape[0] > max_edges:
        rng = np.random.default_rng(seed)
        indices = rng.choice(np.arange(x.shape[0]), size=max_edges, replace=False)
        x = x[indices]
        y = y[indices]
        families = [families[int(index)] for index in indices]
    return {"x": x, "y": y, "families": families}


def fit_ranker(
    train: dict[str, Any],
    *,
    requested: str,
    seed: int,
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    l2_regularization: float,
) -> tuple[str, Any]:
    x = train["x"]
    y = train["y"]
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    sample_weight = np.where(y == 1, neg / pos, 1.0)
    if requested in {"auto", "lightgbm"}:
        try:
            from lightgbm import LGBMClassifier  # type: ignore

            model = LGBMClassifier(
                n_estimators=max_iter,
                learning_rate=learning_rate,
                num_leaves=max_leaf_nodes,
                random_state=seed,
                class_weight={0: 1.0, 1: neg / pos},
                objective="binary",
                verbosity=-1,
            )
            model.fit(x, y)
            return "lightgbm", model
        except ModuleNotFoundError:
            if requested == "lightgbm":
                raise SystemExit("lightgbm is not installed in this environment")
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return "sklearn_hist_gradient_boosting_fallback", model


def predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-scores))


def evaluate_edge_thresholds(model: Any, matrix: dict[str, Any]) -> dict[str, Any]:
    if matrix["x"].shape[0] == 0:
        return {"status": "empty"}
    probs = predict_proba(model, matrix["x"])
    y = matrix["y"]
    rows = []
    for threshold in [0.5, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]:
        pred = probs >= threshold
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        rows.append(
            {
                "threshold": threshold,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "pred_rate": round(float(pred.mean()), 6),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return {
        "status": "ok",
        "edge_count": int(y.shape[0]),
        "positive_rate": round(float(y.mean()), 6),
        "grid": rows,
    }


def selected_items(manifest: dict[str, Any], *, offset: int, limit: int | None, doc_ids: list[str] | None) -> list[dict[str, Any]]:
    items = list(manifest.get("items") or [])
    items = sorted(items, key=lambda item: str(item.get("doc_id") or ""))
    if doc_ids:
        wanted = set(doc_ids)
        items = [item for item in items if str(item.get("doc_id")) in wanted]
    end = offset + limit if limit is not None else None
    return items[offset:end]


def parse_eval_configs(values: list[str] | None) -> list[EvalConfig]:
    configs: list[EvalConfig] = []
    for value in values or []:
        name, body, list_threshold, cap = value.split(":")
        configs.append(EvalConfig(name, float(body), float(list_threshold), int(cap)))
    return configs


def project_config(
    *,
    config: EvalConfig,
    items: list[dict[str, Any]],
    selected_lookup: dict[str, dict[str, Any]],
    ranker: Any,
    torch: Any,
    feature_plan: dict[str, Any],
    variant: str,
    overlay_policy: str,
    max_overlay_reading_gap: int,
    output_dir: Path,
    skip_existing: bool,
    render: bool,
    run_paragraph_audit: bool,
    compile_pdf: bool,
    compile_engine: str,
    compile_timeout: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        doc_id = str(item.get("doc_id") or "")
        doc_dir = output_dir / doc_id
        if skip_existing and (doc_dir / "projection_report.json").exists():
            rows.append(json.loads((doc_dir / "projection_report.json").read_text(encoding="utf-8"))["summary"])
            print(f"[{config.name} {index}/{len(items)}] skip existing {doc_id}")
            continue
        try:
            row = process_doc_with_ranker(
                item,
                selected_lookup=selected_lookup,
                ranker=ranker,
                torch=torch,
                feature_plan=feature_plan,
                variant=variant,
                config=config,
                overlay_policy=overlay_policy,
                max_overlay_reading_gap=max_overlay_reading_gap,
                output_dir=doc_dir,
                render=render,
                run_paragraph_audit=run_paragraph_audit,
                compile_pdf=compile_pdf,
                compile_engine=compile_engine,
                compile_timeout=compile_timeout,
            )
            rows.append(row)
            print(
                f"[{config.name} {index}/{len(items)}] {doc_id} added={row['model_added_owner_merge_count']} "
                f"changed={row['generated_tex_changed']} det_missing={row.get('deterministic_missing_merge_rate')} "
                f"ranker_missing={row.get('learned_missing_merge_rate')}"
            )
        except Exception as exc:
            error = {"doc_id": doc_id, "error": type(exc).__name__, "message": str(exc)}
            errors.append(error)
            print(f"[{config.name} {index}/{len(items)}] ERROR {doc_id}: {type(exc).__name__}: {exc}")
    return rows, errors


def process_doc_with_ranker(
    item: dict[str, Any],
    *,
    selected_lookup: dict[str, dict[str, Any]],
    ranker: Any,
    torch: Any,
    feature_plan: dict[str, Any],
    variant: str,
    config: EvalConfig,
    overlay_policy: str,
    max_overlay_reading_gap: int,
    output_dir: Path,
    render: bool,
    run_paragraph_audit: bool,
    compile_pdf: bool,
    compile_engine: str,
    compile_timeout: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_id = str(item["doc_id"])
    selected = selected_lookup.get(doc_id, {})
    graph_path = Path(item["graph_path"])
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph_payload = json.loads(Path(data.source_graph_view).read_text(encoding="utf-8"))
    source_paths = resolve_source_paths(item, selected, graph_payload)

    edge_attr = data.edge_attr.clone()
    if variant != "all_features":
        apply_ablation(edge_attr, ablation=variant, feature_plan=feature_plan)
    probs = predict_proba(ranker, edge_attr.cpu().numpy()).tolist()
    predicted = select_predicted_owner_merges(
        graph_payload,
        data.edge_records,
        probs,
        threshold=min(config.body_threshold, config.list_threshold),
        family_thresholds={
            "BODY_TEXT_CONTINUATION": config.body_threshold,
            "LIST_CONTINUATION": config.list_threshold,
        },
        allowed_families=ALLOWED_FAMILIES,
        overlay_policy=overlay_policy,
        max_overlay_reading_gap=max_overlay_reading_gap,
        max_added_owner_merges_per_doc=config.cap,
    )

    deterministic_payload = build_v8_from_middle(
        doc_id=doc_id,
        middle_json_path=source_paths["middle_json"],
        content_list_json_path=source_paths.get("content_list_json"),
        style_content_list_json_path=source_paths.get("style_content_list_json"),
        middle_block_source=str(graph_payload.get("source", {}).get("middle_block_source") or "preproc_blocks"),
    )
    learned_payload = build_projected_payload(
        doc_id=doc_id,
        graph_payload=graph_payload,
        source_paths=source_paths,
        predicted=predicted,
        deterministic_payload=deterministic_payload,
    )
    dump_json(output_dir / f"{doc_id}_content_list_v8_deterministic.json", strip_diagnostics(deterministic_payload))
    dump_json(output_dir / f"{doc_id}_content_list_v8_ranker_plus_deterministic.json", strip_diagnostics(learned_payload))

    projection_report = {
        "schema_version": "v8_residual_ranker_doc_projection_v1",
        "doc_id": doc_id,
        "graph_path": str(graph_path),
        "source_paths": {key: str(value) for key, value in source_paths.items() if value},
        "body_threshold": config.body_threshold,
        "list_threshold": config.list_threshold,
        "cap": config.cap,
        "overlay_policy": overlay_policy,
        "predicted_atomic_merge_count": len(predicted["predicted_atomic_edges"]),
        "predicted_cross_owner_atomic_merge_count": len(predicted["cross_owner_atomic_edges"]),
        "model_added_owner_merges": predicted["accepted_owner_merges"],
        "rejection_reason_counts": predicted["rejection_reason_counts"],
        "strict_candidate_count": predicted["strict_candidate_count"],
        "deterministic_merge_count": len(deterministic_payload.get("merge_decisions") or []),
        "learned_plus_deterministic_merge_count": len(learned_payload.get("merge_decisions") or []),
    }
    dump_json(output_dir / "prediction_projection_sidecar.json", projection_report)

    render_summaries: dict[str, Any] = {}
    if render:
        render_summaries["deterministic"] = render_payload(
            doc_id,
            deterministic_payload,
            source_paths,
            output_dir / "deterministic",
            compile_pdf=compile_pdf,
            compile_engine=compile_engine,
            compile_timeout=compile_timeout,
        )
        render_summaries["learned_plus_deterministic"] = render_payload(
            doc_id,
            learned_payload,
            source_paths,
            output_dir / "ranker_plus_deterministic",
            compile_pdf=compile_pdf,
            compile_engine=compile_engine,
            compile_timeout=compile_timeout,
        )
        if run_paragraph_audit and source_paths.get("source_tex"):
            for mode, tex_dir in (
                ("deterministic", output_dir / "deterministic"),
                ("learned_plus_deterministic", output_dir / "ranker_plus_deterministic"),
            ):
                audit_dir = tex_dir / "paragraph_audit"
                run_paragraph_preservation_audit(
                    source_tex=source_paths["source_tex"],
                    generated_tex=tex_dir / "generated.tex",
                    doc_id=doc_id,
                    output_dir=audit_dir,
                )
                audit = json.loads((audit_dir / "paragraph_preservation_against_tex.json").read_text(encoding="utf-8"))
                render_summaries[mode]["paragraph_audit_summary"] = audit.get("summary", {})

    generated_tex_changed = False
    if render:
        det_tex = (output_dir / "deterministic" / "generated.tex").read_text(encoding="utf-8", errors="ignore")
        ranker_tex = (output_dir / "ranker_plus_deterministic" / "generated.tex").read_text(encoding="utf-8", errors="ignore")
        generated_tex_changed = normalize_tex_for_hash(det_tex) != normalize_tex_for_hash(ranker_tex)

    summary = summarize_doc(
        doc_id=doc_id,
        projection_report=projection_report,
        render_summaries=render_summaries,
        generated_tex_changed=generated_tex_changed,
    )
    dump_json(output_dir / "projection_report.json", {"projection": projection_report, "render": render_summaries, "summary": summary})
    return summary


def resolve_source_paths(item: dict[str, Any], selected: dict[str, Any], graph_payload: dict[str, Any]) -> dict[str, Path]:
    from tools.v8_atomic.project_predictions_to_v8 import resolve_source_paths as _resolve

    return _resolve(item, selected, graph_payload)


def summarize_doc(
    *,
    doc_id: str,
    projection_report: dict[str, Any],
    render_summaries: dict[str, Any],
    generated_tex_changed: bool,
) -> dict[str, Any]:
    from tools.v8_atomic.project_predictions_to_v8 import summarize_doc as _summarize

    return _summarize(
        doc_id=doc_id,
        projection_report=projection_report,
        render_summaries=render_summaries,
        generated_tex_changed=generated_tex_changed,
    )


def add_utility_fields(summary: dict[str, Any]) -> None:
    agg = summary.get("aggregate") or {}
    det_missing = agg.get("mean_deterministic_missing_merge_rate")
    learned_missing = agg.get("mean_learned_missing_merge_rate")
    det_wrong = agg.get("mean_deterministic_wrong_merge_rate")
    learned_wrong = agg.get("mean_learned_wrong_merge_rate")
    det_cov = agg.get("mean_deterministic_body_source_coverage_rate")
    learned_cov = agg.get("mean_learned_body_source_coverage_rate")
    if det_cov is None or learned_cov is None:
        det_cov = agg.get("mean_deterministic_source_coverage_rate")
        learned_cov = agg.get("mean_learned_source_coverage_rate")
    if None in {det_missing, learned_missing, det_wrong, learned_wrong}:
        return
    missing_reduction = float(det_missing) - float(learned_missing)
    wrong_increase = float(learned_wrong) - float(det_wrong)
    agg["missing_reduction"] = round(missing_reduction, 6)
    agg["wrong_increase"] = round(wrong_increase, 6)
    for lam in (3, 5, 10):
        agg[f"utility_{lam}"] = round(missing_reduction - lam * wrong_increase, 6)
    agg["wrong_merge_constraint_pass"] = bool(float(learned_wrong) <= float(det_wrong) + 1e-12)
    if det_cov is not None and learned_cov is not None:
        agg["source_coverage_constraint_pass"] = bool(float(learned_cov) >= float(det_cov) - 1e-12)


def config_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    add_utility_fields(summary)
    agg = summary.get("aggregate") or {}
    return {
        "name": summary["name"],
        "body_threshold": summary["body_threshold"],
        "list_threshold": summary["list_threshold"],
        "cap": summary["cap"],
        "doc_count": summary["doc_count"],
        "error_count": summary["error_count"],
        "model_added_owner_merge_total": agg.get("model_added_owner_merge_total"),
        "generated_tex_changed_count": agg.get("generated_tex_changed_count"),
        "det_missing": agg.get("mean_deterministic_missing_merge_rate"),
        "learned_missing": agg.get("mean_learned_missing_merge_rate"),
        "missing_reduction": agg.get("missing_reduction"),
        "det_wrong": agg.get("mean_deterministic_wrong_merge_rate"),
        "learned_wrong": agg.get("mean_learned_wrong_merge_rate"),
        "wrong_increase": agg.get("wrong_increase"),
        "det_source_coverage": agg.get("mean_deterministic_source_coverage_rate"),
        "learned_source_coverage": agg.get("mean_learned_source_coverage_rate"),
        "det_body_source_coverage": agg.get("mean_deterministic_body_source_coverage_rate"),
        "learned_body_source_coverage": agg.get("mean_learned_body_source_coverage_rate"),
        "det_paragraph_delta": agg.get("mean_deterministic_paragraph_count_delta"),
        "learned_paragraph_delta": agg.get("mean_learned_paragraph_count_delta"),
        "utility_3": agg.get("utility_3"),
        "utility_5": agg.get("utility_5"),
        "utility_10": agg.get("utility_10"),
        "wrong_merge_constraint_pass": agg.get("wrong_merge_constraint_pass"),
        "source_coverage_constraint_pass": agg.get("source_coverage_constraint_pass"),
    }


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [
        row
        for row in rows
        if row.get("wrong_merge_constraint_pass")
        and row.get("source_coverage_constraint_pass")
        and (row.get("missing_reduction") or 0) > 0
        and (row.get("utility_5") or -999) > 0
    ]
    if not viable:
        return {
            "status": "not_ready_for_production",
            "reason": "No threshold/cap configuration satisfied wrong_merge <= deterministic, source_coverage >= deterministic, missing reduction > 0, and utility_5 > 0.",
        }
    best = max(viable, key=lambda row: (row.get("utility_5") or -999, row.get("missing_reduction") or 0))
    return {
        "status": "candidate_overlay_found",
        "best_config": best["name"],
        "reason": "At least one configuration passed the conservative production constraints.",
    }


def write_report(path: Path, final: dict[str, Any]) -> None:
    lines = [
        "# Residual Ranker Overlay Report",
        "",
        "## Status",
        f"- model_backend: {final['model_backend']}",
        f"- policy: {final['policy']}",
        f"- variant: {final['variant']}",
        f"- train_edges: {final['train_edges']}",
        f"- val_edges: {final['val_edges']}",
        "- production v8 deterministic path changed: No",
        "- training project GNN: No",
        "- MinerU rerun: No",
        "",
        "## Edge-Level Threshold Probe",
    ]
    for row in (final.get("edge_threshold_summary") or {}).get("grid", []):
        lines.append(
            f"- threshold={row['threshold']}: P={row['precision']} R={row['recall']} "
            f"F1={row['f1']} pred_rate={row['pred_rate']}"
        )
    lines.extend(
        [
            "",
            "## Projection Results",
            "| config | body tau | list tau | cap | added | changed | missing reduction | wrong increase | utility_3 | utility_5 | utility_10 | wrong<=det | source>=det |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in final["runs"]:
        lines.append(
            f"| {row['name']} | {row['body_threshold']} | {row['list_threshold']} | {row['cap']} | "
            f"{row['model_added_owner_merge_total']} | {row['generated_tex_changed_count']} | "
            f"{row['missing_reduction']} | {row['wrong_increase']} | {row['utility_3']} | "
            f"{row['utility_5']} | {row['utility_10']} | {row['wrong_merge_constraint_pass']} | "
            f"{row['source_coverage_constraint_pass']} |"
        )
    decision = final["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            f"- status: {decision['status']}",
            f"- reason: {decision['reason']}",
            "",
            "A learned residual overlay should only continue if it satisfies the conservative constraint: "
            "wrong_merge_rate must not exceed deterministic v8, source coverage must not drop, and utility_5 must be positive.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
