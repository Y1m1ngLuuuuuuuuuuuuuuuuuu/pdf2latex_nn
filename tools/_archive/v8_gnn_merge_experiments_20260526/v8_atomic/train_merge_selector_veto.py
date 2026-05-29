#!/usr/bin/env python3
"""Train and evaluate a v8 atomic MERGE selector/veto branch.

This is a branch-local experiment for the v8 atomic MERGE route.  Unlike the
residual overlay probe, this selector is allowed to both:

* keep or veto deterministic v8 merge decisions when a scored candidate exists;
* add a very small number of high-confidence residual BODY/LIST continuations.

It does not modify v8 JSON sources, graph tensors, labels, checkpoints, or the
production deterministic v8 reconstruction path.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.mineru_v8_reflow import (  # noqa: E402
    MergeDecision,
    build_diagnostics,
    build_v8_from_middle,
    dump_json,
    materialize_v8_items,
)
from src.perception.v8_atomic_merge import load_and_order_v8_blocks  # noqa: E402
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
    load_manifest,
    load_selected_lookup,
    normalize_tex_for_hash,
    render_payload,
    resolve_source_paths,
    run_paragraph_preservation_audit,
    strict_continuation_overlay_gate,
    strip_diagnostics,
)
from tools.v8_atomic.train_residual_ranker_overlay import (  # noqa: E402
    evaluate_edge_thresholds,
    fit_ranker,
    predict_proba,
)


RESIDUAL_FAMILIES = {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION"}
DEFAULT_CONFIGS = (
    "keep_relaxed_add_strict:0.45:0.99:0.97:1",
    "keep_mid_add_strict:0.60:0.99:0.97:1",
    "keep_strict_add_strict:0.75:0.99:0.97:1",
    "keep_mid_add_balanced:0.60:0.985:0.965:2",
    "keep_high_add_balanced:0.70:0.985:0.965:2",
    "keep_mid_add_recall:0.60:0.975:0.955:3",
)


@dataclass(frozen=True)
class SelectorConfig:
    name: str
    deterministic_threshold: float
    body_threshold: float
    list_threshold: float
    residual_cap: int


@dataclass
class Candidate:
    src_block_id: str
    dst_block_id: str
    probability: float
    source: str
    family: str
    src_reading_order: int
    dst_reading_order: int
    edge_id: str | None = None
    src_atomic_id: str | None = None
    dst_atomic_id: str | None = None
    src_text: str | None = None
    dst_text: str | None = None
    reason: str | None = None
    evidence: dict[str, Any] | None = None


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
    parser.add_argument("--component-size-cap", type=int, default=12)
    parser.add_argument("--configs", nargs="*", default=list(DEFAULT_CONFIGS))
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

    model_name, selector = fit_ranker(
        train_matrix,
        requested=args.model,
        seed=args.seed,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
    )
    val_summary = evaluate_edge_thresholds(selector, val_matrix)

    manifest = load_manifest(args.manifest)
    selected_lookup = load_selected_lookup(manifest)
    items = selected_items(manifest, offset=args.offset, limit=args.limit, doc_ids=args.doc_ids)
    configs = parse_configs(args.configs)

    run_rows: list[dict[str, Any]] = []
    for config in configs:
        config_dir = args.output_dir / config.name
        rows, errors = project_config(
            config=config,
            items=items,
            selected_lookup=selected_lookup,
            selector=selector,
            torch=torch,
            feature_plan=feature_plan,
            variant=args.variant,
            overlay_policy=args.overlay_policy,
            max_overlay_reading_gap=args.max_overlay_reading_gap,
            component_size_cap=args.component_size_cap,
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
            "deterministic_threshold": config.deterministic_threshold,
            "body_threshold": config.body_threshold,
            "list_threshold": config.list_threshold,
            "residual_cap": config.residual_cap,
            "doc_count": len(rows),
            "error_count": len(errors),
            "aggregate": aggregate_selector_rows(rows),
            "rows": rows,
            "errors": errors,
        }
        add_utility_fields(summary)
        dump_json(config_dir / "summary.json", summary)
        write_csv(config_dir / "summary.csv", rows)
        run_rows.append(config_summary_row(summary))

    final = {
        "schema_version": "v8_merge_selector_veto_v1",
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
        "runs": run_rows,
        "decision": decide(run_rows),
    }
    dump_json(args.output_dir / "summary.json", final)
    write_csv(args.output_dir / "merge_selector_veto_runs.csv", run_rows)
    write_report(args.output_dir / "MERGE_SELECTOR_VETO_REPORT.md", final)
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
    if not xs:
        return {"x": np.empty((0, 0), dtype=np.float32), "y": np.empty((0,), dtype=np.int64)}
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_edges and x.shape[0] > max_edges:
        rng = np.random.default_rng(seed)
        indices = rng.choice(np.arange(x.shape[0]), size=max_edges, replace=False)
        x = x[indices]
        y = y[indices]
    return {"x": x, "y": y}


def selected_items(manifest: dict[str, Any], *, offset: int, limit: int | None, doc_ids: list[str] | None) -> list[dict[str, Any]]:
    items = sorted(list(manifest.get("items") or []), key=lambda item: str(item.get("doc_id") or ""))
    if doc_ids:
        wanted = set(doc_ids)
        items = [item for item in items if str(item.get("doc_id")) in wanted]
    end = offset + limit if limit is not None else None
    return items[offset:end]


def parse_configs(values: list[str]) -> list[SelectorConfig]:
    configs: list[SelectorConfig] = []
    for value in values:
        name, det_tau, body_tau, list_tau, cap = value.split(":")
        configs.append(SelectorConfig(name, float(det_tau), float(body_tau), float(list_tau), int(cap)))
    return configs


def project_config(
    *,
    config: SelectorConfig,
    items: list[dict[str, Any]],
    selected_lookup: dict[str, dict[str, Any]],
    selector: Any,
    torch: Any,
    feature_plan: dict[str, Any],
    variant: str,
    overlay_policy: str,
    max_overlay_reading_gap: int,
    component_size_cap: int,
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
        if skip_existing and (doc_dir / "selector_report.json").exists():
            rows.append(json.loads((doc_dir / "selector_report.json").read_text(encoding="utf-8"))["summary"])
            print(f"[{config.name} {index}/{len(items)}] skip existing {doc_id}")
            continue
        try:
            row = process_doc(
                item,
                selected_lookup=selected_lookup,
                selector=selector,
                torch=torch,
                feature_plan=feature_plan,
                variant=variant,
                config=config,
                overlay_policy=overlay_policy,
                max_overlay_reading_gap=max_overlay_reading_gap,
                component_size_cap=component_size_cap,
                output_dir=doc_dir,
                render=render,
                run_paragraph_audit=run_paragraph_audit,
                compile_pdf=compile_pdf,
                compile_engine=compile_engine,
                compile_timeout=compile_timeout,
            )
            rows.append(row)
            print(
                f"[{config.name} {index}/{len(items)}] {doc_id} "
                f"kept_det={row['selector_kept_deterministic_count']} vetoed={row['selector_vetoed_deterministic_count']} "
                f"added={row['selector_added_residual_count']} changed={row['generated_tex_changed']}"
            )
        except Exception as exc:
            error = {"doc_id": doc_id, "error": type(exc).__name__, "message": str(exc)}
            errors.append(error)
            print(f"[{config.name} {index}/{len(items)}] ERROR {doc_id}: {type(exc).__name__}: {exc}")
    return rows, errors


def process_doc(
    item: dict[str, Any],
    *,
    selected_lookup: dict[str, dict[str, Any]],
    selector: Any,
    torch: Any,
    feature_plan: dict[str, Any],
    variant: str,
    config: SelectorConfig,
    overlay_policy: str,
    max_overlay_reading_gap: int,
    component_size_cap: int,
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
    probs = predict_proba(selector, edge_attr.cpu().numpy()).tolist()

    deterministic_payload = build_v8_from_middle(
        doc_id=doc_id,
        middle_json_path=source_paths["middle_json"],
        content_list_json_path=source_paths.get("content_list_json"),
        style_content_list_json_path=source_paths.get("style_content_list_json"),
        middle_block_source=str(graph_payload.get("source", {}).get("middle_block_source") or "preproc_blocks"),
    )
    selector_result = select_owner_merges(
        graph_payload=graph_payload,
        edge_records=data.edge_records,
        probs=probs,
        deterministic_payload=deterministic_payload,
        config=config,
        overlay_policy=overlay_policy,
        max_overlay_reading_gap=max_overlay_reading_gap,
        component_size_cap=component_size_cap,
    )
    selector_payload = build_selector_payload(
        doc_id=doc_id,
        graph_payload=graph_payload,
        source_paths=source_paths,
        decisions=selector_result["selected_decisions"],
    )

    dump_json(output_dir / f"{doc_id}_content_list_v8_deterministic.json", strip_diagnostics(deterministic_payload))
    dump_json(output_dir / f"{doc_id}_content_list_v8_selector_veto.json", strip_diagnostics(selector_payload))

    selector_sidecar = {
        "schema_version": "v8_merge_selector_doc_sidecar_v1",
        "doc_id": doc_id,
        "graph_path": str(graph_path),
        "config": config.__dict__,
        "source_paths": {key: str(value) for key, value in source_paths.items() if value},
        **{key: value for key, value in selector_result.items() if key != "selected_decisions"},
        "deterministic_merge_count": len(deterministic_payload.get("merge_decisions") or []),
        "selector_merge_count": len(selector_payload.get("merge_decisions") or []),
    }
    dump_json(output_dir / "selector_projection_sidecar.json", selector_sidecar)

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
        render_summaries["selector_veto"] = render_payload(
            doc_id,
            selector_payload,
            source_paths,
            output_dir / "selector_veto",
            compile_pdf=compile_pdf,
            compile_engine=compile_engine,
            compile_timeout=compile_timeout,
        )
        if run_paragraph_audit and source_paths.get("source_tex"):
            for mode, tex_dir in (
                ("deterministic", output_dir / "deterministic"),
                ("selector_veto", output_dir / "selector_veto"),
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
        selector_tex = (output_dir / "selector_veto" / "generated.tex").read_text(encoding="utf-8", errors="ignore")
        generated_tex_changed = normalize_tex_for_hash(det_tex) != normalize_tex_for_hash(selector_tex)

    summary = summarize_selector_doc(
        doc_id=doc_id,
        sidecar=selector_sidecar,
        render_summaries=render_summaries,
        generated_tex_changed=generated_tex_changed,
    )
    dump_json(output_dir / "selector_report.json", {"selector": selector_sidecar, "render": render_summaries, "summary": summary})
    return summary


def select_owner_merges(
    *,
    graph_payload: dict[str, Any],
    edge_records: list[dict[str, Any]],
    probs: list[float],
    deterministic_payload: dict[str, Any],
    config: SelectorConfig,
    overlay_policy: str,
    max_overlay_reading_gap: int,
    component_size_cap: int,
) -> dict[str, Any]:
    nodes = {str(node.get("atomic_id")): node for node in graph_payload.get("nodes") or []}
    edge_payloads = {str(edge.get("edge_id")): edge for edge in graph_payload.get("candidate_edges") or []}
    det_decisions = [decision_from_json(row) for row in deterministic_payload.get("merge_decisions") or []]
    det_pairs = {(decision.src_block_id, decision.dst_block_id) for decision in det_decisions}
    det_decision_by_pair = {(decision.src_block_id, decision.dst_block_id): decision for decision in det_decisions}

    scored_by_pair: dict[tuple[str, str], Candidate] = {}
    residual_candidates: list[Candidate] = []
    rejection_reasons: Counter[str] = Counter()
    for rec, prob in zip(edge_records, probs, strict=False):
        src = nodes.get(str(rec.get("src")))
        dst = nodes.get(str(rec.get("dst")))
        edge_payload = edge_payloads.get(str(rec.get("edge_id")), {})
        if not src or not dst:
            rejection_reasons["missing_endpoint"] += 1
            continue
        src_block = str(src.get("source_middle_block_id") or "")
        dst_block = str(dst.get("source_middle_block_id") or "")
        if not src_block or not dst_block or src_block == dst_block:
            rejection_reasons["missing_or_same_owner"] += 1
            continue
        family = str(rec.get("candidate_family") or edge_payload.get("candidate_family") or "")
        candidate = Candidate(
            src_block_id=src_block,
            dst_block_id=dst_block,
            probability=float(prob),
            source="deterministic_scored" if (src_block, dst_block) in det_pairs else "residual_scored",
            family=family,
            src_reading_order=int(src.get("reading_order") or 0),
            dst_reading_order=int(dst.get("reading_order") or 0),
            edge_id=str(rec.get("edge_id") or ""),
            src_atomic_id=str(rec.get("src") or ""),
            dst_atomic_id=str(rec.get("dst") or ""),
            src_text=str(src.get("text") or ""),
            dst_text=str(dst.get("text") or ""),
            evidence={
                "candidate_family": family,
                "edge_id": rec.get("edge_id"),
                "src_atomic_id": rec.get("src"),
                "dst_atomic_id": rec.get("dst"),
                "probability": round(float(prob), 6),
                "layout_scope": edge_payload.get("layout_scope"),
                "reading_order_gap": edge_payload.get("reading_order_gap"),
                "skipped_channels": edge_payload.get("skipped_channels"),
            },
        )
        pair = (src_block, dst_block)
        old = scored_by_pair.get(pair)
        if old is None or candidate.probability > old.probability:
            scored_by_pair[pair] = candidate
        if pair not in det_pairs:
            body_tau = config.body_threshold if family == "BODY_TEXT_CONTINUATION" else config.list_threshold
            if family not in RESIDUAL_FAMILIES or candidate.probability < body_tau:
                rejection_reasons["residual_below_threshold_or_family_disabled"] += 1
                continue
            ok, reasons = strict_continuation_overlay_gate(
                rec=rec,
                edge_payload=edge_payload,
                src=src,
                dst=dst,
                probability=float(prob),
                family_threshold=body_tau,
                max_reading_gap=max_overlay_reading_gap,
                conservative=overlay_policy == "strict_continuation_v2_conservative",
            )
            if not ok:
                rejection_reasons.update(Counter({f"residual:{key}": value for key, value in reasons.items()}))
                continue
            residual_candidates.append(candidate)

    selected_candidates: list[Candidate] = []
    vetoed: list[Candidate] = []
    unscored_kept: list[MergeDecision] = []
    for decision in det_decisions:
        pair = (decision.src_block_id, decision.dst_block_id)
        scored = scored_by_pair.get(pair)
        if scored is None:
            unscored_kept.append(decision)
            continue
        if scored.probability >= config.deterministic_threshold:
            scored.source = "deterministic_kept"
            scored.reason = "selector_keep_deterministic"
            selected_candidates.append(scored)
        else:
            scored.source = "deterministic_vetoed"
            scored.reason = "selector_veto_deterministic"
            vetoed.append(scored)

    residual_candidates = sorted(residual_candidates, key=lambda value: (-value.probability, value.src_reading_order))
    residual_candidates = residual_candidates[: max(0, config.residual_cap)]
    for candidate in residual_candidates:
        candidate.source = "residual_added"
        candidate.reason = f"selector_add_{candidate.family.lower()}"
        selected_candidates.append(candidate)

    solver = solve_merge_set(
        selected_candidates=selected_candidates,
        unscored_kept=unscored_kept,
        original_decision_by_pair=det_decision_by_pair,
        component_size_cap=component_size_cap,
    )
    return {
        "selected_decisions": solver["decisions"],
        "selected_candidates": [candidate_to_json(candidate) for candidate in solver["selected_candidates"]],
        "unscored_deterministic_kept": [decision.to_json() for decision in unscored_kept],
        "vetoed_deterministic": [candidate_to_json(candidate) for candidate in vetoed],
        "solver_rejected": solver["solver_rejected"],
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "scored_deterministic_count": sum(1 for pair in det_pairs if pair in scored_by_pair),
        "unscored_deterministic_count": len(unscored_kept),
        "selector_kept_deterministic_count": sum(1 for candidate in solver["selected_candidates"] if candidate.source == "deterministic_kept"),
        "selector_added_residual_count": sum(1 for candidate in solver["selected_candidates"] if candidate.source == "residual_added"),
        "selector_vetoed_deterministic_count": len(vetoed),
        "residual_candidate_count": len(residual_candidates),
    }


def solve_merge_set(
    *,
    selected_candidates: list[Candidate],
    unscored_kept: list[MergeDecision],
    original_decision_by_pair: dict[tuple[str, str], MergeDecision],
    component_size_cap: int,
) -> dict[str, Any]:
    outgoing = {decision.src_block_id for decision in unscored_kept}
    incoming = {decision.dst_block_id for decision in unscored_kept}
    parent: dict[str, str] = {}
    size: Counter[str] = Counter()
    for decision in unscored_kept:
        union(parent, size, decision.src_block_id, decision.dst_block_id)

    selected: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    ordered = sorted(
        selected_candidates,
        key=lambda candidate: (
            0 if candidate.source == "deterministic_kept" else 1,
            candidate.src_reading_order,
            candidate.dst_reading_order,
            -candidate.probability,
        ),
    )
    for candidate in ordered:
        reason = solver_reject_reason(candidate, outgoing=outgoing, incoming=incoming, parent=parent, size=size, cap=component_size_cap)
        if reason:
            rejected.append({**candidate_to_json(candidate), "solver_reject_reason": reason})
            continue
        outgoing.add(candidate.src_block_id)
        incoming.add(candidate.dst_block_id)
        union(parent, size, candidate.src_block_id, candidate.dst_block_id)
        selected.append(candidate)

    decisions = list(unscored_kept)
    for candidate in selected:
        original = original_decision_by_pair.get((candidate.src_block_id, candidate.dst_block_id))
        evidence = dict(candidate.evidence or {})
        if original is not None:
            evidence["original_deterministic_reason"] = original.reason
            evidence["original_deterministic_confidence"] = original.confidence
        decisions.append(
            MergeDecision(
                src_block_id=candidate.src_block_id,
                dst_block_id=candidate.dst_block_id,
                reason=candidate.reason or candidate.source,
                confidence=float(candidate.probability),
                evidence=evidence,
            )
        )
    decisions = sorted(decisions, key=lambda decision: (decision.src_block_id, decision.dst_block_id, decision.reason))
    return {"decisions": decisions, "selected_candidates": selected, "solver_rejected": rejected}


def solver_reject_reason(
    candidate: Candidate,
    *,
    outgoing: set[str],
    incoming: set[str],
    parent: dict[str, str],
    size: Counter[str],
    cap: int,
) -> str | None:
    if candidate.dst_reading_order <= candidate.src_reading_order:
        return "reverse_or_same_reading_order"
    if candidate.src_block_id in outgoing:
        return "source_already_has_outgoing"
    if candidate.dst_block_id in incoming:
        return "destination_already_has_incoming"
    root_a = find(parent, candidate.src_block_id)
    root_b = find(parent, candidate.dst_block_id)
    if root_a == root_b:
        return "cycle"
    component_size = size[root_a] + size[root_b]
    if component_size > cap:
        return "component_size_cap"
    return None


def find(parent: dict[str, str], value: str) -> str:
    if value not in parent:
        parent[value] = value
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def union(parent: dict[str, str], size: Counter[str], a: str, b: str) -> None:
    root_a = find(parent, a)
    root_b = find(parent, b)
    if root_a == root_b:
        return
    if size[root_a] <= 0:
        size[root_a] = 1
    if size[root_b] <= 0:
        size[root_b] = 1
    if size[root_a] < size[root_b]:
        root_a, root_b = root_b, root_a
    parent[root_b] = root_a
    size[root_a] += size[root_b]


def candidate_to_json(candidate: Candidate) -> dict[str, Any]:
    return {
        "src_block_id": candidate.src_block_id,
        "dst_block_id": candidate.dst_block_id,
        "probability": round(float(candidate.probability), 6),
        "source": candidate.source,
        "family": candidate.family,
        "src_reading_order": candidate.src_reading_order,
        "dst_reading_order": candidate.dst_reading_order,
        "edge_id": candidate.edge_id,
        "src_atomic_id": candidate.src_atomic_id,
        "dst_atomic_id": candidate.dst_atomic_id,
        "src_text": candidate.src_text,
        "dst_text": candidate.dst_text,
        "reason": candidate.reason,
        "evidence": candidate.evidence or {},
    }


def decision_from_json(row: dict[str, Any]) -> MergeDecision:
    return MergeDecision(
        src_block_id=str(row.get("src_block_id")),
        dst_block_id=str(row.get("dst_block_id")),
        reason=str(row.get("reason") or "deterministic"),
        confidence=float(row.get("confidence") or 0.0),
        evidence=row.get("evidence") if isinstance(row.get("evidence"), dict) else {},
    )


def build_selector_payload(
    *,
    doc_id: str,
    graph_payload: dict[str, Any],
    source_paths: dict[str, Path],
    decisions: list[MergeDecision],
) -> dict[str, Any]:
    source = graph_payload.get("source") if isinstance(graph_payload.get("source"), dict) else {}
    ordered_blocks = load_and_order_v8_blocks(
        doc_id=doc_id,
        middle_json_path=source_paths["middle_json"],
        content_list_json_path=source_paths.get("content_list_json"),
        style_content_list_json_path=source_paths.get("style_content_list_json"),
        middle_block_source=str(source.get("middle_block_source") or "preproc_blocks"),
    )
    items = materialize_v8_items(ordered_blocks, decisions)
    diagnostics = build_diagnostics(
        doc_id=doc_id,
        middle_json_path=source_paths["middle_json"],
        content_list_json_path=source_paths.get("content_list_json"),
        source=str(source.get("middle_block_source") or "preproc_blocks"),
        ordered_blocks=ordered_blocks,
        merge_decisions=decisions,
        items=items,
        debug_page=None,
    )
    return {
        "schema_version": "content_list_v8_reflow_v1",
        "doc_id": doc_id,
        "source": {
            "middle_json": str(source_paths["middle_json"]),
            "content_list_json": str(source_paths.get("content_list_json")) if source_paths.get("content_list_json") else None,
            "style_content_list_json": str(source_paths.get("style_content_list_json"))
            if source_paths.get("style_content_list_json")
            else None,
            "middle_block_source": str(source.get("middle_block_source") or "preproc_blocks"),
            "projection": "selector_veto",
        },
        "items": items,
        "atomic_blocks": [block.to_json() for block in ordered_blocks],
        "merge_decisions": [decision.to_json() for decision in decisions],
        "diagnostics": diagnostics,
    }


def summarize_selector_doc(
    *,
    doc_id: str,
    sidecar: dict[str, Any],
    render_summaries: dict[str, Any],
    generated_tex_changed: bool,
) -> dict[str, Any]:
    det_audit = (render_summaries.get("deterministic") or {}).get("paragraph_audit_summary") or {}
    selector_audit = (render_summaries.get("selector_veto") or {}).get("paragraph_audit_summary") or {}
    return {
        "doc_id": doc_id,
        "deterministic_merge_count": sidecar["deterministic_merge_count"],
        "selector_merge_count": sidecar["selector_merge_count"],
        "scored_deterministic_count": sidecar["scored_deterministic_count"],
        "unscored_deterministic_count": sidecar["unscored_deterministic_count"],
        "selector_kept_deterministic_count": sidecar["selector_kept_deterministic_count"],
        "selector_vetoed_deterministic_count": sidecar["selector_vetoed_deterministic_count"],
        "selector_added_residual_count": sidecar["selector_added_residual_count"],
        "solver_rejected_count": len(sidecar.get("solver_rejected") or []),
        "generated_tex_changed": generated_tex_changed,
        "deterministic_missing_merge_rate": det_audit.get("missing_merge_rate_among_covered"),
        "selector_missing_merge_rate": selector_audit.get("missing_merge_rate_among_covered"),
        "deterministic_wrong_merge_rate": det_audit.get("wrong_merge_rate_among_generated"),
        "selector_wrong_merge_rate": selector_audit.get("wrong_merge_rate_among_generated"),
        "deterministic_paragraph_count_delta": det_audit.get("paragraph_count_delta"),
        "selector_paragraph_count_delta": selector_audit.get("paragraph_count_delta"),
        "deterministic_source_coverage_rate": det_audit.get("source_coverage_rate"),
        "selector_source_coverage_rate": selector_audit.get("source_coverage_rate"),
        "deterministic_body_source_coverage_rate": det_audit.get("body_source_coverage_rate"),
        "selector_body_source_coverage_rate": selector_audit.get("body_source_coverage_rate"),
    }


def aggregate_selector_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    aggregate: dict[str, Any] = {
        "generated_tex_changed_count": sum(1 for row in rows if row.get("generated_tex_changed")),
        "selector_added_residual_total": sum(int(row.get("selector_added_residual_count") or 0) for row in rows),
        "selector_vetoed_deterministic_total": sum(int(row.get("selector_vetoed_deterministic_count") or 0) for row in rows),
        "selector_kept_deterministic_total": sum(int(row.get("selector_kept_deterministic_count") or 0) for row in rows),
        "unscored_deterministic_total": sum(int(row.get("unscored_deterministic_count") or 0) for row in rows),
    }
    for key in (
        "deterministic_missing_merge_rate",
        "selector_missing_merge_rate",
        "deterministic_wrong_merge_rate",
        "selector_wrong_merge_rate",
        "deterministic_source_coverage_rate",
        "selector_source_coverage_rate",
        "deterministic_body_source_coverage_rate",
        "selector_body_source_coverage_rate",
        "deterministic_paragraph_count_delta",
        "selector_paragraph_count_delta",
    ):
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        if vals:
            aggregate[f"mean_{key}"] = round(sum(vals) / len(vals), 6)
    return aggregate


def add_utility_fields(summary: dict[str, Any]) -> None:
    agg = summary.get("aggregate") or {}
    det_missing = agg.get("mean_deterministic_missing_merge_rate")
    sel_missing = agg.get("mean_selector_missing_merge_rate")
    det_wrong = agg.get("mean_deterministic_wrong_merge_rate")
    sel_wrong = agg.get("mean_selector_wrong_merge_rate")
    det_cov = agg.get("mean_deterministic_body_source_coverage_rate")
    sel_cov = agg.get("mean_selector_body_source_coverage_rate")
    if det_cov is None or sel_cov is None:
        det_cov = agg.get("mean_deterministic_source_coverage_rate")
        sel_cov = agg.get("mean_selector_source_coverage_rate")
    if None in {det_missing, sel_missing, det_wrong, sel_wrong}:
        return
    missing_reduction = float(det_missing) - float(sel_missing)
    wrong_increase = float(sel_wrong) - float(det_wrong)
    agg["missing_reduction"] = round(missing_reduction, 6)
    agg["wrong_increase"] = round(wrong_increase, 6)
    for lam in (3, 5, 10):
        agg[f"utility_{lam}"] = round(missing_reduction - lam * wrong_increase, 6)
    agg["wrong_merge_constraint_pass"] = bool(float(sel_wrong) <= float(det_wrong) + 1e-12)
    if det_cov is not None and sel_cov is not None:
        agg["source_coverage_constraint_pass"] = bool(float(sel_cov) >= float(det_cov) - 1e-12)


def config_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    add_utility_fields(summary)
    agg = summary.get("aggregate") or {}
    return {
        "name": summary["name"],
        "deterministic_threshold": summary["deterministic_threshold"],
        "body_threshold": summary["body_threshold"],
        "list_threshold": summary["list_threshold"],
        "residual_cap": summary["residual_cap"],
        "doc_count": summary["doc_count"],
        "error_count": summary["error_count"],
        "selector_vetoed_deterministic_total": agg.get("selector_vetoed_deterministic_total"),
        "selector_added_residual_total": agg.get("selector_added_residual_total"),
        "generated_tex_changed_count": agg.get("generated_tex_changed_count"),
        "det_missing": agg.get("mean_deterministic_missing_merge_rate"),
        "selector_missing": agg.get("mean_selector_missing_merge_rate"),
        "missing_reduction": agg.get("missing_reduction"),
        "det_wrong": agg.get("mean_deterministic_wrong_merge_rate"),
        "selector_wrong": agg.get("mean_selector_wrong_merge_rate"),
        "wrong_increase": agg.get("wrong_increase"),
        "det_source_coverage": agg.get("mean_deterministic_source_coverage_rate"),
        "selector_source_coverage": agg.get("mean_selector_source_coverage_rate"),
        "det_body_source_coverage": agg.get("mean_deterministic_body_source_coverage_rate"),
        "selector_body_source_coverage": agg.get("mean_selector_body_source_coverage_rate"),
        "det_paragraph_delta": agg.get("mean_deterministic_paragraph_count_delta"),
        "selector_paragraph_delta": agg.get("mean_selector_paragraph_count_delta"),
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
            "status": "selector_not_ready_for_production",
            "reason": "No selector configuration satisfied wrong_merge <= deterministic, source_coverage >= deterministic, missing reduction > 0, and utility_5 > 0.",
        }
    best = max(viable, key=lambda row: (row.get("utility_5") or -999, row.get("missing_reduction") or 0))
    return {
        "status": "selector_candidate_found",
        "best_config": best["name"],
        "reason": "At least one selector/veto configuration passed the conservative production constraints.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, final: dict[str, Any]) -> None:
    lines = [
        "# Merge Selector / Veto Report",
        "",
        "## Status",
        f"- model_backend: {final['model_backend']}",
        f"- policy: {final['policy']}",
        f"- variant: {final['variant']}",
        f"- train_edges: {final['train_edges']}",
        f"- val_edges: {final['val_edges']}",
        "- production v8 deterministic path changed: No",
        "- project GNN training: No",
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
            "## Selector/Veto Results",
            "| config | det tau | body tau | list tau | cap | vetoed det | added residual | changed | missing reduction | wrong increase | utility_3 | utility_5 | utility_10 | wrong<=det | source>=det |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in final["runs"]:
        lines.append(
            f"| {row['name']} | {row['deterministic_threshold']} | {row['body_threshold']} | "
            f"{row['list_threshold']} | {row['residual_cap']} | {row['selector_vetoed_deterministic_total']} | "
            f"{row['selector_added_residual_total']} | {row['generated_tex_changed_count']} | "
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
            "The selector/veto branch is allowed to improve production only if wrong_merge_rate does not exceed deterministic v8, source coverage does not drop, and utility_5 is positive.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
