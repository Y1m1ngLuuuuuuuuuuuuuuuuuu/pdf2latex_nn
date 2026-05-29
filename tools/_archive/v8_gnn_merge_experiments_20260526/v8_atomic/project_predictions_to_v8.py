#!/usr/bin/env python3
"""Project v8 atomic MERGE predictions back to v8 logical owners.

This is an evaluation-only bridge for the optional v8 atomic MERGE branch:

  atomic graph + trained MERGE checkpoint
    -> predicted atomic continuation edges
    -> owner-level v8 MergeDecision sidecar
    -> patched content_list_v8 payload
    -> normal v8 renderer
    -> paragraph-preservation comparison against TeX

It does not modify MinerU output, v8 JSON sources, graph tensors, labels, or the
default v8 renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline.run_v8_layout_reconstruction import ensure_v8_math_compatibility  # noqa: E402
from src.adapters.mineru_v8_document_ir import convert_v8_payload_to_document_ir  # noqa: E402
from src.evaluation.compile_eval import compile_latex  # noqa: E402
from src.generation.ir_renderer import IRLatexRenderConfig  # noqa: E402
from src.generation.render_surface import render_original_like_document  # noqa: E402
from src.generation.v8_style_detector import detect_v8_style  # noqa: E402
from src.ir.serialization import write_json  # noqa: E402
from src.perception.mineru_v8_reflow import (  # noqa: E402
    MergeDecision,
    build_diagnostics,
    build_v8_from_middle,
    dump_json,
    materialize_v8_items,
)
from src.perception.v8_atomic_merge import load_and_order_v8_blocks  # noqa: E402
from src.reasoning.front_matter_extractor import extract_front_matter  # noqa: E402
from src.reasoning.v8_render_tree import build_v8_render_tree  # noqa: E402
from tools.v8_atomic.diagnose_shortcut_feature_ablation import build_feature_plan  # noqa: E402
from tools.v8_atomic.train_v8_atomic_merge_ablation import (  # noqa: E402
    define_model_class,
    transformed_edge_attr,
)


ALLOWED_FAMILIES = {
    "body_list_focus": {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION"},
    "body_list_float_skip_focus": {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION", "FLOAT_SKIP_CONTINUATION"},
}

STRICT_CONTINUATION_FAMILIES = {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION"}
STRICT_ALLOWED_LAYOUT_SCOPES = {"same_column", "cross_page"}
STRICT_FORBIDDEN_CHANNELS = {
    "HEADING",
    "DISPLAY_MATH",
    "FRONT_MATTER",
    "REFERENCE_ITEM",
    "CAPTION",
    "FLOAT_PROXY",
    "PAGE_FURNITURE",
    "UNKNOWN",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--overlay-policy",
        choices=("legacy_family_threshold", "strict_continuation_v1", "strict_continuation_v2_conservative"),
        default="strict_continuation_v1",
        help="Owner projection policy. strict_continuation_v1 keeps v8 deterministic as mainline and uses GNN only as a high-precision overlay.",
    )
    parser.add_argument("--body-threshold", type=float, help="Optional family-specific threshold for BODY_TEXT_CONTINUATION.")
    parser.add_argument("--list-threshold", type=float, help="Optional family-specific threshold for LIST_CONTINUATION.")
    parser.add_argument("--max-overlay-reading-gap", type=int, default=8)
    parser.add_argument(
        "--max-added-owner-merges-per-doc",
        type=int,
        help="Optional high-precision guard: keep only the top-K learned owner merges per document after strict gating.",
    )
    parser.add_argument(
        "--residual-overlay-allowlist",
        type=Path,
        help="Optional residual-target allowlist from build_residual_overlay_allowlist.py. When set, learned overlay only accepts edges near listed residual gaps.",
    )
    parser.add_argument("--residual-allowlist-min-overlap", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--doc-ids", nargs="*")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-paragraph-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-engine", default="auto")
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("This projection tool requires torch and torch_geometric") from exc

    define_model_class(torch=torch, nn=nn, F=F, SAGEConv=SAGEConv)
    from tools.v8_atomic import train_v8_atomic_merge_ablation as trainer  # noqa: WPS433

    device = resolve_device(args.device, torch=torch)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = args.policy or str(checkpoint.get("policy") or "body_list_focus")
    variant = args.variant or str(checkpoint.get("variant") or "all_features")
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("best_threshold", 0.5))
    family_thresholds = {
        "BODY_TEXT_CONTINUATION": float(args.body_threshold if args.body_threshold is not None else threshold),
        "LIST_CONTINUATION": float(args.list_threshold if args.list_threshold is not None else threshold),
    }
    allowed_families = ALLOWED_FAMILIES.get(policy)
    if not allowed_families:
        raise SystemExit(f"Unsupported policy for projection: {policy}")

    manifest = load_manifest(args.manifest)
    selected_lookup = load_selected_lookup(manifest)
    residual_allowlist = load_residual_overlay_allowlist(args.residual_overlay_allowlist)
    items = list(manifest.get("items") or [])
    items = sorted(items, key=lambda item: str(item.get("doc_id") or ""))
    if args.doc_ids:
        wanted = set(args.doc_ids)
        items = [item for item in items if str(item.get("doc_id")) in wanted]
    end = args.offset + args.limit if args.limit is not None else None
    items = items[args.offset:end]
    if not items:
        raise SystemExit("No graph manifest items selected")

    sample_graph = Path(items[0]["graph_path"])
    feature_plan = build_feature_plan(sample_graph, torch=torch)
    model = trainer.AtomicMergeGNN(
        node_dim=int(checkpoint.get("node_dim")),
        edge_dim=int(checkpoint.get("edge_dim")),
        hidden_dim=int(checkpoint.get("hidden_dim", 64)),
        num_layers=int(checkpoint.get("num_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        doc_id = str(item.get("doc_id") or "").strip()
        out_doc = args.output_dir / doc_id
        if args.skip_existing and (out_doc / "projection_report.json").exists():
            rows.append(json.loads((out_doc / "projection_report.json").read_text(encoding="utf-8"))["summary"])
            print(f"[{index}/{len(items)}] skip existing {doc_id}")
            continue
        try:
            row = process_doc(
                item,
                selected_lookup=selected_lookup,
                model=model,
                torch=torch,
                device=device,
                feature_plan=feature_plan,
                variant=variant,
                threshold=threshold,
                family_thresholds=family_thresholds,
                allowed_families=allowed_families,
                overlay_policy=args.overlay_policy,
                max_overlay_reading_gap=args.max_overlay_reading_gap,
                max_added_owner_merges_per_doc=args.max_added_owner_merges_per_doc,
                residual_targets=residual_allowlist.get(doc_id),
                residual_allowlist_min_overlap=args.residual_allowlist_min_overlap,
                output_dir=out_doc,
                render=args.render,
                run_paragraph_audit=args.run_paragraph_audit,
                compile_pdf=args.compile,
                compile_engine=args.compile_engine,
                compile_timeout=args.compile_timeout,
            )
            rows.append(row)
            print(
                f"[{index}/{len(items)}] {doc_id} added={row['model_added_owner_merge_count']} "
                f"changed={row['generated_tex_changed']} det_missing={row.get('deterministic_missing_merge_rate')} "
                f"learned_missing={row.get('learned_missing_merge_rate')}"
            )
        except Exception as exc:  # keep batch moving
            error = {"doc_id": doc_id, "error": type(exc).__name__, "message": str(exc)}
            errors.append(error)
            print(f"[{index}/{len(items)}] ERROR {doc_id}: {type(exc).__name__}: {exc}")

    summary = {
        "schema_version": "v8_atomic_prediction_projection_eval_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "policy": policy,
        "variant": variant,
        "threshold": threshold,
        "family_thresholds": family_thresholds,
        "overlay_policy": args.overlay_policy,
        "max_overlay_reading_gap": args.max_overlay_reading_gap,
        "max_added_owner_merges_per_doc": args.max_added_owner_merges_per_doc,
        "residual_overlay_allowlist": str(args.residual_overlay_allowlist) if args.residual_overlay_allowlist else None,
        "residual_allowlist_min_overlap": args.residual_allowlist_min_overlap,
        "doc_count": len(rows),
        "error_count": len(errors),
        "aggregate": aggregate_rows(rows),
        "rows": rows,
        "errors": errors,
    }
    dump_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "summary.csv", rows)
    write_report(args.output_dir / "V8_ATOMIC_PREDICTION_PROJECTION_EVAL_REPORT.md", summary)
    return 0


def process_doc(
    item: dict[str, Any],
    *,
    selected_lookup: dict[str, dict[str, Any]],
    model: Any,
    torch: Any,
    device: Any,
    feature_plan: dict[str, Any],
    variant: str,
    threshold: float,
    family_thresholds: dict[str, float],
    allowed_families: set[str],
    overlay_policy: str,
    max_overlay_reading_gap: int,
    max_added_owner_merges_per_doc: int | None,
    residual_targets: list[dict[str, Any]] | None,
    residual_allowlist_min_overlap: int,
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

    probs = score_edges(data, model=model, torch=torch, device=device, feature_plan=feature_plan, variant=variant)
    predicted = select_predicted_owner_merges(
        graph_payload,
        data.edge_records,
        probs,
        threshold=threshold,
        family_thresholds=family_thresholds,
        allowed_families=allowed_families,
        overlay_policy=overlay_policy,
        max_overlay_reading_gap=max_overlay_reading_gap,
        max_added_owner_merges_per_doc=max_added_owner_merges_per_doc,
        residual_targets=residual_targets,
        residual_allowlist_min_overlap=residual_allowlist_min_overlap,
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
    dump_json(output_dir / f"{doc_id}_content_list_v8_learned_plus_deterministic.json", strip_diagnostics(learned_payload))

    projection_report = {
        "schema_version": "v8_atomic_projection_doc_report_v1",
        "doc_id": doc_id,
        "graph_path": str(graph_path),
        "source_paths": {key: str(value) for key, value in source_paths.items() if value},
        "threshold": threshold,
        "family_thresholds": family_thresholds,
        "overlay_policy": overlay_policy,
        "max_added_owner_merges_per_doc": max_added_owner_merges_per_doc,
        "allowed_families": sorted(allowed_families),
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
            output_dir / "learned_plus_deterministic",
            compile_pdf=compile_pdf,
            compile_engine=compile_engine,
            compile_timeout=compile_timeout,
        )
        if run_paragraph_audit and source_paths.get("source_tex"):
            for mode in ("deterministic", "learned_plus_deterministic"):
                audit_dir = output_dir / mode / "paragraph_audit"
                run_paragraph_preservation_audit(
                    source_tex=source_paths["source_tex"],
                    generated_tex=output_dir / mode / "generated.tex",
                    doc_id=doc_id,
                    output_dir=audit_dir,
                )
                audit = json.loads((audit_dir / "paragraph_preservation_against_tex.json").read_text(encoding="utf-8"))
                render_summaries[mode]["paragraph_audit_summary"] = audit.get("summary", {})

    generated_tex_changed = False
    if render:
        det_tex = (output_dir / "deterministic" / "generated.tex").read_text(encoding="utf-8", errors="ignore")
        learned_tex = (output_dir / "learned_plus_deterministic" / "generated.tex").read_text(encoding="utf-8", errors="ignore")
        generated_tex_changed = normalize_tex_for_hash(det_tex) != normalize_tex_for_hash(learned_tex)

    summary = summarize_doc(
        doc_id=doc_id,
        projection_report=projection_report,
        render_summaries=render_summaries,
        generated_tex_changed=generated_tex_changed,
    )
    dump_json(output_dir / "projection_report.json", {"projection": projection_report, "render": render_summaries, "summary": summary})
    return summary


def score_edges(data: Any, *, model: Any, torch: Any, device: Any, feature_plan: dict[str, Any], variant: str) -> list[float]:
    with torch.no_grad():
        cpu_data = data
        data_gpu = cpu_data.clone().to(device)
        edge_attr = transformed_edge_attr(data_gpu, variant, feature_plan=feature_plan, torch=torch).to(device)
        logits = model(data_gpu, edge_attr)
        return torch.sigmoid(logits).detach().cpu().tolist()


def select_predicted_owner_merges(
    graph_payload: dict[str, Any],
    edge_records: list[dict[str, Any]],
    probs: list[float],
    *,
    threshold: float,
    family_thresholds: dict[str, float],
    allowed_families: set[str],
    overlay_policy: str,
    max_overlay_reading_gap: int,
    max_added_owner_merges_per_doc: int | None = None,
    residual_targets: list[dict[str, Any]] | None = None,
    residual_allowlist_min_overlap: int = 2,
) -> dict[str, Any]:
    nodes = {str(node.get("atomic_id")): node for node in graph_payload.get("nodes") or []}
    edge_payloads = {str(edge.get("edge_id")): edge for edge in graph_payload.get("candidate_edges") or []}
    candidates: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    strict_candidate_count = 0
    for rec, prob in zip(edge_records, probs, strict=False):
        family = str(rec.get("candidate_family") or "")
        edge_payload = edge_payloads.get(str(rec.get("edge_id")), {})
        family_threshold = float(family_thresholds.get(family, threshold))
        if family not in allowed_families or float(prob) < family_threshold:
            rejection_reasons["below_threshold_or_family_disabled"] += 1
            continue
        src = nodes.get(str(rec.get("src")))
        dst = nodes.get(str(rec.get("dst")))
        if not src or not dst:
            rejection_reasons["missing_endpoint"] += 1
            continue
        if overlay_policy in {"strict_continuation_v1", "strict_continuation_v2_conservative"}:
            ok, reasons = strict_continuation_overlay_gate(
                rec=rec,
                edge_payload=edge_payload,
                src=src,
                dst=dst,
                probability=float(prob),
                family_threshold=family_threshold,
                max_reading_gap=max_overlay_reading_gap,
                conservative=overlay_policy == "strict_continuation_v2_conservative",
            )
            if not ok:
                rejection_reasons.update(reasons)
                continue
            strict_candidate_count += 1
        elif overlay_policy != "legacy_family_threshold":
            raise ValueError(f"unsupported overlay policy: {overlay_policy}")
        src_block = str(src.get("source_middle_block_id") or "")
        dst_block = str(dst.get("source_middle_block_id") or "")
        if not src_block or not dst_block:
            rejection_reasons["missing_owner"] += 1
            continue
        atomic = {
            "edge_id": rec.get("edge_id"),
            "src_atomic_id": rec.get("src"),
            "dst_atomic_id": rec.get("dst"),
            "src_block_id": src_block,
            "dst_block_id": dst_block,
            "src_text": src.get("text"),
            "dst_text": dst.get("text"),
            "candidate_family": family,
            "probability": round(float(prob), 6),
            "threshold": family_threshold,
            "overlay_policy": overlay_policy,
            "layout_scope": edge_payload.get("layout_scope"),
            "reading_order_gap": edge_payload.get("reading_order_gap"),
            "skipped_channels": edge_payload.get("skipped_channels"),
            "src_reading_order": int(src.get("reading_order") or 0),
            "dst_reading_order": int(dst.get("reading_order") or 0),
        }
        if residual_targets is not None and not matches_residual_target(
            atomic,
            residual_targets,
            min_overlap=residual_allowlist_min_overlap,
        ):
            rejection_reasons["not_in_residual_allowlist"] += 1
            continue
        candidates.append(atomic)
    cross_owner = [edge for edge in candidates if edge["src_block_id"] != edge["dst_block_id"]]
    accepted = greedy_owner_edges(cross_owner, max_count=max_added_owner_merges_per_doc)
    return {
        "predicted_atomic_edges": candidates,
        "cross_owner_atomic_edges": cross_owner,
        "accepted_owner_merges": accepted,
        "strict_candidate_count": strict_candidate_count,
        "residual_allowlist_target_count": len(residual_targets) if residual_targets is not None else None,
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
    }


def strict_continuation_overlay_gate(
    *,
    rec: dict[str, Any],
    edge_payload: dict[str, Any],
    src: dict[str, Any],
    dst: dict[str, Any],
    probability: float,
    family_threshold: float,
    max_reading_gap: int,
    conservative: bool = False,
) -> tuple[bool, Counter[str]]:
    """High-precision learned overlay gate.

    The default v8 deterministic merge remains the main path.  This gate only
    lets the GNN add conservative body/list continuation edges.
    """

    reasons: Counter[str] = Counter()
    family = str(rec.get("candidate_family") or edge_payload.get("candidate_family") or "")
    src_channel = str(src.get("channel") or "")
    dst_channel = str(dst.get("channel") or "")
    layout_scope = str(edge_payload.get("layout_scope") or "")
    skipped_channels = [str(value) for value in (edge_payload.get("skipped_channels") or [])]
    reading_gap = _safe_int(edge_payload.get("reading_order_gap"), 9999)

    if family not in STRICT_CONTINUATION_FAMILIES:
        reasons[f"disabled_family:{family or 'unknown'}"] += 1
    if src_channel != dst_channel or src_channel not in {"BODY_TEXT", "LIST_ITEM"}:
        reasons[f"endpoint_channel:{src_channel}->{dst_channel}"] += 1
    if layout_scope not in STRICT_ALLOWED_LAYOUT_SCOPES:
        reasons[f"layout_scope:{layout_scope or 'unknown'}"] += 1
    if reading_gap <= 0 or reading_gap > max_reading_gap:
        reasons["reading_gap_not_close"] += 1
    forbidden = sorted({channel for channel in skipped_channels if channel in STRICT_FORBIDDEN_CHANNELS})
    for channel in forbidden:
        reasons[f"skipped_barrier:{channel}"] += 1
    if any(channel in STRICT_FORBIDDEN_CHANNELS for channel in (src_channel, dst_channel)):
        reasons["forbidden_endpoint"] += 1

    src_text = str(src.get("text") or "")
    dst_text = str(dst.get("text") or "")
    src_open = _is_merge_tail_continuable(src_text)
    dst_cont = _starts_lowercase(dst_text) or _starts_parenthetical(dst_text)
    if not conservative:
        dst_cont = dst_cont or _starts_continuation_word(dst_text)
    if not src_open:
        reasons["src_not_open_ended"] += 1
    if not dst_cont:
        reasons["dst_not_continuation_like"] += 1

    if not _style_compatible(src, dst):
        reasons["style_incompatible"] += 1

    if probability < family_threshold:
        reasons["below_family_threshold"] += 1
    return not reasons, reasons


def greedy_owner_edges(edges: list[dict[str, Any]], *, max_count: int | None = None) -> list[dict[str, Any]]:
    best_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        key = (edge["src_block_id"], edge["dst_block_id"])
        old = best_by_pair.get(key)
        if old is None or edge["probability"] > old["probability"]:
            best_by_pair[key] = edge
    ordered = sorted(
        best_by_pair.values(),
        key=lambda edge: (edge["src_reading_order"], edge["dst_reading_order"], -edge["probability"]),
    )
    outgoing: set[str] = set()
    incoming: set[str] = set()
    accepted: list[dict[str, Any]] = []
    for edge in ordered:
        src = edge["src_block_id"]
        dst = edge["dst_block_id"]
        if src in outgoing or dst in incoming:
            continue
        if edge["dst_reading_order"] <= edge["src_reading_order"]:
            continue
        outgoing.add(src)
        incoming.add(dst)
        accepted.append(edge)
    if max_count is not None and max_count >= 0 and len(accepted) > max_count:
        accepted = sorted(accepted, key=lambda edge: (-float(edge.get("probability") or 0.0), edge["src_reading_order"]))[:max_count]
        accepted = sorted(accepted, key=lambda edge: (edge["src_reading_order"], edge["dst_reading_order"], -edge["probability"]))
    return accepted


def matches_residual_target(edge: dict[str, Any], targets: list[dict[str, Any]], *, min_overlap: int) -> bool:
    src_tokens = set(_overlay_tokenize(str(edge.get("src_text") or "")))
    dst_tokens = set(_overlay_tokenize(str(edge.get("dst_text") or "")))
    if not src_tokens or not dst_tokens:
        return False
    for target in targets:
        left_tokens = set(_overlay_tokenize(str(target.get("left_tail") or "")))
        right_tokens = set(_overlay_tokenize(str(target.get("right_head") or "")))
        if not left_tokens or not right_tokens:
            continue
        left_overlap = len(src_tokens & left_tokens)
        right_overlap = len(dst_tokens & right_tokens)
        if left_overlap >= min_overlap and right_overlap >= max(1, min_overlap - 1):
            edge["residual_target_id"] = target.get("target_id")
            edge["residual_target_reason"] = target.get("reason")
            edge["residual_target_left_tail"] = target.get("left_tail")
            edge["residual_target_right_head"] = target.get("right_head")
            return True
    return False


def _overlay_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_projected_payload(
    *,
    doc_id: str,
    graph_payload: dict[str, Any],
    source_paths: dict[str, Path],
    predicted: dict[str, Any],
    deterministic_payload: dict[str, Any],
) -> dict[str, Any]:
    source = graph_payload.get("source") if isinstance(graph_payload.get("source"), dict) else {}
    ordered_blocks = load_and_order_v8_blocks(
        doc_id=doc_id,
        middle_json_path=source_paths["middle_json"],
        content_list_json_path=source_paths.get("content_list_json"),
        style_content_list_json_path=source_paths.get("style_content_list_json"),
        middle_block_source=str(source.get("middle_block_source") or "preproc_blocks"),
    )
    decisions = [decision_from_json(row) for row in deterministic_payload.get("merge_decisions") or []]
    existing_pairs = {(decision.src_block_id, decision.dst_block_id) for decision in decisions}
    for edge in predicted["accepted_owner_merges"]:
        pair = (edge["src_block_id"], edge["dst_block_id"])
        if pair in existing_pairs:
            continue
        decisions.append(
            MergeDecision(
                src_block_id=edge["src_block_id"],
                dst_block_id=edge["dst_block_id"],
                reason=f"gnn_atomic_{edge['candidate_family'].lower()}",
                confidence=float(edge["probability"]),
                evidence={
                    "atomic_edge_id": edge.get("edge_id"),
                    "src_atomic_id": edge.get("src_atomic_id"),
                    "dst_atomic_id": edge.get("dst_atomic_id"),
                    "probability": edge.get("probability"),
                    "src_text": edge.get("src_text"),
                    "dst_text": edge.get("dst_text"),
                },
            )
        )
        existing_pairs.add(pair)
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
            "style_content_list_json": str(source_paths.get("style_content_list_json")) if source_paths.get("style_content_list_json") else None,
            "middle_block_source": str(source.get("middle_block_source") or "preproc_blocks"),
            "projection": "learned_plus_deterministic",
        },
        "items": items,
        "atomic_blocks": [block.to_json() for block in ordered_blocks],
        "merge_decisions": [decision.to_json() for decision in decisions],
        "diagnostics": diagnostics,
    }


def decision_from_json(row: dict[str, Any]) -> MergeDecision:
    return MergeDecision(
        src_block_id=str(row.get("src_block_id")),
        dst_block_id=str(row.get("dst_block_id")),
        reason=str(row.get("reason") or "deterministic"),
        confidence=float(row.get("confidence") or 0.0),
        evidence=row.get("evidence") if isinstance(row.get("evidence"), dict) else {},
    )


def render_payload(
    doc_id: str,
    payload: dict[str, Any],
    source_paths: dict[str, Path],
    output_dir: Path,
    *,
    compile_pdf: bool,
    compile_engine: str,
    compile_timeout: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    v8_path = output_dir / f"{doc_id}_content_list_v8.json"
    dump_json(v8_path, strip_diagnostics(payload))
    if source_paths.get("pdf") and source_paths["pdf"].exists():
        shutil.copy2(source_paths["pdf"], output_dir / "original.pdf")
    document = convert_v8_payload_to_document_ir(payload, source_path=v8_path, pdf_path=source_paths.get("pdf"), doc_id=doc_id)
    write_json(output_dir / "document_ir.json", document)
    front_matter = extract_front_matter(document)
    write_json(output_dir / "front_matter_diag.json", front_matter.to_diagnostic())
    tree = build_v8_render_tree(document, document_ir_path=str(output_dir / "document_ir.json"), front_matter=front_matter)
    write_json(output_dir / "render_tree_ir.json", tree)
    style, style_diagnostics = detect_v8_style(document, tree=tree)
    write_json(output_dir / "style_profile.json", style)
    write_json(output_dir / "v8_style_detector_diag.json", style_diagnostics)
    tex = render_original_like_document(
        document,
        tree,
        style=style,
        config=IRLatexRenderConfig(
            title=None,
            include_maketitle=False,
            front_matter_mode="original_like",
            table_asset_output_dir=output_dir / "assets",
            figure_asset_output_dir=output_dir / "assets",
            table_asset_latex_prefix="assets",
            figure_asset_latex_prefix="assets",
        ),
        resolve_citations=True,
        source_tex_path=source_paths.get("source_tex"),
    )
    tex = ensure_v8_math_compatibility(tex)
    tex_path = output_dir / "generated.tex"
    tex_path.write_text(tex, encoding="utf-8")
    compile_report: dict[str, Any] = {"success": "not_run", "skipped": True}
    if compile_pdf:
        compile_report = compile_latex(tex_path, output_dir=output_dir, engine=compile_engine, timeout=compile_timeout, passes=2)
    write_json(output_dir / "compile_report.json", compile_report)
    return {
        "v8_content_json": str(v8_path),
        "generated_tex": str(tex_path),
        "document_node_count": len(document.nodes),
        "render_tree_node_count": len(tree.nodes),
        "compile": compile_report,
        "tex_hash": hash_text(normalize_tex_for_hash(tex)),
    }


def run_paragraph_preservation_audit(*, source_tex: Path, generated_tex: Path, doc_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "tools/audit/check_paragraph_preservation_against_tex.py"),
        "--source-tex",
        str(source_tex),
        "--generated-tex",
        str(generated_tex),
        "--doc-id",
        doc_id,
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)


def resolve_source_paths(item: dict[str, Any], selected: dict[str, Any], graph_payload: dict[str, Any]) -> dict[str, Path]:
    source = graph_payload.get("source") if isinstance(graph_payload.get("source"), dict) else {}
    paths = item.get("paths") if isinstance(item.get("paths"), dict) else {}
    return {
        "middle_json": first_existing(paths.get("middle_json"), source.get("middle_json")),
        "content_list_json": first_existing(paths.get("content_list_json"), source.get("content_list_json")),
        "style_content_list_json": first_existing(paths.get("style_content_list_json"), source.get("style_content_list_json")),
        "source_tex": first_existing(
            paths.get("source_tex"),
            selected.get("tex_path"),
            selected.get("main_tex"),
            find_main_tex(selected.get("tex_source_dir"), doc_id=str(item.get("doc_id") or selected.get("doc_id") or "")),
        ),
        "pdf": first_existing(selected.get("pdf_path")),
    }


def first_existing(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(str(value))
        if path.exists():
            return path
    return None


def find_main_tex(tex_source_dir: Any, *, doc_id: str) -> Path | None:
    if not tex_source_dir:
        return None
    root = Path(str(tex_source_dir))
    if not root.exists():
        return None
    preferred = ["main.tex", "paper.tex", "ms.tex", "article.tex", "source.tex", "root.tex"]
    if doc_id:
        preferred.append(f"{doc_id}.tex")
    for name in preferred:
        path = root / name
        if path.exists():
            return path
    candidates = sorted(root.rglob("*.tex"))
    roots: list[Path] = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:80000]
        except OSError:
            continue
        if "\\documentclass" in text or "\\documentstyle" in text:
            roots.append(path)
    if roots:
        return sorted(roots, key=lambda value: (len(str(value)), str(value)))[0]
    return candidates[0] if candidates else None


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a manifest object")
    return payload


def load_selected_lookup(graph_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected_path = Path(str(graph_manifest.get("selected_manifest") or ""))
    if not selected_path.exists():
        return {}
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    return {str(item.get("doc_id")): item for item in items if isinstance(item, dict)}


def load_residual_overlay_allowlist(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, list[dict[str, Any]]] = {}
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "")
        targets = item.get("targets") if isinstance(item.get("targets"), list) else []
        lookup[doc_id] = targets
    return lookup


def strip_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "diagnostics"}


def summarize_doc(
    *,
    doc_id: str,
    projection_report: dict[str, Any],
    render_summaries: dict[str, Any],
    generated_tex_changed: bool,
) -> dict[str, Any]:
    det_audit = (render_summaries.get("deterministic") or {}).get("paragraph_audit_summary") or {}
    learned_audit = (render_summaries.get("learned_plus_deterministic") or {}).get("paragraph_audit_summary") or {}
    return {
        "doc_id": doc_id,
        "deterministic_merge_count": projection_report["deterministic_merge_count"],
        "learned_plus_deterministic_merge_count": projection_report["learned_plus_deterministic_merge_count"],
        "model_predicted_atomic_merge_count": projection_report["predicted_atomic_merge_count"],
        "model_predicted_cross_owner_atomic_merge_count": projection_report["predicted_cross_owner_atomic_merge_count"],
        "strict_overlay_candidate_count": projection_report.get("strict_candidate_count"),
        "model_added_owner_merge_count": len(projection_report["model_added_owner_merges"]),
        "rejection_reason_counts": projection_report.get("rejection_reason_counts", {}),
        "generated_tex_changed": generated_tex_changed,
        "deterministic_missing_merge_rate": det_audit.get("missing_merge_rate_among_covered"),
        "learned_missing_merge_rate": learned_audit.get("missing_merge_rate_among_covered"),
        "deterministic_wrong_merge_rate": det_audit.get("wrong_merge_rate_among_generated"),
        "learned_wrong_merge_rate": learned_audit.get("wrong_merge_rate_among_generated"),
        "deterministic_paragraph_count_delta": det_audit.get("paragraph_count_delta"),
        "learned_paragraph_count_delta": learned_audit.get("paragraph_count_delta"),
        "deterministic_source_coverage_rate": det_audit.get("source_coverage_rate"),
        "learned_source_coverage_rate": learned_audit.get("source_coverage_rate"),
        "deterministic_body_source_coverage_rate": det_audit.get("body_source_coverage_rate"),
        "learned_body_source_coverage_rate": learned_audit.get("body_source_coverage_rate"),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [
        "deterministic_missing_merge_rate",
        "learned_missing_merge_rate",
        "deterministic_wrong_merge_rate",
        "learned_wrong_merge_rate",
        "deterministic_source_coverage_rate",
        "learned_source_coverage_rate",
        "deterministic_body_source_coverage_rate",
        "learned_body_source_coverage_rate",
        "deterministic_paragraph_count_delta",
        "learned_paragraph_count_delta",
    ]
    aggregate: dict[str, Any] = {
        "generated_tex_changed_count": sum(1 for row in rows if row.get("generated_tex_changed")),
        "model_added_owner_merge_total": sum(int(row.get("model_added_owner_merge_count") or 0) for row in rows),
        "strict_overlay_candidate_total": sum(int(row.get("strict_overlay_candidate_count") or 0) for row in rows),
        "model_predicted_cross_owner_atomic_merge_total": sum(
            int(row.get("model_predicted_cross_owner_atomic_merge_count") or 0) for row in rows
        ),
    }
    for key in keys:
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        if vals:
            aggregate[f"mean_{key}"] = round(sum(vals) / len(vals), 6)
    if "mean_deterministic_missing_merge_rate" in aggregate and "mean_learned_missing_merge_rate" in aggregate:
        aggregate["delta_missing_merge_rate_learned_minus_deterministic"] = round(
            aggregate["mean_learned_missing_merge_rate"] - aggregate["mean_deterministic_missing_merge_rate"],
            6,
        )
    if "mean_deterministic_wrong_merge_rate" in aggregate and "mean_learned_wrong_merge_rate" in aggregate:
        aggregate["delta_wrong_merge_rate_learned_minus_deterministic"] = round(
            aggregate["mean_learned_wrong_merge_rate"] - aggregate["mean_deterministic_wrong_merge_rate"],
            6,
        )
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = __import__("csv").DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    agg = summary.get("aggregate") or {}
    rows = summary.get("rows") or []
    lines = [
        "# V8 Atomic Prediction Projection Evaluation",
        "",
        "## Status",
        f"- doc_count: {summary.get('doc_count')}",
        f"- error_count: {summary.get('error_count')}",
        f"- policy: {summary.get('policy')}",
        f"- overlay_policy: {summary.get('overlay_policy')}",
        f"- variant: {summary.get('variant')}",
        f"- threshold: {summary.get('threshold')}",
        f"- family_thresholds: {summary.get('family_thresholds')}",
        f"- max_added_owner_merges_per_doc: {summary.get('max_added_owner_merges_per_doc')}",
        "- training: No",
        "- MinerU rerun: No",
        "- graph rebuild/relabel: No",
        "",
        "## Aggregate",
        f"- generated_tex_changed_count: {agg.get('generated_tex_changed_count')}",
        f"- model_added_owner_merge_total: {agg.get('model_added_owner_merge_total')}",
        f"- strict_overlay_candidate_total: {agg.get('strict_overlay_candidate_total')}",
        f"- mean deterministic missing-merge rate: {agg.get('mean_deterministic_missing_merge_rate')}",
        f"- mean learned missing-merge rate: {agg.get('mean_learned_missing_merge_rate')}",
        f"- delta missing learned-deterministic: {agg.get('delta_missing_merge_rate_learned_minus_deterministic')}",
        f"- mean deterministic wrong-merge rate: {agg.get('mean_deterministic_wrong_merge_rate')}",
        f"- mean learned wrong-merge rate: {agg.get('mean_learned_wrong_merge_rate')}",
        f"- delta wrong learned-deterministic: {agg.get('delta_wrong_merge_rate_learned_minus_deterministic')}",
        "",
        "## Per-doc",
        "| doc_id | added owner merges | tex changed | det missing | learned missing | det wrong | learned wrong |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {doc_id} | {model_added_owner_merge_count} | {generated_tex_changed} | "
            "{deterministic_missing_merge_rate} | {learned_missing_merge_rate} | "
            "{deterministic_wrong_merge_rate} | {learned_wrong_merge_rate} |".format(**row)
        )
    if summary.get("errors"):
        lines.extend(["", "## Errors"])
        for error in summary["errors"]:
            lines.append(f"- {error.get('doc_id')}: {error.get('error')} {error.get('message')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_tex_for_hash(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())


def _is_open_ended(text: str) -> bool:
    value = _normalize_space(text)
    if not value:
        return False
    if value.endswith("-"):
        return True
    if _tail_ends_soft_punctuation(value):
        return True
    if _tail_abbrev_like(value):
        return True
    return not bool(__import__("re").search(r"[.!?。！？;；:]$|[.!?。！？][\"')\]}]*$", value))


def _is_merge_tail_continuable(text: str) -> bool:
    value = _normalize_space(text)
    if not value:
        return False
    if value.endswith("-"):
        return True
    if _tail_ends_soft_punctuation(value):
        return True
    if _tail_abbrev_like(value):
        return True
    return not _tail_ends_hard_terminal(value)


def _tail_ends_hard_terminal(text: str) -> bool:
    value = _normalize_space(text)
    if not value:
        return False
    if _tail_abbrev_like(value):
        return False
    return bool(re.search(r"[.!?。！？][\"')\]}]*$", value))


def _tail_ends_soft_punctuation(text: str) -> bool:
    return bool(re.search(r"[,，;；:：][\"')\]}]*$", _normalize_space(text)))


def _tail_abbrev_like(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)(?:\b(?:et al|e\.g|i\.e|fig|figs|sec|secs|eq|eqs|tab|tabs|alg|algs|no|nos|vs|cf|dr|mr|mrs|ms|prof|inc|ltd|corp)\.)$",
            _normalize_space(text).rstrip(),
        )
    )


def _starts_lowercase(text: str) -> bool:
    return bool(__import__("re").match(r"^[a-zα-ω]", _normalize_space(text).lstrip("([{")))


def _starts_parenthetical(text: str) -> bool:
    return _normalize_space(text).lstrip().startswith("(")


def _starts_continuation_word(text: str) -> bool:
    value = _normalize_space(text).lstrip("([{").casefold()
    return bool(
        __import__("re").match(
            r"^(and|or|but|for|to|of|in|on|with|which|where|while|because|that|than|from|as|by)\b",
            value,
        )
    )


def _style_compatible(src: dict[str, Any], dst: dict[str, Any]) -> bool:
    src_meta = src.get("metadata") if isinstance(src.get("metadata"), dict) else {}
    dst_meta = dst.get("metadata") if isinstance(dst.get("metadata"), dict) else {}
    src_font = _safe_float(src_meta.get("font_size"), 0.0)
    dst_font = _safe_float(dst_meta.get("font_size"), 0.0)
    if src_font > 0 and dst_font > 0 and abs(src_font - dst_font) > max(1.0, 0.14 * max(src_font, dst_font)):
        return False
    src_bold = _safe_float(src_meta.get("bold_ratio"), 0.0)
    dst_bold = _safe_float(dst_meta.get("bold_ratio"), 0.0)
    if src_bold > 0 or dst_bold > 0:
        if abs(src_bold - dst_bold) > 0.75:
            return False
    return True


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def resolve_device(value: str, *, torch: Any) -> Any:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    raise SystemExit(main())
