#!/usr/bin/env python3
"""Project middle-fragment MERGE predictions back to full-v7 logical owners.

This tool is intentionally audit-only.  It runs a fragment-level MERGE model on
the middle-derived pseudo-v7 graphs, groups predicted fragment continuations by
their original v7 logical owner, and optionally writes patched full-v7 JSON
files for downstream visibility checks.  It does not mutate the source v7 JSON,
graph.pt files, labels, decoder, or renderer.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ir.serialization import write_json  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402


MERGE = 0
PARENT_CHILD = 1
NONE = 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--doc-ids", nargs="*", default=[])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--merge-decision", choices=["argmax", "threshold"], default="argmax")
    parser.add_argument("--merge-threshold", type=float, default=0.50)
    parser.add_argument(
        "--patch-policy",
        choices=["predicted_complete_owner", "gold_middle_owner"],
        default="predicted_complete_owner",
        help=(
            "predicted_complete_owner patches a v7 owner only when all adjacent "
            "strong fragments under that owner are predicted as connected. "
            "gold_middle_owner is diagnostic upper-bound text patching."
        ),
    )
    parser.add_argument("--write-patched-content", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-name", default="middlefrag_projection_to_v7_20260523")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_rows(args.fragment_manifest)
    rows = stable_filter_rows(rows, doc_ids=args.doc_ids, limit=args.limit)
    if not rows:
        raise ValueError(f"No fragment rows selected from {args.fragment_manifest}")

    full_rows = load_full_manifest_map(args.full_manifest) if args.full_manifest else {}
    device = resolve_device(args.device, torch=torch)
    model, checkpoint_meta = load_model(args.checkpoint, device=device, torch=torch)

    summary_rows: list[dict[str, Any]] = []
    patched_manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        doc_id = row_doc_id(row)
        try:
            result = process_one_doc(
                row,
                full_row=full_rows.get(doc_id),
                model=model,
                checkpoint_meta=checkpoint_meta,
                device=device,
                torch=torch,
                args=args,
            )
            summary_rows.append(result["summary"])
            if result.get("patched_manifest_row"):
                patched_manifest_rows.append(result["patched_manifest_row"])
            print(
                f"[{index}/{len(rows)}] {doc_id} "
                f"merge_f1={result['summary'].get('fragment_merge_f1')} "
                f"patches={result['summary'].get('owner_text_patch_count')} "
                f"cross_owner={result['summary'].get('predicted_cross_owner_merge_edges')}"
            )
        except Exception as exc:  # noqa: BLE001 - audit batch should keep going.
            failure = {"doc_id": doc_id, "error": repr(exc)}
            failures.append(failure)
            print(f"[{index}/{len(rows)}] {doc_id} ERROR {exc!r}")

    write_summary(args, summary_rows, patched_manifest_rows, failures, checkpoint_meta)
    return 0 if not failures else 1


def process_one_doc(
    row: dict[str, Any],
    *,
    full_row: dict[str, Any] | None,
    model: Any,
    checkpoint_meta: dict[str, Any],
    device: Any,
    torch: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    doc_id = row_doc_id(row)
    graph_path = Path(str(row["graph_path"]))
    view_path = Path(str(row["middle_fragment_view"]))
    graph = torch.load(graph_path, map_location=device, weights_only=False)
    graph = graph.to(device)
    with torch.no_grad():
        logits = model(graph).detach().cpu()
        probs = torch.softmax(logits, dim=-1)

    edge_index = graph.edge_index.detach().cpu().long()
    y = getattr(graph, "y", None)
    y_cpu = y.detach().cpu().long() if y is not None else None
    fragment_ids = graph_fragment_ids(graph)
    fragment_to_gnn = {fragment_id: index for index, fragment_id in enumerate(fragment_ids)}
    edge_lookup = edge_position_lookup(edge_index)
    predicted_positions = predicted_merge_positions(
        probs,
        decision=args.merge_decision,
        threshold=args.merge_threshold,
    )
    gold_positions = set(torch.nonzero(y_cpu == MERGE, as_tuple=False).flatten().tolist()) if y_cpu is not None else set()
    merge_metrics = binary_edge_metrics(predicted_positions, gold_positions, edge_count=int(edge_index.shape[1]))

    view = load_json(view_path)
    fragments = {str(fragment["fragment_id"]): fragment for fragment in view.get("fragments", [])}
    owner_groups = group_fragments_by_owner(view.get("fragments", []), allowed_fragment_ids=set(fragment_to_gnn))
    predicted_pair_keys = edge_pair_keys(edge_index, predicted_positions, fragment_ids)
    gold_pair_keys = edge_pair_keys(edge_index, gold_positions, fragment_ids)

    owner_reports, projection_totals, cross_owner_pairs = project_predictions_to_owners(
        owner_groups,
        fragments=fragments,
        predicted_pair_keys=predicted_pair_keys,
        gold_pair_keys=gold_pair_keys,
    )

    source_v7_path = source_v7_path_for(row, view, full_row)
    source_v7_payload = load_json(source_v7_path) if source_v7_path and source_v7_path.exists() else None
    patch_payload: dict[str, Any] | None = None
    patch_records: list[dict[str, Any]] = []
    if args.write_patched_content and isinstance(source_v7_payload, dict):
        patch_payload, patch_records = patch_v7_payload(
            source_v7_payload,
            owner_reports,
            patch_policy=args.patch_policy,
            doc_id=doc_id,
        )

    doc_dir = args.output_dir / "per_doc" / safe_filename(doc_id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    projection_path = doc_dir / "fragment_to_v7_projection.json"
    write_json(
        projection_path,
        {
            "schema_version": "middle_fragment_to_v7_projection_v1",
            "doc_id": doc_id,
            "graph_path": str(graph_path),
            "middle_fragment_view": str(view_path),
            "source_v7_json": str(source_v7_path) if source_v7_path else None,
            "checkpoint": str(args.checkpoint),
            "checkpoint_meta": checkpoint_meta,
            "merge_decision": args.merge_decision,
            "merge_threshold": args.merge_threshold,
            "fragment_edge_metrics": merge_metrics,
            "projection_totals": projection_totals,
            "cross_owner_predicted_pairs": cross_owner_pairs,
            "owner_reports": owner_reports,
            "patch_policy": args.patch_policy,
            "patch_records": patch_records,
        },
    )

    patched_content_path: Path | None = None
    if patch_payload is not None:
        patched_content_dir = args.output_dir / "patched_content_v7"
        patched_content_dir.mkdir(parents=True, exist_ok=True)
        patched_content_path = patched_content_dir / f"{safe_filename(doc_id)}_content_list_v7_styles_middlefrag_projected.json"
        write_json(patched_content_path, patch_payload)

    patched_manifest_row = None
    if patched_content_path and full_row:
        patched_manifest_row = dict(full_row)
        patched_manifest_row["document_id"] = doc_id
        patched_manifest_row["content_json"] = str(patched_content_path.resolve())
        patched_manifest_row["middle_fragment_projection"] = str(projection_path.resolve())
        patched_manifest_row["middle_fragment_projection_source_content_json"] = str(source_v7_path) if source_v7_path else None
        patched_manifest_row["middle_fragment_projection_patch_count"] = len(patch_records)

    summary = {
        "doc_id": doc_id,
        "graph_nodes": int(getattr(graph, "num_nodes", 0) or 0),
        "graph_edges": int(edge_index.shape[1]),
        "fragment_merge_tp": merge_metrics["tp"],
        "fragment_merge_fp": merge_metrics["fp"],
        "fragment_merge_fn": merge_metrics["fn"],
        "fragment_merge_precision": merge_metrics["precision"],
        "fragment_merge_recall": merge_metrics["recall"],
        "fragment_merge_f1": merge_metrics["f1"],
        **projection_totals,
        "owner_text_patch_count": len(patch_records),
        "patched_content_json": str(patched_content_path) if patched_content_path else None,
        "projection_report": str(projection_path),
    }
    return {"summary": summary, "patched_manifest_row": patched_manifest_row}


def load_model(checkpoint_path: Path, *, device: Any, torch: Any) -> tuple[Any, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_config = config if isinstance(config, EdgeGATConfig) else EdgeGATConfig()
    model = EdgeRelationGAT(model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    meta = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_type": checkpoint.get("checkpoint_type") if isinstance(checkpoint, dict) else None,
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "edge_dim": int(getattr(model_config, "edge_dim", 0)),
        "prediction_architecture": str(getattr(model_config, "prediction_architecture", "")),
        "merge_gate_mode": str(getattr(model_config, "merge_gate_mode", "")),
    }
    return model, meta


def graph_fragment_ids(graph: Any) -> list[str]:
    values = getattr(graph, "gnn_to_v7_id", None) or getattr(graph, "gnn_to_v7_ids", None)
    if values is None:
        raise ValueError("graph missing gnn_to_v7_id(s) bridge")
    result: list[str] = []
    for index, value in enumerate(values):
        if isinstance(value, (list, tuple)):
            result.append(str(value[0]) if value else f"gnn_{index}")
        else:
            result.append(str(value))
    return result


def predicted_merge_positions(probs: Any, *, decision: str, threshold: float) -> set[int]:
    if decision == "argmax":
        return set(int(index) for index in (probs.argmax(dim=-1) == MERGE).nonzero(as_tuple=False).flatten().tolist())
    return set(int(index) for index in (probs[:, MERGE] >= float(threshold)).nonzero(as_tuple=False).flatten().tolist())


def edge_position_lookup(edge_index: Any) -> dict[tuple[int, int], list[int]]:
    lookup: dict[tuple[int, int], list[int]] = defaultdict(list)
    for pos in range(int(edge_index.shape[1])):
        lookup[(int(edge_index[0, pos]), int(edge_index[1, pos]))].append(pos)
    return lookup


def edge_pair_keys(edge_index: Any, positions: set[int], fragment_ids: list[str]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for pos in positions:
        src = fragment_ids[int(edge_index[0, pos])]
        dst = fragment_ids[int(edge_index[1, pos])]
        keys.add((src, dst))
        keys.add((dst, src))
    return keys


def binary_edge_metrics(predicted: set[int], gold: set[int], *, edge_count: int) -> dict[str, Any]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    tn = max(0, int(edge_count) - tp - fp - fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
    }


def group_fragments_by_owner(fragments: list[dict[str, Any]], *, allowed_fragment_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fragment in fragments:
        fragment_id = str(fragment.get("fragment_id") or "")
        if fragment_id not in allowed_fragment_ids:
            continue
        owner = str(fragment.get("owner_middle_block_id") or "")
        if not owner:
            continue
        groups[owner].append(fragment)
    for owner, items in groups.items():
        groups[owner] = sorted(items, key=lambda item: int(item.get("order_in_owner") or 0))
    return groups


def project_predictions_to_owners(
    owner_groups: dict[str, list[dict[str, Any]]],
    *,
    fragments: dict[str, dict[str, Any]],
    predicted_pair_keys: set[tuple[str, str]],
    gold_pair_keys: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    owner_reports: list[dict[str, Any]] = []
    totals = Counter()
    cross_owner_predicted: dict[tuple[str, str], dict[str, Any]] = {}
    for owner_id, group in sorted(owner_groups.items()):
        totals["owners_total"] += 1
        if len(group) > 1:
            totals["owners_with_multiple_graph_fragments"] += 1
        adjacent = [(str(left["fragment_id"]), str(right["fragment_id"])) for left, right in zip(group, group[1:])]
        if not adjacent:
            continue
        predicted_adjacent = [pair for pair in adjacent if pair in predicted_pair_keys or (pair[1], pair[0]) in predicted_pair_keys]
        gold_adjacent = [pair for pair in adjacent if pair in gold_pair_keys or (pair[1], pair[0]) in gold_pair_keys]
        if len(predicted_adjacent) == len(adjacent):
            totals["owners_all_adjacent_predicted"] += 1
        elif predicted_adjacent:
            totals["owners_partial_adjacent_predicted"] += 1
        else:
            totals["owners_no_adjacent_predicted"] += 1

        components = connected_components([str(item["fragment_id"]) for item in group], predicted_adjacent)
        v7_ids = owner_v7_ids_for_group(group)
        owner_reports.append(
            {
                "owner_middle_block_id": owner_id,
                "owner_v7_ids": v7_ids,
                "merge_channel": dominant_value(str(item.get("merge_channel") or "") for item in group),
                "fragment_count": len(group),
                "adjacent_edge_count": len(adjacent),
                "predicted_adjacent_edge_count": len(predicted_adjacent),
                "gold_adjacent_edge_count": len(gold_adjacent),
                "all_adjacent_predicted": len(predicted_adjacent) == len(adjacent),
                "predicted_components": [
                    {
                        "fragment_ids": component,
                        "text": join_fragment_text(fragments[fid] for fid in component if fid in fragments),
                    }
                    for component in components
                ],
                "gold_middle_text": join_fragment_text(group),
            }
        )
    # Cross-owner predictions are diagnostic; labels do not request these, but a
    # model may still emit them at inference.
    for src, dst in predicted_pair_keys:
        src_owner = str(fragments.get(src, {}).get("owner_middle_block_id") or "")
        dst_owner = str(fragments.get(dst, {}).get("owner_middle_block_id") or "")
        if src_owner and dst_owner and src_owner != dst_owner:
            key = tuple(sorted((src_owner, dst_owner)))
            if key not in cross_owner_predicted:
                src_fragment = fragments.get(src, {})
                dst_fragment = fragments.get(dst, {})
                cross_owner_predicted[key] = {
                    "owner_middle_block_ids": list(key),
                    "src_fragment_id": src,
                    "dst_fragment_id": dst,
                    "src_owner_v7_ids": list(src_fragment.get("owner_v7_ids") or []),
                    "dst_owner_v7_ids": list(dst_fragment.get("owner_v7_ids") or []),
                    "src_text_preview": str(src_fragment.get("text") or "")[:180],
                    "dst_text_preview": str(dst_fragment.get("text") or "")[:180],
                }
    totals["predicted_cross_owner_merge_edges"] = len(cross_owner_predicted)
    totals["predicted_internal_owner_adjacent_merge_edges"] = sum(
        int(row["predicted_adjacent_edge_count"]) for row in owner_reports
    )
    return owner_reports, dict(totals), list(cross_owner_predicted.values())


def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    order = {node: index for index, node in enumerate(nodes)}
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for src, dst in edges:
        if src in adjacency and dst in adjacency:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        queue: deque[str] = deque([node])
        seen.add(node)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for nxt in sorted(adjacency[current], key=lambda item: order[item]):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(sorted(component, key=lambda item: order[item]))
    return components


def patch_v7_payload(
    payload: dict[str, Any],
    owner_reports: list[dict[str, Any]],
    *,
    patch_policy: str,
    doc_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    patched = json.loads(json.dumps(payload))
    items = patched.get("items") if isinstance(patched.get("items"), list) else []
    id_to_item: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("node_id", "id", "block_id"):
            value = str(item.get(key) or "")
            if value:
                id_to_item[value] = item
    patch_records: list[dict[str, Any]] = []
    for report in owner_reports:
        if not report.get("owner_v7_ids"):
            continue
        if patch_policy == "predicted_complete_owner" and not report.get("all_adjacent_predicted"):
            continue
        replacement = str(report.get("gold_middle_text") or "").strip()
        if not replacement:
            continue
        for v7_id in report.get("owner_v7_ids") or []:
            item = id_to_item.get(str(v7_id))
            if not item:
                continue
            old = str(item.get("text") or item.get("content") or "").strip()
            if normalize_text(old) == normalize_text(replacement):
                continue
            if "text" in item or "content" not in item:
                item["text"] = replacement
            else:
                item["content"] = replacement
            item.setdefault("metadata", {})
            if isinstance(item["metadata"], dict):
                item["metadata"]["middle_fragment_projection_patch"] = True
                item["metadata"]["middle_fragment_projection_policy"] = patch_policy
            patch_records.append(
                {
                    "doc_id": doc_id,
                    "v7_id": str(v7_id),
                    "owner_middle_block_id": report.get("owner_middle_block_id"),
                    "old_text_preview": old[:240],
                    "new_text_preview": replacement[:240],
                    "old_len": len(old),
                    "new_len": len(replacement),
                    "fragment_count": report.get("fragment_count"),
                    "predicted_adjacent_edge_count": report.get("predicted_adjacent_edge_count"),
                }
            )
    patched.setdefault("metadata", {})
    if isinstance(patched["metadata"], dict):
        patched["metadata"]["middle_fragment_projection"] = {
            "schema_version": "middle_fragment_projection_patch_v1",
            "patch_policy": patch_policy,
            "patch_count": len(patch_records),
        }
    return patched, patch_records


def source_v7_path_for(row: dict[str, Any], view: dict[str, Any], full_row: dict[str, Any] | None) -> Path | None:
    for value in (
        view.get("v7_json"),
        row.get("v7_json"),
        row.get("source_v7_json"),
        full_row.get("content_json") if full_row else None,
    ):
        if value:
            return Path(str(value))
    return None


def owner_v7_ids_for_group(group: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for item in group:
        for v7_id in item.get("owner_v7_ids") or []:
            v7_id = str(v7_id)
            if v7_id and v7_id not in values:
                values.append(v7_id)
    return values


def join_fragment_text(items: Any) -> str:
    parts = [str(item.get("text") or "").strip() for item in items if str(item.get("text") or "").strip()]
    text = " ".join(parts)
    text = re.sub(r"([\\w])\\-\\s+([a-z])", r"\\1\\2", text)
    return normalize_space(text)


def write_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    patched_manifest_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    checkpoint_meta: dict[str, Any],
) -> None:
    summary = {
        "schema_version": "middle_fragment_to_v7_projection_summary_v1",
        "run_name": args.run_name,
        "fragment_manifest": str(args.fragment_manifest),
        "full_manifest": str(args.full_manifest) if args.full_manifest else None,
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": checkpoint_meta,
        "merge_decision": args.merge_decision,
        "merge_threshold": args.merge_threshold,
        "patch_policy": args.patch_policy,
        "doc_count": len(rows),
        "failure_count": len(failures),
        "aggregate": aggregate_summary(rows),
        "documents": rows,
        "failures": failures,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "summary.csv", rows)
    if patched_manifest_rows:
        patched_manifest = {
            "schema_version": "middle_fragment_projected_full_v7_manifest_v1",
            "source_fragment_manifest": str(args.fragment_manifest),
            "source_full_manifest": str(args.full_manifest) if args.full_manifest else None,
            "run_name": args.run_name,
            "documents": patched_manifest_rows,
        }
        write_json(args.output_dir / "projected_full_v7_manifest.json", patched_manifest)
    write_report(args.output_dir / "MIDDLEFRAG_TO_V7_PROJECTION_REPORT.md", summary, bool(patched_manifest_rows))


def aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "fragment_merge_tp",
        "fragment_merge_fp",
        "fragment_merge_fn",
        "predicted_internal_owner_adjacent_merge_edges",
        "predicted_cross_owner_merge_edges",
        "owners_total",
        "owners_with_multiple_graph_fragments",
        "owners_all_adjacent_predicted",
        "owners_partial_adjacent_predicted",
        "owners_no_adjacent_predicted",
        "owner_text_patch_count",
    ]
    totals = {key: sum(int(row.get(key) or 0) for row in rows) for key in keys}
    precision = safe_div(totals["fragment_merge_tp"], totals["fragment_merge_tp"] + totals["fragment_merge_fp"])
    recall = safe_div(totals["fragment_merge_tp"], totals["fragment_merge_tp"] + totals["fragment_merge_fn"])
    totals.update(
        {
            "fragment_merge_precision": precision,
            "fragment_merge_recall": recall,
            "fragment_merge_f1": safe_div(2 * precision * recall, precision + recall),
        }
    )
    return totals


def write_report(path: Path, summary: dict[str, Any], has_patched_manifest: bool) -> None:
    aggregate = summary.get("aggregate", {})
    lines = [
        "# Middle-Fragment to V7 Projection Report",
        "",
        "## Status",
        f"- docs: `{summary.get('doc_count')}`",
        f"- failures: `{summary.get('failure_count')}`",
        f"- checkpoint: `{summary.get('checkpoint')}`",
        f"- decision: `{summary.get('merge_decision')}`",
        f"- patch policy: `{summary.get('patch_policy')}`",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in (
        "fragment_merge_precision",
        "fragment_merge_recall",
        "fragment_merge_f1",
        "fragment_merge_tp",
        "fragment_merge_fp",
        "fragment_merge_fn",
        "owners_with_multiple_graph_fragments",
        "owners_all_adjacent_predicted",
        "owners_partial_adjacent_predicted",
        "predicted_internal_owner_adjacent_merge_edges",
        "predicted_cross_owner_merge_edges",
        "owner_text_patch_count",
    ):
        value = aggregate.get(key)
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- `predicted_internal_owner_adjacent_merge_edges` means the fragment model recovers lines/spans inside an existing v7 logical owner.",
        "- `predicted_cross_owner_merge_edges` is the part that could become a new full-v7 owner-level MERGE edge.",
        "- `owner_text_patch_count` estimates whether rebuilding owner text from predicted complete fragment groups would change full-v7 text and therefore potentially change generated.tex.",
        "",
        "## Artifacts",
        f"- per-doc projection: `{path.parent / 'per_doc'}`",
        f"- patched content JSONs: `{path.parent / 'patched_content_v7'}`",
    ]
    if has_patched_manifest:
        lines.append(f"- projected full-v7 manifest: `{path.parent / 'projected_full_v7_manifest.json'}`")
    else:
        lines.append("- projected full-v7 manifest: not written")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = (
            payload.get("documents")
            or payload.get("items")
            or payload.get("records")
            or payload.get("data")
            or []
        )
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def load_full_manifest_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = load_manifest_rows(path)
    return {row_doc_id(row): row for row in rows if row_doc_id(row)}


def stable_filter_rows(rows: list[dict[str, Any]], *, doc_ids: list[str], limit: int | None) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: row_doc_id(row))
    wanted = {str(doc_id) for doc_id in doc_ids if str(doc_id)}
    if wanted:
        selected = [row for row in selected if row_doc_id(row) in wanted]
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected


def row_doc_id(row: dict[str, Any]) -> str:
    for key in ("document_id", "doc_id", "id", "paper_id", "arxiv_id"):
        value = str(row.get(key) or "")
        if value:
            return value
    return ""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(value: str, *, torch: Any) -> Any:
    if value == "cuda":
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dominant_value(values: Any) -> str:
    counter = Counter(value for value in values if value)
    return counter.most_common(1)[0][0] if counter else ""


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\W+", "", str(text or "").casefold())


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "doc"


if __name__ == "__main__":
    raise SystemExit(main())
