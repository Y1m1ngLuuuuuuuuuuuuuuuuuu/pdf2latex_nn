#!/usr/bin/env python3
"""Audit missing same-TeX candidate edges and below-threshold MERGE positives.

This is read-only. It loads existing labeled graphs, an existing channel-aware
MERGE audit, and an optional checkpoint to attribute which MERGE labels or
candidate families are suppressed by the current decision threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.batch_visual_qa_inference import load_model, resolve_device  # noqa: E402


MERGE = 0
BODY_LIKE_CHANNELS = {"BODY_TEXT", "LIST_ITEM", "REFERENCE_ITEM"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def doc_id_from_graph(path: Path) -> str:
    name = path.name
    for suffix in ("_v7_relabel_labeled_graph.pt", "_labeled_graph.pt", "_graph.pt", ".pt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def graph_paths(graph_dir: Path, *, limit: int | None, doc_ids: set[str] | None) -> list[Path]:
    paths = sorted(graph_dir.glob("*.pt"))
    selected = []
    for path in paths:
        doc_id = doc_id_from_graph(path)
        if doc_ids and doc_id not in doc_ids:
            continue
        selected.append(path)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def edge_audit_map(channel_audit_dir: Path, doc_id: str) -> dict[int, dict[str, Any]]:
    path = channel_audit_dir / "per_doc" / doc_id / "merge_candidate_audit.json"
    if not path.exists():
        return {}
    payload = load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else []
    by_edge: dict[int, dict[str, Any]] = {}
    for record in records or []:
        try:
            by_edge[int(record["edge_pos"])] = record
        except Exception:
            continue
    return by_edge


def probability_bucket(value: float) -> str:
    if value < 0.05:
        return "0.00-0.05"
    if value < 0.10:
        return "0.05-0.10"
    if value < 0.20:
        return "0.10-0.20"
    if value < 0.30:
        return "0.20-0.30"
    if value < 0.37:
        return "0.30-0.37"
    if value < 0.50:
        return "0.37-0.50"
    if value < 0.75:
        return "0.50-0.75"
    return "0.75-1.00"


def parse_missing_channel_key(key: str) -> tuple[str, str, str]:
    reason, _, channels = str(key).partition(":")
    src, _, dst = channels.partition("->")
    return reason or "UNKNOWN", src or "UNKNOWN", dst or "UNKNOWN"


def body_like_missing_count(channel_stats: dict[str, int]) -> int:
    total = 0
    for key, value in channel_stats.items():
        _, src, dst = parse_missing_channel_key(key)
        if src in BODY_LIKE_CHANNELS and dst in BODY_LIKE_CHANNELS:
            total += int(value)
    return total


def model_probs(model: Any, graph: Any, *, device: Any, torch: Any) -> Any:
    with torch.no_grad():
        data = graph.to(device)
        logits = model(data)
        return torch.softmax(logits.detach().cpu(), dim=-1)


def audit_one(
    graph_path: Path,
    *,
    channel_audit_dir: Path,
    model: Any | None,
    device: Any | None,
    tau_merge: float,
    torch: Any,
) -> dict[str, Any]:
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    doc_id = doc_id_from_graph(graph_path)
    audit_by_edge = edge_audit_map(channel_audit_dir, doc_id)
    schema = getattr(graph, "alignment_schema", {}) or {}
    channel_payload = schema.get("channel_supervision_audit") or {}
    missing_stats = {
        str(key): int(value)
        for key, value in (channel_payload.get("missing_same_tex_candidate_edge_stats") or {}).items()
    }
    missing_channel_stats = {
        str(key): int(value)
        for key, value in (channel_payload.get("missing_same_tex_candidate_edge_channel_stats") or {}).items()
    }
    missing_examples = channel_payload.get("missing_same_tex_candidate_edge_examples") or []
    body_like_missing = body_like_missing_count(missing_channel_stats)

    below_rows: list[dict[str, Any]] = []
    below_family_counts: Counter[str] = Counter()
    below_strength_counts: Counter[str] = Counter()
    below_prob_buckets: Counter[str] = Counter()
    old_merge_count = 0
    strong_old_merge_count = 0
    old_merge_below_count = 0
    strong_old_merge_below_count = 0
    merge_prob_values: list[float] = []
    probs = None
    if model is not None:
        probs = model_probs(model, graph, device=device, torch=torch)
    labels = graph.y.detach().cpu() if hasattr(graph.y, "detach") else torch.as_tensor(graph.y)
    edge_index = graph.edge_index.detach().cpu()
    for edge_pos in range(int(edge_index.shape[1])):
        label = int(labels[edge_pos].item())
        if label != MERGE:
            continue
        old_merge_count += 1
        record = audit_by_edge.get(edge_pos, {})
        family = str(record.get("relation_family") or "UNKNOWN")
        strength = str(record.get("label_strength") or "UNKNOWN")
        if strength == "strong":
            strong_old_merge_count += 1
        merge_prob = None
        if probs is not None:
            merge_prob = float(probs[edge_pos, MERGE].item())
            merge_prob_values.append(merge_prob)
            if merge_prob < tau_merge:
                old_merge_below_count += 1
                if strength == "strong":
                    strong_old_merge_below_count += 1
                below_family_counts[family] += 1
                below_strength_counts[strength] += 1
                below_prob_buckets[probability_bucket(merge_prob)] += 1
                if len(below_rows) < 80:
                    below_rows.append(
                        {
                            "edge_pos": edge_pos,
                            "src": int(edge_index[0, edge_pos].item()),
                            "dst": int(edge_index[1, edge_pos].item()),
                            "merge_prob": merge_prob,
                            "relation_family": family,
                            "label_strength": strength,
                            "src_channel": record.get("src_channel"),
                            "dst_channel": record.get("dst_channel"),
                            "src_text_preview": record.get("src_text_preview"),
                            "dst_text_preview": record.get("dst_text_preview"),
                        }
                    )

    return {
        "schema_version": "missing_below_threshold_merge_doc_audit_v1",
        "doc_id": doc_id,
        "graph_path": str(graph_path),
        "old_merge_count": old_merge_count,
        "strong_old_merge_count": strong_old_merge_count,
        "old_merge_below_threshold_count": old_merge_below_count,
        "strong_old_merge_below_threshold_count": strong_old_merge_below_count,
        "tau_merge": tau_merge,
        "merge_prob_min": min(merge_prob_values) if merge_prob_values else None,
        "merge_prob_mean": (sum(merge_prob_values) / len(merge_prob_values)) if merge_prob_values else None,
        "merge_prob_max": max(merge_prob_values) if merge_prob_values else None,
        "below_threshold_family_counts": dict(sorted(below_family_counts.items())),
        "below_threshold_strength_counts": dict(sorted(below_strength_counts.items())),
        "below_threshold_prob_buckets": dict(sorted(below_prob_buckets.items())),
        "below_threshold_examples": below_rows,
        "missing_candidate_stats": missing_stats,
        "missing_candidate_channel_stats": missing_channel_stats,
        "missing_candidate_total": int(sum(missing_stats.values())),
        "missing_candidate_body_like_total": int(body_like_missing),
        "missing_candidate_examples": missing_examples[:80],
    }


def ratio(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "doc_id",
        "old_merge_count",
        "strong_old_merge_count",
        "old_merge_below_threshold_count",
        "old_merge_below_threshold_ratio",
        "strong_old_merge_below_threshold_count",
        "strong_old_merge_below_threshold_ratio",
        "missing_candidate_total",
        "missing_candidate_body_like_total",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(path: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Missing-Candidate / Below-Threshold MERGE Audit",
        "",
        "## Scope",
        "",
        f"- Docs audited: {summary['doc_count']}",
        f"- Checkpoint: `{summary.get('checkpoint') or 'not used'}`",
        f"- tau_merge: {summary['tau_merge']}",
        "- Read-only audit: no graph labels were rewritten; no training, relabel, rebuild, MinerU, or E2E was run.",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| old MERGE edges | {agg['old_merge_count']} |",
        f"| old MERGE below threshold | {agg['old_merge_below_threshold_count']} ({agg['old_merge_below_threshold_ratio']:.4f}) |",
        f"| strong old MERGE edges | {agg['strong_old_merge_count']} |",
        f"| strong old MERGE below threshold | {agg['strong_old_merge_below_threshold_count']} ({agg['strong_old_merge_below_threshold_ratio']:.4f}) |",
        f"| missing same-TeX candidate edges | {agg['missing_candidate_total']} |",
        f"| missing body/list/reference candidate edges | {agg['missing_candidate_body_like_total']} |",
        "",
        "## Below-Threshold Families",
        "",
        "| family | count |",
        "| --- | ---: |",
    ]
    for family, count in sorted(agg["below_threshold_family_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{family}` | {count} |")
    lines.extend(["", "## Missing Candidate Reasons", "", "| reason | count |", "| --- | ---: |"])
    for reason, count in sorted(agg["missing_candidate_stats"].items(), key=lambda item: (-item[1], item[0]))[:20]:
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## Missing Candidate Channel Reasons", "", "| reason/channel | count |", "| --- | ---: |"])
    for reason, count in sorted(agg["missing_candidate_channel_stats"].items(), key=lambda item: (-item[1], item[0]))[:30]:
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## Interpretation", "", summary["interpretation"], "", "## Artifacts", ""])
    lines.extend(
        [
            "- Per-doc audit: `per_doc/<doc_id>/missing_below_threshold_merge.json`",
            "- Aggregate JSON: `summary.json`",
            "- Aggregate CSV: `summary.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--channel-audit-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tau-merge", type=float, default=0.30)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--doc-ids", nargs="*")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    doc_ids = set(args.doc_ids) if args.doc_ids else None
    paths = graph_paths(args.graph_dir, limit=args.limit, doc_ids=doc_ids)
    if not paths:
        raise SystemExit(f"No graph .pt files found in {args.graph_dir}")

    device = resolve_device(args.device, torch=torch)
    model = None
    if args.checkpoint:
        model = load_model(args.checkpoint, device=device, torch=torch)

    rows: list[dict[str, Any]] = []
    missing_stats_total: Counter[str] = Counter()
    missing_channel_total: Counter[str] = Counter()
    below_family_total: Counter[str] = Counter()
    below_strength_total: Counter[str] = Counter()
    below_bucket_total: Counter[str] = Counter()

    for path in paths:
        payload = audit_one(
            path,
            channel_audit_dir=args.channel_audit_dir,
            model=model,
            device=device,
            tau_merge=args.tau_merge,
            torch=torch,
        )
        doc_id = str(payload["doc_id"])
        dump_json(args.output_dir / "per_doc" / doc_id / "missing_below_threshold_merge.json", payload)
        missing_stats_total.update(payload["missing_candidate_stats"])
        missing_channel_total.update(payload["missing_candidate_channel_stats"])
        below_family_total.update(payload["below_threshold_family_counts"])
        below_strength_total.update(payload["below_threshold_strength_counts"])
        below_bucket_total.update(payload["below_threshold_prob_buckets"])
        old_merge = int(payload["old_merge_count"])
        strong_old = int(payload["strong_old_merge_count"])
        rows.append(
            {
                "doc_id": doc_id,
                "old_merge_count": old_merge,
                "strong_old_merge_count": strong_old,
                "old_merge_below_threshold_count": int(payload["old_merge_below_threshold_count"]),
                "old_merge_below_threshold_ratio": ratio(int(payload["old_merge_below_threshold_count"]), old_merge),
                "strong_old_merge_below_threshold_count": int(payload["strong_old_merge_below_threshold_count"]),
                "strong_old_merge_below_threshold_ratio": ratio(
                    int(payload["strong_old_merge_below_threshold_count"]),
                    strong_old,
                ),
                "missing_candidate_total": int(payload["missing_candidate_total"]),
                "missing_candidate_body_like_total": int(payload["missing_candidate_body_like_total"]),
            }
        )

    old_merge_count = sum(int(row["old_merge_count"]) for row in rows)
    strong_old_merge_count = sum(int(row["strong_old_merge_count"]) for row in rows)
    below_count = sum(int(row["old_merge_below_threshold_count"]) for row in rows)
    strong_below_count = sum(int(row["strong_old_merge_below_threshold_count"]) for row in rows)
    missing_total = sum(int(row["missing_candidate_total"]) for row in rows)
    missing_body_like = sum(int(row["missing_candidate_body_like_total"]) for row in rows)
    if below_count > strong_below_count:
        interpretation = (
            "Most below-threshold pressure is not from strong body/list MERGE labels alone. Inspect family-specific "
            "probability calibration before lowering tau_merge."
        )
    elif strong_below_count:
        interpretation = (
            "A non-trivial part of strong MERGE supervision is below the current threshold; a small class-weight or "
            "threshold diagnostic is justified, but only with per-family precision checks."
        )
    elif missing_body_like:
        interpretation = (
            "The bigger issue is candidate-edge generation: body/list/reference same-TeX continuations are absent "
            "from the graph, so retraining alone cannot recover them."
        )
    else:
        interpretation = "No strong evidence that below-threshold or missing-candidate MERGE is the current main bottleneck."

    summary = {
        "schema_version": "missing_below_threshold_merge_audit_summary_v1",
        "doc_count": len(rows),
        "graph_dir": str(args.graph_dir),
        "channel_audit_dir": str(args.channel_audit_dir),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "tau_merge": args.tau_merge,
        "aggregate": {
            "old_merge_count": old_merge_count,
            "old_merge_below_threshold_count": below_count,
            "old_merge_below_threshold_ratio": ratio(below_count, old_merge_count),
            "strong_old_merge_count": strong_old_merge_count,
            "strong_old_merge_below_threshold_count": strong_below_count,
            "strong_old_merge_below_threshold_ratio": ratio(strong_below_count, strong_old_merge_count),
            "below_threshold_family_counts": dict(sorted(below_family_total.items())),
            "below_threshold_strength_counts": dict(sorted(below_strength_total.items())),
            "below_threshold_prob_buckets": dict(sorted(below_bucket_total.items())),
            "missing_candidate_total": missing_total,
            "missing_candidate_body_like_total": missing_body_like,
            "missing_candidate_stats": dict(sorted(missing_stats_total.items())),
            "missing_candidate_channel_stats": dict(sorted(missing_channel_total.items())),
        },
        "rows": rows,
        "interpretation": interpretation,
    }
    dump_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "summary.csv", rows)
    write_report(args.output_dir / "MISSING_CANDIDATE_BELOW_THRESHOLD_MERGE_AUDIT_REPORT.md", summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
