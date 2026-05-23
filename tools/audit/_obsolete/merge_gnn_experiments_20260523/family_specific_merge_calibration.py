#!/usr/bin/env python3
"""Read-only family-specific MERGE threshold calibration.

The script separates BODY/LIST continuation from REFERENCE continuation and
weak/exempt endpoint families. It does not train, relabel, rebuild, or write
graph tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.batch_visual_qa_inference import load_model, resolve_device  # noqa: E402


MERGE = 0
FAMILIES = {
    "body_list": {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION"},
    "reference": {"REFERENCE_ENDPOINT"},
    "weak_masked": {
        "WEAK_SAME_TEX",
        "FORMULA_LEAD_IN",
        "FORMULA_CONTEXT",
        "FLOAT_SKIP_CONTINUATION",
        "FLOAT_PROXY_ENDPOINT",
        "CAPTION_ENDPOINT",
        "MASKED_UNKNOWN",
    },
    "layout_mismatch": {"LAYOUT_SCOPE_MISMATCH"},
}


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
    selected: list[Path] = []
    for path in sorted(graph_dir.glob("*.pt")):
        doc_id = doc_id_from_graph(path)
        if doc_ids and doc_id not in doc_ids:
            continue
        selected.append(path)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def edge_audit_records(channel_audit_dir: Path, doc_id: str) -> dict[int, dict[str, Any]]:
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


def model_probs(model: Any, graph: Any, *, device: Any, torch: Any) -> Any:
    with torch.no_grad():
        data = graph.to(device)
        logits = model(data)
        return torch.softmax(logits.detach().cpu(), dim=-1)


def thresholds() -> list[float]:
    return [round(i / 100.0, 2) for i in range(0, 96)]


def pr_at(scores: list[tuple[float, bool]], threshold: float) -> dict[str, float | int]:
    tp = fp = fn = 0
    for prob, positive in scores:
        pred = prob >= threshold
        if pred and positive:
            tp += 1
        elif pred and not positive:
            fp += 1
        elif (not pred) and positive:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_positive": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def best_by_f1(scores: list[tuple[float, bool]]) -> dict[str, float | int]:
    rows = [pr_at(scores, tau) for tau in thresholds()]
    return max(rows, key=lambda row: (float(row["f1"]), float(row["precision"]), float(row["recall"])))


def best_for_precision(scores: list[tuple[float, bool]], floor: float) -> dict[str, float | int] | None:
    rows = [row for row in (pr_at(scores, tau) for tau in thresholds()) if float(row["precision"]) >= floor and row["tp"] > 0]
    if not rows:
        return None
    return max(rows, key=lambda row: (float(row["recall"]), float(row["f1"]), -float(row["threshold"])))


def family_group(family: str) -> str:
    for group, values in FAMILIES.items():
        if family in values:
            return group
    if family == "HARD_NEGATIVE":
        return "hard_negative"
    return "other"


def collect_rows(
    graph_dir: Path,
    channel_audit_dir: Path,
    checkpoint: Path,
    *,
    limit: int | None,
    doc_ids: set[str] | None,
    device_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    device = resolve_device(device_name, torch=torch)
    model = load_model(checkpoint, device=device, torch=torch)
    rows: list[dict[str, Any]] = []
    doc_stats: dict[str, Any] = {}

    for graph_path in graph_paths(graph_dir, limit=limit, doc_ids=doc_ids):
        doc_id = doc_id_from_graph(graph_path)
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        probs = model_probs(model, graph, device=device, torch=torch)
        labels = graph.y.detach().cpu() if hasattr(graph.y, "detach") else torch.as_tensor(graph.y)
        audit = edge_audit_records(channel_audit_dir, doc_id)
        counts = defaultdict(int)
        for edge_pos in range(int(graph.edge_index.shape[1])):
            label = int(labels[edge_pos].item())
            record = audit.get(edge_pos)
            family = str(record.get("relation_family") if record else "HARD_NEGATIVE")
            group = family_group(family)
            positive = label == MERGE
            prob = float(probs[edge_pos, MERGE].item())
            counts[f"{group}_candidate"] += 1
            if positive:
                counts[f"{group}_positive"] += 1
            rows.append(
                {
                    "doc_id": doc_id,
                    "edge_pos": edge_pos,
                    "merge_prob": prob,
                    "gold_merge": positive,
                    "relation_family": family,
                    "family_group": group,
                    "label_strength": str(record.get("label_strength") if record else "hard_negative"),
                    "src_channel": record.get("src_channel") if record else None,
                    "dst_channel": record.get("dst_channel") if record else None,
                    "src_text_preview": record.get("src_text_preview") if record else None,
                    "dst_text_preview": record.get("dst_text_preview") if record else None,
                }
            )
        doc_stats[doc_id] = dict(counts)
    return rows, doc_stats


def calibrate(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    scores = [(float(row["merge_prob"]), bool(row["gold_merge"])) for row in rows if row["family_group"] == group]
    positives = sum(1 for _, positive in scores if positive)
    payload: dict[str, Any] = {
        "candidate_count": len(scores),
        "positive_count": positives,
        "positive_rate": positives / len(scores) if scores else 0.0,
        "current_tau_0_30": pr_at(scores, 0.30) if scores else None,
        "best_f1": best_by_f1(scores) if scores else None,
        "precision_floor": {},
    }
    for floor in (0.70, 0.80, 0.90, 0.95):
        payload["precision_floor"][f"{floor:.2f}"] = best_for_precision(scores, floor)
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "family_group",
        "candidate_count",
        "positive_count",
        "positive_rate",
        "current_tau",
        "current_precision",
        "current_recall",
        "current_f1",
        "best_f1_tau",
        "best_f1_precision",
        "best_f1_recall",
        "best_f1",
        "tau_at_precision_0_80",
        "recall_at_precision_0_80",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Family-Specific MERGE Calibration Report",
        "",
        "## Scope",
        "",
        f"- Docs audited: {payload['doc_count']}",
        f"- Checkpoint: `{payload['checkpoint']}`",
        "- Read-only: no training, relabel, rebuild, MinerU, or E2E.",
        "",
        "## Calibration",
        "",
        "| family | candidates | positives | current P/R/F1 @0.30 | best F1 tau/P/R/F1 | tau for P>=0.80 |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for group, cal in payload["calibration"].items():
        cur = cal.get("current_tau_0_30") or {}
        best = cal.get("best_f1") or {}
        p80 = (cal.get("precision_floor") or {}).get("0.80") or {}
        lines.append(
            "| `{group}` | {cand} | {pos} | {cp:.3f}/{cr:.3f}/{cf:.3f} | {bt:.2f}/{bp:.3f}/{br:.3f}/{bf:.3f} | {p80tau} |".format(
                group=group,
                cand=cal.get("candidate_count", 0),
                pos=cal.get("positive_count", 0),
                cp=float(cur.get("precision", 0.0)),
                cr=float(cur.get("recall", 0.0)),
                cf=float(cur.get("f1", 0.0)),
                bt=float(best.get("threshold", 0.0)),
                bp=float(best.get("precision", 0.0)),
                br=float(best.get("recall", 0.0)),
                bf=float(best.get("f1", 0.0)),
                p80tau=("N/A" if not p80 else f"{float(p80['threshold']):.2f} recall={float(p80['recall']):.3f}"),
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            payload["recommendation"],
            "",
            "## Artifacts",
            "",
            "- `summary.json`",
            "- `summary.csv`",
            "- `examples.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--channel-audit-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--doc-ids", nargs="*")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    rows, doc_stats = collect_rows(
        args.graph_dir,
        args.channel_audit_dir,
        args.checkpoint,
        limit=args.limit,
        doc_ids=set(args.doc_ids) if args.doc_ids else None,
        device_name=args.device,
    )
    groups = ["body_list", "reference", "weak_masked", "layout_mismatch", "hard_negative", "other"]
    calibration = {group: calibrate(rows, group) for group in groups}
    csv_rows: list[dict[str, Any]] = []
    for group, cal in calibration.items():
        cur = cal.get("current_tau_0_30") or {}
        best = cal.get("best_f1") or {}
        p80 = (cal.get("precision_floor") or {}).get("0.80") or {}
        csv_rows.append(
            {
                "family_group": group,
                "candidate_count": cal.get("candidate_count", 0),
                "positive_count": cal.get("positive_count", 0),
                "positive_rate": cal.get("positive_rate", 0.0),
                "current_tau": 0.30,
                "current_precision": cur.get("precision"),
                "current_recall": cur.get("recall"),
                "current_f1": cur.get("f1"),
                "best_f1_tau": best.get("threshold"),
                "best_f1_precision": best.get("precision"),
                "best_f1_recall": best.get("recall"),
                "best_f1": best.get("f1"),
                "tau_at_precision_0_80": p80.get("threshold") if p80 else None,
                "recall_at_precision_0_80": p80.get("recall") if p80 else None,
            }
        )

    examples = {
        "body_list_false_negatives_at_0_30": sorted(
            [
                row
                for row in rows
                if row["family_group"] == "body_list" and row["gold_merge"] and float(row["merge_prob"]) < 0.30
            ],
            key=lambda row: -float(row["merge_prob"]),
        )[:50],
        "body_list_false_positives_at_0_30": sorted(
            [
                row
                for row in rows
                if row["family_group"] == "body_list" and (not row["gold_merge"]) and float(row["merge_prob"]) >= 0.30
            ],
            key=lambda row: -float(row["merge_prob"]),
        )[:50],
        "reference_false_negatives_at_0_30": sorted(
            [
                row
                for row in rows
                if row["family_group"] == "reference" and row["gold_merge"] and float(row["merge_prob"]) < 0.30
            ],
            key=lambda row: -float(row["merge_prob"]),
        )[:50],
    }

    body = calibration["body_list"]
    ref = calibration["reference"]
    body_best = body.get("best_f1") or {}
    ref_best = ref.get("best_f1") or {}
    recommendation = (
        "Use separate operating points: calibrate BODY_TEXT/LIST continuation independently from REFERENCE continuation. "
        f"In this audit BODY/LIST best-F1 tau is {float(body_best.get('threshold', 0.0)):.2f}, while REFERENCE best-F1 tau is "
        f"{float(ref_best.get('threshold', 0.0)):.2f}. Keep missing-candidate handling as a narrow candidate-edge audit, not the main branch."
    )
    payload = {
        "schema_version": "family_specific_merge_calibration_v1",
        "doc_count": len(doc_stats),
        "edge_count": len(rows),
        "checkpoint": str(args.checkpoint),
        "graph_dir": str(args.graph_dir),
        "channel_audit_dir": str(args.channel_audit_dir),
        "calibration": calibration,
        "doc_stats": doc_stats,
        "recommendation": recommendation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "summary.json", payload)
    dump_json(args.output_dir / "examples.json", examples)
    write_csv(args.output_dir / "summary.csv", csv_rows)
    report(args.output_dir / "FAMILY_SPECIFIC_MERGE_CALIBRATION_REPORT.md", payload)
    print(json.dumps({"calibration": calibration, "recommendation": recommendation}, ensure_ascii=False, indent=2)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
