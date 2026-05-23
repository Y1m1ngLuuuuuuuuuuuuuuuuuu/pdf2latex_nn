#!/usr/bin/env python3
"""Dry-run channel-aware MERGE v2 labels without rewriting graphs.

This script is intentionally audit-only. It replays the current labeler on
existing graph/content/TeX artifacts, reads the MERGE v2 sidecar fields emitted
by ``AlignmentLabeler.edge_supervision_record()``, and writes proposed label
statistics plus compact per-document records. It never saves a graph or changes
``graph.y``.
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

from src.reasoning.label_generator import AlignmentLabeler, AlignmentLabelerConfig, TexRelationLabel, text_preview


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_path(value: str | None, *, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def doc_id_from_graph(path: Path) -> str:
    name = path.name
    for suffix in ("_v7_truthgen_labeled_graph.pt", "_v7_relabel_labeled_graph.pt", "_labeled_graph.pt", "_graph.pt", ".pt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def manifest_items(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    return []


def graph_schema_paths(graph_path: Path) -> tuple[Path | None, Path | None]:
    import torch

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    schema = getattr(graph, "alignment_schema", {}) or {}
    content_path = schema.get("content_json_path")
    tex_path = schema.get("tex_path")
    return (
        resolve_path(str(content_path), root=Path.cwd()) if content_path else None,
        resolve_path(str(tex_path), root=Path.cwd()) if tex_path else None,
    )


def find_graph_path(item: dict[str, Any], *, root: Path, graph_dir: Path | None) -> Path | None:
    for key in ("graph_path", "graph_pt", "labeled_graph_path", "labeled_graph", "graph"):
        path = resolve_path(item.get(key), root=root)
        if path and path.exists():
            return path
    doc_id = str(item.get("doc_id") or item.get("paper_id") or "")
    if graph_dir and doc_id:
        matches = sorted(graph_dir.glob(f"{doc_id}*.pt"))
        if matches:
            return matches[0]
    return None


def label_name(value: int) -> str:
    if value == int(TexRelationLabel.MERGE):
        return "MERGE"
    if value == int(TexRelationLabel.PARENT_CHILD):
        return "PARENT_CHILD"
    return "NONE"


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def audit_doc(item: dict[str, Any], *, output_dir: Path, graph_dir: Path | None, root: Path) -> dict[str, Any]:
    import torch

    graph_path = find_graph_path(item, root=root, graph_dir=graph_dir)
    if graph_path is None:
        raise FileNotFoundError(f"missing graph: {item}")
    doc_id = str(item.get("doc_id") or item.get("paper_id") or doc_id_from_graph(graph_path))
    content_json = resolve_path(item.get("content_json") or item.get("content_json_path"), root=root)
    tex_path = resolve_path(item.get("tex_path") or item.get("source_tex") or item.get("main_tex"), root=root)
    if content_json is None or tex_path is None:
        schema_content, schema_tex = graph_schema_paths(graph_path)
        content_json = content_json or schema_content
        tex_path = tex_path or schema_tex
    if content_json is None or not content_json.exists():
        raise FileNotFoundError(f"missing content JSON for {doc_id}: {content_json}")
    if tex_path is None or not tex_path.exists():
        raise FileNotFoundError(f"missing TeX for {doc_id}: {tex_path}")

    labeler = AlignmentLabeler(
        content_json_path=content_json,
        tex_path=tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(
            audit_supervision_channels=True,
            emit_supervision_weights=False,
            abort_on_bad_alignment=False,
        ),
    )
    graph = labeler.run(output_graph_path=None, overwrite=False)
    edge_index = graph.edge_index.detach().cpu()
    labels = graph.y.detach().cpu() if hasattr(graph.y, "detach") else torch.as_tensor(graph.y)

    family_counts: Counter[str] = Counter()
    strength_counts: Counter[str] = Counter()
    old_label_counts: Counter[str] = Counter()
    old_merge_family_counts: Counter[str] = Counter()
    old_merge_strength_counts: Counter[str] = Counter()
    relevant_records: list[dict[str, Any]] = []

    for edge_pos in range(int(edge_index.shape[1])):
        src = int(edge_index[0, edge_pos].item())
        dst = int(edge_index[1, edge_pos].item())
        old_value = int(labels[edge_pos].item())
        old_label = label_name(old_value)
        record = labeler.edge_supervision_record(src, dst, label=old_value, edge_pos=edge_pos)
        family = str(record.get("merge_relation_family") or "MASKED_UNKNOWN")
        strength = str(record.get("merge_label_strength") or "masked")
        train_mask = bool(record.get("proposed_merge_train_mask"))
        loss_weight = float(record.get("proposed_merge_loss_weight") or 0.0)
        family_counts[family] += 1
        strength_counts[strength] += 1
        old_label_counts[old_label] += 1
        if old_label == "MERGE":
            old_merge_family_counts[family] += 1
            old_merge_strength_counts[strength] += 1
        if old_label == "MERGE" or record.get("same_tex_node") or family not in {"HARD_NEGATIVE", "MASKED_UNKNOWN"}:
            relevant_records.append(
                {
                    "edge_pos": int(edge_pos),
                    "src": int(src),
                    "dst": int(dst),
                    "old_label": old_label,
                    "merge_relation_family": family,
                    "merge_label_strength": strength,
                    "proposed_merge_train_mask": train_mask,
                    "proposed_merge_loss_weight": loss_weight,
                    "same_tex_node": bool(record.get("same_tex_node")),
                    "src_channel": record.get("src_channel"),
                    "dst_channel": record.get("dst_channel"),
                    "src_strength": record.get("src_strength"),
                    "dst_strength": record.get("dst_strength"),
                    "src_text_preview": text_preview(record.get("src_text_preview") or ""),
                    "dst_text_preview": text_preview(record.get("dst_text_preview") or ""),
                }
            )

    channel_payload = labeler.channel_supervision_audit_payload()
    missing_same_tex = sum(int(value) for value in channel_payload.get("missing_same_tex_candidate_edge_stats", {}).values())
    same_tex_edge_count = sum(
        int(count)
        for family, count in family_counts.items()
        if family in {
            "BODY_TEXT_CONTINUATION",
            "LIST_CONTINUATION",
            "REFERENCE_CONTINUATION",
            "FORMULA_LEAD_IN",
            "FORMULA_CONTEXT",
            "CODE_OR_PROMPT_LIKE",
            "CAPTION_TABLE_ISH",
            "FLOAT_PROXY_ENDPOINT",
            "CAPTION_ENDPOINT",
            "WEAK_SAME_TEX",
        }
    )
    merge_candidate_edge_recall = safe_ratio(same_tex_edge_count, same_tex_edge_count + missing_same_tex)
    candidate_edge_recall = getattr(graph, "candidate_edge_recall", None)

    breakdown = {
        "schema_version": "merge_v2_family_breakdown_v1",
        "doc_id": doc_id,
        "graph_path": str(graph_path),
        "content_json_path": str(content_json),
        "tex_path": str(tex_path),
        "edge_count": int(edge_index.shape[1]),
        "old_label_counts": dict(sorted(old_label_counts.items())),
        "merge_relation_family_counts": dict(sorted(family_counts.items())),
        "merge_label_strength_counts": dict(sorted(strength_counts.items())),
        "old_merge_family_counts": dict(sorted(old_merge_family_counts.items())),
        "old_merge_strength_counts": dict(sorted(old_merge_strength_counts.items())),
        "old_merge_count": int(old_label_counts.get("MERGE", 0)),
        "proposed_strong_body_text_merge_count": int(old_merge_family_counts.get("BODY_TEXT_CONTINUATION", 0)),
        "proposed_strong_list_merge_count": int(old_merge_family_counts.get("LIST_CONTINUATION", 0)),
        "proposed_reference_continuation_count": int(old_merge_family_counts.get("REFERENCE_CONTINUATION", 0)),
        "proposed_formula_masked_count": int(
            old_merge_family_counts.get("FORMULA_LEAD_IN", 0) + old_merge_family_counts.get("FORMULA_CONTEXT", 0)
        ),
        "proposed_code_prompt_masked_count": int(old_merge_family_counts.get("CODE_OR_PROMPT_LIKE", 0)),
        "proposed_caption_table_masked_count": int(
            old_merge_family_counts.get("CAPTION_TABLE_ISH", 0) + old_merge_family_counts.get("CAPTION_ENDPOINT", 0)
        ),
        "proposed_weak_same_tex_masked_count": int(old_merge_family_counts.get("WEAK_SAME_TEX", 0)),
        "layout_scope_mismatch_hard_negative_count": int(family_counts.get("LAYOUT_SCOPE_MISMATCH", 0)),
        "final_trainable_merge_positive_count": int(
            old_merge_family_counts.get("BODY_TEXT_CONTINUATION", 0) + old_merge_family_counts.get("LIST_CONTINUATION", 0)
        ),
        "masked_edge_count": int(strength_counts.get("masked", 0) + strength_counts.get("exempt", 0)),
        "old_merge_masked_edge_count": int(
            old_merge_strength_counts.get("masked", 0) + old_merge_strength_counts.get("exempt", 0)
        ),
        "hard_negative_count": int(strength_counts.get("hard_negative", 0)),
        "candidate_edge_recall": float(candidate_edge_recall) if candidate_edge_recall is not None else None,
        "merge_candidate_edge_recall": merge_candidate_edge_recall,
        "missing_same_tex_candidate_edge_count": int(missing_same_tex),
    }

    doc_dir = output_dir / "per_doc" / doc_id
    dump_json(
        doc_dir / "merge_v2_label_audit.json",
        {
            "schema_version": "merge_v2_label_audit_v1",
            "doc_id": doc_id,
            "records": relevant_records,
        },
    )
    dump_json(doc_dir / "merge_v2_family_breakdown.json", breakdown)
    return breakdown


def summarize(rows: list[dict[str, Any]], *, run_name: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    aggregate: Counter[str] = Counter()
    for row in rows:
        for key in (
            "old_merge_count",
            "proposed_strong_body_text_merge_count",
            "proposed_strong_list_merge_count",
            "proposed_reference_continuation_count",
            "proposed_formula_masked_count",
            "proposed_code_prompt_masked_count",
            "proposed_caption_table_masked_count",
            "proposed_weak_same_tex_masked_count",
            "layout_scope_mismatch_hard_negative_count",
            "final_trainable_merge_positive_count",
            "masked_edge_count",
            "old_merge_masked_edge_count",
            "hard_negative_count",
            "missing_same_tex_candidate_edge_count",
        ):
            aggregate[key] += int(row.get(key) or 0)
    recalls = [float(row["candidate_edge_recall"]) for row in rows if row.get("candidate_edge_recall") is not None]
    merge_recalls = [float(row["merge_candidate_edge_recall"]) for row in rows if row.get("merge_candidate_edge_recall") is not None]
    return {
        "schema_version": "merge_v2_relabel_dryrun_summary_v1",
        "run_name": run_name,
        "doc_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "aggregate": {
            **dict(aggregate),
            "candidate_edge_recall_mean": sum(recalls) / len(recalls) if recalls else None,
            "merge_candidate_edge_recall_mean": sum(merge_recalls) / len(merge_recalls) if merge_recalls else None,
        },
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "doc_id",
        "old_merge_count",
        "proposed_strong_body_text_merge_count",
        "proposed_strong_list_merge_count",
        "proposed_reference_continuation_count",
        "proposed_formula_masked_count",
        "proposed_code_prompt_masked_count",
        "proposed_caption_table_masked_count",
        "proposed_weak_same_tex_masked_count",
        "layout_scope_mismatch_hard_negative_count",
        "final_trainable_merge_positive_count",
        "old_merge_masked_edge_count",
        "masked_edge_count",
        "hard_negative_count",
        "candidate_edge_recall",
        "merge_candidate_edge_recall",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    root = Path.cwd()
    items = manifest_items(args.manifest)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No manifest items: {args.manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for item in items:
        try:
            rows.append(audit_doc(item, output_dir=args.output_dir, graph_dir=args.graph_dir, root=root))
        except Exception as exc:
            errors.append({"item": str(item), "error": repr(exc)})
    summary = summarize(rows, run_name=args.run_name, errors=errors)
    dump_json(args.output_dir / f"{args.run_name}_merge_v2_summary.json", summary)
    write_csv(args.output_dir / f"{args.run_name}_merge_v2_summary.csv", rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
