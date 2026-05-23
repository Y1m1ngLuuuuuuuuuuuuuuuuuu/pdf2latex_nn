#!/usr/bin/env python3
"""Audit MERGE supervision channels without rewriting labels or graphs.

This tool replays the existing PDF/TeX alignment for already-built graphs and
emits sidecar summaries describing which MERGE-like candidates are strong,
weak, masked, or hard negatives. It intentionally never saves a graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reasoning.label_generator import (
    AlignmentLabeler,
    AlignmentLabelerConfig,
    TexRelationLabel,
    same_layout_scope_can_merge,
    text_preview,
    tex_node_type_name,
)


ALLOWED_FAMILIES = {
    "BODY_TEXT_CONTINUATION",
    "LIST_CONTINUATION",
    "FORMULA_LEAD_IN",
    "FORMULA_CONTEXT",
    "FLOAT_SKIP_CONTINUATION",
    "WEAK_SAME_TEX",
    "LAYOUT_SCOPE_MISMATCH",
    "FLOAT_PROXY_ENDPOINT",
    "CAPTION_ENDPOINT",
    "REFERENCE_ENDPOINT",
    "HARD_NEGATIVE",
    "MASKED_UNKNOWN",
}

FORMULA_LEAD_IN_RE = re.compile(
    r"^\s*(where|with|given|let|for|such that|s\.t\.|subject to|and|or|while|if)\b",
    re.IGNORECASE,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def manifest_items(manifest: Path | None) -> list[dict[str, Any]]:
    if manifest is None or not manifest.exists():
        return []
    payload = load_json(manifest)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def resolve_path(value: str | None, *, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def doc_id_from_graph(path: Path) -> str:
    name = path.name
    for suffix in ("_v7_relabel_labeled_graph.pt", "_labeled_graph.pt", "_graph.pt", ".pt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def graph_items_from_dir(graph_dir: Path, *, limit: int | None = None, doc_ids: set[str] | None = None) -> list[dict[str, Any]]:
    paths = sorted(graph_dir.glob("*.pt"))
    items: list[dict[str, Any]] = []
    for path in paths:
        doc_id = doc_id_from_graph(path)
        if doc_ids and doc_id not in doc_ids:
            continue
        items.append({"doc_id": doc_id, "graph_path": str(path)})
        if limit is not None and len(items) >= limit:
            break
    return items


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


def formula_lead_in(text_a: str, text_b: str) -> bool:
    combined = " ".join(part for part in (text_a, text_b) if part)
    return bool(FORMULA_LEAD_IN_RE.match(combined))


def endpoint_family(src_channel: str, dst_channel: str, src_text: str, dst_text: str) -> str | None:
    channels = {src_channel, dst_channel}
    if "FLOAT_PROXY" in channels:
        return "FLOAT_PROXY_ENDPOINT"
    if "CAPTION" in channels:
        return "CAPTION_ENDPOINT"
    if "REFERENCE_ITEM" in channels:
        return "REFERENCE_ENDPOINT"
    if "DISPLAY_MATH" in channels:
        return "FORMULA_LEAD_IN" if formula_lead_in(src_text, dst_text) else "FORMULA_CONTEXT"
    return None


def normalize_relation_family(
    *,
    old_label: str,
    raw_family: str,
    same_tex: bool,
    same_tex_reason: str | None,
    layout_scope_mismatch: bool,
    src_channel: str,
    dst_channel: str,
    src_text: str,
    dst_text: str,
) -> str:
    endpoint = endpoint_family(src_channel, dst_channel, src_text, dst_text)
    if endpoint:
        if endpoint == "FLOAT_PROXY_ENDPOINT" and same_tex_reason == "FLOAT_INTERRUPTED":
            return "FLOAT_SKIP_CONTINUATION"
        return endpoint

    if layout_scope_mismatch or same_tex_reason in {"GEOMETRY_GATE", "LAYOUT_SCOPE_MISMATCH"}:
        return "LAYOUT_SCOPE_MISMATCH"

    if old_label == "MERGE":
        if "LIST" in {src_channel, dst_channel} or raw_family in {"LIST_ITEM_CONTINUATION", "LIST_CONTINUATION"}:
            return "LIST_CONTINUATION"
        if src_channel == "BODY_TEXT" and dst_channel == "BODY_TEXT":
            return "BODY_TEXT_CONTINUATION"
        if same_tex:
            return "WEAK_SAME_TEX"
        return "MASKED_UNKNOWN"

    if same_tex:
        if same_tex_reason in {"FLOAT_INTERRUPTED"}:
            return "FLOAT_SKIP_CONTINUATION"
        return "WEAK_SAME_TEX"

    if src_channel == "BODY_TEXT" and dst_channel == "BODY_TEXT":
        return "HARD_NEGATIVE"
    if src_channel in {"FRONT_MATTER", "PAGE_FURNITURE", "NOISE_OR_NO_RENDER"} or dst_channel in {
        "FRONT_MATTER",
        "PAGE_FURNITURE",
        "NOISE_OR_NO_RENDER",
    }:
        return "MASKED_UNKNOWN"
    return "HARD_NEGATIVE"


def strength_for_family(family: str, *, old_label: str) -> tuple[str, bool, float]:
    if family in {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION"} and old_label == "MERGE":
        return "strong", True, 1.0
    if family in {"FORMULA_LEAD_IN", "FORMULA_CONTEXT", "WEAK_SAME_TEX", "FLOAT_SKIP_CONTINUATION"}:
        return "masked", False, 0.0
    if family in {"FLOAT_PROXY_ENDPOINT", "CAPTION_ENDPOINT"}:
        return "exempt", False, 0.0
    if family == "REFERENCE_ENDPOINT":
        return ("weak", True, 0.2) if old_label == "MERGE" else ("hard_negative", True, 1.0)
    if family == "LAYOUT_SCOPE_MISMATCH":
        return "hard_negative", True, 1.0
    if family == "MASKED_UNKNOWN":
        return "masked", False, 0.0
    return "hard_negative", True, 1.0


def audit_one(item: dict[str, Any], *, output_root: Path, graph_dir: Path | None, root: Path) -> dict[str, Any]:
    import torch

    doc_id = str(item.get("doc_id") or item.get("paper_id") or "")
    graph_path = find_graph_path(item, root=root, graph_dir=graph_dir)
    if graph_path is None or not graph_path.exists():
        raise FileNotFoundError(f"Missing graph for {doc_id or item}")
    if not doc_id:
        doc_id = doc_id_from_graph(graph_path)

    content_json = resolve_path(item.get("content_json") or item.get("content_json_path"), root=root)
    tex_path = resolve_path(item.get("tex_path") or item.get("source_tex") or item.get("main_tex"), root=root)
    if content_json is None or tex_path is None:
        schema_content, schema_tex = graph_schema_paths(graph_path)
        content_json = content_json or schema_content
        tex_path = tex_path or schema_tex
    if content_json is None or not content_json.exists():
        raise FileNotFoundError(f"Missing content JSON for {doc_id}: {content_json}")
    if tex_path is None or not tex_path.exists():
        raise FileNotFoundError(f"Missing TeX source for {doc_id}: {tex_path}")

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

    records: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    strength_counts: Counter[str] = Counter()
    old_label_counts: Counter[str] = Counter()
    old_merge_family_counts: Counter[str] = Counter()
    old_merge_weight_counts: Counter[str] = Counter()
    interesting_count = 0

    for edge_pos in range(int(edge_index.shape[1])):
        src = int(edge_index[0, edge_pos].item())
        dst = int(edge_index[1, edge_pos].item())
        old_label_value = int(labels[edge_pos].item())
        old_label = "MERGE" if old_label_value == int(TexRelationLabel.MERGE) else (
            "PARENT_CHILD" if old_label_value == int(TexRelationLabel.PARENT_CHILD) else "NONE"
        )

        src_record = labeler.node_alignment_record(src)
        dst_record = labeler.node_alignment_record(dst)
        src_match = labeler.matches[src] if 0 <= src < len(labeler.matches) else None
        dst_match = labeler.matches[dst] if 0 <= dst < len(labeler.matches) else None
        src_tex = labeler.tex_nodes.get(src_match.tex_id) if src_match is not None and src_match.tex_id else None
        dst_tex = labeler.tex_nodes.get(dst_match.tex_id) if dst_match is not None and dst_match.tex_id else None
        same_tex = bool(src_match and dst_match and src_match.tex_id and src_match.tex_id == dst_match.tex_id)
        raw_record = labeler.edge_supervision_record(src, dst, label=old_label_value, edge_pos=edge_pos)
        same_tex_reason = None
        if same_tex and old_label != "MERGE":
            same_tex_reason = labeler.same_tex_not_merged_reason(src, dst, edge_pos=edge_pos)

        layout_scope_mismatch = False
        if same_tex:
            try:
                layout_scope_mismatch = not same_layout_scope_can_merge(labeler.pdf_nodes[src].item, labeler.pdf_nodes[dst].item)
            except Exception:
                layout_scope_mismatch = False

        family = normalize_relation_family(
            old_label=old_label,
            raw_family=str(raw_record.get("relation_family") or ""),
            same_tex=same_tex,
            same_tex_reason=same_tex_reason,
            layout_scope_mismatch=layout_scope_mismatch,
            src_channel=str(src_record["alignment_channel"]),
            dst_channel=str(dst_record["alignment_channel"]),
            src_text=labeler.pdf_nodes[src].text,
            dst_text=labeler.pdf_nodes[dst].text,
        )
        if family not in ALLOWED_FAMILIES:
            family = "MASKED_UNKNOWN"
        label_strength, train_mask, loss_weight = strength_for_family(family, old_label=old_label)

        family_counts[family] += 1
        strength_counts[label_strength] += 1
        old_label_counts[old_label] += 1
        if old_label == "MERGE":
            old_merge_family_counts[family] += 1
            old_merge_weight_counts[f"{loss_weight:.1f}"] += 1

        include_record = old_label == "MERGE" or same_tex or family not in {"HARD_NEGATIVE"}
        if include_record:
            interesting_count += 1
            records.append(
                {
                    "edge_pos": edge_pos,
                    "src": src,
                    "dst": dst,
                    "old_label": old_label,
                    "relation_family": family,
                    "label_strength": label_strength,
                    "proposed_loss_weight": loss_weight,
                    "train_mask_proposal": train_mask,
                    "same_tex_node": same_tex,
                    "same_tex_reason": same_tex_reason,
                    "layout_scope_mismatch": layout_scope_mismatch,
                    "src_channel": src_record["alignment_channel"],
                    "dst_channel": dst_record["alignment_channel"],
                    "src_strength": src_record["alignment_strength"],
                    "dst_strength": dst_record["alignment_strength"],
                    "src_tex_id": src_match.tex_id if src_match is not None else None,
                    "dst_tex_id": dst_match.tex_id if dst_match is not None else None,
                    "src_tex_node_type": tex_node_type_name(src_tex),
                    "dst_tex_node_type": tex_node_type_name(dst_tex),
                    "src_text_preview": text_preview(labeler.pdf_nodes[src].text),
                    "dst_text_preview": text_preview(labeler.pdf_nodes[dst].text),
                }
            )

    doc_dir = output_root / "per_doc" / doc_id
    audit_payload = {
        "schema_version": "channel_aware_merge_candidate_audit_v1",
        "doc_id": doc_id,
        "graph_path": str(graph_path),
        "content_json_path": str(content_json),
        "tex_path": str(tex_path),
        "edge_count": int(edge_index.shape[1]),
        "included_record_count": len(records),
        "records": records,
    }
    breakdown = {
        "schema_version": "channel_aware_merge_family_breakdown_v1",
        "doc_id": doc_id,
        "edge_count": int(edge_index.shape[1]),
        "interesting_edge_count": interesting_count,
        "old_label_counts": dict(sorted(old_label_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "label_strength_counts": dict(sorted(strength_counts.items())),
        "old_merge_family_counts": dict(sorted(old_merge_family_counts.items())),
        "old_merge_loss_weight_counts": dict(sorted(old_merge_weight_counts.items())),
        "old_merge_count": int(old_label_counts.get("MERGE", 0)),
        "old_merge_strong_body_list_count": int(
            old_merge_family_counts.get("BODY_TEXT_CONTINUATION", 0)
            + old_merge_family_counts.get("LIST_CONTINUATION", 0)
        ),
        "old_merge_mask_or_exempt_count": int(
            sum(
                count
                for family_name, count in old_merge_family_counts.items()
                if strength_for_family(family_name, old_label="MERGE")[2] == 0.0
            )
        ),
        "node_alignment_channel_stats": labeler.channel_supervision_audit_payload().get("node_alignment_channel_stats", {}),
    }
    dump_json(doc_dir / "merge_candidate_audit.json", audit_payload)
    dump_json(doc_dir / "merge_family_breakdown.json", breakdown)
    return breakdown


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "doc_id",
        "edge_count",
        "old_merge_count",
        "old_merge_strong_body_list_count",
        "old_merge_strong_body_list_ratio",
        "old_merge_weak_same_tex_count",
        "old_merge_formula_count",
        "old_merge_layout_scope_mismatch_count",
        "old_merge_mask_or_exempt_count",
        "old_merge_mask_or_exempt_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_report(path: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Channel-Aware MERGE Label Audit Report",
        "",
        "## Scope",
        "",
        f"- Docs audited: {summary['doc_count']}",
        f"- Graph dir: `{summary.get('graph_dir') or 'manifest'}`",
        "- This pass is read-only: no graph labels were rewritten, no training/relabel/rebuild/MinerU/E2E was run.",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| old MERGE edges | {agg['old_merge_count']} |",
        f"| strong BODY/LIST continuation | {agg['old_merge_strong_body_list_count']} ({agg['old_merge_strong_body_list_ratio']:.4f}) |",
        f"| weak same-TeX | {agg['old_merge_weak_same_tex_count']} ({agg['old_merge_weak_same_tex_ratio']:.4f}) |",
        f"| formula lead-in/context | {agg['old_merge_formula_count']} ({agg['old_merge_formula_ratio']:.4f}) |",
        f"| layout scope mismatch | {agg['old_merge_layout_scope_mismatch_count']} ({agg['old_merge_layout_scope_mismatch_ratio']:.4f}) |",
        f"| proposed mask/exempt among old MERGE | {agg['old_merge_mask_or_exempt_count']} ({agg['old_merge_mask_or_exempt_ratio']:.4f}) |",
        f"| proposed old MERGE effective loss mass | {agg['old_merge_effective_loss_mass']:.2f} / {agg['old_merge_count']} |",
        "",
        "## Family Counts Among Old MERGE",
        "",
        "| relation_family | count |",
        "| --- | ---: |",
    ]
    for family, count in sorted(agg["old_merge_family_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{family}` | {count} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            summary["recommendation"],
            "",
            "## Artifacts",
            "",
            "- Per-doc audit: `per_doc/<doc_id>/merge_candidate_audit.json`",
            "- Per-doc family breakdown: `per_doc/<doc_id>/merge_family_breakdown.json`",
            "- Aggregate JSON: `summary.json`",
            "- Aggregate CSV: `summary.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-ids", nargs="*", default=None)
    args = parser.parse_args()

    root = Path.cwd()
    doc_ids = set(args.doc_ids) if args.doc_ids else None
    items = manifest_items(args.manifest)
    if doc_ids:
        items = [
            item
            for item in items
            if str(item.get("doc_id") or item.get("paper_id") or doc_id_from_graph(Path(str(item.get("graph_path", "")))))
            in doc_ids
        ]
    if args.limit is not None and items:
        items = items[: args.limit]
    if not items:
        if args.graph_dir is None:
            raise SystemExit("No manifest items found and --graph-dir was not provided.")
        items = graph_items_from_dir(args.graph_dir, limit=args.limit, doc_ids=doc_ids)
    if not items:
        raise SystemExit("No items to audit.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    family_aggregate: Counter[str] = Counter()
    old_merge_family_aggregate: Counter[str] = Counter()
    old_merge_loss_mass = 0.0
    old_merge_count = 0
    errors: list[dict[str, str]] = []

    for item in items:
        try:
            breakdown = audit_one(item, output_root=args.output_dir, graph_dir=args.graph_dir, root=root)
        except Exception as exc:  # keep batch audit moving
            errors.append({"item": str(item), "error": repr(exc)})
            continue
        old_merge = int(breakdown["old_merge_count"])
        old_merge_count += old_merge
        family_aggregate.update(breakdown["family_counts"])
        old_merge_family_aggregate.update(breakdown["old_merge_family_counts"])
        for family_name, count in breakdown["old_merge_family_counts"].items():
            old_merge_loss_mass += strength_for_family(family_name, old_label="MERGE")[2] * int(count)

        weak_same = int(breakdown["old_merge_family_counts"].get("WEAK_SAME_TEX", 0))
        formula_count = int(
            breakdown["old_merge_family_counts"].get("FORMULA_LEAD_IN", 0)
            + breakdown["old_merge_family_counts"].get("FORMULA_CONTEXT", 0)
        )
        layout_mismatch = int(breakdown["old_merge_family_counts"].get("LAYOUT_SCOPE_MISMATCH", 0))
        strong_body_list = int(breakdown["old_merge_strong_body_list_count"])
        masked = int(breakdown["old_merge_mask_or_exempt_count"])
        rows.append(
            {
                "doc_id": breakdown["doc_id"],
                "edge_count": breakdown["edge_count"],
                "old_merge_count": old_merge,
                "old_merge_strong_body_list_count": strong_body_list,
                "old_merge_strong_body_list_ratio": f"{ratio(strong_body_list, old_merge):.6f}",
                "old_merge_weak_same_tex_count": weak_same,
                "old_merge_formula_count": formula_count,
                "old_merge_layout_scope_mismatch_count": layout_mismatch,
                "old_merge_mask_or_exempt_count": masked,
                "old_merge_mask_or_exempt_ratio": f"{ratio(masked, old_merge):.6f}",
            }
        )

    strong_body_list_total = int(
        old_merge_family_aggregate.get("BODY_TEXT_CONTINUATION", 0)
        + old_merge_family_aggregate.get("LIST_CONTINUATION", 0)
    )
    weak_same_total = int(old_merge_family_aggregate.get("WEAK_SAME_TEX", 0))
    formula_total = int(
        old_merge_family_aggregate.get("FORMULA_LEAD_IN", 0)
        + old_merge_family_aggregate.get("FORMULA_CONTEXT", 0)
    )
    layout_mismatch_total = int(old_merge_family_aggregate.get("LAYOUT_SCOPE_MISMATCH", 0))
    mask_or_exempt_total = int(
        sum(
            count
            for family_name, count in old_merge_family_aggregate.items()
            if strength_for_family(family_name, old_label="MERGE")[2] == 0.0
        )
    )
    recommendation = (
        "Recommendation: run a small MERGE-only relabel/retrain diagnostic only after freezing this mask/weight policy. "
        "Do not lower tau_merge globally and do not expand layout_scope_mismatch into positives."
    )
    if old_merge_count and ratio(mask_or_exempt_total, old_merge_count) < 0.05:
        recommendation = (
            "Recommendation: this audit shows little old-MERGE noise under the proposed mask/weight policy; prioritize "
            "missing-candidate and threshold diagnostics before MERGE-only retraining."
        )

    summary = {
        "schema_version": "channel_aware_merge_label_audit_summary_v1",
        "doc_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "graph_dir": str(args.graph_dir) if args.graph_dir else None,
        "manifest": str(args.manifest) if args.manifest else None,
        "aggregate": {
            "family_counts": dict(sorted(family_aggregate.items())),
            "old_merge_family_counts": dict(sorted(old_merge_family_aggregate.items())),
            "old_merge_count": old_merge_count,
            "old_merge_strong_body_list_count": strong_body_list_total,
            "old_merge_strong_body_list_ratio": ratio(strong_body_list_total, old_merge_count),
            "old_merge_weak_same_tex_count": weak_same_total,
            "old_merge_weak_same_tex_ratio": ratio(weak_same_total, old_merge_count),
            "old_merge_formula_count": formula_total,
            "old_merge_formula_ratio": ratio(formula_total, old_merge_count),
            "old_merge_layout_scope_mismatch_count": layout_mismatch_total,
            "old_merge_layout_scope_mismatch_ratio": ratio(layout_mismatch_total, old_merge_count),
            "old_merge_mask_or_exempt_count": mask_or_exempt_total,
            "old_merge_mask_or_exempt_ratio": ratio(mask_or_exempt_total, old_merge_count),
            "old_merge_effective_loss_mass": old_merge_loss_mass,
        },
        "rows": rows,
        "recommendation": recommendation,
    }
    dump_json(args.output_dir / "summary.json", summary)
    write_summary_csv(args.output_dir / "summary.csv", rows)
    markdown_report(args.output_dir / "CHANNEL_AWARE_MERGE_LABEL_AUDIT_REPORT.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
