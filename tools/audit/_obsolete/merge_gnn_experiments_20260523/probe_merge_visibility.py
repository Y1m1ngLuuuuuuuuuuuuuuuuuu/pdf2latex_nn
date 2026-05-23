#!/usr/bin/env python3
"""Probe whether rejected MERGE candidates visibly affect generated structure.

This is a targeted audit tool, not a training or relabel script.  It reuses an
existing skip-compile E2E run, selects a few likely continuation candidates per
document, forces each candidate through the decoder one at a time, and compares
the forced output against a same-tool baseline render.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.batch_visual_qa_inference import (  # noqa: E402
    infer_document_title,
    load_node_records,
    render_decoded_tree_with_ir_backend,
)
from src.evaluation.comparison_structure import latex_file_to_comparison  # noqa: E402
from src.evaluation.structure_metrics import evaluate_comparison_structures  # noqa: E402
from src.reasoning.postprocess import (  # noqa: E402
    MERGE,
    ResolvedNode,
    TreeDecoder,
    TreeDecoderConfig,
    build_heading_skeleton,
    can_contract_merge_records,
    canonical_render_type,
    layout_scope_mismatch_is_only_band_boundary,
    merge_crosses_intermediate_list_marker,
    merge_crosses_section_boundary,
    node_physical_index,
    record_ends_with_hyphen,
    record_has_merge_excluded_layout_role,
    record_is_open_sentence,
    record_starts_with_list_marker,
    record_starts_with_lowercase_continuation,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-docs", type=int, default=10)
    parser.add_argument("--max-candidates-per-doc", type=int, default=6)
    parser.add_argument("--max-reading-gap", type=float, default=3.0)
    parser.add_argument("--merge-threshold", type=float, default=0.30)
    parser.add_argument("--parent-threshold", type=float, default=0.79)
    parser.add_argument(
        "--reasons",
        nargs="+",
        default=["hard_gate_layout_scope_mismatch", "section_boundary", "below_threshold"],
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    rows: list[dict[str, Any]] = []
    for doc_dir in list_doc_dirs(args.e2e_output_dir)[: args.limit_docs]:
        rows.extend(probe_document(doc_dir, args, torch=torch))

    summary = summarize(rows)
    payload = {"schema_version": "merge_visibility_probe_v1", "summary": summary, "rows": rows}
    json_path = args.output_dir / "MERGE_VISIBILITY_PROBE_20260522.json"
    md_path = args.output_dir / "MERGE_VISIBILITY_PROBE_20260522.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(md_path, args, summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(md_path)
    return 0


def list_doc_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "e2e_record.json").exists())


def probe_document(doc_dir: Path, args: argparse.Namespace, *, torch: Any) -> list[dict[str, Any]]:
    record = load_json(doc_dir / "e2e_record.json")
    doc_id = str(record["document_id"])
    content_json = Path(str(record["source_content_json"]))
    graph_path = Path(str(record["source_graph"]))
    source_tex = Path(str(record.get("source_tex") or ""))
    edge_logits_path = Path(str(record["edge_logits"]))
    predicted_relations_path = Path(str(record["predicted_relations"]))

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    logits = torch.load(edge_logits_path, map_location="cpu", weights_only=False)
    node_records = load_node_records(content_json, data)
    edge_index = data.edge_index.detach().cpu()

    base_tex, base_trace, base_metrics = render_and_score(
        node_records,
        edge_index,
        logits,
        content_json=content_json,
        source_tex=source_tex,
        predicted_relations_path=predicted_relations_path,
        doc_id=doc_id,
        merge_threshold=args.merge_threshold,
        parent_threshold=args.parent_threshold,
        forced_pair=None,
    )
    base_hash = normalize_tex_hash(base_tex)

    candidates = select_candidates(
        node_records,
        edge_index,
        logits,
        merge_threshold=args.merge_threshold,
        parent_threshold=args.parent_threshold,
        reasons=set(args.reasons),
        max_candidates=args.max_candidates_per_doc,
        max_reading_gap=args.max_reading_gap,
        torch=torch,
    )

    rows: list[dict[str, Any]] = []
    changed_dir = args.output_dir / "changed_cases"
    for candidate in candidates:
        pair = (int(candidate["source"]), int(candidate["target"]))
        forced_tex, forced_trace, forced_metrics = render_and_score(
            node_records,
            edge_index,
            logits,
            content_json=content_json,
            source_tex=source_tex,
            predicted_relations_path=predicted_relations_path,
            doc_id=doc_id,
            merge_threshold=args.merge_threshold,
            parent_threshold=args.parent_threshold,
            forced_pair=pair,
        )
        forced_hash = normalize_tex_hash(forced_tex)
        metric_delta = metric_deltas(base_metrics, forced_metrics)
        visible = forced_hash != base_hash or any(
            key.startswith("delta_") and abs(value) > 1e-12
            for key, value in metric_delta.items()
            if value is not None
        )
        row = {
            "doc_id": doc_id,
            **candidate,
            "visible": visible,
            "generated_tex_changed": forced_hash != base_hash,
            "base_tex_hash": base_hash,
            "forced_tex_hash": forced_hash,
            "base_render_tree_merged_nodes": (base_trace.get("merge_decoding") or {}).get("merged_supernode_count"),
            "forced_render_tree_merged_nodes": (forced_trace.get("merge_decoding") or {}).get("merged_supernode_count"),
            "forced_merge_edge_count": (forced_trace.get("merge_decoding") or {}).get("forced_merge_edge_count"),
            **metric_delta,
        }
        if visible:
            changed_dir.mkdir(parents=True, exist_ok=True)
            safe = f"{doc_id}_{pair[0]}_{pair[1]}".replace("/", "_")
            (changed_dir / f"{safe}.tex").write_text(forced_tex, encoding="utf-8")
        rows.append(row)
    return rows


def select_candidates(
    node_records: list[dict[str, Any]],
    edge_index: Any,
    logits: Any,
    *,
    merge_threshold: float,
    parent_threshold: float,
    reasons: set[str],
    max_candidates: int,
    max_reading_gap: float,
    torch: Any,
) -> list[dict[str, Any]]:
    decoder = TreeDecoder(TreeDecoderConfig(merge_threshold=merge_threshold, parent_threshold=parent_threshold))
    probs = decoder.edge_probabilities(logits)
    raw_nodes = {idx: ResolvedNode(node_id=idx, record=dict(record), merged_node_ids=[idx]) for idx, record in enumerate(node_records)}
    skeleton = build_heading_skeleton(raw_nodes, mode="stack")
    selected: list[dict[str, Any]] = []
    for edge_pos in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        if source == target or source < 0 or target < 0 or source >= len(node_records) or target >= len(node_records):
            continue
        left = node_records[source]
        right = node_records[target]
        if not likely_visible_text_continuation(left, right, max_reading_gap=max_reading_gap):
            continue
        reason = reject_reason(node_records, left, right, source, target, skeleton, merge_threshold, probs[edge_pos, MERGE])
        if reason not in reasons:
            continue
        selected.append(
            {
                "edge_pos": edge_pos,
                "source": source,
                "target": target,
                "reason": reason,
                "merge_prob": float(probs[edge_pos, MERGE].item()),
                "source_preview": preview_text(left),
                "target_preview": preview_text(right),
            }
        )
    selected.sort(key=lambda row: (reason_priority(str(row["reason"])), -float(row["merge_prob"]), int(row["source"])))
    return selected[:max_candidates]


def likely_visible_text_continuation(left: dict[str, Any], right: dict[str, Any], *, max_reading_gap: float) -> bool:
    if canonical_render_type(left) != "text" or canonical_render_type(right) != "text":
        return False
    if record_has_merge_excluded_layout_role(left) or record_has_merge_excluded_layout_role(right):
        return False
    if record_starts_with_list_marker(right):
        return False
    left_index = node_physical_index(left)
    right_index = node_physical_index(right)
    if left_index is None or right_index is None:
        return False
    if not (0.0 < float(right_index) - float(left_index) <= max_reading_gap):
        return False
    return (record_ends_with_hyphen(left) or record_is_open_sentence(left)) and record_starts_with_lowercase_continuation(right)


def reject_reason(
    node_records: list[dict[str, Any]],
    left: dict[str, Any],
    right: dict[str, Any],
    source: int,
    target: int,
    skeleton: Any,
    merge_threshold: float,
    merge_prob: Any,
) -> str:
    if merge_crosses_section_boundary(source, target, skeleton):
        return "section_boundary"
    if merge_crosses_intermediate_list_marker(source, target, node_records):
        return "intermediate_list_marker"
    if not can_contract_merge_records(left, right):
        if layout_scope_mismatch_is_only_band_boundary(left, right):
            return "hard_gate_layout_scope_mismatch"
        return "hard_gate_can_merge"
    if float(merge_prob.item() if hasattr(merge_prob, "item") else merge_prob) < merge_threshold:
        return "below_threshold"
    return "already_accepted_or_other"


def render_and_score(
    node_records: list[dict[str, Any]],
    edge_index: Any,
    logits: Any,
    *,
    content_json: Path,
    source_tex: Path,
    predicted_relations_path: Path,
    doc_id: str,
    merge_threshold: float,
    parent_threshold: float,
    forced_pair: tuple[int, int] | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    config = TreeDecoderConfig(
        merge_threshold=merge_threshold,
        parent_threshold=parent_threshold,
        heading_skeleton_mode="stack",
        forced_merge_pairs=tuple([forced_pair] if forced_pair else []),
    )
    decoder = TreeDecoder(config)
    root = decoder.decode(node_records, edge_index, logits)
    tex = render_decoded_tree_with_ir_backend(
        root,
        node_records=node_records,
        content_json=content_json,
        pdf_path=None,
        source_tex_path=source_tex if source_tex.exists() else None,
        document_id=doc_id,
        title=infer_document_title(node_records),
        document_metadata=None,
        predicted_relations_path=predicted_relations_path,
        decoder_trace=decoder.last_trace,
        attribution_output_path=None,
        table_asset_output_dir=None,
        figure_asset_output_dir=None,
        asset_latex_prefix="assets",
    )
    metrics: dict[str, Any] = {}
    if source_tex.exists():
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tex_path = Path(tmp) / "generated.tex"
            tex_path.write_text(tex, encoding="utf-8")
            gold = latex_file_to_comparison(source_tex, doc_id=doc_id)
            pred = latex_file_to_comparison(tex_path, doc_id=doc_id)
            metrics = evaluate_comparison_structures(gold.to_dict(), pred.to_dict(), match_threshold=0.58)
    return tex, decoder.last_trace, metrics


def metric_deltas(base: dict[str, Any], forced: dict[str, Any]) -> dict[str, float | None]:
    keys = [
        "macro_structure_score",
        "paragraph_boundary_f1",
        "paragraph_text_coverage_f1",
        "section_attachment_body_no_float_f1",
        "heading_tree_accuracy",
        "float_caption_attachment_accuracy",
        "generated_structure_validity",
    ]
    out: dict[str, float | None] = {}
    for key in keys:
        base_value = metric_value(base, key)
        forced_value = metric_value(forced, key)
        out[f"base_{key}"] = base_value
        out[f"forced_{key}"] = forced_value
        out[f"delta_{key}"] = None if base_value is None or forced_value is None else forced_value - base_value
    return out


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        for nested in ("score", "f1", "accuracy"):
            if nested in value:
                return metric_value({key: value[nested]}, key)
        return None
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def normalize_tex_hash(tex: str) -> str:
    text = re.sub(r"%[^\n]*", "", tex)
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def preview_text(record: dict[str, Any], limit: int = 120) -> str:
    text = str(record.get("merged_text") or record.get("text") or "")
    return " ".join(text.split())[:limit]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reason_priority(reason: str) -> int:
    order = {
        "hard_gate_layout_scope_mismatch": 0,
        "section_boundary": 1,
        "below_threshold": 2,
    }
    return order.get(reason, 9)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    visible = [row for row in rows if row.get("visible")]
    by_reason: dict[str, int] = {}
    visible_by_reason: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason"))
        by_reason[reason] = by_reason.get(reason, 0) + 1
        if row.get("visible"):
            visible_by_reason[reason] = visible_by_reason.get(reason, 0) + 1
    return {
        "probed_candidates": len(rows),
        "visible_candidates": len(visible),
        "generated_tex_changed": sum(1 for row in rows if row.get("generated_tex_changed")),
        "by_reason": by_reason,
        "visible_by_reason": visible_by_reason,
        "max_delta_paragraph_boundary_f1": max(
            [abs(float(row.get("delta_paragraph_boundary_f1") or 0.0)) for row in rows],
            default=0.0,
        ),
    }


def write_report(path: Path, args: argparse.Namespace, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    visible = [row for row in rows if row.get("visible")]
    lines = [
        "# MERGE Visibility Probe 20260522",
        "",
        "## Scope",
        f"- Existing E2E run: `{args.e2e_output_dir}`",
        "- No training, relabel, MinerU, API, CompHRDoc, or compile was run.",
        "- Each candidate is forced one at a time through decoder and renderer, then compared against a same-tool baseline.",
        "",
        "## Summary",
        f"- probed candidates: {summary['probed_candidates']}",
        f"- visible candidates: {summary['visible_candidates']}",
        f"- generated.tex changed: {summary['generated_tex_changed']}",
        f"- max |delta paragraph_boundary_f1|: {summary['max_delta_paragraph_boundary_f1']:.6f}",
        f"- by reason: `{json.dumps(summary['by_reason'], ensure_ascii=False)}`",
        f"- visible by reason: `{json.dumps(summary['visible_by_reason'], ensure_ascii=False)}`",
        "",
        "## Visible Candidates",
        "| doc_id | source | target | reason | merge_prob | tex_changed | delta_para_boundary | source preview | target preview |",
        "| --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in visible[:40]:
        lines.append(
            "| {} | {} | {} | {} | {:.4f} | {} | {:.6f} | {} | {} |".format(
                row.get("doc_id"),
                row.get("source"),
                row.get("target"),
                row.get("reason"),
                float(row.get("merge_prob") or 0.0),
                row.get("generated_tex_changed"),
                float(row.get("delta_paragraph_boundary_f1") or 0.0),
                str(row.get("source_preview", "")).replace("|", " "),
                str(row.get("target_preview", "")).replace("|", " "),
            )
        )
    if not visible:
        lines.append("| none |  |  |  |  |  |  |  |  |")
    lines.append("")
    lines.append("## Interpretation")
    if visible:
        lines.append("At least one forced MERGE candidate visibly changed output or metrics. Inspect the visible rows before changing labels or decoder policy.")
    else:
        lines.append("None of the sampled high-likelihood rejected MERGE candidates changed generated.tex or paragraph metrics. On this subset, missing MERGE candidates are mostly invisible after renderer consumption, so label changes should target only cases proven visible on a larger/harder probe.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
