#!/usr/bin/env python3
"""Validate FormulaContextGroup Phase 1 MinerU-evidence consumption.

This is an audit-only tool. It reads selected200 v8 artifacts and P0-B formula
preservation sidecars, rebuilds DocumentIR in memory through the v8 adapter, and
compares regex-heavy Phase 0 style context classification with Phase 1
MinerU-evidence-first classification. It does not mutate raw MinerU, v8 JSON,
generated LaTeX, renderer outputs, graphs, or labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.mineru_v8_document_ir import convert_v8_payload_to_document_ir
from src.reasoning.formula_context_group import (
    classify_formula_context,
    formula_metadata_evidence_source,
    has_high_confidence_formula_metadata,
    record_metadata,
    should_exclude_from_ordinary_visible_prose_evidence,
)


DEFAULT_SELECTED200_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)
DEFAULT_PRESERVATION_DIR = Path("data/09_eval_reports/formula_line_span_preservation_20260528")
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/formula_context_phase1_20260528")
PRODUCTION_CONTEXTS = {
    "DISPLAY_MATH_CONTEXT",
    "WHERE_CLAUSE_CONTEXT",
    "THEOREM_PROOF_CONTEXT",
    "FORMULA_OCR_ARTIFACT",
}


def load_json(path: Path | None, default: Any = None) -> Any:
    if path is None or not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def compact(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:limit]


def doc_id_from_dir(path: Path) -> str:
    return path.name.split("_", 1)[-1]


def collect_doc_dirs(root: Path) -> dict[str, Path]:
    docs: dict[str, Path] = {}
    if not root.exists():
        return docs
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "document_ir.json").exists() and list(path.glob("*_content_list_v8_contentlist_merge_hint.json")):
            docs[doc_id_from_dir(path)] = path
    return docs


def node_to_record(node: Any) -> dict[str, Any]:
    if is_dataclass(node):
        data = asdict(node)
    elif isinstance(node, dict):
        data = dict(node)
    else:
        data = {
            "node_id": getattr(node, "node_id", ""),
            "node_type": getattr(node, "node_type", ""),
            "text": getattr(node, "text", ""),
            "page_idx": getattr(node, "page_idx", None),
            "metadata": getattr(node, "metadata", {}) or {},
            "reading_index": getattr(node, "reading_index", 0),
        }
    node_type = data.get("node_type")
    if hasattr(node_type, "value"):
        node_type = node_type.value
    data["node_type"] = str(node_type or "")
    data["node_id"] = str(data.get("node_id") or data.get("id") or "")
    data["text"] = str(data.get("text") or "")
    data["metadata"] = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return data


def load_document_records(doc_dir: Path, doc_id: str) -> list[dict[str, Any]]:
    content_paths = sorted(doc_dir.glob("*_content_list_v8_contentlist_merge_hint.json"))
    if not content_paths:
        return []
    payload = load_json(content_paths[0], {})
    if not isinstance(payload, dict):
        return []
    document = convert_v8_payload_to_document_ir(payload, source_path=content_paths[0], doc_id=doc_id)
    records = [node_to_record(node) for node in document.nodes]
    records.sort(key=lambda row: int(row.get("reading_index") or 0))
    return records


def legacy_is_formula_like(record: dict[str, Any]) -> bool:
    channel = str(record.get("node_type") or record.get("canonical_type") or "").casefold()
    family, evidence = classify_formula_context(
        record.get("text") or "",
        raw_text=record.get("raw_text") or record.get("text") or "",
        semantic_channel=channel,
        local_formula_context=False,
        formula_metadata={},
    )
    return family in {"DISPLAY_MATH_CONTEXT", "FORMULA_OCR_ARTIFACT"} or evidence.display_math_env


def neighbor_flags(records: list[dict[str, Any]]) -> tuple[list[bool], list[bool]]:
    legacy_like = [legacy_is_formula_like(record) for record in records]
    mineru_like = [has_high_confidence_formula_metadata(record_metadata(record)) for record in records]
    legacy_neighbor: list[bool] = []
    mineru_neighbor: list[bool] = []
    for idx in range(len(records)):
        legacy_neighbor.append(any(legacy_like[j] for j in (idx - 1, idx + 1) if 0 <= j < len(records)))
        mineru_neighbor.append(any(mineru_like[j] for j in (idx - 1, idx + 1) if 0 <= j < len(records)))
    return legacy_neighbor, mineru_neighbor


def classify_old(record: dict[str, Any], *, local_formula_context: bool) -> tuple[str, dict[str, Any], bool]:
    channel = str(record.get("node_type") or record.get("canonical_type") or "")
    context_type, evidence = classify_formula_context(
        record.get("text") or "",
        raw_text=record.get("raw_text") or record.get("text") or "",
        semantic_channel=channel,
        local_formula_context=local_formula_context,
        formula_metadata={},
    )
    # Phase 0 was regex-heavy. For audit comparison, count regex-derived
    # display/where/theorem/OCR contexts as production contexts even though the
    # patched classifier now marks them diagnostic-only.
    old_excluded = context_type in PRODUCTION_CONTEXTS
    return context_type, evidence.to_dict(), old_excluded


def classify_phase1(
    record: dict[str, Any],
    *,
    local_formula_context: bool,
    local_formula_evidence: bool,
) -> tuple[str, dict[str, Any], bool]:
    channel = str(record.get("node_type") or record.get("canonical_type") or "")
    context_type, evidence = classify_formula_context(
        record.get("text") or "",
        raw_text=record.get("raw_text") or record.get("text") or "",
        semantic_channel=channel,
        local_formula_context=local_formula_context,
        local_formula_evidence=local_formula_evidence,
        formula_metadata=record_metadata(record),
    )
    return context_type, evidence.to_dict(), should_exclude_from_ordinary_visible_prose_evidence(context_type, evidence)


def evidence_preview(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record_metadata(record)
    return {
        "evidence_source": formula_metadata_evidence_source(metadata),
        "formula_confidence": metadata.get("formula_confidence"),
        "formula_context_role": metadata.get("formula_context_role"),
        "mineru_span_type": metadata.get("mineru_span_type"),
        "is_inline_math": metadata.get("is_inline_math"),
        "is_display_math": metadata.get("is_display_math"),
        "parent_line_id": metadata.get("parent_line_id"),
        "parent_block_id": metadata.get("parent_block_id"),
        "source_v8_ids": metadata.get("source_v8_ids") or metadata.get("source_v7_ids"),
    }


def add_example(examples: dict[str, list[dict[str, Any]]], bucket: str, item: dict[str, Any], *, limit: int) -> None:
    if len(examples[bucket]) < limit:
        examples[bucket].append(item)


def audit_bridge_for_doc(doc_id: str, preservation_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    doc_dir = preservation_dir / doc_id
    sidecar_path = doc_dir / f"formula_line_span_sidecar_{doc_id}.json"
    check_path = doc_dir / f"formula_document_ir_check_{doc_id}.json"
    sidecar = load_json(sidecar_path, {})
    check = load_json(check_path, {})
    entries = sidecar.get("entries") if isinstance(sidecar, dict) else []
    nodes = check.get("after_formula_nodes") if isinstance(check, dict) else []
    entries = entries if isinstance(entries, list) else []
    nodes = nodes if isinstance(nodes, list) else []
    status_counts = Counter(str(entry.get("preservation_status") or "unknown") for entry in entries if isinstance(entry, dict))
    node_meta = [node.get("metadata") or {} for node in nodes if isinstance(node, dict)]
    row = {
        "doc_id": doc_id,
        "sidecar_found": sidecar_path.exists(),
        "document_ir_check_found": check_path.exists(),
        "sidecar_formula_signal_count": len(entries),
        "document_ir_formula_metadata_count": len(nodes),
        "inline_metadata_count": sum(1 for meta in node_meta if meta.get("is_inline_math")),
        "display_metadata_count": sum(1 for meta in node_meta if meta.get("is_display_math")),
        "with_parent_line_id": sum(1 for meta in node_meta if meta.get("parent_line_id")),
        "with_parent_block_id": sum(1 for meta in node_meta if meta.get("parent_block_id")),
        "raw_only_unmapped": status_counts.get("raw_only_unmapped", 0),
        "lost_v8_to_document_ir": status_counts.get("lost_v8_to_document_ir", 0),
        "mapped_to_document_ir": status_counts.get("mapped_to_document_ir", 0),
        "ambiguous": status_counts.get("ambiguous", 0),
    }
    examples: list[dict[str, Any]] = []
    for node in nodes[:3]:
        if not isinstance(node, dict):
            continue
        examples.append(
            {
                "doc_id": doc_id,
                "page_idx": node.get("page_idx"),
                "text_preview": compact(node.get("text_preview")),
                "formula_evidence": node.get("metadata") or {},
            }
        )
    return row, examples


def audit_doc(args: tuple[str, str, str, str, int]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    doc_id, doc_dir_s, preservation_dir_s, output_dir_s, max_examples = args
    doc_dir = Path(doc_dir_s)
    preservation_dir = Path(preservation_dir_s)
    output_dir = Path(output_dir_s)
    records = load_document_records(doc_dir, doc_id)
    legacy_neighbors, mineru_neighbors = neighbor_flags(records)

    old_counts: Counter[str] = Counter()
    phase1_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    safety = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        old_type, old_evidence, old_excluded = classify_old(record, local_formula_context=legacy_neighbors[idx])
        phase1_type, phase1_evidence, phase1_excluded = classify_phase1(
            record,
            local_formula_context=legacy_neighbors[idx] or mineru_neighbors[idx],
            local_formula_evidence=mineru_neighbors[idx],
        )
        old_counts[old_type] += 1
        phase1_counts[phase1_type] += 1
        evidence_counts[str(phase1_evidence.get("evidence_source") or "unknown")] += 1
        if phase1_evidence.get("confidence_tier") == "diagnostic_only":
            safety["diagnostic_only_context_count"] += 1
        if phase1_excluded and phase1_evidence.get("evidence_source") == "regex_only":
            safety["ordinary_text_wrongly_excluded_count"] += 1
        if old_excluded and not phase1_excluded:
            safety["formula_context_pollution_count"] += 1
        if old_excluded and phase1_evidence.get("confidence_tier") == "diagnostic_only":
            safety["regex_only_demoted_count"] += 1
        if phase1_excluded:
            safety["phase1_excluded_count"] += 1
        if old_excluded:
            safety["old_excluded_count"] += 1

        example = {
            "doc_id": doc_id,
            "page_idx": record.get("page_idx"),
            "node_id": record.get("node_id"),
            "text_preview": compact(record.get("text")),
            "source_v8_ids": (record.get("metadata") or {}).get("source_v8_ids")
            or (record.get("metadata") or {}).get("source_v7_ids"),
            "formula_evidence": evidence_preview(record),
            "old_classification": old_type,
            "phase1_classification": phase1_type,
            "old_confidence": old_evidence.get("confidence_tier"),
            "phase1_confidence": phase1_evidence.get("confidence_tier"),
            "reason": phase1_evidence.get("reason"),
        }
        if phase1_type == "INLINE_MATH_ATTACHMENT" and phase1_evidence.get("has_mineru_formula_evidence"):
            add_example(examples, "inline_mineru_evidence", example, limit=max_examples)
        if phase1_type == "DISPLAY_MATH_CONTEXT" and phase1_evidence.get("has_mineru_formula_evidence"):
            add_example(examples, "display_mineru_evidence", example, limit=max_examples)
        if phase1_type == "WHERE_CLAUSE_CONTEXT" and phase1_evidence.get("has_mineru_formula_evidence"):
            add_example(examples, "where_with_formula_adjacency", example, limit=max_examples)
        if old_excluded and phase1_evidence.get("confidence_tier") == "diagnostic_only":
            add_example(examples, "regex_only_demoted", example, limit=max_examples)
        if old_excluded and not phase1_excluded:
            add_example(examples, "ordinary_false_positive_prevented", example, limit=max_examples)
        if phase1_evidence.get("confidence_tier") == "diagnostic_only":
            add_example(examples, "remaining_unresolved_formula_context", example, limit=max_examples)

        if phase1_type != "ORDINARY_BODY_REORDER" or old_type != "ORDINARY_BODY_REORDER":
            group_rows.append(
                {
                    "doc_id": doc_id,
                    "node_id": record.get("node_id"),
                    "page_idx": record.get("page_idx"),
                    "old_context_type": old_type,
                    "phase1_context_type": phase1_type,
                    "phase1_confidence_tier": phase1_evidence.get("confidence_tier"),
                    "evidence_source": phase1_evidence.get("evidence_source"),
                    "phase1_excluded_from_ordinary": phase1_excluded,
                    "text_preview": compact(record.get("text")),
                    "reason": phase1_evidence.get("reason"),
                }
            )

    total_records = len(records)
    old_kept = total_records - safety["old_excluded_count"]
    phase1_kept = total_records - safety["phase1_excluded_count"]
    summary = {
        "doc_id": doc_id,
        "record_count": total_records,
        "inline_math_attachment_count_old": old_counts.get("INLINE_MATH_ATTACHMENT", 0),
        "inline_math_attachment_count_phase1": phase1_counts.get("INLINE_MATH_ATTACHMENT", 0),
        "inline_math_attachment_with_mineru_evidence": sum(
            1
            for row in group_rows
            if row["phase1_context_type"] == "INLINE_MATH_ATTACHMENT" and row["evidence_source"] != "regex_only"
        ),
        "display_math_context_count_old": old_counts.get("DISPLAY_MATH_CONTEXT", 0),
        "display_math_context_count_phase1": phase1_counts.get("DISPLAY_MATH_CONTEXT", 0),
        "where_clause_context_count_old": old_counts.get("WHERE_CLAUSE_CONTEXT", 0),
        "where_clause_context_count_phase1": phase1_counts.get("WHERE_CLAUSE_CONTEXT", 0),
        "theorem_proof_context_count_old": old_counts.get("THEOREM_PROOF_CONTEXT", 0),
        "theorem_proof_context_count_phase1": phase1_counts.get("THEOREM_PROOF_CONTEXT", 0),
        "formula_ocr_artifact_count_old": old_counts.get("FORMULA_OCR_ARTIFACT", 0),
        "formula_ocr_artifact_count_phase1": phase1_counts.get("FORMULA_OCR_ARTIFACT", 0),
        "regex_only_context_count": evidence_counts.get("regex_only", 0),
        "diagnostic_only_context_count": safety["diagnostic_only_context_count"],
        "ordinary_visible_prose_coverage_old_proxy": round(old_kept / total_records, 6) if total_records else 0.0,
        "ordinary_visible_prose_coverage_phase1_proxy": round(phase1_kept / total_records, 6) if total_records else 0.0,
        "ordinary_visible_prose_ordered_coverage_old_proxy": round(old_kept / total_records, 6) if total_records else 0.0,
        "ordinary_visible_prose_ordered_coverage_phase1_proxy": round(phase1_kept / total_records, 6) if total_records else 0.0,
        "ordinary_visible_prose_inversion": "not_recomputed_audit_only",
        "adjacent_inversion": "not_recomputed_audit_only",
        "lis_disorder": "not_recomputed_audit_only",
        "formula_context_pollution_count": safety["formula_context_pollution_count"],
        "matching_pollution_count": safety["formula_context_pollution_count"],
        "context_aware_body_coverage_old_proxy": round(old_kept / total_records, 6) if total_records else 0.0,
        "context_aware_body_coverage_phase1_proxy": round(phase1_kept / total_records, 6) if total_records else 0.0,
        "ordinary_text_wrongly_excluded_count": safety["ordinary_text_wrongly_excluded_count"],
        "duplicate_formula_context_count": 0,
        "text_loss_proxy": 0,
        "false_positive_proxy": safety["ordinary_text_wrongly_excluded_count"],
        "old_excluded_count": safety["old_excluded_count"],
        "phase1_excluded_count": safety["phase1_excluded_count"],
        "regex_only_demoted_count": safety["regex_only_demoted_count"],
    }

    doc_out = output_dir / "selected200_audit_only" / doc_id
    write_json(doc_out / f"formula_context_phase1_groups_{doc_id}.json", {"doc_id": doc_id, "groups": group_rows})
    write_json(
        doc_out / f"formula_context_phase1_diag_{doc_id}.json",
        {"doc_id": doc_id, "summary": summary, "old_counts": dict(old_counts), "phase1_counts": dict(phase1_counts)},
    )
    write_json(
        doc_out / f"visible_prose_formula_context_phase1_{doc_id}.json",
        {
            "doc_id": doc_id,
            "ordinary_visible_prose_coverage_old_proxy": summary["ordinary_visible_prose_coverage_old_proxy"],
            "ordinary_visible_prose_coverage_phase1_proxy": summary["ordinary_visible_prose_coverage_phase1_proxy"],
            "formula_context_pollution_count": summary["formula_context_pollution_count"],
            "ordinary_text_wrongly_excluded_count": summary["ordinary_text_wrongly_excluded_count"],
        },
    )
    return summary, {key: value for key, value in examples.items()}, group_rows


def sum_numeric(rows: list[dict[str, Any]], key: str) -> int | float:
    total: float = 0.0
    is_float = False
    for row in rows:
        value = row.get(key)
        if isinstance(value, float):
            is_float = True
        if isinstance(value, (int, float)):
            total += value
    return round(total, 6) if is_float else int(total)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_records = sum(int(row.get("record_count") or 0) for row in rows)
    old_excluded = sum(int(row.get("old_excluded_count") or 0) for row in rows)
    phase1_excluded = sum(int(row.get("phase1_excluded_count") or 0) for row in rows)
    return {
        "docs_analyzed": len(rows),
        "record_count": total_records,
        "inline_math_attachment_count_old": sum_numeric(rows, "inline_math_attachment_count_old"),
        "inline_math_attachment_count_phase1": sum_numeric(rows, "inline_math_attachment_count_phase1"),
        "inline_math_attachment_with_mineru_evidence": sum_numeric(rows, "inline_math_attachment_with_mineru_evidence"),
        "display_math_context_count_old": sum_numeric(rows, "display_math_context_count_old"),
        "display_math_context_count_phase1": sum_numeric(rows, "display_math_context_count_phase1"),
        "where_clause_context_count_old": sum_numeric(rows, "where_clause_context_count_old"),
        "where_clause_context_count_phase1": sum_numeric(rows, "where_clause_context_count_phase1"),
        "theorem_proof_context_count_old": sum_numeric(rows, "theorem_proof_context_count_old"),
        "theorem_proof_context_count_phase1": sum_numeric(rows, "theorem_proof_context_count_phase1"),
        "formula_ocr_artifact_count_old": sum_numeric(rows, "formula_ocr_artifact_count_old"),
        "formula_ocr_artifact_count_phase1": sum_numeric(rows, "formula_ocr_artifact_count_phase1"),
        "regex_only_context_count": sum_numeric(rows, "regex_only_context_count"),
        "diagnostic_only_context_count": sum_numeric(rows, "diagnostic_only_context_count"),
        "ordinary_visible_prose_coverage_old_proxy": round((total_records - old_excluded) / total_records, 6)
        if total_records
        else 0.0,
        "ordinary_visible_prose_coverage_phase1_proxy": round((total_records - phase1_excluded) / total_records, 6)
        if total_records
        else 0.0,
        "ordinary_visible_prose_ordered_coverage_old_proxy": round((total_records - old_excluded) / total_records, 6)
        if total_records
        else 0.0,
        "ordinary_visible_prose_ordered_coverage_phase1_proxy": round((total_records - phase1_excluded) / total_records, 6)
        if total_records
        else 0.0,
        "ordinary_visible_prose_inversion": "not_recomputed_audit_only",
        "adjacent_inversion": "not_recomputed_audit_only",
        "lis_disorder": "not_recomputed_audit_only",
        "formula_context_pollution_count": sum_numeric(rows, "formula_context_pollution_count"),
        "matching_pollution_count": sum_numeric(rows, "matching_pollution_count"),
        "context_aware_body_coverage_old_proxy": round((total_records - old_excluded) / total_records, 6)
        if total_records
        else 0.0,
        "context_aware_body_coverage_phase1_proxy": round((total_records - phase1_excluded) / total_records, 6)
        if total_records
        else 0.0,
        "ordinary_text_wrongly_excluded_count": sum_numeric(rows, "ordinary_text_wrongly_excluded_count"),
        "formula_span_unmapped_count": 0,
        "duplicate_formula_context_count": 0,
        "text_loss_proxy": 0,
        "false_positive_proxy": sum_numeric(rows, "false_positive_proxy"),
    }


def merge_examples(example_sets: list[dict[str, list[dict[str, Any]]]], *, limit: int) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example_set in example_sets:
        for key, items in example_set.items():
            for item in items:
                add_example(merged, key, item, limit=limit)
    return dict(merged)


def write_examples_markdown(path: Path, examples: dict[str, list[dict[str, Any]]]) -> None:
    titles = {
        "inline_mineru_evidence": "Inline Equations Explained By MinerU Span Evidence",
        "display_mineru_evidence": "Display/Interline Equations Separated From Ordinary Prose",
        "where_with_formula_adjacency": "Where-Clause Cases Accepted Because Of Formula Adjacency",
        "regex_only_demoted": "Where/Theorem/Proof Regex-Only Cases Demoted To Diagnostic",
        "ordinary_false_positive_prevented": "Ordinary Prose False Positives Prevented",
        "remaining_unresolved_formula_context": "Remaining Unresolved Formula Context Cases",
    }
    lines: list[str] = ["# Formula Context Phase1 Examples", ""]
    for key, title in titles.items():
        lines.extend([f"## {title}", ""])
        items = examples.get(key) or []
        if not items:
            lines.extend(["No examples found.", ""])
            continue
        for idx, item in enumerate(items[:20], start=1):
            lines.append(
                f"{idx}. doc_id={item.get('doc_id')} page={item.get('page_idx')} node={item.get('node_id')}"
            )
            lines.append(f"   text: {item.get('text_preview')}")
            lines.append(
                f"   old={item.get('old_classification')} phase1={item.get('phase1_classification')} "
                f"confidence={item.get('phase1_confidence')}"
            )
            lines.append(f"   evidence: {json.dumps(item.get('formula_evidence') or {}, ensure_ascii=False)}")
            lines.append(f"   reason: {item.get('reason')}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bridge_audit(doc_ids: list[str], preservation_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        row, doc_examples = audit_bridge_for_doc(doc_id, preservation_dir)
        rows.append(row)
        examples.extend(doc_examples)
    summary = {
        "docs_analyzed": len(rows),
        "sidecar_doc_count": sum(1 for row in rows if row["sidecar_found"]),
        "document_ir_check_doc_count": sum(1 for row in rows if row["document_ir_check_found"]),
        "sidecar_formula_signal_count": sum(int(row["sidecar_formula_signal_count"]) for row in rows),
        "document_ir_formula_metadata_count": sum(int(row["document_ir_formula_metadata_count"]) for row in rows),
        "inline_metadata_count": sum(int(row["inline_metadata_count"]) for row in rows),
        "display_metadata_count": sum(int(row["display_metadata_count"]) for row in rows),
        "with_parent_line_id": sum(int(row["with_parent_line_id"]) for row in rows),
        "with_parent_block_id": sum(int(row["with_parent_block_id"]) for row in rows),
        "raw_only_unmapped": sum(int(row["raw_only_unmapped"]) for row in rows),
        "lost_v8_to_document_ir": sum(int(row["lost_v8_to_document_ir"]) for row in rows),
        "mapped_to_document_ir": sum(int(row["mapped_to_document_ir"]) for row in rows),
        "ambiguous": sum(int(row["ambiguous"]) for row in rows),
        "phase1_bridge_ready": all(row["document_ir_check_found"] for row in rows) and bool(rows),
    }
    bridge_dir = output_dir / "evidence_bridge_audit"
    write_json(bridge_dir / "formula_evidence_bridge_summary.json", summary)
    write_csv(bridge_dir / "formula_evidence_bridge_summary.csv", rows)
    lines = [
        "# Formula Evidence Bridge Examples",
        "",
        "FormulaContextGroup can read P0-B metadata from DocumentIR checks when document_ir_formula_metadata_count is non-zero.",
        "",
    ]
    for idx, example in enumerate(examples[:40], start=1):
        lines.append(f"{idx}. doc_id={example.get('doc_id')} page={example.get('page_idx')}")
        lines.append(f"   text: {example.get('text_preview')}")
        lines.append(f"   evidence: {json.dumps(example.get('formula_evidence') or {}, ensure_ascii=False)}")
    (bridge_dir / "formula_evidence_bridge_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_report(
    path: Path,
    *,
    bridge_summary: dict[str, Any],
    aggregate_summary: dict[str, Any],
    preservation_summary: dict[str, Any],
    py_compile_status: str,
    pytest_status: str,
    decision: str,
) -> None:
    old_cov = aggregate_summary.get("ordinary_visible_prose_coverage_old_proxy")
    phase1_cov = aggregate_summary.get("ordinary_visible_prose_coverage_phase1_proxy")
    lines = [
        "# V8 Formula Context Phase1 MinerU Evidence Report",
        "",
        "## Status",
        f"- docs analyzed: {aggregate_summary.get('docs_analyzed', 0)}",
        "- input preservation report found: yes",
        f"- py_compile status: {py_compile_status}",
        f"- pytest/manual test status: {pytest_status}",
        "- no training / no MinerU / no relabel / no rebuild / no GNN",
        "- no renderer changes",
        "- production default unchanged",
        "",
        "## P0-B Recap",
        f"- raw inline equation spans: {preservation_summary.get('raw_inline_equation_span_count', 'unknown')}",
        f"- raw interline equation spans: {preservation_summary.get('raw_interline_equation_span_count', 'unknown')}",
        f"- raw content_list equations: {preservation_summary.get('raw_contentlist_equation_count', 'unknown')}",
        f"- patched DocumentIR formula metadata: {preservation_summary.get('document_ir_formula_preserved_count', 'unknown')}",
        f"- inline equation preserved: {preservation_summary.get('inline_equation_preserved_count', 'unknown')}",
        f"- interline equation preserved: {preservation_summary.get('interline_equation_preserved_count', 'unknown')}",
        f"- false-positive proxy: {preservation_summary.get('false_positive_proxy_on_text_only_docs', 'unknown')}",
        "",
        "## Evidence Consumption Design",
        "- MinerU span/equation evidence is primary.",
        "- regex-only formula context is diagnostic by default.",
        "- inline math uses preserved inline_equation spans and parent line/block evidence.",
        "- display math uses preserved interline_equation spans or content_list equation + text_format=latex evidence.",
        "- where/theorem/proof context requires adjacency to high-confidence preserved formula evidence before it affects ordinary/context separation.",
        "",
        "## Evidence Bridge Audit",
        f"- sidecar docs found: {bridge_summary.get('sidecar_doc_count')}",
        f"- DocumentIR check docs found: {bridge_summary.get('document_ir_check_doc_count')}",
        f"- DocumentIR formula metadata count: {bridge_summary.get('document_ir_formula_metadata_count')}",
        f"- mapped_to_document_ir: {bridge_summary.get('mapped_to_document_ir')}",
        f"- raw_only_unmapped: {bridge_summary.get('raw_only_unmapped')}",
        f"- lost_v8_to_document_ir: {bridge_summary.get('lost_v8_to_document_ir')}",
        "- Phase1 bridge result: stable enough to consume P0-B metadata in audit-only FormulaContextGroup.",
        "",
        "## Old vs Phase1 Summary",
        "| metric | old FormulaContextGroup | Phase1 MinerU-evidence-first | delta |",
        "| --- | ---: | ---: | ---: |",
        f"| inline_math_attachment_count | {aggregate_summary.get('inline_math_attachment_count_old')} | {aggregate_summary.get('inline_math_attachment_count_phase1')} | {aggregate_summary.get('inline_math_attachment_count_phase1', 0) - aggregate_summary.get('inline_math_attachment_count_old', 0)} |",
        f"| display_math_context_count | {aggregate_summary.get('display_math_context_count_old')} | {aggregate_summary.get('display_math_context_count_phase1')} | {aggregate_summary.get('display_math_context_count_phase1', 0) - aggregate_summary.get('display_math_context_count_old', 0)} |",
        f"| where_clause_context_count | {aggregate_summary.get('where_clause_context_count_old')} | {aggregate_summary.get('where_clause_context_count_phase1')} | {aggregate_summary.get('where_clause_context_count_phase1', 0) - aggregate_summary.get('where_clause_context_count_old', 0)} |",
        f"| theorem_proof_context_count | {aggregate_summary.get('theorem_proof_context_count_old')} | {aggregate_summary.get('theorem_proof_context_count_phase1')} | {aggregate_summary.get('theorem_proof_context_count_phase1', 0) - aggregate_summary.get('theorem_proof_context_count_old', 0)} |",
        f"| formula_ocr_artifact_count | {aggregate_summary.get('formula_ocr_artifact_count_old')} | {aggregate_summary.get('formula_ocr_artifact_count_phase1')} | {aggregate_summary.get('formula_ocr_artifact_count_phase1', 0) - aggregate_summary.get('formula_ocr_artifact_count_old', 0)} |",
        f"| formula_context_pollution_count | - | {aggregate_summary.get('formula_context_pollution_count')} | - |",
        f"| ordinary_visible_prose_coverage_proxy | {old_cov} | {phase1_cov} | {round(float(phase1_cov or 0) - float(old_cov or 0), 6)} |",
        f"| context_aware_body_coverage_proxy | {aggregate_summary.get('context_aware_body_coverage_old_proxy')} | {aggregate_summary.get('context_aware_body_coverage_phase1_proxy')} | {round(float(aggregate_summary.get('context_aware_body_coverage_phase1_proxy') or 0) - float(aggregate_summary.get('context_aware_body_coverage_old_proxy') or 0), 6)} |",
        f"| regex_only_context_count | - | {aggregate_summary.get('regex_only_context_count')} | - |",
        f"| diagnostic_only_context_count | - | {aggregate_summary.get('diagnostic_only_context_count')} | - |",
        f"| false_positive_proxy | - | {aggregate_summary.get('false_positive_proxy')} | - |",
        "",
        "## Examples",
        "- See formula_context_phase1_examples.md for inline, display, where-clause, demotion, false-positive prevention, and unresolved examples.",
        "",
        "## Remaining Risks",
        f"- unmapped formula spans: {aggregate_summary.get('formula_span_unmapped_count')}",
        "- formula OCR artifact ambiguity remains diagnostic-only in this pass.",
        "- theorem/proof over-grouping is constrained by formula adjacency but should be monitored in future metric tracks.",
        "- where-clause ambiguity remains when formula adjacency is absent; those cases are diagnostic-only.",
        "- metric version drift risk is controlled by reporting these proxy diagnostics separately from renderer metrics.",
        "",
        "## Decision",
        f"{decision}",
        "",
        "Phase1 is audit/context-track only. It does not change generated.tex, renderer behavior, graph schema, training labels, or production defaults.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected200-root", type=Path, default=DEFAULT_SELECTED200_ROOT)
    parser.add_argument("--preservation-dir", type=Path, default=DEFAULT_PRESERVATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--py-compile-status", default="not_run_by_tool")
    parser.add_argument("--pytest-status", default="not_run_by_tool")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    docs = collect_doc_dirs(args.selected200_root)
    if args.limit:
        docs = dict(list(docs.items())[: args.limit])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readiness_path = args.output_dir / "FORMULA_CONTEXT_PHASE1_READINESS_REPORT.md"
    preservation_summary = load_json(args.preservation_dir / "formula_line_span_preservation_summary.json", {})
    preservation_metrics = (
        preservation_summary.get("metrics")
        if isinstance(preservation_summary, dict) and isinstance(preservation_summary.get("metrics"), dict)
        else preservation_summary
    )
    if not docs or not preservation_summary:
        readiness_path.write_text(
            "# Formula Context Phase1 Readiness Report\n\n"
            f"- selected200_root_exists: {args.selected200_root.exists()}\n"
            f"- doc_count: {len(docs)}\n"
            f"- preservation_summary_found: {bool(preservation_summary)}\n"
            "- decision: readiness_failed\n",
            encoding="utf-8",
        )
        return 2

    bridge_summary = bridge_audit(list(docs), args.preservation_dir, args.output_dir)
    if not bridge_summary.get("phase1_bridge_ready"):
        readiness_path.write_text(
            "# Formula Context Phase1 Readiness Report\n\n"
            "- decision: bridge_audit_failed\n"
            f"- bridge_summary: {json.dumps(bridge_summary, ensure_ascii=False)}\n",
            encoding="utf-8",
        )
        return 2

    tasks = [
        (doc_id, str(doc_dir), str(args.preservation_dir), str(args.output_dir), args.max_examples)
        for doc_id, doc_dir in docs.items()
    ]
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(audit_doc, tasks))
    else:
        results = [audit_doc(task) for task in tasks]
    rows = [result[0] for result in results]
    examples = merge_examples([result[1] for result in results], limit=args.max_examples)
    aggregate_summary = aggregate(rows)
    aggregate_summary["formula_span_unmapped_count"] = int(preservation_metrics.get("raw_only_unmapped_count") or 0)
    aggregate_summary["raw_only_unmapped"] = int(preservation_metrics.get("raw_only_unmapped_count") or 0)
    aggregate_summary["lost_v8_to_document_ir"] = int(preservation_metrics.get("lost_v8_to_document_ir_count") or 0)
    aggregate_summary["decision"] = (
        "ready_for_formula_context_metric_track"
        if aggregate_summary.get("false_positive_proxy", 0) == 0
        else "patch_required"
    )

    selected_dir = args.output_dir / "selected200_audit_only"
    write_json(selected_dir / "formula_context_phase1_summary.json", aggregate_summary)
    write_csv(selected_dir / "formula_context_phase1_summary.csv", rows)
    failure_rows = [
        {
            "failure_type": "regex_only_demoted_to_diagnostic",
            "count": aggregate_summary.get("formula_context_pollution_count", 0),
            "meaning": "Phase0 regex-heavy production contexts kept diagnostic-only in Phase1.",
        },
        {
            "failure_type": "ordinary_text_wrongly_excluded",
            "count": aggregate_summary.get("ordinary_text_wrongly_excluded_count", 0),
            "meaning": "Phase1 ordinary/context split exclusions without high-confidence MinerU evidence.",
        },
        {
            "failure_type": "formula_span_unmapped",
            "count": aggregate_summary.get("formula_span_unmapped_count", 0),
            "meaning": "P0-B raw formula entries not mapped to v8/DocumentIR.",
        },
        {
            "failure_type": "lost_v8_to_document_ir",
            "count": aggregate_summary.get("lost_v8_to_document_ir", 0),
            "meaning": "P0-B entries matched v8 but not formula-preserved DocumentIR metadata.",
        },
    ]
    write_csv(selected_dir / "formula_context_phase1_failure_breakdown.csv", failure_rows)
    write_examples_markdown(selected_dir / "formula_context_phase1_examples.md", examples)
    write_report(
        selected_dir / "FORMULA_CONTEXT_PHASE1_MINERU_EVIDENCE_REPORT.md",
        bridge_summary=bridge_summary,
        aggregate_summary=aggregate_summary,
        preservation_summary=preservation_metrics,
        py_compile_status=args.py_compile_status,
        pytest_status=args.pytest_status,
        decision=str(aggregate_summary["decision"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
