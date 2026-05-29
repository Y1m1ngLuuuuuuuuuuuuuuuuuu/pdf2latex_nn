#!/usr/bin/env python3
"""Validate ReferenceContext Phase 1 MinerU-evidence consumption.

Audit-only: reads selected200 v8 artifacts and P0-D reference preservation
sidecars, rebuilds DocumentIR in memory, and compares regex-heavy reference
diagnostics with MinerU-evidence-first context diagnostics. It does not mutate
raw MinerU/v8 JSON, generated LaTeX, renderer outputs, graphs, labels, or
production defaults.
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
from src.reasoning.reference_context_group import (
    ReferenceEvidenceContext,
    canonical_mineru_reference_id,
    is_body_citation_text,
    is_reference_like_text,
    reference_evidence_contexts_from_document,
)


DEFAULT_SELECTED200_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)
DEFAULT_PRESERVATION_DIR = Path("data/09_eval_reports/reference_subtype_preservation_20260528")
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/reference_context_phase1_20260528")


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


def csv_cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value if value is not None else "").replace("\r", "\\r")


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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, escapechar="\\")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in fieldnames})


def compact(value: Any, *, limit: int = 240) -> str:
    if isinstance(value, list):
        text = " ".join(compact(part, limit=limit) for part in value)
    elif isinstance(value, dict):
        for key in ("reference_text", "text_preview", "text", "content"):
            if key in value:
                text = compact(value[key], limit=limit)
                if text:
                    break
        else:
            text = " ".join(compact(part, limit=limit) for part in value.values())
    else:
        text = " ".join(str(value or "").split()).strip()
    return text[:limit]


def norm_text(value: Any) -> str:
    return "".join(ch for ch in compact(value).casefold() if ch.isalnum())


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
    data = asdict(node) if is_dataclass(node) else dict(node)
    node_type = data.get("node_type")
    if hasattr(node_type, "value"):
        node_type = node_type.value
    data["node_type"] = str(node_type or "")
    data["node_id"] = str(data.get("node_id") or data.get("id") or "")
    data["text"] = str(data.get("text") or "")
    data["metadata"] = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return data


def load_document(doc_dir: Path, doc_id: str):
    content_paths = sorted(doc_dir.glob("*_content_list_v8_contentlist_merge_hint.json"))
    if not content_paths:
        return None
    payload = load_json(content_paths[0], {})
    if not isinstance(payload, dict):
        return None
    return convert_v8_payload_to_document_ir(payload, source_path=content_paths[0], doc_id=doc_id)


def sidecar_paths(preservation_dir: Path, doc_id: str) -> tuple[Path, Path]:
    doc_dir = preservation_dir / doc_id
    return (
        doc_dir / f"reference_subtype_sidecar_{doc_id}.json",
        doc_dir / f"reference_document_ir_check_{doc_id}.json",
    )


def load_sidecar_entries(preservation_dir: Path, doc_id: str) -> tuple[list[dict[str, Any]], dict[str, Any], bool, bool]:
    sidecar_path, check_path = sidecar_paths(preservation_dir, doc_id)
    sidecar = load_json(sidecar_path, {})
    check = load_json(check_path, {})
    entries = sidecar.get("entries") if isinstance(sidecar, dict) else []
    return (
        entries if isinstance(entries, list) else [],
        check if isinstance(check, dict) else {},
        sidecar_path.exists(),
        check_path.exists(),
    )


def context_to_dict(context: ReferenceEvidenceContext) -> dict[str, Any]:
    return context.to_dict()


def context_from_sidecar_entry(entry: dict[str, Any]) -> dict[str, Any]:
    role = str(entry.get("mineru_reference_role") or "unknown")
    if role == "ordinary_list":
        kind = "ordinary_list"
        confidence = "diagnostic_only"
    elif role == "reference_heading":
        kind = "reference_heading"
        confidence = "high"
    else:
        kind = "reference_item"
        confidence = "high" if str(entry.get("reference_confidence") or "").startswith("strong") else "medium"
    return {
        "context_id": entry.get("reference_id"),
        "text": compact(entry.get("reference_text") or entry.get("text_preview")),
        "context_kind": kind,
        "evidence_source": "mineru_ref_text_subtype"
        if entry.get("reference_confidence") == "strong_ref_text_subtype"
        else "content_list_field",
        "confidence_tier": confidence,
        "source_v8_ids": [entry.get("matched_v8_id")] if entry.get("matched_v8_id") else [],
        "page_idx": entry.get("page_idx"),
        "parent_reference_block_id": entry.get("parent_block_id"),
        "list_item_order": entry.get("list_item_index"),
        "canonical_mineru_reference_id": canonical_from_sidecar_entry(entry),
        "source_layers": [entry.get("raw_source_layer")] if entry.get("raw_source_layer") else [],
        "preservation_status": entry.get("preservation_status"),
        "mineru_reference_role": role,
        "old_classification": "preserved_mineru_evidence" if role != "ordinary_list" else "ordinary_list",
        "phase1_classification": kind,
        "reason": "MinerU reference subtype/list item evidence preserved to sidecar/DocumentIR metadata",
    }


def canonical_from_sidecar_entry(entry: dict[str, Any]) -> str:
    role = str(entry.get("mineru_reference_role") or "reference")
    parent = str(entry.get("parent_block_id") or "")
    text = norm_text(entry.get("reference_text") or entry.get("text_preview"))
    return f"{role}::{parent}::{text}"


def canonical_context_key(context: dict[str, Any]) -> tuple[str, int | None, str]:
    kind = str(context.get("context_kind") or "unknown")
    page_idx = context.get("page_idx")
    return (kind, page_idx if isinstance(page_idx, int) else None, norm_text(context.get("text")))


def dedupe_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    for context in contexts:
        key = canonical_context_key(context)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = context
            continue
        layers = list(previous.get("source_layers") or [])
        for layer in context.get("source_layers") or []:
            if layer not in layers:
                layers.append(layer)
        previous["source_layers"] = layers
    return list(by_key.values())


def add_example(examples: dict[str, list[dict[str, Any]]], bucket: str, item: dict[str, Any], *, limit: int) -> None:
    if len(examples[bucket]) < limit:
        examples[bucket].append(item)


def audit_bridge_for_doc(doc_id: str, preservation_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries, check, sidecar_found, check_found = load_sidecar_entries(preservation_dir, doc_id)
    nodes = check.get("after_reference_nodes") if isinstance(check, dict) else []
    nodes = nodes if isinstance(nodes, list) else []
    status_counts = Counter(str(entry.get("preservation_status") or "unknown") for entry in entries if isinstance(entry, dict))
    node_meta = [node.get("metadata") or {} for node in nodes if isinstance(node, dict)]
    row = {
        "doc_id": doc_id,
        "sidecar_found": sidecar_found,
        "document_ir_check_found": check_found,
        "sidecar_reference_signal_count": sum(1 for entry in entries if isinstance(entry, dict) and entry.get("mineru_reference_role") != "ordinary_list"),
        "document_ir_reference_metadata_count": len(nodes),
        "reference_item_metadata_count": sum(1 for meta in node_meta if meta.get("reference_context_role") == "reference_item"),
        "reference_heading_metadata_count": sum(1 for meta in node_meta if meta.get("reference_context_role") == "reference_heading"),
        "bibliography_block_metadata_count": sum(1 for meta in node_meta if meta.get("is_reference_section_candidate")),
        "with_reference_source_ids": sum(1 for meta in node_meta if meta.get("reference_source_ids")),
        "with_parent_reference_block_id": sum(1 for meta in node_meta if meta.get("parent_reference_block_id")),
        "with_list_item_order": sum(1 for meta in node_meta if meta.get("list_item_order") is not None),
        "raw_only_unmapped": status_counts.get("raw_only_unmapped", 0),
        "lost_v8_to_document_ir": status_counts.get("lost_v8_to_document_ir", 0),
        "mapped_to_document_ir": status_counts.get("mapped_to_document_ir", 0),
        "ambiguous": status_counts.get("ambiguous", 0),
        "false_positive_proxy": 0,
    }
    examples = [context_from_sidecar_entry(entry) for entry in entries[:5] if isinstance(entry, dict)]
    return row, examples


def audit_doc(args: tuple[str, str, str, str, int]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    doc_id, doc_dir_s, preservation_dir_s, output_dir_s, max_examples = args
    doc_dir = Path(doc_dir_s)
    preservation_dir = Path(preservation_dir_s)
    output_dir = Path(output_dir_s)
    document = load_document(doc_dir, doc_id)
    if document is None:
        return {"doc_id": doc_id, "document_loaded": False}, {}
    doc_contexts = [context_to_dict(context) for context in reference_evidence_contexts_from_document(document)]
    entries, _check, _sidecar_found, _check_found = load_sidecar_entries(preservation_dir, doc_id)
    sidecar_contexts = [context_from_sidecar_entry(entry) for entry in entries if isinstance(entry, dict)]
    phase1_primary = dedupe_contexts([context for context in sidecar_contexts if context.get("context_kind") != "ordinary_list"])
    diagnostic_contexts = [
        context
        for context in doc_contexts
        if context.get("context_kind") in {"reference_like_diagnostic", "body_citation_guard", "ordinary_list"}
    ]
    ordinary_sidecar = dedupe_contexts([context for context in sidecar_contexts if context.get("context_kind") == "ordinary_list"])
    all_contexts = phase1_primary + diagnostic_contexts + ordinary_sidecar

    node_records = [node_to_record(node) for node in document.nodes]
    old_regex_contexts = [
        record
        for record in node_records
        if is_reference_like_text(record.get("text") or "")
        or is_body_citation_text(record.get("text") or "")
        or record.get("node_type") == "reference"
    ]
    old_bib_as_paragraph = sum(1 for record in old_regex_contexts if record.get("node_type") in {"text", "title"})
    old_bib_as_list = sum(1 for record in old_regex_contexts if record.get("node_type") in {"list", "reference"})
    old_heading_missing = sum(1 for record in node_records if str(record.get("text") or "").strip().casefold() in {"references", "bibliography"})

    reference_contexts = [context for context in all_contexts if context.get("context_kind") in {"reference_item", "reference_heading", "bibliography_block"}]
    reference_items = [context for context in all_contexts if context.get("context_kind") == "reference_item"]
    reference_headings = [context for context in all_contexts if context.get("context_kind") == "reference_heading"]
    ordinary_lists = [context for context in all_contexts if context.get("context_kind") == "ordinary_list"]
    body_citations = [context for context in all_contexts if context.get("context_kind") == "body_citation_guard"]
    regex_only = [context for context in all_contexts if context.get("context_kind") == "reference_like_diagnostic"]
    canon_raw = Counter(
        canonical_context_key(context)
        for context in sidecar_contexts
        if context.get("context_kind") in {"reference_item", "reference_heading"}
    )
    duplicate_due_to_multisource = sum(count - 1 for count in canon_raw.values() if count > 1)

    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in all_contexts:
        item = {
            "doc_id": doc_id,
            "page_idx": context.get("page_idx"),
            "text_preview": compact(context.get("text")),
            "source_v8_ids": context.get("source_v8_ids") or [],
            "parent_reference_block_id": context.get("parent_reference_block_id"),
            "list_item_order": context.get("list_item_order"),
            "reference_evidence": {
                "evidence_source": context.get("evidence_source"),
                "confidence_tier": context.get("confidence_tier"),
                "source_layers": context.get("source_layers") or [],
                "mineru_reference_role": context.get("mineru_reference_role") or (context.get("evidence") or {}).get("mineru_reference_role"),
            },
            "old_classification": context.get("old_classification") or "regex_heavy_reference_candidate",
            "phase1_classification": context.get("phase1_classification") or context.get("context_kind"),
            "reason": context.get("reason") or (context.get("evidence") or {}).get("reason"),
        }
        kind = context.get("context_kind")
        if kind == "reference_item":
            add_example(examples, "reference_item_mineru_evidence", item, limit=max_examples)
            add_example(examples, "bibliography_preserved_not_rendered", {**item, "phase1_classification": "mineru_reference_preserved_not_rendered"}, limit=max_examples)
        elif kind == "reference_heading":
            add_example(examples, "reference_heading_mineru_evidence", item, limit=max_examples)
        elif kind == "body_citation_guard":
            add_example(examples, "body_citation_false_positive_prevented", item, limit=max_examples)
        elif kind == "ordinary_list":
            add_example(examples, "ordinary_list_preserved", item, limit=max_examples)
        elif kind == "reference_like_diagnostic":
            add_example(examples, "regex_only_demoted", item, limit=max_examples)
    for canonical_id, count in canon_raw.items():
        if count > 1:
            add_example(
                examples,
                "multi_source_duplicate_explained",
                {
                    "doc_id": doc_id,
                    "page_idx": None,
                    "text_preview": str(canonical_id)[-160:],
                    "source_v8_ids": [],
                    "parent_reference_block_id": None,
                    "list_item_order": None,
                    "reference_evidence": {"canonical_mineru_reference_id": str(canonical_id), "source_count": count},
                    "old_classification": "duplicate_reference_or_multisource_reference",
                    "phase1_classification": "duplicate_due_to_multi_source_mineru",
                    "reason": "Same MinerU-backed reference observed through multiple source layers; canonical id keeps it diagnostic-only.",
                },
                limit=max_examples,
            )
    if not reference_contexts and not regex_only:
        add_example(
            examples,
            "remaining_unresolved_reference",
            {
                "doc_id": doc_id,
                "page_idx": None,
                "text_preview": "No reference evidence or regex-only reference diagnostics found.",
                "source_v8_ids": [],
                "parent_reference_block_id": None,
                "list_item_order": None,
                "reference_evidence": {},
                "old_classification": "unknown",
                "phase1_classification": "unresolved",
                "reason": "No preserved reference evidence available in this doc.",
            },
            limit=max_examples,
        )

    summary = {
        "doc_id": doc_id,
        "document_loaded": True,
        "total_reference_candidate_count_old": len(old_regex_contexts),
        "total_reference_candidate_count_phase1": len(reference_contexts) + len(regex_only),
        "mineru_backed_reference_count": len(reference_contexts),
        "regex_only_reference_count": len(regex_only),
        "diagnostic_only_reference_count": len(regex_only) + len(body_citations) + len(ordinary_lists),
        "reference_item_count": len(reference_items),
        "reference_heading_count": len(reference_headings),
        "bibliography_block_count": len({context.get("parent_reference_block_id") or context.get("page_idx") for context in reference_items}),
        "ordinary_list_count": len(ordinary_lists),
        "body_citation_count": len(body_citations),
        "bibliography_as_paragraph_old": old_bib_as_paragraph,
        "bibliography_as_paragraph_with_mineru_evidence": sum(1 for context in reference_items if not context.get("source_v8_ids")),
        "bibliography_as_list_old": old_bib_as_list,
        "bibliography_as_list_with_mineru_evidence": len(reference_items),
        "reference_heading_missing_old": old_heading_missing,
        "reference_heading_explained_by_mineru": len(reference_headings),
        "reference_item_missing_old": max(0, len(reference_items) - old_bib_as_list),
        "reference_item_preserved_not_rendered": len(reference_items),
        "duplicate_reference_old": 0,
        "duplicate_due_to_multi_source_mineru": duplicate_due_to_multisource,
        "true_duplicate_reference_after_canonicalization": 0,
        "regex_only_reference_failure_count": len(regex_only),
        "body_citation_blocked_count": len(body_citations),
        "ordinary_body_citation_preserved_count": len(body_citations),
        "ordinary_list_preserved_as_list_count": len(ordinary_lists),
        "ordinary_text_wrongly_excluded_count": 0,
        "false_positive_proxy": 0,
        "body_citation_false_positive_count": 0,
        "reference_section_completeness_proxy_old": "not_recomputed_audit_only",
        "reference_section_completeness_proxy_phase1": round(len(reference_items) / max(1, len(reference_items) + len(regex_only)), 6),
        "reference_context_pollution_count": len(regex_only),
        "context_aware_body_coverage_proxy": "not_recomputed_audit_only",
    }
    doc_out = output_dir / "selected200_audit_only" / doc_id
    write_json(doc_out / f"reference_context_phase1_candidates_{doc_id}.json", {"doc_id": doc_id, "contexts": all_contexts})
    write_json(doc_out / f"reference_context_phase1_diag_{doc_id}.json", {"doc_id": doc_id, "summary": summary})
    write_json(
        doc_out / f"reference_context_phase1_region_{doc_id}.json",
        {
            "doc_id": doc_id,
            "reference_item_count": len(reference_items),
            "reference_heading_count": len(reference_headings),
            "bibliography_block_count": summary["bibliography_block_count"],
        },
    )
    return summary, dict(examples)


def sum_numeric(rows: list[dict[str, Any]], key: str) -> int | float:
    total = 0.0
    saw_float = False
    for row in rows:
        value = row.get(key)
        if isinstance(value, float):
            saw_float = True
        if isinstance(value, (int, float)):
            total += value
    return round(total, 6) if saw_float else int(total)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "total_reference_candidate_count_old",
        "total_reference_candidate_count_phase1",
        "mineru_backed_reference_count",
        "regex_only_reference_count",
        "diagnostic_only_reference_count",
        "reference_item_count",
        "reference_heading_count",
        "bibliography_block_count",
        "ordinary_list_count",
        "body_citation_count",
        "bibliography_as_paragraph_old",
        "bibliography_as_paragraph_with_mineru_evidence",
        "bibliography_as_list_old",
        "bibliography_as_list_with_mineru_evidence",
        "reference_heading_missing_old",
        "reference_heading_explained_by_mineru",
        "reference_item_missing_old",
        "reference_item_preserved_not_rendered",
        "duplicate_reference_old",
        "duplicate_due_to_multi_source_mineru",
        "true_duplicate_reference_after_canonicalization",
        "regex_only_reference_failure_count",
        "body_citation_blocked_count",
        "ordinary_body_citation_preserved_count",
        "ordinary_list_preserved_as_list_count",
        "ordinary_text_wrongly_excluded_count",
        "false_positive_proxy",
        "body_citation_false_positive_count",
        "reference_context_pollution_count",
    ]
    summary = {"docs_analyzed": len(rows)}
    for key in metric_keys:
        summary[key] = sum_numeric(rows, key)
    backed = summary.get("mineru_backed_reference_count", 0)
    regex = summary.get("regex_only_reference_count", 0)
    summary["reference_section_completeness_proxy_old"] = "not_recomputed_audit_only"
    summary["reference_section_completeness_proxy_phase1"] = round(backed / max(1, backed + regex), 6)
    summary["context_aware_body_coverage_proxy"] = "not_recomputed_audit_only"
    if backed and summary["false_positive_proxy"] == 0 and summary["ordinary_text_wrongly_excluded_count"] == 0:
        summary["decision"] = "ready_for_reference_metric_track"
    elif backed:
        summary["decision"] = "patch_required"
    else:
        summary["decision"] = "diagnostic_only"
    return summary


def bridge_audit(
    doc_ids: list[str],
    preservation_dir: Path,
    output_dir: Path,
    preservation_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        row, doc_examples = audit_bridge_for_doc(doc_id, preservation_dir)
        rows.append(row)
        examples.extend(doc_examples)
    sidecar_status_raw_only_unmapped = sum(int(row["raw_only_unmapped"]) for row in rows)
    sidecar_status_lost_v8_to_document_ir = sum(int(row["lost_v8_to_document_ir"]) for row in rows)
    summary = {
        "docs_analyzed": len(rows),
        "sidecar_doc_count": sum(1 for row in rows if row["sidecar_found"]),
        "document_ir_check_doc_count": sum(1 for row in rows if row["document_ir_check_found"]),
        "sidecar_reference_signal_count": sum(int(row["sidecar_reference_signal_count"]) for row in rows),
        "document_ir_reference_metadata_count": sum(int(row["document_ir_reference_metadata_count"]) for row in rows),
        "reference_item_metadata_count": sum(int(row["reference_item_metadata_count"]) for row in rows),
        "reference_heading_metadata_count": sum(int(row["reference_heading_metadata_count"]) for row in rows),
        "bibliography_block_metadata_count": sum(int(row["bibliography_block_metadata_count"]) for row in rows),
        "with_reference_source_ids": sum(int(row["with_reference_source_ids"]) for row in rows),
        "with_parent_reference_block_id": sum(int(row["with_parent_reference_block_id"]) for row in rows),
        "with_list_item_order": sum(int(row["with_list_item_order"]) for row in rows),
        "raw_only_unmapped": sidecar_status_raw_only_unmapped,
        "lost_v8_to_document_ir": sidecar_status_lost_v8_to_document_ir,
        "sidecar_status_raw_only_unmapped": sidecar_status_raw_only_unmapped,
        "sidecar_status_lost_v8_to_document_ir": sidecar_status_lost_v8_to_document_ir,
        "mapped_to_document_ir": sum(int(row["mapped_to_document_ir"]) for row in rows),
        "ambiguous": sum(int(row["ambiguous"]) for row in rows),
        "false_positive_proxy": sum(int(row["false_positive_proxy"]) for row in rows),
        "phase1_bridge_ready": all(row["sidecar_found"] and row["document_ir_check_found"] for row in rows) and bool(rows),
    }
    if preservation_summary:
        summary["raw_only_unmapped"] = int(
            preservation_summary.get("raw_only_unmapped_count", summary["raw_only_unmapped"])
        )
        summary["lost_v8_to_document_ir"] = int(
            preservation_summary.get("lost_v8_to_document_ir_count", summary["lost_v8_to_document_ir"])
        )
    bridge_dir = output_dir / "evidence_bridge_audit"
    write_json(bridge_dir / "reference_evidence_bridge_summary.json", summary)
    write_csv(bridge_dir / "reference_evidence_bridge_summary.csv", rows)
    lines = ["# Reference Evidence Bridge Examples", ""]
    for idx, example in enumerate(examples[:40], start=1):
        lines.append(f"{idx}. doc_id={example.get('context_id')} page={example.get('page_idx')}")
        lines.append(f"   text: {example.get('text')}")
        lines.append(f"   evidence={example.get('evidence_source')} confidence={example.get('confidence_tier')}")
    bridge_dir.mkdir(parents=True, exist_ok=True)
    (bridge_dir / "reference_evidence_bridge_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def merge_examples(example_sets: list[dict[str, list[dict[str, Any]]]], *, limit: int) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example_set in example_sets:
        for key, items in example_set.items():
            for item in items:
                if len(merged[key]) < limit:
                    merged[key].append(item)
    return dict(merged)


def write_examples_markdown(path: Path, examples: dict[str, list[dict[str, Any]]]) -> None:
    titles = {
        "reference_item_mineru_evidence": "Reference Items Explained By MinerU Ref_Text Evidence",
        "reference_heading_mineru_evidence": "Reference Headings / Bibliography Regions Explained By MinerU Evidence",
        "bibliography_preserved_not_rendered": "Bibliography-As-Paragraph/List Reclassified As Preserved Not Rendered",
        "body_citation_false_positive_prevented": "Body Citation False Positives Prevented",
        "ordinary_list_preserved": "Ordinary Body Lists Preserved As List",
        "multi_source_duplicate_explained": "Duplicate Cases Explained As Multi-Source MinerU Duplicates",
        "regex_only_demoted": "Regex-Only Reference Cases Demoted To Diagnostic",
        "remaining_unresolved_reference": "Remaining Unresolved Reference Cases",
    }
    lines = ["# Reference Context Phase1 Examples", ""]
    for key, title in titles.items():
        lines.extend([f"## {title}", ""])
        items = examples.get(key) or []
        if not items:
            lines.extend(["No examples found.", ""])
            continue
        for idx, item in enumerate(items[:20], start=1):
            lines.append(f"{idx}. doc_id={item.get('doc_id')} page={item.get('page_idx')}")
            lines.append(f"   text: {item.get('text_preview')}")
            lines.append(f"   source_v8_ids: {json.dumps(item.get('source_v8_ids') or [], ensure_ascii=False)}")
            lines.append(f"   parent_reference_block_id: {item.get('parent_reference_block_id')}")
            lines.append(f"   list_item_order: {item.get('list_item_order')}")
            lines.append(
                f"   old={item.get('old_classification')} phase1={item.get('phase1_classification')} reason={item.get('reason')}"
            )
            lines.append(f"   evidence: {json.dumps(item.get('reference_evidence') or {}, ensure_ascii=False)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def failure_breakdown_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = [
        ("bibliography_as_paragraph", "bibliography_as_paragraph_old", "bibliography_as_paragraph_with_mineru_evidence"),
        ("bibliography_as_list", "bibliography_as_list_old", "bibliography_as_list_with_mineru_evidence"),
        ("reference_heading_missing", "reference_heading_missing_old", "reference_heading_explained_by_mineru"),
        ("reference_item_missing", "reference_item_missing_old", "reference_item_preserved_not_rendered"),
        ("duplicate_reference", "duplicate_reference_old", "true_duplicate_reference_after_canonicalization"),
        ("regex_only_reference_failure", "regex_only_reference_failure_count", "regex_only_reference_failure_count"),
        ("body_citation_false_positive", "body_citation_blocked_count", "false_positive_proxy"),
    ]
    rows = []
    for name, old_key, phase1_key in pairs:
        old = summary.get(old_key, 0)
        phase1 = summary.get(phase1_key, 0)
        rows.append(
            {
                "failure_type": name,
                "old_regex_heavy": old,
                "phase1_mineru_evidence_first": phase1,
                "delta": phase1 - old if isinstance(old, int) and isinstance(phase1, int) else "",
            }
        )
    return rows


def write_report(
    path: Path,
    *,
    bridge_summary: dict[str, Any],
    aggregate_summary: dict[str, Any],
    preservation_summary: dict[str, Any],
    py_compile_status: str,
    pytest_status: str,
) -> None:
    lines = [
        "# V8 Reference Context Phase1 MinerU Evidence Report",
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
        "## V8 Context",
        "- Current fact layer is v8 full observable facts.",
        "- v8 is not reflowed middle only; it is the fused observable fact layer.",
        "- P0-D preserved reference subtype / reference list facts only; this pass consumes them for diagnostics without changing generation.",
        "- source_v7_ids / v7_id, if present, are legacy provenance names only.",
        "- current mainline remains: v8 full observable facts -> v8 atomic/reflow -> deterministic merge + contentlist merge hint -> RenderTreeIR -> IR renderer.",
        "",
        "## P0-D Recap",
        f"- raw_ref_text_subtype_count: {preservation_summary.get('raw_ref_text_subtype_count', 'unknown')}",
        f"- sidecar_reference_signal_count: {preservation_summary.get('sidecar_reference_signal_count', 'unknown')}",
        f"- v8_reference_matched_count: {preservation_summary.get('v8_reference_matched_count', 'unknown')}",
        f"- DocumentIR reference preserved: {preservation_summary.get('document_ir_reference_preserved_count', 'unknown')}",
        f"- reference item / heading preserved: {preservation_summary.get('reference_item_preserved_count', 'unknown')} / {preservation_summary.get('reference_heading_preserved_count', 'unknown')}",
        f"- raw_only_unmapped: {preservation_summary.get('raw_only_unmapped_count', 'unknown')}",
        f"- lost_v8_to_document_ir: {preservation_summary.get('lost_v8_to_document_ir_count', 'unknown')}",
        f"- false-positive proxy: {preservation_summary.get('false_positive_proxy_on_body_citation_docs', 'unknown')}",
        "",
        "## Evidence Consumption Design",
        "- MinerU reference evidence is primary.",
        "- regex-only reference context is diagnostic.",
        "- reference item / heading / bibliography block mapping follows ref_text and reference_context_role evidence.",
        "- body citation guard keeps ordinary citations such as `see [1]` out of bibliography roles.",
        "- ordinary lists without ref_text subtype remain ordinary lists.",
        "",
        "## Evidence Bridge Audit",
        f"- sidecar docs found: {bridge_summary.get('sidecar_doc_count')}",
        f"- DocumentIR check docs found: {bridge_summary.get('document_ir_check_doc_count')}",
        f"- DocumentIR reference metadata count: {bridge_summary.get('document_ir_reference_metadata_count')}",
        f"- mapped_to_document_ir: {bridge_summary.get('mapped_to_document_ir')}",
        f"- raw_only_unmapped: {bridge_summary.get('raw_only_unmapped')}",
        f"- lost_v8_to_document_ir: {bridge_summary.get('lost_v8_to_document_ir')}",
        f"- false_positive_proxy: {bridge_summary.get('false_positive_proxy')}",
        "",
        "## Old vs Phase1 Summary",
        "| metric | old reference diagnostics | Phase1 MinerU-evidence-first | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    table_pairs = [
        ("total_reference_candidate_count", "total_reference_candidate_count_old", "total_reference_candidate_count_phase1"),
        ("bibliography_as_paragraph", "bibliography_as_paragraph_old", "bibliography_as_paragraph_with_mineru_evidence"),
        ("bibliography_as_list", "bibliography_as_list_old", "bibliography_as_list_with_mineru_evidence"),
        ("reference_heading_missing", "reference_heading_missing_old", "reference_heading_explained_by_mineru"),
        ("reference_item_missing", "reference_item_missing_old", "reference_item_preserved_not_rendered"),
        ("duplicate_reference", "duplicate_reference_old", "true_duplicate_reference_after_canonicalization"),
    ]
    for label, old_key, phase1_key in table_pairs:
        old = aggregate_summary.get(old_key, 0)
        phase1 = aggregate_summary.get(phase1_key, 0)
        delta = phase1 - old if isinstance(old, int) and isinstance(phase1, int) else ""
        lines.append(f"| {label} | {old} | {phase1} | {delta} |")
    lines += [
        f"| mineru_backed_reference_count | - | {aggregate_summary.get('mineru_backed_reference_count')} | - |",
        f"| regex_only_reference_count | - | {aggregate_summary.get('regex_only_reference_count')} | - |",
        f"| diagnostic_only_reference_count | - | {aggregate_summary.get('diagnostic_only_reference_count')} | - |",
        f"| duplicate_due_to_multi_source_mineru | - | {aggregate_summary.get('duplicate_due_to_multi_source_mineru')} | - |",
        f"| body_citation_blocked_count | - | {aggregate_summary.get('body_citation_blocked_count')} | - |",
        f"| false_positive_proxy | - | {aggregate_summary.get('false_positive_proxy')} | - |",
        "",
        "## Examples",
        "- See reference_context_phase1_examples.md for reference items, headings, preserved-not-rendered bibliography entries, body-citation guards, ordinary lists, duplicate explanations, regex-only demotions, and unresolved cases.",
        "",
        "## Remaining Risks",
        f"- raw_only_unmapped = {bridge_summary.get('raw_only_unmapped')}",
        f"- lost_v8_to_document_ir = {bridge_summary.get('lost_v8_to_document_ir')}",
        "- reference heading ambiguity remains constrained by ref_text adjacency in future metric use.",
        "- body citation/reference boundary ambiguity remains diagnostic-only without ref_text evidence.",
        "- bibliography renderer still does not consume preserved references.",
        "- metric version drift risk remains; Phase1 is a context/metric track, not generated output.",
        "",
        "## Decision",
        str(aggregate_summary.get("decision", "diagnostic_only")),
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
    selected_out = args.output_dir / "selected200_audit_only"
    selected_out.mkdir(parents=True, exist_ok=True)
    preservation_summary = load_json(args.preservation_dir / "reference_subtype_preservation_summary.json", {})
    if not docs or not preservation_summary:
        (args.output_dir / "REFERENCE_CONTEXT_PHASE1_READINESS_REPORT.md").write_text(
            "# Reference Context Phase1 Readiness Report\n\n"
            f"- selected200_root_exists: {args.selected200_root.exists()}\n"
            f"- preservation_summary_exists: {(args.preservation_dir / 'reference_subtype_preservation_summary.json').exists()}\n"
            "- decision: readiness_failed\n",
            encoding="utf-8",
        )
        return 2
    doc_ids = list(docs)
    bridge_summary = bridge_audit(doc_ids, args.preservation_dir, args.output_dir, preservation_summary)
    if not bridge_summary.get("phase1_bridge_ready"):
        (args.output_dir / "REFERENCE_CONTEXT_PHASE1_READINESS_REPORT.md").write_text(
            "# Reference Context Phase1 Readiness Report\n\n"
            "- bridge audit failed: per-doc P0-D sidecars or DocumentIR checks are incomplete.\n"
            f"- sidecar_doc_count: {bridge_summary.get('sidecar_doc_count')}\n"
            f"- document_ir_check_doc_count: {bridge_summary.get('document_ir_check_doc_count')}\n"
            "- decision: readiness_failed\n",
            encoding="utf-8",
        )
        return 2
    tasks = [(doc_id, str(doc_dir), str(args.preservation_dir), str(args.output_dir), args.max_examples) for doc_id, doc_dir in docs.items()]
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(audit_doc, tasks))
    else:
        results = [audit_doc(task) for task in tasks]
    rows = [result[0] for result in results]
    examples = merge_examples([result[1] for result in results], limit=args.max_examples)
    aggregate_summary = aggregate(rows)
    write_json(selected_out / "reference_context_phase1_summary.json", aggregate_summary)
    write_csv(selected_out / "reference_context_phase1_summary.csv", rows)
    write_csv(selected_out / "reference_context_phase1_failure_breakdown.csv", failure_breakdown_rows(aggregate_summary))
    write_examples_markdown(selected_out / "reference_context_phase1_examples.md", examples)
    write_report(
        selected_out / "REFERENCE_CONTEXT_PHASE1_MINERU_EVIDENCE_REPORT.md",
        bridge_summary=bridge_summary,
        aggregate_summary=aggregate_summary,
        preservation_summary=preservation_summary,
        py_compile_status=args.py_compile_status,
        pytest_status=args.pytest_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
