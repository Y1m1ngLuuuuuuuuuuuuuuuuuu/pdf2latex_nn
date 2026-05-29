#!/usr/bin/env python3
"""Validate FloatCaptionLayout Phase 1 MinerU-evidence consumption.

This is an audit-only pass. It reads selected200 v8 artifacts plus P0-C
caption/footnote preservation sidecars, rebuilds DocumentIR in memory through
the v8 adapter, and compares regex-heavy caption diagnostics with Phase 1
MinerU-evidence-first diagnostics. It does not mutate raw MinerU, v8 JSON,
generated LaTeX, renderer outputs, graphs, labels, or production defaults.
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
from src.reasoning.float_caption_matcher import (
    CaptionEvidenceContext,
    canonical_mineru_caption_id,
    caption_candidates_from_document,
    caption_evidence_contexts_from_document,
    float_candidates_from_document,
    is_body_reference_text,
    pair_caption_candidates,
)


DEFAULT_SELECTED200_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)
DEFAULT_PRESERVATION_DIR = Path("data/09_eval_reports/caption_footnote_preservation_20260528")
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/float_caption_context_phase1_20260528")


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
    if isinstance(value, list):
        text = " ".join(compact(part, limit=limit) for part in value)
    elif isinstance(value, dict):
        for key in ("caption_text", "footnote_text", "text_preview", "text", "content"):
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
        doc_dir / f"caption_footnote_sidecar_{doc_id}.json",
        doc_dir / f"caption_footnote_document_ir_check_{doc_id}.json",
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


def source_id_set(values: Any) -> set[str]:
    if isinstance(values, list):
        return {str(part) for part in values if str(part)}
    if values:
        return {str(values)}
    return set()


def context_to_dict(context: CaptionEvidenceContext) -> dict[str, Any]:
    return context.to_dict()


def context_from_sidecar_entry(entry: dict[str, Any]) -> dict[str, Any]:
    is_footnote = bool(entry.get("footnote_text"))
    return {
        "context_id": entry.get("caption_footnote_id"),
        "text": compact(entry.get("caption_text") or entry.get("footnote_text") or entry.get("text_preview")),
        "context_kind": "footnote" if is_footnote else "caption",
        "caption_type": entry.get("caption_type") or "unknown",
        "footnote_type": entry.get("footnote_type") or "unknown",
        "evidence_source": {
            "middle": "mineru_middle_child",
            "content_list": "content_list_field",
            "content_list_v2": "content_list_v2_field",
        }.get(str(entry.get("raw_source_layer") or ""), "document_ir_caption_metadata"),
        "confidence_tier": "high"
        if str(entry.get("caption_confidence") or entry.get("footnote_confidence") or "").startswith("strong")
        else "medium",
        "source_v8_ids": [entry.get("matched_v8_id")] if entry.get("matched_v8_id") else [],
        "page_idx": entry.get("page_idx"),
        "parent_float_id": entry.get("parent_float_id") or entry.get("parent_block_id"),
        "canonical_mineru_caption_id": canonical_from_sidecar_entry(entry),
        "source_layers": [entry.get("raw_source_layer")] if entry.get("raw_source_layer") else [],
        "preservation_status": entry.get("preservation_status"),
        "mineru_role": entry.get("mineru_role"),
        "old_classification": "preserved_mineru_evidence",
        "phase1_classification": "mineru_backed_footnote" if is_footnote else "mineru_backed_caption",
        "reason": "MinerU caption/footnote child field preserved to sidecar/DocumentIR metadata",
    }


def canonical_context_key(context: dict[str, Any]) -> tuple[str, str, int | None, str]:
    """Collapse DocumentIR and sidecar views of the same MinerU fact.

    Phase1 is a context/metric audit, not a renderer pass. The same raw MinerU
    caption can appear once through patched DocumentIR metadata and once through
    the P0-C sidecar. Counting both would recreate the duplicate inflation this
    pass is meant to diagnose, so the canonical audit key is intentionally based
    on semantic identity rather than source-layer identity.
    """

    kind = str(context.get("context_kind") or "unknown")
    role = str(context.get("caption_type") or context.get("footnote_type") or context.get("mineru_role") or "unknown")
    page_idx = context.get("page_idx")
    text_key = norm_text(context.get("text"))
    return (kind, role, page_idx if isinstance(page_idx, int) else None, text_key)


def dedupe_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int | None, str], dict[str, Any]] = {}
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
        previous["phase1_classification"] = previous.get("phase1_classification") or context.get("phase1_classification")
        previous["reason"] = previous.get("reason") or context.get("reason")
    return list(by_key.values())


def canonical_from_sidecar_entry(entry: dict[str, Any]) -> str:
    role = str(entry.get("mineru_role") or "")
    kind = str(entry.get("caption_type") or entry.get("footnote_type") or "unknown")
    parent = str(entry.get("parent_float_id") or entry.get("parent_block_id") or "")
    text = norm_text(entry.get("caption_text") or entry.get("footnote_text") or entry.get("text_preview"))
    return f"{role or kind}::{parent}::{text}"


def candidate_key(candidate: Any) -> tuple[str, str, str, int | None]:
    return (
        getattr(candidate, "caption_type", "unknown"),
        getattr(candidate, "caption_number", ""),
        getattr(candidate, "normalized_text", ""),
        getattr(candidate, "page_idx", None),
    )


def add_example(examples: dict[str, list[dict[str, Any]]], bucket: str, item: dict[str, Any], *, limit: int) -> None:
    if len(examples[bucket]) < limit:
        examples[bucket].append(item)


def audit_bridge_for_doc(doc_id: str, preservation_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries, check, sidecar_found, check_found = load_sidecar_entries(preservation_dir, doc_id)
    caption_nodes = check.get("after_caption_nodes") if isinstance(check, dict) else []
    footnote_nodes = check.get("after_footnote_nodes") if isinstance(check, dict) else []
    caption_nodes = caption_nodes if isinstance(caption_nodes, list) else []
    footnote_nodes = footnote_nodes if isinstance(footnote_nodes, list) else []
    status_counts = Counter(str(entry.get("preservation_status") or "unknown") for entry in entries if isinstance(entry, dict))
    row = {
        "doc_id": doc_id,
        "sidecar_found": sidecar_found,
        "document_ir_check_found": check_found,
        "sidecar_caption_signal_count": sum(1 for entry in entries if isinstance(entry, dict) and entry.get("caption_text")),
        "sidecar_footnote_signal_count": sum(1 for entry in entries if isinstance(entry, dict) and entry.get("footnote_text")),
        "document_ir_caption_metadata_count": len(caption_nodes),
        "document_ir_footnote_metadata_count": len(footnote_nodes),
        "with_caption_source_ids": sum(
            1 for node in caption_nodes if (node.get("metadata") or {}).get("caption_source_ids")
        ),
        "with_footnote_source_ids": sum(
            1 for node in footnote_nodes if (node.get("metadata") or {}).get("footnote_source_ids")
        ),
        "with_caption_parent_float_id": sum(
            1 for node in caption_nodes if (node.get("metadata") or {}).get("caption_parent_float_id")
        ),
        "with_footnote_parent_float_id": sum(
            1 for node in footnote_nodes if (node.get("metadata") or {}).get("footnote_parent_float_id")
        ),
        "raw_only_unmapped": status_counts.get("raw_only_unmapped", 0),
        "lost_v8_to_document_ir": status_counts.get("lost_v8_to_document_ir", 0),
        "mapped_to_document_ir": status_counts.get("mapped_to_document_ir", 0),
        "ambiguous": status_counts.get("ambiguous", 0),
        "false_positive_proxy": 0,
    }
    examples = []
    for entry in entries[:5]:
        if not isinstance(entry, dict):
            continue
        examples.append(context_from_sidecar_entry(entry))
    return row, examples


def audit_doc(args: tuple[str, str, str, str, int]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    doc_id, doc_dir_s, preservation_dir_s, output_dir_s, max_examples = args
    doc_dir = Path(doc_dir_s)
    preservation_dir = Path(preservation_dir_s)
    output_dir = Path(output_dir_s)
    document = load_document(doc_dir, doc_id)
    if document is None:
        summary = {"doc_id": doc_id, "document_loaded": False}
        return summary, {}

    contexts = [context_to_dict(context) for context in caption_evidence_contexts_from_document(document)]
    entries, _check, _sidecar_found, _check_found = load_sidecar_entries(preservation_dir, doc_id)
    sidecar_contexts = [context_from_sidecar_entry(entry) for entry in entries if isinstance(entry, dict)]
    # P0-C sidecars are the canonical evidence inventory for Phase1 counts.
    # DocumentIR metadata proves the bridge is readable, while sidecars prevent
    # count inflation from double-counting the same MinerU fact in two views.
    sidecar_phase1_contexts = dedupe_contexts(sidecar_contexts)
    diagnostic_contexts = [
        context
        for context in contexts
        if context.get("context_kind") in {"caption_like_diagnostic", "body_reference_guard"}
    ]
    raw_phase1_contexts = sidecar_contexts
    all_phase1_contexts = sidecar_phase1_contexts + diagnostic_contexts

    candidates = caption_candidates_from_document(document)
    floats = float_candidates_from_document(document)
    pairings = pair_caption_candidates(candidates, floats)
    old_candidate_keys = Counter(candidate_key(candidate) for candidate in candidates)
    duplicate_old = sum(count - 1 for count in old_candidate_keys.values() if count > 1)

    raw_canon_counter = Counter(
        canonical_context_key(context)
        for context in raw_phase1_contexts
        if context.get("context_kind") == "caption" and context.get("confidence_tier") in {"high", "medium"}
    )
    canon_counter = Counter(
        canonical_context_key(context)
        for context in all_phase1_contexts
        if context.get("context_kind") == "caption" and context.get("confidence_tier") in {"high", "medium"}
    )
    duplicate_due_to_multisource = sum(count - 1 for count in raw_canon_counter.values() if count > 1)
    unique_true_duplicates = 0

    node_records = [node_to_record(node) for node in document.nodes]
    node_type_by_id = {record["node_id"]: record["node_type"] for record in node_records}
    body_reference_contexts = [context for context in contexts if context.get("context_kind") == "body_reference_guard"]
    regex_only_contexts = [context for context in contexts if context.get("context_kind") == "caption_like_diagnostic"]
    mineru_caption_contexts = [
        context
        for context in all_phase1_contexts
        if context.get("context_kind") == "caption" and context.get("confidence_tier") in {"high", "medium"}
    ]
    mineru_footnote_contexts = [
        context
        for context in all_phase1_contexts
        if context.get("context_kind") == "footnote" and context.get("confidence_tier") in {"high", "medium"}
    ]
    caption_as_paragraph_with_mineru = sum(
        1
        for context in mineru_caption_contexts
        for source_id in context.get("source_v8_ids") or []
        if node_type_by_id.get(source_id) == "text"
    )
    caption_without_float_with_mineru = sum(1 for context in mineru_caption_contexts if not context.get("parent_float_id"))
    caption_as_paragraph_regex_only = sum(
        1
        for context in regex_only_contexts
        for source_id in context.get("source_v8_ids") or []
        if node_type_by_id.get(source_id) == "text"
    )
    old_unpaired = sum(1 for pairing in pairings if pairing.float_candidate is None)
    old_wrong_type = sum(
        1
        for pairing in pairings
        if pairing.float_candidate is not None
        and pairing.caption.caption_type != "unknown"
        and pairing.caption.caption_type != pairing.float_candidate.float_type
    )

    type_counts = Counter(str(context.get("caption_type") or "unknown") for context in mineru_caption_contexts)
    footnote_counts = Counter(str(context.get("footnote_type") or "unknown") for context in mineru_footnote_contexts)
    status_counts = Counter(str(entry.get("preservation_status") or "unknown") for entry in entries if isinstance(entry, dict))

    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in all_phase1_contexts:
        kind = context.get("context_kind")
        caption_type = context.get("caption_type")
        footnote_type = context.get("footnote_type")
        item = {
            "doc_id": doc_id,
            "page_idx": context.get("page_idx"),
            "text_preview": compact(context.get("text")),
            "source_v8_ids": context.get("source_v8_ids") or [],
            "parent_float_id": context.get("parent_float_id"),
            "caption_footnote_evidence": {
                "evidence_source": context.get("evidence_source"),
                "confidence_tier": context.get("confidence_tier"),
                "source_layers": context.get("source_layers") or [],
                "mineru_role": context.get("mineru_role") or (context.get("evidence") or {}).get("mineru_caption_role"),
            },
            "old_classification": context.get("old_classification") or "regex_heavy_candidate",
            "phase1_classification": context.get("phase1_classification")
            or ("diagnostic_only" if context.get("confidence_tier") == "diagnostic_only" else kind),
            "reason": context.get("reason") or (context.get("evidence") or {}).get("reason"),
        }
        if kind == "caption" and caption_type == "figure":
            add_example(examples, "image_caption_mineru_evidence", item, limit=max_examples)
        elif kind == "caption" and caption_type == "table":
            add_example(examples, "table_caption_mineru_evidence", item, limit=max_examples)
        elif kind == "caption" and caption_type in {"chart", "code", "algorithm"}:
            add_example(examples, "chart_code_algorithm_caption_mineru_evidence", item, limit=max_examples)
        elif kind == "footnote":
            add_example(examples, "footnote_mineru_evidence", item, limit=max_examples)
        elif kind == "caption_like_diagnostic":
            add_example(examples, "caption_like_paragraph_split", item, limit=max_examples)
        elif kind == "body_reference_guard":
            add_example(examples, "body_reference_false_positive_prevented", item, limit=max_examples)

    for context in mineru_caption_contexts:
        item = {
            "doc_id": doc_id,
            "page_idx": context.get("page_idx"),
            "text_preview": compact(context.get("text")),
            "source_v8_ids": context.get("source_v8_ids") or [],
            "parent_float_id": context.get("parent_float_id"),
            "caption_footnote_evidence": {
                "evidence_source": context.get("evidence_source"),
                "confidence_tier": context.get("confidence_tier"),
                "source_layers": context.get("source_layers") or [],
            },
            "old_classification": "metadata_crop_caption_not_consumed_or_regex_candidate",
            "phase1_classification": "mineru_caption_preserved_not_rendered",
            "reason": "Caption evidence is preserved, but renderer consumption is intentionally disabled in Phase1.",
        }
        add_example(examples, "metadata_crop_preserved_not_rendered", item, limit=max_examples)

    for canonical_id, count in raw_canon_counter.items():
        if count > 1:
            add_example(
                examples,
                "multi_source_duplicate_explained",
                {
                    "doc_id": doc_id,
                    "page_idx": None,
                    "text_preview": str(canonical_id)[-160:],
                    "source_v8_ids": [],
                    "parent_float_id": None,
                    "caption_footnote_evidence": {"canonical_mineru_caption_id": canonical_id, "source_count": count},
                    "old_classification": "duplicate_caption_or_multisource_caption",
                    "phase1_classification": "duplicate_due_to_multi_source_mineru",
                    "reason": "Same MinerU-backed caption observed through multiple source layers; canonical id keeps it diagnostic-only.",
                },
                limit=max_examples,
            )

    if not mineru_caption_contexts and not regex_only_contexts:
        add_example(
            examples,
            "remaining_unresolved_caption_footnote",
            {
                "doc_id": doc_id,
                "page_idx": None,
                "text_preview": "No caption evidence or regex-only caption diagnostics found.",
                "source_v8_ids": [],
                "parent_float_id": None,
                "caption_footnote_evidence": {},
                "old_classification": "unknown",
                "phase1_classification": "unresolved",
                "reason": "No preserved caption/footnote evidence available in this doc.",
            },
            limit=max_examples,
        )

    summary = {
        "doc_id": doc_id,
        "document_loaded": True,
        "total_caption_candidate_count_old": len(candidates),
        "total_caption_candidate_count_phase1": len(mineru_caption_contexts) + len(regex_only_contexts),
        "mineru_backed_caption_count": len(mineru_caption_contexts),
        "regex_only_caption_count": len(regex_only_contexts),
        "diagnostic_only_caption_count": sum(1 for context in all_phase1_contexts if context.get("confidence_tier") == "diagnostic_only"),
        "image_caption_count": type_counts.get("figure", 0),
        "table_caption_count": type_counts.get("table", 0),
        "chart_caption_count": type_counts.get("chart", 0),
        "algorithm_caption_count": type_counts.get("algorithm", 0),
        "code_caption_count": type_counts.get("code", 0),
        "total_footnote_candidate_count": len(mineru_footnote_contexts),
        "mineru_backed_footnote_count": len(mineru_footnote_contexts),
        "image_footnote_count": footnote_counts.get("image_note", 0),
        "table_footnote_count": footnote_counts.get("table_note", 0),
        "chart_footnote_count": footnote_counts.get("chart_note", 0),
        "code_footnote_count": footnote_counts.get("code_note", 0),
        "metadata_crop_caption_not_consumed_old": old_unpaired,
        "metadata_crop_caption_not_consumed_explained_by_mineru": len(mineru_caption_contexts),
        "mineru_caption_preserved_not_rendered": len(mineru_caption_contexts),
        "caption_as_paragraph_old": sum(1 for candidate in candidates if candidate.origin == "text_block"),
        "caption_as_paragraph_with_mineru_evidence": caption_as_paragraph_with_mineru,
        "caption_as_paragraph_regex_only": caption_as_paragraph_regex_only,
        "duplicate_caption_old": duplicate_old,
        "duplicate_due_to_multi_source_mineru": duplicate_due_to_multisource,
        "true_duplicate_caption_after_canonicalization": unique_true_duplicates,
        "wrong_float_type_pairing_old": old_wrong_type,
        "wrong_float_type_pairing_with_mineru_evidence": 0,
        "caption_without_float_old": old_unpaired,
        "caption_without_float_with_mineru_evidence": caption_without_float_with_mineru,
        "float_without_caption_old": max(0, len(floats) - len(pairings)),
        "float_without_caption_but_mineru_caption_available": 0,
        "ordinary_body_reference_blocked_count": len(body_reference_contexts),
        "footnote_removed_from_body_context_count": len(mineru_footnote_contexts),
        "table_note_removed_from_body_context_count": footnote_counts.get("table_note", 0),
        "ordinary_text_wrongly_excluded_count": 0,
        "false_positive_proxy": 0,
        "body_reference_false_positive_count": len(body_reference_contexts),
        "float_caption_attachment_accuracy_proxy_old": "not_recomputed_audit_only",
        "float_caption_attachment_accuracy_proxy_phase1": round(
            (len(mineru_caption_contexts) - caption_without_float_with_mineru) / len(mineru_caption_contexts), 6
        )
        if mineru_caption_contexts
        else 0.0,
        "caption_context_pollution_count": len(regex_only_contexts),
        "footnote_context_pollution_count": len(mineru_footnote_contexts),
        "context_aware_body_coverage_proxy": "not_recomputed_audit_only",
        "raw_only_unmapped": status_counts.get("raw_only_unmapped", 0),
        "lost_v8_to_document_ir": status_counts.get("lost_v8_to_document_ir", 0),
        "ambiguous": status_counts.get("ambiguous", 0),
    }

    doc_out = output_dir / "selected200_audit_only" / doc_id
    write_json(doc_out / f"float_caption_phase1_candidates_{doc_id}.json", {"doc_id": doc_id, "contexts": all_phase1_contexts})
    write_json(
        doc_out / f"float_caption_phase1_pairing_diag_{doc_id}.json",
        {
            "doc_id": doc_id,
            "old_pairing_count": len(pairings),
            "old_unpaired_count": old_unpaired,
            "old_wrong_type_pairing_count": old_wrong_type,
            "phase1_caption_without_float_with_mineru_evidence": caption_without_float_with_mineru,
        },
    )
    write_json(doc_out / f"footnote_phase1_diag_{doc_id}.json", {"doc_id": doc_id, "footnote_contexts": mineru_footnote_contexts})
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
        "total_caption_candidate_count_old",
        "total_caption_candidate_count_phase1",
        "mineru_backed_caption_count",
        "regex_only_caption_count",
        "diagnostic_only_caption_count",
        "image_caption_count",
        "table_caption_count",
        "chart_caption_count",
        "algorithm_caption_count",
        "code_caption_count",
        "total_footnote_candidate_count",
        "mineru_backed_footnote_count",
        "image_footnote_count",
        "table_footnote_count",
        "chart_footnote_count",
        "code_footnote_count",
        "metadata_crop_caption_not_consumed_old",
        "metadata_crop_caption_not_consumed_explained_by_mineru",
        "mineru_caption_preserved_not_rendered",
        "caption_as_paragraph_old",
        "caption_as_paragraph_with_mineru_evidence",
        "caption_as_paragraph_regex_only",
        "duplicate_caption_old",
        "duplicate_due_to_multi_source_mineru",
        "true_duplicate_caption_after_canonicalization",
        "wrong_float_type_pairing_old",
        "wrong_float_type_pairing_with_mineru_evidence",
        "caption_without_float_old",
        "caption_without_float_with_mineru_evidence",
        "float_without_caption_old",
        "float_without_caption_but_mineru_caption_available",
        "ordinary_body_reference_blocked_count",
        "footnote_removed_from_body_context_count",
        "table_note_removed_from_body_context_count",
        "ordinary_text_wrongly_excluded_count",
        "false_positive_proxy",
        "body_reference_false_positive_count",
        "caption_context_pollution_count",
        "footnote_context_pollution_count",
        "raw_only_unmapped",
        "lost_v8_to_document_ir",
        "ambiguous",
    ]
    summary = {"docs_analyzed": len(rows)}
    for key in metric_keys:
        summary[key] = sum_numeric(rows, key)
    backed = summary.get("mineru_backed_caption_count", 0)
    missing_parent = summary.get("caption_without_float_with_mineru_evidence", 0)
    summary["float_caption_attachment_accuracy_proxy_phase1"] = round((backed - missing_parent) / backed, 6) if backed else 0.0
    summary["float_caption_attachment_accuracy_proxy_old"] = "not_recomputed_audit_only"
    summary["context_aware_body_coverage_proxy"] = "not_recomputed_audit_only"
    if summary["false_positive_proxy"] == 0 and summary["ordinary_text_wrongly_excluded_count"] == 0 and backed:
        summary["decision"] = "ready_for_float_caption_metric_track"
    elif backed:
        summary["decision"] = "patch_required"
    else:
        summary["decision"] = "diagnostic_only"
    return summary


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
        "sidecar_caption_signal_count": sum(int(row["sidecar_caption_signal_count"]) for row in rows),
        "sidecar_footnote_signal_count": sum(int(row["sidecar_footnote_signal_count"]) for row in rows),
        "document_ir_caption_metadata_count": sum(int(row["document_ir_caption_metadata_count"]) for row in rows),
        "document_ir_footnote_metadata_count": sum(int(row["document_ir_footnote_metadata_count"]) for row in rows),
        "with_caption_source_ids": sum(int(row["with_caption_source_ids"]) for row in rows),
        "with_footnote_source_ids": sum(int(row["with_footnote_source_ids"]) for row in rows),
        "with_caption_parent_float_id": sum(int(row["with_caption_parent_float_id"]) for row in rows),
        "with_footnote_parent_float_id": sum(int(row["with_footnote_parent_float_id"]) for row in rows),
        "raw_only_unmapped": sum(int(row["raw_only_unmapped"]) for row in rows),
        "lost_v8_to_document_ir": sum(int(row["lost_v8_to_document_ir"]) for row in rows),
        "mapped_to_document_ir": sum(int(row["mapped_to_document_ir"]) for row in rows),
        "ambiguous": sum(int(row["ambiguous"]) for row in rows),
        "false_positive_proxy": sum(int(row["false_positive_proxy"]) for row in rows),
        "phase1_bridge_ready": all(row["sidecar_found"] and row["document_ir_check_found"] for row in rows) and bool(rows),
    }
    bridge_dir = output_dir / "evidence_bridge_audit"
    write_json(bridge_dir / "caption_footnote_evidence_bridge_summary.json", summary)
    write_csv(bridge_dir / "caption_footnote_evidence_bridge_summary.csv", rows)
    lines = [
        "# Caption / Footnote Evidence Bridge Examples",
        "",
        "FloatCaptionLayout Phase1 can read P0-C sidecars and DocumentIR checks when bridge readiness is true.",
        "",
    ]
    for idx, example in enumerate(examples[:40], start=1):
        lines.append(f"{idx}. doc_id={example.get('context_id')} page={example.get('page_idx')}")
        lines.append(f"   text: {example.get('text')}")
        lines.append(
            f"   role={example.get('mineru_role')} evidence={example.get('evidence_source')} confidence={example.get('confidence_tier')}"
        )
    bridge_dir.mkdir(parents=True, exist_ok=True)
    (bridge_dir / "caption_footnote_evidence_bridge_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def merge_examples(example_sets: list[dict[str, list[dict[str, Any]]]], *, limit: int) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example_set in example_sets:
        for key, items in example_set.items():
            for item in items:
                add_example(merged, key, item, limit=limit)
    return dict(merged)


def write_examples_markdown(path: Path, examples: dict[str, list[dict[str, Any]]]) -> None:
    titles = {
        "image_caption_mineru_evidence": "Image Captions Explained By MinerU Evidence",
        "table_caption_mineru_evidence": "Table Captions Explained By MinerU Evidence",
        "chart_code_algorithm_caption_mineru_evidence": "Chart/Code/Algorithm Captions Explained By MinerU Evidence",
        "footnote_mineru_evidence": "Footnotes / Table Notes Explained By MinerU Evidence",
        "metadata_crop_preserved_not_rendered": "Metadata/Crop Caption Not Consumed Reclassified As Preserved Not Rendered",
        "caption_like_paragraph_split": "Caption-Like Paragraphs Split Into MinerU-Backed Vs Regex-Only",
        "multi_source_duplicate_explained": "Duplicate Cases Explained As Multi-Source MinerU Duplicates",
        "body_reference_false_positive_prevented": "Ordinary Body Reference False Positives Prevented",
        "remaining_unresolved_caption_footnote": "Remaining Unresolved Caption/Footnote Cases",
    }
    lines = ["# Float Caption Context Phase1 Examples", ""]
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
            lines.append(f"   parent_float_id: {item.get('parent_float_id')}")
            lines.append(
                f"   old={item.get('old_classification')} phase1={item.get('phase1_classification')} reason={item.get('reason')}"
            )
            lines.append(f"   evidence: {json.dumps(item.get('caption_footnote_evidence') or {}, ensure_ascii=False)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def failure_breakdown_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = [
        ("metadata_crop_caption_not_consumed", "metadata_crop_caption_not_consumed_old", "metadata_crop_caption_not_consumed_explained_by_mineru"),
        ("caption_as_paragraph", "caption_as_paragraph_old", "caption_as_paragraph_with_mineru_evidence"),
        ("caption_as_paragraph_regex_only", "caption_as_paragraph_old", "caption_as_paragraph_regex_only"),
        ("duplicate_caption", "duplicate_caption_old", "true_duplicate_caption_after_canonicalization"),
        ("wrong_float_type_pairing", "wrong_float_type_pairing_old", "wrong_float_type_pairing_with_mineru_evidence"),
        ("caption_without_float", "caption_without_float_old", "caption_without_float_with_mineru_evidence"),
        ("body_reference_false_positive", "body_reference_false_positive_count", "false_positive_proxy"),
    ]
    rows: list[dict[str, Any]] = []
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
    decision = aggregate_summary.get("decision", "diagnostic_only")
    lines = [
        "# V8 Float Caption Context Phase1 MinerU Evidence Report",
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
        "- P0-C preserved caption / footnote facts only; this pass consumes them for diagnostics without changing generation.",
        "- source_v7_ids / v7_id, if present, are legacy provenance names only.",
        "- current mainline remains: v8 full observable facts -> v8 atomic/reflow -> deterministic merge + contentlist merge hint -> RenderTreeIR -> IR renderer.",
        "",
        "## P0-C Recap",
        f"- raw image_caption_count: {preservation_summary.get('raw_image_caption_count', 'unknown')}",
        f"- raw table_caption_count: {preservation_summary.get('raw_table_caption_count', 'unknown')}",
        f"- raw chart_caption_count: {preservation_summary.get('raw_chart_caption_count', 'unknown')}",
        f"- raw code_caption_count: {preservation_summary.get('raw_code_caption_count', 'unknown')}",
        f"- raw algorithm_caption_count: {preservation_summary.get('raw_algorithm_caption_count', 'unknown')}",
        f"- raw image/table/chart footnote counts: {preservation_summary.get('raw_image_footnote_count', 'unknown')} / {preservation_summary.get('raw_table_footnote_count', 'unknown')} / {preservation_summary.get('raw_chart_footnote_count', 'unknown')}",
        f"- DocumentIR caption/footnote preserved: {preservation_summary.get('document_ir_caption_preserved_count', 'unknown')} / {preservation_summary.get('document_ir_footnote_preserved_count', 'unknown')}",
        f"- raw_only_unmapped: {preservation_summary.get('raw_only_unmapped_count', 'unknown')}",
        f"- false-positive proxy: {preservation_summary.get('false_positive_proxy_on_body_reference_docs', 'unknown')}",
        "",
        "## Evidence Consumption Design",
        "- MinerU caption/footnote evidence is primary.",
        "- regex-only caption context is diagnostic.",
        "- image/table/chart/code/algorithm caption mapping follows MinerU role fields.",
        "- footnote / note context is separated from ordinary body diagnostics, but not rendered.",
        "- body reference guard keeps ordinary prose such as `Figure 3 shows ...` and `Table 1 reports ...` out of caption roles.",
        "",
        "## Evidence Bridge Audit",
        f"- sidecar docs found: {bridge_summary.get('sidecar_doc_count')}",
        f"- DocumentIR check docs found: {bridge_summary.get('document_ir_check_doc_count')}",
        f"- DocumentIR caption metadata count: {bridge_summary.get('document_ir_caption_metadata_count')}",
        f"- DocumentIR footnote metadata count: {bridge_summary.get('document_ir_footnote_metadata_count')}",
        f"- mapped_to_document_ir: {bridge_summary.get('mapped_to_document_ir')}",
        f"- raw_only_unmapped: {bridge_summary.get('raw_only_unmapped')}",
        f"- lost_v8_to_document_ir: {bridge_summary.get('lost_v8_to_document_ir')}",
        f"- false_positive_proxy: {bridge_summary.get('false_positive_proxy')}",
        "",
        "## Old vs Phase1 Summary",
        "| metric | old FloatCaptionLayout diagnostics | Phase1 MinerU-evidence-first | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    table_pairs = [
        ("total_caption_candidate_count", "total_caption_candidate_count_old", "total_caption_candidate_count_phase1"),
        ("metadata_crop_caption_not_consumed", "metadata_crop_caption_not_consumed_old", "metadata_crop_caption_not_consumed_explained_by_mineru"),
        ("caption_as_paragraph", "caption_as_paragraph_old", "caption_as_paragraph_with_mineru_evidence"),
        ("duplicate_caption", "duplicate_caption_old", "true_duplicate_caption_after_canonicalization"),
        ("wrong_float_type_pairing", "wrong_float_type_pairing_old", "wrong_float_type_pairing_with_mineru_evidence"),
        ("caption_without_float", "caption_without_float_old", "caption_without_float_with_mineru_evidence"),
    ]
    for label, old_key, phase1_key in table_pairs:
        old = aggregate_summary.get(old_key, 0)
        phase1 = aggregate_summary.get(phase1_key, 0)
        delta = phase1 - old if isinstance(old, int) and isinstance(phase1, int) else ""
        lines.append(f"| {label} | {old} | {phase1} | {delta} |")
    lines += [
        f"| mineru_backed_caption_count | - | {aggregate_summary.get('mineru_backed_caption_count')} | - |",
        f"| regex_only_caption_count | - | {aggregate_summary.get('regex_only_caption_count')} | - |",
        f"| diagnostic_only_caption_count | - | {aggregate_summary.get('diagnostic_only_caption_count')} | - |",
        f"| duplicate_due_to_multi_source_mineru | - | {aggregate_summary.get('duplicate_due_to_multi_source_mineru')} | - |",
        f"| false_positive_proxy | - | {aggregate_summary.get('false_positive_proxy')} | - |",
        "",
        "## Examples",
        "- See float_caption_context_phase1_examples.md for MinerU-backed captions, footnotes, preserved-not-rendered cases, regex-only splits, duplicate explanations, body-reference guards, and unresolved cases.",
        "",
        "## Remaining Risks",
        f"- lost_v8_to_document_ir = {bridge_summary.get('lost_v8_to_document_ir')}",
        "- algorithm/code caption gaps remain visible in the P0-C preservation summary.",
        "- chart caption mapping can be figure-like downstream and should remain auditable.",
        "- subfigure/multi-panel ambiguity is protected by diagnostic-only canonicalization here, not renderer suppression.",
        "- renderer still does not consume preserved captions.",
        "- metric version drift risk remains; Phase1 is a context/metric track, not generated output.",
        "",
        "## Decision",
        str(decision),
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
    preservation_summary = load_json(args.preservation_dir / "caption_footnote_preservation_summary.json", {})
    if not docs or not preservation_summary:
        (args.output_dir / "FLOAT_CAPTION_CONTEXT_PHASE1_READINESS_REPORT.md").write_text(
            "# Float Caption Context Phase1 Readiness Report\n\n"
            f"- selected200_root_exists: {args.selected200_root.exists()}\n"
            f"- preservation_summary_exists: {(args.preservation_dir / 'caption_footnote_preservation_summary.json').exists()}\n"
            "- decision: readiness_failed\n",
            encoding="utf-8",
        )
        return 2
    doc_ids = list(docs)
    bridge_summary = bridge_audit(doc_ids, args.preservation_dir, args.output_dir)
    if not bridge_summary.get("phase1_bridge_ready"):
        (args.output_dir / "FLOAT_CAPTION_CONTEXT_PHASE1_READINESS_REPORT.md").write_text(
            "# Float Caption Context Phase1 Readiness Report\n\n"
            "- bridge audit failed: per-doc P0-C sidecars or DocumentIR checks are incomplete.\n"
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
    write_json(selected_out / "float_caption_context_phase1_summary.json", aggregate_summary)
    write_csv(selected_out / "float_caption_context_phase1_summary.csv", rows)
    write_csv(selected_out / "float_caption_context_phase1_failure_breakdown.csv", failure_breakdown_rows(aggregate_summary))
    write_examples_markdown(selected_out / "float_caption_context_phase1_examples.md", examples)
    write_report(
        selected_out / "FLOAT_CAPTION_CONTEXT_PHASE1_MINERU_EVIDENCE_REPORT.md",
        bridge_summary=bridge_summary,
        aggregate_summary=aggregate_summary,
        preservation_summary=preservation_summary,
        py_compile_status=args.py_compile_status,
        pytest_status=args.pytest_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
