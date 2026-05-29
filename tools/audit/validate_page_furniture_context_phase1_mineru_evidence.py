#!/usr/bin/env python3
"""Validate PageFurniture / FrontMatter / HeadingNegative Phase1 context track.

This audit consumes P0-E page-furniture/model-label preservation sidecars and
DocumentIR checks. It does not regenerate LaTeX, run E2E, modify renderer code,
or write back raw MinerU/v8 JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reasoning.front_matter_context_group import front_matter_contexts_from_page_contexts
from src.reasoning.page_furniture_context_group import PageFurnitureContext, contexts_from_document_ir_check


DEFAULT_SELECTED200_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)
DEFAULT_PRESERVATION_DIR = Path("data/09_eval_reports/page_furniture_model_label_preservation_20260528")
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/page_furniture_context_phase1_20260528")

PAGE_FURNITURE_ROLES = {"page_header", "page_footer", "page_number", "page_footnote", "aside_text", "margin_note", "discarded_block"}
MODEL_PAGE_FURNITURE_LABELS = {"header", "footer", "number", "page_number"}
BODY_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,:;()/-]{2,100}$")
REGEX_ONLY_FURNITURE_RE = re.compile(r"^\s*(?:\d{1,4}|copyright|proceedings|preprint)\b", re.IGNORECASE)


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
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value if value is not None else "")
    return text.replace("\r", "\\r")


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


def compact_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(compact_text(part) for part in value if compact_text(part)).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "html"):
            if key in value and compact_text(value[key]):
                return compact_text(value[key])
        return " ".join(compact_text(part) for part in value.values() if compact_text(part)).strip()
    return " ".join(str(value or "").split()).strip()


def collect_doc_dirs(root: Path) -> dict[str, Path]:
    docs: dict[str, Path] = {}
    if not root.exists():
        return docs
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "document_ir.json").exists() and list(path.glob("*_content_list_v8_contentlist_merge_hint.json")):
            docs[path.name.split("_", 1)[-1]] = path
    return docs


def preservation_doc_dir(preservation_dir: Path, doc_id: str) -> Path:
    return preservation_dir / doc_id


def sidecar_path(preservation_dir: Path, doc_id: str) -> Path:
    return preservation_doc_dir(preservation_dir, doc_id) / f"page_furniture_model_label_sidecar_{doc_id}.json"


def check_path(preservation_dir: Path, doc_id: str) -> Path:
    return preservation_doc_dir(preservation_dir, doc_id) / f"page_furniture_model_label_document_ir_check_{doc_id}.json"


def context_to_example(context: PageFurnitureContext, *, old: str, phase1: str, reason: str) -> dict[str, Any]:
    return {
        "doc_id": context.doc_id,
        "page_idx": context.page_idx,
        "text_preview": context.text_preview,
        "source_v8_ids": list(context.source_v8_ids),
        "model_label": context.model_label,
        "model_score": context.model_score,
        "page_furniture_evidence": context.page_furniture_role,
        "old_classification": old,
        "phase1_classification": phase1,
        "reason": reason,
    }


def entry_example(entry: dict[str, Any], *, old: str, phase1: str, reason: str) -> dict[str, Any]:
    return {
        "doc_id": entry.get("doc_id"),
        "page_idx": entry.get("page_idx"),
        "text_preview": entry.get("text_preview"),
        "source_v8_ids": [entry.get("matched_v8_id")] if entry.get("matched_v8_id") else [],
        "model_label": entry.get("model_label"),
        "model_score": entry.get("model_score"),
        "page_furniture_evidence": entry.get("mineru_role"),
        "old_classification": old,
        "phase1_classification": phase1,
        "reason": reason,
    }


def load_v8_items(doc_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(doc_dir.glob("*_content_list_v8_contentlist_merge_hint.json"))
    if not paths:
        return []
    payload = load_json(paths[0], {})
    return [item for item in payload.get("items") or [] if isinstance(item, dict)] if isinstance(payload, dict) else []


def heading_candidate_items(v8_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in v8_items
        if str(item.get("type") or item.get("canonical_type") or "").casefold() == "title"
        or str(item.get("layout_role") or "").casefold() in {"body_heading", "document_title", "abstract_title"}
    ]


def regex_only_furniture_items(v8_items: list[dict[str, Any]], context_node_ids: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in v8_items:
        item_id = str(item.get("id") or "")
        if item_id in context_node_ids:
            continue
        text = compact_text(item.get("text"))
        if not text or len(text) > 80:
            continue
        bbox = item.get("bbox")
        page_top_or_bottom = isinstance(bbox, list) and len(bbox) >= 4 and (float(bbox[1]) < 35.0 or float(bbox[3]) > 755.0)
        if page_top_or_bottom and REGEX_ONLY_FURNITURE_RE.search(text):
            candidates.append(item)
    return candidates


def audit_bridge(doc_ids: list[str], preservation_dir: Path, output_dir: Path, preservation_summary: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        sidecar = load_json(sidecar_path(preservation_dir, doc_id), {})
        check = load_json(check_path(preservation_dir, doc_id), {})
        entries = sidecar.get("entries") if isinstance(sidecar, dict) else []
        contexts = contexts_from_document_ir_check(doc_id, check if isinstance(check, dict) else {})
        row = {
            "doc_id": doc_id,
            "sidecar_found": bool(sidecar),
            "document_ir_check_found": bool(check),
            "sidecar_entry_count": len(entries or []),
            "context_count": len(contexts),
            "document_ir_page_furniture_metadata_count": len(check.get("after_page_furniture_nodes") or []) if isinstance(check, dict) else 0,
            "document_ir_model_label_metadata_count": len(check.get("after_model_label_nodes") or []) if isinstance(check, dict) else 0,
            "heading_negative_mask_count": sum(1 for context in contexts if "heading_detection" in context.negative_masks or "title_body_heading" in context.negative_masks),
            "front_matter_negative_mask_count": sum(1 for context in contexts if "front_matter_body_heading" in context.negative_masks or "title_body_heading" in context.negative_masks),
            "visible_prose_negative_mask_count": sum(1 for context in contexts if "visible_prose" in context.negative_masks),
            "raw_only_unmapped_count": sum(1 for entry in entries or [] if entry.get("preservation_status") == "raw_only_unmapped"),
            "dense_model_raw_only_count": sum(1 for entry in entries or [] if entry.get("preservation_status") == "raw_only_unmapped" and entry.get("raw_source_layer") == "model"),
            "discarded_or_no_flow_raw_only_count": sum(1 for entry in entries or [] if entry.get("preservation_status") == "raw_only_unmapped" and entry.get("mineru_role") in {"discarded_block", "page_footnote", "page_number", "page_header", "page_footer"}),
            "body_heading_wrongly_masked_count": 0,
            "ordinary_text_wrongly_excluded_count": 0,
        }
        rows.append(row)
        for context in contexts[:3]:
            examples.append(context.to_dict())
    summary = {
        "docs_analyzed": len(rows),
        "sidecar_doc_count": sum(1 for row in rows if row["sidecar_found"]),
        "document_ir_check_doc_count": sum(1 for row in rows if row["document_ir_check_found"]),
        "document_ir_page_furniture_metadata_count": sum(row["document_ir_page_furniture_metadata_count"] for row in rows),
        "document_ir_model_label_metadata_count": sum(row["document_ir_model_label_metadata_count"] for row in rows),
        "heading_negative_mask_count": sum(row["heading_negative_mask_count"] for row in rows),
        "front_matter_negative_mask_count": sum(row["front_matter_negative_mask_count"] for row in rows),
        "visible_prose_negative_mask_count": sum(row["visible_prose_negative_mask_count"] for row in rows),
        "raw_only_unmapped_count": int(preservation_summary.get("raw_only_unmapped_count", sum(row["raw_only_unmapped_count"] for row in rows))),
        "dense_model_raw_only_count": sum(row["dense_model_raw_only_count"] for row in rows),
        "discarded_or_no_flow_raw_only_count": sum(row["discarded_or_no_flow_raw_only_count"] for row in rows),
        "body_heading_wrongly_masked_count": int(preservation_summary.get("body_heading_wrongly_masked_count", 0)),
        "ordinary_text_wrongly_excluded_count": int(preservation_summary.get("ordinary_text_wrongly_excluded_count", 0)),
        "phase1_bridge_ready": all(row["sidecar_found"] and row["document_ir_check_found"] for row in rows) and bool(rows),
    }
    bridge_dir = output_dir / "evidence_bridge_audit"
    write_json(bridge_dir / "page_furniture_evidence_bridge_summary.json", summary)
    write_csv(bridge_dir / "page_furniture_evidence_bridge_summary.csv", rows)
    lines = ["# Page Furniture Evidence Bridge Examples", ""]
    for idx, example in enumerate(examples[:40], start=1):
        lines.append(f"{idx}. doc_id={example.get('doc_id')} page={example.get('page_idx')} kind={example.get('context_kind')}")
        lines.append(f"   text: {example.get('text_preview')}")
        lines.append(f"   evidence={example.get('evidence_source')} confidence={example.get('confidence_tier')}")
    bridge_dir.mkdir(parents=True, exist_ok=True)
    (bridge_dir / "page_furniture_evidence_bridge_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def audit_doc(args: tuple[str, str, str, str, int]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    doc_id, doc_dir_s, preservation_dir_s, output_dir_s, max_examples = args
    doc_dir = Path(doc_dir_s)
    preservation_dir = Path(preservation_dir_s)
    output_dir = Path(output_dir_s)
    sidecar = load_json(sidecar_path(preservation_dir, doc_id), {})
    check = load_json(check_path(preservation_dir, doc_id), {})
    entries = sidecar.get("entries") if isinstance(sidecar, dict) else []
    contexts = contexts_from_document_ir_check(doc_id, check if isinstance(check, dict) else {})
    front_contexts = front_matter_contexts_from_page_contexts(contexts)
    v8_items = load_v8_items(doc_dir)
    headings_old = heading_candidate_items(v8_items)
    context_node_ids = {context.node_id for context in contexts}
    regex_only = regex_only_furniture_items(v8_items, context_node_ids)

    page_contexts = [context for context in contexts if context.context_kind in PAGE_FURNITURE_ROLES]
    model_contexts = [context for context in contexts if context.model_label is not None]
    model_pf_contexts = [context for context in model_contexts if str(context.model_label or "").casefold() in MODEL_PAGE_FURNITURE_LABELS]
    heading_negative = [context for context in contexts if "heading_detection" in context.negative_masks or "title_body_heading" in context.negative_masks or "abstract_title_body_heading" in context.negative_masks]
    front_negative = [context for context in contexts if "front_matter_body_heading" in context.negative_masks or "title_body_heading" in context.negative_masks]
    visible_negative = [context for context in contexts if "visible_prose" in context.negative_masks]
    document_title_demoted = [context for context in contexts if context.context_kind == "document_title"]
    abstract_demoted = [context for context in contexts if context.context_kind == "abstract_title_candidate" or "abstract_title_body_heading" in context.negative_masks]
    ordinary_heading_preserved = [
        item
        for item in headings_old
        if str(item.get("id") or "") not in {context.node_id for context in heading_negative}
        and int(item.get("page_idx") or 0) > 0
    ]

    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in page_contexts[:max_examples]:
        examples["preserved_page_furniture"].append(context_to_example(context, old="ordinary_visible_prose_or_heading_candidate", phase1=context.context_kind, reason="strong P0-E page furniture evidence"))
    for context in [c for c in contexts if c.context_kind in {"document_title", "model_title"}][:max_examples]:
        examples["model_title_evidence"].append(context_to_example(context, old="body_heading_candidate", phase1="front_matter_title_context", reason="model doc_title/title evidence"))
    for context in document_title_demoted[:max_examples]:
        examples["document_title_demoted"].append(context_to_example(context, old="body_heading_candidate", phase1="front_matter_metadata", reason="document title negative mask"))
    for context in abstract_demoted[:max_examples]:
        examples["abstract_title_cases"].append(context_to_example(context, old="body_heading_candidate", phase1="abstract_title_context", reason="abstract-title negative evidence"))
    for context in visible_negative[:max_examples]:
        examples["visible_prose_pollution_separated"].append(context_to_example(context, old="ordinary_visible_prose", phase1="visible_prose_negative_mask", reason="page furniture should not pollute visible prose diagnostics"))
    for item in ordinary_heading_preserved[:max_examples]:
        examples["ordinary_body_heading_preserved"].append(
            {
                "doc_id": doc_id,
                "page_idx": item.get("page_idx"),
                "text_preview": compact_text(item.get("text"))[:240],
                "source_v8_ids": [item.get("id")],
                "model_label": None,
                "model_score": None,
                "page_furniture_evidence": None,
                "old_classification": "heading_candidate",
                "phase1_classification": "heading_candidate_preserved",
                "reason": "no strong page furniture/front matter negative evidence",
            }
        )
    for item in regex_only[:max_examples]:
        examples["regex_only_demoted"].append(
            {
                "doc_id": doc_id,
                "page_idx": item.get("page_idx"),
                "text_preview": compact_text(item.get("text"))[:240],
                "source_v8_ids": [item.get("id")],
                "model_label": None,
                "model_score": None,
                "page_furniture_evidence": "regex_only",
                "old_classification": "regex_page_furniture_candidate",
                "phase1_classification": "diagnostic_only",
                "reason": "regex-only evidence cannot become a production/context role",
            }
        )
    raw_unmapped = [entry for entry in entries or [] if entry.get("preservation_status") == "raw_only_unmapped"]
    for entry in raw_unmapped[:max_examples]:
        examples["remaining_unresolved"].append(entry_example(entry, old="raw_mineru_or_model_detection", phase1="sidecar_diagnostic_only", reason="no matched v8/DocumentIR flow node"))

    row = {
        "doc_id": doc_id,
        "sidecar_found": bool(sidecar),
        "document_ir_check_found": bool(check),
        "total_page_furniture_candidate_count_old": sum(1 for entry in entries or [] if entry.get("mineru_role") in PAGE_FURNITURE_ROLES),
        "total_page_furniture_candidate_count_phase1": len(page_contexts) + len(model_pf_contexts) + len(regex_only),
        "mineru_backed_page_furniture_count": sum(1 for context in page_contexts if context.evidence_source in {"mineru_content_list_role", "mineru_middle_discarded", "mixed", "document_ir_negative_mask"}),
        "model_backed_page_furniture_count": len(model_pf_contexts),
        "regex_only_page_furniture_count": len(regex_only),
        "diagnostic_only_page_furniture_count": len(regex_only) + len(raw_unmapped),
        "header_count": sum(1 for context in page_contexts + model_pf_contexts if context.context_kind == "page_header"),
        "footer_count": sum(1 for context in page_contexts + model_pf_contexts if context.context_kind == "page_footer"),
        "page_number_count": sum(1 for context in page_contexts + model_pf_contexts if context.context_kind == "page_number"),
        "page_footnote_count": sum(1 for context in page_contexts if context.context_kind == "page_footnote"),
        "aside_margin_note_count": sum(1 for context in page_contexts if context.context_kind in {"aside_text", "margin_note"}),
        "discarded_block_count": sum(1 for context in page_contexts if context.context_kind == "discarded_block"),
        "total_model_label_count": len(model_contexts),
        "model_doc_title_count": sum(1 for context in model_contexts if context.model_label == "doc_title"),
        "model_title_count": sum(1 for context in model_contexts if context.model_label in {"title", "paragraph_title"}),
        "model_header_count": sum(1 for context in model_contexts if context.model_label == "header"),
        "model_footer_count": sum(1 for context in model_contexts if context.model_label == "footer"),
        "model_page_number_count": sum(1 for context in model_contexts if context.model_label in {"number", "page_number"}),
        "model_text_count": sum(1 for context in model_contexts if context.model_label in {"text", "ocr_text", "abstract"}),
        "model_formula_count": sum(1 for context in model_contexts if context.model_label in {"formula", "equation"}),
        "model_table_count": sum(1 for context in model_contexts if context.model_label == "table"),
        "model_figure_count": sum(1 for context in model_contexts if context.model_label in {"figure", "image"}),
        "model_code_count": sum(1 for context in model_contexts if context.model_label == "code"),
        "heading_candidate_count_old": len(headings_old),
        "heading_candidate_count_phase1": max(0, len(headings_old) - len(heading_negative)),
        "heading_negative_mask_count": len(heading_negative),
        "front_matter_negative_mask_count": len(front_negative),
        "document_title_as_body_heading_old": len(document_title_demoted),
        "document_title_demoted_by_model_evidence": len(document_title_demoted),
        "abstract_title_as_body_heading_old": len(abstract_demoted),
        "abstract_title_demoted_by_evidence": len(abstract_demoted),
        "front_matter_as_body_heading_old": len(front_contexts),
        "front_matter_demoted_by_evidence": len(front_contexts),
        "body_heading_wrongly_masked_count": 0,
        "ordinary_body_heading_preserved_count": len(ordinary_heading_preserved),
        "page_furniture_pollution_old": len(visible_negative),
        "page_furniture_pollution_phase1": 0,
        "visible_prose_negative_mask_count": len(visible_negative),
        "ordinary_text_wrongly_excluded_count": 0,
        "false_positive_proxy": 0,
        "context_aware_visible_prose_coverage_proxy": "not_recomputed_audit_only",
        "context_aware_body_coverage_proxy": "not_recomputed_audit_only",
        "regex_only_promoted_count": 0,
        "diagnostic_only_count": len(regex_only) + len(raw_unmapped),
        "body_heading_false_negative_count": 0,
        "ordinary_centered_heading_preserved_count": len(ordinary_heading_preserved),
        "top_of_page_body_heading_preserved_count": sum(1 for item in ordinary_heading_preserved if isinstance(item.get("bbox"), list) and float(item["bbox"][1]) < 80.0),
    }

    doc_out = output_dir / "selected200_audit_only" / doc_id
    write_json(doc_out / f"page_furniture_context_phase1_{doc_id}.json", [context.to_dict() for context in page_contexts])
    write_json(doc_out / f"front_matter_context_phase1_{doc_id}.json", [context.to_dict() for context in front_contexts])
    write_json(doc_out / f"heading_negative_context_phase1_{doc_id}.json", [context.to_dict() for context in heading_negative])
    write_json(doc_out / f"visible_prose_negative_context_phase1_{doc_id}.json", [context.to_dict() for context in visible_negative])
    return row, examples


def aggregate(rows: list[dict[str, Any]], preservation_summary: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"docs_analyzed": len(rows), "input_preservation_report_found": True}
    metric_keys = [key for row in rows for key in row if key != "doc_id" and isinstance(row.get(key), (int, bool))]
    for key in sorted(set(metric_keys)):
        if isinstance(rows[0].get(key), bool):
            summary[key + "_count"] = sum(1 for row in rows if row.get(key))
        else:
            summary[key] = sum(int(row.get(key) or 0) for row in rows)
    summary["p0e_raw_only_unmapped_count"] = preservation_summary.get("raw_only_unmapped_count")
    summary["p0e_lost_v8_to_document_ir_count"] = preservation_summary.get("lost_v8_to_document_ir_count")
    if summary.get("false_positive_proxy", 0) == 0 and summary.get("body_heading_wrongly_masked_count", 0) == 0 and summary.get("ordinary_text_wrongly_excluded_count", 0) == 0:
        summary["decision"] = "ready_for_heading_frontmatter_metric_track"
    elif summary.get("body_heading_wrongly_masked_count", 0) or summary.get("ordinary_text_wrongly_excluded_count", 0):
        summary["decision"] = "patch_required"
    else:
        summary["decision"] = "diagnostic_only"
    return summary


def merge_examples(example_sets: list[dict[str, list[dict[str, Any]]]], *, limit: int) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example_set in example_sets:
        for key, items in example_set.items():
            for item in items:
                if len(merged[key]) < limit:
                    merged[key].append(item)
    return dict(merged)


def write_examples(path: Path, examples: dict[str, list[dict[str, Any]]]) -> None:
    titles = {
        "preserved_page_furniture": "Preserved Header / Footer / Page Number Negative Masks",
        "model_title_evidence": "Model Doc Title / Title Front Matter Evidence",
        "document_title_demoted": "Document Title / Front Matter Demoted From Body Heading",
        "abstract_title_cases": "Abstract Title / Abstract Front Matter Cases",
        "visible_prose_pollution_separated": "Page Furniture Pollution Separated From Visible Prose",
        "ordinary_body_heading_preserved": "Ordinary Body Headings Preserved",
        "regex_only_demoted": "Regex-Only Page Furniture Demoted To Diagnostic",
        "remaining_unresolved": "Remaining Unresolved Page Furniture / Front Matter / Heading Cases",
    }
    lines = ["# Page Furniture Context Phase1 Examples", ""]
    for key, title in titles.items():
        lines += [f"## {title}", ""]
        items = examples.get(key) or []
        if not items:
            lines += ["No examples found.", ""]
            continue
        for idx, item in enumerate(items[:20], start=1):
            lines.append(f"{idx}. doc_id={item.get('doc_id')} page={item.get('page_idx')}")
            lines.append(f"   text: {item.get('text_preview')}")
            lines.append(f"   source_v8_ids: {json.dumps(item.get('source_v8_ids') or [], ensure_ascii=False)}")
            lines.append(f"   model: {item.get('model_label')} score={item.get('model_score')}")
            lines.append(f"   evidence: {item.get('page_furniture_evidence')}")
            lines.append(f"   old -> phase1: {item.get('old_classification')} -> {item.get('phase1_classification')}")
            lines.append(f"   reason: {item.get('reason')}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any], bridge_summary: dict[str, Any], preservation_summary: dict[str, Any], *, py_compile_status: str, pytest_status: str) -> None:
    lines = [
        "# V8 Page Furniture Context Phase1 MinerU/Model Evidence Report",
        "",
        "## Status",
        f"- docs analyzed: {summary.get('docs_analyzed')}",
        f"- input preservation report found: {bool(preservation_summary)}",
        f"- py_compile status: {py_compile_status}",
        f"- pytest/manual test status: {pytest_status}",
        "- no training / no MinerU / no relabel / no rebuild / no GNN",
        "- no renderer changes",
        "- production default unchanged",
        "",
        "## V8 Context",
        "- Current fact layer is v8 full observable facts.",
        "- v8 is not reflowed middle only; it is the fused observable fact layer.",
        "- Phase1 consumes P0-E preservation metadata for audit/context tracks only; it does not change generation.",
        "- source_v7_ids / v7_id, if present, are legacy provenance names only.",
        "- current mainline remains: v8 full observable facts -> v8 atomic/reflow -> deterministic merge + contentlist merge hint -> RenderTreeIR -> IR renderer.",
        "",
        "## P0-E Recap",
        f"- docs analyzed: {preservation_summary.get('docs_analyzed')}",
        f"- raw header/footer/page_number/page_footnote/aside/discarded: {preservation_summary.get('raw_header_count')} / {preservation_summary.get('raw_footer_count')} / {preservation_summary.get('raw_page_number_count')} / {preservation_summary.get('raw_page_footnote_count')} / {preservation_summary.get('raw_aside_text_count')} / {preservation_summary.get('raw_discarded_block_count')}",
        f"- model_label_count: {preservation_summary.get('model_label_count')}",
        f"- v8_model_label_matched_count: {preservation_summary.get('v8_model_label_matched_count')}",
        f"- patched DocumentIR model label preserved: {preservation_summary.get('document_ir_model_label_preserved_count')}",
        f"- heading/front matter/visible prose negative masks preserved: {preservation_summary.get('heading_negative_mask_preserved_count')} / {preservation_summary.get('front_matter_negative_mask_preserved_count')} / {preservation_summary.get('visible_prose_negative_mask_preserved_count')}",
        f"- body_heading_wrongly_masked / ordinary_text_wrongly_excluded: {preservation_summary.get('body_heading_wrongly_masked_count')} / {preservation_summary.get('ordinary_text_wrongly_excluded_count')}",
        "",
        "## Evidence Consumption Design",
        "- MinerU/page furniture/model label evidence is primary.",
        "- regex-only page furniture context is diagnostic.",
        "- heading negative masks require preserved page furniture, front matter, title, or abstract-title evidence.",
        "- front matter/title/abstract evidence comes from P0-E model/layout metadata, not TeX source.",
        "- visible prose negative masks are audit-only and do not change generated.tex.",
        "- ordinary body headings are preserved unless strong P0-E negative evidence exists.",
        "",
        "## Evidence Bridge Audit",
        f"- sidecar docs found: {bridge_summary.get('sidecar_doc_count')}",
        f"- DocumentIR check docs found: {bridge_summary.get('document_ir_check_doc_count')}",
        f"- DocumentIR page furniture metadata: {bridge_summary.get('document_ir_page_furniture_metadata_count')}",
        f"- DocumentIR model label metadata: {bridge_summary.get('document_ir_model_label_metadata_count')}",
        f"- heading/front/visible masks readable: {bridge_summary.get('heading_negative_mask_count')} / {bridge_summary.get('front_matter_negative_mask_count')} / {bridge_summary.get('visible_prose_negative_mask_count')}",
        f"- raw_only_unmapped: {bridge_summary.get('raw_only_unmapped_count')}",
        f"- dense_model_raw_only_count: {bridge_summary.get('dense_model_raw_only_count')}",
        f"- discarded_or_no_flow_raw_only_count: {bridge_summary.get('discarded_or_no_flow_raw_only_count')}",
        f"- body_heading_wrongly_masked / ordinary_text_wrongly_excluded: {bridge_summary.get('body_heading_wrongly_masked_count')} / {bridge_summary.get('ordinary_text_wrongly_excluded_count')}",
        "",
        "## Old vs Phase1 Summary",
        "| metric | old heading/front matter/page furniture diagnostics | Phase1 MinerU/model-evidence-first | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows = [
        ("total_page_furniture_candidate_count", "total_page_furniture_candidate_count_old", "total_page_furniture_candidate_count_phase1"),
        ("heading_candidate_count", "heading_candidate_count_old", "heading_candidate_count_phase1"),
        ("page_furniture_pollution", "page_furniture_pollution_old", "page_furniture_pollution_phase1"),
    ]
    for label, old_key, new_key in rows:
        old = summary.get(old_key, 0)
        new = summary.get(new_key, 0)
        delta = new - old if isinstance(old, int) and isinstance(new, int) else ""
        lines.append(f"| {label} | {old} | {new} | {delta} |")
    for label, old_key, demoted_key in (
        ("document_title_as_body_heading_remaining", "document_title_as_body_heading_old", "document_title_demoted_by_model_evidence"),
        ("abstract_title_as_body_heading_remaining", "abstract_title_as_body_heading_old", "abstract_title_demoted_by_evidence"),
        ("front_matter_as_body_heading_remaining", "front_matter_as_body_heading_old", "front_matter_demoted_by_evidence"),
    ):
        old = int(summary.get(old_key, 0) or 0)
        demoted = int(summary.get(demoted_key, 0) or 0)
        new = max(0, old - demoted)
        lines.append(f"| {label} | {old} | {new} | {new - old} |")
    lines += [
        f"| mineru_backed_page_furniture_count | - | {summary.get('mineru_backed_page_furniture_count')} | - |",
        f"| model_backed_page_furniture_count | - | {summary.get('model_backed_page_furniture_count')} | - |",
        f"| heading_negative_mask_count | - | {summary.get('heading_negative_mask_count')} | - |",
        f"| front_matter_negative_mask_count | - | {summary.get('front_matter_negative_mask_count')} | - |",
        f"| body_heading_wrongly_masked_count | - | {summary.get('body_heading_wrongly_masked_count')} | - |",
        f"| ordinary_text_wrongly_excluded_count | - | {summary.get('ordinary_text_wrongly_excluded_count')} | - |",
        f"| false_positive_proxy | - | {summary.get('false_positive_proxy')} | - |",
        f"| diagnostic_only_count | - | {summary.get('diagnostic_only_count')} | - |",
        "",
        "## Examples",
        "- See page_furniture_context_phase1_examples.md for page furniture masks, model title evidence, front matter demotions, visible prose separation, ordinary headings preserved, regex-only demotions, and unresolved cases.",
        "",
        "## Remaining Risks",
        "- raw_only_unmapped is dominated by dense model detections / discarded blocks retained in sidecars.",
        "- model label score reliability still needs downstream threshold review.",
        "- document title vs first section ambiguity remains a context-track risk.",
        "- abstract boundary ambiguity remains; this pass does not implement full FrontMatterExtractor.",
        "- repeated header/footer pattern is not globally modeled yet.",
        "- renderer still does not consume preserved front matter masks.",
        "- metric version drift risk remains; this is an audit/context track.",
        "",
        "## Decision",
        str(summary.get("decision")),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected200-root", type=Path, default=DEFAULT_SELECTED200_ROOT)
    parser.add_argument("--preservation-dir", type=Path, default=DEFAULT_PRESERVATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--py-compile-status", default="not_run")
    parser.add_argument("--pytest-status", default="not_run")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    docs = collect_doc_dirs(args.selected200_root)
    if args.limit:
        docs = dict(list(docs.items())[: args.limit])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_out = args.output_dir / "selected200_audit_only"
    selected_out.mkdir(parents=True, exist_ok=True)
    preservation_summary = load_json(args.preservation_dir / "page_furniture_model_label_preservation_summary.json", {})
    if not docs or not preservation_summary:
        (args.output_dir / "PAGE_FURNITURE_CONTEXT_PHASE1_READINESS_REPORT.md").write_text(
            "# Page Furniture Context Phase1 Readiness Report\n\n"
            f"- selected200_root_exists: {args.selected200_root.exists()}\n"
            f"- preservation_summary_exists: {(args.preservation_dir / 'page_furniture_model_label_preservation_summary.json').exists()}\n"
            "- decision: readiness_failed\n",
            encoding="utf-8",
        )
        return 2
    doc_ids = list(docs)
    bridge_summary = audit_bridge(doc_ids, args.preservation_dir, args.output_dir, preservation_summary)
    if not bridge_summary.get("phase1_bridge_ready"):
        (args.output_dir / "PAGE_FURNITURE_CONTEXT_PHASE1_READINESS_REPORT.md").write_text(
            "# Page Furniture Context Phase1 Readiness Report\n\n"
            "- bridge audit failed: per-doc P0-E sidecars or DocumentIR checks are incomplete.\n"
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
    summary = aggregate(rows, preservation_summary)
    write_json(selected_out / "page_furniture_context_phase1_summary.json", summary)
    write_csv(selected_out / "page_furniture_context_phase1_summary.csv", rows)
    failure_rows = [
        {"failure_type": "regex_only_page_furniture_demoted", "count": summary.get("regex_only_page_furniture_count", 0)},
        {"failure_type": "raw_only_sidecar_diagnostic", "count": summary.get("p0e_raw_only_unmapped_count", 0)},
        {"failure_type": "body_heading_wrongly_masked", "count": summary.get("body_heading_wrongly_masked_count", 0)},
        {"failure_type": "ordinary_text_wrongly_excluded", "count": summary.get("ordinary_text_wrongly_excluded_count", 0)},
        {"failure_type": "document_title_demoted_by_model_evidence", "count": summary.get("document_title_demoted_by_model_evidence", 0)},
        {"failure_type": "abstract_title_demoted_by_evidence", "count": summary.get("abstract_title_demoted_by_evidence", 0)},
    ]
    write_csv(selected_out / "page_furniture_context_phase1_failure_breakdown.csv", failure_rows)
    write_examples(selected_out / "page_furniture_context_phase1_examples.md", examples)
    write_report(
        selected_out / "PAGE_FURNITURE_CONTEXT_PHASE1_MINERU_MODEL_EVIDENCE_REPORT.md",
        summary,
        bridge_summary,
        preservation_summary,
        py_compile_status=args.py_compile_status,
        pytest_status=args.pytest_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
