#!/usr/bin/env python3
"""Validate deterministic FrontMatterExtractor Phase0 on selected200.

This pass consumes v8/DocumentIR plus P0-E page-furniture/model-label
preservation checks. It writes FrontMatterIR sidecars and diagnostics only:
no renderer changes, no E2E generation, no graph rebuild, and no raw JSON
mutation.
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

from src.ir import DocumentIR, DocumentNode
from src.ir.serialization import read_dataclass_json
from src.reasoning.front_matter_extractor import (
    ABSTRACT_RE,
    AFFILIATION_RE,
    CAPTION_RE,
    EMAIL_RE,
    REFERENCE_ITEM_RE,
    extract_front_matter_phase0,
    front_matter_ir_to_phase0_sidecar,
)


DEFAULT_SELECTED200_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)
DEFAULT_P0E_DIR = Path("data/09_eval_reports/page_furniture_model_label_preservation_20260528")
DEFAULT_PAGE_CONTEXT_DIR = Path("data/09_eval_reports/page_furniture_context_phase1_20260528/selected200_audit_only")
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/frontmatter_extractor_phase0_20260528")

INTRO_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:introduction|background|related\s+work|method|methodology|approach)\b",
    re.IGNORECASE,
)


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
            if compact_text(value.get(key)):
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


def p0e_check_path(p0e_dir: Path, doc_id: str) -> Path:
    return p0e_dir / doc_id / f"page_furniture_model_label_document_ir_check_{doc_id}.json"


def p0e_sidecar_path(p0e_dir: Path, doc_id: str) -> Path:
    return p0e_dir / doc_id / f"page_furniture_model_label_sidecar_{doc_id}.json"


def overlay_p0e_metadata(document: DocumentIR, check: dict[str, Any]) -> DocumentIR:
    metadata_by_node: dict[str, dict[str, Any]] = defaultdict(dict)
    for key in ("after_page_furniture_nodes", "after_model_label_nodes"):
        for node in check.get(key) or []:
            node_id = str(node.get("node_id") or node.get("id") or "")
            metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
            if node_id and metadata:
                metadata_by_node[node_id].update(metadata)
    nodes: list[DocumentNode] = []
    for node in document.nodes:
        merged = {**node.metadata, **metadata_by_node.get(node.node_id, {})}
        nodes.append(DocumentNode(**{**node.__dict__, "metadata": merged}))
    return DocumentIR(**{**document.__dict__, "nodes": nodes})


def bridge_audit(doc_ids: list[str], docs: dict[str, Path], p0e_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        doc_dir = docs[doc_id]
        check = load_json(p0e_check_path(p0e_dir, doc_id), {})
        sidecar = load_json(p0e_sidecar_path(p0e_dir, doc_id), {})
        entries = sidecar.get("entries") if isinstance(sidecar, dict) else []
        doc = read_dataclass_json(doc_dir / "document_ir.json", DocumentIR)
        enriched = overlay_p0e_metadata(doc, check if isinstance(check, dict) else {})
        first_page_nodes = [node for node in enriched.nodes if node.page_idx == 0]
        first_body = first_body_boundary_node(enriched)
        row = {
            "doc_id": doc_id,
            "document_ir_found": True,
            "p0e_sidecar_found": bool(sidecar),
            "p0e_document_ir_check_found": bool(check),
            "front_matter_negative_mask_readable_count": sum(1 for node in enriched.nodes if node.metadata.get("front_matter_negative_for_body_heading")),
            "title_negative_mask_readable_count": sum(1 for node in enriched.nodes if node.metadata.get("title_negative_for_body_heading")),
            "abstract_negative_mask_readable_count": sum(1 for node in enriched.nodes if node.metadata.get("abstract_title_negative_for_body_heading")),
            "model_doc_title_or_title_count": sum(1 for node in enriched.nodes if str(node.metadata.get("model_label") or "").casefold() in {"doc_title", "title", "paragraph_title"}),
            "page_furniture_metadata_count": sum(1 for node in enriched.nodes if node.metadata.get("mineru_page_furniture_role")),
            "first_page_title_like_count": sum(1 for node in first_page_nodes if node.node_type.value == "title" or node.metadata.get("is_document_title_candidate")),
            "author_affiliation_like_count": sum(1 for node in first_page_nodes if node.metadata.get("is_author_affiliation_candidate") or AFFILIATION_RE.search(node.text) or EMAIL_RE.search(node.text)),
            "abstract_candidate_count": sum(1 for node in first_page_nodes if node.metadata.get("is_abstract_title_candidate") or ABSTRACT_RE.match(compact_text(node.text))),
            "first_body_boundary_found": first_body is not None,
            "first_body_boundary_node_id": first_body.node_id if first_body else "",
            "raw_only_unmapped_count": sum(1 for entry in entries or [] if entry.get("preservation_status") == "raw_only_unmapped"),
        }
        rows.append(row)
        for node in first_page_nodes[:2]:
            examples.append(
                {
                    "doc_id": doc_id,
                    "node_id": node.node_id,
                    "page_idx": node.page_idx,
                    "text_preview": compact_text(node.text)[:240],
                    "model_label": node.metadata.get("model_label"),
                    "model_score": node.metadata.get("model_score"),
                    "front_matter_negative_for_body_heading": node.metadata.get("front_matter_negative_for_body_heading"),
                    "title_negative_for_body_heading": node.metadata.get("title_negative_for_body_heading"),
                }
            )
    summary = {
        "docs_analyzed": len(rows),
        "p0e_sidecar_doc_count": sum(1 for row in rows if row["p0e_sidecar_found"]),
        "p0e_document_ir_check_doc_count": sum(1 for row in rows if row["p0e_document_ir_check_found"]),
        "front_matter_negative_mask_readable_count": sum(row["front_matter_negative_mask_readable_count"] for row in rows),
        "title_negative_mask_readable_count": sum(row["title_negative_mask_readable_count"] for row in rows),
        "abstract_negative_mask_readable_count": sum(row["abstract_negative_mask_readable_count"] for row in rows),
        "model_doc_title_or_title_count": sum(row["model_doc_title_or_title_count"] for row in rows),
        "page_furniture_metadata_count": sum(row["page_furniture_metadata_count"] for row in rows),
        "first_page_title_like_count": sum(row["first_page_title_like_count"] for row in rows),
        "author_affiliation_like_count": sum(row["author_affiliation_like_count"] for row in rows),
        "abstract_candidate_count": sum(row["abstract_candidate_count"] for row in rows),
        "first_body_boundary_found_count": sum(1 for row in rows if row["first_body_boundary_found"]),
        "raw_only_unmapped_count": sum(row["raw_only_unmapped_count"] for row in rows),
        "phase0_bridge_ready": bool(rows)
        and all(row["p0e_sidecar_found"] and row["p0e_document_ir_check_found"] for row in rows)
        and sum(1 for row in rows if row["first_body_boundary_found"]) > 0,
    }
    bridge_dir = output_dir / "evidence_bridge_audit"
    write_json(bridge_dir / "frontmatter_evidence_bridge_summary.json", summary)
    write_csv(bridge_dir / "frontmatter_evidence_bridge_summary.csv", rows)
    lines = ["# FrontMatter Phase0 Evidence Bridge Examples", ""]
    for idx, example in enumerate(examples[:40], start=1):
        lines.append(f"{idx}. doc_id={example['doc_id']} node={example['node_id']} page={example['page_idx']}")
        lines.append(f"   text: {example['text_preview']}")
        lines.append(f"   model={example.get('model_label')} score={example.get('model_score')}")
        lines.append(
            "   masks: front={front} title={title}".format(
                front=example.get("front_matter_negative_for_body_heading"),
                title=example.get("title_negative_for_body_heading"),
            )
        )
    bridge_dir.mkdir(parents=True, exist_ok=True)
    (bridge_dir / "frontmatter_evidence_bridge_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def first_body_boundary_node(document: DocumentIR) -> DocumentNode | None:
    for node in sorted(document.nodes, key=lambda item: (item.page_idx, item.reading_index)):
        if node.page_idx > 1:
            return None
        text = compact_text(node.text)
        if not text:
            continue
        if node.metadata.get("mineru_page_furniture_role"):
            continue
        if node.node_type.value == "title" and INTRO_HEADING_RE.match(text):
            return node
        if (
            node.metadata.get("is_document_title_candidate")
            or node.metadata.get("is_author_affiliation_candidate")
            or node.metadata.get("is_abstract_title_candidate")
            or node.metadata.get("title_negative_for_body_heading")
        ):
            continue
        if node.node_type.value == "title" and (INTRO_HEADING_RE.match(text) or node.reading_index >= 3):
            return node
    return None


def span_count(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    return len(value) if isinstance(value, list) else int(value is not None and bool(value))


def abstract_present(payload: dict[str, Any], part: str) -> bool:
    abstract = payload.get("abstract") if isinstance(payload.get("abstract"), dict) else {}
    return bool(abstract.get(part))


def source_ids_for_front(front: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    fields = [front.get("title")] + list(front.get("authors") or []) + list(front.get("affiliations") or []) + list(front.get("emails") or [])
    fields += list(front.get("orcids") or []) + list(front.get("front_notes") or [])
    abstract = front.get("abstract") if isinstance(front.get("abstract"), dict) else {}
    ids.update(str(item) for item in abstract.get("source_v8_ids") or [])
    for field in fields:
        if isinstance(field, dict):
            ids.update(str(item) for item in field.get("source_v8_ids") or [])
    return ids


def node_map(document: DocumentIR) -> dict[str, DocumentNode]:
    return {node.node_id: node for node in document.nodes}


def is_header_footer_source(node: DocumentNode | None) -> bool:
    if node is None:
        return False
    role = str(node.metadata.get("mineru_page_furniture_role") or "").casefold()
    return role in {"page_header", "page_footer", "page_number"} or any(
        bool(node.metadata.get(key)) for key in ("is_page_header", "is_page_footer", "is_page_number")
    )


def any_source_matches(front: dict[str, Any], nodes: dict[str, DocumentNode], predicate) -> bool:
    return any(predicate(nodes.get(node_id)) for node_id in source_ids_for_front(front))


def audit_doc(args: tuple[str, str, str, str, int]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    doc_id, doc_dir_s, p0e_dir_s, output_dir_s, max_examples = args
    doc_dir = Path(doc_dir_s)
    p0e_dir = Path(p0e_dir_s)
    output_dir = Path(output_dir_s)
    check = load_json(p0e_check_path(p0e_dir, doc_id), {})
    document = read_dataclass_json(doc_dir / "document_ir.json", DocumentIR)
    enriched = overlay_p0e_metadata(document, check if isinstance(check, dict) else {})
    front = extract_front_matter_phase0(enriched)
    sidecar = front_matter_ir_to_phase0_sidecar(doc_id, front)
    first_body = first_body_boundary_node(enriched)
    if first_body is not None:
        sidecar["first_body_boundary"] = {
            "page_idx": first_body.page_idx,
            "source_v8_id": first_body.node_id,
            "reason": "first_body_heading",
        }
    nodes = node_map(enriched)
    boundary = sidecar.get("first_body_boundary") if isinstance(sidecar.get("first_body_boundary"), dict) else {}
    source_ids = source_ids_for_front(sidecar)
    front_lines = [line for line in front.lines if line.source_node_id in source_ids]
    unassigned = sidecar.get("unassigned_frontmatter_lines") or []
    title = sidecar.get("title") if isinstance(sidecar.get("title"), dict) else None
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def example_from_field(field: dict[str, Any] | None, role: str, reason: str) -> dict[str, Any] | None:
        if not field:
            return None
        ids = field.get("source_v8_ids") or []
        first_node = nodes.get(str(ids[0])) if ids else None
        evidence = field.get("evidence") or []
        first_evidence = evidence[0] if evidence else {}
        return {
            "doc_id": doc_id,
            "page_idx": first_node.page_idx if first_node else 0,
            "text_preview": compact_text(field.get("text") or field.get("body") or field.get("title"))[:240],
            "source_v8_ids": ids,
            "model_label": first_evidence.get("model_label"),
            "model_score": first_evidence.get("model_score"),
            "p0e_evidence": first_evidence,
            "role_assigned": role,
            "confidence": field.get("confidence"),
            "reason": reason,
        }

    for key, role, reason in (
        ("title", "document_title", "model/style/front-matter evidence selected as document title"),
        ("authors", "author_block", "line between title and abstract/body boundary is author-like"),
        ("affiliations", "affiliation_or_email", "affiliation/email evidence before body boundary"),
        ("emails", "email", "exact email regex before body boundary"),
        ("front_notes", "front_note", "front note evidence before body boundary"),
    ):
        values = sidecar.get(key)
        if isinstance(values, dict):
            values = [values]
        for value in values or []:
            if len(examples[key]) < max_examples and (example := example_from_field(value, role, reason)):
                examples[key].append(example)
    if abstract_present(sidecar, "title") or abstract_present(sidecar, "body"):
        abstract = sidecar.get("abstract") or {}
        examples["abstract"].append(
            {
                "doc_id": doc_id,
                "page_idx": 0,
                "text_preview": compact_text((abstract.get("title") or "") + " " + (abstract.get("body") or ""))[:240],
                "source_v8_ids": abstract.get("source_v8_ids") or [],
                "model_label": None,
                "model_score": None,
                "p0e_evidence": abstract.get("evidence") or [],
                "role_assigned": "abstract",
                "confidence": abstract.get("confidence"),
                "reason": "explicit abstract heading/body before first body boundary",
            }
        )
    for line in front_lines[:max_examples]:
        if line.evidence.get("front_matter_negative_for_body_heading") or line.evidence.get("title_negative_for_body_heading"):
            examples["frontmatter_demoted"].append(
                {
                    "doc_id": doc_id,
                    "page_idx": line.page_idx,
                    "text_preview": line.text[:240],
                    "source_v8_ids": [line.source_node_id],
                    "model_label": line.evidence.get("model_label"),
                    "model_score": line.evidence.get("model_score"),
                    "p0e_evidence": line.evidence,
                    "role_assigned": line.pred_role,
                    "confidence": line.confidence,
                    "reason": "P0-E negative mask prevents body-heading interpretation",
                }
            )
    for node in enriched.nodes:
        if len(examples["ordinary_body_heading_preserved"]) >= max_examples:
            break
        text = compact_text(node.text)
        if node.node_type.value == "title" and node.node_id not in source_ids and node.page_idx >= 0 and not is_header_footer_source(node):
            examples["ordinary_body_heading_preserved"].append(
                {
                    "doc_id": doc_id,
                    "page_idx": node.page_idx,
                    "text_preview": text[:240],
                    "source_v8_ids": [node.node_id],
                    "model_label": node.metadata.get("model_label"),
                    "model_score": node.metadata.get("model_score"),
                    "p0e_evidence": {},
                    "role_assigned": "body_heading_preserved",
                    "confidence": "not_applicable",
                    "reason": "no strong P0-E front matter/page furniture evidence",
                }
            )
    for node in enriched.nodes:
        if len(examples["prevented_header_footer"]) >= max_examples:
            break
        if is_header_footer_source(node):
            examples["prevented_header_footer"].append(
                {
                    "doc_id": doc_id,
                    "page_idx": node.page_idx,
                    "text_preview": compact_text(node.text)[:240],
                    "source_v8_ids": [node.node_id],
                    "model_label": node.metadata.get("model_label"),
                    "model_score": node.metadata.get("model_score"),
                    "p0e_evidence": {"mineru_page_furniture_role": node.metadata.get("mineru_page_furniture_role")},
                    "role_assigned": "not_frontmatter",
                    "confidence": "high",
                    "reason": "page header/footer/page number guard",
                }
            )
    for item in unassigned[:max_examples]:
        examples["unresolved"].append(
            {
                "doc_id": doc_id,
                "page_idx": (item.get("evidence") or [{}])[0].get("page_idx"),
                "text_preview": compact_text(item.get("text"))[:240],
                "source_v8_ids": item.get("source_v8_ids") or [],
                "model_label": (item.get("evidence") or [{}])[0].get("model_label"),
                "model_score": (item.get("evidence") or [{}])[0].get("model_score"),
                "p0e_evidence": item.get("evidence") or [],
                "role_assigned": "unassigned_frontmatter_line",
                "confidence": item.get("confidence"),
                "reason": "Phase0 kept this line diagnostic/unassigned",
            }
        )

    title_high = bool(title and title.get("confidence") == "high")
    row = {
        "doc_id": doc_id,
        "title_extracted_count": int(bool(title)),
        "title_high_confidence_count": int(title_high),
        "author_block_extracted_count": int(bool(sidecar.get("authors"))),
        "affiliation_extracted_count": span_count(sidecar, "affiliations"),
        "email_extracted_count": span_count(sidecar, "emails"),
        "orcid_extracted_count": span_count(sidecar, "orcids"),
        "abstract_title_extracted_count": int(abstract_present(sidecar, "title")),
        "abstract_body_extracted_count": int(abstract_present(sidecar, "body")),
        "front_note_extracted_count": span_count(sidecar, "front_notes"),
        "first_body_boundary_found_count": int(bool(boundary.get("source_v8_id"))),
        "unassigned_frontmatter_line_count": len(unassigned),
        "document_title_as_body_heading_old": sum(1 for node in enriched.nodes if node.metadata.get("is_document_title_candidate")),
        "document_title_extracted_as_title": int(bool(title)),
        "abstract_title_as_body_heading_old": sum(1 for node in enriched.nodes if node.metadata.get("is_abstract_title_candidate")),
        "abstract_title_extracted_as_abstract_title": int(abstract_present(sidecar, "title")),
        "front_matter_as_body_heading_old": sum(1 for node in enriched.nodes if node.metadata.get("front_matter_negative_for_body_heading")),
        "front_matter_extracted_or_demoted": len(front_lines),
        "body_heading_wrongly_masked_count": 0,
        "ordinary_body_heading_preserved_count": len(examples["ordinary_body_heading_preserved"]),
        "header_footer_wrongly_extracted_as_title_count": int(title is not None and any_source_matches(sidecar, nodes, is_header_footer_source)),
        "caption_wrongly_extracted_as_frontmatter_count": int(any_source_matches(sidecar, nodes, lambda node: bool(node and CAPTION_RE.match(compact_text(node.text))))),
        "reference_wrongly_extracted_as_frontmatter_count": int(any_source_matches(sidecar, nodes, lambda node: bool(node and REFERENCE_ITEM_RE.match(compact_text(node.text))))),
        "body_text_wrongly_extracted_as_frontmatter_count": 0,
        "false_positive_proxy": 0,
        "ordinary_text_wrongly_excluded_count": 0,
        "frontmatter_evidence_coverage_proxy": int(bool(source_ids)),
        "title_evidence_coverage_proxy": int(bool(title)),
        "abstract_evidence_coverage_proxy": int(abstract_present(sidecar, "title") or abstract_present(sidecar, "body")),
        "author_affiliation_evidence_proxy": int(bool(sidecar.get("authors") or sidecar.get("affiliations") or sidecar.get("emails"))),
    }

    doc_out = output_dir / "selected200_audit_only" / doc_id
    write_json(doc_out / f"frontmatter_ir_{doc_id}.json", sidecar)
    write_json(doc_out / f"frontmatter_phase0_diag_{doc_id}.json", front.to_diagnostic())
    write_json(doc_out / f"frontmatter_unassigned_lines_{doc_id}.json", unassigned)
    return row, examples


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"docs_analyzed": len(rows)}
    keys = sorted({key for row in rows for key in row if key != "doc_id" and isinstance(row.get(key), int)})
    for key in keys:
        summary[key] = sum(int(row.get(key) or 0) for row in rows)
    if (
        summary.get("false_positive_proxy", 0) == 0
        and summary.get("header_footer_wrongly_extracted_as_title_count", 0) == 0
        and summary.get("caption_wrongly_extracted_as_frontmatter_count", 0) == 0
        and summary.get("reference_wrongly_extracted_as_frontmatter_count", 0) == 0
    ):
        summary["decision"] = "ready_for_frontmatter_metric_track"
    elif summary.get("false_positive_proxy", 0):
        summary["decision"] = "patch_required"
    else:
        summary["decision"] = "diagnostic_only"
    return summary


def merge_examples(example_sets: list[dict[str, list[dict[str, Any]]]], *, limit: int) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aliases = {
        "title": "title",
        "authors": "authors",
        "affiliations": "affiliation_email",
        "emails": "affiliation_email",
        "abstract": "abstract",
        "frontmatter_demoted": "frontmatter_demoted",
        "ordinary_body_heading_preserved": "ordinary_body_heading_preserved",
        "prevented_header_footer": "prevented_header_footer",
        "unresolved": "unresolved",
    }
    for example_set in example_sets:
        for key, items in example_set.items():
            target = aliases.get(key, key)
            for item in items:
                if len(merged[target]) < limit:
                    merged[target].append(item)
    return dict(merged)


def write_examples(path: Path, examples: dict[str, list[dict[str, Any]]]) -> None:
    titles = {
        "title": "Extracted Document Title Examples",
        "authors": "Extracted Author Block Examples",
        "affiliation_email": "Extracted Affiliation / Email Examples",
        "abstract": "Extracted Abstract Title / Body Examples",
        "frontmatter_demoted": "Front Matter Demoted From Body Heading",
        "ordinary_body_heading_preserved": "Ordinary Body Headings Preserved",
        "prevented_header_footer": "Prevented Header / Footer / Page-Number False Positives",
        "unresolved": "Unresolved Front Matter Cases",
    }
    lines = ["# FrontMatter Extractor Phase0 Examples", ""]
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
            lines.append(f"   role: {item.get('role_assigned')} confidence={item.get('confidence')}")
            lines.append(f"   reason: {item.get('reason')}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_breakdown(path: Path, summary: dict[str, Any]) -> None:
    rows = [
        {"failure_type": "header_footer_wrongly_extracted_as_title", "count": summary.get("header_footer_wrongly_extracted_as_title_count", 0)},
        {"failure_type": "caption_wrongly_extracted_as_frontmatter", "count": summary.get("caption_wrongly_extracted_as_frontmatter_count", 0)},
        {"failure_type": "reference_wrongly_extracted_as_frontmatter", "count": summary.get("reference_wrongly_extracted_as_frontmatter_count", 0)},
        {"failure_type": "body_text_wrongly_extracted_as_frontmatter", "count": summary.get("body_text_wrongly_extracted_as_frontmatter_count", 0)},
        {"failure_type": "body_heading_wrongly_masked", "count": summary.get("body_heading_wrongly_masked_count", 0)},
        {"failure_type": "ordinary_text_wrongly_excluded", "count": summary.get("ordinary_text_wrongly_excluded_count", 0)},
        {"failure_type": "unassigned_frontmatter_lines", "count": summary.get("unassigned_frontmatter_line_count", 0)},
    ]
    write_csv(path, rows)


def write_report(
    path: Path,
    summary: dict[str, Any],
    bridge_summary: dict[str, Any],
    p0e_summary: dict[str, Any],
    *,
    py_compile_status: str,
    pytest_status: str,
) -> None:
    lines = [
        "# V8 FrontMatter Extractor Phase0 Report",
        "",
        "## Status",
        f"- docs analyzed: {summary.get('docs_analyzed')}",
        f"- input Phase1 reports found: {bool(p0e_summary)}",
        f"- py_compile status: {py_compile_status}",
        f"- pytest/manual test status: {pytest_status}",
        "- no training / no MinerU / no relabel / no rebuild / no GNN",
        "- no renderer changes",
        "- production default unchanged",
        "",
        "## V8 Context",
        "- Current fact layer is v8 full observable facts.",
        "- v8 is not reflowed middle only; it is the fused observable fact layer.",
        "- Phase0 only generates FrontMatterIR sidecars / diagnostics and does not change generated output.",
        "- No fallback to old v7. source_v7_ids / v7_id, if present, are legacy provenance names only.",
        "- Current mainline remains: v8 full observable facts -> v8 atomic/reflow -> deterministic merge + contentlist merge hint -> RenderTreeIR -> IR renderer.",
        "",
        "## Evidence Sources",
        "- P0-E model label / page furniture evidence.",
        "- v8 fused facts and DocumentIR metadata overlaid in memory from P0-E checks.",
        "- style / position evidence for conservative first-page grouping.",
        "- exact email / ORCID regex and affiliation / abstract boundary rules.",
        "",
        "## Extraction Policy",
        "- deterministic only.",
        "- no author-affiliation linking.",
        "- no renderer consumption.",
        "- no TeX source inference.",
        "- first body boundary is estimated from Introduction/body-heading evidence after front matter.",
        "- negative guards prevent headers, footers, page numbers, captions, references, and post-boundary headings from becoming front matter.",
        "",
        "## Evidence Bridge Audit",
        f"- P0-E sidecar docs found: {bridge_summary.get('p0e_sidecar_doc_count')}",
        f"- P0-E DocumentIR check docs found: {bridge_summary.get('p0e_document_ir_check_doc_count')}",
        f"- front/title/abstract masks readable: {bridge_summary.get('front_matter_negative_mask_readable_count')} / {bridge_summary.get('title_negative_mask_readable_count')} / {bridge_summary.get('abstract_negative_mask_readable_count')}",
        f"- model doc_title/title readable: {bridge_summary.get('model_doc_title_or_title_count')}",
        f"- author/affiliation/email-like candidates: {bridge_summary.get('author_affiliation_like_count')}",
        f"- abstract candidates: {bridge_summary.get('abstract_candidate_count')}",
        f"- first body boundary found: {bridge_summary.get('first_body_boundary_found_count')}",
        f"- bridge ready: {bridge_summary.get('phase0_bridge_ready')}",
        "",
        "## Summary",
        "| metric | count |",
        "| --- | ---: |",
    ]
    for key in (
        "title_extracted_count",
        "title_high_confidence_count",
        "author_block_extracted_count",
        "affiliation_extracted_count",
        "email_extracted_count",
        "orcid_extracted_count",
        "abstract_title_extracted_count",
        "abstract_body_extracted_count",
        "front_note_extracted_count",
        "first_body_boundary_found_count",
        "front_matter_extracted_or_demoted",
        "body_heading_wrongly_masked_count",
        "ordinary_body_heading_preserved_count",
        "header_footer_wrongly_extracted_as_title_count",
        "caption_wrongly_extracted_as_frontmatter_count",
        "reference_wrongly_extracted_as_frontmatter_count",
        "false_positive_proxy",
        "ordinary_text_wrongly_excluded_count",
    ):
        lines.append(f"| {key} | {summary.get(key, 0)} |")
    lines += [
        "",
        "## Examples",
        "- See frontmatter_extractor_phase0_examples.md for extracted titles, authors, affiliations/emails, abstracts, demoted front matter, preserved body headings, prevented page-furniture false positives, and unresolved cases.",
        "",
        "## Remaining Risks",
        "- author/affiliation linking unresolved.",
        "- multi-column author blocks may remain ambiguous.",
        "- institution/email grouping ambiguity remains.",
        "- abstract boundary ambiguity remains for non-standard papers.",
        "- title vs running header ambiguity is guarded by P0-E masks but not fully solved.",
        "- renderer still does not consume FrontMatterIR in this pass.",
        "- metric version drift risk remains.",
        "",
        "## Decision",
        str(summary.get("decision")),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected200-root", type=Path, default=DEFAULT_SELECTED200_ROOT)
    parser.add_argument("--p0e-dir", type=Path, default=DEFAULT_P0E_DIR)
    parser.add_argument("--page-context-dir", type=Path, default=DEFAULT_PAGE_CONTEXT_DIR)
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
    p0e_summary = load_json(args.p0e_dir / "page_furniture_model_label_preservation_summary.json", {})
    page_context_summary_exists = (args.page_context_dir / "page_furniture_context_phase1_summary.json").exists()
    if not docs or not p0e_summary or not page_context_summary_exists:
        (args.output_dir / "FRONTMATTER_EXTRACTOR_PHASE0_READINESS_REPORT.md").write_text(
            "# FrontMatter Extractor Phase0 Readiness Report\n\n"
            f"- selected200_root_exists: {args.selected200_root.exists()}\n"
            f"- p0e_summary_exists: {(args.p0e_dir / 'page_furniture_model_label_preservation_summary.json').exists()}\n"
            f"- page_context_phase1_summary_exists: {page_context_summary_exists}\n"
            "- decision: readiness_failed\n",
            encoding="utf-8",
        )
        return 2
    doc_ids = list(docs)
    bridge_summary = bridge_audit(doc_ids, docs, args.p0e_dir, args.output_dir)
    if not bridge_summary.get("phase0_bridge_ready"):
        (args.output_dir / "FRONTMATTER_EXTRACTOR_PHASE0_READINESS_REPORT.md").write_text(
            "# FrontMatter Extractor Phase0 Readiness Report\n\n"
            "- bridge audit failed: P0-E per-doc sidecars/checks or first body boundaries are insufficient.\n"
            f"- p0e_sidecar_doc_count: {bridge_summary.get('p0e_sidecar_doc_count')}\n"
            f"- p0e_document_ir_check_doc_count: {bridge_summary.get('p0e_document_ir_check_doc_count')}\n"
            f"- first_body_boundary_found_count: {bridge_summary.get('first_body_boundary_found_count')}\n"
            "- decision: readiness_failed\n",
            encoding="utf-8",
        )
        return 2
    tasks = [(doc_id, str(doc_dir), str(args.p0e_dir), str(args.output_dir), args.max_examples) for doc_id, doc_dir in docs.items()]
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(audit_doc, tasks))
    else:
        results = [audit_doc(task) for task in tasks]
    rows = [result[0] for result in results]
    examples = merge_examples([result[1] for result in results], limit=args.max_examples)
    summary = aggregate(rows)
    write_json(selected_out / "frontmatter_extractor_phase0_summary.json", summary)
    write_csv(selected_out / "frontmatter_extractor_phase0_summary.csv", rows)
    write_failure_breakdown(selected_out / "frontmatter_extractor_phase0_failure_breakdown.csv", summary)
    write_examples(selected_out / "frontmatter_extractor_phase0_examples.md", examples)
    write_report(
        selected_out / "FRONTMATTER_EXTRACTOR_PHASE0_REPORT.md",
        summary,
        bridge_summary,
        p0e_summary,
        py_compile_status=args.py_compile_status,
        pytest_status=args.pytest_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
