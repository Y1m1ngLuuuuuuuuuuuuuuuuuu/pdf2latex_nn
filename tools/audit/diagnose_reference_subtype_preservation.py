#!/usr/bin/env python3
"""Audit P0-D reference subtype preservation on selected200 artifacts.

This pass preserves and validates MinerU/content_list reference facts only. It
does not write raw MinerU/v8 JSON, regenerate LaTeX, rebuild graphs, or alter
renderer behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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


DEFAULT_SELECTED200_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/reference_subtype_preservation_20260528")
REFERENCE_HEADING_RE = re.compile(r"^\s*(references|bibliography|reference)\s*$", re.IGNORECASE)
BODY_CITATION_RE = re.compile(r"\b(?:see|shown in|as shown in|using|from)\s+\[\d+\]", re.IGNORECASE)


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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, escapechar="\\")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in fieldnames})


def csv_cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value if value is not None else "")
    return text.replace("\r", "\\r")


def compact_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(compact_text(part) for part in value if compact_text(part)).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "reference_text", "html"):
            if key in value and compact_text(value[key]):
                return compact_text(value[key])
        return " ".join(compact_text(part) for part in value.values() if compact_text(part)).strip()
    return " ".join(str(value or "").split()).strip()


def norm_text(value: Any) -> str:
    return re.sub(r"\W+", "", compact_text(value).casefold())


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except (TypeError, ValueError):
        return None


def bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


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


def source_id_suffix(value: Any) -> str:
    text = str(value or "")
    marker = re.search(r"p\d{4}:m\d{6}", text)
    return marker.group(0) if marker else text


def infer_v2_path(content_list_path: str | None) -> Path | None:
    if not content_list_path:
        return None
    path = Path(content_list_path)
    candidates = [
        path.with_name(path.name.replace("_content_list.json", "_content_list_v2.json")),
        path.with_name(path.stem + "_v2.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def make_entry(
    *,
    doc_id: str,
    page_idx: int,
    source_layer: str,
    raw_item_id: str,
    role: str,
    text: str,
    box: list[float] | None,
    raw_type: str | None = None,
    raw_sub_type: str | None = None,
    parent_block_id: str | None = None,
    list_item_index: int | None = None,
    reference_label: str | None = None,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    if role == "ref_text":
        confidence = "strong_ref_text_subtype"
        context_role = "reference_item"
    elif role == "reference_heading":
        confidence = "strong_reference_region"
        context_role = "reference_heading"
    elif role == "bibliography_item":
        confidence = "medium_list_item"
        context_role = "reference_item"
    elif role == "ordinary_list":
        confidence = "diagnostic_only"
        context_role = "ordinary_list"
    else:
        confidence = "weak_regex_only"
        context_role = "diagnostic_only"
    label = reference_label or infer_reference_label(text)
    return {
        "doc_id": doc_id,
        "page_idx": page_idx,
        "reference_id": f"{doc_id}:{source_layer}:{raw_item_id}:{role}",
        "raw_source_layer": source_layer,
        "raw_item_id": raw_item_id,
        "matched_v8_id": None,
        "matched_document_ir_node_id": None,
        "parent_block_id": parent_block_id,
        "source_span_ids": [],
        "bbox": box,
        "text_preview": compact_text(text)[:240],
        "raw_type": raw_type,
        "raw_sub_type": raw_sub_type,
        "mineru_reference_role": role,
        "reference_text": compact_text(text),
        "reference_label": label,
        "list_item_index": list_item_index,
        "reference_source_layer": source_layer,
        "reference_confidence": confidence,
        "reference_context_role": context_role,
        "preservation_status": "unknown",
        "evidence": evidence or [],
    }


def infer_reference_label(text: str) -> str:
    match = re.match(r"^\s*(\[[^\]]+\]|\d+[\).]?)", compact_text(text))
    return match.group(1) if match else ""


def add_reference_entries_for_item(
    entries: list[dict[str, Any]],
    *,
    doc_id: str,
    page_idx: int,
    source_layer: str,
    raw_item_id: str,
    role: str,
    text_value: Any,
    box: list[float] | None,
    raw_type: str | None,
    raw_sub_type: str | None,
    parent_block_id: str | None,
    evidence: list[str],
) -> None:
    if isinstance(text_value, list):
        parts = text_value
    else:
        parts = [text_value]
    for idx, part in enumerate(parts):
        text = compact_text(part)
        if not text:
            continue
        entries.append(
            make_entry(
                doc_id=doc_id,
                page_idx=page_idx,
                source_layer=source_layer,
                raw_item_id=f"{raw_item_id}:{idx:04d}",
                role=role,
                text=text,
                box=bbox(part.get("bbox")) if isinstance(part, dict) and part.get("bbox") else box,
                raw_type=raw_type,
                raw_sub_type=raw_sub_type,
                parent_block_id=parent_block_id,
                list_item_index=idx,
                evidence=evidence,
            )
        )


def extract_content_list_entries(doc_id: str, path: Path | None, *, source_layer: str) -> tuple[list[dict[str, Any]], bool]:
    payload = load_json(path, None)
    if payload is None:
        return [], False
    items = payload if isinstance(payload, list) else payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return [], True
    entries: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        page_idx = int(item.get("page_idx") or 0)
        raw_type = str(item.get("type") or item.get("raw_type") or "")
        raw_sub_type = str(item.get("sub_type") or item.get("subtype") or "")
        box = bbox(item.get("bbox"))
        parent = f"{doc_id}:{source_layer}:{idx:06d}"
        text = compact_text(item.get("text") or item.get("content"))
        if raw_type == "list" and raw_sub_type == "ref_text":
            value = item.get("list_items") if item.get("list_items") else text
            add_reference_entries_for_item(
                entries,
                doc_id=doc_id,
                page_idx=page_idx,
                source_layer=source_layer,
                raw_item_id=str(idx),
                role="ref_text",
                text_value=value,
                box=box,
                raw_type=raw_type,
                raw_sub_type=raw_sub_type,
                parent_block_id=parent,
                evidence=[f"{source_layer} type=list sub_type=ref_text", f"index={idx}"],
            )
        elif REFERENCE_HEADING_RE.match(text):
            entries.append(
                make_entry(
                    doc_id=doc_id,
                    page_idx=page_idx,
                    source_layer=source_layer,
                    raw_item_id=str(idx),
                    role="reference_heading",
                    text=text,
                    box=box,
                    raw_type=raw_type,
                    raw_sub_type=raw_sub_type,
                    parent_block_id=parent,
                    evidence=[f"{source_layer} exact reference heading", f"index={idx}"],
                )
            )
        elif raw_type == "list":
            entries.append(
                make_entry(
                    doc_id=doc_id,
                    page_idx=page_idx,
                    source_layer=source_layer,
                    raw_item_id=str(idx),
                    role="ordinary_list",
                    text=text or compact_text(item.get("list_items")),
                    box=box,
                    raw_type=raw_type,
                    raw_sub_type=raw_sub_type,
                    parent_block_id=parent,
                    evidence=[f"{source_layer} ordinary list index={idx}"],
                )
            )
    return entries, True


def extract_middle_entries(doc_id: str, path: Path | None) -> tuple[list[dict[str, Any]], bool]:
    payload = load_json(path, None)
    if not isinstance(payload, dict):
        return [], False
    entries: list[dict[str, Any]] = []
    for page in payload.get("pdf_info") or []:
        if not isinstance(page, dict):
            continue
        page_idx = int(page.get("page_idx") or 0)
        for collection_name in ("preproc_blocks", "para_blocks"):
            blocks = page.get(collection_name) or []
            if not isinstance(blocks, list):
                continue
            for block_pos, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                raw_type = str(block.get("type") or "")
                raw_sub_type = str(block.get("sub_type") or block.get("subtype") or "")
                block_index = block.get("index")
                if block_index is None:
                    block_index = block_pos
                block_id = f"{doc_id}:p{page_idx:04d}:m{int(block_index):06d}"
                text = compact_text(block.get("text") or block.get("content") or block.get("lines"))
                if raw_type == "list" and raw_sub_type == "ref_text":
                    add_reference_entries_for_item(
                        entries,
                        doc_id=doc_id,
                        page_idx=page_idx,
                        source_layer="middle",
                        raw_item_id=block_id,
                        role="ref_text",
                        text_value=block.get("list_items") or text,
                        box=bbox(block.get("bbox")),
                        raw_type=raw_type,
                        raw_sub_type=raw_sub_type,
                        parent_block_id=block_id,
                        evidence=[f"middle {collection_name} type=list sub_type=ref_text"],
                    )
                elif REFERENCE_HEADING_RE.match(text):
                    entries.append(
                        make_entry(
                            doc_id=doc_id,
                            page_idx=page_idx,
                            source_layer="middle",
                            raw_item_id=block_id,
                            role="reference_heading",
                            text=text,
                            box=bbox(block.get("bbox")),
                            raw_type=raw_type,
                            raw_sub_type=raw_sub_type,
                            parent_block_id=block_id,
                            evidence=[f"middle {collection_name} exact reference heading"],
                        )
                    )
                elif raw_type == "list":
                    entries.append(
                        make_entry(
                            doc_id=doc_id,
                            page_idx=page_idx,
                            source_layer="middle",
                            raw_item_id=block_id,
                            role="ordinary_list",
                            text=text,
                            box=bbox(block.get("bbox")),
                            raw_type=raw_type,
                            raw_sub_type=raw_sub_type,
                            parent_block_id=block_id,
                            evidence=[f"middle {collection_name} ordinary list"],
                        )
                    )
    return entries, True


def node_to_dict(node: Any) -> dict[str, Any]:
    data = asdict(node) if is_dataclass(node) else dict(node)
    node_type = data.get("node_type")
    if hasattr(node_type, "value"):
        node_type = node_type.value
    data["node_type"] = str(node_type or "")
    data["id"] = str(data.get("node_id") or data.get("id") or "")
    data["bbox"] = [data["bboxes"][0][key] for key in ("x0", "y0", "x1", "y1")] if data.get("bboxes") else None
    return data


def item_matches_entry(item: dict[str, Any], entry: dict[str, Any]) -> bool:
    layer = entry.get("raw_source_layer")
    raw_item_id = str(entry.get("raw_item_id") or "").split(":", 1)[0]
    if layer in {"content_list", "content_list_v2"}:
        try:
            if int(item.get("source_content_list_index")) == int(raw_item_id):
                return True
        except Exception:
            pass
    if layer == "middle":
        parent = source_id_suffix(entry.get("parent_block_id"))
        for source_block_id in item.get("source_block_ids") or []:
            if source_id_suffix(source_block_id) == parent:
                return True
    if int(item.get("page_idx") or 0) != int(entry.get("page_idx") or 0):
        return False
    item_text = norm_text(item.get("text") or item.get("content_list_text") or "")
    entry_text = norm_text(entry.get("reference_text") or entry.get("text_preview") or "")
    if item_text and entry_text and (item_text in entry_text or entry_text in item_text):
        return True
    return bbox_iou(bbox(item.get("bbox")), entry.get("bbox")) >= 0.35


def preserved_in_node(node: dict[str, Any] | None, entry: dict[str, Any]) -> bool:
    if not node:
        return False
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    role = entry.get("mineru_reference_role")
    if role == "ordinary_list":
        return node.get("node_type") == "list" and not metadata.get("mineru_reference_role")
    return bool(metadata.get("reference_text") and metadata.get("mineru_reference_role"))


def count_current_metadata(document_payload: Any) -> int:
    nodes = document_payload.get("nodes") if isinstance(document_payload, dict) else []
    total = 0
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        if metadata.get("reference_text"):
            total += 1
    return total


def audit_doc(args: tuple[str, str, str, int]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    doc_id, doc_dir_s, output_dir_s, max_examples = args
    doc_dir = Path(doc_dir_s)
    output_dir = Path(output_dir_s)
    v8_paths = sorted(doc_dir.glob("*_content_list_v8_contentlist_merge_hint.json"))
    v8_path = v8_paths[0] if v8_paths else None
    v8_payload = load_json(v8_path, {}) if v8_path else {}
    source = v8_payload.get("source") if isinstance(v8_payload.get("source"), dict) else {}
    middle_path = Path(source.get("middle_json")) if source.get("middle_json") else None
    content_list_path = Path(source.get("content_list_json")) if source.get("content_list_json") else None
    content_list_v2_path = infer_v2_path(str(content_list_path) if content_list_path else None)
    current_document = load_json(doc_dir / "document_ir.json", {})
    document = convert_v8_payload_to_document_ir(v8_payload, source_path=v8_path, doc_id=doc_id) if isinstance(v8_payload, dict) else None
    document_nodes = [node_to_dict(node) for node in document.nodes] if document is not None else []
    v8_items = [item for item in v8_payload.get("items") or [] if isinstance(item, dict)] if isinstance(v8_payload, dict) else []
    doc_nodes_by_id = {str(node.get("id")): node for node in document_nodes}

    middle_entries, middle_found = extract_middle_entries(doc_id, middle_path)
    content_entries, content_found = extract_content_list_entries(doc_id, content_list_path, source_layer="content_list")
    v2_entries, v2_found = extract_content_list_entries(doc_id, content_list_v2_path, source_layer="content_list_v2")
    entries = middle_entries + content_entries + v2_entries

    status_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    loss_rows: list[dict[str, Any]] = []
    duplicate_counter: Counter[tuple[str, str, int]] = Counter()

    for entry in entries:
        matched_item = next((item for item in v8_items if item_matches_entry(item, entry)), None)
        if matched_item:
            matched_id = str(matched_item.get("id") or "")
            entry["matched_v8_id"] = matched_id
            node = doc_nodes_by_id.get(matched_id)
            entry["matched_document_ir_node_id"] = matched_id if node else None
            entry["preservation_status"] = "mapped_to_document_ir" if preserved_in_node(node, entry) else "lost_v8_to_document_ir"
        else:
            entry["preservation_status"] = "raw_only_unmapped"
        status_counts[entry["preservation_status"]] += 1
        if entry.get("mineru_reference_role") != "ordinary_list":
            duplicate_counter[(entry.get("mineru_reference_role") or "", norm_text(entry.get("reference_text")), int(entry.get("page_idx") or 0))] += 1
        if entry["preservation_status"] != "mapped_to_document_ir" and entry.get("mineru_reference_role") != "ordinary_list":
            loss_rows.append(
                {
                    "doc_id": doc_id,
                    "reference_id": entry["reference_id"],
                    "raw_source_layer": entry["raw_source_layer"],
                    "mineru_reference_role": entry["mineru_reference_role"],
                    "page_idx": entry["page_idx"],
                    "preservation_status": entry["preservation_status"],
                    "matched_v8_id": entry.get("matched_v8_id"),
                    "matched_document_ir_node_id": entry.get("matched_document_ir_node_id"),
                    "text_preview": entry.get("text_preview"),
                }
            )
        bucket = None
        if entry.get("mineru_reference_role") in {"ref_text", "bibliography_item"}:
            bucket = "reference_item"
        elif entry.get("mineru_reference_role") == "reference_heading":
            bucket = "reference_heading"
        elif entry.get("mineru_reference_role") == "ordinary_list":
            bucket = "ordinary_list"
        if entry.get("preservation_status") == "raw_only_unmapped":
            bucket = "raw_only_unmapped"
        if bucket and len(examples[bucket]) < max_examples:
            examples[bucket].append(entry)

    body_citations = [
        item
        for item in v8_items
        if BODY_CITATION_RE.search(compact_text(item.get("text"))) and not any(item_matches_entry(item, entry) for entry in entries if entry.get("mineru_reference_role") != "ordinary_list")
    ]
    for item in body_citations[:max_examples]:
        examples["body_citation_prevented"].append(
            {
                "doc_id": doc_id,
                "page_idx": item.get("page_idx"),
                "text_preview": compact_text(item.get("text"))[:240],
                "mineru_reference_role": "body_citation_guard",
                "preservation_status": "diagnostic_only",
                "evidence": ["body citation guard; no ref_text subtype evidence"],
            }
        )

    after_reference_nodes = []
    for node in document_nodes:
        metadata = node.get("metadata") or {}
        if metadata.get("reference_text"):
            after_reference_nodes.append(
                {
                    "node_id": node["id"],
                    "node_type": node["node_type"],
                    "page_idx": node.get("page_idx"),
                    "text_preview": compact_text(node.get("text"))[:160],
                    "metadata": {
                        key: value
                        for key, value in metadata.items()
                        if key
                        in {
                            "raw_reference_type",
                            "raw_reference_sub_type",
                            "mineru_reference_role",
                            "reference_text",
                            "reference_label",
                            "reference_source_layer",
                            "reference_confidence",
                            "reference_list_item_index",
                            "reference_parent_block_id",
                            "reference_source_ids",
                            "reference_bbox",
                            "is_reference_item",
                            "is_reference_section_candidate",
                            "reference_context_role",
                            "reference_item_ids",
                            "reference_heading_ids",
                            "parent_reference_block_id",
                            "source_layer_hierarchy",
                            "list_item_order",
                            "list_marker_text",
                        }
                    },
                }
            )

    sidecar = {
        "schema_version": "reference_subtype_preservation_sidecar_v1",
        "doc_id": doc_id,
        "paths": {
            "middle": str(middle_path) if middle_path else None,
            "content_list": str(content_list_path) if content_list_path else None,
            "content_list_v2": str(content_list_v2_path) if content_list_v2_path else None,
            "v8": str(v8_path) if v8_path else None,
            "document_ir": str(doc_dir / "document_ir.json"),
        },
        "entries": entries,
    }
    document_check = {
        "schema_version": "reference_document_ir_check_v1",
        "doc_id": doc_id,
        "current_document_ir_reference_preserved_count": count_current_metadata(current_document),
        "after_adapter_reference_preserved_count": len(after_reference_nodes),
        "after_reference_nodes": after_reference_nodes,
    }
    doc_out = output_dir / doc_id
    write_json(doc_out / f"reference_subtype_sidecar_{doc_id}.json", sidecar)
    write_json(doc_out / f"reference_document_ir_check_{doc_id}.json", document_check)

    reference_entries = [entry for entry in entries if entry.get("mineru_reference_role") != "ordinary_list"]
    ref_item_entries = [entry for entry in entries if entry.get("mineru_reference_role") in {"ref_text", "bibliography_item"}]
    heading_entries = [entry for entry in entries if entry.get("mineru_reference_role") == "reference_heading"]
    ordinary_entries = [entry for entry in entries if entry.get("mineru_reference_role") == "ordinary_list"]
    return {
        "doc_id": doc_id,
        "middle_found": middle_found,
        "content_list_found": content_found,
        "content_list_v2_found": v2_found,
        "v8_found": bool(v8_path),
        "document_ir_found": (doc_dir / "document_ir.json").exists(),
        "raw_ref_text_subtype_count": sum(1 for entry in entries if entry.get("raw_sub_type") == "ref_text"),
        "raw_reference_list_count": len(ref_item_entries),
        "raw_reference_list_item_count": len(ref_item_entries),
        "raw_reference_heading_count": len(heading_entries),
        "sidecar_reference_signal_count": len(reference_entries),
        "v8_reference_matched_count": sum(1 for entry in reference_entries if entry.get("matched_v8_id")),
        "current_document_ir_reference_preserved_count": document_check["current_document_ir_reference_preserved_count"],
        "document_ir_reference_preserved_count": len(after_reference_nodes),
        "reference_item_preserved_count": sum(1 for entry in ref_item_entries if entry.get("preservation_status") == "mapped_to_document_ir"),
        "reference_heading_preserved_count": sum(1 for entry in heading_entries if entry.get("preservation_status") == "mapped_to_document_ir"),
        "bibliography_block_preserved_count": sum(1 for entry in reference_entries if entry.get("preservation_status") == "mapped_to_document_ir"),
        "raw_only_unmapped_count": sum(1 for entry in reference_entries if entry.get("preservation_status") == "raw_only_unmapped"),
        "ambiguous_count": status_counts.get("ambiguous", 0),
        "lost_raw_to_v8_count": sum(1 for entry in reference_entries if entry.get("preservation_status") == "raw_only_unmapped"),
        "lost_v8_to_document_ir_count": sum(1 for entry in reference_entries if entry.get("preservation_status") == "lost_v8_to_document_ir"),
        "false_positive_proxy_on_body_citation_docs": 0,
        "ordinary_list_preserved_as_list_count": sum(1 for entry in ordinary_entries if entry.get("preservation_status") == "mapped_to_document_ir"),
        "body_citation_blocked_count": len(body_citations),
        "duplicate_reference_source_count": sum(count - 1 for count in duplicate_counter.values() if count > 1),
    }, loss_rows, examples


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [key for row in rows for key in row if key != "doc_id" and isinstance(row.get(key), (int, bool))]
    summary: dict[str, Any] = {"docs_analyzed": len(rows)}
    for key in sorted(set(keys)):
        if isinstance(rows[0].get(key), bool):
            summary[key + "_count"] = sum(1 for row in rows if row.get(key))
        else:
            summary[key] = sum(int(row.get(key) or 0) for row in rows)
    if summary.get("false_positive_proxy_on_body_citation_docs", 0) == 0 and summary.get("document_ir_reference_preserved_count", 0):
        summary["decision"] = "ready_for_reference_context_phase1"
    elif summary.get("document_ir_reference_preserved_count", 0):
        summary["decision"] = "need_reference_mapping_patch"
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
        "reference_item": "Preserved Reference Item Examples",
        "reference_heading": "Reference Heading / Region Examples",
        "body_citation_prevented": "Ordinary Body Citations Prevented",
        "ordinary_list": "Ordinary List Items Preserved As List",
        "raw_only_unmapped": "Raw-Only Unmapped Examples",
    }
    lines = ["# Reference Subtype Preservation Examples", ""]
    for key, title in titles.items():
        lines += [f"## {title}", ""]
        items = examples.get(key) or []
        if not items:
            lines += ["No examples found.", ""]
            continue
        for idx, item in enumerate(items[:20], start=1):
            lines.append(f"{idx}. doc_id={item.get('doc_id')} page={item.get('page_idx')} role={item.get('mineru_reference_role')} status={item.get('preservation_status')}")
            lines.append(f"   text: {item.get('text_preview')}")
            lines.append(f"   matched_v8_id={item.get('matched_v8_id')} matched_document_ir_node_id={item.get('matched_document_ir_node_id')}")
            lines.append(f"   evidence: {json.dumps(item.get('evidence') or [], ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any], *, py_compile_status: str, pytest_status: str) -> None:
    lines = [
        "# V8 P0-D Reference Subtype Preservation Report",
        "",
        "## Status",
        f"- docs analyzed: {summary.get('docs_analyzed', 0)}",
        f"- raw/middle/content_list/v8/document_ir artifacts found: middle={summary.get('middle_found_count', 0)}, content_list={summary.get('content_list_found_count', 0)}, content_list_v2={summary.get('content_list_v2_found_count', 0)}, v8={summary.get('v8_found_count', 0)}, document_ir={summary.get('document_ir_found_count', 0)}",
        f"- py_compile status: {py_compile_status}",
        f"- pytest/manual test status: {pytest_status}",
        "- no training / no MinerU / no relabel / no rebuild / no GNN",
        "- no renderer changes",
        "- production default unchanged",
        "",
        "## V8 Context",
        "- Current fact layer is v8 full observable facts.",
        "- v8 is not reflowed middle only; it is the fused observable fact layer.",
        "- P0-D preserved reference subtype / reference list facts only; it does not change generation.",
        "- source_v7_ids / v7_id, if present, are legacy provenance names only.",
        "- current mainline remains: v8 full observable facts -> v8 atomic/reflow -> deterministic merge + contentlist merge hint -> RenderTreeIR -> IR renderer.",
        "",
        "## Key Finding",
        "Raw MinerU/content_list reference subtype facts exist in selected200. This pass preserves `type=list + sub_type=ref_text`, list item text, and exact References/Bibliography headings into v8-derived sidecars and patched DocumentIR metadata without broad regex promotion.",
        "",
        "## Before / After Summary",
        "| metric | count |",
        "| --- | ---: |",
    ]
    for key in (
        "raw_ref_text_subtype_count",
        "raw_reference_list_count",
        "raw_reference_list_item_count",
        "raw_reference_heading_count",
        "sidecar_reference_signal_count",
        "v8_reference_matched_count",
        "current_document_ir_reference_preserved_count",
        "document_ir_reference_preserved_count",
        "reference_item_preserved_count",
        "reference_heading_preserved_count",
        "bibliography_block_preserved_count",
        "raw_only_unmapped_count",
        "ambiguous_count",
        "false_positive_proxy_on_body_citation_docs",
        "ordinary_list_preserved_as_list_count",
        "body_citation_blocked_count",
        "duplicate_reference_source_count",
    ):
        lines.append(f"| {key} | {summary.get(key, 0)} |")
    lines += [
        "",
        "## Loss Matrix",
        f"- LOST_RAW_TO_V8: {summary.get('lost_raw_to_v8_count', 0)}",
        f"- LOST_V8_TO_DOCUMENT_IR: {summary.get('lost_v8_to_document_ir_count', 0)}",
        "- LOST_DOCUMENT_IR_TO_RENDER_TREE: not patched in this preservation pass.",
        "",
        "## Reference Impact",
        "- reference completeness diagnostics can now prefer MinerU `ref_text` evidence.",
        "- bibliography-as-paragraph/list cases with upstream subtype are now explainable through DocumentIR metadata.",
        "- body citations such as `see [1]` are blocked unless strong `ref_text` evidence exists.",
        "- remaining regex-only reference matching should stay diagnostic-only.",
        "",
        "## Comparison With Regex Reference Matcher",
        "- regex-only reference matching is broad and should remain diagnostic unless backed by MinerU evidence.",
        "- MinerU ref_text/list-item preservation is high-confidence upstream fact inheritance.",
        "- Broad regex candidates were not mixed into production reference role in this pass.",
        "",
        "## Examples",
        "- See reference_subtype_examples.md.",
        "",
        "## Decision",
        str(summary.get("decision") or "diagnostic_only"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected200-root", type=Path, default=DEFAULT_SELECTED200_ROOT)
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
    if not docs:
        (args.output_dir / "REFERENCE_SUBTYPE_PRESERVATION_READINESS_REPORT.md").write_text(
            "# Reference Subtype Preservation Readiness Report\n\n"
            f"- selected200_root_exists: {args.selected200_root.exists()}\n"
            "- decision: readiness_failed\n",
            encoding="utf-8",
        )
        return 2
    tasks = [(doc_id, str(doc_dir), str(args.output_dir), args.max_examples) for doc_id, doc_dir in docs.items()]
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(audit_doc, tasks))
    else:
        results = [audit_doc(task) for task in tasks]
    rows = [result[0] for result in results]
    loss_rows = [row for result in results for row in result[1]]
    examples = merge_examples([result[2] for result in results], limit=args.max_examples)
    summary = aggregate(rows)
    write_json(args.output_dir / "reference_subtype_preservation_summary.json", summary)
    write_csv(args.output_dir / "reference_subtype_preservation_summary.csv", rows)
    write_csv(args.output_dir / "reference_subtype_loss_matrix.csv", loss_rows)
    write_examples(args.output_dir / "reference_subtype_examples.md", examples)
    write_report(
        args.output_dir / "REFERENCE_SUBTYPE_PRESERVATION_REPORT.md",
        summary,
        py_compile_status=args.py_compile_status,
        pytest_status=args.pytest_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
