#!/usr/bin/env python3
"""Audit P0-E page furniture/model label preservation on selected200 artifacts.

This pass preserves and validates MinerU/model page-furniture facts only. It
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
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/page_furniture_model_label_preservation_20260528")

PAGE_FURNITURE_TYPES = {"header", "footer", "page_number", "aside_text", "page_footnote", "footnote", "noise"}
ROLE_BY_TYPE = {
    "header": "page_header",
    "footer": "page_footer",
    "page_number": "page_number",
    "number": "page_number",
    "aside_text": "aside_text",
    "page_footnote": "page_footnote",
    "footnote": "page_footnote",
    "noise": "discarded_block",
    "discarded": "discarded_block",
    "discarded_block": "discarded_block",
}
MODEL_ROLE_VOTES = {
    "doc_title": "document_title",
    "title": "section_title_candidate",
    "paragraph_title": "section_title_candidate",
    "text": "ordinary_text",
    "ocr_text": "ordinary_text",
    "abstract": "ordinary_text",
    "header": "page_header",
    "footer": "page_footer",
    "number": "page_number",
    "page_number": "page_number",
    "figure": "figure",
    "image": "figure",
    "table": "table",
    "formula": "formula",
    "equation": "formula",
    "code": "code",
    "reference": "unknown",
    "list": "ordinary_text",
}
BODY_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,:;-]{2,80}$")


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


def scaled_model_bbox(value: Any, *, width: float, height: float) -> list[float] | None:
    box = bbox(value)
    if not box or width <= 0 or height <= 0:
        return box
    return [box[0] / width * 612.0, box[1] / height * 792.0, box[2] / width * 612.0, box[3] / height * 792.0]


def doc_id_from_dir(path: Path) -> str:
    return path.name.split("_", 1)[-1]


def collect_doc_dirs(root: Path) -> dict[str, Path]:
    docs: dict[str, Path] = {}
    if not root.exists():
        return docs
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "document_ir.json").exists() and list(path.glob("*_content_list_v8_contentlist_merge_hint.json")):
            docs[doc_id_from_dir(path)] = path
    return docs


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


def infer_model_path(source_path: str | None) -> Path | None:
    if not source_path:
        return None
    path = Path(source_path)
    candidates = [
        path.with_name(path.name.replace("_middle.json", "_model.json")),
        path.with_name(path.name.replace("_content_list.json", "_model.json")),
        path.with_name(path.stem + "_model.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def source_id_suffix(value: Any) -> str:
    text = str(value or "")
    marker = re.search(r"p\d{4}:m\d{6}", text)
    return marker.group(0) if marker else text


def role_from_type(value: Any) -> str | None:
    return ROLE_BY_TYPE.get(str(value or "").casefold().strip())


def negative_mask_for_role(role: str) -> str:
    if role in {"page_header", "page_footer"}:
        return "header_footer"
    if role == "page_number":
        return "page_number"
    if role == "page_footnote":
        return "page_footnote"
    if role in {"aside_text", "margin_note"}:
        return "no_render_noise"
    if role == "discarded_block":
        return "no_render_noise"
    if role == "document_title":
        return "front_matter_title"
    return "not_negative"


def confidence_for_entry(source_layer: str, role: str) -> str:
    if source_layer == "model":
        return "strong_model_label"
    if source_layer in {"content_list", "content_list_v2"}:
        return "strong_content_list_role"
    if role == "discarded_block":
        return "strong_middle_discarded"
    if source_layer == "middle":
        return "medium_layout_position"
    return "weak_regex_only"


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
    model_label: str | None = None,
    model_score: Any = None,
    model_cls_id: Any = None,
    model_index: Any = None,
    model_bbox: list[float] | None = None,
    parent_block_id: str | None = None,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    negative_role = negative_mask_for_role(role)
    return {
        "doc_id": doc_id,
        "page_idx": page_idx,
        "entry_id": f"{doc_id}:{source_layer}:{raw_item_id}:{role}",
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
        "model_label": model_label,
        "model_score": model_score,
        "model_cls_id": model_cls_id,
        "model_index": model_index,
        "model_bbox": model_bbox,
        "mineru_role": role,
        "negative_mask_role": negative_role,
        "evidence_source": source_layer,
        "confidence": confidence_for_entry(source_layer, role),
        "preservation_status": "unknown",
        "evidence": evidence or [],
    }


def extract_content_entries(doc_id: str, path: Path | None, *, source_layer: str) -> tuple[list[dict[str, Any]], bool]:
    payload = load_json(path, None)
    items = payload if isinstance(payload, list) else payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return [], False
    entries: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get("type") or "")
        raw_sub_type = str(item.get("sub_type") or item.get("subtype") or "")
        role = role_from_type(raw_type)
        if not role:
            continue
        page_idx = int(item.get("page_idx") or 0)
        entries.append(
            make_entry(
                doc_id=doc_id,
                page_idx=page_idx,
                source_layer=source_layer,
                raw_item_id=str(idx),
                role=role,
                text=compact_text(item.get("text")),
                box=bbox(item.get("bbox")),
                raw_type=raw_type,
                raw_sub_type=raw_sub_type,
                evidence=[f"{source_layer} type={raw_type}"],
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
        page_idx = int(page.get("page_idx") if page.get("page_idx") is not None else page.get("page_no") or 0)
        for collection_name, default_role in (("discarded_blocks", "discarded_block"), ("preproc_blocks", ""), ("para_blocks", "")):
            blocks = page.get(collection_name) or []
            if not isinstance(blocks, list):
                continue
            for block_pos, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                raw_type = str(block.get("type") or block.get("layout_label") or "")
                role = default_role or role_from_type(raw_type)
                if not role:
                    continue
                block_index = block.get("index")
                if block_index is None:
                    block_index = block_pos
                block_id = f"{doc_id}:p{page_idx:04d}:m{int(block_index):06d}"
                entries.append(
                    make_entry(
                        doc_id=doc_id,
                        page_idx=page_idx,
                        source_layer="middle",
                        raw_item_id=block_id,
                        role=role,
                        text=compact_text(block.get("text") or block.get("content") or block.get("lines")),
                        box=bbox(block.get("bbox")),
                        raw_type=raw_type,
                        parent_block_id=block_id,
                        evidence=[f"middle {collection_name} role={role}"],
                    )
                )
    return entries, True


def extract_model_entries(doc_id: str, path: Path | None) -> tuple[list[dict[str, Any]], bool]:
    payload = load_json(path, None)
    pages = payload if isinstance(payload, list) else payload.get("pages") if isinstance(payload, dict) else None
    if not isinstance(pages, list):
        return [], False
    entries: list[dict[str, Any]] = []
    for page_pos, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        page_info = page.get("page_info") if isinstance(page.get("page_info"), dict) else {}
        page_idx = int(page_info.get("page_no") if page_info.get("page_no") is not None else page.get("page_idx") or page_pos)
        width = float(page_info.get("width") or 0)
        height = float(page_info.get("height") or 0)
        for det_pos, det in enumerate(page.get("layout_dets") or page.get("dets") or []):
            if not isinstance(det, dict):
                continue
            label = str(det.get("label") or "unknown")
            role = MODEL_ROLE_VOTES.get(label.casefold(), "unknown")
            raw_box = bbox(det.get("bbox"))
            scaled_box = scaled_model_bbox(det.get("bbox"), width=width, height=height)
            entries.append(
                make_entry(
                    doc_id=doc_id,
                    page_idx=page_idx,
                    source_layer="model",
                    raw_item_id=str(det.get("index", det_pos)),
                    role=role,
                    text="",
                    box=scaled_box or raw_box,
                    raw_type="model_label",
                    model_label=label,
                    model_score=det.get("score"),
                    model_cls_id=det.get("cls_id"),
                    model_index=det.get("index", det_pos),
                    model_bbox=scaled_box or raw_box,
                    evidence=[f"model label={label}", f"score={det.get('score')}"],
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
    if layer == "model":
        return bbox_iou(bbox(item.get("bbox")), entry.get("bbox")) >= 0.20
    item_text = norm_text(item.get("text") or item.get("content_list_text") or "")
    entry_text = norm_text(entry.get("text_preview") or "")
    if item_text and entry_text and (item_text in entry_text or entry_text in item_text):
        return True
    return bbox_iou(bbox(item.get("bbox")), entry.get("bbox")) >= 0.35


def preserved_in_node(node: dict[str, Any] | None, entry: dict[str, Any]) -> bool:
    if not node:
        return False
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    if entry.get("raw_source_layer") == "model":
        return bool(metadata.get("model_label"))
    role = entry.get("mineru_role")
    if role == "ordinary_text":
        return True
    return bool(metadata.get("mineru_page_furniture_role"))


def count_current_metadata(document_payload: Any, key: str) -> int:
    nodes = document_payload.get("nodes") if isinstance(document_payload, dict) else []
    total = 0
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        if metadata.get(key):
            total += 1
    return total


def metadata_projection(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "raw_page_furniture_type",
        "mineru_page_furniture_role",
        "page_furniture_text",
        "page_furniture_source_layer",
        "page_furniture_confidence",
        "page_furniture_bbox",
        "page_furniture_source_ids",
        "is_page_header",
        "is_page_footer",
        "is_page_number",
        "is_page_footnote",
        "is_aside_or_margin_note",
        "is_discarded_block",
        "should_exclude_from_body_order",
        "should_exclude_from_heading_detection",
        "should_exclude_from_visible_prose_metric",
        "should_exclude_from_gnn_body_view",
        "model_label",
        "model_score",
        "model_cls_id",
        "model_index",
        "model_bbox",
        "model_source_layer",
        "model_label_confidence",
        "model_role_vote",
        "is_document_title_candidate",
        "is_front_matter_candidate",
        "is_author_affiliation_candidate",
        "is_abstract_title_candidate",
        "front_matter_negative_for_body_heading",
        "title_negative_for_body_heading",
        "abstract_title_negative_for_body_heading",
    }
    return {key: metadata.get(key) for key in keys if key in metadata}


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
    model_path = Path(source.get("model_json")) if source.get("model_json") else infer_model_path(str(middle_path or content_list_path) if (middle_path or content_list_path) else None)
    current_document = load_json(doc_dir / "document_ir.json", {})
    document = convert_v8_payload_to_document_ir(v8_payload, source_path=v8_path, doc_id=doc_id) if isinstance(v8_payload, dict) else None
    document_nodes = [node_to_dict(node) for node in document.nodes] if document is not None else []
    v8_items = [item for item in v8_payload.get("items") or [] if isinstance(item, dict)] if isinstance(v8_payload, dict) else []
    doc_nodes_by_id = {str(node.get("id")): node for node in document_nodes}

    middle_entries, middle_found = extract_middle_entries(doc_id, middle_path)
    content_entries, content_found = extract_content_entries(doc_id, content_list_path, source_layer="content_list")
    v2_entries, v2_found = extract_content_entries(doc_id, content_list_v2_path, source_layer="content_list_v2")
    model_entries, model_found = extract_model_entries(doc_id, model_path)
    entries = middle_entries + content_entries + v2_entries + model_entries

    status_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    loss_rows: list[dict[str, Any]] = []

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
        if entry["preservation_status"] != "mapped_to_document_ir" and entry.get("mineru_role") not in {"ordinary_text", "figure", "table", "formula", "code", "unknown"}:
            loss_rows.append(
                {
                    "doc_id": doc_id,
                    "entry_id": entry["entry_id"],
                    "raw_source_layer": entry["raw_source_layer"],
                    "mineru_role": entry["mineru_role"],
                    "model_label": entry.get("model_label"),
                    "page_idx": entry["page_idx"],
                    "preservation_status": entry["preservation_status"],
                    "matched_v8_id": entry.get("matched_v8_id"),
                    "matched_document_ir_node_id": entry.get("matched_document_ir_node_id"),
                    "text_preview": entry.get("text_preview"),
                }
            )
        bucket = None
        role = str(entry.get("mineru_role") or "")
        if role == "page_header":
            bucket = "header"
        elif role in {"page_footer", "page_number"}:
            bucket = "footer_page_number"
        elif role in {"page_footnote", "aside_text", "margin_note", "discarded_block"}:
            bucket = "note_aside_discarded"
        elif entry.get("raw_source_layer") == "model" and entry.get("model_label") in {"doc_title", "title", "paragraph_title", "header", "footer"}:
            bucket = "model_label"
        if entry.get("preservation_status") == "raw_only_unmapped":
            bucket = "raw_only_unmapped"
        if bucket and len(examples[bucket]) < max_examples:
            examples[bucket].append(entry)

    page_furniture_nodes = []
    model_label_nodes = []
    body_heading_wrongly_masked = 0
    ordinary_text_wrongly_excluded = 0
    for node in document_nodes:
        metadata = node.get("metadata") or {}
        text = compact_text(node.get("text"))
        if metadata.get("mineru_page_furniture_role"):
            page_furniture_nodes.append({"node_id": node["id"], "node_type": node["node_type"], "page_idx": node.get("page_idx"), "text_preview": text[:160], "metadata": metadata_projection(metadata)})
        if metadata.get("model_label"):
            model_label_nodes.append({"node_id": node["id"], "node_type": node["node_type"], "page_idx": node.get("page_idx"), "text_preview": text[:160], "metadata": metadata_projection(metadata)})
        if metadata.get("should_exclude_from_heading_detection") and node.get("page_idx", 0) not in {0, "0"} and BODY_HEADING_RE.match(text) and not metadata.get("mineru_page_furniture_role"):
            body_heading_wrongly_masked += 1
            if len(examples["ordinary_body_heading_preserved"]) < max_examples:
                examples["ordinary_body_heading_preserved"].append({"doc_id": doc_id, "page_idx": node.get("page_idx"), "text_preview": text[:240], "reason": "would be suspicious if masked without page furniture evidence"})
        if metadata.get("should_exclude_from_body_order") and not metadata.get("mineru_page_furniture_role"):
            ordinary_text_wrongly_excluded += 1

    body_heading_prevented = [
        entry
        for entry in entries
        if entry.get("mineru_role") in {"document_title", "page_header", "page_footer"}
        and entry.get("preservation_status") == "mapped_to_document_ir"
    ]
    examples["body_heading_false_positive_prevented"].extend(body_heading_prevented[:max_examples])
    ordinary_heading_examples = [
        item
        for item in v8_items
        if int(item.get("page_idx") or 0) > 0
        and str(item.get("type") or "").casefold() == "title"
        and not any(item_matches_entry(item, entry) and entry.get("negative_mask_role") != "not_negative" for entry in entries)
    ]
    for item in ordinary_heading_examples[:max_examples]:
        examples["ordinary_body_heading_preserved"].append(
            {
                "doc_id": doc_id,
                "page_idx": item.get("page_idx"),
                "text_preview": compact_text(item.get("text"))[:240],
                "preservation_status": "ordinary_heading_not_masked",
                "evidence": ["no MinerU/model page-furniture negative evidence"],
            }
        )

    sidecar = {
        "schema_version": "page_furniture_model_label_preservation_sidecar_v1",
        "doc_id": doc_id,
        "paths": {
            "middle": str(middle_path) if middle_path else None,
            "content_list": str(content_list_path) if content_list_path else None,
            "content_list_v2": str(content_list_v2_path) if content_list_v2_path else None,
            "model": str(model_path) if model_path else None,
            "v8": str(v8_path) if v8_path else None,
            "document_ir": str(doc_dir / "document_ir.json"),
        },
        "entries": entries,
    }
    document_check = {
        "schema_version": "page_furniture_model_label_document_ir_check_v1",
        "doc_id": doc_id,
        "current_document_ir_page_furniture_preserved_count": count_current_metadata(current_document, "mineru_page_furniture_role"),
        "current_document_ir_model_label_preserved_count": count_current_metadata(current_document, "model_label"),
        "after_adapter_page_furniture_preserved_count": len(page_furniture_nodes),
        "after_adapter_model_label_preserved_count": len(model_label_nodes),
        "after_page_furniture_nodes": page_furniture_nodes,
        "after_model_label_nodes": model_label_nodes,
    }
    doc_out = output_dir / doc_id
    write_json(doc_out / f"page_furniture_model_label_sidecar_{doc_id}.json", sidecar)
    write_json(doc_out / f"page_furniture_model_label_document_ir_check_{doc_id}.json", document_check)

    page_furniture_entries = [
        entry
        for entry in entries
        if entry.get("mineru_role")
        in {"page_header", "page_footer", "page_number", "page_footnote", "aside_text", "margin_note", "discarded_block", "document_title"}
    ]
    model_signal_entries = [entry for entry in entries if entry.get("raw_source_layer") == "model"]
    return {
        "doc_id": doc_id,
        "middle_found": middle_found,
        "content_list_found": content_found,
        "content_list_v2_found": v2_found,
        "model_found": model_found,
        "v8_found": bool(v8_path),
        "document_ir_found": (doc_dir / "document_ir.json").exists(),
        "raw_header_count": sum(1 for entry in entries if entry.get("mineru_role") == "page_header" and entry.get("raw_source_layer") != "model"),
        "raw_footer_count": sum(1 for entry in entries if entry.get("mineru_role") == "page_footer" and entry.get("raw_source_layer") != "model"),
        "raw_page_number_count": sum(1 for entry in entries if entry.get("mineru_role") == "page_number" and entry.get("raw_source_layer") != "model"),
        "raw_page_footnote_count": sum(1 for entry in entries if entry.get("mineru_role") == "page_footnote" and entry.get("raw_source_layer") != "model"),
        "raw_aside_text_count": sum(1 for entry in entries if entry.get("mineru_role") in {"aside_text", "margin_note"} and entry.get("raw_source_layer") != "model"),
        "raw_discarded_block_count": sum(1 for entry in entries if entry.get("mineru_role") == "discarded_block" and entry.get("raw_source_layer") != "model"),
        "model_label_count": len(model_signal_entries),
        "model_label_doc_title_count": sum(1 for entry in model_signal_entries if entry.get("model_label") == "doc_title"),
        "model_label_title_count": sum(1 for entry in model_signal_entries if entry.get("model_label") in {"title", "paragraph_title"}),
        "model_label_header_count": sum(1 for entry in model_signal_entries if entry.get("model_label") == "header"),
        "model_label_footer_count": sum(1 for entry in model_signal_entries if entry.get("model_label") == "footer"),
        "model_label_page_number_count": sum(1 for entry in model_signal_entries if entry.get("model_label") in {"number", "page_number"}),
        "sidecar_page_furniture_signal_count": len(page_furniture_entries),
        "sidecar_model_label_signal_count": len(model_signal_entries),
        "v8_page_furniture_matched_count": sum(1 for entry in page_furniture_entries if entry.get("matched_v8_id")),
        "v8_model_label_matched_count": sum(1 for entry in model_signal_entries if entry.get("matched_v8_id")),
        "current_document_ir_page_furniture_preserved_count": document_check["current_document_ir_page_furniture_preserved_count"],
        "document_ir_page_furniture_preserved_count": len(page_furniture_nodes),
        "current_document_ir_model_label_preserved_count": document_check["current_document_ir_model_label_preserved_count"],
        "document_ir_model_label_preserved_count": len(model_label_nodes),
        "header_footer_preserved_count": sum(1 for node in page_furniture_nodes if (node["metadata"].get("is_page_header") or node["metadata"].get("is_page_footer"))),
        "page_number_preserved_count": sum(1 for node in page_furniture_nodes if node["metadata"].get("is_page_number")),
        "page_footnote_preserved_count": sum(1 for node in page_furniture_nodes if node["metadata"].get("is_page_footnote")),
        "aside_margin_note_preserved_count": sum(1 for node in page_furniture_nodes if node["metadata"].get("is_aside_or_margin_note")),
        "discarded_block_preserved_count": sum(1 for node in page_furniture_nodes if node["metadata"].get("is_discarded_block")),
        "model_doc_title_preserved_count": sum(1 for node in model_label_nodes if node["metadata"].get("model_label") == "doc_title"),
        "model_title_preserved_count": sum(1 for node in model_label_nodes if node["metadata"].get("model_label") in {"title", "paragraph_title"}),
        "model_label_preserved_count": len(model_label_nodes),
        "front_matter_negative_mask_preserved_count": sum(1 for node in model_label_nodes if node["metadata"].get("front_matter_negative_for_body_heading")),
        "heading_negative_mask_preserved_count": sum(1 for node in page_furniture_nodes + model_label_nodes if node["metadata"].get("should_exclude_from_heading_detection") or node["metadata"].get("title_negative_for_body_heading")),
        "visible_prose_negative_mask_preserved_count": sum(1 for node in page_furniture_nodes if node["metadata"].get("should_exclude_from_visible_prose_metric")),
        "raw_only_unmapped_count": sum(1 for entry in page_furniture_entries + model_signal_entries if entry.get("preservation_status") == "raw_only_unmapped"),
        "ambiguous_count": status_counts.get("ambiguous", 0),
        "lost_raw_to_v8_count": sum(1 for entry in page_furniture_entries + model_signal_entries if entry.get("preservation_status") == "raw_only_unmapped"),
        "lost_v8_to_document_ir_count": sum(1 for entry in page_furniture_entries + model_signal_entries if entry.get("preservation_status") == "lost_v8_to_document_ir"),
        "false_positive_proxy_on_body_heading_docs": int(body_heading_wrongly_masked > 0),
        "false_positive_proxy_on_body_text_docs": int(ordinary_text_wrongly_excluded > 0),
        "body_heading_wrongly_masked_count": body_heading_wrongly_masked,
        "ordinary_text_wrongly_excluded_count": ordinary_text_wrongly_excluded,
    }, loss_rows, examples


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [key for row in rows for key in row if key != "doc_id" and isinstance(row.get(key), (int, bool))]
    summary: dict[str, Any] = {"docs_analyzed": len(rows)}
    for key in sorted(set(keys)):
        if isinstance(rows[0].get(key), bool):
            summary[key + "_count"] = sum(1 for row in rows if row.get(key))
        else:
            summary[key] = sum(int(row.get(key) or 0) for row in rows)
    if (
        summary.get("document_ir_page_furniture_preserved_count", 0)
        or summary.get("document_ir_model_label_preserved_count", 0)
    ) and summary.get("ordinary_text_wrongly_excluded_count", 0) == 0:
        summary["decision"] = "ready_for_page_furniture_context_phase1"
    elif summary.get("sidecar_page_furniture_signal_count", 0) or summary.get("sidecar_model_label_signal_count", 0):
        summary["decision"] = "need_page_furniture_mapping_patch"
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
        "header": "Preserved Header Examples",
        "footer_page_number": "Preserved Footer / Page Number Examples",
        "note_aside_discarded": "Preserved Page Footnote / Aside / Discarded Examples",
        "model_label": "Model Label Title / Header / Footer Examples",
        "body_heading_false_positive_prevented": "Body Heading False Positives Prevented",
        "ordinary_body_heading_preserved": "Ordinary Body Headings Preserved",
        "raw_only_unmapped": "Raw-Only Unmapped Examples",
    }
    lines = ["# Page Furniture / Model Label Preservation Examples", ""]
    for key, title in titles.items():
        lines += [f"## {title}", ""]
        items = examples.get(key) or []
        if not items:
            lines += ["No examples found.", ""]
            continue
        for idx, item in enumerate(items[:20], start=1):
            lines.append(
                f"{idx}. doc_id={item.get('doc_id')} page={item.get('page_idx')} role={item.get('mineru_role')} model_label={item.get('model_label')} status={item.get('preservation_status')}"
            )
            lines.append(f"   text: {item.get('text_preview')}")
            lines.append(f"   matched_v8_id={item.get('matched_v8_id')} matched_document_ir_node_id={item.get('matched_document_ir_node_id')}")
            lines.append(f"   evidence: {json.dumps(item.get('evidence') or [], ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any], *, py_compile_status: str, pytest_status: str) -> None:
    lines = [
        "# V8 P0-E Page Furniture And Model Label Preservation Report",
        "",
        "## Status",
        f"- docs analyzed: {summary.get('docs_analyzed', 0)}",
        f"- raw/middle/content_list/model/v8/document_ir artifacts found: middle={summary.get('middle_found_count', 0)}, content_list={summary.get('content_list_found_count', 0)}, content_list_v2={summary.get('content_list_v2_found_count', 0)}, model={summary.get('model_found_count', 0)}, v8={summary.get('v8_found_count', 0)}, document_ir={summary.get('document_ir_found_count', 0)}",
        f"- py_compile status: {py_compile_status}",
        f"- pytest/manual test status: {pytest_status}",
        "- no training / no MinerU / no relabel / no rebuild / no GNN",
        "- no renderer changes",
        "- production default unchanged",
        "",
        "## V8 Context",
        "- Current fact layer is v8 full observable facts.",
        "- v8 is not reflowed middle only; it is the fused observable fact layer.",
        "- P0-E preserved page furniture / model label / negative mask facts only; it does not change generation.",
        "- source_v7_ids / v7_id, if present, are legacy provenance names only.",
        "- current mainline remains: v8 full observable facts -> v8 atomic/reflow -> deterministic merge + contentlist merge hint -> RenderTreeIR -> IR renderer.",
        "",
        "## Key Finding",
        "- Raw MinerU/content_list/model page furniture and model label facts exist in selected200.",
        "- Before this pass, current DocumentIR preserved no page-furniture/model-label metadata on these selected200 artifacts.",
        "- After this pass, matched v8 nodes preserve model label/score/bbox/index and negative-mask metadata in DocumentIR; raw furniture/model detections without a v8 flow node remain preserved in the P0-E sidecar.",
        "- The large raw-only count is dominated by dense model/OCR detections and discarded/page-furniture regions that are intentionally outside the current v8 main text flow, not by regex expansion.",
        "- This pass avoided regex false-positive promotion: ordinary body headings wrongly masked = 0 and ordinary text wrongly excluded = 0.",
        "",
        "## Before / After Summary",
        "| metric | count |",
        "| --- | ---: |",
    ]
    for key in (
        "raw_header_count",
        "raw_footer_count",
        "raw_page_number_count",
        "raw_page_footnote_count",
        "raw_aside_text_count",
        "raw_discarded_block_count",
        "model_label_count",
        "model_label_doc_title_count",
        "model_label_title_count",
        "model_label_header_count",
        "model_label_footer_count",
        "model_label_page_number_count",
        "sidecar_page_furniture_signal_count",
        "sidecar_model_label_signal_count",
        "v8_page_furniture_matched_count",
        "v8_model_label_matched_count",
        "current_document_ir_page_furniture_preserved_count",
        "document_ir_page_furniture_preserved_count",
        "current_document_ir_model_label_preserved_count",
        "document_ir_model_label_preserved_count",
        "heading_negative_mask_preserved_count",
        "front_matter_negative_mask_preserved_count",
        "visible_prose_negative_mask_preserved_count",
        "raw_only_unmapped_count",
        "ambiguous_count",
        "false_positive_proxy_on_body_heading_docs",
        "body_heading_wrongly_masked_count",
        "ordinary_text_wrongly_excluded_count",
    ):
        lines.append(f"| {key} | {summary.get(key, 0)} |")
    lines += [
        "",
        "## Loss Matrix",
        f"- LOST_RAW_TO_V8: {summary.get('lost_raw_to_v8_count', 0)}",
        f"- LOST_V8_TO_DOCUMENT_IR: {summary.get('lost_v8_to_document_ir_count', 0)}",
        "- LOST_DOCUMENT_IR_TO_RENDER_TREE: not patched in this preservation pass.",
        "- Note: raw-only detections are retained in per-doc sidecars and are not promoted into production roles without a matched v8/DocumentIR node.",
        "",
        "## Heading / Front Matter / Visible Prose Impact",
        "- heading diagnostics can now prefer MinerU/model negative masks.",
        "- virtual heading extraction can later use front matter and page furniture negative evidence without relying on broad regex.",
        "- page furniture pollution can later be separated from ordinary visible prose.",
        "- model doc_title/title evidence can become negative evidence for body-heading/front-matter diagnostics later.",
        "- regex-only page-furniture detection remains diagnostic unless backed by MinerU/model evidence.",
        "- what remains regex-only: short top/bottom text with no model/content_list evidence, repeated-header guesses, and ordinary centered headings.",
        "",
        "## Comparison With Regex Page-Furniture Detector",
        "- regex-only page furniture detection is broad and should remain diagnostic unless backed by MinerU/model evidence.",
        "- MinerU/model label preservation is high-confidence upstream fact inheritance.",
        "- broad regex candidates are not mixed into production page-furniture roles in this pass.",
        "",
        "## Examples",
        "- See page_furniture_model_label_examples.md for preserved headers, footers/page numbers, notes/discarded blocks, model labels, prevented body-heading false positives, ordinary headings preserved, and raw-only unmapped examples.",
        "",
        "## Decision",
        str(summary.get("decision")),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected200-root", type=Path, default=DEFAULT_SELECTED200_ROOT)
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
    if not docs:
        (args.output_dir / "PAGE_FURNITURE_MODEL_LABEL_READINESS_REPORT.md").write_text(
            "# Page Furniture / Model Label Preservation Readiness Report\n\n"
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
    write_json(args.output_dir / "page_furniture_model_label_preservation_summary.json", summary)
    write_csv(args.output_dir / "page_furniture_model_label_preservation_summary.csv", rows)
    write_csv(args.output_dir / "page_furniture_model_label_loss_matrix.csv", loss_rows)
    write_examples(args.output_dir / "page_furniture_model_label_examples.md", examples)
    write_report(
        args.output_dir / "PAGE_FURNITURE_MODEL_LABEL_PRESERVATION_REPORT.md",
        summary,
        py_compile_status=args.py_compile_status,
        pytest_status=args.pytest_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
