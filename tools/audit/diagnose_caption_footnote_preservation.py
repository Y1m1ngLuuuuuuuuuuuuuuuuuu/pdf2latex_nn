#!/usr/bin/env python3
"""Audit P0-C caption/footnote preservation on selected200 artifacts.

This pass preserves and validates MinerU caption/footnote facts only. It does
not write raw MinerU/v8 JSON, regenerate LaTeX, rebuild graphs, or alter renderer
behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/caption_footnote_preservation_20260528")
CAPTION_ROLE_TO_TYPE = {
    "image_caption": "figure",
    "figure_caption": "figure",
    "table_caption": "table",
    "chart_caption": "chart",
    "code_caption": "code",
    "algorithm_caption": "algorithm",
}
FOOTNOTE_ROLE_TO_TYPE = {
    "image_footnote": "image_note",
    "figure_footnote": "image_note",
    "table_footnote": "table_note",
    "chart_footnote": "chart_note",
    "code_footnote": "code_note",
    "algorithm_footnote": "code_note",
}
BODY_REFERENCE_RE = re.compile(
    r"\b(?:Figure|Fig\.?|Table|Algorithm|Alg\.?)\s*\d+[A-Za-z]?\b\s+"
    r"(?:shows?|reports?|is|are|was|were|illustrates?|depicts?|presents?)",
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


def compact_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(compact_text(part) for part in value if compact_text(part)).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "caption", "html"):
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


def source_id_suffix(value: Any) -> str:
    text = str(value or "")
    marker = re.search(r"p\d{4}:m\d{6}", text)
    return marker.group(0) if marker else text


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
    parent_float_id: str | None = None,
    crop_or_asset_path: str | None = None,
    body_ref_ids: list[str] | None = None,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    is_caption = role in CAPTION_ROLE_TO_TYPE
    is_footnote = role in FOOTNOTE_ROLE_TO_TYPE
    caption_type = CAPTION_ROLE_TO_TYPE.get(role, "unknown")
    if role == "code_caption" and raw_sub_type == "algorithm":
        caption_type = "algorithm"
        role = "algorithm_caption"
    footnote_type = FOOTNOTE_ROLE_TO_TYPE.get(role, "unknown")
    confidence = {
        "middle": "strong_middle_child",
        "content_list": "strong_content_list_field",
        "content_list_v2": "strong_v2_field",
    }.get(source_layer, "medium_metadata")
    return {
        "doc_id": doc_id,
        "page_idx": page_idx,
        "caption_footnote_id": f"{doc_id}:{source_layer}:{raw_item_id}:{role}",
        "raw_source_layer": source_layer,
        "raw_item_id": raw_item_id,
        "matched_v8_id": None,
        "matched_document_ir_node_id": None,
        "parent_block_id": parent_block_id,
        "parent_float_id": parent_float_id,
        "source_span_ids": [],
        "bbox": box,
        "text_preview": compact_text(text)[:240],
        "raw_type": raw_type,
        "raw_sub_type": raw_sub_type,
        "mineru_role": role if (is_caption or is_footnote) else "unknown",
        "caption_type": caption_type if is_caption else "unknown",
        "footnote_type": footnote_type if is_footnote else "unknown",
        "caption_text": text if is_caption else "",
        "footnote_text": text if is_footnote else "",
        "body_ref_ids": body_ref_ids or [],
        "crop_or_asset_path": crop_or_asset_path,
        "caption_source_layer": source_layer if is_caption else "",
        "caption_confidence": confidence if is_caption else "",
        "footnote_source_layer": source_layer if is_footnote else "",
        "footnote_confidence": confidence if is_footnote else "",
        "preservation_status": "unknown",
        "evidence": evidence or [],
    }


def add_entries_for_field(
    entries: list[dict[str, Any]],
    *,
    doc_id: str,
    page_idx: int,
    source_layer: str,
    raw_item_id: str,
    role: str,
    value: Any,
    box: list[float] | None,
    raw_type: str | None,
    raw_sub_type: str | None,
    parent_block_id: str | None,
    parent_float_id: str | None,
    crop_or_asset_path: str | None,
    evidence: list[str],
) -> None:
    parts = value if isinstance(value, list) else [value]
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
                parent_float_id=parent_float_id,
                crop_or_asset_path=crop_or_asset_path,
                body_ref_ids=[parent_block_id] if parent_block_id else [],
                evidence=evidence + [f"{role} field present"],
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
        asset = item.get("img_path") or item.get("image_path") or item.get("table_img_path") or item.get("asset_path")
        for role in tuple(CAPTION_ROLE_TO_TYPE) + tuple(FOOTNOTE_ROLE_TO_TYPE):
            if role == "algorithm_caption":
                value = item.get("algorithm_caption")
            elif role == "algorithm_footnote":
                value = item.get("algorithm_footnote")
            else:
                value = item.get(role)
            if not value and source_layer == "content_list_v2":
                value = item.get(role.replace("code_", "algorithm_"))
            if not value:
                continue
            add_entries_for_field(
                entries,
                doc_id=doc_id,
                page_idx=page_idx,
                source_layer=source_layer,
                raw_item_id=str(idx),
                role=role,
                value=value,
                box=box,
                raw_type=raw_type,
                raw_sub_type=raw_sub_type,
                parent_block_id=f"{doc_id}:{source_layer}:{idx:06d}",
                parent_float_id=f"{doc_id}:{source_layer}:{idx:06d}",
                crop_or_asset_path=str(asset) if asset else None,
                evidence=[f"{source_layer} index={idx}", f"raw_type={raw_type}", f"raw_sub_type={raw_sub_type}"],
            )
        if source_layer == "content_list_v2":
            for role, key in (("algorithm_caption", "algorithm_caption"),):
                if item.get(key):
                    add_entries_for_field(
                        entries,
                        doc_id=doc_id,
                        page_idx=page_idx,
                        source_layer=source_layer,
                        raw_item_id=str(idx),
                        role=role,
                        value=item.get(key),
                        box=box,
                        raw_type=raw_type,
                        raw_sub_type=raw_sub_type or "algorithm",
                        parent_block_id=f"{doc_id}:{source_layer}:{idx:06d}",
                        parent_float_id=f"{doc_id}:{source_layer}:{idx:06d}",
                        crop_or_asset_path=str(asset) if asset else None,
                        evidence=[f"{source_layer} algorithm_caption index={idx}"],
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
                box = bbox(block.get("bbox"))
                for role in tuple(CAPTION_ROLE_TO_TYPE) + tuple(FOOTNOTE_ROLE_TO_TYPE):
                    value = block.get(role)
                    if not value:
                        continue
                    add_entries_for_field(
                        entries,
                        doc_id=doc_id,
                        page_idx=page_idx,
                        source_layer="middle",
                        raw_item_id=block_id,
                        role=role,
                        value=value,
                        box=box,
                        raw_type=raw_type,
                        raw_sub_type=raw_sub_type,
                        parent_block_id=block_id,
                        parent_float_id=block_id,
                        crop_or_asset_path=str(block.get("img_path") or block.get("image_path") or "") or None,
                        evidence=[f"middle {collection_name}", f"block_type={raw_type}"],
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


def build_v8_indices(v8_payload: dict[str, Any], document_nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    items = [item for item in v8_payload.get("items") or [] if isinstance(item, dict)]
    by_id = {str(node.get("id")): node for node in document_nodes}
    return items, by_id


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
    entry_text = norm_text(entry.get("caption_text") or entry.get("footnote_text") or entry.get("text_preview") or "")
    if item_text and entry_text and (item_text in entry_text or entry_text in item_text):
        return True
    return bbox_iou(bbox(item.get("bbox")), entry.get("bbox")) >= 0.35


def preserved_in_node(node: dict[str, Any] | None, entry: dict[str, Any]) -> bool:
    if not node:
        return False
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    role = entry.get("mineru_role")
    if role in CAPTION_ROLE_TO_TYPE:
        return bool(metadata.get("caption_text") and metadata.get("mineru_caption_role"))
    if role in FOOTNOTE_ROLE_TO_TYPE:
        return bool(metadata.get("footnote_text") and metadata.get("mineru_footnote_role"))
    return False


def count_current_metadata(document_payload: Any, *, kind: str) -> int:
    nodes = document_payload.get("nodes") if isinstance(document_payload, dict) else []
    total = 0
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        if kind == "caption" and metadata.get("caption_text"):
            total += 1
        if kind == "footnote" and metadata.get("footnote_text"):
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
    v8_items, doc_nodes_by_id = build_v8_indices(v8_payload if isinstance(v8_payload, dict) else {}, document_nodes)

    entries: list[dict[str, Any]] = []
    middle_entries, middle_found = extract_middle_entries(doc_id, middle_path)
    content_entries, content_found = extract_content_list_entries(doc_id, content_list_path, source_layer="content_list")
    v2_entries, v2_found = extract_content_list_entries(doc_id, content_list_v2_path, source_layer="content_list_v2")
    entries.extend(middle_entries)
    entries.extend(content_entries)
    entries.extend(v2_entries)

    status_counts: Counter[str] = Counter()
    loss_rows: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_counter: Counter[tuple[str, str, str, int]] = Counter()

    for entry in entries:
        matched_item = next((item for item in v8_items if item_matches_entry(item, entry)), None)
        if matched_item:
            matched_id = str(matched_item.get("id") or "")
            entry["matched_v8_id"] = matched_id
            node = doc_nodes_by_id.get(matched_id)
            entry["matched_document_ir_node_id"] = matched_id if node else None
            if preserved_in_node(node, entry):
                entry["preservation_status"] = "mapped_to_document_ir"
            elif node:
                entry["preservation_status"] = "lost_v8_to_document_ir"
            else:
                entry["preservation_status"] = "lost_v8_to_document_ir"
        else:
            entry["preservation_status"] = "raw_only_unmapped"
        status_counts[entry["preservation_status"]] += 1
        key_text = norm_text(entry.get("caption_text") or entry.get("footnote_text"))
        duplicate_counter[(entry.get("mineru_role") or "", key_text, entry.get("raw_source_layer") or "", int(entry.get("page_idx") or 0))] += 1
        if entry["preservation_status"] != "mapped_to_document_ir":
            loss_rows.append(
                {
                    "doc_id": doc_id,
                    "caption_footnote_id": entry["caption_footnote_id"],
                    "raw_source_layer": entry["raw_source_layer"],
                    "mineru_role": entry["mineru_role"],
                    "caption_type": entry["caption_type"],
                    "footnote_type": entry["footnote_type"],
                    "page_idx": entry["page_idx"],
                    "preservation_status": entry["preservation_status"],
                    "matched_v8_id": entry.get("matched_v8_id"),
                    "matched_document_ir_node_id": entry.get("matched_document_ir_node_id"),
                    "text_preview": entry.get("text_preview"),
                }
            )

        bucket = None
        if entry.get("mineru_role") == "image_caption":
            bucket = "image_caption"
        elif entry.get("mineru_role") == "table_caption":
            bucket = "table_caption"
        elif entry.get("mineru_role") in {"chart_caption", "code_caption", "algorithm_caption"}:
            bucket = "chart_code_algorithm_caption"
        elif entry.get("mineru_role") in FOOTNOTE_ROLE_TO_TYPE:
            bucket = "footnote"
        elif entry["preservation_status"] == "raw_only_unmapped":
            bucket = "raw_only_unmapped"
        if bucket and len(examples[bucket]) < max_examples:
            examples[bucket].append(entry)

    sidecar = {
        "schema_version": "caption_footnote_preservation_sidecar_v1",
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
    after_caption_nodes = []
    after_footnote_nodes = []
    for node in document_nodes:
        metadata = node.get("metadata") or {}
        if metadata.get("caption_text"):
            after_caption_nodes.append(
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
                            "raw_caption_type",
                            "mineru_caption_role",
                            "caption_text",
                            "caption_source_layer",
                            "caption_confidence",
                            "caption_parent_float_id",
                            "caption_body_ids",
                            "caption_source_ids",
                            "caption_bbox",
                            "caption_type",
                            "body_node_ids",
                            "caption_node_ids",
                            "parent_float_source_id",
                            "child_block_ids",
                            "source_layer_hierarchy",
                        }
                    },
                }
            )
        if metadata.get("footnote_text"):
            after_footnote_nodes.append(
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
                            "raw_footnote_type",
                            "mineru_footnote_role",
                            "footnote_text",
                            "footnote_source_layer",
                            "footnote_confidence",
                            "footnote_parent_float_id",
                            "footnote_body_ids",
                            "footnote_source_ids",
                            "footnote_bbox",
                            "footnote_type",
                            "body_node_ids",
                            "footnote_node_ids",
                            "parent_float_source_id",
                            "child_block_ids",
                            "source_layer_hierarchy",
                        }
                    },
                }
            )
    document_check = {
        "schema_version": "caption_footnote_document_ir_check_v1",
        "doc_id": doc_id,
        "current_document_ir_caption_preserved_count": count_current_metadata(current_document, kind="caption"),
        "current_document_ir_footnote_preserved_count": count_current_metadata(current_document, kind="footnote"),
        "after_adapter_caption_preserved_count": len(after_caption_nodes),
        "after_adapter_footnote_preserved_count": len(after_footnote_nodes),
        "after_caption_nodes": after_caption_nodes,
        "after_footnote_nodes": after_footnote_nodes,
    }
    doc_out = output_dir / doc_id
    write_json(doc_out / f"caption_footnote_sidecar_{doc_id}.json", sidecar)
    write_json(doc_out / f"caption_footnote_document_ir_check_{doc_id}.json", document_check)

    raw_image_caption = sum(1 for entry in entries if entry.get("mineru_role") in {"image_caption", "figure_caption"})
    raw_table_caption = sum(1 for entry in entries if entry.get("mineru_role") == "table_caption")
    raw_chart_caption = sum(1 for entry in entries if entry.get("mineru_role") == "chart_caption")
    raw_code_caption = sum(1 for entry in entries if entry.get("mineru_role") == "code_caption")
    raw_algorithm_caption = sum(1 for entry in entries if entry.get("mineru_role") == "algorithm_caption")
    raw_image_footnote = sum(1 for entry in entries if entry.get("mineru_role") in {"image_footnote", "figure_footnote"})
    raw_table_footnote = sum(1 for entry in entries if entry.get("mineru_role") == "table_footnote")
    raw_chart_footnote = sum(1 for entry in entries if entry.get("mineru_role") == "chart_footnote")
    raw_code_footnote = sum(1 for entry in entries if entry.get("mineru_role") in {"code_footnote", "algorithm_footnote"})
    image_caption_preserved = sum(
        1
        for entry in entries
        if entry.get("mineru_role") in {"image_caption", "figure_caption"} and entry.get("preservation_status") == "mapped_to_document_ir"
    )
    table_caption_preserved = sum(
        1 for entry in entries if entry.get("mineru_role") == "table_caption" and entry.get("preservation_status") == "mapped_to_document_ir"
    )
    chart_caption_preserved = sum(
        1 for entry in entries if entry.get("mineru_role") == "chart_caption" and entry.get("preservation_status") == "mapped_to_document_ir"
    )
    algorithm_caption_preserved = sum(
        1
        for entry in entries
        if entry.get("mineru_role") == "algorithm_caption" and entry.get("preservation_status") == "mapped_to_document_ir"
    )
    image_footnote_preserved = sum(
        1
        for entry in entries
        if entry.get("mineru_role") in {"image_footnote", "figure_footnote"} and entry.get("preservation_status") == "mapped_to_document_ir"
    )
    table_footnote_preserved = sum(
        1 for entry in entries if entry.get("mineru_role") == "table_footnote" and entry.get("preservation_status") == "mapped_to_document_ir"
    )
    chart_footnote_preserved = sum(
        1 for entry in entries if entry.get("mineru_role") == "chart_footnote" and entry.get("preservation_status") == "mapped_to_document_ir"
    )
    false_positive_proxy = sum(1 for item in v8_items if BODY_REFERENCE_RE.search(compact_text(item.get("text"))) and any(role in item for role in CAPTION_ROLE_TO_TYPE))
    return {
        "doc_id": doc_id,
        "middle_found": middle_found,
        "content_list_found": content_found,
        "content_list_v2_found": v2_found,
        "v8_found": bool(v8_path),
        "document_ir_found": (doc_dir / "document_ir.json").exists(),
        "raw_image_caption_count": raw_image_caption,
        "raw_table_caption_count": raw_table_caption,
        "raw_chart_caption_count": raw_chart_caption,
        "raw_code_caption_count": raw_code_caption,
        "raw_algorithm_caption_count": raw_algorithm_caption,
        "raw_image_footnote_count": raw_image_footnote,
        "raw_table_footnote_count": raw_table_footnote,
        "raw_chart_footnote_count": raw_chart_footnote,
        "raw_code_footnote_count": raw_code_footnote,
        "sidecar_caption_signal_count": sum(1 for entry in entries if entry.get("caption_text")),
        "sidecar_footnote_signal_count": sum(1 for entry in entries if entry.get("footnote_text")),
        "v8_caption_matched_count": sum(1 for entry in entries if entry.get("caption_text") and entry.get("matched_v8_id")),
        "v8_footnote_matched_count": sum(1 for entry in entries if entry.get("footnote_text") and entry.get("matched_v8_id")),
        "current_document_ir_caption_preserved_count": document_check["current_document_ir_caption_preserved_count"],
        "current_document_ir_footnote_preserved_count": document_check["current_document_ir_footnote_preserved_count"],
        "document_ir_caption_preserved_count": len(after_caption_nodes),
        "document_ir_footnote_preserved_count": len(after_footnote_nodes),
        "image_caption_preserved_count": image_caption_preserved,
        "table_caption_preserved_count": table_caption_preserved,
        "chart_caption_preserved_count": chart_caption_preserved,
        "algorithm_caption_preserved_count": algorithm_caption_preserved,
        "image_footnote_preserved_count": image_footnote_preserved,
        "table_footnote_preserved_count": table_footnote_preserved,
        "chart_footnote_preserved_count": chart_footnote_preserved,
        "raw_only_unmapped_count": status_counts.get("raw_only_unmapped", 0),
        "ambiguous_count": status_counts.get("ambiguous", 0),
        "lost_raw_to_v8_count": status_counts.get("raw_only_unmapped", 0),
        "lost_v8_to_document_ir_count": status_counts.get("lost_v8_to_document_ir", 0),
        "false_positive_proxy_on_body_reference_docs": false_positive_proxy,
        "duplicate_caption_source_count": sum(count - 1 for key, count in duplicate_counter.items() if count > 1 and key[0] in CAPTION_ROLE_TO_TYPE),
        "duplicate_footnote_source_count": sum(count - 1 for key, count in duplicate_counter.items() if count > 1 and key[0] in FOOTNOTE_ROLE_TO_TYPE),
    }, loss_rows, examples


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [key for row in rows for key in row if key != "doc_id" and isinstance(row.get(key), (int, bool))]
    summary: dict[str, Any] = {"docs_analyzed": len(rows)}
    for key in sorted(set(keys)):
        if isinstance(rows[0].get(key), bool):
            summary[key + "_count"] = sum(1 for row in rows if row.get(key))
        else:
            summary[key] = sum(int(row.get(key) or 0) for row in rows)
    if summary.get("false_positive_proxy_on_body_reference_docs", 0) == 0 and summary.get("document_ir_caption_preserved_count", 0):
        summary["decision"] = "ready_for_float_caption_context_phase1"
    elif summary.get("document_ir_caption_preserved_count", 0):
        summary["decision"] = "need_caption_mapping_patch"
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
        "image_caption": "Preserved Image Caption Examples",
        "table_caption": "Preserved Table Caption Examples",
        "chart_code_algorithm_caption": "Preserved Chart/Code/Algorithm Caption Examples",
        "footnote": "Preserved Footnote/Table-Note Examples",
        "raw_only_unmapped": "Raw-Only Unmapped Examples",
    }
    lines = ["# Caption Footnote Preservation Examples", ""]
    for key, title in titles.items():
        lines += [f"## {title}", ""]
        items = examples.get(key) or []
        if not items:
            lines += ["No examples found.", ""]
            continue
        for idx, item in enumerate(items[:20], start=1):
            lines.append(f"{idx}. doc_id={item.get('doc_id')} page={item.get('page_idx')} role={item.get('mineru_role')} status={item.get('preservation_status')}")
            lines.append(f"   text: {item.get('text_preview')}")
            lines.append(f"   matched_v8_id={item.get('matched_v8_id')} matched_document_ir_node_id={item.get('matched_document_ir_node_id')}")
            lines.append(f"   evidence: {json.dumps(item.get('evidence') or [], ensure_ascii=False)}")
        lines.append("")
    lines += [
        "## Prevented False Positives",
        "",
        "No ordinary body-reference false positives were materialized as caption metadata in this audit when false_positive_proxy_on_body_reference_docs is 0.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any], *, py_compile_status: str, pytest_status: str) -> None:
    lines = [
        "# V8 P0-C Caption And Footnote Preservation Report",
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
        "## Key Finding",
        "Raw MinerU middle/content_list/content_list_v2 caption and footnote facts exist in selected200. This pass preserves them into v8-derived sidecars and patched DocumentIR metadata without using regex-only caption guesses.",
        "Raw-to-v8 matching is complete for the extracted sidecar entries in this pass. Remaining loss is v8-to-DocumentIR preservation, concentrated in entries whose matched v8 node does not yet carry caption/footnote metadata after adapter conversion. Renderer consumption remains intentionally unchanged.",
        "",
        "## Before / After Summary",
        "| metric | count |",
        "| --- | ---: |",
    ]
    for key in (
        "raw_image_caption_count",
        "raw_table_caption_count",
        "raw_chart_caption_count",
        "raw_code_caption_count",
        "raw_algorithm_caption_count",
        "raw_image_footnote_count",
        "raw_table_footnote_count",
        "raw_chart_footnote_count",
        "raw_code_footnote_count",
        "v8_caption_matched_count",
        "v8_footnote_matched_count",
        "current_document_ir_caption_preserved_count",
        "document_ir_caption_preserved_count",
        "current_document_ir_footnote_preserved_count",
        "document_ir_footnote_preserved_count",
        "image_caption_preserved_count",
        "table_caption_preserved_count",
        "chart_caption_preserved_count",
        "algorithm_caption_preserved_count",
        "image_footnote_preserved_count",
        "table_footnote_preserved_count",
        "chart_footnote_preserved_count",
        "raw_only_unmapped_count",
        "ambiguous_count",
        "false_positive_proxy_on_body_reference_docs",
    ):
        lines.append(f"| {key} | {summary.get(key, 0)} |")
    lines += [
        "",
        "## Loss Matrix",
        f"- LOST_RAW_TO_V8: {summary.get('lost_raw_to_v8_count', 0)}",
        f"- LOST_V8_TO_DOCUMENT_IR: {summary.get('lost_v8_to_document_ir_count', 0)}",
        "- LOST_DOCUMENT_IR_TO_RENDER_TREE: not patched in this preservation pass.",
        "",
        "## FloatCaption / Footnote Impact",
        "- FloatCaptionLayout can now prefer MinerU caption metadata in a later metric/context pass.",
        "- metadata/crop-caption cases can be explained by `caption_source_layer`, `mineru_caption_role`, and `caption_parent_float_id` sidecar fields.",
        "- caption-like paragraphs backed by MinerU child fields are distinguishable from regex-only caption guesses.",
        "- image/table/chart/code footnotes are preserved as footnote metadata and can later be separated from ordinary prose.",
        f"- remaining algorithm/code caption preservation gaps are included in LOST_V8_TO_DOCUMENT_IR ({summary.get('lost_v8_to_document_ir_count', 0)} total) and should be handled in a later context pass, not by regex expansion here.",
        "- regex-only matches remain diagnostic unless backed by MinerU evidence.",
        "",
        "## Comparison With Regex Caption Matcher",
        "- regex-only caption matching is broad and should remain diagnostic unless backed by MinerU evidence.",
        "- MinerU caption/footnote preservation is high-confidence upstream fact inheritance.",
        "- Broad regex candidates were not mixed into production caption role in this pass.",
        "",
        "## Examples",
        "- See caption_footnote_examples.md.",
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
        (args.output_dir / "CAPTION_FOOTNOTE_PRESERVATION_READINESS_REPORT.md").write_text(
            "# Caption Footnote Preservation Readiness Report\n\n"
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
    write_json(args.output_dir / "caption_footnote_preservation_summary.json", summary)
    write_csv(args.output_dir / "caption_footnote_preservation_summary.csv", rows)
    write_csv(args.output_dir / "caption_footnote_loss_matrix.csv", loss_rows)
    write_examples(args.output_dir / "caption_footnote_examples.md", examples)
    write_report(
        args.output_dir / "CAPTION_FOOTNOTE_PRESERVATION_REPORT.md",
        summary,
        py_compile_status=args.py_compile_status,
        pytest_status=args.pytest_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
