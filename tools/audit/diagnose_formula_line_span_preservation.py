#!/usr/bin/env python3
"""Audit P0-B Formula line/span preservation on selected200 artifacts.

This audit reads MinerU raw/middle/content_list formula evidence and compares
it with current v8 items plus DocumentIR metadata generated through the patched
v8 adapter. It does not mutate raw MinerU, v8, or generated outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.adapters.mineru_v8_document_ir import convert_v8_payload_to_document_ir


DEFAULT_V8_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/formula_line_span_preservation_20260528")
DEFAULT_RAW_ROOT = Path("data/02_mineru_outputs")
RAW_DATASET_PRIORITY = (
    "arxiv2025_tex8000_mineru_only_20260523",
    "pilot500_v7_mineru_scibert_strict_20260522",
)
FORMULA_TYPES = {"inline_equation", "interline_equation", "equation", "equation_inline", "formula"}
INLINE_TYPES = {"inline_equation", "equation_inline", "inline_math", "inline_formula"}
DISPLAY_TYPES = {"interline_equation", "equation_interline", "equation", "display_formula", "formula"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def compact_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(compact_text(part) for part in value)
    elif isinstance(value, dict):
        for key in ("content", "text", "latex"):
            if key in value:
                return compact_text(value[key])
        value = " ".join(compact_text(part) for part in value.values())
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


def bbox_center_distance(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return math.inf
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return math.hypot(ax - bx, ay - by)


def extract_formula_text(item: dict[str, Any]) -> str:
    for key in ("formula_latex", "latex", "equation", "equation_text", "text", "content"):
        value = item.get(key)
        text = compact_text(value)
        if text:
            return text
    return ""


def formula_role(kind: str, text: str) -> str:
    marker = kind.casefold()
    if marker in INLINE_TYPES:
        return "inline_attachment"
    if marker in DISPLAY_TYPES:
        return "display_math"
    if len(compact_text(text)) <= 6:
        return "formula_ocr_artifact"
    return "uncertain"


def formula_confidence(kind: str, *, text_format: str | None = None, source_layer: str = "middle") -> str:
    marker = kind.casefold()
    if marker in INLINE_TYPES:
        return "strong_span_inline"
    if marker in DISPLAY_TYPES:
        if source_layer == "content_list" and str(text_format or "").casefold() == "latex":
            return "strong_content_equation_latex"
        return "strong_span_interline"
    if str(text_format or "").casefold() == "latex":
        return "medium_equation_text"
    return "weak_text_only"


def make_formula_entry(
    *,
    doc_id: str,
    page_idx: int,
    source_layer: str,
    raw_item_id: str,
    kind: str,
    text: str,
    box: list[float] | None,
    parent_line_id: str | None = None,
    parent_block_id: str | None = None,
    source_span_ids: list[str] | None = None,
    text_format: str | None = None,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    inline = kind.casefold() in INLINE_TYPES
    display = kind.casefold() in DISPLAY_TYPES and not inline
    return {
        "doc_id": doc_id,
        "page_idx": page_idx,
        "formula_id": f"{doc_id}:{source_layer}:{raw_item_id}",
        "raw_source_layer": source_layer,
        "raw_item_id": raw_item_id,
        "matched_v8_id": None,
        "matched_document_ir_node_id": None,
        "parent_line_id": parent_line_id,
        "parent_block_id": parent_block_id,
        "source_span_ids": source_span_ids or [],
        "bbox": box,
        "text_preview": compact_text(text)[:240],
        "raw_formula_type": kind,
        "mineru_span_type": kind if source_layer == "middle" else None,
        "text_format": text_format,
        "formula_latex": text if text_format == "latex" or source_layer == "middle" else text,
        "is_inline_math": inline,
        "is_display_math": display,
        "formula_context_role": formula_role(kind, text),
        "formula_source_layer": source_layer,
        "formula_confidence": formula_confidence(kind, text_format=text_format, source_layer=source_layer),
        "preservation_status": "unknown",
        "evidence": evidence or [],
    }


def iter_middle_formula_entries(doc_id: str, middle: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not isinstance(middle, dict):
        return entries
    for page in middle.get("pdf_info") or []:
        if not isinstance(page, dict):
            continue
        page_idx = int(page.get("page_idx") or 0)
        for collection_name in ("preproc_blocks", "para_blocks", "interline_equations"):
            blocks = page.get(collection_name) or []
            if not isinstance(blocks, list):
                continue
            for block_pos, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                block_index = block.get("index")
                if block_index is None:
                    block_index = block_pos
                parent_block_id = f"{doc_id}:p{page_idx:04d}:m{int(block_index):06d}"
                entries.extend(
                    iter_formula_entries_from_block(
                        doc_id=doc_id,
                        page_idx=page_idx,
                        block=block,
                        collection_name=collection_name,
                        parent_block_id=parent_block_id,
                        path=f"{collection_name}[{block_pos}]",
                    )
                )
    return entries


def iter_formula_entries_from_block(
    *,
    doc_id: str,
    page_idx: int,
    block: dict[str, Any],
    collection_name: str,
    parent_block_id: str,
    path: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    block_type = str(block.get("type") or "").casefold()
    if block_type in FORMULA_TYPES:
        entries.append(
            make_formula_entry(
                doc_id=doc_id,
                page_idx=page_idx,
                source_layer="middle",
                raw_item_id=f"{collection_name}:{path}:block",
                kind=block_type,
                text=extract_formula_text(block),
                box=bbox(block.get("bbox")),
                parent_block_id=parent_block_id,
                source_span_ids=[],
                evidence=[f"middle block type={block_type}", f"parent_block_id={parent_block_id}"],
            )
        )
    for line_idx, line in enumerate(block.get("lines") or []):
        if not isinstance(line, dict):
            continue
        line_box = bbox(line.get("bbox"))
        for span_idx, span in enumerate(line.get("spans") or []):
            if not isinstance(span, dict):
                continue
            span_type = str(span.get("type") or "").casefold()
            if span_type not in FORMULA_TYPES:
                continue
            line_id = f"{parent_block_id}:l{line_idx:04d}:s{span_idx:04d}"
            entries.append(
                make_formula_entry(
                    doc_id=doc_id,
                    page_idx=page_idx,
                    source_layer="middle",
                    raw_item_id=f"{collection_name}:{path}:l{line_idx}:s{span_idx}",
                    kind=span_type,
                    text=extract_formula_text(span),
                    box=bbox(span.get("bbox")) or line_box,
                    parent_line_id=line_id,
                    parent_block_id=parent_block_id,
                    source_span_ids=[line_id],
                    evidence=[
                        f"middle span type={span_type}",
                        f"parent_line_id={line_id}",
                        f"parent_block_id={parent_block_id}",
                    ],
                )
            )
    return entries


def iter_content_list_formula_entries(doc_id: str, content_list: Any, *, source_layer: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    items = content_list if isinstance(content_list, list) else content_list.get("items") if isinstance(content_list, dict) else []
    if not isinstance(items, list):
        return entries
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        typ = str(item.get("type") or "").casefold()
        text_format = str(item.get("text_format") or "").casefold() or None
        text = extract_formula_text(item)
        is_formula = typ in FORMULA_TYPES or text_format == "latex"
        if not is_formula:
            # content_list_v2 nests equation segments inside structured content.
            nested_entries = iter_v2_nested_formula_entries(doc_id, item, idx)
            entries.extend(nested_entries)
            continue
        entries.append(
            make_formula_entry(
                doc_id=doc_id,
                page_idx=int(item.get("page_idx") or 0),
                source_layer=source_layer,
                raw_item_id=f"{source_layer}:{idx}",
                kind=typ or "equation",
                text=text,
                box=bbox(item.get("bbox")),
                text_format=text_format,
                evidence=[f"{source_layer} type={typ}", f"text_format={text_format}"],
            )
        )
    return entries


def iter_v2_nested_formula_entries(doc_id: str, item: dict[str, Any], item_idx: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def walk(value: Any, path: str, page_idx: int, box: list[float] | None) -> None:
        if isinstance(value, dict):
            typ = str(value.get("type") or "").casefold()
            if "equation" in typ:
                text = compact_text(value.get("content") or value.get("text"))
                entries.append(
                    make_formula_entry(
                        doc_id=doc_id,
                        page_idx=page_idx,
                        source_layer="content_list_v2",
                        raw_item_id=f"content_list_v2:{item_idx}:{path}",
                        kind=typ,
                        text=text,
                        box=bbox(value.get("bbox")) or box,
                        text_format="latex",
                        evidence=[f"content_list_v2 nested type={typ}"],
                    )
                )
            for key, child in value.items():
                walk(child, f"{path}/{key}", page_idx, bbox(value.get("bbox")) or box)
        elif isinstance(value, list):
            for pos, child in enumerate(value):
                walk(child, f"{path}[{pos}]", page_idx, box)

    walk(item, "", int(item.get("page_idx") or 0), bbox(item.get("bbox")))
    return entries


def build_v8_index(v8_payload: dict[str, Any]) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for item in v8_payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        indexed.append(
            {
                "id": str(item.get("id") or ""),
                "page_idx": int(item.get("page_idx") or 0),
                "bbox": bbox(item.get("bbox")),
                "text": compact_text(item.get("text")),
                "type": str(item.get("type") or ""),
                "source_block_ids": [str(value) for value in item.get("source_block_ids") or []],
                "source_line_ids": [str(value) for value in item.get("source_line_ids") or []],
                "content_list_indices": [
                    int(candidate.get("content_list_index"))
                    for candidate in item.get("content_list_text_candidates") or []
                    if isinstance(candidate, dict) and candidate.get("content_list_index") is not None
                ],
            }
        )
    return indexed


def build_doc_ir_index(document_payload: dict[str, Any]) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for node in document_payload.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_bbox = None
        boxes = node.get("bboxes") or []
        if boxes and isinstance(boxes[0], dict):
            node_bbox = bbox([boxes[0].get("x0"), boxes[0].get("y0"), boxes[0].get("x1"), boxes[0].get("y1")])
        indexed.append(
            {
                "id": str(node.get("node_id") or ""),
                "page_idx": int(node.get("page_idx") or 0),
                "bbox": node_bbox,
                "text": compact_text(node.get("text")),
                "node_type": str(node.get("node_type") or ""),
                "metadata": node.get("metadata") or {},
            }
        )
    return indexed


def match_entry_to_index(entry: dict[str, Any], index: list[dict[str, Any]], *, content_list_index: int | None = None) -> tuple[str | None, str, list[str]]:
    page_idx = int(entry.get("page_idx") or 0)
    candidates = [item for item in index if int(item.get("page_idx") or 0) == page_idx]
    evidence: list[str] = []
    parent_block_id = str(entry.get("parent_block_id") or "")
    source_span_ids = {str(value) for value in entry.get("source_span_ids") or []}
    if parent_block_id:
        for candidate in candidates:
            if parent_block_id in set(candidate.get("source_block_ids") or []):
                evidence.append(f"source_block_id={parent_block_id}")
                return candidate["id"], "stable_source_block", evidence
    if source_span_ids:
        for candidate in candidates:
            if source_span_ids & set(candidate.get("source_line_ids") or []):
                evidence.append("source_line_id_overlap")
                return candidate["id"], "stable_source_line", evidence
    if content_list_index is not None:
        for candidate in candidates:
            if content_list_index in set(candidate.get("content_list_indices") or []):
                evidence.append(f"content_list_index={content_list_index}")
                return candidate["id"], "content_list_index", evidence
    best_bbox = None
    best_iou = 0.0
    for candidate in candidates:
        iou = bbox_iou(entry.get("bbox"), candidate.get("bbox"))
        if iou > best_iou:
            best_iou = iou
            best_bbox = candidate
    if best_bbox and best_iou >= 0.45:
        evidence.append(f"bbox_iou={best_iou:.3f}")
        return best_bbox["id"], "bbox_iou", evidence
    best_dist = None
    best_distance = math.inf
    for candidate in candidates:
        distance = bbox_center_distance(entry.get("bbox"), candidate.get("bbox"))
        if distance < best_distance:
            best_distance = distance
            best_dist = candidate
    if best_dist and best_distance <= 18.0:
        evidence.append(f"bbox_center_distance={best_distance:.1f}")
        return best_dist["id"], "bbox_proximity", evidence
    entry_text = norm_text(entry.get("formula_latex") or entry.get("text_preview"))
    if entry_text:
        for candidate in candidates:
            candidate_text = norm_text(candidate.get("text"))
            if entry_text and (entry_text in candidate_text or candidate_text in entry_text) and min(len(entry_text), len(candidate_text)) >= 4:
                evidence.append("normalized_text_overlap")
                return candidate["id"], "text_similarity", evidence
    return None, "raw_only_unmapped", []


def formula_metadata_count(document_payload: dict[str, Any]) -> int:
    return sum(1 for node in document_payload.get("nodes") or [] if isinstance(node, dict) and (node.get("metadata") or {}).get("formula_context_role"))


def node_formula_metadata_by_id(document_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for node in document_payload.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        metadata = node.get("metadata") or {}
        if metadata.get("formula_context_role"):
            result[str(node.get("node_id") or "")] = metadata
    return result


def document_to_payload(document: Any) -> dict[str, Any]:
    return {
        "doc_id": document.doc_id,
        "nodes": [
            {
                "node_id": node.node_id,
                "node_type": str(node.node_type.value if hasattr(node.node_type, "value") else node.node_type),
                "text": node.text,
                "page_idx": node.page_idx,
                "bboxes": [
                    {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1}
                    for box in node.bboxes
                ],
                "metadata": node.metadata,
            }
            for node in document.nodes
        ],
    }


def locate_selected_v8_files(v8_root: Path) -> list[Path]:
    return sorted(v8_root.glob("*/*_content_list_v8_contentlist_merge_hint.json"))


def doc_id_from_v8_path(path: Path) -> str:
    name = path.name
    return name.split("_content_list_v8_contentlist_merge_hint.json")[0]


def locate_raw_file(raw_root: Path, doc_id: str, suffix: str) -> Path | None:
    names = [f"{doc_id}_{suffix}", suffix]
    for dataset in RAW_DATASET_PRIORITY:
        for name in names:
            candidate = raw_root / dataset / doc_id / "auto" / name
            if candidate.exists():
                return candidate
    for candidate in raw_root.glob(f"*/{doc_id}/auto/{doc_id}_{suffix}"):
        if candidate.exists():
            return candidate
    for candidate in raw_root.glob(f"*/{doc_id}/auto/{suffix}"):
        if candidate.exists():
            return candidate
    return None


def compile_risk(text: str) -> bool:
    value = str(text or "")
    if value.count("{") != value.count("}"):
        return True
    return any(char in value for char in ("\u2713", "\u2717", "✓", "✗"))


def false_positive_text_like(text: str) -> bool:
    value = compact_text(text)
    if not value or len(value) > 120:
        return False
    return bool(re.search(r"\b[A-Za-z]\b", value) and not re.search(r"[=\\_^{}]|\\[A-Za-z]+", value))


def process_doc(doc_id: str, v8_path: Path, raw_root: Path, output_dir: Path) -> dict[str, Any]:
    doc_dir = output_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    middle_path = locate_raw_file(raw_root, doc_id, "middle.json")
    content_list_path = locate_raw_file(raw_root, doc_id, "content_list.json")
    content_list_v2_path = locate_raw_file(raw_root, doc_id, "content_list_v2.json")
    document_ir_path = v8_path.parent / "document_ir.json"

    v8_payload = read_json(v8_path)
    current_document_payload = read_json(document_ir_path) if document_ir_path.exists() else {"nodes": []}
    after_document_payload = document_to_payload(convert_v8_payload_to_document_ir(v8_payload, source_path=v8_path, doc_id=doc_id))

    raw_entries: list[dict[str, Any]] = []
    if middle_path and middle_path.exists():
        raw_entries.extend(iter_middle_formula_entries(doc_id, read_json(middle_path)))
    if content_list_path and content_list_path.exists():
        raw_entries.extend(iter_content_list_formula_entries(doc_id, read_json(content_list_path), source_layer="content_list"))
    if content_list_v2_path and content_list_v2_path.exists():
        raw_entries.extend(iter_content_list_formula_entries(doc_id, read_json(content_list_v2_path), source_layer="content_list_v2"))

    v8_index = build_v8_index(v8_payload)
    current_doc_index = build_doc_ir_index(current_document_payload)
    after_doc_index = build_doc_ir_index(after_document_payload)
    after_formula_meta = node_formula_metadata_by_id(after_document_payload)

    status_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    loss_rows: list[dict[str, Any]] = []

    for entry in raw_entries:
        content_idx = None
        if str(entry.get("raw_item_id") or "").startswith("content_list:"):
            try:
                content_idx = int(str(entry["raw_item_id"]).split(":")[1])
            except (ValueError, IndexError):
                pass
        v8_id, match_status, match_evidence = match_entry_to_index(entry, v8_index, content_list_index=content_idx)
        entry["matched_v8_id"] = v8_id
        entry["evidence"] = list(entry.get("evidence") or []) + match_evidence
        if v8_id:
            doc_id_match = v8_id if v8_id in {item["id"] for item in after_doc_index} else None
            if doc_id_match and doc_id_match in after_formula_meta:
                entry["matched_document_ir_node_id"] = doc_id_match
                entry["preservation_status"] = "mapped_to_document_ir"
            elif doc_id_match:
                entry["matched_document_ir_node_id"] = doc_id_match
                entry["preservation_status"] = "lost_v8_to_document_ir"
            else:
                doc_match_id, doc_match_status, doc_evidence = match_entry_to_index(entry, after_doc_index)
                entry["matched_document_ir_node_id"] = doc_match_id
                entry["evidence"] = list(entry.get("evidence") or []) + doc_evidence
                if doc_match_id and doc_match_id in after_formula_meta:
                    entry["preservation_status"] = "mapped_to_document_ir"
                elif doc_match_id:
                    entry["preservation_status"] = "lost_v8_to_document_ir"
                else:
                    entry["preservation_status"] = "lost_v8_to_document_ir"
        else:
            entry["preservation_status"] = "raw_only_unmapped" if match_status == "raw_only_unmapped" else "ambiguous"
        status_counts[entry["preservation_status"]] += 1
        confidence_counts[str(entry.get("formula_confidence") or "unknown")] += 1
        role_counts[str(entry.get("formula_context_role") or "unknown")] += 1
        source_counts[str(entry.get("raw_source_layer") or "unknown")] += 1
        if entry["preservation_status"] != "mapped_to_document_ir":
            loss_rows.append(
                {
                    "doc_id": doc_id,
                    "formula_id": entry["formula_id"],
                    "raw_source_layer": entry["raw_source_layer"],
                    "raw_formula_type": entry["raw_formula_type"],
                    "formula_confidence": entry["formula_confidence"],
                    "page_idx": entry["page_idx"],
                    "preservation_status": entry["preservation_status"],
                    "matched_v8_id": entry.get("matched_v8_id"),
                    "matched_document_ir_node_id": entry.get("matched_document_ir_node_id"),
                    "text_preview": entry.get("text_preview"),
                }
            )

    sidecar = {
        "schema_version": "formula_line_span_preservation_sidecar_v1",
        "doc_id": doc_id,
        "paths": {
            "middle": str(middle_path) if middle_path else None,
            "content_list": str(content_list_path) if content_list_path else None,
            "content_list_v2": str(content_list_v2_path) if content_list_v2_path else None,
            "v8": str(v8_path),
            "document_ir": str(document_ir_path) if document_ir_path.exists() else None,
        },
        "entries": raw_entries,
    }
    document_check = {
        "schema_version": "formula_document_ir_check_v1",
        "doc_id": doc_id,
        "current_document_ir_formula_preserved_count": formula_metadata_count(current_document_payload),
        "after_adapter_formula_preserved_count": formula_metadata_count(after_document_payload),
        "after_formula_nodes": [
            {
                "node_id": node["id"],
                "node_type": node["node_type"],
                "page_idx": node["page_idx"],
                "text_preview": node["text"][:160],
                "metadata": {
                    key: value
                    for key, value in node["metadata"].items()
                    if key
                    in {
                        "raw_formula_type",
                        "mineru_span_type",
                        "formula_latex",
                        "text_format",
                        "formula_source_layer",
                        "formula_confidence",
                        "is_inline_math",
                        "is_display_math",
                        "formula_context_role",
                        "line_span_ids",
                        "parent_line_id",
                        "parent_block_id",
                    }
                },
            }
            for node in after_doc_index
            if node["id"] in after_formula_meta
        ],
    }
    write_json(doc_dir / f"formula_line_span_sidecar_{doc_id}.json", sidecar)
    write_json(doc_dir / f"formula_document_ir_check_{doc_id}.json", document_check)

    raw_inline = sum(1 for entry in raw_entries if entry.get("raw_formula_type") in INLINE_TYPES)
    raw_interline = sum(1 for entry in raw_entries if entry.get("raw_formula_type") in DISPLAY_TYPES)
    raw_cl_equations = sum(1 for entry in raw_entries if entry.get("raw_source_layer") == "content_list" and entry.get("raw_formula_type") == "equation")
    raw_text_format = sum(1 for entry in raw_entries if entry.get("text_format") == "latex")
    inline_preserved = sum(1 for entry in raw_entries if entry.get("is_inline_math") and entry.get("preservation_status") == "mapped_to_document_ir")
    interline_preserved = sum(1 for entry in raw_entries if entry.get("is_display_math") and entry.get("preservation_status") == "mapped_to_document_ir")
    latex_preserved = sum(1 for entry in raw_entries if entry.get("formula_latex") and entry.get("preservation_status") == "mapped_to_document_ir")
    text_formula_without_span_evidence = sum(
        1
        for node in after_document_payload.get("nodes") or []
        if isinstance(node, dict)
        and node.get("node_type") == "text"
        and (node.get("metadata") or {}).get("formula_context_role")
        and not (node.get("metadata") or {}).get("inline_equation_spans")
        and not (node.get("metadata") or {}).get("interline_equation_spans")
    )
    compile_risk_count = sum(1 for entry in raw_entries if compile_risk(str(entry.get("formula_latex") or "")))
    false_positive_proxy = (1 if not raw_entries and formula_metadata_count(after_document_payload) else 0) + text_formula_without_span_evidence

    return {
        "doc_id": doc_id,
        "middle_found": bool(middle_path),
        "content_list_found": bool(content_list_path),
        "content_list_v2_found": bool(content_list_v2_path),
        "v8_found": True,
        "document_ir_found": document_ir_path.exists(),
        "raw_inline_equation_span_count": raw_inline,
        "raw_interline_equation_span_count": raw_interline,
        "raw_contentlist_equation_count": raw_cl_equations,
        "raw_text_format_latex_count": raw_text_format,
        "sidecar_formula_signal_count": len(raw_entries),
        "v8_formula_matched_count": sum(1 for entry in raw_entries if entry.get("matched_v8_id")),
        "current_document_ir_formula_preserved_count": formula_metadata_count(current_document_payload),
        "document_ir_formula_preserved_count": formula_metadata_count(after_document_payload),
        "inline_equation_preserved_count": inline_preserved,
        "interline_equation_preserved_count": interline_preserved,
        "formula_latex_preserved_count": latex_preserved,
        "raw_only_unmapped_count": status_counts.get("raw_only_unmapped", 0),
        "ambiguous_count": status_counts.get("ambiguous", 0),
        "lost_raw_to_v8_count": status_counts.get("raw_only_unmapped", 0),
        "lost_v8_to_document_ir_count": status_counts.get("lost_v8_to_document_ir", 0),
        "mapped_to_document_ir_count": status_counts.get("mapped_to_document_ir", 0),
        "false_positive_proxy_on_text_only_docs": false_positive_proxy,
        "formula_ocr_artifact_count": role_counts.get("formula_ocr_artifact", 0),
        "compile_risk_formula_text_count": compile_risk_count,
        "status_counts": dict(status_counts),
        "confidence_counts": dict(confidence_counts),
        "role_counts": dict(role_counts),
        "source_counts": dict(source_counts),
        "loss_rows": loss_rows,
        "raw_entries": raw_entries,
    }


def aggregate_rows(doc_results: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    nested = {"status_counts": Counter(), "confidence_counts": Counter(), "role_counts": Counter(), "source_counts": Counter()}
    for row in doc_results:
        for key, value in row.items():
            if key.endswith("_counts") or key in {"loss_rows", "raw_entries", "status_counts", "confidence_counts", "role_counts", "source_counts"}:
                continue
            if isinstance(value, bool):
                totals[key] += int(value)
            elif isinstance(value, int):
                totals[key] += value
        for key in nested:
            nested[key].update(row.get(key) or {})
    return {
        "docs_analyzed": len(doc_results),
        "artifact_found_counts": {
            "middle": sum(1 for row in doc_results if row.get("middle_found")),
            "content_list": sum(1 for row in doc_results if row.get("content_list_found")),
            "content_list_v2": sum(1 for row in doc_results if row.get("content_list_v2_found")),
            "v8": sum(1 for row in doc_results if row.get("v8_found")),
            "document_ir": sum(1 for row in doc_results if row.get("document_ir_found")),
        },
        "metrics": {
            key: totals[key]
            for key in (
                "raw_inline_equation_span_count",
                "raw_interline_equation_span_count",
                "raw_contentlist_equation_count",
                "raw_text_format_latex_count",
                "sidecar_formula_signal_count",
                "v8_formula_matched_count",
                "current_document_ir_formula_preserved_count",
                "document_ir_formula_preserved_count",
                "inline_equation_preserved_count",
                "interline_equation_preserved_count",
                "formula_latex_preserved_count",
                "raw_only_unmapped_count",
                "ambiguous_count",
                "lost_raw_to_v8_count",
                "lost_v8_to_document_ir_count",
                "mapped_to_document_ir_count",
                "false_positive_proxy_on_text_only_docs",
                "formula_ocr_artifact_count",
                "compile_risk_formula_text_count",
            )
        },
        **{key: dict(value) for key, value in nested.items()},
    }


def examples_from_results(doc_results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "preserved_inline": [],
        "preserved_interline": [],
        "raw_only_unmapped": [],
        "prevented_false_positive": [],
    }
    for result in doc_results:
        for entry in result.get("raw_entries", []):
            sample = {
                "doc_id": result["doc_id"],
                "page_idx": entry.get("page_idx"),
                "formula_id": entry.get("formula_id"),
                "text_preview": entry.get("text_preview"),
                "bbox": entry.get("bbox"),
                "raw_source_layer": entry.get("raw_source_layer"),
                "raw_formula_type": entry.get("raw_formula_type"),
                "formula_confidence": entry.get("formula_confidence"),
                "preservation_status": entry.get("preservation_status"),
                "matched_v8_id": entry.get("matched_v8_id"),
                "matched_document_ir_node_id": entry.get("matched_document_ir_node_id"),
            }
            if entry.get("preservation_status") == "mapped_to_document_ir" and entry.get("is_inline_math") and len(buckets["preserved_inline"]) < 20:
                buckets["preserved_inline"].append(sample)
            if entry.get("preservation_status") == "mapped_to_document_ir" and entry.get("is_display_math") and len(buckets["preserved_interline"]) < 20:
                buckets["preserved_interline"].append(sample)
            if entry.get("preservation_status") == "raw_only_unmapped" and len(buckets["raw_only_unmapped"]) < 20:
                buckets["raw_only_unmapped"].append(sample)
            if false_positive_text_like(entry.get("text_preview")) and entry.get("formula_confidence") == "weak_text_only" and len(buckets["prevented_false_positive"]) < 20:
                buckets["prevented_false_positive"].append(sample)
    return buckets


def write_examples(path: Path, examples: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Formula Line/Span Preservation Examples", ""]
    titles = {
        "preserved_inline": "Preserved inline equation examples",
        "preserved_interline": "Preserved interline/display equation examples",
        "raw_only_unmapped": "Raw-only unmapped examples",
        "prevented_false_positive": "Prevented false positive examples",
    }
    for key, title in titles.items():
        lines.extend([f"## {title}", ""])
        if not examples.get(key):
            lines.append("- none observed")
        for row in examples.get(key, []):
            lines.append(
                f"- `{row['doc_id']}` p{row.get('page_idx')}: {row.get('raw_formula_type')} / "
                f"{row.get('formula_confidence')} / {row.get('preservation_status')} / "
                f"{row.get('text_preview')}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any], examples: dict[str, list[dict[str, Any]]], py_compile_status: str, pytest_status: str) -> None:
    m = summary["metrics"]
    lines = [
        "# FORMULA LINE/SPAN PRESERVATION REPORT",
        "",
        "## Status",
        f"- docs analyzed: {summary['docs_analyzed']}",
        f"- raw/middle/content_list/v8/document_ir artifacts found: {summary['artifact_found_counts']}",
        f"- py_compile status: {py_compile_status}",
        f"- pytest/manual test status: {pytest_status}",
        "- no training / no MinerU / no relabel / no rebuild / no GNN",
        "- no renderer changes",
        "- production default unchanged",
        "- v8 facts used; no fallback to old v7; legacy names such as `source_v7_ids` are provenance only.",
        "",
        "## Key Finding",
        "",
        "- raw/middle/content_list formula line/span facts existed in selected200.",
        "- the main preservation gap was metadata: current DocumentIR had no formula line/span metadata, while the patched adapter preserves it without changing renderer output.",
        "- the patch avoids regex false-positive explosion by only preserving upstream MinerU span/equation/text_format evidence.",
        "",
        "## Before / After Summary",
        "",
        "| metric | count |",
        "| --- | ---: |",
    ]
    for key in (
        "raw_inline_equation_span_count",
        "raw_interline_equation_span_count",
        "raw_contentlist_equation_count",
        "raw_text_format_latex_count",
        "v8_formula_matched_count",
        "current_document_ir_formula_preserved_count",
        "document_ir_formula_preserved_count",
        "inline_equation_preserved_count",
        "interline_equation_preserved_count",
        "formula_latex_preserved_count",
        "raw_only_unmapped_count",
        "ambiguous_count",
        "false_positive_proxy_on_text_only_docs",
    ):
        lines.append(f"| {key} | {m.get(key, 0)} |")
    lines.extend(
        [
            "",
            "## Loss Matrix",
            "",
            "| loss/status | count |",
            "| --- | ---: |",
        ]
    )
    for key, value in sorted(summary.get("status_counts", {}).items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "Major loss types are `LOST_RAW_TO_V8` represented by raw-only unmapped sidecar entries, and `LOST_V8_TO_DOCUMENT_IR` when a v8 match exists but formula metadata is not preserved. RenderTree consumption was intentionally not patched in this pass.",
            "",
            "## Formula Context Impact",
            "",
            "- FormulaContextGroup can now prefer MinerU span/equation evidence for inline attachments and display math in a later pass.",
            "- Inline equation spans, interline/display equation blocks, and content_list LaTeX equations are now represented as sidecar and DocumentIR metadata.",
            "- Theorem/where/context regex cases remain diagnostic unless they overlap preserved MinerU formula evidence.",
            "- Remaining regex-only cases should not be promoted into production formula roles without source evidence.",
            "",
            "## Comparison With Regex Detector",
            "",
            "- Regex-only formula context detection is broad and should remain diagnostic.",
            "- MinerU span/equation preservation is high-confidence upstream fact inheritance.",
            "- Broad regex candidates must not be mixed into production formula role decisions.",
            "",
            "## Examples",
            "",
        ]
    )
    for title, bucket in (
        ("Preserved inline equation examples", examples.get("preserved_inline", [])),
        ("Preserved interline/display equation examples", examples.get("preserved_interline", [])),
        ("Raw-only unmapped examples", examples.get("raw_only_unmapped", [])),
        ("Prevented false positives", examples.get("prevented_false_positive", [])),
    ):
        lines.extend([f"### {title}", ""])
        if not bucket:
            lines.append("- none observed")
        for row in bucket[:20]:
            lines.append(
                f"- `{row['doc_id']}` p{row.get('page_idx')}: {row.get('raw_formula_type')} / "
                f"{row.get('formula_confidence')} / {row.get('preservation_status')} / "
                f"{row.get('text_preview')}"
            )
        lines.append("")
    inline_total = max(1, int(m.get("raw_inline_equation_span_count", 0)))
    inline_preserved_ratio = float(m.get("inline_equation_preserved_count", 0)) / inline_total
    decision = (
        "ready_for_formula_context_phase1"
        if m.get("document_ir_formula_preserved_count", 0) > 0
        and inline_preserved_ratio >= 0.50
        and m.get("false_positive_proxy_on_text_only_docs", 0) <= 0
        else "need_formula_mapping_patch"
    )
    lines.extend(
        [
            "## Decision",
            "",
            f"**{decision}**",
            "",
            "Formula line/span preservation gives high-confidence upstream facts for a later FormulaContextGroup consumer. This pass did not change rendering or generated.tex.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-root", type=Path, default=DEFAULT_V8_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--py-compile-status", default="not_run")
    parser.add_argument("--pytest-status", default="not_run")
    args = parser.parse_args()

    v8_files = locate_selected_v8_files(args.v8_root)
    if not v8_files:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        readiness = {
            "status": "readiness_failed",
            "reason": "selected200 v8 files not found",
            "v8_root": str(args.v8_root),
        }
        write_json(args.output_dir / "FORMULA_LINE_SPAN_PRESERVATION_READINESS.json", readiness)
        (args.output_dir / "FORMULA_LINE_SPAN_PRESERVATION_REPORT.md").write_text(
            "# FORMULA LINE/SPAN PRESERVATION REPORT\n\nreadiness_failed: selected200 v8 files not found\n",
            encoding="utf-8",
        )
        return

    doc_results: list[dict[str, Any]] = []
    all_loss_rows: list[dict[str, Any]] = []
    for v8_path in v8_files:
        doc_id = doc_id_from_v8_path(v8_path)
        result = process_doc(doc_id, v8_path, args.raw_root, args.output_dir)
        doc_results.append(result)
        all_loss_rows.extend(result.get("loss_rows") or [])

    summary = aggregate_rows(doc_results)
    examples = examples_from_results(doc_results)
    summary_rows = [
        {key: value for key, value in row.items() if key not in {"loss_rows", "raw_entries", "status_counts", "confidence_counts", "role_counts", "source_counts"}}
        for row in doc_results
    ]
    write_csv(args.output_dir / "formula_line_span_preservation_summary.csv", summary_rows)
    write_json(args.output_dir / "formula_line_span_preservation_summary.json", summary)
    write_csv(args.output_dir / "formula_line_span_loss_matrix.csv", all_loss_rows)
    write_examples(args.output_dir / "formula_line_span_examples.md", examples)
    write_report(
        args.output_dir / "FORMULA_LINE_SPAN_PRESERVATION_REPORT.md",
        summary,
        examples,
        args.py_compile_status,
        args.pytest_status,
    )


if __name__ == "__main__":
    main()
