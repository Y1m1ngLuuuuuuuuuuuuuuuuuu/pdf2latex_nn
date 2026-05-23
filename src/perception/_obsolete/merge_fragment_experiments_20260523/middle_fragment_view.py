"""Build a middle.json-derived fragment view for MERGE supervision.

MinerU's ``content_list`` is already a logical block layer.  For paragraph
continuation learning we sometimes need the earlier line/span evidence stored
in ``middle.json``: each fragment has its own text and bbox, while still
projecting back to the content-list/v7 logical owner.

This module does not mutate v7 and does not participate in the main graph
pipeline unless a caller explicitly feeds the generated pseudo-v7 fragment
payload to graph construction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.adapters.mineru_v7_document_ir import stable_node_id


BODY_MERGE_ROLES = {"body_text", "list_item", "reference_list", "reference_item"}
BODY_MERGE_TYPES = {"paragraph", "text", "list", "reference", "references", "bibliography"}
NON_BODY_LAYERS = {"metadata_layer", "float_layer", "math_layer", "annotation_layer", "noise_layer"}
NON_BODY_ROLES = {
    "document_title",
    "front_matter",
    "affiliation",
    "author",
    "authors",
    "email",
    "figure_caption",
    "table_caption",
    "algorithm_caption",
    "footnote",
    "noise",
}
MATH_TYPES = {"equation", "equation_interline", "display_formula", "formula", "inline_math"}
FLOAT_TYPES = {"figure", "image", "chart", "table", "algorithm"}


@dataclass(frozen=True)
class MiddleFragmentBuildResult:
    """Serializable outputs for one document."""

    doc_id: str
    fragment_view: dict[str, Any]
    fragment_v7_payload: dict[str, Any]
    merge_labels: dict[str, Any]
    summary: dict[str, Any]


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_middle_fragment_view(
    *,
    doc_id: str,
    middle_json_path: Path,
    v7_json_path: Path | None = None,
    middle_block_source: str = "para_blocks",
) -> MiddleFragmentBuildResult:
    """Build a fragment view and a pseudo-v7 payload from one middle.json."""

    middle_payload = load_json(middle_json_path)
    v7_payload = load_json(v7_json_path) if v7_json_path is not None and v7_json_path.exists() else {}
    v7_items = normalize_v7_items(v7_payload)
    logical_blocks = extract_logical_blocks(middle_payload, doc_id=doc_id, source=middle_block_source)
    owner_records = [attach_v7_owner(block, v7_items) for block in logical_blocks]
    fragments = build_fragments(owner_records)
    merge_edges = build_fragment_merge_edges(fragments)
    fragment_v7_payload = build_fragment_v7_payload(
        doc_id=doc_id,
        middle_json_path=middle_json_path,
        v7_json_path=v7_json_path,
        source_v7_payload=v7_payload,
        fragments=fragments,
    )
    fragment_view = {
        "schema_version": "middle_fragment_view_v1",
        "doc_id": doc_id,
        "middle_json": str(middle_json_path),
        "v7_json": str(v7_json_path) if v7_json_path is not None else None,
        "middle_block_source": middle_block_source,
        "logical_blocks": owner_records,
        "fragments": fragments,
        "owner_projection": owner_projection(fragments),
    }
    merge_payload = {
        "schema_version": "middle_fragment_merge_labels_v1",
        "doc_id": doc_id,
        "label_space": {"MERGE": 0, "PARENT_CHILD": 1, "NONE": 2},
        "positive_merge_edges": merge_edges,
    }
    summary = summarize_fragment_view(
        doc_id=doc_id,
        logical_blocks=owner_records,
        fragments=fragments,
        merge_edges=merge_edges,
        middle_json_path=middle_json_path,
        v7_json_path=v7_json_path,
    )
    return MiddleFragmentBuildResult(
        doc_id=doc_id,
        fragment_view=fragment_view,
        fragment_v7_payload=fragment_v7_payload,
        merge_labels=merge_payload,
        summary=summary,
    )


def normalize_v7_items(payload: Any) -> list[dict[str, Any]]:
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        record = dict(item)
        record["_v7_index"] = index
        record["_v7_node_id"] = stable_node_id(record, fallback_position=index)
        normalized.append(record)
    return normalized


def extract_logical_blocks(payload: Any, *, doc_id: str, source: str = "para_blocks") -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("middle.json payload must be an object")
    pdf_info = payload.get("pdf_info")
    if not isinstance(pdf_info, list):
        raise ValueError("middle.json missing pdf_info list")
    logical: list[dict[str, Any]] = []
    for page_idx, page in enumerate(pdf_info):
        if not isinstance(page, dict):
            continue
        blocks = page.get(source)
        if not isinstance(blocks, list):
            continue
        for local_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            block_page_idx = int_or_none(block.get("page_idx"))
            if block_page_idx is None:
                block_page_idx = page_idx
            middle_index = int_or_none(block.get("index"))
            if middle_index is None:
                middle_index = local_index
            fragments = extract_fragments(block, page_idx=block_page_idx)
            text = normalize_space(" ".join(fragment["text"] for fragment in fragments if fragment.get("text")))
            logical.append(
                {
                    "doc_id": doc_id,
                    "middle_block_id": f"p{block_page_idx:04d}_b{middle_index:06d}",
                    "source": source,
                    "source_page_idx": block_page_idx,
                    "middle_index": middle_index,
                    "type": str(block.get("type") or ""),
                    "score": block.get("score"),
                    "bbox": list(block.get("bbox") or []),
                    "line_count": len(block.get("lines") or []),
                    "span_count": len(fragments),
                    "text": text,
                    "has_cross_page": bool(block.get("cross_page"))
                    or any(bool(fragment.get("cross_page")) for fragment in fragments),
                    "has_cross_column": bool(block.get("cross_column"))
                    or any(bool(fragment.get("cross_column")) for fragment in fragments),
                    "fragments": fragments,
                }
            )
    return logical


def extract_fragments(block: dict[str, Any], *, page_idx: int) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    lines = block.get("lines")
    if isinstance(lines, list):
        for line_idx, line in enumerate(lines):
            if not isinstance(line, dict):
                continue
            spans = line.get("spans")
            if not isinstance(spans, list):
                spans = []
            if not spans:
                text = text_from_any(line)
                if text:
                    fragments.append(fragment_record(line, page_idx=page_idx, line_idx=line_idx, span_idx=None, text=text))
                continue
            for span_idx, span in enumerate(spans):
                if not isinstance(span, dict):
                    continue
                text = text_from_any(span)
                if not text:
                    continue
                record = fragment_record(span, page_idx=page_idx, line_idx=line_idx, span_idx=span_idx, text=text)
                if not record.get("line_bbox"):
                    record["line_bbox"] = list(line.get("bbox") or [])
                record["cross_page"] = bool(record.get("cross_page") or line.get("cross_page"))
                record["cross_column"] = bool(record.get("cross_column") or line.get("cross_column"))
                fragments.append(record)
    if fragments:
        return fragments
    text = text_from_any(block)
    if text:
        return [fragment_record(block, page_idx=page_idx, line_idx=None, span_idx=None, text=text)]
    return []


def fragment_record(
    source: dict[str, Any],
    *,
    page_idx: int,
    line_idx: int | None,
    span_idx: int | None,
    text: str,
) -> dict[str, Any]:
    return {
        "page_idx": page_idx,
        "line_idx": line_idx,
        "span_idx": span_idx,
        "type": str(source.get("type") or ""),
        "text": normalize_space(text),
        "bbox": list(source.get("bbox") or []),
        "line_bbox": list(source.get("line_bbox") or []),
        "cross_page": bool(source.get("cross_page")),
        "cross_column": bool(source.get("cross_column")),
        "score": source.get("score"),
    }


def text_from_any(value: Any) -> str:
    if isinstance(value, str):
        return normalize_space(value)
    if isinstance(value, dict):
        for key in ("content", "text", "latex", "html"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return normalize_space(inner)
        for key in ("spans", "lines", "children"):
            inner = value.get(key)
            if isinstance(inner, list):
                text = normalize_space(" ".join(text_from_any(item) for item in inner))
                if text:
                    return text
    if isinstance(value, list):
        return normalize_space(" ".join(text_from_any(item) for item in value))
    return ""


def attach_v7_owner(block: dict[str, Any], v7_items: list[dict[str, Any]]) -> dict[str, Any]:
    mapping = map_middle_block_to_v7(block, v7_items)
    owner_items = [v7_items[index] for index in mapping["mapped_v7_indices"] if 0 <= index < len(v7_items)]
    owner = owner_items[0] if owner_items else {}
    return {
        **block,
        **mapping,
        "owner_layout_layer": str(owner.get("layout_layer") or ""),
        "owner_layout_role": str(owner.get("layout_role") or ""),
        "owner_type": str(owner.get("type") or owner.get("canonical_type") or ""),
        "owner_raw_type": str(owner.get("raw_type") or owner.get("type") or ""),
    }


def map_middle_block_to_v7(block: dict[str, Any], v7_items: list[dict[str, Any]]) -> dict[str, Any]:
    if not v7_items:
        return {"mapped_v7_ids": [], "mapped_v7_indices": [], "mapping_method": "missing_v7", "mapping_score": 0.0}
    page_idx = int(block.get("source_page_idx") or 0)
    middle_index = int(block.get("middle_index") or 0)
    exact = [
        item
        for item in v7_items
        if int_or_none(item.get("mineru_page_idx"), item.get("page_idx")) == page_idx
        and int_or_none(item.get("mineru_block_idx"), item.get("original_index")) == middle_index
    ]
    if exact:
        return v7_mapping_payload(exact, method="page_block_index", score=1.0)
    text = compact_text(block.get("text") or "")
    if len(text) < 8:
        return {"mapped_v7_ids": [], "mapped_v7_indices": [], "mapping_method": "empty_or_short_text", "mapping_score": 0.0}
    candidates = [
        item for item in v7_items if int_or_none(item.get("page_idx"), item.get("mineru_page_idx")) in {page_idx, page_idx + 1}
    ]
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        item_text = compact_text(v7_text(item))
        if len(item_text) < 8:
            continue
        score = containment_similarity(text, item_text)
        scored.append((score, item))
    if not scored:
        return {"mapped_v7_ids": [], "mapped_v7_indices": [], "mapping_method": "no_text_candidate", "mapping_score": 0.0}
    scored.sort(key=lambda entry: entry[0], reverse=True)
    best_score, best = scored[0]
    if best_score < 0.65:
        return {
            "mapped_v7_ids": [],
            "mapped_v7_indices": [],
            "mapping_method": "low_text_similarity",
            "mapping_score": round(float(best_score), 4),
        }
    return v7_mapping_payload([best], method="text_similarity", score=best_score)


def build_fragments(owner_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    global_order = 0
    for block in owner_blocks:
        for local_order, source in enumerate(block.get("fragments") or []):
            if not isinstance(source, dict) or not str(source.get("text") or "").strip():
                continue
            owner_v7_ids = list(block.get("mapped_v7_ids") or [])
            owner_v7_indices = list(block.get("mapped_v7_indices") or [])
            fragment_id = (
                f"{block['middle_block_id']}_"
                f"l{none_safe_int(source.get('line_idx')):04d}_"
                f"s{none_safe_int(source.get('span_idx')):04d}"
            )
            fragments.append(
                {
                    "fragment_id": fragment_id,
                    "fragment_index": global_order,
                    "global_order": global_order,
                    "owner_middle_block_id": block["middle_block_id"],
                    "owner_v7_ids": owner_v7_ids,
                    "owner_v7_indices": owner_v7_indices,
                    "owner_mapping_method": block.get("mapping_method"),
                    "owner_mapping_score": block.get("mapping_score"),
                    "order_in_owner": local_order,
                    "text": str(source.get("text") or ""),
                    "bbox": list(source.get("bbox") or source.get("line_bbox") or block.get("bbox") or []),
                    "page_idx": int_or_none(source.get("page_idx"), block.get("source_page_idx")) or 0,
                    "line_idx": source.get("line_idx"),
                    "span_idx": source.get("span_idx"),
                    "fragment_cross_page": bool(source.get("cross_page")),
                    "fragment_cross_column": bool(source.get("cross_column")),
                    "owner_has_cross_page": bool(block.get("has_cross_page")),
                    "owner_has_cross_column": bool(block.get("has_cross_column")),
                    "cross_page": bool(source.get("cross_page") or block.get("has_cross_page")),
                    "cross_column": bool(source.get("cross_column") or block.get("has_cross_column")),
                    "middle_type": block.get("type"),
                    "owner_layout_layer": block.get("owner_layout_layer"),
                    "owner_layout_role": block.get("owner_layout_role"),
                    "owner_type": block.get("owner_type"),
                    "owner_raw_type": block.get("owner_raw_type"),
                    "merge_channel": fragment_merge_channel(block),
                }
            )
            global_order += 1
    return fragments


def build_fragment_merge_edges(fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_owner: dict[str, list[dict[str, Any]]] = {}
    for fragment in fragments:
        by_owner.setdefault(str(fragment["owner_middle_block_id"]), []).append(fragment)
    edges: list[dict[str, Any]] = []
    for owner_id, group in by_owner.items():
        group = sorted(group, key=lambda item: int(item["order_in_owner"]))
        for left, right in zip(group, group[1:]):
            channel = str(left.get("merge_channel") or "MASKED_UNKNOWN")
            if channel != str(right.get("merge_channel") or "MASKED_UNKNOWN"):
                channel = "MASKED_UNKNOWN"
            strength = "strong" if channel in {"BODY_TEXT", "LIST_ITEM", "REFERENCE_ITEM"} else "masked"
            if strength != "strong":
                continue
            edges.append(
                {
                    "src_fragment_id": left["fragment_id"],
                    "dst_fragment_id": right["fragment_id"],
                    "src_fragment_index": int(left["fragment_index"]),
                    "dst_fragment_index": int(right["fragment_index"]),
                    "owner_middle_block_id": owner_id,
                    "owner_v7_ids": left.get("owner_v7_ids") or right.get("owner_v7_ids") or [],
                    "label": "MERGE",
                    "label_id": 0,
                    "label_strength": strength,
                    "merge_relation_family": f"{channel}_CONTINUATION",
                    "reason": "adjacent_fragments_same_middle_owner",
                }
            )
    return edges


def fragment_merge_channel(block: dict[str, Any]) -> str:
    layer = str(block.get("owner_layout_layer") or "").casefold()
    role = str(block.get("owner_layout_role") or "").casefold()
    owner_type = str(block.get("owner_type") or block.get("owner_raw_type") or block.get("type") or "").casefold()
    if layer in NON_BODY_LAYERS or role in NON_BODY_ROLES:
        return "MASKED_UNKNOWN"
    if owner_type in MATH_TYPES:
        return "FORMULA"
    if owner_type in FLOAT_TYPES:
        return "FLOAT"
    if "caption" in role:
        return "CAPTION"
    if role in {"reference_list", "reference_item"} or owner_type in {"reference", "references", "bibliography"}:
        return "REFERENCE_ITEM"
    if role == "list_item" or owner_type == "list":
        return "LIST_ITEM"
    if role in BODY_MERGE_ROLES or owner_type in BODY_MERGE_TYPES:
        return "BODY_TEXT"
    return "MASKED_UNKNOWN"


def build_fragment_v7_payload(
    *,
    doc_id: str,
    middle_json_path: Path,
    v7_json_path: Path | None,
    source_v7_payload: Any,
    fragments: list[dict[str, Any]],
) -> dict[str, Any]:
    source_config = source_v7_payload.get("config") if isinstance(source_v7_payload, dict) else {}
    style_source_pdf = source_v7_payload.get("style_source_pdf") if isinstance(source_v7_payload, dict) else None
    return {
        "schema_version": "content_v7_middle_fragment_with_styles",
        "source_format": "mineru_middle_fragment_view",
        "doc_id": doc_id,
        "source_path": str(middle_json_path),
        "source_v7_path": str(v7_json_path) if v7_json_path is not None else None,
        "style_source_pdf": style_source_pdf,
        "config": {
            "fragment_view": True,
            "preserve_v7": True,
            "source_config": source_config,
            "merge_paragraphs": False,
            "fragment_owner_projection": True,
        },
        "style_config": {
            "source": "middle_fragment_fallback",
            "note": "Fragments inherit only text/bbox; font spans are unavailable unless a later enrichment pass is run.",
        },
        "items": [fragment_to_v7_item(fragment) for fragment in fragments],
    }


def fragment_to_v7_item(fragment: dict[str, Any]) -> dict[str, Any]:
    channel = str(fragment.get("merge_channel") or "")
    item_type = {
        "LIST_ITEM": "list",
        "REFERENCE_ITEM": "reference",
        "FORMULA": "equation_interline",
        "FLOAT": "image",
        "CAPTION": "paragraph",
    }.get(channel, "paragraph")
    text = str(fragment.get("text") or "")
    fragment_id = str(fragment["fragment_id"])
    return {
        "node_id": fragment_id,
        "id": fragment_id,
        "block_id": fragment_id,
        "global_order": int(fragment["global_order"]),
        "column_fix_global_order": int(fragment["global_order"]),
        "layout_flow_order": int(fragment["global_order"]),
        "page_idx": int(fragment.get("page_idx") or 0),
        "type": item_type,
        "raw_type": "middle_fragment",
        "canonical_type": item_type,
        "layout_layer": fragment.get("owner_layout_layer") or "main_text_flow",
        "layout_role": fragment.get("owner_layout_role") or ("list_item" if item_type == "list" else "body_text"),
        "bbox": list(fragment.get("bbox") or []),
        "text": text,
        "text_for_embedding": text,
        "content": text,
        "style_spans": [
            {
                "text": text,
                "font_name": "",
                "font_size": 0.0,
                "is_bold": False,
                "is_italic": False,
                "is_inline_math": channel == "FORMULA",
                "is_inline_code": False,
                "char_count": len(text),
                "bbox": list(fragment.get("bbox") or []),
            }
        ],
        "style_extract_status": "middle_fragment_fallback",
        "fragment_id": fragment_id,
        "middle_fragment_view": True,
        "owner_middle_block_id": fragment["owner_middle_block_id"],
        "owner_v7_ids": list(fragment.get("owner_v7_ids") or []),
        "owner_v7_indices": list(fragment.get("owner_v7_indices") or []),
        "middle_line_idx": fragment.get("line_idx"),
        "middle_span_idx": fragment.get("span_idx"),
        "middle_fragment_cross_page": bool(fragment.get("fragment_cross_page")),
        "middle_fragment_cross_column": bool(fragment.get("fragment_cross_column")),
        "middle_owner_has_cross_page": bool(fragment.get("owner_has_cross_page")),
        "middle_owner_has_cross_column": bool(fragment.get("owner_has_cross_column")),
        "middle_cross_page": bool(fragment.get("cross_page")),
        "middle_cross_column": bool(fragment.get("cross_column")),
        "merge_channel": channel,
    }


def owner_projection(fragments: list[dict[str, Any]]) -> dict[str, Any]:
    by_middle: dict[str, list[str]] = {}
    by_v7: dict[str, list[str]] = {}
    for fragment in fragments:
        fid = str(fragment["fragment_id"])
        by_middle.setdefault(str(fragment["owner_middle_block_id"]), []).append(fid)
        for v7_id in fragment.get("owner_v7_ids") or []:
            by_v7.setdefault(str(v7_id), []).append(fid)
    return {
        "middle_block_to_fragment_ids": by_middle,
        "v7_id_to_fragment_ids": by_v7,
    }


def summarize_fragment_view(
    *,
    doc_id: str,
    logical_blocks: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
    merge_edges: list[dict[str, Any]],
    middle_json_path: Path,
    v7_json_path: Path | None,
) -> dict[str, Any]:
    channel_counts: dict[str, int] = {}
    mapping_counts: dict[str, int] = {}
    for fragment in fragments:
        channel = str(fragment.get("merge_channel") or "MASKED_UNKNOWN")
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
    for block in logical_blocks:
        method = str(block.get("mapping_method") or "unknown")
        mapping_counts[method] = mapping_counts.get(method, 0) + 1
    return {
        "schema_version": "middle_fragment_view_summary_v1",
        "doc_id": doc_id,
        "middle_json": str(middle_json_path),
        "v7_json": str(v7_json_path) if v7_json_path is not None else None,
        "logical_block_count": len(logical_blocks),
        "fragment_count": len(fragments),
        "positive_merge_edge_count": len(merge_edges),
        "cross_page_fragment_count": sum(1 for fragment in fragments if fragment.get("cross_page")),
        "cross_column_fragment_count": sum(1 for fragment in fragments if fragment.get("cross_column")),
        "channel_counts": dict(sorted(channel_counts.items())),
        "mapping_method_counts": dict(sorted(mapping_counts.items())),
    }


def v7_mapping_payload(items: list[dict[str, Any]], *, method: str, score: float) -> dict[str, Any]:
    return {
        "mapped_v7_ids": [str(item["_v7_node_id"]) for item in items],
        "mapped_v7_indices": [int(item["_v7_index"]) for item in items],
        "mapping_method": method,
        "mapping_score": round(float(score), 4),
    }


def v7_text(item: dict[str, Any]) -> str:
    for key in ("text_for_embedding", "text", "content", "latex", "text_preview"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_space(value)
    return ""


def containment_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left in right:
        return min(1.0, len(left) / max(len(right), 1) + 0.25)
    if right in left:
        return min(1.0, len(right) / max(len(left), 1) + 0.25)
    limit = min(len(left), len(right), 500)
    return SequenceMatcher(None, left[:limit], right[:limit]).ratio()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def compact_text(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(text or "").casefold())


def int_or_none(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int):
            return value
        try:
            if value is not None and str(value).strip():
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def none_safe_int(value: Any) -> int:
    parsed = int_or_none(value)
    return parsed if parsed is not None else 0
