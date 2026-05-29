"""Adapter from v8 middle-derived content into the stable DocumentIR contract.

V8 deliberately emits v7-compatible ``items`` so that style extraction,
rendering, and evaluation can reuse the existing IR backend.  This wrapper is
thin by design: it normalizes v8 provenance/type metadata, preserves PDF-point
coordinates, then delegates item conversion to the v7 adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

from src.adapters.mineru_v7_document_ir import (
    MinerUV7DocumentIRAdapterConfig,
    convert_v7_payload_to_document_ir,
)
from src.ir import CoordinateSpace, DocumentIR
from src.ir.serialization import read_json, write_json


def load_v8_document_ir(
    content_json_path: Path,
    *,
    pdf_path: Path | None = None,
    doc_id: str | None = None,
) -> DocumentIR:
    payload = read_json(content_json_path)
    if not isinstance(payload, dict):
        raise ValueError(f"v8 content payload must be a JSON object: {content_json_path}")
    return convert_v8_payload_to_document_ir(
        payload,
        source_path=content_json_path,
        pdf_path=pdf_path,
        doc_id=doc_id,
    )


def convert_v8_payload_to_document_ir(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
    pdf_path: Path | None = None,
    doc_id: str | None = None,
) -> DocumentIR:
    page_width, page_height = infer_page_size(payload)
    stable_doc_id = doc_id or str(payload.get("doc_id") or "document")
    adapter_payload = {
        "schema_version": str(payload.get("schema_version") or "content_list_v8_reflow_v1"),
        "source_format": "mineru_middle_v8_reflow",
        "doc_id": stable_doc_id,
        "items": normalize_v8_items_for_adapter(payload),
        "style_source_pdf": str(pdf_path) if pdf_path is not None else None,
        "source": payload.get("source"),
        "v8_diagnostics_summary": {
            "atomic_block_count": len(payload.get("atomic_blocks") or []),
            "merge_count": len(payload.get("merge_decisions") or []),
        },
    }
    return convert_v7_payload_to_document_ir(
        adapter_payload,
        source_path=source_path,
        pdf_path=pdf_path,
        doc_id=stable_doc_id,
        config=MinerUV7DocumentIRAdapterConfig(
            require_styles=False,
            coordinate_space=CoordinateSpace.PDF_POINTS,
            default_page_width=page_width,
            default_page_height=page_height,
            extractor_name="mineru_v8_reflow",
        ),
    )


def write_v8_document_ir(
    content_json_path: Path,
    output_path: Path,
    *,
    pdf_path: Path | None = None,
    doc_id: str | None = None,
) -> DocumentIR:
    document = load_v8_document_ir(content_json_path, pdf_path=pdf_path, doc_id=doc_id)
    write_json(output_path, document)
    return document


def normalize_v8_items_for_adapter(payload: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    content_item_cache: dict[str, list[Any] | None] = {}
    middle_formula_cache: dict[str, dict[str, list[dict[str, Any]]] | None] = {}
    middle_caption_footnote_cache: dict[str, dict[str, list[dict[str, Any]]] | None] = {}
    middle_page_furniture_cache: dict[str, dict[str, list[dict[str, Any]]] | None] = {}
    model_label_cache: dict[str, list[dict[str, Any]] | None] = {}
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    source_content_list_json = source.get("content_list_json") if isinstance(source, dict) else None
    source_middle_json = source.get("middle_json") if isinstance(source, dict) else None
    source_model_json = source.get("model_json") if isinstance(source, dict) else None
    if not source_model_json:
        source_model_json = infer_model_json_path(source_middle_json or source_content_list_json)
    for index, item in enumerate(payload.get("items") or []):
        if not isinstance(item, dict):
            continue
        record = dict(item)
        if source_content_list_json and not record.get("content_list_json"):
            record["content_list_json"] = source_content_list_json
        if source_middle_json and not record.get("middle_json"):
            record["middle_json"] = source_middle_json
        if source_model_json and not record.get("model_json"):
            record["model_json"] = str(source_model_json)
        enrich_algorithm_provenance_from_content_list(record, content_item_cache)
        enrich_formula_provenance_from_content_list(record, content_item_cache)
        enrich_formula_provenance_from_middle(record, middle_formula_cache)
        enrich_caption_footnote_provenance_from_content_list(record, content_item_cache)
        enrich_caption_footnote_provenance_from_middle(record, middle_caption_footnote_cache)
        enrich_reference_provenance_from_content_list(record, content_item_cache)
        enrich_page_furniture_provenance_from_content_list(record, content_item_cache)
        enrich_page_furniture_provenance_from_middle(record, middle_page_furniture_cache)
        enrich_model_label_provenance(record, model_label_cache)
        raw_type = str(record.get("type") or record.get("raw_type") or "text")
        canonical_type = canonical_type_for_v8_item(record, index=index)
        record["id"] = str(record.get("id") or f"v8_{index:06d}")
        record["global_order"] = int(record.get("global_order") or record.get("reading_order") or index)
        record["raw_type"] = raw_type
        record["canonical_type"] = canonical_type
        record["layout_role"] = layout_role_for_v8_item(record, canonical_type=canonical_type, index=index)
        record["layout_layer"] = layout_layer_for_v8_item(record, canonical_type=canonical_type, index=index)
        record["v8_source"] = record.get("v8_source") or "middle_preproc_reflow"
        record["source_format"] = "mineru_middle_v8_reflow"
        if "source_block_ids" in record:
            record["mineru_middle_block_ids"] = record["source_block_ids"]
        if "reading_order" in record and "global_order" not in record:
            record["global_order"] = record["reading_order"]
        if canonical_type == "algorithm":
            apply_algorithm_metadata(record)
        if is_formula_item(record):
            apply_formula_metadata(record)
        if has_caption_footnote_item_metadata(record):
            apply_caption_footnote_metadata(record)
        if is_reference_item(record) or is_reference_heading_item(record):
            apply_reference_metadata(record)
        if has_page_furniture_or_model_label_metadata(record):
            apply_page_furniture_model_metadata(record)
        normalized.append(record)
    return normalized


def canonical_type_for_v8_item(item: dict[str, Any], *, index: int) -> str:
    raw = str(item.get("type") or "").casefold().strip()
    text = compact_text(item.get("text"))
    if raw == "text":
        if is_reference_like(text):
            return "ref_text"
        if is_document_title_like(item, index=index):
            return "title"
        return "paragraph"
    if raw == "abstract":
        return "paragraph"
    if raw == "ref_text":
        return "ref_text"
    if is_reference_item(item):
        return "ref_text"
    if raw == "interline_equation":
        return "equation_interline"
    if raw == "chart":
        return "figure"
    if raw == "image":
        return "figure"
    if is_algorithm_item(item):
        return "algorithm"
    return raw or "paragraph"


def layout_role_for_v8_item(item: dict[str, Any], *, canonical_type: str, index: int) -> str:
    text = compact_text(item.get("text"))
    raw = str(item.get("type") or "").casefold().strip()
    if int(item.get("page_idx") or 0) == 0 and index <= 8:
        lower = text.casefold()
        if index == 0 and text:
            return "document_title"
        if index == 1 and looks_like_author_line(text):
            return "author"
        if "@" in text:
            return "email"
        if looks_like_affiliation(text):
            return "affiliation"
        if lower.startswith("keywords"):
            return "front_matter"
    if canonical_type == "title":
        if is_front_title_position(item, index=index):
            return "document_title"
        if text.casefold().strip(" .:") == "abstract":
            return "abstract_title"
        return "body_heading"
    if raw == "abstract":
        return "abstract_body"
    if canonical_type == "ref_text":
        return "reference_item"
    if canonical_type in {"figure", "table", "algorithm"}:
        return canonical_type
    if canonical_type == "equation_interline":
        return "display_formula"
    return "body_text"


def layout_layer_for_v8_item(item: dict[str, Any], *, canonical_type: str, index: int) -> str:
    role = layout_role_for_v8_item(item, canonical_type=canonical_type, index=index)
    if role in {"document_title", "author", "authors", "affiliation", "email", "front_matter", "abstract_title", "abstract_body"}:
        return "metadata_layer"
    return "main_text_flow"


def infer_page_size(payload: dict[str, Any]) -> tuple[float, float]:
    sizes: list[tuple[float, float]] = []
    for block in payload.get("atomic_blocks") or []:
        if not isinstance(block, dict):
            continue
        size = block.get("page_size")
        if isinstance(size, list) and len(size) >= 2:
            try:
                sizes.append((float(size[0]), float(size[1])))
            except (TypeError, ValueError):
                pass
    if not sizes:
        return 612.0, 792.0
    widths = sorted(width for width, _ in sizes)
    heights = sorted(height for _, height in sizes)
    return widths[len(widths) // 2], heights[len(heights) // 2]


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def is_reference_like(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    return bool(value.startswith("[") or value[:1].isdigit() and any(year in value for year in (" 19", " 20")))


def is_document_title_like(item: dict[str, Any], *, index: int) -> bool:
    text = compact_text(item.get("text"))
    if index > 2 or int(item.get("page_idx") or 0) != 0:
        return False
    if not text or len(text) < 20:
        return False
    bbox = item.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return float(bbox[1]) < 180.0
    return True


def is_front_title_position(item: dict[str, Any], *, index: int) -> bool:
    return int(item.get("page_idx") or 0) == 0 and index <= 2


def has_algorithm_caption(item: dict[str, Any]) -> bool:
    captions = item.get("algorithm_caption") or item.get("code_caption")
    if isinstance(captions, list):
        return any(str(value).strip() for value in captions)
    return bool(str(captions or "").strip())


ALGORITHM_SUBTYPE_VALUES = {"algorithm", "alg", "pseudocode", "procedure"}
ALGORITHM_CONTENT_KEYS = ("algorithm_content", "code_body")
ALGORITHM_CAPTION_KEYS = ("algorithm_caption", "code_caption")
ALGORITHM_PROVENANCE_COPY_KEYS = (
    "sub_type",
    "subtype",
    "raw_sub_type",
    "mineru_subtype",
    "algorithm_content",
    "algorithm_caption",
    "algorithm_footnote",
    "code_body",
    "code_caption",
    "code_footnote",
)

FORMULA_INLINE_TYPES = {"inline_equation", "equation_inline", "inline_math", "inline_formula", "math_inline"}
FORMULA_DISPLAY_TYPES = {
    "equation",
    "equation_interline",
    "interline_equation",
    "display_formula",
    "formula",
}
FORMULA_TEXT_KEYS = ("formula_latex", "latex", "equation", "equation_text", "text", "content")
FORMULA_PROVENANCE_COPY_KEYS = (
    "type",
    "raw_formula_type",
    "mineru_span_type",
    "text_format",
    "formula_latex",
    "latex",
    "equation",
    "equation_text",
    "inline_equation_spans",
    "interline_equation_spans",
    "line_span_ids",
    "parent_line_id",
    "parent_block_id",
    "line_id",
    "span_id",
    "span_type",
    "span_bbox",
    "line_bbox",
    "span_text",
    "span_latex",
)

CAPTION_ROLE_TO_TYPE = {
    "image_caption": "figure",
    "figure_caption": "figure",
    "chart_caption": "chart",
    "table_caption": "table",
    "code_caption": "code",
    "algorithm_caption": "algorithm",
}
FOOTNOTE_ROLE_TO_TYPE = {
    "image_footnote": "image_note",
    "figure_footnote": "image_note",
    "chart_footnote": "chart_note",
    "table_footnote": "table_note",
    "code_footnote": "code_note",
    "algorithm_footnote": "code_note",
    "page_footnote": "page_note",
}
CAPTION_PROVENANCE_COPY_KEYS = (
    "image_caption",
    "figure_caption",
    "table_caption",
    "chart_caption",
    "code_caption",
    "algorithm_caption",
    "caption",
    "caption_text",
    "caption_bbox",
    "caption_node_ids",
    "caption_source_ids",
    "caption_parent_float_id",
    "caption_body_ids",
    "body_node_ids",
    "image_body",
    "table_body",
    "chart_body",
    "code_body",
    "algorithm_content",
    "img_path",
    "image_path",
    "figure_path",
    "asset_path",
)
FOOTNOTE_PROVENANCE_COPY_KEYS = (
    "image_footnote",
    "figure_footnote",
    "table_footnote",
    "chart_footnote",
    "code_footnote",
    "algorithm_footnote",
    "footnote_text",
    "footnote_bbox",
    "footnote_node_ids",
    "footnote_source_ids",
    "footnote_parent_float_id",
    "footnote_body_ids",
)

REFERENCE_SUBTYPE_VALUES = {"ref_text", "reference", "references", "bibliography", "bibliography_item"}
REFERENCE_LIST_TYPES = {"list", "ordered_list", "unordered_list", "reference", "ref_text"}
REFERENCE_PROVENANCE_COPY_KEYS = (
    "type",
    "sub_type",
    "subtype",
    "raw_sub_type",
    "list_items",
    "list_marker",
    "list_marker_text",
    "reference_text",
    "reference_label",
    "reference_items",
    "reference_source_ids",
    "reference_bbox",
)
REFERENCE_HEADING_RE = re.compile(r"^\s*(references|bibliography|reference)\s*$", re.IGNORECASE)

PAGE_FURNITURE_CONTENT_TYPES = {"header", "footer", "page_number", "aside_text", "page_footnote", "footnote", "noise"}
PAGE_FURNITURE_ROLES = {
    "header": "page_header",
    "page_header": "page_header",
    "footer": "page_footer",
    "page_footer": "page_footer",
    "page_number": "page_number",
    "number": "page_number",
    "aside_text": "aside_text",
    "page_aside_text": "aside_text",
    "margin_note": "margin_note",
    "marginnote": "margin_note",
    "side_note": "margin_note",
    "sidenote": "margin_note",
    "page_footnote": "page_footnote",
    "footnote": "page_footnote",
    "discarded_block": "discarded_block",
    "discarded": "discarded_block",
    "noise": "discarded_block",
}
MODEL_ROLE_VOTES = {
    "doc_title": "doc_title",
    "title": "title",
    "paragraph_title": "title",
    "text": "text",
    "ocr_text": "text",
    "abstract": "text",
    "header": "header",
    "footer": "footer",
    "page_number": "page_number",
    "number": "page_number",
    "figure": "figure",
    "image": "figure",
    "table": "table",
    "formula": "formula",
    "equation": "formula",
    "code": "code",
    "reference": "reference",
    "list": "list",
}
PAGE_FURNITURE_PROVENANCE_COPY_KEYS = (
    "type",
    "sub_type",
    "subtype",
    "text",
    "bbox",
    "page_idx",
    "layout_label",
    "sub_layout",
)


def normalize_marker(value: Any) -> str:
    return str(value or "").casefold().strip()


def nonempty_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(nonempty_value(part) for part in value)
    if isinstance(value, dict):
        return any(nonempty_value(part) for part in value.values())
    return value not in (None, "", [], {})


def is_reference_subtype_value(value: Any) -> bool:
    return normalize_marker(value) in REFERENCE_SUBTYPE_VALUES


def is_reference_item(item: dict[str, Any]) -> bool:
    """Return true only for upstream reference-list subtype evidence.

    Body citations such as "see [1]" and ordinary numbered lists must not become
    reference items without MinerU/content_list subtype evidence.
    """

    if normalize_marker(item.get("mineru_reference_role")) in {"ref_text", "reference_list", "bibliography_item"}:
        return True
    if any(is_reference_subtype_value(item.get(key)) for key in ("type", "raw_type", "canonical_type")):
        return True
    if any(is_reference_subtype_value(item.get(key)) for key in ("sub_type", "subtype", "raw_sub_type", "mineru_subtype")):
        raw = normalize_marker(item.get("type") or item.get("raw_type") or item.get("content_list_type"))
        return raw in REFERENCE_LIST_TYPES or raw in REFERENCE_SUBTYPE_VALUES
    if is_reference_subtype_value(item.get("content_list_type")):
        return True
    return False


def is_reference_heading_item(item: dict[str, Any]) -> bool:
    role = normalize_marker(item.get("mineru_reference_role"))
    if role == "reference_heading":
        return True
    if normalize_marker(item.get("reference_context_role")) == "reference_heading":
        return True
    return bool(REFERENCE_HEADING_RE.match(compact_text(item.get("text"))))


def reference_source_layer(item: dict[str, Any]) -> str:
    source = str(item.get("reference_source_layer") or "").strip()
    if source:
        return source
    if item.get("source_content_list_index") is not None or item.get("content_list_text_candidates"):
        return "content_list"
    if item.get("source_middle_indices") or item.get("mineru_middle_block_ids") or item.get("source_block_ids"):
        return "middle"
    return "metadata"


def reference_confidence(item: dict[str, Any]) -> str:
    if any(is_reference_subtype_value(item.get(key)) for key in ("type", "raw_type", "canonical_type", "content_list_type")):
        return "strong_ref_text_subtype"
    if is_reference_subtype_value(item.get("raw_sub_type")) or is_reference_subtype_value(item.get("mineru_subtype")):
        return "strong_ref_text_subtype"
    if normalize_marker(item.get("mineru_reference_role")) in {"reference_heading", "reference_list"}:
        return "strong_reference_region"
    if nonempty_value(item.get("list_items")) or nonempty_value(item.get("reference_items")):
        return "medium_list_item"
    return "weak_regex_only"


def reference_context_role(item: dict[str, Any]) -> str:
    if is_reference_heading_item(item) and not is_reference_item(item):
        return "reference_heading"
    if is_reference_item(item):
        return "reference_item"
    if normalize_marker(item.get("mineru_reference_role")) == "reference_list":
        return "bibliography_block"
    if normalize_marker(item.get("content_list_type")) == "list":
        return "ordinary_list"
    return "diagnostic_only"


def reference_text_value(item: dict[str, Any]) -> str:
    if nonempty_value(item.get("reference_text")):
        return text_value(item.get("reference_text"))
    if nonempty_value(item.get("list_items")):
        return text_value(item.get("list_items"))
    if nonempty_value(item.get("reference_items")):
        return text_value(item.get("reference_items"))
    return text_value(item.get("text"))


def reference_label_value(item: dict[str, Any]) -> str:
    for key in ("reference_label", "list_marker_text", "list_marker", "label"):
        if nonempty_value(item.get(key)):
            return text_value(item.get(key))
    text = compact_text(reference_text_value(item))
    match = re.match(r"^\s*(\[[^\]]+\]|\d+[\).]?)", text)
    return match.group(1) if match else ""


def _reference_source_ids_for_item(item: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("reference_source_ids", "source_line_ids", "source_block_ids"):
        value = item.get(key)
        if isinstance(value, list):
            ids.extend(str(part) for part in value if str(part))
        elif value:
            ids.append(str(value))
    if item.get("id"):
        ids.append(str(item["id"]))
    return list(dict.fromkeys(ids))


def apply_reference_metadata(item: dict[str, Any]) -> None:
    layer = reference_source_layer(item)
    role = "reference_heading" if is_reference_heading_item(item) and not is_reference_item(item) else "ref_text"
    if normalize_marker(item.get("mineru_reference_role")) in {"reference_list", "bibliography_item"}:
        role = normalize_marker(item.get("mineru_reference_role"))
    raw_subtype = item.get("sub_type") or item.get("subtype") or item.get("raw_sub_type") or item.get("mineru_subtype")
    if raw_subtype is not None:
        item.setdefault("raw_reference_sub_type", raw_subtype)
        item.setdefault("raw_sub_type", raw_subtype)
        item.setdefault("mineru_subtype", raw_subtype)
    item["raw_reference_type"] = item.get("content_list_type") or item.get("type") or item.get("raw_type") or role
    item["mineru_reference_role"] = role
    item["reference_text"] = reference_text_value(item)
    item["reference_label"] = reference_label_value(item)
    item["reference_source_layer"] = layer
    item["reference_confidence"] = reference_confidence(item)
    item["reference_list_item_index"] = item.get("reference_list_item_index", item.get("source_content_list_index"))
    if not item.get("reference_parent_block_id"):
        source_blocks = item.get("source_block_ids")
        if isinstance(source_blocks, list) and source_blocks:
            item["reference_parent_block_id"] = str(source_blocks[0])
    item.setdefault("reference_source_ids", _reference_source_ids_for_item(item))
    item.setdefault("reference_bbox", item.get("bbox"))
    item["is_reference_item"] = bool(is_reference_item(item))
    item["is_reference_section_candidate"] = bool(is_reference_item(item) or is_reference_heading_item(item))
    item["reference_context_role"] = reference_context_role(item)
    if item.get("reference_source_ids"):
        item.setdefault("reference_item_ids", list(item.get("reference_source_ids") or []))
    if item["reference_context_role"] == "reference_heading" and item.get("id"):
        item.setdefault("reference_heading_ids", [str(item["id"])])
    if item.get("reference_parent_block_id"):
        item.setdefault("parent_reference_block_id", item["reference_parent_block_id"])
    item.setdefault("list_item_order", item.get("source_content_list_index"))
    if item.get("reference_label"):
        item.setdefault("list_marker_text", item["reference_label"])
    item.setdefault("source_layer_hierarchy", layer)
    if not compact_text(item.get("text")) and item.get("reference_text"):
        item["text"] = item["reference_text"]
        item["text_source"] = item.get("text_source") or "reference_metadata"


def infer_model_json_path(source_path: Any) -> str | None:
    if not source_path:
        return None
    path = Path(str(source_path))
    candidates = [
        path.with_name(path.name.replace("_middle.json", "_model.json")),
        path.with_name(path.name.replace("_content_list.json", "_model.json")),
        path.with_name(path.stem + "_model.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0]) if candidates else None


def page_furniture_role_from_marker(value: Any) -> str | None:
    marker = normalize_marker(value)
    return PAGE_FURNITURE_ROLES.get(marker)


def model_role_vote_from_label(value: Any) -> str:
    return MODEL_ROLE_VOTES.get(normalize_marker(value), "unknown")


def has_page_furniture_or_model_label_metadata(item: dict[str, Any]) -> bool:
    if item.get("model_label") is not None:
        return True
    if item.get("mineru_page_furniture_role") or item.get("raw_page_furniture_type"):
        return True
    for key in ("content_list_type", "type", "raw_type", "layout_role"):
        if page_furniture_role_from_marker(item.get(key)):
            return True
    return False


def page_furniture_source_layer(item: dict[str, Any]) -> str:
    source = str(item.get("page_furniture_source_layer") or "").strip()
    if source:
        return source
    if item.get("source_content_list_index") is not None or item.get("content_list_text_candidates"):
        return "content_list"
    if item.get("source_middle_indices") or item.get("mineru_middle_block_ids") or item.get("source_block_ids"):
        return "middle"
    if item.get("model_label") is not None:
        return "model"
    return "metadata"


def page_furniture_confidence(item: dict[str, Any]) -> str:
    if item.get("model_label") is not None and page_furniture_role_from_marker(item.get("model_label")):
        return "strong_model_label"
    if page_furniture_role_from_marker(item.get("content_list_type")) or page_furniture_role_from_marker(item.get("type")):
        return "strong_content_list_role"
    if normalize_marker(item.get("mineru_page_furniture_role")) == "discarded_block":
        return "strong_middle_discarded"
    if item.get("model_label") is not None:
        return "strong_model_label"
    return "medium_layout_position"


def page_furniture_text_value(item: dict[str, Any]) -> str:
    for key in ("page_furniture_text", "content_list_text", "middle_text", "text"):
        if nonempty_value(item.get(key)):
            return text_value(item.get(key))
    return ""


def page_furniture_source_ids(item: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("page_furniture_source_ids", "source_line_ids", "source_block_ids"):
        value = item.get(key)
        if isinstance(value, list):
            ids.extend(str(part) for part in value if str(part))
        elif value:
            ids.append(str(value))
    if item.get("id"):
        ids.append(str(item["id"]))
    return list(dict.fromkeys(ids))


def page_furniture_role(item: dict[str, Any]) -> str:
    for key in ("mineru_page_furniture_role", "raw_page_furniture_type", "content_list_type", "type", "raw_type", "layout_role"):
        role = page_furniture_role_from_marker(item.get(key))
        if role:
            return role
    model_role = page_furniture_role_from_marker(item.get("model_label"))
    if model_role:
        return model_role
    return "unknown"


def is_front_matter_title_evidence(item: dict[str, Any]) -> bool:
    vote = model_role_vote_from_label(item.get("model_label"))
    page_idx = int(item.get("page_idx") or 0)
    if vote in {"doc_title", "title"} and page_idx == 0:
        return True
    role = normalize_marker(item.get("layout_role"))
    return page_idx == 0 and role in {"document_title", "abstract_title", "front_matter"}


def apply_page_furniture_model_metadata(item: dict[str, Any]) -> None:
    model_label = item.get("model_label")
    if model_label is not None:
        item["model_label"] = str(model_label)
        if item.get("model_score") is not None:
            try:
                item["model_score"] = float(item["model_score"])
            except (TypeError, ValueError):
                item["model_score"] = item.get("model_score")
        item.setdefault("model_source_layer", "model")
        item["model_role_vote"] = model_role_vote_from_label(model_label)
        item["model_label_confidence"] = "strong_model_label"
        if item.get("model_bbox") is None and item.get("bbox") is not None:
            item["model_bbox"] = item.get("bbox")

    role = page_furniture_role(item)
    is_page_furniture = role in {
        "page_header",
        "page_footer",
        "page_number",
        "page_footnote",
        "aside_text",
        "margin_note",
        "discarded_block",
    }
    if is_page_furniture:
        item["raw_page_furniture_type"] = item.get("raw_page_furniture_type") or item.get("content_list_type") or item.get("type") or role
        item["mineru_page_furniture_role"] = role
        item["page_furniture_text"] = page_furniture_text_value(item)
        item["page_furniture_source_layer"] = page_furniture_source_layer(item)
        item["page_furniture_confidence"] = page_furniture_confidence(item)
        item.setdefault("page_furniture_bbox", item.get("bbox"))
        item.setdefault("page_furniture_source_ids", page_furniture_source_ids(item))

    item["is_page_header"] = role == "page_header"
    item["is_page_footer"] = role == "page_footer"
    item["is_page_number"] = role == "page_number"
    item["is_page_footnote"] = role == "page_footnote"
    item["is_aside_or_margin_note"] = role in {"aside_text", "margin_note"}
    item["is_discarded_block"] = role == "discarded_block"

    exclude_body = role in {"page_header", "page_footer", "page_number", "page_footnote", "discarded_block"}
    if item["is_aside_or_margin_note"]:
        exclude_body = True
    item["should_exclude_from_body_order"] = bool(exclude_body)
    item["should_exclude_from_heading_detection"] = bool(
        exclude_body or is_front_matter_title_evidence(item)
    )
    item["should_exclude_from_visible_prose_metric"] = bool(exclude_body)
    item["should_exclude_from_gnn_body_view"] = bool(exclude_body)

    vote = model_role_vote_from_label(item.get("model_label"))
    item["is_document_title_candidate"] = bool(vote == "doc_title" or normalize_marker(item.get("layout_role")) == "document_title")
    item["is_front_matter_candidate"] = bool(is_front_matter_title_evidence(item))
    item["is_author_affiliation_candidate"] = bool(normalize_marker(item.get("layout_role")) in {"author", "affiliation", "email"})
    item["is_abstract_title_candidate"] = bool(normalize_marker(item.get("layout_role")) == "abstract_title")
    item["front_matter_negative_for_body_heading"] = bool(item["is_front_matter_candidate"])
    item["title_negative_for_body_heading"] = bool(item["is_document_title_candidate"] or (vote == "title" and int(item.get("page_idx") or 0) == 0))
    item["abstract_title_negative_for_body_heading"] = bool(item["is_abstract_title_candidate"])


def first_nonempty_value(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = item.get(key)
        if nonempty_value(value):
            return value
    return None


def text_value(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(text_value(part) for part in value if text_value(part)).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "caption", "latex"):
            if nonempty_value(value.get(key)):
                return text_value(value.get(key))
        return " ".join(text_value(part) for part in value.values() if text_value(part)).strip()
    if value is None:
        return ""
    return str(value).strip()


def is_algorithm_subtype_value(value: Any) -> bool:
    return normalize_marker(value) in ALGORITHM_SUBTYPE_VALUES


def has_algorithm_subtype(item: dict[str, Any]) -> bool:
    return any(
        is_algorithm_subtype_value(item.get(key))
        for key in ("type", "canonical_type", "content_list_type", "sub_type", "subtype", "raw_sub_type", "mineru_subtype")
    )


def has_explicit_algorithm_subtype(item: dict[str, Any]) -> bool:
    return any(
        is_algorithm_subtype_value(item.get(key))
        for key in ("type", "content_list_type", "sub_type", "subtype", "raw_sub_type", "mineru_subtype")
    )


def has_algorithm_content(item: dict[str, Any]) -> bool:
    return nonempty_value(first_nonempty_value(item, ALGORITHM_CONTENT_KEYS))


def is_algorithm_item(item: dict[str, Any]) -> bool:
    """Return true only for explicit MinerU algorithm/code subtype evidence.

    This intentionally avoids text-only keyword guesses such as
    "Algorithm 1 shows ..."; those stay in the audit detector instead of the
    production v8 adapter.
    """

    raw = normalize_marker(item.get("type") or item.get("raw_type"))
    if has_algorithm_subtype(item):
        return True
    if raw == "code" and has_algorithm_caption(item):
        return True
    if raw == "code" and has_algorithm_content(item) and (
        is_algorithm_subtype_value(item.get("sub_type"))
        or is_algorithm_subtype_value(item.get("raw_sub_type"))
        or is_algorithm_subtype_value(item.get("mineru_subtype"))
    ):
        return True
    return False


def algorithm_confidence(item: dict[str, Any]) -> str:
    if has_explicit_algorithm_subtype(item):
        return "strong_subtype"
    if has_algorithm_caption(item):
        return "medium_caption"
    return "weak_text_only"


def algorithm_origin(item: dict[str, Any]) -> str:
    source = str(item.get("algorithm_origin") or "").strip()
    if source:
        return source
    if item.get("source_content_list_index") is not None:
        return "raw_content_list"
    if item.get("source_middle_indices") or item.get("mineru_middle_block_ids"):
        return "middle"
    if has_algorithm_caption(item):
        return "inferred_from_caption"
    return "metadata"


def algorithm_text_value(item: dict[str, Any]) -> str:
    value = first_nonempty_value(item, ALGORITHM_CONTENT_KEYS)
    if isinstance(value, list):
        return "\n".join(str(part).strip() for part in value if str(part).strip()).strip()
    if value is None:
        return ""
    return str(value).strip()


def apply_algorithm_metadata(item: dict[str, Any]) -> None:
    subtype = item.get("sub_type") or item.get("subtype") or item.get("raw_sub_type") or item.get("mineru_subtype")
    if subtype is not None:
        item.setdefault("raw_sub_type", subtype)
        item.setdefault("mineru_subtype", subtype)
    item["is_algorithm_subtype"] = bool(has_algorithm_subtype(item))
    item["algorithm_origin"] = algorithm_origin(item)
    item["algorithm_confidence"] = algorithm_confidence(item)
    if item.get("source_block_ids") and not item.get("algorithm_body_ids"):
        item["algorithm_body_ids"] = list(item.get("source_block_ids") or [])
    if item.get("source_line_ids") and not item.get("algorithm_caption_ids") and has_algorithm_caption(item):
        item["algorithm_caption_ids"] = list(item.get("source_line_ids") or [])
    if not compact_text(item.get("text")):
        body_text = algorithm_text_value(item)
        if body_text:
            item["text"] = body_text
            item["text_source"] = item.get("text_source") or "algorithm_body_metadata"


def is_formula_type_value(value: Any) -> bool:
    marker = normalize_marker(value)
    return marker in FORMULA_INLINE_TYPES or marker in FORMULA_DISPLAY_TYPES


def is_inline_formula_type_value(value: Any) -> bool:
    return normalize_marker(value) in FORMULA_INLINE_TYPES


def is_display_formula_type_value(value: Any) -> bool:
    return normalize_marker(value) in FORMULA_DISPLAY_TYPES


def has_formula_latex_format(item: dict[str, Any]) -> bool:
    return normalize_marker(item.get("text_format")) == "latex"


def is_formula_item(item: dict[str, Any]) -> bool:
    """Return true only for upstream formula/equation evidence.

    This deliberately excludes ordinary prose that merely contains variables,
    citations, or punctuation. Regex-only math guesses belong in diagnostics,
    not in v8 adapter preservation.
    """

    keys = ("type", "raw_type", "canonical_type", "content_list_type", "mineru_span_type", "span_type", "raw_formula_type")
    if any(is_formula_type_value(item.get(key)) for key in keys):
        return True
    if has_formula_latex_format(item) and any(nonempty_value(item.get(key)) for key in FORMULA_TEXT_KEYS):
        return True
    if nonempty_value(item.get("inline_equation_spans")) or nonempty_value(item.get("interline_equation_spans")):
        return True
    return False


def formula_text_value(item: dict[str, Any]) -> str:
    value = first_nonempty_value(item, FORMULA_TEXT_KEYS)
    if isinstance(value, list):
        return " ".join(str(part).strip() for part in value if str(part).strip()).strip()
    if isinstance(value, dict):
        return " ".join(str(part).strip() for part in value.values() if str(part).strip()).strip()
    if value is None:
        return ""
    return str(value).strip()


def formula_confidence(item: dict[str, Any]) -> str:
    if nonempty_value(item.get("inline_equation_spans")):
        return "strong_span_inline"
    if nonempty_value(item.get("interline_equation_spans")):
        return "strong_span_interline"
    if any(is_inline_formula_type_value(item.get(key)) for key in ("type", "raw_type", "canonical_type", "mineru_span_type", "span_type")):
        return "strong_span_inline"
    if any(is_display_formula_type_value(item.get(key)) for key in ("type", "raw_type", "canonical_type", "mineru_span_type", "span_type")):
        return "strong_span_interline"
    if normalize_marker(item.get("content_list_type")) == "equation" and has_formula_latex_format(item):
        return "strong_content_equation_latex"
    if has_formula_latex_format(item) or normalize_marker(item.get("content_list_type")) == "equation":
        return "medium_equation_text"
    return "weak_text_only"


def formula_source_layer(item: dict[str, Any]) -> str:
    source = str(item.get("formula_source_layer") or "").strip()
    if source:
        return source
    if item.get("source_content_list_index") is not None or item.get("content_list_text_candidates"):
        return "content_list"
    if item.get("source_middle_indices") or item.get("mineru_middle_block_ids") or item.get("source_lines"):
        return "middle"
    return "metadata"


def formula_context_role(item: dict[str, Any]) -> str:
    confidence = formula_confidence(item)
    text = formula_text_value(item)
    bbox = item.get("bbox")
    if confidence == "strong_span_inline":
        return "inline_attachment"
    if confidence in {"strong_span_interline", "strong_content_equation_latex"}:
        return "display_math"
    if normalize_marker(item.get("content_list_type")) == "equation":
        return "equation_block"
    if len(text) <= 6 and isinstance(bbox, list):
        return "formula_ocr_artifact"
    return "uncertain"


def _line_span_ids_from_item(item: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("line_span_ids", "source_line_ids"):
        value = item.get(key)
        if isinstance(value, list):
            ids.extend(str(part) for part in value if str(part))
        elif value:
            ids.append(str(value))
    for line in item.get("source_lines") or []:
        if isinstance(line, dict) and line.get("line_id"):
            ids.append(str(line["line_id"]))
    for key in ("inline_equation_spans", "interline_equation_spans"):
        for span in item.get(key) or []:
            if isinstance(span, dict):
                span_id = span.get("span_id") or span.get("line_id")
                if span_id:
                    ids.append(str(span_id))
            elif span:
                ids.append(str(span))
    return list(dict.fromkeys(ids))


def apply_formula_metadata(item: dict[str, Any]) -> None:
    raw_type = item.get("raw_formula_type") or item.get("mineru_span_type") or item.get("span_type") or item.get("type") or item.get("raw_type")
    if raw_type is not None:
        item.setdefault("raw_formula_type", raw_type)
    if item.get("span_type") is not None:
        item.setdefault("mineru_span_type", item.get("span_type"))
    elif is_formula_type_value(raw_type):
        item.setdefault("mineru_span_type", raw_type)

    text_format = item.get("text_format")
    if text_format is not None:
        item["text_format"] = str(text_format)

    latex = formula_text_value(item)
    if latex:
        item.setdefault("formula_latex", latex)

    line_span_ids = _line_span_ids_from_item(item)
    if line_span_ids:
        item.setdefault("line_span_ids", line_span_ids)
        if not item.get("parent_line_id"):
            item["parent_line_id"] = line_span_ids[0]

    if not item.get("parent_block_id"):
        source_blocks = item.get("source_block_ids")
        if isinstance(source_blocks, list) and source_blocks:
            item["parent_block_id"] = str(source_blocks[0])

    source_lines = item.get("source_lines")
    if isinstance(source_lines, list) and source_lines:
        first_line = next((line for line in source_lines if isinstance(line, dict)), None)
        if first_line is not None:
            item.setdefault("line_id", first_line.get("line_id"))
            item.setdefault("span_id", first_line.get("line_id"))
            item.setdefault("line_bbox", first_line.get("bbox"))
            item.setdefault("span_bbox", first_line.get("bbox"))
            item.setdefault("span_text", first_line.get("text"))
            item.setdefault("span_latex", first_line.get("text"))

    inline = bool(nonempty_value(item.get("inline_equation_spans"))) or any(
        is_inline_formula_type_value(item.get(key)) for key in ("type", "raw_type", "canonical_type", "mineru_span_type", "span_type")
    )
    display = bool(nonempty_value(item.get("interline_equation_spans"))) or any(
        is_display_formula_type_value(item.get(key)) for key in ("type", "raw_type", "canonical_type", "mineru_span_type", "span_type")
    )
    if normalize_marker(item.get("content_list_type")) == "equation" and not inline:
        display = True
    item["is_inline_math"] = bool(inline)
    item["is_display_math"] = bool(display)
    if inline:
        item.setdefault("inline_equation_spans", line_span_ids or ([item.get("id")] if item.get("id") else []))
    if display:
        item.setdefault("interline_equation_spans", line_span_ids or ([item.get("id")] if item.get("id") else []))
    item["formula_source_layer"] = formula_source_layer(item)
    item["formula_confidence"] = formula_confidence(item)
    item["formula_context_role"] = formula_context_role(item)


def load_content_items_cached(path: str, cache: dict[str, list[Any] | None]) -> list[Any] | None:
    if path in cache:
        return cache[path]
    try:
        payload = read_json(Path(path))
    except Exception:
        cache[path] = None
        return None
    items = payload if isinstance(payload, list) else payload.get("items") if isinstance(payload, dict) else None
    cache[path] = items if isinstance(items, list) else None
    return cache[path]


def enrich_algorithm_provenance_from_content_list(item: dict[str, Any], cache: dict[str, list[Any] | None]) -> None:
    path = item.get("content_list_json")
    index = item.get("source_content_list_index")
    if not path or index is None:
        return
    try:
        source_index = int(index)
    except (TypeError, ValueError):
        return
    raw_items = load_content_items_cached(str(path), cache)
    if raw_items is None or source_index < 0 or source_index >= len(raw_items):
        return
    raw_item = raw_items[source_index]
    if not isinstance(raw_item, dict):
        return
    raw_subtype = raw_item.get("sub_type") or raw_item.get("subtype")
    if raw_subtype is not None:
        item.setdefault("raw_sub_type", raw_subtype)
        item.setdefault("mineru_subtype", raw_subtype)
    for key in ALGORITHM_PROVENANCE_COPY_KEYS:
        value = raw_item.get(key)
        if nonempty_value(value):
            item.setdefault(key, value)
    raw_type = raw_item.get("type")
    if raw_type is not None:
        item.setdefault("content_list_raw_type", raw_type)
    if is_algorithm_item(item):
        item.setdefault("algorithm_origin", "raw_content_list")


def enrich_formula_provenance_from_content_list(item: dict[str, Any], cache: dict[str, list[Any] | None]) -> None:
    path = item.get("content_list_json")
    index = item.get("source_content_list_index")
    if not path or index is None:
        return
    try:
        source_index = int(index)
    except (TypeError, ValueError):
        return
    raw_items = load_content_items_cached(str(path), cache)
    if raw_items is None or source_index < 0 or source_index >= len(raw_items):
        return
    raw_item = raw_items[source_index]
    if not isinstance(raw_item, dict):
        return
    raw_type = raw_item.get("type")
    if raw_type is not None:
        item.setdefault("content_list_type", raw_type)
        if is_formula_type_value(raw_type):
            item.setdefault("raw_formula_type", raw_type)
    for key in FORMULA_PROVENANCE_COPY_KEYS:
        value = raw_item.get(key)
        if nonempty_value(value):
            item.setdefault(key, value)
    if is_formula_item(item):
        item.setdefault("formula_source_layer", "content_list")


def load_middle_formula_index_cached(path: str, cache: dict[str, dict[str, list[dict[str, Any]]] | None]) -> dict[str, list[dict[str, Any]]] | None:
    if path in cache:
        return cache[path]
    try:
        payload = read_json(Path(path))
    except Exception:
        cache[path] = None
        return None
    cache[path] = build_middle_formula_index(payload)
    return cache[path]


def build_middle_formula_index(payload: Any) -> dict[str, list[dict[str, Any]]]:
    by_block: dict[str, list[dict[str, Any]]] = {}
    by_line: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(payload, dict):
        return {"by_block": [], "by_line": []}
    doc_id = str(payload.get("doc_id") or "")
    # Most MinerU middle files do not store doc_id; infer later through source ids.
    for page in payload.get("pdf_info") or []:
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
                block_stub = f"p{page_idx:04d}:m{int(block_index):06d}"
                block_type = normalize_marker(block.get("type"))
                if is_formula_type_value(block_type):
                    entry = {
                        "span_id_suffix": f"{block_stub}:block",
                        "block_id_suffix": block_stub,
                        "page_idx": page_idx,
                        "bbox": block.get("bbox"),
                        "span_type": block_type,
                        "text": formula_text_value(block),
                    }
                    by_block.setdefault(block_stub, []).append(entry)
                for line_idx, line in enumerate(block.get("lines") or []):
                    if not isinstance(line, dict):
                        continue
                    for span_idx, span in enumerate(line.get("spans") or []):
                        if not isinstance(span, dict):
                            continue
                        span_type = normalize_marker(span.get("type"))
                        if not is_formula_type_value(span_type):
                            continue
                        line_stub = f"{block_stub}:l{line_idx:04d}:s{span_idx:04d}"
                        entry = {
                            "span_id_suffix": line_stub,
                            "line_id_suffix": line_stub,
                            "block_id_suffix": block_stub,
                            "page_idx": page_idx,
                            "bbox": span.get("bbox") or line.get("bbox"),
                            "line_bbox": line.get("bbox"),
                            "span_type": span_type,
                            "text": formula_text_value(span),
                        }
                        by_line.setdefault(line_stub, []).append(entry)
                        by_block.setdefault(block_stub, []).append(entry)
    return {"by_block": by_block, "by_line": by_line}


def source_id_suffix(value: Any) -> str:
    text = str(value or "")
    marker = re.search(r"p\d{4}:m\d{6}(?::l\d{4}:s\d{4})?", text)
    return marker.group(0) if marker else text


def span_entry_for_metadata(entry: dict[str, Any], *, full_id: str | None = None) -> dict[str, Any]:
    return {
        "span_id": full_id or entry.get("line_id_suffix") or entry.get("span_id_suffix"),
        "parent_block_id": entry.get("block_id_suffix"),
        "span_type": entry.get("span_type"),
        "bbox": entry.get("bbox"),
        "line_bbox": entry.get("line_bbox"),
        "text": entry.get("text"),
        "latex": entry.get("text"),
    }


def enrich_formula_provenance_from_middle(item: dict[str, Any], cache: dict[str, dict[str, list[dict[str, Any]]] | None]) -> None:
    path = item.get("middle_json")
    if not path:
        return
    index = load_middle_formula_index_cached(str(path), cache)
    if not index:
        return
    by_block = index.get("by_block") or {}
    by_line = index.get("by_line") or {}
    inline_spans: list[dict[str, Any]] = []
    interline_spans: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_line_id in item.get("source_line_ids") or []:
        suffix = source_id_suffix(source_line_id)
        for entry in by_line.get(suffix, []):
            key = str(entry.get("span_id_suffix"))
            if key in seen:
                continue
            seen.add(key)
            full_id = str(source_line_id)
            target = inline_spans if is_inline_formula_type_value(entry.get("span_type")) else interline_spans
            target.append(span_entry_for_metadata(entry, full_id=full_id))
    for source_block_id in item.get("source_block_ids") or []:
        suffix = source_id_suffix(source_block_id)
        for entry in by_block.get(suffix, []):
            key = str(entry.get("span_id_suffix"))
            if key in seen:
                continue
            seen.add(key)
            full_id = f"{source_block_id}:{entry.get('line_id_suffix', entry.get('span_id_suffix', 'block')).split(':', 2)[-1]}"
            target = inline_spans if is_inline_formula_type_value(entry.get("span_type")) else interline_spans
            target.append(span_entry_for_metadata(entry, full_id=full_id))
    if inline_spans:
        item.setdefault("inline_equation_spans", inline_spans)
    if interline_spans:
        item.setdefault("interline_equation_spans", interline_spans)
    if inline_spans or interline_spans:
        item.setdefault("formula_source_layer", "middle")
        first = (inline_spans or interline_spans)[0]
        item.setdefault("mineru_span_type", first.get("span_type"))
        item.setdefault("span_type", first.get("span_type"))
        if not item.get("formula_latex") and not compact_text(item.get("text")):
            item["formula_latex"] = first.get("latex")


def caption_type_from_role(role: str, item: dict[str, Any] | None = None) -> str:
    marker = normalize_marker(role)
    if marker == "code_caption" and item and is_algorithm_item(item):
        return "algorithm"
    return CAPTION_ROLE_TO_TYPE.get(marker, "unknown")


def footnote_type_from_role(role: str) -> str:
    return FOOTNOTE_ROLE_TO_TYPE.get(normalize_marker(role), "unknown")


def caption_role_for_item(item: dict[str, Any]) -> str | None:
    for key in ("mineru_caption_role", "raw_caption_type"):
        marker = normalize_marker(item.get(key))
        if marker in CAPTION_ROLE_TO_TYPE:
            return marker
    for key in ("algorithm_caption", "code_caption", "table_caption", "image_caption", "figure_caption", "chart_caption"):
        if nonempty_value(item.get(key)):
            if key == "code_caption" and is_algorithm_item(item):
                return "algorithm_caption"
            return key
    if nonempty_value(item.get("caption_text")) and normalize_marker(item.get("caption_type")) in {
        "figure",
        "table",
        "chart",
        "algorithm",
        "code",
    }:
        caption_type = normalize_marker(item.get("caption_type"))
        return {
            "figure": "image_caption",
            "table": "table_caption",
            "chart": "chart_caption",
            "algorithm": "algorithm_caption",
            "code": "code_caption",
        }.get(caption_type)
    return None


def footnote_role_for_item(item: dict[str, Any]) -> str | None:
    for key in ("mineru_footnote_role", "raw_footnote_type"):
        marker = normalize_marker(item.get(key))
        if marker in FOOTNOTE_ROLE_TO_TYPE:
            return marker
    for key in ("algorithm_footnote", "code_footnote", "table_footnote", "image_footnote", "figure_footnote", "chart_footnote"):
        if nonempty_value(item.get(key)):
            return key
    if nonempty_value(item.get("footnote_text")):
        footnote_type = normalize_marker(item.get("footnote_type"))
        return {
            "image_note": "image_footnote",
            "table_note": "table_footnote",
            "chart_note": "chart_footnote",
            "code_note": "code_footnote",
            "page_note": "page_footnote",
        }.get(footnote_type)
    return None


def caption_text_value(item: dict[str, Any]) -> str:
    role = caption_role_for_item(item)
    if role and nonempty_value(item.get(role)):
        return text_value(item.get(role))
    return text_value(first_nonempty_value(item, ("caption_text", "caption")))


def footnote_text_value(item: dict[str, Any]) -> str:
    role = footnote_role_for_item(item)
    if role and nonempty_value(item.get(role)):
        return text_value(item.get(role))
    return text_value(first_nonempty_value(item, ("footnote_text",)))


def has_caption_footnote_item_metadata(item: dict[str, Any]) -> bool:
    return bool(caption_role_for_item(item) or footnote_role_for_item(item))


def caption_source_layer(item: dict[str, Any]) -> str:
    source = str(item.get("caption_source_layer") or item.get("footnote_source_layer") or "").strip()
    if source:
        return source
    if item.get("source_content_list_index") is not None or item.get("content_list_text_candidates"):
        return "content_list"
    if item.get("source_middle_indices") or item.get("mineru_middle_block_ids") or item.get("source_block_ids"):
        return "middle"
    return "metadata"


def caption_confidence_for_layer(layer: str) -> str:
    if layer == "middle":
        return "strong_middle_child"
    if layer == "content_list":
        return "strong_content_list_field"
    if layer == "content_list_v2":
        return "strong_v2_field"
    return "medium_metadata"


def _source_ids_for_item(item: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("source_line_ids", "source_block_ids", "caption_source_ids", "footnote_source_ids"):
        value = item.get(key)
        if isinstance(value, list):
            ids.extend(str(part) for part in value if str(part))
        elif value:
            ids.append(str(value))
    if item.get("id"):
        ids.append(str(item["id"]))
    return list(dict.fromkeys(ids))


def apply_caption_footnote_metadata(item: dict[str, Any]) -> None:
    caption_role = caption_role_for_item(item)
    if caption_role:
        layer = str(item.get("caption_source_layer") or caption_source_layer(item))
        caption_type = caption_type_from_role(caption_role, item)
        item["raw_caption_type"] = caption_role
        item["mineru_caption_role"] = caption_role
        item["caption_text"] = caption_text_value(item)
        item["caption_source_layer"] = layer
        item["caption_confidence"] = str(item.get("caption_confidence") or caption_confidence_for_layer(layer))
        item["caption_type"] = caption_type
        item.setdefault("caption_bbox", item.get("bbox"))
        item.setdefault("caption_source_ids", _source_ids_for_item(item))
        if not item.get("caption_parent_float_id"):
            source_blocks = item.get("source_block_ids")
            if isinstance(source_blocks, list) and source_blocks:
                item["caption_parent_float_id"] = str(source_blocks[0])
        if item.get("source_block_ids") and not item.get("caption_body_ids"):
            item["caption_body_ids"] = list(item.get("source_block_ids") or [])

    footnote_role = footnote_role_for_item(item)
    if footnote_role:
        layer = str(item.get("footnote_source_layer") or caption_source_layer(item))
        item["raw_footnote_type"] = footnote_role
        item["mineru_footnote_role"] = footnote_role
        item["footnote_text"] = footnote_text_value(item)
        item["footnote_source_layer"] = layer
        item["footnote_confidence"] = str(item.get("footnote_confidence") or caption_confidence_for_layer(layer))
        item["footnote_type"] = footnote_type_from_role(footnote_role)
        item.setdefault("footnote_bbox", item.get("bbox"))
        item.setdefault("footnote_source_ids", _source_ids_for_item(item))
        if not item.get("footnote_parent_float_id"):
            source_blocks = item.get("source_block_ids")
            if isinstance(source_blocks, list) and source_blocks:
                item["footnote_parent_float_id"] = str(source_blocks[0])
        if item.get("source_block_ids") and not item.get("footnote_body_ids"):
            item["footnote_body_ids"] = list(item.get("source_block_ids") or [])

    body_ids = list(dict.fromkeys(str(part) for part in (item.get("caption_body_ids") or item.get("footnote_body_ids") or item.get("source_block_ids") or []) if str(part)))
    caption_ids = list(dict.fromkeys(str(part) for part in (item.get("caption_source_ids") or item.get("source_line_ids") or []) if str(part)))
    footnote_ids = list(dict.fromkeys(str(part) for part in (item.get("footnote_source_ids") or []) if str(part)))
    if body_ids:
        item.setdefault("body_node_ids", body_ids)
    if caption_ids:
        item.setdefault("caption_node_ids", caption_ids)
    if footnote_ids:
        item.setdefault("footnote_node_ids", footnote_ids)
    parent_id = item.get("caption_parent_float_id") or item.get("footnote_parent_float_id")
    if parent_id:
        item.setdefault("parent_float_source_id", parent_id)
    child_ids = list(dict.fromkeys(body_ids + caption_ids + footnote_ids))
    if child_ids:
        item.setdefault("child_block_ids", child_ids)
    item.setdefault("source_layer_hierarchy", caption_source_layer(item))


def enrich_caption_footnote_provenance_from_content_list(item: dict[str, Any], cache: dict[str, list[Any] | None]) -> None:
    path = item.get("content_list_json")
    index = item.get("source_content_list_index")
    if not path or index is None:
        return
    try:
        source_index = int(index)
    except (TypeError, ValueError):
        return
    raw_items = load_content_items_cached(str(path), cache)
    if raw_items is None or source_index < 0 or source_index >= len(raw_items):
        return
    raw_item = raw_items[source_index]
    if not isinstance(raw_item, dict):
        return
    raw_type = raw_item.get("type")
    if raw_type is not None:
        item.setdefault("content_list_type", raw_type)
    raw_subtype = raw_item.get("sub_type") or raw_item.get("subtype")
    if raw_subtype is not None:
        item.setdefault("raw_sub_type", raw_subtype)
        item.setdefault("mineru_subtype", raw_subtype)
    for key in CAPTION_PROVENANCE_COPY_KEYS + FOOTNOTE_PROVENANCE_COPY_KEYS:
        value = raw_item.get(key)
        if nonempty_value(value):
            item.setdefault(key, value)
    if any(nonempty_value(raw_item.get(key)) for key in CAPTION_ROLE_TO_TYPE):
        item.setdefault("caption_source_layer", "content_list")
    if any(nonempty_value(raw_item.get(key)) for key in FOOTNOTE_ROLE_TO_TYPE):
        item.setdefault("footnote_source_layer", "content_list")


def enrich_reference_provenance_from_content_list(item: dict[str, Any], cache: dict[str, list[Any] | None]) -> None:
    path = item.get("content_list_json")
    index = item.get("source_content_list_index")
    if not path or index is None:
        return
    try:
        source_index = int(index)
    except (TypeError, ValueError):
        return
    raw_items = load_content_items_cached(str(path), cache)
    if raw_items is None or source_index < 0 or source_index >= len(raw_items):
        return
    raw_item = raw_items[source_index]
    if not isinstance(raw_item, dict):
        return
    raw_type = raw_item.get("type")
    raw_subtype = raw_item.get("sub_type") or raw_item.get("subtype")
    if raw_type is not None:
        item.setdefault("content_list_type", raw_type)
    if raw_subtype is not None:
        item.setdefault("raw_sub_type", raw_subtype)
        item.setdefault("mineru_subtype", raw_subtype)
    for key in REFERENCE_PROVENANCE_COPY_KEYS:
        value = raw_item.get(key)
        if nonempty_value(value):
            item.setdefault(key, value)
    if normalize_marker(raw_type) in {"list", "ref_text"} and is_reference_subtype_value(raw_subtype):
        item.setdefault("mineru_reference_role", "ref_text")
        item.setdefault("reference_source_layer", "content_list")
        item.setdefault("reference_confidence", "strong_ref_text_subtype")
    elif REFERENCE_HEADING_RE.match(compact_text(raw_item.get("text"))):
        item.setdefault("mineru_reference_role", "reference_heading")
        item.setdefault("reference_source_layer", "content_list")
        item.setdefault("reference_confidence", "strong_reference_region")


def enrich_page_furniture_provenance_from_content_list(item: dict[str, Any], cache: dict[str, list[Any] | None]) -> None:
    path = item.get("content_list_json")
    index = item.get("source_content_list_index")
    if not path or index is None:
        return
    try:
        source_index = int(index)
    except (TypeError, ValueError):
        return
    raw_items = load_content_items_cached(str(path), cache)
    if raw_items is None or source_index < 0 or source_index >= len(raw_items):
        return
    raw_item = raw_items[source_index]
    if not isinstance(raw_item, dict):
        return
    raw_type = raw_item.get("type")
    raw_subtype = raw_item.get("sub_type") or raw_item.get("subtype")
    if raw_type is not None:
        item.setdefault("content_list_type", raw_type)
    if raw_subtype is not None:
        item.setdefault("raw_sub_type", raw_subtype)
        item.setdefault("mineru_subtype", raw_subtype)
    role = page_furniture_role_from_marker(raw_type)
    if not role:
        return
    for key in PAGE_FURNITURE_PROVENANCE_COPY_KEYS:
        value = raw_item.get(key)
        if nonempty_value(value):
            if key == "text":
                item.setdefault("content_list_text", value)
            else:
                item.setdefault(key, value)
    item.setdefault("raw_page_furniture_type", raw_type)
    item.setdefault("mineru_page_furniture_role", role)
    item.setdefault("page_furniture_source_layer", "content_list")
    item.setdefault("page_furniture_confidence", "strong_content_list_role")
    item.setdefault("page_furniture_bbox", raw_item.get("bbox") or item.get("bbox"))


def load_middle_page_furniture_index_cached(path: str, cache: dict[str, dict[str, list[dict[str, Any]]] | None]) -> dict[str, list[dict[str, Any]]] | None:
    if path in cache:
        return cache[path]
    try:
        payload = read_json(Path(path))
    except Exception:
        cache[path] = None
        return None
    cache[path] = build_middle_page_furniture_index(payload)
    return cache[path]


def build_middle_page_furniture_index(payload: Any) -> dict[str, list[dict[str, Any]]]:
    by_block: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(payload, dict):
        return {"by_block": by_block}
    for page in payload.get("pdf_info") or []:
        if not isinstance(page, dict):
            continue
        page_idx = int(page.get("page_idx") or page.get("page_no") or 0)
        for collection_name, role in (("discarded_blocks", "discarded_block"), ("preproc_blocks", ""), ("para_blocks", "")):
            blocks = page.get(collection_name) or []
            if not isinstance(blocks, list):
                continue
            for block_pos, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                block_index = block.get("index")
                if block_index is None:
                    block_index = block_pos
                block_stub = f"p{page_idx:04d}:m{int(block_index):06d}"
                block_role = role or page_furniture_role_from_marker(block.get("type")) or page_furniture_role_from_marker(block.get("layout_label"))
                if not block_role:
                    continue
                by_block.setdefault(block_stub, []).append(
                    {
                        "role": block_role,
                        "bbox": block.get("bbox"),
                        "text": text_value(block),
                        "page_idx": page_idx,
                        "block_id_suffix": block_stub,
                        "source_layer": "middle",
                        "confidence": "strong_middle_discarded" if block_role == "discarded_block" else "medium_layout_position",
                    }
                )
    return {"by_block": by_block}


def enrich_page_furniture_provenance_from_middle(item: dict[str, Any], cache: dict[str, dict[str, list[dict[str, Any]]] | None]) -> None:
    path = item.get("middle_json")
    if not path:
        return
    index = load_middle_page_furniture_index_cached(str(path), cache)
    if not index:
        return
    by_block = index.get("by_block") or {}
    for source_block_id in item.get("source_block_ids") or []:
        suffix = source_id_suffix(source_block_id)
        entries = by_block.get(suffix) or []
        if not entries:
            continue
        entry = entries[0]
        role = str(entry.get("role") or "")
        item.setdefault("mineru_page_furniture_role", role)
        item.setdefault("raw_page_furniture_type", role)
        item.setdefault("page_furniture_source_layer", "middle")
        item.setdefault("page_furniture_confidence", entry.get("confidence") or "medium_layout_position")
        item.setdefault("page_furniture_bbox", entry.get("bbox") or item.get("bbox"))
        item.setdefault("middle_text", entry.get("text"))
        item.setdefault("page_furniture_source_ids", [str(source_block_id)])
        break


def load_model_labels_cached(path: str, cache: dict[str, list[dict[str, Any]] | None]) -> list[dict[str, Any]] | None:
    if path in cache:
        return cache[path]
    try:
        payload = read_json(Path(path))
    except Exception:
        cache[path] = None
        return None
    cache[path] = flatten_model_labels(payload)
    return cache[path]


def flatten_model_labels(payload: Any) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    pages = payload if isinstance(payload, list) else payload.get("pages") if isinstance(payload, dict) else []
    if not isinstance(pages, list):
        return labels
    for page_pos, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        page_info = page.get("page_info") if isinstance(page.get("page_info"), dict) else {}
        page_idx = int(page_info.get("page_no") if page_info.get("page_no") is not None else page.get("page_idx") or page_pos)
        page_width = float(page_info.get("width") or 0.0)
        page_height = float(page_info.get("height") or 0.0)
        for det_pos, det in enumerate(page.get("layout_dets") or page.get("dets") or []):
            if not isinstance(det, dict):
                continue
            raw_bbox = det.get("bbox")
            scaled_bbox = scale_model_bbox(raw_bbox, page_width=page_width, page_height=page_height)
            labels.append(
                {
                    "page_idx": page_idx,
                    "label": det.get("label"),
                    "score": det.get("score"),
                    "cls_id": det.get("cls_id"),
                    "index": det.get("index", det_pos),
                    "bbox": raw_bbox,
                    "scaled_bbox": scaled_bbox,
                }
            )
    return labels


def scale_model_bbox(value: Any, *, page_width: float, page_height: float) -> list[float] | None:
    box = as_float_bbox(value)
    if not box:
        return None
    if page_width <= 0 or page_height <= 0:
        return box
    return [box[0] / page_width * 612.0, box[1] / page_height * 792.0, box[2] / page_width * 612.0, box[3] / page_height * 792.0]


def as_float_bbox(value: Any) -> list[float] | None:
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


def enrich_model_label_provenance(item: dict[str, Any], cache: dict[str, list[dict[str, Any]] | None]) -> None:
    path = item.get("model_json")
    if not path:
        return
    labels = load_model_labels_cached(str(path), cache)
    if not labels:
        return
    item_page = int(item.get("page_idx") or 0)
    item_bbox = as_float_bbox(item.get("bbox"))
    best: tuple[float, dict[str, Any]] | None = None
    for label in labels:
        if int(label.get("page_idx") or 0) != item_page:
            continue
        score = max(
            bbox_iou(item_bbox, as_float_bbox(label.get("bbox"))),
            bbox_iou(item_bbox, as_float_bbox(label.get("scaled_bbox"))),
        )
        if best is None or score > best[0]:
            best = (score, label)
    if best is None or best[0] < 0.20:
        return
    label = best[1]
    item.setdefault("model_label", label.get("label"))
    item.setdefault("model_score", label.get("score"))
    item.setdefault("model_cls_id", label.get("cls_id"))
    item.setdefault("model_index", label.get("index"))
    item.setdefault("model_bbox", label.get("scaled_bbox") or label.get("bbox"))
    item.setdefault("model_source_layer", "model")
    item.setdefault("model_label_match_iou", round(best[0], 6))


def load_middle_caption_footnote_index_cached(path: str, cache: dict[str, dict[str, list[dict[str, Any]]] | None]) -> dict[str, list[dict[str, Any]]] | None:
    if path in cache:
        return cache[path]
    try:
        payload = read_json(Path(path))
    except Exception:
        cache[path] = None
        return None
    cache[path] = build_middle_caption_footnote_index(payload)
    return cache[path]


def _caption_footnote_entries_from_value(
    *,
    role: str,
    value: Any,
    page_idx: int,
    block_stub: str,
    block_bbox: Any,
    source_layer: str = "middle",
) -> list[dict[str, Any]]:
    parts = value if isinstance(value, list) else [value]
    entries: list[dict[str, Any]] = []
    for idx, part in enumerate(parts):
        text = text_value(part)
        if not text:
            continue
        entries.append(
            {
                "role": role,
                "text": text,
                "bbox": part.get("bbox") if isinstance(part, dict) and part.get("bbox") else block_bbox,
                "page_idx": page_idx,
                "block_id_suffix": block_stub,
                "source_id_suffix": f"{block_stub}:{role}:{idx:04d}",
                "source_layer": source_layer,
            }
        )
    return entries


def build_middle_caption_footnote_index(payload: Any) -> dict[str, list[dict[str, Any]]]:
    by_block: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(payload, dict):
        return {"by_block": by_block}
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
                block_index = block.get("index")
                if block_index is None:
                    block_index = block_pos
                block_stub = f"p{page_idx:04d}:m{int(block_index):06d}"
                entries: list[dict[str, Any]] = []
                for role in tuple(CAPTION_ROLE_TO_TYPE) + tuple(FOOTNOTE_ROLE_TO_TYPE):
                    value = block.get(role)
                    if nonempty_value(value):
                        entries.extend(
                            _caption_footnote_entries_from_value(
                                role=role,
                                value=value,
                                page_idx=page_idx,
                                block_stub=block_stub,
                                block_bbox=block.get("bbox"),
                            )
                        )
                    for child in block.get(role) or [] if isinstance(block.get(role), list) else []:
                        if isinstance(child, dict) and nonempty_value(child.get("lines")):
                            entries.extend(
                                _caption_footnote_entries_from_value(
                                    role=role,
                                    value=" ".join(text_value(line) for line in child.get("lines") or []),
                                    page_idx=page_idx,
                                    block_stub=block_stub,
                                    block_bbox=child.get("bbox") or block.get("bbox"),
                                )
                            )
                for child_key in (
                    "image_caption",
                    "image_footnote",
                    "table_caption",
                    "table_footnote",
                    "chart_caption",
                    "chart_footnote",
                    "code_caption",
                    "code_footnote",
                ):
                    value = block.get(child_key)
                    if nonempty_value(value):
                        entries.extend(
                            _caption_footnote_entries_from_value(
                                role=child_key,
                                value=value,
                                page_idx=page_idx,
                                block_stub=block_stub,
                                block_bbox=block.get("bbox"),
                            )
                        )
                if entries:
                    by_block.setdefault(block_stub, []).extend(entries)
    return {"by_block": by_block}


def enrich_caption_footnote_provenance_from_middle(item: dict[str, Any], cache: dict[str, dict[str, list[dict[str, Any]]] | None]) -> None:
    path = item.get("middle_json")
    if not path:
        return
    index = load_middle_caption_footnote_index_cached(str(path), cache)
    if not index:
        return
    by_block = index.get("by_block") or {}
    for source_block_id in item.get("source_block_ids") or []:
        suffix = source_id_suffix(source_block_id)
        entries = by_block.get(suffix) or []
        if not entries:
            continue
        for entry in entries:
            role = str(entry.get("role") or "")
            text = str(entry.get("text") or "")
            if role in CAPTION_ROLE_TO_TYPE:
                item.setdefault(role, text)
                item.setdefault("caption_source_layer", "middle")
                item.setdefault("caption_parent_float_id", str(source_block_id))
                item.setdefault("caption_bbox", entry.get("bbox"))
                item.setdefault("caption_source_ids", [f"{source_block_id}:{role}"])
            elif role in FOOTNOTE_ROLE_TO_TYPE:
                item.setdefault(role, text)
                item.setdefault("footnote_source_layer", "middle")
                item.setdefault("footnote_parent_float_id", str(source_block_id))
                item.setdefault("footnote_bbox", entry.get("bbox"))
                item.setdefault("footnote_source_ids", [f"{source_block_id}:{role}"])


AFFILIATION_HINT_RE = re.compile(
    r"\b(university|department|dept\.?|institute|school|college|laboratory|lab\b|fordham|ny\s+\d{5})\b",
    re.IGNORECASE,
)


def looks_like_affiliation(text: str) -> bool:
    return bool(AFFILIATION_HINT_RE.search(text))


def looks_like_author_line(text: str) -> bool:
    if not text or "@" in text or looks_like_affiliation(text):
        return False
    if len(text) > 180:
        return False
    pieces = [piece.strip() for piece in re.split(r",|\band\b|&|;", text) if piece.strip()]
    if len(pieces) < 2:
        pieces = [piece for piece in text.split("  ") if piece.strip()]
    return len(pieces) >= 2 and sum(1 for piece in pieces if re.search(r"[A-Z][a-z]+", piece)) >= 2
