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
    for index, item in enumerate(payload.get("items") or []):
        if not isinstance(item, dict):
            continue
        record = dict(item)
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
    if raw == "interline_equation":
        return "equation_interline"
    if raw == "chart":
        return "figure"
    if raw == "image":
        return "figure"
    if raw == "code" and has_algorithm_caption(item):
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
