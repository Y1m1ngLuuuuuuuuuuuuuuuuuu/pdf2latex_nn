#!/usr/bin/env python3
"""Diagnose v8 float/caption candidates and experimental layout sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR
from src.reasoning.float_caption_layout import build_float_caption_layout_sidecars
from src.reasoning.float_caption_matcher import (
    caption_candidates_from_document,
    caption_evidence_contexts_from_document,
    float_candidates_from_document,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-ir", help="Path to a DocumentIR JSON payload.")
    parser.add_argument("--content-json", help="Path to a v8 content-list-like JSON payload.")
    parser.add_argument("--output", required=True, help="Output diagnostic JSON path.")
    args = parser.parse_args()

    document = load_document_ir(Path(args.document_ir or args.content_json))
    sidecars = build_float_caption_layout_sidecars(document)
    output = {
        "doc_id": document.doc_id,
        "v8_only": True,
        "legacy_field_names_are_provenance_only": True,
        "caption_candidate_count": len(caption_candidates_from_document(document)),
        "caption_evidence_context_count": len(caption_evidence_contexts_from_document(document)),
        "mineru_backed_caption_count": sum(
            1
            for context in caption_evidence_contexts_from_document(document)
            if context.context_kind == "caption" and context.confidence_tier == "high"
        ),
        "regex_only_caption_count": sum(
            1
            for context in caption_evidence_contexts_from_document(document)
            if context.context_kind == "caption_like_diagnostic" and context.evidence_source == "regex_only"
        ),
        "float_candidate_count": len(float_candidates_from_document(document)),
        **sidecars.to_diagnostic(),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def load_document_ir(path: Path) -> DocumentIR:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "nodes" in payload and "pages" in payload:
        return _document_ir_from_payload(payload)
    if isinstance(payload, list):
        return _document_ir_from_content_list(path.stem, payload)
    if isinstance(payload, dict) and isinstance(payload.get("content"), list):
        return _document_ir_from_content_list(path.stem, payload["content"])
    raise ValueError(f"Unsupported v8/DocumentIR payload: {path}")


def _document_ir_from_payload(payload: dict[str, Any]) -> DocumentIR:
    pages = [
        PageIR(
            page_idx=int(page.get("page_idx", page.get("page", 0))),
            width=float(page.get("width", 1000)),
            height=float(page.get("height", 1000)),
            node_ids=list(page.get("node_ids") or []),
        )
        for page in payload.get("pages", [])
    ]
    nodes = [_node_from_payload(node, index) for index, node in enumerate(payload.get("nodes", []))]
    return DocumentIR(
        doc_id=str(payload.get("doc_id") or payload.get("id") or "document"),
        pages=pages or [PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        source_pdf=payload.get("source_pdf"),
        reading_order=list(payload.get("reading_order") or [node.node_id for node in nodes]),
        provenance=dict(payload.get("provenance") or {}),
        metadata=dict(payload.get("metadata") or {}),
    )


def _document_ir_from_content_list(doc_id: str, records: list[dict[str, Any]]) -> DocumentIR:
    nodes = [_node_from_payload(record, index) for index, record in enumerate(records)]
    page_ids: dict[int, list[str]] = {}
    for node in nodes:
        page_ids.setdefault(node.page_idx, []).append(node.node_id)
    pages = [
        PageIR(
            page_idx=page_idx,
            width=1000,
            height=1000,
            node_ids=node_ids,
        )
        for page_idx, node_ids in sorted(page_ids.items())
    ]
    return DocumentIR(doc_id=doc_id, pages=pages, nodes=nodes, reading_order=[node.node_id for node in nodes])


def _node_from_payload(payload: dict[str, Any], index: int) -> DocumentNode:
    node_type = _block_type(payload.get("node_type") or payload.get("type") or payload.get("block_type"))
    node_id = str(payload.get("node_id") or payload.get("id") or payload.get("block_id") or f"n{index}")
    page_idx = int(payload.get("page_idx", payload.get("page", 0)) or 0)
    bbox = payload.get("bbox") or payload.get("bboxes")
    boxes = []
    if isinstance(bbox, list) and len(bbox) == 4 and not isinstance(bbox[0], list):
        boxes = [BBox.from_list(bbox)]
    elif isinstance(bbox, list):
        for item in bbox:
            if isinstance(item, list) and len(item) == 4:
                boxes.append(BBox.from_list(item))
    return DocumentNode(
        node_id=node_id,
        node_type=node_type,
        text=str(payload.get("text") or payload.get("content") or ""),
        page_idx=page_idx,
        bboxes=boxes,
        reading_index=int(payload.get("reading_index", payload.get("index", index)) or index),
        raw_type=str(payload.get("raw_type") or payload.get("type") or "") or None,
        metadata=dict(payload.get("metadata") or {key: value for key, value in payload.items() if key not in {"text", "content"}}),
    )


def _block_type(value: Any) -> BlockType:
    normalized = str(value or "").casefold()
    if "figure" in normalized or "image" in normalized:
        return BlockType.FIGURE
    if "table" in normalized:
        return BlockType.TABLE
    if "algorithm" in normalized:
        return BlockType.ALGORITHM
    if "caption" in normalized:
        return BlockType.TEXT
    try:
        return BlockType(normalized)
    except ValueError:
        return BlockType.TEXT


if __name__ == "__main__":
    main()
