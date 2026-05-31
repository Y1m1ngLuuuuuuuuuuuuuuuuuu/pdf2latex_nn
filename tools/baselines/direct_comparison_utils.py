from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from src.evaluation.comparison_structure import (
    ROOT_ID,
    ComparisonBlock,
    ComparisonDocument,
    caption_kind_and_label,
    count_markdown_inline_math,
    extract_cross_refs,
    extract_markdown_citations,
    normalize_caption_for_compare,
    normalize_for_compare,
)


REFERENCE_HEADING_RE = re.compile(r"^(references|bibliography|参考文献)\s*$", re.I)
ABSTRACT_HEADING_RE = re.compile(r"^abstract\s*$", re.I)
CAPTION_PREFIX_RE = re.compile(
    r"^\s*(fig(?:ure)?|table|alg(?:orithm)?)\s*\.?\s+"
    r"(?:S?\d+(?:\.\d+)*(?:\([A-Za-z]\))?|[IVXLCDM]+|[A-Za-z]\d+(?:\.\d+)*)?"
    r"\s*[:.\-–—]\s+\S",
    re.I,
)
TAG_RE = re.compile(r"<[^>]+>")


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    return " ".join(text.split())


def flatten_content(value: Any) -> str:
    """Flatten MinerU/content-list fragments without interpreting them as LaTeX."""

    if value is None:
        return ""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(part for part in (flatten_content(item) for item in value) if part)
    if not isinstance(value, dict):
        return clean_text(value)

    raw_type = str(value.get("type") or value.get("sub_type") or "").casefold()
    if raw_type in {
        "equation_inline",
        "inline_equation",
        "inline_formula",
        "formula_inline",
    }:
        return "[MATH]"
    if raw_type in {
        "equation_interline",
        "interline_equation",
        "display_formula",
        "formula",
    }:
        return "[DISPLAY_MATH]"

    for key in ("text", "content", "latex", "html", "md", "markdown"):
        if key in value:
            flattened = flatten_content(value.get(key))
            if flattened:
                return flattened

    parts: list[str] = []
    for key, child in value.items():
        if key in {"bbox", "page_idx", "page", "index", "id"}:
            continue
        if key.endswith("content") or key in {"spans", "lines", "items", "cells", "rows"}:
            part = flatten_content(child)
            if part:
                parts.append(part)
    return " ".join(parts)


def flatten_mineru_lines(lines: Any) -> str:
    if not isinstance(lines, list):
        return flatten_content(lines)
    parts: list[str] = []
    for line in lines:
        if isinstance(line, dict) and isinstance(line.get("spans"), list):
            line_text = flatten_content(line.get("spans"))
        else:
            line_text = flatten_content(line)
        if line_text:
            parts.append(line_text)
    return " ".join(parts)


def doc_id_from_path(path: Path) -> str:
    name = path.stem
    match = re.search(r"(\d{4}\.\d{4,5})", str(path))
    return match.group(1) if match else name


class DirectComparisonBuilder:
    def __init__(self, *, doc_id: str, source_format: str, source_path: Path | None = None) -> None:
        self.doc_id = doc_id
        self.source_format = source_format
        self.source_path = str(source_path) if source_path else None
        self.blocks: list[ComparisonBlock] = []
        self.section_stack: dict[int, str] = {}
        self.next_id = 1
        self.seen_document_title = False
        self.in_references = False
        self.in_abstract = False
        self.active_abstract_id: str | None = None
        self.last_float_by_page: dict[int, ComparisonBlock] = {}

    def document(self, *, metadata: dict[str, Any] | None = None) -> ComparisonDocument:
        return ComparisonDocument(
            doc_id=self.doc_id,
            source_format=self.source_format,
            source_path=self.source_path,
            blocks=self.blocks,
            metadata=metadata or {},
        )

    def add_heading(self, text: str, level: int = 1, *, metadata: dict[str, Any] | None = None) -> ComparisonBlock | None:
        text = clean_text(text)
        if not text:
            return None
        if ABSTRACT_HEADING_RE.match(text):
            block = self.add_block("abstract", "Abstract", level=1, parent_id=ROOT_ID, marker="abstract_heading", metadata=metadata)
            self.in_abstract = True
            self.in_references = False
            self.active_abstract_id = block.block_id
            return block
        block = self.add_block("heading", text, level=max(1, min(level, 5)), parent_id=self.section_parent(level), metadata=metadata)
        self.section_stack = {key: value for key, value in self.section_stack.items() if key < level}
        self.section_stack[level] = block.block_id
        self.in_references = bool(REFERENCE_HEADING_RE.match(text))
        self.in_abstract = False
        self.active_abstract_id = None
        return block

    def add_textual(self, text: str, *, metadata: dict[str, Any] | None = None) -> ComparisonBlock | None:
        text = clean_text(text)
        if not text:
            return None
        if REFERENCE_HEADING_RE.match(text):
            return self.add_heading(text, 1, metadata=metadata)
        if ABSTRACT_HEADING_RE.match(text):
            return self.add_heading(text, 1, metadata=metadata)
        if self.in_references:
            return self.add_block("reference_item", text, metadata=metadata)
        if self.in_abstract:
            return self.add_block("abstract", text, parent_id=self.active_abstract_id or self.current_parent_id(), metadata=metadata)
        return self.add_block("paragraph", text, metadata=metadata)

    def add_float(self, block_type: str, text: str = "", *, page_idx: int = -1, marker: str | None = None, metadata: dict[str, Any] | None = None) -> ComparisonBlock:
        block = self.add_block(block_type, text, marker=marker, metadata=metadata)
        if page_idx >= 0:
            self.last_float_by_page[page_idx] = block
        return block

    def add_caption(self, text: str, *, page_idx: int = -1, metadata: dict[str, Any] | None = None) -> ComparisonBlock | None:
        text = clean_text(text)
        if not text:
            return None
        parent = self.last_float_by_page.get(page_idx) or self.last_float_by_page.get(-1)
        parent_type = parent.block_type if parent else None
        kind, label = caption_kind_and_label(text, parent_type)
        parent_id = parent.block_id if parent and (kind is None or parent.block_type == kind) else None
        return self.add_block("caption", text, parent_id=parent_id, marker=kind, label=label, metadata=metadata)

    def add_block(
        self,
        block_type: str,
        text: str,
        *,
        level: int | None = None,
        parent_id: str | None = None,
        marker: str | None = None,
        label: str | None = None,
        display_math_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ComparisonBlock:
        block_id = f"B_{self.next_id:06d}"
        self.next_id += 1
        clean_value = clean_text(text)
        parent = self.current_parent_id() if parent_id is None else parent_id
        normalized = normalize_caption_for_compare(clean_value, marker) if block_type == "caption" else normalize_for_compare(clean_value)
        block = ComparisonBlock(
            block_id=block_id,
            block_type=block_type,
            order=len(self.blocks),
            text=clean_value,
            normalized_text=normalized,
            level=level,
            parent_id=None if parent == ROOT_ID else parent,
            marker=marker,
            label=label,
            citations=extract_markdown_citations(clean_value),
            cross_refs=extract_cross_refs(clean_value),
            inline_math_count=count_markdown_inline_math(clean_value),
            display_math_count=display_math_count,
            metadata=metadata or {},
        )
        self.blocks.append(block)
        return block

    def maybe_document_title(self, text: str, *, level: int | None = None, metadata: dict[str, Any] | None = None) -> bool:
        text = clean_text(text)
        if not text or self.seen_document_title or self.blocks:
            return False
        if level not in {None, 0, 1}:
            return False
        if REFERENCE_HEADING_RE.match(text) or ABSTRACT_HEADING_RE.match(text):
            return False
        self.add_block("document_title", text, parent_id=ROOT_ID, marker="direct_title", metadata=metadata)
        self.seen_document_title = True
        return True

    def current_parent_id(self) -> str:
        if self.section_stack:
            return self.section_stack[max(self.section_stack)]
        return ROOT_ID

    def section_parent(self, level: int) -> str:
        lower = [key for key in self.section_stack if key < level]
        return self.section_stack[max(lower)] if lower else ROOT_ID


def source_metadata(raw_type: str, page_idx: Any = None, bbox: Any = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source_type": raw_type}
    if page_idx is not None:
        metadata["page_idx"] = page_idx
    if bbox is not None:
        metadata["bbox"] = bbox
    if extra:
        metadata.update(extra)
    return metadata


def looks_like_caption(text: str) -> bool:
    return bool(CAPTION_PREFIX_RE.match(str(text or "")))
