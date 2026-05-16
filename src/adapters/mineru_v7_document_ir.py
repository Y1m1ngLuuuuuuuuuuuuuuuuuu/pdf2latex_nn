"""Adapter from MinerU v7 styled content JSON to stable DocumentIR.

The v7 content JSON is a PDF-frontend implementation detail.  This adapter is
the single boundary where v7 field names are translated into the stable
DocumentIR contract consumed by style extraction, citation repair, decoding and
rendering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.ir import (
    BBox,
    BlockType,
    CoordinateSpace,
    DocumentIR,
    DocumentNode,
    PageIR,
    SourceRef,
    StyleSpan,
)
from src.ir.serialization import write_json
from src.ir.validators import validate_document_ir
from src.generation.table_assets import annotate_table_group_records
from src.perception.reading_order import annotate_duplicate_contained_continuations, is_duplicate_shadow_record
from src.perception.title_features import title_numbering_info
from src.pipeline.v7_contract import assert_v7_content_json, read_json_payload


PAGE_FOOTNOTE_TYPES = {"page_footnote", "footnote", "foot_note", "image_footnote", "table_footnote", "chart_footnote"}
MARGIN_NOTE_TYPES = {"aside_text", "page_aside_text", "margin_note", "marginnote", "side_note", "sidenote", "sidebar"}
HEADER_FOOTER_TYPES = {"page_number", "header", "footer", "page_header", "page_footer"}
TOC_TYPES = {"toc", "index", "table_of_contents"}
TEXT_TYPES = {"paragraph", "text", "paragraph_text", "phonetic"}
TITLE_TYPES = {"title", "section", "subsection", "subsubsection", "heading"}
DISPLAY_EQUATION_TYPES = {"equation", "equation_interline", "interline_equation", "display_formula", "formula"}
INLINE_MATH_TYPES = {"inline_math", "inline_formula", "math_inline", "equation_inline", "inline_equation"}
TABLE_TYPES = {"table"}
FIGURE_TYPES = {"figure", "image", "chart", "seal"}
ALGORITHM_TYPES = {"algorithm"}
LIST_TYPES = {"list", "item", "itemize", "enumerate"}
CODE_TYPES = {"code"}
REFERENCE_TYPES = {"reference", "references", "bibliography", "ref_text"}
CAPTION_TYPES_BY_BLOCK = {
    "image_caption": BlockType.FIGURE,
    "figure_caption": BlockType.FIGURE,
    "chart_caption": BlockType.FIGURE,
    "table_caption": BlockType.TABLE,
    "code_caption": BlockType.CODE,
    "algorithm_caption": BlockType.ALGORITHM,
}
RAW_TYPE_KEYS = ("canonical_type", "type", "raw_type", "category", "block_type")


@dataclass(frozen=True)
class MinerUV7DocumentIRAdapterConfig:
    require_styles: bool = True
    coordinate_space: CoordinateSpace = CoordinateSpace.PAGE_NORMALIZED_1000
    default_page_width: float = 1000.0
    default_page_height: float = 1000.0
    include_raw_block: bool = True
    include_raw_item: bool = False
    extractor_name: str = "mineru_v7"


class MinerUV7DocumentIRAdapter:
    """Convert one v7 content payload into DocumentIR."""

    def __init__(self, config: MinerUV7DocumentIRAdapterConfig | None = None) -> None:
        self.config = config or MinerUV7DocumentIRAdapterConfig()

    def load(
        self,
        content_json_path: Path,
        *,
        pdf_path: Path | None = None,
        doc_id: str | None = None,
    ) -> DocumentIR:
        payload = assert_v7_content_json(Path(content_json_path), require_styles=self.config.require_styles)
        return self.convert_payload(
            payload,
            source_path=Path(content_json_path),
            pdf_path=pdf_path,
            doc_id=doc_id,
        )

    def convert_payload(
        self,
        payload: dict[str, Any],
        *,
        source_path: Path | None = None,
        pdf_path: Path | None = None,
        doc_id: str | None = None,
    ) -> DocumentIR:
        items = annotate_duplicate_contained_continuations([item for item in payload.get("items", []) if isinstance(item, dict)])
        items = [item for item in items if not is_duplicate_shadow_record(item)]
        items = annotate_table_group_records(items)
        stable_doc_id = doc_id or infer_doc_id(payload, source_path, pdf_path)
        node_id_map: dict[int, str] = {}
        used_node_ids: set[str] = set()
        nodes: list[DocumentNode] = []

        for position, item in enumerate(items):
            base_node_id = stable_node_id(item, fallback_position=position)
            node_id = dedupe_node_id(base_node_id, used_node_ids)
            node_id_map[id(item)] = node_id
            used_node_ids.add(node_id)
            node = self._convert_item(
                item,
                node_id=node_id,
                position=position,
                source_path=source_path,
                pdf_path=pdf_path,
                schema_version=str(payload.get("schema_version") or ""),
            )
            nodes.append(node)

        nodes = annotate_table_group_nodes(nodes)

        nodes = sorted(nodes, key=lambda node: (node.reading_index, node.page_idx, node.node_id))
        reading_order = [node.node_id for node in nodes]
        pages = build_pages(nodes, self.config.default_page_width, self.config.default_page_height, self.config.coordinate_space)

        document = DocumentIR(
            doc_id=stable_doc_id,
            source_pdf=str(pdf_path) if pdf_path is not None else payload.get("style_source_pdf"),
            pages=pages,
            nodes=nodes,
            coordinate_space=self.config.coordinate_space,
            reading_order=reading_order,
            provenance={
                "adapter": "MinerUV7DocumentIRAdapter",
                "adapter_version": "v1",
                "source_path": str(source_path) if source_path is not None else None,
                "source_schema_version": payload.get("schema_version"),
                "source_format": payload.get("source_format"),
                "style_source_pdf": payload.get("style_source_pdf"),
            },
            metadata={
                "v7_config": payload.get("config"),
                "style_config": payload.get("style_config"),
                "item_count": len(items),
            },
        )
        validate_document_ir(document)
        return document

    def _convert_item(
        self,
        item: dict[str, Any],
        *,
        node_id: str,
        position: int,
        source_path: Path | None,
        pdf_path: Path | None,
        schema_version: str,
    ) -> DocumentNode:
        page_idx = int_value(item.get("page_idx"), 0)
        bboxes = [BBox.from_list(list(chunk)) for chunk in iter_bbox_chunks(item.get("bbox"))]
        if not bboxes:
            bboxes = [BBox(0.0, 0.0, 0.0, 0.0)]
        node_type = map_v7_type_to_block_type(item)
        reading_index = int_value(
            item.get("global_order"),
            int_value(item.get("layout_flow_order"), int_value(item.get("column_fix_global_order"), position)),
        )
        text, text_repairs = clean_node_text_ocr_debris(text_from_v7_item(item))
        metadata = metadata_from_v7_item(item, include_raw_block=self.config.include_raw_block, include_raw_item=self.config.include_raw_item)
        if text_repairs:
            metadata["ocr_text_repairs"] = text_repairs
            metadata["original_ocr_text"] = text_from_v7_item(item)
        if source_path is not None:
            metadata.setdefault("source_json", str(source_path))
            metadata.setdefault("source_json_dir", str(source_path.parent))
            metadata.setdefault("asset_base_dir", str(source_path.parent))
        if pdf_path is not None:
            metadata.setdefault("source_pdf", str(pdf_path))
        features = features_from_v7_item(item)
        numbering = title_numbering_info(text)
        if bool(numbering.get("has_numbering")):
            features["title_numbering_level"] = int(numbering["level"]) if numbering.get("level") is not None else None
            features["title_numbering_style"] = str(numbering.get("style") or "none")
            features["title_numbering_token"] = str(numbering.get("token") or "")
            metadata["title_numbering"] = {
                **numbering,
                "path": list(numbering.get("path") or ()),
            }
        spans = [style_span_from_v7(span) for span in item.get("style_spans", []) if isinstance(span, dict)]
        spans, span_repairs = clean_style_spans_for_node(spans, node_text=text)
        if span_repairs:
            metadata["style_span_repairs"] = span_repairs
            metadata["noisy_style_span_repaired"] = True
        source_refs = [
            SourceRef(
                path=str(source_path) if source_path is not None else None,
                page_idx=page_idx,
                extractor=self.config.extractor_name,
                version=schema_version,
                metadata={
                    "pdf_path": str(pdf_path) if pdf_path is not None else None,
                    "mineru_page_idx": item.get("mineru_page_idx"),
                    "mineru_block_idx": item.get("mineru_block_idx"),
                },
            )
        ]

        return DocumentNode(
            node_id=node_id,
            node_type=node_type,
            text=text,
            page_idx=page_idx,
            bboxes=bboxes,
            reading_index=reading_index,
            raw_type=str(item.get("raw_type") or item.get("type") or ""),
            list_type=str(item.get("list_type")) if item.get("list_type") is not None else None,
            spans=spans,
            flags=flags_from_v7_item(item),
            features=features,
            source_refs=source_refs,
            metadata=metadata,
        )


def load_v7_document_ir(
    content_json_path: Path,
    *,
    pdf_path: Path | None = None,
    doc_id: str | None = None,
    config: MinerUV7DocumentIRAdapterConfig | None = None,
) -> DocumentIR:
    return MinerUV7DocumentIRAdapter(config).load(content_json_path, pdf_path=pdf_path, doc_id=doc_id)


def convert_v7_payload_to_document_ir(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
    pdf_path: Path | None = None,
    doc_id: str | None = None,
    config: MinerUV7DocumentIRAdapterConfig | None = None,
) -> DocumentIR:
    return MinerUV7DocumentIRAdapter(config).convert_payload(payload, source_path=source_path, pdf_path=pdf_path, doc_id=doc_id)


def write_v7_document_ir(
    content_json_path: Path,
    output_path: Path,
    *,
    pdf_path: Path | None = None,
    doc_id: str | None = None,
    config: MinerUV7DocumentIRAdapterConfig | None = None,
) -> DocumentIR:
    document = load_v7_document_ir(content_json_path, pdf_path=pdf_path, doc_id=doc_id, config=config)
    write_json(output_path, document)
    return document


def infer_doc_id(payload: dict[str, Any], source_path: Path | None, pdf_path: Path | None) -> str:
    for key in ("doc_id", "document_id", "paper_id", "arxiv_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if pdf_path is not None:
        return Path(pdf_path).stem
    if source_path is not None:
        name = Path(source_path).name
        for suffix in ("_content_list_v7_styles.json", "_content_list_v7.json", ".json"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return Path(source_path).stem
    return "document"


def stable_node_id(item: dict[str, Any], *, fallback_position: int) -> str:
    for key in ("node_id", "id", "block_id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return sanitize_node_id(value.strip())
        if isinstance(value, int):
            return f"v7_{value:06d}"
    order = int_value(item.get("global_order"), fallback_position)
    page = int_value(item.get("page_idx"), 0)
    mineru_block = int_value(item.get("mineru_block_idx"), int_value(item.get("original_index"), -1))
    if mineru_block >= 0:
        return f"v7_p{page:04d}_b{mineru_block:06d}"
    return f"v7_{order:06d}"


def sanitize_node_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)
    return cleaned or "node"


def dedupe_node_id(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    suffix = 2
    while f"{base}_{suffix}" in used:
        suffix += 1
    return f"{base}_{suffix}"


def build_pages(
    nodes: list[DocumentNode],
    default_width: float,
    default_height: float,
    coordinate_space: CoordinateSpace,
) -> list[PageIR]:
    max_page = max((node.page_idx for node in nodes), default=0)
    node_ids_by_page: dict[int, list[str]] = {page_idx: [] for page_idx in range(max_page + 1)}
    for node in sorted(nodes, key=lambda item: (item.page_idx, item.reading_index)):
        node_ids_by_page.setdefault(node.page_idx, []).append(node.node_id)
    return [
        PageIR(
            page_idx=page_idx,
            width=default_width,
            height=default_height,
            node_ids=node_ids_by_page.get(page_idx, []),
            coordinate_space=coordinate_space,
        )
        for page_idx in range(max_page + 1)
    ]


def map_v7_type_to_block_type(item: dict[str, Any]) -> BlockType:
    if str(item.get("list_type") or "").casefold() == "reference_list":
        return BlockType.REFERENCE
    layer = str(item.get("layout_layer") or "").casefold()
    role = str(item.get("layout_role") or "").casefold()
    raw = raw_v7_type(item)
    if raw in CAPTION_TYPES_BY_BLOCK:
        return CAPTION_TYPES_BY_BLOCK[raw]
    if raw in PAGE_FOOTNOTE_TYPES or role in {"footnote", "page_footnote", "image_footnote", "table_footnote", "chart_footnote"}:
        return BlockType.FOOTNOTE
    if raw in MARGIN_NOTE_TYPES or role in {"margin_note", "marginnote", "side_note", "sidenote", "aside_text", "page_aside_text"}:
        return BlockType.MARGIN_NOTE
    if layer == "noise_layer" or raw in HEADER_FOOTER_TYPES or role == "noise":
        return BlockType.HEADER_FOOTER
    if raw in TOC_TYPES or role in {"toc_title", "toc_entry"}:
        return BlockType.TOC
    if raw in TEXT_TYPES:
        return BlockType.TEXT
    if raw in TITLE_TYPES:
        return BlockType.TITLE
    if raw in DISPLAY_EQUATION_TYPES:
        return BlockType.EQUATION
    if raw in INLINE_MATH_TYPES:
        return BlockType.INLINE_MATH
    if raw in TABLE_TYPES:
        return BlockType.TABLE
    if raw in FIGURE_TYPES:
        return BlockType.FIGURE
    if raw in ALGORITHM_TYPES:
        return BlockType.ALGORITHM
    if raw in LIST_TYPES:
        return BlockType.LIST
    if raw in CODE_TYPES:
        return BlockType.CODE
    if raw in REFERENCE_TYPES:
        return BlockType.REFERENCE
    return BlockType.OTHER if not text_from_v7_item(item) else BlockType.TEXT


def raw_v7_type(item: dict[str, Any]) -> str:
    for key in RAW_TYPE_KEYS:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).casefold().strip()
    return ""


def text_from_v7_item(item: dict[str, Any]) -> str:
    if str(item.get("canonical_type") or item.get("type") or item.get("raw_type") or "").casefold() == "table":
        caption = item.get("table_group_caption") or item.get("table_caption")
        if isinstance(caption, list):
            caption_text = " ".join(str(part).strip() for part in caption if str(part).strip()).strip()
        else:
            caption_text = str(caption or "").strip()
        # Keep table cell OCR out of the semantic text surface.  The raw body is
        # still preserved in metadata for crop/table reconstruction, but the IR
        # node text should represent the table as a visual object plus caption.
        if caption_text:
            return caption_text
        for key in ("text", "text_for_embedding", "text_preview"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                extracted = table_caption_from_text(value)
                if extracted:
                    return extracted
        block = item.get("block")
        if isinstance(block, dict):
            extracted = table_caption_from_text(text_from_block(block))
            if extracted:
                return extracted
        return ""
    for key in ("text", "text_for_embedding", "text_preview"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    block = item.get("block")
    if isinstance(block, dict):
        return text_from_block(block)
    return ""


def text_from_block(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(text_from_content_segment(segment) for segment in content)
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        for key in ("paragraph_content", "title_content", "content"):
            value = content.get(key)
            if isinstance(value, list):
                return "".join(text_from_content_segment(segment) for segment in value)
    return ""


TABLE_CAPTION_RE = re.compile(
    r"^\s*(?P<caption>Table\s+(?:[0-9]+|[IVXLCDM]+|[A-Z])(?:\.[0-9A-Za-z]+)*\s*[:.]\s+.+)",
    re.IGNORECASE,
)


def table_caption_from_text(text: str) -> str:
    value = " ".join(str(text or "").replace("\r", "\n").split())
    if not value:
        return ""
    match = TABLE_CAPTION_RE.match(value)
    if not match:
        return ""
    caption = clean_table_caption_markup(match.group("caption").strip())
    # OCR table bodies are often appended after the caption.  Retain enough text
    # for numbering and a useful caption, but avoid dumping full table cells into
    # the semantic surface.
    return safe_table_caption_excerpt(caption)


def clean_table_caption_markup(text: str) -> str:
    value = str(text or "")
    # MinerU can emit small OCR math fragments such as ``\mathrm { N }`` inside
    # table notes.  A later hard cut can otherwise leave ``\mathrm {`` dangling
    # inside ``\caption{...}``.  Captions only need readable text and numbering,
    # so unwrap simple LaTeX font/math commands before any truncation.
    previous = None
    while previous != value:
        previous = value
        value = re.sub(r"\\[A-Za-z]+\s*\{\s*([^{}]*?)\s*\}", r"\1", value)
    value = re.sub(r"\\[A-Za-z]+", " ", value)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def safe_table_caption_excerpt(text: str, *, limit: int = 360) -> str:
    value = " ".join(str(text or "").split()).strip()
    if len(value) <= limit:
        return value.rstrip(" ,;:")
    sentence_ends = [match.end() for match in re.finditer(r"[.!?]\s+", value)]
    usable = [end for end in sentence_ends if 80 <= end <= limit]
    if usable:
        return value[: usable[-1]].rstrip(" ,;:")
    cut = value.rfind(" ", 0, limit)
    if cut < 80:
        cut = limit
    return value[:cut].rstrip(" ,;:")


def text_from_content_segment(segment: Any) -> str:
    if isinstance(segment, str):
        return segment
    if isinstance(segment, dict):
        for key in ("content", "text"):
            value = segment.get(key)
            if isinstance(value, str):
                return value
    return ""


def iter_bbox_chunks(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, list) or len(value) < 4:
        return []
    chunks: list[tuple[float, float, float, float]] = []
    usable_len = len(value) - (len(value) % 4)
    for index in range(0, usable_len, 4):
        chunk = value[index : index + 4]
        try:
            chunks.append((float(chunk[0]), float(chunk[1]), float(chunk[2]), float(chunk[3])))
        except (TypeError, ValueError):
            continue
    return chunks


def style_span_from_v7(span: dict[str, Any]) -> StyleSpan:
    bbox = None
    chunks = iter_bbox_chunks(span.get("bbox"))
    if chunks:
        bbox = BBox.from_list(list(chunks[0]))
    return StyleSpan(
        text=str(span.get("text") or ""),
        font_name=str(span.get("font_name")) if span.get("font_name") is not None else None,
        font_size=float_value(span.get("font_size")),
        is_bold=bool(span.get("is_bold")),
        is_italic=bool(span.get("is_italic")),
        is_inline_math=bool(span.get("is_inline_math")),
        is_inline_code=bool(span.get("is_inline_code")),
        bbox=bbox,
    )


SPLIT_LETTER_NOISE_PREFIX_RE = re.compile(r"^\s*(?:[a-z]\s+){1,8}(?=[A-Z0-9(⋆\u2022\u25E6])")
PUNCT_SPLIT_LETTER_NUMBERING_PREFIX_RE = re.compile(
    r"^\s*[,.;:(){}\[\]\\/\-–—]*\s*(?:[a-z]\s*){1,10}[\s,.;:(){}\[\]\\/\-–—]*(?=(?:[IVXLCDM]+|[A-Z]|\d+(?:\.\d+)*)\.?\s*)",
    re.IGNORECASE,
)
SPLIT_LETTER_DEBRIS_RE = re.compile(r"^\s*(?:[a-z]\s*){1,12}\s*$")
SPLIT_LETTER_WITH_SPACES_RE = re.compile(r"^\s*(?:[A-Za-z]\s+){1,12}[A-Za-z]?\s*$")
LOWERCASE_PUNCT_DEBRIS_RE = re.compile(r"^[a-z]{1,4}\W+$")
NODE_SPLIT_LETTER_NOISE_PREFIX_RE = re.compile(r"^\s*(?P<prefix>(?:[a-z]\s+){1,8})(?P<body>[A-Z0-9(⋆\u2022\u25E6].*)$", re.DOTALL)
NODE_TRAILING_SPLIT_LETTER_NOISE_RE = re.compile(
    r"^(?P<body>.*[.!?;:])\s*(?P<suffix>(?:[a-z]\s*){1,4})\s*$",
    re.DOTALL,
)


def clean_node_text_ocr_debris(text: str) -> tuple[str, list[dict[str, str]]]:
    """Remove only high-confidence OCR debris from a MinerU node text field.

    The canonical MinerU text is usually cleaner than PyMuPDF spans, so this
    function is intentionally narrow: strip split lowercase prefixes before a
    normal sentence/list marker and split lowercase tails after a completed
    sentence.  Full raw text is preserved in metadata by the caller.
    """

    value = str(text or "")
    repairs: list[dict[str, str]] = []
    match = NODE_SPLIT_LETTER_NOISE_PREFIX_RE.match(value)
    if match and len(match.group("body").strip()) >= 20:
        removed = match.group("prefix").strip()
        value = match.group("body").lstrip()
        repairs.append({"kind": "node_split_letter_ocr_prefix", "removed": removed})
    match = NODE_TRAILING_SPLIT_LETTER_NOISE_RE.match(value)
    if match and len(match.group("body").strip()) >= 40:
        suffix = match.group("suffix").strip()
        # Do not strip common one-letter sentence endings in math prose unless
        # the suffix is visually separated from the terminal punctuation.
        if suffix and _is_split_letter_debris(suffix):
            value = match.group("body").rstrip()
            repairs.append({"kind": "node_split_letter_ocr_suffix", "removed": suffix})
    return value, repairs


def strip_split_letter_ocr_noise_prefix(text: str, *, node_text: str = "") -> tuple[str, str | None]:
    """Remove short split-letter OCR debris prepended to otherwise clean text.

    PyMuPDF spans occasionally include visual leftovers such as ``"y p p p g
    Although ..."`` while the MinerU block text is already clean
    (``"Although ..."``).  Keep the node and style span, but drop the prefix so
    the renderer does not faithfully print OCR debris.
    """

    value = str(text or "")
    aligned = _strip_prefix_by_node_alignment(value, node_text=node_text)
    if aligned is not None:
        stripped, removed = aligned
        return stripped, removed
    match = SPLIT_LETTER_NOISE_PREFIX_RE.match(value)
    if not match:
        match = PUNCT_SPLIT_LETTER_NUMBERING_PREFIX_RE.match(value)
        if not match:
            return value, None
    prefix = match.group(0)
    stripped = value[match.end() :]
    normalized_node = _compact_alignment_text(node_text)
    normalized_stripped = _compact_alignment_text(stripped)
    if normalized_node and not normalized_node.startswith(normalized_stripped[: min(len(normalized_stripped), 80)]):
        return value, None
    return stripped.lstrip(), prefix.strip()


def _strip_prefix_by_node_alignment(text: str, *, node_text: str = "") -> tuple[str, str] | None:
    """Drop short OCR prefixes when the rest of the span aligns to node text.

    MinerU's block text is often clean while PyMuPDF spans include visual
    leftovers from math/formula glyphs, e.g. ``, pp y g p Another ...`` or
    ``) g We can ...``.  Regexes alone are brittle here, so use the clean node
    text as an anchor and only remove a prefix when the remaining compact text
    starts exactly like the node.
    """

    value = str(text or "")
    compact_node = _compact_alignment_text(node_text)
    compact_value = _compact_alignment_text(value)
    if len(compact_node) < 12 or len(compact_value) <= len(compact_node[:12]):
        return None
    if compact_value.startswith(compact_node[: min(len(compact_node), 24)]):
        return None
    anchor_index: int | None = None
    for anchor_len in (28, 24, 20, 16, 12):
        if len(compact_node) < anchor_len:
            continue
        index = compact_value.find(compact_node[:anchor_len])
        if 0 < index <= 14:
            anchor_index = index
            break
    if anchor_index is None:
        return None
    char_index = _original_offset_for_compact_index(value, anchor_index)
    if char_index is None or char_index <= 0 or char_index > min(36, max(1, len(value) // 3)):
        return None
    removed = value[:char_index].strip()
    stripped = value[char_index:].lstrip()
    if not removed or len(_compact_alignment_text(removed)) > 14:
        return None
    if len(stripped) < 20:
        return None
    return stripped, removed


def _original_offset_for_compact_index(text: str, compact_index: int) -> int | None:
    seen = 0
    for position, char in enumerate(text):
        if char.isalnum():
            if seen == compact_index:
                return position
            seen += 1
    return None


def _compact_alignment_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", str(text or "")).casefold()


def _is_split_letter_debris(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    compact = _compact_alignment_text(value)
    if not compact or len(compact) > 12:
        return False
    if SPLIT_LETTER_WITH_SPACES_RE.match(value):
        return True
    if LOWERCASE_PUNCT_DEBRIS_RE.match(value):
        return True
    if len(compact) == 1 and re.fullmatch(r"\W*[a-z]\W*", value):
        return True
    # Single lowercase OCR leftovers such as ``g`` or ``i`` are common around
    # italic/math boundaries.  Keep real short words such as ``and`` or math
    # fragments such as ``30%`` even when local span alignment is imperfect.
    return len(compact) == 1 and value == value.lower() and bool(SPLIT_LETTER_DEBRIS_RE.match(value))


def _span_aligns_after_cursor(compact_node_text: str, compact_span_text: str, cursor: int) -> tuple[bool, int]:
    if not compact_span_text:
        return True, cursor
    index = compact_node_text.find(compact_span_text, max(cursor, 0))
    if index < 0:
        # Small OCR hyphenation differences can move a span a little behind the
        # monotonic cursor.  Allow a tiny backoff, but do not globally search the
        # whole node because that would preserve trailing one-letter debris just
        # because the same letter appeared earlier in the paragraph.
        index = compact_node_text.find(compact_span_text, max(cursor - 8, 0))
    if index < 0:
        return False, cursor
    return True, max(cursor, index + len(compact_span_text))


def clean_style_spans_for_node(spans: list[StyleSpan], *, node_text: str) -> tuple[list[StyleSpan], list[dict[str, str]]]:
    cleaned: list[StyleSpan] = []
    repairs: list[dict[str, str]] = []
    compact_node = _compact_alignment_text(node_text)
    cursor = 0
    for span in spans:
        text, removed = strip_split_letter_ocr_noise_prefix(span.text, node_text=node_text)
        if removed:
            repairs.append({"kind": "split_letter_ocr_prefix", "removed": removed, "original": span.text[:160]})
            span = replace(span, text=text)
        compact_span = _compact_alignment_text(text)
        if compact_node and not compact_span and re.fullmatch(r"\W+", str(text or "").strip()):
            repairs.append({"kind": "punctuation_ocr_span", "removed": text.strip(), "original": span.text[:160]})
            continue
        if compact_node and compact_span and _is_split_letter_debris(text):
            local_start = max(cursor - 2, 0)
            local_end = min(len(compact_node), cursor + max(4, len(compact_span) + 3))
            if compact_span not in compact_node[local_start:local_end]:
                repairs.append({"kind": "split_letter_ocr_span", "removed": text.strip(), "original": span.text[:160]})
                continue
        aligns, next_cursor = _span_aligns_after_cursor(compact_node, compact_span, cursor)
        if compact_node and not aligns and _is_split_letter_debris(text):
            repairs.append({"kind": "split_letter_ocr_span", "removed": text.strip(), "original": span.text[:160]})
            continue
        cursor = next_cursor
        cleaned.append(span)
    return cleaned, repairs


def flags_from_v7_item(item: dict[str, Any]) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for key in (
        "has_list_marker",
        "is_main_flow_candidate",
        "is_heading_candidate",
        "is_title_candidate",
        "is_toc",
        "is_noise",
        "no_render",
        "render_skip",
        "duplicate_shadow",
    ):
        value = item.get(key)
        if isinstance(value, bool):
            flags[key] = value
    return flags


def features_from_v7_item(item: dict[str, Any]) -> dict[str, float | int | bool | str | None]:
    keys = (
        "global_order",
        "column_fix_global_order",
        "column_fix_page_order",
        "layout_flow_order",
        "layout_band_id",
        "layout_band_global_id",
        "layout_band_global_order",
        "layout_is_band_boundary",
        "style_baseline_size",
        "style_span_count",
        "heading_level",
        "run_in_heading_level",
        "relative_font_size",
        "font_size",
        "page_width",
        "page_height",
        "footnote_marker",
        "footnote_label",
        "margin_note_side",
    )
    features: dict[str, float | int | bool | str | None] = {}
    for key in keys:
        value = item.get(key)
        if isinstance(value, (int, float, bool, str)) or value is None:
            features[key] = value
    list_marker = item.get("list_marker")
    if isinstance(list_marker, dict):
        features["list_marker_type"] = str(list_marker.get("type") or "")
    return features


def metadata_from_v7_item(item: dict[str, Any], *, include_raw_block: bool, include_raw_item: bool) -> dict[str, Any]:
    metadata_keys = (
        "type",
        "raw_type",
        "canonical_type",
        "category",
        "block_type",
        "text_level",
        "level",
        "subtype",
        "label",
        "original_index",
        "mineru_page_idx",
        "mineru_block_idx",
        "source_page_idxs",
        "layout_layer",
        "layout_role",
        "layout_band_type",
        "layout_band_column",
        "column_fix_span",
        "list_marker",
        "list_item_id",
        "reference_items",
        "img_path",
        "figure_asset_path",
        "image_asset_path",
        "image_path",
        "figure_path",
        "asset_path",
        "table_caption",
        "table_footnote",
        "table_body",
        "table_group_id",
        "table_group_member_ids",
        "table_group_member_index",
        "table_group_size",
        "table_group_primary",
        "table_group_bbox",
        "table_group_caption",
        "table_group_render_strategy",
        "figure_group_id",
        "image_group_id",
        "figure_group_member_ids",
        "image_group_member_ids",
        "figure_group_member_index",
        "image_group_member_index",
        "figure_group_size",
        "image_group_size",
        "figure_group_primary",
        "image_group_primary",
        "figure_group_bbox",
        "image_group_bbox",
        "figure_group_caption",
        "image_group_caption",
        "figure_group_render_strategy",
        "image_group_render_strategy",
        "figure_caption",
        "image_caption",
        "chart_caption",
        "table_caption",
        "code_caption",
        "algorithm_caption",
        "image_footnote",
        "chart_footnote",
        "code_footnote",
        "ref_text",
        "seal_text",
        "footnote_marker",
        "footnote_label",
        "footnote_anchor",
        "margin_note_side",
        "aside_text",
        "style_extract_status",
        "is_main_flow_candidate",
        "run_in_heading",
        "run_in_heading_number",
        "run_in_heading_text",
        "run_in_heading_body",
        "run_in_heading_level",
    )
    metadata = {key: item.get(key) for key in metadata_keys if key in item}
    if include_raw_block and isinstance(item.get("block"), dict):
        metadata["block"] = item["block"]
    if include_raw_item:
        metadata["raw_v7_item"] = dict(item)
    return metadata


def annotate_table_group_nodes(nodes: list[DocumentNode]) -> list[DocumentNode]:
    """Replace visual group member ids with stable DocumentNode ids."""

    by_raw_identifier: dict[str, str] = {}
    for node in nodes:
        for value in (
            node.node_id,
            node.metadata.get("global_order"),
            node.metadata.get("original_index"),
            node.metadata.get("mineru_block_idx"),
        ):
            if value is not None:
                by_raw_identifier[str(value)] = node.node_id
    updated: list[DocumentNode] = []
    for node in nodes:
        if node.node_type not in {BlockType.TABLE, BlockType.FIGURE}:
            updated.append(node)
            continue
        if "table_group_id" not in node.metadata and "figure_group_id" not in node.metadata and "image_group_id" not in node.metadata:
            updated.append(node)
            continue
        metadata = dict(node.metadata)
        members = metadata.get("table_group_member_ids")
        if isinstance(members, list):
            metadata["table_group_member_node_ids"] = [
                by_raw_identifier.get(str(value), str(value)) for value in members
            ]
        figure_members = metadata.get("figure_group_member_ids") or metadata.get("image_group_member_ids")
        if isinstance(figure_members, list):
            metadata["figure_group_member_node_ids"] = [
                by_raw_identifier.get(str(value), str(value)) for value in figure_members
            ]
        metadata["source_pdf"] = node.source_refs[0].metadata.get("pdf_path") if node.source_refs else None
        updated.append(replace(node, metadata=metadata))
    return updated


def int_value(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def float_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def load_v7_payload_without_contract(path: Path) -> dict[str, Any]:
    payload = read_json_payload(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
