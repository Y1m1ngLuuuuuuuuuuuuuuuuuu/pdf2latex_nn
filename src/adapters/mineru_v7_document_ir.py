"""Adapter from MinerU v7 styled content JSON to stable DocumentIR.

The v7 content JSON is a PDF-frontend implementation detail.  This adapter is
the single boundary where v7 field names are translated into the stable
DocumentIR contract consumed by style extraction, citation repair, decoding and
rendering.
"""

from __future__ import annotations

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
from src.pipeline.v7_contract import assert_v7_content_json, read_json_payload


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
        text = text_from_v7_item(item)
        metadata = metadata_from_v7_item(item, include_raw_block=self.config.include_raw_block, include_raw_item=self.config.include_raw_item)
        if source_path is not None:
            metadata.setdefault("source_json", str(source_path))
            metadata.setdefault("source_json_dir", str(source_path.parent))
            metadata.setdefault("asset_base_dir", str(source_path.parent))
        if pdf_path is not None:
            metadata.setdefault("source_pdf", str(pdf_path))
        features = features_from_v7_item(item)
        spans = [style_span_from_v7(span) for span in item.get("style_spans", []) if isinstance(span, dict)]
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
    raw = str(item.get("canonical_type") or item.get("type") or item.get("raw_type") or "").casefold()
    if raw in {"page_footnote", "footnote", "foot_note"} or role in {"footnote", "page_footnote"}:
        return BlockType.FOOTNOTE
    if raw in {"margin_note", "marginnote", "side_note", "sidenote", "sidebar"} or role in {"margin_note", "marginnote", "side_note", "sidenote"}:
        return BlockType.MARGIN_NOTE
    if layer == "noise_layer" or raw in {"page_number", "header", "footer"} or role == "noise":
        return BlockType.HEADER_FOOTER
    if raw in {"toc", "index", "table_of_contents"} or role in {"toc_title", "toc_entry"}:
        return BlockType.TOC
    if raw in {"paragraph", "text", "paragraph_text"}:
        return BlockType.TEXT
    if raw in {"title", "section", "subsection", "subsubsection", "heading"}:
        return BlockType.TITLE
    if raw in {"equation", "equation_interline", "interline_equation", "display_formula", "formula"}:
        return BlockType.EQUATION
    if raw in {"inline_math", "inline_formula", "math_inline"}:
        return BlockType.INLINE_MATH
    if raw in {"table"}:
        return BlockType.TABLE
    if raw in {"figure", "image", "chart"}:
        return BlockType.FIGURE
    if raw in {"algorithm"}:
        return BlockType.ALGORITHM
    if raw in {"list", "item", "itemize", "enumerate"}:
        return BlockType.LIST
    if raw in {"code"}:
        return BlockType.CODE
    if raw in {"reference", "references", "bibliography"}:
        return BlockType.REFERENCE
    return BlockType.OTHER if not text_from_v7_item(item) else BlockType.TEXT


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
        return caption_text
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


def flags_from_v7_item(item: dict[str, Any]) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for key in (
        "has_list_marker",
        "is_main_flow_candidate",
        "is_heading_candidate",
        "is_title_candidate",
        "is_toc",
        "is_noise",
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
        "footnote_marker",
        "footnote_label",
        "footnote_anchor",
        "margin_note_side",
        "style_extract_status",
        "canonical_type",
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
