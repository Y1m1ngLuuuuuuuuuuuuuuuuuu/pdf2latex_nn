"""Build a v8 logical content layer directly from MinerU raw ``middle.json``.

The v7/content-list path consumes MinerU's already-merged logical blocks.  In
some papers MinerU merges continuation text before our reading-order correction
has a chance to run, so a page-local order error becomes a wrong paragraph
owner.  The v8 path intentionally starts earlier: it reads ``preproc_blocks``
from ``middle.json``, rebuilds a page/column reading order, and only then
performs conservative continuation merges.

This module does not mutate v7, does not build a GNN view, and does not depend
on graph labels.  It emits a standalone v8 JSON payload plus diagnostics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TEXT_LIKE_TYPES = {"text", "title", "interline_equation", "equation", "list", "abstract"}
NON_MERGE_TYPES = {"title", "table", "figure", "image", "chart", "algorithm", "equation", "interline_equation"}
FLOAT_SKIP_TYPES = {"table", "figure", "image", "chart", "algorithm"}
SENTENCE_END_RE = re.compile(r"[.!?。！？;；:]$|[.!?。！？][\"')\\]}]*$")
LOWERCASE_START_RE = re.compile(r"^[a-zα-ω]")
CONTINUATION_START_RE = re.compile(
    r"^(?:and|or|but|for|to|of|in|on|with|which|where|while|because|that|than|from|as|by|into|through|under|over)\\b",
    re.IGNORECASE,
)


@dataclass
class V8Line:
    """Line/span evidence extracted from one middle block."""

    line_id: str
    text: str
    page_idx: int
    bbox: list[float]
    source_block_id: str
    line_idx: int
    span_idx: int | None
    cross_page: bool = False
    cross_column: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "text": self.text,
            "page_idx": self.page_idx,
            "bbox": self.bbox,
            "source_block_id": self.source_block_id,
            "line_idx": self.line_idx,
            "span_idx": self.span_idx,
            "cross_page": self.cross_page,
            "cross_column": self.cross_column,
        }


@dataclass
class V8Block:
    """Atomic raw block before v8 continuation merging."""

    block_id: str
    doc_id: str
    page_idx: int
    middle_index: int
    type: str
    bbox: list[float]
    text: str
    page_size: list[float]
    lines: list[V8Line] = field(default_factory=list)
    score: float | None = None
    column_id: int | None = None
    is_full_width: bool = False
    reading_order: int | None = None
    order_key: tuple[Any, ...] = field(default_factory=tuple)
    source: str = "preproc_blocks"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "doc_id": self.doc_id,
            "page_idx": self.page_idx,
            "middle_index": self.middle_index,
            "type": self.type,
            "bbox": self.bbox,
            "text": self.text,
            "page_size": self.page_size,
            "score": self.score,
            "column_id": self.column_id,
            "is_full_width": self.is_full_width,
            "reading_order": self.reading_order,
            "order_key": list(self.order_key),
            "source": self.source,
            "metadata": self.metadata,
            "lines": [line.to_json() for line in self.lines],
        }


@dataclass
class MergeDecision:
    """A v8 continuation merge between two atomic middle blocks."""

    src_block_id: str
    dst_block_id: str
    reason: str
    confidence: float
    evidence: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "src_block_id": self.src_block_id,
            "dst_block_id": self.dst_block_id,
            "reason": self.reason,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_v8_from_middle(
    *,
    doc_id: str,
    middle_json_path: Path,
    content_list_json_path: Path | None = None,
    style_content_list_json_path: Path | None = None,
    middle_block_source: str = "preproc_blocks",
    debug_page: int | None = None,
) -> dict[str, Any]:
    """Convert one MinerU middle.json into a standalone v8 payload."""

    payload = load_json(middle_json_path)
    blocks = extract_atomic_blocks(payload, doc_id=doc_id, source=middle_block_source)
    if content_list_json_path is not None:
        augment_blocks_from_content_list(
            blocks,
            load_json(content_list_json_path),
            content_list_json_path=content_list_json_path,
        )
    if style_content_list_json_path is not None:
        attach_styles_from_content_list(
            blocks,
            load_json(style_content_list_json_path),
            style_content_list_json_path=style_content_list_json_path,
        )
    assign_layout_columns(blocks)
    ordered_blocks = rebuild_reading_order(blocks)
    merge_decisions = infer_continuation_merges(ordered_blocks)
    items = materialize_v8_items(ordered_blocks, merge_decisions)
    diagnostics = build_diagnostics(
        doc_id=doc_id,
        middle_json_path=middle_json_path,
        content_list_json_path=content_list_json_path,
        source=middle_block_source,
        ordered_blocks=ordered_blocks,
        merge_decisions=merge_decisions,
        items=items,
        debug_page=debug_page,
    )
    return {
        "schema_version": "content_list_v8_reflow_v1",
        "doc_id": doc_id,
        "source": {
            "middle_json": str(middle_json_path),
            "content_list_json": str(content_list_json_path) if content_list_json_path is not None else None,
            "style_content_list_json": str(style_content_list_json_path) if style_content_list_json_path is not None else None,
            "middle_block_source": middle_block_source,
        },
        "items": items,
        "atomic_blocks": [block.to_json() for block in ordered_blocks],
        "merge_decisions": [decision.to_json() for decision in merge_decisions],
        "diagnostics": diagnostics,
    }


CONTENT_LIST_COPY_KEYS = {
    "img_path",
    "table_caption",
    "table_footnote",
    "table_body",
    "chart_caption",
    "chart_footnote",
    "image_caption",
    "image_footnote",
    "figure_caption",
    "figure_footnote",
    "code_caption",
    "code_footnote",
    "algorithm_caption",
    "algorithm_footnote",
    "html",
}
CONTENT_FLOAT_TYPES = {"table", "chart", "figure", "image", "algorithm", "code"}
CONTENT_TEXT_TYPES = {"text", "paragraph", "title", "abstract", "ref_text", "reference"}


def augment_blocks_from_content_list(
    blocks: list[V8Block],
    content_payload: Any,
    *,
    content_list_json_path: Path,
) -> None:
    """Attach MinerU content-list float/table assets to middle blocks.

    Middle JSON is better for raw line order, while content_list keeps the
    image/table asset path and caption/body fields.  V8 uses middle for order
    and imports only matched asset metadata as a sidecar.
    """

    raw_items = content_payload if isinstance(content_payload, list) else content_payload.get("items", [])
    if not isinstance(raw_items, list):
        return
    candidates = build_content_list_candidates(raw_items)
    float_candidates = [candidate for candidate in candidates if candidate["type"] in CONTENT_FLOAT_TYPES]

    matched_indices: set[int] = set()
    for block in blocks:
        block_type = normalize_content_type(block.type)
        if block_type not in CONTENT_FLOAT_TYPES or len(block.bbox) != 4:
            continue
        block_bbox_1000 = pdf_points_bbox_to_1000(block.bbox, block.page_size)
        best: dict[str, Any] | None = None
        best_score = 0.0
        for candidate in float_candidates:
            if candidate["index"] in matched_indices:
                continue
            if candidate["page_idx"] != block.page_idx:
                continue
            if not content_types_compatible(block_type, candidate["type"]):
                continue
            iou = bbox_iou(block_bbox_1000, candidate["bbox"])
            center_score = bbox_center_closeness(block_bbox_1000, candidate["bbox"])
            score = iou + 0.2 * center_score
            if score > best_score:
                best_score = score
                best = candidate
        if best is None or best_score < 0.08:
            continue
        matched_indices.add(int(best["index"]))
        item = best["item"]
        block.metadata.update(
            {
                "content_list_json": str(content_list_json_path),
                "asset_base_dir": str(content_list_json_path.parent),
                "source_content_list_index": best["index"],
                "content_list_type": item.get("type"),
                "content_list_bbox_1000": best["bbox"],
                "content_list_match_score": round(best_score, 4),
            }
        )
        for key in CONTENT_LIST_COPY_KEYS:
            value = item.get(key)
            if value not in (None, "", [], {}):
                if key == "img_path" and isinstance(value, str):
                    raw_path = Path(value)
                    if not raw_path.is_absolute():
                        value = str((content_list_json_path.parent / raw_path).resolve())
                block.metadata[key] = value
        caption = first_text_value(item.get("figure_caption") or item.get("image_caption") or item.get("chart_caption"))
        if caption and normalize_content_type(block.type) in {"figure", "chart"}:
            block.metadata.setdefault("figure_caption", caption)
        table_caption = first_text_value(item.get("table_caption"))
        if table_caption and normalize_content_type(block.type) == "table":
            block.metadata["table_caption"] = table_caption

    attach_semantic_text_from_content_list(
        blocks,
        candidates,
        content_list_json_path=content_list_json_path,
    )


def build_content_list_candidates(raw_items: list[Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        bbox = normalize_bbox(item.get("bbox"))
        if not bbox:
            continue
        item_type = normalize_content_type(item.get("type"))
        candidates.append(
            {
                "index": index,
                "type": item_type,
                "page_idx": int_or_default(item.get("page_idx"), 0),
                "bbox": bbox,
                "item": item,
                "text": content_list_text(item),
            }
        )
    return candidates


def attach_semantic_text_from_content_list(
    blocks: list[V8Block],
    candidates: list[dict[str, Any]],
    *,
    content_list_json_path: Path,
) -> None:
    """Attach text/title semantic hints without letting content_list drive order.

    MinerU content_list already contains logical pre-merges, so v8 must not use
    it as an owner of paragraph continuation.  A strong same-bbox match is still
    useful for cleaner titles/captions and for traceability.
    """

    matched_indices: set[int] = set()
    text_candidates = [
        candidate
        for candidate in candidates
        if candidate["type"] in CONTENT_TEXT_TYPES and str(candidate.get("text") or "").strip()
    ]
    for block in blocks:
        block_type = normalize_content_type(block.type)
        if block_type not in TEXT_LIKE_TYPES or len(block.bbox) != 4:
            continue
        block_bbox_1000 = pdf_points_bbox_to_1000(block.bbox, block.page_size)
        best: dict[str, Any] | None = None
        best_score = 0.0
        best_iou = 0.0
        for candidate in text_candidates:
            if candidate["index"] in matched_indices:
                continue
            if candidate["page_idx"] != block.page_idx:
                continue
            if not text_content_types_compatible(block_type, candidate["type"]):
                continue
            iou = bbox_iou(block_bbox_1000, candidate["bbox"])
            center_score = bbox_center_closeness(block_bbox_1000, candidate["bbox"])
            score = iou + 0.2 * center_score
            if score > best_score:
                best_score = score
                best_iou = iou
                best = candidate
        if best is None or (best_iou < 0.18 and best_score < 0.30):
            continue
        matched_indices.add(int(best["index"]))
        text = str(best.get("text") or "").strip()
        if not text:
            continue
        block.metadata.update(
            {
                "content_list_json": str(content_list_json_path),
                "source_content_list_index": best["index"],
                "content_list_type": best["type"],
                "content_list_bbox_1000": best["bbox"],
                "content_list_match_score": round(best_score, 4),
                "content_list_text": text,
                "content_list_text_iou": round(best_iou, 4),
            }
        )
        copy_style_metadata_from_content_item(block.metadata, best["item"])


def attach_styles_from_content_list(
    blocks: list[V8Block],
    style_payload: Any,
    *,
    style_content_list_json_path: Path,
) -> None:
    """Attach PyMuPDF-derived style spans to v8 blocks by bbox.

    ``middle.json`` usually has line geometry but no font-weight/font-size
    details.  Existing v7-style content lists carry that information.  This
    sidecar must not replace v8 text/order; it only enriches matched blocks.
    """

    raw_items = style_payload if isinstance(style_payload, list) else style_payload.get("items", [])
    if not isinstance(raw_items, list):
        return
    candidates = [
        candidate
        for candidate in build_content_list_candidates(raw_items)
        if candidate["type"] in (CONTENT_TEXT_TYPES | CONTENT_FLOAT_TYPES)
    ]
    matched_indices: set[int] = set()
    for block in blocks:
        block_type = normalize_content_type(block.type)
        if block_type not in (TEXT_LIKE_TYPES | CONTENT_FLOAT_TYPES) or len(block.bbox) != 4:
            continue
        block_bbox_1000 = pdf_points_bbox_to_1000(block.bbox, block.page_size)
        best: dict[str, Any] | None = None
        best_score = 0.0
        best_iou = 0.0
        for candidate in candidates:
            if candidate["index"] in matched_indices:
                continue
            if candidate["page_idx"] != block.page_idx:
                continue
            candidate_type = candidate["type"]
            if block_type in TEXT_LIKE_TYPES:
                compatible = text_content_types_compatible(block_type, candidate_type)
            else:
                compatible = content_types_compatible(block_type, candidate_type)
            if not compatible:
                continue
            iou = bbox_iou(block_bbox_1000, candidate["bbox"])
            center_score = bbox_center_closeness(block_bbox_1000, candidate["bbox"])
            score = iou + 0.2 * center_score
            if score > best_score:
                best_score = score
                best_iou = iou
                best = candidate
        if best is None or (best_iou < 0.18 and best_score < 0.30):
            continue
        matched_indices.add(int(best["index"]))
        block.metadata.update(
            {
                "style_content_list_json": str(style_content_list_json_path),
                "style_content_list_index": best["index"],
                "style_content_list_match_score": round(best_score, 4),
                "style_content_list_iou": round(best_iou, 4),
            }
        )
        copy_style_metadata_from_content_item(block.metadata, best["item"])


def copy_style_metadata_from_content_item(target: dict[str, Any], item: dict[str, Any]) -> None:
    spans = item.get("style_spans")
    if isinstance(spans, list) and spans:
        target["style_spans"] = spans
    for key in ("style_extract_status", "style_config", "font_size", "bold_ratio", "relative_font_size", "style_baseline_size"):
        value = item.get(key)
        if value not in (None, "", [], {}):
            target[key] = value


def content_list_text(item: dict[str, Any]) -> str:
    for key in ("text", "text_for_embedding", "text_preview"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_space(value)
    return ""


def normalize_content_type(value: Any) -> str:
    raw = str(value or "").casefold().strip()
    return "figure" if raw == "image" else raw


def first_text_value(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(part).strip() for part in value if str(part).strip()).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def content_types_compatible(block_type: str, content_type: str) -> bool:
    if block_type == content_type:
        return True
    if block_type in {"figure", "image"} and content_type in {"figure", "image"}:
        return True
    if block_type in {"chart", "figure"} and content_type in {"chart", "figure"}:
        return True
    return False


def text_content_types_compatible(block_type: str, content_type: str) -> bool:
    if block_type == "title":
        return content_type in {"title", "text"}
    if block_type == "abstract":
        return content_type in {"abstract", "text", "paragraph"}
    if block_type in {"text", "list"}:
        return content_type in {"text", "paragraph", "abstract", "ref_text", "reference"}
    if block_type in {"ref_text", "reference"}:
        return content_type in {"ref_text", "reference", "text"}
    return content_type in CONTENT_TEXT_TYPES


def pdf_points_bbox_to_1000(bbox: list[float], page_size: list[float]) -> list[float]:
    if len(bbox) != 4 or len(page_size) < 2:
        return bbox
    width, height = float(page_size[0] or 0), float(page_size[1] or 0)
    if width <= 0 or height <= 0:
        return bbox
    return [
        bbox[0] / width * 1000.0,
        bbox[1] / height * 1000.0,
        bbox[2] / width * 1000.0,
        bbox[3] / height * 1000.0,
    ]


def bbox_iou(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def bbox_center_closeness(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return max(0.0, 1.0 - distance / 1000.0)


def extract_atomic_blocks(payload: Any, *, doc_id: str, source: str) -> list[V8Block]:
    if not isinstance(payload, dict):
        raise ValueError("middle.json payload must be an object")
    pdf_info = payload.get("pdf_info")
    if not isinstance(pdf_info, list):
        raise ValueError("middle.json missing pdf_info list")

    blocks: list[V8Block] = []
    for page_pos, page in enumerate(pdf_info):
        if not isinstance(page, dict):
            continue
        page_idx = int_or_default(page.get("page_idx"), page_pos)
        page_size = normalize_page_size(page.get("page_size"))
        raw_blocks = page.get(source)
        if not isinstance(raw_blocks, list):
            continue
        for local_idx, raw in enumerate(raw_blocks):
            if not isinstance(raw, dict):
                continue
            middle_index = int_or_default(raw.get("index"), local_idx)
            block_id = f"{doc_id}:p{page_idx:04d}:m{middle_index:06d}"
            lines = extract_lines(raw, block_id=block_id, page_idx=page_idx)
            text = normalize_space(" ".join(line.text for line in lines))
            bbox = normalize_bbox(raw.get("bbox"))
            blocks.append(
                V8Block(
                    block_id=block_id,
                    doc_id=doc_id,
                    page_idx=page_idx,
                    middle_index=middle_index,
                    type=str(raw.get("type") or "unknown"),
                    bbox=bbox,
                    text=text,
                    page_size=page_size,
                    lines=lines,
                    score=float_or_none(raw.get("score")),
                    source=source,
                )
            )
    return blocks


def extract_lines(raw: dict[str, Any], *, block_id: str, page_idx: int) -> list[V8Line]:
    lines: list[V8Line] = []
    raw_lines = raw.get("lines")
    if isinstance(raw_lines, list):
        for line_idx, raw_line in enumerate(raw_lines):
            if not isinstance(raw_line, dict):
                continue
            spans = raw_line.get("spans")
            if isinstance(spans, list) and spans:
                # Most MinerU lines contain one text span.  If multiple spans
                # exist, preserve each span as line evidence but keep the same
                # line index for traceability.
                for span_idx, span in enumerate(spans):
                    if not isinstance(span, dict):
                        continue
                    text = text_from_any(span)
                    if not text:
                        continue
                    line_bbox = normalize_bbox(raw_line.get("bbox")) or normalize_bbox(span.get("bbox"))
                    lines.append(
                        V8Line(
                            line_id=f"{block_id}:l{line_idx:04d}:s{span_idx:04d}",
                            text=text,
                            page_idx=page_idx,
                            bbox=line_bbox,
                            source_block_id=block_id,
                            line_idx=line_idx,
                            span_idx=span_idx,
                            cross_page=bool(raw_line.get("cross_page") or span.get("cross_page")),
                            cross_column=bool(raw_line.get("cross_column") or span.get("cross_column")),
                        )
                    )
                continue
            text = text_from_any(raw_line)
            if text:
                lines.append(
                    V8Line(
                        line_id=f"{block_id}:l{line_idx:04d}",
                        text=text,
                        page_idx=page_idx,
                        bbox=normalize_bbox(raw_line.get("bbox")),
                        source_block_id=block_id,
                        line_idx=line_idx,
                        span_idx=None,
                        cross_page=bool(raw_line.get("cross_page")),
                        cross_column=bool(raw_line.get("cross_column")),
                    )
                )
    if lines:
        return lines
    text = text_from_any(raw)
    if not text:
        return []
    return [
        V8Line(
            line_id=f"{block_id}:l0000",
            text=text,
            page_idx=page_idx,
            bbox=normalize_bbox(raw.get("bbox")),
            source_block_id=block_id,
            line_idx=0,
            span_idx=None,
            cross_page=bool(raw.get("cross_page")),
            cross_column=bool(raw.get("cross_column")),
        )
    ]


def assign_layout_columns(blocks: list[V8Block]) -> None:
    by_page: dict[int, list[V8Block]] = {}
    for block in blocks:
        by_page.setdefault(block.page_idx, []).append(block)

    for page_blocks in by_page.values():
        page_width = max((block.page_size[0] for block in page_blocks if block.page_size), default=0.0)
        if not page_width:
            page_width = max((block.bbox[2] for block in page_blocks if len(block.bbox) == 4), default=1.0)
        for block in page_blocks:
            if len(block.bbox) != 4:
                block.column_id = None
                block.is_full_width = False
                continue
            x0, _, x1, _ = block.bbox
            width = max(0.0, x1 - x0)
            center = (x0 + x1) / 2.0
            block.is_full_width = width >= page_width * 0.68 or (x0 <= page_width * 0.22 and x1 >= page_width * 0.78)
            if block.is_full_width:
                block.column_id = -1
            elif center < page_width * 0.52:
                block.column_id = 0
            else:
                block.column_id = 1


def rebuild_reading_order(blocks: list[V8Block]) -> list[V8Block]:
    """Rebuild document order from raw page blocks.

    The important correction is page-local: after wide table/figure material at
    the top of a two-column page, non-wide body content is ordered by column
    (left column top-to-bottom, then right column top-to-bottom).  This prevents
    a short right-column continuation line from being attached to the previous
    left-column top fragment purely because their y coordinates are similar.
    """

    ordered: list[V8Block] = []
    by_page: dict[int, list[V8Block]] = {}
    for block in blocks:
        by_page.setdefault(block.page_idx, []).append(block)

    for page_idx in sorted(by_page):
        page_blocks = by_page[page_idx]
        page_order = order_page_blocks(page_blocks)
        ordered.extend(page_order)

    for idx, block in enumerate(ordered):
        block.reading_order = idx
    return ordered


def order_page_blocks(page_blocks: list[V8Block]) -> list[V8Block]:
    text_blocks = [b for b in page_blocks if b.type in TEXT_LIKE_TYPES and len(b.bbox) == 4]
    has_two_columns = bool({b.column_id for b in text_blocks if b.column_id in {0, 1}} == {0, 1})
    if not has_two_columns:
        for block in page_blocks:
            block.order_key = (block.page_idx, bbox_y0(block), bbox_x0(block), block.middle_index)
        return sorted(page_blocks, key=lambda b: b.order_key)

    full_width = [b for b in page_blocks if b.is_full_width]
    column_blocks = [b for b in page_blocks if not b.is_full_width]
    # Wide figures/tables/section bands are page-spanning anchors.  Keep their
    # y-position as band separators instead of blindly moving every wide object
    # to the page top.
    full_width_sorted = sorted(full_width, key=lambda b: (bbox_y0(b), bbox_x0(b), b.middle_index))
    emitted: list[V8Block] = []
    remaining_columns = sorted(column_blocks, key=lambda b: (bbox_y0(b), bbox_x0(b), b.middle_index))

    for wide in full_width_sorted:
        before = [b for b in remaining_columns if bbox_y0(b) < bbox_y0(wide)]
        if before:
            emitted.extend(order_column_band(before))
            before_ids = {b.block_id for b in before}
            remaining_columns = [b for b in remaining_columns if b.block_id not in before_ids]
        emitted.append(wide)
    if remaining_columns:
        emitted.extend(order_column_band(remaining_columns))

    for rank, block in enumerate(emitted):
        column_rank = 0 if block.is_full_width else (3 if block.column_id is None else 1 + int(block.column_id))
        block.order_key = (block.page_idx, column_rank, rank, bbox_y0(block), bbox_x0(block), block.middle_index)
    return emitted


def order_column_band(blocks: list[V8Block]) -> list[V8Block]:
    ordered: list[V8Block] = []
    for column_id in (0, 1, None):
        members = [b for b in blocks if b.column_id == column_id]
        ordered.extend(sorted(members, key=lambda b: (bbox_y0(b), bbox_x0(b), b.middle_index)))
    return ordered


def infer_continuation_merges(ordered_blocks: list[V8Block]) -> list[MergeDecision]:
    decisions: list[MergeDecision] = []
    last_text: V8Block | None = None
    for block in ordered_blocks:
        if not is_body_merge_candidate(block):
            if block.type in FLOAT_SKIP_TYPES:
                # Floats/tables can be skipped between a previous open text
                # block and the next continuation, so do not reset last_text.
                continue
            # Headings, equations, captions, and unknown text barriers should
            # not allow a paragraph continuation to jump across them.
            last_text = None
            continue
        if last_text is not None:
            decision = continuation_decision(last_text, block)
            if decision is not None:
                decisions.append(decision)
        last_text = block
    return decisions


def continuation_decision(prev: V8Block, curr: V8Block) -> MergeDecision | None:
    if prev.block_id == curr.block_id:
        return None
    if prev.type in NON_MERGE_TYPES or curr.type in NON_MERGE_TYPES:
        return None
    if not prev.text or not curr.text:
        return None
    if not is_open_ended(prev.text):
        return None
    if not starts_like_continuation(curr.text):
        return None

    same_column = prev.page_idx == curr.page_idx and prev.column_id == curr.column_id
    cross_column = prev.page_idx == curr.page_idx and prev.column_id in {0, 1} and curr.column_id in {0, 1} and curr.column_id > prev.column_id
    cross_page = curr.page_idx == prev.page_idx + 1
    close_vertical = same_column and vertical_gap(prev, curr) <= body_line_height(prev, curr) * 2.8
    prev_near_page_bottom = bbox_y1(prev) >= page_height(prev) * 0.82
    curr_near_column_top = bbox_y0(curr) <= page_height(curr) * 0.62

    if close_vertical:
        return make_decision(prev, curr, "same_column_open_sentence", 0.86)
    if cross_column and prev_near_page_bottom and curr_near_column_top:
        return make_decision(prev, curr, "cross_column_open_sentence", 0.82)
    if cross_page and curr.text and (prev_near_page_bottom or curr_near_column_top):
        return make_decision(prev, curr, "cross_page_open_sentence", 0.78)
    return None


def make_decision(prev: V8Block, curr: V8Block, reason: str, confidence: float) -> MergeDecision:
    return MergeDecision(
        src_block_id=prev.block_id,
        dst_block_id=curr.block_id,
        reason=reason,
        confidence=confidence,
        evidence={
            "prev_text_tail": prev.text[-120:],
            "curr_text_head": curr.text[:120],
            "prev_page_idx": prev.page_idx,
            "curr_page_idx": curr.page_idx,
            "prev_column_id": prev.column_id,
            "curr_column_id": curr.column_id,
            "prev_bbox": prev.bbox,
            "curr_bbox": curr.bbox,
            "prev_open_ended": is_open_ended(prev.text),
            "curr_starts_continuation": starts_like_continuation(curr.text),
            "vertical_gap": vertical_gap(prev, curr),
        },
    )


def materialize_v8_items(ordered_blocks: list[V8Block], decisions: list[MergeDecision]) -> list[dict[str, Any]]:
    merge_dst_to_src = {decision.dst_block_id: decision.src_block_id for decision in decisions}
    merge_src_to_dst: dict[str, str] = {decision.src_block_id: decision.dst_block_id for decision in decisions}
    decision_by_pair = {(decision.src_block_id, decision.dst_block_id): decision for decision in decisions}
    block_by_id = {block.block_id: block for block in ordered_blocks}
    consumed: set[str] = set()
    items: list[dict[str, Any]] = []

    for block in ordered_blocks:
        if block.block_id in consumed or block.block_id in merge_dst_to_src:
            continue
        chain = [block]
        cursor = block
        while cursor.block_id in merge_src_to_dst:
            next_id = merge_src_to_dst[cursor.block_id]
            next_block = block_by_id.get(next_id)
            if next_block is None or next_block.block_id in consumed:
                break
            chain.append(next_block)
            cursor = next_block

        for member in chain:
            consumed.add(member.block_id)
        first = chain[0]
        text = chain[0].text
        for prev, curr in zip(chain, chain[1:], strict=False):
            text = join_continuation_text(text, curr.text)
        middle_text = normalize_visual_text(text)
        text_source = "middle_reflow_merged" if len(chain) > 1 else "middle_reflow"
        if len(chain) == 1:
            semantic_text = str(first.metadata.get("content_list_text") or "").strip()
            if should_use_content_list_text(first, semantic_text, middle_text):
                text = normalize_visual_text(semantic_text)
                text_source = "content_list_bbox_match"
            else:
                text = middle_text
        else:
            text = middle_text
        bbox = bbox_union([member.bbox for member in chain])
        item = {
            "id": f"v8_{len(items):06d}",
            "type": first.type,
            "text": text,
            "page_idx": first.page_idx,
            "bbox": bbox,
            "page_width": first.page_size[0] if len(first.page_size) >= 2 else None,
            "page_height": first.page_size[1] if len(first.page_size) >= 2 else None,
            "source_block_ids": [member.block_id for member in chain],
            "source_middle_indices": [member.middle_index for member in chain],
            "source_page_indices": [member.page_idx for member in chain],
            "source_line_ids": [line.line_id for member in chain for line in member.lines],
            "source_lines": [line.to_json() for member in chain for line in member.lines],
            "reading_order": len(items),
            "column_id": first.column_id,
            "merge_chain_length": len(chain),
            "merge_reasons": [
                decision_by_pair[(prev.block_id, curr.block_id)].reason
                for prev, curr in zip(chain, chain[1:], strict=False)
                if (prev.block_id, curr.block_id) in decision_by_pair
            ],
            "continued_from_previous_page": len({member.page_idx for member in chain}) > 1
            and chain[0].page_idx < chain[-1].page_idx,
            "v8_source": "middle_preproc_reflow",
            "text_source": text_source,
        }
        if text != middle_text:
            item["middle_text"] = middle_text
        content_text_candidates = [
            {
                "source_block_id": member.block_id,
                "content_list_index": member.metadata.get("source_content_list_index"),
                "content_list_type": member.metadata.get("content_list_type"),
                "content_list_text": member.metadata.get("content_list_text"),
                "content_list_match_score": member.metadata.get("content_list_match_score"),
            }
            for member in chain
            if member.metadata.get("content_list_text")
        ]
        if content_text_candidates:
            item["content_list_text_candidates"] = content_text_candidates
        for key, value in first.metadata.items():
            item[key] = value
        items.append(item)
    return items


def should_use_content_list_text(block: V8Block, semantic_text: str, middle_text: str) -> bool:
    if not semantic_text:
        return False
    if normalize_content_type(block.type) == "title":
        return True
    score = float(block.metadata.get("content_list_match_score") or 0.0)
    iou = float(block.metadata.get("content_list_text_iou") or 0.0)
    if iou < 0.35 and score < 0.45:
        return False
    semantic_len = len(semantic_text)
    middle_len = max(1, len(middle_text))
    # content_list may already contain a pre-merged paragraph.  Only replace
    # body text when lengths are in the same neighborhood.
    if semantic_len > middle_len * 1.35 or semantic_len < middle_len * 0.65:
        return False
    return True


def build_diagnostics(
    *,
    doc_id: str,
    middle_json_path: Path,
    content_list_json_path: Path | None,
    source: str,
    ordered_blocks: list[V8Block],
    merge_decisions: list[MergeDecision],
    items: list[dict[str, Any]],
    debug_page: int | None,
) -> dict[str, Any]:
    page_orders: dict[str, list[dict[str, Any]]] = {}
    pages = sorted({block.page_idx for block in ordered_blocks})
    for page_idx in pages:
        if debug_page is not None and page_idx != debug_page:
            continue
        page_orders[str(page_idx)] = [
            {
                "reading_order": block.reading_order,
                "block_id": block.block_id,
                "middle_index": block.middle_index,
                "type": block.type,
                "column_id": block.column_id,
                "is_full_width": block.is_full_width,
                "bbox": block.bbox,
                "text_preview": block.text[:180],
            }
            for block in ordered_blocks
            if block.page_idx == page_idx
        ]

    return {
        "schema_version": "v8_reflow_diagnostics_v1",
        "doc_id": doc_id,
        "middle_json": str(middle_json_path),
        "content_list_json": str(content_list_json_path) if content_list_json_path is not None else None,
        "middle_block_source": source,
        "block_count": len(ordered_blocks),
        "item_count": len(items),
        "merge_count": len(merge_decisions),
        "merge_decisions": [decision.to_json() for decision in merge_decisions],
        "page_orders": page_orders,
        "merge_reason_counts": count_by(decision.reason for decision in merge_decisions),
    }


def is_body_merge_candidate(block: V8Block) -> bool:
    if block.type in NON_MERGE_TYPES:
        return False
    return block.type in TEXT_LIKE_TYPES and bool(block.text)


def is_open_ended(text: str) -> bool:
    clean = normalize_space(text)
    if not clean:
        return False
    if clean.endswith("-"):
        return True
    if SENTENCE_END_RE.search(clean):
        return False
    # A long text block ending without terminal punctuation is usually a
    # paragraph fragment at a column/page boundary.
    return len(clean) >= 24


def starts_like_continuation(text: str) -> bool:
    clean = normalize_space(text).lstrip("([{")
    if not clean:
        return False
    if LOWERCASE_START_RE.search(clean):
        return True
    if CONTINUATION_START_RE.search(clean):
        return True
    # Hyphenated previous line often resumes with an uppercase acronym or
    # dataset token, e.g. "CI-" -> "CEVSE2024".
    return bool(re.match(r"^[A-Z]{2,}[A-Za-z0-9-]*[,)]?", clean))


def join_continuation_text(prev: str, curr: str) -> str:
    prev_clean = normalize_space(prev)
    curr_clean = normalize_space(curr)
    if not prev_clean:
        return curr_clean
    if not curr_clean:
        return prev_clean
    if prev_clean.endswith("-") and curr_clean and re.match(r"^[A-Za-z]", curr_clean):
        return prev_clean[:-1] + curr_clean
    return prev_clean + " " + curr_clean


def normalize_visual_text(text: str) -> str:
    value = normalize_space(text)
    # MinerU middle keeps visual line-break hyphenation such as
    # "chal- lenges".  Remove the artificial split before rendering.
    value = re.sub(r"\b([A-Z]?[a-z]{2,})-\s+([a-z]{2,})\b", r"\1\2", value)
    value = re.sub(r"\b([A-Z]?[a-z]{2,})-\s+([A-Z][a-z]{2,})\b", r"\1\2", value)
    value = re.sub(r"\b([A-Z]{2})-\s+([A-Z][A-Za-z0-9]{2,})\b", r"\1\2", value)
    return normalize_space(value)


def vertical_gap(prev: V8Block, curr: V8Block) -> float:
    if len(prev.bbox) != 4 or len(curr.bbox) != 4:
        return float("inf")
    return curr.bbox[1] - prev.bbox[3]


def body_line_height(prev: V8Block, curr: V8Block) -> float:
    heights = []
    for block in (prev, curr):
        for line in block.lines:
            if len(line.bbox) == 4:
                heights.append(max(1.0, line.bbox[3] - line.bbox[1]))
    if not heights:
        return 12.0
    return sorted(heights)[len(heights) // 2]


def page_height(block: V8Block) -> float:
    if len(block.page_size) >= 2:
        return max(1.0, float(block.page_size[1]))
    if len(block.bbox) == 4:
        return max(1.0, float(block.bbox[3]))
    return 792.0


def bbox_x0(block: V8Block) -> float:
    return block.bbox[0] if len(block.bbox) == 4 else 0.0


def bbox_y0(block: V8Block) -> float:
    return block.bbox[1] if len(block.bbox) == 4 else 0.0


def bbox_y1(block: V8Block) -> float:
    return block.bbox[3] if len(block.bbox) == 4 else 0.0


def bbox_union(bboxes: list[list[float]]) -> list[float]:
    valid = [bbox for bbox in bboxes if len(bbox) == 4]
    if not valid:
        return []
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]


def text_from_any(value: Any) -> str:
    if isinstance(value, str):
        return normalize_space(value)
    if isinstance(value, dict):
        for key in ("content", "text", "latex", "html"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return normalize_space(inner)
        for key in ("spans", "lines"):
            inner = value.get(key)
            if isinstance(inner, list):
                return normalize_space(" ".join(text_from_any(item) for item in inner))
    if isinstance(value, list):
        return normalize_space(" ".join(text_from_any(item) for item in value))
    return ""


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_bbox(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        return []
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return []


def normalize_page_size(value: Any) -> list[float]:
    if isinstance(value, list) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            pass
    return []


def int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def count_by(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
