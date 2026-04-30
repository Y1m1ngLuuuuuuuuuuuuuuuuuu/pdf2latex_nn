"""Extract and merge inline style spans from PDFs for content v3 nodes."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BOLD_FONT_PATTERNS = ("bold", "bd", "blk", "black", "heavy", "demi", "medi")
ITALIC_FONT_PATTERNS = ("ital", "italic", "oblique", "slant", "slanted")
CODE_FONT_PATTERNS = ("mono", "courier", "consolas", "typewriter", "cmtt", "lmmono")
MATH_FONT_PATTERNS = (
    "cmmi",
    "cmsy",
    "cmex",
    "msam",
    "msbm",
    "stixmath",
    "latinmodernmath",
    "xitsmath",
    "texgyrepagella-math",
    "texgyretermes-math",
    "texgyrebonum-math",
    "texgyreschola-math",
    "mathematicalpi",
    "mtmi",
    "mtsy",
)
MATH_CHARS = set("∑∫√∞≈≠≤≥±×÷∂∇αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ_^{}=<>|")
SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")


@dataclass(frozen=True)
class StyleConfig:
    clip_margin: float = 2.0
    size_bucket: float = 0.5


def enrich_content_v3_with_styles(input_path: Path, pdf_path: Path, output_path: Path, config: StyleConfig | None = None) -> dict[str, Any]:
    """Write a v3 JSON copy with style span sequences added per item."""

    import fitz

    cfg = config or StyleConfig()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Expected {input_path} to contain an items list")

    doc = fitz.open(pdf_path)
    try:
        for item in items:
            raw_spans = extract_raw_spans_for_item(doc, item, cfg)
            merged = merge_raw_spans(raw_spans)
            item["style_spans"] = merged
            item["style_baseline_size"] = baseline_font_size(merged)
            item["style_span_count"] = len(merged)
            item["style_extract_status"] = "ok" if merged else "empty"
    finally:
        doc.close()

    payload["schema_version"] = "content_v3_with_styles"
    payload["style_source_pdf"] = str(pdf_path)
    payload["style_config"] = {
        "clip_margin": cfg.clip_margin,
        "size_bucket": cfg.size_bucket,
        "math_font_patterns": list(MATH_FONT_PATTERNS),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def extract_raw_spans_for_item(doc: Any, item: dict[str, Any], config: StyleConfig) -> list[dict[str, Any]]:
    spans = []
    bboxes = list(iter_bbox_chunks(item.get("bbox")))
    pages = item.get("source_page_idxs")
    if not isinstance(pages, list) or len(pages) != len(bboxes):
        pages = [item.get("page_idx")] * len(bboxes)

    for page_idx, bbox in zip(pages, bboxes):
        if not isinstance(page_idx, int) or page_idx < 0 or page_idx >= len(doc):
            continue
        page = doc[page_idx]
        clip = normalized_bbox_to_page_rect(page, bbox, config.clip_margin)
        try:
            blocks = page.get_text("rawdict", clip=clip).get("blocks", [])
        except (ValueError, RuntimeError):
            continue
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = reconstruct_raw_span_text(span)
                    if not text.strip():
                        continue
                    font_name = normalize_font_name(str(span.get("font") or ""))
                    size = bucket_font_size(float(span.get("size") or 0.0), config.size_bucket)
                    flags = int(span.get("flags") or 0)
                    spans.append(
                        {
                            "text": text,
                            "font_name": font_name,
                            "font_size": size,
                            "is_bold": is_bold_font(flags, font_name),
                            "is_italic": is_italic_font(flags, font_name),
                            "is_inline_math": is_inline_math_font(font_name, text),
                            "is_inline_code": is_inline_code_font(font_name),
                        }
                    )
    return spans


def reconstruct_raw_span_text(span: dict[str, Any]) -> str:
    """Rebuild span text from raw characters, preserving inferred word spaces."""

    chars = span.get("chars")
    if not isinstance(chars, list) or not chars:
        return str(span.get("text") or "")

    size = float(span.get("size") or 0.0)
    gap_threshold = max(1.0, size * 0.22)
    parts: list[str] = []
    previous_x1: float | None = None
    previous_char = ""

    for char in chars:
        value = str(char.get("c") or "")
        if not value:
            continue
        bbox = char.get("bbox")
        x0 = float(bbox[0]) if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else None
        x1 = float(bbox[2]) if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else None
        if (
            parts
            and previous_x1 is not None
            and x0 is not None
            and x0 - previous_x1 > gap_threshold
            and previous_char not in " ([{/%“‘"
            and value not in " .,;:!?)]}%”’"
        ):
            parts.append(" ")
        parts.append(value)
        if x1 is not None:
            previous_x1 = x1
        previous_char = value
    return "".join(parts)


def merge_raw_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for span in spans:
        key = style_key(span)
        if current is not None and style_key(current) == key:
            current["text"] = join_span_text(current["text"], span["text"])
            current["char_count"] += len(span["text"])
            continue
        if current is not None:
            merged.append(current)
        current = {
            "text": span["text"],
            "font_name": span["font_name"],
            "font_size": span["font_size"],
            "is_bold": span["is_bold"],
            "is_italic": span["is_italic"],
            "is_inline_math": span["is_inline_math"],
            "is_inline_code": span["is_inline_code"],
            "char_count": len(span["text"]),
        }
    if current is not None:
        merged.append(current)
    return merged


def join_span_text(left: str, right: str) -> str:
    """Join adjacent PyMuPDF fragments while restoring readable word spacing."""

    if not left:
        return right
    if not right:
        return left
    if left.endswith((" ", "\n")) or right.startswith((" ", "\n")):
        return left + right

    lch = left[-1]
    rch = right[0]
    if lch == "-":
        return left + right
    if rch in ".,;:!?)]}%”’":
        return left + right
    if lch in "([{/%“‘":
        return left + right
    if _is_word_char(lch) and _is_word_char(rch):
        return left + " " + right
    if _is_word_char(lch) and rch in "([{“‘":
        return left + " " + right
    if lch == "." and len(left) >= 2 and left[-2].isdigit() and rch.isdigit():
        return left + right
    if lch in ".,;:!?)]}%”’" and _is_word_char(rch):
        return left + " " + right
    return left + right


def _is_word_char(char: str) -> bool:
    return char.isalnum()


def style_key(span: dict[str, Any]) -> tuple[Any, ...]:
    return (
        span.get("font_name"),
        span.get("font_size"),
        span.get("is_bold"),
        span.get("is_italic"),
        span.get("is_inline_math"),
        span.get("is_inline_code"),
    )


def baseline_font_size(spans: list[dict[str, Any]]) -> float | None:
    counts: Counter[float] = Counter()
    for span in spans:
        size = span.get("font_size")
        if size is None:
            continue
        counts[float(size)] += max(1, int(span.get("char_count") or len(str(span.get("text") or ""))))
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def normalize_font_name(font_name: str) -> str:
    name = font_name or ""
    name = SUBSET_PREFIX_RE.sub("", name)
    return name.lower()


def bucket_font_size(size: float, bucket: float = 0.5) -> float:
    if bucket <= 0:
        return size
    return round(size / bucket) * bucket


def is_bold_font(flags: int, font_name: str) -> bool:
    return bool(flags & 16) or any(pattern in font_name for pattern in BOLD_FONT_PATTERNS)


def is_italic_font(flags: int, font_name: str) -> bool:
    return bool(flags & 2) or any(pattern in font_name for pattern in ITALIC_FONT_PATTERNS)


def is_inline_code_font(font_name: str) -> bool:
    return any(pattern in font_name for pattern in CODE_FONT_PATTERNS)


def is_inline_math_font(font_name: str, text: str) -> bool:
    if any(pattern in font_name for pattern in MATH_FONT_PATTERNS):
        return True
    if "symbol" in font_name and looks_like_math_text(text):
        return True
    return False


def looks_like_math_text(text: str) -> bool:
    return any(char in MATH_CHARS for char in text)


def normalized_bbox_to_page_rect(page: Any, bbox: tuple[float, float, float, float], margin: float) -> Any:
    import fitz

    width = float(page.rect.width)
    height = float(page.rect.height)
    x0, y0, x1, y1 = bbox
    rect = fitz.Rect(
        x0 / 1000.0 * width - margin,
        y0 / 1000.0 * height - margin,
        x1 / 1000.0 * width + margin,
        y1 / 1000.0 * height + margin,
    )
    return rect & page.rect


def iter_bbox_chunks(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, list) or len(value) < 4:
        return []
    chunks = []
    usable_len = len(value) - (len(value) % 4)
    for idx in range(0, usable_len, 4):
        chunk = value[idx : idx + 4]
        chunks.append((float(chunk[0]), float(chunk[1]), float(chunk[2]), float(chunk[3])))
    return chunks
