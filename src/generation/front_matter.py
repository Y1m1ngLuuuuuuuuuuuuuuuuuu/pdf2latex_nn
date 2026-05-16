"""Front-matter rendering helpers for original-like LaTeX output.

The PDF front matter is often one visual block: MinerU may provide one bbox for
all authors and affiliations, while PyMuPDF spans still preserve line-level
geometry.  This module reconstructs that visual grouping without trying to
solve author semantics perfectly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.generation.latex_renderer import escape_latex, render_text_with_inline_latex
from src.ir import BBox, BlockType, DocumentNode, StyleProfile, StyleSpan


EMAIL_RE = re.compile(r"[\w.+\-]+@[\w.\-]+\.[A-Za-z]{2,}")
AFFILIATION_RE = re.compile(
    r"\b("
    r"affiliation|university|college|institute|department|school|faculty|"
    r"laboratory|lab\b|center|centre|academy|hospital|corporation|inc\.?|ltd\.?"
    r")\b",
    re.IGNORECASE,
)
CORRESPONDING_RE = re.compile(r"\b(corresponding|correspondence|contact)\b", re.IGNORECASE)
AUTHOR_BREAK_RE = re.compile(
    r"\s+(?=(?:\d+\s*)?(?:Department|School|College|University|Institute|Laboratory|Faculty|Center|Centre)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FrontMatterLine:
    text: str
    role: str
    font_size: float | None = None


@dataclass
class _LineBucket:
    spans: list[StyleSpan]
    y_center: float
    height: float


def render_document_title_original_like(
    text: str,
    source_nodes: list[DocumentNode],
    style: StyleProfile | None = None,
) -> str:
    value = _clean_inline_space(text)
    if not value:
        return ""
    font_size = _front_matter_font_size(source_nodes, style, multiplier=1.65, minimum=14.0)
    line_height = font_size * 1.28
    return "\n".join(
        [
            r"\begin{center}",
            rf"{{\fontsize{{{font_size:.2f}pt}}{{{line_height:.2f}pt}}\selectfont\bfseries\begin{{minipage}}{{0.94\textwidth}}\centering {render_text_with_inline_latex(value)}\par\end{{minipage}}}}",
            r"\end{center}",
        ]
    )


def render_author_block_original_like(
    text: str,
    source_nodes: list[DocumentNode],
    style: StyleProfile | None = None,
) -> str:
    author_nodes = [node for node in source_nodes if _classify_node_as_author_box(node)]
    grid = _render_author_grid(author_nodes, style)
    if grid:
        return grid

    lines = author_lines_from_nodes(author_nodes)
    if not lines:
        lines = author_lines_from_text(text)
    if not lines:
        return ""

    body: list[str] = [r"\begin{center}", r"\begin{minipage}{0.94\textwidth}", r"\centering"]
    for index, line in enumerate(lines):
        rendered = _render_front_matter_line(line, style)
        if not rendered:
            continue
        spacing = _line_spacing_after(line, next_line=lines[index + 1] if index + 1 < len(lines) else None)
        body.append(rendered + spacing)
    body.extend([r"\end{minipage}", r"\end{center}"])
    return "\n".join(body)


def _render_author_grid(source_nodes: list[DocumentNode], style: StyleProfile | None) -> str:
    boxes = [(node, node.bboxes[0]) for node in source_nodes if node.bboxes and _classify_node_as_author_box(node)]
    if len(boxes) < 2:
        return ""
    rows = _cluster_author_boxes_into_rows(boxes)
    if not rows:
        return ""
    body: list[str] = [r"\begin{center}"]
    for row in rows:
        columns = len(row)
        column_width = min(0.94 / max(columns, 1), 0.30)
        body.append(r"\noindent")
        cells: list[str] = []
        for node, _box in row:
            cell_lines = author_lines_from_nodes([node]) or author_lines_from_text(node.text)
            cell_body = r"\\".join(_render_front_matter_line(line, style) for line in cell_lines if line.text.strip())
            if not cell_body:
                continue
            cells.append(
                rf"\begin{{minipage}}[t]{{{column_width:.3f}\textwidth}}\centering {cell_body}\end{{minipage}}"
            )
        if cells:
            body.append(r"\hfill ".join(cells))
            body.append(r"\par\vspace{4pt}")
    body.append(r"\end{center}")
    return "\n".join(body)


def _classify_node_as_author_box(node: DocumentNode) -> bool:
    if node.node_type == BlockType.TITLE:
        return False
    layer = str(node.metadata.get("layout_layer") or "").casefold()
    role = str(node.metadata.get("layout_role") or "").casefold()
    if role in {"title", "document_title", "paper_title"}:
        return False
    if layer == "metadata_layer" and role in {"affiliation", "author", "authors", "date", "email", "correspondence"}:
        return True
    return _classify_author_line(node.text) in {"author", "affiliation", "email", "correspondence"}


def _cluster_author_boxes_into_rows(
    boxes: list[tuple[DocumentNode, BBox]],
) -> list[list[tuple[DocumentNode, BBox]]]:
    ordered = sorted(boxes, key=lambda item: (((item[1].y0 + item[1].y1) / 2.0), item[1].x0))
    rows: list[list[tuple[DocumentNode, BBox]]] = []
    for item in ordered:
        _node, box = item
        y_center = (box.y0 + box.y1) / 2.0
        height = max(box.y1 - box.y0, 1.0)
        target: list[tuple[DocumentNode, BBox]] | None = None
        for row in rows:
            row_center = sum((candidate_box.y0 + candidate_box.y1) / 2.0 for _candidate, candidate_box in row) / len(row)
            row_height = max(candidate_box.y1 - candidate_box.y0 for _candidate, candidate_box in row)
            if abs(row_center - y_center) <= max(8.0, 0.8 * max(row_height, height)):
                target = row
                break
        if target is None:
            rows.append([item])
        else:
            target.append(item)
    return [sorted(row, key=lambda item: item[1].x0) for row in rows]


def author_lines_from_nodes(nodes: list[DocumentNode]) -> list[FrontMatterLine]:
    positioned_spans = [span for node in nodes for span in node.spans if span.bbox is not None and (span.text or "").strip()]
    if positioned_spans:
        buckets = _cluster_spans_into_visual_lines(positioned_spans)
        lines: list[FrontMatterLine] = []
        for bucket in buckets:
            text = _join_spans_on_line(bucket.spans)
            if text:
                lines.append(FrontMatterLine(text=text, role=_classify_author_line(text), font_size=_median_font_size(bucket.spans)))
        return _dedupe_lines(lines)

    # Some MinerU/PyMuPDF paths preserve font/style but not per-span bbox.  This
    # is common in author blocks: bold spans are names, while the following
    # roman span contains affiliation and email.  Using the raw node text would
    # split "Dartmouth College" at "College" and pollute the author name.
    styled_lines: list[FrontMatterLine] = []
    for node in nodes:
        styled_lines.extend(_author_lines_from_style_spans(node.spans, fallback_text=node.text))
    return _dedupe_lines(styled_lines)


def author_lines_from_text(text: str) -> list[FrontMatterLine]:
    value = str(text or "").strip()
    if not value:
        return []
    raw_lines = [line.strip() for line in re.split(r"\n+", value) if line.strip()]
    if len(raw_lines) <= 1:
        raw_lines = _soft_split_author_block(value)
    return _dedupe_lines(
        [FrontMatterLine(text=_clean_inline_space(line), role=_classify_author_line(line)) for line in raw_lines if line.strip()]
    )


def _cluster_spans_into_visual_lines(spans: list[StyleSpan]) -> list[_LineBucket]:
    ordered = sorted(spans, key=lambda span: (_span_y_center(span), span.bbox.x0 if span.bbox else 0.0))
    buckets: list[_LineBucket] = []
    for span in ordered:
        if span.bbox is None:
            continue
        y_center = _span_y_center(span)
        height = max(span.bbox.y1 - span.bbox.y0, 1.0)
        target = _nearest_line_bucket(buckets, y_center, height)
        if target is None:
            buckets.append(_LineBucket(spans=[span], y_center=y_center, height=height))
            continue
        target.spans.append(span)
        count = len(target.spans)
        target.y_center = ((target.y_center * (count - 1)) + y_center) / count
        target.height = max(target.height, height)
    return sorted(buckets, key=lambda bucket: bucket.y_center)


def _nearest_line_bucket(buckets: list[_LineBucket], y_center: float, height: float) -> _LineBucket | None:
    best: tuple[float, _LineBucket] | None = None
    for bucket in buckets:
        tolerance = max(2.0, 0.45 * (bucket.height + height) / 2.0)
        distance = abs(bucket.y_center - y_center)
        if distance <= tolerance and (best is None or distance < best[0]):
            best = (distance, bucket)
    return best[1] if best is not None else None


def _join_spans_on_line(spans: list[StyleSpan]) -> str:
    ordered = sorted(spans, key=lambda span: span.bbox.x0 if span.bbox else 0.0)
    pieces: list[str] = []
    previous_box: BBox | None = None
    previous_size: float | None = None
    for span in ordered:
        text = str(span.text or "")
        if not text.strip():
            continue
        if pieces and previous_box is not None and span.bbox is not None:
            gap = span.bbox.x0 - previous_box.x1
            space_threshold = max(1.2, (previous_size or span.font_size or 9.0) * 0.18)
            if gap > space_threshold and not pieces[-1].endswith((" ", "-", "/", "(")) and not text.startswith((" ", ",", ".", ")", ":", ";")):
                pieces.append(" ")
        pieces.append(text)
        previous_box = span.bbox
        previous_size = span.font_size
    return _clean_inline_space("".join(pieces))


def _soft_split_author_block(text: str) -> list[str]:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return []
    structured = _split_author_affiliation_email_text(value)
    if len(structured) > 1:
        return [line.text for line in structured]
    protected = AUTHOR_BREAK_RE.sub("\n", value)
    protected = re.sub(r"\s+(?=(?:e-?mail|email|emails?)\s*:)", "\n", protected, flags=re.IGNORECASE)
    protected = re.sub(r"\s+(?=\*?\s*Corresponding\b)", "\n", protected, flags=re.IGNORECASE)
    lines = [line.strip(" ;,") for line in protected.splitlines() if line.strip(" ;,")]
    if len(lines) > 1:
        return lines
    return [value]


def _author_lines_from_style_spans(spans: list[StyleSpan], *, fallback_text: str = "") -> list[FrontMatterLine]:
    usable = [span for span in spans if (span.text or "").strip()]
    if not usable:
        return []
    author_spans: list[StyleSpan] = []
    remainder: list[str] = []
    seen_remainder = False
    for span in usable:
        text = _clean_inline_space(span.text)
        if not text:
            continue
        if not seen_remainder and _span_looks_like_author_name(span, text):
            author_spans.append(span)
            continue
        seen_remainder = True
        remainder.append(text)

    lines: list[FrontMatterLine] = []
    author_text = _clean_inline_space(" ".join(span.text for span in author_spans))
    if author_text:
        lines.append(FrontMatterLine(text=author_text, role="author", font_size=_median_font_size(author_spans)))

    remainder_text = _clean_inline_space(" ".join(remainder))
    if remainder_text:
        lines.extend(_split_affiliation_email_text(remainder_text))

    if lines:
        return lines
    return _split_author_affiliation_email_text(fallback_text)


def _span_looks_like_author_name(span: StyleSpan, text: str) -> bool:
    if EMAIL_RE.search(text) or AFFILIATION_RE.search(text):
        return False
    font_name = str(span.font_name or "").casefold()
    return bool(span.is_bold or "bold" in font_name or "cmbx" in font_name or font_name.endswith("bd"))


def _split_author_affiliation_email_text(text: str) -> list[FrontMatterLine]:
    value = _clean_inline_space(text)
    if not value:
        return []
    before_email, email_text = _split_email_tail(value)
    author_text, affiliation_text = _split_author_affiliation_prefix(before_email)
    lines: list[FrontMatterLine] = []
    if author_text:
        lines.append(FrontMatterLine(text=author_text, role="author"))
    if affiliation_text:
        lines.append(FrontMatterLine(text=affiliation_text, role="affiliation"))
    if email_text:
        lines.append(FrontMatterLine(text=email_text, role="email"))
    return lines


def _split_affiliation_email_text(text: str) -> list[FrontMatterLine]:
    affiliation, email_text = _split_email_tail(text)
    lines: list[FrontMatterLine] = []
    if affiliation:
        lines.append(FrontMatterLine(text=affiliation, role="affiliation"))
    if email_text:
        lines.append(FrontMatterLine(text=email_text, role="email"))
    return lines


def _split_email_tail(text: str) -> tuple[str, str]:
    value = _clean_inline_space(text)
    matches = list(EMAIL_RE.finditer(value))
    if not matches:
        return value, ""
    first = matches[0]
    prefix = value[: first.start()].strip(" ,;")
    emails = " ".join(match.group(0) for match in matches)
    return prefix, emails


def _split_author_affiliation_prefix(text: str) -> tuple[str, str]:
    value = _clean_inline_space(text)
    if not value:
        return "", ""
    tokens = value.split()
    lower_tokens = [token.strip(",.;:").casefold() for token in tokens]
    institution_keywords = {
        "university",
        "college",
        "institute",
        "department",
        "school",
        "faculty",
        "laboratory",
        "lab",
        "center",
        "centre",
        "academy",
        "hospital",
    }
    keyword_index = next((index for index, token in enumerate(lower_tokens) if token in institution_keywords), None)
    if keyword_index is None:
        return value, ""
    if keyword_index == 0:
        return "", value
    # For suffix-style institutions ("New York University", "Michigan State
    # University", "Dartmouth College"), include the capitalized institution
    # prefix instead of leaving it attached to the author name.  Two-token
    # author names are by far the dominant case, so never steal below index 2.
    start = keyword_index
    if keyword_index >= 4:
        start = keyword_index - 2
    elif keyword_index >= 3:
        start = keyword_index - 1
    elif keyword_index >= 2:
        start = keyword_index - 1
    start = max(start, 0)
    author = " ".join(tokens[:start]).strip(" ,;")
    affiliation = " ".join(tokens[start:]).strip(" ,;")
    if not author:
        return "", value
    return author, affiliation


def _classify_author_line(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return "author"
    if CORRESPONDING_RE.search(value):
        return "correspondence"
    if EMAIL_RE.search(value):
        return "email"
    if AFFILIATION_RE.search(value):
        return "affiliation"
    if re.match(r"^[*\d,\s]+(?:Department|School|College|University|Institute|Laboratory|Faculty|Center|Centre)\b", value, re.IGNORECASE):
        return "affiliation"
    return "author"


def _render_front_matter_line(line: FrontMatterLine, style: StyleProfile | None) -> str:
    text = _clean_inline_space(line.text)
    if not text:
        return ""
    if line.role == "email":
        return rf"{{\small\texttt{{{escape_latex(text)}}}}}"
    if line.role == "correspondence":
        return rf"{{\footnotesize {render_text_with_inline_latex(text)}}}"
    if line.role == "affiliation":
        return rf"{{\small {render_text_with_inline_latex(text)}}}"
    font_size = line.font_size or _style_body_font_size(style) or 10.0
    author_size = max(font_size, (_style_body_font_size(style) or font_size) * 1.05)
    return rf"{{\fontsize{{{author_size:.2f}pt}}{{{(author_size * 1.2):.2f}pt}}\selectfont {render_text_with_inline_latex(text)}}}"


def _line_spacing_after(line: FrontMatterLine, *, next_line: FrontMatterLine | None) -> str:
    if next_line is None:
        return ""
    if line.role == "author" and next_line.role != "author":
        return r"\\[4pt]"
    if line.role != next_line.role:
        return r"\\[2pt]"
    return r"\\"


def _front_matter_font_size(
    source_nodes: list[DocumentNode],
    style: StyleProfile | None,
    *,
    multiplier: float,
    minimum: float,
) -> float:
    span_sizes = [
        float(span.font_size)
        for node in source_nodes
        for span in node.spans
        if span.font_size is not None and (span.text or "").strip()
    ]
    if span_sizes:
        return max(max(span_sizes), minimum)
    body = _style_body_font_size(style) or 10.0
    return max(body * multiplier, minimum)


def _style_body_font_size(style: StyleProfile | None) -> float | None:
    if style is None:
        return None
    value = (style.renderer_options or {}).get("body_font_size")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _median_font_size(spans: list[StyleSpan]) -> float | None:
    values = sorted(float(span.font_size) for span in spans if span.font_size is not None)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _span_y_center(span: StyleSpan) -> float:
    if span.bbox is None:
        return 0.0
    return (span.bbox.y0 + span.bbox.y1) / 2.0


def _dedupe_lines(lines: list[FrontMatterLine]) -> list[FrontMatterLine]:
    result: list[FrontMatterLine] = []
    seen: set[str] = set()
    for line in lines:
        key = re.sub(r"\s+", " ", line.text).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(line)
    return result


def _clean_inline_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()
