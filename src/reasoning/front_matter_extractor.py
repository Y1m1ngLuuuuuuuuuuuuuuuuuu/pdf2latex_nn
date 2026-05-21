"""Deterministic front-matter extraction for decoder/generator use.

This module builds a conservative FrontMatterIR from the complete DocumentIR.
It does not mutate v7 records, does not enter the GNN view, and does not try to
solve exact author-affiliation-email linking.  Its job is to preserve visible
front matter and keep it out of the body heading tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Literal

from src.ir import BBox, BlockType, DocumentIR, DocumentNode, StyleSpan


FrontMatterRole = Literal[
    "TITLE",
    "AUTHOR",
    "AFFILIATION",
    "EMAIL",
    "ORCID",
    "FRONT_NOTE",
    "ABSTRACT_TITLE",
    "ABSTRACT_BODY",
    "BODY",
    "OTHER",
]

EMAIL_RE = re.compile(r"(?:[\w.+\-]+|\{[\w.,+\-\s]+\})@[\w.\-]+\.[A-Za-z]{2,}")
ABSTRACT_RE = re.compile(r"^\s*(?:abstract|摘要)\s*[:.\-—]?\s*(?P<body>.*)$", re.IGNORECASE)
BODY_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:\d+(?:\.\d+){0,2}\.?\s+[A-Z])|"
    r"(?:[IVXLCDM]{1,6}\.?\s+[A-Z])|"
    r"(?:[A-Z]\.\s+[A-Z][A-Za-z])|"
    r"(?:Introduction|Related\s+Work|Background|Methodology?|Experiments?|Results?|Discussion|Conclusion)"
    r")",
    re.IGNORECASE,
)
CAPTION_RE = re.compile(r"^\s*(?:fig(?:ure)?|table|tab\.?|algorithm)\s+[A-Z]?\d+", re.IGNORECASE)
REFERENCE_ITEM_RE = re.compile(r"^\s*(?:\[\d+\]|\d+\.\s+)[A-Z][A-Za-z,\s.'’-]+(?:19|20)\d{2}")
ORCID_RE = re.compile(r"\bORCID\b|\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", re.IGNORECASE)
AFFILIATION_RE = re.compile(
    r"\b("
    r"university|univ\.?|institute|inst\.?|department|dept\.?|school|college|"
    r"laboratory|lab\b|faculty|research\s+center|centre|academy|hospital|"
    r"microsoft|google|meta|openai|yale|zhejiang|tsinghua|tokyo|"
    r"大学|学院|研究所|实验室|中心|系|国家重点实验室"
    r")\b",
    re.IGNORECASE,
)
FRONT_NOTE_RE = re.compile(
    r"\b(equal\s+contribution|corresponding\s+author|data\s*&\s*models|code|project\s+page|funding)\b",
    re.IGNORECASE,
)
AUTHOR_SEPARATOR_RE = re.compile(r"\s*(?:,|;|\band\b|&|·|•)\s*", re.IGNORECASE)
VERB_LIKE_RE = re.compile(r"\b(is|are|was|were|has|have|shows?|proposes?|presents?|demonstrates?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class FrontMatterLine:
    line_id: str
    source_node_id: str
    text: str
    page_idx: int
    bbox: BBox | None
    font_size: float | None
    font_size_vs_body: float
    bold_ratio: float
    centeredness: float
    x_span_ratio: float
    y_position_norm: float
    line_order: float
    source: str
    role_scores: dict[str, float] = field(default_factory=dict)
    pred_role: FrontMatterRole = "OTHER"
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrontMatterSpan:
    role: FrontMatterRole
    text: str
    source_node_ids: list[str]
    line_ids: list[str]
    confidence: float
    bbox: BBox | None = None


@dataclass(frozen=True)
class FrontMatterAbstract:
    title: FrontMatterSpan | None
    body: FrontMatterSpan | None


@dataclass(frozen=True)
class FrontMatterRegion:
    page_idx: int
    start_order: float
    end_order: float
    body_start_order: float | None = None
    source_node_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FrontMatterIR:
    title: FrontMatterSpan | None
    authors: list[FrontMatterSpan]
    affiliations: list[FrontMatterSpan]
    emails: list[FrontMatterSpan]
    notes: list[FrontMatterSpan]
    abstract: FrontMatterAbstract | None
    misc: list[FrontMatterSpan]
    region: FrontMatterRegion | None
    lines: list[FrontMatterLine]
    warnings: list[str] = field(default_factory=list)

    @property
    def consumed_source_node_ids(self) -> set[str]:
        ids: set[str] = set()
        for span in self.all_spans():
            ids.update(span.source_node_ids)
        return ids

    def all_spans(self) -> list[FrontMatterSpan]:
        spans: list[FrontMatterSpan] = []
        if self.title is not None:
            spans.append(self.title)
        spans.extend(self.authors)
        spans.extend(self.affiliations)
        spans.extend(self.emails)
        spans.extend(self.notes)
        if self.abstract is not None:
            if self.abstract.title is not None:
                spans.append(self.abstract.title)
            if self.abstract.body is not None:
                spans.append(self.abstract.body)
        spans.extend(self.misc)
        return spans

    def negative_source_node_ids(self) -> set[str]:
        return self.consumed_source_node_ids

    def to_diagnostic(self) -> dict[str, Any]:
        return {
            "front_matter_region": None
            if self.region is None
            else {
                "page_idx": self.region.page_idx,
                "start_order": self.region.start_order,
                "end_order": self.region.end_order,
                "body_start_order": self.region.body_start_order,
                "source_node_ids": self.region.source_node_ids,
            },
            "lines": [
                {
                    "line_id": line.line_id,
                    "source_node_id": line.source_node_id,
                    "text": line.text,
                    "pred_role": line.pred_role,
                    "confidence": line.confidence,
                    "scores": line.role_scores,
                    "evidence": line.evidence,
                }
                for line in self.lines
            ],
            "groups": {
                "title": _span_diag(self.title),
                "authors": [_span_diag(span) for span in self.authors],
                "affiliations": [_span_diag(span) for span in self.affiliations],
                "emails": [_span_diag(span) for span in self.emails],
                "notes": [_span_diag(span) for span in self.notes],
                "abstract": None
                if self.abstract is None
                else {
                    "title": _span_diag(self.abstract.title),
                    "body": _span_diag(self.abstract.body),
                },
                "misc": [_span_diag(span) for span in self.misc],
            },
            "warnings": list(self.warnings),
        }


class FrontMatterLineBuilder:
    def __init__(self, document: DocumentIR) -> None:
        self.document = document
        self.body_font_size = _body_font_size(document.nodes)
        self.page_width_by_idx = {page.page_idx: float(page.width or 1000.0) for page in document.pages}
        self.page_height_by_idx = {page.page_idx: float(page.height or 1000.0) for page in document.pages}

    def build(self) -> list[FrontMatterLine]:
        body_start_order = _first_body_heading_order(self.document.nodes)
        front_nodes = [
            node
            for node in sorted(self.document.nodes, key=lambda item: (item.page_idx, item.reading_index))
            if self._node_in_front_region(node, body_start_order=body_start_order)
        ]
        lines: list[FrontMatterLine] = []
        for node in front_nodes:
            lines.extend(self._lines_from_node(node))
        return sorted(lines, key=lambda line: (line.page_idx, line.line_order, line.line_id))

    def _node_in_front_region(self, node: DocumentNode, *, body_start_order: int | None) -> bool:
        if node.page_idx > 1:
            return False
        if body_start_order is not None and node.reading_index >= body_start_order:
            return False
        layer = str(node.metadata.get("layout_layer") or "").casefold()
        role = str(node.metadata.get("layout_role") or "").casefold()
        if layer == "metadata_layer" or role in {
            "front_matter",
            "front_matter_title",
            "document_title",
            "paper_title",
            "author",
            "authors",
            "author_block",
            "affiliation",
            "email",
            "correspondence",
            "abstract",
            "abstract_title",
            "abstract_body",
        }:
            return True
        if node.page_idx == 0 and node.reading_index <= 30:
            text = _clean_space(node.text)
            if EMAIL_RE.search(text) or ABSTRACT_RE.match(text) or AFFILIATION_RE.search(text):
                return True
            if node.node_type == BlockType.TITLE and not CAPTION_RE.match(text):
                return True
            y = _node_y0_norm(node, self.page_height_by_idx.get(node.page_idx, 1000.0))
            return y <= 0.62
        return False

    def _lines_from_node(self, node: DocumentNode) -> list[FrontMatterLine]:
        span_lines = self._lines_from_positioned_spans(node)
        if span_lines:
            return span_lines
        raw_lines = [line.strip() for line in str(node.text or "").replace("\r", "\n").splitlines() if line.strip()]
        if not raw_lines:
            raw_lines = _split_inline_abstract_text(str(node.text or ""))
        elif len(raw_lines) == 1:
            raw_lines = _split_inline_abstract_text(raw_lines[0])
        if not raw_lines:
            return []
        result: list[FrontMatterLine] = []
        for index, text in enumerate(raw_lines):
            result.append(self._line_from_text(node, text=text, line_index=index, source="text_split"))
        return result

    def _lines_from_positioned_spans(self, node: DocumentNode) -> list[FrontMatterLine]:
        spans = [span for span in node.spans if span.bbox is not None and str(span.text or "").strip()]
        if not spans:
            return []
        buckets = _cluster_spans_into_lines(spans)
        result: list[FrontMatterLine] = []
        for index, bucket in enumerate(buckets):
            text = _join_spans(bucket)
            split_texts = _split_inline_abstract_text(text)
            if len(split_texts) > 1:
                for split_index, split_text in enumerate(split_texts):
                    result.append(
                        self._line_from_text(
                            node,
                            text=split_text,
                            line_index=index + (split_index / 10.0),
                            source="style_span_inline_split",
                            bbox=_union_bbox([span.bbox for span in bucket if span.bbox is not None]),
                            font_size=_median([span.font_size for span in bucket if span.font_size is not None]),
                            bold_ratio=_bold_ratio(bucket),
                        )
                    )
                continue
            result.append(
                self._line_from_text(
                    node,
                    text=text,
                    line_index=index,
                    source="style_span",
                    bbox=_union_bbox([span.bbox for span in bucket if span.bbox is not None]),
                    font_size=_median([span.font_size for span in bucket if span.font_size is not None]),
                    bold_ratio=_bold_ratio(bucket),
                )
            )
        return result

    def _line_from_text(
        self,
        node: DocumentNode,
        *,
        text: str,
        line_index: float,
        source: str,
        bbox: BBox | None = None,
        font_size: float | None = None,
        bold_ratio: float | None = None,
    ) -> FrontMatterLine:
        bbox = bbox or (node.bboxes[0] if node.bboxes else None)
        font_size = font_size or _node_font_size(node)
        bold_ratio = _node_bold_ratio(node) if bold_ratio is None else bold_ratio
        page_width = self.page_width_by_idx.get(node.page_idx, 1000.0)
        page_height = self.page_height_by_idx.get(node.page_idx, 1000.0)
        centeredness = _centeredness(bbox, page_width)
        x_span_ratio = _x_span_ratio(bbox, page_width)
        y_position_norm = _node_y0_norm(node, page_height) if bbox is None else bbox.y0 / max(page_height, 1.0)
        line_order = float(node.reading_index) + float(line_index) / 100.0
        return FrontMatterLine(
            line_id=f"{node.node_id}:line:{line_index}",
            source_node_id=node.node_id,
            text=_clean_space(text),
            page_idx=node.page_idx,
            bbox=bbox,
            font_size=font_size,
            font_size_vs_body=(font_size / self.body_font_size) if font_size and self.body_font_size else 0.0,
            bold_ratio=bold_ratio,
            centeredness=centeredness,
            x_span_ratio=x_span_ratio,
            y_position_norm=y_position_norm,
            line_order=line_order,
            source=source,
            evidence={},
        )


class RuleBasedFrontMatterSequenceTagger:
    def tag(self, lines: list[FrontMatterLine]) -> list[FrontMatterLine]:
        tagged: list[FrontMatterLine] = []
        state: FrontMatterRole = "TITLE"
        seen_title = False
        seen_abstract = False
        for index, line in enumerate(lines):
            scores, evidence = _role_scores(line, state=state, seen_title=seen_title, seen_abstract=seen_abstract, index=index)
            role = _decode_role(scores, state=state)
            confidence = _score_confidence(scores, role)
            if role == "TITLE":
                seen_title = True
                state = "TITLE"
            elif role in {"AUTHOR", "AFFILIATION", "EMAIL", "ORCID", "FRONT_NOTE"}:
                state = role
            elif role == "ABSTRACT_TITLE":
                seen_abstract = True
                state = "ABSTRACT_BODY"
            elif role == "ABSTRACT_BODY":
                state = "ABSTRACT_BODY"
            elif role == "BODY":
                state = "BODY"
            tagged.append(
                FrontMatterLine(
                    **{
                        **line.__dict__,
                        "role_scores": scores,
                        "pred_role": role,
                        "confidence": confidence,
                        "evidence": evidence,
                    }
                )
            )
        return tagged


class FrontMatterIRBuilder:
    def build(self, lines: list[FrontMatterLine]) -> FrontMatterIR:
        usable = [line for line in lines if line.pred_role not in {"BODY", "OTHER"} and line.text]
        warnings: list[str] = []
        if not usable:
            return FrontMatterIR(None, [], [], [], [], None, [], None, lines, warnings)
        title = _span_from_lines("TITLE", [line for line in usable if line.pred_role == "TITLE"])
        authors = _group_consecutive_lines([line for line in usable if line.pred_role == "AUTHOR"], "AUTHOR")
        affiliations = _group_consecutive_lines([line for line in usable if line.pred_role == "AFFILIATION"], "AFFILIATION")
        emails = _group_consecutive_lines([line for line in usable if line.pred_role in {"EMAIL", "ORCID"}], "EMAIL")
        notes = _group_consecutive_lines([line for line in usable if line.pred_role == "FRONT_NOTE"], "FRONT_NOTE")
        abstract_title = _span_from_lines("ABSTRACT_TITLE", [line for line in usable if line.pred_role == "ABSTRACT_TITLE"])
        abstract_body = _span_from_lines("ABSTRACT_BODY", [line for line in usable if line.pred_role == "ABSTRACT_BODY"])
        abstract = FrontMatterAbstract(abstract_title, abstract_body) if abstract_title or abstract_body else None
        misc = _group_consecutive_lines([line for line in usable if line.pred_role == "OTHER"], "OTHER")
        if title is None and (authors or affiliations or emails or abstract):
            warnings.append("missing_title_candidate")
        if authors and affiliations and _low_confidence(authors + affiliations):
            warnings.append("low_confidence_author_affiliation_split")
        region = FrontMatterRegion(
            page_idx=min(line.page_idx for line in usable),
            start_order=min(line.line_order for line in usable),
            end_order=max(line.line_order for line in usable),
            body_start_order=_first_line_order(lines, "BODY"),
            source_node_ids=sorted({line.source_node_id for line in usable}),
        )
        return FrontMatterIR(title, authors, affiliations, emails, notes, abstract, misc, region, lines, warnings)


def extract_front_matter(document: DocumentIR) -> FrontMatterIR:
    lines = FrontMatterLineBuilder(document).build()
    tagged = RuleBasedFrontMatterSequenceTagger().tag(lines)
    return FrontMatterIRBuilder().build(tagged)


def _role_scores(
    line: FrontMatterLine,
    *,
    state: FrontMatterRole,
    seen_title: bool,
    seen_abstract: bool,
    index: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    text = line.text
    lower = text.casefold()
    token_count = len(re.findall(r"\w+", text))
    has_email = bool(EMAIL_RE.search(text))
    is_abstract = bool(ABSTRACT_RE.match(text)) and len(text.split()) <= 4
    affiliation = bool(AFFILIATION_RE.search(text))
    front_note = bool(FRONT_NOTE_RE.search(text))
    caption = bool(CAPTION_RE.match(text))
    reference = bool(REFERENCE_ITEM_RE.match(text))
    sentence_like = bool(VERB_LIKE_RE.search(text)) and token_count >= 8
    author_like = _looks_like_author_line(text)
    title_like = (
        index <= 3
        and token_count >= 2
        and token_count <= 26
        and line.font_size_vs_body >= 1.05
        and line.centeredness >= 0.45
        and not has_email
        and not affiliation
    )
    scores = {
        "TITLE": 0.0,
        "AUTHOR": 0.0,
        "AFFILIATION": 0.0,
        "EMAIL": 0.0,
        "ORCID": 0.0,
        "FRONT_NOTE": 0.0,
        "ABSTRACT_TITLE": 0.0,
        "ABSTRACT_BODY": 0.0,
        "BODY": 0.0,
        "OTHER": 0.0,
    }
    if title_like:
        scores["TITLE"] += 3.0 + min(1.5, max(0.0, line.font_size_vs_body - 1.0) * 2.0)
    if seen_title and author_like and line.font_size_vs_body < 1.25:
        scores["TITLE"] -= 1.4
    if line.bold_ratio >= 0.5 and index <= 4 and not seen_title:
        scores["TITLE"] += 0.5
    if author_like and seen_title and not seen_abstract:
        scores["AUTHOR"] += 2.8
    if affiliation and seen_title and not seen_abstract:
        scores["AFFILIATION"] += 3.2
    if has_email:
        scores["EMAIL"] += 5.0
    if ORCID_RE.search(text):
        scores["ORCID"] += 4.0
    if front_note and not seen_abstract:
        scores["FRONT_NOTE"] += 2.5
    if is_abstract:
        scores["ABSTRACT_TITLE"] += 5.0
    elif seen_abstract and state == "ABSTRACT_BODY" and not BODY_HEADING_RE.match(text):
        scores["ABSTRACT_BODY"] += 3.0
    if BODY_HEADING_RE.match(text) and seen_abstract:
        scores["BODY"] += 5.0
    if not seen_title and index <= 1 and line.font_size_vs_body < 1.0:
        scores["TITLE"] -= 1.0
    if caption or reference:
        for role in ("TITLE", "AUTHOR", "AFFILIATION", "ABSTRACT_TITLE"):
            scores[role] -= 5.0
    if sentence_like and not seen_abstract:
        scores["TITLE"] -= 1.0
        scores["AUTHOR"] -= 1.0
    if state == "BODY":
        scores["BODY"] += 5.0
    if token_count > 35:
        scores["TITLE"] -= 2.0
        scores["AUTHOR"] -= 1.0
        if seen_abstract:
            scores["ABSTRACT_BODY"] += 1.0
    scores["OTHER"] += 0.1
    evidence = {
        "token_count": token_count,
        "has_email": has_email,
        "is_abstract_title": is_abstract,
        "affiliation_keyword": affiliation,
        "front_note_keyword": front_note,
        "author_like": author_like,
        "title_like": title_like,
        "caption_like": caption,
        "reference_like": reference,
        "sentence_like": sentence_like,
        "font_size_vs_body": line.font_size_vs_body,
        "bold_ratio": line.bold_ratio,
        "centeredness": line.centeredness,
    }
    return scores, evidence


def _decode_role(scores: dict[str, float], *, state: FrontMatterRole) -> FrontMatterRole:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    role, score = ordered[0]
    if score < 1.0:
        return "OTHER"
    if state == "ABSTRACT_BODY" and role in {"AUTHOR", "AFFILIATION", "TITLE"}:
        return "ABSTRACT_BODY"
    if state == "BODY" and role not in {"BODY", "OTHER"}:
        return "BODY"
    return role  # type: ignore[return-value]


def _score_confidence(scores: dict[str, float], role: FrontMatterRole) -> float:
    values = sorted(scores.values(), reverse=True)
    if not values:
        return 0.0
    margin = values[0] - (values[1] if len(values) > 1 else 0.0)
    return max(0.0, min(0.99, 0.45 + 0.12 * values[0] + 0.08 * margin))


def _looks_like_author_line(text: str) -> bool:
    value = _clean_space(text)
    if not value or EMAIL_RE.search(value) or AFFILIATION_RE.search(value) or FRONT_NOTE_RE.search(value):
        return False
    if len(value) > 180 or VERB_LIKE_RE.search(value):
        return False
    parts = [part for part in AUTHOR_SEPARATOR_RE.split(value) if part.strip()]
    if len(parts) >= 2:
        return sum(1 for part in parts if _name_like_token_sequence(part)) >= 2
    return _name_like_token_sequence(value) and len(value.split()) <= 6


def _name_like_token_sequence(text: str) -> bool:
    tokens = [token.strip(" ,.;:0123456789*†‡§¶") for token in str(text or "").split()]
    tokens = [token for token in tokens if token]
    if not 1 <= len(tokens) <= 5:
        return False
    capitalized = sum(1 for token in tokens if re.match(r"^[A-ZÀ-ÖØ-ÞĀ-ſ][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ'’-]*$", token))
    initials = sum(1 for token in tokens if re.match(r"^[A-Z]\.?$", token))
    return capitalized + initials >= max(1, min(2, len(tokens)))


def _split_inline_abstract_text(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    match = ABSTRACT_RE.match(value)
    if match and match.group("body"):
        return ["Abstract", match.group("body").strip()]
    return [value]


def _first_body_heading_order(nodes: list[DocumentNode]) -> int | None:
    for node in sorted(nodes, key=lambda item: (item.page_idx, item.reading_index)):
        if node.page_idx > 1:
            break
        text = _clean_space(node.text)
        layer = str(node.metadata.get("layout_layer") or "").casefold()
        role = str(node.metadata.get("layout_role") or "").casefold()
        if layer == "metadata_layer" or role in {"front_matter", "author", "affiliation", "abstract", "abstract_title", "abstract_body"}:
            continue
        if node.node_type == BlockType.TITLE and BODY_HEADING_RE.match(text) and not ABSTRACT_RE.match(text):
            return node.reading_index
    return None


def _span_from_lines(role: FrontMatterRole, lines: list[FrontMatterLine]) -> FrontMatterSpan | None:
    if not lines:
        return None
    ordered = sorted(lines, key=lambda line: line.line_order)
    return FrontMatterSpan(
        role=role,
        text="\n".join(line.text for line in ordered if line.text),
        source_node_ids=sorted({line.source_node_id for line in ordered}),
        line_ids=[line.line_id for line in ordered],
        confidence=sum(line.confidence for line in ordered) / max(1, len(ordered)),
        bbox=_union_bbox([line.bbox for line in ordered if line.bbox is not None]),
    )


def _group_consecutive_lines(lines: list[FrontMatterLine], role: FrontMatterRole) -> list[FrontMatterSpan]:
    if not lines:
        return []
    ordered = sorted(lines, key=lambda line: line.line_order)
    groups: list[list[FrontMatterLine]] = []
    current: list[FrontMatterLine] = []
    previous_order: float | None = None
    for line in ordered:
        if previous_order is not None and line.line_order - previous_order > 1.25 and current:
            groups.append(current)
            current = []
        current.append(line)
        previous_order = line.line_order
    if current:
        groups.append(current)
    return [span for group in groups if (span := _span_from_lines(role, group)) is not None]


def _low_confidence(spans: list[FrontMatterSpan]) -> bool:
    return any(span.confidence < 0.68 for span in spans)


def _first_line_order(lines: list[FrontMatterLine], role: FrontMatterRole) -> float | None:
    orders = [line.line_order for line in lines if line.pred_role == role]
    return min(orders) if orders else None


def _span_diag(span: FrontMatterSpan | None) -> dict[str, Any] | None:
    if span is None:
        return None
    return {
        "role": span.role,
        "text": span.text,
        "source_node_ids": span.source_node_ids,
        "line_ids": span.line_ids,
        "confidence": span.confidence,
        "bbox": None if span.bbox is None else span.bbox.to_list(),
    }


def _cluster_spans_into_lines(spans: list[StyleSpan]) -> list[list[StyleSpan]]:
    ordered = sorted(spans, key=lambda span: (_bbox_center_y(span.bbox), span.bbox.x0 if span.bbox else 0.0))
    buckets: list[list[StyleSpan]] = []
    centers: list[float] = []
    heights: list[float] = []
    for span in ordered:
        if span.bbox is None:
            continue
        center = _bbox_center_y(span.bbox)
        height = max(span.bbox.y1 - span.bbox.y0, 1.0)
        target = None
        for idx, bucket_center in enumerate(centers):
            if abs(bucket_center - center) <= max(2.0, 0.45 * max(heights[idx], height)):
                target = idx
                break
        if target is None:
            buckets.append([span])
            centers.append(center)
            heights.append(height)
        else:
            buckets[target].append(span)
            centers[target] = sum(_bbox_center_y(item.bbox) for item in buckets[target] if item.bbox) / len(buckets[target])
            heights[target] = max(heights[target], height)
    return [bucket for _center, bucket in sorted(zip(centers, buckets), key=lambda item: item[0])]


def _join_spans(spans: list[StyleSpan]) -> str:
    ordered = sorted(spans, key=lambda span: span.bbox.x0 if span.bbox else 0.0)
    pieces: list[str] = []
    previous: BBox | None = None
    for span in ordered:
        text = str(span.text or "")
        if not text.strip():
            continue
        if pieces and previous is not None and span.bbox is not None:
            gap = span.bbox.x0 - previous.x1
            if gap > max(1.5, (span.font_size or 9.0) * 0.20) and not text.startswith((" ", ",", ".", ")", ":")):
                pieces.append(" ")
        pieces.append(text)
        previous = span.bbox
    return _clean_space("".join(pieces))


def _body_font_size(nodes: list[DocumentNode]) -> float:
    values = [
        _node_font_size(node)
        for node in nodes
        if node.node_type == BlockType.TEXT
        and str(node.metadata.get("layout_layer") or "").casefold() not in {"metadata_layer", "noise_layer"}
        and _node_font_size(node) is not None
    ]
    return float(median(values)) if values else 10.0


def _node_font_size(node: DocumentNode) -> float | None:
    span_values = [float(span.font_size) for span in node.spans if span.font_size is not None and str(span.text or "").strip()]
    if span_values:
        return float(median(span_values))
    for key in ("style_baseline_size", "font_size", "median_font_size"):
        value = node.metadata.get(key) or node.features.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _node_bold_ratio(node: DocumentNode) -> float:
    spans = [span for span in node.spans if str(span.text or "").strip()]
    if spans:
        return sum(1 for span in spans if span.is_bold or "bold" in str(span.font_name or "").casefold()) / len(spans)
    value = node.features.get("bold_ratio") or node.metadata.get("bold_ratio")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _bold_ratio(spans: list[StyleSpan]) -> float:
    usable = [span for span in spans if str(span.text or "").strip()]
    return sum(1 for span in usable if span.is_bold or "bold" in str(span.font_name or "").casefold()) / max(1, len(usable))


def _bbox_center_y(bbox: BBox | None) -> float:
    return ((bbox.y0 + bbox.y1) / 2.0) if bbox else 0.0


def _centeredness(bbox: BBox | None, page_width: float) -> float:
    if bbox is None or page_width <= 0:
        return 0.0
    center = (bbox.x0 + bbox.x1) / 2.0
    return max(0.0, 1.0 - abs(center - (page_width / 2.0)) / max(page_width / 2.0, 1.0))


def _x_span_ratio(bbox: BBox | None, page_width: float) -> float:
    if bbox is None or page_width <= 0:
        return 0.0
    return max(0.0, min(1.0, (bbox.x1 - bbox.x0) / page_width))


def _node_y0_norm(node: DocumentNode, page_height: float) -> float:
    if not node.bboxes or page_height <= 0:
        return 0.0
    return node.bboxes[0].y0 / page_height


def _union_bbox(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    return BBox(
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def _median(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return float(median(usable)) if usable else None


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()
