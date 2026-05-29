"""Algorithm and pseudocode region candidate extraction for v8 facts.

This module is intentionally sidecar-only. It reads v8 full observable facts
(`content_list_v8...json` and `document_ir.json`) and returns candidate records;
it does not mutate v8 facts, graph tensors, RenderTreeIR, or renderer behavior.
Legacy names such as ``source_v7_ids`` are treated as opaque provenance only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any, Iterable


ALGORITHM_CAPTION_RE = re.compile(
    r"^\s*(?P<label>Algorithm|Alg\.?|Procedure|Pseudocode|Method)\s*"
    r"(?P<number>(?:\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?|[IVXLCDM]+(?:\([a-zA-Z0-9]+\))?))?"
    r"\s*(?P<sep>[:.\-–—])?\s*(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)

FALSE_ALGORITHM_REFERENCE_RE = re.compile(
    r"^\s*(?:"
    r"(?:as\s+)?shown\s+in\s+|"
    r"see\s+|according\s+to\s+|refer\s+to\s+|"
    r"we\s+use\s+|we\s+apply\s+|using\s+|in\s+"
    r")?(?:Algorithm|Alg\.?|Procedure)\s+"
    r"(?:\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?|[IVXLCDM]+(?:\([a-zA-Z0-9]+\))?)"
    r"\s+(?:shows?|is\s+used|are\s+used|uses?|illustrates?|describes?|presents?|"
    r"can\s+be\s+seen|reports?|summari[sz]es?)\b",
    re.IGNORECASE,
)

ANCHOR_RE = re.compile(
    r"^\s*(Input|Output|Require|Ensure|Initialization|Initialisation|Initialize|Return)\s*:",
    re.IGNORECASE,
)
BODY_KEYWORD_RE = re.compile(
    r"\b(for|while|if|else|repeat|until|return|break|continue|function|procedure|"
    r"input|output|require|ensure|initialize|initialization|initialisation|end\s+if|"
    r"end\s+for|end\s+while)\b",
    re.IGNORECASE,
)
CONTROL_FLOW_LINE_RE = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(for|while|if|else|repeat|until|return|break|continue|function|procedure)\b",
    re.IGNORECASE,
)
STRONG_CONTROL_FLOW_LINE_RE = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(?:"
    r"if\b|while\b|return\b|function\b|procedure\b|repeat\b|until\b|else\b|"
    r"for\s+(?:each\b|all\b|[a-zA-Z_]\w*\s*(?:=|←|<-|\\gets|\\leftarrow|in\b|to\b))"
    r")",
    re.IGNORECASE,
)
LINE_NUMBER_RE = re.compile(r"^\s*(?:\d+[.)]|\(\d+\)|[ivxlcdm]+[.)])\s+\S+", re.IGNORECASE)
ASSIGNMENT_RE = re.compile(r"(:=|<-|←|=|\\leftarrow|\\gets)")
CODE_CONFIG_RE = re.compile(r"\b(choice|float|int|bool|range|true|false|null|none)\s*\(|\w+\s*=")
RISKY_UNICODE_RE = re.compile(r"[✓✗✘✔×□■●▲▶→⇒≤≥∈∞−∑∏∂∇ϵηρΓ]")
MATH_GLYPH_RE = re.compile(r"[α-ωΑ-ΩϵηρΓ∑∏∂∇∞≤≥≈≠]")
UNESCAPED_SPECIAL_RE = re.compile(r"(?<!\\)[#%&_]")


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class AlgorithmCandidate:
    candidate_id: str
    doc_id: str
    page_idx: int | None
    source_v8_ids: list[str]
    text: str
    text_preview: str
    bbox: list[float] | None
    candidate_type: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    compile_risk_flags: list[str] = field(default_factory=list)
    raw_type: str = ""
    current_role: str | None = None
    current_canonical_type: str | None = None
    source: str = "v8_fact"
    caption_number: str | None = None
    body_score: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["normalized_text"] = normalize_text(self.text)
        return payload


@dataclass
class AlgorithmRegionCandidate:
    region_id: str
    doc_id: str
    page_idx: int | None
    bbox_union: list[float] | None
    source_v8_ids: list[str]
    caption_candidate_id: str | None
    body_candidate_ids: list[str]
    region_type: str
    confidence: float
    evidence: list[str]
    compile_risk_flags: list[str]
    recommended_render_policy: str
    failure_hint: str | None = None
    text_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compact(text: Any, limit: int = 320) -> str:
    return " ".join(str(text or "").split())[:limit]


def normalize_text(text: Any) -> str:
    value = str(text or "").casefold()
    value = re.sub(r"\[math\]|\$[^$]*\$|\\[a-zA-Z]+", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_bbox(value: Any) -> BBox | None:
    if isinstance(value, dict):
        if all(key in value for key in ("x0", "y0", "x1", "y1")):
            return BBox(float(value["x0"]), float(value["y0"]), float(value["x1"]), float(value["y1"]))
    if isinstance(value, list):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            return BBox(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        boxes = [box for item in value if (box := parse_bbox(item)) is not None]
        if boxes:
            return bbox_union(boxes)
    return None


def bbox_union(boxes: Iterable[BBox]) -> BBox | None:
    box_list = list(boxes)
    if not box_list:
        return None
    return BBox(
        min(box.x0 for box in box_list),
        min(box.y0 for box in box_list),
        max(box.x1 for box in box_list),
        max(box.y1 for box in box_list),
    )


def bbox_to_list(box: BBox | None) -> list[float] | None:
    if box is None:
        return None
    return [round(box.x0, 3), round(box.y0, 3), round(box.x1, 3), round(box.y1, 3)]


def parse_algorithm_caption(text: Any) -> tuple[bool, str | None, list[str]]:
    value = " ".join(str(text or "").split()).strip()
    if not value or FALSE_ALGORITHM_REFERENCE_RE.match(value):
        return False, None, ["false_algorithm_reference"] if value else []
    match = ALGORITHM_CAPTION_RE.match(value)
    if not match:
        return False, None, []
    label = (match.group("label") or "").casefold().rstrip(".")
    number = match.group("number")
    sep = match.group("sep")
    body = (match.group("body") or "").strip()
    if label in {"algorithm", "alg", "procedure"} and not (number or sep):
        return False, None, []
    if label in {"method", "pseudocode"} and sep not in {":", ".", "-", "–", "—"}:
        return False, None, []
    if body.casefold().startswith(("shows ", "is used", "are used", "uses ", "can be seen")):
        return False, None, ["false_algorithm_reference"]
    evidence = [f"caption_label={label}"]
    if number:
        evidence.append(f"caption_number={number}")
    if sep:
        evidence.append("caption_separator")
    return True, number, evidence


def first_nonempty_line(text: Any) -> str:
    for line in str(text or "").splitlines():
        if line.strip():
            return line.strip()
    return str(text or "").strip()


def compile_risk_flags(text: Any) -> list[str]:
    value = str(text or "")
    flags: list[str] = []
    if RISKY_UNICODE_RE.search(value):
        flags.append("unicode_math_or_symbol")
    if UNESCAPED_SPECIAL_RE.search(value):
        flags.append("unescaped_special_char")
    if MATH_GLYPH_RE.search(value):
        flags.append("math_glyph_in_text_mode")
    if value.count("{") != value.count("}"):
        flags.append("unmatched_brace")
    if "\\includegraphics" in value and "algorithm" in value.casefold():
        flags.append("includegraphics_algorithm_crop")
    if CODE_CONFIG_RE.search(value) and UNESCAPED_SPECIAL_RE.search(value):
        flags.append("code_config_special_chars")
    return sorted(set(flags))


def algorithm_body_score(text: Any, *, raw_type: str = "", style_spans: list[dict[str, Any]] | None = None) -> tuple[int, list[str]]:
    value = str(text or "")
    raw = raw_type.casefold()
    lines = [line for line in value.splitlines() if line.strip()]
    keyword_hits = len(BODY_KEYWORD_RE.findall(value))
    anchor_hits = len([line for line in lines if ANCHOR_RE.match(line)])
    numbered_hits = len([line for line in lines if LINE_NUMBER_RE.match(line)])
    control_flow_hits = len([line for line in lines if CONTROL_FLOW_LINE_RE.match(line)])
    assignment_hits = len(ASSIGNMENT_RE.findall(value))
    short_code_lines = sum(1 for line in lines if len(line.strip()) <= 100 and BODY_KEYWORD_RE.search(line))
    evidence: list[str] = []
    score = 0
    if raw in {"algorithm", "code"}:
        score += 5
        evidence.append("raw_type_code_or_algorithm")
    if anchor_hits:
        score += 4 + min(3, anchor_hits)
        evidence.append(f"anchor_lines={anchor_hits}")
    if numbered_hits:
        score += 2 + min(3, numbered_hits)
        evidence.append(f"line_number_pattern={numbered_hits}")
    structural_context = bool(anchor_hits or numbered_hits or control_flow_hits >= 2 or raw in {"algorithm", "code"} or CODE_CONFIG_RE.search(value))
    if keyword_hits >= 3 and structural_context:
        score += min(6, keyword_hits)
        evidence.append(f"algorithm_keyword_hits={keyword_hits}")
    elif keyword_hits and (anchor_hits or numbered_hits or raw in {"algorithm", "code"}):
        score += keyword_hits
        evidence.append(f"algorithm_keyword_hits={keyword_hits}")
    if control_flow_hits and (assignment_hits or keyword_hits >= 2 or len(lines) <= 2):
        score += 5 + min(3, control_flow_hits)
        evidence.append(f"control_flow_lines={control_flow_hits}")
    if assignment_hits >= 2 and (anchor_hits or keyword_hits >= 2 or CODE_CONFIG_RE.search(value)):
        score += 2
        evidence.append(f"assignment_like_lines={assignment_hits}")
    if short_code_lines >= 2:
        score += 2
        evidence.append(f"short_code_like_lines={short_code_lines}")
    if CODE_CONFIG_RE.search(value) and (assignment_hits >= 2 or "," in value):
        score += 4
        evidence.append("code_config_syntax")
    if style_spans:
        fonts = " ".join(str(span.get("font_name") or "").casefold() for span in style_spans)
        if any(token in fonts for token in ("mono", "cour", "cmtt", "typewriter")):
            score += 3
            evidence.append("monospace_style")
        bold_labels = sum(
            1
            for span in style_spans
            if span.get("is_bold")
            and str(span.get("text") or "").strip().rstrip(":").casefold()
            in {"input", "output", "require", "ensure", "return", "initialization", "initialisation"}
        )
        if bold_labels:
            score += 2
            evidence.append(f"bold_pseudocode_labels={bold_labels}")
    return score, evidence


def detect_algorithm_candidates(
    content_payload: dict[str, Any] | None,
    document_ir: dict[str, Any] | None,
    *,
    doc_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Detect algorithm-like candidates and grouped regions from v8 facts."""

    doc = doc_id or str((content_payload or {}).get("doc_id") or (document_ir or {}).get("doc_id") or "")
    records = _iter_v8_records(content_payload or {}, document_ir or {}, doc)
    candidates: list[AlgorithmCandidate] = []
    seen: set[str] = set()
    for record in records:
        candidate = _candidate_from_record(record)
        if candidate is None or candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        candidates.append(candidate)

    regions = group_algorithm_regions(candidates, doc_id=doc)
    caption_candidates = [candidate.to_dict() for candidate in candidates if candidate.candidate_type == "ALGORITHM_CAPTION"]
    body_candidates = [
        candidate.to_dict()
        for candidate in candidates
        if candidate.candidate_type in {"ALGORITHM_BODY", "PSEUDOCODE_BODY", "CODE_CONFIG_BLOCK", "ALGORITHM_AS_PARAGRAPH", "ALGORITHM_AS_TABLE_LIKE"}
    ]
    risks = [
        {
            "doc_id": candidate.doc_id,
            "candidate_id": candidate.candidate_id,
            "page_idx": candidate.page_idx,
            "text": candidate.text_preview,
            "risk_reasons": candidate.compile_risk_flags,
            "candidate_type": candidate.candidate_type,
            "source_v8_ids": candidate.source_v8_ids,
        }
        for candidate in candidates
        if candidate.compile_risk_flags and candidate.candidate_type != "ALGORITHM_AS_PARAGRAPH"
    ]
    return {
        "algorithm_region_candidates": [region.to_dict() for region in regions],
        "algorithm_caption_candidates": caption_candidates,
        "algorithm_body_candidates": body_candidates,
        "all_candidates": [candidate.to_dict() for candidate in candidates],
        "pseudocode_compile_risk": risks,
    }


def group_algorithm_regions(candidates: list[AlgorithmCandidate], *, doc_id: str | None = None) -> list[AlgorithmRegionCandidate]:
    captions = [candidate for candidate in candidates if candidate.candidate_type == "ALGORITHM_CAPTION"]
    bodies: list[AlgorithmCandidate] = []
    for candidate in candidates:
        if candidate.candidate_type in {"ALGORITHM_BODY", "PSEUDOCODE_BODY", "CODE_CONFIG_BLOCK", "ALGORITHM_AS_TABLE_LIKE"}:
            bodies.append(candidate)
        elif candidate.candidate_type == "ALGORITHM_AS_PARAGRAPH" and candidate.body_score >= 11:
            bodies.append(candidate)
    used_bodies: set[str] = set()
    regions: list[AlgorithmRegionCandidate] = []

    for caption in captions:
        nearby = [
            body
            for body in bodies
            if body.candidate_id not in used_bodies and _nearby(caption, body, max_gap=180.0)
        ]
        nearby.sort(key=lambda body: (_same_page_penalty(caption, body), _vertical_gap(caption, body)))
        selected = nearby[:8]
        for body in selected:
            used_bodies.add(body.candidate_id)
        regions.append(_build_region(doc_id or caption.doc_id, caption, selected))

    remaining = [body for body in bodies if body.candidate_id not in used_bodies]
    for group in _group_body_candidates(remaining):
        regions.append(_build_region(doc_id or group[0].doc_id, None, group))
    return regions


def _iter_v8_records(content_payload: dict[str, Any], document_ir: dict[str, Any], doc_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(content_payload.get("items") or []):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or f"item_{idx:06d}")
        text = _combined_item_text(item)
        records.append(
            {
                "record_id": item_id,
                "doc_id": doc_id,
                "page_idx": item.get("page_idx"),
                "bbox": item.get("bbox"),
                "text": text,
                "raw_type": str(item.get("type") or item.get("raw_type") or ""),
                "current_role": item.get("layout_role") or item.get("canonical_type") or item.get("type"),
                "current_canonical_type": item.get("canonical_type") or item.get("content_list_type"),
                "source_v8_ids": [item_id],
                "style_spans": item.get("style_spans") or [],
                "source": "v8_content_item",
            }
        )
        for line_idx, line in enumerate(item.get("source_lines") or []):
            if not isinstance(line, dict) or not str(line.get("text") or "").strip():
                continue
            line_id = str(line.get("line_id") or f"{item_id}:line:{line_idx:04d}")
            records.append(
                {
                    "record_id": line_id,
                    "doc_id": doc_id,
                    "page_idx": line.get("page_idx", item.get("page_idx")),
                    "bbox": line.get("bbox"),
                    "text": line.get("text") or "",
                    "raw_type": str(item.get("type") or item.get("raw_type") or ""),
                    "current_role": "source_line",
                    "current_canonical_type": item.get("canonical_type") or item.get("content_list_type"),
                    "source_v8_ids": [line_id, item_id],
                    "style_spans": [],
                    "source": "v8_source_line",
                }
            )
    for idx, node in enumerate(document_ir.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        metadata = node.get("metadata") or {}
        node_id = str(node.get("node_id") or f"node_{idx:06d}")
        records.append(
            {
                "record_id": node_id,
                "doc_id": doc_id,
                "page_idx": node.get("page_idx"),
                "bbox": node.get("bboxes") or node.get("bbox"),
                "text": _combined_node_text(node),
                "raw_type": str(node.get("raw_type") or metadata.get("raw_type") or node.get("node_type") or ""),
                "current_role": metadata.get("layout_role") or metadata.get("canonical_type") or node.get("node_type"),
                "current_canonical_type": metadata.get("canonical_type"),
                "source_v8_ids": [node_id],
                "style_spans": node.get("spans") or [],
                "source": "document_ir",
            }
        )
    return records


def _candidate_from_record(record: dict[str, Any]) -> AlgorithmCandidate | None:
    text = str(record.get("text") or "")
    first = first_nonempty_line(text)
    is_caption, number, caption_evidence = parse_algorithm_caption(first)
    if "false_algorithm_reference" in caption_evidence:
        return AlgorithmCandidate(
            candidate_id=_candidate_id(record, "false_ref"),
            doc_id=str(record.get("doc_id") or ""),
            page_idx=_maybe_int(record.get("page_idx")),
            source_v8_ids=list(record.get("source_v8_ids") or []),
            text=text,
            text_preview=compact(text),
            bbox=bbox_to_list(parse_bbox(record.get("bbox"))),
            candidate_type="FALSE_ALGORITHM_REFERENCE",
            confidence=0.2,
            evidence=caption_evidence,
            compile_risk_flags=[],
            raw_type=str(record.get("raw_type") or ""),
            current_role=record.get("current_role"),
            current_canonical_type=record.get("current_canonical_type"),
            source=str(record.get("source") or "v8_fact"),
            caption_number=number,
        )
    raw_type = str(record.get("raw_type") or "")
    body_score, body_evidence = algorithm_body_score(text, raw_type=raw_type, style_spans=record.get("style_spans") or [])
    risks = compile_risk_flags(text)
    role_blob = " ".join(str(record.get(key) or "") for key in ("raw_type", "current_role", "current_canonical_type")).casefold()
    if is_caption:
        candidate_type = "ALGORITHM_CAPTION"
        confidence = 0.86 + min(0.1, 0.02 * len(caption_evidence))
        evidence = caption_evidence
    elif body_score >= 6:
        paragraph_role = (
            ("paragraph" in role_blob or "text" in role_blob or "body" in role_blob)
            and str(record.get("source") or "") != "v8_source_line"
        )
        strong_control = any(STRONG_CONTROL_FLOW_LINE_RE.match(line) for line in str(text or "").splitlines() if line.strip())
        if paragraph_role and body_score < 9 and not CODE_CONFIG_RE.search(text) and not strong_control:
            return None
        if "table" in role_blob:
            candidate_type = "ALGORITHM_AS_TABLE_LIKE"
        elif CODE_CONFIG_RE.search(text):
            candidate_type = "CODE_CONFIG_BLOCK"
        elif paragraph_role:
            candidate_type = "ALGORITHM_AS_PARAGRAPH"
        elif body_score >= 9:
            candidate_type = "ALGORITHM_BODY"
        else:
            candidate_type = "PSEUDOCODE_BODY"
        confidence = min(0.96, 0.45 + body_score / 20.0)
        evidence = body_evidence
    elif risks and ("algorithm" in text.casefold() or "input:" in text.casefold() or "output:" in text.casefold()):
        candidate_type = "COMPILE_RISK_PSEUDOCODE"
        confidence = 0.5
        evidence = ["compile_risk_only"]
    else:
        return None
    return AlgorithmCandidate(
        candidate_id=_candidate_id(record, candidate_type),
        doc_id=str(record.get("doc_id") or ""),
        page_idx=_maybe_int(record.get("page_idx")),
        source_v8_ids=list(record.get("source_v8_ids") or []),
        text=text,
        text_preview=compact(text, 500),
        bbox=bbox_to_list(parse_bbox(record.get("bbox"))),
        candidate_type=candidate_type,
        confidence=round(confidence, 3),
        evidence=evidence,
        compile_risk_flags=risks,
        raw_type=raw_type,
        current_role=record.get("current_role"),
        current_canonical_type=record.get("current_canonical_type"),
        source=str(record.get("source") or "v8_fact"),
        caption_number=number,
        body_score=body_score,
    )


def _combined_item_text(item: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("text", "content_list_text", "algorithm_caption", "code_caption", "caption"):
        value = item.get(key)
        if isinstance(value, list):
            pieces.extend(str(part) for part in value if str(part).strip())
        elif str(value or "").strip():
            pieces.append(str(value))
    for span in item.get("style_spans") or []:
        if isinstance(span, dict) and str(span.get("text") or "").strip():
            pieces.append(str(span.get("text")))
    return "\n".join(pieces)


def _combined_node_text(node: dict[str, Any]) -> str:
    pieces = [str(node.get("text") or "")]
    metadata = node.get("metadata") or {}
    for key in ("algorithm_caption", "code_caption", "caption", "figure_caption", "table_caption"):
        value = metadata.get(key)
        if isinstance(value, list):
            pieces.extend(str(part) for part in value if str(part).strip())
        elif str(value or "").strip():
            pieces.append(str(value))
    for span in node.get("spans") or []:
        if isinstance(span, dict) and str(span.get("text") or "").strip():
            pieces.append(str(span.get("text")))
    return "\n".join(piece for piece in pieces if piece)


def _candidate_id(record: dict[str, Any], kind: str) -> str:
    base = f"{record.get('doc_id')}|{record.get('record_id')}|{kind}|{compact(record.get('text'), 80)}"
    digest = hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"algcand_{digest}"


def _maybe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _box(candidate: AlgorithmCandidate) -> BBox | None:
    return parse_bbox(candidate.bbox)


def _same_page_penalty(left: AlgorithmCandidate, right: AlgorithmCandidate) -> int:
    return 0 if left.page_idx == right.page_idx else 1


def _vertical_gap(left: AlgorithmCandidate, right: AlgorithmCandidate) -> float:
    lb = _box(left)
    rb = _box(right)
    if lb is None or rb is None:
        return 9999.0
    if lb.y1 < rb.y0:
        return rb.y0 - lb.y1
    if rb.y1 < lb.y0:
        return lb.y0 - rb.y1
    return 0.0


def _x_overlap_ratio(left: AlgorithmCandidate, right: AlgorithmCandidate) -> float:
    lb = _box(left)
    rb = _box(right)
    if lb is None or rb is None:
        return 0.0
    overlap = max(0.0, min(lb.x1, rb.x1) - max(lb.x0, rb.x0))
    denom = max(1.0, min(lb.width, rb.width))
    return overlap / denom


def _nearby(left: AlgorithmCandidate, right: AlgorithmCandidate, *, max_gap: float) -> bool:
    if left.page_idx != right.page_idx:
        return False
    gap = _vertical_gap(left, right)
    return gap <= max_gap and (_x_overlap_ratio(left, right) >= 0.25 or gap <= 40)


def _group_body_candidates(bodies: list[AlgorithmCandidate]) -> list[list[AlgorithmCandidate]]:
    ordered = sorted(bodies, key=lambda item: (item.page_idx if item.page_idx is not None else 9999, (_box(item).y0 if _box(item) else 1e9), (_box(item).x0 if _box(item) else 1e9)))
    groups: list[list[AlgorithmCandidate]] = []
    for body in ordered:
        if not groups:
            groups.append([body])
            continue
        prev = groups[-1][-1]
        if _nearby(prev, body, max_gap=45.0):
            groups[-1].append(body)
        else:
            groups.append([body])
    return groups


def _build_region(doc_id: str, caption: AlgorithmCandidate | None, bodies: list[AlgorithmCandidate]) -> AlgorithmRegionCandidate:
    members = ([caption] if caption is not None else []) + bodies
    boxes = [box for member in members if (box := _box(member)) is not None]
    source_ids: list[str] = []
    risks: list[str] = []
    evidence: list[str] = []
    for member in members:
        source_ids.extend(member.source_v8_ids)
        risks.extend(member.compile_risk_flags)
        evidence.extend(member.evidence)
    source_ids = list(dict.fromkeys(source_ids))
    risks = sorted(set(risks))
    region_type = "algorithm" if caption and bodies else "pseudocode" if bodies else "uncertain"
    if bodies and any(body.candidate_type == "CODE_CONFIG_BLOCK" for body in bodies):
        region_type = "code_config"
    confidence = 0.0
    if caption:
        confidence += caption.confidence * 0.45
    if bodies:
        confidence += min(0.5, sum(body.confidence for body in bodies) / max(1, len(bodies)) * 0.55)
    if caption and bodies:
        confidence += 0.08
    confidence = min(0.98, max(0.3, confidence))
    if not bodies:
        failure_hint = "CAPTION_EXISTS_BODY_MISSING"
    elif not caption:
        failure_hint = "BODY_EXISTS_CAPTION_MISSING"
    else:
        failure_hint = None
    policy = recommended_render_policy(region_type, risks)
    region_key = f"{doc_id}|{caption.candidate_id if caption else ''}|{','.join(body.candidate_id for body in bodies)}"
    region_id = f"algregion_{hashlib.sha1(region_key.encode()).hexdigest()[:12]}"
    return AlgorithmRegionCandidate(
        region_id=region_id,
        doc_id=doc_id,
        page_idx=(caption.page_idx if caption is not None else bodies[0].page_idx if bodies else None),
        bbox_union=bbox_to_list(bbox_union(boxes)),
        source_v8_ids=source_ids,
        caption_candidate_id=caption.candidate_id if caption else None,
        body_candidate_ids=[body.candidate_id for body in bodies],
        region_type=region_type,
        confidence=round(confidence, 3),
        evidence=sorted(set(evidence)),
        compile_risk_flags=risks,
        recommended_render_policy=policy,
        failure_hint=failure_hint,
        text_preview=compact(" ".join(member.text for member in members), 500),
    )


def recommended_render_policy(region_type: str, risk_flags: list[str]) -> str:
    if "includegraphics_algorithm_crop" in risk_flags:
        return "crop_fallback"
    if risk_flags:
        return "verbatim_fallback"
    if region_type == "algorithm":
        return "algorithm_env_later"
    if region_type in {"pseudocode", "code_config"}:
        return "plain_text_fallback"
    return "diagnostic_only"
