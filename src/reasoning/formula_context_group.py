"""Decoder-side formula and theorem context grouping primitives.

These structures are deliberately IR-side only.  They describe how formula-like
fragments should be interpreted by the decoder/renderer without mutating full
v7 records, GNN views, or graph labels.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


FormulaContextType = Literal[
    "INLINE_MATH_ATTACHMENT",
    "DISPLAY_MATH_CONTEXT",
    "WHERE_CLAUSE_CONTEXT",
    "THEOREM_PROOF_CONTEXT",
    "FORMULA_OCR_ARTIFACT",
    "ORDINARY_BODY_REORDER",
    "UNCERTAIN",
]

FormulaEvidenceSource = Literal[
    "mineru_span",
    "content_list_equation",
    "document_ir_formula_metadata",
    "regex_only",
    "mixed",
]
FormulaConfidenceTier = Literal["high", "medium", "low", "diagnostic_only"]


THEOREM_KEYWORD_RE = re.compile(
    r"^\s*(?:theorem|lemma|proof|definition|corollary|proposition|remark|example|claim|assumption)"
    r"(?:\s+[A-Za-z]?\d+(?:\.\d+)*|\s*\([^)]+\))?\s*[:.]",
    re.IGNORECASE,
)
THEOREM_PREFIX_RE = re.compile(
    r"^\s*(?P<label>(?:theorem|lemma|proof|definition|corollary|proposition|remark|example|claim|assumption)"
    r"(?:\s+[A-Za-z]?\d+(?:\.\d+)*|\s*\([^)]+\))?\s*[:.])",
    re.IGNORECASE,
)
WHERE_CLAUSE_RE = re.compile(r"^\s*(?:where|in which|subject to|s\.t\.|such that|其中[，,]?)\b", re.IGNORECASE)
WHERE_NEGATIVE_START_RE = re.compile(
    r"^\s*(?:with|within|without|whereas|which|while|when|we|whose)\b",
    re.IGNORECASE,
)
DISPLAY_MATH_ENV_RE = re.compile(r"\\begin\{(?:equation|align|gather|multline|split)\*?\}", re.IGNORECASE)
INLINE_MATH_RE = re.compile(r"\$[^$]{1,120}\$|\\\([^)]{1,120}\\\)|\\(?:alpha|beta|gamma|lambda|theta|sigma|frac|sum|int)\b")
EQUATION_NUMBER_RE = re.compile(r"^\s*\(?[A-Za-z]?\d+(?:\.\d+)*\)?\s*$")
MATH_SYMBOL_CHARS = set("=<>^_{}[]()+*/|∑∫≤≥≈≠±×·")


@dataclass(frozen=True)
class FormulaContextEvidence:
    normalized_text: str
    token_count: int
    symbol_count: int
    semantic_channel: str | None = None
    starts_where_clause: bool = False
    theorem_like: bool = False
    display_math_env: bool = False
    inline_math_marker: bool = False
    equation_number_like: bool = False
    short_fragment: bool = False
    local_formula_context: bool = False
    negative_where_start: bool = False
    evidence_source: FormulaEvidenceSource = "regex_only"
    formula_confidence: str | None = None
    formula_context_role: str | None = None
    has_mineru_formula_evidence: bool = False
    confidence_tier: FormulaConfidenceTier = "low"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InlineMathAttachment:
    paragraph_node_id: str
    inline_formula_node_ids: list[str]
    insertion_position: Literal["before", "after", "between", "uncertain"] = "uncertain"
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TheoremProofContext:
    label_text: str
    body_node_ids: list[str]
    source_v7_ids: list[str]
    render_policy: Literal["bold_inline_label", "theorem_like_block", "plain_paragraph_fallback"] = (
        "bold_inline_label"
    )
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WhereClauseContext:
    lead_in_node_ids: list[str]
    display_math_node_ids: list[str]
    where_clause_node_ids: list[str]
    render_policy: Literal["paragraph_displaymath_where"] = "paragraph_displaymath_where"
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FormulaContextGroup:
    group_id: str
    context_type: FormulaContextType
    source_v7_ids: list[str]
    text_before_ids: list[str] = field(default_factory=list)
    formula_ids: list[str] = field(default_factory=list)
    text_after_ids: list[str] = field(default_factory=list)
    theorem_label_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    render_policy: str = "plain_paragraph_fallback"
    confidence_tier: FormulaConfidenceTier = "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_context_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def context_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def math_symbol_count(text: str) -> int:
    return sum(1 for char in str(text or "") if char in MATH_SYMBOL_CHARS)


def record_text(record: Any) -> str:
    if hasattr(record, "text"):
        return str(getattr(record, "text") or "")
    if isinstance(record, dict):
        return str(record.get("text") or record.get("raw_text") or record.get("content") or "")
    return str(record or "")


def record_id(record: Any, fallback: str = "") -> str:
    if hasattr(record, "node_id"):
        return str(getattr(record, "node_id"))
    if hasattr(record, "block_id"):
        return str(getattr(record, "block_id"))
    if isinstance(record, dict):
        for key in ("node_id", "block_id", "id", "line_id"):
            if record.get(key) is not None:
                return str(record[key])
    return fallback


def record_channel(record: Any) -> str | None:
    if hasattr(record, "semantic_channel"):
        return str(getattr(record, "semantic_channel") or "")
    if hasattr(record, "node_type"):
        return str(getattr(record, "node_type") or "")
    if isinstance(record, dict):
        for key in ("semantic_channel", "node_type", "type", "role", "layout_role", "canonical_type"):
            if record.get(key):
                return str(record[key])
    return None


def record_metadata(record: Any) -> dict[str, Any]:
    """Return formula-relevant metadata without assuming a concrete IR class."""

    if hasattr(record, "metadata") and isinstance(getattr(record, "metadata"), dict):
        return dict(getattr(record, "metadata") or {})
    if isinstance(record, dict):
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            merged = dict(metadata)
            for key in (
                "raw_formula_type",
                "mineru_span_type",
                "formula_latex",
                "text_format",
                "formula_source_layer",
                "formula_confidence",
                "formula_context_role",
                "inline_equation_spans",
                "interline_equation_spans",
                "is_inline_math",
                "is_display_math",
                "parent_line_id",
                "parent_block_id",
            ):
                if key in record and key not in merged:
                    merged[key] = record[key]
            return merged
        return {
            key: record[key]
            for key in (
                "raw_formula_type",
                "mineru_span_type",
                "formula_latex",
                "text_format",
                "formula_source_layer",
                "formula_confidence",
                "formula_context_role",
                "inline_equation_spans",
                "interline_equation_spans",
                "is_inline_math",
                "is_display_math",
                "parent_line_id",
                "parent_block_id",
            )
            if key in record
        }
    return {}


def _metadata_bool(metadata: dict[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def _metadata_list_present(metadata: dict[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if isinstance(value, list):
        return bool(value)
    return bool(value)


def formula_metadata_evidence_source(metadata: dict[str, Any]) -> FormulaEvidenceSource:
    span_type = str(metadata.get("mineru_span_type") or metadata.get("span_type") or "").casefold()
    confidence = str(metadata.get("formula_confidence") or "").casefold()
    raw_formula_type = str(metadata.get("raw_formula_type") or "").casefold()
    text_format = str(metadata.get("text_format") or "").casefold()
    has_span = (
        span_type in {"inline_equation", "interline_equation"}
        or confidence in {"strong_span_inline", "strong_span_interline"}
        or _metadata_list_present(metadata, "inline_equation_spans")
        or _metadata_list_present(metadata, "interline_equation_spans")
    )
    has_content_equation = (
        raw_formula_type == "equation"
        or confidence == "strong_content_equation_latex"
        or text_format == "latex"
    )
    if has_span and has_content_equation:
        return "mixed"
    if has_span:
        return "mineru_span"
    if has_content_equation:
        return "content_list_equation"
    if metadata.get("formula_context_role") or metadata.get("formula_latex"):
        return "document_ir_formula_metadata"
    return "regex_only"


def has_high_confidence_formula_metadata(metadata: dict[str, Any]) -> bool:
    return formula_metadata_evidence_source(metadata) != "regex_only"


def _evidence_from_metadata(
    *,
    metadata: dict[str, Any],
    normalized: str,
    token_count: int,
    symbol_count: int,
    semantic_channel: str | None,
    starts_where: bool,
    theorem_like: bool,
    display_env: bool,
    inline_marker: bool,
    equation_number_like: bool,
    short_fragment: bool,
    local_formula_context: bool,
    negative_where: bool,
) -> tuple[FormulaContextType, FormulaContextEvidence] | None:
    if not metadata:
        return None

    source = formula_metadata_evidence_source(metadata)
    if source == "regex_only":
        return None

    span_type = str(metadata.get("mineru_span_type") or metadata.get("span_type") or "").casefold()
    confidence = str(metadata.get("formula_confidence") or "").casefold()
    role = str(metadata.get("formula_context_role") or "").casefold()
    is_inline = (
        _metadata_bool(metadata, "is_inline_math")
        or span_type == "inline_equation"
        or confidence == "strong_span_inline"
        or role == "inline_attachment"
        or _metadata_list_present(metadata, "inline_equation_spans")
    )
    is_display = (
        _metadata_bool(metadata, "is_display_math")
        or span_type == "interline_equation"
        or confidence in {"strong_span_interline", "strong_content_equation_latex"}
        or role in {"display_math", "equation_block"}
        or _metadata_list_present(metadata, "interline_equation_spans")
    )
    is_uncertain = role in {"formula_ocr_artifact", "uncertain"} or confidence in {
        "medium_equation_text",
        "weak_text_only",
    }

    if is_display:
        family: FormulaContextType = "DISPLAY_MATH_CONTEXT"
        reason = "high-confidence MinerU display/interline equation evidence"
    elif is_inline:
        family = "INLINE_MATH_ATTACHMENT"
        reason = "high-confidence MinerU inline equation span evidence"
    elif is_uncertain:
        family = "FORMULA_OCR_ARTIFACT"
        reason = "formula metadata marked uncertain or OCR-artifact-like"
    else:
        family = "DISPLAY_MATH_CONTEXT"
        reason = "DocumentIR formula metadata evidence"

    return family, FormulaContextEvidence(
        normalized_text=normalized,
        token_count=token_count,
        symbol_count=symbol_count,
        semantic_channel=semantic_channel,
        starts_where_clause=starts_where,
        theorem_like=theorem_like,
        display_math_env=display_env,
        inline_math_marker=inline_marker,
        equation_number_like=equation_number_like,
        short_fragment=short_fragment,
        local_formula_context=local_formula_context,
        negative_where_start=negative_where,
        evidence_source=source,
        formula_confidence=str(metadata.get("formula_confidence") or "") or None,
        formula_context_role=str(metadata.get("formula_context_role") or "") or None,
        has_mineru_formula_evidence=True,
        confidence_tier="high" if not is_uncertain else "medium",
        reason=reason,
    )


def classify_formula_context(
    text: Any,
    *,
    raw_text: Any | None = None,
    semantic_channel: str | None = None,
    local_formula_context: bool = False,
    local_formula_evidence: bool = False,
    formula_metadata: dict[str, Any] | None = None,
) -> tuple[FormulaContextType, FormulaContextEvidence]:
    raw = str(raw_text if raw_text is not None else text or "")
    normalized = normalize_context_text(text)
    lower = normalized.lower()
    tokens = context_tokens(normalized)
    token_count = len(tokens)
    symbol_count = math_symbol_count(normalized) + raw.count("\\")
    starts_where = bool(WHERE_CLAUSE_RE.match(lower))
    negative_where = bool(WHERE_NEGATIVE_START_RE.match(lower))
    theorem_like = bool(THEOREM_KEYWORD_RE.match(lower))
    display_env = bool(DISPLAY_MATH_ENV_RE.search(raw))
    inline_marker = bool(INLINE_MATH_RE.search(raw))
    equation_number_like = bool(EQUATION_NUMBER_RE.match(normalized))
    short_fragment = token_count <= 4 and symbol_count >= 2

    metadata_result = _evidence_from_metadata(
        metadata=formula_metadata or {},
        normalized=normalized,
        token_count=token_count,
        symbol_count=symbol_count,
        semantic_channel=semantic_channel,
        starts_where=starts_where,
        theorem_like=theorem_like,
        display_env=display_env,
        inline_marker=inline_marker,
        equation_number_like=equation_number_like,
        short_fragment=short_fragment,
        local_formula_context=local_formula_context,
        negative_where=negative_where,
    )
    if metadata_result is not None:
        return metadata_result

    family: FormulaContextType
    reason: str
    channel = str(semantic_channel or "").casefold()
    if display_env:
        family, reason = "DISPLAY_MATH_CONTEXT", "contains display math environment"
    elif channel in {"display_math", "equation", "math_context"}:
        if starts_where and not negative_where:
            family, reason = "WHERE_CLAUSE_CONTEXT", "where-like text in math channel"
        elif token_count <= 20 and symbol_count >= max(4, token_count // 2):
            family, reason = "FORMULA_OCR_ARTIFACT", "symbol-heavy short math-channel fragment"
        else:
            family, reason = "DISPLAY_MATH_CONTEXT", "math-channel display context"
    elif starts_where and not negative_where and local_formula_context and (
        symbol_count >= 2
        or re.search(r"\b(?:denotes?|represents?|is defined as|are defined as|satisfies|subject to)\b", lower)
    ):
        family, reason = "WHERE_CLAUSE_CONTEXT", "where/subject-to equation context"
    elif theorem_like:
        family, reason = "THEOREM_PROOF_CONTEXT", "theorem/proof keyword context"
    elif inline_marker and token_count <= 18:
        family, reason = "INLINE_MATH_ATTACHMENT", "short inline math marker context"
    elif short_fragment or (token_count <= 12 and symbol_count >= max(5, token_count)):
        family, reason = "FORMULA_OCR_ARTIFACT", "short symbol-heavy formula residue"
    elif equation_number_like:
        family, reason = "FORMULA_OCR_ARTIFACT", "equation-number-like standalone fragment"
    else:
        family, reason = "ORDINARY_BODY_REORDER", "ordinary prose or non-formula context"

    confidence = confidence_for_context(family, _FormulaEvidenceProxy(
        theorem_like=theorem_like,
        starts_where_clause=starts_where,
        display_math_env=display_env,
        semantic_channel=semantic_channel,
        inline_math_marker=inline_marker,
        short_fragment=short_fragment,
        equation_number_like=equation_number_like,
        local_formula_context=local_formula_context,
        negative_where_start=negative_where,
    ))
    context_has_mineru_neighbor = local_formula_evidence and family in {
        "WHERE_CLAUSE_CONTEXT",
        "THEOREM_PROOF_CONTEXT",
    }
    confidence_tier: FormulaConfidenceTier = confidence_tier_for_score(confidence)
    evidence_source: FormulaEvidenceSource = "mixed" if context_has_mineru_neighbor else "regex_only"
    has_mineru_formula_evidence = context_has_mineru_neighbor
    if family in {
        "WHERE_CLAUSE_CONTEXT",
        "THEOREM_PROOF_CONTEXT",
        "INLINE_MATH_ATTACHMENT",
        "FORMULA_OCR_ARTIFACT",
    } and not context_has_mineru_neighbor:
        confidence_tier = "diagnostic_only"
        reason = f"{reason}; regex-only evidence kept diagnostic-only"
    elif context_has_mineru_neighbor:
        confidence_tier = "high"
        reason = f"{reason}; accepted because adjacent high-confidence MinerU formula evidence exists"
    evidence = FormulaContextEvidence(
        normalized_text=normalized,
        token_count=token_count,
        symbol_count=symbol_count,
        semantic_channel=semantic_channel,
        starts_where_clause=starts_where,
        theorem_like=theorem_like,
        display_math_env=display_env,
        inline_math_marker=inline_marker,
        equation_number_like=equation_number_like,
        short_fragment=short_fragment,
        local_formula_context=local_formula_context,
        negative_where_start=negative_where,
        evidence_source=evidence_source,
        formula_confidence=None,
        formula_context_role=None,
        has_mineru_formula_evidence=has_mineru_formula_evidence,
        confidence_tier=confidence_tier,
        reason=reason,
    )
    return family, evidence


def classify_record_formula_context(record: Any) -> tuple[FormulaContextType, FormulaContextEvidence]:
    metadata = record_metadata(record)
    return classify_formula_context(
        record_text(record),
        raw_text=getattr(record, "raw_text", None) if not isinstance(record, dict) else record.get("raw_text"),
        semantic_channel=record_channel(record),
        local_formula_context=bool(is_formula_like_record(record)),
        local_formula_evidence=has_high_confidence_formula_metadata(metadata),
        formula_metadata=metadata,
    )


def should_exclude_from_ordinary_visible_prose(context_type: str) -> bool:
    return context_type in {
        "DISPLAY_MATH_CONTEXT",
        "WHERE_CLAUSE_CONTEXT",
        "THEOREM_PROOF_CONTEXT",
        "FORMULA_OCR_ARTIFACT",
    }


def should_exclude_from_ordinary_visible_prose_evidence(
    context_type: str,
    evidence: FormulaContextEvidence,
) -> bool:
    return (
        should_exclude_from_ordinary_visible_prose(context_type)
        and evidence.confidence_tier == "high"
        and evidence.evidence_source != "regex_only"
    )


def is_formula_like_record(record: Any) -> bool:
    if has_high_confidence_formula_metadata(record_metadata(record)):
        return True
    channel = str(record_channel(record) or "").casefold()
    if channel in {"display_math", "equation", "math_context"}:
        return True
    text = record_text(record)
    family, evidence = classify_formula_context(text, raw_text=text, semantic_channel=channel, local_formula_context=False)
    return family in {"DISPLAY_MATH_CONTEXT", "FORMULA_OCR_ARTIFACT"} or evidence.display_math_env


def has_local_formula_neighbor(records: list[Any], index: int) -> bool:
    for neighbor_index in (index - 1, index + 1):
        if 0 <= neighbor_index < len(records) and is_formula_like_record(records[neighbor_index]):
            return True
    return False


def has_local_formula_evidence_neighbor(records: list[Any], index: int) -> bool:
    for neighbor_index in (index - 1, index + 1):
        if 0 <= neighbor_index < len(records) and has_high_confidence_formula_metadata(record_metadata(records[neighbor_index])):
            return True
    return False


def build_formula_context_groups(records: list[Any], *, group_prefix: str = "fcg") -> list[FormulaContextGroup]:
    """Create conservative local FormulaContextGroups from ordered records."""

    groups: list[FormulaContextGroup] = []
    for idx, record in enumerate(records):
        text = record_text(record)
        context_type, evidence = classify_formula_context(
            text,
            raw_text=getattr(record, "raw_text", None) if not isinstance(record, dict) else record.get("raw_text"),
            semantic_channel=record_channel(record),
            local_formula_context=has_local_formula_neighbor(records, idx),
            local_formula_evidence=has_local_formula_evidence_neighbor(records, idx),
            formula_metadata=record_metadata(record),
        )
        if context_type not in {
            "DISPLAY_MATH_CONTEXT",
            "WHERE_CLAUSE_CONTEXT",
            "THEOREM_PROOF_CONTEXT",
            "FORMULA_OCR_ARTIFACT",
        }:
            continue
        if evidence.confidence_tier != "high" or evidence.evidence_source == "regex_only":
            continue
        before = records[idx - 1] if idx > 0 else None
        after = records[idx + 1] if idx + 1 < len(records) else None
        record_id_value = record_id(record, f"record_{idx:04d}")
        before_id = record_id(before, f"record_{idx-1:04d}") if before is not None else None
        after_id = record_id(after, f"record_{idx+1:04d}") if after is not None else None
        render_policy = render_policy_for_context(context_type)
        groups.append(
            FormulaContextGroup(
                group_id=f"{group_prefix}_{len(groups):04d}",
                context_type=context_type,
                source_v7_ids=[record_id_value],
                text_before_ids=[before_id] if before_id and context_type != "THEOREM_PROOF_CONTEXT" else [],
                formula_ids=[record_id_value] if context_type in {"DISPLAY_MATH_CONTEXT", "FORMULA_OCR_ARTIFACT"} else [],
                text_after_ids=[after_id] if after_id and context_type in {"DISPLAY_MATH_CONTEXT", "WHERE_CLAUSE_CONTEXT"} else [],
                theorem_label_ids=[record_id_value] if context_type == "THEOREM_PROOF_CONTEXT" else [],
                confidence=confidence_for_context(context_type, evidence),
                evidence=evidence.to_dict(),
                render_policy=render_policy,
                confidence_tier=evidence.confidence_tier,
            )
        )
    return groups


def confidence_for_context(context_type: FormulaContextType, evidence: FormulaContextEvidence) -> float:
    if context_type == "THEOREM_PROOF_CONTEXT":
        return 0.90 if evidence.theorem_like else 0.65
    if context_type == "WHERE_CLAUSE_CONTEXT":
        if evidence.negative_where_start:
            return 0.0
        if evidence.starts_where_clause and evidence.local_formula_context:
            return 0.88
        return 0.55
    if context_type == "DISPLAY_MATH_CONTEXT":
        return 0.88 if evidence.display_math_env or evidence.semantic_channel in {"display_math", "equation"} else 0.70
    if context_type == "INLINE_MATH_ATTACHMENT":
        return 0.82 if evidence.inline_math_marker else 0.55
    if context_type == "FORMULA_OCR_ARTIFACT":
        return 0.72 if evidence.short_fragment or evidence.equation_number_like else 0.55
    return 0.50


@dataclass(frozen=True)
class _FormulaEvidenceProxy:
    theorem_like: bool = False
    starts_where_clause: bool = False
    display_math_env: bool = False
    semantic_channel: str | None = None
    inline_math_marker: bool = False
    short_fragment: bool = False
    equation_number_like: bool = False
    local_formula_context: bool = False
    negative_where_start: bool = False


def confidence_tier_for_score(score: float) -> Literal["high", "medium", "low"]:
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def render_policy_for_context(context_type: FormulaContextType) -> str:
    if context_type == "INLINE_MATH_ATTACHMENT":
        return "inline_math_span_attachment"
    if context_type == "DISPLAY_MATH_CONTEXT":
        return "paragraph_displaymath_context"
    if context_type == "WHERE_CLAUSE_CONTEXT":
        return "paragraph_displaymath_where"
    if context_type == "THEOREM_PROOF_CONTEXT":
        return "bold_inline_label"
    if context_type == "FORMULA_OCR_ARTIFACT":
        return "artifact_cleaning_required"
    return "plain_paragraph_fallback"
