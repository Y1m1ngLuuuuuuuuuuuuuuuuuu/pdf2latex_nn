"""Bounded visual/text probes for v7 layout layer resolution.

The probes in this module are deliberately weak classifiers.  They return
evidence with a confidence and a scope, while the caller owns the final state
transition.  This prevents generic words such as "research" from directly
turning body headings into affiliation metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


ProbeScope = Literal["any", "front_matter_only"]
ProbeStrength = Literal["strong", "weak"]


@dataclass(frozen=True)
class LayoutProbe:
    role: str
    confidence: float
    reason: str
    scope: ProbeScope = "any"
    strength: ProbeStrength = "weak"


EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
ORCID_RE = re.compile(r"\b(?:orcid[:\s]*)?\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", re.IGNORECASE)
FOOTER_NOISE_RE = re.compile(
    r"^\s*(?:submission\s+to\b|proceedings\s+of\b|copyright\b|©\b|.*\bwith\s+the\s+authors\b)",
    re.IGNORECASE,
)
FLOAT_CAPTION_RE = re.compile(r"^\s*(?P<kind>fig\.?|figure|table|algorithm)\s*\.?\s*(?P<number>[A-Za-z]?\d+(?:\.\d+)*)\s*[:.\-]", re.IGNORECASE)
FOOTNOTE_MARKER_RE = re.compile(r"^\s*(?:\d{1,3}|[*†‡§¶]|[¹²³⁴⁵⁶⁷⁸⁹⁰]+)\s+")
ABSTRACT_RE = re.compile(r"^\s*abstract\b", re.IGNORECASE)
KEYWORDS_RE = re.compile(r"^\s*(?:index\s+terms|keywords?)\b", re.IGNORECASE)
FRONT_MATTER_LABEL_RE = re.compile(r"^\s*(?:author|authors|affiliation)\b", re.IGNORECASE)
STRONG_AFFILIATION_RE = re.compile(
    r"\b("
    r"affiliation|department|university|institute|school|college|faculty|"
    r"laborator(?:y|ies)|academy|cnrs|inria|google|microsoft"
    r")\b",
    re.IGNORECASE,
)
WEAK_AFFILIATION_RE = re.compile(
    r"\b(research|lab|centre|center|norway|china|usa|uk|germany|france|italy)\b",
    re.IGNORECASE,
)


def collect_layout_probes(node: dict[str, Any], *, text: str | None = None) -> list[LayoutProbe]:
    """Return layout evidence for a page object without resolving final role."""

    value = " ".join(str(text if text is not None else _node_text(node)).split())
    probes: list[LayoutProbe] = []
    if not value:
        return probes

    if FOOTER_NOISE_RE.match(value) and len(value) <= 140:
        probes.append(LayoutProbe("footer", 0.98, "footer_copyright_or_submission", "any", "strong"))
    caption_match = FLOAT_CAPTION_RE.match(value)
    if caption_match:
        probes.append(LayoutProbe(f"{caption_match.group('kind').lower().rstrip('.')}_caption", 0.96, "float_caption_label", "any", "strong"))
    if FOOTNOTE_MARKER_RE.match(value):
        probes.append(LayoutProbe("footnote", 0.72, "footnote_marker_prefix", "any", "weak"))
    if EMAIL_RE.search(value):
        probes.append(LayoutProbe("affiliation", 0.96, "email", "front_matter_only", "strong"))
    if ORCID_RE.search(value):
        probes.append(LayoutProbe("front_matter", 0.95, "orcid", "front_matter_only", "strong"))
    if ABSTRACT_RE.match(value):
        probes.append(LayoutProbe("abstract", 0.96, "abstract_label", "front_matter_only", "strong"))
    if KEYWORDS_RE.match(value):
        probes.append(LayoutProbe("front_matter", 0.90, "keywords_or_index_terms", "front_matter_only", "strong"))
    if FRONT_MATTER_LABEL_RE.match(value):
        probes.append(LayoutProbe("front_matter", 0.88, "front_matter_label", "front_matter_only", "strong"))
    if STRONG_AFFILIATION_RE.search(value):
        probes.append(LayoutProbe("affiliation", 0.88, "strong_affiliation_keyword", "front_matter_only", "strong"))
    if WEAK_AFFILIATION_RE.search(value):
        probes.append(LayoutProbe("affiliation", 0.45, "weak_affiliation_keyword", "front_matter_only", "weak"))
    return sorted(probes, key=lambda probe: probe.confidence, reverse=True)


def best_layout_probe(node: dict[str, Any], *, text: str | None = None) -> LayoutProbe | None:
    probes = collect_layout_probes(node, text=text)
    return probes[0] if probes else None


def has_strong_layout_probe(
    node: dict[str, Any],
    *,
    text: str | None = None,
    roles: set[str] | frozenset[str] | None = None,
    min_confidence: float = 0.80,
) -> bool:
    for probe in collect_layout_probes(node, text=text):
        if probe.strength != "strong" or probe.confidence < min_confidence:
            continue
        if roles is not None and probe.role not in roles:
            continue
        return True
    return False


def _node_text(node: dict[str, Any]) -> str:
    for key in ("text_for_embedding", "text", "content", "latex"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""
