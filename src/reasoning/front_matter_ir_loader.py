"""Load deterministic FrontMatterIR Phase0 sidecars.

The Phase0 extractor writes a compact JSON sidecar for audit/context tracks.
This loader converts that sidecar into the existing ``FrontMatterIR`` dataclasses
so renderer experiments can consume the same contract without re-running
extraction or mutating DocumentIR/v8 artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ir.serialization import read_json
from src.reasoning.front_matter_extractor import FrontMatterAbstract, FrontMatterIR, FrontMatterRegion, FrontMatterSpan


CONFIDENCE_BY_TIER = {
    "high": 0.95,
    "medium": 0.75,
    "low": 0.45,
    "diagnostic_only": 0.0,
}


def load_front_matter_ir_sidecar(path: str | Path) -> FrontMatterIR:
    """Load a Phase0 FrontMatterIR sidecar.

    The sidecar schema stores human-readable confidence tiers.  Renderer code
    already expects numeric confidence on spans, so we normalize tiers here.
    Unknown or diagnostic-only entries are kept with low confidence; renderer
    policy decides whether they are safe to consume.
    """

    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"FrontMatterIR sidecar must be a JSON object: {path}")

    title = _span_from_payload(payload.get("title"), role="TITLE")
    authors = [_span_from_payload(item, role="AUTHOR") for item in _list_payload(payload.get("authors"))]
    affiliations = [_span_from_payload(item, role="AFFILIATION") for item in _list_payload(payload.get("affiliations"))]
    emails = [_span_from_payload(item, role="EMAIL") for item in _list_payload(payload.get("emails"))]
    notes = [_span_from_payload(item, role="FRONT_NOTE") for item in _list_payload(payload.get("front_notes"))]

    abstract_payload = payload.get("abstract")
    abstract = _abstract_from_payload(abstract_payload)
    region = _region_from_payload(payload.get("first_body_boundary"))

    return FrontMatterIR(
        title=title,
        authors=[span for span in authors if span.text.strip()],
        affiliations=[span for span in affiliations if span.text.strip()],
        emails=[span for span in emails if span.text.strip()],
        notes=[span for span in notes if span.text.strip()],
        abstract=abstract,
        misc=[],
        region=region,
        lines=[],
        warnings=[],
    )


def _span_from_payload(value: Any, *, role: str) -> FrontMatterSpan:
    if isinstance(value, str):
        return FrontMatterSpan(role=role, text=value, source_node_ids=[], line_ids=[], confidence=0.0)
    if not isinstance(value, dict):
        return FrontMatterSpan(role=role, text="", source_node_ids=[], line_ids=[], confidence=0.0)
    text = str(value.get("text") or "")
    source_ids = [str(item) for item in value.get("source_v8_ids") or value.get("source_node_ids") or [] if item]
    line_ids = [
        str(item.get("line_id"))
        for item in _list_payload(value.get("evidence"))
        if isinstance(item, dict) and item.get("line_id")
    ]
    confidence = _confidence(value.get("confidence"))
    if confidence <= 0 and value.get("evidence"):
        evidence_confidences = [
            _confidence(item.get("confidence_tier")) for item in _list_payload(value.get("evidence")) if isinstance(item, dict)
        ]
        if evidence_confidences:
            confidence = max(evidence_confidences)
    return FrontMatterSpan(
        role=role,
        text=text,
        source_node_ids=source_ids,
        line_ids=line_ids,
        confidence=confidence,
        bbox=None,
    )


def _abstract_from_payload(value: Any) -> FrontMatterAbstract | None:
    if not isinstance(value, dict):
        return None
    title_payload = value.get("title")
    body_payload: Any
    if isinstance(value.get("body"), dict):
        body_payload = value.get("body")
    else:
        body_payload = {
            "text": value.get("body") or "",
            "source_v8_ids": value.get("source_v8_ids") or [],
            "confidence": value.get("confidence"),
            "evidence": value.get("evidence") or [],
        }
    title = _span_from_payload(title_payload, role="ABSTRACT_TITLE") if title_payload else None
    body = _span_from_payload(body_payload, role="ABSTRACT_BODY") if body_payload else None
    if title is None and (body is None or not body.text.strip()):
        return None
    return FrontMatterAbstract(title=title, body=body)


def _region_from_payload(value: Any) -> FrontMatterRegion | None:
    if not isinstance(value, dict):
        return None
    source_id = value.get("source_v8_id")
    source_ids = [str(source_id)] if source_id else []
    return FrontMatterRegion(
        page_idx=int(value.get("page_idx") or 0),
        start_order=0.0,
        end_order=0.0,
        body_start_order=None,
        source_node_ids=source_ids,
    )


def _confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    tier = str(value or "").strip().casefold()
    return CONFIDENCE_BY_TIER.get(tier, 0.0)


def _list_payload(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
