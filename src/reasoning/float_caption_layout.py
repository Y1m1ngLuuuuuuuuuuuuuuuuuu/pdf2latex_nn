"""Experimental v8 FloatCaptionLayout pass.

The pass promotes v8 caption-like facts into structured float/caption render
metadata.  It is opt-in and does not mutate v8 facts, graph views, graph
schema, or deterministic v8 merge decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from src.ir import BlockType, DocumentIR, RenderRole, RenderTreeIR, RenderTreeNode
from src.reasoning.float_caption_matcher import (
    CaptionCandidate,
    CaptionPairing,
    caption_candidates_from_document,
    caption_to_record,
    float_candidates_from_document,
    pairing_to_record,
    pair_caption_candidates,
)


@dataclass(frozen=True)
class FloatCaptionLayoutResult:
    promoted_captions: list[dict[str, Any]] = field(default_factory=list)
    pairings: list[dict[str, Any]] = field(default_factory=list)
    placeholder_floats: list[dict[str, Any]] = field(default_factory=list)
    duplicate_caption_suppression: list[dict[str, Any]] = field(default_factory=list)
    crop_caption_separation: list[dict[str, Any]] = field(default_factory=list)
    consumed_caption_paragraphs: list[dict[str, Any]] = field(default_factory=list)
    canonical_caption_clusters: list[dict[str, Any]] = field(default_factory=list)
    noncanonical_suppressed_candidates: list[dict[str, Any]] = field(default_factory=list)
    subfigure_like_risk_review: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_diagnostic(self) -> dict[str, Any]:
        return {
            "promoted_captions": self.promoted_captions,
            "float_caption_pairings": self.pairings,
            "placeholder_floats": self.placeholder_floats,
            "duplicate_caption_suppression": self.duplicate_caption_suppression,
            "crop_caption_separation": self.crop_caption_separation,
            "consumed_caption_paragraphs": self.consumed_caption_paragraphs,
            "canonical_caption_clusters": self.canonical_caption_clusters,
            "noncanonical_suppressed_candidates": self.noncanonical_suppressed_candidates,
            "subfigure_like_risk_review": self.subfigure_like_risk_review,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class _CanonicalSelection:
    pairings: list[CaptionPairing] = field(default_factory=list)
    clusters: list[dict[str, Any]] = field(default_factory=list)
    suppressed: list[dict[str, Any]] = field(default_factory=list)
    subfigure_like_risk_review: list[dict[str, Any]] = field(default_factory=list)


def build_float_caption_layout_sidecars(document: DocumentIR) -> FloatCaptionLayoutResult:
    """Return candidate/pairing diagnostics without changing render output."""

    pairings = pair_caption_candidates(caption_candidates_from_document(document), float_candidates_from_document(document))
    selection = select_canonical_caption_pairings(pairings, doc_id=document.doc_id)
    canonical_pairings = selection.pairings
    return FloatCaptionLayoutResult(
        promoted_captions=[_caption_record(pairing.caption, candidate_class=_candidate_class(pairing.caption)) for pairing in canonical_pairings],
        pairings=[pairing_to_record(pairing) for pairing in pairings],
        placeholder_floats=[
            _placeholder_record(pairing)
            for pairing in canonical_pairings
            if pairing.float_candidate is None and pairing.caption.confidence >= 0.86
        ],
        duplicate_caption_suppression=selection.suppressed,
        crop_caption_separation=[
            {
                "caption_id": pairing.caption.caption_id,
                "source_v8_ids": pairing.caption.source_v8_ids,
                "caption_type": pairing.caption.caption_type,
                "caption_number": pairing.caption.caption_number,
                "crop_may_include_caption": pairing.caption.origin == "crop_metadata",
                "reason": "crop_metadata_caption_materialized_as_structured_caption",
            }
            for pairing in canonical_pairings
            if pairing.caption.origin == "crop_metadata"
        ],
        consumed_caption_paragraphs=[
            {
                "caption_id": pairing.caption.caption_id,
                "source_v8_ids": pairing.caption.source_v8_ids,
                "caption_type": pairing.caption.caption_type,
                "caption_number": pairing.caption.caption_number,
                "reason": "caption_promoted_to_float_caption_layout",
            }
            for pairing in canonical_pairings
            if pairing.caption.origin == "text_block"
        ],
        canonical_caption_clusters=selection.clusters,
        noncanonical_suppressed_candidates=selection.suppressed,
        subfigure_like_risk_review=selection.subfigure_like_risk_review,
    )


def apply_float_caption_layout(
    document: DocumentIR,
    tree: RenderTreeIR,
    *,
    enabled: bool = False,
) -> tuple[RenderTreeIR, FloatCaptionLayoutResult]:
    """Apply the experimental layout pass when explicitly enabled.

    When ``enabled`` is false, only sidecar diagnostics are produced and the
    original tree object is returned unchanged.
    """

    result = build_float_caption_layout_sidecars(document)
    if not enabled:
        return tree, result

    pairings = pair_caption_candidates(caption_candidates_from_document(document), float_candidates_from_document(document))
    selection = select_canonical_caption_pairings(pairings, doc_id=document.doc_id)
    canonical_pairings = selection.pairings
    nodes_by_id = {node.render_id: node for node in tree.nodes}
    source_to_render_ids: dict[str, list[str]] = {}
    parent_by_child_id: dict[str, str] = {}
    for node in tree.nodes:
        for source_id in node.source_node_ids:
            source_to_render_ids.setdefault(source_id, []).append(node.render_id)
        for child_id in node.children:
            parent_by_child_id[child_id] = node.render_id

    updated_nodes: dict[str, RenderTreeNode] = dict(nodes_by_id)
    consumed_caption_source_ids: set[str] = set()
    placeholder_insertions: list[tuple[str, str, CaptionPairing]] = []
    promoted: list[dict[str, Any]] = []
    consumed: list[dict[str, Any]] = []
    placeholders: list[dict[str, Any]] = []
    crop_separation: list[dict[str, Any]] = []

    for pairing in canonical_pairings:
        caption = pairing.caption
        if caption.confidence < 0.78:
            continue
        if pairing.float_candidate is None:
            if caption.confidence >= 0.86:
                placeholder_id = f"float_caption_placeholder_{_safe_render_id(caption.caption_id)}"
                placeholder = _placeholder_node(placeholder_id, caption)
                updated_nodes[placeholder_id] = placeholder
                parent_id = _parent_for_caption(caption, source_to_render_ids, parent_by_child_id, tree.root_id)
                placeholder_insertions.append((parent_id, placeholder_id, pairing))
                placeholders.append(_placeholder_record(pairing, render_id=placeholder_id, parent_id=parent_id))
                consumed_caption_source_ids.update(caption.source_v8_ids)
                consumed.append(_consumed_record(caption, "placeholder_float_created"))
                promoted.append(_promotion_record(caption, pairing, render_id=placeholder_id))
            continue

        float_render_id = _render_id_for_float(pairing.float_candidate.source_v8_ids, source_to_render_ids, updated_nodes)
        if float_render_id is None:
            if caption.confidence >= 0.86:
                placeholder_id = f"float_caption_materialized_{_safe_render_id(caption.caption_id)}"
                placeholder = _placeholder_node(
                    placeholder_id,
                    caption,
                    source_v8_ids=pairing.float_candidate.source_v8_ids,
                    paired_float_id=pairing.float_candidate.float_id,
                )
                updated_nodes[placeholder_id] = placeholder
                parent_id = _parent_for_caption(caption, source_to_render_ids, parent_by_child_id, tree.root_id)
                placeholder_insertions.append((parent_id, placeholder_id, pairing))
                placeholders.append(
                    _placeholder_record(
                        pairing,
                        render_id=placeholder_id,
                        parent_id=parent_id,
                        reason="paired_float_render_node_missing_materialized_caption",
                    )
                )
                consumed_caption_source_ids.update(caption.source_v8_ids)
                consumed.append(_consumed_record(caption, "paired_float_render_node_missing_materialized_caption", render_id=placeholder_id))
                promoted.append(_promotion_record(caption, pairing, render_id=placeholder_id))
            continue
        old = updated_nodes[float_render_id]
        attrs = dict(old.attributes)
        attrs["float_caption_layout"] = True
        attrs["float_caption_layout_caption"] = caption_to_record(caption)
        attrs["float_caption_layout_pairing"] = pairing_to_record(pairing)
        attrs["crop_may_include_caption"] = caption.origin == "crop_metadata"
        updated_nodes[float_render_id] = RenderTreeNode(
            render_id=old.render_id,
            role=old.role,
            source_node_ids=list(old.source_node_ids),
            text=caption.text,
            latex=old.latex,
            children=list(old.children),
            attributes=attrs,
        )
        consumed_caption_source_ids.update(caption.source_v8_ids)
        consumed.append(_consumed_record(caption, "caption_attached_to_existing_float", render_id=float_render_id))
        promoted.append(_promotion_record(caption, pairing, render_id=float_render_id))
        if caption.origin == "crop_metadata":
            crop_separation.append(
                {
                    "caption_id": caption.caption_id,
                    "source_v8_ids": caption.source_v8_ids,
                    "paired_float_id": pairing.float_candidate.float_id,
                    "render_id": float_render_id,
                    "crop_may_include_caption": True,
                    "reason": "crop_metadata_caption_materialized_as_structured_caption",
                }
            )

    for source_id in consumed_caption_source_ids:
        for render_id in source_to_render_ids.get(source_id, []):
            node = updated_nodes.get(render_id)
            if node is None or node.role in {RenderRole.FIGURE, RenderRole.TABLE, RenderRole.ALGORITHM}:
                continue
            attrs = dict(node.attributes)
            attrs["float_caption_consumed"] = True
            attrs["float_caption_suppression_reason"] = "promoted_to_structured_caption"
            updated_nodes[render_id] = RenderTreeNode(
                render_id=node.render_id,
                role=node.role,
                source_node_ids=list(node.source_node_ids),
                text=node.text,
                latex=node.latex,
                children=list(node.children),
                attributes=attrs,
            )

    final_nodes = _insert_placeholders(tree, updated_nodes, placeholder_insertions)
    final_result = FloatCaptionLayoutResult(
        promoted_captions=promoted,
        pairings=[pairing_to_record(pairing) for pairing in pairings],
        placeholder_floats=placeholders,
        duplicate_caption_suppression=selection.suppressed,
        crop_caption_separation=crop_separation,
        consumed_caption_paragraphs=consumed,
        canonical_caption_clusters=selection.clusters,
        noncanonical_suppressed_candidates=selection.suppressed,
        subfigure_like_risk_review=selection.subfigure_like_risk_review,
        notes=["experimental_float_caption_layout_enabled"],
    )
    return (
        RenderTreeIR(
            doc_id=tree.doc_id,
            root_id=tree.root_id,
            nodes=final_nodes,
            document_ir_path=tree.document_ir_path,
            schema_version=tree.schema_version,
            predicted_relations_path=tree.predicted_relations_path,
            style_profile_path=tree.style_profile_path,
            metadata={
                **tree.metadata,
                "experimental_float_caption_layout_enabled": True,
                "float_caption_layout_diag": final_result.to_diagnostic(),
            },
        ),
        final_result,
    )


def select_canonical_caption_pairings(pairings: list[CaptionPairing], *, doc_id: str = "") -> _CanonicalSelection:
    """Select one materializable caption per true-caption cluster.

    Candidate discovery intentionally remains broad, but materialization must be
    conservative.  This selection runs after pairing so text-block and
    metadata/crop copies attached to the same float collapse to one canonical
    caption, while subfigure markers such as 2(a) and 2(b) remain distinct.
    """

    by_key: dict[tuple[str, str, str, str | None, str, int | None, str], CaptionPairing] = {}
    suppressed: list[dict[str, Any]] = []
    risk_review: list[dict[str, Any]] = []
    cluster_members: dict[tuple[str, str, str, str | None, str, int | None, str], list[CaptionPairing]] = {}

    for pairing in pairings:
        caption = pairing.caption
        candidate_class = _candidate_class(caption)
        if candidate_class in {"PANEL_LABEL", "SYNTHETIC_FALLBACK_CAPTION", "BODY_REFERENCE_FALSE_POSITIVE"}:
            suppressed.append(_suppression_record(pairing, reason=f"{candidate_class.lower()}_not_materialized", kept_caption_id=None))
            risk_review.append(_risk_review_record(pairing, candidate_class))
            continue
        key = _canonical_cluster_key(pairing, doc_id=doc_id)
        cluster_members.setdefault(key, []).append(pairing)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = pairing
            continue
        winner, loser = _choose_canonical_pairing(previous, pairing)
        by_key[key] = winner
        suppressed.append(_suppression_record(loser, reason="duplicate_suppressed_by_canonical", kept_caption_id=winner.caption.caption_id))

    canonical_pairings = sorted(
        by_key.values(),
        key=lambda item: (
            item.caption.page_idx if item.caption.page_idx is not None else -1,
            item.caption.caption_id,
        ),
    )
    clusters = [
        _cluster_record(key, by_key[key], cluster_members.get(key, []))
        for key in sorted(by_key, key=lambda item: (item[0], item[1], item[2], item[3] or "", item[4], item[5] if item[5] is not None else -1, item[6]))
    ]
    return _CanonicalSelection(
        pairings=canonical_pairings,
        clusters=clusters,
        suppressed=suppressed,
        subfigure_like_risk_review=risk_review,
    )


def _placeholder_node(
    render_id: str,
    caption: CaptionCandidate,
    *,
    source_v8_ids: list[str] | None = None,
    paired_float_id: str | None = None,
) -> RenderTreeNode:
    role = {
        "figure": RenderRole.FIGURE,
        "table": RenderRole.TABLE,
        "algorithm": RenderRole.ALGORITHM,
    }.get(caption.caption_type, RenderRole.FIGURE)
    return RenderTreeNode(
        render_id=render_id,
        role=role,
        source_node_ids=list(source_v8_ids or []),
        text=caption.text,
        attributes={
            "float_caption_layout": True,
            "placeholder_float": True,
            "paired_float_id": paired_float_id,
            "caption_type": caption.caption_type,
            "caption_number": caption.caption_number,
            "source_v8_ids": list(caption.source_v8_ids),
            "float_caption_layout_caption": caption_to_record(caption),
            "render_policy": "placeholder_float_with_structured_caption",
        },
    )


def _insert_placeholders(
    tree: RenderTreeIR,
    updated_nodes: dict[str, RenderTreeNode],
    insertions: list[tuple[str, str, CaptionPairing]],
) -> list[RenderTreeNode]:
    insert_by_parent: dict[str, list[str]] = {}
    for parent_id, placeholder_id, _pairing in insertions:
        insert_by_parent.setdefault(parent_id, []).append(placeholder_id)
    result: list[RenderTreeNode] = []
    existing_ids = {node.render_id for node in tree.nodes}
    for original in tree.nodes:
        node = updated_nodes[original.render_id]
        children = list(node.children)
        additions = insert_by_parent.get(node.render_id, [])
        if additions:
            children.extend(additions)
        result.append(
            RenderTreeNode(
                render_id=node.render_id,
                role=node.role,
                source_node_ids=list(node.source_node_ids),
                text=node.text,
                latex=node.latex,
                children=list(dict.fromkeys(children)),
                attributes=dict(node.attributes),
            )
        )
    for render_id, node in updated_nodes.items():
        if render_id not in existing_ids:
            result.append(node)
    return result


def _render_id_for_float(
    source_ids: list[str],
    source_to_render_ids: dict[str, list[str]],
    nodes: dict[str, RenderTreeNode],
) -> str | None:
    wanted_roles = {RenderRole.FIGURE, RenderRole.TABLE, RenderRole.ALGORITHM}
    for source_id in source_ids:
        for render_id in source_to_render_ids.get(source_id, []):
            node = nodes.get(render_id)
            if node is not None and node.role in wanted_roles:
                return render_id
    return None


def _parent_for_caption(
    caption: CaptionCandidate,
    source_to_render_ids: dict[str, list[str]],
    parent_by_child_id: dict[str, str],
    root_id: str,
) -> str:
    for source_id in caption.source_v8_ids:
        for render_id in source_to_render_ids.get(source_id, []):
            return parent_by_child_id.get(render_id, root_id)
    return root_id


def _promotion_record(caption: CaptionCandidate, pairing: CaptionPairing, *, render_id: str) -> dict[str, Any]:
    return {
        **_caption_record(caption, candidate_class=_candidate_class(caption)),
        "render_id": render_id,
        "paired_float_id": pairing.float_candidate.float_id if pairing.float_candidate else None,
        "pairing_confidence": pairing.confidence,
        "promotion_reason": pairing.reason,
        "render_policy": "structured_caption",
        "consumed_original_nodes": list(caption.source_v8_ids),
        "placeholder_created": pairing.float_candidate is None,
        "duplicate_suppressed": False,
        "crop_may_include_caption": caption.origin == "crop_metadata",
    }


def _placeholder_record(
    pairing: CaptionPairing,
    *,
    render_id: str | None = None,
    parent_id: str | None = None,
    reason: str = "high_confidence_caption_without_float",
) -> dict[str, Any]:
    return {
        "caption_id": pairing.caption.caption_id,
        "caption_type": pairing.caption.caption_type,
        "caption_number": pairing.caption.caption_number,
        "source_v8_ids": pairing.caption.source_v8_ids,
        "render_id": render_id,
        "parent_id": parent_id,
        "reason": reason,
        "render_policy": "placeholder_float_with_structured_caption",
    }


def _consumed_record(caption: CaptionCandidate, reason: str, *, render_id: str | None = None) -> dict[str, Any]:
    return {
        "caption_id": caption.caption_id,
        "source_v8_ids": list(caption.source_v8_ids),
        "caption_type": caption.caption_type,
        "caption_number": caption.caption_number,
        "render_id": render_id,
        "reason": reason,
    }


def _safe_render_id(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in str(value or "caption")).strip("_")


def _canonical_cluster_key(pairing: CaptionPairing, *, doc_id: str) -> tuple[str, str, str, str | None, str, int | None, str]:
    caption = pairing.caption
    paired_float_id = _paired_float_cluster_id(pairing)
    return (
        str(doc_id or ""),
        caption.caption_type or "unknown",
        _main_caption_number(caption.caption_number),
        _subfigure_marker(caption.caption_number),
        _cluster_text_key(caption.normalized_text or caption.text),
        caption.page_idx,
        paired_float_id,
    )


def _paired_float_cluster_id(pairing: CaptionPairing) -> str:
    """Return the float identity used by canonical caption selection.

    Patch3 exposed that the same real caption can be paired to multiple
    neighboring crop/float nodes.  The candidate should still materialize once.
    Subfigure identities remain protected by the separate subfigure marker in
    the cluster key.
    """

    caption = pairing.caption
    text_key = _cluster_text_key(caption.normalized_text or caption.text)
    if text_key and len(re.sub(r"[^0-9a-z]+", "", text_key.casefold())) >= 8:
        return "same_visible_caption_identity"
    return pairing.float_candidate.float_id if pairing.float_candidate else "unpaired"


def _choose_canonical_pairing(left: CaptionPairing, right: CaptionPairing) -> tuple[CaptionPairing, CaptionPairing]:
    return (left, right) if _canonical_rank(left) <= _canonical_rank(right) else (right, left)


def _canonical_rank(pairing: CaptionPairing) -> tuple[int, float, int, str]:
    caption = pairing.caption
    origin_rank = {
        "text_block": 0,
        "caption_metadata": 1,
        "float_metadata": 2,
        "crop_metadata": 3,
        "unknown": 4,
    }.get(caption.origin, 4)
    source_bonus = 0 if caption.source_v8_ids else 1
    return (origin_rank, -float(caption.confidence or 0.0), source_bonus, caption.caption_id)


def _number_key(value: str | None) -> str:
    value = str(value or "").strip().casefold()
    return value


def _cluster_text_key(text: str | None) -> str:
    value = " ".join(str(text or "").casefold().split())
    value = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", value)
    value = re.sub(r"[{}]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .:;,-–—")


def _caption_materialization_risk(caption: CaptionCandidate) -> str | None:
    text = _cluster_text_key(caption.normalized_text or caption.text)
    compact = re.sub(r"[^0-9a-z]+", "", text.casefold())
    if not compact:
        return "empty_caption_text"
    if compact in {
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "figure",
        "fig",
        "table",
        "algorithm",
        "reconstructionplaceholder",
        "tablereconstructionplaceholder",
        "figurereconstructionplaceholder",
    }:
        return "subfigure_panel_label_or_generic_caption"
    if re.fullmatch(r"\(?[a-z]\)?", text.strip(), flags=re.IGNORECASE):
        return "subfigure_panel_label_or_generic_caption"
    return None


def _candidate_class(caption: CaptionCandidate) -> str:
    text = _cluster_text_key(caption.normalized_text or caption.text)
    compact = re.sub(r"[^0-9a-z]+", "", text.casefold())
    if caption.evidence.get("false_positive_guard"):
        return "BODY_REFERENCE_FALSE_POSITIVE"
    if _subfigure_marker(caption.caption_number):
        return "SUBFIGURE_CAPTION"
    if compact in {"a", "b", "c", "d", "e", "f", "left", "right", "upper", "lower"}:
        return "PANEL_LABEL"
    if _looks_like_panel_label_sequence(text):
        return "PANEL_LABEL"
    if re.match(r"^\(?[a-z]\)?\s+", text.strip(), flags=re.IGNORECASE) and not caption.caption_number:
        return "PANEL_LABEL"
    if re.fullmatch(r"\(?[a-z]\)?", text.strip(), flags=re.IGNORECASE):
        return "PANEL_LABEL"
    if compact in {
        "figure",
        "fig",
        "table",
        "algorithm",
        "reconstructionplaceholder",
        "tablereconstructionplaceholder",
        "figurereconstructionplaceholder",
    }:
        return "SYNTHETIC_FALLBACK_CAPTION"
    return "REAL_CAPTION"


def _looks_like_panel_label_sequence(text: str) -> bool:
    value = " ".join(str(text or "").casefold().split()).strip()
    if not value:
        return False
    panel_token = r"(?:\([a-z]\)|[a-z]\))"
    return bool(
        re.fullmatch(rf"{panel_token}(?:\s+{panel_token}){{1,7}}", value, flags=re.IGNORECASE)
        or re.fullmatch(r"[a-z](?:\s+[a-z]){1,7}", value, flags=re.IGNORECASE)
    )


def _caption_record(caption: CaptionCandidate, *, candidate_class: str) -> dict[str, Any]:
    record = caption_to_record(caption)
    record["caption_candidate_class"] = candidate_class
    record["main_caption_number"] = _main_caption_number(caption.caption_number)
    record["subfigure_marker"] = _subfigure_marker(caption.caption_number)
    return record


def _suppression_record(pairing: CaptionPairing, *, reason: str, kept_caption_id: str | None) -> dict[str, Any]:
    caption = pairing.caption
    return {
        "caption_id": caption.caption_id,
        "kept_caption_id": kept_caption_id,
        "caption_type": caption.caption_type,
        "caption_number": caption.caption_number,
        "caption_candidate_class": _duplicate_candidate_class(caption),
        "subfigure_marker": _subfigure_marker(caption.caption_number),
        "normalized_caption_text": caption.normalized_text,
        "source_v8_ids": list(caption.source_v8_ids),
        "page_idx": caption.page_idx,
        "origin": caption.origin,
        "paired_float_id": pairing.float_candidate.float_id if pairing.float_candidate else None,
        "reason": reason,
    }


def _risk_review_record(pairing: CaptionPairing, risk: str) -> dict[str, Any]:
    caption = pairing.caption
    return {
        "caption_id": caption.caption_id,
        "caption_type": caption.caption_type,
        "caption_number": caption.caption_number,
        "caption_candidate_class": _candidate_class(caption),
        "subfigure_marker": _subfigure_marker(caption.caption_number),
        "normalized_caption_text": caption.normalized_text,
        "text": caption.text,
        "origin": caption.origin,
        "paired_float_id": pairing.float_candidate.float_id if pairing.float_candidate else None,
        "risk": risk,
        "review_reason": "panel-only or generic caption text is not materialized as a standalone structured caption",
    }


def _cluster_record(
    key: tuple[str, str, str, str | None, str, int | None, str],
    canonical: CaptionPairing,
    members: list[CaptionPairing],
) -> dict[str, Any]:
    caption = canonical.caption
    return {
        "cluster_key": list(key),
        "doc_id": key[0],
        "canonical_caption_id": caption.caption_id,
        "canonical_origin": caption.origin,
        "caption_type": caption.caption_type,
        "caption_number": caption.caption_number,
        "caption_candidate_class": _candidate_class(caption),
        "main_caption_number": _main_caption_number(caption.caption_number),
        "subfigure_marker": _subfigure_marker(caption.caption_number),
        "normalized_caption_text": caption.normalized_text,
        "paired_float_id": canonical.float_candidate.float_id if canonical.float_candidate else None,
        "member_count": len(members),
        "member_caption_ids": [member.caption.caption_id for member in members],
        "member_origins": sorted({member.caption.origin for member in members}),
        "source_v8_id_sets": [member.caption.source_v8_ids for member in members],
        "bbox_available": any(member.caption.bbox for member in members),
    }


def _subfigure_marker(value: str | None) -> str | None:
    match = re.search(r"\(([a-zA-Z0-9]+)\)\s*$", str(value or ""))
    return match.group(1).casefold() if match else None


def _main_caption_number(value: str | None) -> str:
    return re.sub(r"\([a-zA-Z0-9]+\)\s*$", "", str(value or "").strip()).casefold()


def _duplicate_candidate_class(caption: CaptionCandidate) -> str:
    base = _candidate_class(caption)
    if base != "REAL_CAPTION":
        return base
    if caption.origin == "crop_metadata":
        return "CROP_METADATA_DUPLICATE"
    if caption.origin in {"caption_metadata", "float_metadata"}:
        return "METADATA_DUPLICATE"
    return "METADATA_DUPLICATE"
