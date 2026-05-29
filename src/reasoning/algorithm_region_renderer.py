"""Experimental AlgorithmRegion renderer wiring for v8.

This module consumes only high-confidence algorithm evidence preserved by the
MinerU v8 adapter. It does not use broad keyword detector candidates and it
does not mutate raw MinerU/v8 JSON, graph tensors, or deterministic merge
decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from src.ir import BlockType, DocumentIR, DocumentNode, RenderRole, RenderTreeIR, RenderTreeNode


ALLOWED_PRESERVATION_STATUSES = {"mapped_to_document_ir", "mapped_to_v8"}
ALLOWED_CONFIDENCE = {"strong_subtype", "strong_v2_algorithm", "medium_caption"}
FALSE_ALGORITHM_REFERENCE_RE = re.compile(
    r"\b(?:Algorithm|Alg\.?)\s+\d+(?:\.\d+)*(?:\([A-Za-z0-9]+\))?\s+"
    r"(?:shows?|is\s+used|are\s+used|uses?|illustrates?|describes?|presents?)\b",
    re.IGNORECASE,
)
ALGORITHM_CAPTION_RE = re.compile(
    r"^\s*(?:Algorithm|Alg\.?|Procedure)\s*"
    r"(?P<number>\d+(?:\.\d+)*(?:\([A-Za-z0-9]+\))?|[IVXLCDM]+)?"
    r"\s*[:.\-–—]?\s*(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)
SPECIAL_CHAR_RE = re.compile(r"(?<!\\)[#%&_]")
RISKY_UNICODE_RE = re.compile(r"[✓✗✘✔×□■●▲▶→⇒≤≥∈∞−∑∏∂∇ϵηρΓ]")


@dataclass(frozen=True)
class AlgorithmRegionIR:
    region_id: str
    doc_id: str
    page_idx: int | None
    source_v8_ids: list[str] = field(default_factory=list)
    document_ir_node_ids: list[str] = field(default_factory=list)
    bbox: list[float] | None = None
    algorithm_caption: str = ""
    algorithm_body: str = ""
    algorithm_origin: str = "metadata"
    algorithm_confidence: str = "weak_text_only"
    preservation_status: str = "unknown"
    render_policy: str = "diagnostic_only"
    compile_risk_flags: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlgorithmRegionRenderResult:
    regions: list[AlgorithmRegionIR] = field(default_factory=list)
    rendered_regions: list[dict[str, Any]] = field(default_factory=list)
    consumed_nodes: list[dict[str, Any]] = field(default_factory=list)
    compile_risks: list[dict[str, Any]] = field(default_factory=list)
    render_policies: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_only: list[dict[str, Any]] = field(default_factory=list)
    false_positive_blocked: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_diagnostic(self) -> dict[str, Any]:
        return {
            "algorithm_regions": [region.to_record() for region in self.regions],
            "rendered_regions": self.rendered_regions,
            "consumed_nodes": self.consumed_nodes,
            "compile_risks": self.compile_risks,
            "render_policies": self.render_policies,
            "diagnostic_only": self.diagnostic_only,
            "false_positive_blocked": self.false_positive_blocked,
            "notes": self.notes,
        }


def build_algorithm_region_sidecars(document: DocumentIR) -> AlgorithmRegionRenderResult:
    regions: list[AlgorithmRegionIR] = []
    false_positive_blocked: list[dict[str, Any]] = []
    diagnostic_only: list[dict[str, Any]] = []
    for node in sorted(document.nodes, key=lambda item: (item.reading_index, item.page_idx, item.node_id)):
        region, blocked_reason = region_from_document_node(node, document=document)
        if blocked_reason:
            false_positive_blocked.append(
                {
                    "node_id": node.node_id,
                    "page_idx": node.page_idx,
                    "text_preview": compact_text(node.text),
                    "reason": blocked_reason,
                }
            )
            continue
        if region is None:
            continue
        regions.append(region)
        if region.render_policy == "diagnostic_only":
            diagnostic_only.append(region.to_record())
    return AlgorithmRegionRenderResult(
        regions=regions,
        compile_risks=[
            {
                "region_id": region.region_id,
                "doc_id": region.doc_id,
                "page_idx": region.page_idx,
                "compile_risk_flags": list(region.compile_risk_flags),
                "render_policy": region.render_policy,
            }
            for region in regions
            if region.compile_risk_flags
        ],
        render_policies=[
            {
                "region_id": region.region_id,
                "render_policy": region.render_policy,
                "algorithm_confidence": region.algorithm_confidence,
                "preservation_status": region.preservation_status,
            }
            for region in regions
        ],
        diagnostic_only=diagnostic_only,
        false_positive_blocked=false_positive_blocked,
    )


def apply_algorithm_region_renderer(
    document: DocumentIR,
    tree: RenderTreeIR,
    *,
    enabled: bool = False,
) -> tuple[RenderTreeIR, AlgorithmRegionRenderResult]:
    """Apply experimental AlgorithmRegion materialization when enabled."""

    base_result = build_algorithm_region_sidecars(document)
    if not enabled:
        return tree, base_result

    nodes_by_id = {node.render_id: node for node in tree.nodes}
    source_to_render_ids: dict[str, list[str]] = {}
    parent_by_child_id: dict[str, str] = {}
    for node in tree.nodes:
        for source_id in node.source_node_ids:
            source_to_render_ids.setdefault(source_id, []).append(node.render_id)
        for child_id in node.children:
            parent_by_child_id[child_id] = node.render_id

    updated_nodes = dict(nodes_by_id)
    insertions: list[tuple[str, str, str | None]] = []
    consumed_render_ids: set[str] = set()
    rendered_regions: list[dict[str, Any]] = []
    consumed_nodes: list[dict[str, Any]] = []

    for region in base_result.regions:
        if not region_is_renderable(region):
            continue
        render_id = f"algorithm_region_{safe_id(region.region_id)}"
        source_render_ids = [
            render_id
            for source_id in region.document_ir_node_ids
            for render_id in source_to_render_ids.get(source_id, [])
            if render_id in updated_nodes
        ]
        parent_id = parent_by_child_id.get(source_render_ids[0], tree.root_id) if source_render_ids else tree.root_id
        insert_after = source_render_ids[0] if source_render_ids else None
        updated_nodes[render_id] = RenderTreeNode(
            render_id=render_id,
            role=RenderRole.ALGORITHM,
            source_node_ids=list(region.document_ir_node_ids),
            text=region.algorithm_body or region.algorithm_caption,
            attributes={
                "algorithm_region_phase0": True,
                "algorithm_region": region.to_record(),
                "algorithm_caption": region.algorithm_caption,
                "algorithm_body": region.algorithm_body,
                "render_policy": region.render_policy,
                "compile_risk_flags": list(region.compile_risk_flags),
            },
        )
        insertions.append((parent_id, render_id, insert_after))
        consumed_render_ids.update(source_render_ids)
        rendered_regions.append(
            {
                "region_id": region.region_id,
                "render_id": render_id,
                "source_render_ids": source_render_ids,
                "render_policy": region.render_policy,
                "algorithm_confidence": region.algorithm_confidence,
            }
        )
        for source_id in region.document_ir_node_ids:
            consumed_nodes.append(
                {
                    "region_id": region.region_id,
                    "source_node_id": source_id,
                    "reason": "algorithm_region_phase0_materialized",
                    "render_policy": region.render_policy,
                }
            )

    for render_id in consumed_render_ids:
        node = updated_nodes.get(render_id)
        if node is None:
            continue
        attrs = dict(node.attributes)
        attrs["algorithm_region_consumed"] = True
        attrs["algorithm_region_suppression_reason"] = "materialized_as_algorithm_region_phase0"
        updated_nodes[render_id] = RenderTreeNode(
            render_id=node.render_id,
            role=node.role,
            source_node_ids=list(node.source_node_ids),
            text=node.text,
            latex=node.latex,
            children=list(node.children),
            attributes=attrs,
        )

    final_nodes = insert_algorithm_nodes(tree, updated_nodes, insertions, consumed_render_ids)
    result = AlgorithmRegionRenderResult(
        regions=base_result.regions,
        rendered_regions=rendered_regions,
        consumed_nodes=consumed_nodes,
        compile_risks=base_result.compile_risks,
        render_policies=base_result.render_policies,
        diagnostic_only=base_result.diagnostic_only,
        false_positive_blocked=base_result.false_positive_blocked,
        notes=["experimental_algorithm_region_renderer_phase0_enabled"],
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
                "experimental_algorithm_region_renderer_phase0_enabled": True,
                "algorithm_region_renderer_diag": result.to_diagnostic(),
            },
        ),
        result,
    )


def region_from_document_node(node: DocumentNode, *, document: DocumentIR) -> tuple[AlgorithmRegionIR | None, str | None]:
    metadata = dict(node.metadata or {})
    text = compact_text(node.text, limit=1000)
    if FALSE_ALGORITHM_REFERENCE_RE.search(text) and not metadata_algorithm_evidence(metadata):
        return None, "ordinary_algorithm_reference"
    confidence = str(metadata.get("algorithm_confidence") or "").strip()
    if node.node_type != BlockType.ALGORITHM and not metadata_algorithm_evidence(metadata):
        return None, None
    if confidence == "weak_text_only":
        return None, None
    if confidence not in ALLOWED_CONFIDENCE:
        confidence = "strong_subtype" if metadata_algorithm_evidence(metadata) else "weak_text_only"
    if confidence not in ALLOWED_CONFIDENCE:
        return None, None
    origin = str(metadata.get("algorithm_origin") or "metadata")
    caption = first_metadata_text(metadata, ("algorithm_caption", "code_caption"))
    body = first_metadata_text(metadata, ("algorithm_content", "code_body")) or text
    if not caption:
        caption = caption_from_text(body)
    body = strip_caption_from_body(body, caption)
    risks = compile_risk_flags(body)
    policy = render_policy_for_region(node, caption=caption, body=body, risks=risks)
    bbox = bbox_to_list(node.bboxes[0]) if node.bboxes else None
    source_v8_ids = list(metadata.get("source_v8_ids") or metadata.get("source_refs") or [])
    if not source_v8_ids:
        source_v8_ids = [node.node_id]
    region = AlgorithmRegionIR(
        region_id=f"{document.doc_id}_{node.node_id}",
        doc_id=document.doc_id,
        page_idx=node.page_idx,
        source_v8_ids=[str(value) for value in source_v8_ids],
        document_ir_node_ids=[node.node_id],
        bbox=bbox,
        algorithm_caption=caption,
        algorithm_body=body,
        algorithm_origin=origin,
        algorithm_confidence=confidence,
        preservation_status="mapped_to_document_ir",
        render_policy=policy,
        compile_risk_flags=risks,
        evidence=algorithm_evidence(metadata),
    )
    return region, None


def render_policy_for_region(node: DocumentNode, *, caption: str, body: str, risks: list[str]) -> str:
    if node.bboxes and not risks:
        return "crop_fallback"
    if body.strip():
        return "verbatim_fallback"
    if caption.strip():
        return "caption_only_placeholder"
    return "diagnostic_only"


def region_is_renderable(region: AlgorithmRegionIR) -> bool:
    return (
        region.preservation_status in ALLOWED_PRESERVATION_STATUSES
        and region.algorithm_confidence in ALLOWED_CONFIDENCE
        and region.render_policy in {"crop_fallback", "verbatim_fallback", "caption_only_placeholder"}
    )


def insert_algorithm_nodes(
    tree: RenderTreeIR,
    updated_nodes: dict[str, RenderTreeNode],
    insertions: list[tuple[str, str, str | None]],
    consumed_render_ids: set[str],
) -> list[RenderTreeNode]:
    by_parent: dict[str, list[tuple[str, str | None]]] = {}
    for parent_id, render_id, insert_after in insertions:
        by_parent.setdefault(parent_id, []).append((render_id, insert_after))
    existing_ids = {node.render_id for node in tree.nodes}
    result: list[RenderTreeNode] = []
    for original in tree.nodes:
        node = updated_nodes[original.render_id]
        children: list[str] = []
        additions = by_parent.get(node.render_id, [])
        added: set[str] = set()
        for child_id in node.children:
            if child_id in consumed_render_ids:
                for render_id, anchor_id in additions:
                    if anchor_id == child_id and render_id not in added:
                        children.append(render_id)
                        added.add(render_id)
                continue
            children.append(child_id)
            for render_id, anchor_id in additions:
                if anchor_id == child_id and render_id not in added:
                    children.append(render_id)
                    added.add(render_id)
        for render_id, anchor_id in additions:
            if render_id not in added:
                children.append(render_id)
                added.add(render_id)
        if node.render_id not in consumed_render_ids:
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


def metadata_algorithm_evidence(metadata: dict[str, Any]) -> bool:
    return bool(
        metadata.get("is_algorithm_subtype")
        or normalize(metadata.get("raw_sub_type")) == "algorithm"
        or normalize(metadata.get("mineru_subtype")) == "algorithm"
        or normalize(metadata.get("content_list_type")) == "algorithm"
        or metadata.get("algorithm_content")
        or metadata.get("algorithm_caption")
        or metadata.get("code_body")
        or metadata.get("code_caption")
    )


def algorithm_evidence(metadata: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    for key in (
        "is_algorithm_subtype",
        "raw_sub_type",
        "mineru_subtype",
        "content_list_type",
        "algorithm_content",
        "algorithm_caption",
        "code_body",
        "code_caption",
    ):
        if metadata.get(key):
            evidence.append(key)
    return evidence


def first_metadata_text(metadata: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, list):
            joined = "\n".join(str(part).strip() for part in value if str(part).strip()).strip()
            if joined:
                return joined
        elif isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def caption_from_text(text: str) -> str:
    first = first_nonempty_line(text)
    match = ALGORITHM_CAPTION_RE.match(first)
    if not match:
        return ""
    body = (match.group("body") or "").strip()
    number = match.group("number") or ""
    return " ".join(part for part in [f"Algorithm {number}".strip(), body] if part).strip()


def strip_caption_from_body(body: str, caption: str) -> str:
    lines = [line for line in str(body or "").splitlines()]
    if not lines:
        return ""
    first = first_nonempty_line(body)
    if caption and first and normalize(first).startswith(normalize(caption)[:32]):
        return "\n".join(line for line in lines[1:] if line.strip()).strip()
    if ALGORITHM_CAPTION_RE.match(first):
        return "\n".join(line for line in lines[1:] if line.strip()).strip()
    return body.strip()


def compile_risk_flags(text: str) -> list[str]:
    flags: list[str] = []
    if SPECIAL_CHAR_RE.search(text):
        flags.append("special_chars_escaped_by_phase0")
    if RISKY_UNICODE_RE.search(text):
        flags.append("unicode_symbol_sanitized_by_phase0")
    if text.count("{") != text.count("}"):
        flags.append("unbalanced_brace_escaped_by_phase0")
    return flags


def bbox_to_list(bbox: Any) -> list[float] | None:
    values = [getattr(bbox, name, None) for name in ("x0", "y0", "x1", "y1")]
    if all(value is not None for value in values):
        return [round(float(value), 3) for value in values]
    return None


def first_nonempty_line(text: str) -> str:
    for line in str(text or "").splitlines():
        if line.strip():
            return line.strip()
    return str(text or "").strip()


def compact_text(text: Any, *, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def normalize(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def safe_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "algorithm")).strip("_")
