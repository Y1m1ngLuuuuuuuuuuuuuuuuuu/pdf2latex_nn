"""Build RenderTreeIR directly from v8 DocumentIR.

This is the decoder-side half of the v8 path.  It consumes the complete
DocumentIR, removes extracted front matter from the body tree, builds a simple
heading stack, and preserves source ids for the existing full-v7-first renderer.
No GNN view or graph artifact is involved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.ir import BlockType, DocumentIR, DocumentNode, RenderRole, RenderTreeIR, RenderTreeNode
from src.reasoning.front_matter_extractor import FrontMatterIR, FrontMatterSpan, extract_front_matter
from src.reasoning.v8_heading_style_stack import V8HeadingStyleResolver, build_v8_heading_style_resolver


HEADING_NUMBER_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+){0,3})\.?\s+(?P<title>.+)$")
REFERENCE_HEADING_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)
ABSTRACT_HEADING_RE = re.compile(r"^\s*(abstract|摘要)\s*$", re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*(?:[•●▪*-]|\d+[.)])\s+")
ORDERED_LIST_RE = re.compile(r"^\s*\d+[.)]\s+")
TOP_LEVEL_UNNUMBERED_HEADINGS = {
    "introduction",
    "related work",
    "background",
    "methodology",
    "method",
    "methods",
    "experiments",
    "experimental setup",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
}


@dataclass
class _MutableRenderNode:
    render_id: str
    role: RenderRole
    source_node_ids: list[str] = field(default_factory=list)
    text: str | None = None
    latex: str | None = None
    children: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def frozen(self) -> RenderTreeNode:
        return RenderTreeNode(
            render_id=self.render_id,
            role=self.role,
            source_node_ids=list(self.source_node_ids),
            text=self.text,
            latex=self.latex,
            children=list(self.children),
            attributes=dict(self.attributes),
        )


def build_v8_render_tree(
    document: DocumentIR,
    *,
    document_ir_path: str,
    front_matter: FrontMatterIR | None = None,
    enable_float_caption_layout: bool = False,
    enable_algorithm_region_renderer: bool = False,
) -> RenderTreeIR:
    """Build a hierarchical RenderTreeIR for v8 reconstruction."""

    extracted_front_matter = front_matter or extract_front_matter(document)
    abstract_layout_mode = front_matter_abstract_layout_mode(extracted_front_matter, document)
    consumed_front_ids = set(extracted_front_matter.consumed_source_node_ids)
    heading_style_resolver = build_v8_heading_style_resolver(document, excluded_source_ids=consumed_front_ids)
    mutable_nodes: dict[str, _MutableRenderNode] = {
        "root": _MutableRenderNode(render_id="root", role=RenderRole.ROOT)
    }
    heading_stack: list[tuple[int, str]] = []
    references_render_id: str | None = None
    add_explicit_front_matter_nodes(
        mutable_nodes,
        extracted_front_matter,
        abstract_layout_mode=abstract_layout_mode,
        document=document,
    )
    column_abstract_source_ids = set()
    if abstract_layout_mode == "column_flow" and extracted_front_matter.abstract is not None:
        column_abstract_source_ids.update(span_source_ids(extracted_front_matter.abstract.title))
        column_abstract_source_ids.update(span_source_ids(extracted_front_matter.abstract.body))
    inserted_column_abstract = False

    for source in sorted(document.nodes, key=lambda item: (item.reading_index, item.page_idx, item.node_id)):
        if (
            abstract_layout_mode == "column_flow"
            and not inserted_column_abstract
            and source.node_id in column_abstract_source_ids
            and extracted_front_matter.abstract is not None
        ):
            add_abstract_render_node(
                mutable_nodes,
                extracted_front_matter.abstract.title,
                extracted_front_matter.abstract.body,
                abstract_layout_mode=abstract_layout_mode,
                document=document,
                render_id="v8_front_abstract",
            )
            inserted_column_abstract = True
            continue
        if source.node_id in consumed_front_ids:
            continue
        role, level, heading_evidence = role_for_source_node(
            source,
            document=document,
            heading_style_resolver=heading_style_resolver,
        )
        render_id = f"r_{source.node_id}"
        source_node_ids = [] if role == RenderRole.REFERENCES else [source.node_id]
        render_text = None if role == RenderRole.REFERENCES else source.text
        attributes = {
            "v8_render_tree": True,
            "source_node_type": source.node_type.value,
            "source_raw_type": source.raw_type,
            "page_idx": source.page_idx,
            "reading_index": source.reading_index,
        }
        if role == RenderRole.LIST_ITEM and heading_evidence:
            attributes.update(heading_evidence)
        elif heading_evidence:
            attributes["heading_level_evidence"] = heading_evidence
        if role == RenderRole.REFERENCES:
            attributes["reference_heading_source_node_id"] = source.node_id
            attributes["reference_heading_text"] = source.text
        mutable_nodes[render_id] = _MutableRenderNode(
            render_id=render_id,
            role=role,
            source_node_ids=source_node_ids,
            text=render_text,
            attributes=attributes,
        )

        if role in {RenderRole.SECTION, RenderRole.SUBSECTION, RenderRole.SUBSUBSECTION, RenderRole.REFERENCES}:
            if role == RenderRole.REFERENCES:
                level = 1
                references_render_id = render_id
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            parent_id = heading_stack[-1][1] if heading_stack else "root"
            mutable_nodes[parent_id].children.append(render_id)
            heading_stack.append((level, render_id))
            continue

        if role == RenderRole.REFERENCE_ITEM and references_render_id is not None:
            mutable_nodes[references_render_id].children.append(render_id)
            continue

        parent_id = heading_stack[-1][1] if heading_stack else "root"
        mutable_nodes[parent_id].children.append(render_id)

    tree = RenderTreeIR(
        doc_id=document.doc_id,
        root_id="root",
        nodes=[node.frozen() for node in mutable_nodes.values()],
        document_ir_path=document_ir_path,
        metadata={
            "builder": "v8_render_tree",
            "front_matter_consumed_source_ids": sorted(consumed_front_ids),
            "front_matter_abstract_layout_mode": abstract_layout_mode,
            "front_matter_diag": extracted_front_matter.to_diagnostic(),
            "heading_style_registry": heading_style_resolver.to_diagnostic(),
        },
    )
    if enable_float_caption_layout:
        from src.reasoning.float_caption_layout import apply_float_caption_layout

        tree, _layout_result = apply_float_caption_layout(document, tree, enabled=True)
    if enable_algorithm_region_renderer:
        from src.reasoning.algorithm_region_renderer import apply_algorithm_region_renderer

        tree, _algorithm_result = apply_algorithm_region_renderer(document, tree, enabled=True)
    return tree


def add_explicit_front_matter_nodes(
    mutable_nodes: dict[str, _MutableRenderNode],
    front_matter: FrontMatterIR,
    *,
    abstract_layout_mode: str,
    document: DocumentIR,
) -> None:
    root = mutable_nodes["root"]
    if front_matter.title is not None:
        render_id = "v8_front_title"
        mutable_nodes[render_id] = _MutableRenderNode(
            render_id=render_id,
            role=RenderRole.DOCUMENT_TITLE,
            source_node_ids=front_matter.title.source_node_ids,
            text=front_matter.title.text,
            attributes={"v8_front_matter": True, "front_matter_role": "title"},
        )
        root.children.append(render_id)

    author_spans = [
        *front_matter.authors,
        *front_matter.affiliations,
        *front_matter.emails,
        *front_matter.notes,
        *front_matter.misc,
    ]
    if author_spans:
        render_id = "v8_front_author_block"
        mutable_nodes[render_id] = _MutableRenderNode(
            render_id=render_id,
            role=RenderRole.AUTHOR_BLOCK,
            source_node_ids=[node_id for span in author_spans for node_id in span.source_node_ids],
            text="\n".join(span.text for span in author_spans if span.text.strip()),
            attributes={"v8_front_matter": True, "front_matter_role": "author_block"},
        )
        root.children.append(render_id)

    if abstract_layout_mode == "full_width" and front_matter.abstract is not None:
        add_abstract_render_node(
            mutable_nodes,
            front_matter.abstract.title,
            front_matter.abstract.body,
            abstract_layout_mode=abstract_layout_mode,
            document=document,
            render_id="v8_front_abstract",
        )


def add_abstract_render_node(
    mutable_nodes: dict[str, _MutableRenderNode],
    title: FrontMatterSpan | None,
    body: FrontMatterSpan | None,
    *,
    abstract_layout_mode: str,
    document: DocumentIR,
    render_id: str,
) -> None:
    if body is None or not body.text.strip():
        return
    source_node_ids = [*span_source_ids(title), *span_source_ids(body)]
    mutable_nodes[render_id] = _MutableRenderNode(
        render_id=render_id,
        role=RenderRole.ABSTRACT,
        source_node_ids=source_node_ids,
        text=body.text,
        attributes={
            "v8_front_matter": True,
            "front_matter_role": "abstract",
            "abstract_layout_mode": abstract_layout_mode,
        },
    )
    mutable_nodes["root"].children.append(render_id)


def span_source_ids(span: FrontMatterSpan | None) -> list[str]:
    return list(span.source_node_ids) if span is not None else []


def front_matter_abstract_layout_mode(front_matter: FrontMatterIR, document: DocumentIR) -> str:
    if front_matter.abstract is None:
        return "none"
    source_ids = [*span_source_ids(front_matter.abstract.title), *span_source_ids(front_matter.abstract.body)]
    source_nodes = [node for node in document.nodes if node.node_id in set(source_ids)]
    if not source_nodes:
        return "full_width"
    boxes = [box for node in source_nodes for box in node.bboxes]
    if not boxes:
        return "full_width"
    page_width = max((page.width for page in document.pages if page.page_idx == source_nodes[0].page_idx), default=612.0)
    x0 = min(box.x0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    width_ratio = (x1 - x0) / page_width if page_width else 1.0
    columns = {
        node.features.get("column_id")
        for node in source_nodes
        if node.features.get("column_id") not in (None, -1, "-1")
    }
    if width_ratio < 0.62 and (columns or x1 <= page_width * 0.55 or x0 >= page_width * 0.45):
        return "column_flow"
    return "full_width"


def role_for_source_node(
    node: DocumentNode,
    *,
    document: DocumentIR | None = None,
    heading_style_resolver: V8HeadingStyleResolver | None = None,
) -> tuple[RenderRole, int, dict[str, Any]]:
    text = clean_text(node.text)
    if node.node_type == BlockType.TITLE:
        if REFERENCE_HEADING_RE.match(text):
            return RenderRole.REFERENCES, 1, {"rule": "reference_heading"}
        if ABSTRACT_HEADING_RE.match(text):
            return RenderRole.ABSTRACT, 1, {"rule": "abstract_heading"}
        if heading_style_resolver is not None:
            level, evidence = heading_style_resolver.resolve(node)
        else:
            level, evidence = heading_level_from_node(node, document=document)
        return render_role_from_heading_level(level), level, evidence
    if node.node_type == BlockType.TABLE:
        return RenderRole.TABLE, 0, {}
    if node.node_type == BlockType.FIGURE:
        return RenderRole.FIGURE, 0, {}
    if node.node_type == BlockType.ALGORITHM:
        return RenderRole.ALGORITHM, 0, {}
    if node.node_type == BlockType.EQUATION:
        return RenderRole.DISPLAY_EQUATION, 0, {}
    if node.node_type == BlockType.CODE:
        return RenderRole.CODE, 0, {}
    if node.node_type == BlockType.LIST or BULLET_RE.match(text):
        ordered = _node_is_ordered_list_item(node, text)
        return RenderRole.LIST_ITEM, 0, {
            "ordered": ordered,
            "list_marker_source": "ordered_prefix" if ordered else "bullet_prefix_or_list_type",
        }
    if node.node_type == BlockType.REFERENCE:
        return RenderRole.REFERENCE_ITEM, 0, {}
    if node.node_type == BlockType.FOOTNOTE:
        return RenderRole.FOOTNOTE, 0, {}
    if node.node_type == BlockType.TOC:
        return RenderRole.TOC_PLACEHOLDER, 0, {}
    return RenderRole.PARAGRAPH, 0, {}


def _node_is_ordered_list_item(node: DocumentNode, text: str) -> bool:
    list_type = str(node.list_type or node.metadata.get("list_type") or "").casefold()
    if list_type in {"ordered", "enumerate", "numbered", "number", "alpha", "roman"}:
        return True
    if list_type in {"unordered", "itemize", "bullet", "bulleted"}:
        return False
    return bool(ORDERED_LIST_RE.match(text))


def heading_level_from_node(node: DocumentNode, *, document: DocumentIR | None = None) -> tuple[int, dict[str, Any]]:
    text = clean_text(node.text)
    match = HEADING_NUMBER_RE.match(text)
    if match:
        depth = len([part for part in match.group("num").split(".") if part])
        level = min(max(depth, 1), 3)
        return level, {"rule": "numbering_depth", "numbering_depth": depth, "level": level}

    normalized = normalize_heading_text(text)
    centered, centered_evidence = is_visually_centered_heading(node, document=document)
    if normalized in TOP_LEVEL_UNNUMBERED_HEADINGS:
        return 1, {"rule": "known_top_level_heading", "normalized_text": normalized, **centered_evidence}
    if centered:
        return 1, {"rule": "document_local_centered_heading", "normalized_text": normalized, **centered_evidence}
    return 2, {"rule": "left_aligned_unnumbered_heading", "normalized_text": normalized, **centered_evidence}


def normalize_heading_text(text: str) -> str:
    return clean_text(text).casefold().strip(" .:")


def is_visually_centered_heading(node: DocumentNode, *, document: DocumentIR | None = None) -> tuple[bool, dict[str, Any]]:
    bbox = node.bboxes[0] if node.bboxes else None
    page_width = page_width_for_node(node, document=document)
    if bbox is None or page_width <= 0:
        return False, {"center_rule": "missing_bbox_or_page_width"}
    center = (bbox.x0 + bbox.x1) / 2.0
    width = max(0.0, bbox.x1 - bbox.x0)
    column_id = node.features.get("column_id")
    if column_id in (-1, "-1") or width >= page_width * 0.50:
        target_center = page_width / 2.0
        column_kind = "full_width"
    elif center < page_width * 0.52:
        target_center = page_width * 0.25
        column_kind = "left_column"
    else:
        target_center = page_width * 0.75
        column_kind = "right_column"
    distance = abs(center - target_center)
    threshold = max(24.0, page_width * 0.050)
    x0 = float(bbox.x0)
    if column_kind == "left_column":
        left_aligned_margin = x0 <= page_width * 0.12
    elif column_kind == "right_column":
        left_aligned_margin = abs(x0 - page_width * 0.515) <= page_width * 0.055
    else:
        left_aligned_margin = False
    centered = distance <= threshold and not left_aligned_margin
    return centered, {
        "center_rule": "column_center_distance",
        "column_kind": column_kind,
        "bbox_center_x": round(center, 3),
        "target_center_x": round(target_center, 3),
        "center_distance": round(distance, 3),
        "center_threshold": round(threshold, 3),
        "left_aligned_margin": left_aligned_margin,
    }


def page_width_for_node(node: DocumentNode, *, document: DocumentIR | None = None) -> float:
    value = node.features.get("page_width") or node.metadata.get("page_width")
    try:
        width = float(value)
        if width > 0:
            return width
    except (TypeError, ValueError):
        pass
    if document is not None:
        for page in document.pages:
            if page.page_idx == node.page_idx and page.width > 0:
                return float(page.width)
    if node.bboxes:
        return max(1.0, float(node.bboxes[0].x1))
    return 612.0


def render_role_from_heading_level(level: int) -> RenderRole:
    if level <= 1:
        return RenderRole.SECTION
    if level == 2:
        return RenderRole.SUBSECTION
    return RenderRole.SUBSUBSECTION


def clean_text(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip()
