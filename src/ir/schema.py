"""Stable contracts between PDF extraction, TeX labeling, GNNs, and rendering.

The classes in this module are intentionally lightweight.  They describe the
files that cross module boundaries; they do not perform extraction, labeling,
model inference, or rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


DOCUMENT_IR_SCHEMA_VERSION = "document_ir_v1"
GRAPH_INPUT_SCHEMA_VERSION = "graph_input_v1"
GRAPH_LABELS_SCHEMA_VERSION = "graph_labels_v1"
PREDICTED_RELATIONS_SCHEMA_VERSION = "predicted_relations_v1"
RENDER_TREE_SCHEMA_VERSION = "render_tree_ir_v1"
STYLE_PROFILE_SCHEMA_VERSION = "style_profile_v1"


class ContractError(ValueError):
    """Raised when an IR payload violates the stable boundary contract."""


class BlockType(str, Enum):
    TEXT = "text"
    TITLE = "title"
    EQUATION = "equation"
    INLINE_MATH = "inline_math"
    TABLE = "table"
    FIGURE = "figure"
    ALGORITHM = "algorithm"
    LIST = "list"
    CODE = "code"
    REFERENCE = "reference"
    TOC = "toc"
    HEADER_FOOTER = "header_footer"
    OTHER = "other"


class RelationLabel(int, Enum):
    MERGE = 0
    PARENT_CHILD = 1
    NONE = 2


class RenderRole(str, Enum):
    ROOT = "root"
    DOCUMENT_TITLE = "document_title"
    AUTHOR_BLOCK = "author_block"
    ABSTRACT = "abstract"
    SECTION = "section"
    SUBSECTION = "subsection"
    SUBSUBSECTION = "subsubsection"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    DISPLAY_EQUATION = "display_equation"
    INLINE_MATH = "inline_math"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"
    ALGORITHM = "algorithm"
    CODE = "code"
    REFERENCES = "references"
    REFERENCE_ITEM = "reference_item"
    TOC_PLACEHOLDER = "toc_placeholder"
    RAW_LATEX = "raw_latex"
    UNKNOWN = "unknown"


class RendererMode(str, Enum):
    ORIGINAL_LIKE = "original_like"
    JOURNAL_TEMPLATE = "journal_template"
    LEARNED_STYLE = "learned_style"


class CoordinateSpace(str, Enum):
    PAGE_NORMALIZED_1000 = "page_normalized_1000"
    PDF_POINTS = "pdf_points"


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_list(cls, values: list[float] | tuple[float, float, float, float]) -> "BBox":
        if len(values) != 4:
            raise ContractError(f"bbox must have four values, got {len(values)}")
        return cls(float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass(frozen=True)
class SourceRef:
    path: str | None = None
    page_idx: int | None = None
    extractor: str | None = None
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StyleSpan:
    text: str
    font_name: str | None = None
    font_size: float | None = None
    is_bold: bool = False
    is_italic: bool = False
    is_inline_math: bool = False
    is_inline_code: bool = False
    bbox: BBox | None = None
    source: SourceRef | None = None


@dataclass(frozen=True)
class DocumentNode:
    node_id: str
    node_type: BlockType
    text: str
    page_idx: int
    bboxes: list[BBox]
    reading_index: int
    raw_type: str | None = None
    list_type: str | None = None
    spans: list[StyleSpan] = field(default_factory=list)
    children_hint: list[str] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)
    features: dict[str, float | int | bool | str | None] = field(default_factory=dict)
    source_refs: list[SourceRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PageIR:
    page_idx: int
    width: float
    height: float
    node_ids: list[str] = field(default_factory=list)
    coordinate_space: CoordinateSpace = CoordinateSpace.PAGE_NORMALIZED_1000
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentIR:
    doc_id: str
    pages: list[PageIR]
    nodes: list[DocumentNode]
    schema_version: str = DOCUMENT_IR_SCHEMA_VERSION
    source_pdf: str | None = None
    coordinate_space: CoordinateSpace = CoordinateSpace.PAGE_NORMALIZED_1000
    reading_order: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphTensorRef:
    path: str
    tensor_name: str
    shape: list[int]
    dtype: str
    description: str | None = None


@dataclass(frozen=True)
class GraphInput:
    doc_id: str
    document_ir_path: str
    graph_path: str
    node_ids: list[str]
    edge_ids: list[str]
    x: GraphTensorRef
    edge_index: GraphTensorRef
    edge_attr: GraphTensorRef
    schema_version: str = GRAPH_INPUT_SCHEMA_VERSION
    graph_schema_version: str = "graph_v7"
    feature_schema_version: str = "feature_schema_v0"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignmentEntry:
    tex_id: str
    pdf_node_ids: list[str]
    score: float | None = None
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphLabels:
    doc_id: str
    graph_input_path: str
    edge_ids: list[str]
    y: list[RelationLabel]
    alignments: list[AlignmentEntry] = field(default_factory=list)
    schema_version: str = GRAPH_LABELS_SCHEMA_VERSION
    label_vocab: list[str] = field(
        default_factory=lambda: [
            RelationLabel.MERGE.name,
            RelationLabel.PARENT_CHILD.name,
            RelationLabel.NONE.name,
        ]
    )
    quality_report: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictedRelations:
    doc_id: str
    graph_input_path: str
    edge_ids: list[str]
    predicted_labels: list[RelationLabel]
    probabilities: list[list[float]] = field(default_factory=list)
    logits: list[list[float]] = field(default_factory=list)
    schema_version: str = PREDICTED_RELATIONS_SCHEMA_VERSION
    model_version: str | None = None
    threshold_config: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderTreeNode:
    render_id: str
    role: RenderRole
    source_node_ids: list[str] = field(default_factory=list)
    text: str | None = None
    latex: str | None = None
    children: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderTreeIR:
    doc_id: str
    root_id: str
    nodes: list[RenderTreeNode]
    document_ir_path: str
    schema_version: str = RENDER_TREE_SCHEMA_VERSION
    predicted_relations_path: str | None = None
    style_profile_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StyleProfile:
    profile_id: str
    mode: RendererMode
    schema_version: str = STYLE_PROFILE_SCHEMA_VERSION
    documentclass: str = "article"
    documentclass_options: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=lambda: ["amsmath", "amssymb", "graphicx", "float"])
    macros: list[str] = field(default_factory=list)
    page_layout: dict[str, Any] = field(default_factory=dict)
    role_styles: dict[str, dict[str, Any]] = field(default_factory=dict)
    renderer_options: dict[str, Any] = field(default_factory=dict)
    learned_from: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
