"""Stable perception IR and feature schema definitions.

This module intentionally contains structure only: enums, dataclasses, and
dimension constants shared by MinerU, PyMuPDF, BERT, GNN, validators, and
generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "feature_schema_v0"
COORDINATE_SPACE = "page_normalized_1000"
PAGE_COORD_MIN = 0.0
PAGE_COORD_MAX = 1000.0
SCIBERT_DIM = 768


class BlockType(str, Enum):
    TEXT = "text"
    TITLE = "title"
    EQUATION = "equation"
    TABLE = "table"
    FIGURE = "figure"
    ALGORITHM = "algorithm"
    LIST = "list"
    CODE = "code"
    REFERENCE = "reference"
    OTHER = "other"


class RawSource(str, Enum):
    MINERU = "mineru"
    PYMUPDF = "pymupdf"
    DERIVED = "derived"
    MANUAL = "manual"


class EdgeRelation(str, Enum):
    NEXT_READING_ORDER = "next_reading_order"
    SAME_PAGE_NEXT = "same_page_next"
    CROSS_PAGE_NEXT = "cross_page_next"
    CROSS_COLUMN_NEXT = "cross_column_next"
    FLOAT_SEPARATED_CONTINUATION = "float_separated_continuation"
    PARENT_CHILD = "parent_child"
    CAPTION_OF = "caption_of"
    REFERENCE_ITEM_OF = "reference_item_of"


FEATURE_TYPE_VOCAB = [block_type.value for block_type in BlockType]
GEOMETRY_FIELDS = ["x_start_local", "y_start_page", "x_end_local", "y_end_page"]
DERIVED_STAT_FIELDS = ["macro_position", "aspect_ratio", "text_density"]
EDGE_ATTR_FIELDS = [
    "semantic_cosine",
    "delta_x_start",
    "delta_y_start",
    "delta_x_end",
    "delta_y_end",
    "vertical_gap",
    "horizontal_overlap",
    "same_page",
    "same_column",
    "cross_page",
    "cross_column",
    "same_type",
    "source_ends_with_hyphen",
    "source_has_terminal_punctuation",
    "target_starts_lowercase",
    "is_forward_edge",
]
NON_TEXT_DENSITY_TYPES = {
    BlockType.EQUATION.value,
    BlockType.TABLE.value,
    BlockType.FIGURE.value,
    BlockType.ALGORITHM.value,
    BlockType.CODE.value,
}
PLACEHOLDER_TEXT = {
    BlockType.EQUATION.value: "[EQUATION]",
    BlockType.TABLE.value: "[TABLE]",
    BlockType.FIGURE.value: "[FIGURE]",
    BlockType.ALGORITHM.value: "[ALGORITHM]",
    BlockType.CODE.value: "[CODE]",
    BlockType.REFERENCE.value: "[REFERENCE]",
    BlockType.OTHER.value: "[EMPTY]",
    BlockType.TEXT.value: "[EMPTY]",
    BlockType.TITLE.value: "[EMPTY]",
    BlockType.LIST.value: "[EMPTY]",
}


@dataclass(frozen=True)
class BBox:
    """A rectangle in normalized page coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class TextSpan:
    """A consecutive run of text sharing the same style state."""

    text: str
    font_name: str | None = None
    font_size: float | None = None
    is_bold: bool = False
    is_italic: bool = False
    is_inline_math: bool = False
    is_inline_code: bool = False
    bbox: BBox | None = None
    source: RawSource = RawSource.PYMUPDF


@dataclass(frozen=True)
class Line:
    """A visual text line belonging to a page and optionally a block."""

    line_id: str
    page_idx: int
    bbox: BBox
    text: str
    spans: list[TextSpan] = field(default_factory=list)


@dataclass(frozen=True)
class ReferenceItem:
    """One bibliographic entry inside a reference block."""

    text: str
    raw_index: int | None = None
    bbox: BBox | None = None


@dataclass(frozen=True)
class Block:
    """A graph node candidate extracted from one or more visual boxes."""

    block_id: str
    global_order: int
    block_type: BlockType
    page_idx: int
    bboxes: list[BBox]
    text: str = ""
    raw_type: str | None = None
    list_type: str | None = None
    column_id: int | None = None
    is_full_width: bool = False
    merge_count: int = 1
    source_page_idxs: list[int] = field(default_factory=list)
    source_visual_orders: list[int] = field(default_factory=list)
    source_original_indexes: list[int] = field(default_factory=list)
    reference_items: list[ReferenceItem] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    spans: list[TextSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Page:
    """A normalized PDF page."""

    page_idx: int
    width: float
    height: float
    blocks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeCandidate:
    """A candidate relation between two block nodes."""

    source_block_id: str
    target_block_id: str
    relation: EdgeRelation
    features: dict[str, float | int | bool | str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureTensorSchema:
    """Fixed feature layout for PyG node tensors."""

    semantic_dim: int = SCIBERT_DIM
    type_vocab: list[str] = field(default_factory=lambda: list(FEATURE_TYPE_VOCAB))
    geometry_fields: list[str] = field(default_factory=lambda: list(GEOMETRY_FIELDS))
    derived_stat_fields: list[str] = field(default_factory=lambda: list(DERIVED_STAT_FIELDS))

    @property
    def type_dim(self) -> int:
        return len(self.type_vocab)

    @property
    def node_feature_dim(self) -> int:
        return self.semantic_dim + self.type_dim + len(self.geometry_fields) + len(self.derived_stat_fields)

    @property
    def edge_attr_fields(self) -> list[str]:
        return list(EDGE_ATTR_FIELDS)

    @property
    def edge_attr_dim(self) -> int:
        return len(self.edge_attr_fields)


@dataclass(frozen=True)
class Document:
    """The unified per-PDF IR consumed by validation, graphing, and generation."""

    document_id: str
    pages: list[Page]
    blocks: list[Block]
    edges: list[EdgeCandidate] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    coordinate_space: str = COORDINATE_SPACE
    source_pdf: str | None = None
    feature_schema: FeatureTensorSchema = field(default_factory=FeatureTensorSchema)
    metadata: dict[str, Any] = field(default_factory=dict)
