"""Build PyTorch Geometric graphs from MinerU content v7 JSON."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.perception.schema import (
    COLUMN_FEATURE_FIELDS,
    DERIVED_STAT_FIELDS,
    EDGE_ATTR_FIELDS,
    FEATURE_TYPE_VOCAB,
    GEOMETRY_FIELDS,
    NON_TEXT_DENSITY_TYPES,
    PLACEHOLDER_TEXT,
    SCIBERT_DIM,
    SCROLL_GEOMETRY_FIELDS,
    SEQUENCE_POSITION_FIELDS,
    STYLE_STAT_FIELDS,
    TITLE_STRUCTURE_FIELDS,
)
from src.perception.reading_order import fuse_micro_nodes
from src.perception.xy_cut import reading_order_ranks as regime_reading_order_ranks
from src.perception.xy_cut import sort_node_indices_by_reading_order
from src.perception.title_features import title_pattern_flags
from src.pipeline.v7_contract import V7_GRAPH_SCHEMA_VERSION, V7_PIPELINE_VERSION

PAGE_SIZE = 1000.0
FULL_WIDTH_THRESHOLD = 620.0
TYPE_VOCAB = FEATURE_TYPE_VOCAB
EDGE_SOURCE_TYPES = [
    "sequential_forced",
    "sequential",
    "spatial_down",
    "spatial_right",
    "same_column_long_sight",
    "float_skip",
    "scope_anchor",
    "list_run_scope",
]
LIST_MARKER_RE = re.compile(r"^\s*(?:[\u2022\u25E6\u25CB\u25AA\-\*]|\d+[\.\)]|[a-zA-Z][\.\)])\s+")
STRUCTURAL_SKIP_TYPES = {"figure", "table", "algorithm", "equation"}
TEXT_FLOW_TYPES = {"text", "list", "reference"}
AUXILIARY_TYPES = {"page_header", "header", "page_footer", "footer", "page_number"}


@dataclass(frozen=True)
class ColumnFrame:
    x_min: float
    x_max: float

    @property
    def width(self) -> float:
        return max(1.0, self.x_max - self.x_min)

    def normalize_x(self, x: float) -> float:
        return (x - self.x_min) / self.width


@dataclass(frozen=True)
class PageFrames:
    left: ColumnFrame | None
    right: ColumnFrame | None


@dataclass(frozen=True)
class LogicalBox:
    item_index: int
    chunk_index: int
    page_idx: int
    bbox: tuple[float, float, float, float]
    full_span: bool

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class ScrollBox:
    item_index: int
    page_idx: int
    bbox: tuple[float, float, float, float]
    column_id: int
    column_width: float
    local_x0: float
    local_x1: float
    pseudo_y0: float
    pseudo_y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def local_cx(self) -> float:
        return (self.local_x0 + self.local_x1) / 2.0

    @property
    def pseudo_cy(self) -> float:
        return (self.pseudo_y0 + self.pseudo_y1) / 2.0


@dataclass(frozen=True)
class ScrollLayout:
    boxes: list[ScrollBox | None]
    ranks: list[int]
    total_scroll_height: float


@dataclass(frozen=True)
class GraphBuildConfig:
    model_path: Path
    max_length: int = 512
    stride: int = 384
    batch_size: int = 16
    bidirectional_edges: bool = True
    sequential_window: int = 15
    spatial_k: int = 3
    long_sight_window: int = 40
    scope_anchor_window: int = 80
    float_skip_window: int = 40
    fuse_micro_nodes: bool = False


def load_content_v7(path: Path) -> list[dict[str, Any]]:
    data = _load_content_v7_payload(path)
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Expected {path} to contain an items list")
    return [item for item in items if isinstance(item, dict)]


def _load_content_v7_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a v7 content object")
    return data


def build_graph_from_content_v7(input_path: Path, output_path: Path, config: GraphBuildConfig) -> Any:
    """Embed v7 nodes, concatenate geometry, and save a PyG Data object."""

    import torch
    from torch_geometric.data import Data

    payload = _load_content_v7_payload(input_path)
    raw_items = load_content_v7(input_path)
    items = fuse_micro_nodes(raw_items) if config.fuse_micro_nodes else raw_items
    texts = [text_for_embedding(item) for item in items]
    regime_order = sort_node_indices_by_reading_order(items)
    regime_ranks = _ranks_from_order(regime_order, len(items))
    column_ids = infer_column_ids(items)
    scroll_layout = build_scroll_layout(items, reading_order_indices=regime_order, column_ids=column_ids)
    semantic = embed_texts_scibert_cls(texts, config)
    type_onehot = build_type_onehot_matrix(items)
    geometry = build_geometry_matrix(items)
    scroll_geometry = build_scroll_geometry_matrix(items, scroll_layout=scroll_layout)
    stats = build_derived_stats_matrix(items, reading_order_ranks=regime_ranks)
    style_stats = build_style_stats_matrix(items)
    sequence_position = build_sequence_position_matrix(items, reading_order_ranks=regime_ranks)
    column_features = build_column_onehot_matrix(items, column_ids=column_ids)
    title_structure = build_title_structure_matrix(items)
    x = torch.cat(
        [
            semantic,
            type_onehot,
            geometry,
            scroll_geometry,
            stats,
            style_stats,
            sequence_position,
            column_features,
            title_structure,
        ],
        dim=1,
    )
    edge_pairs = build_candidate_edge_pairs(
        items,
        sequential_window=config.sequential_window,
        spatial_k=config.spatial_k,
        bidirectional=config.bidirectional_edges,
        reading_order_indices=regime_order,
        column_ids=column_ids,
        long_sight_window=config.long_sight_window,
        scope_anchor_window=config.scope_anchor_window,
        float_skip_window=config.float_skip_window,
    )
    edge_index = build_edge_index_from_pairs(edge_pairs)
    edge_attr = build_edge_attr_matrix(
        items,
        semantic,
        edge_pairs=edge_pairs,
        reading_order_ranks=regime_ranks,
        scroll_layout=scroll_layout,
    )
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_source_types = [source_type for _, _, source_type in edge_pairs]
    data.node_records = make_node_records(
        items,
        column_ids=column_ids,
        reading_order_ranks=regime_ranks,
        scroll_layout=scroll_layout,
    )
    data.feature_schema = build_node_feature_schema()
    data.edge_attr_schema = {
        "dim": len(EDGE_ATTR_FIELDS),
        "fields": EDGE_ATTR_FIELDS,
        "topology": {
            "strategy": "dual_view_knn",
            "sequential_window": config.sequential_window,
            "spatial_k": config.spatial_k,
            "long_sight_window": config.long_sight_window,
            "scope_anchor_window": config.scope_anchor_window,
            "float_skip_window": config.float_skip_window,
            "edge_source_types": EDGE_SOURCE_TYPES,
        },
    }
    data.source_path = str(input_path)
    data.model_path = str(config.model_path)
    data.pipeline_version = V7_PIPELINE_VERSION
    data.graph_schema_version = V7_GRAPH_SCHEMA_VERSION
    data.content_schema_version = str(payload.get("schema_version") or "")
    data.micro_fusion_applied = bool(config.fuse_micro_nodes)
    data.micro_fusion_node_count_before = len(raw_items)
    data.micro_fusion_node_count_after = len(items)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    return data


def embed_texts_scibert_cls(texts: list[str], config: GraphBuildConfig) -> Any:
    """Return an N x 768 tensor using mean pooled window CLS vectors."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(config.model_path), local_files_only=True).to(device)
    model.eval()

    vectors = []
    body_len = config.max_length - 2
    stride = min(config.stride, body_len)

    with torch.no_grad():
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                vectors.append(torch.zeros(model.config.hidden_size, dtype=torch.float32))
                continue

            windows = []
            start = 0
            while start < len(token_ids):
                chunk = token_ids[start : start + body_len]
                windows.append(tokenizer.build_inputs_with_special_tokens(chunk))
                if start + body_len >= len(token_ids):
                    break
                start += stride

            cls_vectors = []
            for batch_start in range(0, len(windows), config.batch_size):
                batch = windows[batch_start : batch_start + config.batch_size]
                padded = tokenizer.pad({"input_ids": batch}, padding=True, return_tensors="pt")
                padded = {key: value.to(device) for key, value in padded.items()}
                outputs = model(**padded)
                cls_vectors.append(outputs.last_hidden_state[:, 0, :].detach().cpu())
            vectors.append(torch.cat(cls_vectors, dim=0).mean(dim=0))

    return torch.stack(vectors, dim=0)


def build_type_onehot_matrix(items: list[dict[str, Any]]) -> Any:
    import torch

    rows = []
    for item in items:
        type_name = canonical_type(item)
        row = [0.0] * len(TYPE_VOCAB)
        row[TYPE_VOCAB.index(type_name)] = 1.0
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def build_geometry_matrix(items: list[dict[str, Any]]) -> Any:
    """Return N x 4 geometry tensor using first/last bbox local coordinates."""

    import torch

    page_frames = infer_page_frames(items)
    rows = []
    for item in items:
        chunks = list(iter_bbox_chunks(item.get("bbox")))
        if not chunks:
            rows.append([0.0, 0.0, 0.0, 0.0])
            continue

        pages = item.get("source_page_idxs")
        if not isinstance(pages, list) or not pages:
            pages = [item.get("page_idx")]
        first_page = int(pages[0]) if isinstance(pages[0], int) else int(item.get("page_idx") or 0)
        last_page = int(pages[-1]) if isinstance(pages[-1], int) else first_page
        first = chunks[0]
        last = chunks[-1]
        x_start = normalize_x_in_local_frame(first[0], first, first_page, page_frames, bool(item.get("is_full_width")))
        y_start = first[1] / PAGE_SIZE
        x_end = normalize_x_in_local_frame(last[2], last, last_page, page_frames, bool(item.get("is_full_width")))
        y_end = last[3] / PAGE_SIZE
        rows.append([x_start, y_start, x_end, y_end])
    return torch.tensor(rows, dtype=torch.float32)


def build_scroll_geometry_matrix(
    items: list[dict[str, Any]],
    *,
    reading_order_indices: list[int] | None = None,
    column_ids: list[int] | None = None,
    scroll_layout: ScrollLayout | None = None,
) -> Any:
    """Return local/page width, font-relative height, and global pseudo-y/index features."""

    import torch

    if scroll_layout is None:
        scroll_layout = build_scroll_layout(items, reading_order_indices=reading_order_indices, column_ids=column_ids)
    body_height = infer_document_body_font_size(items) or infer_document_baseline_font_size(items) or 1.0
    total_nodes = max(1, len(items) - 1)
    total_scroll_height = max(1.0, scroll_layout.total_scroll_height)
    rows = []
    for idx, _ in enumerate(items):
        box = scroll_layout.boxes[idx] if idx < len(scroll_layout.boxes) else None
        rank = scroll_layout.ranks[idx] if idx < len(scroll_layout.ranks) else idx
        norm_index = rank / total_nodes
        if box is None:
            rows.append([0.0, 0.0, 0.0, 0.0, norm_index])
            continue
        norm_width = box.width / max(1.0, box.column_width)
        norm_width_page = box.width / PAGE_SIZE
        norm_height = box.height / body_height
        norm_pseudo_y = clamp01(box.pseudo_y0 / total_scroll_height)
        rows.append([norm_width, norm_width_page, norm_height, norm_pseudo_y, norm_index])
    if not rows:
        return torch.empty((0, len(SCROLL_GEOMETRY_FIELDS)), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def build_derived_stats_matrix(items: list[dict[str, Any]], *, reading_order_ranks: list[int] | None = None) -> Any:
    import torch

    total_nodes = max(1, len(items) - 1)
    if reading_order_ranks is None:
        reading_order_ranks = list(range(len(items)))
    rows = []
    for idx, item in enumerate(items):
        chunks = list(iter_bbox_chunks(item.get("bbox")))
        rank = reading_order_ranks[idx] if idx < len(reading_order_ranks) else idx
        macro_position = rank / total_nodes
        total_width = sum(max(0.0, bbox[2] - bbox[0]) for bbox in chunks)
        total_height = sum(max(0.0, bbox[3] - bbox[1]) for bbox in chunks)
        area_sum = sum(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]) for bbox in chunks)
        aspect_ratio = total_height / max(total_width, 1.0)
        type_name = canonical_type(item)
        if type_name in NON_TEXT_DENSITY_TYPES:
            text_density = 0.0
        else:
            char_count = len(str(item.get("text_for_embedding") or ""))
            text_density = char_count / max(area_sum, 1.0)
        rows.append([macro_position, aspect_ratio, text_density])
    return torch.tensor(rows, dtype=torch.float32)


def build_style_stats_matrix(items: list[dict[str, Any]]) -> Any:
    import torch

    body_size = infer_document_body_font_size(items)
    rows = []
    for item in items:
        baseline = _item_font_size(item)
        baseline_norm = baseline / 100.0 if baseline > 0 else 0.0
        font_size_vs_body = (baseline - body_size) / body_size if baseline > 0 and body_size > 0 else 0.0
        rows.append(
            [
                baseline_norm,
                font_size_vs_body,
                _style_char_ratio(item, "is_bold"),
                _style_char_ratio(item, "is_italic"),
                _style_char_ratio(item, "is_inline_math"),
                _style_char_ratio(item, "is_inline_code"),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def build_sequence_position_matrix(items: list[dict[str, Any]], *, reading_order_ranks: list[int] | None = None) -> Any:
    """Return N x 16 sinusoidal encodings of the regime-state reading index."""

    import torch

    dim = len(SEQUENCE_POSITION_FIELDS)
    if not items:
        return torch.empty((0, dim), dtype=torch.float32)

    rows = []
    if reading_order_ranks is None:
        reading_order_ranks = regime_reading_order_ranks(items)
    for idx, _ in enumerate(items):
        order_idx = reading_order_ranks[idx] if idx < len(reading_order_ranks) else idx
        row = []
        for pair_idx in range(dim // 2):
            denominator = 10000.0 ** (2.0 * pair_idx / dim)
            angle = order_idx / denominator
            row.extend([math.sin(angle), math.cos(angle)])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def build_column_onehot_matrix(items: list[dict[str, Any]], *, column_ids: list[int] | None = None) -> Any:
    """Return N x 3 one-hot column IDs: left, right, or full/single-column."""

    import torch

    if column_ids is None:
        column_ids = infer_column_ids(items)
    rows = []
    for column_id in column_ids:
        normalized = column_id if column_id in {0, 1} else 2
        row = [0.0, 0.0, 0.0]
        row[normalized] = 1.0
        rows.append(row)
    if not rows:
        return torch.empty((0, len(COLUMN_FEATURE_FIELDS)), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def build_title_structure_matrix(items: list[dict[str, Any]]) -> Any:
    """Return relative font size plus heading-number regex probes."""

    import torch

    base_font_size = infer_document_body_font_size(items) or infer_document_baseline_font_size(items) or 1.0
    rows = []
    for item in items:
        font_size = _item_font_size(item)
        relative_font_size = font_size / base_font_size if font_size > 0 and base_font_size > 0 else 0.0
        is_h1, is_h2 = title_pattern_flags(str(item.get("text_for_embedding") or item.get("text") or ""))
        rows.append([relative_font_size, is_h1, is_h2])
    if not rows:
        return torch.empty((0, len(TITLE_STRUCTURE_FIELDS)), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def infer_document_body_font_size(items: list[dict[str, Any]]) -> float:
    weighted: dict[float, int] = {}
    for item in items:
        if canonical_type(item) != "text":
            continue
        size = _item_font_size(item)
        if size <= 0:
            continue
        weight = max(1, len(str(item.get("text_for_embedding") or "")))
        weighted[size] = weighted.get(size, 0) + weight
    if not weighted:
        return 0.0
    return max(weighted.items(), key=lambda item: item[1])[0]


def infer_document_baseline_font_size(items: list[dict[str, Any]]) -> float:
    weighted: dict[float, int] = {}
    for item in items:
        size = _item_font_size(item)
        if size <= 0:
            continue
        weight = max(1, len(str(item.get("text_for_embedding") or "")))
        weighted[size] = weighted.get(size, 0) + weight
    if not weighted:
        return 0.0
    return max(weighted.items(), key=lambda item: item[1])[0]


def infer_page_frames(items: list[dict[str, Any]]) -> dict[int, PageFrames]:
    """Infer left/right column coordinate frames from item bbox chunks."""

    by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for item in items:
        pages = item.get("source_page_idxs")
        chunks = list(iter_bbox_chunks(item.get("bbox")))
        if not isinstance(pages, list) or len(pages) != len(chunks):
            pages = [item.get("page_idx")] * len(chunks)
        for page, bbox in zip(pages, chunks):
            if not isinstance(page, int):
                continue
            if bbox[2] - bbox[0] >= 620.0:
                continue
            by_page.setdefault(page, []).append(bbox)

    frames = {}
    for page, boxes in by_page.items():
        frames[page] = infer_page_frame(boxes)
    return frames


def infer_column_ids(items: list[dict[str, Any]]) -> list[int]:
    """Infer coarse column IDs from each page's X-center clusters.

    IDs are 0 for left column, 1 for right column, and 2 for full-width or
    single-column blocks. This is intentionally coarse: the model needs a
    stable logical cue, not a brittle exact column detector.
    """

    page_frames = infer_page_frames(items)
    column_ids: list[int] = []
    for item in items:
        bbox = _first_bbox(item.get("bbox"))
        page_idx = _first_item_page(item)
        if bbox is None or page_idx is None:
            column_ids.append(2)
            continue

        frames = page_frames.get(page_idx)
        if bool(item.get("is_full_width")) or bbox[2] - bbox[0] >= FULL_WIDTH_THRESHOLD:
            column_ids.append(2)
            continue
        if frames is None or frames.left is None or frames.right is None:
            column_ids.append(2)
            continue
        if _bbox_spans_column_gutter(bbox, frames):
            column_ids.append(2)
            continue

        center = (bbox[0] + bbox[2]) / 2.0
        left_center = (frames.left.x_min + frames.left.x_max) / 2.0
        right_center = (frames.right.x_min + frames.right.x_max) / 2.0
        column_ids.append(0 if abs(center - left_center) <= abs(center - right_center) else 1)
    return column_ids


def infer_page_frame(boxes: list[tuple[float, float, float, float]]) -> PageFrames:
    centers = sorted(((box[0] + box[2]) / 2.0 for box in boxes))
    if len(centers) < 4:
        return PageFrames(left=None, right=None)

    best_gap = 0.0
    best_index = -1
    for idx, (left, right) in enumerate(zip(centers, centers[1:])):
        if idx + 1 < 2 or len(centers) - idx - 1 < 2:
            continue
        gap = right - left
        if gap > best_gap:
            best_gap = gap
            best_index = idx
    if best_gap < 130.0 or best_index < 0:
        return PageFrames(left=None, right=None)

    split = (centers[best_index] + centers[best_index + 1]) / 2.0
    left_boxes = [box for box in boxes if (box[0] + box[2]) / 2.0 < split]
    right_boxes = [box for box in boxes if (box[0] + box[2]) / 2.0 >= split]
    return PageFrames(left=frame_from_boxes(left_boxes), right=frame_from_boxes(right_boxes))


def frame_from_boxes(boxes: list[tuple[float, float, float, float]]) -> ColumnFrame | None:
    if not boxes:
        return None
    return ColumnFrame(x_min=min(box[0] for box in boxes), x_max=max(box[2] for box in boxes))


def normalize_x_in_local_frame(
    x: float,
    bbox: tuple[float, float, float, float],
    page_idx: int,
    page_frames: dict[int, PageFrames],
    is_full_width: bool,
) -> float:
    if is_full_width:
        return x / PAGE_SIZE
    frames = page_frames.get(page_idx)
    if frames is None or frames.left is None or frames.right is None:
        return x / PAGE_SIZE
    center = (bbox[0] + bbox[2]) / 2.0
    left_center = (frames.left.x_min + frames.left.x_max) / 2.0
    right_center = (frames.right.x_min + frames.right.x_max) / 2.0
    frame = frames.left if abs(center - left_center) <= abs(center - right_center) else frames.right
    return frame.normalize_x(x)


def _bbox_spans_column_gutter(bbox: tuple[float, float, float, float], frames: PageFrames) -> bool:
    if frames.left is None or frames.right is None:
        return False
    return bbox[0] <= frames.left.x_max and bbox[2] >= frames.right.x_min


def _ranks_from_order(order: list[int], node_count: int) -> list[int]:
    ranks = list(range(node_count))
    for rank, node_idx in enumerate(_valid_reading_order_indices(order, node_count)):
        ranks[node_idx] = rank
    return ranks


def _valid_reading_order_indices(order: list[int] | None, node_count: int) -> list[int]:
    if order is None or len(order) != node_count or sorted(order) != list(range(node_count)):
        return list(range(node_count))
    return list(order)


def build_sequential_edge_index(node_count: int, *, bidirectional: bool = True) -> Any:
    import torch

    edges = build_sequential_edge_pairs(node_count, bidirectional=bidirectional)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_edge_index_from_pairs(edge_pairs: list[tuple[int, int, str]]) -> Any:
    import torch

    if not edge_pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([(source, target) for source, target, _ in edge_pairs], dtype=torch.long).t().contiguous()


def build_node_feature_schema() -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    start = 0

    def add(name: str, dim: int, **metadata: Any) -> None:
        nonlocal start
        schema[name] = {"start": start, "end": start + dim, "dim": dim, **metadata}
        start += dim

    add("semantic", SCIBERT_DIM, source="SciBERT CLS window mean")
    add("type_onehot", len(TYPE_VOCAB), vocab=TYPE_VOCAB)
    add("geometry", len(GEOMETRY_FIELDS), fields=GEOMETRY_FIELDS)
    add(
        "scroll_geometry",
        len(SCROLL_GEOMETRY_FIELDS),
        fields=SCROLL_GEOMETRY_FIELDS,
        source="pseudo-y long-scroll projection plus local column/font normalization",
    )
    add("derived_stats", len(DERIVED_STAT_FIELDS), fields=DERIVED_STAT_FIELDS)
    add("style_stats", len(STYLE_STAT_FIELDS), fields=STYLE_STAT_FIELDS)
    add("sequence_position", len(SEQUENCE_POSITION_FIELDS), fields=SEQUENCE_POSITION_FIELDS, source="single/double-column regime reading index")
    add("column_features", len(COLUMN_FEATURE_FIELDS), fields=COLUMN_FEATURE_FIELDS, source="page x-center column clustering")
    add("title_structure", len(TITLE_STRUCTURE_FIELDS), fields=TITLE_STRUCTURE_FIELDS, source="font-size ratio and heading regex probes")
    return schema


def build_sequential_edge_pairs(node_count: int, *, bidirectional: bool = True) -> list[tuple[int, int]]:
    edges = []
    for idx in range(max(0, node_count - 1)):
        edges.append((idx, idx + 1))
        if bidirectional:
            edges.append((idx + 1, idx))
    return edges


def build_candidate_edge_pairs(
    items: list[dict[str, Any]],
    *,
    sequential_window: int = 3,
    spatial_k: int = 3,
    bidirectional: bool = True,
    reading_order_indices: list[int] | None = None,
    column_ids: list[int] | None = None,
    long_sight_window: int = 40,
    scope_anchor_window: int = 80,
    float_skip_window: int = 40,
) -> list[tuple[int, int, str]]:
    """Build candidate edges with high positive-edge recall.

    The base graph is still dual-view: reading-order windows plus spatial
    sight lines. Extra long-range edges cover two known blind spots:
    section/list scope edges and text continuations separated by large floats.
    These rules are PDF-only and must run for both training and inference; they
    do not inspect TeX truth labels.
    """

    edge_pairs: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    def add_edge(source_idx: int, target_idx: int, source_type: str) -> None:
        if source_idx == target_idx:
            return
        key = (source_idx, target_idx)
        if key in seen:
            return
        seen.add(key)
        edge_pairs.append((source_idx, target_idx, source_type))

    node_count = len(items)
    order = sort_node_indices_by_reading_order(items) if reading_order_indices is None else _valid_reading_order_indices(reading_order_indices, node_count)
    ranks = _ranks_from_order(order, node_count)
    if column_ids is None:
        column_ids = infer_column_ids(items)
    for pos in range(max(0, node_count - 1)):
        source_idx = order[pos]
        target_idx = order[pos + 1]
        add_edge(source_idx, target_idx, "sequential_forced")
        if bidirectional:
            add_edge(target_idx, source_idx, "sequential_forced")

    window = max(0, int(sequential_window))
    for source_pos, source_idx in enumerate(order):
        start = max(0, source_pos - window) if bidirectional else source_pos + 1
        end = min(node_count, source_pos + window + 1)
        for target_pos in range(start, end):
            add_edge(source_idx, order[target_pos], "sequential")

    if spatial_k > 0:
        centers = [_node_center(item) for item in items]
        pages = [_first_item_page(item) for item in items]
        for source_idx, (source_center, source_page) in enumerate(zip(centers, pages)):
            if source_center is None or source_page is None:
                continue
            down_candidates = []
            right_candidates = []
            for target_idx, (target_center, target_page) in enumerate(zip(centers, pages)):
                if target_idx == source_idx or target_center is None or target_page != source_page:
                    continue
                distance = _center_distance(source_center, target_center)
                if target_center[1] > source_center[1]:
                    down_candidates.append((distance, target_idx))
                if target_center[0] > source_center[0]:
                    right_candidates.append((distance, target_idx))
            for _, target_idx in sorted(down_candidates)[:spatial_k]:
                add_edge(source_idx, target_idx, "spatial_down")
            for _, target_idx in sorted(right_candidates)[:spatial_k]:
                add_edge(source_idx, target_idx, "spatial_right")

    add_same_column_long_sight_edges(
        items,
        order=order,
        column_ids=column_ids,
        add_edge=add_edge,
        max_window=long_sight_window,
    )
    add_float_skip_edges(
        items,
        order=order,
        add_edge=add_edge,
        max_window=float_skip_window,
        bidirectional=bidirectional,
    )
    add_scope_anchor_edges(
        items,
        order=order,
        ranks=ranks,
        add_edge=add_edge,
        max_window=scope_anchor_window,
    )

    return edge_pairs


def add_same_column_long_sight_edges(
    items: list[dict[str, Any]],
    *,
    order: list[int],
    column_ids: list[int],
    add_edge: Any,
    max_window: int,
) -> None:
    """Add sparse long sight-line edges inside the same logical column."""

    if max_window <= 0:
        return
    for source_pos, source_idx in enumerate(order):
        source = items[source_idx]
        source_bbox = _last_bbox(source.get("bbox"))
        source_page = _last_item_page(source)
        source_column = column_ids[source_idx] if source_idx < len(column_ids) else 2
        if source_bbox is None or source_page is None or not _is_flow_candidate(source):
            continue
        found = 0
        for target_pos in range(source_pos + 1, min(len(order), source_pos + max_window + 1)):
            target_idx = order[target_pos]
            target = items[target_idx]
            if _is_heading_item(target):
                break
            target_bbox = _first_bbox(target.get("bbox"))
            target_page = _first_item_page(target)
            target_column = column_ids[target_idx] if target_idx < len(column_ids) else 2
            if target_bbox is None or target_page != source_page or target_column != source_column:
                continue
            if not _is_flow_candidate(target):
                continue
            if target_bbox[1] <= source_bbox[3]:
                continue
            add_edge(source_idx, target_idx, "same_column_long_sight")
            found += 1
            if found >= 2:
                break


def add_float_skip_edges(
    items: list[dict[str, Any]],
    *,
    order: list[int],
    add_edge: Any,
    max_window: int,
    bidirectional: bool,
) -> None:
    """Connect text before and after large non-text barriers."""

    if max_window <= 0:
        return
    for source_pos, source_idx in enumerate(order):
        source = items[source_idx]
        if not _is_text_continuation_source(source):
            continue
        skipped_structural = False
        skipped_nodes = 0
        for target_pos in range(source_pos + 1, min(len(order), source_pos + max_window + 1)):
            target_idx = order[target_pos]
            target = items[target_idx]
            if _is_heading_item(target):
                break
            if _is_structural_skip_item(target):
                skipped_structural = True
                skipped_nodes += 1
                continue
            if not _is_text_continuation_target(target):
                skipped_nodes += 1
                continue
            if skipped_structural or skipped_nodes >= 4:
                add_edge(source_idx, target_idx, "float_skip")
                if bidirectional:
                    add_edge(target_idx, source_idx, "float_skip")
            break


def add_scope_anchor_edges(
    items: list[dict[str, Any]],
    *,
    order: list[int],
    ranks: list[int],
    add_edge: Any,
    max_window: int,
) -> None:
    """Add heading/reference/list anchors to their local logical scope."""

    if max_window <= 0:
        return
    for source_pos, source_idx in enumerate(order):
        source = items[source_idx]
        if _is_reference_heading(source):
            for target_idx in _iter_scope_targets(items, order, source_pos, max_window=max_window, reference_only=True):
                add_edge(source_idx, target_idx, "scope_anchor")
            continue
        if _is_heading_item(source):
            for target_idx in _iter_scope_targets(items, order, source_pos, max_window=max_window):
                add_edge(source_idx, target_idx, "scope_anchor")
            continue
        if _is_list_item_like(source):
            for target_idx in _iter_list_run_targets(items, order, source_pos, max_window=max_window):
                add_edge(source_idx, target_idx, "list_run_scope")


def _iter_scope_targets(
    items: list[dict[str, Any]],
    order: list[int],
    source_pos: int,
    *,
    max_window: int,
    reference_only: bool = False,
) -> list[int]:
    targets: list[int] = []
    for target_pos in range(source_pos + 1, min(len(order), source_pos + max_window + 1)):
        target_idx = order[target_pos]
        target = items[target_idx]
        if _is_heading_item(target):
            break
        if _is_auxiliary_item(target):
            continue
        if reference_only and canonical_type(target) != "reference":
            continue
        if not reference_only and not _is_scope_child_candidate(target):
            continue
        targets.append(target_idx)
    return targets


def _iter_list_run_targets(
    items: list[dict[str, Any]],
    order: list[int],
    source_pos: int,
    *,
    max_window: int,
) -> list[int]:
    targets: list[int] = []
    saw_next_marker = False
    for target_pos in range(source_pos + 1, min(len(order), source_pos + max_window + 1)):
        target_idx = order[target_pos]
        target = items[target_idx]
        if _is_heading_item(target):
            break
        if _is_list_item_like(target):
            saw_next_marker = True
            targets.append(target_idx)
            continue
        if saw_next_marker and canonical_type(target) == "equation":
            targets.append(target_idx)
            continue
        if saw_next_marker and _is_text_continuation_target(target) and not _is_list_item_like(target):
            break
    return targets


def build_edge_attr_matrix(
    items: list[dict[str, Any]],
    semantic: Any,
    *,
    edge_pairs: list[tuple[int, int, str]] | None = None,
    reading_order_ranks: list[int] | None = None,
    scroll_layout: ScrollLayout | None = None,
) -> Any:
    """Return 15-dimensional edge features aligned with candidate edge_index."""

    import torch
    import torch.nn.functional as F

    if edge_pairs is None:
        edge_pairs = build_candidate_edge_pairs(items)
    if reading_order_ranks is None:
        reading_order_ranks = regime_reading_order_ranks(items)
    if scroll_layout is None:
        scroll_layout = build_scroll_layout(items, reading_order_indices=None)

    rows = []
    for source_idx, target_idx, source_type in edge_pairs:
        source = items[source_idx]
        target = items[target_idx]
        semantic_cosine = float(F.cosine_similarity(semantic[source_idx], semantic[target_idx], dim=0).item())
        source_bbox = _last_bbox(source.get("bbox"))
        target_bbox = _first_bbox(target.get("bbox"))
        source_bbox = source_bbox or (0.0, 0.0, 0.0, 0.0)
        target_bbox = target_bbox or (0.0, 0.0, 0.0, 0.0)
        source_scroll = scroll_layout.boxes[source_idx] if source_idx < len(scroll_layout.boxes) else None
        target_scroll = scroll_layout.boxes[target_idx] if target_idx < len(scroll_layout.boxes) else None
        delta_y_gap = scroll_delta_y_gap(source_scroll, target_scroll, fallback_source=source_bbox, fallback_target=target_bbox)
        delta_x_left = scroll_delta_x_left(source_scroll, target_scroll, fallback_source=source_bbox, fallback_target=target_bbox)
        source_center = _bbox_center(source_bbox)
        target_center = _bbox_center(target_bbox)
        center_distance = scroll_center_distance(
            source_scroll,
            target_scroll,
            fallback_source=source_center,
            fallback_target=target_center,
        )
        source_height = max(1.0, source_bbox[3] - source_bbox[1])
        target_height = max(1.0, target_bbox[3] - target_bbox[1])
        y_overlap_ratio = bbox_y_overlap_ratio(source_bbox, target_bbox)
        has_x_gutter = float(
            y_overlap_ratio > 0.3 and bbox_x_gap(source_bbox, target_bbox) > 0.03 * PAGE_SIZE
        )

        source_rank = reading_order_ranks[source_idx] if source_idx < len(reading_order_ranks) else source_idx
        target_rank = reading_order_ranks[target_idx] if target_idx < len(reading_order_ranks) else target_idx
        index_delta = float(target_rank - source_rank)
        index_bins = index_delta_bins(index_delta)
        rows.append(
            [
                semantic_cosine,
                delta_y_gap,
                delta_x_left,
                float(abs(delta_x_left) < 0.01),
                center_distance / PAGE_SIZE,
                _item_font_size(target) - _item_font_size(source),
                float(_item_is_bold(source) and not _item_is_bold(target)),
                target_height / source_height,
                y_overlap_ratio,
                has_x_gutter,
                *index_bins,
            ]
        )
    if not rows:
        return torch.empty((0, len(EDGE_ATTR_FIELDS)), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def make_node_records(
    items: list[dict[str, Any]],
    *,
    column_ids: list[int] | None = None,
    reading_order_ranks: list[int] | None = None,
    scroll_layout: ScrollLayout | None = None,
) -> list[dict[str, Any]]:
    if column_ids is None:
        column_ids = infer_column_ids(items)
    if reading_order_ranks is None:
        reading_order_ranks = regime_reading_order_ranks(items)
    if scroll_layout is None:
        scroll_layout = build_scroll_layout(items, column_ids=column_ids)
    records = []
    for idx, item in enumerate(items):
        scroll_box = scroll_layout.boxes[idx] if idx < len(scroll_layout.boxes) else None
        records.append(
            {
                "global_order": item.get("global_order"),
                "type": item.get("type"),
                "raw_type": item.get("raw_type"),
                "list_type": item.get("list_type"),
                "canonical_type": canonical_type(item),
                "column_id_inferred": column_ids[idx] if idx < len(column_ids) else 2,
                "regime_reading_order": reading_order_ranks[idx] if idx < len(reading_order_ranks) else idx,
                "dag_reading_order": reading_order_ranks[idx] if idx < len(reading_order_ranks) else idx,
                "xycut_reading_order": reading_order_ranks[idx] if idx < len(reading_order_ranks) else idx,
                "pseudo_y0": scroll_box.pseudo_y0 if scroll_box is not None else None,
                "pseudo_y1": scroll_box.pseudo_y1 if scroll_box is not None else None,
                "scroll_total_height": scroll_layout.total_scroll_height,
                "page_idx": item.get("page_idx"),
                "visual_order": item.get("visual_order"),
                "merge_count": item.get("merge_count"),
                "source_page_idxs": item.get("source_page_idxs"),
                "source_visual_orders": item.get("source_visual_orders"),
                "bbox": item.get("bbox"),
                "text_preview": str(item.get("text_for_embedding") or "")[:200],
                "reference_items": item.get("reference_items"),
            }
        )
    return records


def build_logical_center_pairs(
    items: list[dict[str, Any]],
    *,
    full_span_ratio: float = 0.60,
) -> list[tuple[tuple[float, float] | None, tuple[float, float] | None]]:
    """Return first/last centers in a reading-flow coordinate system.

    Double-column regions are "unrolled" into a single logical vertical axis:
    the left column occupies the first half of that region and the right column
    occupies the second half. This makes left-column bottoms and right-column
    tops close in relation space without erasing local indentation within each
    column.
    """

    chunk_boxes = collect_logical_boxes(items)
    if not chunk_boxes:
        return [(None, None) for _ in items]

    local_centers: dict[tuple[int, int], tuple[float, float]] = {}
    page_heights: dict[int, float] = {}
    for page_idx in sorted({box.page_idx for box in chunk_boxes}):
        page_boxes = [box for box in chunk_boxes if box.page_idx == page_idx]
        centers, logical_height = build_page_logical_centers(page_boxes, full_span_ratio=full_span_ratio)
        local_centers.update(centers)
        page_heights[page_idx] = logical_height

    page_offsets: dict[int, float] = {}
    cursor = 0.0
    for page_idx in sorted(page_heights):
        page_offsets[page_idx] = cursor
        cursor += max(PAGE_SIZE, page_heights[page_idx])

    centers_by_chunk: dict[tuple[int, int], tuple[float, float]] = {}
    page_by_chunk = {(box.item_index, box.chunk_index): box.page_idx for box in chunk_boxes}
    for key, center in local_centers.items():
        page_idx = page_by_chunk.get(key, 0)
        centers_by_chunk[key] = (center[0], center[1] + page_offsets.get(page_idx, 0.0))

    pairs: list[tuple[tuple[float, float] | None, tuple[float, float] | None]] = []
    for item_index, item in enumerate(items):
        chunk_count = len(iter_bbox_chunks(item.get("bbox")))
        if chunk_count == 0:
            pairs.append((None, None))
            continue
        first = centers_by_chunk.get((item_index, 0))
        last = centers_by_chunk.get((item_index, chunk_count - 1))
        pairs.append((first, last or first))
    return pairs


def build_scroll_layout(
    items: list[dict[str, Any]],
    *,
    reading_order_indices: list[int] | None = None,
    column_ids: list[int] | None = None,
) -> ScrollLayout:
    """Project page-local bboxes onto a one-dimensional long-scroll axis."""

    node_count = len(items)
    if column_ids is None:
        column_ids = infer_column_ids(items)
    order = _valid_reading_order_indices(
        sort_node_indices_by_reading_order(items) if reading_order_indices is None else reading_order_indices,
        node_count,
    )
    ranks = _ranks_from_order(order, node_count)
    page_frames = infer_page_frames(items)
    page_widths = infer_page_content_widths(items)
    boxes: list[ScrollBox | None] = [None] * node_count

    current_page: int | None = None
    page_base = 0.0
    page_column_offset = 0.0
    page_max_local_y = 0.0
    max_pseudo_y = 0.0
    last_half_column: int | None = None
    previous_bbox: tuple[float, float, float, float] | None = None

    for item_idx in order:
        item = items[item_idx]
        bbox = _first_bbox(item.get("bbox"))
        page_idx = _first_item_page(item)
        if bbox is None or page_idx is None:
            continue

        if current_page is None or page_idx != current_page:
            if current_page is not None:
                page_base = max(page_base, max_pseudo_y)
            current_page = page_idx
            page_column_offset = 0.0
            page_max_local_y = 0.0
            last_half_column = None
            previous_bbox = None

        column_id = column_ids[item_idx] if item_idx < len(column_ids) else 2
        explicit_column = _explicit_column_label(item)
        if explicit_column is not None:
            column_id = explicit_column
        if _is_scroll_column_wrap(bbox, previous_bbox, column_id=column_id, previous_column_id=last_half_column):
            page_column_offset = max(page_column_offset, page_max_local_y)

        column_width = column_width_for_item(item, bbox, page_idx, column_id, page_frames, page_widths)
        local_x0, local_x1 = local_x_span_for_item(item, bbox, page_idx, column_id, page_frames, page_widths)
        pseudo_y0 = page_base + page_column_offset + bbox[1]
        pseudo_y1 = page_base + page_column_offset + bbox[3]
        boxes[item_idx] = ScrollBox(
            item_index=item_idx,
            page_idx=page_idx,
            bbox=bbox,
            column_id=column_id,
            column_width=column_width,
            local_x0=local_x0,
            local_x1=local_x1,
            pseudo_y0=pseudo_y0,
            pseudo_y1=pseudo_y1,
        )

        page_max_local_y = max(page_max_local_y, bbox[3])
        max_pseudo_y = max(max_pseudo_y, pseudo_y1)
        if column_id in {0, 1}:
            last_half_column = column_id
        previous_bbox = bbox

    return ScrollLayout(boxes=boxes, ranks=ranks, total_scroll_height=max(1.0, max_pseudo_y))


def _is_scroll_column_wrap(
    bbox: tuple[float, float, float, float],
    previous_bbox: tuple[float, float, float, float] | None,
    *,
    column_id: int,
    previous_column_id: int | None,
    x_shift_threshold: float = 0.20 * PAGE_SIZE,
) -> bool:
    if previous_bbox is None:
        return False
    if column_id not in {0, 1} or previous_column_id not in {0, 1}:
        return False
    if column_id == previous_column_id:
        return False
    if column_id == 1 and previous_column_id == 0:
        return True
    previous_center = (previous_bbox[0] + previous_bbox[2]) / 2.0
    current_center = (bbox[0] + bbox[2]) / 2.0
    return bbox[1] < previous_bbox[1] and abs(current_center - previous_center) >= x_shift_threshold


def infer_page_content_widths(items: list[dict[str, Any]]) -> dict[int, float]:
    by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for item in items:
        bbox = _first_bbox(item.get("bbox"))
        page_idx = _first_item_page(item)
        if bbox is None or page_idx is None:
            continue
        by_page.setdefault(page_idx, []).append(bbox)
    widths = {}
    for page_idx, boxes in by_page.items():
        widths[page_idx] = max(1.0, max(bbox[2] for bbox in boxes) - min(bbox[0] for bbox in boxes))
    return widths


def column_width_for_item(
    item: dict[str, Any],
    bbox: tuple[float, float, float, float],
    page_idx: int,
    column_id: int,
    page_frames: dict[int, PageFrames],
    page_widths: dict[int, float],
) -> float:
    frame = column_frame_for_item(item, bbox, page_idx, column_id, page_frames)
    if frame is not None:
        return frame.width
    return max(1.0, page_widths.get(page_idx, PAGE_SIZE))


def local_x_span_for_item(
    item: dict[str, Any],
    bbox: tuple[float, float, float, float],
    page_idx: int,
    column_id: int,
    page_frames: dict[int, PageFrames],
    page_widths: dict[int, float],
) -> tuple[float, float]:
    frame = column_frame_for_item(item, bbox, page_idx, column_id, page_frames)
    if frame is None:
        return bbox[0], bbox[2]
    return frame.normalize_x(bbox[0]) * PAGE_SIZE, frame.normalize_x(bbox[2]) * PAGE_SIZE


def column_frame_for_item(
    item: dict[str, Any],
    bbox: tuple[float, float, float, float],
    page_idx: int,
    column_id: int,
    page_frames: dict[int, PageFrames],
) -> ColumnFrame | None:
    if column_id == 2 or bool(item.get("is_full_width")):
        return None
    frames = page_frames.get(page_idx)
    if frames is None:
        return None
    if column_id == 0:
        return frames.left
    if column_id == 1:
        return frames.right
    return None


def _explicit_column_label(item: dict[str, Any]) -> int | None:
    label = item.get("column_fix_column")
    if label == "LEFT_COL":
        return 0
    if label == "RIGHT_COL":
        return 1
    if item.get("column_fix_span") == "FULL_SPAN":
        return 2
    return None


def scroll_delta_y_gap(
    source: ScrollBox | None,
    target: ScrollBox | None,
    *,
    fallback_source: tuple[float, float, float, float],
    fallback_target: tuple[float, float, float, float],
) -> float:
    if source is None or target is None:
        return (fallback_target[1] - fallback_source[3]) / PAGE_SIZE
    return (target.pseudo_y0 - source.pseudo_y1) / PAGE_SIZE


def scroll_delta_x_left(
    source: ScrollBox | None,
    target: ScrollBox | None,
    *,
    fallback_source: tuple[float, float, float, float],
    fallback_target: tuple[float, float, float, float],
) -> float:
    if source is None or target is None:
        return (fallback_target[0] - fallback_source[0]) / PAGE_SIZE
    return (target.local_x0 - source.local_x0) / PAGE_SIZE


def scroll_center_distance(
    source: ScrollBox | None,
    target: ScrollBox | None,
    *,
    fallback_source: tuple[float, float],
    fallback_target: tuple[float, float],
) -> float:
    if source is None or target is None:
        return _center_distance(fallback_source, fallback_target)
    return _center_distance((source.local_cx, source.pseudo_cy), (target.local_cx, target.pseudo_cy))


def bbox_y_overlap_ratio(
    source: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    *,
    eps: float = 1e-6,
) -> float:
    intersection = max(0.0, min(source[3], target[3]) - max(source[1], target[1]))
    min_height = min(max(0.0, source[3] - source[1]), max(0.0, target[3] - target[1]))
    return intersection / (min_height + eps)


def bbox_x_gap(source: tuple[float, float, float, float], target: tuple[float, float, float, float]) -> float:
    return max(source[0], target[0]) - min(source[2], target[2])


def index_delta_bins(index_delta: float) -> list[float]:
    if index_delta <= 0:
        return [0.0, 0.0, 0.0, 0.0, 1.0]
    if index_delta == 1:
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    if index_delta == 2:
        return [0.0, 1.0, 0.0, 0.0, 0.0]
    if 3 <= index_delta <= 5:
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0, 0.0]


def collect_logical_boxes(items: list[dict[str, Any]]) -> list[LogicalBox]:
    boxes: list[LogicalBox] = []
    for item_index, item in enumerate(items):
        chunks = iter_bbox_chunks(item.get("bbox"))
        if not chunks:
            continue
        pages = item.get("source_page_idxs")
        if not isinstance(pages, list) or len(pages) != len(chunks):
            pages = [item.get("page_idx")] * len(chunks)
        for chunk_index, (bbox, page) in enumerate(zip(chunks, pages)):
            if not isinstance(page, int):
                continue
            boxes.append(
                LogicalBox(
                    item_index=item_index,
                    chunk_index=chunk_index,
                    page_idx=page,
                    bbox=bbox,
                    full_span=bool(item.get("is_full_width")) or (bbox[2] - bbox[0]) >= FULL_WIDTH_THRESHOLD,
                )
            )
    return boxes


def build_page_logical_centers(
    boxes: list[LogicalBox],
    *,
    full_span_ratio: float,
) -> tuple[dict[tuple[int, int], tuple[float, float]], float]:
    if not boxes:
        return {}, PAGE_SIZE

    min_x = min(box.x0 for box in boxes)
    max_x = max(box.x1 for box in boxes)
    page_width = max(1.0, max_x - min_x)
    center_x = min_x + page_width / 2.0
    mode_blocks: list[tuple[str, list[LogicalBox]]] = []
    current_mode: str | None = None
    current_block: list[LogicalBox] = []
    for box in sorted(boxes, key=lambda item: (item.y0, item.x0, item.item_index, item.chunk_index)):
        mode = "SINGLE" if box.full_span or (box.x1 - box.x0) > full_span_ratio * page_width else "DOUBLE"
        if current_mode is not None and mode != current_mode and current_block:
            mode_blocks.append((current_mode, current_block))
            current_block = []
        current_mode = mode
        current_block.append(box)
    if current_mode is not None and current_block:
        mode_blocks.append((current_mode, current_block))

    centers: dict[tuple[int, int], tuple[float, float]] = {}
    logical_cursor = 0.0
    previous_y1: float | None = None
    for mode, block in mode_blocks:
        block_y0 = min(box.y0 for box in block)
        block_y1 = max(box.y1 for box in block)
        if previous_y1 is not None and block_y0 > previous_y1:
            logical_cursor += block_y0 - previous_y1
        block_height = max(1.0, block_y1 - block_y0)
        if mode == "DOUBLE":
            write_double_column_logical_centers(
                block,
                centers,
                center_x=center_x,
                page_width=page_width,
                block_y0=block_y0,
                block_height=block_height,
                logical_cursor=logical_cursor,
            )
        else:
            for box in block:
                y_rel = clamp01((box.cy - block_y0) / block_height)
                x_rel = clamp01((box.cx - min_x) / page_width)
                centers[(box.item_index, box.chunk_index)] = (x_rel * PAGE_SIZE, logical_cursor + y_rel * block_height)
        logical_cursor += block_height
        previous_y1 = block_y1

    return centers, max(PAGE_SIZE, logical_cursor)


def write_double_column_logical_centers(
    block: list[LogicalBox],
    centers: dict[tuple[int, int], tuple[float, float]],
    *,
    center_x: float,
    page_width: float,
    block_y0: float,
    block_height: float,
    logical_cursor: float,
) -> None:
    left = [box for box in block if box.cx <= center_x]
    right = [box for box in block if box.cx > center_x]
    frames = {
        0: frame_from_logical_boxes(left) or frame_from_logical_boxes(block),
        1: frame_from_logical_boxes(right) or frame_from_logical_boxes(block),
    }
    column_count = 2 if left and right else 1
    for box in block:
        column = 0 if column_count == 1 or box.cx <= center_x else 1
        frame = frames[column] or frame_from_logical_boxes(block)
        y_rel = clamp01((box.cy - block_y0) / block_height)
        logical_progress = (column + y_rel) / column_count
        x_rel = clamp01((box.cx - frame.x_min) / frame.width) if frame is not None else 0.5
        centers[(box.item_index, box.chunk_index)] = (x_rel * PAGE_SIZE, logical_cursor + logical_progress * block_height)


def frame_from_logical_boxes(boxes: list[LogicalBox]) -> ColumnFrame | None:
    if not boxes:
        return None
    return ColumnFrame(x_min=min(box.x0 for box in boxes), x_max=max(box.x1 for box in boxes))


def logical_center_distance(
    source: tuple[float, float] | None,
    target: tuple[float, float] | None,
    *,
    fallback_source: tuple[float, float],
    fallback_target: tuple[float, float],
) -> float:
    if source is None or target is None:
        return _center_distance(fallback_source, fallback_target)
    return _center_distance(source, target)


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def iter_bbox_chunks(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, list) or len(value) < 4:
        return []
    chunks = []
    usable_len = len(value) - (len(value) % 4)
    for idx in range(0, usable_len, 4):
        chunk = value[idx : idx + 4]
        chunks.append((float(chunk[0]), float(chunk[1]), float(chunk[2]), float(chunk[3])))
    return chunks


def _first_bbox(value: Any) -> tuple[float, float, float, float] | None:
    chunks = iter_bbox_chunks(value)
    return chunks[0] if chunks else None


def _last_bbox(value: Any) -> tuple[float, float, float, float] | None:
    chunks = iter_bbox_chunks(value)
    return chunks[-1] if chunks else None


def _node_center(item: dict[str, Any]) -> tuple[float, float] | None:
    bbox = _first_bbox(item.get("bbox"))
    if bbox is None:
        return None
    return _bbox_center(bbox)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _center_distance(source: tuple[float, float], target: tuple[float, float]) -> float:
    return math.sqrt((target[0] - source[0]) ** 2 + (target[1] - source[1]) ** 2)


def _first_item_page(item: dict[str, Any]) -> int | None:
    pages = item.get("source_page_idxs")
    if isinstance(pages, list) and pages and isinstance(pages[0], int):
        return pages[0]
    page = item.get("page_idx")
    return page if isinstance(page, int) else None


def _last_item_page(item: dict[str, Any]) -> int | None:
    pages = item.get("source_page_idxs")
    if isinstance(pages, list) and pages and isinstance(pages[-1], int):
        return pages[-1]
    page = item.get("page_idx")
    return page if isinstance(page, int) else None


def _item_font_size(item: dict[str, Any]) -> float:
    value = item.get("style_baseline_size")
    if isinstance(value, (int, float)):
        return float(value)
    spans = item.get("style_spans")
    if not isinstance(spans, list):
        return 0.0
    weighted: dict[float, int] = {}
    for span in spans:
        if not isinstance(span, dict):
            continue
        size = span.get("font_size")
        if not isinstance(size, (int, float)):
            continue
        weight = int(span.get("char_count") or len(str(span.get("text") or "")) or 1)
        weighted[float(size)] = weighted.get(float(size), 0) + max(1, weight)
    if not weighted:
        return 0.0
    return max(weighted.items(), key=lambda item: item[1])[0]


def _item_is_bold(item: dict[str, Any]) -> bool:
    spans = item.get("style_spans")
    if not isinstance(spans, list):
        return False
    bold_chars = 0
    total_chars = 0
    for span in spans:
        if not isinstance(span, dict):
            continue
        count = int(span.get("char_count") or len(str(span.get("text") or "")) or 1)
        total_chars += count
        if span.get("is_bold"):
            bold_chars += count
    return total_chars > 0 and bold_chars / total_chars >= 0.5


def _is_auxiliary_item(item: dict[str, Any]) -> bool:
    raw_type = str(item.get("type") or item.get("raw_type") or "").lower()
    canonical = canonical_type(item)
    return raw_type in AUXILIARY_TYPES or canonical in AUXILIARY_TYPES


def _is_heading_item(item: dict[str, Any]) -> bool:
    return canonical_type(item) == "title"


def _is_reference_heading(item: dict[str, Any]) -> bool:
    if not _is_heading_item(item):
        return False
    text = _item_text(item).strip().lower()
    normalized = re.sub(r"[^a-z]", "", text)
    return normalized in {"references", "bibliography"}


def _is_scope_child_candidate(item: dict[str, Any]) -> bool:
    if _is_auxiliary_item(item):
        return False
    return canonical_type(item) in {"text", "list", "equation", "figure", "table", "algorithm", "code", "reference"}


def _is_flow_candidate(item: dict[str, Any]) -> bool:
    return canonical_type(item) in {"text", "list", "equation", "reference"}


def _is_text_continuation_source(item: dict[str, Any]) -> bool:
    if canonical_type(item) not in TEXT_FLOW_TYPES:
        return False
    text = _item_text(item).strip()
    return bool(text)


def _is_text_continuation_target(item: dict[str, Any]) -> bool:
    return canonical_type(item) in TEXT_FLOW_TYPES and bool(_item_text(item).strip())


def _is_structural_skip_item(item: dict[str, Any]) -> bool:
    return canonical_type(item) in STRUCTURAL_SKIP_TYPES


def _is_list_item_like(item: dict[str, Any]) -> bool:
    if item.get("list_marker"):
        return True
    if canonical_type(item) == "list":
        return True
    return bool(LIST_MARKER_RE.match(_item_text(item)))


def _item_text(item: dict[str, Any]) -> str:
    return str(item.get("text_for_embedding") or item.get("text") or item.get("content") or "")


def _style_char_ratio(item: dict[str, Any], flag: str) -> float:
    spans = item.get("style_spans")
    if not isinstance(spans, list):
        return 0.0
    flagged = 0
    total = 0
    for span in spans:
        if not isinstance(span, dict):
            continue
        count = int(span.get("char_count") or len(str(span.get("text") or "")) or 1)
        total += count
        if span.get(flag):
            flagged += count
    return flagged / total if total > 0 else 0.0


def canonical_type(value: Any) -> str:
    list_type = ""
    if isinstance(value, dict):
        list_type = str(value.get("list_type") or "").lower()
        raw = str(value.get("type") or value.get("raw_type") or value.get("canonical_type") or "").lower()
    else:
        raw = str(value or "").lower()
    if list_type == "reference_list":
        return "reference"
    if raw in {"paragraph", "text"}:
        return "text"
    if raw == "title":
        return "title"
    if raw in {"equation", "equation_interline", "interline_equation", "display_formula"}:
        return "equation"
    if raw == "table":
        return "table"
    if raw in {"figure", "image", "chart"}:
        return "figure"
    if raw == "algorithm":
        return "algorithm"
    if raw == "list":
        return "list"
    if raw == "code":
        return "code"
    if raw in {"reference", "references", "bibliography"}:
        return "reference"
    return "other"


def text_for_embedding(item: dict[str, Any]) -> str:
    type_name = canonical_type(item)
    if type_name == "reference":
        return PLACEHOLDER_TEXT[type_name]
    text = str(item.get("text_for_embedding") or "").strip()
    if text:
        return text
    return PLACEHOLDER_TEXT[type_name]
