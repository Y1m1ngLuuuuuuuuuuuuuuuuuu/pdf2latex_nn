"""Build PyTorch Geometric graphs from merged MinerU content v3 JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAGE_SIZE = 1000.0
TYPE_VOCAB = ["text", "title", "equation", "table", "figure", "list", "other"]
NON_TEXT_DENSITY_TYPES = {"equation", "table", "figure"}
PLACEHOLDER_TEXT = {
    "equation": "[EQUATION]",
    "table": "[TABLE]",
    "figure": "[FIGURE]",
    "other": "[EMPTY]",
    "text": "[EMPTY]",
    "title": "[EMPTY]",
    "list": "[EMPTY]",
}


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
class GraphBuildConfig:
    model_path: Path
    max_length: int = 512
    stride: int = 384
    batch_size: int = 16
    bidirectional_edges: bool = True


def load_content_v3(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Expected {path} to contain an items list")
    return [item for item in items if isinstance(item, dict)]


def build_graph_from_content_v3(input_path: Path, output_path: Path, config: GraphBuildConfig) -> Any:
    """Embed v3 nodes, concatenate geometry, and save a PyG Data object."""

    import torch
    from torch_geometric.data import Data

    items = load_content_v3(input_path)
    texts = [text_for_embedding(item) for item in items]
    semantic = embed_texts_scibert_cls(texts, config)
    type_onehot = build_type_onehot_matrix(items)
    geometry = build_geometry_matrix(items)
    stats = build_derived_stats_matrix(items)
    x = torch.cat([semantic, type_onehot, geometry, stats], dim=1)
    edge_index = build_sequential_edge_index(len(items), bidirectional=config.bidirectional_edges)
    data = Data(x=x, edge_index=edge_index)
    data.node_records = make_node_records(items)
    data.feature_schema = {
        "semantic": {"start": 0, "end": 768, "dim": 768, "source": "SciBERT CLS window mean"},
        "type_onehot": {
            "start": 768,
            "end": 775,
            "dim": 7,
            "vocab": TYPE_VOCAB,
        },
        "geometry": {
            "start": 775,
            "end": 779,
            "dim": 4,
            "fields": ["x_start_local", "y_start_page", "x_end_local", "y_end_page"],
        },
        "derived_stats": {
            "start": 779,
            "end": 782,
            "dim": 3,
            "fields": ["macro_position", "aspect_ratio", "text_density"],
        },
    }
    data.source_path = str(input_path)
    data.model_path = str(config.model_path)
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
        type_name = canonical_type(item.get("type"))
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


def build_derived_stats_matrix(items: list[dict[str, Any]]) -> Any:
    import torch

    total_nodes = max(1, len(items) - 1)
    rows = []
    for idx, item in enumerate(items):
        chunks = list(iter_bbox_chunks(item.get("bbox")))
        macro_position = idx / total_nodes
        total_width = sum(max(0.0, bbox[2] - bbox[0]) for bbox in chunks)
        total_height = sum(max(0.0, bbox[3] - bbox[1]) for bbox in chunks)
        area_sum = sum(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]) for bbox in chunks)
        aspect_ratio = total_height / max(total_width, 1.0)
        type_name = canonical_type(item.get("type"))
        if type_name in NON_TEXT_DENSITY_TYPES:
            text_density = 0.0
        else:
            char_count = len(str(item.get("text_for_embedding") or ""))
            text_density = char_count / max(area_sum, 1.0)
        rows.append([macro_position, aspect_ratio, text_density])
    return torch.tensor(rows, dtype=torch.float32)


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


def build_sequential_edge_index(node_count: int, *, bidirectional: bool = True) -> Any:
    import torch

    edges = []
    for idx in range(max(0, node_count - 1)):
        edges.append((idx, idx + 1))
        if bidirectional:
            edges.append((idx + 1, idx))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_node_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in items:
        records.append(
            {
                "global_order": item.get("global_order"),
                "type": item.get("type"),
                "canonical_type": canonical_type(item.get("type")),
                "page_idx": item.get("page_idx"),
                "visual_order": item.get("visual_order"),
                "merge_count": item.get("merge_count"),
                "source_page_idxs": item.get("source_page_idxs"),
                "source_visual_orders": item.get("source_visual_orders"),
                "bbox": item.get("bbox"),
                "text_preview": str(item.get("text_for_embedding") or "")[:200],
            }
        )
    return records


def iter_bbox_chunks(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, list) or len(value) < 4:
        return []
    chunks = []
    usable_len = len(value) - (len(value) % 4)
    for idx in range(0, usable_len, 4):
        chunk = value[idx : idx + 4]
        chunks.append((float(chunk[0]), float(chunk[1]), float(chunk[2]), float(chunk[3])))
    return chunks


def canonical_type(value: Any) -> str:
    raw = str(value or "").lower()
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
    if raw == "list":
        return "list"
    return "other"


def text_for_embedding(item: dict[str, Any]) -> str:
    text = str(item.get("text_for_embedding") or "").strip()
    if text:
        return text
    return PLACEHOLDER_TEXT[canonical_type(item.get("type"))]
