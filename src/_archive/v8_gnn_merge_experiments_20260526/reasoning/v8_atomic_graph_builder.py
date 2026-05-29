"""Build PyG graph objects for v8 middle-derived atomic MERGE training.

This graph family is intentionally separate from the v7 graph schema.  It uses
atomic line/span fragments from MinerU middle.json as graph nodes and only
supervises local MERGE/NONE edges.  PARENT_CHILD is kept as an unused label id
for compatibility with existing 3-class edge-training utilities.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

CHANNELS = [
    "BODY_TEXT",
    "LIST_ITEM",
    "REFERENCE_ITEM",
    "HEADING",
    "CAPTION",
    "DISPLAY_MATH",
    "FLOAT_PROXY",
    "FRONT_MATTER",
    "PAGE_FURNITURE",
    "UNKNOWN",
]

RAW_TYPES = [
    "text",
    "title",
    "list",
    "reference",
    "equation",
    "formula",
    "figure",
    "table",
    "algorithm",
    "caption",
    "code",
    "unknown",
]

EDGE_FAMILIES = [
    "BODY_TEXT_CONTINUATION",
    "LIST_CONTINUATION",
    "REFERENCE_CONTINUATION",
    "FLOAT_SKIP_CONTINUATION",
    "FORMULA_CONTEXT",
    "LAYOUT_SCOPE_MISMATCH",
    "MASKED_UNKNOWN",
]

LAYOUT_SCOPES = [
    "same_column",
    "same_page_cross_column",
    "skip_over_float",
    "cross_page",
    "cross_column",
    "layout_scope_mismatch",
    "unknown",
]

COLUMN_TRANSITIONS = [
    "same_column_down",
    "left_to_right_column",
    "right_to_next_page_left",
    "cross_page_same_column",
    "cross_page_column_reset",
    "other",
]

LABEL_IDS = {
    "MERGE": 0,
    "PARENT_CHILD_UNUSED": 1,
    "NONE": 2,
    "UNKNOWN": 2,
}

LABEL_NAMES = {
    0: "MERGE",
    1: "PARENT_CHILD_UNUSED",
    2: "NONE_OR_MASKED_UNKNOWN",
}

NODE_FEATURE_SCHEMA = (
    [f"channel:{name}" for name in CHANNELS]
    + [f"raw_type:{name}" for name in RAW_TYPES]
    + [
        "bbox_x0_norm",
        "bbox_y0_norm",
        "bbox_x1_norm",
        "bbox_y1_norm",
        "bbox_w_norm",
        "bbox_h_norm",
        "bbox_cx_norm",
        "bbox_cy_norm",
        "page_index_norm",
        "reading_order_norm",
        "block_reading_order_norm",
        "column_left",
        "column_right",
        "column_full_width",
        "column_unknown",
        "text_char_count_norm",
        "text_token_count_norm",
        "text_ends_hyphen",
        "text_open_ended",
        "text_starts_lowercase",
        "text_starts_digit",
        "text_caption_like",
        "text_reference_like",
        "text_math_like",
        "style_font_size_norm",
        "style_bold_ratio",
    ]
)

EDGE_ATTR_SCHEMA = (
    [f"family:{name}" for name in EDGE_FAMILIES]
    + [f"layout_scope:{name}" for name in LAYOUT_SCOPES]
    + [f"column_transition:{name}" for name in COLUMN_TRANSITIONS]
    + [
        "reading_order_gap_norm",
        "page_delta_norm",
        "same_page",
        "same_column",
        "vertical_gap_norm",
        "x_overlap_ratio",
        "skipped_count_norm",
        "skipped_float_count_norm",
        "skipped_formula_count_norm",
        "skipped_caption_count_norm",
        "src_tail_open",
        "src_tail_hyphen",
        "dst_head_lowercase",
        "dst_head_parenthetical",
        "dst_head_citation_like",
        "rel_dx0_norm",
        "rel_dx1_norm",
        "rel_dcx_norm",
        "rel_dy0_norm",
        "rel_dy1_norm",
        "rel_dcy_norm",
        "left_alignment_abs_delta_norm",
        "right_alignment_abs_delta_norm",
        "center_alignment_abs_delta_norm",
        "width_log_ratio",
        "height_log_ratio",
        "area_log_ratio",
        "y_overlap_ratio",
        "vertical_gap_by_line_height",
        "same_middle_block",
        "same_content_owner",
        "same_style_content_owner",
        "line_index_gap_norm",
        "span_index_gap_norm",
        "src_font_size_norm",
        "dst_font_size_norm",
        "font_size_delta_norm",
        "font_size_abs_delta_norm",
        "same_font_size_bucket",
        "src_bold_ratio",
        "dst_bold_ratio",
        "bold_ratio_delta",
        "bold_ratio_abs_delta",
        "same_bold_state",
        "cross_page_bottom_to_top",
        "src_bottom_page_norm",
        "dst_top_page_norm",
        "src_tail_comma",
        "src_tail_alpha_end",
        "src_tail_hard_terminal",
        "src_tail_soft_punctuation",
        "src_tail_abbrev_like",
        "src_tail_citation_closed",
        "dst_tail_hard_terminal",
        "dst_tail_soft_punctuation",
        "src_unclosed_parenthesis",
        "src_unclosed_bracket",
        "src_unclosed_quote",
        "src_tail_after_math_symbol",
        "src_tail_last_token_stopword",
        "dst_head_first_token_stopword",
        "dst_starts_punctuation",
        "dst_starts_closing_bracket",
        "dst_head_conjunction",
        "dst_head_preposition",
        "dst_head_uppercase",
        "dst_head_continuation_word",
        "src_near_column_bottom",
        "dst_near_column_top",
        "same_column_flow_lane",
        "skipped_figure_count_norm",
        "skipped_table_count_norm",
        "skipped_algorithm_count_norm",
        "skipped_code_count_norm",
        "skipped_wide_float_count_norm",
        "skipped_full_width_float_count_norm",
        "skipped_max_float_width_ratio",
        "skipped_has_caption",
        "skipped_display_math_between",
        "src_before_float_dst_after_float",
        "src_prev2_body_text_count_norm",
        "src_prev2_list_count_norm",
        "src_prev2_heading_count_norm",
        "src_prev2_formula_count_norm",
        "src_prev2_float_count_norm",
        "src_prev2_caption_count_norm",
        "src_prev2_reference_count_norm",
        "src_prev2_front_matter_count_norm",
        "src_next2_body_text_count_norm",
        "src_next2_list_count_norm",
        "src_next2_heading_count_norm",
        "src_next2_formula_count_norm",
        "src_next2_float_count_norm",
        "src_next2_caption_count_norm",
        "src_next2_reference_count_norm",
        "src_next2_front_matter_count_norm",
        "dst_prev2_body_text_count_norm",
        "dst_prev2_list_count_norm",
        "dst_prev2_heading_count_norm",
        "dst_prev2_formula_count_norm",
        "dst_prev2_float_count_norm",
        "dst_prev2_caption_count_norm",
        "dst_prev2_reference_count_norm",
        "dst_prev2_front_matter_count_norm",
        "dst_next2_body_text_count_norm",
        "dst_next2_list_count_norm",
        "dst_next2_heading_count_norm",
        "dst_next2_formula_count_norm",
        "dst_next2_float_count_norm",
        "dst_next2_caption_count_norm",
        "dst_next2_reference_count_norm",
        "dst_next2_front_matter_count_norm",
        "between_node_count_norm",
        "between_body_text_count_norm",
        "between_list_count_norm",
        "between_barrier_count_norm",
        "src_prev_gap_by_line_height",
        "src_next_gap_by_line_height",
        "dst_prev_gap_by_line_height",
        "dst_next_gap_by_line_height",
        "candidate_gap_vs_src_next_gap",
        "candidate_gap_vs_dst_prev_gap",
        "same_indent_bucket",
        "font_size_ratio_clipped",
        "bold_state_transition",
    ]
)


def build_v8_atomic_pyg_data(
    graph_payload: dict[str, Any],
    label_payload: dict[str, Any] | None = None,
    *,
    source_graph_path: str | None = None,
    source_label_path: str | None = None,
) -> Any:
    """Convert a v8 atomic graph-view payload and label sidecar to PyG Data."""

    try:
        import torch
        from torch_geometric.data import Data
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise ModuleNotFoundError("v8 atomic graph conversion requires torch and torch_geometric") from exc

    nodes = list(graph_payload.get("nodes") or [])
    edges = list(graph_payload.get("candidate_edges") or [])
    labels_by_edge = _labels_by_edge_id(label_payload)
    node_index = {str(node.get("atomic_id")): idx for idx, node in enumerate(nodes)}
    num_pages = _infer_num_pages(nodes)

    x = torch.tensor([_node_features(node, nodes, num_pages) for node in nodes], dtype=torch.float32)

    edge_pairs: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    y_values: list[int] = []
    train_mask_values: list[bool] = []
    loss_weight_values: list[float] = []
    merge_candidate_values: list[bool] = []
    edge_records: list[dict[str, Any]] = []

    for edge in edges:
        src_id = str(edge.get("src_atomic_id") or edge.get("src") or "")
        dst_id = str(edge.get("dst_atomic_id") or edge.get("dst") or "")
        if src_id not in node_index or dst_id not in node_index:
            continue
        label = labels_by_edge.get(str(edge.get("edge_id")), {})
        label_name = str(label.get("label") or "UNKNOWN")
        y = int(LABEL_IDS.get(label_name, 2))
        train_mask = bool(label.get("train_mask", False))
        weight = float(label.get("proposed_loss_weight", 0.0) or 0.0)
        family = str(edge.get("candidate_family") or label.get("relation_family") or "MASKED_UNKNOWN")

        edge_pairs.append([node_index[src_id], node_index[dst_id]])
        src_node = nodes[node_index[src_id]]
        dst_node = nodes[node_index[dst_id]]
        edge_attrs.append(_edge_features(edge, src_node, dst_node))
        y_values.append(y)
        train_mask_values.append(train_mask)
        loss_weight_values.append(weight)
        merge_candidate_values.append(family in {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION", "FLOAT_SKIP_CONTINUATION"})
        edge_records.append(
            {
                "edge_id": edge.get("edge_id"),
                "src": src_id,
                "dst": dst_id,
                "candidate_family": family,
                "label": label_name,
                "label_strength": label.get("label_strength"),
                "train_mask": train_mask,
                "proposed_loss_weight": weight,
            }
        )

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(EDGE_ATTR_SCHEMA)), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(y_values, dtype=torch.long),
        edge_train_mask=torch.tensor(train_mask_values, dtype=torch.bool),
        edge_loss_weight=torch.tensor(loss_weight_values, dtype=torch.float32),
        merge_candidate_mask=torch.tensor(merge_candidate_values, dtype=torch.bool),
        message_edge_mask=torch.ones(edge_index.shape[1], dtype=torch.bool),
    )
    data.doc_id = str(graph_payload.get("doc_id") or "")
    data.graph_schema_version = "v8_atomic_merge_graph_v1_4"
    data.edge_relative_geometry_version = "v1.4"
    data.label_schema = {"labels": LABEL_NAMES, "parent_child_label_is_unused": True}
    data.node_feature_schema = list(NODE_FEATURE_SCHEMA)
    data.edge_attr_schema = list(EDGE_ATTR_SCHEMA)
    data.node_records = _compact_node_records(nodes)
    data.edge_records = edge_records
    data.source_graph_view = source_graph_path
    data.source_labels = source_label_path
    data.graph_content_hash = _stable_hash({"nodes": data.node_records, "edges": edge_records})
    return data


def summarize_v8_atomic_graph_payload(graph_payload: dict[str, Any], label_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    nodes = list(graph_payload.get("nodes") or [])
    edges = list(graph_payload.get("candidate_edges") or [])
    labels = list((label_payload or {}).get("edge_labels") or [])
    channel_counts = Counter(str(node.get("channel") or "UNKNOWN") for node in nodes)
    family_counts = Counter(str(edge.get("candidate_family") or "MASKED_UNKNOWN") for edge in edges)
    label_counts = Counter(str(label.get("label") or "UNKNOWN") for label in labels)
    strength_counts = Counter(str(label.get("label_strength") or "unknown") for label in labels)
    trainable = [label for label in labels if label.get("train_mask")]
    trainable_merge = [label for label in trainable if label.get("label") == "MERGE"]
    trainable_none = [label for label in trainable if label.get("label") == "NONE"]
    return {
        "doc_id": graph_payload.get("doc_id"),
        "node_count": len(nodes),
        "candidate_edge_count": len(edges),
        "edge_label_count": len(labels),
        "trainable_edge_count": len(trainable),
        "trainable_merge_positive_count": len(trainable_merge),
        "trainable_none_negative_count": len(trainable_none),
        "channel_counts": dict(sorted(channel_counts.items())),
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "label_strength_counts": dict(sorted(strength_counts.items())),
        "node_feature_dim": len(NODE_FEATURE_SCHEMA),
        "edge_attr_dim": len(EDGE_ATTR_SCHEMA),
    }


def save_v8_atomic_pyg_data(data: Any, path: str | Path) -> None:
    import torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)


def _labels_by_edge_id(label_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    labels = (label_payload or {}).get("edge_labels") or []
    return {str(label.get("edge_id")): label for label in labels}


def _node_features(node: dict[str, Any], all_nodes: list[dict[str, Any]], num_pages: int) -> list[float]:
    channel = str(node.get("channel") or "UNKNOWN")
    raw_type = _normalize_raw_type(str(node.get("raw_type") or node.get("type") or "unknown"))
    page_idx = _safe_float(node.get("page_idx"), 0.0)
    bbox = node.get("bbox") if isinstance(node.get("bbox"), list) else None
    page_size = node.get("page_size") if isinstance(node.get("page_size"), list) else None
    page_w = max(1.0, _safe_float((page_size or [1, 1])[0], 1.0))
    page_h = max(1.0, _safe_float((page_size or [1, 1])[1], 1.0))
    x0, y0, x1, y1 = _bbox4(bbox)
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    cx = x0 + w / 2.0
    cy = y0 + h / 2.0
    max_ro = max(1.0, max(_safe_float(n.get("reading_order"), 0.0) for n in all_nodes))
    max_block_ro = max(1.0, max(_safe_float(n.get("block_reading_order"), 0.0) for n in all_nodes))
    column = str(node.get("column_id") or "unknown").lower()
    is_full = bool(node.get("is_full_width"))
    text = str(node.get("text") or "")
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    stripped = text.strip()
    style = node.get("style") if isinstance(node.get("style"), dict) else {}
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    font_size = _safe_float(style.get("font_size") or metadata.get("font_size"), 0.0)
    bold_ratio = _safe_float(style.get("bold_ratio") or metadata.get("bold_ratio"), 0.0)

    return (
        _one_hot(channel, CHANNELS)
        + _one_hot(raw_type, RAW_TYPES)
        + [
            x0 / page_w,
            y0 / page_h,
            x1 / page_w,
            y1 / page_h,
            w / page_w,
            h / page_h,
            cx / page_w,
            cy / page_h,
            page_idx / max(1.0, float(num_pages - 1)),
            _safe_float(node.get("reading_order"), 0.0) / max_ro,
            _safe_float(node.get("block_reading_order"), 0.0) / max_block_ro,
            1.0 if column in {"0", "left", "l"} and not is_full else 0.0,
            1.0 if column in {"1", "right", "r"} and not is_full else 0.0,
            1.0 if is_full else 0.0,
            1.0 if column in {"unknown", "", "none"} and not is_full else 0.0,
            min(len(stripped), 300) / 300.0,
            min(len(tokens), 80) / 80.0,
            1.0 if stripped.endswith("-") else 0.0,
            1.0 if stripped and stripped[-1:] not in ".?!:;]" else 0.0,
            1.0 if stripped[:1].islower() else 0.0,
            1.0 if stripped[:1].isdigit() else 0.0,
            1.0 if re.match(r"(?i)^(fig\\.?|figure|tab\\.?|table|algorithm)\\s+[A-Z0-9IVX]", stripped) else 0.0,
            1.0 if re.match(r"^\\s*(\\[\\d+\\]|\\d+\\.|[A-Z][a-z]+,\\s+[A-Z])", stripped) else 0.0,
            1.0 if any(sym in stripped for sym in ("=", "\\", "∑", "∫", "≤", "≥")) else 0.0,
            min(font_size, 30.0) / 30.0,
            max(0.0, min(bold_ratio, 1.0)),
        ]
    )


def _edge_features(edge: dict[str, Any], src_node: dict[str, Any], dst_node: dict[str, Any]) -> list[float]:
    family = str(edge.get("candidate_family") or "MASKED_UNKNOWN")
    scope = str(edge.get("layout_scope") or "unknown")
    skipped_channels = edge.get("skipped_channels") if isinstance(edge.get("skipped_channels"), list) else []
    feature_meta = edge.get("features") if isinstance(edge.get("features"), dict) else {}
    column_transition = str(feature_meta.get("column_transition_type") or "other")
    skipped_count = len(skipped_channels)
    src_tail = str(edge.get("src_tail") or "")
    dst_head = str(edge.get("dst_head") or "")
    vertical_gap = _safe_float(edge.get("vertical_gap"), 0.0)
    x_overlap = _safe_float(edge.get("x_overlap_ratio"), 0.0)
    page_gap = abs(_safe_float(edge.get("page_delta", edge.get("page_gap")), 0.0))
    order_gap = abs(_safe_float(edge.get("reading_order_gap"), 0.0))
    src_box = _node_geometry(src_node)
    dst_box = _node_geometry(dst_node)
    src_style = _node_style(src_node)
    dst_style = _node_style(dst_node)
    page_w = max(1.0, max(src_box["page_w"], dst_box["page_w"]))
    page_h = max(1.0, max(src_box["page_h"], dst_box["page_h"]))
    src_w = max(0.0, src_box["w"])
    dst_w = max(0.0, dst_box["w"])
    src_h = max(0.0, src_box["h"])
    dst_h = max(0.0, dst_box["h"])
    src_area = src_w * src_h
    dst_area = dst_w * dst_h
    y_overlap = _interval_overlap_ratio(src_box["y0"], src_box["y1"], dst_box["y0"], dst_box["y1"])
    line_height = max(1.0, src_h, dst_h)
    same_middle_block = _same_nonempty(src_node.get("source_middle_block_id"), dst_node.get("source_middle_block_id"))
    same_content_owner = _same_nonempty(src_node.get("source_content_list_index"), dst_node.get("source_content_list_index"))
    same_style_content_owner = _same_nonempty(src_node.get("style_content_list_index"), dst_node.get("style_content_list_index"))
    src_line = _safe_float(src_node.get("line_index"), -1.0)
    dst_line = _safe_float(dst_node.get("line_index"), -1.0)
    src_span = _safe_float(src_node.get("span_index"), -1.0)
    dst_span = _safe_float(dst_node.get("span_index"), -1.0)
    line_gap = abs(dst_line - src_line) if src_line >= 0 and dst_line >= 0 else 0.0
    span_gap = abs(dst_span - src_span) if src_span >= 0 and dst_span >= 0 else 0.0
    src_font = src_style["font_size"]
    dst_font = dst_style["font_size"]
    src_font_norm = _font_size_norm(src_font)
    dst_font_norm = _font_size_norm(dst_font)
    font_delta = dst_font - src_font if src_font > 0 and dst_font > 0 else 0.0
    font_delta_norm = max(-1.0, min(font_delta / 12.0, 1.0))
    same_font_bucket = _same_font_bucket(src_font, dst_font)
    src_bold = max(0.0, min(src_style["bold_ratio"], 1.0))
    dst_bold = max(0.0, min(dst_style["bold_ratio"], 1.0))
    bold_delta = max(-1.0, min(dst_bold - src_bold, 1.0))
    src_bottom_norm = src_box["y1"] / max(1.0, src_box["page_h"])
    dst_top_norm = dst_box["y0"] / max(1.0, dst_box["page_h"])
    page_delta_value = _safe_float(edge.get("page_delta", edge.get("page_gap")), 0.0)
    cross_page_bottom_to_top = bool(
        int(page_delta_value) == 1
        and src_bottom_norm >= 0.65
        and dst_top_norm <= 0.35
    )
    same_indent_bucket = abs(dst_box["x0"] - src_box["x0"]) <= max(2.0, 0.012 * page_w)
    font_size_ratio = (dst_font / src_font) if src_font > 0 and dst_font > 0 else 1.0
    font_size_ratio_clipped = max(0.0, min(font_size_ratio, 3.0)) / 3.0
    bold_state_transition = float(int(dst_bold >= 0.5) - int(src_bold >= 0.5))
    return (
        _one_hot(family, EDGE_FAMILIES)
        + _one_hot(scope, LAYOUT_SCOPES)
        + _one_hot(column_transition, COLUMN_TRANSITIONS)
        + [
            min(order_gap, 20.0) / 20.0,
            min(page_gap, 5.0) / 5.0,
            1.0 if edge.get("same_page") else 0.0,
            1.0 if edge.get("same_column") else 0.0,
            max(-1.0, min(vertical_gap / 1000.0, 1.0)),
            max(0.0, min(x_overlap, 1.0)),
            min(skipped_count, 10) / 10.0,
            min(sum(1 for c in skipped_channels if c == "FLOAT_PROXY"), 5) / 5.0,
            min(sum(1 for c in skipped_channels if c == "DISPLAY_MATH"), 5) / 5.0,
            min(sum(1 for c in skipped_channels if c == "CAPTION"), 5) / 5.0,
            1.0 if src_tail and src_tail[-1:] not in ".?!:;]" else 0.0,
            1.0 if src_tail.endswith("-") else 0.0,
            1.0 if dst_head[:1].islower() else 0.0,
            1.0 if dst_head.startswith("(") else 0.0,
            1.0 if re.match(r"^\(?\s*(Table|Tab\.|Fig\.|Figure|\[?\d+\])", dst_head) else 0.0,
            max(-1.0, min((dst_box["x0"] - src_box["x0"]) / page_w, 1.0)),
            max(-1.0, min((dst_box["x1"] - src_box["x1"]) / page_w, 1.0)),
            max(-1.0, min((dst_box["cx"] - src_box["cx"]) / page_w, 1.0)),
            max(-1.0, min((dst_box["y0"] - src_box["y0"]) / page_h, 1.0)),
            max(-1.0, min((dst_box["y1"] - src_box["y1"]) / page_h, 1.0)),
            max(-1.0, min((dst_box["cy"] - src_box["cy"]) / page_h, 1.0)),
            min(abs(dst_box["x0"] - src_box["x0"]) / page_w, 1.0),
            min(abs(dst_box["x1"] - src_box["x1"]) / page_w, 1.0),
            min(abs(dst_box["cx"] - src_box["cx"]) / page_w, 1.0),
            _safe_log_ratio(dst_w + 1.0, src_w + 1.0, denom=3.0),
            _safe_log_ratio(dst_h + 1.0, src_h + 1.0, denom=3.0),
            _safe_log_ratio(dst_area + 1.0, src_area + 1.0, denom=5.0),
            y_overlap,
            max(-1.0, min(vertical_gap / line_height, 1.0)),
            1.0 if same_middle_block else 0.0,
            1.0 if same_content_owner else 0.0,
            1.0 if same_style_content_owner else 0.0,
            min(line_gap, 20.0) / 20.0,
            min(span_gap, 20.0) / 20.0,
            src_font_norm,
            dst_font_norm,
            font_delta_norm,
            abs(font_delta_norm),
            1.0 if same_font_bucket else 0.0,
            src_bold,
            dst_bold,
            bold_delta,
            abs(bold_delta),
            1.0 if (src_bold >= 0.5) == (dst_bold >= 0.5) else 0.0,
            1.0 if cross_page_bottom_to_top else 0.0,
            max(0.0, min(src_bottom_norm, 1.0)),
            max(0.0, min(dst_top_norm, 1.0)),
            1.0 if feature_meta.get("src_comma_ended") else 0.0,
            1.0 if feature_meta.get("src_alpha_ended") else 0.0,
            1.0 if feature_meta.get("src_tail_hard_terminal") else 0.0,
            1.0 if feature_meta.get("src_tail_soft_punctuation") else 0.0,
            1.0 if feature_meta.get("src_tail_abbrev_like") else 0.0,
            1.0 if feature_meta.get("src_tail_citation_closed") else 0.0,
            1.0 if feature_meta.get("dst_tail_hard_terminal") else 0.0,
            1.0 if feature_meta.get("dst_tail_soft_punctuation") else 0.0,
            1.0 if feature_meta.get("src_unclosed_parenthesis") else 0.0,
            1.0 if feature_meta.get("src_unclosed_bracket") else 0.0,
            1.0 if feature_meta.get("src_unclosed_quote") else 0.0,
            1.0 if feature_meta.get("src_tail_after_math_symbol") else 0.0,
            1.0 if feature_meta.get("src_tail_last_token_stopword") else 0.0,
            1.0 if feature_meta.get("dst_head_first_token_stopword") else 0.0,
            1.0 if feature_meta.get("dst_starts_punctuation") else 0.0,
            1.0 if feature_meta.get("dst_starts_closing_bracket") else 0.0,
            1.0 if feature_meta.get("dst_head_conjunction") else 0.0,
            1.0 if feature_meta.get("dst_head_preposition") else 0.0,
            1.0 if feature_meta.get("dst_uppercase_start") else 0.0,
            1.0 if feature_meta.get("dst_continuation_word_start") else 0.0,
            1.0 if feature_meta.get("src_near_column_bottom") else 0.0,
            1.0 if feature_meta.get("dst_near_column_top") else 0.0,
            1.0 if feature_meta.get("same_column_flow_lane") else 0.0,
            _count_feature_norm(feature_meta, "skipped_figure_count", denom=5.0),
            _count_feature_norm(feature_meta, "skipped_table_count", denom=5.0),
            _count_feature_norm(feature_meta, "skipped_algorithm_count", denom=5.0),
            _count_feature_norm(feature_meta, "skipped_code_count", denom=5.0),
            _count_feature_norm(feature_meta, "skipped_wide_float_count", denom=5.0),
            _count_feature_norm(feature_meta, "skipped_full_width_float_count", denom=5.0),
            max(0.0, min(_safe_float(feature_meta.get("skipped_max_float_width_ratio"), 0.0), 1.0)),
            1.0 if feature_meta.get("skipped_has_caption") else 0.0,
            1.0 if feature_meta.get("skipped_display_math_between") else 0.0,
            1.0 if feature_meta.get("src_before_float_dst_after_float") else 0.0,
            *[_count_feature_norm(feature_meta, key, denom=2.0) for key in _LOCAL_WINDOW_FEATURE_KEYS],
            _count_feature_norm(feature_meta, "between_node_count", denom=10.0),
            _count_feature_norm(feature_meta, "between_body_text_count", denom=5.0),
            _count_feature_norm(feature_meta, "between_list_count", denom=5.0),
            _count_feature_norm(feature_meta, "between_barrier_count", denom=5.0),
            _ratio_feature_clip(feature_meta, "src_prev_gap_by_line_height"),
            _ratio_feature_clip(feature_meta, "src_next_gap_by_line_height"),
            _ratio_feature_clip(feature_meta, "dst_prev_gap_by_line_height"),
            _ratio_feature_clip(feature_meta, "dst_next_gap_by_line_height"),
            _ratio_feature_clip(feature_meta, "candidate_gap_vs_src_next_gap"),
            _ratio_feature_clip(feature_meta, "candidate_gap_vs_dst_prev_gap"),
            1.0 if same_indent_bucket else 0.0,
            font_size_ratio_clipped,
            bold_state_transition,
        ]
    )


_LOCAL_WINDOW_FEATURE_KEYS = [
    f"{prefix}_{kind}_count"
    for prefix in ("src_prev2", "src_next2", "dst_prev2", "dst_next2")
    for kind in ("body_text", "list", "heading", "formula", "float", "caption", "reference", "front_matter")
]


def _count_feature_norm(features: dict[str, Any], key: str, *, denom: float) -> float:
    return max(0.0, min(_safe_float(features.get(key), 0.0), denom)) / max(denom, 1.0)


def _ratio_feature_clip(features: dict[str, Any], key: str, *, denom: float = 5.0) -> float:
    value = _safe_float(features.get(key), 0.0)
    return max(-1.0, min(value / max(denom, 1.0), 1.0))


def _node_geometry(node: dict[str, Any]) -> dict[str, float]:
    bbox = node.get("bbox") if isinstance(node.get("bbox"), list) else None
    page_size = node.get("page_size") if isinstance(node.get("page_size"), list) else None
    page_w = max(1.0, _safe_float((page_size or [1, 1])[0], 1.0))
    page_h = max(1.0, _safe_float((page_size or [1, 1])[1], 1.0))
    x0, y0, x1, y1 = _bbox4(bbox)
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "w": w,
        "h": h,
        "cx": x0 + w / 2.0,
        "cy": y0 + h / 2.0,
        "page_w": page_w,
        "page_h": page_h,
    }


def _node_style(node: dict[str, Any]) -> dict[str, float]:
    style = node.get("style") if isinstance(node.get("style"), dict) else {}
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    font_size = _safe_float(
        style.get("font_size")
        or metadata.get("font_size")
        or node.get("font_size"),
        0.0,
    )
    bold_ratio = _safe_float(
        style.get("bold_ratio")
        or metadata.get("bold_ratio")
        or node.get("bold_ratio"),
        0.0,
    )
    relative_font_size = _safe_float(
        style.get("relative_font_size")
        or metadata.get("relative_font_size")
        or node.get("relative_font_size"),
        0.0,
    )
    return {
        "font_size": font_size,
        "bold_ratio": bold_ratio,
        "relative_font_size": relative_font_size,
    }


def _interval_overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1.0, min(max(0.0, a1 - a0), max(0.0, b1 - b0)))
    return max(0.0, min(overlap / denom, 1.0))


def _safe_log_ratio(numerator: float, denominator: float, *, denom: float) -> float:
    numerator = max(numerator, 1e-6)
    denominator = max(denominator, 1e-6)
    return max(-1.0, min(math.log(numerator / denominator) / denom, 1.0))


def _font_size_norm(value: float) -> float:
    if value <= 0:
        return 0.0
    return max(0.0, min(value, 30.0) / 30.0)


def _same_font_bucket(src_font: float, dst_font: float) -> bool:
    if src_font <= 0 or dst_font <= 0:
        return False
    return abs(src_font - dst_font) <= max(0.5, 0.06 * max(src_font, dst_font))


def _same_nonempty(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    left_s = str(left)
    right_s = str(right)
    if left_s == "" or right_s == "":
        return False
    return left_s == right_s


def _one_hot(value: str, vocabulary: list[str]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocabulary]


def _normalize_raw_type(value: str) -> str:
    value = value.lower().strip()
    if value in RAW_TYPES:
        return value
    if value in {"para", "paragraph"}:
        return "text"
    if "equation" in value:
        return "equation"
    if "formula" in value:
        return "formula"
    if "reference" in value or value == "ref":
        return "reference"
    return "unknown"


def _bbox4(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, list) and len(value) >= 4:
        return tuple(_safe_float(v, 0.0) for v in value[:4])  # type: ignore[return-value]
    return (0.0, 0.0, 0.0, 0.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_num_pages(nodes: list[dict[str, Any]]) -> int:
    if not nodes:
        return 1
    return int(max(_safe_float(node.get("page_idx"), 0.0) for node in nodes)) + 1


def _compact_node_records(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for idx, node in enumerate(nodes):
        compact.append(
            {
                "node_index": idx,
                "atomic_id": node.get("atomic_id"),
                "text_preview": str(node.get("text") or "")[:160],
                "channel": node.get("channel"),
                "page_idx": node.get("page_idx"),
                "bbox": node.get("bbox"),
                "reading_order": node.get("reading_order"),
                "source_middle_block_id": node.get("source_middle_block_id"),
                "source_line_id": node.get("source_line_id"),
                "source_v7_id": node.get("source_v7_id"),
            }
        )
    return compact


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
