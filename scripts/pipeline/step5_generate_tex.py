#!/usr/bin/env python3
"""Generate LaTeX from a graph checkpoint via TreeDecoder."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.reading_order import fuse_micro_nodes  # noqa: E402
from src.pipeline.v7_contract import assert_v7_content_json, assert_v7_graph_data  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402
from src.reasoning.postprocess import TreeDecoder, TreeDecoderConfig  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="Input PyG graph .pt")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint .pth/.pt")
    parser.add_argument("--output-tex", type=Path, required=True, help="Generated .tex path")
    parser.add_argument("--content-json", type=Path, help="Optional content_v7_styles.json with full node text")
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--parent-threshold", type=float, default=0.0)
    parser.add_argument("--sibling-threshold", type=float, default=0.5)
    parser.add_argument("--title", default=None)
    parser.add_argument("--logits-output", type=Path, help="Optional tensor path for raw edge logits")
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.graph, map_location=device, weights_only=False)
    assert_v7_graph_data(data, args.graph)
    if args.content_json is not None:
        assert_v7_content_json(args.content_json, require_styles=True)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    config = checkpoint_compatible_config(config if isinstance(config, EdgeGATConfig) else EdgeGATConfig(), state_dict)
    model = EdgeRelationGAT(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(data.to(device)).detach().cpu()
    if args.logits_output:
        args.logits_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(logits, args.logits_output)

    node_records = load_node_records(args.content_json, data)
    decoder = TreeDecoder(
        TreeDecoderConfig(
            merge_threshold=args.merge_threshold,
            parent_threshold=args.parent_threshold,
            sibling_threshold=args.sibling_threshold,
        )
    )
    root = decoder.decode(node_records, data.edge_index.detach().cpu(), logits)
    document_title = args.title or infer_document_title(node_records)
    tex = decoder.render_document(root, title=document_title)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(tex, encoding="utf-8")
    pred_counts = torch.bincount(torch.clamp(logits.argmax(dim=-1), max=2), minlength=3).tolist()
    print(f"wrote {args.output_tex}")
    print(f"nodes={len(node_records)} edges={int(data.edge_index.shape[1])}")
    print(f"predicted_argmax_counts={{0: {pred_counts[0]}, 1: {pred_counts[1]}, 2: {pred_counts[2]}}}")
    return 0


def checkpoint_compatible_config(config: EdgeGATConfig, state_dict: Any) -> EdgeGATConfig:
    """Adapt model dimensions to legacy checkpoints without changing weights."""

    layout_weight = state_dict.get("projector.layout.0.weight") if isinstance(state_dict, dict) else None
    legacy_head_weight = state_dict.get("edge_head.3.weight") if isinstance(state_dict, dict) else None
    if layout_weight is not None:
        checkpoint_layout_dim = int(layout_weight.shape[1])
        if checkpoint_layout_dim != config.node_projector.layout_input_dim:
            config = replace(
                config,
                node_projector=replace(config.node_projector, layout_input_dim_override=checkpoint_layout_dim),
            )
    checkpoint_edge_dim = infer_checkpoint_edge_dim(state_dict, config=config)
    if checkpoint_edge_dim is not None and checkpoint_edge_dim != config.edge_dim:
        config = replace(config, edge_dim=checkpoint_edge_dim)
    if legacy_head_weight is not None and "edge_head.4.weight" not in state_dict and "edge_head.12.weight" not in state_dict:
        first_head_weight = state_dict.get("edge_head.0.weight")
        if first_head_weight is not None:
            config = replace(
                config,
                predictor_hidden_dims=(int(first_head_weight.shape[0]),),
                predictor_layer_norm=False,
            )
        if int(legacy_head_weight.shape[0]) != config.num_classes:
            config = replace(config, num_classes=int(legacy_head_weight.shape[0]))
    return config


def infer_checkpoint_edge_dim(state_dict: Any, *, config: EdgeGATConfig) -> int | None:
    if not isinstance(state_dict, dict):
        return None
    for key in ("convs.0.lin_edge.weight", "convs.0.lin_edge.lin.weight"):
        weight = state_dict.get(key)
        if weight is not None and getattr(weight, "ndim", 0) == 2:
            return int(weight.shape[1])
    first_head_weight = state_dict.get("edge_head.0.weight")
    if first_head_weight is not None and getattr(first_head_weight, "ndim", 0) == 2:
        node_relation_dim = int(config.hidden_dim) * int(config.heads) * 4
        inferred = int(first_head_weight.shape[1]) - node_relation_dim
        if inferred > 0:
            return inferred
    return None


def load_node_records(content_json: Path | None, data: Any) -> list[dict[str, Any]]:
    graph_records = [dict(record) for record in list(getattr(data, "node_records", [])) if isinstance(record, dict)]
    if content_json is not None:
        payload = json.loads(content_json.read_text(encoding="utf-8"))
        items = payload.get("items", payload if isinstance(payload, list) else [])
        if not isinstance(items, list):
            raise ValueError(f"Expected content JSON with an items list: {content_json}")
        raw_records = [record_from_content_item(item) for item in items if isinstance(item, dict)]
        records = select_records_for_graph(raw_records, data)
        if len(records) != int(data.num_nodes):
            raise ValueError(f"content records ({len(records)}) do not match graph.num_nodes ({int(data.num_nodes)})")
        if len(graph_records) == len(records):
            records = [merge_graph_node_metadata(content_record, graph_record) for content_record, graph_record in zip(records, graph_records)]
        return records
    if graph_records:
        return graph_records
    return [{} for _ in range(int(data.num_nodes))]


def select_records_for_graph(records: list[dict[str, Any]], data: Any) -> list[dict[str, Any]]:
    """Match content JSON records to graph nodes, trying micro-fusion when needed."""

    expected = int(data.num_nodes)
    fused_records: list[dict[str, Any]] | None = None
    if bool(getattr(data, "micro_fusion_applied", False)):
        fused_records = fuse_micro_nodes(records)
        if len(fused_records) == expected:
            return fused_records
    if len(records) == expected:
        return records
    fused_records = fused_records if fused_records is not None else fuse_micro_nodes(records)
    if len(fused_records) == expected:
        return fused_records
    return records


def record_from_content_item(item: dict[str, Any]) -> dict[str, Any]:
    record = dict(item)
    text = item.get("text_for_embedding") or item.get("text") or item.get("content") or item.get("latex") or ""
    record["text"] = str(text)
    if "canonical_type" not in record:
        record["canonical_type"] = canonical_content_type(item)
    return record


def merge_graph_node_metadata(content_record: dict[str, Any], graph_record: dict[str, Any]) -> dict[str, Any]:
    merged = dict(content_record)
    for key, value in graph_record.items():
        if key in {"text", "text_for_embedding", "content", "latex", "html", "reference_items", "merged_records"}:
            continue
        if key not in merged or merged[key] in (None, "", []):
            merged[key] = value
        elif key in {
            "regime_reading_order",
            "dag_reading_order",
            "xycut_reading_order",
            "global_order",
            "reading_order",
            "original_order",
            "original_index",
            "column_id",
            "is_full_width",
            "style_baseline_size",
        }:
            merged[key] = value
    return merged


def infer_document_title(records: list[dict[str, Any]]) -> str | None:
    """Use the first real title-like content block as the document title."""

    for record in records[:20]:
        if canonical_content_type(record) != "title":
            continue
        text = str(record.get("text") or record.get("text_for_embedding") or "").strip()
        if not text:
            continue
        normalized = normalized_title_key(text)
        if normalized in {"abstract", "keywords", "introduction", "references", "bibliography"}:
            continue
        if looks_like_arxiv_identifier(text):
            continue
        return text
    return None


def normalized_title_key(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum())


def looks_like_arxiv_identifier(text: str) -> bool:
    stripped = text.strip()
    compact = "".join(char for char in stripped if char.isdigit() or char in ".v")
    return bool(re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", compact))


def canonical_content_type(value: Any) -> str:
    list_type = ""
    if isinstance(value, dict):
        list_type = str(value.get("list_type") or "").lower()
        raw = str(value.get("canonical_type") or value.get("type") or value.get("raw_type") or "").lower()
    else:
        raw = str(value or "").lower()
    if list_type == "reference_list":
        return "reference"
    if raw in {"paragraph", "text"}:
        return "text"
    if raw in {"title", "section", "subsection", "subsubsection"}:
        return "title"
    if raw in {"equation", "equation_interline", "interline_equation", "display_formula", "formula"}:
        return "equation"
    if raw in {"inline_math", "inline_formula", "math_inline"}:
        return "inline_math"
    if raw in {"table"}:
        return "table"
    if raw in {"figure", "image", "chart"}:
        return "figure"
    if raw in {"algorithm"}:
        return "algorithm"
    if raw in {"list", "item", "itemize", "enumerate"}:
        return "list"
    if raw in {"code"}:
        return "code"
    if raw in {"reference", "references", "bibliography"}:
        return "reference"
    return "text"


if __name__ == "__main__":
    raise SystemExit(main())
