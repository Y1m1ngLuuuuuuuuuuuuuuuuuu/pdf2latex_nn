#!/usr/bin/env python3
"""Run edge-relation inference and render a LaTeX document."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.v7_contract import assert_v7_graph_data  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402
from src.reasoning.postprocess import TreeDecoder, TreeDecoderConfig  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="Input PyG graph .pt")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint from step4")
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.graph, map_location=device, weights_only=False)
    assert_v7_graph_data(data, args.graph)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_config = checkpoint_compatible_config(config if isinstance(config, EdgeGATConfig) else EdgeGATConfig(), state_dict)
    model = EdgeRelationGAT(model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(data.to(device)).detach().cpu()
    node_records = list(getattr(data, "node_records", []))
    if not node_records:
        node_records = [{} for _ in range(int(data.num_nodes))]
    decoder = TreeDecoder(
        TreeDecoderConfig(
            merge_threshold=args.threshold,
            parent_threshold=args.threshold,
            sibling_threshold=args.threshold,
        )
    )
    root = decoder.decode(node_records, data.edge_index.detach().cpu(), logits)
    tex = decoder.render_document(root)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(tex, encoding="utf-8")
    print(f"wrote {args.output_tex}")
    print(f"decoded_nodes={len(node_records)}")
    return 0


def checkpoint_compatible_config(config: EdgeGATConfig, state_dict: Any) -> EdgeGATConfig:
    """Adapt default config to old checkpoints that serialized only weights."""

    if not isinstance(state_dict, dict):
        return config
    layout_weight = state_dict.get("projector.layout.0.weight")
    if layout_weight is not None:
        checkpoint_layout_dim = int(layout_weight.shape[1])
        if checkpoint_layout_dim != config.node_projector.layout_input_dim:
            config = replace(
                config,
                node_projector=replace(config.node_projector, layout_input_dim_override=checkpoint_layout_dim),
            )
    checkpoint_edge_dim = infer_checkpoint_edge_dim(state_dict, config=config)
    if checkpoint_edge_dim is not None and checkpoint_edge_dim != effective_edge_dim(config):
        extra_dim = gaussian_extra_dim(config)
        raw_edge_dim = checkpoint_edge_dim - extra_dim
        if raw_edge_dim > 0:
            config = replace(config, edge_dim=raw_edge_dim)
    legacy_head_weight = state_dict.get("edge_head.3.weight")
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


def gaussian_extra_dim(config: EdgeGATConfig) -> int:
    mode = getattr(config, "gaussian_edge_feature_mode", "none")
    if mode in (None, "", "none"):
        return 0
    if mode == "center":
        return 1
    return 0


def effective_edge_dim(config: EdgeGATConfig) -> int:
    return int(config.edge_dim) + gaussian_extra_dim(config)


if __name__ == "__main__":
    raise SystemExit(main())
