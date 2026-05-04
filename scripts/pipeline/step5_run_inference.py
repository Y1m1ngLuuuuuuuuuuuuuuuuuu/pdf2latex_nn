#!/usr/bin/env python3
"""Run edge-relation inference and render a LaTeX document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.latex_renderer import RenderConfig, render_latex_document  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402
from src.reasoning.postprocess import build_resolved_tree, decode_relations_with_arborescence  # noqa: E402


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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    model = EdgeRelationGAT(config if isinstance(config, EdgeGATConfig) else EdgeGATConfig()).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(data.to(device)).detach().cpu()
    decoded = decode_relations_with_arborescence(data.edge_index.detach().cpu(), logits, threshold=args.threshold, num_nodes=data.num_nodes)
    root = build_resolved_tree(list(getattr(data, "node_records", [])), decoded)
    tex = render_latex_document(root, RenderConfig())
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(tex, encoding="utf-8")
    print(f"wrote {args.output_tex}")
    print(f"decoded_edges={len(decoded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
