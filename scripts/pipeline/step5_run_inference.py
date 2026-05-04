#!/usr/bin/env python3
"""Run edge-relation inference and render a LaTeX document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    model = EdgeRelationGAT(config if isinstance(config, EdgeGATConfig) else EdgeGATConfig()).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
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


if __name__ == "__main__":
    raise SystemExit(main())
