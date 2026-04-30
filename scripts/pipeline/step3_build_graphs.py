#!/usr/bin/env python3
"""Build PyG graph feature tensors from content v3 JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.graph_builder import GraphBuildConfig, build_graph_from_content_v3  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="*_content_list_v3.json file")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt graph path")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local SciBERT model directory, e.g. models/huggingface/allenai/scibert_scivocab_uncased",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--directed", action="store_true", help="Use only forward sequential edges")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = GraphBuildConfig(
        model_path=args.model_path,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
        bidirectional_edges=not args.directed,
    )
    data = build_graph_from_content_v3(args.input, args.output, config)
    print(f"wrote {args.output}")
    print(f"x_shape={tuple(data.x.shape)}")
    print(f"edge_index_shape={tuple(data.edge_index.shape)}")
    print(f"nodes={data.num_nodes} edges={data.edge_index.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
