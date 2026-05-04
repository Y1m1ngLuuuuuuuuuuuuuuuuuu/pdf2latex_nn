#!/usr/bin/env python3
"""Inject TeX-derived edge labels into a PyG document graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO_ROOT))

from src.reasoning.label_generator import AlignmentLabeler, AlignmentLabelerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-json", type=Path, required=True, help="MinerU content_v4_styles.json")
    parser.add_argument("--tex", type=Path, required=True, help="Main TeX source file")
    parser.add_argument("--graph", type=Path, required=True, help="Input graph .pt to label")
    parser.add_argument("--output", type=Path, help="Optional output graph .pt; defaults to overwriting --graph")
    parser.add_argument("--mapping-output", type=Path, help="Optional JSON dump of PDF-to-TeX fuzzy matches")
    parser.add_argument("--similarity-threshold", type=float, default=65.0)
    parser.add_argument("--max-orphan-ratio", type=float, default=0.30)
    parser.add_argument("--abort-on-bad-alignment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AlignmentLabelerConfig(
        similarity_threshold=args.similarity_threshold,
        max_orphan_ratio=args.max_orphan_ratio,
        abort_on_bad_alignment=args.abort_on_bad_alignment,
        output_mapping_json=args.mapping_output,
    )
    labeler = AlignmentLabeler(
        content_json_path=args.content_json,
        tex_path=args.tex,
        graph_path=args.graph,
        config=config,
    )
    graph = labeler.run(output_graph_path=args.output, overwrite=args.output is None)
    counts = torch.bincount(graph.y.detach().cpu(), minlength=4).tolist()
    orphan_count = sum(1 for match in labeler.matches if not match.tex_id)
    orphan_ratio = orphan_count / max(1, len(labeler.matches))
    print(f"saved_graph={args.output or args.graph}")
    print(f"num_nodes={int(graph.num_nodes)} num_edges={int(graph.edge_index.shape[1])}")
    print(f"label_counts={{0: {counts[0]}, 1: {counts[1]}, 2: {counts[2]}, 3: {counts[3]}}}")
    print(f"orphan_count={orphan_count} orphan_ratio={orphan_ratio:.2%}")


if __name__ == "__main__":
    main()
