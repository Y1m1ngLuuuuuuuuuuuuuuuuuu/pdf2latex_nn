#!/usr/bin/env python3
"""Generate LaTeX from a graph checkpoint via TreeDecoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402
from src.reasoning.postprocess import TreeDecoder, TreeDecoderConfig  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="Input PyG graph .pt")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint .pth/.pt")
    parser.add_argument("--output-tex", type=Path, required=True, help="Generated .tex path")
    parser.add_argument("--content-json", type=Path, help="Optional content_v4_styles.json with full node text")
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    model = EdgeRelationGAT(config if isinstance(config, EdgeGATConfig) else EdgeGATConfig()).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
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
    tex = decoder.render_document(root, title=args.title)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(tex, encoding="utf-8")
    pred_counts = torch.bincount(logits.argmax(dim=-1), minlength=4).tolist()
    print(f"wrote {args.output_tex}")
    print(f"nodes={len(node_records)} edges={int(data.edge_index.shape[1])}")
    print(f"predicted_argmax_counts={{0: {pred_counts[0]}, 1: {pred_counts[1]}, 2: {pred_counts[2]}, 3: {pred_counts[3]}}}")
    return 0


def load_node_records(content_json: Path | None, data: Any) -> list[dict[str, Any]]:
    if content_json is not None:
        payload = json.loads(content_json.read_text(encoding="utf-8"))
        items = payload.get("items", payload if isinstance(payload, list) else [])
        if not isinstance(items, list):
            raise ValueError(f"Expected content JSON with an items list: {content_json}")
        records = [record_from_content_item(item) for item in items if isinstance(item, dict)]
        if len(records) != int(data.num_nodes):
            raise ValueError(f"content records ({len(records)}) do not match graph.num_nodes ({int(data.num_nodes)})")
        return records
    node_records = list(getattr(data, "node_records", []))
    if node_records:
        return [dict(record) for record in node_records]
    return [{} for _ in range(int(data.num_nodes))]


def record_from_content_item(item: dict[str, Any]) -> dict[str, Any]:
    record = dict(item)
    text = item.get("text_for_embedding") or item.get("text") or item.get("content") or item.get("latex") or ""
    record["text"] = str(text)
    if "canonical_type" not in record:
        record["canonical_type"] = canonical_content_type(item.get("type") or item.get("raw_type"))
    return record


def canonical_content_type(value: Any) -> str:
    raw = str(value or "").lower()
    if raw in {"paragraph", "text"}:
        return "text"
    if raw in {"title", "section", "subsection", "subsubsection"}:
        return "title"
    if raw in {"equation", "equation_interline", "interline_equation", "display_formula", "formula"}:
        return "equation"
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
