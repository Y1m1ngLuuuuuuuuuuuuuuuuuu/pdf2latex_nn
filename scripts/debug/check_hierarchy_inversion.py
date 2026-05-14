#!/usr/bin/env python3
"""Diagnose physically reversed GNN Parent-Child predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402
from src.reasoning.postprocess import PARENT_CHILD, node_physical_index  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-json", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--logits", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu", help="Torch device. Default: cpu")
    parser.add_argument("--max-text", type=int, default=120)
    parser.add_argument("--parent-threshold", type=float, default=0.0)
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
    data = torch.load(args.graph, map_location=args.device, weights_only=False)
    records = load_node_records(args.content_json)
    if len(records) != int(data.num_nodes):
        raise ValueError(f"content records ({len(records)}) do not match graph.num_nodes ({int(data.num_nodes)})")

    logits = load_logits(args, data)
    probs = scores_to_probabilities(logits)
    labels = probs.argmax(dim=-1)
    edge_index = data.edge_index.detach().cpu()

    inversions = []
    parent_count = 0
    for edge_pos in range(edge_index.shape[1]):
        if int(labels[edge_pos].item()) != PARENT_CHILD:
            continue
        score = float(probs[edge_pos, PARENT_CHILD].item())
        if score < args.parent_threshold:
            continue
        parent_count += 1
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        source_index = node_physical_index(records[source])
        target_index = node_physical_index(records[target])
        if source_index is None or target_index is None or source_index <= target_index:
            continue
        inversions.append((edge_pos, source, target, source_index, target_index, score))

    print(f"graph={args.graph}")
    print(f"content_json={args.content_json}")
    print(f"nodes={len(records)} edges={int(edge_index.shape[1])}")
    print(f"predicted_parent_edges={parent_count}")
    print(f"inverted_parent_edges={len(inversions)}")
    print()

    for edge_pos, source, target, source_index, target_index, score in inversions:
        source_record = records[source]
        target_record = records[target]
        print(
            f"[edge {edge_pos}] score={score:.4f} "
            f"parent={source} physical={source_index:g} -> child={target} physical={target_index:g}"
        )
        print(f"  parent type={record_type(source_record)} text={preview_text(source_record, args.max_text)}")
        print(f"  child  type={record_type(target_record)} text={preview_text(target_record, args.max_text)}")
        print()
    return 0


def load_node_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", payload if isinstance(payload, list) else [])
    if not isinstance(items, list):
        raise ValueError(f"Expected JSON with an items list: {path}")
    records = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        record = dict(item)
        record.setdefault("global_order", index)
        record["text"] = str(item.get("text_for_embedding") or item.get("text") or item.get("content") or "")
        records.append(record)
    return records


def load_logits(args: argparse.Namespace, data: Any) -> Any:
    import torch

    if args.logits and args.logits.exists():
        return torch.load(args.logits, map_location="cpu", weights_only=False)
    if not args.checkpoint or not args.checkpoint.exists():
        raise FileNotFoundError(f"No logits found and checkpoint is missing: {args.checkpoint}")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    model = EdgeRelationGAT(config if isinstance(config, EdgeGATConfig) else EdgeGATConfig()).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        return model(data.to(device)).detach().cpu()


def scores_to_probabilities(scores: Any) -> Any:
    import torch
    import torch.nn.functional as F

    probs = scores.detach().cpu().to(dtype=torch.float32)
    row_sums = probs.sum(dim=1) if probs.numel() else torch.empty(0)
    if probs.numel() and not (torch.all(probs >= 0.0) and torch.all((row_sums > 0.99) & (row_sums < 1.01))):
        probs = F.softmax(probs, dim=-1)
    return torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)


def record_type(record: dict[str, Any]) -> str:
    return str(record.get("canonical_type") or record.get("type") or record.get("raw_type") or "unknown")


def preview_text(record: dict[str, Any], max_len: int) -> str:
    text = " ".join(str(record.get("text") or record.get("text_for_embedding") or record.get("text_preview") or "").split())
    if len(text) <= max_len:
        return repr(text)
    return repr(text[: max_len - 3] + "...")


if __name__ == "__main__":
    raise SystemExit(main())
