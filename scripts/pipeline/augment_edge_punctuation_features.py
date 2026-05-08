#!/usr/bin/env python3
"""Append lightweight edge features to existing v7 graph manifests.

This avoids rerunning MinerU/PyMuPDF/SciBERT when only edge_attr schema grows
from older cached dimensions to the current schema. The script copies each
graph, appends or refreshes:

1. source_ends_with_terminal_punctuation
2. source_ends_with_hyphen
3. same_layout_layer
4. same_layout_band
5. same_band_column
6. band_order_delta
7. crosses_band_boundary

and writes a new manifest pointing to the augmented graph copies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.reading_order import fuse_micro_nodes  # noqa: E402
from src.perception.schema import EDGE_ATTR_FIELDS  # noqa: E402
from src.reasoning.graph_builder import (  # noqa: E402
    layout_band_column_id,
    layout_band_id,
    layout_band_order,
    layout_layer_name,
    source_ends_with_hyphen,
    source_ends_with_terminal_punctuation,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--graph-output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
    records = load_records(args.input_manifest)
    args.graph_output_dir.mkdir(parents=True, exist_ok=True)
    output_records = []
    for index, record in enumerate(records, start=1):
        doc_id = str(record.get("document_id") or Path(str(record["graph_path"])).stem)
        source_graph = Path(str(record["graph_path"]))
        output_graph = (args.graph_output_dir / f"{safe_filename(doc_id)}_edge22.pt").resolve()
        if output_graph.exists() and not args.force:
            output_records.append({**record, "graph_path": str(output_graph), "source_graph_path": str(source_graph)})
            continue

        graph = torch.load(source_graph, map_location="cpu", weights_only=False)
        items = load_content_items(Path(str(record["content_json"])), graph=graph)
        graph.edge_attr = append_punctuation_features(graph, items, torch=torch)
        graph.edge_attr_schema = update_edge_attr_schema(getattr(graph, "edge_attr_schema", None))
        output_graph.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, output_graph)
        output_records.append({**record, "graph_path": str(output_graph), "source_graph_path": str(source_graph)})
        if index == 1 or index == len(records) or index % 50 == 0:
            print(f"[{index}/{len(records)}] wrote={output_graph}", flush=True)

    payload = {
        "schema_version": "v7_punctuation_augmented_manifest_v1",
        "source_manifest": str(args.input_manifest),
        "num_documents": len(output_records),
        "edge_attr_fields": list(EDGE_ATTR_FIELDS),
        "documents": output_records,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote manifest={args.output_manifest}")
    print(f"documents={len(output_records)} edge_attr_dim={len(EDGE_ATTR_FIELDS)}")
    return 0


def append_punctuation_features(graph: Any, items: list[dict[str, Any]], *, torch: Any) -> Any:
    edge_index = graph.edge_index.detach().cpu()
    punctuation_rows = []
    layout_rows = []
    for edge_pos in range(int(edge_index.shape[1])):
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        source_item = items[source] if 0 <= source < len(items) else {}
        target_item = items[target] if 0 <= target < len(items) else {}
        punctuation_rows.append(
            [
                source_ends_with_terminal_punctuation(source_item),
                source_ends_with_hyphen(source_item),
            ]
        )
        source_band = layout_band_id(source_item)
        target_band = layout_band_id(target_item)
        source_column = layout_band_column_id(source_item)
        target_column = layout_band_column_id(target_item)
        layout_rows.append(
            [
                float(layout_layer_name(source_item) == layout_layer_name(target_item)),
                float(source_band is not None and target_band is not None and source_band == target_band),
                float(source_column is not None and target_column is not None and source_column == target_column),
                max(-1.0, min(1.0, (layout_band_order(target_item) - layout_band_order(source_item)) / 10.0)),
                float(source_band is not None and target_band is not None and source_band != target_band),
            ]
        )
    punctuation = torch.tensor(punctuation_rows, dtype=torch.float32)
    layout = torch.tensor(layout_rows, dtype=torch.float32)
    edge_attr = graph.edge_attr.detach().cpu().to(dtype=torch.float32)
    current_dim = int(edge_attr.shape[1]) if edge_attr.ndim == 2 else 0
    target_dim = len(EDGE_ATTR_FIELDS)
    if current_dim == target_dim:
        edge_attr = edge_attr.clone()
        edge_attr[:, 15:17] = punctuation
        edge_attr[:, 17:22] = layout
        return edge_attr
    if current_dim == 17:
        edge_attr = edge_attr.clone()
        edge_attr[:, 15:17] = punctuation
        return torch.cat([edge_attr, layout], dim=1)
    if current_dim == 15:
        return torch.cat([edge_attr, punctuation, layout], dim=1)
    raise ValueError(f"Unsupported edge_attr dim {tuple(edge_attr.shape)}; expected 15, 17, or {target_dim}")


def update_edge_attr_schema(schema: Any) -> dict[str, Any]:
    payload = dict(schema) if isinstance(schema, dict) else {}
    payload["dim"] = len(EDGE_ATTR_FIELDS)
    payload["fields"] = list(EDGE_ATTR_FIELDS)
    return payload


def load_content_items(path: Path, *, graph: Any) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", payload if isinstance(payload, list) else [])
    if not isinstance(items, list):
        raise ValueError(f"Expected content JSON items list: {path}")
    items = [item for item in items if isinstance(item, dict)]
    expected = int(graph.num_nodes)
    if bool(getattr(graph, "micro_fusion_applied", False)):
        fused = fuse_micro_nodes(items)
        if len(fused) == expected:
            return fused
    if len(items) == expected:
        return items
    fused = fuse_micro_nodes(items)
    if len(fused) == expected:
        return fused
    raise ValueError(f"content nodes ({len(items)}) do not match graph nodes ({expected}): {path}")


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Expected manifest list or documents list: {path}")
    return [record for record in records if isinstance(record, dict) and record.get("graph_path") and record.get("content_json")]


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
