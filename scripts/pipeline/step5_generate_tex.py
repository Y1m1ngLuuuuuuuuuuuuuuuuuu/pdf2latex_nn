#!/usr/bin/env python3
"""Generate LaTeX from a graph checkpoint through the canonical IR renderer."""

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

from src.adapters.mineru_v7_document_ir import stable_node_id  # noqa: E402
from src.perception.gnn_view_adapter import GNNViewAdapterConfig, build_gnn_view  # noqa: E402
from src.perception.reading_order import filter_graph_content_items, fuse_micro_nodes  # noqa: E402
from src.pipeline.v7_contract import assert_v7_content_json, assert_v7_graph_data  # noqa: E402
from src.generation.table_assets import annotate_table_group_records  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402
from src.reasoning.postprocess import TreeDecoder, TreeDecoderConfig  # noqa: E402
from src.reasoning.prediction_io import write_predicted_relations  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="Input PyG graph .pt")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint .pth/.pt")
    parser.add_argument("--output-tex", type=Path, required=True, help="Generated .tex path")
    parser.add_argument("--content-json", type=Path, help="Optional content_v7_styles.json with full node text")
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--parent-threshold", type=float, default=0.0)
    parser.add_argument("--sibling-threshold", type=float, default=0.5)
    parser.add_argument(
        "--heading-skeleton-mode",
        choices=["stack"],
        default="stack",
        help="Canonical decoder mode. Only stack is supported.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--logits-output", type=Path, help="Optional tensor path for raw edge logits")
    parser.add_argument(
        "--predicted-relations-output",
        type=Path,
        help="Optional JSON sidecar for raw per-edge GNN predictions. Defaults next to --output-tex.",
    )
    parser.add_argument("--source-pdf", type=Path, help="Optional source PDF used for table/figure crops")
    parser.add_argument("--source-tex", type=Path, help="Optional source TeX used for citation/float style sidecars")
    parser.add_argument("--asset-dir", type=Path, help="Directory for generated table/figure crop assets")
    parser.add_argument("--asset-latex-prefix", default="assets", help="LaTeX path prefix for generated assets")
    parser.add_argument(
        "--renderer",
        choices=["ir"],
        default="ir",
        help="Production renderer. Only the full-v7 IR surface is supported.",
    )
    parser.add_argument(
        "--render-crops",
        dest="render_table_crops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate visual crop images for tables and figures from "
            "--source-pdf or the v7 content JSON source PDF. Enabled by "
            "default; use --no-render-crops to disable."
        ),
    )
    parser.add_argument(
        "--render-table-crops",
        dest="render_table_crops",
        action=argparse.BooleanOptionalAction,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.graph, map_location=device, weights_only=False)
    assert_v7_graph_data(data, args.graph)
    if args.content_json is not None:
        assert_v7_content_json(args.content_json, require_styles=True)
    resolved_source_pdf = args.source_pdf or (source_pdf_from_content_json(args.content_json) if args.content_json else None)
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
    predicted_relations_path = args.predicted_relations_output or (args.output_tex.parent / "predicted_relations.json")
    write_predicted_relations(
        predicted_relations_path,
        doc_id=document_id_from_content_json(args.content_json, fallback=args.output_tex.stem) if args.content_json else args.output_tex.stem,
        graph_path=str(args.graph),
        edge_index=data.edge_index.detach().cpu(),
        scores=logits,
        threshold_config={
            "merge": float(args.merge_threshold),
            "parent_child": float(args.parent_threshold),
            "sibling": float(args.sibling_threshold),
        },
        model_version=str(args.checkpoint),
        include_logits=False,
    )

    node_records = load_node_records(args.content_json, data)
    decoder = TreeDecoder(
        TreeDecoderConfig(
            merge_threshold=args.merge_threshold,
            parent_threshold=args.parent_threshold,
            sibling_threshold=args.sibling_threshold,
            heading_skeleton_mode=args.heading_skeleton_mode,
            source_pdf=str(resolved_source_pdf) if args.render_table_crops and resolved_source_pdf else None,
            table_asset_output_dir=(
                str(args.asset_dir or (args.output_tex.parent / "assets"))
                if args.render_table_crops and resolved_source_pdf
                else None
            ),
            figure_asset_output_dir=(
                str(args.asset_dir or (args.output_tex.parent / "assets"))
                if args.render_table_crops and resolved_source_pdf
                else None
            ),
            table_asset_latex_prefix=args.asset_latex_prefix,
            figure_asset_latex_prefix=args.asset_latex_prefix,
        )
    )
    root = decoder.decode(node_records, data.edge_index.detach().cpu(), logits)
    document_title = args.title or infer_document_title(node_records)
    if args.content_json is None:
        raise ValueError("--renderer ir requires --content-json so graph predictions can map back to the full v7 IR")
    from scripts.pipeline.batch_visual_qa_inference import render_decoded_tree_with_ir_backend  # noqa: PLC0415

    tex = render_decoded_tree_with_ir_backend(
        root,
        node_records=node_records,
        content_json=args.content_json,
        pdf_path=resolved_source_pdf,
        source_tex_path=args.source_tex,
        document_id=document_id_from_content_json(args.content_json, fallback=args.output_tex.stem),
        title=document_title,
        document_metadata=getattr(data, "document_metadata", None),
        predicted_relations_path=predicted_relations_path,
        table_asset_output_dir=(args.asset_dir or (args.output_tex.parent / "assets")) if args.render_table_crops else None,
        figure_asset_output_dir=(args.asset_dir or (args.output_tex.parent / "assets")) if args.render_table_crops else None,
        asset_latex_prefix=args.asset_latex_prefix,
        decoder_trace=decoder.last_trace,
        attribution_output_path=args.output_tex.parent / "relation_trace_report.json",
    )
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(tex, encoding="utf-8")
    pred_counts = torch.bincount(torch.clamp(logits.argmax(dim=-1), max=2), minlength=3).tolist()
    print(f"wrote {args.output_tex}")
    print(f"wrote {predicted_relations_path}")
    print(f"nodes={len(node_records)} edges={int(data.edge_index.shape[1])}")
    print(f"predicted_argmax_counts={{0: {pred_counts[0]}, 1: {pred_counts[1]}, 2: {pred_counts[2]}}}")
    return 0


def checkpoint_compatible_config(config: EdgeGATConfig, state_dict: Any) -> EdgeGATConfig:
    """Adapt model dimensions to checkpoint tensor shapes without changing weights."""

    layout_weight = state_dict.get("projector.layout.0.weight") if isinstance(state_dict, dict) else None
    checkpoint_head_weight = state_dict.get("edge_head.3.weight") if isinstance(state_dict, dict) else None
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
    if checkpoint_head_weight is not None and "edge_head.4.weight" not in state_dict and "edge_head.12.weight" not in state_dict:
        first_head_weight = state_dict.get("edge_head.0.weight")
        if first_head_weight is not None:
            config = replace(
                config,
                predictor_hidden_dims=(int(first_head_weight.shape[0]),),
                predictor_layer_norm=False,
            )
        if int(checkpoint_head_weight.shape[0]) != config.num_classes:
            config = replace(config, num_classes=int(checkpoint_head_weight.shape[0]))
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


def gaussian_extra_dim(config: EdgeGATConfig) -> int:
    mode = getattr(config, "gaussian_edge_feature_mode", "none")
    if mode in (None, "", "none"):
        return 0
    if mode == "center":
        return 1
    return 0


def effective_edge_dim(config: EdgeGATConfig) -> int:
    return int(config.edge_dim) + gaussian_extra_dim(config)


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
        return annotate_table_group_records(records)
    if graph_records:
        return annotate_table_group_records(graph_records)
    return [{} for _ in range(int(data.num_nodes))]


def select_records_for_graph(records: list[dict[str, Any]], data: Any) -> list[dict[str, Any]]:
    """Match content JSON records to graph nodes with graph-side v7 ids.

    The graph stores the exact ``gnn_idx -> v7 node id(s)`` mapping that was
    used when ``edge_index`` and logits were produced.  Inference must use that
    mapping as the source of truth; matching only by node count is unsafe
    because a reordered GNN view would silently attach predicted edges to the
    wrong full-v7 records.
    """

    expected = int(data.num_nodes)
    graph_source_ids = graph_gnn_to_v7_ids(data)
    view = build_gnn_view(
        records,
        config=GNNViewAdapterConfig(fuse_micro_nodes=bool(getattr(data, "micro_fusion_applied", False))),
    )
    if graph_source_ids:
        if len(graph_source_ids) != expected:
            raise ValueError(
                "graph gnn_to_v7_ids length does not match graph.num_nodes: "
                f"{len(graph_source_ids)} != {expected}"
            )
        fallback_view = build_gnn_view(
            records,
            config=GNNViewAdapterConfig(fuse_micro_nodes=not bool(getattr(data, "micro_fusion_applied", False))),
        )
        current_source_ids = source_ids_for_view(view.gnn_items)
        fallback_source_ids = source_ids_for_view(fallback_view.gnn_items)
        if current_source_ids != graph_source_ids and fallback_source_ids != graph_source_ids:
            # The graph-side mapping is still the source of truth for selecting
            # full-v7 records, but a rebuilt adapter view must reproduce the
            # same sequence.  Otherwise the graph was built under a different
            # adapter policy and logits/edge_index cannot be safely bridged.
            raise ValueError(
                "content JSON rebuilt GNN view does not match graph-side gnn_to_v7_ids. "
                "Refusing to render because logits/edge_index could be bridged to the wrong v7 nodes. "
                + format_gnn_mapping_mismatches(graph_source_ids, current_source_ids, fallback_source_ids)
            )
        selected = select_records_by_graph_source_ids(records, graph_source_ids)
        if len(selected) == expected:
            return selected
        raise ValueError(
            "graph gnn_to_v7_ids did not select graph.num_nodes records: "
            f"{len(selected)} != {expected}"
        )
    if len(view.gnn_items) == expected:
        return view.gnn_items
    fallback_view = build_gnn_view(
        records,
        config=GNNViewAdapterConfig(fuse_micro_nodes=not bool(getattr(data, "micro_fusion_applied", False))),
    )
    if len(fallback_view.gnn_items) == expected:
        return fallback_view.gnn_items
    filtered_records = filter_graph_content_items(records)
    if len(filtered_records) == expected:
        return filtered_records
    fused_records: list[dict[str, Any]] | None = None
    if bool(getattr(data, "micro_fusion_applied", False)):
        fused_records = fuse_micro_nodes(filtered_records if filtered_records else records)
        if len(fused_records) == expected:
            return fused_records
    if len(records) == expected:
        return records
    fused_records = fused_records if fused_records is not None else fuse_micro_nodes(filtered_records if filtered_records else records)
    if len(fused_records) == expected:
        return fused_records
    return records


def select_records_by_graph_source_ids(
    records: list[dict[str, Any]], graph_source_ids: list[list[str]]
) -> list[dict[str, Any]]:
    """Select full-v7 records in the exact graph-saved GNN node order.

    The GNN graph already stores the source v7 node id(s) for every graph node.
    That graph-side mapping is the only safe bridge from logits/edge_index back
    to the complete v7 observation layer.  Rebuilding a fresh adapter view at
    inference time is useful for diagnostics, but it must not define order.
    """

    indexed: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, record in enumerate(records):
        node_id = str(record.get("_v7_node_id") or stable_node_id(record, fallback_position=index))
        prepared = dict(record)
        prepared.setdefault("_v7_source_index", index)
        prepared.setdefault("_v7_node_id", node_id)
        prepared.setdefault("_v7_source_indexes", [index])
        prepared.setdefault("_v7_source_node_ids", [node_id])
        indexed[node_id] = (index, prepared)

    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for source_ids in graph_source_ids:
        ids = [source_id for source_id in normalize_v7_id_list(source_ids) if source_id]
        source_records: list[tuple[int, dict[str, Any]]] = []
        for source_id in ids:
            match = indexed.get(source_id)
            if match is None:
                missing.append(source_id)
                continue
            source_records.append(match)
        if source_records:
            selected.append(combine_graph_source_records(source_records, ids))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"graph gnn_to_v7_ids references missing content records: {preview}")
    return selected


def combine_graph_source_records(source_records: list[tuple[int, dict[str, Any]]], source_ids: list[str]) -> dict[str, Any]:
    if len(source_records) == 1:
        index, record = source_records[0]
        combined = dict(record)
        combined["_v7_source_index"] = index
        combined["_v7_node_id"] = source_ids[0] if source_ids else str(record.get("_v7_node_id") or "")
        combined["_v7_source_indexes"] = [index]
        combined["_v7_source_node_ids"] = list(source_ids)
        return combined

    ordered = sorted(source_records, key=lambda pair: pair[0])
    combined = dict(ordered[0][1])
    combined["_v7_source_index"] = ordered[0][0]
    combined["_v7_node_id"] = source_ids[0]
    combined["_v7_source_indexes"] = [index for index, _ in ordered]
    combined["_v7_source_node_ids"] = list(source_ids)
    combined["merged_records"] = [dict(record) for _, record in ordered[1:]]
    combined["source_node_ids"] = [index for index, _ in ordered]

    text_parts = [record_text(record) for _, record in ordered]
    text = join_graph_source_text(text_parts)
    if text:
        combined["text"] = text
        combined["text_for_embedding"] = text

    bbox = union_bboxes([record.get("bbox") for _, record in ordered])
    if bbox is not None:
        combined["bbox"] = bbox
    return combined


def record_text(record: dict[str, Any]) -> str:
    for key in ("text", "text_for_embedding", "content", "latex"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def join_graph_source_text(parts: list[str]) -> str:
    text = ""
    for part in parts:
        if not part:
            continue
        if not text:
            text = part
            continue
        if text.endswith("-"):
            text = text[:-1] + part.lstrip()
        elif re.match(r"^[,.;:!?%)\]}]", part):
            text += part
        else:
            text += " " + part
    return text.strip()


def union_bboxes(values: list[Any]) -> list[float] | None:
    bboxes: list[list[float]] = []
    for value in values:
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                bboxes.append([float(value[0]), float(value[1]), float(value[2]), float(value[3])])
            except (TypeError, ValueError):
                continue
    if not bboxes:
        return None
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def graph_gnn_to_v7_ids(data: Any) -> list[list[str]]:
    values = getattr(data, "gnn_to_v7_ids", None)
    if values:
        normalized = [normalize_v7_id_list(value) for value in list(values)]
        if any(normalized):
            return normalized
    primary_values = getattr(data, "gnn_to_v7_id", None)
    if primary_values:
        return [[str(value)] for value in list(primary_values) if str(value)]
    return []


def source_ids_for_view(items: list[dict[str, Any]]) -> list[list[str]]:
    return [normalize_v7_id_list(item.get("_v7_source_node_ids") or item.get("_v7_node_id")) for item in items]


def normalize_v7_id_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item:
                result.append(item)
        return list(dict.fromkeys(result))
    return []


def format_gnn_mapping_mismatches(
    expected: list[list[str]],
    current: list[list[str]],
    fallback: list[list[str]],
    *,
    limit: int = 8,
) -> str:
    mismatches: list[str] = []
    max_len = max(len(expected), len(current), len(fallback))
    for index in range(max_len):
        exp = expected[index] if index < len(expected) else ["<missing>"]
        cur = current[index] if index < len(current) else ["<missing>"]
        alt = fallback[index] if index < len(fallback) else ["<missing>"]
        if exp == cur or exp == alt:
            continue
        mismatches.append(f"idx={index}: graph={exp} current={cur} fallback={alt}")
        if len(mismatches) >= limit:
            break
    suffix = f"; showing {len(mismatches)} mismatch(es)" if mismatches else "; length mismatch only"
    return suffix + (": " + " | ".join(mismatches) if mismatches else "")


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
        if key in {
            "text",
            "text_for_embedding",
            "content",
            "latex",
            "html",
            "reference_items",
            "merged_text",
            "merged_records",
            "source_node_ids",
        }:
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


def source_pdf_from_content_json(content_json: Path) -> Path | None:
    payload = json.loads(content_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("style_source_pdf", "source_pdf", "pdf_path"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return Path(value)
    return None


def document_id_from_content_json(content_json: Path, *, fallback: str) -> str:
    payload = json.loads(content_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("document_id", "doc_id", "paper_id", "arxiv_id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return fallback


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
    if raw in {"toc", "toc_title", "toc_entry", "index", "table_of_contents"}:
        return "toc"
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
