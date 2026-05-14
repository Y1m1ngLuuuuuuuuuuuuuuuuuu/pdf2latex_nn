"""Derive the graph-visible GNN view from the complete MinerU v7 IR.

The v7 content JSON is the complete observation layer used by generation.  It
must retain metadata, floats, annotations and true page furniture.  The GNN
needs a narrower training/inference view, but that view must keep a stable map
back to the full v7 nodes so predicted edges can be rendered against the full
document IR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.adapters.mineru_v7_document_ir import stable_node_id
from src.perception.reading_order import (
    annotate_duplicate_contained_continuations,
    fuse_micro_nodes,
    is_duplicate_shadow_record,
    is_toc_record,
)

EXCLUDED_LAYERS = {"noise_layer", "annotation_layer"}
DEFAULT_EXCLUDED_METADATA_ROLES = {
    "document_title",
    "front_matter_title",
    "author",
    "authors",
    "affiliation",
    "email",
    "date",
    "correspondence",
    "abstract",
    "abstract_title",
    "front_matter",
}


@dataclass(frozen=True)
class GNNView:
    """A graph-visible node sequence plus a reversible map to full v7 nodes."""

    gnn_items: list[dict[str, Any]]
    gnn_to_v7_index: list[int]
    gnn_to_v7_id: list[str]
    gnn_to_v7_ids: list[list[str]]
    v7_index_to_gnn_idx: dict[int, int]
    v7_id_to_gnn_idx: dict[str, int]
    excluded_items_summary: dict[str, Any]


@dataclass(frozen=True)
class GNNViewAdapterConfig:
    """Policy for building the GNN view.

    ``include_metadata`` defaults to false because front matter is rendered
    directly from full v7 IR, not learned as body structure.  Floats stay in the
    graph by default so caption/float relations remain classifiable, while
    message-passing masks prevent them from polluting text representations.
    """

    include_metadata: bool = False
    include_float: bool = True
    include_toc: bool = False
    include_annotations: bool = False
    include_noise: bool = False
    fuse_micro_nodes: bool = False


def build_gnn_view(
    full_items: list[dict[str, Any]],
    *,
    config: GNNViewAdapterConfig | None = None,
) -> GNNView:
    cfg = config or GNNViewAdapterConfig()
    normalized = _normalized_full_items(full_items)
    annotated = annotate_duplicate_contained_continuations(normalized)

    included: list[dict[str, Any]] = []
    excluded: list[tuple[int, dict[str, Any], str]] = []
    for item in annotated:
        include, reason = _include_item_in_gnn_view(item, cfg)
        if include:
            included.append(item)
        else:
            excluded.append((int(item["_v7_source_index"]), item, reason))

    gnn_items = _apply_micro_fusion(included) if cfg.fuse_micro_nodes else included
    for gnn_idx, item in enumerate(gnn_items):
        _attach_gnn_mapping_metadata(item, gnn_idx=gnn_idx, source_items=included)

    return GNNView(
        gnn_items=gnn_items,
        gnn_to_v7_index=[_primary_v7_index(item) for item in gnn_items],
        gnn_to_v7_id=[_primary_v7_id(item) for item in gnn_items],
        gnn_to_v7_ids=[_source_v7_ids(item) for item in gnn_items],
        v7_index_to_gnn_idx=_v7_index_to_gnn_idx(gnn_items),
        v7_id_to_gnn_idx=_v7_id_to_gnn_idx(gnn_items),
        excluded_items_summary=_excluded_summary(annotated, included, excluded, cfg),
    )


def _normalized_full_items(full_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(full_items):
        if not isinstance(item, dict):
            continue
        record = dict(item)
        record["_v7_source_index"] = index
        node_id = stable_node_id(record, fallback_position=index)
        record["_v7_node_id"] = node_id
        record["_v7_source_indexes"] = [index]
        record["_v7_source_node_ids"] = [node_id]
        normalized.append(record)
    return normalized


def _include_item_in_gnn_view(item: dict[str, Any], cfg: GNNViewAdapterConfig) -> tuple[bool, str]:
    layer = str(item.get("layout_layer") or "").casefold()
    role = str(item.get("layout_role") or item.get("role") or item.get("semantic_role") or "").casefold()
    raw_type = str(item.get("canonical_type") or item.get("type") or item.get("raw_type") or "").casefold()
    if is_duplicate_shadow_record(item):
        return False, "duplicate_shadow"
    if is_toc_record(item) and not cfg.include_toc:
        return False, "toc"
    if layer == "noise_layer" and not cfg.include_noise:
        return False, "noise_layer"
    if layer == "annotation_layer" and not cfg.include_annotations:
        return False, "annotation_layer"
    if layer == "metadata_layer" and not cfg.include_metadata:
        return False, f"metadata:{role or raw_type or 'unknown'}"
    if layer == "float_layer" and not cfg.include_float:
        return False, f"float:{role or raw_type or 'unknown'}"
    return True, "included"


def _apply_micro_fusion(included: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fused = fuse_micro_nodes(included)
    for item in fused:
        if not isinstance(item, dict):
            continue
        source_positions = item.get("source_node_indexes")
        if isinstance(source_positions, list) and source_positions:
            source_items = [
                included[int(position)]
                for position in source_positions
                if isinstance(position, int) and 0 <= int(position) < len(included)
            ]
            source_indexes = [
                index
                for source_item in source_items
                for index in source_item.get("_v7_source_indexes", [])
                if isinstance(index, int)
            ]
            source_ids = [
                node_id
                for source_item in source_items
                for node_id in source_item.get("_v7_source_node_ids", [])
                if isinstance(node_id, str)
            ]
            if source_indexes:
                item["_v7_source_indexes"] = list(dict.fromkeys(source_indexes))
            if source_ids:
                item["_v7_source_node_ids"] = list(dict.fromkeys(source_ids))
            if source_indexes:
                item["_v7_source_index"] = source_indexes[0]
            if source_ids:
                item["_v7_node_id"] = source_ids[0]
        elif "_v7_source_index" not in item:
            # Defensive fallback for unexpected fused records.
            item["_v7_source_index"] = len(fused)
            item["_v7_node_id"] = stable_node_id(item, fallback_position=len(fused))
            item["_v7_source_indexes"] = [int(item["_v7_source_index"])]
            item["_v7_source_node_ids"] = [str(item["_v7_node_id"])]
    return fused


def _attach_gnn_mapping_metadata(item: dict[str, Any], *, gnn_idx: int, source_items: list[dict[str, Any]]) -> None:
    item["_gnn_view_index"] = gnn_idx
    source_indexes = item.get("_v7_source_indexes")
    if not isinstance(source_indexes, list) or not source_indexes:
        source_index = int(item.get("_v7_source_index", gnn_idx))
        item["_v7_source_indexes"] = [source_index]
    source_ids = item.get("_v7_source_node_ids")
    if not isinstance(source_ids, list) or not source_ids:
        source_id = str(item.get("_v7_node_id") or stable_node_id(item, fallback_position=gnn_idx))
        item["_v7_source_node_ids"] = [source_id]
    item["_v7_source_index"] = _primary_v7_index(item)
    item["_v7_node_id"] = _primary_v7_id(item)


def _primary_v7_index(item: dict[str, Any]) -> int:
    values = item.get("_v7_source_indexes")
    if isinstance(values, list):
        for value in values:
            if isinstance(value, int):
                return value
    value = item.get("_v7_source_index")
    return int(value) if isinstance(value, int) else 0


def _primary_v7_id(item: dict[str, Any]) -> str:
    values = item.get("_v7_source_node_ids")
    if isinstance(values, list):
        for value in values:
            if isinstance(value, str) and value:
                return value
    value = item.get("_v7_node_id")
    return str(value) if value else stable_node_id(item, fallback_position=_primary_v7_index(item))


def _source_v7_ids(item: dict[str, Any]) -> list[str]:
    values = item.get("_v7_source_node_ids")
    ids = [value for value in values if isinstance(value, str) and value] if isinstance(values, list) else []
    return list(dict.fromkeys(ids or [_primary_v7_id(item)]))


def _v7_index_to_gnn_idx(items: list[dict[str, Any]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for gnn_idx, item in enumerate(items):
        values = item.get("_v7_source_indexes")
        indexes = [value for value in values if isinstance(value, int)] if isinstance(values, list) else []
        for source_index in indexes or [_primary_v7_index(item)]:
            mapping[source_index] = gnn_idx
    return mapping


def _v7_id_to_gnn_idx(items: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for gnn_idx, item in enumerate(items):
        for node_id in _source_v7_ids(item):
            mapping[node_id] = gnn_idx
    return mapping


def _excluded_summary(
    annotated: list[dict[str, Any]],
    included: list[dict[str, Any]],
    excluded: list[tuple[int, dict[str, Any], str]],
    cfg: GNNViewAdapterConfig,
) -> dict[str, Any]:
    by_reason: dict[str, int] = {}
    by_layer: dict[str, int] = {}
    by_type: dict[str, int] = {}
    sample: list[dict[str, Any]] = []
    for index, item, reason in excluded:
        by_reason[reason] = by_reason.get(reason, 0) + 1
        layer = str(item.get("layout_layer") or "unknown")
        raw_type = str(item.get("canonical_type") or item.get("type") or item.get("raw_type") or "unknown")
        by_layer[layer] = by_layer.get(layer, 0) + 1
        by_type[raw_type] = by_type.get(raw_type, 0) + 1
        if len(sample) < 12:
            sample.append(
                {
                    "v7_index": index,
                    "v7_node_id": item.get("_v7_node_id"),
                    "reason": reason,
                    "layout_layer": item.get("layout_layer"),
                    "layout_role": item.get("layout_role"),
                    "type": item.get("type"),
                    "text_preview": str(item.get("text_for_embedding") or item.get("text") or "")[:120],
                }
            )
    return {
        "schema_version": "gnn_view_adapter_v1",
        "policy": {
            "include_metadata": cfg.include_metadata,
            "include_float": cfg.include_float,
            "include_toc": cfg.include_toc,
            "include_annotations": cfg.include_annotations,
            "include_noise": cfg.include_noise,
            "fuse_micro_nodes": cfg.fuse_micro_nodes,
        },
        "full_node_count": len(annotated),
        "gnn_node_count": len(included),
        "excluded_node_count": len(excluded),
        "excluded_by_reason": by_reason,
        "excluded_by_layer": by_layer,
        "excluded_by_type": by_type,
        "excluded_sample": sample,
    }
