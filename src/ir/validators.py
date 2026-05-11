"""Validation helpers for stable IR boundary payloads."""

from __future__ import annotations

from collections import Counter

from .schema import (
    DOCUMENT_IR_SCHEMA_VERSION,
    GRAPH_INPUT_SCHEMA_VERSION,
    GRAPH_LABELS_SCHEMA_VERSION,
    PREDICTED_RELATIONS_SCHEMA_VERSION,
    RENDER_TREE_SCHEMA_VERSION,
    STYLE_PROFILE_SCHEMA_VERSION,
    BBox,
    ContractError,
    DocumentIR,
    GraphInput,
    GraphLabels,
    PredictedRelations,
    RelationLabel,
    RenderTreeIR,
    StyleProfile,
)


def validate_bbox(bbox: BBox) -> None:
    if bbox.x0 > bbox.x1 or bbox.y0 > bbox.y1:
        raise ContractError(f"Invalid bbox bounds: {bbox}")


def validate_document_ir(document: DocumentIR) -> None:
    if document.schema_version != DOCUMENT_IR_SCHEMA_VERSION:
        raise ContractError(f"Expected {DOCUMENT_IR_SCHEMA_VERSION}, got {document.schema_version!r}")
    if not document.doc_id:
        raise ContractError("DocumentIR.doc_id is required")

    page_indexes = [page.page_idx for page in document.pages]
    if page_indexes != list(range(len(document.pages))):
        raise ContractError(f"Page indexes must be 0-based and contiguous, got {page_indexes[:10]}")

    node_ids = [node.node_id for node in document.nodes]
    _assert_unique(node_ids, "DocumentIR.nodes.node_id")
    node_id_set = set(node_ids)

    reading_order = document.reading_order or node_ids
    if set(reading_order) != node_id_set:
        missing = sorted(node_id_set.difference(reading_order))[:10]
        extra = sorted(set(reading_order).difference(node_id_set))[:10]
        raise ContractError(f"reading_order must cover all nodes exactly once; missing={missing} extra={extra}")

    for node in document.nodes:
        if not node.node_id:
            raise ContractError("DocumentNode.node_id is required")
        if node.page_idx not in page_indexes:
            raise ContractError(f"Node {node.node_id} references invalid page_idx={node.page_idx}")
        if node.reading_index < 0:
            raise ContractError(f"Node {node.node_id} has negative reading_index={node.reading_index}")
        if not node.bboxes:
            raise ContractError(f"Node {node.node_id} must keep at least one bbox")
        for bbox in node.bboxes:
            validate_bbox(bbox)

    for page in document.pages:
        for node_id in page.node_ids:
            if node_id not in node_id_set:
                raise ContractError(f"Page {page.page_idx} references unknown node_id={node_id}")


def validate_graph_input(graph: GraphInput) -> None:
    if graph.schema_version != GRAPH_INPUT_SCHEMA_VERSION:
        raise ContractError(f"Expected {GRAPH_INPUT_SCHEMA_VERSION}, got {graph.schema_version!r}")
    if not graph.node_ids:
        raise ContractError("GraphInput.node_ids must not be empty")
    if not graph.edge_ids:
        raise ContractError("GraphInput.edge_ids must not be empty")
    _assert_unique(graph.node_ids, "GraphInput.node_ids")
    _assert_unique(graph.edge_ids, "GraphInput.edge_ids")
    if graph.edge_index.shape[:1] != [2]:
        raise ContractError(f"edge_index first dimension must be 2, got {graph.edge_index.shape}")
    if graph.edge_index.shape[-1] != len(graph.edge_ids):
        raise ContractError("edge_index edge dimension must match edge_ids length")
    if graph.x.shape[:1] != [len(graph.node_ids)]:
        raise ContractError("x first dimension must match node_ids length")
    if graph.edge_attr.shape[:1] != [len(graph.edge_ids)]:
        raise ContractError("edge_attr first dimension must match edge_ids length")


def validate_graph_labels(labels: GraphLabels) -> None:
    if labels.schema_version != GRAPH_LABELS_SCHEMA_VERSION:
        raise ContractError(f"Expected {GRAPH_LABELS_SCHEMA_VERSION}, got {labels.schema_version!r}")
    if len(labels.edge_ids) != len(labels.y):
        raise ContractError("GraphLabels.edge_ids and y must have identical length")
    _assert_unique(labels.edge_ids, "GraphLabels.edge_ids")
    for label in labels.y:
        if not isinstance(label, RelationLabel):
            raise ContractError(f"Unknown relation label: {label!r}")


def validate_predicted_relations(predictions: PredictedRelations) -> None:
    if predictions.schema_version != PREDICTED_RELATIONS_SCHEMA_VERSION:
        raise ContractError(f"Expected {PREDICTED_RELATIONS_SCHEMA_VERSION}, got {predictions.schema_version!r}")
    if len(predictions.edge_ids) != len(predictions.predicted_labels):
        raise ContractError("PredictedRelations.edge_ids and predicted_labels must have identical length")
    _assert_unique(predictions.edge_ids, "PredictedRelations.edge_ids")
    if predictions.probabilities and len(predictions.probabilities) != len(predictions.edge_ids):
        raise ContractError("probabilities length must match edge_ids length")
    if predictions.logits and len(predictions.logits) != len(predictions.edge_ids):
        raise ContractError("logits length must match edge_ids length")
    for label in predictions.predicted_labels:
        if not isinstance(label, RelationLabel):
            raise ContractError(f"Unknown predicted relation label: {label!r}")


def validate_render_tree(tree: RenderTreeIR) -> None:
    if tree.schema_version != RENDER_TREE_SCHEMA_VERSION:
        raise ContractError(f"Expected {RENDER_TREE_SCHEMA_VERSION}, got {tree.schema_version!r}")
    node_ids = [node.render_id for node in tree.nodes]
    _assert_unique(node_ids, "RenderTreeIR.nodes.render_id")
    node_id_set = set(node_ids)
    if tree.root_id not in node_id_set:
        raise ContractError(f"root_id={tree.root_id!r} is not present in render nodes")
    for node in tree.nodes:
        for child_id in node.children:
            if child_id not in node_id_set:
                raise ContractError(f"Render node {node.render_id} references unknown child {child_id}")
    _assert_acyclic(tree.root_id, {node.render_id: node.children for node in tree.nodes})


def validate_style_profile(profile: StyleProfile) -> None:
    if profile.schema_version != STYLE_PROFILE_SCHEMA_VERSION:
        raise ContractError(f"Expected {STYLE_PROFILE_SCHEMA_VERSION}, got {profile.schema_version!r}")
    if not profile.profile_id:
        raise ContractError("StyleProfile.profile_id is required")
    if not profile.documentclass:
        raise ContractError("StyleProfile.documentclass is required")


def _assert_unique(values: list[str], field_name: str) -> None:
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ContractError(f"{field_name} contains duplicates: {duplicates[:10]}")


def _assert_acyclic(root_id: str, children_by_id: dict[str, list[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ContractError(f"Render tree has a cycle at {node_id}")
        if node_id in visited:
            return
        visiting.add(node_id)
        for child_id in children_by_id.get(node_id, []):
            visit(child_id)
        visiting.remove(node_id)
        visited.add(node_id)

    visit(root_id)
