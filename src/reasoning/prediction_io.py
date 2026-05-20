"""Serialization helpers for GNN edge predictions.

The GNN emits per-candidate-edge logits.  These logits are not the final render
tree: the decoder still applies merge contraction, heading-scope constraints,
and parent-edge selection before building RenderTreeIR.  This module writes a
small, auditable sidecar for the raw model output so inference runs can be
debugged without treating the GNN view as the render source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ir import PredictedRelations, RelationLabel
from src.ir.serialization import write_json
from src.ir.validators import validate_predicted_relations


LABELS = (RelationLabel.MERGE, RelationLabel.PARENT_CHILD, RelationLabel.NONE)


def write_predicted_relations(
    path: str | Path,
    *,
    doc_id: str,
    graph_path: str,
    edge_index: Any,
    scores: Any,
    threshold_config: dict[str, float] | None = None,
    model_version: str | None = None,
    include_logits: bool = False,
) -> PredictedRelations:
    """Write raw per-edge GNN predictions to ``path`` and return the payload."""

    payload = predicted_relations_from_scores(
        doc_id=doc_id,
        graph_path=graph_path,
        edge_index=edge_index,
        scores=scores,
        threshold_config=threshold_config,
        model_version=model_version,
        include_logits=include_logits,
    )
    validate_predicted_relations(payload)
    write_json(path, payload)
    return payload


def predicted_relations_from_scores(
    *,
    doc_id: str,
    graph_path: str,
    edge_index: Any,
    scores: Any,
    threshold_config: dict[str, float] | None = None,
    model_version: str | None = None,
    include_logits: bool = False,
) -> PredictedRelations:
    """Build a PredictedRelations contract from logits or probability rows."""

    probs = edge_probabilities(scores)
    labels = [label_from_index(int(index)) for index in probs.argmax(dim=-1).tolist()]
    edges = edge_pairs(edge_index)
    edge_ids = [f"e{index}:{source}->{target}" for index, (source, target) in enumerate(edges)]
    logits_payload = scores.detach().cpu().to(dtype=probs.dtype).tolist() if include_logits else []
    return PredictedRelations(
        doc_id=doc_id,
        graph_input_path=str(graph_path),
        edge_ids=edge_ids,
        predicted_labels=labels,
        probabilities=probs.tolist(),
        logits=logits_payload,
        model_version=model_version,
        threshold_config=threshold_config or {},
        metadata={
            "edge_index": [[source, target] for source, target in edges],
            "label_names": [label.name for label in LABELS],
            "prediction_basis": "raw_argmax_before_decoder_constraints",
            "decoder_note": (
                "TreeDecoder consumes these probabilities with thresholds, "
                "merge contraction, heading skeleton constraints, and parent "
                "selection before RenderTreeIR is built."
            ),
        },
    )


def edge_probabilities(scores: Any) -> Any:
    """Accept logits or probabilities and return CPU float probability rows."""

    import torch
    import torch.nn.functional as F

    if scores.numel() == 0:
        return scores.detach().cpu().to(dtype=torch.float32)
    probs = scores.detach().cpu().to(dtype=torch.float32)
    if probs.ndim != 2 or int(probs.shape[1]) < 3:
        raise ValueError("Expected edge scores with shape [num_edges, >=3]")
    row_sums = probs.sum(dim=1)
    is_probability_like = torch.all(probs >= 0.0) and torch.all((row_sums > 0.99) & (row_sums < 1.01))
    if not is_probability_like:
        probs = F.softmax(probs, dim=-1)
    return torch.nan_to_num(probs[:, :3], nan=0.0, posinf=1.0, neginf=0.0)


def edge_pairs(edge_index: Any) -> list[tuple[int, int]]:
    normalized = edge_index.detach().cpu()
    if normalized.ndim != 2 or int(normalized.shape[0]) != 2:
        raise ValueError("Expected edge_index with shape [2, num_edges]")
    return [(int(normalized[0, index].item()), int(normalized[1, index].item())) for index in range(int(normalized.shape[1]))]


def label_from_index(index: int) -> RelationLabel:
    if index == int(RelationLabel.MERGE):
        return RelationLabel.MERGE
    if index == int(RelationLabel.PARENT_CHILD):
        return RelationLabel.PARENT_CHILD
    return RelationLabel.NONE
