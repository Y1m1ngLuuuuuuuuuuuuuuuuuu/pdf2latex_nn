"""Label structural relations between path-encoded TeX nodes."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from src.reasoning.tex_ast_builder import tex_nodes_by_id


class TexRelationLabel(IntEnum):
    MERGE = 0
    PARENT_CHILD = 1
    NONE = 2
    SIBLING = 2  # Deprecated compatibility alias: sibling is now a derived postprocess relation.


def label_tex_relation(
    tex_id_a: str | None,
    tex_id_b: str | None,
    ast_nodes: dict[str, dict[str, Any]] | list[dict[str, Any]] | dict[str, Any],
    *,
    adjacent_siblings_only: bool = True,
    directed_parent_child: bool = False,
) -> TexRelationLabel:
    """Return Merge/Parent-Child/None using path-encoded TeX nodes.

    Sibling is intentionally not a supervised GNN class anymore. Nodes that
    share a parent but require no structural action are labeled NONE; sibling
    order is derived later from reading order inside the resolved tree.
    """

    if not tex_id_a or not tex_id_b:
        return TexRelationLabel.NONE

    nodes = tex_nodes_by_id(ast_nodes) if not _looks_like_node_index(ast_nodes) else ast_nodes
    node_a = nodes.get(tex_id_a)  # type: ignore[union-attr]
    node_b = nodes.get(tex_id_b)  # type: ignore[union-attr]
    if node_a is None or node_b is None:
        return TexRelationLabel.NONE

    if tex_id_a == tex_id_b:
        return TexRelationLabel.MERGE

    path_a = tuple(node_a.get("path") or ())
    path_b = tuple(node_b.get("path") or ())
    if not path_a or not path_b:
        return TexRelationLabel.NONE

    if path_a == path_b[:-1]:
        return TexRelationLabel.PARENT_CHILD
    if not directed_parent_child and path_b == path_a[:-1]:
        return TexRelationLabel.PARENT_CHILD

    return TexRelationLabel.NONE


def label_pdf_edge(
    source_pdf_id: str,
    target_pdf_id: str,
    pdf_to_tex: dict[str, str],
    ast_nodes: dict[str, dict[str, Any]] | list[dict[str, Any]] | dict[str, Any],
    *,
    adjacent_siblings_only: bool = True,
    directed_parent_child: bool = False,
) -> TexRelationLabel:
    return label_tex_relation(
        pdf_to_tex.get(source_pdf_id),
        pdf_to_tex.get(target_pdf_id),
        ast_nodes,
        adjacent_siblings_only=adjacent_siblings_only,
        directed_parent_child=directed_parent_child,
    )


def label_pdf_edges(
    edge_pairs: list[tuple[str, str]],
    pdf_to_tex: dict[str, str],
    ast_nodes: dict[str, dict[str, Any]] | list[dict[str, Any]] | dict[str, Any],
    *,
    adjacent_siblings_only: bool = True,
    directed_parent_child: bool = False,
) -> list[int]:
    return [
        int(
            label_pdf_edge(
                source_pdf_id,
                target_pdf_id,
                pdf_to_tex,
                ast_nodes,
                adjacent_siblings_only=adjacent_siblings_only,
                directed_parent_child=directed_parent_child,
            )
        )
        for source_pdf_id, target_pdf_id in edge_pairs
    ]


def _looks_like_node_index(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(node, dict) for node in value.values())
