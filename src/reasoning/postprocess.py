"""Decode edge probabilities into a self-consistent document tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MERGE = 0
PARENT_CHILD = 1
SIBLING = 2
NONE = 3


@dataclass(frozen=True)
class DecodedEdge:
    source: int
    target: int
    label: int
    score: float


@dataclass
class ResolvedNode:
    node_id: int
    record: dict[str, Any]
    merged_node_ids: list[int] = field(default_factory=list)
    children: list["ResolvedNode"] = field(default_factory=list)
    sibling_after: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        parts = [str(self.record.get("text") or self.record.get("text_for_embedding") or self.record.get("text_preview") or "").strip()]
        for record in self.record.get("merged_records", []):
            text = str(record.get("text") or record.get("text_for_embedding") or record.get("text_preview") or "").strip()
            if text:
                parts.append(text)
        return merge_text_fragments(parts)


def greedy_decode_relations(
    edge_index: Any,
    scores: Any,
    *,
    threshold: float = 0.5,
    num_nodes: int | None = None,
) -> list[DecodedEdge]:
    """Greedily select non-conflicting Merge/Parent/Sibling edges.

    `scores` may be logits or probabilities with shape [E, 4]. The decoder
    removes cycles from parent-child edges and keeps only one structural label
    for each unordered pair.
    """

    import torch
    import torch.nn.functional as F

    if scores.numel() == 0:
        return []
    probs = scores.detach().cpu()
    if probs.ndim != 2 or probs.shape[1] < 4:
        raise ValueError("Expected edge scores with shape [num_edges, 4]")
    row_sums = probs.sum(dim=1)
    if not torch.all((row_sums > 0.99) & (row_sums < 1.01)):
        probs = F.softmax(probs, dim=-1)
    edge_index = edge_index.detach().cpu()
    node_count = num_nodes or int(edge_index.max().item() + 1)

    candidates = []
    for edge_pos in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        label = int(probs[edge_pos].argmax().item())
        score = float(probs[edge_pos, label].item())
        if label == NONE or score < threshold or source == target:
            continue
        candidates.append(DecodedEdge(source=source, target=target, label=label, score=score))
    candidates.sort(key=lambda edge: edge.score, reverse=True)

    union_find = UnionFind(node_count)
    selected: list[DecodedEdge] = []
    selected_pairs: set[tuple[int, int]] = set()
    parent_of: dict[int, int] = {}
    parent_edges: set[tuple[int, int]] = set()

    for edge in candidates:
        pair = tuple(sorted((edge.source, edge.target)))
        if edge.label == MERGE:
            if union_find.find(edge.source) == union_find.find(edge.target):
                continue
            union_find.union(edge.source, edge.target)
            selected.append(edge)
            selected_pairs.add(pair)

    for edge in candidates:
        if edge.label == MERGE:
            continue
        source = union_find.find(edge.source)
        target = union_find.find(edge.target)
        if source == target:
            continue
        pair = tuple(sorted((source, target)))
        if pair in selected_pairs:
            continue
        if edge.label == PARENT_CHILD:
            if target in parent_of:
                continue
            if creates_cycle(source, target, parent_edges):
                continue
            parent_of[target] = source
            parent_edges.add((source, target))
        selected.append(DecodedEdge(source=source, target=target, label=edge.label, score=edge.score))
        selected_pairs.add(pair)

    return selected


def build_resolved_tree(node_records: list[dict[str, Any]], decoded_edges: list[DecodedEdge]) -> ResolvedNode:
    """Convert decoded graph edges into a ROOT-backed tree of ResolvedNode objects."""

    union_find = UnionFind(len(node_records))
    for edge in decoded_edges:
        if edge.label == MERGE:
            union_find.union(edge.source, edge.target)

    groups: dict[int, list[int]] = {}
    for idx in range(len(node_records)):
        groups.setdefault(union_find.find(idx), []).append(idx)

    nodes: dict[int, ResolvedNode] = {}
    old_to_group: dict[int, int] = {}
    for group_id, members in groups.items():
        members = sorted(members)
        primary = members[0]
        record = dict(node_records[primary])
        record["merged_records"] = [node_records[idx] for idx in members[1:]]
        resolved = ResolvedNode(node_id=group_id, record=record, merged_node_ids=members)
        nodes[group_id] = resolved
        for member in members:
            old_to_group[member] = group_id

    parent_of: dict[int, int] = {}
    sibling_after: dict[int, list[int]] = {}
    for edge in decoded_edges:
        source = old_to_group.get(edge.source, edge.source)
        target = old_to_group.get(edge.target, edge.target)
        if source == target:
            continue
        if edge.label == PARENT_CHILD:
            if target not in parent_of and not creates_cycle(source, target, {(p, c) for c, p in parent_of.items()}):
                parent_of[target] = source
        elif edge.label == SIBLING:
            sibling_after.setdefault(source, []).append(target)

    for child, parent in parent_of.items():
        if parent in nodes and child in nodes:
            nodes[parent].children.append(nodes[child])
    for source, targets in sibling_after.items():
        if source in nodes:
            nodes[source].sibling_after.extend(targets)

    root = ResolvedNode(node_id=-1, record={"type": "root", "text": "ROOT"}, merged_node_ids=[])
    for node_id in sorted(nodes, key=lambda idx: min(nodes[idx].merged_node_ids or [idx])):
        if node_id not in parent_of:
            root.children.append(nodes[node_id])
    sort_tree_children(root)
    return root


def sort_tree_children(node: ResolvedNode) -> None:
    node.children.sort(key=lambda child: min(child.merged_node_ids or [child.node_id]))
    for child in node.children:
        sort_tree_children(child)


def creates_cycle(parent: int, child: int, edges: set[tuple[int, int]]) -> bool:
    stack = [child]
    seen = set()
    while stack:
        current = stack.pop()
        if current == parent:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(target for source, target in edges if source == current)
    return False


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def merge_text_fragments(parts: list[str]) -> str:
    text = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not text:
            text = part
        elif text.endswith("-") and part[:1].islower():
            text = text[:-1] + part
        else:
            text += " " + part
    return " ".join(text.split())
