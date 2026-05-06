"""Decode edge probabilities into a self-consistent document tree."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.perception.xy_cut import sort_nodes_by_reading_order
from src.perception.title_features import strip_title_numbering, title_numbering_level


MERGE = 0
PARENT_CHILD = 1
NONE = 2
SIBLING = 2  # Deprecated compatibility alias: sibling is derived from reading order, not decoded.
VIRTUAL_ROOT = "__ROOT__"
SECTION_COMMANDS = ["section", "subsection", "subsubsection", "paragraph", "subparagraph"]
DISPLAY_MATH_ENVS = {"equation", "align", "gather", "eqnarray", "flalign", "multline"}
MERGE_COMPATIBLE_TYPES = {"text", "equation", "reference"}
DEFAULT_PREAMBLE_COMMANDS = (r"\providecommand{\mathbfcal}[1]{\mathbf{\mathcal{#1}}}",)
LIST_MARKER_RE = re.compile(r"^\s*(?P<marker>[\u2022\u25E6\u25CB\u25AA\-\*]|\d+\.|[a-zA-Z]\.)\s+")
ORDERED_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\.|[a-zA-Z]\.)\s+")
NUMERIC_ID_RE = re.compile(r"\d+")
PSEUDOCODE_START_RE = re.compile(
    r"^\s*(?:Algorithm\s*\d+\b|Input\s*:|Output\s*:|Require\s*:|Ensure\s*:)",
    re.IGNORECASE,
)
PSEUDOCODE_BREAK_RE = re.compile(
    r"\s+(?=(?:Input|Output|Require|Ensure)\s*:|Algorithm\s*\d+\b|(?:for|while|if|else|elif|return|end)\b)",
    re.IGNORECASE,
)
VERBATIM_END_RE = re.compile(r"\\end\s*\{\s*verbatim\s*\}", re.IGNORECASE)
ALGORITHM_CAPTION_RE = re.compile(r"^\s*Algorithm\s*(?:\d+)?\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
PSEUDOCODE_IO_RE = re.compile(r"^\s*(Input|Require|Output|Ensure)\s*:\s*(.*)$", re.IGNORECASE)
PSEUDOCODE_FOR_RE = re.compile(r"^\s*for\s+(.+?)(?:\s+do)?\s*$", re.IGNORECASE)
PSEUDOCODE_WHILE_RE = re.compile(r"^\s*while\s+(.+?)(?:\s+do)?\s*$", re.IGNORECASE)
PSEUDOCODE_IF_RE = re.compile(r"^\s*if\s+(.+?)(?:\s+then)?\s*$", re.IGNORECASE)
PSEUDOCODE_RETURN_RE = re.compile(r"^\s*return\s+(.+)$", re.IGNORECASE)
PSEUDOCODE_END_RE = re.compile(r"^\s*end(?:\s+(for|if|while))?\s*$", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^\s*(Table\s*\d*[:.\-]?\s*[^\n]+)", re.IGNORECASE)
LATEX_MATH_MARKER_RE = re.compile(r"(\\[A-Za-z]+|[_^{}]|[<>=+\-*/]|\\\(|\\\[)")


@dataclass(frozen=True)
class DecodedEdge:
    source: int
    target: int
    label: int
    score: float


@dataclass(frozen=True)
class TreeDecoderConfig:
    merge_threshold: float = 0.5
    parent_threshold: float = 0.0
    sibling_threshold: float = 0.5  # Deprecated no-op; kept for CLI/checkpoint compatibility.
    require_merge_argmax: bool = True
    require_parent_argmax: bool = False
    include_sibling_edges: bool = False  # Deprecated no-op; sibling order comes from reading order.
    abstract_root_weight: float = 100.0
    reference_parent_weight: float = 100.0
    enforce_parent_causality: bool = True
    text_text_parent_weight_scale: float = 0.05
    text_text_parent_weight_bias: float = -0.05
    merge_parallel_y_overlap_ratio: float = 0.10
    merge_gutter_threshold: float = 30.0
    merge_gutter_page_width_ratio: float = 0.05
    document_class: str = "article"
    packages: tuple[str, ...] = ("graphicx", "amsmath", "amssymb", "booktabs", "hyperref", "float", "algorithm", "algpseudocode")


@dataclass
class ResolvedNode:
    node_id: int
    record: dict[str, Any]
    merged_node_ids: list[int] = field(default_factory=list)
    children: list["ResolvedNode"] = field(default_factory=list)
    sibling_after: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        explicit = self.record.get("merged_text")
        if explicit:
            return str(explicit).strip()
        parts = [node_record_text(self.record).strip()]
        for record in self.record.get("merged_records", []):
            text = node_record_text(record).strip()
            if text:
                parts.append(text)
        return merge_text_fragments(parts)


@dataclass
class ContractedGraph:
    nodes: dict[int, ResolvedNode]
    old_to_super: dict[int, int]
    merge_edges: list[DecodedEdge]


@dataclass(frozen=True)
class HeadingSkeleton:
    """Physical reading-flow heading tree and section scopes."""

    heading_ids: frozenset[int]
    heading_levels: dict[int, int]
    heading_parent: dict[int, int | None]
    scope_by_node: dict[int, int | None]


class TreeDecoder:
    """Three-stage tree decoder: merge contraction, NetworkX MSA, DFS render."""

    def __init__(self, config: TreeDecoderConfig | None = None) -> None:
        self.config = config or TreeDecoderConfig()

    def decode(
        self,
        node_records: list[dict[str, Any]],
        edge_index: Any,
        scores: Any,
    ) -> ResolvedNode:
        """Decode model edge scores directly into a ROOT-backed tree."""

        probs = self.edge_probabilities(scores)
        raw_skeleton = build_heading_skeleton(
            {
                index: ResolvedNode(node_id=index, record=dict(record), merged_node_ids=[index])
                for index, record in enumerate(node_records)
            }
        )
        contracted = self.contract_merge_nodes(
            node_records,
            edge_index,
            probs,
            raw_skeleton=raw_skeleton,
        )
        contracted = self.semantic_title_deduplication(contracted)
        skeleton = build_heading_skeleton(contracted.nodes)
        parent_edges = self.maximum_parent_arborescence(contracted, edge_index, probs, skeleton=skeleton)
        return self.build_skeleton_tree(contracted, skeleton, parent_edges)

    def decode_edges(self, edge_index: Any, scores: Any, *, num_nodes: int | None = None) -> list[DecodedEdge]:
        """Return decoded edges while preserving the legacy function API."""

        probs = self.edge_probabilities(scores)
        node_count = resolve_num_nodes(edge_index, scores, num_nodes=num_nodes)
        node_records = [{"type": "text", "text": "", "_disable_domain_priors": True} for _ in range(node_count)]
        contracted = self.contract_merge_nodes(node_records, edge_index, probs)
        contracted = self.semantic_title_deduplication(contracted)
        parent_edges = self.maximum_parent_arborescence(contracted, edge_index, probs)
        return contracted.merge_edges + parent_edges

    def edge_probabilities(self, scores: Any) -> Any:
        """Accept logits or probabilities and return CPU probability rows."""

        import torch
        import torch.nn.functional as F

        if scores.numel() == 0:
            return scores.detach().cpu()
        probs = scores.detach().cpu().to(dtype=torch.float32)
        if probs.ndim != 2 or int(probs.shape[1]) < 3:
            raise ValueError("Expected edge scores with shape [num_edges, >=3]")
        row_sums = probs.sum(dim=1)
        is_probability_like = torch.all(probs >= 0.0) and torch.all((row_sums > 0.99) & (row_sums < 1.01))
        if not is_probability_like:
            probs = F.softmax(probs, dim=-1)
        return torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

    def contract_merge_nodes(
        self,
        node_records: list[dict[str, Any]],
        edge_index: Any,
        probs: Any,
        *,
        raw_skeleton: HeadingSkeleton | None = None,
    ) -> ContractedGraph:
        """Stage 1: contract high-confidence Merge components into supernodes."""

        edge_index = normalize_edge_index(edge_index)
        if int(probs.shape[0]) != int(edge_index.shape[1]):
            raise ValueError("scores rows must match edge_index edge count")
        union_find = UnionFind(len(node_records))
        merge_edges: list[DecodedEdge] = []

        for edge_pos in range(edge_index.shape[1]):
            source = int(edge_index[0, edge_pos].item())
            target = int(edge_index[1, edge_pos].item())
            if not valid_node_pair(source, target, len(node_records)) or source == target:
                continue
            if raw_skeleton is not None and merge_crosses_section_boundary(source, target, raw_skeleton):
                continue
            if merge_crosses_intermediate_list_marker(source, target, node_records):
                continue
            if not self.can_merge(node_records[source], node_records[target]):
                continue
            merge_score = float(probs[edge_pos, MERGE].item())
            label = int(probs[edge_pos].argmax().item())
            if merge_score < self.config.merge_threshold:
                continue
            if self.config.require_merge_argmax and label != MERGE:
                continue
            if union_find.find(source) != union_find.find(target):
                union_find.union(source, target)
                merge_edges.append(DecodedEdge(source=source, target=target, label=MERGE, score=merge_score))

        groups: dict[int, list[int]] = {}
        for index in range(len(node_records)):
            groups.setdefault(union_find.find(index), []).append(index)

        nodes: dict[int, ResolvedNode] = {}
        old_to_super: dict[int, int] = {}
        for root, members in groups.items():
            canonical_id = min(members)
            ordered_members = sorted(members)
            primary = ordered_members[0]
            merged_text = merge_text_fragments([node_record_text(node_records[idx]) for idx in ordered_members])
            record = dict(node_records[primary])
            record["merged_text"] = merged_text
            record["merged_records"] = [dict(node_records[idx]) for idx in ordered_members[1:]]
            record["source_node_ids"] = ordered_members
            nodes[canonical_id] = ResolvedNode(
                node_id=canonical_id,
                record=record,
                merged_node_ids=ordered_members,
            )
            for member in ordered_members:
                old_to_super[member] = canonical_id

        return ContractedGraph(nodes=nodes, old_to_super=old_to_super, merge_edges=merge_edges)

    def can_merge(self, node_u: dict[str, Any], node_v: dict[str, Any]) -> bool:
        """Physical and semantic hard gates for high-confidence Merge edges."""

        return can_contract_merge_records(
            node_u,
            node_v,
            parallel_y_overlap_ratio=self.config.merge_parallel_y_overlap_ratio,
            gutter_threshold=self.config.merge_gutter_threshold,
            gutter_page_width_ratio=self.config.merge_gutter_page_width_ratio,
        )

    def semantic_title_deduplication(self, contracted: ContractedGraph) -> ContractedGraph:
        """Alias duplicate title supernodes before NetworkX tree decoding."""

        seen_titles: dict[str, int] = {}
        alias_map: dict[int, int] = {}

        ordered_node_ids = sorted(
            contracted.nodes,
            key=lambda node_id: min(contracted.nodes[node_id].merged_node_ids or [node_id]),
        )
        for node_id in ordered_node_ids:
            node = contracted.nodes[node_id]
            if canonical_render_type(node.record) != "title":
                continue
            normalized_title = normalize_title_text_for_dedup(node.text)
            if len(normalized_title.replace(" ", "")) < 3:
                continue
            if normalized_title in seen_titles:
                alias_map[node_id] = seen_titles[normalized_title]
            else:
                seen_titles[normalized_title] = node_id

        if not alias_map:
            return contracted

        def resolve_alias(node_id: int) -> int:
            seen: set[int] = set()
            while node_id in alias_map and node_id not in seen:
                seen.add(node_id)
                node_id = alias_map[node_id]
            return node_id

        pruned_nodes = {
            node_id: node
            for node_id, node in contracted.nodes.items()
            if node_id not in alias_map
        }
        rerouted_old_to_super = {
            raw_node_id: resolve_alias(super_node_id)
            for raw_node_id, super_node_id in contracted.old_to_super.items()
        }
        rerouted_merge_edges = reroute_decoded_edges(contracted.merge_edges, alias_map, resolve_alias)
        return ContractedGraph(
            nodes=pruned_nodes,
            old_to_super=rerouted_old_to_super,
            merge_edges=rerouted_merge_edges,
        )

    def maximum_parent_arborescence(
        self,
        contracted: ContractedGraph,
        edge_index: Any,
        probs: Any,
        *,
        skeleton: HeadingSkeleton | None = None,
    ) -> list[DecodedEdge]:
        """Stage 2: decode Parent-Child relations with NetworkX MSA."""

        import networkx as nx
        from networkx.algorithms.tree.branchings import maximum_spanning_arborescence

        parent_candidates = self.collect_parent_candidate_edges(contracted, edge_index, probs)
        parent_candidates = self.enforce_reference_topology(contracted.nodes, parent_candidates)
        if self.config.enforce_parent_causality:
            parent_candidates = self.apply_causality_barrier(contracted.nodes, parent_candidates)
        if skeleton is not None:
            parent_candidates = self.apply_structure_barrier(contracted.nodes, parent_candidates, skeleton)

        graph = nx.DiGraph()
        graph.add_node(VIRTUAL_ROOT)
        for node_id in sorted(contracted.nodes):
            root_score = self.root_prior_score(contracted.nodes[node_id])
            graph.add_node(node_id)
            graph.add_edge(
                VIRTUAL_ROOT,
                node_id,
                weight=root_score,
                score=root_score,
                label=NONE,
                synthetic_root=True,
                domain_prior="abstract_root" if root_score > 0.0 else None,
            )

        for edge in parent_candidates:
            if edge.source not in contracted.nodes or edge.target not in contracted.nodes:
                continue
            if edge.source == edge.target:
                continue
            if graph.has_edge(edge.source, edge.target) and float(graph[edge.source][edge.target]["weight"]) >= edge.score:
                continue
            graph.add_edge(
                edge.source,
                edge.target,
                weight=edge.score,
                score=edge.score,
                label=PARENT_CHILD,
                synthetic_root=False,
            )

        if len(contracted.nodes) == 0:
            return []
        arborescence = maximum_spanning_arborescence(graph, attr="weight", default=0.0, preserve_attrs=True)
        decoded: list[DecodedEdge] = []
        for source, target, attrs in arborescence.edges(data=True):
            if source == VIRTUAL_ROOT or attrs.get("synthetic_root"):
                continue
            decoded.append(
                DecodedEdge(
                    source=int(source),
                    target=int(target),
                    label=PARENT_CHILD,
                    score=float(attrs.get("score", attrs.get("weight", 0.0))),
                )
            )
        decoded.sort(key=lambda edge: (min(contracted.nodes[edge.target].merged_node_ids), edge.source, edge.target))
        return decoded

    def collect_parent_candidate_edges(
        self,
        contracted: ContractedGraph,
        edge_index: Any,
        probs: Any,
    ) -> list[DecodedEdge]:
        """Collect model-scored Parent-Child candidates after contraction and priors."""

        edge_index = normalize_edge_index(edge_index)
        candidates: list[DecodedEdge] = []
        for edge_pos in range(edge_index.shape[1]):
            raw_source = int(edge_index[0, edge_pos].item())
            raw_target = int(edge_index[1, edge_pos].item())
            source = contracted.old_to_super.get(raw_source)
            target = contracted.old_to_super.get(raw_target)
            if source is None or target is None or source == target:
                continue
            raw_parent_score = float(probs[edge_pos, PARENT_CHILD].item())
            parent_score = self.parent_prior_score(contracted.nodes[source], contracted.nodes[target], raw_parent_score)
            label = int(probs[edge_pos].argmax().item())
            if parent_score < self.config.parent_threshold:
                continue
            if self.config.require_parent_argmax and label != PARENT_CHILD:
                continue
            candidates.append(DecodedEdge(source=source, target=target, label=PARENT_CHILD, score=parent_score))
        return keep_highest_scoring_edges(candidates)

    def apply_causality_barrier(
        self,
        nodes: dict[int, ResolvedNode],
        edges: list[DecodedEdge],
    ) -> list[DecodedEdge]:
        """Drop Parent-Child edges where a later block is predicted as parent of an earlier block."""

        filtered: list[DecodedEdge] = []
        for edge in edges:
            if edge.label != PARENT_CHILD:
                filtered.append(edge)
                continue
            source = nodes.get(edge.source)
            target = nodes.get(edge.target)
            if source is None or target is None:
                continue
            if violates_parent_child_causality(source, target):
                continue
            filtered.append(edge)
        return keep_highest_scoring_edges(filtered)

    def apply_structure_barrier(
        self,
        nodes: dict[int, ResolvedNode],
        edges: list[DecodedEdge],
        skeleton: HeadingSkeleton,
    ) -> list[DecodedEdge]:
        """Keep GNN parent edges inside the physical section scope only."""

        filtered: list[DecodedEdge] = []
        for edge in edges:
            if edge.label != PARENT_CHILD:
                filtered.append(edge)
                continue
            source = nodes.get(edge.source)
            target = nodes.get(edge.target)
            if source is None or target is None:
                continue
            if violates_structural_parent_child(source, target, skeleton):
                continue
            filtered.append(edge)
        return keep_highest_scoring_edges(filtered)

    def enforce_reference_topology(
        self,
        nodes: dict[int, ResolvedNode],
        edges: list[DecodedEdge],
    ) -> list[DecodedEdge]:
        """Force the tail bibliography region under its References/Bibliography title."""

        ordered_node_ids = sorted(
            nodes,
            key=lambda node_id: min(nodes[node_id].merged_node_ids or [node_id]),
        )
        ref_anchor_id: int | None = None
        ref_anchor_index: int | None = None
        for index, node_id in enumerate(ordered_node_ids):
            node = nodes[node_id]
            if canonical_render_type(node.record) != "title":
                continue
            normalized = normalize_structural_heading_text(node.text)
            if normalized in {"references", "bibliography"}:
                ref_anchor_id = node_id
                ref_anchor_index = index
                break

        if ref_anchor_id is None or ref_anchor_index is None:
            return edges

        captured_ids: set[int] = set()
        for node_id in ordered_node_ids[ref_anchor_index + 1:]:
            node = nodes[node_id]
            if is_appendix_stop_title(node):
                break
            if is_page_noise_node(node):
                continue
            captured_ids.add(node_id)

        if not captured_ids:
            return edges

        rerouted: list[DecodedEdge] = []
        for edge in edges:
            if edge.source == edge.target:
                continue
            if edge.source in captured_ids and edge.source != ref_anchor_id:
                continue
            if edge.target in captured_ids and edge.source != ref_anchor_id:
                continue
            rerouted.append(edge)

        for target in sorted(captured_ids, key=lambda node_id: min(nodes[node_id].merged_node_ids or [node_id])):
            if target == ref_anchor_id:
                continue
            rerouted.append(
                DecodedEdge(
                    source=ref_anchor_id,
                    target=target,
                    label=PARENT_CHILD,
                    score=self.config.reference_parent_weight,
                )
            )

        return keep_highest_scoring_edges(rerouted)

    def root_prior_score(self, node: ResolvedNode) -> float:
        if node.record.get("_disable_domain_priors"):
            return 0.0
        if is_abstract_root_candidate(node):
            return self.config.abstract_root_weight
        return 0.0

    def parent_prior_score(self, source: ResolvedNode, target: ResolvedNode, raw_score: float) -> float:
        if source.record.get("_disable_domain_priors") or target.record.get("_disable_domain_priors"):
            return raw_score
        if canonical_render_type(source.record) == "text" and canonical_render_type(target.record) == "text":
            return (
                raw_score * self.config.text_text_parent_weight_scale
                + self.config.text_text_parent_weight_bias
            )
        return raw_score

    def decode_sibling_edges(
        self,
        contracted: ContractedGraph,
        edge_index: Any,
        probs: Any,
        *,
        skeleton: HeadingSkeleton | None = None,
    ) -> list[DecodedEdge]:
        """Deprecated: sibling is no longer a supervised/decoded edge class."""

        return []

    def build_skeleton_tree(
        self,
        contracted: ContractedGraph,
        skeleton: HeadingSkeleton,
        decoded_edges: list[DecodedEdge],
    ) -> ResolvedNode:
        """Attach headings by the physical stack, then attach flesh locally."""

        nodes = {node_id: clone_resolved_node(node) for node_id, node in contracted.nodes.items()}
        parent_of: dict[int, int] = {}

        for heading_id, parent_id in skeleton.heading_parent.items():
            if heading_id in nodes and parent_id in nodes:
                parent_of[heading_id] = int(parent_id)

        best_parent_edge: dict[int, DecodedEdge] = {}
        for edge in decoded_edges:
            if edge.source == edge.target:
                continue
            if edge.label != PARENT_CHILD or edge.source not in nodes or edge.target not in nodes:
                continue
            source = nodes[edge.source]
            target = nodes[edge.target]
            if violates_structural_parent_child(source, target, skeleton):
                continue
            if edge.target not in best_parent_edge or edge.score > best_parent_edge[edge.target].score:
                best_parent_edge[edge.target] = edge

        for target_id, edge in best_parent_edge.items():
            if target_id in skeleton.heading_ids:
                continue
            parent_of[target_id] = edge.source

        for node_id in nodes:
            if node_id in parent_of or node_id in skeleton.heading_ids:
                continue
            if is_page_noise_node(nodes[node_id]):
                continue
            scope_id = skeleton.scope_by_node.get(node_id)
            if scope_id in nodes:
                parent_of[node_id] = int(scope_id)

        for child_id, parent_id in parent_of.items():
            if child_id in nodes and parent_id in nodes and child_id != parent_id:
                nodes[parent_id].children.append(nodes[child_id])

        root = ResolvedNode(node_id=-1, record={"type": "root", "text": "ROOT"}, merged_node_ids=[])
        for node_id in sorted(nodes, key=lambda idx: node_reading_order_key(nodes[idx])):
            if node_id not in parent_of:
                root.children.append(nodes[node_id])
        sort_tree_children(root)
        return root

    def build_tree(self, contracted: ContractedGraph, decoded_edges: list[DecodedEdge]) -> ResolvedNode:
        """Attach contracted nodes under the arborescence, adding virtual ROOT."""

        nodes = {node_id: clone_resolved_node(node) for node_id, node in contracted.nodes.items()}
        parent_of: dict[int, int] = {}

        for edge in decoded_edges:
            if edge.source == edge.target:
                continue
            if edge.label == PARENT_CHILD and edge.source in nodes and edge.target in nodes:
                parent_of[edge.target] = edge.source

        for child_id, parent_id in parent_of.items():
            nodes[parent_id].children.append(nodes[child_id])

        root = ResolvedNode(node_id=-1, record={"type": "root", "text": "ROOT"}, merged_node_ids=[])
        for node_id in sorted(nodes, key=lambda idx: min(nodes[idx].merged_node_ids or [idx])):
            if node_id not in parent_of:
                root.children.append(nodes[node_id])
        sort_tree_children(root)
        return root

    def render_document(self, root: ResolvedNode, *, title: str | None = None) -> str:
        """Stage 3: render a resolved tree to a compilable LaTeX document."""

        body_root = root_without_redundant_document_title(root, title) if title else root
        lines = [rf"\documentclass{{{self.config.document_class}}}"]
        for package in self.config.packages:
            lines.append(rf"\usepackage{{{package}}}")
        lines.extend(DEFAULT_PREAMBLE_COMMANDS)
        lines.append("")
        if title:
            lines.extend([rf"\title{{{escape_latex(title)}}}", r"\date{}", ""])
        lines.append(r"\begin{document}")
        if title:
            lines.append(r"\maketitle")
            lines.append("")
        body = self.render_node(body_root, depth=0, is_root=True).strip()
        if body:
            lines.append(body)
            lines.append("")
        lines.append(r"\end{document}")
        return "\n".join(lines).rstrip() + "\n"

    def render_node(self, node: ResolvedNode, *, depth: int = 0, is_root: bool = False) -> str:
        if is_root:
            return "\n\n".join(
                rendered
                for rendered in self.render_child_blocks_with_dynamic_lists(node.children, depth=0)
                if rendered
            )

        block_type = canonical_render_type(node.record)
        text = node.text
        children = sorted_render_children(node.children)
        if is_algorithm_like_node(node.record, node_verbatim_text(node)):
            return render_algorithm_block(node_verbatim_text(node))
        if block_type == "title":
            parts = [render_title(text, depth=depth)] if text else []
            parts.extend(self.render_child_blocks_with_dynamic_lists(children, depth=depth + 1))
            return "\n\n".join(part for part in parts if part)
        if block_type == "equation":
            return render_equation(text)
        if block_type == "inline_math":
            return render_inline_math(text)
        if block_type == "table":
            return render_table_placeholder(node.record, node_verbatim_text(node), node_id=node.node_id)
        if block_type == "figure":
            caption = escape_latex(text) if text else "Figure"
            return "\\begin{figure}[htbp]\n\\centering\n% image placeholder\n" + rf"\caption{{{caption}}}" + "\n\\end{figure}"
        if block_type == "reference":
            return render_references(node.record, text)
        if block_type == "list":
            return self.render_list(node, depth=depth)

        parts = [render_textual_node(node)] if text else []
        parts.extend(self.render_child_blocks_with_dynamic_lists(children, depth=depth + 1))
        return "\n\n".join(part for part in parts if part)

    def render_child_blocks_with_dynamic_lists(self, children: list[ResolvedNode], *, depth: int) -> list[str]:
        rendered: list[str] = []
        index = 0
        child_list = sorted_render_children(children)
        while index < len(child_list):
            child = child_list[index]
            if canonical_render_type(child.record) == "reference":
                run: list[ResolvedNode] = []
                while index < len(child_list) and canonical_render_type(child_list[index].record) == "reference":
                    run.append(child_list[index])
                    index += 1
                rendered.append(render_reference_run(run))
                continue
            list_environment = list_environment_for_node(child)
            if list_environment is not None:
                entries: list[tuple[ResolvedNode, list[ResolvedNode]]] = []
                item_node = child
                continuations: list[ResolvedNode] = []
                index += 1
                while index < len(child_list):
                    current = child_list[index]
                    current_environment = list_environment_for_node(current)
                    if current_environment is not None:
                        entries.append((item_node, continuations))
                        item_node = current
                        continuations = []
                        index += 1
                        continue
                    if is_list_item_continuation_node(current):
                        continuations.append(current)
                        index += 1
                        continue
                    break
                entries.append((item_node, continuations))
                rendered.append(self.render_dynamic_list_entries(entries, environment=list_environment, depth=depth))
                continue
            block = self.render_node(child, depth=depth).strip()
            if block:
                rendered.append(block)
            index += 1
        return rendered

    def render_dynamic_list_entries(
        self,
        entries: list[tuple[ResolvedNode, list[ResolvedNode]]],
        *,
        environment: str,
        depth: int,
    ) -> str:
        lines = [rf"\begin{{{environment}}}"]
        for item, continuations in entries:
            if list_environment_for_node(item) is not None:
                item_body = render_textual_node_without_list_marker(item) if item.text else ""
            else:
                item_body = self.render_list_item(item, depth=depth + 1)
            nested = self.render_child_blocks_with_dynamic_lists(item.children, depth=depth + 1)
            continuation_blocks = [self.render_node(node, depth=depth + 1).strip() for node in continuations]
            body_parts = [item_body, *nested, *continuation_blocks]
            item_body = "\n".join(part for part in body_parts if part).strip()
            lines.append(rf"\item {item_body}".rstrip())
        lines.append(rf"\end{{{environment}}}")
        return "\n".join(lines)

    def render_dynamic_list_group(self, items: list[ResolvedNode], *, environment: str, depth: int) -> str:
        entries = [(item, []) for item in items]
        return self.render_dynamic_list_entries(entries, environment=environment, depth=depth)

    def render_dynamic_itemize(self, items: list[ResolvedNode], *, depth: int) -> str:
        return self.render_dynamic_list_group(items, environment="itemize", depth=depth)

    def render_list(self, node: ResolvedNode, *, depth: int = 0) -> str:
        children = sorted_render_children(node.children)
        environment = list_environment_for_record(node.record, fallback_text=node.text)
        if not children:
            item_body = render_textual_node_without_list_marker(node) if node.text else ""
            return rf"\begin{{{environment}}}" + "\n" + rf"\item {item_body}".rstrip() + "\n" + rf"\end{{{environment}}}"
        if children:
            first_child_environment = list_environment_for_node(children[0])
            if first_child_environment is not None:
                environment = first_child_environment
        entries: list[tuple[ResolvedNode, list[ResolvedNode]]] = []
        item_node: ResolvedNode | None = None
        continuations: list[ResolvedNode] = []
        for child in children:
            if list_environment_for_node(child) is not None:
                if item_node is not None:
                    entries.append((item_node, continuations))
                item_node = child
                continuations = []
                continue
            if item_node is not None and is_list_item_continuation_node(child):
                continuations.append(child)
                continue
            if item_node is not None:
                entries.append((item_node, continuations))
                item_node = None
                continuations = []
            entries.append((child, []))
        if item_node is not None:
            entries.append((item_node, continuations))
        return self.render_dynamic_list_entries(entries, environment=environment, depth=depth)

    def render_list_item(self, node: ResolvedNode, *, depth: int = 0) -> str:
        block_type = canonical_render_type(node.record)
        if is_algorithm_like_node(node.record, node_verbatim_text(node)):
            item_body = render_algorithm_block(node_verbatim_text(node))
        elif block_type == "equation":
            item_body = render_equation(node.text)
        elif block_type == "inline_math":
            item_body = render_inline_math(node.text)
        else:
            item_body = render_textual_node_without_list_marker(node) if node.text else ""
        nested = [self.render_node(grandchild, depth=depth + 1).strip() for grandchild in sorted_render_children(node.children)]
        if nested:
            item_body = (item_body + "\n" + "\n".join(part for part in nested if part)).strip()
        return item_body


def greedy_decode_relations(
    edge_index: Any,
    scores: Any,
    *,
    threshold: float = 0.5,
    num_nodes: int | None = None,
) -> list[DecodedEdge]:
    """Backward-compatible alias for the NetworkX arborescence decoder."""

    return decode_relations_with_arborescence(edge_index, scores, threshold=threshold, num_nodes=num_nodes)


def decode_relations_with_arborescence(
    edge_index: Any,
    scores: Any,
    *,
    threshold: float = 0.5,
    num_nodes: int | None = None,
) -> list[DecodedEdge]:
    """Decode relation probabilities through TreeDecoder's first two stages."""

    decoder = TreeDecoder(
        TreeDecoderConfig(
            merge_threshold=threshold,
            parent_threshold=threshold,
            sibling_threshold=threshold,
        )
    )
    return decoder.decode_edges(edge_index, scores, num_nodes=num_nodes)


def build_resolved_tree(node_records: list[dict[str, Any]], decoded_edges: list[DecodedEdge]) -> ResolvedNode:
    """Convert decoded graph edges into a ROOT-backed tree of ResolvedNode objects."""

    union_find = UnionFind(len(node_records))
    merge_edges = [edge for edge in decoded_edges if edge.label == MERGE]
    for edge in merge_edges:
        if valid_node_pair(edge.source, edge.target, len(node_records)):
            union_find.union(edge.source, edge.target)

    groups: dict[int, list[int]] = {}
    for idx in range(len(node_records)):
        groups.setdefault(union_find.find(idx), []).append(idx)

    nodes: dict[int, ResolvedNode] = {}
    old_to_group: dict[int, int] = {}
    for members in groups.values():
        members = sorted(members)
        canonical_id = min(members)
        merged_text = merge_text_fragments([node_record_text(node_records[idx]) for idx in members])
        record = dict(node_records[members[0]])
        record["merged_text"] = merged_text
        record["merged_records"] = [dict(node_records[idx]) for idx in members[1:]]
        record["source_node_ids"] = members
        nodes[canonical_id] = ResolvedNode(node_id=canonical_id, record=record, merged_node_ids=members)
        for member in members:
            old_to_group[member] = canonical_id

    contracted = ContractedGraph(nodes=nodes, old_to_super=old_to_group, merge_edges=merge_edges)
    non_merge_edges = [
        DecodedEdge(
            source=old_to_group.get(edge.source, edge.source),
            target=old_to_group.get(edge.target, edge.target),
            label=edge.label,
            score=edge.score,
        )
        for edge in decoded_edges
        if edge.label != MERGE
    ]
    return TreeDecoder().build_tree(contracted, non_merge_edges)


def normalize_edge_index(edge_index: Any) -> Any:
    edge_index = edge_index.detach().cpu()
    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError("Expected edge_index with shape [2, num_edges]")
    return edge_index


def resolve_num_nodes(edge_index: Any, scores: Any, *, num_nodes: int | None = None) -> int:
    if num_nodes is not None:
        return int(num_nodes)
    edge_index = normalize_edge_index(edge_index)
    if edge_index.numel() == 0:
        return 0
    return int(edge_index.max().item()) + 1


def valid_node_pair(source: int, target: int, node_count: int) -> bool:
    return 0 <= source < node_count and 0 <= target < node_count


def clone_resolved_node(node: ResolvedNode) -> ResolvedNode:
    return ResolvedNode(
        node_id=node.node_id,
        record=dict(node.record),
        merged_node_ids=list(node.merged_node_ids),
        children=[],
        sibling_after=list(node.sibling_after),
    )


def sort_tree_children(node: ResolvedNode) -> None:
    node.children = sorted_render_children(node.children)
    for child in node.children:
        sort_tree_children(child)


def build_heading_skeleton(nodes: dict[int, ResolvedNode]) -> HeadingSkeleton:
    """Build a deterministic heading tree from physical reading order."""

    if not nodes:
        return HeadingSkeleton(frozenset(), {}, {}, {})

    ordered_ids = sorted(nodes, key=lambda node_id: node_reading_order_key(nodes[node_id]))
    body_font_size = infer_body_font_size_from_nodes(nodes.values())
    heading_ids: set[int] = set()
    heading_levels: dict[int, int] = {}
    heading_parent: dict[int, int | None] = {}
    scope_by_node: dict[int, int | None] = {}
    stack: list[tuple[int, int]] = []

    for order_pos, node_id in enumerate(ordered_ids):
        node = nodes[node_id]
        if is_heading_candidate_node(node, body_font_size=body_font_size, order_pos=order_pos):
            level = heading_stack_level(node, body_font_size=body_font_size, order_pos=order_pos)
            node.record["_skeleton_heading_level"] = level
            node.record["canonical_type"] = "title"
            heading_ids.add(node_id)
            heading_levels[node_id] = level
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_id = stack[-1][1] if stack else None
            heading_parent[node_id] = parent_id
            scope_by_node[node_id] = node_id
            stack.append((level, node_id))
            continue
        scope_by_node[node_id] = stack[-1][1] if stack else None

    return HeadingSkeleton(
        heading_ids=frozenset(heading_ids),
        heading_levels=heading_levels,
        heading_parent=heading_parent,
        scope_by_node=scope_by_node,
    )


def is_heading_candidate_node(node: ResolvedNode, *, body_font_size: float, order_pos: int) -> bool:
    record = node.record
    text = " ".join(node.text.split())
    if not text or is_page_noise_node(node):
        return False
    block_type = canonical_render_type(record)
    if block_type == "title":
        return True
    if block_type not in {"text", "reference"}:
        return False
    font_size = node_font_size(record)
    is_bold = node_is_bold(record)
    if (
        title_numbering_level(text) is not None
        and looks_like_standalone_heading(text)
        and (is_bold or (body_font_size > 0 and font_size >= body_font_size * 1.05))
    ):
        return True
    if body_font_size > 0 and font_size >= body_font_size * 1.12 and looks_like_standalone_heading(text):
        return True
    if body_font_size > 0 and font_size >= body_font_size * 1.05 and is_bold and looks_like_standalone_heading(text):
        return True
    return False


def heading_stack_level(node: ResolvedNode, *, body_font_size: float, order_pos: int) -> int:
    text = " ".join(node.text.split())
    explicit_level = title_numbering_level(text)
    if explicit_level is not None:
        return explicit_level

    raw_type = str(
        node.record.get("type")
        or node.record.get("raw_type")
        or node.record.get("block_type")
        or ""
    ).casefold()
    if raw_type == "section":
        return 1
    if raw_type == "subsection":
        return 2
    if raw_type == "subsubsection":
        return 3

    normalized = normalize_structural_heading_text(text)
    if normalized in {"abstract", "references", "bibliography"} or normalized.startswith("appendix"):
        return 1

    font_size = node_font_size(node.record)
    if order_pos == 0 and body_font_size > 0 and font_size >= body_font_size * 1.25 and len(text) >= 25:
        return 0
    if body_font_size > 0 and font_size >= body_font_size * 1.15:
        return 1
    if body_font_size > 0 and font_size >= body_font_size * 1.03:
        return 2
    return 1


def looks_like_standalone_heading(text: str) -> bool:
    value = " ".join(str(text or "").split())
    if not value or len(value) > 160:
        return False
    if LIST_MARKER_RE.match(value) and title_numbering_level(value) is None:
        return False
    if "@" in value or "\\@" in value or value.count(",") >= 2:
        return False
    if ":" in value and not value.rstrip().endswith(":"):
        return False
    if value.endswith((".", "。", "?", "!", "？", "！")):
        return False
    return True


def infer_body_font_size_from_nodes(nodes: Any) -> float:
    weighted: dict[float, int] = {}
    fallback: dict[float, int] = {}
    for node in nodes:
        record = getattr(node, "record", node if isinstance(node, dict) else {})
        size = node_font_size(record)
        if size <= 0:
            continue
        text_len = max(1, len(node_record_text(record)))
        fallback[size] = fallback.get(size, 0) + text_len
        if canonical_render_type(record) == "text":
            weighted[size] = weighted.get(size, 0) + text_len
    source = weighted or fallback
    if not source:
        return 0.0
    return max(source.items(), key=lambda item: item[1])[0]


def node_font_size(record: dict[str, Any]) -> float:
    for key in ("style_baseline_size", "font_size", "baseline_font_size"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    spans = record.get("style_spans")
    if not isinstance(spans, list):
        return 0.0
    weighted: dict[float, int] = {}
    for span in spans:
        if not isinstance(span, dict):
            continue
        size = span.get("font_size")
        if not isinstance(size, (int, float)):
            continue
        weight = int(span.get("char_count") or len(str(span.get("text") or "")) or 1)
        weighted[float(size)] = weighted.get(float(size), 0) + max(1, weight)
    if not weighted:
        return 0.0
    return max(weighted.items(), key=lambda item: item[1])[0]


def node_is_bold(record: dict[str, Any]) -> bool:
    spans = record.get("style_spans")
    if not isinstance(spans, list):
        return False
    bold_chars = 0
    total_chars = 0
    for span in spans:
        if not isinstance(span, dict):
            continue
        count = int(span.get("char_count") or len(str(span.get("text") or "")) or 1)
        total_chars += count
        if span.get("is_bold"):
            bold_chars += count
    return total_chars > 0 and bold_chars / total_chars >= 0.5


def merge_crosses_section_boundary(source: int, target: int, skeleton: HeadingSkeleton) -> bool:
    if source in skeleton.heading_ids or target in skeleton.heading_ids:
        return True
    return skeleton.scope_by_node.get(source) != skeleton.scope_by_node.get(target)


def sibling_crosses_section_boundary(source: int, target: int, skeleton: HeadingSkeleton) -> bool:
    if source in skeleton.heading_ids or target in skeleton.heading_ids:
        return True
    return skeleton.scope_by_node.get(source) != skeleton.scope_by_node.get(target)


def violates_structural_parent_child(source: ResolvedNode, target: ResolvedNode, skeleton: HeadingSkeleton) -> bool:
    source_id = source.node_id
    target_id = target.node_id
    if target_id in skeleton.heading_ids:
        return True
    if source_id in skeleton.heading_ids:
        return skeleton.scope_by_node.get(target_id) != source_id
    if canonical_render_type(target.record) == "title":
        return True
    return skeleton.scope_by_node.get(source_id) != skeleton.scope_by_node.get(target_id)


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


def node_record_text(record: dict[str, Any]) -> str:
    return str(
        record.get("merged_text")
        or record.get("text_for_embedding")
        or record.get("text")
        or record.get("text_preview")
        or record.get("latex")
        or ""
    )


def node_record_verbatim_text(record: dict[str, Any]) -> str:
    for key in ("text", "content", "latex", "text_for_embedding", "text_preview"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def node_verbatim_text(node: ResolvedNode) -> str:
    parts = []
    primary = node_record_verbatim_text(node.record)
    if primary:
        parts.append(primary)
    for record in node.record.get("merged_records", []):
        if isinstance(record, dict):
            text = node_record_verbatim_text(record)
            if text:
                parts.append(text)
    if parts:
        return "\n".join(part.strip("\n") for part in parts if part.strip("\n")).strip()
    return node.text


def sorted_render_children(children: list[ResolvedNode] | tuple[ResolvedNode, ...] | None) -> list[ResolvedNode]:
    child_list = list(children or [])
    if any(has_explicit_reading_order(getattr(child, "record", {})) for child in child_list):
        return sorted(child_list, key=node_reading_order_key)
    if any(record_has_bbox(getattr(child, "record", {})) for child in child_list):
        return sort_nodes_by_reading_order(child_list, fallback_key=node_reading_order_key)
    return sorted(child_list, key=node_reading_order_key)


def has_explicit_reading_order(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    for key in ("regime_reading_order", "dag_reading_order", "global_order", "reading_order", "original_order", "original_index", "index"):
        if numeric_value(record.get(key)) is not None:
            return True
    return False


def node_reading_order_key(node: Any) -> tuple[int, float, float, str]:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    for key in ("regime_reading_order", "dag_reading_order", "xycut_reading_order", "global_order", "reading_order", "original_order", "original_index", "index"):
        value = numeric_value(record.get(key))
        if value is not None:
            return (0, value, 0.0, "")

    source_id = min_numeric_sequence(record.get("source_node_ids"))
    if source_id is not None:
        return (1, source_id, 0.0, "")

    merged_ids = getattr(node, "merged_node_ids", None)
    merged_id = min_numeric_sequence(merged_ids)
    if merged_id is not None:
        return (1, merged_id, 0.0, "")

    node_id = numeric_value(getattr(node, "node_id", None))
    if node_id is not None and node_id >= 0:
        return (1, node_id, 0.0, "")

    page = numeric_value(record.get("page_idx"))
    visual = numeric_value(record.get("visual_order"))
    if page is not None or visual is not None:
        return (2, page if page is not None else 0.0, visual if visual is not None else 0.0, "")

    for key in ("id", "node_id", "block_id"):
        value = numeric_value(record.get(key))
        if value is not None:
            return (3, value, 0.0, "")

    return (4, 0.0, 0.0, "")


def node_physical_index(node: Any) -> float | None:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    for key in ("regime_reading_order", "dag_reading_order", "xycut_reading_order", "global_order", "reading_order", "original_order", "original_index", "index"):
        value = numeric_value(record.get(key))
        if value is not None:
            return value

    source_id = min_numeric_sequence(record.get("source_node_ids"))
    if source_id is not None:
        return source_id

    merged_ids = getattr(node, "merged_node_ids", None)
    merged_id = min_numeric_sequence(merged_ids)
    if merged_id is not None:
        return merged_id

    node_id = numeric_value(getattr(node, "node_id", None))
    if node_id is not None and node_id >= 0:
        return node_id

    for key in ("id", "node_id", "block_id"):
        value = numeric_value(record.get(key))
        if value is not None:
            return value

    page = numeric_value(record.get("page_idx"))
    visual = numeric_value(record.get("visual_order"))
    if page is not None or visual is not None:
        return (page if page is not None else 0.0) * 1_000_000.0 + (visual if visual is not None else 0.0)

    return None


def record_has_bbox(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    value = record.get("bbox")
    return isinstance(value, (list, tuple)) and len(value) >= 4


def violates_parent_child_causality(parent: Any, child: Any) -> bool:
    if is_floating_child_for_causality(child):
        return False
    parent_index = node_physical_index(parent)
    child_index = node_physical_index(child)
    return parent_index is not None and child_index is not None and parent_index > child_index


def is_floating_child_for_causality(node: Any) -> bool:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    return canonical_render_type(record) in {"figure", "table"}


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = NUMERIC_ID_RE.search(value)
        if match:
            return float(match.group(0))
    return None


def min_numeric_sequence(value: Any) -> float | None:
    if not isinstance(value, (list, tuple)):
        return None
    numbers = [number for number in (numeric_value(item) for item in value) if number is not None]
    return min(numbers) if numbers else None


def node_text_for_sort(node: Any) -> str:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    return str(
        getattr(node, "text", "")
        or record.get("text")
        or record.get("text_for_embedding")
        or record.get("text_preview")
        or ""
    )


def merge_text_fragments(parts: list[str]) -> str:
    text = ""
    for part in parts:
        part = " ".join(str(part or "").split())
        if not part:
            continue
        if not text:
            text = part
            continue
        if should_join_without_space(text, part):
            if text.endswith("-") and part[:1].islower():
                text = text[:-1] + part
            else:
                text += part
        else:
            text += " " + part
    return " ".join(text.split())


def should_join_without_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left.endswith("-") and right[:1].islower():
        return True
    if right[0] in ",.;:!?%)]}，。；：！？、》）】":
        return True
    if left[-1] in "([{（《【":
        return True
    if is_cjk(left[-1]) and is_cjk(right[0]):
        return True
    return False


def is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def canonical_render_type(record: dict[str, Any]) -> str:
    if str(record.get("list_type") or "").lower() == "reference_list":
        return "reference"
    raw = str(record.get("canonical_type") or record.get("type") or record.get("raw_type") or record.get("block_type") or "").lower()
    if raw in {"paragraph", "text", "paragraph_text", "body"}:
        return "text"
    if raw in {"title", "section", "subsection", "subsubsection", "heading"}:
        return "title"
    if raw in {"equation", "equation_interline", "interline_equation", "display_formula", "formula"}:
        return "equation"
    if raw in {"inline_math", "inline_formula", "math_inline"}:
        return "inline_math"
    if raw == "table":
        return "table"
    if raw in {"figure", "image", "chart"}:
        return "figure"
    if raw == "algorithm":
        return "algorithm"
    if raw in {"list", "item", "itemize", "enumerate"}:
        return "list"
    if raw == "code":
        return "code"
    if raw in {"reference", "references", "bibliography"}:
        return "reference"
    return "text"


def can_contract_merge_records(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    parallel_y_overlap_ratio: float = 0.10,
    gutter_threshold: float = 30.0,
    gutter_page_width_ratio: float = 0.05,
) -> bool:
    left_type = canonical_render_type(left)
    right_type = canonical_render_type(right)
    if record_starts_with_list_marker(right):
        return False
    if left_type != right_type or left_type not in MERGE_COMPATIBLE_TYPES:
        return False
    return not crosses_column_gutter_barrier(
        left,
        right,
        parallel_y_overlap_ratio=parallel_y_overlap_ratio,
        gutter_threshold=gutter_threshold,
        gutter_page_width_ratio=gutter_page_width_ratio,
    )


def record_starts_with_list_marker(record: dict[str, Any]) -> bool:
    return bool(LIST_MARKER_RE.match(node_record_text(record)))


def merge_crosses_intermediate_list_marker(source: int, target: int, node_records: list[dict[str, Any]]) -> bool:
    lower, upper = sorted((source, target))
    if upper - lower <= 1:
        return False
    for index in range(lower + 1, upper):
        if 0 <= index < len(node_records) and record_starts_with_list_marker(node_records[index]):
            return True
    return False


def crosses_column_gutter_barrier(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    parallel_y_overlap_ratio: float = 0.10,
    gutter_threshold: float = 30.0,
    gutter_page_width_ratio: float = 0.05,
) -> bool:
    """Reject Merge when two boxes sit side-by-side across a clear gutter."""

    left_bbox = merge_barrier_bbox(left)
    right_bbox = merge_barrier_bbox(right)
    if left_bbox is None or right_bbox is None:
        return False
    left_page = merge_barrier_page(left)
    right_page = merge_barrier_page(right)
    if left_page is not None and right_page is not None and left_page != right_page:
        return False

    y_intersect = max(0.0, min(left_bbox[3], right_bbox[3]) - max(left_bbox[1], right_bbox[1]))
    left_height = max(1.0, left_bbox[3] - left_bbox[1])
    right_height = max(1.0, right_bbox[3] - right_bbox[1])
    is_parallel = (
        y_intersect > 0.0
        and (
            y_intersect / left_height >= parallel_y_overlap_ratio
            or y_intersect / right_height >= parallel_y_overlap_ratio
        )
    )
    if not is_parallel:
        return False

    x_gap = max(left_bbox[0], right_bbox[0]) - min(left_bbox[2], right_bbox[2])
    if x_gap <= 0.0:
        return False

    dynamic_threshold = gutter_threshold
    page_width = merge_barrier_page_width(left, right)
    if page_width is not None and page_width > 0:
        dynamic_threshold = min(gutter_threshold, page_width * gutter_page_width_ratio)
    return x_gap > dynamic_threshold


def merge_barrier_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    chunks = bbox_chunks(record.get("bbox"))
    if not chunks:
        return None
    return chunks[0]


def merge_barrier_page_width(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    for record in (left, right):
        for key in ("page_width", "width", "page_w"):
            value = numeric_value(record.get(key))
            if value is not None and value > 0:
                return value
    return 1000.0


def merge_barrier_page(record: dict[str, Any]) -> float | None:
    pages = record.get("source_page_idxs")
    if isinstance(pages, list) and pages:
        value = numeric_value(pages[0])
        if value is not None:
            return value
    for key in ("page_idx", "page", "page_id"):
        value = numeric_value(record.get(key))
        if value is not None:
            return value
    return None


def bbox_chunks(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    usable_len = len(value) - (len(value) % 4)
    chunks = []
    for idx in range(0, usable_len, 4):
        try:
            x0, y0, x1, y1 = (float(value[idx + offset]) for offset in range(4))
        except (TypeError, ValueError):
            continue
        if x1 < x0 or y1 < y0:
            continue
        chunks.append((x0, y0, x1, y1))
    return chunks


def is_bullet_list_candidate(node: ResolvedNode) -> bool:
    return list_environment_for_node(node) is not None


def is_list_item_continuation_node(node: ResolvedNode) -> bool:
    """Allow display objects to stay inside the current list item."""

    block_type = canonical_render_type(node.record)
    return block_type in {"equation", "inline_math", "table", "figure", "algorithm", "code"}


def list_environment_for_node(node: ResolvedNode) -> str | None:
    record = node.record
    block_type = canonical_render_type(record)
    text = node.text
    if block_type == "list" and not node.children:
        return list_environment_for_record(record, fallback_text=text)
    if block_type != "text":
        return None
    return list_environment_for_text(text)


def list_environment_for_record(record: dict[str, Any], *, fallback_text: str = "") -> str:
    explicit = str(record.get("list_type") or record.get("list_style") or record.get("enum_type") or "").casefold()
    if explicit in {"ordered", "enumerate", "numbered", "number", "alpha", "roman"}:
        return "enumerate"
    text = fallback_text or node_record_text(record)
    return list_environment_for_text(text) or "itemize"


def list_environment_for_text(text: str) -> str | None:
    value = str(text or "")
    if not LIST_MARKER_RE.match(value):
        return None
    return "enumerate" if ORDERED_LIST_MARKER_RE.match(value) else "itemize"


def strip_list_marker(text: str) -> str:
    return LIST_MARKER_RE.sub("", str(text or ""), count=1).strip()


def normalize_title_text_for_dedup(text: str) -> str:
    lowered = str(text or "").casefold().strip()
    without_punctuation = "".join(
        char for char in lowered if not unicodedata.category(char).startswith("P")
    )
    return " ".join(without_punctuation.split())


def reroute_decoded_edges(
    edges: list[DecodedEdge],
    alias_map: dict[int, int],
    resolve_alias: Any,
) -> list[DecodedEdge]:
    rerouted: list[DecodedEdge] = []
    for edge in edges:
        source = resolve_alias(edge.source) if edge.source in alias_map else edge.source
        target = resolve_alias(edge.target) if edge.target in alias_map else edge.target
        if source == target:
            continue
        rerouted.append(DecodedEdge(source=source, target=target, label=edge.label, score=edge.score))
    return rerouted


def keep_highest_scoring_edges(edges: list[DecodedEdge]) -> list[DecodedEdge]:
    best: dict[tuple[int, int, int], DecodedEdge] = {}
    for edge in edges:
        if edge.source == edge.target:
            continue
        key = (edge.source, edge.target, edge.label)
        if key not in best or edge.score > best[key].score:
            best[key] = edge
    return sorted(best.values(), key=lambda edge: (edge.source, edge.target, -edge.score))


def normalize_structural_heading_text(text: str) -> str:
    return normalize_title_text_for_dedup(text)


def root_without_redundant_document_title(root: ResolvedNode, title: str | None) -> ResolvedNode:
    if not title:
        return root
    title_key = normalize_structural_heading_text(title)
    if not title_key:
        return root

    replacement_children: list[ResolvedNode] = []
    skipped = False
    for child in sorted_render_children(root.children):
        child_key = normalize_structural_heading_text(child.text)
        if not skipped and canonical_render_type(child.record) == "title" and child_key == title_key:
            replacement_children.extend(sorted_render_children(child.children))
            skipped = True
            continue
        replacement_children.append(child)

    if not skipped:
        return root
    body_root = ResolvedNode(node_id=root.node_id, record=dict(root.record), merged_node_ids=list(root.merged_node_ids))
    body_root.children = replacement_children
    body_root.sibling_after = list(root.sibling_after)
    return body_root


def is_appendix_stop_title(node: ResolvedNode) -> bool:
    if canonical_render_type(node.record) != "title":
        return False
    return "appendix" in normalize_structural_heading_text(node.text).split()


def is_page_noise_node(node: ResolvedNode) -> bool:
    raw_type = str(
        node.record.get("canonical_type")
        or node.record.get("type")
        or node.record.get("raw_type")
        or node.record.get("block_type")
        or ""
    ).casefold()
    if raw_type in {
        "header",
        "footer",
        "page_header",
        "page_footer",
        "page_number",
        "page_num",
        "page_no",
        "pagenum",
        "noise",
    }:
        return True
    normalized = normalize_structural_heading_text(node.text).replace(" ", "")
    return bool(normalized) and normalized.isdigit() and len(normalized) <= 4


def is_abstract_root_candidate(node: ResolvedNode) -> bool:
    block_type = canonical_render_type(node.record)
    if block_type not in {"title", "text"}:
        return False
    text = " ".join(node.text.split())
    return bool(re.search(r"\babstract\b", text[:120], flags=re.IGNORECASE))


def render_title(text: str, *, depth: int) -> str:
    command = title_command(text, depth=depth)
    title_text = strip_title_numbering(text)
    return rf"\{command}{{{escape_latex(title_text)}}}"


def title_command(text: str, *, depth: int) -> str:
    numbered_level = title_numbering_level(text)
    if numbered_level is not None:
        return SECTION_COMMANDS[min(numbered_level - 1, len(SECTION_COMMANDS) - 1)]
    return SECTION_COMMANDS[min(max(0, depth), len(SECTION_COMMANDS) - 1)]


def render_equation(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return "\\[\n\n\\]"
    if stripped.startswith("\\[") or stripped.startswith("$$"):
        return stripped
    begin_match = re.match(r"\\begin\{([^}]+)\}", stripped)
    if begin_match and begin_match.group(1).rstrip("*") in DISPLAY_MATH_ENVS:
        return stripped
    return "\\[\n" + stripped + "\n\\]"


def render_inline_math(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return "$$"
    if stripped.startswith("$") or stripped.startswith(r"\("):
        return stripped
    return "$" + stripped + "$"


def render_textual_node(node: ResolvedNode) -> str:
    records = [node.record] + [record for record in node.record.get("merged_records", []) if isinstance(record, dict)]
    rendered_parts: list[str] = []
    used_structured_content = False
    for record in records:
        rendered = render_textual_content(record, node_record_text(record))
        if extract_content_segments(record):
            used_structured_content = True
        if rendered:
            rendered_parts.append(rendered)
    if used_structured_content and rendered_parts:
        return merge_latex_fragments(rendered_parts)
    return escape_latex(node.text)


def render_textual_node_without_list_marker(node: ResolvedNode) -> str:
    records = [node.record] + [record for record in node.record.get("merged_records", []) if isinstance(record, dict)]
    rendered_parts: list[str] = []
    used_structured_content = False
    marker_stripped = False
    for record in records:
        prepared_record = strip_list_marker_from_record(record) if not marker_stripped else record
        if prepared_record is not record:
            marker_stripped = True
        rendered = render_textual_content(prepared_record, node_record_text(prepared_record))
        if extract_content_segments(prepared_record):
            used_structured_content = True
        if rendered:
            rendered_parts.append(rendered)
    if rendered_parts:
        if used_structured_content:
            return merge_latex_fragments(rendered_parts)
        return normalize_latex_text(" ".join(rendered_parts))
    return escape_latex(strip_list_marker(node.text))


def strip_list_marker_from_record(record: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(record)
    changed = False
    for key in ("merged_text", "text_for_embedding", "text", "text_preview"):
        value = prepared.get(key)
        if isinstance(value, str) and LIST_MARKER_RE.match(value):
            prepared[key] = strip_list_marker(value)
            changed = True
            break

    block = prepared.get("block")
    if isinstance(block, dict):
        block_copy = dict(block)
        content = block_copy.get("content")
        content_copy: Any = content
        segments = extract_content_segments(prepared)
        if segments:
            stripped_segments = []
            stripped = False
            for segment in segments:
                segment_copy = dict(segment)
                if not stripped and str(segment_copy.get("type") or "").lower() == "text":
                    content_text = str(segment_copy.get("content") or segment_copy.get("text") or "")
                    if LIST_MARKER_RE.match(content_text):
                        replacement = strip_list_marker(content_text)
                        if "content" in segment_copy:
                            segment_copy["content"] = replacement
                        else:
                            segment_copy["text"] = replacement
                        stripped = True
                        changed = True
                stripped_segments.append(segment_copy)
            if isinstance(content, dict):
                content_copy = dict(content)
                for key in ("paragraph_content", "title_content", "content"):
                    if isinstance(content_copy.get(key), list):
                        content_copy[key] = stripped_segments
                        break
            elif isinstance(content, list):
                content_copy = stripped_segments
            block_copy["content"] = content_copy
            prepared["block"] = block_copy
    return prepared if changed else record


def render_textual_content(record: dict[str, Any], fallback_text: str) -> str:
    segments = extract_content_segments(record)
    if not segments:
        return escape_latex(fallback_text)
    rendered: list[str] = []
    for segment in segments:
        segment_type = str(segment.get("type") or "").lower()
        content = str(segment.get("content") or segment.get("text") or "")
        if not content:
            continue
        if segment_type in {"equation_inline", "inline_equation", "inline_math", "inline_formula"}:
            rendered.append(render_inline_math(content))
        elif segment_type in {"equation_interline", "interline_equation", "display_formula", "formula", "equation"}:
            rendered.append("\n\n" + render_equation(content) + "\n\n")
        else:
            rendered.append(escape_latex(content))
    return normalize_latex_text("".join(rendered))


def extract_content_segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    block = record.get("block")
    if not isinstance(block, dict):
        return []
    content = block.get("content")
    if isinstance(content, dict):
        for key in ("paragraph_content", "title_content", "content"):
            value = content.get(key)
            if isinstance(value, list):
                return [segment for segment in value if isinstance(segment, dict)]
    if isinstance(content, list):
        return [segment for segment in content if isinstance(segment, dict)]
    return []


def merge_latex_fragments(parts: list[str]) -> str:
    text = ""
    for part in parts:
        part = normalize_latex_text(part)
        if not part:
            continue
        if not text:
            text = part
            continue
        if should_join_without_space(text, part):
            text += part
        else:
            text += " " + part
    return normalize_latex_text(text)


def normalize_latex_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(text)).strip()


def render_references(record: dict[str, Any], fallback_text: str) -> str:
    items = collect_reference_items(record)
    if not items:
        items = [line.strip() for line in fallback_text.split("\n") if line.strip()] or ([fallback_text.strip()] if fallback_text.strip() else [])
    if not items:
        return ""
    lines = [r"\begin{thebibliography}{99}"]
    for idx, item in enumerate(items, start=1):
        lines.append(rf"\bibitem{{ref{idx}}} {escape_latex(item)}")
    lines.append(r"\end{thebibliography}")
    return "\n".join(lines)


def render_reference_run(nodes: list[ResolvedNode]) -> str:
    if not nodes:
        return ""
    primary = dict(nodes[0].record)
    merged_records = list(primary.get("merged_records") or [])
    for node in nodes[1:]:
        merged_records.append(node.record)
        merged_records.extend(node.record.get("merged_records") or [])
    primary["merged_records"] = merged_records
    fallback = "\n".join(node.text for node in nodes if node.text)
    return render_references(primary, fallback)


def collect_reference_items(record: dict[str, Any]) -> list[str]:
    """Collect reference items from a primary record and any merged records."""

    items = normalize_reference_items(record.get("reference_items"))
    for merged_record in record.get("merged_records", []):
        if isinstance(merged_record, dict):
            items.extend(normalize_reference_items(merged_record.get("reference_items")))
    return [item for item in items if item]


def normalize_reference_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        text = item.get("text") if isinstance(item, dict) else item
        text = str(text or "").strip()
        if text:
            items.append(text)
    return items


def render_verbatim_like(text: str, label: str) -> str:
    if not text:
        return f"% empty {label} block"
    return "\\begin{verbatim}\n" + safe_verbatim_text(text.strip()) + "\n\\end{verbatim}"


def is_algorithm_like_node(record: dict[str, Any], text: str) -> bool:
    if canonical_render_type(record) in {"algorithm", "code"}:
        return True
    return bool(PSEUDOCODE_START_RE.match(str(text or "")))


def render_algorithm_block(text: str) -> str:
    caption, commands = parse_pseudo_code(text)
    lines = [r"\begin{algorithm}[H]"]
    if caption:
        lines.append(rf"\caption{{{escape_latex(caption)}}}")
    lines.append(r"\begin{algorithmic}[1]")
    lines.extend(commands or [r"\State " + format_algorithmic_text(text)])
    lines.append(r"\end{algorithmic}")
    lines.append(r"\end{algorithm}")
    return "\n".join(lines)


def parse_pseudo_code(text: str) -> tuple[str | None, list[str]]:
    body = restore_algorithm_line_breaks(text)
    raw_lines = [line.strip() for line in body.split("\n") if line.strip()]
    caption: str | None = None
    commands: list[str] = []
    block_stack: list[str] = []

    for raw_line in raw_lines:
        line = strip_pseudocode_line_number(raw_line)
        caption_match = ALGORITHM_CAPTION_RE.match(line)
        if caption_match and caption is None:
            caption = caption_match.group(1).strip() or "Algorithm"
            continue

        io_match = PSEUDOCODE_IO_RE.match(line)
        if io_match:
            kind, content = io_match.group(1).casefold(), io_match.group(2).strip()
            command = r"\Require" if kind in {"input", "require"} else r"\Ensure"
            commands.append(rf"{command} {format_algorithmic_text(content)}")
            continue

        end_match = PSEUDOCODE_END_RE.match(line)
        if end_match:
            close_kind = end_match.group(1)
            commands.append(close_algorithmic_block(block_stack, close_kind))
            continue

        for_match = PSEUDOCODE_FOR_RE.match(line)
        if for_match:
            commands.append(rf"\For{{{format_algorithmic_text(for_match.group(1).strip())}}}")
            block_stack.append("for")
            continue

        while_match = PSEUDOCODE_WHILE_RE.match(line)
        if while_match:
            commands.append(rf"\While{{{format_algorithmic_text(while_match.group(1).strip())}}}")
            block_stack.append("while")
            continue

        if_match = PSEUDOCODE_IF_RE.match(line)
        if if_match:
            commands.append(rf"\If{{{format_algorithmic_text(if_match.group(1).strip())}}}")
            block_stack.append("if")
            continue

        return_match = PSEUDOCODE_RETURN_RE.match(line)
        if return_match:
            commands.append(rf"\State \Return {format_algorithmic_text(return_match.group(1).strip())}")
            continue

        commands.append(rf"\State {format_algorithmic_text(line)}")

    while block_stack:
        commands.append(close_algorithmic_block(block_stack, None))
    return caption, commands


def strip_pseudocode_line_number(line: str) -> str:
    return re.sub(r"^\s*\d+\s*[:.)]\s*", "", line).strip()


def close_algorithmic_block(block_stack: list[str], close_kind: str | None) -> str:
    normalized = str(close_kind or "").casefold()
    if normalized in block_stack:
        block_stack.pop(len(block_stack) - 1 - block_stack[::-1].index(normalized))
        kind = normalized
    elif block_stack:
        kind = block_stack.pop()
    else:
        kind = normalized or "for"
    if kind == "if":
        return r"\EndIf"
    if kind == "while":
        return r"\EndWhile"
    return r"\EndFor"


def format_algorithmic_text(text: str) -> str:
    prepared = normalize_algorithm_math_text(text)
    if not prepared:
        return ""
    if LATEX_MATH_MARKER_RE.search(prepared):
        return r"\(\displaystyle " + escape_algorithm_math_text(prepared) + r"\)"
    return escape_latex(prepared)


def normalize_algorithm_math_text(text: str) -> str:
    normalized = "".join(ALGORITHM_MATH_UNICODE_REPLACEMENTS.get(char, char) for char in str(text or ""))
    normalized = normalized.replace("<-", r"\gets").replace("->", r"\to")
    return " ".join(normalized.split())


def escape_algorithm_math_text(text: str) -> str:
    return (
        str(text)
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def restore_algorithm_line_breaks(text: str) -> str:
    body = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" in body:
        return body
    body = PSEUDOCODE_BREAK_RE.sub("\n", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def sanitize_verbatim_body(text: str) -> str:
    sanitized = VERBATIM_END_RE.sub(r"\\end {verbatim}", str(text or ""))
    return "".join(_safe_code_verbatim_char(char) for char in sanitized)


def render_table_placeholder(record: dict[str, Any], text: str, *, node_id: int | None = None) -> str:
    table_id = table_node_identifier(record, node_id=node_id)
    bbox = format_table_bbox(record.get("bbox"))
    caption = extract_table_caption(text) or "Table reconstruction placeholder"
    todo = f"% [TODO_TABLE_RECONSTRUCT: BBOX={bbox}, ID={table_id}]"
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            todo,
            rf"\caption{{{escape_latex(caption)}}}",
            r"\end{table}",
        ]
    )


def table_node_identifier(record: dict[str, Any], *, node_id: int | None = None) -> str:
    for key in ("id", "block_id", "table_id"):
        value = record.get(key)
        if value:
            return str(value)
    value = record.get("global_order")
    if value is not None:
        return f"table_{value}"
    if node_id is not None and node_id >= 0:
        return f"table_{node_id}"
    return "table_unknown"


def format_table_bbox(value: Any) -> str:
    if not isinstance(value, list) or len(value) < 4:
        return "UNKNOWN"
    try:
        coords = [float(coord) for coord in value[:4]]
    except (TypeError, ValueError):
        return "UNKNOWN"
    return "(" + ", ".join(format_bbox_number(coord) for coord in coords) + ")"


def format_bbox_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def extract_table_caption(text: str) -> str | None:
    match = TABLE_CAPTION_RE.search(str(text or ""))
    if not match:
        return None
    return " ".join(match.group(1).split())


def _safe_code_verbatim_char(char: str) -> str:
    if ord(char) < 128:
        return char
    if char in CODE_UNICODE_REPLACEMENTS:
        return CODE_UNICODE_REPLACEMENTS[char]
    ascii_fallback = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode("ascii")
    return ascii_fallback or "?"


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(_escape_latex_char(char, replacements) for char in str(text))


def _escape_latex_char(char: str, replacements: dict[str, str]) -> str:
    if char in UNICODE_LATEX_REPLACEMENTS:
        return UNICODE_LATEX_REPLACEMENTS[char]
    if char in replacements:
        return replacements[char]
    if ord(char) < 128:
        return char
    ascii_fallback = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode("ascii")
    if ascii_fallback:
        return "".join(replacements.get(fallback_char, fallback_char) for fallback_char in ascii_fallback)
    return "?"


def safe_verbatim_text(text: str) -> str:
    return "".join(_safe_verbatim_char(char) for char in str(text))


def _safe_verbatim_char(char: str) -> str:
    if ord(char) < 128:
        return char
    if char in UNICODE_LATEX_REPLACEMENTS:
        return UNICODE_LATEX_REPLACEMENTS[char]
    ascii_fallback = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode("ascii")
    return ascii_fallback or "?"


ALGORITHM_MATH_UNICODE_REPLACEMENTS = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ϵ": r"\epsilon",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "±": r"\pm",
    "×": r"\times",
    "÷": r"\div",
    "∞": r"\infty",
    "∂": r"\partial",
    "∇": r"\nabla",
    "∑": r"\sum",
    "∫": r"\int",
    "∈": r"\in",
    "∉": r"\notin",
    "∋": r"\ni",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "⊃": r"\supset",
    "⊇": r"\supseteq",
    "∪": r"\cup",
    "∩": r"\cap",
    "∧": r"\wedge",
    "∨": r"\vee",
    "¬": r"\neg",
    "∀": r"\forall",
    "∃": r"\exists",
    "∅": r"\emptyset",
    "∝": r"\propto",
    "∼": r"\sim",
    "≃": r"\simeq",
    "≅": r"\cong",
    "≡": r"\equiv",
    "≪": r"\ll",
    "≫": r"\gg",
    "⋅": r"\cdot",
    "·": r"\cdot",
    "∗": r"*",
    "√": r"\sqrt{}",
    "→": r"\to",
    "←": r"\gets",
    "↔": r"\leftrightarrow",
    "⟶": r"\longrightarrow",
    "⟵": r"\longleftarrow",
    "⇔": r"\Leftrightarrow",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
}


CODE_UNICODE_REPLACEMENTS = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ϵ": "epsilon",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Γ": "Gamma",
    "Δ": "Delta",
    "Θ": "Theta",
    "Λ": "Lambda",
    "Ξ": "Xi",
    "Π": "Pi",
    "Σ": "Sigma",
    "Φ": "Phi",
    "Ψ": "Psi",
    "Ω": "Omega",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "≈": "~=",
    "±": "+/-",
    "×": "x",
    "÷": "/",
    "∞": "inf",
    "∂": "partial",
    "∇": "nabla",
    "∑": "sum",
    "∫": "int",
    "∈": " in ",
    "∉": " notin ",
    "∋": " contains ",
    "⊂": " subset ",
    "⊆": " subseteq ",
    "⊃": " superset ",
    "⊇": " superseteq ",
    "∪": " union ",
    "∩": " inter ",
    "∧": " and ",
    "∨": " or ",
    "¬": "not ",
    "∀": "forall ",
    "∃": "exists ",
    "∅": "empty",
    "∝": "propto",
    "∼": "~",
    "≃": "~=",
    "≅": "~=",
    "≡": "==",
    "≪": "<<",
    "≫": ">>",
    "⋅": "*",
    "·": "*",
    "∗": "*",
    "√": "sqrt",
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "⟶": "->",
    "⟵": "<-",
    "⇔": "<=>",
    "⇒": "=>",
    "⇐": "<=",
    "′": "'",
    "″": "''",
    "°": "deg",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "•": "*",
    "–": "-",
    "—": "---",
    "−": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}


UNICODE_LATEX_REPLACEMENTS = {
    "α": r"\ensuremath{\alpha}",
    "β": r"\ensuremath{\beta}",
    "γ": r"\ensuremath{\gamma}",
    "δ": r"\ensuremath{\delta}",
    "ϵ": r"\ensuremath{\epsilon}",
    "ε": r"\ensuremath{\epsilon}",
    "ζ": r"\ensuremath{\zeta}",
    "η": r"\ensuremath{\eta}",
    "θ": r"\ensuremath{\theta}",
    "ι": r"\ensuremath{\iota}",
    "κ": r"\ensuremath{\kappa}",
    "λ": r"\ensuremath{\lambda}",
    "μ": r"\ensuremath{\mu}",
    "ν": r"\ensuremath{\nu}",
    "ξ": r"\ensuremath{\xi}",
    "π": r"\ensuremath{\pi}",
    "ρ": r"\ensuremath{\rho}",
    "σ": r"\ensuremath{\sigma}",
    "τ": r"\ensuremath{\tau}",
    "υ": r"\ensuremath{\upsilon}",
    "φ": r"\ensuremath{\phi}",
    "χ": r"\ensuremath{\chi}",
    "ψ": r"\ensuremath{\psi}",
    "ω": r"\ensuremath{\omega}",
    "Γ": r"\ensuremath{\Gamma}",
    "Δ": r"\ensuremath{\Delta}",
    "Θ": r"\ensuremath{\Theta}",
    "Λ": r"\ensuremath{\Lambda}",
    "Ξ": r"\ensuremath{\Xi}",
    "Π": r"\ensuremath{\Pi}",
    "Σ": r"\ensuremath{\Sigma}",
    "Φ": r"\ensuremath{\Phi}",
    "Ψ": r"\ensuremath{\Psi}",
    "Ω": r"\ensuremath{\Omega}",
    "≤": r"\ensuremath{\leq}",
    "≥": r"\ensuremath{\geq}",
    "≠": r"\ensuremath{\neq}",
    "≈": r"\ensuremath{\approx}",
    "±": r"\ensuremath{\pm}",
    "×": r"\ensuremath{\times}",
    "÷": r"\ensuremath{\div}",
    "∞": r"\ensuremath{\infty}",
    "∂": r"\ensuremath{\partial}",
    "∇": r"\ensuremath{\nabla}",
    "∑": r"\ensuremath{\sum}",
    "∫": r"\ensuremath{\int}",
    "∈": r"\ensuremath{\in}",
    "∉": r"\ensuremath{\notin}",
    "∋": r"\ensuremath{\ni}",
    "⊂": r"\ensuremath{\subset}",
    "⊆": r"\ensuremath{\subseteq}",
    "⊃": r"\ensuremath{\supset}",
    "⊇": r"\ensuremath{\supseteq}",
    "∪": r"\ensuremath{\cup}",
    "∩": r"\ensuremath{\cap}",
    "∧": r"\ensuremath{\wedge}",
    "∨": r"\ensuremath{\vee}",
    "¬": r"\ensuremath{\neg}",
    "∀": r"\ensuremath{\forall}",
    "∃": r"\ensuremath{\exists}",
    "∅": r"\ensuremath{\emptyset}",
    "∝": r"\ensuremath{\propto}",
    "∼": r"\ensuremath{\sim}",
    "≃": r"\ensuremath{\simeq}",
    "≅": r"\ensuremath{\cong}",
    "≡": r"\ensuremath{\equiv}",
    "≪": r"\ensuremath{\ll}",
    "≫": r"\ensuremath{\gg}",
    "⋅": r"\ensuremath{\cdot}",
    "·": r"\ensuremath{\cdot}",
    "∗": r"\ensuremath{*}",
    "√": r"\ensuremath{\sqrt{\ }}",
    "→": r"\ensuremath{\rightarrow}",
    "←": r"\ensuremath{\leftarrow}",
    "↔": r"\ensuremath{\leftrightarrow}",
    "⟶": r"\ensuremath{\longrightarrow}",
    "⟵": r"\ensuremath{\longleftarrow}",
    "⇔": r"\ensuremath{\Leftrightarrow}",
    "⇒": r"\ensuremath{\Rightarrow}",
    "⇐": r"\ensuremath{\Leftarrow}",
    "′": r"\ensuremath{'}",
    "″": r"\ensuremath{''}",
    "°": r"\ensuremath{^\circ}",
    "¹": r"\ensuremath{^1}",
    "²": r"\ensuremath{^2}",
    "³": r"\ensuremath{^3}",
    "⁰": r"\ensuremath{^0}",
    "⁴": r"\ensuremath{^4}",
    "⁵": r"\ensuremath{^5}",
    "⁶": r"\ensuremath{^6}",
    "⁷": r"\ensuremath{^7}",
    "⁸": r"\ensuremath{^8}",
    "⁹": r"\ensuremath{^9}",
    "₀": r"\ensuremath{_0}",
    "₁": r"\ensuremath{_1}",
    "₂": r"\ensuremath{_2}",
    "₃": r"\ensuremath{_3}",
    "₄": r"\ensuremath{_4}",
    "₅": r"\ensuremath{_5}",
    "₆": r"\ensuremath{_6}",
    "₇": r"\ensuremath{_7}",
    "₈": r"\ensuremath{_8}",
    "₉": r"\ensuremath{_9}",
    "–": "--",
    "—": "---",
    "−": r"\ensuremath{-}",
    "•": r"\textbullet{}",
    "“": "``",
    "”": "''",
    "‘": "`",
    "’": "'",
    "´": "'",
}
