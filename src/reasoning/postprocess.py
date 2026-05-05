"""Decode edge probabilities into a self-consistent document tree."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


MERGE = 0
PARENT_CHILD = 1
SIBLING = 2
NONE = 3
VIRTUAL_ROOT = "__ROOT__"
SECTION_COMMANDS = ["section", "subsection", "subsubsection", "paragraph", "subparagraph"]
DISPLAY_MATH_ENVS = {"equation", "align", "gather", "eqnarray", "flalign", "multline"}


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
    sibling_threshold: float = 0.5
    require_merge_argmax: bool = True
    require_parent_argmax: bool = True
    include_sibling_edges: bool = True
    document_class: str = "article"
    packages: tuple[str, ...] = ("graphicx", "amsmath", "amssymb", "booktabs", "hyperref")


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
        contracted = self.contract_merge_nodes(node_records, edge_index, probs)
        parent_edges = self.maximum_parent_arborescence(contracted, edge_index, probs)
        sibling_edges = self.decode_sibling_edges(contracted, edge_index, probs)
        return self.build_tree(contracted, parent_edges + sibling_edges)

    def decode_edges(self, edge_index: Any, scores: Any, *, num_nodes: int | None = None) -> list[DecodedEdge]:
        """Return decoded edges while preserving the legacy function API."""

        probs = self.edge_probabilities(scores)
        node_count = resolve_num_nodes(edge_index, scores, num_nodes=num_nodes)
        node_records = [{"type": "text", "text": ""} for _ in range(node_count)]
        contracted = self.contract_merge_nodes(node_records, edge_index, probs)
        parent_edges = self.maximum_parent_arborescence(contracted, edge_index, probs)
        sibling_edges = self.decode_sibling_edges(contracted, edge_index, probs)
        return contracted.merge_edges + parent_edges + sibling_edges

    def edge_probabilities(self, scores: Any) -> Any:
        """Accept logits or probabilities and return CPU probability rows."""

        import torch
        import torch.nn.functional as F

        if scores.numel() == 0:
            return scores.detach().cpu()
        probs = scores.detach().cpu().to(dtype=torch.float32)
        if probs.ndim != 2 or int(probs.shape[1]) < 4:
            raise ValueError("Expected edge scores with shape [num_edges, 4]")
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

    def maximum_parent_arborescence(
        self,
        contracted: ContractedGraph,
        edge_index: Any,
        probs: Any,
    ) -> list[DecodedEdge]:
        """Stage 2: decode Parent-Child relations with NetworkX MSA."""

        import networkx as nx
        from networkx.algorithms.tree.branchings import maximum_spanning_arborescence

        graph = nx.DiGraph()
        graph.add_node(VIRTUAL_ROOT)
        for node_id in sorted(contracted.nodes):
            graph.add_node(node_id)
            graph.add_edge(
                VIRTUAL_ROOT,
                node_id,
                weight=0.0,
                score=0.0,
                label=NONE,
                synthetic_root=True,
            )

        edge_index = normalize_edge_index(edge_index)
        for edge_pos in range(edge_index.shape[1]):
            raw_source = int(edge_index[0, edge_pos].item())
            raw_target = int(edge_index[1, edge_pos].item())
            source = contracted.old_to_super.get(raw_source)
            target = contracted.old_to_super.get(raw_target)
            if source is None or target is None or source == target:
                continue
            parent_score = float(probs[edge_pos, PARENT_CHILD].item())
            label = int(probs[edge_pos].argmax().item())
            if parent_score < self.config.parent_threshold:
                continue
            if self.config.require_parent_argmax and label != PARENT_CHILD:
                continue
            if graph.has_edge(source, target) and float(graph[source][target]["weight"]) >= parent_score:
                continue
            graph.add_edge(
                source,
                target,
                weight=parent_score,
                score=parent_score,
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

    def decode_sibling_edges(self, contracted: ContractedGraph, edge_index: Any, probs: Any) -> list[DecodedEdge]:
        if not self.config.include_sibling_edges:
            return []
        edge_index = normalize_edge_index(edge_index)
        decoded: list[DecodedEdge] = []
        for edge_pos in range(edge_index.shape[1]):
            raw_source = int(edge_index[0, edge_pos].item())
            raw_target = int(edge_index[1, edge_pos].item())
            source = contracted.old_to_super.get(raw_source)
            target = contracted.old_to_super.get(raw_target)
            if source is None or target is None or source == target:
                continue
            score = float(probs[edge_pos, SIBLING].item())
            if int(probs[edge_pos].argmax().item()) != SIBLING or score < self.config.sibling_threshold:
                continue
            decoded.append(DecodedEdge(source=source, target=target, label=SIBLING, score=score))
        return decoded

    def build_tree(self, contracted: ContractedGraph, decoded_edges: list[DecodedEdge]) -> ResolvedNode:
        """Attach contracted nodes under the arborescence, adding virtual ROOT."""

        nodes = {node_id: clone_resolved_node(node) for node_id, node in contracted.nodes.items()}
        parent_of: dict[int, int] = {}
        sibling_after: dict[int, list[int]] = {}

        for edge in decoded_edges:
            if edge.source == edge.target:
                continue
            if edge.label == PARENT_CHILD and edge.source in nodes and edge.target in nodes:
                parent_of[edge.target] = edge.source
            elif edge.label == SIBLING and edge.source in nodes and edge.target in nodes:
                sibling_after.setdefault(edge.source, []).append(edge.target)

        for child_id, parent_id in parent_of.items():
            nodes[parent_id].children.append(nodes[child_id])
        for source, targets in sibling_after.items():
            nodes[source].sibling_after.extend(sorted(set(targets)))

        root = ResolvedNode(node_id=-1, record={"type": "root", "text": "ROOT"}, merged_node_ids=[])
        for node_id in sorted(nodes, key=lambda idx: min(nodes[idx].merged_node_ids or [idx])):
            if node_id not in parent_of:
                root.children.append(nodes[node_id])
        sort_tree_children(root)
        return root

    def render_document(self, root: ResolvedNode, *, title: str | None = None) -> str:
        """Stage 3: render a resolved tree to a compilable LaTeX document."""

        lines = [rf"\documentclass{{{self.config.document_class}}}"]
        for package in self.config.packages:
            lines.append(rf"\usepackage{{{package}}}")
        lines.append("")
        if title:
            lines.extend([rf"\title{{{escape_latex(title)}}}", r"\date{}", ""])
        lines.append(r"\begin{document}")
        if title:
            lines.append(r"\maketitle")
            lines.append("")
        body = self.render_node(root, depth=0, is_root=True).strip()
        if body:
            lines.append(body)
            lines.append("")
        lines.append(r"\end{document}")
        return "\n".join(lines).rstrip() + "\n"

    def render_node(self, node: ResolvedNode, *, depth: int = 0, is_root: bool = False) -> str:
        if is_root:
            return "\n\n".join(
                rendered
                for child in node.children
                if (rendered := self.render_node(child, depth=0).strip())
            )

        block_type = canonical_render_type(node.record)
        text = node.text
        children = list(node.children)
        if block_type == "title":
            command = SECTION_COMMANDS[min(depth, len(SECTION_COMMANDS) - 1)]
            parts = [rf"\{command}{{{escape_latex(text)}}}"] if text else []
            parts.extend(self.render_node(child, depth=depth + 1).strip() for child in children)
            return "\n\n".join(part for part in parts if part)
        if block_type == "equation":
            return render_equation(text)
        if block_type in {"table", "algorithm", "code"}:
            return render_verbatim_like(text, block_type)
        if block_type == "figure":
            caption = escape_latex(text) if text else "Figure"
            return "\\begin{figure}[htbp]\n\\centering\n% image placeholder\n" + rf"\caption{{{caption}}}" + "\n\\end{figure}"
        if block_type == "reference":
            return render_references(node.record, text)
        if block_type == "list":
            return self.render_list(node, depth=depth)

        parts = [escape_latex(text)] if text else []
        parts.extend(self.render_node(child, depth=depth + 1).strip() for child in children)
        return "\n\n".join(part for part in parts if part)

    def render_list(self, node: ResolvedNode, *, depth: int = 0) -> str:
        if not node.children:
            return "\\begin{itemize}\n" + rf"\item {escape_latex(node.text)}" + "\n\\end{itemize}"
        lines = [r"\begin{itemize}"]
        for child in node.children:
            child_text = escape_latex(child.text) if child.text else ""
            nested = [self.render_node(grandchild, depth=depth + 1).strip() for grandchild in child.children]
            item = (child_text + ("\n" + "\n".join(part for part in nested if part) if nested else "")).strip()
            lines.append(rf"\item {item}".rstrip())
        lines.append(r"\end{itemize}")
        return "\n".join(lines)


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
    node.children.sort(key=lambda child: min(child.merged_node_ids or [child.node_id]))
    for child in node.children:
        sort_tree_children(child)


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
    raw = str(record.get("canonical_type") or record.get("type") or record.get("raw_type") or record.get("block_type") or "").lower()
    if raw in {"paragraph", "text", "paragraph_text", "body"}:
        return "text"
    if raw in {"title", "section", "subsection", "subsubsection", "heading"}:
        return "title"
    if raw in {"equation", "equation_interline", "display_formula", "formula"}:
        return "equation"
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


def render_equation(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "\\[\n\n\\]"
    if stripped.startswith("\\[") or stripped.startswith("$$"):
        return stripped
    begin_match = re.match(r"\\begin\{([^}]+)\}", stripped)
    if begin_match and begin_match.group(1).rstrip("*") in DISPLAY_MATH_ENVS:
        return stripped
    return "\\[\n" + stripped + "\n\\]"


def render_references(record: dict[str, Any], fallback_text: str) -> str:
    references = record.get("reference_items")
    if isinstance(references, list) and references:
        items = [str(item.get("text") if isinstance(item, dict) else item).strip() for item in references]
    else:
        items = [line.strip() for line in fallback_text.split("\n") if line.strip()] or ([fallback_text.strip()] if fallback_text.strip() else [])
    if not items:
        return ""
    lines = [r"\begin{thebibliography}{99}"]
    for idx, item in enumerate(items, start=1):
        lines.append(rf"\bibitem{{ref{idx}}} {escape_latex(item)}")
    lines.append(r"\end{thebibliography}")
    return "\n".join(lines)


def render_verbatim_like(text: str, label: str) -> str:
    if not text:
        return f"% empty {label} block"
    return "\\begin{verbatim}\n" + text.strip() + "\n\\end{verbatim}"


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
