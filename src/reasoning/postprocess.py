"""Decode edge probabilities into a self-consistent document tree."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.generation.citations import strip_reference_label
from src.generation.table_assets import ensure_figure_asset, ensure_table_pdf_crop, table_caption_text
from src.perception.xy_cut import sort_nodes_by_reading_order
from src.perception.title_features import is_front_matter_date_text, strip_title_numbering, title_numbering_level
from src.reasoning.heading_skeleton import (
    HeadingEvidence,
    HeadingStyleProfile,
    collect_heading_evidence,
    learn_heading_style_profile,
)
from src.reasoning.layout_state_machine import LayoutParseResult, parse_layout_state_machine


MERGE = 0
PARENT_CHILD = 1
NONE = 2
SIBLING = 2  # Deprecated compatibility alias: sibling is derived from reading order, not decoded.
VIRTUAL_ROOT = "__ROOT__"
SECTION_COMMANDS = ["section", "subsection", "subsubsection", "paragraph", "subparagraph"]
DISPLAY_MATH_ENVS = {"equation", "align", "gather", "eqnarray", "flalign", "multline"}
MERGE_COMPATIBLE_TYPES = {"text", "reference"}
NON_PARENT_RENDER_TYPES = {"equation", "inline_math", "algorithm", "code"}
DEFAULT_PREAMBLE_COMMANDS = (r"\providecommand{\mathbfcal}[1]{\mathbf{\mathcal{#1}}}",)
LATEX_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
AUTHOR_BIOGRAPHY_ROLE_TOKENS = ("author_bio", "authorbiography", "biograph", "backmatter")
INLINE_MATH_COMMANDS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Phi",
    "Psi",
    "Omega",
    "mathrm",
    "mathbf",
    "mathit",
    "mathsf",
    "mathtt",
    "mathcal",
    "mathbfcal",
    "operatorname",
    "frac",
    "dfrac",
    "tfrac",
    "sqrt",
    "left",
    "right",
    "leq",
    "geq",
    "neq",
    "approx",
    "sim",
    "simeq",
    "times",
    "cdot",
    "pm",
    "mp",
    "prime",
    "partial",
    "nabla",
    "sum",
    "prod",
    "int",
    "in",
    "notin",
}
LIST_MARKER_RE = re.compile(r"^\s*(?P<marker>[\u2022\u25E6\u25CB\u25AA\-\*]|\d+\.|[a-zA-Z]\.)\s+")
ORDERED_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\.|[a-zA-Z]\.)\s+")
NUMERIC_ID_RE = re.compile(r"\d+")
MERGE_TERMINAL_PUNCT_RE = re.compile(r"[.!?。！？:;；]\s*(?:[)\]”’\"}]*)$")
MERGE_TRAILING_HYPHEN_RE = re.compile(r"[-‐‑‒–—]\s*$")
MERGE_CONTINUATION_START_RE = re.compile(
    r"^\s*(?:[a-z,;:)\]}]|and\b|or\b|where\b|which\b|that\b|while\b|because\b|for\b|in\b|of\b|to\b|the\b)",
    re.IGNORECASE,
)
NOTE_MARKER_RE = re.compile(
    r"^\s*(?:(?:\[(?P<bracket>[0-9A-Za-z*†‡§¶]+)\])|(?:\((?P<paren>[0-9A-Za-z*†‡§¶]+)\))|(?P<bare>[0-9]{1,3}|[*†‡§¶]))[\s:.\-]*"
)
NUMERIC_PAREN_PREFIX_RE = re.compile(r"^\s*\d+\)\s+")
NUMERIC_PREFIX_RE = re.compile(r"^\s*\d+[\.\)]\s+")
ORDERED_NUMERIC_DOT_PREFIX_RE = re.compile(r"^\s*(\d+)\.\s+")
DOTTED_NUMERIC_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)+\.?\s+")
APPENDIX_DOTTED_PREFIX_RE = re.compile(r"^\s*[A-Z](?:\.\d+)+\.?\s+")
ALPHA_PREFIX_RE = re.compile(r"^\s*[A-Za-z][\.\)]\s+")
ROMAN_PREFIX_RE = re.compile(r"^\s*[IVXLCDM]+[\.\)]\s+", re.IGNORECASE)
CUSTOM_COLON_PREFIX_RE = re.compile(r"^\s*[^\s:]{1,24}(?:\s+[\w\-\.]+){0,3}\s*:\s+")
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
FLOAT_CAPTION_RE = re.compile(
    r"^\s*(?P<kind>Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?)"
    r"\s*(?P<number>\d+(?:\.\d+)*[A-Za-z]?)?\s*[:.\-]\s*(?P<body>.+)?",
    re.IGNORECASE,
)
LATEX_MATH_MARKER_RE = re.compile(r"(\\[A-Za-z]+|[_^{}]|[<>=+\-*/]|\\\(|\\\[)")
MATH_COMMAND_RE = re.compile(r"\\([A-Za-z]+)\*?")
ALGORITHM_CODE_MARKER_RE = re.compile(r"([{};]|(?:\+\+|--|==|!=|&&|\|\|))")
BARE_OPERATOR_EQUATION_RE = re.compile(r"^\\(?:arc)?(?:sin|cos|tan)\s*=")
GREEK_CONTEXT_RE = re.compile(r"[αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ]")
GREEK_TO_LATEX = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
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
    "Α": "A",
    "Β": "B",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Θ": r"\Theta",
    "Ι": "I",
    "Κ": "K",
    "Λ": r"\Lambda",
    "Μ": "M",
    "Ν": "N",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Ρ": "P",
    "Σ": r"\Sigma",
    "Τ": "T",
    "Υ": r"\Upsilon",
    "Φ": r"\Phi",
    "Χ": "X",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
}


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
    merge_continuation_threshold: float = 0.90
    merge_hyphen_threshold: float = 0.85
    document_class: str = "article"
    packages: tuple[str, ...] = (
        "graphicx",
        "amsmath",
        "amssymb",
        "mathrsfs",
        "booktabs",
        "hyperref",
        "float",
        "algorithm",
        "algpseudocode",
    )
    source_pdf: str | None = None
    table_asset_output_dir: str | None = None
    figure_asset_output_dir: str | None = None
    table_asset_latex_prefix: str = "assets"
    figure_asset_latex_prefix: str = "assets"
    heading_skeleton_mode: str = "legacy"


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
    parent_by_node: dict[int, int] = field(default_factory=dict)
    render_hints: dict[int, dict[str, Any]] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HeadingDecision:
    """Conservative render/structure decision for one title-like node."""

    is_structural: bool
    level: int
    render_as_paragraph: bool
    reason: str


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
        heading_mode = normalize_heading_skeleton_mode(self.config.heading_skeleton_mode)
        raw_nodes = {
            index: ResolvedNode(node_id=index, record=dict(record), merged_node_ids=[index])
            for index, record in enumerate(node_records)
        }
        raw_skeleton = None if heading_mode == "off" else build_heading_skeleton(raw_nodes, mode=heading_mode)
        contracted = self.contract_merge_nodes(
            node_records,
            edge_index,
            probs,
            raw_skeleton=raw_skeleton,
        )
        contracted = self.semantic_title_deduplication(contracted)
        if heading_mode == "off":
            parent_edges = self.maximum_parent_arborescence(contracted, edge_index, probs, skeleton=None)
            return self.build_tree(contracted, parent_edges)
        skeleton = build_heading_skeleton(contracted.nodes, mode=heading_mode)
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
            merge_threshold = self.merge_threshold_for(node_records[source], node_records[target])
            if merge_score < merge_threshold:
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

    def merge_threshold_for(self, node_u: dict[str, Any], node_v: dict[str, Any]) -> float:
        """Return an adaptive merge threshold for obvious paragraph continuations.

        The global threshold is often raised to protect precision.  That should
        not block near-certain typographic continuations, especially hyphenated
        line/page breaks such as ``fluctua-`` + ``tion``.  We only relax the
        threshold for adjacent reading-order pairs that already passed every
        physical hard gate in ``can_merge``.
        """

        threshold = float(self.config.merge_threshold)
        if not records_are_adjacent_in_reading_order(node_u, node_v):
            return threshold
        if record_ends_with_hyphen(node_u) and record_starts_like_continuation(node_v):
            return min(threshold, float(self.config.merge_hyphen_threshold))
        if record_is_open_sentence(node_u) and record_starts_like_continuation(node_v):
            return min(threshold, float(self.config.merge_continuation_threshold))
        return threshold

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
                node.record["_appendix_heading"] = True
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

        for node_id, hints in skeleton.render_hints.items():
            if node_id in nodes:
                nodes[node_id].record.update(hints)

        for child_id, parent_id in skeleton.parent_by_node.items():
            if child_id in nodes and parent_id in nodes and child_id != parent_id:
                parent_of[child_id] = int(parent_id)

        for heading_id, parent_id in skeleton.heading_parent.items():
            if heading_id not in parent_of and heading_id in nodes and parent_id in nodes:
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

        if normalize_heading_skeleton_mode(self.config.heading_skeleton_mode) == "stack":
            self.apply_strict_scope_defaults(nodes, parent_of, skeleton)

        for target_id, edge in best_parent_edge.items():
            if target_id in skeleton.heading_ids:
                continue
            if normalize_heading_skeleton_mode(self.config.heading_skeleton_mode) == "stack":
                # In stack mode the active heading state machine owns section
                # attachment.  GNN parent edges remain useful for diagnostics
                # and non-stack modes, but they must not re-parent body nodes
                # away from the current heading scope.
                if target_id in parent_of:
                    continue
                if not stack_mode_allows_local_parent(edge, nodes, skeleton):
                    continue
            elif target_id in parent_of:
                continue
            parent_of[target_id] = edge.source

        self.apply_scope_fallbacks(nodes, parent_of, skeleton)

        apply_float_caption_grouping(nodes, parent_of, skeleton)
        enforce_numbered_list_parent_continuity(nodes, parent_of)

        for child_id, parent_id in parent_of.items():
            if child_id in nodes and parent_id in nodes and child_id != parent_id:
                nodes[parent_id].children.append(nodes[child_id])

        root = ResolvedNode(node_id=-1, record={"type": "root", "text": "ROOT"}, merged_node_ids=[])
        for node_id in sorted(nodes, key=lambda idx: node_reading_order_key(nodes[idx])):
            if node_id not in parent_of:
                root.children.append(nodes[node_id])
        sort_tree_children(root)
        return root

    def apply_strict_scope_defaults(
        self,
        nodes: dict[int, ResolvedNode],
        parent_of: dict[int, int],
        skeleton: HeadingSkeleton,
    ) -> None:
        """Attach body nodes to the active heading before local GNN edges.

        In stack mode, the global heading state machine owns section scope.
        GNN parent edges may still refine local non-heading relations, but they
        do not get first chance to steal paragraphs across the outline.
        """

        for node_id in nodes:
            if node_id in parent_of or node_id in skeleton.heading_ids:
                continue
            if is_page_noise_node(nodes[node_id]):
                continue
            scope_id = skeleton.scope_by_node.get(node_id)
            if scope_id in nodes and scope_id != node_id:
                parent_of[node_id] = int(scope_id)

    def apply_scope_fallbacks(
        self,
        nodes: dict[int, ResolvedNode],
        parent_of: dict[int, int],
        skeleton: HeadingSkeleton,
    ) -> None:
        """Attach any remaining body nodes to their physical heading scope."""

        for node_id in nodes:
            if node_id in parent_of or node_id in skeleton.heading_ids:
                continue
            if is_page_noise_node(nodes[node_id]):
                continue
            scope_id = skeleton.scope_by_node.get(node_id)
            if scope_id in nodes and scope_id != node_id:
                parent_of[node_id] = int(scope_id)

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

    def render_document(
        self,
        root: ResolvedNode,
        *,
        title: str | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Stage 3: render a resolved tree to a compilable LaTeX document."""

        body_root = root_without_redundant_document_title(root, title) if title else root
        body_root = root_with_document_toc(body_root, document_metadata)
        apply_heading_render_policy(body_root)
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

        if node.record.get("_consumed_as_float_caption"):
            return ""
        if layout_layer_name(node.record) == "noise_layer":
            return ""
        block_type = canonical_render_type(node.record)
        text = node.text
        children = sorted_render_children(node.children)
        if is_algorithm_like_node(node.record, node_verbatim_text(node)):
            return render_algorithm_block(node_verbatim_text(node))
        if is_toc_title_node(node.record, text):
            return render_toc()
        if block_type == "toc":
            return ""
        if node.record.get("run_in_heading"):
            parts = render_run_in_heading_node(node, depth=depth)
            parts.extend(self.render_child_blocks_with_dynamic_lists(children, depth=depth + 1))
            return "\n\n".join(part for part in parts if part)
        if block_type == "title" and is_front_matter_date_text(text):
            parts = [render_textual_node(node)] if text else []
            parts.extend(self.render_child_blocks_with_dynamic_lists(children, depth=depth + 1))
            return "\n\n".join(part for part in parts if part)
        if block_type == "title":
            parts = [render_title(text, depth=depth, record=node.record)] if text else []
            parts.extend(self.render_child_blocks_with_dynamic_lists(children, depth=depth + 1))
            return "\n\n".join(part for part in parts if part)
        if block_type == "equation":
            parts = [render_equation(text)]
            parts.extend(self.render_child_blocks_with_dynamic_lists(children, depth=depth + 1))
            return "\n\n".join(part for part in parts if part)
        if block_type == "inline_math":
            parts = [render_inline_math(text)]
            parts.extend(self.render_child_blocks_with_dynamic_lists(children, depth=depth + 1))
            return "\n\n".join(part for part in parts if part)
        if block_type == "table":
            return render_table_placeholder(
                node.record,
                node_verbatim_text(node),
                node_id=node.node_id,
                source_pdf=self.config.source_pdf,
                asset_output_dir=self.config.table_asset_output_dir,
                asset_latex_prefix=self.config.table_asset_latex_prefix,
            )
        if block_type == "figure":
            if int(node.record.get("figure_group_size") or node.record.get("image_group_size") or 1) > 1 and node.record.get("figure_group_primary") is False:
                return ""
            raw_caption = node.record.get("figure_group_caption") or node.record.get("image_group_caption") or text
            caption = render_text_with_inline_latex(raw_caption) if raw_caption else "Figure"
            asset_path = ensure_figure_asset(
                node.record,
                source_pdf=self.config.source_pdf,
                asset_output_dir=self.config.figure_asset_output_dir or self.config.table_asset_output_dir,
                asset_latex_prefix=self.config.figure_asset_latex_prefix,
            )
            graphic_line = (
                rf"\includegraphics[width={figure_include_width(node.record)}\linewidth]{{{asset_path}}}"
                if asset_path
                else figure_placeholder(node.record, node_id=node.node_id)
            )
            return "\n".join(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    graphic_line,
                    rf"\caption{{{caption}}}",
                    r"\end{figure}",
                ]
            )
        if block_type == "reference":
            return render_references(node.record, text)
        if block_type == "footnote":
            return rf"\footnote{{{render_text_with_inline_latex(strip_note_marker(text)[0])}}}" if text else ""
        if block_type == "margin_note":
            return rf"\marginpar{{\footnotesize {render_text_with_inline_latex(strip_note_marker(text)[0])}}}" if text else ""
        if block_type == "list":
            return self.render_list(node, depth=depth)

        parts = [render_textual_node(node)] if text else []
        parts.extend(self.render_child_blocks_with_dynamic_lists(children, depth=depth + 1))
        return "\n\n".join(part for part in parts if part)

    def render_child_blocks_with_dynamic_lists(self, children: list[ResolvedNode], *, depth: int) -> list[str]:
        rendered: list[str] = []
        index = 0
        child_list = defer_sentence_interrupting_float_nodes(sorted_render_children(children))
        while index < len(child_list):
            child = child_list[index]
            if is_page_noise_node(child):
                index += 1
                continue
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
                    if is_page_noise_node(current):
                        index += 1
                        continue
                    current_environment = list_environment_for_node(current)
                    if current_environment is not None:
                        entries.append((item_node, continuations))
                        item_node = current
                        continuations = []
                        index += 1
                        continue
                    previous_continuation = continuations[-1] if continuations else None
                    if is_list_item_continuation_node(
                        current,
                        item_node=item_node,
                        previous_continuation=previous_continuation,
                    ):
                        continuations.append(current)
                        index += 1
                        continue
                    break
                entries.append((item_node, continuations))
                rendered.append(self.render_dynamic_list_entries(entries, environment=list_environment, depth=depth))
                continue
            implicit_items, next_index = collect_implicit_indented_list_after_lead(child, child_list, index + 1)
            if implicit_items:
                block = self.render_node(child, depth=depth).strip()
                if block:
                    append_nonredundant_rendered(rendered, block)
                append_nonredundant_rendered(rendered, self.render_dynamic_list_group(implicit_items, environment="itemize", depth=depth))
                index = next_index
                continue
            block = self.render_node(child, depth=depth).strip()
            if block:
                append_nonredundant_rendered(rendered, block)
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
            extra_nodes = unique_nodes_by_id([*item.children, *continuations])
            ordered_extra_nodes = defer_sentence_interrupting_float_nodes(sorted_render_children(extra_nodes))
            extra_blocks = [self.render_node(node, depth=depth + 1).strip() for node in ordered_extra_nodes]
            body_parts = [item_body, *extra_blocks]
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
        children = defer_sentence_interrupting_float_nodes(sorted_render_children(node.children))
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
            if is_page_noise_node(child):
                continue
            if list_environment_for_node(child) is not None:
                if item_node is not None:
                    entries.append((item_node, continuations))
                item_node = child
                continuations = []
                continue
            if item_node is not None and is_list_item_continuation_node(child, item_node=item_node):
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
    decoder = TreeDecoder()
    if has_layout_state_signals(node.record for node in contracted.nodes.values()):
        skeleton = build_heading_skeleton(contracted.nodes)
        return decoder.build_skeleton_tree(contracted, skeleton, non_merge_edges)
    return decoder.build_tree(contracted, non_merge_edges)


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


def normalize_heading_skeleton_mode(mode: str | None) -> str:
    value = str(mode or "legacy").casefold().replace("-", "_")
    aliases = {
        "none": "off",
        "disabled": "off",
        "disable": "off",
        "old": "off",
        "current": "legacy",
        "default": "legacy",
        "strict": "stack",
        "stack_strict": "stack",
    }
    value = aliases.get(value, value)
    if value not in {"off", "legacy", "stack"}:
        raise ValueError(f"Unknown heading skeleton mode: {mode!r}")
    return value


def build_heading_skeleton(nodes: dict[int, ResolvedNode], *, mode: str = "legacy") -> HeadingSkeleton:
    """Build a deterministic heading tree from physical reading order."""

    if not nodes:
        return HeadingSkeleton(frozenset(), {}, {}, {})
    mode = normalize_heading_skeleton_mode(mode)
    if mode == "off":
        return HeadingSkeleton(frozenset(), {}, {}, {})

    records_by_id = {node_id: node.record for node_id, node in nodes.items()}
    text_by_id = {node_id: node.text for node_id, node in nodes.items()}
    evidence_by_id = collect_heading_evidence(records_by_id, text_by_id=text_by_id)
    profile = learn_heading_style_profile(
        evidence_by_id,
        body_font_size=infer_body_font_size_from_nodes(nodes.values()),
    )
    for node_id, evidence in evidence_by_id.items():
        if node_id in nodes:
            nodes[node_id].record["_heading_evidence_score"] = round(float(evidence.score), 4)
            nodes[node_id].record["_heading_numbering_style"] = evidence.numbering_style
            if evidence.numbering_level is not None:
                nodes[node_id].record["_heading_numbering_level"] = evidence.numbering_level
    for node in nodes.values():
        node.record["_heading_profile_body_font_size"] = round(float(profile.body_font_size), 4)

    layout_result: LayoutParseResult | None = None
    if has_layout_state_signals(node.record for node in nodes.values()):
        layout_result = parse_layout_state_machine(
            {node_id: node.record for node_id, node in nodes.items()},
            text_by_id={node_id: node.text for node_id, node in nodes.items()},
        )
        for node_id, hints in layout_result.render_hints.items():
            if node_id in nodes:
                nodes[node_id].record.update(hints)

    if mode == "stack":
        return build_strict_heading_stack_skeleton(
            nodes,
            evidence_by_id=evidence_by_id,
            profile=profile,
            layout_result=layout_result,
        )

    if layout_result is not None:
        if layout_result.heading_ids or layout_result.parent_by_node:
            for node_id in layout_result.heading_ids:
                if node_id not in nodes:
                    continue
                nodes[node_id].record["_skeleton_heading_level"] = layout_result.heading_levels.get(node_id, 1)
                nodes[node_id].record["canonical_type"] = "title"
            return HeadingSkeleton(
                heading_ids=frozenset(layout_result.heading_ids),
                heading_levels=layout_result.heading_levels,
                heading_parent=layout_result.heading_parent,
                scope_by_node=layout_result.scope_by_node,
                parent_by_node=layout_result.parent_by_node,
                render_hints=layout_result.render_hints,
                events=layout_result.events,
            )

    ordered_ids = sorted(nodes, key=lambda node_id: node_reading_order_key(nodes[node_id]))
    body_font_size = infer_body_font_size_from_nodes(nodes.values())
    heading_decisions = resolve_heading_decisions(nodes, ordered_ids=ordered_ids, body_font_size=body_font_size)
    heading_ids: set[int] = set()
    heading_levels: dict[int, int] = {}
    heading_parent: dict[int, int | None] = {}
    scope_by_node: dict[int, int | None] = {}
    stack: list[tuple[int, int]] = []

    for order_pos, node_id in enumerate(ordered_ids):
        node = nodes[node_id]
        decision = heading_decisions.get(node_id)
        if decision is not None:
            apply_heading_decision(node, decision)
        if decision is not None and decision.is_structural:
            level = decision.level
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


def build_strict_heading_stack_skeleton(
    nodes: dict[int, ResolvedNode],
    *,
    evidence_by_id: dict[int, HeadingEvidence],
    profile: HeadingStyleProfile,
    layout_result: LayoutParseResult | None = None,
) -> HeadingSkeleton:
    """Build a document-global outline without using GNN parent edges.

    This is the production candidate for `--heading-skeleton-mode stack`:
    the stack owns heading parentage and section scope, while GNN edges may
    only refine local non-heading relations later in ``build_skeleton_tree``.
    """

    ordered_ids = sorted(nodes, key=lambda node_id: node_reading_order_key(nodes[node_id]))
    heading_ids: set[int] = set()
    heading_levels: dict[int, int] = {}
    heading_parent: dict[int, int | None] = {}
    scope_by_node: dict[int, int | None] = {}
    parent_by_node: dict[int, int] = {}
    render_hints: dict[int, dict[str, Any]] = {}
    events: list[str] = []
    stack: list[tuple[int, int]] = []
    seen_body_heading = False
    references_open = False
    appendix_open = False
    layout_heading_ids = set(layout_result.heading_ids) if layout_result is not None else set()
    layout_heading_levels = dict(layout_result.heading_levels) if layout_result is not None else {}

    for effective_pos, node_id in enumerate(ordered_ids):
        node = nodes[node_id]
        evidence = evidence_by_id.get(node_id)
        if is_page_noise_node(node):
            events.append(f"{node_id}:skip-noise")
            continue
        if strict_node_is_front_matter(node) and not seen_body_heading and not strict_is_abstract_heading(node):
            scope_by_node[node_id] = None
            events.append(f"{node_id}:front-matter")
            continue

        decision = strict_heading_stack_decision(
            node,
            evidence=evidence,
            profile=profile,
            effective_pos=effective_pos,
            current_level=stack[-1][0] if stack else 0,
            seen_body_heading=seen_body_heading,
            references_open=references_open,
            appendix_open=appendix_open,
            layout_heading_level=layout_heading_levels.get(node_id),
            layout_heading_signal=node_id in layout_heading_ids,
        )
        if decision is not None:
            apply_heading_decision(node, decision)
            if decision.is_structural:
                level = max(1, int(decision.level))
                node.record["_skeleton_heading_level"] = level
                node.record["_heading_render_level"] = level
                node.record["canonical_type"] = "title"
                heading_ids.add(node_id)
                heading_levels[node_id] = level
                if strict_is_references_heading(node):
                    references_open = True
                    appendix_open = False
                    stack.clear()
                elif strict_is_appendix_heading(node):
                    appendix_open = True
                    stack.clear()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                parent_id = stack[-1][1] if stack else None
                heading_parent[node_id] = parent_id
                if parent_id is not None:
                    parent_by_node[node_id] = parent_id
                scope_by_node[node_id] = node_id
                stack.append((level, node_id))
                seen_body_heading = True
                events.append(f"{node_id}:strict-heading level={level} parent={parent_id} reason={decision.reason}")
                continue
            render_hints[node_id] = dict(node.record)
            events.append(f"{node_id}:strict-non-heading reason={decision.reason}")

        current_scope = stack[-1][1] if stack else None
        scope_by_node[node_id] = current_scope
        if current_scope is not None and strict_node_participates_in_section_scope(node):
            parent_by_node[node_id] = current_scope
            events.append(f"{node_id}:strict-attach scope={current_scope}")
        else:
            events.append(f"{node_id}:strict-root-or-layer")

    return HeadingSkeleton(
        heading_ids=frozenset(heading_ids),
        heading_levels=heading_levels,
        heading_parent=heading_parent,
        scope_by_node=scope_by_node,
        parent_by_node=parent_by_node,
        render_hints=render_hints,
        events=events,
    )


def strict_heading_stack_decision(
    node: ResolvedNode,
    *,
    evidence: HeadingEvidence | None,
    profile: HeadingStyleProfile,
    effective_pos: int,
    current_level: int,
    seen_body_heading: bool,
    references_open: bool,
    appendix_open: bool,
    layout_heading_level: int | None = None,
    layout_heading_signal: bool = False,
) -> HeadingDecision | None:
    text = " ".join(node.text.split())
    if not text or is_page_noise_node(node):
        return None
    if is_toc_title_node(node.record, text) or strict_node_is_toc_entry(node):
        return HeadingDecision(False, 1, False, "toc")
    if strict_node_is_float_caption(node) or canonical_render_type(node.record) in {"figure", "table", "equation", "inline_math", "algorithm", "code"}:
        return None
    if node.record.get("run_in_heading"):
        level = int(numeric_value(node.record.get("run_in_heading_level")) or title_numbering_level(text) or max(2, current_level + 1 if current_level else 2))
        return HeadingDecision(True, min(max(1, level), 5), False, "run-in-heading")
    if strict_is_references_heading(node):
        return HeadingDecision(True, 1, False, "references-scope")
    if strict_is_appendix_heading(node):
        return HeadingDecision(True, 1, False, "appendix-scope")
    if strict_is_abstract_heading(node):
        return HeadingDecision(True, 1, False, "abstract-scope")
    if strict_is_acknowledgement_heading(node):
        return HeadingDecision(True, 1, False, "acknowledgement-scope")
    if strict_is_likely_document_title(node, evidence=evidence, effective_pos=effective_pos, seen_body_heading=seen_body_heading):
        return None
    if strict_rejects_heading_like_text(text):
        return None
    if strict_node_is_front_matter(node) and not seen_body_heading:
        return None
    if strict_node_is_list_item(node):
        if strict_numbered_heading_override(
            node,
            evidence=evidence,
            profile=profile,
            seen_body_heading=seen_body_heading,
            layout_heading_signal=layout_heading_signal,
        ):
            level = strict_heading_level(node, evidence=evidence, profile=profile, current_level=current_level, effective_pos=effective_pos)
            return HeadingDecision(True, level, False, "numbered-heading-over-list-role")
        return None
    if strict_node_is_reference_item(node):
        return None

    if evidence is None:
        return None
    prefix_kind = evidence.numbering_style
    prefix_level = evidence.numbering_level
    block_type = canonical_render_type(node.record)
    role = node_layout_role(node.record)
    standalone = looks_like_standalone_heading(text)
    physical_gate = strict_physical_heading_gate(node, evidence=evidence, profile=profile)
    numbered = prefix_level is not None and prefix_kind not in {"numeric_paren", "custom_colon"}
    strong_numbered = strict_has_strong_numbering_signal(evidence, node=node, layout_heading_signal=layout_heading_signal)

    if strong_numbered and standalone:
        level = strict_heading_level(node, evidence=evidence, profile=profile, current_level=current_level, effective_pos=effective_pos)
        return HeadingDecision(True, level, False, f"numbered-{prefix_kind}")
    if physical_gate and standalone and (block_type == "title" or role == "heading" or layout_heading_signal):
        level = strict_heading_level(node, evidence=evidence, profile=profile, current_level=current_level, effective_pos=effective_pos)
        reason = "physical-title-heading" if not layout_heading_signal else "layout-physical-heading"
        return HeadingDecision(True, level, False, reason)
    if numbered and not standalone:
        # Numbered but sentence-like nodes are list or run-in candidates, not
        # standalone section titles unless upstream has already split them.
        return HeadingDecision(False, max(1, prefix_level or 1), True, "numbered-non-standalone")
    if references_open or appendix_open:
        return None
    return None


def strict_heading_level(
    node: ResolvedNode,
    *,
    evidence: HeadingEvidence | None,
    profile: HeadingStyleProfile,
    current_level: int,
    effective_pos: int,
) -> int:
    text = " ".join(node.text.split())
    explicit = title_numbering_level(text)
    if explicit is not None:
        return max(1, min(explicit, 5))
    if evidence is not None and evidence.numbering_level is not None:
        if evidence.numbering_style == "alpha" and current_level >= 1:
            return max(2, min(current_level + 1, 5))
        return max(1, min(evidence.numbering_level, 5))
    raw_type = str(node.record.get("type") or node.record.get("raw_type") or node.record.get("block_type") or "").casefold()
    if raw_type == "section":
        return 1
    if raw_type == "subsection":
        return 2
    if raw_type == "subsubsection":
        return 3
    if evidence is not None:
        cluster = round(evidence.relative_font_size * 20.0) / 20.0
        if cluster in profile.level_by_font_cluster:
            return max(1, min(profile.level_by_font_cluster[cluster], 5))
        if evidence.relative_font_size >= 1.18:
            return 1
        if evidence.relative_font_size >= 1.05:
            return 2 if current_level else 1
    if is_local_subheading_layout(node.record):
        return 2 if current_level else 1
    return heading_stack_level_without_numbering(node, body_font_size=profile.body_font_size, order_pos=effective_pos)


def strict_style_supports_heading(evidence: HeadingEvidence, profile: HeadingStyleProfile) -> bool:
    if evidence.relative_font_size >= 1.10:
        return True
    cluster = round(evidence.relative_font_size * 20.0) / 20.0
    if cluster in profile.level_by_font_cluster and evidence.relative_font_size >= 1.03:
        return True
    return evidence.is_bold and evidence.is_line_isolated and evidence.relative_font_size >= 1.0


def strict_physical_heading_gate(
    node: ResolvedNode,
    *,
    evidence: HeadingEvidence | None,
    profile: HeadingStyleProfile,
) -> bool:
    """Physical gate for unnumbered heading promotion.

    Numbering can promote a heading on its own, but free-form headings must show
    a real document-local style jump.  This is the "font-size step" barrier the
    stack decoder uses to avoid bold run-in paragraph starts becoming sections.
    """

    relative = evidence.relative_font_size if evidence is not None else 0.0
    if relative >= 1.15:
        return True
    font_size = node_font_size(node.record)
    if profile.body_font_size > 0 and font_size >= profile.body_font_size * 1.15:
        return True
    return False


def strict_has_strong_numbering_signal(
    evidence: HeadingEvidence,
    *,
    node: ResolvedNode,
    layout_heading_signal: bool,
) -> bool:
    """Return true for numbering patterns strong enough to define a heading."""

    if evidence.numbering_level is None:
        return False
    if evidence.numbering_style in {"numeric_paren", "custom_colon"}:
        return False
    if evidence.numbering_style in {"dotted_numeric", "appendix_dotted", "bare_numbered", "roman", "numeric"}:
        return True
    if evidence.numbering_style == "alpha":
        return layout_heading_signal or canonical_render_type(node.record) == "title" or node_is_bold(node.record)
    return False


def strict_numbered_heading_override(
    node: ResolvedNode,
    *,
    evidence: HeadingEvidence | None,
    profile: HeadingStyleProfile,
    seen_body_heading: bool,
    layout_heading_signal: bool = False,
) -> bool:
    if evidence is None or evidence.numbering_level is None:
        return False
    text = " ".join(node.text.split())
    if evidence.numbering_style not in {"numeric", "dotted_numeric", "bare_numbered", "roman", "appendix_dotted", "alpha"}:
        return False
    if not looks_like_standalone_heading(text):
        return False
    if not strict_has_strong_numbering_signal(evidence, node=node, layout_heading_signal=layout_heading_signal):
        return False
    if canonical_render_type(node.record) == "title":
        return True
    if node_is_bold(node.record) and (evidence.relative_font_size >= 1.0 or seen_body_heading):
        return True
    return strict_physical_heading_gate(node, evidence=evidence, profile=profile)


def strict_node_participates_in_section_scope(node: ResolvedNode) -> bool:
    role = node_layout_role(node.record)
    if is_page_noise_node(node):
        return False
    if strict_node_is_front_matter(node):
        return role in {"abstract_body", "abstract_paragraph"}
    if strict_node_is_toc_entry(node):
        return False
    return True


def strict_node_is_front_matter(node: ResolvedNode) -> bool:
    layer = layout_layer_name(node.record).casefold()
    role = node_layout_role(node.record)
    if layer == "metadata_layer":
        return role in {
            "",
            "front_matter",
            "document_title",
            "front_matter_title",
            "author",
            "authors",
            "affiliation",
            "date",
            "email",
            "correspondence",
            "abstract",
            "abstract_title",
        }
    return role in {"document_title", "front_matter_title", "author", "authors", "affiliation", "date", "email", "correspondence"}


def strict_node_is_toc_entry(node: ResolvedNode) -> bool:
    role = node_layout_role(node.record)
    return role in {"toc", "toc_title", "toc_entry"} or canonical_render_type(node.record) == "toc"


def strict_node_is_float_caption(node: ResolvedNode) -> bool:
    return float_caption_kind(node) in {"figure", "table"}


def strict_node_is_list_item(node: ResolvedNode) -> bool:
    role = node_layout_role(node.record)
    return role in {"list_item", "list"} or canonical_render_type(node.record) == "list" or bool(node.record.get("_render_as_list_item"))


def strict_node_is_reference_item(node: ResolvedNode) -> bool:
    return canonical_render_type(node.record) == "reference" and not strict_is_references_heading(node)


def strict_rejects_heading_like_text(text: str) -> bool:
    value = " ".join(str(text or "").split())
    if not value:
        return True
    math_markers = value.count("[MATH]") + value.count("\\") + value.count("{") + value.count("}")
    if len(value) > 72 and math_markers >= 2:
        return True
    if value.startswith("[MATH]") and len(value) > 36:
        return True
    if len(value) > 150 and title_numbering_level(value) is None:
        return True
    return False


def strict_is_likely_document_title(
    node: ResolvedNode,
    *,
    evidence: HeadingEvidence | None,
    effective_pos: int,
    seen_body_heading: bool,
) -> bool:
    if seen_body_heading or effective_pos > 4:
        return False
    text = " ".join(node.text.split())
    if not text or title_numbering_level(text) is not None:
        return False
    role = node_layout_role(node.record)
    if role in {"document_title", "front_matter_title"}:
        return True
    if canonical_render_type(node.record) != "title":
        return False
    relative = evidence.relative_font_size if evidence is not None else 0.0
    return len(text) >= 32 and (relative >= 1.12 or node_is_bold(node.record))


def strict_is_abstract_heading(node: ResolvedNode) -> bool:
    normalized = normalize_structural_heading_text(node.text)
    role = node_layout_role(node.record)
    compact_text = " ".join(node.text.split())
    return normalized == "abstract" or role == "abstract_title" or (role == "abstract" and len(compact_text) <= 32)


def strict_is_references_heading(node: ResolvedNode) -> bool:
    normalized = normalize_structural_heading_text(node.text)
    return normalized in {"references", "bibliography"}


def strict_is_appendix_heading(node: ResolvedNode) -> bool:
    normalized = normalize_structural_heading_text(node.text)
    return normalized.startswith("appendix") or bool(node.record.get("_appendix_heading"))


def strict_is_acknowledgement_heading(node: ResolvedNode) -> bool:
    normalized = normalize_structural_heading_text(node.text).replace(" ", "")
    return normalized in {"acknowledgment", "acknowledgement", "acknowledgments", "acknowledgements"}


def has_layout_state_signals(records: Any) -> bool:
    signal_keys = {
        "layout_role",
        "layout_layer",
        "layout_band_id",
        "layout_band_type",
        "layout_band_column",
        "layout_flow_order",
        "regime_reading_order",
        "dag_reading_order",
        "column_fix_span",
        "column_fix_column",
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        if any(key in record for key in signal_keys):
            return True
    return False


def apply_heading_render_policy(root: ResolvedNode) -> None:
    """Mark uncertain title-like nodes as paragraph headings before rendering."""

    nodes = {index: node for index, node in enumerate(iter_resolved_tree_nodes(root)) if node.node_id != -1}
    if not nodes:
        return
    ordered_ids = sorted(nodes, key=lambda node_id: node_reading_order_key(nodes[node_id]))
    body_font_size = infer_body_font_size_from_nodes(nodes.values())
    for node_id, decision in resolve_heading_decisions(nodes, ordered_ids=ordered_ids, body_font_size=body_font_size).items():
        if nodes[node_id].record.get("_layout_state_locked") or nodes[node_id].record.get("_skeleton_heading_level") is not None:
            continue
        apply_heading_decision(nodes[node_id], decision)


def iter_resolved_tree_nodes(root: ResolvedNode) -> list[ResolvedNode]:
    nodes = [root]
    for child in sorted_render_children(root.children):
        nodes.extend(iter_resolved_tree_nodes(child))
    return nodes


def resolve_heading_decisions(
    nodes: dict[int, ResolvedNode],
    *,
    ordered_ids: list[int],
    body_font_size: float,
) -> dict[int, HeadingDecision]:
    """Resolve title-like nodes by style clusters and prefix-pattern stability.

    This deliberately avoids domain-specific prefixes such as "Q1" or "Case".
    It asks whether a node belongs to a stable visual/prefix system inside the
    current document.  Unstable or custom-prefixed title nodes become safe
    ``paragraph*`` headings instead of exploding the section tree.
    """

    candidates: list[tuple[int, int]] = []
    style_groups: dict[tuple[float, int], list[int]] = {}
    prefix_by_id: dict[int, tuple[str, int | None]] = {}
    order_lookup = {node_id: pos for pos, node_id in enumerate(ordered_ids)}

    for node_id in ordered_ids:
        node = nodes[node_id]
        order_pos = order_lookup.get(node_id, 0)
        if not is_heading_candidate_node(node, body_font_size=body_font_size, order_pos=order_pos):
            continue
        candidates.append((node_id, order_pos))
        style_groups.setdefault(heading_style_key(node, body_font_size=body_font_size), []).append(node_id)
        prefix_by_id[node_id] = heading_prefix_signature(node.text)

    dominant_prefix: dict[tuple[float, int], str] = {}
    mixed_prefix: set[tuple[float, int]] = set()
    for style_key, group_ids in style_groups.items():
        counts: dict[str, int] = {}
        for node_id in group_ids:
            prefix_kind, _ = prefix_by_id[node_id]
            counts[prefix_kind] = counts.get(prefix_kind, 0) + 1
        if counts:
            dominant_prefix[style_key] = max(counts.items(), key=lambda item: (item[1], item[0] == "freeform"))[0]
        if len(counts) > 1:
            mixed_prefix.add(style_key)

    decisions: dict[int, HeadingDecision] = {}
    for node_id, order_pos in candidates:
        node = nodes[node_id]
        style_key = heading_style_key(node, body_font_size=body_font_size)
        prefix_kind, prefix_level = prefix_by_id[node_id]
        special = is_special_structural_title(node.text)
        level = heading_decision_level(
            node,
            prefix_kind=prefix_kind,
            prefix_level=prefix_level,
            body_font_size=body_font_size,
            order_pos=order_pos,
        )

        if is_numbered_list_continuation_heading(node_id, nodes=nodes, ordered_ids=ordered_ids):
            decisions[node_id] = HeadingDecision(
                is_structural=False,
                level=max(1, level),
                render_as_paragraph=False,
                reason="numbered-list-continuation",
            )
            continue

        demote_reason = ""
        if not special:
            if prefix_kind in {"custom_colon", "numeric_paren"}:
                demote_reason = "custom-prefix"
            elif (
                style_key in mixed_prefix
                and prefix_kind != dominant_prefix.get(style_key)
                and prefix_kind not in {"freeform", "numeric", "bare_numbered", "dotted_numeric", "appendix_dotted", "roman"}
            ):
                demote_reason = "style-prefix-shift"
            elif (
                prefix_kind == "freeform"
                and node_layout_role(node.record) != "heading"
                and heading_font_ratio(node, body_font_size=body_font_size) <= 1.04
                and node_is_bold(node.record)
            ):
                demote_reason = "body-size-bold-heading"

        if demote_reason:
            decisions[node_id] = HeadingDecision(
                is_structural=False,
                level=max(1, level),
                render_as_paragraph=True,
                reason=demote_reason,
            )
        else:
            decisions[node_id] = HeadingDecision(
                is_structural=True,
                level=max(0, level),
                render_as_paragraph=False,
                reason="structural",
            )
    return decisions


def heading_decision_level(
    node: ResolvedNode,
    *,
    prefix_kind: str,
    prefix_level: int | None,
    body_font_size: float,
    order_pos: int,
) -> int:
    """Resolve heading depth with document-global style before prefix shortcuts.

    A bare numeric prefix like ``3.`` is ambiguous in papers: it can be a real
    section number, but MinerU also emits run-in/list headings as title blocks.
    When the layout classifier already says the title-like node is a list item,
    keep it structural but demote it under the active section instead of letting
    the prefix alone make it a top-level ``section``.
    """

    if prefix_kind == "numeric" and node_layout_role(node.record) in {"list_item", "list"}:
        return max(2, min(3, heading_stack_level_without_numbering(node, body_font_size=body_font_size, order_pos=order_pos)))
    if (
        prefix_kind == "freeform"
        and node_layout_role(node.record) == "heading"
        and canonical_render_type(node.record) == "title"
        and is_local_subheading_layout(node.record)
    ):
        return 2
    if prefix_level is not None:
        return prefix_level
    return heading_stack_level(node, body_font_size=body_font_size, order_pos=order_pos)


def apply_heading_decision(node: ResolvedNode, decision: HeadingDecision) -> None:
    node.record["_heading_policy_reason"] = decision.reason
    if decision.reason == "numbered-list-continuation":
        node.record.pop("_render_as_paragraph_heading", None)
        node.record.pop("_heading_unnumbered", None)
        node.record.pop("_heading_render_level", None)
        node.record["_render_as_list_item"] = True
        return
    node.record.pop("_render_as_list_item", None)
    if decision.render_as_paragraph:
        node.record["_heading_render_level"] = int(decision.level)
        node.record["_render_as_paragraph_heading"] = True
        node.record.pop("_heading_unnumbered", None)
    else:
        node.record.pop("_render_as_paragraph_heading", None)
        prefix_kind, _ = heading_prefix_signature(node.text)
        if "_skeleton_heading_level" in node.record or prefix_kind != "freeform":
            node.record["_heading_render_level"] = int(decision.level)
        else:
            node.record.pop("_heading_render_level", None)
        if (
            prefix_kind == "freeform"
            and node_layout_role(node.record) == "heading"
            and is_local_subheading_layout(node.record)
        ):
            node.record["_heading_unnumbered"] = True
        else:
            node.record.pop("_heading_unnumbered", None)


def heading_style_key(node: ResolvedNode, *, body_font_size: float) -> tuple[float, int]:
    ratio = heading_font_ratio(node, body_font_size=body_font_size)
    return (round(ratio * 20.0) / 20.0, int(node_is_bold(node.record)))


def heading_font_ratio(node: ResolvedNode, *, body_font_size: float) -> float:
    font_size = node_font_size(node.record)
    if body_font_size > 0 and font_size > 0:
        return font_size / body_font_size
    return font_size


def heading_prefix_signature(text: str) -> tuple[str, int | None]:
    value = " ".join(str(text or "").split())
    if not value:
        return ("empty", None)
    if APPENDIX_DOTTED_PREFIX_RE.match(value):
        token = value.split(maxsplit=1)[0].rstrip(".")
        return ("appendix_dotted", max(2, token.count(".") + 1))
    if DOTTED_NUMERIC_PREFIX_RE.match(value):
        token = value.split(maxsplit=1)[0].rstrip(".")
        return ("dotted_numeric", max(2, token.count(".") + 1))
    if NUMERIC_PAREN_PREFIX_RE.match(value):
        return ("numeric_paren", 2)
    if NUMERIC_PREFIX_RE.match(value):
        return ("numeric", 1)
    numbered_level = title_numbering_level(value)
    if numbered_level is not None:
        return ("bare_numbered", numbered_level)
    if ROMAN_PREFIX_RE.match(value):
        return ("roman", 1)
    if ALPHA_PREFIX_RE.match(value):
        return ("alpha", 2)
    if CUSTOM_COLON_PREFIX_RE.match(value):
        return ("custom_colon", None)
    return ("freeform", None)


def is_numbered_list_continuation_heading(
    node_id: int,
    *,
    nodes: dict[int, ResolvedNode],
    ordered_ids: list[int],
    max_effective_gap: int = 12,
) -> bool:
    node = nodes[node_id]
    current_number = ordered_numeric_dot_number(node.text)
    if current_number is None or current_number <= 1:
        return False
    if node_layout_role(node.record) not in {"list_item", "list"}:
        return False
    try:
        current_position = ordered_ids.index(node_id)
    except ValueError:
        return False
    effective_gap = 0
    for previous_id in reversed(ordered_ids[:current_position]):
        previous = nodes[previous_id]
        if is_page_noise_node(previous):
            continue
        effective_gap += 1
        if effective_gap > max_effective_gap:
            break
        previous_number = ordered_numeric_dot_number(previous.text)
        if previous_number == current_number - 1 and record_is_numbered_list_like(previous.record, previous.text):
            return True
        if previous_number == 1 and current_number > 2:
            break
    return False


def ordered_numeric_dot_number(text: str) -> int | None:
    match = ORDERED_NUMERIC_DOT_PREFIX_RE.match(" ".join(str(text or "").split()))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def record_is_numbered_list_like(record: dict[str, Any], text: str) -> bool:
    if record.get("_render_as_list_item"):
        return True
    if node_layout_role(record) in {"list_item", "list"} and ordered_numeric_dot_number(text) is not None:
        return True
    return False


def is_special_structural_title(text: str) -> bool:
    normalized = normalize_structural_heading_text(text)
    return normalized in {"abstract", "references", "bibliography"} or normalized.startswith("appendix")


def is_heading_candidate_node(node: ResolvedNode, *, body_font_size: float, order_pos: int) -> bool:
    record = node.record
    text = " ".join(node.text.split())
    if not text or is_page_noise_node(node):
        return False
    if record.get("run_in_heading"):
        return True
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
    return heading_stack_level_without_numbering(node, body_font_size=body_font_size, order_pos=order_pos)


def heading_stack_level_without_numbering(node: ResolvedNode, *, body_font_size: float, order_pos: int) -> int:
    text = " ".join(node.text.split())

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


def node_layout_role(record: dict[str, Any]) -> str:
    return str(
        record.get("layout_role")
        or record.get("role")
        or record.get("semantic_role")
        or ""
    ).casefold()


def is_local_subheading_layout(record: dict[str, Any]) -> bool:
    band_type = str(record.get("layout_band_type") or "").casefold()
    band_column = str(record.get("layout_band_column") or "").casefold()
    column_id = numeric_value(record.get("layout_band_column_id"))
    boundary = bool(record.get("layout_is_band_boundary"))
    if not band_type and not band_column and column_id is None:
        return False
    if band_type == "double_column":
        return True
    if band_column in {"left", "right"}:
        return True
    if column_id is not None and int(column_id) in {0, 1}:
        return True
    return not boundary and band_type not in {"full_span", "single_column"}


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
    source_type = canonical_render_type(source.record)
    target_type = canonical_render_type(target.record)
    if source_type in NON_PARENT_RENDER_TYPES:
        return True
    if target_id in skeleton.heading_ids:
        return True
    if source_id in skeleton.heading_ids:
        return skeleton.scope_by_node.get(target_id) != source_id
    if target_type == "title":
        return True
    return skeleton.scope_by_node.get(source_id) != skeleton.scope_by_node.get(target_id)


def stack_mode_allows_local_parent(
    edge: DecodedEdge,
    nodes: dict[int, ResolvedNode],
    skeleton: HeadingSkeleton,
) -> bool:
    """Return true when a GNN parent edge is a local refinement, not outline structure."""

    source = nodes.get(edge.source)
    target = nodes.get(edge.target)
    if source is None or target is None:
        return False
    source_id = source.node_id
    target_id = target.node_id
    if source_id in skeleton.heading_ids or target_id in skeleton.heading_ids:
        return False
    if skeleton.scope_by_node.get(source_id) != skeleton.scope_by_node.get(target_id):
        return False

    source_type = canonical_render_type(source.record)
    target_type = canonical_render_type(target.record)
    target_caption_kind = float_caption_kind(target)
    if source_type in {"figure", "table"} and target_caption_kind == source_type:
        return True
    if target_type in {"equation", "inline_math", "algorithm", "code"} and source_type in {"text", "list", "reference"}:
        return True
    if source_type == "list" and target_type in {"text", "equation", "inline_math"}:
        return True
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


def defer_sentence_interrupting_float_nodes(children: list[ResolvedNode]) -> list[ResolvedNode]:
    """Move floats after a local text continuation when they split an open sentence."""

    if len(children) < 3:
        return children

    reordered: list[ResolvedNode] = []
    index = 0
    changed = False
    while index < len(children):
        child = children[index]
        previous = reordered[-1] if reordered else None
        next_node = children[index + 1] if index + 1 < len(children) else None
        if previous is not None and next_node is not None and is_sentence_interrupting_float_node(previous, child, next_node):
            floating_nodes: list[ResolvedNode] = []
            while index < len(children) and is_float_render_node(children[index]):
                floating_nodes.append(children[index])
                index += 1

            continuation_nodes: list[ResolvedNode] = []
            while index < len(children):
                candidate = children[index]
                if is_float_render_node(candidate) or not is_text_flow_render_node(candidate):
                    break
                if not record_starts_like_continuation(candidate.record):
                    break
                continuation_nodes.append(candidate)
                index += 1
                if not record_is_open_sentence(candidate.record):
                    break

            if continuation_nodes:
                reordered.extend(continuation_nodes)
                reordered.extend(floating_nodes)
                changed = True
            else:
                reordered.extend(floating_nodes)
            continue

        reordered.append(child)
        index += 1
    return reordered if changed else children


def is_sentence_interrupting_float_node(previous: ResolvedNode, floating: ResolvedNode, next_node: ResolvedNode) -> bool:
    return (
        is_text_flow_render_node(previous)
        and is_float_render_node(floating)
        and is_text_flow_render_node(next_node)
        and record_is_open_sentence(previous.record)
        and record_starts_like_continuation(next_node.record)
    )


def is_float_render_node(node: ResolvedNode) -> bool:
    return canonical_render_type(node.record) in {"table", "figure"}


def is_text_flow_render_node(node: ResolvedNode) -> bool:
    if is_page_noise_node(node):
        return False
    return canonical_render_type(node.record) in {"text", "list", "reference"}


def unique_nodes_by_id(nodes: list[ResolvedNode]) -> list[ResolvedNode]:
    seen: set[int] = set()
    unique: list[ResolvedNode] = []
    for node in nodes:
        key = node.node_id
        if key in seen:
            continue
        seen.add(key)
        unique.append(node)
    return unique


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
    if raw in {"toc", "toc_title", "toc_entry", "index", "table_of_contents"}:
        return "toc"
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
    if raw in {"page_footnote", "footnote", "foot_note"}:
        return "footnote"
    if raw in {"margin_note", "marginnote", "side_note", "sidenote", "sidebar"}:
        return "margin_note"
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
    if record_is_author_biography_or_backmatter(left) or record_is_author_biography_or_backmatter(right):
        return False
    if not same_layout_scope_can_contract_merge(left, right):
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


def records_are_adjacent_in_reading_order(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_index = node_physical_index(left)
    right_index = node_physical_index(right)
    if left_index is None or right_index is None:
        return False
    return 0 < right_index - left_index <= 1.01


def record_ends_with_hyphen(record: dict[str, Any]) -> bool:
    return bool(MERGE_TRAILING_HYPHEN_RE.search(" ".join(node_record_text(record).split())))


def record_is_open_sentence(record: dict[str, Any]) -> bool:
    text = " ".join(node_record_text(record).split())
    if not text:
        return False
    if record_ends_with_hyphen(record):
        return True
    return MERGE_TERMINAL_PUNCT_RE.search(text) is None


def record_starts_like_continuation(record: dict[str, Any]) -> bool:
    text = " ".join(node_record_text(record).split())
    return bool(text and MERGE_CONTINUATION_START_RE.match(text))


def same_layout_scope_can_contract_merge(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_layer = layout_layer_name(left)
    right_layer = layout_layer_name(right)
    if left_layer == "noise_layer" or right_layer == "noise_layer":
        return False
    if left_layer != right_layer:
        return False
    if left_layer not in {"main_text_flow", "math_layer"} and canonical_render_type(left) != "reference":
        return False
    left_band = layout_band_id(left)
    right_band = layout_band_id(right)
    if left_band is not None and right_band is not None and left_band != right_band:
        return False
    return True


def layout_layer_name(record: dict[str, Any]) -> str:
    return str(record.get("layout_layer") or "main_text_flow")


def record_is_author_biography_or_backmatter(record: dict[str, Any]) -> bool:
    raw = str(record.get("canonical_type") or record.get("type") or record.get("raw_type") or "").casefold()
    role = node_layout_role(record)
    layer = layout_layer_name(record).casefold()
    list_type = str(record.get("list_type") or "").casefold()
    haystack = " ".join((raw, role, layer, list_type)).replace("-", "_").replace(" ", "_")
    if any(token in haystack for token in AUTHOR_BIOGRAPHY_ROLE_TOKENS):
        return True
    return record_text_looks_like_author_biography(node_record_text(record))


def record_text_looks_like_author_biography(text: str) -> bool:
    compact = " ".join(str(text or "").split())
    if len(compact) < 60:
        return False
    first_clause = compact[:140]
    if not re.match(r"^[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,4}\s+(?:is|was|received|earned)\b", first_clause):
        return False
    lower = compact.casefold()
    return any(
        signal in lower
        for signal in (
            " is a ",
            " is the ",
            " received ",
            " earned ",
            " ph.d",
            " m.s.",
            " b.s.",
            " university",
            " research",
        )
    )


def layout_band_id(record: dict[str, Any]) -> int | None:
    value = record.get("layout_band_global_id")
    if value is None:
        value = record.get("layout_band_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def is_list_item_continuation_node(
    node: ResolvedNode,
    *,
    item_node: ResolvedNode | None = None,
    previous_continuation: ResolvedNode | None = None,
) -> bool:
    """Allow display objects to stay inside the current list item."""

    block_type = canonical_render_type(node.record)
    if block_type in {"equation", "inline_math", "table", "figure", "algorithm", "code"}:
        return True
    if previous_continuation is not None and is_post_display_list_continuation(node, previous_continuation):
        return True
    if item_node is not None and item_node.record.get("_render_as_list_item") and block_type == "text":
        if list_environment_for_node(node) is not None:
            return False
        if node_layout_role(node.record) == "heading":
            return False
        return is_hanging_list_continuation(node, item_node)
    return False


def is_post_display_list_continuation(node: ResolvedNode, previous: ResolvedNode) -> bool:
    if canonical_render_type(node.record) != "text":
        return False
    if list_environment_for_node(node) is not None:
        return False
    if node_layout_role(node.record) == "heading":
        return False
    previous_type = canonical_render_type(previous.record)
    if previous_type not in {"equation", "inline_math", "table", "figure", "algorithm", "code"}:
        return False
    text = " ".join(node.text.split())
    if not text:
        return False
    return bool(re.match(r"^(where|which|this|these|therefore|thus|for example|for instance)\b", text, re.IGNORECASE))


def is_hanging_list_continuation(node: ResolvedNode, item_node: ResolvedNode) -> bool:
    node_bbox = merge_barrier_bbox(node.record)
    item_bbox = merge_barrier_bbox(item_node.record)
    if node_bbox is None or item_bbox is None:
        return False
    page = merge_barrier_page(node.record)
    item_page = merge_barrier_page(item_node.record)
    if page is not None and item_page is not None and page != item_page:
        return False
    page_width = max(1.0, merge_barrier_page_width(node.record, item_node.record) or 1000.0)
    indent_threshold = max(12.0, 0.018 * page_width)
    if node_bbox[0] >= item_bbox[0] + indent_threshold:
        return True
    item_text = " ".join(item_node.text.split())
    node_text = " ".join(node.text.split())
    if item_text and not item_text.rstrip().endswith((".", "?", "!", "。", "？", "！", ":")) and node_text[:1].islower():
        vertical_gap = node_bbox[1] - item_bbox[3]
        return -2.0 <= vertical_gap <= max(24.0, 0.04 * page_width) and node_bbox[0] >= item_bbox[0] - 4.0
    return False


def list_environment_for_node(node: ResolvedNode) -> str | None:
    record = node.record
    block_type = canonical_render_type(record)
    text = node.text
    if record.get("_render_as_list_item"):
        return list_environment_for_record(record, fallback_text=text)
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


def strip_note_marker(text: str) -> tuple[str, str | None]:
    value = str(text or "").strip()
    match = NOTE_MARKER_RE.match(value)
    if not match:
        return value, None
    marker = next((group for group in match.groups() if group), None)
    return value[match.end() :].strip(), marker


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
    normalized = normalize_structural_heading_text(node.text)
    if "appendix" in normalized.split():
        return True
    return looks_like_appendix_letter_title(node.text)


def looks_like_appendix_letter_title(text: str) -> bool:
    """Return true for appendix headings like ``A Details`` after References.

    Some conference styles switch into ``\appendix`` and render headings as
    ``A Title`` without the word "Appendix".  This helper is intentionally used
    only after a References/Bibliography anchor so ordinary early headings such
    as "A Study ..." are not globally treated as appendix material.
    """

    value = " ".join(str(text or "").strip().split())
    if not value:
        return False
    return re.match(r"^[A-Z](?:\.\d+)*\.?\s+\S", value) is not None


def is_page_noise_node(node: ResolvedNode) -> bool:
    if node_layout_role(node.record) == "noise" or layout_layer_name(node.record) == "noise_layer":
        return True
    raw_type = str(
        node.record.get("type")
        or node.record.get("raw_type")
        or node.record.get("block_type")
        or node.record.get("canonical_type")
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


def apply_float_caption_grouping(
    nodes: dict[int, ResolvedNode],
    parent_of: dict[int, int],
    skeleton: HeadingSkeleton,
) -> None:
    """Attach visual float captions to nearby floats before tree materialization.

    GNN parent edges are deliberately conservative around floating objects: TeX
    source order and PDF placement often disagree.  This pass uses only physical
    layout evidence plus shallow caption labels and writes the caption back onto
    the primary float record, so the IR renderer emits a single figure/table
    block rather than a loose caption paragraph.
    """

    if not nodes:
        return
    float_ids = [
        node_id
        for node_id, node in nodes.items()
        if canonical_render_type(node.record) in {"figure", "table"} and not is_page_noise_node(node)
    ]
    if not float_ids:
        return

    caption_ids = [
        node_id
        for node_id in sorted(nodes, key=lambda item: node_reading_order_key(nodes[item]))
        if float_caption_kind(nodes[node_id]) in {"figure", "table"} and not is_page_noise_node(nodes[node_id])
    ]
    consumed_caption_ids: set[int] = set()
    for caption_id in caption_ids:
        if caption_id in consumed_caption_ids:
            continue
        caption_node = nodes[caption_id]
        kind = float_caption_kind(caption_node)
        if kind not in {"figure", "table"}:
            continue
        group = nearest_float_group_for_caption(caption_node, kind=kind, nodes=nodes, float_ids=float_ids)
        if not group:
            continue
        primary_id = choose_float_group_primary(nodes, group, caption_node)
        write_float_group_metadata(nodes, group, primary_id=primary_id, caption_node=caption_node, kind=kind)
        caption_node.record["_consumed_as_float_caption"] = True
        caption_node.record["canonical_type"] = "caption"
        parent_of[caption_id] = primary_id
        consumed_caption_ids.add(caption_id)
        for member_id in group:
            if member_id != primary_id:
                parent_of[member_id] = primary_id
        scope_id = skeleton.scope_by_node.get(caption_id)
        if scope_id in nodes and primary_id not in parent_of:
            parent_of[primary_id] = int(scope_id)


def float_caption_kind(node: ResolvedNode) -> str | None:
    role = node_layout_role(node.record)
    if "figure_caption" in role or role in {"caption", "image_caption"}:
        return "figure"
    if "table_caption" in role:
        return "table"
    match = FLOAT_CAPTION_RE.match(" ".join(node.text.split()))
    if not match:
        return None
    kind = match.group("kind").casefold().rstrip(".")
    if kind in {"figure", "fig"}:
        return "figure"
    if kind in {"table", "tab"}:
        return "table"
    return None


def nearest_float_group_for_caption(
    caption_node: ResolvedNode,
    *,
    kind: str,
    nodes: dict[int, ResolvedNode],
    float_ids: list[int],
) -> list[int]:
    caption_box = merge_barrier_bbox(caption_node.record)
    caption_page = merge_barrier_page(caption_node.record)
    if caption_box is None:
        return []
    candidates: list[tuple[float, int]] = []
    for float_id in float_ids:
        float_node = nodes[float_id]
        if canonical_render_type(float_node.record) != kind:
            continue
        if caption_page is not None and merge_barrier_page(float_node.record) not in {None, caption_page}:
            continue
        float_box = merge_barrier_bbox(float_node.record)
        if float_box is None:
            continue
        score = caption_float_distance_score(caption_box, float_box, caption_node.record, float_node.record)
        if score is not None:
            candidates.append((score, float_id))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0])
    primary_id = candidates[0][1]
    primary_box = merge_barrier_bbox(nodes[primary_id].record)
    if primary_box is None:
        return [primary_id]
    group = {primary_id}
    for score, candidate_id in candidates[1:]:
        candidate_box = merge_barrier_bbox(nodes[candidate_id].record)
        if candidate_box is None:
            continue
        if score > max(140.0, 0.18 * max(float_page_height(caption_node.record), 1.0)):
            continue
        if floats_share_caption_band(primary_box, candidate_box, caption_box, caption_node.record):
            group.add(candidate_id)
    return sorted(group, key=lambda node_id: node_reading_order_key(nodes[node_id]))


def caption_float_distance_score(
    caption_box: tuple[float, float, float, float],
    float_box: tuple[float, float, float, float],
    caption_record: dict[str, Any],
    float_record: dict[str, Any],
) -> float | None:
    page_width = max(float_page_width(caption_record, float_record), 1.0)
    page_height = max(float_page_height(caption_record, float_record), 1.0)
    x_overlap = bbox_x_overlap_ratio(caption_box, float_box)
    x_gap = bbox_x_gap(caption_box, float_box)
    below_gap = caption_box[1] - float_box[3]
    above_gap = float_box[1] - caption_box[3]
    vertical_gap = below_gap if below_gap >= 0 else above_gap if above_gap >= 0 else 0.0
    if vertical_gap > max(160.0, 0.20 * page_height):
        return None
    if x_overlap < 0.08 and x_gap > max(80.0, 0.12 * page_width):
        return None
    caption_center_x = (caption_box[0] + caption_box[2]) / 2.0
    float_center_x = (float_box[0] + float_box[2]) / 2.0
    alignment_penalty = 0.0 if x_overlap >= 0.25 else min(abs(caption_center_x - float_center_x) / page_width, 1.0) * 40.0
    return vertical_gap + alignment_penalty


def floats_share_caption_band(
    primary_box: tuple[float, float, float, float],
    candidate_box: tuple[float, float, float, float],
    caption_box: tuple[float, float, float, float],
    caption_record: dict[str, Any],
) -> bool:
    page_width = max(float_page_width(caption_record), 1.0)
    page_height = max(float_page_height(caption_record), 1.0)
    if y_overlap_ratio(primary_box, candidate_box) >= 0.18:
        return True
    primary_gap = min(abs(caption_box[1] - primary_box[3]), abs(primary_box[1] - caption_box[3]))
    candidate_gap = min(abs(caption_box[1] - candidate_box[3]), abs(candidate_box[1] - caption_box[3]))
    if abs(candidate_gap - primary_gap) > max(55.0, 0.08 * page_height):
        return False
    union_width = max(primary_box[2], candidate_box[2]) - min(primary_box[0], candidate_box[0])
    return union_width <= page_width * 0.98


def choose_float_group_primary(
    nodes: dict[int, ResolvedNode],
    group: list[int],
    caption_node: ResolvedNode,
) -> int:
    caption_order = node_physical_index(caption_node)
    if caption_order is not None:
        before_caption = [
            node_id
            for node_id in group
            if (node_physical_index(nodes[node_id]) is not None and node_physical_index(nodes[node_id]) <= caption_order)
        ]
        if before_caption:
            return max(before_caption, key=lambda node_id: node_reading_order_key(nodes[node_id]))
    return max(group, key=lambda node_id: node_reading_order_key(nodes[node_id]))


def write_float_group_metadata(
    nodes: dict[int, ResolvedNode],
    group: list[int],
    *,
    primary_id: int,
    caption_node: ResolvedNode,
    kind: str,
) -> None:
    boxes = [merge_barrier_bbox(nodes[node_id].record) for node_id in group]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return
    union = [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
    caption = " ".join(caption_node.text.split())
    group_id = existing_float_group_id(nodes, group, kind) or f"{kind}_group_auto_{primary_id}"
    member_ids = sorted(group, key=lambda node_id: node_reading_order_key(nodes[node_id]))
    prefix = "figure" if kind == "figure" else "table"
    for member_index, node_id in enumerate(member_ids):
        record = nodes[node_id].record
        record[f"{prefix}_group_id"] = group_id
        record[f"{prefix}_group_member_index"] = member_index
        record[f"{prefix}_group_size"] = len(member_ids)
        record[f"{prefix}_group_primary"] = node_id == primary_id
        record[f"{prefix}_group_bbox"] = list(union)
        record[f"{prefix}_group_caption"] = caption
        record[f"{prefix}_group_source_node_ids"] = member_ids
        if kind == "figure":
            record["image_group_id"] = group_id
            record["image_group_member_index"] = member_index
            record["image_group_size"] = len(member_ids)
            record["image_group_primary"] = node_id == primary_id
            record["image_group_bbox"] = list(union)
            record["image_group_caption"] = caption
            record["image_group_source_node_ids"] = member_ids
        if node_id == primary_id:
            record["merged_text"] = caption


def existing_float_group_id(nodes: dict[int, ResolvedNode], group: list[int], kind: str) -> str | None:
    keys = ("figure_group_id", "image_group_id") if kind == "figure" else ("table_group_id",)
    for node_id in group:
        for key in keys:
            value = nodes[node_id].record.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return None


def bbox_x_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    min_width = max(min(left[2] - left[0], right[2] - right[0]), 1e-6)
    return intersection / min_width


def y_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    min_height = max(min(left[3] - left[1], right[3] - right[1]), 1e-6)
    return intersection / min_height


def bbox_x_gap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    return max(max(left[0], right[0]) - min(left[2], right[2]), 0.0)


def float_page_width(*records: dict[str, Any]) -> float:
    for record in records:
        value = numeric_value(record.get("page_width"))
        if value is not None and value > 0:
            return value
    return 1000.0


def float_page_height(*records: dict[str, Any]) -> float:
    for record in records:
        value = numeric_value(record.get("page_height"))
        if value is not None and value > 0:
            return value
    return 1000.0


def enforce_numbered_list_parent_continuity(nodes: dict[int, ResolvedNode], parent_of: dict[int, int]) -> None:
    """Attach cross-page numbered items to the same parent as the previous item."""

    ordered_ids = sorted(nodes, key=lambda node_id: node_reading_order_key(nodes[node_id]))
    last_numbered_item: dict[int, tuple[int, int]] = {}
    effective_position = 0
    for node_id in ordered_ids:
        node = nodes[node_id]
        if is_page_noise_node(node):
            continue
        effective_position += 1
        number = ordered_numeric_dot_number(node.text)
        if number is None:
            continue
        if not record_is_numbered_list_like(node.record, node.text):
            continue
        if number > 1:
            previous = last_numbered_item.get(number - 1)
            if previous is not None:
                previous_id, previous_position = previous
                previous_parent = parent_of.get(previous_id)
                if previous_parent is not None and effective_position - previous_position <= 12:
                    parent_of[node_id] = previous_parent
        last_numbered_item[number] = (node_id, effective_position)


def collect_implicit_indented_list_after_lead(
    lead: ResolvedNode,
    children: list[ResolvedNode],
    start_index: int,
) -> tuple[list[ResolvedNode], int]:
    """Recover bullet-list items whose visible marker was dropped by OCR."""

    if not is_implicit_list_lead(lead):
        return ([], start_index)
    anchor_bbox = merge_barrier_bbox(lead.record)
    if anchor_bbox is None:
        return ([], start_index)
    anchor_page = merge_barrier_page(lead.record)
    min_indent = anchor_bbox[0] + 24.0
    items: list[ResolvedNode] = []
    index = start_index
    while index < len(children) and is_page_noise_node(children[index]):
        index += 1
    if index < len(children) and list_environment_for_node(children[index]) is not None:
        return ([], start_index)
    while index < len(children):
        candidate = children[index]
        if is_page_noise_node(candidate):
            index += 1
            continue
        if not is_implicit_indented_list_item(candidate, min_indent=min_indent, anchor_page=anchor_page):
            break
        items.append(candidate)
        index += 1
    if len(items) < 2:
        return ([], start_index)
    return (items, index)


def is_implicit_list_lead(node: ResolvedNode) -> bool:
    if canonical_render_type(node.record) != "text":
        return False
    text = " ".join(node.text.split()).strip()
    if not text:
        return False
    return text.endswith(":") or normalize_structural_heading_text(text) == "where"


def is_implicit_indented_list_item(
    node: ResolvedNode,
    *,
    min_indent: float,
    anchor_page: float | None,
) -> bool:
    if canonical_render_type(node.record) != "text":
        return False
    if list_environment_for_node(node) is not None:
        return True
    if node.children:
        return False
    bbox = merge_barrier_bbox(node.record)
    if bbox is None or bbox[0] < min_indent:
        return False
    page = merge_barrier_page(node.record)
    if anchor_page is not None and page is not None and page != anchor_page:
        return False
    text = " ".join(node.text.split()).strip()
    if not text or len(text) > 260:
        return False
    if text.endswith((".", "。", "?", "!", "？", "！")) and ":" not in text[:80]:
        return False
    return True


def is_abstract_root_candidate(node: ResolvedNode) -> bool:
    block_type = canonical_render_type(node.record)
    if block_type not in {"title", "text"}:
        return False
    text = " ".join(node.text.split())
    return bool(re.search(r"\babstract\b", text[:120], flags=re.IGNORECASE))


def render_title(text: str, *, depth: int, record: dict[str, Any] | None = None) -> str:
    record = record or {}
    if record.get("_render_as_paragraph_heading"):
        return rf"\paragraph*{{{escape_latex(str(text or '').strip())}}}"
    command = title_command(text, depth=depth, record=record)
    title_text = strip_title_numbering(text)
    star = "*" if record.get("_heading_unnumbered") or is_unnumbered_frontmatter_title(text) else ""
    return rf"\{command}{star}{{{escape_latex(title_text)}}}"


def render_run_in_heading_node(node: ResolvedNode, *, depth: int) -> list[str]:
    """Render a paragraph bbox that starts with an inline visual heading."""

    title_text = str(node.record.get("run_in_heading_text") or "").strip()
    if not title_text:
        title_text = strip_run_in_heading_from_text(node.text)[0]
    body_text = str(node.record.get("run_in_heading_body") or "").strip()
    if not body_text:
        _title, body_text = strip_run_in_heading_from_text(node.text)

    parts: list[str] = []
    if title_text:
        parts.append(render_title(title_text, depth=depth, record=node.record))
    if body_text:
        parts.append(render_text_with_inline_latex(body_text))
    return parts


def strip_run_in_heading_from_text(text: str) -> tuple[str, str]:
    value = " ".join(str(text or "").split())
    match = re.match(r"^\s*\d+(?:\.\d+)+\.?\s+(.+?\.)\s+(.*)$", value)
    if not match:
        return value, ""
    return (re.sub(r"[\s.]+$", "", match.group(1)).strip(), match.group(2).strip())


def is_unnumbered_frontmatter_title(text: str) -> bool:
    normalized = normalize_structural_heading_text(text)
    return normalized in {"abstract", "references", "bibliography"} or normalized.startswith("appendix")


def is_toc_title_node(record: dict[str, Any], text: str) -> bool:
    role = str(record.get("layout_role") or "").casefold()
    canonical = str(record.get("canonical_type") or "").casefold()
    if role == "toc_title" or canonical == "toc_title":
        return True
    raw = str(record.get("type") or record.get("raw_type") or record.get("block_type") or "").casefold()
    normalized = re.sub(r"[^a-z]+", "", str(text or "").casefold())
    return raw in {"title", "section", "heading"} and normalized in {"contents", "tableofcontents"}


def render_toc() -> str:
    return r"\tableofcontents"


def root_with_document_toc(root: ResolvedNode, document_metadata: dict[str, Any] | None) -> ResolvedNode:
    if not isinstance(document_metadata, dict) or not document_metadata.get("has_toc"):
        return root
    if tree_contains_toc(root):
        return root

    toc_order = numeric_value(document_metadata.get("toc_order"))
    toc_page = numeric_value(document_metadata.get("toc_page_idx"))
    record: dict[str, Any] = {
        "type": "toc",
        "canonical_type": "toc",
        "layout_role": "toc_title",
        "layout_layer": "metadata_layer",
        "text": "",
        "global_order": toc_order if toc_order is not None else 0,
        "regime_reading_order": toc_order if toc_order is not None else 0,
        "page_idx": toc_page if toc_page is not None else None,
        "_layout_state_locked": True,
    }
    toc_node = ResolvedNode(node_id=-1000001, record=record, merged_node_ids=[])
    patched = ResolvedNode(
        node_id=root.node_id,
        record=dict(root.record),
        merged_node_ids=list(root.merged_node_ids),
        children=list(root.children) + [toc_node],
        sibling_after=list(root.sibling_after),
    )
    sort_tree_children(patched)
    return patched


def tree_contains_toc(node: ResolvedNode) -> bool:
    if canonical_render_type(node.record) == "toc" or is_toc_title_node(node.record, node.text):
        return True
    return any(tree_contains_toc(child) for child in node.children)


def title_command(text: str, *, depth: int, record: dict[str, Any] | None = None) -> str:
    record = record or {}
    render_level = numeric_value(record.get("_heading_render_level"))
    if render_level is None:
        render_level = numeric_value(record.get("_skeleton_heading_level"))
    if render_level is not None and render_level >= 1:
        return SECTION_COMMANDS[min(int(render_level) - 1, len(SECTION_COMMANDS) - 1)]
    numbered_level = title_numbering_level(text)
    if numbered_level is not None:
        return SECTION_COMMANDS[min(numbered_level - 1, len(SECTION_COMMANDS) - 1)]
    return SECTION_COMMANDS[min(max(0, depth), len(SECTION_COMMANDS) - 1)]


def render_equation(text: str) -> str:
    stripped = strip_latex_control_chars(text).strip()
    if not stripped:
        return "\\[\n\n\\]"
    multi_tag_render = render_multi_tag_equation(stripped)
    if multi_tag_render:
        return multi_tag_render
    if stripped.startswith("\\[") or stripped.startswith("$$"):
        return stripped
    begin_match = re.match(r"\\begin\{([^}]+)\}", stripped)
    if begin_match and begin_match.group(1).rstrip("*") in DISPLAY_MATH_ENVS:
        return stripped
    body, tag = split_trailing_equation_number(stripped)
    if tag is not None:
        return "\\begin{equation}\n" + body + rf" \tag{{{tag}}}" + "\n\\end{equation}"
    if TAG_RE.search(stripped):
        return "\\begin{equation}\n" + stripped + "\n\\end{equation}"
    if should_render_as_align(stripped):
        return "\\begin{align}\n" + stripped + "\n\\end{align}"
    return "\\[\n" + stripped + "\n\\]"


TAG_RE = re.compile(r"\\tag\s*\{([^{}]+)\}")
TRAILING_EQUATION_NUMBER_RE = re.compile(r"^(?P<body>.+?)\s*(?:\((?P<tag>[A-Za-z]?\d+(?:\.\d+)*)\))\s*$", re.DOTALL)


def render_multi_tag_equation(text: str) -> str | None:
    """Render OCR-fused display formulas containing more than one equation tag.

    MinerU occasionally merges consecutive display equations into one block:
    ``expr_a \\tag{1} expr_b \\tag{2}``.  Wrapping that in ``\\[...\\]`` makes
    amsmath fail with "Multiple \\tag".  Splitting at tag boundaries preserves
    compilation and keeps the visual order.
    """
    stripped = text.strip()
    if stripped.startswith("\\[") and stripped.endswith("\\]"):
        stripped = stripped[2:-2].strip()
    elif stripped.startswith("$$") and stripped.endswith("$$"):
        stripped = stripped[2:-2].strip()

    matches = list(TAG_RE.finditer(stripped))
    if len(matches) <= 1:
        return None

    rows: list[tuple[str, str | None]] = []
    cursor = 0
    for match in matches:
        expr = stripped[cursor : match.start()].strip()
        tag = match.group(1).strip()
        if expr:
            rows.append((expr, tag))
        elif rows:
            prev_expr, _ = rows[-1]
            rows[-1] = (prev_expr, tag)
        cursor = match.end()

    tail = stripped[cursor:].strip()
    if tail:
        rows.append((tail, None))

    if len(rows) <= 1:
        return None

    rendered_rows = []
    for expr, tag in rows:
        rendered_rows.append(f"{expr} \\tag{{{tag}}}" if tag else expr)
    return "\\begin{align}\n" + "\\\\\n".join(rendered_rows) + "\n\\end{align}"


def split_trailing_equation_number(text: str) -> tuple[str, str | None]:
    stripped = text.strip()
    if TAG_RE.search(stripped):
        return stripped, None
    match = TRAILING_EQUATION_NUMBER_RE.match(stripped)
    if not match:
        return stripped, None
    body = match.group("body").strip()
    tag = match.group("tag").strip()
    if not body or not tag:
        return stripped, None
    return body, tag


def should_render_as_align(text: str) -> bool:
    stripped = text.strip()
    if "\\\\" in stripped and ("&" in stripped or "\n" in stripped):
        return True
    rows = [row.strip() for row in stripped.splitlines() if row.strip()]
    return len(rows) > 1 and any("&" in row for row in rows)


def render_inline_math(text: str) -> str:
    stripped = normalize_duplicate_math_command_slashes(strip_latex_control_chars(text).strip())
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
            append_nonredundant_rendered(rendered_parts, rendered)
    if used_structured_content and rendered_parts:
        return merge_latex_fragments(rendered_parts)
    if rendered_parts:
        return normalize_latex_text(" ".join(rendered_parts))
    return render_text_with_inline_latex(node.text)


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
            append_nonredundant_rendered(rendered_parts, rendered)
    if rendered_parts:
        if used_structured_content:
            return merge_latex_fragments(rendered_parts)
        return normalize_latex_text(" ".join(rendered_parts))
    return escape_latex(strip_list_marker(node.text))


def append_nonredundant_rendered(parts: list[str], rendered: str) -> None:
    if not is_redundant_rendered_text(rendered, parts):
        parts.append(rendered)


def is_redundant_rendered_text(rendered: str, previous_parts: Sequence[str], *, min_chars: int = 60) -> bool:
    """Detect OCR/layout duplicates without suppressing ordinary repeated phrases."""
    key = rendered_text_dedupe_key(rendered)
    if len(key) < min_chars:
        return False
    for previous in previous_parts[-4:]:
        previous_key = rendered_text_dedupe_key(previous)
        if len(previous_key) >= len(key) and key in previous_key:
            return True
    return False


def rendered_text_dedupe_key(value: str) -> str:
    text = re.sub(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^{}]*)\}", r"\2", str(value or ""))
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


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
        return render_text_with_inline_latex(fallback_text)
    rendered: list[str] = []
    plain_context = ""
    for segment in segments:
        segment_type = str(segment.get("type") or "").lower()
        content = str(segment.get("content") or segment.get("text") or "")
        if not content:
            continue
        if segment_type in {"equation_inline", "inline_equation", "inline_math", "inline_formula"}:
            repaired_content, insert_as_marker = repair_inline_math_ocr_segment(content, plain_context)
            if insert_as_marker:
                marker = "as: " if rendered and rendered[-1].endswith((" ", "\n")) else " as: "
                rendered.append(render_text_with_inline_latex(marker, strip=False))
                plain_context += marker
            rendered.append(render_inline_math(repaired_content))
        elif segment_type in {"equation_interline", "interline_equation", "display_formula", "formula", "equation"}:
            rendered.append("\n\n" + render_equation(content) + "\n\n")
        else:
            rendered.append(render_text_with_inline_latex(content, strip=False))
            plain_context += content
    return normalize_latex_text("".join(rendered))


def repair_inline_math_ocr_segment(content: str, left_context: str) -> tuple[str, bool]:
    """Repair narrow OCR cases where prose was swallowed into inline math.

    MinerU occasionally reads a run such as ``as: η = ...`` as the standalone
    math command ``\\arcsin = ...``.  That shape is not a meaningful inline
    equation by itself: inverse trig functions need an argument before equality.
    When the nearby prose already introduced a Greek variable, reuse that
    variable as the left-hand side and optionally restore the missing ``as:``.
    """

    value = strip_latex_control_chars(content).strip()
    if not BARE_OPERATOR_EQUATION_RE.match(value):
        return value, False
    replacement = last_greek_variable_in_context(left_context)
    if not replacement:
        return value, False
    repaired = BARE_OPERATOR_EQUATION_RE.sub(lambda _match: replacement + " =", value, count=1)
    return repaired, should_insert_as_marker_before_repaired_math(left_context)


def last_greek_variable_in_context(text: str) -> str | None:
    matches = list(GREEK_CONTEXT_RE.finditer(str(text or "")))
    if not matches:
        return None
    return GREEK_TO_LATEX.get(matches[-1].group(0))


def should_insert_as_marker_before_repaired_math(text: str) -> bool:
    normalized = " ".join(str(text or "").split()).casefold()
    if not normalized or normalized.endswith((":", " as", " as:")):
        return False
    return normalized.endswith(("modeled", "modelled", "defined", "expressed", "written", "given"))


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


def normalize_duplicate_math_command_slashes(text: str) -> str:
    return re.sub(r"\\\\([A-Za-z]+)", r"\\\1", str(text or ""))


def render_text_with_inline_latex(text: str, *, strip: bool = True) -> str:
    """Escape prose while preserving inline TeX math fragments embedded in it.

    MinerU sometimes keeps raw snippets such as ``\\mathrm { p }`` inside an
    otherwise ordinary caption/text block instead of emitting an ``inline_math``
    segment. Treat only explicit math delimiters and known math commands as raw
    math; everything else still goes through normal LaTeX escaping.
    """

    value = strip_latex_control_chars(text)
    if not value:
        return ""
    rendered: list[str] = []
    cursor = 0
    while cursor < len(value):
        span = find_next_inline_latex_span(value, cursor)
        if span is None:
            rendered.append(escape_latex(value[cursor:]))
            break
        start, end = span
        if start > cursor:
            rendered.append(escape_latex(value[cursor:start]))
        raw_math = value[start:end].strip()
        if raw_math:
            raw_math, trailing_punctuation = split_trailing_inline_math_punctuation(raw_math)
            rendered.append(render_inline_math(raw_math))
            if trailing_punctuation:
                rendered.append(escape_latex(trailing_punctuation))
        cursor = end
    output = re.sub(r"\n{3,}", "\n\n", "".join(rendered))
    return output.strip() if strip else output


def find_next_inline_latex_span(text: str, start_index: int) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    dollar = text.find("$", start_index)
    if dollar >= 0 and not text.startswith("$$", dollar):
        end = find_unescaped(text, "$", dollar + 1)
        if end is not None:
            candidates.append((dollar, end + 1))
    paren = text.find(r"\(", start_index)
    if paren >= 0:
        end = text.find(r"\)", paren + 2)
        if end >= 0:
            candidates.append((paren, end + 2))
    command_match = find_next_math_command(text, start_index)
    if command_match is not None:
        command_start, _command_name = command_match
        end = consume_bare_latex_math(text, command_start)
        if end > command_start:
            candidates.append((command_start, end))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])


def split_trailing_inline_math_punctuation(text: str) -> tuple[str, str]:
    value = str(text or "").strip()
    if len(value) < 2 or value.startswith("$") or value.startswith(r"\("):
        return value, ""
    if value[-1] not in ".,;:":
        return value, ""
    if len(value) >= 2 and value[-2].isdigit() and value[-1] == ".":
        return value, ""
    return value[:-1].rstrip(), value[-1]


def find_unescaped(text: str, needle: str, start_index: int) -> int | None:
    index = text.find(needle, start_index)
    while index >= 0:
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return index
        index = text.find(needle, index + 1)
    return None


def find_next_math_command(text: str, start_index: int) -> tuple[int, str] | None:
    for match in MATH_COMMAND_RE.finditer(text, start_index):
        command_name = match.group(1)
        if command_name in INLINE_MATH_COMMANDS:
            command_start = match.start()
            if command_start > start_index and text[command_start - 1] == "\\":
                command_start -= 1
            return command_start, command_name
    return None


def consume_bare_latex_math(text: str, start_index: int) -> int:
    index = start_index
    brace_depth = 0
    saw_command = False
    while index < len(text):
        char = text[index]
        if char == "\\":
            if index + 2 < len(text) and text[index + 1] == "\\" and text[index + 2].isalpha():
                command = MATH_COMMAND_RE.match(text, index + 1)
                if command:
                    saw_command = True
                    index = command.end()
                    continue
            command = MATH_COMMAND_RE.match(text, index)
            if command:
                saw_command = True
                index = command.end()
                continue
            if index + 1 < len(text):
                saw_command = True
                index += 2
                continue
            break
        if char == "{":
            brace_depth += 1
            index += 1
            continue
        if char == "}":
            if brace_depth <= 0:
                break
            brace_depth -= 1
            index += 1
            continue
        if brace_depth > 0:
            index += 1
            continue
        if char.isspace():
            next_index = index + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if next_index < len(text) and (text[next_index] == "\\" or text[next_index] in "{}_^+-=*/<>,.()[]|"):
                index = next_index
                continue
            break
        if char in "_^+-=*/<>,.()[]|" or char.isdigit():
            index += 1
            continue
        break
    return index if saw_command else start_index


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
        text = strip_reference_label(str(text or "").strip())
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
            commands.append(rf"\For{{{format_algorithmic_text(for_match.group(1).strip(), allow_math=False)}}}")
            block_stack.append("for")
            continue

        while_match = PSEUDOCODE_WHILE_RE.match(line)
        if while_match:
            commands.append(rf"\While{{{format_algorithmic_text(while_match.group(1).strip(), allow_math=False)}}}")
            block_stack.append("while")
            continue

        if_match = PSEUDOCODE_IF_RE.match(line)
        if if_match:
            commands.append(rf"\If{{{format_algorithmic_text(if_match.group(1).strip(), allow_math=False)}}}")
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


def format_algorithmic_text(text: str, *, allow_math: bool = True) -> str:
    prepared = normalize_algorithm_math_text(text)
    if not prepared:
        return ""
    if is_algorithm_code_like(prepared) or not allow_math:
        return r"\texttt{" + escape_algorithm_code_text(prepared) + r"}"
    if allow_math and LATEX_MATH_MARKER_RE.search(prepared):
        return r"\(\displaystyle " + escape_algorithm_math_text(prepared) + r"\)"
    return escape_latex(prepared)


def is_algorithm_code_like(text: str) -> bool:
    """Detect C-like pseudo-code that must not be wrapped in math mode."""

    return bool(ALGORITHM_CODE_MARKER_RE.search(str(text or "")))


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


def escape_algorithm_code_text(text: str) -> str:
    safe = "".join(_safe_code_verbatim_char(char) for char in str(text or ""))
    return escape_latex(safe)


def restore_algorithm_line_breaks(text: str) -> str:
    body = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" in body:
        return body
    body = PSEUDOCODE_BREAK_RE.sub("\n", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def sanitize_verbatim_body(text: str) -> str:
    sanitized = VERBATIM_END_RE.sub(r"\\end {verbatim}", str(text or ""))
    return "".join(_safe_code_verbatim_char(char) for char in sanitized)


def render_table_placeholder(
    record: dict[str, Any],
    text: str,
    *,
    node_id: int | None = None,
    source_pdf: str | None = None,
    asset_output_dir: str | None = None,
    asset_latex_prefix: str = "assets",
) -> str:
    if int(record.get("table_group_size") or 1) > 1 and record.get("table_group_primary") is False:
        return ""
    table_id = table_node_identifier(record, node_id=node_id)
    bbox = format_table_bbox(record.get("table_group_bbox") or record.get("bbox"))
    caption = table_caption_text(record) or extract_table_caption(text) or "Table reconstruction placeholder"
    graphic = ensure_table_pdf_crop(
        record,
        source_pdf=source_pdf or cfg_source_pdf(record),
        asset_output_dir=asset_output_dir,
        asset_latex_prefix=asset_latex_prefix,
    )
    todo = f"% [TODO_TABLE_RECONSTRUCT: BBOX={bbox}, ID={table_id}]"
    graphic_line = rf"\includegraphics[width=\linewidth]{{{graphic}}}" if graphic else todo
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            graphic_line,
            rf"\caption{{{escape_latex(caption)}}}",
            r"\end{table}",
        ]
    )


def figure_placeholder(record: dict[str, Any], *, node_id: int | None = None) -> str:
    figure_id = figure_node_identifier(record, node_id=node_id)
    bbox = format_table_bbox(record.get("figure_group_bbox") or record.get("image_group_bbox") or record.get("bbox"))
    return f"% [TODO_FIGURE_RECONSTRUCT: BBOX={bbox}, ID={figure_id}]"


def figure_include_width(record: dict[str, Any]) -> str:
    bbox = record.get("figure_group_bbox") or record.get("image_group_bbox") or record.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return "0.95"
    try:
        width = float(bbox[2]) - float(bbox[0])
    except (TypeError, ValueError):
        return "0.95"
    page_width = numeric_value(record.get("page_width")) or 1000.0
    ratio = max(min(width / max(page_width, 1.0), 0.98), 0.25)
    return f"{ratio:.3f}"


def figure_node_identifier(record: dict[str, Any], *, node_id: int | None = None) -> str:
    for key in ("figure_group_id", "image_group_id", "id", "block_id", "figure_id", "image_id"):
        value = record.get(key)
        if value:
            return str(value)
    value = record.get("global_order")
    if value is not None:
        return f"figure_{value}"
    if node_id is not None and node_id >= 0:
        return f"figure_{node_id}"
    return "figure_unknown"


def cfg_source_pdf(record: dict[str, Any]) -> str | None:
    for key in ("source_pdf", "pdf_path", "style_source_pdf"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def table_node_identifier(record: dict[str, Any], *, node_id: int | None = None) -> str:
    for key in ("table_group_id", "id", "block_id", "table_id"):
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
    return "".join(_escape_latex_char(char, replacements) for char in strip_latex_control_chars(text))


def strip_latex_control_chars(text: Any) -> str:
    """Remove non-printing OCR/control bytes that TeX cannot compile."""

    return LATEX_CONTROL_CHAR_RE.sub("", str(text or ""))


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
