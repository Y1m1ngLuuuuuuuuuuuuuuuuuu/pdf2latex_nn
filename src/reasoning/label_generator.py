"""Generate supervised edge labels from PDF-to-TeX alignments."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.perception.reading_order import fuse_micro_nodes
from src.perception.title_features import title_numbering_level
from src.reasoning.latex_flattener import LatexFlattenerConfig, flatten_latex_file
from src.reasoning.tex_ast_builder import build_tex_ast_from_file, tex_nodes_by_id
from src.reasoning.tex_relation_labeler import TexRelationLabel, label_tex_relation


@dataclass(frozen=True)
class LabelGeneratorConfig:
    similarity_threshold: float = 0.55
    adjacent_siblings_only: bool = True
    directed_parent_child: bool = False
    orphan_label: int = int(TexRelationLabel.NONE)
    max_orphan_ratio: float = 0.15
    min_aligned_nodes: int = 1
    abort_on_bad_alignment: bool = True


class AlignmentQualityError(RuntimeError):
    """Raised when PDF-to-TeX alignment is too poor for supervised training."""


class LayoutBreakerException(AlignmentQualityError):
    """Raised when a TeX construct is known to poison text-to-layout alignment."""


@dataclass(frozen=True)
class OrphanAlignment:
    node_index: int
    node_key: str
    reason: str
    score: float | None = None


@dataclass(frozen=True)
class LabelGenerationResult:
    data: Any
    label_counts: dict[int, int]
    orphan_alignments: list[OrphanAlignment]


@dataclass(frozen=True)
class TexAlignmentNode:
    tex_id: str
    node_type: str
    text: str
    clean: str
    path_ids: tuple[str, ...]
    parent_id: str | None = None
    source_name: str | None = None
    source_span: tuple[int, int] | None = None

    @property
    def clean_text(self) -> str:
        return self.clean


@dataclass(frozen=True)
class PdfAlignmentNode:
    node_index: int
    text: str
    clean: str
    item: dict[str, Any]


@dataclass(frozen=True)
class AlignmentMatch:
    pdf_node_index: int
    tex_id: str | None
    score: float


@dataclass(frozen=True)
class VisualHierarchy:
    parent_by_node: dict[int, int | None]
    child_rank_by_node: dict[int, int]
    heading_ids: frozenset[int]
    heading_levels: dict[int, int]
    document_title_ids: frozenset[int]
    body_font_size: float


@dataclass(frozen=True)
class AlignmentLabelerConfig:
    similarity_threshold: float = 65.0
    min_clean_chars: int = 3
    max_window_nodes: int = 8
    max_window_chars_ratio: float = 1.8
    score_drop_tolerance: float = 8.0
    tex_lookahead_nodes: int = 4
    tail_absorption_nodes: int = 3
    equation_blind_alignment_window: int = 2
    relation_strategy: str = "visual_first"
    visual_heading_font_scale: float = 1.12
    visual_bold_heading_font_scale: float = 1.05
    max_orphan_ratio: float = 0.15
    max_unmapped_tex_ratio: float = 0.30
    max_isolated_node_ratio: float = 0.85
    min_section_nodes: int = 1
    abort_on_bad_alignment: bool = False
    output_mapping_json: Path | None = None


SECTION_LEVELS = {
    "part": 0,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
    "paragraph": 5,
    "subparagraph": 6,
}
INLINE_MATH_ENV_NAMES = {
    "(",
    "$",
    "math",
}
DISPLAY_MATH_ENV_NAMES = {
    "[",
    "$$",
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "flalign",
    "flalign*",
    "displaymath",
}
MATH_ENV_NAMES = INLINE_MATH_ENV_NAMES | DISPLAY_MATH_ENV_NAMES
LIST_ENV_NAMES = {"itemize", "enumerate", "description"}
STANDARD_LIST_NODE = "list_container"
STANDARD_LIST_ITEM_NODE = "list_item"
STANDARD_SECTION_NODE = "section"
STANDARD_PARAGRAPH_NODE = "paragraph"
STANDARD_EQUATION_NODE = "equation_display"
STANDARD_FIGURE_CAPTION_NODE = "figure_caption"
STANDARD_TABLE_CAPTION_NODE = "table_caption"
ALIGNABLE_TEX_NODE_TYPES = {
    STANDARD_SECTION_NODE,
    STANDARD_PARAGRAPH_NODE,
    STANDARD_EQUATION_NODE,
    STANDARD_LIST_ITEM_NODE,
    STANDARD_FIGURE_CAPTION_NODE,
    STANDARD_TABLE_CAPTION_NODE,
}
POISON_TEX_ENV_NAMES = {
    "tikzpicture",
    "pgfpicture",
    "pgfplots",
    "axis",
    "pspicture",
}
BLOCK_ENV_NAMES = {
    "abstract",
    "algorithm",
    "algorithmic",
    "algorithm2e",
    "caption",
    "figure",
    "figure*",
    "table",
    "table*",
}
CAPTION_PARENT_ENVS = {"figure", "figure*", "table", "table*"}
CONTAINER_ENV_NAMES = {
    "document",
    "center",
    "flushleft",
    "flushright",
    "minipage",
    "small",
    "footnotesize",
    "scriptsize",
    "normalsize",
    "large",
    "Large",
}
SKIP_TEX_NODE_NAMES = {
    "documentclass",
    "usepackage",
    "label",
    "ref",
    "cite",
    "bibliographystyle",
    "bibliography",
}
HEADING_TYPES = {"title", "section", "subsection", "subsubsection", "heading"}
SECTION_TYPE_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}
LIST_MARKER_RE = re.compile(r"^\s*(?:[\u2022\u25E6\u25CB\u25AA\-\*]|\d+[\.\)]|[a-zA-Z][\.\)])\s+")
SENTENCE_END_RE = re.compile(r"[。.!?！？]\s*$")
MERGE_COMPATIBLE_PDF_TYPES = {"text", "equation", "reference"}
NON_HEADING_PDF_TYPES = {
    "equation",
    "equation_interline",
    "interline_equation",
    "display_formula",
    "formula",
    "inline_math",
    "inline_formula",
    "math_inline",
    "table",
    "figure",
    "image",
    "chart",
    "algorithm",
    "code",
    "reference",
    "references",
    "bibliography",
}


class AlignmentLabeler:
    """Align MinerU PDF blocks to TexSoup nodes and inject edge labels into a graph."""

    def __init__(
        self,
        *,
        content_json_path: Path,
        tex_path: Path,
        graph_path: Path,
        config: AlignmentLabelerConfig | None = None,
    ) -> None:
        self.content_json_path = Path(content_json_path)
        self.tex_path = Path(tex_path)
        self.graph_path = Path(graph_path)
        self.config = config or AlignmentLabelerConfig()
        self.tex_nodes: dict[str, TexAlignmentNode] = {}
        self.pdf_nodes: list[PdfAlignmentNode] = []
        self.matches: list[AlignmentMatch] = []
        self.tex_to_pdf_indices: dict[str, list[int]] = {}
        self.visual_hierarchy: VisualHierarchy | None = None
        self.flattener_summary: dict[str, Any] | None = None
        self.alignment_quality: dict[str, Any] = {}

    def run(self, *, output_graph_path: Path | None = None, overwrite: bool = True) -> Any:
        graph = self.load_graph()
        self.pdf_nodes = self.parse_pdf_nodes(
            expected_node_count=int(graph.num_nodes),
            force_micro_fusion=bool(getattr(graph, "micro_fusion_applied", False)),
        )
        if len(self.pdf_nodes) != int(graph.num_nodes):
            raise ValueError(
                f"content node count ({len(self.pdf_nodes)}) does not match graph.num_nodes ({int(graph.num_nodes)})"
            )
        self.tex_nodes = {node.tex_id: node for node in self.parse_tex_nodes()}
        self.matches = self.align_pdf_to_tex()
        self.visual_hierarchy = build_visual_hierarchy(self.pdf_nodes, config=self.config)
        labels = self.build_edge_labels(graph)
        graph.y = labels
        graph.edge_label = labels
        graph.pdf_to_tex = [match.tex_id for match in self.matches]
        graph.pdf_to_tex_scores = [match.score for match in self.matches]
        graph.label_counts = label_counts(labels)
        graph.alignment_schema = {
            "strategy": "texsoup_semantic_ast_sliding_window_v3",
            "similarity_threshold": self.config.similarity_threshold,
            "max_window_nodes": self.config.max_window_nodes,
            "tail_absorption_nodes": self.config.tail_absorption_nodes,
            "equation_blind_alignment_window": self.config.equation_blind_alignment_window,
            "relation_strategy": self.config.relation_strategy,
            "content_json_path": str(self.content_json_path),
            "tex_path": str(self.tex_path),
            "flattener": self.flattener_summary,
        }
        self.assert_alignment_quality(graph=graph, labels=labels)
        if self.config.output_mapping_json is not None:
            self.write_mapping_json(self.config.output_mapping_json)
        destination = self.graph_path if output_graph_path is None and overwrite else output_graph_path
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            import torch

            torch.save(graph, destination)
        return graph

    def load_graph(self) -> Any:
        import torch

        return torch.load(self.graph_path, map_location="cpu", weights_only=False)

    def parse_pdf_nodes(
        self,
        *,
        expected_node_count: int | None = None,
        force_micro_fusion: bool = False,
    ) -> list[PdfAlignmentNode]:
        content = json.loads(self.content_json_path.read_text(encoding="utf-8"))
        items = content.get("items", content if isinstance(content, list) else [])
        if not isinstance(items, list):
            raise ValueError(f"Expected {self.content_json_path} to contain an items list")
        items = [item if isinstance(item, dict) else {"text_for_embedding": str(item)} for item in items]
        if force_micro_fusion:
            fused_items = fuse_micro_nodes(items)
            if expected_node_count is None or len(fused_items) == expected_node_count:
                items = fused_items
        elif expected_node_count is not None and len(items) != expected_node_count:
            fused_items = fuse_micro_nodes(items)
            if len(fused_items) == expected_node_count:
                items = fused_items
        nodes = []
        for index, item in enumerate(items):
            text = pdf_item_text(item)
            nodes.append(PdfAlignmentNode(node_index=index, text=text, clean=clean_text(text), item=item))
        return nodes

    def parse_tex_nodes(self) -> list[TexAlignmentNode]:
        from TexSoup import TexSoup

        flattened = flatten_latex_file(self.tex_path, config=LatexFlattenerConfig(mask_math=False))
        try:
            soup = TexSoup(normalize_display_math_for_texsoup(flattened.content))
            self.flattener_summary = flattened.summary()
            self.flattener_summary["mask_math_fallback"] = False
        except Exception:
            flattened = flatten_latex_file(self.tex_path, config=LatexFlattenerConfig(mask_math=True))
            soup = TexSoup(normalize_display_math_for_texsoup(flattened.content))
            self.flattener_summary = flattened.summary()
            self.flattener_summary["mask_math_fallback"] = True
        builder = _TexSoupPathBuilder(self.config)
        builder.walk_soup(soup)
        return builder.nodes

    def align_pdf_to_tex(self) -> list[AlignmentMatch]:
        tex_sequence = [node for node in self.tex_nodes.values() if self.is_alignable_tex_node(node)]
        matches = [AlignmentMatch(pdf_node_index=node.node_index, tex_id=None, score=0.0) for node in self.pdf_nodes]
        self.tex_to_pdf_indices = {}

        pdf_cursor = 0
        tex_cursor = 0
        while pdf_cursor < len(self.pdf_nodes) and tex_cursor < len(tex_sequence):
            pdf_node = self.pdf_nodes[pdf_cursor]
            if len(pdf_node.clean) < self.config.min_clean_chars:
                pdf_cursor += 1
                continue

            tex_node = tex_sequence[tex_cursor]
            window = self.find_alignment_window(pdf_cursor, tex_node)
            if window is None:
                next_tex_cursor = self.find_better_tex_candidate(pdf_node.clean, tex_sequence, tex_cursor)
                if next_tex_cursor is not None:
                    tex_cursor = next_tex_cursor
                    continue
                pdf_cursor += 1
                continue

            start, end, score = window
            assigned_indices = list(range(start, end + 1))
            next_index = end + 1
            absorbed = 0
            while absorbed < self.config.tail_absorption_nodes and next_index < len(self.pdf_nodes):
                next_pdf = self.pdf_nodes[next_index]
                if len(next_pdf.clean) < self.config.min_clean_chars:
                    break
                if not self.is_tex_fragment(next_pdf.clean, tex_node.clean):
                    break
                assigned_indices.append(next_index)
                next_index += 1
                absorbed += 1

            for node_index in assigned_indices:
                matches[node_index] = AlignmentMatch(
                    pdf_node_index=node_index,
                    tex_id=tex_node.tex_id,
                    score=score,
                )
            self.tex_to_pdf_indices.setdefault(tex_node.tex_id, []).extend(assigned_indices)
            pdf_cursor = next_index
            tex_cursor += 1

        return matches

    def is_alignable_tex_node(self, node: TexAlignmentNode) -> bool:
        if node.node_type == STANDARD_EQUATION_NODE:
            return bool(node.clean)
        if len(node.clean) < self.config.min_clean_chars:
            return False
        if node.node_type == STANDARD_LIST_NODE:
            return False
        return node.node_type in ALIGNABLE_TEX_NODE_TYPES

    def find_alignment_window(self, start_index: int, tex_node: TexAlignmentNode) -> tuple[int, int, float] | None:
        if tex_node.node_type == STANDARD_EQUATION_NODE:
            blind = self.find_blind_equation_window(start_index)
            if blind is not None:
                return blind
        buffer_parts: list[str] = []
        best_score = 0.0
        best_end = start_index
        max_end = min(len(self.pdf_nodes), start_index + max(1, self.config.max_window_nodes))
        for end_index in range(start_index, max_end):
            pdf_node = self.pdf_nodes[end_index]
            if len(pdf_node.clean) < self.config.min_clean_chars:
                continue
            buffer_parts.append(pdf_node.clean)
            buffer = "".join(buffer_parts)
            score = levenshtein_ratio_score(buffer, tex_node.clean)
            if score > best_score:
                best_score = score
                best_end = end_index
            if score >= self.config.similarity_threshold:
                return start_index, end_index, score
            if self.window_is_too_long(buffer, tex_node.clean):
                break
            if best_end < end_index and best_score - score > self.config.score_drop_tolerance:
                break
        if self.is_tex_fragment(self.pdf_nodes[start_index].clean, tex_node.clean):
            return start_index, start_index, fragment_ratio_score(self.pdf_nodes[start_index].clean, tex_node.clean)
        return None

    def find_blind_equation_window(self, start_index: int) -> tuple[int, int, float] | None:
        upper = min(len(self.pdf_nodes), start_index + max(1, self.config.equation_blind_alignment_window + 1))
        for index in range(start_index, upper):
            if canonical_pdf_merge_type(self.pdf_nodes[index].item) == "equation":
                return index, index, 100.0
        return None

    def find_better_tex_candidate(
        self,
        pdf_clean: str,
        tex_sequence: list[TexAlignmentNode],
        tex_cursor: int,
    ) -> int | None:
        current = tex_sequence[tex_cursor]
        current_score = fragment_ratio_score(pdf_clean, current.clean)
        best_cursor: int | None = None
        best_score = current_score
        upper = min(len(tex_sequence), tex_cursor + 1 + max(0, self.config.tex_lookahead_nodes))
        for candidate_cursor in range(tex_cursor + 1, upper):
            candidate = tex_sequence[candidate_cursor]
            score = fragment_ratio_score(pdf_clean, candidate.clean)
            if score > best_score:
                best_score = score
                best_cursor = candidate_cursor
        if best_cursor is None:
            return None
        if best_score >= self.config.similarity_threshold or best_score - current_score >= 15.0:
            return best_cursor
        return None

    def window_is_too_long(self, buffer: str, tex_clean: str) -> bool:
        if not tex_clean:
            return True
        allowed = int(len(tex_clean) * max(1.0, self.config.max_window_chars_ratio)) + 32
        return len(buffer) > allowed

    def is_tex_fragment(self, pdf_clean: str, tex_clean: str) -> bool:
        if len(pdf_clean) < self.config.min_clean_chars or len(tex_clean) < self.config.min_clean_chars:
            return False
        if pdf_clean in tex_clean or tex_clean in pdf_clean:
            return True
        return fragment_ratio_score(pdf_clean, tex_clean) >= max(self.config.similarity_threshold, 92.0)

    def build_edge_labels(self, graph: Any) -> Any:
        import torch

        labels = []
        edge_index = graph.edge_index.detach().cpu()
        for edge_pos in range(edge_index.shape[1]):
            source = int(edge_index[0, edge_pos].item())
            target = int(edge_index[1, edge_pos].item())
            labels.append(self.infer_relation(source, target))
        return torch.tensor(labels, dtype=torch.long)

    def infer_relation(self, source_index: int, target_index: int) -> int:
        if self.same_reference_scope(source_index, target_index):
            return int(TexRelationLabel.MERGE)
        source_match = self.matches[source_index] if 0 <= source_index < len(self.matches) else None
        target_match = self.matches[target_index] if 0 <= target_index < len(self.matches) else None
        if source_match is None or target_match is None or not source_match.tex_id or not target_match.tex_id:
            return int(TexRelationLabel.NONE)
        path_u = self.tex_nodes[source_match.tex_id].path_ids
        path_v = self.tex_nodes[target_match.tex_id].path_ids
        if path_u == path_v:
            if self.same_tex_node_merge_crosses_list_marker(source_index, target_index):
                return int(TexRelationLabel.NONE)
            if same_tex_node_can_merge(self.pdf_nodes[source_index], self.pdf_nodes[target_index]):
                return int(TexRelationLabel.MERGE)
            return int(TexRelationLabel.NONE)
        visual_relation = self.infer_visual_relation(source_index, target_index, source_match, target_match)
        if visual_relation is not None:
            return visual_relation
        if self.config.relation_strategy == "visual_only":
            return int(TexRelationLabel.NONE)
        if self.visual_hierarchy is not None and self.visual_hierarchy.heading_ids:
            return int(TexRelationLabel.NONE)
        if path_v[:-1] == path_u and self.is_first_pdf_anchor(source_index, source_match.tex_id) and self.is_first_pdf_anchor(
            target_index, target_match.tex_id
        ):
            return int(TexRelationLabel.PARENT_CHILD)
        return int(TexRelationLabel.NONE)

    def infer_visual_relation(
        self,
        source_index: int,
        target_index: int,
        source_match: AlignmentMatch,
        target_match: AlignmentMatch,
    ) -> int | None:
        hierarchy = self.visual_hierarchy
        if hierarchy is None or not hierarchy.heading_ids:
            return None
        if not self.is_first_pdf_anchor(source_index, source_match.tex_id) or not self.is_first_pdf_anchor(
            target_index, target_match.tex_id
        ):
            return int(TexRelationLabel.NONE)
        if hierarchy.parent_by_node.get(target_index) == source_index:
            return int(TexRelationLabel.PARENT_CHILD)
        source_parent = hierarchy.parent_by_node.get(source_index)
        target_parent = hierarchy.parent_by_node.get(target_index)
        return int(TexRelationLabel.NONE)

    def is_first_pdf_anchor(self, pdf_index: int, tex_id: str | None) -> bool:
        if not tex_id:
            return False
        indices = self.tex_to_pdf_indices.get(tex_id, [])
        return bool(indices) and pdf_index == min(indices)

    def same_tex_node_merge_crosses_list_marker(self, source_index: int, target_index: int) -> bool:
        lower, upper = sorted((source_index, target_index))
        if upper - lower <= 1:
            return False
        for index in range(lower + 1, upper):
            if 0 <= index < len(self.pdf_nodes) and LIST_MARKER_RE.match(self.pdf_nodes[index].text):
                return True
        return False

    def same_reference_scope(self, source_index: int, target_index: int) -> bool:
        if not (0 <= source_index < len(self.pdf_nodes) and 0 <= target_index < len(self.pdf_nodes)):
            return False
        source = self.pdf_nodes[source_index]
        target = self.pdf_nodes[target_index]
        if canonical_pdf_merge_type(source.item) != "reference" or canonical_pdf_merge_type(target.item) != "reference":
            return False
        return source_index != target_index

    def assert_alignment_quality(self, *, graph: Any | None = None, labels: Any | None = None) -> None:
        orphan_count = sum(1 for match in self.matches if not match.tex_id)
        orphan_ratio = orphan_count / max(1, len(self.matches))
        alignable_tex_nodes = [node for node in self.tex_nodes.values() if self.is_alignable_tex_node(node)]
        mapped_tex_ids = {match.tex_id for match in self.matches if match.tex_id}
        unmapped_tex_count = sum(1 for node in alignable_tex_nodes if node.tex_id not in mapped_tex_ids)
        unmapped_tex_ratio = unmapped_tex_count / max(1, len(alignable_tex_nodes))
        section_count = sum(1 for node in self.tex_nodes.values() if node.node_type == STANDARD_SECTION_NODE)
        paragraph_pdf_count = sum(1 for node in self.pdf_nodes if canonical_pdf_merge_type(node.item) == "text" and len(node.clean) >= self.config.min_clean_chars)
        isolated_count = 0
        isolated_ratio = 0.0
        if graph is not None and labels is not None and hasattr(graph, "edge_index"):
            connected = connected_node_indices(graph.edge_index.detach().cpu(), labels.detach().cpu())
            candidates = [node.node_index for node in self.pdf_nodes if len(node.clean) >= self.config.min_clean_chars]
            isolated_count = sum(1 for node_index in candidates if node_index not in connected)
            isolated_ratio = isolated_count / max(1, len(candidates))
        self.alignment_quality = {
            "orphan_count": orphan_count,
            "num_pdf_nodes": len(self.matches),
            "orphan_ratio": orphan_ratio,
            "max_orphan_ratio": self.config.max_orphan_ratio,
            "alignable_tex_count": len(alignable_tex_nodes),
            "unmapped_tex_count": unmapped_tex_count,
            "unmapped_tex_ratio": unmapped_tex_ratio,
            "max_unmapped_tex_ratio": self.config.max_unmapped_tex_ratio,
            "section_count": section_count,
            "min_section_nodes": self.config.min_section_nodes,
            "isolated_node_count": isolated_count,
            "isolated_node_ratio": isolated_ratio,
            "max_isolated_node_ratio": self.config.max_isolated_node_ratio,
        }
        failures: list[str] = []
        if orphan_ratio > self.config.max_orphan_ratio:
            failures.append(f"orphan_ratio={orphan_ratio:.2%} > {self.config.max_orphan_ratio:.2%}")
        if unmapped_tex_ratio > self.config.max_unmapped_tex_ratio:
            failures.append(f"unmapped_tex_ratio={unmapped_tex_ratio:.2%} > {self.config.max_unmapped_tex_ratio:.2%}")
        if paragraph_pdf_count >= 8 and section_count < self.config.min_section_nodes:
            failures.append(f"section_count={section_count} < min_section_nodes={self.config.min_section_nodes}")
        if isolated_ratio > self.config.max_isolated_node_ratio:
            failures.append(f"isolated_node_ratio={isolated_ratio:.2%} > {self.config.max_isolated_node_ratio:.2%}")
        if failures and self.config.abort_on_bad_alignment:
            raise AlignmentQualityError("bad alignment quality: " + "; ".join(failures))

    def write_mapping_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "alignment_mapping_v1",
            "content_json_path": str(self.content_json_path),
            "tex_path": str(self.tex_path),
            "graph_path": str(self.graph_path),
            "similarity_threshold": self.config.similarity_threshold,
            "flattener": self.flattener_summary,
            "quality": self.alignment_quality,
            "matches": [asdict(match) for match in self.matches],
            "tex_to_pdf_indices": self.tex_to_pdf_indices,
            "visual_hierarchy": visual_hierarchy_payload(self.visual_hierarchy),
            "tex_nodes": [tex_alignment_node_payload(node) for node in self.tex_nodes.values()],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def tex_alignment_node_payload(node: TexAlignmentNode) -> dict[str, Any]:
    payload = asdict(node)
    payload["clean_text"] = node.clean_text
    return payload


def build_visual_hierarchy(nodes: list[PdfAlignmentNode], *, config: AlignmentLabelerConfig) -> VisualHierarchy:
    """Build a lightweight heading/paragraph hierarchy from visual PDF node metadata.

    TeX AST paths are fragile in real arXiv sources. This stack uses only the
    already ordered MinerU/PyMuPDF records: block type, heading prefix, font
    statistics, and short standalone-title cues. It intentionally does not
    model itemize/enumerate; list-like blocks remain ordinary siblings inside
    the nearest section scope.
    """

    body_font_size = infer_body_font_size(nodes)
    heading_candidates = [
        node.node_index
        for node in nodes
        if is_visual_heading_candidate(node, body_font_size=body_font_size, config=config)
    ]
    document_title_ids = infer_document_title_ids(nodes, heading_candidates)
    heading_levels = infer_visual_heading_levels(
        nodes,
        heading_candidates,
        document_title_ids=document_title_ids,
        body_font_size=body_font_size,
        config=config,
    )
    heading_ids = frozenset(node_id for node_id in heading_candidates if node_id not in document_title_ids)
    parent_by_node: dict[int, int | None] = {}
    children_by_parent: dict[int | None, list[int]] = {}
    stack: list[tuple[int, int]] = []

    for node in nodes:
        node_id = node.node_index
        if node_id in document_title_ids:
            parent_by_node[node_id] = None
            children_by_parent.setdefault(None, []).append(node_id)
            continue
        if node_id in heading_ids:
            level = heading_levels[node_id]
            while stack and stack[-1][1] >= level:
                stack.pop()
            parent_id = stack[-1][0] if stack else None
            parent_by_node[node_id] = parent_id
            children_by_parent.setdefault(parent_id, []).append(node_id)
            stack.append((node_id, level))
            continue
        parent_id = stack[-1][0] if stack else None
        parent_by_node[node_id] = parent_id
        children_by_parent.setdefault(parent_id, []).append(node_id)

    child_rank_by_node: dict[int, int] = {}
    for children in children_by_parent.values():
        for rank, node_id in enumerate(children):
            child_rank_by_node[node_id] = rank

    return VisualHierarchy(
        parent_by_node=parent_by_node,
        child_rank_by_node=child_rank_by_node,
        heading_ids=heading_ids,
        heading_levels=heading_levels,
        document_title_ids=frozenset(document_title_ids),
        body_font_size=body_font_size,
    )


def infer_body_font_size(nodes: list[PdfAlignmentNode]) -> float:
    sizes: list[float] = []
    for node in nodes:
        if canonical_pdf_type(node.item) in HEADING_TYPES:
            continue
        size = pdf_font_size(node.item)
        if size > 0:
            text_len = max(1, min(len(node.text.strip()), 200))
            sizes.extend([round(size, 1)] * text_len)
    if not sizes:
        for node in nodes:
            size = pdf_font_size(node.item)
            if size > 0:
                sizes.append(round(size, 1))
    if not sizes:
        return 0.0
    return float(Counter(sizes).most_common(1)[0][0])


def is_visual_heading_candidate(
    node: PdfAlignmentNode,
    *,
    body_font_size: float,
    config: AlignmentLabelerConfig,
) -> bool:
    text = node.text.strip()
    if not text:
        return False
    raw_type = canonical_pdf_type(node.item)
    if raw_type in HEADING_TYPES:
        return True
    if raw_type in NON_HEADING_PDF_TYPES or str(node.item.get("list_type") or "").lower() == "reference_list":
        return False
    if LIST_MARKER_RE.match(text):
        return False
    if title_numbering_level(text) is not None and looks_like_standalone_heading(text):
        return True
    font_size = pdf_font_size(node.item)
    if body_font_size > 0 and font_size >= body_font_size * config.visual_heading_font_scale and looks_like_standalone_heading(text):
        return True
    if (
        body_font_size > 0
        and font_size >= body_font_size * config.visual_bold_heading_font_scale
        and pdf_bold_ratio(node.item) >= 0.45
        and looks_like_standalone_heading(text)
    ):
        return True
    return False


def infer_document_title_ids(nodes: list[PdfAlignmentNode], heading_candidates: list[int]) -> set[int]:
    if not heading_candidates:
        return set()
    by_id = {node.node_index: node for node in nodes}
    first_id = heading_candidates[0]
    first = by_id.get(first_id)
    if first is None:
        return set()
    bbox = first_bbox(first.item.get("bbox"))
    if bbox is None:
        return set()
    page_idx = parse_int(first.item.get("page_idx"))
    text = first.text.strip()
    width = max(0.0, bbox[2] - bbox[0])
    if page_idx == 0 and bbox[1] < 220 and width >= 520 and len(text) >= 35:
        return {first_id}
    return set()


def infer_visual_heading_levels(
    nodes: list[PdfAlignmentNode],
    heading_candidates: list[int],
    *,
    document_title_ids: set[int],
    body_font_size: float,
    config: AlignmentLabelerConfig,
) -> dict[int, int]:
    by_id = {node.node_index: node for node in nodes}
    sized_headings: list[float] = []
    for node_id in heading_candidates:
        if node_id in document_title_ids:
            continue
        node = by_id[node_id]
        if title_numbering_level(node.text) is not None:
            continue
        size = pdf_font_size(node.item)
        if size > 0:
            sized_headings.append(round(size, 1))
    unique_sizes = sorted(set(sized_headings), reverse=True)
    size_to_level = {size: min(index + 1, 3) for index, size in enumerate(unique_sizes)}

    levels: dict[int, int] = {}
    for node_id in heading_candidates:
        node = by_id[node_id]
        if node_id in document_title_ids:
            levels[node_id] = 0
            continue
        numbered_level = title_numbering_level(node.text)
        if numbered_level is not None:
            levels[node_id] = min(max(1, numbered_level), 3)
            continue
        raw_type = canonical_pdf_type(node.item)
        if raw_type in SECTION_TYPE_LEVELS:
            levels[node_id] = SECTION_TYPE_LEVELS[raw_type]
            continue
        size = pdf_font_size(node.item)
        rounded_size = round(size, 1)
        if rounded_size in size_to_level:
            levels[node_id] = size_to_level[rounded_size]
        elif body_font_size > 0 and size >= body_font_size * config.visual_heading_font_scale:
            levels[node_id] = 1
        else:
            levels[node_id] = 1
    return levels


def looks_like_standalone_heading(text: str) -> bool:
    value = " ".join(str(text or "").split())
    if not value:
        return False
    if len(value) > 140:
        return False
    if SENTENCE_END_RE.search(value):
        return False
    if value.count(",") >= 2:
        return False
    return True


def canonical_pdf_type(item: dict[str, Any]) -> str:
    return str(item.get("canonical_type") or item.get("type") or item.get("raw_type") or "").strip().lower()


def canonical_pdf_merge_type(item: dict[str, Any]) -> str:
    """Collapse MinerU/PyMuPDF block names into relation-label merge families."""

    if str(item.get("list_type") or "").lower() == "reference_list":
        return "reference"
    raw = canonical_pdf_type(item)
    if raw in {"paragraph", "text", "paragraph_text", "body", "list", "item"}:
        return "text"
    if raw in {"equation", "equation_interline", "interline_equation", "display_formula", "formula"}:
        return "equation"
    if raw in {"inline_math", "inline_formula", "math_inline"}:
        return "inline_math"
    if raw in {"reference", "references", "bibliography"}:
        return "reference"
    if raw in {"table", "figure", "image", "chart", "algorithm", "code"}:
        return raw
    return "text"


def same_tex_node_can_merge(left: PdfAlignmentNode, right: PdfAlignmentNode) -> bool:
    """Prevent same-TeX scope from collapsing across structural block boundaries.

    A TeX ``item`` may contain both text and display math. Those are siblings
    inside the list item, not one physical paragraph, so they must not become
    MERGE edges even though fuzzy alignment maps them to the same TeX node.
    """

    left_type = canonical_pdf_merge_type(left.item)
    right_type = canonical_pdf_merge_type(right.item)
    if LIST_MARKER_RE.match(right.text):
        return False
    return left_type == right_type and left_type in MERGE_COMPATIBLE_PDF_TYPES


def pdf_font_size(item: dict[str, Any]) -> float:
    for key in (
        "style_baseline_size",
        "baseline_font_size",
        "font_size",
        "font_size_px",
        "avg_font_size",
    ):
        value = numeric_value(item.get(key))
        if value is not None and value > 0:
            return value
    spans = item.get("spans")
    if isinstance(spans, list):
        weighted: list[float] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            size = numeric_value(span.get("font_size") or span.get("size"))
            if size is None or size <= 0:
                continue
            text = stringify_text_payload(span.get("text") or span.get("content"))
            weighted.extend([float(size)] * max(1, min(len(text), 80)))
        if weighted:
            return float(Counter(round(size, 1) for size in weighted).most_common(1)[0][0])
    return 0.0


def pdf_bold_ratio(item: dict[str, Any]) -> float:
    value = numeric_value(item.get("bold_char_ratio"))
    if value is not None:
        return max(0.0, min(1.0, value))
    spans = item.get("spans")
    if not isinstance(spans, list):
        return 0.0
    bold_chars = 0
    total_chars = 0
    for span in spans:
        if not isinstance(span, dict):
            continue
        text = stringify_text_payload(span.get("text") or span.get("content"))
        length = max(0, len(text.strip()))
        if length == 0:
            continue
        total_chars += length
        if bool(span.get("is_bold")):
            bold_chars += length
    return bold_chars / total_chars if total_chars else 0.0


def first_bbox(value: Any) -> tuple[float, float, float, float] | None:
    chunks = bbox_chunks(value)
    return chunks[0] if chunks else None


def bbox_chunks(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    usable_len = len(value) - (len(value) % 4)
    chunks: list[tuple[float, float, float, float]] = []
    for index in range(0, usable_len, 4):
        try:
            chunks.append(tuple(float(part) for part in value[index : index + 4]))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return chunks


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def numeric_value(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def visual_hierarchy_payload(hierarchy: VisualHierarchy | None) -> dict[str, Any] | None:
    if hierarchy is None:
        return None
    return {
        "strategy": "visual_heading_stack_v1",
        "body_font_size": hierarchy.body_font_size,
        "heading_ids": sorted(hierarchy.heading_ids),
        "heading_levels": {str(key): value for key, value in sorted(hierarchy.heading_levels.items())},
        "document_title_ids": sorted(hierarchy.document_title_ids),
        "parent_by_node": {
            str(key): value for key, value in sorted(hierarchy.parent_by_node.items()) if value is not None
        },
    }


def semantic_node_type_for_block(name: str) -> str:
    if name in {"figure", "figure*"}:
        return "figure"
    if name in {"table", "table*"}:
        return "table"
    if name in {"algorithm", "algorithmic", "algorithm2e"}:
        return "algorithm"
    if name == "abstract":
        return STANDARD_PARAGRAPH_NODE
    return STANDARD_PARAGRAPH_NODE


def section_command_base(name: str) -> str | None:
    base = name[:-1] if name.endswith("*") else name
    return base if base in SECTION_LEVELS else None


def normalize_display_math_for_texsoup(tex: str) -> str:
    """Convert display-math delimiters that TexSoup exposes poorly into environments."""

    tex = re.sub(
        r"\\\[(.*?)\\\]",
        lambda match: "\\begin{equation}" + match.group(1) + "\\end{equation}",
        tex,
        flags=re.DOTALL,
    )
    tex = re.sub(
        r"\$\$(.*?)\$\$",
        lambda match: "\\begin{equation}" + match.group(1) + "\\end{equation}",
        tex,
        flags=re.DOTALL,
    )
    return tex


class _TexSoupPathBuilder:
    def __init__(self, config: AlignmentLabelerConfig) -> None:
        self.config = config
        self.nodes: list[TexAlignmentNode] = []
        self.next_id = 1
        self.section_by_level: dict[int, str] = {}
        self.path_by_id: dict[str, tuple[str, ...]] = {"ROOT": ("ROOT",)}
        self.parent_by_id: dict[str, str | None] = {"ROOT": None}

    def walk_soup(self, soup: Any) -> None:
        contents = getattr(soup, "contents", []) or []
        for child in contents:
            if tex_node_name(child) == "document":
                self.walk_children(getattr(child, "contents", []) or [], parent_id="ROOT", parent_env="document")
                return
        self.walk_children(contents, parent_id="ROOT", parent_env=None)

    def walk_children(self, children: list[Any], *, parent_id: str, parent_env: str | None = None) -> None:
        paragraph_buffer: list[str] = []
        for child in children:
            name = tex_node_name(child)
            if name in SKIP_TEX_NODE_NAMES:
                continue
            if name in POISON_TEX_ENV_NAMES:
                raise LayoutBreakerException(f"Encountered complex drawing environment: {name}")
            if name in CONTAINER_ENV_NAMES:
                self.flush_paragraphs(paragraph_buffer, parent_id)
                self.walk_children(getattr(child, "contents", []) or [], parent_id=self.current_parent(parent_id), parent_env=name)
                continue
            section_name = section_command_base(name)
            if section_name is not None:
                self.flush_paragraphs(paragraph_buffer, parent_id)
                node_id = self.add_node(
                    STANDARD_SECTION_NODE,
                    tex_node_text(child),
                    self.section_parent(section_name),
                    source_name=section_name,
                )
                level = SECTION_LEVELS[section_name]
                if node_id is not None:
                    self.section_by_level = {key: value for key, value in self.section_by_level.items() if key < level}
                    self.section_by_level[level] = node_id
                continue
            if name in LIST_ENV_NAMES:
                self.flush_paragraphs(paragraph_buffer, parent_id)
                env_id = self.add_node(STANDARD_LIST_NODE, name, self.current_parent(parent_id), source_name=name)
                if env_id is not None:
                    self.walk_children(getattr(child, "contents", []) or [], parent_id=env_id, parent_env=name)
                continue
            if name == "item":
                self.flush_paragraphs(paragraph_buffer, parent_id)
                self.walk_item_contents(child, parent_id=parent_id)
                continue
            if name in INLINE_MATH_ENV_NAMES:
                paragraph_buffer.append(f" {tex_node_text(child)} ")
                continue
            if name in DISPLAY_MATH_ENV_NAMES:
                self.flush_paragraphs(paragraph_buffer, parent_id)
                self.add_node(STANDARD_EQUATION_NODE, tex_node_text(child) or "[MATH]", self.current_parent(parent_id), source_name=name)
                continue
            if name == "caption":
                self.flush_paragraphs(paragraph_buffer, parent_id)
                caption_type = STANDARD_TABLE_CAPTION_NODE if parent_env in {"table", "table*"} else STANDARD_FIGURE_CAPTION_NODE
                self.add_node(caption_type, tex_node_text(child), self.current_parent(parent_id), source_name=name)
                continue
            if name in {"figure", "figure*", "table", "table*"}:
                self.flush_paragraphs(paragraph_buffer, parent_id)
                self.walk_children(getattr(child, "contents", []) or [], parent_id=self.current_parent(parent_id), parent_env=name)
                continue
            if name in BLOCK_ENV_NAMES:
                self.flush_paragraphs(paragraph_buffer, parent_id)
                block_id = self.add_node(
                    semantic_node_type_for_block(name),
                    tex_node_text(child),
                    self.current_parent(parent_id),
                    source_name=name,
                )
                if block_id is None:
                    self.walk_children(getattr(child, "contents", []) or [], parent_id=self.current_parent(parent_id), parent_env=name)
                continue
            if name == "text":
                paragraph_buffer.append(str(child))
                continue
            if hasattr(child, "contents"):
                text = tex_node_text(child)
                if text:
                    paragraph_buffer.append(f" {text} ")
                else:
                    self.walk_children(getattr(child, "contents", []) or [], parent_id=self.current_parent(parent_id), parent_env=name)
        self.flush_paragraphs(paragraph_buffer, parent_id)

    def walk_item_contents(self, item_node: Any, *, parent_id: str) -> None:
        item_buffer: list[str] = []
        for child in getattr(item_node, "contents", []) or []:
            name = tex_node_name(child)
            if name in SKIP_TEX_NODE_NAMES:
                continue
            if name in POISON_TEX_ENV_NAMES:
                raise LayoutBreakerException(f"Encountered complex drawing environment: {name}")
            if name in INLINE_MATH_ENV_NAMES:
                item_buffer.append(f" {tex_node_text(child)} ")
                continue
            if name in DISPLAY_MATH_ENV_NAMES:
                self.flush_list_item_text(item_buffer, parent_id)
                self.add_node(STANDARD_EQUATION_NODE, tex_node_text(child) or "[MATH]", parent_id, source_name=name)
                continue
            if name in LIST_ENV_NAMES:
                self.flush_list_item_text(item_buffer, parent_id)
                env_id = self.add_node(STANDARD_LIST_NODE, name, parent_id, source_name=name)
                if env_id is not None:
                    self.walk_children(getattr(child, "contents", []) or [], parent_id=env_id, parent_env=name)
                continue
            if name == "text":
                item_buffer.append(str(child))
                continue
            if hasattr(child, "contents"):
                text = tex_node_text(child)
                if text:
                    item_buffer.append(f" {text} ")
                else:
                    self.walk_children(getattr(child, "contents", []) or [], parent_id=parent_id, parent_env=name)
        self.flush_list_item_text(item_buffer, parent_id)

    def flush_list_item_text(self, item_buffer: list[str], parent_id: str) -> None:
        if not item_buffer:
            return
        raw_text = "".join(item_buffer)
        item_buffer.clear()
        for paragraph in re.split(r"(?:\r?\n\s*){2,}", raw_text):
            if paragraph.strip():
                self.add_node(STANDARD_LIST_ITEM_NODE, paragraph, parent_id, source_name="item")

    def add_node(self, node_type: str, text: str, parent_id: str, *, source_name: str | None = None) -> str | None:
        clean = clean_equation_text(text) if node_type == STANDARD_EQUATION_NODE else clean_text(text)
        if len(clean) < self.config.min_clean_chars and not (
            clean == "math" or (node_type == STANDARD_EQUATION_NODE and bool(clean))
        ):
            return None
        tex_id = f"T_{self.next_id:06d}"
        self.next_id += 1
        parent_path = self.path_by_id.get(parent_id, ("ROOT",))
        path = (*parent_path, tex_id)
        self.path_by_id[tex_id] = path
        self.parent_by_id[tex_id] = None if parent_id == "ROOT" else parent_id
        self.nodes.append(
            TexAlignmentNode(
                tex_id=tex_id,
                node_type=node_type,
                text=text.strip(),
                clean=clean,
                path_ids=path,
                parent_id=None if parent_id == "ROOT" else parent_id,
                source_name=source_name,
            )
        )
        return tex_id

    def flush_paragraphs(self, paragraph_buffer: list[str], parent_id: str) -> None:
        if not paragraph_buffer:
            return
        raw_text = "".join(paragraph_buffer)
        paragraph_buffer.clear()
        for paragraph in re.split(r"(?:\r?\n\s*){2,}", raw_text):
            if paragraph.strip():
                self.add_node(STANDARD_PARAGRAPH_NODE, paragraph, self.current_parent(parent_id), source_name="text")

    def section_parent(self, name: str) -> str:
        level = SECTION_LEVELS[name]
        lower = [key for key in self.section_by_level if key < level]
        if not lower:
            return "ROOT"
        return self.section_by_level[max(lower)]

    def current_parent(self, fallback_parent: str) -> str:
        if fallback_parent != "ROOT":
            return fallback_parent
        if not self.section_by_level:
            return "ROOT"
        return self.section_by_level[max(self.section_by_level)]


def clean_text(text: Any) -> str:
    """Aggressively normalize PDF/TeX text for fuzzy alignment."""

    value = str(text or "")
    value = expose_math_payload(value)
    value = value.lower()
    value = re.sub(r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])?", " ", value)
    value = re.sub(r"\\.", " ", value)
    value = re.sub(rf"[^0-9a-z\u4e00-\u9fff]+", "", value)
    return value


def clean_equation_text(text: Any) -> str:
    """Normalize display equations while keeping command-only formulas alignable."""

    value = expose_math_payload(str(text or ""))
    value = re.sub(r"\\([a-zA-Z]+)\*?", r" \1 ", value)
    value = re.sub(r"\\.", " ", value)
    value = value.lower()
    value = re.sub(rf"[^0-9a-z\u4e00-\u9fff]+", "", value)
    return value


def expose_math_payload(value: str) -> str:
    """Remove TeX math wrappers while preserving the symbols' core letters/digits."""

    value = re.sub(
        r"\\begin\{([^}]+)\}(.*?)\\end\{\1\}",
        lambda match: f" {match.group(2)} ",
        value,
        flags=re.DOTALL,
    )
    value = re.sub(r"\$\$(.*?)\$\$", lambda match: f" {match.group(1)} ", value, flags=re.DOTALL)
    value = re.sub(r"\$(.*?)\$", lambda match: f" {match.group(1)} ", value, flags=re.DOTALL)
    value = re.sub(r"\\\[(.*?)\\\]", lambda match: f" {match.group(1)} ", value, flags=re.DOTALL)
    value = re.sub(r"\\\((.*?)\\\)", lambda match: f" {match.group(1)} ", value, flags=re.DOTALL)
    value = re.sub(r"\[math\]", " math ", value, flags=re.IGNORECASE)
    return value


def pdf_item_text(item: dict[str, Any]) -> str:
    """Resolve the best text-bearing field from a MinerU content item."""

    for key in (
        "text_for_embedding",
        "text",
        "content",
        "caption",
        "latex",
        "html",
        "table_body",
        "reference_items",
        "spans",
    ):
        if key in item:
            text = stringify_text_payload(item[key])
            if text.strip():
                return text
    item_type = str(item.get("type") or item.get("raw_type") or "").lower()
    if "equation" in item_type or "formula" in item_type:
        return "[MATH]"
    if "figure" in item_type or "image" in item_type:
        return "[FIGURE]"
    if "table" in item_type:
        return "[TABLE]"
    return ""


def stringify_text_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(stringify_text_payload(item) for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("text", "content", "title", "caption", "latex", "html", "raw"):
            if key in value:
                parts.append(stringify_text_payload(value[key]))
        if parts:
            return " ".join(parts)
        return " ".join(stringify_text_payload(item) for item in value.values())
    return str(value)


def tex_node_name(node: Any) -> str:
    name = getattr(node, "name", None)
    if name is None:
        return "text" if isinstance(node, str) else ""
    return str(name)


def tex_node_text(node: Any) -> str:
    name = tex_node_name(node)
    if name in MATH_ENV_NAMES:
        contents = getattr(node, "contents", None)
        if contents:
            text = " ".join(tex_node_text(child) for child in contents)
            return text or "[MATH]"
        return str(node or "[MATH]")
    if name in SKIP_TEX_NODE_NAMES:
        return ""
    if isinstance(node, str):
        return node
    contents = getattr(node, "contents", None)
    if contents:
        return " ".join(tex_node_text(child) for child in contents)
    args = getattr(node, "args", None)
    if args:
        return str(args)
    return str(node or "")


def label_counts(labels: Any) -> dict[int, int]:
    values = labels.detach().cpu().tolist()
    return {label: values.count(label) for label in range(3)}


def connected_node_indices(edge_index: Any, labels: Any) -> set[int]:
    connected: set[int] = set()
    label_values = labels.tolist()
    for edge_pos, label in enumerate(label_values):
        if int(label) not in {int(TexRelationLabel.MERGE), int(TexRelationLabel.PARENT_CHILD)}:
            continue
        connected.add(int(edge_index[0, edge_pos].item()))
        connected.add(int(edge_index[1, edge_pos].item()))
    return connected


def levenshtein_ratio_score(source: str, target: str) -> float:
    if not source or not target:
        return 0.0
    from rapidfuzz.distance import Levenshtein

    distance = float(Levenshtein.distance(source, target))
    denominator = float(max(len(source), len(target), 1))
    return max(0.0, 100.0 * (1.0 - distance / denominator))


def fragment_ratio_score(source: str, target: str) -> float:
    if not source or not target:
        return 0.0
    from rapidfuzz import fuzz

    if source in target or target in source:
        return 100.0
    return float(fuzz.partial_ratio(source, target))


def load_pdf_to_tex_mapping(path: Path) -> dict[str, Any]:
    """Load a PDF-block to TeX-node alignment mapping from JSON or JSONL."""

    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if path.suffix.lower() == ".jsonl":
        mapping: dict[str, Any] = {}
        for line in text.splitlines():
            record = json.loads(line)
            pdf_id = record.get("pdf_id") or record.get("block_id") or record.get("source")
            tex_id = record.get("tex_id") or record.get("target")
            if pdf_id and tex_id:
                mapping[str(pdf_id)] = {"tex_id": str(tex_id), "score": record.get("score")}
        return mapping
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("alignments"), list):
        mapping: dict[str, Any] = {}
        for record in data["alignments"]:
            pdf_id = record.get("pdf_id") or record.get("block_id") or record.get("source")
            tex_id = record.get("tex_id") or record.get("target")
            if pdf_id and tex_id:
                mapping[str(pdf_id)] = {"tex_id": str(tex_id), "score": record.get("score")}
        return mapping
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain an object mapping or an alignments list")
    return data


def label_graph_edges_from_paths(
    data: Any,
    *,
    tex_path: Path,
    pdf_to_tex_path: Path,
    config: LabelGeneratorConfig | None = None,
    orphan_log_path: Path | None = None,
) -> LabelGenerationResult:
    tex_ast = build_tex_ast_from_file(tex_path)
    pdf_to_tex = load_pdf_to_tex_mapping(pdf_to_tex_path)
    return label_graph_edges(
        data,
        tex_ast=tex_ast,
        pdf_to_tex=pdf_to_tex,
        config=config,
        orphan_log_path=orphan_log_path,
    )


def label_graph_edges(
    data: Any,
    *,
    tex_ast: dict[str, Any] | list[dict[str, Any]] | dict[str, dict[str, Any]],
    pdf_to_tex: dict[str, Any],
    config: LabelGeneratorConfig | None = None,
    orphan_log_path: Path | None = None,
) -> LabelGenerationResult:
    """Attach `data.y` edge labels, falling back to None for orphan nodes."""

    import torch

    cfg = config or LabelGeneratorConfig()
    ast_nodes = tex_nodes_by_id(tex_ast)
    node_records = getattr(data, "node_records", None)
    if not isinstance(node_records, list):
        node_records = [{} for _ in range(int(data.num_nodes))]

    node_tex_ids: dict[int, str | None] = {}
    orphans: dict[int, OrphanAlignment] = {}
    for node_index in range(int(data.num_nodes)):
        tex_id, orphan = resolve_node_tex_id(node_index, node_records[node_index], pdf_to_tex, cfg)
        node_tex_ids[node_index] = tex_id
        if orphan is not None:
            orphans[node_index] = orphan
    orphan_list = list(orphans.values())
    if orphan_log_path is not None:
        write_orphan_log(orphan_log_path, orphan_list)
    assert_alignment_quality(num_nodes=int(data.num_nodes), orphan_count=len(orphan_list), config=cfg)

    labels: list[int] = []
    edge_index = data.edge_index.detach().cpu()
    for edge_pos in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        tex_source = node_tex_ids.get(source)
        tex_target = node_tex_ids.get(target)
        if not tex_source or not tex_target:
            labels.append(cfg.orphan_label)
            continue
        label = label_tex_relation(
            tex_source,
            tex_target,
            ast_nodes,
            adjacent_siblings_only=cfg.adjacent_siblings_only,
            directed_parent_child=cfg.directed_parent_child,
        )
        labels.append(int(label))

    y = torch.tensor(labels, dtype=torch.long)
    data.y = y
    data.edge_label = y
    data.label_schema = {
        "task": "edge_relation_classification",
        "labels": {
            int(TexRelationLabel.MERGE): "merge",
            int(TexRelationLabel.PARENT_CHILD): "parent_child",
            int(TexRelationLabel.NONE): "none",
        },
        "orphan_label": cfg.orphan_label,
        "similarity_threshold": cfg.similarity_threshold,
    }
    data.pdf_to_tex = [node_tex_ids.get(idx) for idx in range(int(data.num_nodes))]

    label_counts = {label: labels.count(label) for label in range(3)}
    return LabelGenerationResult(data=data, label_counts=label_counts, orphan_alignments=orphan_list)


def assert_alignment_quality(*, num_nodes: int, orphan_count: int, config: LabelGeneratorConfig) -> bool:
    aligned_nodes = max(0, num_nodes - orphan_count)
    orphan_ratio = orphan_count / max(1, num_nodes)
    if orphan_ratio > config.max_orphan_ratio or aligned_nodes < config.min_aligned_nodes:
        message = (
            "bad alignment quality: "
            f"orphan_count={orphan_count}, num_nodes={num_nodes}, "
            f"orphan_ratio={orphan_ratio:.2%}, max_orphan_ratio={config.max_orphan_ratio:.2%}, "
            f"aligned_nodes={aligned_nodes}, min_aligned_nodes={config.min_aligned_nodes}"
        )
        if config.abort_on_bad_alignment:
            raise AlignmentQualityError(message)
        return False
    return True


def resolve_node_tex_id(
    node_index: int,
    node_record: dict[str, Any],
    pdf_to_tex: dict[str, Any],
    config: LabelGeneratorConfig,
) -> tuple[str | None, OrphanAlignment | None]:
    node_key = resolve_node_key(node_index, node_record, pdf_to_tex)
    if node_key is None:
        return None, OrphanAlignment(node_index=node_index, node_key=str(node_index), reason="missing_alignment")

    raw_alignment = pdf_to_tex.get(node_key)
    tex_id, score = parse_alignment_value(raw_alignment)
    if not tex_id:
        return None, OrphanAlignment(node_index=node_index, node_key=node_key, reason="missing_tex_id", score=score)
    if score is not None and score < config.similarity_threshold:
        return None, OrphanAlignment(node_index=node_index, node_key=node_key, reason="low_similarity", score=score)
    return tex_id, None


def resolve_node_key(node_index: int, node_record: dict[str, Any], pdf_to_tex: dict[str, Any]) -> str | None:
    candidates: list[str] = []
    for key in ("block_id", "id", "pdf_id", "global_order", "visual_order"):
        value = node_record.get(key)
        if value is not None:
            candidates.append(str(value))
    candidates.extend([str(node_index), f"P_{node_index}", f"P_{node_index + 1}", f"B_{node_index}", f"B_{node_index + 1}"])
    for candidate in candidates:
        if candidate in pdf_to_tex:
            return candidate
    return None


def parse_alignment_value(value: Any) -> tuple[str | None, float | None]:
    if isinstance(value, str):
        return value, None
    if isinstance(value, dict):
        tex_id = value.get("tex_id") or value.get("target") or value.get("id")
        score = value.get("score")
        if score is None:
            score = value.get("similarity")
        if score is None:
            score = value.get("confidence")
        parsed_score = float(score) if isinstance(score, (int, float)) else None
        return str(tex_id) if tex_id else None, parsed_score
    return None, None


def write_orphan_log(path: Path, orphans: list[OrphanAlignment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for orphan in orphans:
            file.write(json.dumps(asdict(orphan), ensure_ascii=False) + "\n")
