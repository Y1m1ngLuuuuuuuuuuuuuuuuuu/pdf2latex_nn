"""Generate supervised edge labels from PDF-to-TeX alignments."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.perception.reading_order import filter_graph_content_items, fuse_micro_nodes, is_toc_record, style_spans_text
from src.perception.title_features import title_numbering_level, title_numbering_path
from src.reasoning.latex_flattener import LatexFlattenerConfig, flatten_latex_file
from src.reasoning.tex_ast_builder import build_tex_ast_from_file, tex_nodes_by_id
from src.reasoning.tex_relation_labeler import TexRelationLabel, label_tex_relation


@dataclass(frozen=True)
class LabelGeneratorConfig:
    similarity_threshold: float = 0.55
    adjacent_siblings_only: bool = True
    directed_parent_child: bool = True
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
    tex_lookahead_nodes: int = 24
    tail_absorption_nodes: int = 3
    equation_blind_alignment_window: int = 2
    relation_strategy: str = "visual_first"
    visual_heading_font_scale: float = 1.12
    visual_bold_heading_font_scale: float = 1.05
    max_orphan_ratio: float = 0.15
    max_unmapped_tex_ratio: float = 0.30
    max_isolated_node_ratio: float = 0.85
    min_visual_parent_label_recall: float = 0.98
    min_section_nodes: int = 1
    abort_on_bad_alignment: bool = False
    output_mapping_json: Path | None = None
    exclude_expected_visual_orphans: bool = True
    page_edge_y_threshold: float = 45.0
    page_bottom_y_threshold: float = 955.0
    max_expected_orphan_clean_chars: int = 24
    caption_fallback_threshold: float = 80.0
    global_text_fallback_threshold: float = 92.0
    global_text_fallback_min_chars: int = 18


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
STANDARD_REFERENCE_NODE = "reference"
ALIGNABLE_TEX_NODE_TYPES = {
    STANDARD_SECTION_NODE,
    STANDARD_PARAGRAPH_NODE,
    STANDARD_EQUATION_NODE,
    STANDARD_LIST_ITEM_NODE,
    STANDARD_FIGURE_CAPTION_NODE,
    STANDARD_TABLE_CAPTION_NODE,
    STANDARD_REFERENCE_NODE,
}
FLOAT_TEX_NODE_TYPES = {
    STANDARD_FIGURE_CAPTION_NODE,
    STANDARD_TABLE_CAPTION_NODE,
}
WEAK_TEX_NODE_TYPES = FLOAT_TEX_NODE_TYPES | {STANDARD_REFERENCE_NODE}
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
    "thebibliography",
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
    "addtocounter",
    "addtolength",
    "bigskip",
    "bibliography",
    "bibliographystyle",
    "centering",
    "clearpage",
    "color",
    "definecolor",
    "documentclass",
    "hfill",
    "hphantom",
    "hrule",
    "hskip",
    "hspace",
    "includegraphics",
    "indent",
    "usepackage",
    "label",
    "linebreak",
    "maketitle",
    "medskip",
    "newcommand",
    "newpage",
    "noindent",
    "pagebreak",
    "pagestyle",
    "par",
    "phantom",
    "protect",
    "raggedleft",
    "raggedright",
    "ref",
    "renewcommand",
    "resizebox",
    "rule",
    "scalebox",
    "setcounter",
    "setlength",
    "settowidth",
    "smallskip",
    "thispagestyle",
    "vfill",
    "vphantom",
    "vrule",
    "vskip",
    "vspace",
    "cite",
}
HEADING_TYPES = {"title", "section", "subsection", "subsubsection", "heading"}
SECTION_TYPE_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}
LIST_MARKER_RE = re.compile(r"^\s*(?:[\u2022\u25E6\u25CB\u25AA\-\*]|\d+[\.\)]|[a-zA-Z][\.\)])\s+")
ORDERED_LIST_MARKER_RE = re.compile(r"^\s*(\d+)[\.\)]\s+")
ALPHA_OR_ROMAN_HEADING_RE = re.compile(r"^\s*([A-Za-z]+)[\.\)]\s+(.*)$")
SENTENCE_END_RE = re.compile(r"[。.!?！？]\s*$")
TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?。！？]\s*(?:[])}\"'”’»]+)?\s*$")
HYPHEN_END_RE = re.compile(r"[-\u2010\u2011\u2012\u2013\u2014]\s*$")
UPPERCASE_START_RE = re.compile(r"^\s*(?:[\"'“‘(\[]\s*)*[A-Z]")
VISIBLE_LIST_INTRO_RE = re.compile(r"[:：]\s*(?:[])}\"'”’»]+)?\s*$")
ALGORITHM_IO_LABEL_RE = re.compile(
    r"^\s*(?:input|output|require|ensure|parameters?|returns?)\s*[:：]\s*$",
    re.IGNORECASE,
)
RUN_IN_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"[A-Z][A-Za-z][A-Za-z0-9/\-\s]{1,48}[.:]"
    r"|[IVXLCDM]{1,8}\."
    r"|[A-Z]\."
    r"|\d+(?:\.\d+)*[\.\)]"
    r")\s+"
)
MERGE_COMPATIBLE_PDF_TYPES = {"text", "equation", "reference"}
CAPTION_TEX_NODE_TYPES = {STANDARD_FIGURE_CAPTION_NODE, STANDARD_TABLE_CAPTION_NODE}
VISUAL_FILE_RE = re.compile(r"(?:[A-Za-z0-9_.+~/-]+)\.(?:png|jpe?g|pdf|eps|svg)", re.IGNORECASE)
VISUAL_OPTION_RE = re.compile(
    r"\b(?:width|height|scale|angle|trim|clip|keepaspectratio)\s*=?\s*"
    r"(?:[0-9.]+)?\s*(?:\\?(?:textwidth|linewidth|columnwidth|paperwidth)|[a-z]+)?",
    re.IGNORECASE,
)
VISUAL_LENGTH_RE = re.compile(r"\\(?:textwidth|linewidth|columnwidth|paperwidth|paperheight|hsize|vsize)\b")
LAYOUT_ONLY_CLEAN_VALUES = {
    "b",
    "c",
    "center",
    "centering",
    "empty",
    "h",
    "hb",
    "hbp",
    "ht",
    "htb",
    "htbp",
    "l",
    "plain",
    "r",
    "t",
    "tb",
    "tbp",
}
LAYOUT_COLOR_CLEAN_VALUES = {
    "black",
    "blue",
    "brown",
    "cyan",
    "gray",
    "green",
    "grey",
    "magenta",
    "orange",
    "pink",
    "purple",
    "red",
    "violet",
    "white",
    "yellow",
}
LAYOUT_DIMENSION_RE = re.compile(
    r"^(?:[tblrc])?(?:\d+(?:\.\d+)?|\d*\.\d+)(?:pt|em|ex|cm|mm|in|pc|px)?$"
)
AUXILIARY_PDF_TYPES = {
    "page_header",
    "page_footer",
    "page_number",
    "header",
    "footer",
    "page_aside_text",
    "page_footnote",
    "footnote",
    "watermark",
}
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
        self._edge_attr: Any | None = None
        self._edge_attr_fields: dict[str, int] = {}

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
            "strategy": "v7_texsoup_ast_sliding_window_v1",
            "pipeline_version": "v7",
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
        graph.alignment_quality = self.alignment_quality
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
        filtered_items = filter_graph_content_items(items)
        if expected_node_count is not None and len(filtered_items) == expected_node_count:
            items = filtered_items
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
                next_tex_cursor = self.find_better_tex_candidate(pdf_node, tex_sequence, tex_cursor)
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

        self.apply_global_caption_fallback(matches)
        self.apply_global_text_fallback(matches)
        self.apply_neighbor_fragment_fallback(matches)
        return matches

    def apply_global_caption_fallback(self, matches: list[AlignmentMatch]) -> None:
        """Weakly recover missed figure/table captions after monotonic alignment.

        Floats are often written far from their rendered location. The main
        sliding-window pass stays monotonic to avoid O(N^2) noise, then this
        narrow fallback only considers caption TeX nodes and still assigns each
        caption at most once.
        """

        if self.config.caption_fallback_threshold <= 0:
            return
        used_tex_ids = {match.tex_id for match in matches if match.tex_id}
        available_captions = [
            node
            for node in self.tex_nodes.values()
            if node.node_type in CAPTION_TEX_NODE_TYPES
            and node.tex_id not in used_tex_ids
            and len(node.clean) >= self.config.min_clean_chars
        ]
        if not available_captions:
            return

        for pdf_node in self.pdf_nodes:
            if matches[pdf_node.node_index].tex_id is not None:
                continue
            if self.is_expected_visual_orphan(pdf_node):
                continue
            if len(pdf_node.clean) < self.config.min_clean_chars:
                continue
            best_caption: TexAlignmentNode | None = None
            best_score = 0.0
            for caption in available_captions:
                score = max(
                    fragment_ratio_score(pdf_node.clean, caption.clean),
                    levenshtein_ratio_score(pdf_node.clean, caption.clean),
                )
                if score > best_score:
                    best_caption = caption
                    best_score = score
            if best_caption is None or best_score < self.config.caption_fallback_threshold:
                continue
            matches[pdf_node.node_index] = AlignmentMatch(
                pdf_node_index=pdf_node.node_index,
                tex_id=best_caption.tex_id,
                score=best_score,
            )
            self.tex_to_pdf_indices.setdefault(best_caption.tex_id, []).append(pdf_node.node_index)
            available_captions = [caption for caption in available_captions if caption.tex_id != best_caption.tex_id]
            if not available_captions:
                return

    def apply_global_text_fallback(self, matches: list[AlignmentMatch]) -> None:
        """Recover high-confidence residual text matches after monotonic pass.

        This is intentionally a narrow orphan-only fallback. It does not replace
        the ordered sliding window, but it rescues body paragraphs displaced by
        floats/tables when a PDF block is a near-literal fragment of a TeX node.
        """

        if self.config.global_text_fallback_threshold <= 0:
            return
        candidates = [
            node
            for node in self.tex_nodes.values()
            if node.node_type in {STANDARD_PARAGRAPH_NODE, STANDARD_LIST_ITEM_NODE}
            and len(node.clean) >= self.config.global_text_fallback_min_chars
        ]
        if not candidates:
            return
        for pdf_node in self.pdf_nodes:
            index = pdf_node.node_index
            if matches[index].tex_id is not None:
                continue
            if self.is_expected_visual_orphan(pdf_node):
                continue
            if canonical_pdf_merge_type(pdf_node.item) not in {"text", "reference"}:
                continue
            if len(pdf_node.clean) < self.config.global_text_fallback_min_chars:
                continue
            best_node: TexAlignmentNode | None = None
            best_score = 0.0
            for tex_node in candidates:
                score = max(
                    fragment_ratio_score(pdf_node.clean, tex_node.clean),
                    levenshtein_ratio_score(pdf_node.clean, tex_node.clean),
                )
                if score > best_score:
                    best_node = tex_node
                    best_score = score
            if best_node is None or best_score < self.config.global_text_fallback_threshold:
                continue
            matches[index] = AlignmentMatch(pdf_node_index=index, tex_id=best_node.tex_id, score=best_score)
            self.tex_to_pdf_indices.setdefault(best_node.tex_id, []).append(index)
            self.tex_to_pdf_indices[best_node.tex_id] = sorted(set(self.tex_to_pdf_indices[best_node.tex_id]))

    def apply_neighbor_fragment_fallback(self, matches: list[AlignmentMatch]) -> None:
        """Recover short orphan fragments that are continuations of neighbors.

        A common MinerU split is a long paragraph block followed by a short
        final line in the next column. If the main sliding pass aligns the long
        block but leaves the short line orphaned, this local fallback can safely
        attach the short line to the same TeX node when it is a literal fragment
        of that TeX text.
        """

        for index, pdf_node in enumerate(self.pdf_nodes):
            if matches[index].tex_id is not None:
                continue
            if self.is_expected_visual_orphan(pdf_node):
                continue
            if len(pdf_node.clean) < self.config.min_clean_chars:
                continue
            for neighbor_index in (index - 1, index + 1):
                if not 0 <= neighbor_index < len(matches):
                    continue
                neighbor_tex_id = matches[neighbor_index].tex_id
                if not neighbor_tex_id:
                    continue
                tex_node = self.tex_nodes.get(neighbor_tex_id)
                if tex_node is None or tex_node.node_type not in {STANDARD_PARAGRAPH_NODE, STANDARD_LIST_ITEM_NODE}:
                    continue
                if not self.is_tex_fragment(pdf_node.clean, tex_node.clean):
                    continue
                score = fragment_ratio_score(pdf_node.clean, tex_node.clean)
                matches[index] = AlignmentMatch(pdf_node_index=index, tex_id=neighbor_tex_id, score=score)
                self.tex_to_pdf_indices.setdefault(neighbor_tex_id, []).append(index)
                self.tex_to_pdf_indices[neighbor_tex_id] = sorted(set(self.tex_to_pdf_indices[neighbor_tex_id]))
                break

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
        if tex_node.node_type == STANDARD_SECTION_NODE and not self.pdf_node_can_match_section(self.pdf_nodes[start_index], tex_node):
            return None
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
        if tex_node.node_type != STANDARD_SECTION_NODE and self.is_tex_fragment(self.pdf_nodes[start_index].clean, tex_node.clean):
            return start_index, start_index, fragment_ratio_score(self.pdf_nodes[start_index].clean, tex_node.clean)
        return None

    def pdf_node_can_match_section(self, pdf_node: PdfAlignmentNode, tex_node: TexAlignmentNode) -> bool:
        """Avoid matching short TeX headings to arbitrary long body paragraphs."""

        raw_type = canonical_pdf_type(pdf_node.item)
        if raw_type in HEADING_TYPES:
            return True
        if len(pdf_node.clean) < self.config.min_clean_chars or len(tex_node.clean) < self.config.min_clean_chars:
            return False
        text = pdf_node.text.strip()
        if LIST_MARKER_RE.match(text):
            return False
        if len(pdf_node.clean) <= max(len(tex_node.clean) * 2 + 8, 32) and fragment_ratio_score(pdf_node.clean, tex_node.clean) >= 92.0:
            return True
        return False

    def find_blind_equation_window(self, start_index: int) -> tuple[int, int, float] | None:
        upper = min(len(self.pdf_nodes), start_index + max(1, self.config.equation_blind_alignment_window + 1))
        for index in range(start_index, upper):
            if canonical_pdf_merge_type(self.pdf_nodes[index].item) == "equation":
                return index, index, 100.0
        return None

    def find_better_tex_candidate(
        self,
        pdf_node: PdfAlignmentNode,
        tex_sequence: list[TexAlignmentNode],
        tex_cursor: int,
    ) -> int | None:
        pdf_clean = pdf_node.clean
        current = tex_sequence[tex_cursor]
        if current.node_type == STANDARD_SECTION_NODE and not self.pdf_node_can_match_section(pdf_node, current):
            return None
        current_score = fragment_ratio_score(pdf_clean, current.clean)
        best_cursor: int | None = None
        best_score = current_score
        upper = min(len(tex_sequence), tex_cursor + 1 + max(0, self.config.tex_lookahead_nodes))
        for candidate_cursor in range(tex_cursor + 1, upper):
            candidate = tex_sequence[candidate_cursor]
            if candidate.node_type == STANDARD_SECTION_NODE and not self.pdf_node_can_match_section(pdf_node, candidate):
                continue
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
        self._edge_attr = graph.edge_attr.detach().cpu() if hasattr(graph, "edge_attr") and graph.edge_attr is not None else None
        self._edge_attr_fields = edge_attr_fields_from_graph(graph)
        for edge_pos in range(edge_index.shape[1]):
            source = int(edge_index[0, edge_pos].item())
            target = int(edge_index[1, edge_pos].item())
            labels.append(self.infer_relation(source, target, edge_pos=edge_pos))
        return torch.tensor(labels, dtype=torch.long)

    def infer_relation(self, source_index: int, target_index: int, *, edge_pos: int | None = None) -> int:
        if self.node_is_unlabelable_visual_orphan(source_index) or self.node_is_unlabelable_visual_orphan(target_index):
            return int(TexRelationLabel.NONE)
        source_item = self.pdf_nodes[source_index].item if 0 <= source_index < len(self.pdf_nodes) else {}
        target_item = self.pdf_nodes[target_index].item if 0 <= target_index < len(self.pdf_nodes) else {}
        if relation_layers_are_incompatible(source_item, target_item):
            return int(TexRelationLabel.NONE)
        visual_relation = self.infer_visual_relation(source_index, target_index)
        if visual_relation is not None:
            return visual_relation
        source_match = self.matches[source_index] if 0 <= source_index < len(self.matches) else None
        target_match = self.matches[target_index] if 0 <= target_index < len(self.matches) else None
        if source_match is None or target_match is None or not source_match.tex_id or not target_match.tex_id:
            return int(TexRelationLabel.NONE)
        path_u = self.tex_nodes[source_match.tex_id].path_ids
        path_v = self.tex_nodes[target_match.tex_id].path_ids
        if path_u == path_v:
            if self.is_document_root_scoped_match(source_match) or self.is_document_root_scoped_match(target_match):
                return int(TexRelationLabel.NONE)
            if not self.same_tex_node_are_adjacent_fragments(source_index, target_index, source_match.tex_id):
                return int(TexRelationLabel.NONE)
            if self.same_tex_node_merge_crosses_list_marker(source_index, target_index):
                return int(TexRelationLabel.NONE)
            if self.same_tex_node_merge_hits_visual_boundary(source_index, target_index, edge_pos=edge_pos):
                return int(TexRelationLabel.NONE)
            if same_tex_node_can_merge(self.pdf_nodes[source_index], self.pdf_nodes[target_index]):
                return int(TexRelationLabel.MERGE)
            return int(TexRelationLabel.NONE)
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
    ) -> int | None:
        hierarchy = self.visual_hierarchy
        if hierarchy is None or not hierarchy.heading_ids:
            return None
        if hierarchy.parent_by_node.get(target_index) == source_index:
            return int(TexRelationLabel.PARENT_CHILD)
        return None

    def is_first_pdf_anchor(self, pdf_index: int, tex_id: str | None) -> bool:
        if not tex_id:
            return False
        indices = self.tex_to_pdf_indices.get(tex_id, [])
        return bool(indices) and pdf_index == min(indices)

    def same_tex_node_are_adjacent_fragments(self, source_index: int, target_index: int, tex_id: str | None) -> bool:
        """Only direct neighboring PDF fragments of one TeX node become MERGE.

        Fuzzy alignment maps one TeX paragraph/list item to a span of PDF
        boxes. MERGE supervision should mark the stitch points inside that
        span, not every candidate edge between arbitrary boxes in the span.
        """

        if not tex_id:
            return False
        indices = sorted(set(self.tex_to_pdf_indices.get(tex_id, [])))
        if source_index not in indices or target_index not in indices:
            return False
        return abs(indices.index(source_index) - indices.index(target_index)) == 1

    def same_tex_node_merge_crosses_list_marker(self, source_index: int, target_index: int) -> bool:
        lower, upper = sorted((source_index, target_index))
        if upper - lower <= 1:
            return False
        for index in range(lower + 1, upper):
            if 0 <= index < len(self.pdf_nodes) and LIST_MARKER_RE.match(self.pdf_nodes[index].text):
                return True
        return False

    def same_tex_node_merge_hits_visual_boundary(
        self,
        source_index: int,
        target_index: int,
        *,
        edge_pos: int | None = None,
    ) -> bool:
        """Reject same-TeX MERGE labels across visible block boundaries.

        TeX paragraphs are often coarse: a single paragraph/list item can
        contain run-in headings, examples, equations, or visually independent
        statements.  Those fragments share a TeX id but should teach the GNN
        NONE, not MERGE.  These barriers are deliberately label-time rules so
        the model never learns physically impossible stitch edges.
        """

        source_node = self.pdf_nodes[source_index]
        target_node = self.pdf_nodes[target_index]
        source_item = source_node.item
        target_item = target_node.item
        source_type = strict_pdf_merge_type(source_item)
        target_type = strict_pdf_merge_type(target_item)

        if LIST_MARKER_RE.match(target_node.text):
            return True
        if source_type != target_type:
            return True
        if target_index < source_index and is_run_in_heading_like(source_node):
            return True
        if is_run_in_heading_like(target_node):
            return True
        if self.same_tex_text_merge_crosses_paragraph_boundary(
            source_node,
            target_node,
            source_type,
            target_type,
            edge_pos=edge_pos,
        ):
            return True
        if self.same_tex_merge_hits_geometry_boundary(source_node, target_node):
            return True
        if edge_pos is not None and self.edge_attr_blocks_same_tex_merge(edge_pos):
            return True
        return False

    def same_tex_text_merge_crosses_paragraph_boundary(
        self,
        source_node: PdfAlignmentNode,
        target_node: PdfAlignmentNode,
        source_type: str,
        target_type: str,
        *,
        edge_pos: int | None = None,
    ) -> bool:
        """Block same-TeX MERGE across obvious visual paragraph boundaries.

        Some TeX parser fallbacks flatten multiple visual paragraphs into one
        coarse TeX paragraph.  Without this guard, adjacent PDF blocks such as
        ``"... finished."`` -> ``"New paragraph starts ..."`` become MERGE
        positives simply because they share that coarse TeX id.  We only apply
        the rule to text-text pairs and keep hyphenated continuations intact.
        """

        if source_type != "text" or target_type != "text":
            return False
        if source_node.node_index == target_node.node_index:
            return False
        source_terminal = 0.0
        source_hyphen = 0.0
        if edge_pos is not None:
            source_terminal = self.edge_attr_value(edge_pos, "source_ends_with_terminal_punctuation")
            source_hyphen = self.edge_attr_value(edge_pos, "source_ends_with_hyphen")
        if source_terminal <= 0.0:
            source_terminal = float(ends_with_terminal_punctuation(source_node.text))
        if source_hyphen <= 0.0:
            source_hyphen = float(ends_with_hyphen(source_node.text))
        if source_hyphen >= 0.5 or source_terminal < 0.5:
            return False
        return starts_with_uppercase_text(target_node.text)

    def edge_attr_blocks_same_tex_merge(self, edge_pos: int) -> bool:
        if self._edge_attr is None or not self._edge_attr_fields:
            return False
        if edge_pos < 0 or edge_pos >= int(self._edge_attr.shape[0]):
            return False
        has_gutter = self.edge_attr_value(edge_pos, "has_x_gutter")
        y_overlap = self.edge_attr_value(edge_pos, "y_overlap_ratio")
        if has_gutter >= 0.5 and y_overlap > 0.3:
            return True
        far = self.edge_attr_value(edge_pos, "index_delta_bin_far")
        reverse = self.edge_attr_value(edge_pos, "index_delta_bin_reverse")
        source_hyphen = self.edge_attr_value(edge_pos, "source_ends_with_hyphen")
        if far >= 0.5 and source_hyphen < 0.5:
            return True
        if reverse >= 0.5:
            return True
        return False

    def same_tex_merge_hits_geometry_boundary(self, source_node: PdfAlignmentNode, target_node: PdfAlignmentNode) -> bool:
        source_bbox = last_bbox(source_node.item.get("bbox"))
        target_bbox = first_bbox(target_node.item.get("bbox"))
        if source_bbox is None or target_bbox is None:
            return False
        if target_node.node_index < source_node.node_index:
            return True
        y_overlap = bbox_y_overlap_ratio(source_bbox, target_bbox)
        if y_overlap > 0.3 and bbox_x_gap(source_bbox, target_bbox) > 30.0:
            return True
        if abs(target_node.node_index - source_node.node_index) > 5 and not ends_with_hyphen(source_node.text):
            return True
        return False

    def edge_attr_value(self, edge_pos: int, field_name: str) -> float:
        column = self._edge_attr_fields.get(field_name)
        if column is None or self._edge_attr is None:
            return 0.0
        return float(self._edge_attr[edge_pos, column].item())

    def same_reference_scope(self, source_index: int, target_index: int) -> bool:
        return False

    def node_is_unlabelable_visual_orphan(self, node_index: int) -> bool:
        if not 0 <= node_index < len(self.pdf_nodes):
            return True
        return self.is_expected_visual_orphan(self.pdf_nodes[node_index])

    def assert_alignment_quality(self, *, graph: Any | None = None, labels: Any | None = None) -> None:
        exempt_visual_orphans = {
            node.node_index
            for node in self.pdf_nodes
            if self.is_expected_visual_orphan(node)
        }
        document_root_scoped = {
            match.pdf_node_index
            for match in self.matches
            if self.is_document_root_scoped_match(match)
        }
        effective_matches = [
            match for match in self.matches if match.pdf_node_index not in exempt_visual_orphans
        ]
        orphan_count = sum(1 for match in effective_matches if not match.tex_id)
        orphan_ratio = orphan_count / max(1, len(effective_matches))
        alignable_tex_nodes = [node for node in self.tex_nodes.values() if self.is_alignable_tex_node(node)]
        main_alignable_tex_nodes = [node for node in alignable_tex_nodes if node.node_type not in WEAK_TEX_NODE_TYPES]
        float_tex_nodes = [node for node in alignable_tex_nodes if node.node_type in FLOAT_TEX_NODE_TYPES]
        weak_tex_nodes = [node for node in alignable_tex_nodes if node.node_type in WEAK_TEX_NODE_TYPES]
        mapped_tex_ids = {match.tex_id for match in self.matches if match.tex_id}
        raw_unmapped_tex_count = sum(1 for node in alignable_tex_nodes if node.tex_id not in mapped_tex_ids)
        raw_unmapped_tex_ratio = raw_unmapped_tex_count / max(1, len(alignable_tex_nodes))
        unmapped_tex_count = sum(1 for node in main_alignable_tex_nodes if node.tex_id not in mapped_tex_ids)
        unmapped_tex_ratio = unmapped_tex_count / max(1, len(main_alignable_tex_nodes))
        unmapped_float_tex_count = sum(1 for node in float_tex_nodes if node.tex_id not in mapped_tex_ids)
        section_count = sum(1 for node in self.tex_nodes.values() if node.node_type == STANDARD_SECTION_NODE)
        paragraph_pdf_count = sum(1 for node in self.pdf_nodes if canonical_pdf_merge_type(node.item) == "text" and len(node.clean) >= self.config.min_clean_chars)
        metadata_node_indices = {
            node.node_index
            for node in self.pdf_nodes
            if layout_layer_name(node.item) == "metadata_layer"
        }
        metadata_orphan_count = sum(
            1 for match in self.matches if match.pdf_node_index in metadata_node_indices and not match.tex_id
        )
        metadata_orphan_ratio = metadata_orphan_count / max(1, len(metadata_node_indices))
        isolated_count = 0
        isolated_ratio = 0.0
        if graph is not None and labels is not None and hasattr(graph, "edge_index"):
            connected = connected_node_indices(graph.edge_index.detach().cpu(), labels.detach().cpu())
            candidates = [
                node.node_index
                for node in self.pdf_nodes
                if len(node.clean) >= self.config.min_clean_chars
                and node.node_index not in exempt_visual_orphans
                and node.node_index not in document_root_scoped
            ]
            isolated_count = sum(1 for node_index in candidates if node_index not in connected)
            isolated_ratio = isolated_count / max(1, len(candidates))
        visual_parent_quality = self.visual_parent_label_quality(graph=graph, labels=labels)
        self.alignment_quality = {
            "orphan_count": orphan_count,
            "raw_orphan_count": sum(1 for match in self.matches if not match.tex_id),
            "num_pdf_nodes": len(self.matches),
            "effective_pdf_nodes": len(effective_matches),
            "orphan_ratio": orphan_ratio,
            "max_orphan_ratio": self.config.max_orphan_ratio,
            "expected_visual_orphan_exempt_count": len(exempt_visual_orphans),
            "document_root_scoped_count": len(document_root_scoped),
            "alignable_tex_count": len(alignable_tex_nodes),
            "main_alignable_tex_count": len(main_alignable_tex_nodes),
            "weak_tex_count": len(weak_tex_nodes),
            "float_tex_count": len(float_tex_nodes),
            "raw_unmapped_tex_count": raw_unmapped_tex_count,
            "raw_unmapped_tex_ratio": raw_unmapped_tex_ratio,
            "unmapped_tex_count": unmapped_tex_count,
            "unmapped_tex_ratio": unmapped_tex_ratio,
            "unmapped_float_tex_count": unmapped_float_tex_count,
            "max_unmapped_tex_ratio": self.config.max_unmapped_tex_ratio,
            "metadata_pdf_node_count": len(metadata_node_indices),
            "metadata_orphan_count": metadata_orphan_count,
            "metadata_orphan_ratio": metadata_orphan_ratio,
            "section_count": section_count,
            "min_section_nodes": self.config.min_section_nodes,
            "isolated_node_count": isolated_count,
            "isolated_node_ratio": isolated_ratio,
            "max_isolated_node_ratio": self.config.max_isolated_node_ratio,
            **visual_parent_quality,
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
        visual_parent_recall = visual_parent_quality.get("visual_parent_label_recall")
        if (
            visual_parent_recall is not None
            and visual_parent_quality.get("visual_parent_pairs", 0) > 0
            and float(visual_parent_recall) < self.config.min_visual_parent_label_recall
        ):
            failures.append(
                "visual_parent_label_recall="
                f"{float(visual_parent_recall):.2%} < {self.config.min_visual_parent_label_recall:.2%}"
            )
        if failures and self.config.abort_on_bad_alignment:
            raise AlignmentQualityError("bad alignment quality: " + "; ".join(failures))

    def visual_parent_label_quality(self, *, graph: Any | None, labels: Any | None) -> dict[str, Any]:
        hierarchy = self.visual_hierarchy
        if hierarchy is None or graph is None or labels is None or not hasattr(graph, "edge_index"):
            return {
                "visual_parent_pairs": 0,
                "visual_parent_labeled_pairs": 0,
                "visual_parent_missing_candidate_pairs": 0,
                "visual_parent_label_recall": None,
                "min_visual_parent_label_recall": self.config.min_visual_parent_label_recall,
            }
        expected_pairs: list[tuple[int, int]] = []
        for child_index, parent_index in hierarchy.parent_by_node.items():
            if parent_index is None:
                continue
            if self.node_is_unlabelable_visual_orphan(parent_index) or self.node_is_unlabelable_visual_orphan(child_index):
                continue
            parent_item = self.pdf_nodes[parent_index].item if 0 <= parent_index < len(self.pdf_nodes) else {}
            child_item = self.pdf_nodes[child_index].item if 0 <= child_index < len(self.pdf_nodes) else {}
            if relation_layers_are_incompatible(parent_item, child_item):
                continue
            if not visual_parent_pair_is_quality_gate_required(
                self.pdf_nodes[parent_index],
                self.pdf_nodes[child_index],
            ):
                continue
            expected_pairs.append((parent_index, child_index))
        if not expected_pairs:
            return {
                "visual_parent_pairs": 0,
                "visual_parent_labeled_pairs": 0,
                "visual_parent_missing_candidate_pairs": 0,
                "visual_parent_label_recall": None,
                "min_visual_parent_label_recall": self.config.min_visual_parent_label_recall,
            }
        edge_index = graph.edge_index.detach().cpu()
        label_values = labels.detach().cpu().tolist()
        edge_labels = {
            (int(edge_index[0, edge_pos].item()), int(edge_index[1, edge_pos].item())): int(label)
            for edge_pos, label in enumerate(label_values)
        }
        labeled = sum(1 for pair in expected_pairs if edge_labels.get(pair) == int(TexRelationLabel.PARENT_CHILD))
        missing_candidate = sum(1 for pair in expected_pairs if pair not in edge_labels)
        return {
            "visual_parent_pairs": len(expected_pairs),
            "visual_parent_labeled_pairs": labeled,
            "visual_parent_missing_candidate_pairs": missing_candidate,
            "visual_parent_label_recall": labeled / max(1, len(expected_pairs)),
            "min_visual_parent_label_recall": self.config.min_visual_parent_label_recall,
        }

    def write_mapping_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        expected_orphan_exemptions = [
            node.node_index for node in self.pdf_nodes if self.is_expected_visual_orphan(node)
        ]
        document_root_scoped = [
            match.pdf_node_index for match in self.matches if self.is_document_root_scoped_match(match)
        ]
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
            "expected_orphan_exemptions": expected_orphan_exemptions,
            "document_root_scoped": document_root_scoped,
            "visual_hierarchy": visual_hierarchy_payload(self.visual_hierarchy),
            "tex_nodes": [tex_alignment_node_payload(node) for node in self.tex_nodes.values()],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def is_expected_visual_orphan(self, node: PdfAlignmentNode) -> bool:
        if not self.config.exclude_expected_visual_orphans:
            return False
        if is_toc_record(node.item):
            return True
        if layout_layer_name(node.item) == "metadata_layer":
            return True
        raw_type = canonical_pdf_type(node.item)
        if raw_type in AUXILIARY_PDF_TYPES:
            return True
        if raw_type in HEADING_TYPES:
            return False
        text = " ".join(node.text.split())
        text_lower = text.casefold()
        if "copyright" in text_lower and "all rights reserved" in text_lower:
            return True
        bbox = first_bbox(node.item.get("bbox"))
        if bbox is None:
            return False
        y0, y1 = bbox[1], bbox[3]
        near_page_edge = y0 <= self.config.page_edge_y_threshold or y1 >= self.config.page_bottom_y_threshold
        if not near_page_edge:
            return False
        clean = node.clean
        if not clean:
            return True
        if len(clean) <= self.config.max_expected_orphan_clean_chars and len(text) <= 80:
            return True
        if re.fullmatch(r"\d{1,4}", clean):
            return True
        return False

    def is_document_root_scoped_match(self, match: AlignmentMatch) -> bool:
        if not match.tex_id:
            return False
        tex_node = self.tex_nodes.get(match.tex_id)
        if tex_node is None:
            return False
        if tex_node.parent_id is not None:
            return False
        if tex_node.node_type == STANDARD_SECTION_NODE:
            return False
        return True


def tex_alignment_node_payload(node: TexAlignmentNode) -> dict[str, Any]:
    payload = asdict(node)
    payload["clean_text"] = node.clean_text
    return payload


def build_visual_hierarchy(nodes: list[PdfAlignmentNode], *, config: AlignmentLabelerConfig) -> VisualHierarchy:
    """Build a lightweight heading/paragraph hierarchy from visual PDF node metadata.

    TeX AST paths are fragile in real arXiv sources. This stack uses only the
    already ordered MinerU/PyMuPDF records: block type, heading prefix, font
    statistics, short standalone-title cues, and visible list-introduction
    anchors.  TeX ``itemize`` / ``enumerate`` containers do not have their own
    PDF boxes, so a paragraph ending with a colon can become the visible proxy
    parent for the following bullet/numbered items.
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
    stack: list[tuple[int, int, tuple[str, ...] | None]] = []
    active_list_parent: int | None = None
    active_list_next_number: int | None = None
    active_list_last_pos = -1
    recent_list_intro_by_scope: dict[int | None, tuple[int, int]] = {}
    in_author_biography_backmatter = False
    effective_pos = 0

    for node in nodes:
        node_id = node.node_index
        layer = layout_layer_name(node.item)
        if layer == "noise_layer" or (layer == "metadata_layer" and not metadata_layer_heading_override(node.item)):
            parent_by_node[node_id] = None
            children_by_parent.setdefault(None, []).append(node_id)
            continue
        if node_id in document_title_ids:
            parent_by_node[node_id] = None
            children_by_parent.setdefault(None, []).append(node_id)
            continue
        effective_pos += 1
        if node_id in heading_ids:
            in_author_biography_backmatter = False
            active_list_parent = None
            active_list_next_number = None
            active_list_last_pos = -1
            level = heading_levels[node_id]
            numbering_path = title_numbering_path(node.text)
            while stack and heading_scope_must_close_before_child(
                parent_text=nodes[stack[-1][0]].text,
                child_text=node.text,
            ):
                stack.pop()
            while stack and stack[-1][1] >= level:
                stack.pop()
            if numbering_path is not None:
                while stack and not heading_numbering_parent_is_compatible(stack[-1][2], numbering_path):
                    stack.pop()
            parent_id = stack[-1][0] if stack else None
            parent_by_node[node_id] = parent_id
            children_by_parent.setdefault(parent_id, []).append(node_id)
            stack.append((node_id, level, numbering_path))
            continue
        list_number = ordered_list_marker_number(node.text)
        parent_id = stack[-1][0] if stack else None
        if item_looks_like_author_biography(node.item) or in_author_biography_backmatter:
            parent_by_node[node_id] = None
            children_by_parent.setdefault(None, []).append(node_id)
            in_author_biography_backmatter = True
            active_list_parent = None
            active_list_next_number = None
            active_list_last_pos = -1
            continue
        if stack and reference_scope_must_close_before_item(parent_text=nodes[stack[-1][0]].text, item=node.item):
            stack.pop()
            parent_id = stack[-1][0] if stack else None
            active_list_parent = None
            active_list_next_number = None
            active_list_last_pos = -1
        marker_like = LIST_MARKER_RE.match(node.text)
        if list_number is not None:
            if (
                active_list_parent is not None
                and active_list_next_number == list_number
                and 0 <= effective_pos - active_list_last_pos <= 18
            ):
                parent_id = active_list_parent
            else:
                parent_id = visible_list_proxy_parent(
                    recent_list_intro_by_scope,
                    scope_parent=parent_id,
                    current_effective_pos=effective_pos,
                ) or parent_id
                active_list_parent = parent_id
            active_list_next_number = list_number + 1
            active_list_last_pos = effective_pos
        elif marker_like:
            if active_list_parent is not None and 0 <= effective_pos - active_list_last_pos <= 18:
                parent_id = active_list_parent
            else:
                parent_id = visible_list_proxy_parent(
                    recent_list_intro_by_scope,
                    scope_parent=parent_id,
                    current_effective_pos=effective_pos,
                ) or parent_id
                active_list_parent = parent_id
            active_list_next_number = None
            active_list_last_pos = effective_pos
        elif text_can_anchor_visible_list(node.text):
            recent_list_intro_by_scope[parent_id] = (node_id, effective_pos)
            active_list_parent = None
            active_list_next_number = None
            active_list_last_pos = -1
        elif str(node.text or "").strip():
            recent_list_intro_by_scope.pop(parent_id, None)
            active_list_parent = None
            active_list_next_number = None
            active_list_last_pos = -1
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


def heading_numbering_parent_is_compatible(parent_path: tuple[str, ...] | None, child_path: tuple[str, ...]) -> bool:
    if not parent_path:
        return True
    if len(parent_path) >= len(child_path):
        return False
    return child_path[: len(parent_path)] == parent_path


def heading_scope_must_close_before_child(*, parent_text: str, child_text: str) -> bool:
    """Close terminal bibliography scopes before later appendix/body headings.

    Visual heading stacks are intentionally layout-based, so they can see a
    bibliography title before an appendix title even when the TeX AST is noisy.
    A ``References`` / ``Bibliography`` heading may parent reference-list
    entries, but it must never become the parent of later appendix or normal
    section headings.
    """

    parent_kind = normalized_heading_keyword(parent_text)
    child_kind = normalized_heading_keyword(child_text)
    if parent_kind in {"acknowledgements", "acknowledgments"}:
        if child_kind in {"references", "bibliography", "appendix"}:
            return True
        return title_numbering_path(child_text) is not None
    if parent_kind not in {"references", "bibliography"}:
        return False
    if child_kind in {"references", "bibliography"}:
        return False
    if title_numbering_path(child_text) is not None:
        return True
    return child_kind in {"appendix", "acknowledgements", "acknowledgments"} or bool(
        str(child_text or "").strip()
    )


def reference_scope_must_close_before_item(*, parent_text: str, item: dict[str, Any]) -> bool:
    parent_kind = normalized_heading_keyword(parent_text)
    if parent_kind not in {"references", "bibliography"}:
        return False
    return canonical_pdf_merge_type(item) != "reference" and not item_looks_like_reference_entry(item)


def visual_parent_pair_is_quality_gate_required(parent: PdfAlignmentNode, child: PdfAlignmentNode) -> bool:
    """Return whether a visual parent edge is mandatory for data quality.

    The quality gate is meant to catch missing train-critical hierarchy edges:
    headings/run-in anchors to body text, equations, lists, and local
    reference-list content.  Floats and complex reference columns can be
    physically interleaved in two-column layouts, so their anchoring is handled
    by dedicated float/reference logic and should not fail a whole document
    just because a section-scope candidate edge was intentionally not sampled.
    """

    child_kind = canonical_pdf_merge_type(child.item)
    if child_kind in {"figure", "image", "chart", "table", "algorithm", "code"}:
        return False
    parent_kind = normalized_heading_keyword(parent.text)
    if parent_kind in {"references", "bibliography"} and item_looks_like_reference_entry(child.item):
        return child_kind == "reference"
    if child_kind == "reference":
        return parent_kind in {"references", "bibliography"}
    if item_looks_like_author_biography(child.item):
        return False
    return True


def item_looks_like_author_biography(item: dict[str, Any]) -> bool:
    text = stringify_text_payload(
        item.get("text_for_embedding") or item.get("text") or item.get("merged_text") or item.get("text_preview")
    )
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


def item_looks_like_reference_entry(item: dict[str, Any]) -> bool:
    if str(item.get("list_type") or "").lower() == "reference_list":
        return True
    text = stringify_text_payload(
        item.get("text_for_embedding") or item.get("text") or item.get("merged_text") or item.get("text_preview")
    )
    compact = " ".join(str(text or "").split())
    if len(compact) < 40:
        return False
    lower = compact.casefold()
    if len(compact) >= 100 and compact.count(",") >= 5:
        return True
    signals = 0
    if re.search(r"\b(?:19|20)\d{2}[a-z]?\b", compact):
        signals += 1
    if any(token in lower for token in ("http://", "https://", "doi", "arxiv", "accessed", "conference", "journal")):
        signals += 1
    if compact.count(",") >= 2 or " et al" in lower:
        signals += 1
    if re.search(r"\bpp\.\s*\d", lower):
        signals += 1
    return signals >= 2


def normalized_heading_keyword(text: str) -> str:
    raw = str(text or "").casefold()
    tokens = re.findall(r"[a-z]+|\d+", raw)
    while tokens and (tokens[0].isdigit() or re.fullmatch(r"[ivxlcdm]+", tokens[0])):
        tokens.pop(0)
    if tokens and tokens[0] in {"references", "bibliography", "acknowledgements", "acknowledgments"}:
        return tokens[0]
    if tokens and tokens[0] == "appendix":
        return "appendix"
    return "".join(tokens)


def ordered_list_marker_number(text: str) -> int | None:
    match = ORDERED_LIST_MARKER_RE.match(str(text or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def text_can_anchor_visible_list(text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    if not stripped or LIST_MARKER_RE.match(stripped):
        return False
    if len(clean_text(stripped)) < 8:
        return False
    return bool(VISIBLE_LIST_INTRO_RE.search(stripped))


def is_algorithm_io_label(text: str) -> bool:
    return bool(ALGORITHM_IO_LABEL_RE.match(str(text or "")))


def is_formula_heavy_scope_heading_text(text: str) -> bool:
    value = " ".join(str(text or "").split())
    if not value:
        return False
    math_tokens = sum(value.count(token) for token in ("\\", "_", "^", "{", "}", "=", "\\tag", "\\mathbf", "\\colon"))
    word_tokens = len(re.findall(r"[A-Za-z]{2,}", value))
    return math_tokens >= 5 and math_tokens >= word_tokens


def visible_list_proxy_parent(
    recent_list_intro_by_scope: dict[int | None, tuple[int, int]],
    *,
    scope_parent: int | None,
    current_effective_pos: int,
) -> int | None:
    proxy = recent_list_intro_by_scope.get(scope_parent)
    if proxy is None:
        return None
    proxy_node, proxy_pos = proxy
    if 0 <= current_effective_pos - proxy_pos <= 4:
        return proxy_node
    return None


def alpha_or_roman_heading_level(text: str) -> int | None:
    match = ALPHA_OR_ROMAN_HEADING_RE.match(" ".join(str(text or "").split()))
    if not match:
        return None
    token = match.group(1).upper()
    tail = match.group(2)
    if len(token) > 1 or (token in {"I", "V", "X", "L", "C", "D", "M"} and looks_like_all_caps_heading(tail)):
        return 1
    return 2


def looks_like_all_caps_heading(text: str) -> bool:
    letters = [char for char in str(text or "") if char.isalpha()]
    if len(letters) < 4:
        return False
    uppercase = sum(1 for char in letters if char.isupper())
    return uppercase / max(1, len(letters)) >= 0.75


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
    if is_algorithm_io_label(text):
        return False
    if node.item.get("run_in_heading"):
        return True
    layer = layout_layer_name(node.item)
    raw_type = canonical_pdf_type(node.item)
    if layer == "noise_layer":
        return False
    if layer == "metadata_layer" and not metadata_layer_heading_override(node.item):
        return False
    if raw_type in HEADING_TYPES:
        return True
    if is_formula_heavy_scope_heading_text(text):
        return False
    if raw_type in NON_HEADING_PDF_TYPES or str(node.item.get("list_type") or "").lower() == "reference_list":
        return False
    if LIST_MARKER_RE.match(text):
        return False
    if title_numbering_level(text) is not None and looks_like_standalone_heading(text):
        return True
    font_size = pdf_font_size(node.item)
    if (
        body_font_size > 0
        and font_size >= body_font_size * max(1.18, config.visual_heading_font_scale)
        and looks_like_standalone_heading(text)
        and looks_like_title_case_heading(text)
    ):
        return True
    if (
        body_font_size > 0
        and font_size >= body_font_size * max(1.12, config.visual_bold_heading_font_scale)
        and pdf_bold_ratio(node.item) >= 0.45
        and looks_like_standalone_heading(text)
        and looks_like_title_case_heading(text)
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
    if (
        (page_idx in {None, 0})
        and bbox[1] < 260
        and canonical_pdf_type(first.item) in HEADING_TYPES
        and title_numbering_path(text) is None
        and any(
            re.sub(r"[^a-z]+", "", candidate.text.casefold()) == "abstract"
            for candidate in nodes[first.node_index + 1 : first.node_index + 8]
        )
    ):
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
        if node.item.get("run_in_heading"):
            value = node.item.get("run_in_heading_level")
            if isinstance(value, (int, float)):
                levels[node_id] = min(max(1, int(value)), 3)
                continue
        numbered_level = title_numbering_level(node.text)
        if numbered_level is not None:
            levels[node_id] = min(max(1, numbered_level), 3)
            continue
        alpha_level = alpha_or_roman_heading_level(node.text)
        if alpha_level is not None:
            levels[node_id] = alpha_level
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


def looks_like_title_case_heading(text: str) -> bool:
    value = " ".join(str(text or "").split())
    if title_numbering_path(value) is not None:
        return True
    tokens = [token.strip("()[]{}:;,.") for token in re.split(r"\s+", value) if token.strip("()[]{}:;,.")]
    meaningful = [token for token in tokens if any(char.isalpha() for char in token)]
    if not meaningful:
        return False
    heading_like = 0
    for token in meaningful:
        letters = [char for char in token if char.isalpha()]
        if not letters:
            continue
        if letters[0].isupper() or sum(char.isupper() for char in letters) / max(1, len(letters)) >= 0.6:
            heading_like += 1
    return heading_like / max(1, len(meaningful)) >= 0.6


def canonical_pdf_type(item: dict[str, Any]) -> str:
    return str(item.get("canonical_type") or item.get("type") or item.get("raw_type") or "").strip().lower()


def canonical_pdf_merge_type(item: dict[str, Any]) -> str:
    """Collapse MinerU/PyMuPDF block names into relation-label merge families."""

    if str(item.get("list_type") or "").lower() == "reference_list":
        return "reference"
    raw = canonical_pdf_type(item)
    if raw in {"toc", "toc_title", "toc_entry", "index", "table_of_contents"}:
        return "toc"
    if raw in {"paragraph", "text", "paragraph_text", "body", "list", "item"}:
        return "text"
    if raw in {"title", "section", "subsection", "subsubsection", "heading"}:
        return "title"
    if raw in {"equation", "equation_interline", "interline_equation", "display_formula", "formula"}:
        return "equation"
    if raw in {"inline_math", "inline_formula", "math_inline"}:
        return "inline_math"
    if raw in {"reference", "references", "bibliography"}:
        return "reference"
    if raw in {"table", "figure", "image", "chart", "algorithm", "code"}:
        return raw
    return "text"


def strict_pdf_merge_type(item: dict[str, Any]) -> str:
    """Return a visual merge family without folding list items into text.

    ``canonical_pdf_merge_type`` is intentionally broad for alignment and
    fallback matching.  MERGE labels need a stricter view: a MinerU/PyMuPDF
    ``list`` block is a structural/list item boundary, not just ordinary body
    text, and it must not be stitched into a neighboring paragraph solely
    because the TeX parser mapped both fragments to the same node.
    """

    if str(item.get("list_type") or "").lower() == "reference_list":
        return "reference"
    raw = canonical_pdf_type(item)
    if raw in {"toc", "toc_title", "toc_entry", "index", "table_of_contents"}:
        return "toc"
    if raw in {"list", "item"}:
        return "list"
    if raw in {"paragraph", "text", "paragraph_text", "body"}:
        return "text"
    if raw in {"title", "section", "subsection", "subsubsection", "heading"}:
        return "title"
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

    left_type = strict_pdf_merge_type(left.item)
    right_type = strict_pdf_merge_type(right.item)
    if LIST_MARKER_RE.match(right.text):
        return False
    if not same_layout_scope_can_merge(left.item, right.item):
        return False
    return left_type == right_type and left_type in MERGE_COMPATIBLE_PDF_TYPES


def is_run_in_heading_like(node: PdfAlignmentNode) -> bool:
    """Detect visually independent run-in headings inside coarse TeX nodes."""

    text = str(node.text or "").strip()
    if not text:
        return False
    if LIST_MARKER_RE.match(text):
        return True
    if not RUN_IN_HEADING_RE.match(text):
        return False
    # Plain subsection titles are already blocked by the title family.  This
    # probe catches paragraph-internal heads such as "Put operation." and
    # "Space complexity." that appear as separate visual blocks.
    return len(clean_text(text)) >= 3


def ends_with_terminal_punctuation(text: str) -> bool:
    return bool(TERMINAL_PUNCTUATION_RE.search(str(text or "").strip()))


def ends_with_hyphen(text: str) -> bool:
    return bool(HYPHEN_END_RE.search(str(text or "").strip()))


def starts_with_uppercase_text(text: str) -> bool:
    return bool(UPPERCASE_START_RE.match(str(text or "")))


def edge_attr_fields_from_graph(graph: Any) -> dict[str, int]:
    schema = getattr(graph, "edge_attr_schema", None)
    if isinstance(schema, dict):
        fields = schema.get("fields")
        if isinstance(fields, list):
            return {str(name): index for index, name in enumerate(fields)}
    return {}


def same_layout_scope_can_merge(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Keep MERGE labels inside the same page-object layer and local band."""

    left_layer = layout_layer_name(left)
    right_layer = layout_layer_name(right)
    if left_layer == "noise_layer" or right_layer == "noise_layer":
        return False
    if left_layer != right_layer:
        return False
    if left_layer not in {"main_text_flow", "math_layer"} and canonical_pdf_merge_type(left) != "reference":
        return False
    left_band = layout_band_id(left)
    right_band = layout_band_id(right)
    if left_band is not None and right_band is not None and left_band != right_band:
        return False
    return True


def relation_layers_are_incompatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Block semantic labels across metadata/noise and main document flow."""

    left_layer = layout_layer_name(left)
    right_layer = layout_layer_name(right)
    if "noise_layer" in {left_layer, right_layer}:
        return True
    if "metadata_layer" in {left_layer, right_layer} and left_layer != right_layer:
        if metadata_layer_heading_override(left) or metadata_layer_heading_override(right):
            return False
        return True
    return False


def metadata_layer_heading_override(item: dict[str, Any]) -> bool:
    raw = canonical_pdf_type(item)
    if raw not in HEADING_TYPES:
        return False
    text = stringify_text_payload(
        item.get("text_for_embedding") or item.get("text") or item.get("merged_text") or item.get("text_preview")
    ).strip()
    if title_numbering_path(text) is not None:
        return True
    normalized = re.sub(r"[^a-z]+", "", text.casefold())
    return normalized in {"abstract", "references", "bibliography", "appendix", "acknowledgements", "acknowledgments"}


def layout_layer_name(item: dict[str, Any]) -> str:
    return str(item.get("layout_layer") or "main_text_flow")


def layout_band_id(item: dict[str, Any]) -> int | None:
    value = item.get("layout_band_global_id")
    if value is None:
        value = item.get("layout_band_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def last_bbox(value: Any) -> tuple[float, float, float, float] | None:
    chunks = bbox_chunks(value)
    return chunks[-1] if chunks else None


def bbox_y_overlap_ratio(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    intersection = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    min_height = max(1.0, min(left[3] - left[1], right[3] - right[1]))
    return intersection / min_height


def bbox_x_gap(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    return max(0.0, max(left[0], right[0]) - min(left[2], right[2]))


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
            if name == "thebibliography":
                self.flush_paragraphs(paragraph_buffer, parent_id)
                self.walk_children(
                    getattr(child, "contents", []) or [],
                    parent_id=self.current_parent(parent_id),
                    parent_env=name,
                )
                continue
            if name == "bibitem":
                self.flush_paragraphs(paragraph_buffer, parent_id, node_type=STANDARD_REFERENCE_NODE)
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
            if parent_env in CAPTION_PARENT_ENVS:
                if hasattr(child, "contents"):
                    self.walk_children(
                        getattr(child, "contents", []) or [],
                        parent_id=self.current_parent(parent_id),
                        parent_env=parent_env,
                    )
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
        final_node_type = STANDARD_REFERENCE_NODE if parent_env == "thebibliography" else STANDARD_PARAGRAPH_NODE
        self.flush_paragraphs(paragraph_buffer, parent_id, node_type=final_node_type)

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
        if is_layout_artifact_node(text, clean, source_name=source_name):
            return None
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

    def flush_paragraphs(
        self,
        paragraph_buffer: list[str],
        parent_id: str,
        *,
        node_type: str = STANDARD_PARAGRAPH_NODE,
    ) -> None:
        if not paragraph_buffer:
            return
        raw_text = "".join(paragraph_buffer)
        paragraph_buffer.clear()
        for paragraph in re.split(r"(?:\r?\n\s*){2,}", raw_text):
            if paragraph.strip():
                self.add_node(node_type, paragraph, self.current_parent(parent_id), source_name="text")

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

    value = strip_visual_artifacts(str(text or ""))
    value = expose_math_payload(value)
    value = value.lower()
    value = re.sub(r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])?", " ", value)
    value = re.sub(r"\\.", " ", value)
    value = re.sub(rf"[^0-9a-z\u4e00-\u9fff]+", "", value)
    return value


def clean_equation_text(text: Any) -> str:
    """Normalize display equations while keeping command-only formulas alignable."""

    value = expose_math_payload(strip_visual_artifacts(str(text or "")))
    value = re.sub(r"\\([a-zA-Z]+)\*?", r" \1 ", value)
    value = re.sub(r"\\.", " ", value)
    value = value.lower()
    value = re.sub(rf"[^0-9a-z\u4e00-\u9fff]+", "", value)
    return value


def strip_visual_artifacts(value: str) -> str:
    """Remove image paths and layout option fragments before text matching."""

    value = VISUAL_FILE_RE.sub(" ", value)
    value = VISUAL_OPTION_RE.sub(" ", value)
    value = VISUAL_LENGTH_RE.sub(" ", value)
    return value


def is_layout_artifact_node(text: Any, clean: str, *, source_name: str | None = None) -> bool:
    """Reject TeX nodes that are only layout parameters, colors, or placement hints."""

    if source_name in SKIP_TEX_NODE_NAMES:
        return True
    raw = str(text or "").strip()
    compact_raw = re.sub(r"\s+", "", raw).lower()
    compact_clean = str(clean or "").lower()
    if not compact_clean:
        return False
    if compact_clean in LAYOUT_ONLY_CLEAN_VALUES or compact_clean in LAYOUT_COLOR_CLEAN_VALUES:
        return True
    if LAYOUT_DIMENSION_RE.fullmatch(compact_clean):
        return True
    if re.fullmatch(r"[tblrc]?\d+(?:pt|em|ex|cm|mm|in|pc|px)?", compact_clean):
        return True
    if re.fullmatch(r"[tblrc]?\d+", compact_clean) and any(
        marker in compact_raw
        for marker in ("textwidth", "linewidth", "columnwidth", "paperwidth", "pt", "em", "cm", "mm", "in")
    ):
        return True
    if re.fullmatch(r"[htbp]+", compact_clean):
        return True
    if compact_raw in {"\\maketitle", "\\centering", "\\noindent", "\\clearpage", "\\newpage"}:
        return True
    return False


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
    span_text = style_spans_text(item.get("style_spans"))
    if span_text.strip():
        return span_text
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
