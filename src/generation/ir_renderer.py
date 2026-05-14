"""Renderer entrypoint for the decoupled backend IR.

This is the first original-like generator surface.  It consumes the stable
interfaces instead of TreeDecoder internals:

DocumentIR + RenderTreeIR + StyleProfile (+ CitationResolution) -> LaTeX.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.generation.citations import CitationResolution, author_year_lookup, replace_citation_markers, strip_reference_label
from src.generation.font_resolver import resolve_pdf_font
from src.generation.front_matter import render_author_block_original_like, render_document_title_original_like
from src.generation.latex_renderer import (
    clean_float_caption_text,
    escape_latex,
    render_algorithm_block,
    render_equation,
    render_figure_block,
    render_figure_minipage_group,
    render_inline_math,
    render_table_placeholder,
    render_text_with_inline_latex,
    safe_verbatim_text,
    strip_list_marker,
)
from src.generation.source_float_layout import SourceFloatLayout, SourceTableLayout
from src.ir import BBox, BlockType, DocumentIR, DocumentNode, RenderRole, RenderTreeIR, RenderTreeNode, StyleProfile, StyleSpan
from src.perception.title_features import strip_title_numbering


CITE_COMMAND_RE = re.compile(r"\\(?:cite|citep|citet|citealp|citeauthor|citeyear|ref|autoref|cref)\*?(?:\[[^\]]*\])?\{[^{}]+\}")
LIST_MARKER_RE = re.compile(r"^\s*(?:[\u2022\u25E6\u25CB\u25AA\-\*]|\d+\.|[a-zA-Z]\.)\s+")
ORDERED_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\.|[a-zA-Z]\.)\s+")
NUMERIC_ID_RE = re.compile(r"\d+")
APPENDIX_TITLE_PREFIX_RE = re.compile(r"^\s*(?:appendix\s+)?[A-Z](?:\.\d+)*\.?\s+", re.IGNORECASE)
MERGE_TRAILING_HYPHEN_RE = re.compile(r"[-‐‑‒–—]\s*$")
REQUIRED_RENDER_PACKAGES = [
    "amsmath",
    "amssymb",
    "graphicx",
    "float",
    "booktabs",
    "hyperref",
    "geometry",
    "enumitem",
    "multicol",
    "caption",
    "titlesec",
    "algorithm",
    "algpseudocode",
]


@dataclass(frozen=True)
class IRLatexRenderConfig:
    title: str | None = None
    include_maketitle: bool = True
    front_matter_mode: str = "maketitle"
    table_asset_output_dir: Path | None = None
    figure_asset_output_dir: Path | None = None
    table_asset_latex_prefix: str = "assets"
    figure_asset_latex_prefix: str = "assets"
    enable_fontspec: bool = False
    render_header_footer: bool = True
    preserve_span_font_family: bool = True
    preserve_span_font_size: bool = True
    preserve_span_scripts: bool = True
    span_font_size_delta_threshold: float = 0.08
    script_font_size_max_ratio: float = 0.85
    script_vertical_offset_ratio: float = 0.12


class OriginalLikeIRLatexRenderer:
    """Render the stable generation IR into a compilable LaTeX document."""

    def __init__(self, config: IRLatexRenderConfig | None = None) -> None:
        self.config = config or IRLatexRenderConfig()
        self._active_style: StyleProfile | None = None
        self._mixed_column_stack = 0
        self._active_notes: _NoteContext | None = None
        self._active_source_float_layout: SourceFloatLayout | None = None
        self._active_cross_refs: _CrossReferenceRegistry | None = None

    def render(
        self,
        document: DocumentIR,
        tree: RenderTreeIR,
        style: StyleProfile,
        citations: CitationResolution | None = None,
        source_float_layout: SourceFloatLayout | None = None,
    ) -> str:
        previous_style = self._active_style
        previous_notes = self._active_notes
        previous_source_float_layout = self._active_source_float_layout
        previous_cross_refs = self._active_cross_refs
        self._active_style = style
        self._active_notes = _NoteContext.from_document(document)
        self._active_source_float_layout = source_float_layout
        self._active_cross_refs = _CrossReferenceRegistry.from_document(document)
        try:
            return self._render_with_active_style(document, tree, style, citations)
        finally:
            self._active_style = previous_style
            self._active_notes = previous_notes
            self._active_source_float_layout = previous_source_float_layout
            self._active_cross_refs = previous_cross_refs

    def _render_with_active_style(
        self,
        document: DocumentIR,
        tree: RenderTreeIR,
        style: StyleProfile,
        citations: CitationResolution | None = None,
    ) -> str:
        tree = self._tree_with_missing_referenced_float_nodes(document, tree)
        document_nodes = {node.node_id: node for node in document.nodes}
        render_nodes = {node.render_id: node for node in tree.nodes}
        root = render_nodes[tree.root_id]

        lines = self._render_preamble(style, citations)
        title = self.config.title or self._infer_title(document)
        use_maketitle = self._use_maketitle()
        if title and use_maketitle:
            lines.extend([rf"\title{{{escape_latex(title)}}}", r"\date{}", ""])
        lines.append(r"\begin{document}")
        if title and use_maketitle:
            lines.append(r"\maketitle")
            lines.append("")
        elif title and self._use_original_like_front_matter() and not _tree_has_role(tree, RenderRole.DOCUMENT_TITLE):
            lines.append(render_document_title_original_like(title, [], style))
            lines.append("")
        rendered_body = self._render_children(root, render_nodes, document_nodes, citations, depth=0)
        if rendered_body:
            lines.append(rendered_body)
            lines.append("")
        orphan_notes = self._render_unanchored_notes()
        if orphan_notes:
            lines.append(orphan_notes)
            lines.append("")
        lines.append(r"\end{document}")
        return "\n".join(lines).rstrip() + "\n"

    def _tree_with_missing_referenced_float_nodes(self, document: DocumentIR, tree: RenderTreeIR) -> RenderTreeIR:
        registry = self._active_cross_refs
        if registry is None:
            return tree
        referenced_labels = registry.referenced_labels(document)
        if not referenced_labels:
            return tree

        used_source_ids = {source_id for node in tree.nodes for source_id in node.source_node_ids}
        document_nodes = {node.node_id: node for node in document.nodes}
        used_figure_groups = {
            group_id
            for source_id in used_source_ids
            for source in [document_nodes.get(source_id)]
            if source is not None and source.node_type == BlockType.FIGURE
            for group_id in [_figure_group_id(source)]
            if group_id
        }
        used_table_groups = {
            str(source.metadata.get("table_group_id"))
            for source_id in used_source_ids
            for source in [document_nodes.get(source_id)]
            if source is not None
            and source.node_type == BlockType.TABLE
            and source.metadata.get("table_group_id") is not None
        }

        additions: list[RenderTreeNode] = []
        for source in sorted(document.nodes, key=lambda item: item.reading_index):
            kind = _document_node_cross_ref_kind(source)
            if kind is None:
                continue
            label = registry.label_for_node(source.node_id, kind=kind)
            if not label or label not in referenced_labels or source.node_id in used_source_ids:
                continue
            if source.node_type == BlockType.FIGURE:
                group_id = _figure_group_id(source)
                if group_id and group_id in used_figure_groups:
                    continue
                if _is_nonprimary_figure_group_member(source):
                    continue
                if group_id:
                    used_figure_groups.add(group_id)
            if source.node_type == BlockType.TABLE:
                group_id = source.metadata.get("table_group_id")
                if group_id is not None and str(group_id) in used_table_groups:
                    continue
                if source.metadata.get("table_group_primary") is False:
                    continue
                if group_id is not None:
                    used_table_groups.add(str(group_id))
            role = _render_role_for_cross_ref_kind(kind)
            if role is None:
                continue
            additions.append(
                RenderTreeNode(
                    render_id=f"full_v7_ref_float_{_safe_render_id(source.node_id)}",
                    role=role,
                    source_node_ids=[source.node_id],
                    attributes={"injected_reason": "referenced_float_missing_from_tree"},
                )
            )
            used_source_ids.add(source.node_id)
        if not additions:
            return tree

        addition_ids = [node.render_id for node in additions]
        updated_nodes: list[RenderTreeNode] = []
        for node in tree.nodes:
            if node.render_id != tree.root_id:
                updated_nodes.append(node)
                continue
            updated_nodes.append(
                RenderTreeNode(
                    render_id=node.render_id,
                    role=node.role,
                    source_node_ids=node.source_node_ids,
                    text=node.text,
                    latex=node.latex,
                    children=list(dict.fromkeys([*node.children, *addition_ids])),
                    attributes=node.attributes,
                )
            )
        return RenderTreeIR(
            doc_id=tree.doc_id,
            root_id=tree.root_id,
            nodes=[*updated_nodes, *additions],
            document_ir_path=tree.document_ir_path,
            predicted_relations_path=tree.predicted_relations_path,
            style_profile_path=tree.style_profile_path,
            metadata={**tree.metadata, "referenced_float_fallback_count": len(additions)},
        )

    def _render_preamble(self, style: StyleProfile, citations: CitationResolution | None = None) -> list[str]:
        options = f"[{','.join(style.documentclass_options)}]" if style.documentclass_options else ""
        lines = [rf"\documentclass{options}{{{style.documentclass}}}"]
        packages = [*style.packages, *REQUIRED_RENDER_PACKAGES]
        if citations is not None and citations.citation_style == "numeric":
            packages.insert(0, "cite")
        elif citations is not None and citations.citation_style == "author_year":
            packages.insert(0, "natbib")
        if _style_column_mode(style) == "mixed":
            packages.append("multicol")
        if self.config.enable_fontspec:
            packages.append("fontspec")
        if self.config.render_header_footer and _header_footer_profile_enabled(style):
            packages.append("fancyhdr")
        for package in _dedupe_preserve_order(packages):
            lines.append(_render_package(package))
        if citations is not None and citations.citation_style == "author_year":
            lines.append(r"\setcitestyle{round}")
        for macro in style.macros:
            lines.append(macro)
        lines.extend(self._render_original_like_layout_commands(style))
        lines.append("")
        return lines

    def _render_original_like_layout_commands(self, style: StyleProfile) -> list[str]:
        options = style.renderer_options or {}
        lines: list[str] = []
        if self.config.enable_fontspec:
            lines.extend(_fontspec_commands(options.get("font_setup")))
        geometry_options = options.get("geometry_options")
        if isinstance(geometry_options, dict) and geometry_options:
            geometry_parts = [f"{key}={value}" for key, value in geometry_options.items() if value]
            if geometry_parts:
                lines.append(rf"\geometry{{{','.join(geometry_parts)}}}")
        if self.config.render_header_footer:
            lines.extend(_header_footer_commands(options.get("header_footer")))
        column_gap_pt = _float_or_none(options.get("column_gap_pt"))
        if column_gap_pt is not None and column_gap_pt > 0:
            lines.append(rf"\setlength{{\columnsep}}{{{_pt(min(column_gap_pt, 48.0))}}}")
        body_font_size = _float_or_none(options.get("body_font_size"))
        if body_font_size:
            lines.append(rf"\AtBeginDocument{{\fontsize{{{body_font_size:.2f}pt}}{{{(body_font_size * 1.2):.2f}pt}}\selectfont}}")
        paragraph_indent = _float_or_none(options.get("paragraph_indent"))
        if paragraph_indent is not None:
            lines.append(rf"\setlength{{\parindent}}{{{_pt(paragraph_indent)}}}")
        paragraph_spacing = _float_or_none(options.get("paragraph_spacing"))
        if paragraph_spacing is not None:
            lines.append(rf"\setlength{{\parskip}}{{{_pt(min(paragraph_spacing, 18.0))}}}")
        display_spacing = options.get("display_math_spacing")
        if isinstance(display_spacing, dict):
            above = _float_or_none(display_spacing.get("above"))
            below = _float_or_none(display_spacing.get("below"))
            if above is not None:
                lines.append(rf"\setlength{{\abovedisplayskip}}{{{_pt(min(above, 24.0))}}}")
            if below is not None:
                lines.append(rf"\setlength{{\belowdisplayskip}}{{{_pt(min(below, 24.0))}}}")
        list_spacing = options.get("list_spacing")
        if isinstance(list_spacing, dict):
            itemsep = _float_or_none(list_spacing.get("itemsep"))
            topsep = _float_or_none(list_spacing.get("topsep"))
            settings = []
            if itemsep is not None:
                settings.append(f"itemsep={_pt(min(itemsep, 18.0))}")
            if topsep is not None:
                settings.append(f"topsep={_pt(min(topsep, 18.0))}")
            if settings:
                lines.append(rf"\setlist{{{','.join(settings)}}}")
        heading_spacing = _heading_spacing_commands(style.role_styles)
        lines.extend(heading_spacing)
        return lines

    def _render_tree_node(
        self,
        node: RenderTreeNode,
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
        citations: CitationResolution | None,
        *,
        depth: int,
    ) -> str:
        role = node.role
        source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
        text = node.latex or node.text or self._source_text(source_nodes, citations)
        text = _clean_render_text(text)

        if role in {RenderRole.ROOT}:
            return self._render_children(node, render_nodes, document_nodes, citations, depth=depth)
        if role in {RenderRole.DOCUMENT_TITLE}:
            if self._use_original_like_front_matter():
                return render_document_title_original_like(text, source_nodes, self._active_style)
            if self.config.include_maketitle:
                return ""
            return render_text_with_citations(text)
        if role in {RenderRole.AUTHOR_BLOCK}:
            if self._use_original_like_front_matter():
                return render_author_block_original_like(text, source_nodes, self._active_style)
            return render_text_with_citations(text)
        if role in {RenderRole.ABSTRACT}:
            return self._render_abstract(node, render_nodes, document_nodes, citations, text, depth=depth)
        if role in {RenderRole.SECTION, RenderRole.SUBSECTION, RenderRole.SUBSUBSECTION}:
            command = {
                RenderRole.SECTION: "section",
                RenderRole.SUBSECTION: "subsection",
                RenderRole.SUBSUBSECTION: "subsubsection",
            }[role]
            heading_text = _clean_appendix_heading_text(text) if node.attributes.get("appendix_heading") else _clean_heading_text(text)
            heading = rf"\{command}{{{render_text_with_citations(heading_text)}}}" if heading_text else ""
            children = self._render_children(node, render_nodes, document_nodes, citations, depth=depth + 1)
            return "\n\n".join(part for part in [heading, children] if part)
        if role in {RenderRole.DISPLAY_EQUATION}:
            body = render_equation(text, label=self._cross_ref_label_for_render_node(node, document_nodes, "equation"))
            children = self._render_children(node, render_nodes, document_nodes, citations, depth=depth + 1)
            return "\n\n".join(part for part in [body, children] if part)
        if role in {RenderRole.INLINE_MATH}:
            return render_inline_math(text)
        if role in {RenderRole.TABLE}:
            return self._render_table(source_nodes, text)
        if role in {RenderRole.FIGURE}:
            return self._render_figure(source_nodes, text, citations, document_nodes=document_nodes)
        if role in {RenderRole.ALGORITHM}:
            return render_algorithm_block(text, label=self._cross_ref_label_for_render_node(node, document_nodes, "algorithm"))
        if role in {RenderRole.CODE}:
            return "\\begin{verbatim}\n" + safe_verbatim_text(text.strip()) + "\n\\end{verbatim}" if text else ""
        if role in {RenderRole.FOOTNOTE}:
            if self._active_notes is not None and source_nodes:
                return ""
            return self._render_standalone_note("footnote", text)
        if role in {RenderRole.MARGIN_NOTE}:
            if self._active_notes is not None and source_nodes:
                return ""
            return self._render_standalone_note("margin_note", text)
        if role in {RenderRole.CAPTION}:
            return render_text_with_citations(text)
        if role in {RenderRole.TOC_PLACEHOLDER}:
            return r"\tableofcontents"
        if role in {RenderRole.REFERENCES}:
            return self._render_bibliography_with_tail(citations, source_nodes, node, render_nodes, document_nodes, depth=depth)
        if role in {RenderRole.LIST}:
            return self._render_list(node, render_nodes, document_nodes, citations, ordered=False, depth=depth)
        if role in {RenderRole.LIST_ITEM}:
            body = self._render_source_nodes(source_nodes, citations, strip_leading_list_marker=True) or render_text_with_citations(strip_list_marker(text))
            children = self._render_children(node, render_nodes, document_nodes, citations, depth=depth + 1)
            return "\n".join(part for part in [body, children] if part)
        if role in {RenderRole.RAW_LATEX}:
            return text

        body = self._render_source_nodes(source_nodes, citations) if source_nodes else self._render_body_text(text)
        children = self._render_children(node, render_nodes, document_nodes, citations, depth=depth + 1)
        return "\n\n".join(part for part in [body, children] if part)

    def _render_abstract(
        self,
        node: RenderTreeNode,
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
        citations: CitationResolution | None,
        text: str,
        *,
        depth: int,
    ) -> str:
        child_ids = _sorted_child_ids(node.children, render_nodes, document_nodes)
        abstract_label = "" if _is_abstract_label(text) else render_text_with_citations(text)
        column_mode = _abstract_column_mode(node, child_ids, render_nodes, document_nodes)
        if column_mode == "double":
            if self._mixed_column_stack > 0:
                children = self._render_children_standard(
                    child_ids,
                    render_nodes,
                    document_nodes,
                    citations,
                    depth=depth + 1,
                )
            else:
                self._mixed_column_stack += 1
                try:
                    children = self._render_children_standard(
                        child_ids,
                        render_nodes,
                        document_nodes,
                        citations,
                        depth=depth + 1,
                    )
                finally:
                    self._mixed_column_stack -= 1
            body = "\n\n".join(part for part in [abstract_label, children] if part)
            if body and self._mixed_column_stack <= 0:
                body = "\\begin{multicols}{2}\n" + body + "\n\\end{multicols}"
        else:
            children = self._render_children_standard(
                child_ids,
                render_nodes,
                document_nodes,
                citations,
                depth=depth + 1,
            )
            body = "\n\n".join(part for part in [abstract_label, children] if part)
        return "\\begin{abstract}\n" + body + "\n\\end{abstract}" if body else ""

    def _render_children(
        self,
        node: RenderTreeNode,
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
        citations: CitationResolution | None,
        *,
        depth: int,
    ) -> str:
        child_ids = _sorted_child_ids(node.children, render_nodes, document_nodes)
        child_ids = _defer_sentence_interrupting_floats(child_ids, render_nodes, document_nodes)
        if self._should_render_mixed_columns(node, child_ids, render_nodes, document_nodes):
            return self._render_children_with_mixed_columns(
                child_ids,
                render_nodes,
                document_nodes,
                citations,
                depth=depth,
            )
        return self._render_children_standard(
            child_ids,
            render_nodes,
            document_nodes,
            citations,
            depth=depth,
        )

    def _render_children_standard(
        self,
        child_ids: list[str],
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
        citations: CitationResolution | None,
        *,
        depth: int,
    ) -> str:
        parts: list[str] = []
        index = 0
        seen_references = False
        appendix_started = False
        while index < len(child_ids):
            child = render_nodes[child_ids[index]]
            if self._is_author_render_node(child, document_nodes):
                run: list[RenderTreeNode] = []
                while index < len(child_ids):
                    candidate = render_nodes[child_ids[index]]
                    if not self._is_author_render_node(candidate, document_nodes):
                        break
                    run.append(candidate)
                    index += 1
                parts.append(self._render_author_block_run(run, document_nodes))
                continue
            if self._is_reference_render_node(child, document_nodes):
                run: list[RenderTreeNode] = []
                while index < len(child_ids):
                    candidate = render_nodes[child_ids[index]]
                    if not self._is_reference_render_node(candidate, document_nodes):
                        break
                    run.append(candidate)
                    index += 1
                parts.append(self._render_reference_run(run, render_nodes, document_nodes, citations))
                seen_references = True
                continue
            if seen_references and not appendix_started and _is_appendix_render_node(child, document_nodes):
                parts.append("\\newpage\n\\appendix")
                appendix_started = True
            if self._should_break_out_of_mixed_columns(child, document_nodes):
                rendered = self._render_mixed_column_breakout_node(
                    child,
                    render_nodes,
                    document_nodes,
                    citations,
                    depth=depth,
                )
                if rendered:
                    parts.append(rendered)
                index += 1
                continue
            list_environment = self._list_environment_for_render_node(child, document_nodes)
            if list_environment is not None:
                run: list[tuple[RenderTreeNode, list[RenderTreeNode]]] = []
                current_attachments: list[RenderTreeNode] | None = None
                while index < len(child_ids):
                    candidate = render_nodes[child_ids[index]]
                    candidate_environment = self._list_environment_for_render_node(candidate, document_nodes)
                    if candidate_environment is not None:
                        current_attachments = []
                        run.append((candidate, current_attachments))
                        index += 1
                        continue
                    if current_attachments is not None and _is_list_continuation_node(candidate, document_nodes):
                        current_attachments.append(candidate)
                        index += 1
                        continue
                    if candidate_environment is None:
                        break
                parts.append(self._render_list_run(run, list_environment, render_nodes, document_nodes, citations, depth=depth))
                continue
            parts.append(self._render_tree_node(child, render_nodes, document_nodes, citations, depth=depth))
            index += 1
        return "\n\n".join(part for part in parts if part)

    def _should_break_out_of_mixed_columns(
        self,
        node: RenderTreeNode,
        document_nodes: dict[str, DocumentNode],
    ) -> bool:
        if self._mixed_column_stack <= 0:
            return False
        if node.role not in {RenderRole.TABLE, RenderRole.FIGURE}:
            return False
        if node.role == RenderRole.TABLE:
            source_layout = self._source_table_layout_for_render_node(node, document_nodes)
            if source_layout is not None and source_layout.width_scope == "page":
                return True
            if source_layout is not None and source_layout.width_scope == "column":
                return False
        width_ratio = _render_node_visual_width_ratio(node, document_nodes)
        return width_ratio is not None and width_ratio >= 0.62

    def _render_mixed_column_breakout_node(
        self,
        node: RenderTreeNode,
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
        citations: CitationResolution | None,
        *,
        depth: int,
    ) -> str:
        # Root mixed-column rendering wraps a logical run in one multicols
        # environment.  A full-width float inside that run has to temporarily
        # escape the environment, otherwise ``\textwidth`` is interpreted inside
        # the narrow column and the crop explodes past the page.
        self._mixed_column_stack -= 1
        try:
            rendered = self._render_tree_node(node, render_nodes, document_nodes, citations, depth=depth)
        finally:
            self._mixed_column_stack += 1
        if not rendered:
            return ""
        return "\\end{multicols}\n\n" + rendered + "\n\n\\begin{multicols}{2}"

    def _render_children_with_mixed_columns(
        self,
        child_ids: list[str],
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
        citations: CitationResolution | None,
        *,
        depth: int,
    ) -> str:
        parts: list[str] = []
        index = 0
        seen_references = False
        appendix_started = False
        while index < len(child_ids):
            child = render_nodes[child_ids[index]]
            band = _render_node_layout_band(child, document_nodes)
            if band.mode != "double" or _is_front_matter_render_node(child, document_nodes):
                full_run = [child_ids[index]]
                index += 1
                while index < len(child_ids):
                    candidate = render_nodes[child_ids[index]]
                    candidate_band = _render_node_layout_band(candidate, document_nodes)
                    if candidate_band.mode == "double" and not _is_front_matter_render_node(candidate, document_nodes):
                        break
                    full_run.append(child_ids[index])
                    index += 1
                if seen_references and not appendix_started and _run_starts_with_appendix(full_run, render_nodes, document_nodes):
                    parts.append("\\newpage\n\\appendix")
                    appendix_started = True
                rendered = self._render_children_standard(
                    full_run,
                    render_nodes,
                    document_nodes,
                    citations,
                    depth=depth,
                )
                if rendered:
                    parts.append(rendered)
                if any(self._is_reference_render_node(render_nodes[child_id], document_nodes) for child_id in full_run if child_id in render_nodes):
                    seen_references = True
                continue

            run = [child_ids[index]]
            index += 1
            while index < len(child_ids):
                candidate = render_nodes[child_ids[index]]
                candidate_band = _render_node_layout_band(candidate, document_nodes)
                if candidate_band.mode != "double":
                    break
                run.append(child_ids[index])
                index += 1

            if seen_references and not appendix_started and _run_starts_with_appendix(run, render_nodes, document_nodes):
                parts.append("\\newpage\n\\appendix")
                appendix_started = True
            self._mixed_column_stack += 1
            try:
                body = self._render_children_standard(
                    run,
                    render_nodes,
                    document_nodes,
                    citations,
                    depth=depth,
                )
            finally:
                self._mixed_column_stack -= 1
            if body:
                parts.append("\\begin{multicols}{2}\n" + body + "\n\\end{multicols}")
            if any(self._is_reference_render_node(render_nodes[child_id], document_nodes) for child_id in run if child_id in render_nodes):
                seen_references = True
        return "\n\n".join(part for part in parts if part)

    def _should_render_mixed_columns(
        self,
        node: RenderTreeNode,
        child_ids: list[str],
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
    ) -> bool:
        if self._mixed_column_stack > 0 or _style_column_mode(self._active_style) != "mixed":
            return False
        # Column environments are physical layout bands, not logical section
        # containers.  Starting a new multicols block inside every section makes
        # LaTeX rebalance each section independently and creates visible
        # horizontal cuts.  Keep section/list trees purely logical; only the root
        # reading flow may open/close column bands.  Abstract is handled by its
        # dedicated renderer because it is a semantic environment with its own
        # layout.
        if node.role != RenderRole.ROOT:
            return False
        return any(
            _render_node_layout_band(render_nodes[child_id], document_nodes).mode == "double"
            for child_id in child_ids
            if child_id in render_nodes and not _is_front_matter_render_node(render_nodes[child_id], document_nodes)
        )

    def _render_list(
        self,
        node: RenderTreeNode,
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
        citations: CitationResolution | None,
        *,
        ordered: bool,
        depth: int,
    ) -> str:
        environment = "enumerate" if ordered or node.attributes.get("ordered") else "itemize"
        lines = [rf"\begin{{{environment}}}"]
        for child_id in _sorted_child_ids(node.children, render_nodes, document_nodes):
            if child_id not in render_nodes:
                continue
            child = render_nodes[child_id]
            body = self._render_tree_node(child, render_nodes, document_nodes, citations, depth=depth + 1)
            lines.append(rf"\item {body}".rstrip())
        lines.append(rf"\end{{{environment}}}")
        return "\n".join(lines)

    def _is_reference_render_node(
        self,
        node: RenderTreeNode,
        document_nodes: dict[str, DocumentNode],
    ) -> bool:
        if node.role in {RenderRole.REFERENCES, RenderRole.REFERENCE_ITEM}:
            return True
        source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
        return bool(source_nodes) and all(source.node_type == BlockType.REFERENCE for source in source_nodes)

    def _is_author_render_node(
        self,
        node: RenderTreeNode,
        document_nodes: dict[str, DocumentNode],
    ) -> bool:
        if node.role == RenderRole.AUTHOR_BLOCK:
            return True
        source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
        return bool(source_nodes) and all(_document_node_is_author_block(source) for source in source_nodes)

    def _render_author_block_run(
        self,
        nodes: list[RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
    ) -> str:
        source_nodes: list[DocumentNode] = []
        fallback_text: list[str] = []
        for node in nodes:
            fallback_text.append(node.latex or node.text or "")
            source_nodes.extend(document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes)
        return render_author_block_original_like(" ".join(fallback_text), source_nodes, self._active_style)

    def _render_reference_run(
        self,
        nodes: list[RenderTreeNode],
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
        citations: CitationResolution | None,
    ) -> str:
        if not nodes:
            return ""
        source_nodes: list[DocumentNode] = []
        for node in nodes:
            source_nodes.extend(document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes)
        return self._render_bibliography_with_tail(citations, source_nodes, nodes[0], render_nodes, document_nodes, depth=0)

    def _render_list_run(
        self,
        nodes: list[tuple[RenderTreeNode, list[RenderTreeNode]]],
        environment: str,
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
        citations: CitationResolution | None,
        *,
        depth: int,
    ) -> str:
        if not nodes:
            return ""
        lines = [rf"\begin{{{environment}}}"]
        for node, attachments in nodes:
            source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
            body = self._render_source_nodes(source_nodes, citations, strip_leading_list_marker=True)
            if not body:
                body = render_text_with_citations(strip_list_marker(node.latex or node.text or ""))
            children = self._render_children(node, render_nodes, document_nodes, citations, depth=depth + 1)
            attachment_body = "\n\n".join(
                self._render_tree_node(attachment, render_nodes, document_nodes, citations, depth=depth + 1)
                for attachment in attachments
            )
            if children:
                body = "\n".join(part for part in [body, children] if part)
            if attachment_body:
                body = "\n\n".join(part for part in [body, attachment_body] if part)
            lines.append(rf"\item {body}".rstrip())
        lines.append(rf"\end{{{environment}}}")
        return "\n".join(lines)

    def _list_environment_for_render_node(
        self,
        node: RenderTreeNode,
        document_nodes: dict[str, DocumentNode],
    ) -> str | None:
        if node.role == RenderRole.LIST:
            return None
        if node.role == RenderRole.LIST_ITEM:
            return "enumerate" if node.attributes.get("ordered") else "itemize"
        source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
        if source_nodes and all(source.node_type == BlockType.LIST for source in source_nodes):
            text = " ".join(source.text for source in source_nodes)
            return _list_environment_for_text(text) or "itemize"
        text = node.text or node.latex or " ".join(source.text for source in source_nodes)
        return _list_environment_for_text(text)

    def _render_source_nodes(
        self,
        nodes: list[DocumentNode],
        citations: CitationResolution | None,
        *,
        strip_leading_list_marker: bool = False,
    ) -> str:
        rendered = [
            self._render_document_node_with_notes(
                node,
                citations,
                strip_leading_list_marker=strip_leading_list_marker and index == 0,
            )
            for index, node in enumerate(nodes)
        ]
        return _merge_rendered_text_fragments(rendered)

    def _render_document_node_with_notes(
        self,
        node: DocumentNode,
        citations: CitationResolution | None,
        *,
        strip_leading_list_marker: bool = False,
    ) -> str:
        rendered = self._render_document_node(node, citations, strip_leading_list_marker=strip_leading_list_marker)
        notes = self._consume_notes_for_source_node(node.node_id)
        if notes:
            note_commands: list[str] = []
            for note in notes:
                rendered = _remove_anchor_marker_from_rendered_text(rendered, note.marker)
                note_commands.append(self._render_note(note))
            joined = "".join(note_commands)
            return rendered + joined if rendered else joined
        return rendered

    def _render_document_node(
        self,
        node: DocumentNode,
        citations: CitationResolution | None,
        *,
        strip_leading_list_marker: bool = False,
    ) -> str:
        text = citations.text_by_node_id.get(node.node_id, node.text) if citations else node.text
        text = _clean_render_text(text)
        if strip_leading_list_marker:
            text = strip_list_marker(text)
        if node.node_type in {BlockType.FOOTNOTE, BlockType.MARGIN_NOTE}:
            return ""
        if node.node_type == BlockType.EQUATION:
            return render_equation(text, label=self._cross_ref_label_for_document_node(node, "equation"))
        if node.node_type == BlockType.INLINE_MATH:
            return render_inline_math(text)
        if node.node_type == BlockType.TABLE:
            return self._render_table([node], text)
        if node.node_type == BlockType.FIGURE:
            return self._render_figure([node], text, citations)
        if node.node_type == BlockType.ALGORITHM:
            return render_algorithm_block(text, label=self._cross_ref_label_for_document_node(node, "algorithm"))
        if node.node_type == BlockType.CODE:
            return "\\begin{verbatim}\n" + safe_verbatim_text(text.strip()) + "\n\\end{verbatim}" if text else ""
        if node.spans:
            return self._render_spans(node, citations, strip_leading_list_marker=strip_leading_list_marker)
        if citations and node.node_id in citations.text_by_node_id:
            return self._render_body_text(text, node=node)
        return self._render_body_text(text, node=node)

    def _consume_notes_for_source_node(self, node_id: str) -> list["_ResolvedNote"]:
        if self._active_notes is None:
            return []
        return self._active_notes.consume_for_anchor(node_id)

    def _render_unanchored_notes(self) -> str:
        if self._active_notes is None:
            return ""
        rendered = [self._render_note(note, anchored=False) for note in self._active_notes.consume_unanchored()]
        return "\n".join(part for part in rendered if part)

    def _render_note(self, note: "_ResolvedNote", *, anchored: bool = True) -> str:
        body = render_text_with_inline_latex(note.text)
        if not body:
            return ""
        if note.kind == "margin_note":
            return rf"\marginpar{{\footnotesize {body}}}"
        if anchored:
            return rf"\footnote{{{body}}}"
        return rf"\footnotetext{{{body}}}"

    def _render_standalone_note(self, kind: str, text: str) -> str:
        body = render_text_with_inline_latex(_strip_note_marker(text)[0])
        if not body:
            return ""
        if kind == "margin_note":
            return rf"\marginpar{{\footnotesize {body}}}"
        return rf"\footnote{{{body}}}"

    def _render_spans(
        self,
        node: DocumentNode,
        citations: CitationResolution | None,
        *,
        strip_leading_list_marker: bool = False,
    ) -> str:
        parts: list[str] = []
        label_to_key = {entry.label: entry.key for entry in citations.entries} if citations else {}
        author_year_to_key = author_year_lookup(citations.entries) if citations else {}
        marker_pending = strip_leading_list_marker
        node_baseline = _node_baseline_font_size(node, self._active_style)
        body_font_class = _body_font_class(self._active_style)
        for span in node.spans:
            text = _clean_render_text(span.text or "")
            if not text:
                continue
            if marker_pending:
                stripped = strip_list_marker(text)
                if stripped == text and LIST_MARKER_RE.match(text + " "):
                    text = ""
                else:
                    text = stripped
                marker_pending = False
                if not text:
                    continue
            if label_to_key or author_year_to_key:
                text, _occurrences, _unresolved = replace_citation_markers(
                    text,
                    label_to_key,
                    author_year_to_key,
                    node_id=node.node_id,
                    enabled=True,
                )
            if span.is_inline_math:
                rendered = render_inline_math(text)
            elif span.is_inline_code:
                rendered = rf"\texttt{{{escape_latex(text)}}}"
            else:
                rendered = render_text_with_citations(text, strip=False)
                rendered = self._apply_span_font_family(rendered, span, body_font_class)
                if span.is_italic:
                    rendered = rf"\textit{{{rendered}}}"
                if span.is_bold:
                    rendered = rf"\textbf{{{rendered}}}"
                script_role = self._span_script_role(span, node, node_baseline)
                marker_note = self._consume_span_marker_note(node.node_id, text, script_role)
                if marker_note is not None:
                    parts.append(self._render_note(marker_note))
                    continue
                if script_role:
                    rendered = _wrap_script(rendered, script_role)
                else:
                    rendered = self._apply_span_font_size(rendered, span, node_baseline)
            parts.append(rendered)
        return self._replace_cross_refs("".join(parts).strip(), node=node)

    def _consume_span_marker_note(self, node_id: str, text: str, script_role: str | None) -> "_ResolvedNote | None":
        if script_role != "superscript" or self._active_notes is None:
            return None
        marker = _inline_note_marker(text)
        if marker is None:
            return None
        return self._active_notes.consume_marker_for_anchor(node_id, marker)

    def _apply_span_font_family(self, rendered: str, span: StyleSpan, body_font_class: str | None) -> str:
        if not self.config.preserve_span_font_family:
            return rendered
        info = resolve_pdf_font(span.font_name)
        if info is None or info.font_class in {"math", body_font_class}:
            return rendered
        if info.font_class == "mono":
            return rf"\texttt{{{rendered}}}"
        if info.font_class == "sans":
            return rf"\textsf{{{rendered}}}"
        if info.font_class == "serif" and body_font_class and body_font_class != "serif":
            return rf"\textrm{{{rendered}}}"
        return rendered

    def _apply_span_font_size(self, rendered: str, span: StyleSpan, baseline_size: float | None) -> str:
        if not self.config.preserve_span_font_size or span.font_size is None or not baseline_size:
            return rendered
        ratio = float(span.font_size) / baseline_size
        if abs(ratio - 1.0) < self.config.span_font_size_delta_threshold:
            return rendered
        line_height = float(span.font_size) * 1.2
        return rf"{{\fontsize{{{float(span.font_size):.2f}pt}}{{{line_height:.2f}pt}}\selectfont {rendered}}}"

    def _span_script_role(self, span: StyleSpan, node: DocumentNode, baseline_size: float | None) -> str | None:
        if not self.config.preserve_span_scripts or span.bbox is None or span.font_size is None or not baseline_size:
            return None
        if float(span.font_size) > baseline_size * self.config.script_font_size_max_ratio:
            return None
        node_box = _node_union_bbox(node)
        if node_box is None:
            return None
        node_height = max(node_box.y1 - node_box.y0, 1e-6)
        span_center = (span.bbox.y0 + span.bbox.y1) / 2.0
        node_center = (node_box.y0 + node_box.y1) / 2.0
        offset_ratio = (span_center - node_center) / node_height
        if offset_ratio <= -self.config.script_vertical_offset_ratio:
            return "superscript"
        if offset_ratio >= self.config.script_vertical_offset_ratio:
            return "subscript"
        return None

    def _render_table(self, source_nodes: list[DocumentNode], text: str) -> str:
        if source_nodes:
            primary = _primary_table_node(source_nodes)
            record = document_node_record(primary)
            source_layout = self._match_source_table_layout(primary, text)
            if source_layout is not None:
                record["source_table_layout"] = source_layout.to_record()
            label = self._cross_ref_label_for_document_node(primary, "table")
            return render_table_placeholder(
                record,
                text or primary.text,
                source_pdf=primary.metadata.get("source_pdf") if self.config.table_asset_output_dir else None,
                asset_output_dir=self.config.table_asset_output_dir,
                asset_latex_prefix=self.config.table_asset_latex_prefix,
                as_nonfloat=self._mixed_column_stack > 0,
                label=label,
            )
        return render_table_placeholder({"type": "table", "text": text}, text, as_nonfloat=self._mixed_column_stack > 0)

    def _match_source_table_layout(self, primary: DocumentNode, text: str) -> SourceTableLayout | None:
        layout = self._active_source_float_layout
        if layout is None:
            return None
        candidates = [
            text,
            primary.text,
            str(primary.metadata.get("table_group_caption") or ""),
            str(primary.metadata.get("table_caption") or ""),
            str(primary.metadata.get("caption") or ""),
        ]
        for caption in candidates:
            match = layout.match_table(caption)
            if match is not None:
                return match
        return None

    def _source_table_layout_for_render_node(
        self,
        node: RenderTreeNode,
        document_nodes: dict[str, DocumentNode],
    ) -> SourceTableLayout | None:
        source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
        if not source_nodes:
            return None
        primary = _primary_table_node(source_nodes)
        return self._match_source_table_layout(primary, node.text or "")

    def _render_figure(
        self,
        source_nodes: list[DocumentNode],
        text: str,
        citations: CitationResolution | None = None,
        *,
        document_nodes: dict[str, DocumentNode] | None = None,
    ) -> str:
        caption = text or "Figure"
        primary = _primary_visual_node(source_nodes, BlockType.FIGURE)
        metadata_caption = _figure_caption_from_metadata(primary, source_nodes)
        if metadata_caption:
            caption = metadata_caption
        caption_from_citations = False
        if not metadata_caption and citations is not None and primary is not None and primary.node_id in citations.text_by_node_id:
            caption = citations.text_by_node_id[primary.node_id]
            caption_from_citations = True
        if source_nodes and not caption_from_citations and not metadata_caption:
            for node in source_nodes:
                value = (
                    node.metadata.get("figure_group_caption")
                    or node.metadata.get("image_group_caption")
                    or node.metadata.get("figure_caption")
                    or node.metadata.get("caption")
                )
                if isinstance(value, str) and value.strip():
                    caption = value.strip()
                    break
        if primary is not None:
            members = _figure_group_members(primary, source_nodes, document_nodes)
            if _is_nonprimary_figure_group_member(primary):
                if not members or primary.node_id != members[0].node_id:
                    return ""
            caption = clean_float_caption_text(caption, "figure") or "Figure"
            if len(members) > 1 and _should_render_figure_minipages(members):
                return render_figure_minipage_group(
                    [document_node_record(member) for member in members],
                    caption,
                    source_pdf=_source_pdf_for_node(primary),
                    asset_output_dir=self.config.figure_asset_output_dir or self.config.table_asset_output_dir,
                    asset_latex_prefix=self.config.figure_asset_latex_prefix,
                    rendered_caption=render_text_with_citations(caption),
                    as_nonfloat=self._mixed_column_stack > 0,
                    label=self._cross_ref_label_for_document_node(primary, "figure"),
                )
        else:
            caption = clean_float_caption_text(caption, "figure") or "Figure"
        record = document_node_record(primary) if primary is not None else {"type": "figure", "text": caption}
        return render_figure_block(
            record,
            caption,
            source_pdf=_source_pdf_for_node(primary) if primary is not None else None,
            asset_output_dir=self.config.figure_asset_output_dir or self.config.table_asset_output_dir,
            asset_latex_prefix=self.config.figure_asset_latex_prefix,
            rendered_caption=render_text_with_citations(caption),
            as_nonfloat=self._mixed_column_stack > 0,
            label=self._cross_ref_label_for_document_node(primary, "figure") if primary is not None else None,
        )

    def _render_bibliography(
        self,
        citations: CitationResolution | None,
        source_nodes: list[DocumentNode],
        node: RenderTreeNode,
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
    ) -> str:
        if citations and citations.raw_bibliography_latex:
            return self._wrap_bibliography_columns(citations.raw_bibliography_latex, source_nodes)
        if citations and citations.entries:
            lines = [r"\begin{thebibliography}{99}"]
            for entry in citations.entries:
                optional = f"[{render_text_with_inline_latex(entry.display_label)}]" if entry.display_label else ""
                lines.append(rf"\bibitem{optional}{{{entry.key}}} {render_text_with_inline_latex(entry.text)}")
            lines.append(r"\end{thebibliography}")
            return self._wrap_bibliography_columns("\n".join(lines), source_nodes)
        if source_nodes:
            lines = [r"\begin{thebibliography}{99}"]
            for index, source in enumerate(source_nodes, start=1):
                lines.append(rf"\bibitem{{ref_{index}}} {render_text_with_inline_latex(strip_reference_label(source.text))}")
            lines.append(r"\end{thebibliography}")
            return self._wrap_bibliography_columns("\n".join(lines), source_nodes)
        return self._render_children(node, render_nodes, document_nodes, citations, depth=0)

    def _render_bibliography_with_tail(
        self,
        citations: CitationResolution | None,
        source_nodes: list[DocumentNode],
        node: RenderTreeNode,
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
        *,
        depth: int,
    ) -> str:
        bibliography = self._render_bibliography(citations, source_nodes, node, render_nodes, document_nodes)
        tail_ids = [
            child_id
            for child_id in _sorted_child_ids(node.children, render_nodes, document_nodes)
            if child_id in render_nodes and not self._is_reference_render_node(render_nodes[child_id], document_nodes)
        ]
        if not tail_ids:
            return bibliography
        is_appendix_tail = _run_starts_with_appendix(tail_ids, render_nodes, document_nodes)
        if is_appendix_tail:
            tail = self._render_children_with_mixed_columns(
                tail_ids,
                render_nodes,
                document_nodes,
                citations,
                depth=depth + 1,
            )
        else:
            tail = self._render_children_standard(tail_ids, render_nodes, document_nodes, citations, depth=depth + 1)
            use_double_columns = _run_should_use_double_columns(tail_ids, render_nodes, document_nodes, self._active_style)
            if self._mixed_column_stack == 0 and use_double_columns:
                self._mixed_column_stack += 1
                try:
                    tail = "\\begin{multicols}{2}\n" + tail + "\n\\end{multicols}"
                finally:
                    self._mixed_column_stack -= 1
        if not tail:
            return bibliography
        marker = "\\newpage\n\\appendix" if is_appendix_tail else "\\newpage"
        return "\n\n".join(part for part in [bibliography, marker, tail] if part)

    def _wrap_bibliography_columns(self, latex: str, source_nodes: list[DocumentNode]) -> str:
        body = str(latex or "").strip()
        if not body:
            return ""
        if self._mixed_column_stack > 0 or r"\begin{multicols}" in body:
            return body
        style = self._active_style
        if style is not None and "twocolumn" in set(style.documentclass_options or []):
            return body
        if _bibliography_should_use_double_columns(source_nodes, style):
            return "\\begin{multicols}{2}\n" + body + "\n\\end{multicols}"
        return body

    def _source_text(self, nodes: list[DocumentNode], citations: CitationResolution | None) -> str:
        return _merge_rendered_text_fragments(
            [citations.text_by_node_id.get(node.node_id, node.text) if citations else node.text for node in nodes]
        )

    def _render_body_text(self, text: str, *, node: DocumentNode | None = None) -> str:
        repaired = self._replace_cross_refs(text, node=node)
        return render_text_with_citations(repaired)

    def _replace_cross_refs(self, text: str, *, node: DocumentNode | None = None) -> str:
        registry = self._active_cross_refs
        if registry is None:
            return str(text or "")
        if node is not None and node.node_type not in {BlockType.TEXT, BlockType.LIST, BlockType.TITLE}:
            return str(text or "")
        return registry.replace_text(str(text or ""))

    def _cross_ref_label_for_document_node(self, node: DocumentNode | None, kind: str | None = None) -> str | None:
        registry = self._active_cross_refs
        if registry is None or node is None:
            return None
        return registry.label_for_node(node.node_id, kind=kind)

    def _cross_ref_label_for_render_node(
        self,
        node: RenderTreeNode,
        document_nodes: dict[str, DocumentNode],
        kind: str | None = None,
    ) -> str | None:
        registry = self._active_cross_refs
        if registry is None:
            return None
        for source_id in node.source_node_ids:
            label = registry.label_for_node(source_id, kind=kind)
            if label:
                return label
        if node.text:
            return registry.label_for_text(kind or _render_role_cross_ref_kind(node.role), node.text)
        return None

    def _infer_title(self, document: DocumentIR) -> str | None:
        for node_id in document.reading_order:
            node = next((item for item in document.nodes if item.node_id == node_id), None)
            if node and node.node_type == BlockType.TITLE and node.text:
                return node.text
        for node in sorted(document.nodes, key=lambda item: item.reading_index):
            if node.node_type == BlockType.TITLE and node.text:
                return node.text
        return None

    def _use_original_like_front_matter(self) -> bool:
        return str(self.config.front_matter_mode or "").casefold() in {"original_like", "original", "visual"}

    def _use_maketitle(self) -> bool:
        return bool(self.config.include_maketitle and not self._use_original_like_front_matter())


def render_text_with_citations(text: str, *, strip: bool = True) -> str:
    """Render text that may already contain semantic ``\\cite{...}`` commands."""

    value = _clean_render_text(text)
    if not value:
        return ""
    parts: list[str] = []
    cursor = 0
    for match in CITE_COMMAND_RE.finditer(value):
        if match.start() > cursor:
            parts.append(render_text_with_inline_latex(value[cursor : match.start()], strip=False))
        parts.append(match.group(0))
        cursor = match.end()
    if cursor < len(value):
        parts.append(render_text_with_inline_latex(value[cursor:], strip=False))
    output = "".join(parts)
    return output.strip() if strip else output


CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_render_text(text: object) -> str:
    value = str(text or "")
    value = CONTROL_CHAR_RE.sub("", value)
    value = value.replace("\r", "\n")
    return value


def _is_abstract_label(text: str) -> bool:
    return " ".join(str(text or "").split()).casefold().strip(" .:;") == "abstract"


def _clean_heading_text(text: str) -> str:
    value = _clean_render_text(text)
    stripped = strip_title_numbering(value)
    return stripped.strip() or value.strip()


def _clean_appendix_heading_text(text: str) -> str:
    value = _clean_render_text(text).strip()
    if not value:
        return ""
    stripped = APPENDIX_TITLE_PREFIX_RE.sub("", value, count=1).strip()
    return stripped or _clean_heading_text(value)


def _merge_rendered_text_fragments(parts: list[str]) -> str:
    text = ""
    for part in parts:
        part = " ".join(str(part or "").split())
        if not part:
            continue
        if not text:
            text = part
            continue
        if _render_join_without_space(text, part):
            if MERGE_TRAILING_HYPHEN_RE.search(text):
                text = MERGE_TRAILING_HYPHEN_RE.sub("", text) + part
            else:
                text += part
        else:
            text += " " + part
    return " ".join(text.split()).strip()


def _render_join_without_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if MERGE_TRAILING_HYPHEN_RE.search(left) and right[:1].islower():
        return True
    if right[0] in ",.;:!?%)]}，。；：！？、》）】":
        return True
    if left[-1] in "([{（《【":
        return True
    return False


def document_node_record(node: DocumentNode) -> dict[str, object]:
    record: dict[str, object] = dict(node.metadata)
    record.setdefault("id", node.node_id)
    record.setdefault("type", node.node_type.value)
    record.setdefault("page_idx", node.page_idx)
    for key in ("page_width", "page_height"):
        if key in node.features:
            record.setdefault(key, node.features[key])
    if node.bboxes:
        record.setdefault("bbox", node.bboxes[0].to_list())
    record.setdefault("text", node.text)
    return record


@dataclass(frozen=True)
class _ResolvedNote:
    node: DocumentNode
    kind: str
    text: str
    marker: str | None
    anchor_id: str | None = None


@dataclass
class _NoteContext:
    by_anchor: dict[str, list[_ResolvedNote]]
    unanchored: list[_ResolvedNote]
    consumed_node_ids: set[str] = field(default_factory=set)

    @classmethod
    def from_document(cls, document: DocumentIR) -> "_NoteContext":
        body_nodes = [
            node
            for node in sorted(document.nodes, key=lambda item: item.reading_index)
            if node.node_type
            not in {
                BlockType.FOOTNOTE,
                BlockType.MARGIN_NOTE,
                BlockType.HEADER_FOOTER,
                BlockType.TOC,
                BlockType.OTHER,
            }
        ]
        body_by_id = {node.node_id: node for node in body_nodes}
        marker_candidates = _body_note_marker_candidates(body_nodes)
        by_anchor: dict[str, list[_ResolvedNote]] = {}
        unanchored: list[_ResolvedNote] = []
        for note_node in sorted(document.nodes, key=lambda item: item.reading_index):
            if note_node.node_type not in {BlockType.FOOTNOTE, BlockType.MARGIN_NOTE}:
                continue
            text, marker = _strip_note_marker(note_node.text, note_node.metadata)
            if not text:
                continue
            kind = "margin_note" if note_node.node_type == BlockType.MARGIN_NOTE else "footnote"
            anchor_id = (
                _explicit_note_anchor(note_node)
                or _marker_note_anchor(note_node, marker, marker_candidates, body_by_id)
                or _nearest_note_anchor(note_node, body_nodes)
            )
            note = _ResolvedNote(node=note_node, kind=kind, text=text, marker=marker, anchor_id=anchor_id)
            if anchor_id:
                by_anchor.setdefault(anchor_id, []).append(note)
            else:
                unanchored.append(note)
        return cls(by_anchor=by_anchor, unanchored=unanchored)

    def consume_for_anchor(self, node_id: str) -> list[_ResolvedNote]:
        notes = [
            note
            for note in self.by_anchor.get(node_id, [])
            if note.node.node_id not in self.consumed_node_ids
        ]
        for note in notes:
            self.consumed_node_ids.add(note.node.node_id)
        return notes

    def consume_marker_for_anchor(self, node_id: str, marker: str) -> _ResolvedNote | None:
        marker_key = _normalize_note_marker(marker)
        if not marker_key:
            return None
        for note in self.by_anchor.get(node_id, []):
            if note.node.node_id in self.consumed_node_ids:
                continue
            if _normalize_note_marker(note.marker) != marker_key:
                continue
            self.consumed_node_ids.add(note.node.node_id)
            return note
        return None

    def consume_unanchored(self) -> list[_ResolvedNote]:
        notes = [
            note
            for note in self.unanchored
            if note.node.node_id not in self.consumed_node_ids
        ]
        for anchored_notes in self.by_anchor.values():
            notes.extend(
                note
                for note in anchored_notes
                if note.node.node_id not in self.consumed_node_ids
            )
        for note in notes:
            self.consumed_node_ids.add(note.node.node_id)
        return notes


_CROSS_REF_NUMBER = r"(?:\d+(?:\.\d+)*[A-Za-z]?|[IVXLCDM]+)"
_CROSS_REF_TEXT_RE = re.compile(
    rf"(?<!\\)\b(?P<name>Figure|Fig\.|Table|Equation|Eq\.|Algorithm)"
    rf"\s*(?P<open>\()?\s*(?P<number>{_CROSS_REF_NUMBER})\s*(?P<close>\))?",
    re.IGNORECASE,
)
_CROSS_REF_SOURCE_RE = {
    "figure": re.compile(rf"\b(?:Figure|Fig\.?)\s*(?P<number>{_CROSS_REF_NUMBER})\b", re.IGNORECASE),
    "table": re.compile(rf"\b(?:Table|Tab\.?)\s*(?P<number>{_CROSS_REF_NUMBER})\b", re.IGNORECASE),
    "algorithm": re.compile(rf"\bAlgorithm\s*(?P<number>{_CROSS_REF_NUMBER})\b", re.IGNORECASE),
    "equation": re.compile(rf"\b(?:Equation|Eq\.?)\s*\(?\s*(?P<number>{_CROSS_REF_NUMBER})\s*\)?", re.IGNORECASE),
}
_EQUATION_TAG_RE = re.compile(r"\\tag\s*\{\s*(?P<number>[^{}]+?)\s*\}")
_TRAILING_EQUATION_NUMBER_RE = re.compile(r"\(\s*(?P<number>\d+(?:\.\d+)*[A-Za-z]?)\s*\)\s*$")


@dataclass(frozen=True)
class _CrossReferenceRegistry:
    labels_by_kind_number: dict[tuple[str, str], str]
    labels_by_node_id: dict[str, str]

    @classmethod
    def from_document(cls, document: DocumentIR) -> "_CrossReferenceRegistry":
        labels_by_kind_number: dict[tuple[str, str], str] = {}
        labels_by_node_id: dict[str, str] = {}
        for node in sorted(document.nodes, key=lambda item: item.reading_index):
            kind = _document_node_cross_ref_kind(node)
            if kind is None:
                continue
            number = _cross_ref_number_for_node(node, kind)
            if not number:
                continue
            normalized_number = _normalize_cross_ref_number(number)
            if not normalized_number:
                continue
            explicit_label = _explicit_cross_ref_label(node, kind)
            label = explicit_label or _default_cross_ref_label(kind, normalized_number)
            key = (kind, normalized_number.casefold())
            labels_by_kind_number.setdefault(key, label)
            labels_by_node_id.setdefault(node.node_id, labels_by_kind_number[key])
        return cls(labels_by_kind_number=labels_by_kind_number, labels_by_node_id=labels_by_node_id)

    def label_for_node(self, node_id: str, *, kind: str | None = None) -> str | None:
        label = self.labels_by_node_id.get(node_id)
        if label is None:
            return None
        if kind is not None and not label.startswith(_cross_ref_label_prefix(kind) + ":"):
            return None
        return label

    def label_for_text(self, kind: str | None, text: str) -> str | None:
        if kind is None:
            return None
        number = _cross_ref_number_from_text(kind, text)
        if not number:
            return None
        return self.labels_by_kind_number.get((kind, _normalize_cross_ref_number(number).casefold()))

    def replace_text(self, text: str) -> str:
        value = str(text or "")
        if not value or not self.labels_by_kind_number:
            return value

        def replacer(match: re.Match[str]) -> str:
            kind = _cross_ref_kind_from_name(match.group("name"))
            number = _normalize_cross_ref_number(match.group("number"))
            label = self.labels_by_kind_number.get((kind, number.casefold()))
            if not label:
                return match.group(0)
            display = match.group("name").rstrip()
            return rf"{display} \ref{{{label}}}"

        return _CROSS_REF_TEXT_RE.sub(replacer, value)

    def referenced_labels(self, document: DocumentIR) -> set[str]:
        labels: set[str] = set()
        if not self.labels_by_kind_number:
            return labels
        for node in document.nodes:
            if node.node_type in {BlockType.FIGURE, BlockType.TABLE, BlockType.EQUATION, BlockType.ALGORITHM, BlockType.REFERENCE}:
                continue
            value = str(node.text or "")
            for match in _CROSS_REF_TEXT_RE.finditer(value):
                kind = _cross_ref_kind_from_name(match.group("name"))
                number = _normalize_cross_ref_number(match.group("number"))
                label = self.labels_by_kind_number.get((kind, number.casefold()))
                if label:
                    labels.add(label)
        return labels


def _document_node_cross_ref_kind(node: DocumentNode) -> str | None:
    if node.node_type == BlockType.FIGURE:
        return "figure"
    if node.node_type == BlockType.TABLE:
        return "table"
    if node.node_type == BlockType.EQUATION:
        return "equation"
    if node.node_type == BlockType.ALGORITHM:
        return "algorithm"
    return None


def _render_role_cross_ref_kind(role: RenderRole) -> str | None:
    if role == RenderRole.FIGURE:
        return "figure"
    if role == RenderRole.TABLE:
        return "table"
    if role == RenderRole.DISPLAY_EQUATION:
        return "equation"
    if role == RenderRole.ALGORITHM:
        return "algorithm"
    return None


def _render_role_for_cross_ref_kind(kind: str) -> RenderRole | None:
    if kind == "figure":
        return RenderRole.FIGURE
    if kind == "table":
        return RenderRole.TABLE
    if kind == "equation":
        return RenderRole.DISPLAY_EQUATION
    if kind == "algorithm":
        return RenderRole.ALGORITHM
    return None


def _cross_ref_kind_from_name(name: str) -> str:
    normalized = str(name or "").casefold().rstrip(".")
    if normalized in {"figure", "fig"}:
        return "figure"
    if normalized in {"equation", "eq"}:
        return "equation"
    if normalized == "algorithm":
        return "algorithm"
    return "table"


def _cross_ref_number_for_node(node: DocumentNode, kind: str) -> str | None:
    for container in (node.metadata, node.features):
        for key in (
            f"{kind}_number",
            f"{kind}_label",
            "equation_number",
            "float_number",
            "caption_number",
            "number",
        ):
            value = container.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(int(value))
            if isinstance(value, str) and value.strip():
                extracted = _cross_ref_number_from_text(kind, value) or value.strip()
                if re.fullmatch(_CROSS_REF_NUMBER, extracted, re.IGNORECASE):
                    return extracted
    for value in _cross_ref_text_candidates(node, kind):
        number = _cross_ref_number_from_text(kind, value)
        if number:
            return number
    if kind == "equation":
        tag_match = _EQUATION_TAG_RE.search(node.text or "")
        if tag_match:
            return tag_match.group("number")
        trailing = _TRAILING_EQUATION_NUMBER_RE.search(node.text or "")
        if trailing:
            return trailing.group("number")
    return None


def _cross_ref_text_candidates(node: DocumentNode, kind: str) -> list[str]:
    keys = {
        "figure": ("figure_group_caption", "image_group_caption", "figure_caption", "caption", "text"),
        "table": ("table_group_caption", "table_caption", "caption", "text"),
        "algorithm": ("algorithm_caption", "caption", "text"),
        "equation": ("equation_label", "caption", "text"),
    }.get(kind, ("text",))
    values: list[str] = []
    for key in keys:
        value = node.metadata.get(key) if key != "text" else node.text
        if isinstance(value, str) and value.strip():
            values.append(value)
    return values


def _cross_ref_number_from_text(kind: str, text: str) -> str | None:
    value = str(text or "")
    pattern = _CROSS_REF_SOURCE_RE.get(kind)
    if pattern is None:
        return None
    match = pattern.search(value)
    if match:
        return match.group("number")
    return None


def _explicit_cross_ref_label(node: DocumentNode, kind: str) -> str | None:
    for container in (node.metadata, node.features):
        for key in ("latex_label", "tex_label", "source_label", "cross_ref_label"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                label = _sanitize_cross_ref_label(value.strip())
                if label and label.startswith(_cross_ref_label_prefix(kind) + ":"):
                    return label
    return None


def _default_cross_ref_label(kind: str, number: str) -> str:
    return f"{_cross_ref_label_prefix(kind)}:{_sanitize_cross_ref_number(number)}"


def _cross_ref_label_prefix(kind: str) -> str:
    return {
        "figure": "fig",
        "table": "tab",
        "equation": "eq",
        "algorithm": "alg",
    }.get(kind, "ref")


def _normalize_cross_ref_number(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().strip("()[]{}"))


def _sanitize_cross_ref_number(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z.:-]+", "_", _normalize_cross_ref_number(value)).strip("_") or "unknown"


def _sanitize_cross_ref_label(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z:._-]+", "_", str(value or "").strip()).strip("_")


def _safe_render_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "").strip()).strip("_") or "unknown"


NOTE_MARKER_RE = re.compile(
    r"^\s*(?:(?:\[(?P<bracket>[0-9A-Za-z*†‡§¶]+)\])|(?:\((?P<paren>[0-9A-Za-z*†‡§¶]+)\))|(?P<bare>[0-9]{1,3}|[*†‡§¶]))[\s:.\-]*"
)
INLINE_NOTE_MARKER_RE = re.compile(r"^\s*(?:\(?\s*)?(?P<marker>[0-9]{1,3}|[*†‡§¶])(?:\s*\)?)\s*$")
RENDERED_NOTE_MARKER_RE = re.compile(
    r"(?:\\raisebox\{[^{}]*\}\{\\scriptsize\s+(?P<raised>[0-9]{1,3}|[*†‡§¶])\}"
    r"|\$?\^\{?(?P<tex>[0-9]{1,3}|[*†‡§¶])\}?\$?"
    r"|(?P<unicode>[⁰¹²³⁴⁵⁶⁷⁸⁹]+))\s*$"
)
SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _strip_note_marker(text: str, metadata: dict[str, object] | None = None) -> tuple[str, str | None]:
    metadata = metadata or {}
    marker_value = metadata.get("footnote_marker") or metadata.get("footnote_label")
    marker = str(marker_value).strip() if marker_value is not None and str(marker_value).strip() else None
    value = str(text or "").strip()
    match = NOTE_MARKER_RE.match(value)
    if match:
        marker = marker or next((group for group in match.groups() if group), None)
        value = value[match.end() :].strip()
    return value, marker


def _normalize_note_marker(marker: object | None) -> str | None:
    value = str(marker or "").strip()
    if not value:
        return None
    value = value.translate(SUPERSCRIPT_DIGITS)
    value = value.strip("[](){}.:;- ")
    return value.casefold() or None


def _inline_note_marker(text: str) -> str | None:
    value = str(text or "").strip().translate(SUPERSCRIPT_DIGITS)
    match = INLINE_NOTE_MARKER_RE.match(value)
    return match.group("marker") if match else None


def _explicit_note_anchor(note_node: DocumentNode) -> str | None:
    value = note_node.metadata.get("footnote_anchor") or note_node.metadata.get("anchor_node_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _body_note_marker_candidates(body_nodes: list[DocumentNode]) -> dict[str, list[DocumentNode]]:
    candidates: dict[str, list[DocumentNode]] = {}
    for node in body_nodes:
        for marker in _node_note_markers(node):
            key = _normalize_note_marker(marker)
            if key:
                candidates.setdefault(key, []).append(node)
    return candidates


def _node_note_markers(node: DocumentNode) -> list[str]:
    markers: list[str] = []
    for key in ("footnote_marker", "footnote_label", "note_marker", "note_label"):
        value = node.metadata.get(key) or node.features.get(key)
        normalized = _normalize_note_marker(value)
        if normalized:
            markers.append(normalized)
    for span in node.spans:
        if span.bbox is None:
            continue
        marker = _inline_note_marker(span.text)
        if marker is None:
            continue
        baseline = _node_baseline_font_size(node, None)
        if baseline and span.font_size and float(span.font_size) > float(baseline) * 0.9:
            continue
        markers.append(marker)
    text_markers = re.findall(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", node.text or "")
    markers.extend(marker.translate(SUPERSCRIPT_DIGITS) for marker in text_markers)
    return markers


def _marker_note_anchor(
    note_node: DocumentNode,
    marker: str | None,
    marker_candidates: dict[str, list[DocumentNode]],
    body_by_id: dict[str, DocumentNode],
) -> str | None:
    key = _normalize_note_marker(marker)
    if not key:
        return None
    candidates = marker_candidates.get(key, [])
    if not candidates:
        return None
    same_page = [candidate for candidate in candidates if candidate.page_idx == note_node.page_idx]
    pool = same_page or candidates
    previous = [candidate for candidate in pool if candidate.reading_index <= note_node.reading_index]
    pool = previous or pool
    best = min(pool, key=lambda item: (_bbox_vertical_distance(note_node, item), abs(note_node.reading_index - item.reading_index)))
    return best.node_id if best.node_id in body_by_id else None


def _nearest_note_anchor(note_node: DocumentNode, body_nodes: list[DocumentNode]) -> str | None:
    previous_same_page = [
        candidate
        for candidate in body_nodes
        if candidate.page_idx == note_node.page_idx and candidate.reading_index < note_node.reading_index
    ]
    if previous_same_page:
        return max(previous_same_page, key=lambda item: item.reading_index).node_id
    previous_any_page = [candidate for candidate in body_nodes if candidate.reading_index < note_node.reading_index]
    if previous_any_page:
        return max(previous_any_page, key=lambda item: item.reading_index).node_id
    same_page = [candidate for candidate in body_nodes if candidate.page_idx == note_node.page_idx]
    if same_page:
        return min(same_page, key=lambda item: _bbox_vertical_distance(note_node, item)).node_id
    return None


def _bbox_vertical_distance(left: DocumentNode, right: DocumentNode) -> float:
    if not left.bboxes or not right.bboxes:
        return abs(left.reading_index - right.reading_index)
    left_box = left.bboxes[0]
    right_box = right.bboxes[0]
    left_center = (left_box.y0 + left_box.y1) / 2.0
    right_center = (right_box.y0 + right_box.y1) / 2.0
    return abs(left_center - right_center)


def _remove_anchor_marker_from_rendered_text(rendered: str, marker: str | None) -> str:
    key = _normalize_note_marker(marker)
    if not key or not rendered:
        return rendered
    match = RENDERED_NOTE_MARKER_RE.search(rendered)
    if not match:
        return rendered
    value = next((group for group in match.groups() if group), "")
    if _normalize_note_marker(value) != key:
        return rendered
    return rendered[: match.start()].rstrip()


def _list_environment_for_text(text: str) -> str | None:
    value = str(text or "")
    if not LIST_MARKER_RE.match(value):
        return None
    return "enumerate" if ORDERED_LIST_MARKER_RE.match(value) else "itemize"


def _is_list_continuation_node(
    node: RenderTreeNode,
    document_nodes: dict[str, DocumentNode],
) -> bool:
    """Allow display material that belongs inside an active list item.

    MinerU often separates equations that logically live under the preceding
    ``1. text`` list item.  If such nodes are siblings in RenderTreeIR, keeping
    them inside the currently open list is safer than closing and reopening the
    environment around every formula.
    """

    if node.role in {
        RenderRole.DISPLAY_EQUATION,
        RenderRole.INLINE_MATH,
        RenderRole.FIGURE,
        RenderRole.TABLE,
        RenderRole.ALGORITHM,
        RenderRole.CODE,
    }:
        return True
    source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
    return bool(source_nodes) and all(
        source.node_type
        in {
            BlockType.EQUATION,
            BlockType.INLINE_MATH,
            BlockType.FIGURE,
            BlockType.TABLE,
            BlockType.ALGORITHM,
            BlockType.CODE,
        }
        for source in source_nodes
    )


def _run_starts_with_appendix(
    child_ids: list[str],
    render_nodes: dict[str, RenderTreeNode],
    document_nodes: dict[str, DocumentNode],
) -> bool:
    for child_id in child_ids:
        node = render_nodes.get(child_id)
        if node is None:
            continue
        return _is_appendix_render_node(node, document_nodes)
    return False


def _run_should_use_double_columns(
    child_ids: list[str],
    render_nodes: dict[str, RenderTreeNode],
    document_nodes: dict[str, DocumentNode],
    style: StyleProfile | None,
) -> bool:
    if style is not None and "twocolumn" in set(style.documentclass_options or []):
        return False
    bands = [
        _render_node_layout_band(render_nodes[child_id], document_nodes)
        for child_id in child_ids
        if child_id in render_nodes
    ]
    if any(band.mode == "double" for band in bands):
        return True
    return _style_column_mode(style) == "two_column"


def _appendix_run_should_use_double_columns(
    child_ids: list[str],
    render_nodes: dict[str, RenderTreeNode],
    document_nodes: dict[str, DocumentNode],
    style: StyleProfile | None,
) -> bool:
    """Decide appendix columns from the appendix's own bbox proportions.

    Appendix material appears after References structurally, but its layout is
    not inherited from References or from the document-wide column mode.  Some
    papers reset appendices to a single column, while others keep the body
    two-column template.  Use the physical widths of the appendix subtree as the
    first-class signal.
    """

    source_nodes = _collect_render_subtree_source_nodes(child_ids, render_nodes, document_nodes)
    bbox_mode = _bbox_width_column_mode(source_nodes)
    if bbox_mode == "double":
        return True
    if bbox_mode == "full":
        return False
    return _run_should_use_double_columns(child_ids, render_nodes, document_nodes, style)


def _collect_render_subtree_source_nodes(
    child_ids: list[str],
    render_nodes: dict[str, RenderTreeNode],
    document_nodes: dict[str, DocumentNode],
) -> list[DocumentNode]:
    result: list[DocumentNode] = []
    visited: set[str] = set()

    def visit(render_id: str) -> None:
        if render_id in visited:
            return
        visited.add(render_id)
        node = render_nodes.get(render_id)
        if node is None:
            return
        for source_id in node.source_node_ids:
            source = document_nodes.get(source_id)
            if source is not None:
                result.append(source)
        for child_id in node.children:
            visit(child_id)

    for child_id in child_ids:
        visit(child_id)
    return result


def _bbox_width_column_mode(nodes: list[DocumentNode]) -> str | None:
    excluded = {BlockType.FIGURE, BlockType.TABLE, BlockType.HEADER_FOOTER, BlockType.FOOTNOTE, BlockType.MARGIN_NOTE, BlockType.TOC}
    preferred = {BlockType.TEXT, BlockType.LIST, BlockType.EQUATION, BlockType.ALGORITHM, BlockType.CODE}
    candidates = [node for node in nodes if node.node_type in preferred and node.bboxes]
    if not candidates:
        candidates = [node for node in nodes if node.node_type not in excluded and node.bboxes]
    ratios: list[float] = []
    for node in candidates:
        page_width = _page_width_for_nodes([node])
        if page_width <= 0:
            continue
        for box in node.bboxes:
            width = max(box.x1 - box.x0, 0.0)
            ratio = width / page_width
            if 0.02 <= ratio <= 1.20:
                ratios.append(ratio)
    if not ratios:
        return None
    ordered = sorted(ratios)
    median = ordered[len(ordered) // 2]
    narrow_share = sum(1 for ratio in ratios if ratio < 0.62) / len(ratios)
    wide_share = sum(1 for ratio in ratios if ratio >= 0.65) / len(ratios)
    if narrow_share >= 0.60 or median < 0.62:
        return "double"
    if wide_share >= 0.50 or median >= 0.65:
        return "full"
    return None


def _is_appendix_render_node(
    node: RenderTreeNode,
    document_nodes: dict[str, DocumentNode],
) -> bool:
    if node.attributes.get("appendix_heading"):
        return True
    if node.role not in {RenderRole.SECTION, RenderRole.SUBSECTION, RenderRole.SUBSUBSECTION}:
        return False
    source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
    return any(
        bool(source.metadata.get("_appendix_heading") or source.features.get("_appendix_heading"))
        for source in source_nodes
    )


def _sorted_child_ids(
    child_ids: list[str],
    render_nodes: dict[str, RenderTreeNode],
    document_nodes: dict[str, DocumentNode],
) -> list[str]:
    return sorted(
        [child_id for child_id in child_ids if child_id in render_nodes],
        key=lambda child_id: _render_node_order_key(render_nodes[child_id], document_nodes),
    )


def _defer_sentence_interrupting_floats(
    child_ids: list[str],
    render_nodes: dict[str, RenderTreeNode],
    document_nodes: dict[str, DocumentNode],
) -> list[str]:
    """Move a table/figure out of the middle of an open sentence."""

    if len(child_ids) < 3:
        return child_ids
    output: list[str] = []
    index = 0
    changed = False
    while index < len(child_ids):
        child_id = child_ids[index]
        child = render_nodes.get(child_id)
        previous = render_nodes.get(output[-1]) if output else None
        next_node = render_nodes.get(child_ids[index + 1]) if index + 1 < len(child_ids) else None
        if (
            child is not None
            and previous is not None
            and next_node is not None
            and _is_sentence_interrupting_float(previous, child, next_node, document_nodes)
        ):
            floats: list[str] = []
            while index < len(child_ids):
                candidate = render_nodes.get(child_ids[index])
                if candidate is None or candidate.role not in {RenderRole.TABLE, RenderRole.FIGURE}:
                    break
                floats.append(child_ids[index])
                index += 1

            continuations: list[str] = []
            while index < len(child_ids):
                candidate = render_nodes.get(child_ids[index])
                if candidate is None or candidate.role in {RenderRole.TABLE, RenderRole.FIGURE}:
                    break
                if not _render_node_starts_like_continuation(candidate, document_nodes):
                    break
                continuations.append(child_ids[index])
                index += 1
                if not _render_node_is_open_sentence(candidate, document_nodes):
                    break

            if continuations:
                output.extend(continuations)
                output.extend(floats)
            else:
                output.extend(floats)
            changed = True
            continue
        output.append(child_id)
        index += 1
    return output if changed else child_ids


def _is_sentence_interrupting_float(
    previous: RenderTreeNode,
    floating: RenderTreeNode,
    next_node: RenderTreeNode,
    document_nodes: dict[str, DocumentNode],
) -> bool:
    if floating.role not in {RenderRole.TABLE, RenderRole.FIGURE}:
        return False
    if not _render_node_is_text_flow(previous, document_nodes):
        return False
    if not _render_node_is_text_flow(next_node, document_nodes):
        return False
    return _render_node_is_open_sentence(previous, document_nodes) and _render_node_starts_like_continuation(next_node, document_nodes)


def _render_node_is_text_flow(node: RenderTreeNode, document_nodes: dict[str, DocumentNode]) -> bool:
    if node.role in {RenderRole.PARAGRAPH, RenderRole.LIST_ITEM, RenderRole.CAPTION}:
        return True
    if node.role == RenderRole.RAW_LATEX:
        return False
    source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
    return bool(source_nodes) and all(
        source.node_type in {BlockType.TEXT, BlockType.LIST, BlockType.REFERENCE}
        for source in source_nodes
    )


def _render_node_is_open_sentence(node: RenderTreeNode, document_nodes: dict[str, DocumentNode]) -> bool:
    text = _render_node_plain_text(node, document_nodes)
    if not text:
        return False
    return not re.search(r"[.!?。！？]\s*(?:[])}\"'”’»]+)?\s*$", text)


def _render_node_starts_like_continuation(node: RenderTreeNode, document_nodes: dict[str, DocumentNode]) -> bool:
    text = _render_node_plain_text(node, document_nodes).strip()
    if not text:
        return False
    return bool(
        re.match(
            r"^(?:[a-z,;:)\]}]|and\b|or\b|with\b|where\b|which\b|that\b|while\b|because\b|for\b|in\b|of\b|to\b|the\b|as\b|from\b|by\b|on\b|using\b|under\b|between\b)",
            text,
            re.IGNORECASE,
        )
    )


def _render_node_plain_text(node: RenderTreeNode, document_nodes: dict[str, DocumentNode]) -> str:
    text = node.latex or node.text or ""
    if text:
        return " ".join(str(text).split())
    parts: list[str] = []
    for source_id in node.source_node_ids:
        source = document_nodes.get(source_id)
        if source is not None and source.text:
            parts.append(source.text)
    return " ".join(" ".join(parts).split())


def _render_node_order_key(
    node: RenderTreeNode,
    document_nodes: dict[str, DocumentNode],
) -> tuple[int, float, str]:
    bias = _numeric_value(node.attributes.get("render_order_bias"))
    if bias is not None:
        return (-1, bias, node.render_id)
    indexes = [
        document_nodes[node_id].reading_index
        for node_id in node.source_node_ids
        if node_id in document_nodes
    ]
    if indexes:
        return (0, float(min(indexes)), node.render_id)
    for key in ("reading_index", "reading_order", "global_order", "index"):
        value = _numeric_value(node.attributes.get(key))
        if value is not None:
            return (1, value, node.render_id)
    value = _numeric_value(node.render_id)
    if value is not None:
        return (2, value, node.render_id)
    return (3, 0.0, node.render_id)


def _tree_has_role(tree: RenderTreeIR, role: RenderRole) -> bool:
    return any(node.role == role for node in tree.nodes)


def _is_front_matter_render_node(
    node: RenderTreeNode,
    document_nodes: dict[str, DocumentNode],
) -> bool:
    if node.role == RenderRole.ABSTRACT:
        # A double-column abstract is part of the first body flow.  Treating it
        # as full front matter forces a separate ``multicols`` environment and
        # makes LaTeX rebalance abstract/body independently even when no
        # full-width blocker exists between them.
        return _render_node_layout_band(node, document_nodes).mode != "double"
    if node.role in {
        RenderRole.DOCUMENT_TITLE,
        RenderRole.AUTHOR_BLOCK,
        RenderRole.TOC_PLACEHOLDER,
    }:
        return True
    source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
    return bool(source_nodes) and all(
        str(source.metadata.get("layout_layer") or "").casefold() == "metadata_layer"
        or source.node_type in {BlockType.TOC, BlockType.HEADER_FOOTER}
        for source in source_nodes
    )


def _document_node_is_author_block(node: DocumentNode) -> bool:
    layer = str(node.metadata.get("layout_layer") or "").casefold()
    role = str(node.metadata.get("layout_role") or "").casefold()
    return layer == "metadata_layer" and role in {"affiliation", "author", "authors", "date", "email", "correspondence"}


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = NUMERIC_ID_RE.search(value)
        if match:
            return float(match.group(0))
    return None


def _primary_table_node(nodes: list[DocumentNode]) -> DocumentNode:
    for node in nodes:
        if node.metadata.get("table_group_primary") is True:
            return node
    return min(nodes, key=lambda node: node.reading_index)


@dataclass(frozen=True)
class _LayoutBand:
    mode: str
    band_id: int | None = None


def _style_column_mode(style: StyleProfile | None) -> str | None:
    if style is None:
        return None
    options = style.renderer_options or {}
    value = options.get("column_mode") or style.page_layout.get("column_mode")
    return str(value).casefold() if value is not None else None


def _render_node_layout_band(
    node: RenderTreeNode,
    document_nodes: dict[str, DocumentNode],
) -> _LayoutBand:
    source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
    if not source_nodes:
        return _LayoutBand("full", None)
    if node.role in {RenderRole.FOOTNOTE, RenderRole.MARGIN_NOTE, RenderRole.REFERENCES, RenderRole.TOC_PLACEHOLDER}:
        return _LayoutBand("full", _first_layout_band_id(source_nodes))
    if node.role in {RenderRole.FIGURE, RenderRole.TABLE}:
        width_ratio = _render_node_visual_width_ratio(node, document_nodes)
        if width_ratio is not None and width_ratio < 0.62:
            return _LayoutBand("double", _first_layout_band_id(source_nodes))
        return _LayoutBand("full", _first_layout_band_id(source_nodes))
    if any(
        source.node_type in {BlockType.FOOTNOTE, BlockType.MARGIN_NOTE, BlockType.HEADER_FOOTER, BlockType.TOC}
        for source in source_nodes
    ):
        return _LayoutBand("full", _first_layout_band_id(source_nodes))
    if any(source.node_type in {BlockType.FIGURE, BlockType.TABLE} for source in source_nodes):
        width_ratio = _render_node_visual_width_ratio(node, document_nodes)
        if width_ratio is not None and width_ratio < 0.62:
            return _LayoutBand("double", _first_layout_band_id(source_nodes))
        return _LayoutBand("full", _first_layout_band_id(source_nodes))
    band_types = {_layout_band_type(source) for source in source_nodes}
    if "double_column" in band_types and "full_span" not in band_types:
        return _LayoutBand("double", _first_layout_band_id(source_nodes))
    if not any(band_types):
        inferred = _infer_layout_band_mode(source_nodes)
        if inferred is not None:
            return _LayoutBand(inferred, _first_layout_band_id(source_nodes))
    return _LayoutBand("full", _first_layout_band_id(source_nodes))


def _abstract_column_mode(
    node: RenderTreeNode,
    child_ids: list[str],
    render_nodes: dict[str, RenderTreeNode],
    document_nodes: dict[str, DocumentNode],
) -> str:
    child_bands = [
        _render_node_layout_band(render_nodes[child_id], document_nodes)
        for child_id in child_ids
        if child_id in render_nodes
    ]
    if any(band.mode == "double" for band in child_bands):
        return "double"
    own_band = _render_node_layout_band(node, document_nodes)
    if own_band.mode == "double":
        return "double"
    source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
    source_mode = _infer_layout_band_mode(source_nodes)
    if source_mode == "double":
        return "double"
    return "full"


def _layout_band_type(node: DocumentNode) -> str:
    value = node.metadata.get("layout_band_type")
    if not value:
        value = node.features.get("layout_band_type")
    return str(value or "").casefold()


def _first_layout_band_id(nodes: list[DocumentNode]) -> int | None:
    for node in nodes:
        for container in (node.features, node.metadata):
            for key in ("layout_band_global_id", "layout_band_id"):
                value = _numeric_value(container.get(key))
                if value is not None:
                    return int(value)
    return None


def _infer_layout_band_mode(nodes: list[DocumentNode]) -> str | None:
    """Infer full-span vs local double-column blocks when v7 band metadata is absent."""

    boxes = [box for node in nodes for box in node.bboxes]
    if not boxes:
        return None
    x0 = min(box.x0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    page_width = _page_width_for_nodes(nodes)
    width = max(x1 - x0, 0.0)
    center = page_width / 2.0
    margin = 0.05 * page_width
    crosses_center = x0 < center - margin and x1 > center + margin
    if width >= 0.65 * page_width or crosses_center:
        return "full"
    if any(node.node_type in {BlockType.TEXT, BlockType.LIST, BlockType.REFERENCE, BlockType.TITLE} for node in nodes):
        return "double"
    return None


def _bibliography_should_use_double_columns(nodes: list[DocumentNode], style: StyleProfile | None) -> bool:
    """Use detected bibliography columns; otherwise follow the body column mode."""

    band_types = {_layout_band_type(node) for node in nodes}
    if "double_column" in band_types:
        return True
    if "full_span" in band_types and "double_column" not in band_types:
        return False
    inferred = _infer_layout_band_mode(nodes)
    if inferred == "double":
        return True
    if inferred == "full":
        return False
    return _style_column_mode(style) in {"two_column", "mixed"}


def _page_width_for_nodes(nodes: list[DocumentNode]) -> float:
    for node in nodes:
        for container in (node.features, node.metadata):
            value = _numeric_value(container.get("page_width"))
            if value and value > 0:
                return value
    boxes = [box for node in nodes for box in node.bboxes]
    if boxes:
        return max(max(box.x1 for box in boxes), 1000.0)
    return 1000.0


def _render_node_visual_width_ratio(
    node: RenderTreeNode,
    document_nodes: dict[str, DocumentNode],
) -> float | None:
    source_nodes = [document_nodes[node_id] for node_id in node.source_node_ids if node_id in document_nodes]
    if not source_nodes:
        return None
    source_nodes = _visual_group_nodes_for_width(node, source_nodes, document_nodes)
    boxes: list[BBox] = []
    for source in source_nodes:
        box = _node_union_bbox(source)
        if box is not None:
            boxes.append(box)
    if not boxes:
        visual_bbox_keys = ("table_group_bbox", "figure_group_bbox", "image_group_bbox", "crop_bbox", "bbox")
        for source in source_nodes:
            for container in (source.metadata, source.features):
                for key in visual_bbox_keys:
                    box = _bbox_from_value(container.get(key))
                    if box is not None:
                        boxes.append(box)
    if not boxes:
        return None
    union = _union_bboxes(boxes)
    page_width = _page_width_for_nodes(source_nodes)
    if page_width <= 0:
        return None
    return max(union.x1 - union.x0, 0.0) / page_width


def _visual_group_nodes_for_width(
    node: RenderTreeNode,
    source_nodes: list[DocumentNode],
    document_nodes: dict[str, DocumentNode],
) -> list[DocumentNode]:
    if node.role == RenderRole.FIGURE:
        group_ids = {value for source in source_nodes for value in (_figure_group_id(source),) if value}
        if group_ids:
            grouped = [
                candidate
                for candidate in document_nodes.values()
                if candidate.node_type == BlockType.FIGURE and _figure_group_id(candidate) in group_ids
            ]
            if grouped:
                return grouped
    if node.role == RenderRole.TABLE:
        group_ids = {
            str(source.metadata.get("table_group_id"))
            for source in source_nodes
            if source.metadata.get("table_group_id") is not None
        }
        if group_ids:
            grouped = [
                candidate
                for candidate in document_nodes.values()
                if candidate.node_type == BlockType.TABLE and str(candidate.metadata.get("table_group_id")) in group_ids
            ]
            if grouped:
                return grouped
    return source_nodes


def _bbox_from_value(value: object) -> BBox | None:
    if isinstance(value, BBox):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return BBox.from_list(value)
        except (TypeError, ValueError):
            return None
    return None


def _union_bboxes(boxes: list[BBox]) -> BBox:
    return BBox(
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def _primary_visual_node(nodes: list[DocumentNode], node_type: BlockType) -> DocumentNode | None:
    typed = [node for node in nodes if node.node_type == node_type]
    if typed:
        for node in typed:
            if node.metadata.get("figure_group_primary") is True:
                return node
        return min(typed, key=lambda node: node.reading_index)
    return min(nodes, key=lambda node: node.reading_index) if nodes else None


def _figure_group_id(node: DocumentNode) -> str | None:
    for key in ("figure_group_id", "image_group_id"):
        value = node.metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _is_nonprimary_figure_group_member(node: DocumentNode) -> bool:
    group_id = _figure_group_id(node)
    if not group_id:
        return False
    group_size = _numeric_value(node.metadata.get("figure_group_size") or node.metadata.get("image_group_size"))
    if group_size is not None and group_size <= 1:
        return False
    primary = node.metadata.get("figure_group_primary")
    if primary is None:
        primary = node.metadata.get("image_group_primary")
    return primary is False


def _figure_group_members(
    primary: DocumentNode,
    source_nodes: list[DocumentNode],
    document_nodes: dict[str, DocumentNode] | None,
) -> list[DocumentNode]:
    group_id = _figure_group_id(primary)
    candidates: dict[str, DocumentNode] = {}
    if group_id and document_nodes:
        for node in document_nodes.values():
            if node.node_type == BlockType.FIGURE and _figure_group_id(node) == group_id:
                candidates[node.node_id] = node
        member_ids = primary.metadata.get("figure_group_member_node_ids") or primary.metadata.get("image_group_member_node_ids")
        if isinstance(member_ids, list):
            for member_id in member_ids:
                node = document_nodes.get(str(member_id))
                if node is not None and node.node_type == BlockType.FIGURE:
                    candidates[node.node_id] = node
    for node in source_nodes:
        if node.node_type != BlockType.FIGURE:
            continue
        if group_id and _figure_group_id(node) != group_id:
            continue
        candidates[node.node_id] = node
    if not candidates:
        return [primary]
    return sorted(candidates.values(), key=_figure_group_sort_key)


def _figure_caption_from_metadata(primary: DocumentNode | None, source_nodes: list[DocumentNode]) -> str:
    candidates = ([primary] if primary is not None else []) + list(source_nodes)
    for node in candidates:
        if node is None:
            continue
        for key in ("figure_group_caption", "image_group_caption", "figure_caption", "image_caption", "caption"):
            value = node.metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _figure_group_sort_key(node: DocumentNode) -> tuple[int, float, float, int]:
    member_index = node.metadata.get("figure_group_member_index")
    if member_index is None:
        member_index = node.metadata.get("image_group_member_index")
    box = _node_union_bbox(node)
    if member_index is not None:
        try:
            return (0, float(member_index), box.x0 if box else 0.0, node.reading_index)
        except (TypeError, ValueError):
            pass
    return (1, box.y0 if box else 0.0, box.x0 if box else 0.0, node.reading_index)


def _should_render_figure_minipages(nodes: list[DocumentNode]) -> bool:
    if len(nodes) < 2:
        return False
    boxes = [(_node_union_bbox(node), node) for node in nodes]
    boxes = [(box, node) for box, node in boxes if box is not None]
    if len(boxes) < 2:
        return False
    page_idxs = {node.page_idx for _, node in boxes}
    if len(page_idxs) > 1:
        return False
    ordered = sorted(boxes, key=lambda item: (item[0].x0, item[0].y0))
    for (left_box, _), (right_box, _) in zip(ordered, ordered[1:]):
        if _bbox_y_overlap_ratio(left_box, right_box) < 0.18:
            continue
        left_center = (left_box.x0 + left_box.x1) / 2.0
        right_center = (right_box.x0 + right_box.x1) / 2.0
        if abs(right_center - left_center) > 0.05 * max(_page_width_for_nodes(nodes), 1.0):
            return True
    return False


def _bbox_y_overlap_ratio(left: BBox, right: BBox) -> float:
    intersection = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    min_height = max(min(left.y1 - left.y0, right.y1 - right.y0), 1e-6)
    return intersection / min_height


def _source_pdf_for_node(node: DocumentNode) -> str | None:
    value = node.metadata.get("source_pdf")
    if isinstance(value, str) and value:
        return value
    for ref in node.source_refs:
        value = ref.metadata.get("pdf_path") if ref.metadata else None
        if isinstance(value, str) and value:
            return value
    return None


def _render_package(package: str) -> str:
    value = str(package or "").strip()
    if not value:
        return ""
    if value.startswith("\\usepackage"):
        return value
    if "[" in value or "]" in value:
        return rf"\usepackage{{{value}}}"
    return rf"\usepackage{{{value}}}"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _pt(value: float) -> str:
    return f"{max(float(value), 0.0):.2f}pt"


def _heading_spacing_commands(role_styles: dict[str, dict[str, object]]) -> list[str]:
    commands: list[str] = []
    role_to_command = {
        "section": "section",
        "subsection": "subsection",
        "subsubsection": "subsubsection",
    }
    for role, command in role_to_command.items():
        style = role_styles.get(role)
        if not isinstance(style, dict):
            continue
        size = _float_or_none(style.get("font_size"))
        if not size:
            continue
        before = max(size * 0.8, 6.0)
        after = max(size * 0.35, 3.0)
        commands.append(rf"\titlespacing*{{\{command}}}{{0pt}}{{{before:.2f}pt}}{{{after:.2f}pt}}")
    return commands


def _header_footer_profile_enabled(style: StyleProfile) -> bool:
    options = style.renderer_options or {}
    header_footer = options.get("header_footer")
    return bool(isinstance(header_footer, dict) and header_footer.get("render_by_default"))


def _header_footer_commands(header_footer: object) -> list[str]:
    if not isinstance(header_footer, dict) or not header_footer.get("render_by_default"):
        return []
    assignments = _header_footer_assignments(header_footer)
    if not assignments:
        return []
    font_size = _float_or_none(header_footer.get("font_size"))
    lines = [r"\pagestyle{fancy}", r"\fancyhf{}", r"\renewcommand{\headrulewidth}{0pt}", r"\renewcommand{\footrulewidth}{0pt}"]
    if font_size:
        lines.append(rf"\setlength{{\headheight}}{{{max(font_size * 1.5, 12.0):.2f}pt}}")
    lines.extend(assignments)
    lines.append(r"\fancypagestyle{plain}{%")
    lines.append(r"\fancyhf{}")
    lines.extend(assignments)
    lines.append(r"}")
    return lines


def _header_footer_assignments(header_footer: dict[str, object]) -> list[str]:
    assignments: list[str] = []
    for key in ("header", "footer"):
        profile = header_footer.get(key)
        if not isinstance(profile, dict) or not profile.get("enabled"):
            continue
        text = str(profile.get("text") or "").strip()
        if not text:
            continue
        command = _fancy_command(str(profile.get("zone") or key), str(profile.get("slot") or "center"))
        assignments.append(rf"\{command}{{{escape_latex(text)}}}")
    page_number = header_footer.get("page_number")
    if isinstance(page_number, dict) and page_number.get("enabled"):
        command = _fancy_command(str(page_number.get("zone") or "footer"), str(page_number.get("slot") or "center"))
        assignment = rf"\{command}{{\thepage}}"
        # Page numbers are more semantically stable than OCR footer text.  If a
        # repeated footer lands in the same slot, prefer the generated counter.
        prefix = assignment.split("{", 1)[0]
        assignments = [item for item in assignments if not item.startswith(prefix + "{")]
        assignments.append(assignment)
    return assignments


def _fancy_command(zone: str, slot: str) -> str:
    zone_name = "fancyhead" if zone == "header" else "fancyfoot"
    slot_name = {"left": "L", "center": "C", "right": "R"}.get(slot, "C")
    return f"{zone_name}[{slot_name}]"


def _node_baseline_font_size(node: DocumentNode, style: StyleProfile | None) -> float | None:
    for key in ("style_baseline_size", "font_size", "baseline_font_size"):
        value = _float_or_none(node.features.get(key))
        if value:
            return value
    sizes = [float(span.font_size) for span in node.spans if span.font_size is not None and (span.text or "").strip()]
    if sizes:
        return _median_float(sizes)
    if style is not None:
        return _float_or_none((style.renderer_options or {}).get("body_font_size"))
    return None


def _body_font_class(style: StyleProfile | None) -> str | None:
    if style is None:
        return None
    value = (style.renderer_options or {}).get("body_font_class")
    return str(value) if value else None


def _node_union_bbox(node: DocumentNode) -> BBox | None:
    if not node.bboxes:
        return None
    return BBox(
        min(box.x0 for box in node.bboxes),
        min(box.y0 for box in node.bboxes),
        max(box.x1 for box in node.bboxes),
        max(box.y1 for box in node.bboxes),
    )


def _median_float(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _wrap_script(rendered: str, script_role: str) -> str:
    if script_role == "superscript":
        return rf"\raisebox{{0.55ex}}{{\scriptsize {rendered}}}"
    if script_role == "subscript":
        return rf"\raisebox{{-0.35ex}}{{\scriptsize {rendered}}}"
    return rendered


def _fontspec_commands(font_setup: object) -> list[str]:
    if not isinstance(font_setup, dict):
        return []
    commands: list[str] = []
    main_font = _safe_font_name(font_setup.get("main_font"))
    sans_font = _safe_font_name(font_setup.get("sans_font"))
    mono_font = _safe_font_name(font_setup.get("mono_font"))
    if main_font:
        commands.append(rf"\setmainfont{{{main_font}}}")
    if sans_font:
        commands.append(rf"\setsansfont{{{sans_font}}}")
    if mono_font:
        commands.append(rf"\setmonofont{{{mono_font}}}")
    return commands


def _safe_font_name(value: object) -> str | None:
    name = str(value or "").strip()
    if not name:
        return None
    # Font names are data from the PDF.  Keep the renderer conservative by
    # refusing TeX control characters; fallback fonts are ordinary names.
    if re.search(r"[\\{}%#&_$^~]", name):
        return None
    return name
