"""Renderer entrypoint for the decoupled backend IR.

This is the first original-like generator surface.  It consumes the stable
interfaces instead of TreeDecoder internals:

DocumentIR + RenderTreeIR + StyleProfile (+ CitationResolution) -> LaTeX.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.generation.citations import CitationResolution, replace_citation_markers, strip_reference_label
from src.generation.font_resolver import resolve_pdf_font
from src.generation.front_matter import render_author_block_original_like, render_document_title_original_like
from src.generation.latex_renderer import (
    escape_latex,
    render_algorithm_block,
    render_equation,
    render_figure_block,
    render_inline_math,
    render_table_placeholder,
    render_text_with_inline_latex,
    safe_verbatim_text,
    strip_list_marker,
)
from src.ir import BBox, BlockType, DocumentIR, DocumentNode, RenderRole, RenderTreeIR, RenderTreeNode, StyleProfile, StyleSpan


CITE_COMMAND_RE = re.compile(r"\\(?:cite|citep|citet|citealp|citeauthor|citeyear|ref|autoref|cref)\*?(?:\[[^\]]*\])?\{[^{}]+\}")
LIST_MARKER_RE = re.compile(r"^\s*(?:[\u2022\u25E6\u25CB\u25AA\-\*]|\d+\.|[a-zA-Z]\.)\s+")
ORDERED_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\.|[a-zA-Z]\.)\s+")
NUMERIC_ID_RE = re.compile(r"\d+")
REQUIRED_RENDER_PACKAGES = [
    "amsmath",
    "amssymb",
    "graphicx",
    "float",
    "booktabs",
    "hyperref",
    "geometry",
    "enumitem",
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

    def render(
        self,
        document: DocumentIR,
        tree: RenderTreeIR,
        style: StyleProfile,
        citations: CitationResolution | None = None,
    ) -> str:
        previous_style = self._active_style
        previous_notes = self._active_notes
        self._active_style = style
        self._active_notes = _NoteContext.from_document(document)
        try:
            return self._render_with_active_style(document, tree, style, citations)
        finally:
            self._active_style = previous_style
            self._active_notes = previous_notes

    def _render_with_active_style(
        self,
        document: DocumentIR,
        tree: RenderTreeIR,
        style: StyleProfile,
        citations: CitationResolution | None = None,
    ) -> str:
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

    def _render_preamble(self, style: StyleProfile, citations: CitationResolution | None = None) -> list[str]:
        options = f"[{','.join(style.documentclass_options)}]" if style.documentclass_options else ""
        lines = [rf"\documentclass{options}{{{style.documentclass}}}"]
        packages = [*style.packages, *REQUIRED_RENDER_PACKAGES]
        if citations is not None and citations.citation_style == "numeric":
            packages.insert(0, "cite")
        if _style_column_mode(style) == "mixed":
            packages.append("multicol")
        if self.config.enable_fontspec:
            packages.append("fontspec")
        if self.config.render_header_footer and _header_footer_profile_enabled(style):
            packages.append("fancyhdr")
        for package in _dedupe_preserve_order(packages):
            lines.append(_render_package(package))
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
            children = self._render_children(node, render_nodes, document_nodes, citations, depth=depth + 1)
            body = "\n\n".join(part for part in [render_text_with_citations(text), children] if part)
            return "\\begin{abstract}\n" + body + "\n\\end{abstract}" if body else ""
        if role in {RenderRole.SECTION, RenderRole.SUBSECTION, RenderRole.SUBSUBSECTION}:
            command = {
                RenderRole.SECTION: "section",
                RenderRole.SUBSECTION: "subsection",
                RenderRole.SUBSUBSECTION: "subsubsection",
            }[role]
            heading = rf"\{command}{{{render_text_with_citations(text)}}}" if text else ""
            children = self._render_children(node, render_nodes, document_nodes, citations, depth=depth + 1)
            return "\n\n".join(part for part in [heading, children] if part)
        if role in {RenderRole.DISPLAY_EQUATION}:
            body = render_equation(text)
            children = self._render_children(node, render_nodes, document_nodes, citations, depth=depth + 1)
            return "\n\n".join(part for part in [body, children] if part)
        if role in {RenderRole.INLINE_MATH}:
            return render_inline_math(text)
        if role in {RenderRole.TABLE}:
            return self._render_table(source_nodes, text)
        if role in {RenderRole.FIGURE}:
            return self._render_figure(source_nodes, text)
        if role in {RenderRole.ALGORITHM}:
            return render_algorithm_block(text)
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
            return self._render_bibliography(citations, source_nodes, node, render_nodes, document_nodes)
        if role in {RenderRole.LIST}:
            return self._render_list(node, render_nodes, document_nodes, citations, ordered=False, depth=depth)
        if role in {RenderRole.LIST_ITEM}:
            body = self._render_source_nodes(source_nodes, citations, strip_leading_list_marker=True) or render_text_with_citations(strip_list_marker(text))
            children = self._render_children(node, render_nodes, document_nodes, citations, depth=depth + 1)
            return "\n".join(part for part in [body, children] if part)
        if role in {RenderRole.RAW_LATEX}:
            return text

        body = self._render_source_nodes(source_nodes, citations) if source_nodes else render_text_with_citations(text)
        children = self._render_children(node, render_nodes, document_nodes, citations, depth=depth + 1)
        return "\n\n".join(part for part in [body, children] if part)

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
        if self._should_render_mixed_columns(child_ids, render_nodes, document_nodes):
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
        while index < len(child_ids):
            child = render_nodes[child_ids[index]]
            if self._is_reference_render_node(child, document_nodes):
                run: list[RenderTreeNode] = []
                while index < len(child_ids):
                    candidate = render_nodes[child_ids[index]]
                    if not self._is_reference_render_node(candidate, document_nodes):
                        break
                    run.append(candidate)
                    index += 1
                parts.append(self._render_reference_run(run, render_nodes, document_nodes, citations))
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
        while index < len(child_ids):
            child = render_nodes[child_ids[index]]
            band = _render_node_layout_band(child, document_nodes)
            if band.mode != "double":
                rendered = self._render_children_standard(
                    [child_ids[index]],
                    render_nodes,
                    document_nodes,
                    citations,
                    depth=depth,
                )
                if rendered:
                    parts.append(rendered)
                index += 1
                continue

            run = [child_ids[index]]
            index += 1
            while index < len(child_ids):
                candidate = render_nodes[child_ids[index]]
                candidate_band = _render_node_layout_band(candidate, document_nodes)
                if candidate_band.mode != "double":
                    break
                if band.band_id is not None and candidate_band.band_id is not None and candidate_band.band_id != band.band_id:
                    break
                run.append(child_ids[index])
                index += 1

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
        return "\n\n".join(part for part in parts if part)

    def _should_render_mixed_columns(
        self,
        child_ids: list[str],
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
    ) -> bool:
        if self._mixed_column_stack > 0 or _style_column_mode(self._active_style) != "mixed":
            return False
        return any(
            _render_node_layout_band(render_nodes[child_id], document_nodes).mode == "double"
            for child_id in child_ids
            if child_id in render_nodes
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
        return self._render_bibliography(citations, source_nodes, nodes[0], render_nodes, document_nodes)

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
        return " ".join(part for part in rendered if part).strip()

    def _render_document_node_with_notes(
        self,
        node: DocumentNode,
        citations: CitationResolution | None,
        *,
        strip_leading_list_marker: bool = False,
    ) -> str:
        rendered = self._render_document_node(node, citations, strip_leading_list_marker=strip_leading_list_marker)
        note_commands = self._render_notes_for_source_node(node.node_id)
        if note_commands:
            return rendered + note_commands if rendered else note_commands
        return rendered

    def _render_document_node(
        self,
        node: DocumentNode,
        citations: CitationResolution | None,
        *,
        strip_leading_list_marker: bool = False,
    ) -> str:
        text = citations.text_by_node_id.get(node.node_id, node.text) if citations else node.text
        if strip_leading_list_marker:
            text = strip_list_marker(text)
        if node.node_type in {BlockType.FOOTNOTE, BlockType.MARGIN_NOTE}:
            return ""
        if node.node_type == BlockType.EQUATION:
            return render_equation(text)
        if node.node_type == BlockType.INLINE_MATH:
            return render_inline_math(text)
        if node.node_type == BlockType.TABLE:
            return self._render_table([node], text)
        if node.node_type == BlockType.FIGURE:
            return self._render_figure([node], text)
        if node.node_type == BlockType.ALGORITHM:
            return render_algorithm_block(text)
        if node.node_type == BlockType.CODE:
            return "\\begin{verbatim}\n" + safe_verbatim_text(text.strip()) + "\n\\end{verbatim}" if text else ""
        if node.spans:
            return self._render_spans(node, citations, strip_leading_list_marker=strip_leading_list_marker)
        if citations and node.node_id in citations.text_by_node_id:
            return render_text_with_citations(text)
        return render_text_with_inline_latex(text)

    def _render_notes_for_source_node(self, node_id: str) -> str:
        if self._active_notes is None:
            return ""
        return "".join(self._render_note(note) for note in self._active_notes.consume_for_anchor(node_id))

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
        marker_pending = strip_leading_list_marker
        node_baseline = _node_baseline_font_size(node, self._active_style)
        body_font_class = _body_font_class(self._active_style)
        for span in node.spans:
            text = span.text or ""
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
            if label_to_key:
                text, _occurrences, _unresolved = replace_citation_markers(
                    text,
                    label_to_key,
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
                if script_role:
                    rendered = _wrap_script(rendered, script_role)
                else:
                    rendered = self._apply_span_font_size(rendered, span, node_baseline)
            parts.append(rendered)
        return "".join(parts).strip()

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
            return render_table_placeholder(
                document_node_record(primary),
                text or primary.text,
                source_pdf=primary.metadata.get("source_pdf") if self.config.table_asset_output_dir else None,
                asset_output_dir=self.config.table_asset_output_dir,
                asset_latex_prefix=self.config.table_asset_latex_prefix,
            )
        return render_table_placeholder({"type": "table", "text": text}, text)

    def _render_figure(self, source_nodes: list[DocumentNode], text: str) -> str:
        caption = text or "Figure"
        primary = _primary_visual_node(source_nodes, BlockType.FIGURE)
        if source_nodes:
            for node in source_nodes:
                value = node.metadata.get("figure_caption") or node.metadata.get("caption")
                if isinstance(value, str) and value.strip():
                    caption = value.strip()
                    break
        record = document_node_record(primary) if primary is not None else {"type": "figure", "text": caption}
        return render_figure_block(
            record,
            caption,
            source_pdf=_source_pdf_for_node(primary) if primary is not None else None,
            asset_output_dir=self.config.figure_asset_output_dir or self.config.table_asset_output_dir,
            asset_latex_prefix=self.config.figure_asset_latex_prefix,
            rendered_caption=render_text_with_citations(caption),
        )

    def _render_bibliography(
        self,
        citations: CitationResolution | None,
        source_nodes: list[DocumentNode],
        node: RenderTreeNode,
        render_nodes: dict[str, RenderTreeNode],
        document_nodes: dict[str, DocumentNode],
    ) -> str:
        if citations and citations.entries:
            lines = [r"\begin{thebibliography}{99}"]
            for entry in citations.entries:
                optional = f"[{render_text_with_inline_latex(entry.display_label)}]" if entry.display_label else ""
                lines.append(rf"\bibitem{optional}{{{entry.key}}} {render_text_with_inline_latex(entry.text)}")
            lines.append(r"\end{thebibliography}")
            return "\n".join(lines)
        if source_nodes:
            lines = [r"\begin{thebibliography}{99}"]
            for index, source in enumerate(source_nodes, start=1):
                lines.append(rf"\bibitem{{ref_{index}}} {render_text_with_inline_latex(strip_reference_label(source.text))}")
            lines.append(r"\end{thebibliography}")
            return "\n".join(lines)
        return self._render_children(node, render_nodes, document_nodes, citations, depth=0)

    def _source_text(self, nodes: list[DocumentNode], citations: CitationResolution | None) -> str:
        return " ".join(citations.text_by_node_id.get(node.node_id, node.text) if citations else node.text for node in nodes).strip()

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

    value = str(text or "")
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
        by_anchor: dict[str, list[_ResolvedNote]] = {}
        unanchored: list[_ResolvedNote] = []
        for note_node in sorted(document.nodes, key=lambda item: item.reading_index):
            if note_node.node_type not in {BlockType.FOOTNOTE, BlockType.MARGIN_NOTE}:
                continue
            text, marker = _strip_note_marker(note_node.text, note_node.metadata)
            if not text:
                continue
            kind = "margin_note" if note_node.node_type == BlockType.MARGIN_NOTE else "footnote"
            note = _ResolvedNote(node=note_node, kind=kind, text=text, marker=marker)
            anchor_id = _explicit_note_anchor(note_node) or _nearest_note_anchor(note_node, body_nodes)
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


NOTE_MARKER_RE = re.compile(
    r"^\s*(?:(?:\[(?P<bracket>[0-9A-Za-z*†‡§¶]+)\])|(?:\((?P<paren>[0-9A-Za-z*†‡§¶]+)\))|(?P<bare>[0-9]{1,3}|[*†‡§¶]))[\s:.\-]*"
)


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


def _explicit_note_anchor(note_node: DocumentNode) -> str | None:
    value = note_node.metadata.get("footnote_anchor") or note_node.metadata.get("anchor_node_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


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


def _sorted_child_ids(
    child_ids: list[str],
    render_nodes: dict[str, RenderTreeNode],
    document_nodes: dict[str, DocumentNode],
) -> list[str]:
    return sorted(
        [child_id for child_id in child_ids if child_id in render_nodes],
        key=lambda child_id: _render_node_order_key(render_nodes[child_id], document_nodes),
    )


def _render_node_order_key(
    node: RenderTreeNode,
    document_nodes: dict[str, DocumentNode],
) -> tuple[int, float, str]:
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
    if node.role in {RenderRole.FIGURE, RenderRole.TABLE, RenderRole.FOOTNOTE, RenderRole.MARGIN_NOTE, RenderRole.REFERENCES, RenderRole.TOC_PLACEHOLDER}:
        return _LayoutBand("full", _first_layout_band_id(source_nodes))
    if any(
        source.node_type in {BlockType.FIGURE, BlockType.TABLE, BlockType.FOOTNOTE, BlockType.MARGIN_NOTE, BlockType.HEADER_FOOTER, BlockType.TOC}
        for source in source_nodes
    ):
        return _LayoutBand("full", _first_layout_band_id(source_nodes))
    band_types = {_layout_band_type(source) for source in source_nodes}
    if "double_column" in band_types and "full_span" not in band_types:
        return _LayoutBand("double", _first_layout_band_id(source_nodes))
    if not any(band_types):
        inferred = _infer_layout_band_mode(source_nodes)
        if inferred is not None:
            return _LayoutBand(inferred, _first_layout_band_id(source_nodes))
    return _LayoutBand("full", _first_layout_band_id(source_nodes))


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


def _primary_visual_node(nodes: list[DocumentNode], node_type: BlockType) -> DocumentNode | None:
    typed = [node for node in nodes if node.node_type == node_type]
    if typed:
        return min(typed, key=lambda node: node.reading_index)
    return min(nodes, key=lambda node: node.reading_index) if nodes else None


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
