from __future__ import annotations

from src.generation.ir_renderers.context import DocumentNodeRenderContext, RenderContext
from src.generation.latex_helpers import escape_latex, render_inline_math, render_text_with_inline_latex, strip_list_marker
from src.ir import BlockType, RenderRole


class TextRenderer:
    roles = frozenset({RenderRole.ROOT, RenderRole.CAPTION, RenderRole.RAW_LATEX})
    block_types = frozenset({BlockType.TEXT, BlockType.LIST, BlockType.TITLE, BlockType.REFERENCE})

    def render(self, context: RenderContext) -> str:
        owner = context.owner
        if context.node.role == RenderRole.ROOT:
            return owner._render_children(context.node, context.render_nodes, context.document_nodes, context.citations, depth=context.depth)
        if context.node.role == RenderRole.CAPTION:
            return owner._render_text_with_citations(context.text)
        if context.node.role == RenderRole.RAW_LATEX:
            return context.text
        return self.render_fallback(context)

    def render_fallback(self, context: RenderContext) -> str:
        owner = context.owner
        body = owner._render_source_nodes(context.source_nodes, context.citations) if context.source_nodes else owner._render_body_text(context.text)
        children = owner._render_children(
            context.node,
            context.render_nodes,
            context.document_nodes,
            context.citations,
            depth=context.depth + 1,
        )
        return "\n\n".join(part for part in [body, children] if part)

    def render_document_node(self, context: DocumentNodeRenderContext) -> str:
        owner = context.owner
        text = strip_list_marker(context.text) if context.strip_leading_list_marker else context.text
        if context.node.spans and owner._should_render_spans_for_node(context.node, context.text):
            return owner._render_spans(context.node, context.citations, strip_leading_list_marker=context.strip_leading_list_marker)
        if context.citations and context.node.node_id in context.citations.text_by_node_id:
            return owner._render_body_text(text, node=context.node)
        return owner._render_body_text(text, node=context.node)


def render_paragraph_with_inline_math_attachments(
    text_before: str,
    inline_formulae: list[str],
    text_after: str = "",
) -> str:
    """Render a paragraph span with attached inline formula fragments.

    The helper is intentionally standalone so Phase 0 tests and future
    RenderTreeIR materialization can use it without changing the production
    renderer's default path.
    """

    parts: list[str] = []
    before = render_text_with_inline_latex(text_before, strip=False)
    after = render_text_with_inline_latex(text_after, strip=False)
    if before:
        parts.append(before.rstrip())
    for formula in inline_formulae:
        rendered = render_inline_math(formula)
        if rendered:
            parts.append(rendered)
    if after:
        parts.append(after.lstrip())
    return " ".join(part for part in parts if part).strip()


def render_theorem_proof_context(
    label_text: str,
    body_text: str = "",
    *,
    fallback_plain: bool = False,
) -> str:
    """Render theorem/proof-like prose without requiring theorem packages."""

    label = str(label_text or "").strip()
    body = str(body_text or "").strip()
    if not label:
        return render_text_with_inline_latex(body)
    if fallback_plain:
        joined = " ".join(part for part in [label, body] if part)
        return render_text_with_inline_latex(joined)
    rendered_label = escape_latex(label)
    rendered_body = render_text_with_inline_latex(body)
    suffix = f" {rendered_body}" if rendered_body else ""
    return rf"\noindent\textbf{{{rendered_label}}}{suffix}"
