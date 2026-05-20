from __future__ import annotations

from src.generation.ir_renderers.context import DocumentNodeRenderContext, RenderContext
from src.generation.latex_helpers import strip_list_marker
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
