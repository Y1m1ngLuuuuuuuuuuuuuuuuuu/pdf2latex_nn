from __future__ import annotations

from src.generation.ir_renderers.context import RenderContext
from src.generation.latex_renderer import strip_list_marker
from src.ir import RenderRole


class ListRenderer:
    roles = frozenset({RenderRole.LIST, RenderRole.LIST_ITEM})

    def render(self, context: RenderContext) -> str:
        owner = context.owner
        if context.node.role == RenderRole.LIST:
            return owner._render_list(
                context.node,
                context.render_nodes,
                context.document_nodes,
                context.citations,
                ordered=False,
                depth=context.depth,
            )
        body = owner._render_source_nodes(
            context.source_nodes,
            context.citations,
            strip_leading_list_marker=True,
        ) or owner._render_text_with_citations(strip_list_marker(context.text))
        children = owner._render_children(
            context.node,
            context.render_nodes,
            context.document_nodes,
            context.citations,
            depth=context.depth + 1,
        )
        return "\n".join(part for part in [body, children] if part)
