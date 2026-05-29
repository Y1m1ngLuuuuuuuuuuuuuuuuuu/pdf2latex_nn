from __future__ import annotations

from src.generation.ir_renderers.context import DocumentNodeRenderContext, RenderContext
from src.ir import BlockType, RenderRole


class FigureRenderer:
    roles = frozenset({RenderRole.FIGURE})
    block_types = frozenset({BlockType.FIGURE})

    def render(self, context: RenderContext) -> str:
        return context.owner._render_figure(
            context.source_nodes,
            context.text,
            context.citations,
            document_nodes=context.document_nodes,
            render_node=context.node,
        )

    def render_document_node(self, context: DocumentNodeRenderContext) -> str:
        return context.owner._render_figure([context.node], context.text, context.citations)
