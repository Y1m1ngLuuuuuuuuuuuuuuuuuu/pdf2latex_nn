from __future__ import annotations

from src.generation.ir_renderers.context import DocumentNodeRenderContext, RenderContext
from src.ir import BlockType, RenderRole


class TableRenderer:
    roles = frozenset({RenderRole.TABLE})
    block_types = frozenset({BlockType.TABLE})

    def render(self, context: RenderContext) -> str:
        return context.owner._render_table(context.source_nodes, context.text, render_node=context.node)

    def render_document_node(self, context: DocumentNodeRenderContext) -> str:
        return context.owner._render_table([context.node], context.text)
