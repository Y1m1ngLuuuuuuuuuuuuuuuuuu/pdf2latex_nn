from __future__ import annotations

from src.generation.ir_renderers.context import RenderContext
from src.ir import RenderRole


class ReferenceRenderer:
    roles = frozenset({RenderRole.REFERENCES})

    def render(self, context: RenderContext) -> str:
        return context.owner._render_bibliography_with_tail(
            context.citations,
            context.source_nodes,
            context.node,
            context.render_nodes,
            context.document_nodes,
            depth=context.depth,
        )
