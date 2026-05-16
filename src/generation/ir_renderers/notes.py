from __future__ import annotations

from src.generation.ir_renderers.context import DocumentNodeRenderContext, RenderContext
from src.ir import BlockType, RenderRole


class NoteRenderer:
    roles = frozenset({RenderRole.FOOTNOTE, RenderRole.MARGIN_NOTE})
    block_types = frozenset({BlockType.FOOTNOTE, BlockType.MARGIN_NOTE})

    def render(self, context: RenderContext) -> str:
        if context.owner._active_notes is not None and context.source_nodes:
            return ""
        kind = "margin_note" if context.node.role == RenderRole.MARGIN_NOTE else "footnote"
        return context.owner._render_standalone_note(kind, context.text)

    def render_document_node(self, context: DocumentNodeRenderContext) -> str:
        return ""
