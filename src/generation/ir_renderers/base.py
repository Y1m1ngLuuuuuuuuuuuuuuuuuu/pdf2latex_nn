from __future__ import annotations

from typing import Protocol

from src.generation.ir_renderers.context import DocumentNodeRenderContext, RenderContext
from src.ir import BlockType, RenderRole


class RoleRenderer(Protocol):
    roles: frozenset[RenderRole]

    def render(self, context: RenderContext) -> str:
        ...


class BlockRenderer(Protocol):
    block_types: frozenset[BlockType]

    def render_document_node(self, context: DocumentNodeRenderContext) -> str:
        ...
