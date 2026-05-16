from __future__ import annotations

from dataclasses import dataclass

from src.generation.ir_renderers.base import BlockRenderer, RoleRenderer
from src.generation.ir_renderers.context import DocumentNodeRenderContext, RenderContext
from src.generation.ir_renderers.figures import FigureRenderer
from src.generation.ir_renderers.front_matter import FrontMatterRenderer
from src.generation.ir_renderers.headings import HeadingRenderer
from src.generation.ir_renderers.lists import ListRenderer
from src.generation.ir_renderers.math import AlgorithmCodeRenderer, MathRenderer
from src.generation.ir_renderers.notes import NoteRenderer
from src.generation.ir_renderers.references import ReferenceRenderer
from src.generation.ir_renderers.tables import TableRenderer
from src.generation.ir_renderers.text import TextRenderer
from src.ir import BlockType, RenderRole


@dataclass(frozen=True)
class IRRendererRegistry:
    """Dispatches RenderRole/BlockType-specific rendering to small renderers."""

    role_renderers: tuple[RoleRenderer, ...]
    block_renderers: tuple[BlockRenderer, ...]
    fallback_renderer: TextRenderer

    def render_tree_node(self, context: RenderContext) -> str:
        for renderer in self.role_renderers:
            if context.node.role in renderer.roles:
                return renderer.render(context)
        return self.fallback_renderer.render_fallback(context)

    def render_document_node(self, context: DocumentNodeRenderContext) -> str:
        for renderer in self.block_renderers:
            if context.node.node_type in renderer.block_types:
                return renderer.render_document_node(context)
        return self.fallback_renderer.render_document_node(context)

    def role_coverage(self) -> set[RenderRole]:
        covered: set[RenderRole] = set()
        for renderer in self.role_renderers:
            covered.update(renderer.roles)
        return covered

    def block_type_coverage(self) -> set[BlockType]:
        covered: set[BlockType] = set()
        for renderer in self.block_renderers:
            covered.update(renderer.block_types)
        return covered


def build_default_registry() -> IRRendererRegistry:
    text = TextRenderer()
    math = MathRenderer()
    algorithm_code = AlgorithmCodeRenderer()
    table = TableRenderer()
    figure = FigureRenderer()
    note = NoteRenderer()
    return IRRendererRegistry(
        role_renderers=(
            text,
            FrontMatterRenderer(),
            HeadingRenderer(),
            math,
            table,
            figure,
            algorithm_code,
            note,
            ReferenceRenderer(),
            ListRenderer(),
        ),
        block_renderers=(
            math,
            table,
            figure,
            algorithm_code,
            note,
            text,
        ),
        fallback_renderer=text,
    )
