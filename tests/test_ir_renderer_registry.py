from __future__ import annotations

from src.generation.ir_renderers import build_default_registry
from src.ir import BlockType, RenderRole


def test_default_ir_renderer_registry_covers_specialized_roles_and_block_types():
    registry = build_default_registry()

    assert {
        RenderRole.DOCUMENT_TITLE,
        RenderRole.AUTHOR_BLOCK,
        RenderRole.ABSTRACT,
        RenderRole.SECTION,
        RenderRole.SUBSECTION,
        RenderRole.SUBSUBSECTION,
        RenderRole.DISPLAY_EQUATION,
        RenderRole.INLINE_MATH,
        RenderRole.FIGURE,
        RenderRole.TABLE,
        RenderRole.ALGORITHM,
        RenderRole.CODE,
        RenderRole.FOOTNOTE,
        RenderRole.MARGIN_NOTE,
        RenderRole.REFERENCES,
        RenderRole.LIST,
        RenderRole.LIST_ITEM,
    }.issubset(registry.role_coverage())

    assert {
        BlockType.TEXT,
        BlockType.TITLE,
        BlockType.EQUATION,
        BlockType.INLINE_MATH,
        BlockType.FIGURE,
        BlockType.TABLE,
        BlockType.ALGORITHM,
        BlockType.CODE,
        BlockType.FOOTNOTE,
        BlockType.MARGIN_NOTE,
        BlockType.LIST,
        BlockType.REFERENCE,
    }.issubset(registry.block_type_coverage())
