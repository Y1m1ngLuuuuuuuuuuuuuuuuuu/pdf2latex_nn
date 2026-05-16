from __future__ import annotations

from src.generation.front_matter import render_author_block_original_like, render_document_title_original_like
from src.generation.ir_renderers.context import RenderContext
from src.ir import RenderRole


class FrontMatterRenderer:
    roles = frozenset(
        {
            RenderRole.DOCUMENT_TITLE,
            RenderRole.AUTHOR_BLOCK,
            RenderRole.ABSTRACT,
            RenderRole.TOC_PLACEHOLDER,
        }
    )

    def render(self, context: RenderContext) -> str:
        owner = context.owner
        role = context.node.role
        if role == RenderRole.DOCUMENT_TITLE:
            if owner._use_original_like_front_matter():
                return render_document_title_original_like(context.text, context.source_nodes, owner._active_style)
            if owner.config.include_maketitle:
                return ""
            return owner._render_text_with_citations(context.text)
        if role == RenderRole.AUTHOR_BLOCK:
            if owner._use_original_like_front_matter():
                return render_author_block_original_like(context.text, context.source_nodes, owner._active_style)
            return owner._render_text_with_citations(context.text)
        if role == RenderRole.ABSTRACT:
            return owner._render_abstract(
                context.node,
                context.render_nodes,
                context.document_nodes,
                context.citations,
                context.text,
                depth=context.depth,
            )
        if role == RenderRole.TOC_PLACEHOLDER:
            return r"\tableofcontents"
        return ""
