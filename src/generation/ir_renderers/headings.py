from __future__ import annotations

from src.generation.ir_renderers.context import RenderContext
from src.ir import RenderRole


class HeadingRenderer:
    roles = frozenset({RenderRole.SECTION, RenderRole.SUBSECTION, RenderRole.SUBSUBSECTION})

    def render(self, context: RenderContext) -> str:
        owner = context.owner
        command = {
            RenderRole.SECTION: "section",
            RenderRole.SUBSECTION: "subsection",
            RenderRole.SUBSUBSECTION: "subsubsection",
        }[context.node.role]
        run_in_split = owner._split_run_in_heading_source(context.source_nodes)
        heading_source_text = run_in_split[0] if run_in_split is not None else context.text
        heading_text = (
            owner._clean_appendix_heading_text(heading_source_text)
            if context.node.attributes.get("appendix_heading")
            else owner._clean_heading_text(heading_source_text)
        )
        heading = rf"\{command}{{{owner._render_text_with_citations(heading_text)}}}" if heading_text else ""
        run_in_body = ""
        if run_in_split is not None:
            run_in_body = owner._render_body_text(run_in_split[1], node=context.source_nodes[0] if context.source_nodes else None)
        children = owner._render_children(
            context.node,
            context.render_nodes,
            context.document_nodes,
            context.citations,
            depth=context.depth + 1,
        )
        return "\n\n".join(part for part in [heading, run_in_body, children] if part)
