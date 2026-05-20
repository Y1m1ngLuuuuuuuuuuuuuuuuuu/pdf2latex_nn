from __future__ import annotations

from src.generation.ir_renderers.context import DocumentNodeRenderContext, RenderContext
from src.generation.latex_helpers import render_algorithm_block, render_equation, render_inline_math, safe_verbatim_text
from src.ir import BlockType, RenderRole


class MathRenderer:
    roles = frozenset({RenderRole.DISPLAY_EQUATION, RenderRole.INLINE_MATH})
    block_types = frozenset({BlockType.EQUATION, BlockType.INLINE_MATH})

    def render(self, context: RenderContext) -> str:
        owner = context.owner
        if context.node.role == RenderRole.DISPLAY_EQUATION:
            body = render_equation(
                context.text,
                label=owner._cross_ref_label_for_render_node(context.node, context.document_nodes, "equation"),
            )
            children = owner._render_children(
                context.node,
                context.render_nodes,
                context.document_nodes,
                context.citations,
                depth=context.depth + 1,
            )
            return "\n\n".join(part for part in [body, children] if part)
        return render_inline_math(context.text)

    def render_document_node(self, context: DocumentNodeRenderContext) -> str:
        if context.node.node_type == BlockType.EQUATION:
            return render_equation(context.text, label=context.owner._cross_ref_label_for_document_node(context.node, "equation"))
        return render_inline_math(context.text)


class AlgorithmCodeRenderer:
    roles = frozenset({RenderRole.ALGORITHM, RenderRole.CODE})
    block_types = frozenset({BlockType.ALGORITHM, BlockType.CODE})

    def render(self, context: RenderContext) -> str:
        if context.node.role == RenderRole.ALGORITHM:
            render_algorithm = getattr(context.owner, "_render_algorithm", None)
            if callable(render_algorithm):
                return render_algorithm(
                    context.source_nodes,
                    context.text,
                    label=context.owner._cross_ref_label_for_render_node(context.node, context.document_nodes, "algorithm"),
                )
            return render_algorithm_block(context.text)
        return "\\begin{verbatim}\n" + safe_verbatim_text(context.text.strip()) + "\n\\end{verbatim}" if context.text else ""

    def render_document_node(self, context: DocumentNodeRenderContext) -> str:
        if context.node.node_type == BlockType.ALGORITHM:
            render_algorithm = getattr(context.owner, "_render_algorithm", None)
            if callable(render_algorithm):
                return render_algorithm(
                    [context.node],
                    context.text,
                    label=context.owner._cross_ref_label_for_document_node(context.node, "algorithm"),
                )
            return render_algorithm_block(context.text)
        return "\\begin{verbatim}\n" + safe_verbatim_text(context.text.strip()) + "\n\\end{verbatim}" if context.text else ""
