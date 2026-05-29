from src.generation.ir_renderers.text import render_paragraph_with_inline_math_attachments
from src.reasoning.formula_context_group import classify_formula_context
from src.reasoning.paragraph_context_group import build_paragraph_context_groups


def test_inline_formula_fragment_attaches_to_paragraph_span() -> None:
    family, evidence = classify_formula_context(r"$x_i$", raw_text=r"$x_i$")

    assert family == "INLINE_MATH_ATTACHMENT"
    assert evidence.inline_math_marker


def test_inline_attachment_renderer_is_compile_safe() -> None:
    rendered = render_paragraph_with_inline_math_attachments("Let", ["x_i"], "denote the input.")

    assert rendered == "Let $x_i$ denote the input."


def test_inline_context_group_preserves_fragment_without_graph_fields() -> None:
    groups = build_paragraph_context_groups(
        [
            {"node_id": "p1", "text": "Let"},
            {"node_id": "m1", "text": r"$x_i$"},
            {"node_id": "p2", "text": "denote the input."},
        ]
    )

    inline_groups = [group for group in groups if group.context_kind == "inline_math_attachment"]
    assert len(inline_groups) == 1
    assert inline_groups[0].context_ids == ["m1"]
    assert "graph" not in inline_groups[0].to_dict()
