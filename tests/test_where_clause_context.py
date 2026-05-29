from src.generation.ir_renderers.math import render_display_math_context
from src.reasoning.formula_context_group import classify_formula_context


def test_where_clause_is_formula_context() -> None:
    family, evidence = classify_formula_context("where x denotes the input and y = f(x).", local_formula_context=True)

    assert family == "WHERE_CLAUSE_CONTEXT"
    assert evidence.starts_where_clause
    assert evidence.confidence_tier == "high"


def test_with_sentence_is_not_where_context() -> None:
    family, evidence = classify_formula_context("With the rapid advancement of large language models, tokens changed.", local_formula_context=True)

    assert family == "ORDINARY_BODY_REORDER"
    assert not evidence.starts_where_clause


def test_display_math_context_renderer_keeps_math_block_separate() -> None:
    rendered = render_display_math_context(
        text_before="The loss is defined as",
        display_math="L = x + y",
        text_after="where x denotes the input.",
    )

    assert "The loss is defined as" in rendered
    assert "\\[\nL = x + y\n\\]" in rendered
    assert "where x denotes the input." in rendered
    assert "L = x + y where" not in rendered
