from src.reasoning.formula_context_group import (
    build_formula_context_groups,
    classify_formula_context,
    classify_record_formula_context,
    should_exclude_from_ordinary_visible_prose_evidence,
)


def test_inline_equation_span_becomes_high_confidence_inline_attachment():
    context_type, evidence = classify_formula_context(
        "The loss is $L$.",
        formula_metadata={
            "is_inline_math": True,
            "mineru_span_type": "inline_equation",
            "formula_confidence": "strong_span_inline",
            "formula_context_role": "inline_attachment",
        },
    )

    assert context_type == "INLINE_MATH_ATTACHMENT"
    assert evidence.confidence_tier == "high"
    assert evidence.evidence_source == "mineru_span"
    assert evidence.has_mineru_formula_evidence is True


def test_interline_equation_span_becomes_display_math_context():
    context_type, evidence = classify_formula_context(
        "E = mc^2",
        formula_metadata={
            "is_display_math": True,
            "mineru_span_type": "interline_equation",
            "formula_confidence": "strong_span_interline",
            "formula_context_role": "display_math",
        },
    )

    assert context_type == "DISPLAY_MATH_CONTEXT"
    assert evidence.confidence_tier == "high"
    assert should_exclude_from_ordinary_visible_prose_evidence(context_type, evidence)


def test_content_list_equation_latex_becomes_display_formula_evidence():
    context_type, evidence = classify_formula_context(
        "\\frac{a}{b}",
        formula_metadata={
            "raw_formula_type": "equation",
            "text_format": "latex",
            "formula_confidence": "strong_content_equation_latex",
        },
    )

    assert context_type == "DISPLAY_MATH_CONTEXT"
    assert evidence.evidence_source in {"content_list_equation", "mixed"}


def test_where_clause_near_display_formula_becomes_formula_context_group():
    records = [
        {
            "node_id": "eq1",
            "text": "x = y + z",
            "metadata": {
                "is_display_math": True,
                "mineru_span_type": "interline_equation",
                "formula_confidence": "strong_span_interline",
            },
        },
        {"node_id": "w1", "text": "where x denotes the input variable."},
    ]

    groups = build_formula_context_groups(records)

    assert any(group.context_type == "DISPLAY_MATH_CONTEXT" for group in groups)
    assert any(group.context_type == "WHERE_CLAUSE_CONTEXT" for group in groups)


def test_where_clause_without_formula_evidence_remains_diagnostic():
    context_type, evidence = classify_formula_context(
        "where x denotes the input variable.",
        local_formula_context=True,
    )

    assert context_type == "WHERE_CLAUSE_CONTEXT"
    assert evidence.confidence_tier == "diagnostic_only"
    assert not should_exclude_from_ordinary_visible_prose_evidence(context_type, evidence)


def test_theorem_near_formula_evidence_becomes_context_group():
    records = [
        {
            "node_id": "eq1",
            "text": "p(x) = q(x)",
            "metadata": {
                "is_display_math": True,
                "mineru_span_type": "interline_equation",
                "formula_confidence": "strong_span_interline",
            },
        },
        {"node_id": "thm1", "text": "Theorem 1. The estimator is consistent."},
    ]

    groups = build_formula_context_groups(records)

    assert any(group.context_type == "DISPLAY_MATH_CONTEXT" for group in groups)
    assert any(group.context_type == "THEOREM_PROOF_CONTEXT" for group in groups)


def test_ordinary_prose_with_variables_is_not_formula_context():
    context_type, evidence = classify_formula_context("Let x be the input and y be the output.")

    assert context_type == "ORDINARY_BODY_REORDER"
    assert evidence.confidence_tier in {"low", "medium"}


def test_regex_only_formula_like_text_remains_diagnostic():
    context_type, evidence = classify_formula_context("$x+y$", local_formula_context=False)

    assert context_type == "INLINE_MATH_ATTACHMENT"
    assert evidence.evidence_source == "regex_only"
    assert evidence.confidence_tier == "diagnostic_only"


def test_record_metadata_is_consumed_without_renderer_or_graph_changes():
    record = {
        "node_id": "n1",
        "text": "a+b",
        "metadata": {
            "is_inline_math": True,
            "mineru_span_type": "inline_equation",
            "formula_confidence": "strong_span_inline",
        },
    }

    context_type, evidence = classify_record_formula_context(record)

    assert context_type == "INLINE_MATH_ATTACHMENT"
    assert evidence.confidence_tier == "high"
