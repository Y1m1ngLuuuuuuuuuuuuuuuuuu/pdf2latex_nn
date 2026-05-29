from src.reasoning.formula_context_group import (
    build_formula_context_groups,
    classify_formula_context,
    should_exclude_from_ordinary_visible_prose,
)


def test_display_equation_is_not_ordinary_merge_context() -> None:
    family, evidence = classify_formula_context(
        r"\begin{equation} L = \sum_i x_i \end{equation}",
        semantic_channel="display_math",
    )

    assert family == "DISPLAY_MATH_CONTEXT"
    assert evidence.display_math_env
    assert should_exclude_from_ordinary_visible_prose(family)


def test_formula_context_groups_do_not_mutate_records() -> None:
    records = [
        {"node_id": "n1", "text": "The objective is defined as"},
        {"node_id": "n2", "text": r"\begin{equation} L=x+y \end{equation}", "semantic_channel": "display_math"},
        {"node_id": "n3", "text": "where x denotes the input."},
    ]
    before = [dict(record) for record in records]

    groups = build_formula_context_groups(records)

    assert records == before
    assert groups
    assert groups[0].context_type == "DISPLAY_MATH_CONTEXT"
    assert groups[0].source_v7_ids == ["n2"]
