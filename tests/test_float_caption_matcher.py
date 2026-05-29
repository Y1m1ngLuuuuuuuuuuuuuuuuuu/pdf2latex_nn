from __future__ import annotations

from src.reasoning.float_caption_matcher import is_caption_like_text, parse_caption_prefix


def test_figure_caption_grammar_accepts_standard_label() -> None:
    match = parse_caption_prefix("Figure 3: Architecture overview")

    assert match is not None
    assert match.caption_type == "figure"
    assert match.caption_number == "3"


def test_subfigure_caption_is_not_dropped() -> None:
    match = parse_caption_prefix("Fig. 2(a): Qualitative examples")

    assert match is not None
    assert match.caption_type == "figure"
    assert match.caption_number == "2(a)"


def test_table_roman_caption_grammar() -> None:
    match = parse_caption_prefix("Table IV: Ablation results")

    assert match is not None
    assert match.caption_type == "table"
    assert match.caption_number == "IV"


def test_algorithm_caption_grammar() -> None:
    match = parse_caption_prefix("Algorithm 1: Training procedure")

    assert match is not None
    assert match.caption_type == "algorithm"
    assert match.caption_number == "1"


def test_caption_false_positive_guard_for_body_reference() -> None:
    assert not is_caption_like_text("Figure 3 shows the architecture overview.")
    assert not is_caption_like_text("As shown in Fig. 2, the model converges.")
    assert not is_caption_like_text("Table 1 reports the result.")
    assert not is_caption_like_text("Algorithm 1 is used for optimization.")

