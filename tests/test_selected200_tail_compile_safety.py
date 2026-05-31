from __future__ import annotations

from src.generation.front_matter import render_author_block_original_like
from src.generation.latex_helpers import escape_latex, render_equation, render_inline_math, render_text_with_inline_latex


def test_selected200_tail_unicode_mappings_from_logs_are_compile_safe() -> None:
    rendered = escape_latex("⋊ ⋆")

    assert r"\ensuremath{\rtimes}" in rendered
    assert r"\ensuremath{\star}" in rendered


def test_selected200_tail_unicode_math_payloads_do_not_leak_raw_glyphs() -> None:
    assert render_inline_math("K⋊Z") == r"\(K\rtimes{}Z\)"
    assert render_inline_math("eta⋆") == r"\(eta\star{}\)"


def test_selected200_tail_ocr_greek_macro_glue_is_split_safely() -> None:
    assert render_inline_math(r"\chiB") == r"\(\chi B\)"
    assert render_inline_math(r"\zetax") == r"\(\zeta x\)"
    assert render_inline_math(r"\iotaQ") == r"\(\iota Q\)"
    assert render_inline_math(r"\inftyP") == r"\(\infty P\)"


def test_selected200_tail_ocr_greek_macro_glue_in_body_text_is_split_safely() -> None:
    rendered = render_text_with_inline_latex(r"mass \chiB and support \inftyP appear")

    assert r"\(\chi B\)" in rendered
    assert r"\(\infty P\)" in rendered


def test_selected200_tail_pmb_without_argument_falls_back_visibly() -> None:
    rendered = render_inline_math(r"\pmb")

    assert not rendered.startswith(r"\(")
    assert r"\textbackslash{}pmb" in rendered


def test_selected200_tail_pmb_leak_in_text_does_not_break_later_words() -> None:
    rendered = render_text_with_inline_latex(r"noise { \pmb x } _ { T }")

    assert r"\textbackslash{}pmb" in rendered
    assert "x" in rendered
    assert not rendered.startswith(r"\(")


def test_selected200_tail_escaped_brace_math_payload_falls_back() -> None:
    rendered = render_inline_math(r"^ * T^{\ \}")

    assert not rendered.startswith(r"\(")
    assert r"\textbackslash{}" in rendered
    assert "T" in rendered


def test_selected200_tail_escaped_brace_display_payload_falls_back() -> None:
    rendered = render_equation(r"T^{\ \}")

    assert "formula_fallback_escaped_display" in rendered
    assert "T" in rendered


def test_selected200_tail_text_mode_accent_leak_in_display_payload_falls_back() -> None:
    rendered = render_equation(r"\frac { \^{\ast} T h e } { L o u i s }")

    assert "formula_fallback_escaped_display" in rendered
    assert r"\textbackslash{}frac" in rendered


def test_selected200_tail_textcircled_in_display_payload_falls_back() -> None:
    rendered = render_equation(r"\widetilde{\alpha} \textcircled { \ r { \alpha } }")

    assert "formula_fallback_escaped_display" in rendered
    assert r"\textbackslash{}textcircled" in rendered


def test_selected200_tail_existing_guards_remain_unchanged() -> None:
    assert render_inline_math(r"\left| x \right|") == r"\(\left| x \right|\)"
    assert "formula_fallback_escaped_display" in render_equation(r"\begin{array}{c} broken & row \end{array}")
    assert "formula_fallback_escaped_display" in render_equation("a & b")


def test_selected200_tail_frontmatter_output_unaffected() -> None:
    rendered = render_author_block_original_like("Alice_Bob", [], None)

    assert "Alice\\_Bob" in rendered
    assert "formula_fallback" not in rendered
