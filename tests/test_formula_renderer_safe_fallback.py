from src.generation.latex_helpers import render_equation, render_inline_math, render_text_with_inline_latex


def test_safe_inline_formula_remains_inline_math():
    assert render_inline_math(r"\alpha + x") == r"\(\alpha + x\)"


def test_safe_display_formula_remains_display_math():
    rendered = render_equation(r"x + y = z")
    assert rendered.startswith("\\[\n")
    assert "x + y = z" in rendered


def test_unbalanced_brace_formula_falls_back_to_escaped_text_block():
    rendered = render_equation(r"\frac{x}{y")
    assert "formula_fallback_escaped_display" in rendered
    assert r"\textbackslash{}frac" in rendered


def test_unmatched_dollar_formula_does_not_emit_raw_dollar_math():
    rendered = render_inline_math(r"$x+y")
    assert r"\$x+y" in rendered
    assert not rendered.startswith("$")


def test_broken_command_with_trailing_backslash_falls_back():
    rendered = render_inline_math("\\alpha\\")
    assert r"\textbackslash{}alpha\textbackslash{}" in rendered


def test_unicode_math_payload_is_normalized():
    assert render_inline_math("𝐴 + ϕ") == r"\(A + \phi\)"


def test_existing_safe_latex_command_is_not_double_escaped():
    assert render_inline_math(r"\alpha") == r"\(\alpha\)"


def test_left_right_safe_pair_remains_safe():
    assert render_inline_math(r"\left(x\right)") == r"\(\left(x\right)\)"


def test_left_without_right_falls_back_safely():
    rendered = render_inline_math(r"\left(x")

    assert not rendered.startswith("$")
    assert r"\textbackslash{}left" in rendered


def test_raw_text_special_chars_escape_in_fallback():
    rendered = render_inline_math(r"\begin{array}{c} a_b & c% \end{array}")
    assert r"\_" in rendered
    assert r"\&" in rendered
    assert r"\%" in rendered
    assert not rendered.startswith("$")


def test_fallback_content_is_not_dropped():
    rendered = render_equation(r"\begin{array}{c} broken & row \end{array}")
    assert "broken" in rendered
    assert "row" in rendered


def test_text_inline_math_with_glued_command_is_split():
    rendered = render_text_with_inline_latex(r"Let $\rhox$ be fixed.")
    assert r"\(\rho x\)" in rendered
