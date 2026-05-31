from __future__ import annotations

from src.generation.front_matter import render_author_block_original_like
from src.generation.latex_helpers import render_equation, render_inline_math, render_text_with_inline_latex


def test_safe_inline_math_uses_renderer_owned_wrapper() -> None:
    assert render_inline_math("x_i + y_i") == r"\(x_i + y_i\)"


def test_existing_dollar_payload_is_normalized_without_raw_dollar_wrapper() -> None:
    rendered = render_inline_math("$x_i+y_i$")

    assert rendered == r"\(x_i+y_i\)"
    assert not rendered.startswith("$")


def test_safe_left_right_absolute_value_remains_math() -> None:
    assert render_inline_math(r"\left| x \right|") == r"\(\left| x \right|\)"


def test_safe_left_right_norm_remains_math() -> None:
    assert render_inline_math(r"\left\| x \right\|") == r"\(\left\| x \right\|\)"


def test_broken_left_right_missing_delimiter_falls_back() -> None:
    rendered = render_inline_math(r"\left| x \right")

    assert not rendered.startswith("$")
    assert not rendered.startswith(r"\(")
    assert r"\textbackslash{}right" in rendered


def test_broken_right_before_dollar_falls_back() -> None:
    rendered = render_inline_math(r"$\left| x \right$")

    assert not rendered.startswith("$")
    assert not rendered.startswith(r"\(")
    assert r"\textbackslash{}right" in rendered
    assert r"\$" in rendered


def test_dangling_right_left_segment_in_text_mode_is_escaped() -> None:
    rendered = render_text_with_inline_latex(r"bad $\right\| \left\|$ segment")

    assert r"\textbackslash{}right" in rendered
    assert r"\textbackslash{}left" in rendered
    assert r"\right" not in rendered.replace(r"\textbackslash{}right", "")


def test_argument_required_math_command_without_argument_falls_back() -> None:
    rendered = render_inline_math(r"\mathbf")

    assert not rendered.startswith(r"\(")
    assert r"\textbackslash{}mathbf" in rendered


def test_argument_required_math_command_with_argument_remains_math() -> None:
    assert render_inline_math(r"\mathbf{e}_{e}") == r"\(\mathbf{e}_{e}\)"


def test_unmatched_raw_dollar_in_text_segment_is_escaped() -> None:
    rendered = render_text_with_inline_latex("price $5 and text")

    assert r"\$5" in rendered


def test_text_fallback_preserves_visible_preview_and_backslashes() -> None:
    rendered = render_inline_math(r"\right\| \left\|")

    assert r"\textbackslash{}right" in rendered
    assert r"\textbackslash{}left" in rendered


def test_formula_sprint2a_display_fallback_still_marks_unsafe_payload() -> None:
    rendered = render_equation(r"\begin{array}{c} broken & row \end{array}")

    assert "formula_fallback_escaped_display" in rendered
    assert "broken" in rendered


def test_unicode_math_normalization_still_works() -> None:
    assert render_inline_math("𝐴 + ϕ") == r"\(A + \phi\)"


def test_frontmatter_renderer_output_unchanged_by_inline_guard() -> None:
    rendered = render_author_block_original_like("Alice_Bob", [], None)

    assert "Alice\\_Bob" in rendered
    assert "formula_fallback" not in rendered
