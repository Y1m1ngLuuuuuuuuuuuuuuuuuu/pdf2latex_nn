from __future__ import annotations

from src.generation.front_matter import render_author_block_original_like
from src.generation.latex_helpers import (
    render_equation,
    render_inline_math,
    should_fallback_noenv_raw_ampersand_display,
)


def test_raw_ampersand_no_environment_falls_back_safely() -> None:
    rendered = render_equation("a & b")

    assert "formula_fallback_escaped_display" in rendered
    assert r"\&" in rendered


def test_fallback_does_not_emit_raw_alignment_tab() -> None:
    rendered = render_equation(r"\left\{ 0, & \mathrm{if}~i \neq j \right\}")

    assert "formula_fallback_escaped_display" in rendered
    assert r"\&" in rendered
    assert " 0, & " not in rendered


def test_safe_display_without_ampersand_remains_display_math() -> None:
    rendered = render_equation("x^2 + y^2 = z^2")

    assert rendered.startswith("\\[\n")
    assert "x^2 + y^2 = z^2" in rendered


def test_aligned_environment_is_not_reclassified_by_noenv_helper() -> None:
    payload = r"\begin{aligned} a &= b \\ c &= d \end{aligned}"

    assert not should_fallback_noenv_raw_ampersand_display(payload)
    assert "formula_fallback_escaped_display" in render_equation(payload)


def test_array_environment_is_not_reclassified_by_noenv_helper() -> None:
    payload = r"\begin{array}{cc} a & b \\ c & d \end{array}"

    assert not should_fallback_noenv_raw_ampersand_display(payload)
    assert "formula_fallback_escaped_display" in render_equation(payload)


def test_malformed_risky_array_still_uses_formula_sprint_fallback() -> None:
    rendered = render_equation(r"\begin{array}{c} broken & row")

    assert "formula_fallback_escaped_display" in rendered
    assert "broken" in rendered


def test_formula_sprint2a_fallback_marker_unchanged() -> None:
    rendered = render_equation(r"\frac{x}{y")

    assert "formula_fallback_escaped_display" in rendered
    assert r"\textbackslash{}frac" in rendered


def test_body_inline_math_guard_unchanged() -> None:
    rendered = render_inline_math(r"\left| x \right$")

    assert r"\textbackslash{}right" in rendered
    assert not rendered.startswith(r"\(")


def test_unicode_helper_behavior_unchanged() -> None:
    assert render_inline_math("𝐴 + ϕ") == r"\(A + \phi\)"


def test_frontmatter_renderer_output_unchanged() -> None:
    rendered = render_author_block_original_like("Alice_Bob", [], None)

    assert "Alice\\_Bob" in rendered
    assert "formula_fallback" not in rendered
