from __future__ import annotations

from src.generation.front_matter import render_author_block_original_like
from src.generation.latex_helpers import escape_latex, render_equation, render_inline_math, render_text_with_inline_latex


def test_scale_unicode_mappings_from_smoke50_logs_are_compile_safe() -> None:
    rendered = escape_latex("ħ ♯ ς ⃗")

    assert r"\ensuremath{\hbar}" in rendered
    assert r"\ensuremath{\sharp}" in rendered
    assert r"\ensuremath{\varsigma}" in rendered
    assert r"\ensuremath{\vec{}}" in rendered


def test_undefined_ocr_partial_macro_is_split_safely() -> None:
    assert render_inline_math(r"\partialR") == r"\(\partial R\)"
    assert render_inline_math(r"\partialt") == r"\(\partial t\)"
    assert render_text_with_inline_latex(r"boundary \partialR exists") == r"boundary \(\partial R\) exists"


def test_undefined_ocr_relation_macros_are_split_safely() -> None:
    assert render_inline_math(r"\capC") == r"\(\cap C\)"
    assert render_inline_math(r"\toG") == r"\(\to G\)"
    assert render_inline_math(r"\subsetP") == r"\(\subset P\)"
    assert render_inline_math(r"\nur") == r"\(\nu r\)"


def test_uppercase_nu_ocr_macro_degrades_to_visible_normal_symbol() -> None:
    assert render_inline_math(r"\Nu (0, \sigma^2)") == r"\(N (0, \sigma^2)\)"


def test_raw_hash_in_math_payload_falls_back_visibly() -> None:
    rendered = render_inline_math(r"a # b")

    assert not rendered.startswith(r"\(")
    assert r"\#" in rendered
    assert "a" in rendered and "b" in rendered


def test_inline_fraction_without_braced_arguments_falls_back() -> None:
    rendered = render_inline_math(r"\Phi = \frac")

    assert not rendered.startswith(r"\(")
    assert r"\textbackslash{}frac" in rendered


def test_display_fraction_without_braced_arguments_falls_back() -> None:
    rendered = render_equation(r"\Phi = \frac")

    assert "formula_fallback_escaped_display" in rendered
    assert r"\textbackslash{}frac" in rendered


def test_repeated_superscript_display_payload_falls_back() -> None:
    rendered = render_equation(r"\boldsymbol{x}_{t}^{0}^{\top} + 1")

    assert "formula_fallback_escaped_display" in rendered
    assert r"\textbackslash{}boldsymbol" in rendered


def test_mathscr_is_normalized_to_standard_mathcal_without_new_package() -> None:
    rendered = render_equation(r"\mathscr{H}_{-} = 1")

    assert r"\mathcal{H}_{-}" in rendered
    assert r"\mathscr" not in rendered


def test_safe_formula_and_existing_guards_remain_unchanged() -> None:
    assert render_inline_math(r"\left| x \right|") == r"\(\left| x \right|\)"
    assert "formula_fallback_escaped_display" in render_equation(r"\begin{array}{c} broken & row \end{array}")
    assert "formula_fallback_escaped_display" in render_equation("a & b")


def test_reference_like_escaped_array_text_no_longer_leaks_broken_frac_math() -> None:
    rendered = render_text_with_inline_latex(r"Choosing \begin{array} { r } { \(\Phi = \frac\) 1 2 }")

    assert r"\textbackslash{}frac" in rendered
    assert r"\(\Phi = \frac\)" not in rendered


def test_frontmatter_output_unaffected_by_compile_hardening() -> None:
    rendered = render_author_block_original_like("Alice_Bob", [], None)

    assert "Alice\\_Bob" in rendered
    assert "formula_fallback" not in rendered
