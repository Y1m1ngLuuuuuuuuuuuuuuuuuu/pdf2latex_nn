from __future__ import annotations

from src.generation.front_matter import render_author_block_original_like
from src.generation.latex_helpers import (
    escape_latex,
    render_equation,
    render_inline_math,
    render_text_with_inline_latex,
)


def test_tail2_unicode_mappings_from_remaining_smoke50_logs() -> None:
    rendered = escape_latex("↷ ∥ ⊔ ⋉ ≺")

    assert r"\ensuremath{\curvearrowright}" in rendered
    assert r"\ensuremath{\parallel}" in rendered
    assert r"\ensuremath{\sqcup}" in rendered
    assert r"\ensuremath{\ltimes}" in rendered
    assert r"\ensuremath{\prec}" in rendered


def test_tail2_unicode_math_payloads_do_not_leak_raw_glyphs() -> None:
    assert render_inline_math("Z ↷") == r"\(Z \curvearrowright{}\)"
    assert render_inline_math("∥R") == r"\(\parallel{}R\)"
    assert render_inline_math("O ⊔ O") == r"\(O \sqcup{} O\)"
    assert render_inline_math("Z⋉X") == r"\(Z\ltimes{}X\)"
    assert render_inline_math("M≺M") == r"\(M\prec{}M\)"


def test_tail2_sharp_unicode_mapping_does_not_glue_to_following_letter() -> None:
    rendered = render_inline_math("♯i")

    assert rendered == r"\(\sharp{}i\)"
    assert r"\sharpi" not in rendered


def test_tail2_greek_ocr_macro_glue_is_split_safely() -> None:
    assert render_inline_math(r"\kappaT") == r"\(\kappa T\)"
    assert render_text_with_inline_latex(r"metric \kappaT appears") == r"metric \(\kappa T\) appears"


def test_tail2_infty_is_not_split_by_in_glue_guard() -> None:
    assert render_inline_math(r"\infty") == r"\(\infty\)"
    assert render_text_with_inline_latex(r"limit \infty exists") == r"limit \(\infty\) exists"


def test_tail2_dangerous_tex_primitives_fallback_visibly() -> None:
    rendered = render_inline_math(r"\infty \aftergroup \egroup ) \mathbb{R}")

    assert not rendered.startswith(r"\(")
    assert r"\textbackslash{}aftergroup" in rendered
    assert r"\textbackslash{}egroup" in rendered
    assert "mathbb" in rendered


def test_tail2_dangerous_display_payload_falls_back() -> None:
    rendered = render_equation(r"\infty \aftergroup \egroup")

    assert "formula_fallback_escaped_display" in rendered
    assert r"\textbackslash{}aftergroup" in rendered


def test_tail2_broken_mathrm_brace_fallback_still_preserves_text() -> None:
    rendered = render_inline_math(r"\mathrm{e l s e \} -")

    assert not rendered.startswith(r"\(")
    assert "e l s e" in rendered
    assert r"\textbackslash{}mathrm" in rendered


def test_tail2_rendered_inline_math_post_guard_catches_broken_mathrm() -> None:
    rendered = render_text_with_inline_latex(r"or \(\mathrm{e l s e \} -\) 4")

    assert r"\(\mathrm" not in rendered
    assert r"\textbackslash{}mathrm" in rendered
    assert "e l s e" in rendered


def test_tail2_reference_like_text_command_quarantine_still_escapes() -> None:
    rendered = render_text_with_inline_latex(r"see DOI \aftergroup \egroup and \kappaT")

    assert r"\textbackslash{}aftergroup" in rendered
    assert r"\textbackslash{}egroup" in rendered
    assert r"\(\kappa T\)" in rendered


def test_tail2_existing_guards_remain_unchanged() -> None:
    assert "formula_fallback_escaped_display" in render_equation(r"\begin{array}{c} broken & row \end{array}")
    assert "formula_fallback_escaped_display" in render_equation("a & b")
    assert render_inline_math(r"\left| x \right|") == r"\(\left| x \right|\)"


def test_tail2_frontmatter_output_unaffected() -> None:
    rendered = render_author_block_original_like("Alice_Bob", [], None)

    assert "Alice\\_Bob" in rendered
    assert "formula_fallback" not in rendered
