from __future__ import annotations

from src.generation.front_matter import FrontMatterLine, _render_front_matter_line
from src.generation.latex_helpers import escape_latex, render_inline_math, render_text_with_inline_latex


def test_unicode_minus_maps_to_text_safe_hyphen() -> None:
    assert escape_latex("a−b") == "a-b"


def test_curly_phi_maps_to_compile_safe_latex() -> None:
    assert escape_latex("ϕ") == r"\ensuremath{\phi}"


def test_greek_sequence_in_text_maps_safely() -> None:
    rendered = escape_latex("α β γ η θ λ μ π ρ σ τ ω ± ×")

    assert rendered == (
        r"\ensuremath{\alpha} \ensuremath{\beta} \ensuremath{\gamma} "
        r"\ensuremath{\eta} \ensuremath{\theta} \ensuremath{\lambda} "
        r"\ensuremath{\mu} \ensuremath{\pi} \ensuremath{\rho} "
        r"\ensuremath{\sigma} \ensuremath{\tau} \ensuremath{\omega} "
        r"\ensuremath{\pm} \ensuremath{\times}"
    )


def test_existing_latex_command_is_rendered_as_math_not_double_escaped() -> None:
    assert render_text_with_inline_latex(r"\alpha") == r"$\alpha$"


def test_existing_inline_math_is_not_corrupted() -> None:
    assert render_text_with_inline_latex(r"$x-\phi$") == r"$x-\phi$"


def test_increment_sign_maps_safely_in_text_and_math() -> None:
    assert escape_latex("∆t") == r"\ensuremath{\Delta}t"
    assert render_inline_math("∆t") == r"$\Delta{}t$"


def test_unicode_minus_maps_safely_inside_math_payload() -> None:
    assert render_inline_math("− C") == "$- C$"


def test_smart_quotes_and_nonbreaking_space_map_safely() -> None:
    assert escape_latex("“quoted”\u00a0text") == "``quoted'' text"


def test_reserved_text_characters_still_escape() -> None:
    assert escape_latex(r"a_b & 50% #1 {x}") == r"a\_b \& 50\% \#1 \{x\}"


def test_frontmatter_line_rendering_uses_safe_unicode_escaping() -> None:
    rendered = _render_front_matter_line(FrontMatterLine("alice_ϕ@example.edu", role="email"), None)

    assert r"\texttt{" in rendered
    assert r"alice\_\ensuremath{\phi}@example.edu" in rendered
