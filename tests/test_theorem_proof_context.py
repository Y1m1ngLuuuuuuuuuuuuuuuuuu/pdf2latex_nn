from src.generation.ir_renderers.text import render_theorem_proof_context
from src.reasoning.formula_context_group import classify_formula_context


def test_theorem_proof_context_is_not_section_heading() -> None:
    family, evidence = classify_formula_context("Theorem 1. Let f be continuous.")

    assert family == "THEOREM_PROOF_CONTEXT"
    assert evidence.theorem_like


def test_theorem_renderer_uses_bold_inline_label_without_package() -> None:
    rendered = render_theorem_proof_context("Theorem 1.", "Let f be continuous.")

    assert rendered.startswith(r"\noindent\textbf{Theorem 1.}")
    assert r"\begin{theorem}" not in rendered
