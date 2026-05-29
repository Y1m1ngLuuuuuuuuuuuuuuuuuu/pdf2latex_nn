from __future__ import annotations

from src.evaluation.comparison_structure import latex_to_comparison
from src.generation.latex_helpers import render_algorithm_region_phase0


def _blocks(tex: str):
    return latex_to_comparison(r"\begin{document}" + tex + r"\end{document}").blocks


def test_figure_algorithm_caption_converts_to_algorithm_block() -> None:
    blocks = _blocks(
        r"""
        \begin{figure}[H]
        \centering
        \fbox{Algorithm body}
        \caption{Algorithm 1: Train model}
        \end{figure}
        """
    )
    assert any(block.block_type == "algorithm" for block in blocks)
    assert any(block.block_type == "caption" and block.marker == "algorithm" for block in blocks)


def test_figure_alg_label_converts_to_algorithm_block_without_caption() -> None:
    blocks = _blocks(
        r"""
        \begin{figure}[H]
        \centering
        \fbox{Algorithm body}
        \label{alg:phase0_1}
        \end{figure}
        """
    )
    assert any(block.block_type == "algorithm" for block in blocks)


def test_ordinary_figure_caption_stays_figure() -> None:
    blocks = _blocks(
        r"""
        \begin{figure}[H]
        \centering
        \fbox{Image}
        \caption{Figure 1: Accuracy}
        \end{figure}
        """
    )
    assert any(block.block_type == "figure" for block in blocks)
    assert not any(block.block_type == "algorithm" for block in blocks)


def test_algorithm_reference_paragraph_is_not_algorithm_block() -> None:
    blocks = _blocks("Algorithm 1 shows the optimization routine.")
    assert any(block.block_type == "paragraph" for block in blocks)
    assert not any(block.block_type == "algorithm" for block in blocks)


def test_phase01_fallback_keeps_alg_label_and_safe_text() -> None:
    tex = render_algorithm_region_phase0(
        caption="",
        body="x_y = 1 % mask & score",
        label="alg:phase0_demo",
        render_policy="verbatim_fallback",
    )
    assert r"\label{alg:phase0_demo}" in tex
    assert r"\%" in tex
    assert r"\&" in tex
    assert r"\_" in tex
    blocks = _blocks(tex)
    assert any(block.block_type == "algorithm" for block in blocks)
