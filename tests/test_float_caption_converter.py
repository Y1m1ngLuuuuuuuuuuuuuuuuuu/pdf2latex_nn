from __future__ import annotations

from src.evaluation.comparison_structure import latex_to_comparison


def _blocks(tex: str) -> list[dict]:
    return latex_to_comparison(tex).to_dict()["blocks"]


def _captions(tex: str) -> list[dict]:
    return [block for block in _blocks(tex) if block["block_type"] == "caption"]


def test_caption_inside_figure_converts_to_typed_caption_block() -> None:
    captions = _captions(
        r"""
        \begin{figure}[H]
        \centering
        \includegraphics{a.png}
        \caption{Architecture overview}
        \end{figure}
        """
    )

    assert len(captions) == 1
    assert captions[0]["text"] == "Architecture overview"
    assert captions[0]["marker"] == "figure"


def test_optional_caption_uses_long_caption_text() -> None:
    captions = _captions(
        r"""
        \begin{table}
        \caption[Short]{Long caption text with details}
        \end{table}
        """
    )

    assert captions[0]["text"] == "Long caption text with details"
    assert captions[0]["marker"] == "table"


def test_multiline_caption_converts_correctly() -> None:
    captions = _captions(
        r"""
        \begin{figure}
        \caption{
          Multi-line caption
          with spacing.
        }
        \end{figure}
        """
    )

    assert captions[0]["text"] == "Multi-line caption with spacing."


def test_caption_with_math_refs_and_nested_braces_keeps_visible_tokens() -> None:
    captions = _captions(
        r"""
        \begin{figure}
        \caption{Performance of $\check { C }_{1}$ on Fig.~\ref{fig:a} with \textbf{bold} text}
        \end{figure}
        """
    )

    caption = captions[0]
    assert "C 1" in caption["text"]
    assert "fig:a" in caption["text"]
    assert "bold" in caption["normalized_text"]


def test_algorithm_caption_converts_to_algorithm_caption() -> None:
    captions = _captions(
        r"""
        \begin{algorithm}[H]
        \caption{Training procedure}
        \end{algorithm}
        """
    )

    assert captions[0]["marker"] == "algorithm"


def test_captionof_placeholder_converts_to_float_caption() -> None:
    captions = _captions(r"\captionof{figure}{Placeholder figure caption}")

    assert captions[0]["marker"] == "figure"
    assert captions[0]["parent_id"]


def test_subfigure_marker_is_preserved_for_matching_metadata() -> None:
    captions = _captions(
        r"""
        \begin{figure}
        \caption{Fig. 2(a): Left panel}
        \end{figure}
        """
    )

    assert captions[0]["label"] == "2(a)"
    assert captions[0]["normalized_text"] == "left panel"


def test_body_reference_is_not_converted_to_caption() -> None:
    blocks = _blocks("Figure 3 shows the architecture overview.")

    assert not [block for block in blocks if block["block_type"] == "caption"]
