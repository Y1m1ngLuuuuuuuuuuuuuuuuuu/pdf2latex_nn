from src.evaluation.comparison_structure import latex_to_comparison, markdown_to_comparison
from src.evaluation.structure_metrics import evaluate_comparison_structures


def test_structure_metrics_score_clean_structure() -> None:
    gold = latex_to_comparison(
        r"""
        \begin{document}
        \section{Intro}
        First paragraph cites \cite{x}.
        \subsection{Method}
        \begin{itemize}
        \item Item A
        \end{itemize}
        \begin{figure}
        \caption{Figure 1: Cat.}
        \end{figure}
        \begin{thebibliography}{9}
        \bibitem{x} X.
        \end{thebibliography}
        \end{document}
        """
    ).to_dict()
    pred = latex_to_comparison(
        r"""
        \begin{document}
        \section{Intro}
        First paragraph cites \cite{x}.
        \subsection{Method}
        \begin{itemize}
        \item Item A
        \end{itemize}
        \begin{figure}
        \caption{Figure 1: Cat.}
        \end{figure}
        \begin{thebibliography}{9}
        \bibitem{x} X.
        \end{thebibliography}
        \end{document}
        """
    ).to_dict()
    metrics = evaluate_comparison_structures(gold, pred)
    assert metrics["heading_tree_accuracy"]["score"] == 1.0
    assert metrics["reading_order_accuracy"]["score"] == 1.0
    assert metrics["paragraph_merge_f1"]["f1"] == 1.0
    assert metrics["paragraph_boundary_f1"]["f1"] == 1.0
    assert metrics["paragraph_text_coverage_f1"]["f1"] == 1.0
    assert metrics["section_attachment_f1"]["f1"] == 1.0
    assert metrics["reference_section_completeness"]["score"] == 1.0
    assert metrics["float_caption_attachment_accuracy"]["score"] == 1.0
    assert metrics["generated_structure_validity"]["is_valid"] is True


def test_structure_metrics_catches_missing_float_container() -> None:
    gold = latex_to_comparison(
        r"""
        \begin{document}
        \section{Intro}
        Body.
        \begin{figure}
        \caption{Figure 1: Cat.}
        \end{figure}
        \end{document}
        """
    ).to_dict()
    pred = markdown_to_comparison(
        """
        # Intro

        Body.

        Figure 1: Cat.
        """
    ).to_dict()
    metrics = evaluate_comparison_structures(gold, pred)
    assert metrics["float_caption_attachment_accuracy"]["score"] == 1.0
    assert metrics["generated_structure_validity"]["is_valid"] is False
    assert any(v["type"] == "caption_parent_not_float" for v in metrics["generated_structure_validity"]["violations"])


def test_structure_metrics_catches_reading_order_inversion() -> None:
    gold = markdown_to_comparison(
        """
        # A

        First paragraph.

        Second paragraph.
        """
    ).to_dict()
    pred = markdown_to_comparison(
        """
        # A

        Second paragraph.

        First paragraph.
        """
    ).to_dict()
    metrics = evaluate_comparison_structures(gold, pred)
    assert metrics["reading_order_accuracy"]["score"] < 1.0


def test_structure_metrics_text_coverage_tolerates_split_paragraphs() -> None:
    gold = markdown_to_comparison(
        """
        # A

        Alpha beta gamma delta epsilon zeta eta theta.
        """
    ).to_dict()
    pred = markdown_to_comparison(
        """
        # A

        Alpha beta gamma delta.

        Epsilon zeta eta theta.
        """
    ).to_dict()
    metrics = evaluate_comparison_structures(gold, pred)
    assert metrics["strict_block_match"]["matched_blocks"] < len(gold["blocks"])
    assert metrics["paragraph_boundary_f1"]["f1"] < 1.0
    assert metrics["paragraph_text_coverage_f1"]["f1"] == 1.0
    assert metrics["section_attachment_f1"]["f1"] == 1.0
