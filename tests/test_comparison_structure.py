from pathlib import Path

from src.evaluation.comparison_structure import (
    latex_to_comparison,
    markdown_to_comparison,
    write_comparison_json,
)


def test_latex_to_comparison_extracts_core_structure(tmp_path: Path) -> None:
    tex = r"""
    \documentclass{article}
    \begin{document}
    \title{Demo}
    \maketitle
    \begin{abstract}
    We study Fig.~1 and cite \cite{smith2020}.
    \end{abstract}
    \section{Introduction}
    First paragraph with $x$ and Figure 1.
    \subsection{Method}
    \begin{enumerate}
    \item First item text.
    \item Second item text.
    \end{enumerate}
    \begin{equation}
    y = x + 1
    \end{equation}
    \begin{figure}
    \includegraphics{demo.png}
    \caption{Figure 1: Demo figure.}
    \end{figure}
    \begin{thebibliography}{9}
    \bibitem{smith2020} Smith. Paper.
    \end{thebibliography}
    \end{document}
    """
    document = latex_to_comparison(tex, doc_id="demo")
    payload = document.to_dict()
    assert payload["schema_version"] == "comparison_structure_v1"
    assert payload["test_items"]["counts"]["headings"] >= 3
    assert payload["test_items"]["counts"]["list_items"] == 2
    assert payload["test_items"]["counts"]["display_math"] == 1
    assert payload["test_items"]["counts"]["captions"] == 1
    assert payload["test_items"]["counts"]["references"] == 1
    assert "smith2020" in payload["test_items"]["citations"]
    assert any(ref["kind"] == "figure" and ref["label"] == "1" for ref in payload["test_items"]["cross_refs"])
    heading_texts = [item["text"] for item in payload["heading_tree"]]
    assert "Introduction" in heading_texts
    assert "Method" in heading_texts


def test_markdown_to_comparison_extracts_nougat_like_structure(tmp_path: Path) -> None:
    markdown = """
    # Introduction

    This paragraph cites [12] and mentions Table 2.

    ## Method

    1. First item
    2. Second item

    $$ y = x + 1 $$

    ![A panel](figure.png)

    Figure 1: Demo figure.

    # References

    [1] Smith. Paper.
    """
    document = markdown_to_comparison(markdown, doc_id="demo-md")
    payload = document.to_dict()
    assert payload["source_format"] == "markdown"
    assert payload["test_items"]["counts"]["headings"] == 3
    assert payload["test_items"]["counts"]["list_items"] == 2
    assert payload["test_items"]["counts"]["display_math"] == 1
    assert payload["test_items"]["counts"]["figures"] == 1
    assert payload["test_items"]["counts"]["captions"] == 1
    assert payload["test_items"]["counts"]["references"] == 1
    assert any(ref["kind"] == "table" and ref["label"] == "2" for ref in payload["test_items"]["cross_refs"])


def test_comparison_json_writer(tmp_path: Path) -> None:
    document = markdown_to_comparison("# A\n\nBody.", doc_id="x")
    output = tmp_path / "x.json"
    write_comparison_json(document, output)
    assert output.exists()
    assert '"reading_order"' in output.read_text(encoding="utf-8")
