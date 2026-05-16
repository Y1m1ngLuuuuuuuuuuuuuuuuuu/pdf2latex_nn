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


def test_markdown_to_comparison_normalizes_nougat_mmd_heading_conventions(tmp_path: Path) -> None:
    markdown = """
    # Paper Title From Nougat

    Jingzhi Gong

    ###### Abstract

    This is the abstract paragraph.

    ## Introduction

    Intro text.

    ### Method

    Method text.

    ## References

    [1] Smith. Paper.
    """
    document = markdown_to_comparison(markdown, doc_id="demo-nougat-mmd")
    payload = document.to_dict()
    assert payload["test_items"]["counts"]["document_titles"] == 1
    assert payload["test_items"]["counts"]["headings"] == 3
    assert payload["heading_tree"][0]["text"] == "Introduction"
    assert payload["heading_tree"][0]["level"] == 1
    assert payload["heading_tree"][1]["text"] == "Method"
    assert payload["heading_tree"][1]["level"] == 2
    assert payload["heading_tree"][2]["text"] == "References"
    assert payload["heading_tree"][2]["level"] == 1
    assert any(block["block_type"] == "abstract" and block["text"] == "This is the abstract paragraph." for block in payload["blocks"])
    assert payload["test_items"]["counts"]["references"] == 1


def test_markdown_to_comparison_treats_reference_bullets_as_references(tmp_path: Path) -> None:
    markdown = """
    # Paper Title

    ###### Abstract

    Abstract text.

    ## Introduction

    Body text.

    References

    * [1] Smith. Paper.
    * Doe and Roe (2024) Another paper.
    """
    document = markdown_to_comparison(markdown, doc_id="demo-nougat-refs")
    payload = document.to_dict()
    assert payload["test_items"]["counts"]["references"] == 2
    assert payload["test_items"]["counts"]["list_items"] == 0
    markers = [block["marker"] for block in payload["blocks"] if block["block_type"] == "reference_item"]
    assert markers[0] == "1"


def test_markdown_to_comparison_handles_mmd_latex_fragments(tmp_path: Path) -> None:
    markdown = r"""
    \section{Overview}

    Text before the formula.

    \[
    y = x + 1
    \]

    \begin{table}
    \caption{Table 2: Scores.}
    \begin{tabular}{cc}
    A & B
    \end{tabular}
    \end{table}

    Algorithm 1: Demo procedure.
    """
    document = markdown_to_comparison(markdown, doc_id="demo-mmd")
    payload = document.to_dict()
    assert payload["test_items"]["counts"]["headings"] == 1
    assert payload["heading_tree"][0]["text"] == "Overview"
    assert payload["test_items"]["counts"]["display_math"] == 1
    assert payload["test_items"]["counts"]["tables"] == 1
    assert payload["test_items"]["counts"]["captions"] == 2
    assert any(block["marker"] == "algorithm" for block in payload["blocks"] if block["block_type"] == "caption")


def test_comparison_json_writer(tmp_path: Path) -> None:
    document = markdown_to_comparison("# A\n\nBody.", doc_id="x")
    output = tmp_path / "x.json"
    write_comparison_json(document, output)
    assert output.exists()
    assert '"reading_order"' in output.read_text(encoding="utf-8")
