from src.reasoning.latex_flattener import (
    create_flat_tex_ast,
    expand_simple_macros,
    flatten_latex_file,
    inject_bbl,
    mask_math_environments,
    strip_comments,
)


def has_texsoup():
    try:
        import TexSoup  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def test_strip_comments_keeps_escaped_percent_and_removes_dead_input():
    tex = r"""
    live \% percent % dropped
    line break \\% comment after linebreak
    % \input{dead}
    """

    stripped = strip_comments(tex)

    assert r"live \% percent" in stripped
    assert "dropped" not in stripped
    assert "dead" not in stripped
    assert "comment after linebreak" not in stripped


def test_flatten_latex_file_recurses_inputs_and_skips_commented_input(tmp_path):
    main = tmp_path / "main.tex"
    section = tmp_path / "sections" / "intro.tex"
    child = tmp_path / "sections" / "child.tex"
    old = tmp_path / "old.tex"
    section.parent.mkdir()
    main.write_text(
        r"""
        % \input{old}
        \input{sections/intro}
        \input{missing_file}
        """,
        encoding="utf-8",
    )
    section.write_text(r"Intro text. \input{child}", encoding="utf-8")
    child.write_text("Child text.", encoding="utf-8")
    old.write_text("OLD SHOULD NOT APPEAR", encoding="utf-8")

    flattened = flatten_latex_file(main)

    assert "Intro text." in flattened.content
    assert "Child text." in flattened.content
    assert "OLD SHOULD NOT APPEAR" not in flattened.content
    assert len(flattened.included_files) == 2
    assert len(flattened.missing_files) == 1
    assert "missing input file" in flattened.warnings[0]


def test_inject_bbl_replaces_bibliography_with_compiled_references(tmp_path):
    (tmp_path / "paper.bbl").write_text(
        r"\begin{thebibliography}{1}\bibitem{a} Smith. Title.\end{thebibliography}",
        encoding="utf-8",
    )

    tex, bbl_path = inject_bbl(r"Body. \bibliography{refs}", tmp_path, "paper")

    assert bbl_path == (tmp_path / "paper.bbl").resolve()
    assert r"\bibliography" not in tex
    assert "Smith. Title." in tex


def test_expand_simple_macros_only_expands_zero_arg_text_aliases():
    tex = r"""
    \newcommand{\model}{GAT}
    \renewcommand{\ours}{PDF-Graph}
    \def\dataset{arXiv}
    \newcommand{\wrap}[1]{#1}
    \model and \ours on \dataset, but not \models.
    """

    expanded, macros = expand_simple_macros(tex)

    assert macros == {"model": "GAT", "ours": "PDF-Graph", "dataset": "arXiv"}
    assert "GAT and PDF-Graph on arXiv" in expanded
    assert r"\models" in expanded
    assert r"\wrap" in expanded


def test_mask_math_environments_before_texsoup():
    tex = r"Text $x+y$ and \[z\] plus \begin{align}a&=b\end{align} done."

    masked = mask_math_environments(tex)

    assert masked.count("[MATH]") == 3
    assert "x+y" not in masked
    assert "a&=b" not in masked


def test_create_flat_tex_ast_parses_flattened_content(tmp_path):
    if not has_texsoup():
        return
    main = tmp_path / "main.tex"
    main.write_text(r"\section{Intro} Body with $x$.", encoding="utf-8")

    ast = create_flat_tex_ast(main)

    assert "Intro" in str(ast)
    assert "[MATH]" in str(ast)
