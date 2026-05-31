from __future__ import annotations

import subprocess
from pathlib import Path

from src.evaluation.compile_eval import run_compile_commands
from src.generation.ir_renderer import IRLatexRenderConfig, OriginalLikeIRLatexRenderer
from src.generation.latex_helpers import escape_latex, quarantine_broken_rendered_inline_math, render_equation, render_inline_math, render_text_with_inline_latex
from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR, RendererMode, RenderRole, RenderTreeIR, RenderTreeNode, StyleProfile


def test_observed_dotted_d_unicode_is_safe_in_text_and_math() -> None:
    assert escape_latex("Ḋ") == r"\.{D}"
    assert render_inline_math("P_{Ḋ}") == r"\(P_{\dot{D}}\)"


def test_observed_unsupported_astrosun_macro_is_normalized() -> None:
    rendered = render_inline_math(r"\varepsilon_{c, \astrosun}")
    assert r"\astrosun" not in rendered
    assert r"\odot" in rendered
    assert rendered.startswith(r"\(")


def test_unbraced_math_accents_fallback_in_inline_math() -> None:
    rendered = render_inline_math(r"\widetilde \mathrm{Gr}")
    assert r"\(\widetilde" not in rendered
    assert r"\textbackslash{}widetilde" in rendered


def test_unbraced_math_accents_in_text_span_do_not_leak_raw_math() -> None:
    rendered = render_text_with_inline_latex(r"ergodic when \(\mu ( \bar\) B )")
    assert r"\(\mu ( \bar\)" not in rendered
    assert r"\textbackslash{}bar" in rendered


def test_post_span_math_quarantine_catches_text_mode_accent_in_math() -> None:
    rendered = quarantine_broken_rendered_inline_math(r"\(\epsilon \ ( \^{\it 4}\)")
    assert r"\(\epsilon \ ( \^{\it 4}\)" not in rendered
    assert r"\textbackslash{}\textasciicircum{}" in rendered


def test_ambiguous_atop_display_math_falls_back() -> None:
    rendered = render_equation(r"a_i \atop a_p")
    assert "formula_fallback_escaped_display" in rendered
    assert r"\begin{align}" not in rendered


def test_malformed_kern_dimension_falls_back() -> None:
    rendered = render_equation(r"\left. x \right. \kern - delimiterspace y")
    assert "formula_fallback_escaped_display" in rendered
    assert r"\kern - delimiterspace" not in rendered or r"\textbackslash{}kern" in rendered


def test_text_mode_unknown_control_sequence_is_escaped() -> None:
    rendered = render_text_with_inline_latex(r"plain \unknowncontrol text")
    assert r"\unknowncontrol" not in rendered
    assert r"\textbackslash{}unknowncontrol" in rendered


def test_table_text_starting_with_initial_is_not_promoted_to_enumerate() -> None:
    node = DocumentNode(
        "table1",
        BlockType.TABLE,
        "M. Malin et al.: Table 1. Date and time UT",
        0,
        [BBox(100, 100, 900, 300)],
        0,
        metadata={"table_body": "<table><tr><td>Date and time UT</td><td>Filter</td></tr></table>"},
    )
    document = DocumentIR(
        doc_id="heldout_tail_table",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["table1"])],
        nodes=[node],
        reading_order=["table1"],
    )
    tree = RenderTreeIR(
        doc_id="heldout_tail_table",
        document_ir_path="document_ir.json",
        root_id="root",
        nodes=[
            RenderTreeNode(render_id="root", role=RenderRole.ROOT, children=["table"]),
            RenderTreeNode(render_id="table", role=RenderRole.TABLE, source_node_ids=["table1"]),
        ],
    )
    style = StyleProfile(profile_id="heldout-tail", mode=RendererMode.ORIGINAL_LIKE)
    tex = OriginalLikeIRLatexRenderer(IRLatexRenderConfig(table_safe_fallback_experimental=True)).render(document, tree, style)
    assert r"\begin{enumerate}" not in tex
    assert r"\begin{table" in tex


def test_compile_runner_replaces_non_utf8_log_bytes(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="ok", stderr=None)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_compile_commands(["latexmk"], tmp_path, timeout=1, passes=1)

    assert result.returncode == 0
    assert calls[0]["encoding"] == "utf-8"
    assert calls[0]["errors"] == "replace"
