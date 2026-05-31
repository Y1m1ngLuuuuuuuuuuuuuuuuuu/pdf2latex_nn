from __future__ import annotations

from pathlib import Path

from src.generation.ir_renderer import IRLatexRenderConfig, OriginalLikeIRLatexRenderer
from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR, RendererMode, RenderRole, RenderTreeIR, RenderTreeNode, StyleProfile


def _style() -> StyleProfile:
    return StyleProfile(profile_id="table-safe-test", mode=RendererMode.ORIGINAL_LIKE)


def _document(nodes: list[DocumentNode]) -> DocumentIR:
    return DocumentIR(
        doc_id="table_safe_doc",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=[node.node_id for node in nodes])],
        nodes=nodes,
        reading_order=[node.node_id for node in nodes],
    )


def _tree(role: RenderRole, source_ids: list[str]) -> RenderTreeIR:
    return RenderTreeIR(
        doc_id="table_safe_doc",
        document_ir_path="document_ir.json",
        root_id="root",
        nodes=[
            RenderTreeNode(render_id="root", role=RenderRole.ROOT, children=["target"]),
            RenderTreeNode(render_id="target", role=role, source_node_ids=source_ids),
        ],
    )


def _table_node(node_id: str = "table1", *, text: str = "Table 1: Results.", metadata: dict | None = None) -> DocumentNode:
    return DocumentNode(
        node_id=node_id,
        node_type=BlockType.TABLE,
        text=text,
        page_idx=0,
        bboxes=[BBox(100, 100, 900, 260)],
        reading_index=0,
        metadata=metadata or {},
    )


def _render(node: DocumentNode, *, asset_dir: Path | None = None) -> str:
    config = IRLatexRenderConfig(
        table_safe_fallback_experimental=True,
        table_asset_output_dir=asset_dir,
        figure_asset_output_dir=asset_dir,
    )
    return OriginalLikeIRLatexRenderer(config).render(_document([node]), _tree(RenderRole.TABLE, [node.node_id]), _style())


def test_existing_table_asset_renders_compile_safe_includegraphics(tmp_path: Path) -> None:
    image = tmp_path / "table.png"
    image.write_bytes(b"not-a-real-png-but-path-exists")
    node = _table_node(metadata={"img_path": str(image), "table_caption": "Table 1: Results."})

    tex = _render(node, asset_dir=tmp_path / "assets")

    assert r"\includegraphics" in tex
    assert "table_table1.png" in tex
    assert r"\caption{Results}" in tex


def test_missing_crop_asset_does_not_emit_broken_includegraphics_path() -> None:
    node = _table_node(metadata={"img_path": "/definitely/missing/table.png", "table_caption": "Table 1: Results."})

    tex = _render(node)

    assert r"\includegraphics" not in tex
    assert "Table region preserved" in tex


def test_simple_rectangular_html_renders_safe_tabular() -> None:
    node = _table_node(
        metadata={
            "table_caption": "Table 2: Safe cells.",
            "table_body": "<table><tr><td>A & B</td><td>50%_ok</td></tr><tr><td>x</td><td>$y$</td></tr></table>",
        }
    )

    tex = _render(node)

    assert r"\begin{tabular}" in tex
    assert r"A \& B" in tex
    assert r"50\%\_ok" in tex
    assert r"\(y\)" in tex
    assert "<table>" not in tex


def test_complex_rowspan_html_falls_back_to_visual_or_placeholder(tmp_path: Path) -> None:
    image = tmp_path / "table.jpg"
    image.write_bytes(b"fake")
    node = _table_node(
        metadata={
            "img_path": str(image),
            "table_caption": "Table 3: Complex.",
            "table_body": "<table><tr><td rowspan=2>A</td><td>B</td></tr><tr><td>C</td></tr></table>",
        }
    )

    tex = _render(node, asset_dir=tmp_path / "assets")

    assert r"\begin{tabular}" not in tex
    assert r"\includegraphics" in tex
    assert "rowspan" not in tex


def test_raw_html_never_emitted_for_malformed_table() -> None:
    node = _table_node(metadata={"table_body": "<table><tr><td>A</td></tr><tr><td>B</td><td>C</td></tr></table>"})

    tex = _render(node)

    assert "<tr>" not in tex
    assert r"\includegraphics{/definitely" not in tex
    assert "Table region preserved" in tex


def test_column_count_mismatch_falls_back() -> None:
    node = _table_node(metadata={"table_body": "<table><tr><td>A</td></tr><tr><td>B</td><td>C</td></tr></table>"})

    tex = _render(node)

    assert r"\begin{tabular}" not in tex
    assert "Table region preserved" in tex


def test_duplicate_table_caption_not_rendered_twice() -> None:
    caption = "Table 1: Same long experimental result caption."
    first = _table_node("table1", text=caption, metadata={"table_caption": caption, "table_body": "<table><tr><td>A</td></tr></table>"})
    second = _table_node("table2", text=caption, metadata={"table_caption": caption, "table_body": "<table><tr><td>B</td></tr></table>"})
    tree = RenderTreeIR(
        doc_id="table_safe_doc",
        document_ir_path="document_ir.json",
        root_id="root",
        nodes=[
            RenderTreeNode(render_id="root", role=RenderRole.ROOT, children=["a", "b"]),
            RenderTreeNode(render_id="a", role=RenderRole.TABLE, source_node_ids=["table1"]),
            RenderTreeNode(render_id="b", role=RenderRole.TABLE, source_node_ids=["table2"]),
        ],
    )
    config = IRLatexRenderConfig(table_safe_fallback_experimental=True)

    tex = OriginalLikeIRLatexRenderer(config).render(_document([first, second]), tree, _style())

    assert tex.count(r"\caption{Same long experimental result caption}") == 1
    assert tex.count(r"\begin{tabular}") == 2


def test_ordinary_body_table_mention_not_promoted() -> None:
    node = DocumentNode(
        "body1",
        BlockType.TEXT,
        "Table 1 shows the result in prose.",
        0,
        [BBox(100, 100, 900, 130)],
        0,
    )
    config = IRLatexRenderConfig(table_safe_fallback_experimental=True)

    tex = OriginalLikeIRLatexRenderer(config).render(_document([node]), _tree(RenderRole.PARAGRAPH, ["body1"]), _style())

    assert r"\begin{table}" not in tex
    assert "Table 1 shows" in tex


def test_no_table_placeholder_without_table_evidence() -> None:
    node = DocumentNode("body1", BlockType.TEXT, "ordinary body", 0, [BBox(0, 0, 10, 10)], 0)
    config = IRLatexRenderConfig(table_safe_fallback_experimental=True)

    tex = OriginalLikeIRLatexRenderer(config).render(_document([node]), _tree(RenderRole.PARAGRAPH, ["body1"]), _style())

    assert "Table region preserved" not in tex
