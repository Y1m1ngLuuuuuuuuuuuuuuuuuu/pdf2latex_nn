#!/usr/bin/env python3
"""Diagnostic-only flat render for one v8 content JSON.

This script is intentionally not the v8 production path.  Use
``scripts/pipeline/run_v8_layout_reconstruction.py`` for the full v8 chain:
middle/content_list -> v8 payload -> DocumentIR -> RenderTreeIR -> renderer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.batch_visual_qa_inference import compile_tex  # noqa: E402
from src.adapters import MinerUV7DocumentIRAdapterConfig, convert_v7_payload_to_document_ir  # noqa: E402
from src.generation.ir_renderer import IRLatexRenderConfig  # noqa: E402
from src.generation.render_surface import render_original_like_document  # noqa: E402
from src.ir import BlockType, CoordinateSpace, DocumentIR, RenderRole, RenderTreeIR, RenderTreeNode  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--v8-json", required=True, type=Path)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pdflatex", default="pdflatex")
    parser.add_argument("--compile-runs", type=int, default=2)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--skip-compile", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = json.loads(args.v8_json.read_text(encoding="utf-8"))
    document = document_ir_from_v8(payload, source_path=args.v8_json, pdf_path=args.pdf, doc_id=args.doc_id)
    tree = flat_render_tree_from_document(document, document_ir_path=str(args.v8_json))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.pdf, args.output_dir / "original.pdf")
    tex = render_original_like_document(
        document,
        tree,
        config=IRLatexRenderConfig(
            title=None,
            include_maketitle=False,
            front_matter_mode="original_like",
            table_asset_output_dir=args.output_dir / "assets",
            figure_asset_output_dir=args.output_dir / "assets",
            table_asset_latex_prefix="assets",
            figure_asset_latex_prefix="assets",
        ),
        resolve_citations=False,
        source_tex_path=None,
    )
    tex = ensure_v8_smoke_math_compatibility(tex)
    tex_path = args.output_dir / "generated.tex"
    tex_path.write_text(tex, encoding="utf-8")
    compile_info: dict[str, Any] = {"skipped": bool(args.skip_compile)}
    if not args.skip_compile:
        compile_info = compile_tex(
            tex_path,
            pdflatex=args.pdflatex,
            runs=args.compile_runs,
            timeout=args.compile_timeout,
        )
    record = {
        "schema_version": "v8_smoke_render_record_v1",
        "doc_id": args.doc_id,
        "v8_json": str(args.v8_json),
        "pdf": str(args.pdf),
        "generated_tex": str(tex_path),
        "generated_pdf": str(args.output_dir / "generated.pdf"),
        "compile": compile_info,
        "node_count": len(document.nodes),
        "render_node_count": len(tree.nodes),
    }
    (args.output_dir / "v8_smoke_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


def document_ir_from_v8(payload: dict[str, Any], *, source_path: Path, pdf_path: Path, doc_id: str) -> DocumentIR:
    page_width, page_height = infer_page_size(payload)
    adapter_payload = {
        "schema_version": "content_v8_reflow_v1",
        "source_format": "mineru_middle_v8_reflow",
        "doc_id": doc_id,
        "items": normalize_v8_items_for_adapter(payload),
        "style_source_pdf": str(pdf_path),
    }
    return convert_v7_payload_to_document_ir(
        adapter_payload,
        source_path=source_path,
        pdf_path=pdf_path,
        doc_id=doc_id,
        config=MinerUV7DocumentIRAdapterConfig(
            require_styles=False,
            coordinate_space=CoordinateSpace.PDF_POINTS,
            default_page_width=page_width,
            default_page_height=page_height,
            extractor_name="mineru_v8_reflow",
        ),
    )


def normalize_v8_items_for_adapter(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("items") or []):
        if not isinstance(item, dict):
            continue
        record = dict(item)
        record.setdefault("id", record.get("id") or f"v8_{index:06d}")
        record.setdefault("global_order", index)
        record.setdefault("raw_type", record.get("type"))
        record.setdefault("canonical_type", record.get("type"))
        record.setdefault("layout_role", "body_text" if record.get("type") == "text" else record.get("type"))
        record.setdefault("v8_source", "middle_reflow")
        items.append(record)
    return items


def infer_page_size(payload: dict[str, Any]) -> tuple[float, float]:
    sizes: list[list[float]] = []
    for block in payload.get("atomic_blocks") or []:
        if not isinstance(block, dict):
            continue
        size = block.get("page_size")
        if isinstance(size, list) and len(size) >= 2:
            try:
                sizes.append([float(size[0]), float(size[1])])
            except (TypeError, ValueError):
                pass
    if not sizes:
        return 612.0, 792.0
    widths = sorted(size[0] for size in sizes)
    heights = sorted(size[1] for size in sizes)
    return widths[len(widths) // 2], heights[len(heights) // 2]


def flat_render_tree_from_document(document: DocumentIR, *, document_ir_path: str) -> RenderTreeIR:
    nodes: list[RenderTreeNode] = []
    children: list[str] = []
    for node in sorted(document.nodes, key=lambda n: (n.reading_index, n.page_idx, n.node_id)):
        role = render_role_for_v8_node(node)
        render_id = f"r_{node.node_id}"
        nodes.append(
            RenderTreeNode(
                render_id=render_id,
                role=role,
                source_node_ids=[node.node_id],
                text=node.text,
                children=[],
                attributes={
                    "v8_smoke_flat_tree": True,
                    "source_type": node.raw_type,
                },
            )
        )
        children.append(render_id)
    nodes.append(RenderTreeNode(render_id="root", role=RenderRole.ROOT, children=children))
    return RenderTreeIR(
        doc_id=document.doc_id,
        root_id="root",
        nodes=nodes,
        document_ir_path=document_ir_path,
        metadata={"adapter": "V8FlatSmokeRenderTree"},
    )


def render_role_for_v8_node(node: Any) -> RenderRole:
    text_key = " ".join(str(node.text or "").split()).casefold().strip(" .:;-")
    if node.node_type == BlockType.TITLE:
        if text_key in {"references", "bibliography"}:
            return RenderRole.REFERENCES
        if text_key == "abstract":
            return RenderRole.ABSTRACT
        return RenderRole.SECTION
    if node.node_type == BlockType.TABLE:
        return RenderRole.TABLE
    if node.node_type == BlockType.FIGURE:
        return RenderRole.FIGURE
    if node.node_type == BlockType.ALGORITHM:
        return RenderRole.ALGORITHM
    if node.node_type == BlockType.EQUATION:
        return RenderRole.DISPLAY_EQUATION
    if node.node_type == BlockType.LIST:
        return RenderRole.LIST_ITEM
    if node.node_type == BlockType.REFERENCE:
        return RenderRole.REFERENCE_ITEM
    return RenderRole.PARAGRAPH


def ensure_v8_smoke_math_compatibility(tex: str) -> str:
    """Patch rare OCR math macros for the v8 smoke path.

    This keeps the experiment self-contained.  The production renderer should
    eventually normalize these in the math/OCR cleanup layer.
    """

    macros: list[str] = []
    if r"\mathbfcal" in tex:
        macros.append(r"\providecommand{\mathbfcal}[1]{\mathcal{#1}}")
    if not macros:
        return tex
    marker = r"\begin{document}"
    if marker not in tex:
        return "\n".join([*macros, tex])
    return tex.replace(marker, "\n".join([*macros, "", marker]), 1)


if __name__ == "__main__":
    raise SystemExit(main())
