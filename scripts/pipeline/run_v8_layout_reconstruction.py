#!/usr/bin/env python3
"""Run the complete v8 reconstruction path for one document.

Path:
  MinerU middle.json (+ optional content_list asset sidecar)
  -> v8 content payload
  -> DocumentIR
  -> v8 RenderTreeIR
  -> original-like LaTeX renderer
  -> optional LaTeX compile
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

from src.adapters.mineru_v8_document_ir import convert_v8_payload_to_document_ir, load_v8_document_ir  # noqa: E402
from src.evaluation.compile_eval import compile_latex  # noqa: E402
from src.generation.ir_renderer import IRLatexRenderConfig  # noqa: E402
from src.generation.render_surface import render_original_like_document  # noqa: E402
from src.generation.v8_style_detector import detect_v8_style  # noqa: E402
from src.ir.serialization import read_json, write_json  # noqa: E402
from src.perception.mineru_v8_reflow import build_v8_from_middle, dump_json  # noqa: E402
from src.reasoning.front_matter_extractor import extract_front_matter  # noqa: E402
from src.reasoning.front_matter_ir_loader import load_front_matter_ir_sidecar  # noqa: E402
from src.reasoning.float_caption_layout import build_float_caption_layout_sidecars  # noqa: E402
from src.reasoning.v8_render_tree import build_v8_render_tree  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--middle-json", type=Path)
    parser.add_argument("--content-list-json", type=Path)
    parser.add_argument("--style-content-list-json", type=Path)
    parser.add_argument(
        "--source-tex",
        type=Path,
        help="Optional source .tex path used only for citation/bibliography and source float layout sidecars.",
    )
    parser.add_argument("--v8-json", type=Path, help="Optional prebuilt v8 content JSON.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--middle-block-source", default="preproc_blocks", choices=("preproc_blocks", "para_blocks"))
    parser.add_argument("--debug-page", type=int)
    parser.add_argument("--compile-engine", default="auto")
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--no-resolve-citations",
        action="store_true",
        help="Disable the standard v7 citation/bibliography repair path.",
    )
    parser.add_argument(
        "--enable-float-caption-layout-experimental",
        action="store_true",
        help="Opt into the experimental v8 FloatCaptionLayout pass. Default production path is unchanged.",
    )
    parser.add_argument(
        "--enable-algorithm-region-renderer-experimental",
        action="store_true",
        help="Opt into the experimental v8 AlgorithmRegion renderer Phase 0. Default production path is unchanged.",
    )
    parser.add_argument(
        "--enable-frontmatter-ir-renderer-experimental",
        action="store_true",
        help="Opt into the experimental FrontMatterIR renderer Phase 0. Default production path is unchanged.",
    )
    parser.add_argument(
        "--frontmatter-ir-sidecar",
        type=Path,
        help="FrontMatterIR Phase0 sidecar to consume when --enable-frontmatter-ir-renderer-experimental is set.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.v8_json is not None:
        v8_payload = read_json(args.v8_json)
        v8_path = args.output_dir / f"{args.doc_id}_content_list_v8.json"
        dump_json(v8_path, v8_payload)
        diagnostics = v8_payload.get("diagnostics", {}) if isinstance(v8_payload, dict) else {}
    else:
        if args.middle_json is None:
            raise SystemExit("--middle-json is required unless --v8-json is provided")
        v8_payload = build_v8_from_middle(
            doc_id=args.doc_id,
            middle_json_path=args.middle_json,
            content_list_json_path=args.content_list_json,
            style_content_list_json_path=args.style_content_list_json,
            middle_block_source=args.middle_block_source,
            debug_page=args.debug_page,
        )
        v8_path = args.output_dir / f"{args.doc_id}_content_list_v8.json"
        diagnostics_path = args.output_dir / f"{args.doc_id}_v8_diagnostics.json"
        dump_json(v8_path, {key: value for key, value in v8_payload.items() if key != "diagnostics"})
        dump_json(diagnostics_path, v8_payload["diagnostics"])
        diagnostics = v8_payload["diagnostics"]

    document = convert_v8_payload_to_document_ir(
        v8_payload,
        source_path=v8_path,
        pdf_path=args.pdf,
        doc_id=args.doc_id,
    )
    document_path = args.output_dir / "document_ir.json"
    write_json(document_path, document)

    if args.enable_frontmatter_ir_renderer_experimental:
        if args.frontmatter_ir_sidecar is None:
            raise SystemExit("--frontmatter-ir-sidecar is required with --enable-frontmatter-ir-renderer-experimental")
        front_matter = load_front_matter_ir_sidecar(args.frontmatter_ir_sidecar)
    else:
        front_matter = extract_front_matter(document)
    write_json(args.output_dir / "front_matter_diag.json", front_matter.to_diagnostic())
    float_caption_sidecars = build_float_caption_layout_sidecars(document)
    write_json(args.output_dir / "float_caption_fix_diag.json", float_caption_sidecars.to_diagnostic())
    tree = build_v8_render_tree(
        document,
        document_ir_path=str(document_path),
        front_matter=front_matter,
        enable_float_caption_layout=args.enable_float_caption_layout_experimental,
        enable_algorithm_region_renderer=args.enable_algorithm_region_renderer_experimental,
    )
    render_tree_path = args.output_dir / "render_tree_ir.json"
    write_json(render_tree_path, tree)
    algorithm_diag = tree.metadata.get("algorithm_region_renderer_diag") if isinstance(tree.metadata, dict) else None
    if isinstance(algorithm_diag, dict):
        write_json(args.output_dir / "algorithm_region_render_diag.json", algorithm_diag)
        write_json(args.output_dir / "algorithm_region_consumed_nodes.json", algorithm_diag.get("consumed_nodes", []))
        write_json(args.output_dir / "algorithm_region_compile_risk.json", algorithm_diag.get("compile_risks", []))
        write_json(args.output_dir / "algorithm_region_render_policy.json", algorithm_diag.get("render_policies", []))
    style, style_diagnostics = detect_v8_style(document, tree=tree)
    style_path = args.output_dir / "style_profile.json"
    style_diag_path = args.output_dir / "v8_style_detector_diag.json"
    write_json(style_path, style)
    write_json(style_diag_path, style_diagnostics)

    shutil.copy2(args.pdf, args.output_dir / "original.pdf")
    tex = render_original_like_document(
        document,
        tree,
        style=style,
        config=IRLatexRenderConfig(
            title=None,
            include_maketitle=False,
            front_matter_mode="original_like",
            table_asset_output_dir=args.output_dir / "assets",
            figure_asset_output_dir=args.output_dir / "assets",
            table_asset_latex_prefix="assets",
            figure_asset_latex_prefix="assets",
            front_matter_ir=front_matter if args.enable_frontmatter_ir_renderer_experimental else None,
            front_matter_renderer_experimental=args.enable_frontmatter_ir_renderer_experimental,
        ),
        resolve_citations=not args.no_resolve_citations,
        source_tex_path=args.source_tex,
    )
    tex = ensure_v8_math_compatibility(tex)
    tex_path = args.output_dir / "generated.tex"
    tex_path.write_text(tex, encoding="utf-8")

    compile_report: dict[str, Any] = {"success": "not_run", "skipped": True}
    if not args.skip_compile:
        compile_report = compile_latex(
            tex_path,
            output_dir=args.output_dir,
            engine=args.compile_engine,
            timeout=args.compile_timeout,
            passes=2,
        )
    write_json(args.output_dir / "compile_report.json", compile_report)

    record = {
        "schema_version": "v8_layout_reconstruction_record_v1",
        "doc_id": args.doc_id,
        "middle_json": str(args.middle_json) if args.middle_json else None,
        "content_list_json": str(args.content_list_json) if args.content_list_json else None,
        "style_content_list_json": str(args.style_content_list_json) if args.style_content_list_json else None,
        "source_tex": str(args.source_tex) if args.source_tex else None,
        "resolve_citations": not args.no_resolve_citations,
        "v8_content_json": str(v8_path),
        "document_ir": str(document_path),
        "render_tree_ir": str(render_tree_path),
        "style_profile": str(style_path),
        "style_detector_diag": str(style_diag_path),
        "generated_tex": str(tex_path),
        "generated_pdf": str(args.output_dir / "generated.pdf"),
        "compile": compile_report,
        "v8_diagnostics_summary": {
            "item_count": diagnostics.get("item_count") if isinstance(diagnostics, dict) else None,
            "merge_count": diagnostics.get("merge_count") if isinstance(diagnostics, dict) else None,
            "merge_reason_counts": diagnostics.get("merge_reason_counts") if isinstance(diagnostics, dict) else None,
        },
        "document_node_count": len(document.nodes),
        "render_tree_node_count": len(tree.nodes),
        "float_caption_layout_experimental_enabled": args.enable_float_caption_layout_experimental,
        "float_caption_layout_sidecar": str(args.output_dir / "float_caption_fix_diag.json"),
        "algorithm_region_renderer_experimental_enabled": args.enable_algorithm_region_renderer_experimental,
        "algorithm_region_renderer_sidecar": str(args.output_dir / "algorithm_region_render_diag.json")
        if isinstance(algorithm_diag, dict)
        else None,
        "frontmatter_ir_renderer_experimental_enabled": args.enable_frontmatter_ir_renderer_experimental,
        "frontmatter_ir_sidecar": str(args.frontmatter_ir_sidecar) if args.frontmatter_ir_sidecar else None,
    }
    write_json(args.output_dir / "v8_layout_reconstruction_record.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


def ensure_v8_math_compatibility(tex: str) -> str:
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
