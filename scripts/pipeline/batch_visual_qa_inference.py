#!/usr/bin/env python3
"""Batch end-to-end inference for human visual QA.

The script intentionally does not touch training data, graph schemas, or model
checkpoints.  It picks documents from the held-out test split, renders LaTeX
with the current TreeDecoder, compiles it with pdflatex, and stores each
generated PDF next to the original PDF for side-by-side inspection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.step5_generate_tex import (  # noqa: E402
    checkpoint_compatible_config,
    infer_document_title,
    load_node_records,
)
from scripts.pipeline.train_edge_gnn_full import split_indices  # noqa: E402
from src.adapters import MinerUV7DocumentIRAdapterConfig, convert_v7_payload_to_document_ir  # noqa: E402
from src.generation.render_surface import render_original_like_document  # noqa: E402
from src.ir import BlockType, DocumentIR, DocumentNode, RenderRole, RenderTreeIR, RenderTreeNode  # noqa: E402
from src.pipeline.v7_contract import assert_v7_content_json, assert_v7_graph_data  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402
from src.reasoning.postprocess import (  # noqa: E402
    ResolvedNode,
    TreeDecoder,
    TreeDecoderConfig,
    apply_heading_render_policy,
    canonical_render_type,
    root_with_document_toc,
    root_without_redundant_document_title,
    sorted_render_children,
)
from src.reasoning.prediction_io import write_predicted_relations  # noqa: E402
from src.adapters.mineru_v7_document_ir import stable_node_id  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--document-id", action="append", default=[], help="Optional explicit document id; repeatable.")
    parser.add_argument("--merge-threshold", type=float, default=0.42)
    parser.add_argument("--parent-threshold", type=float, default=0.53)
    parser.add_argument("--require-merge-argmax", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-parent-argmax", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--enable-layout-scope-continuation-merge",
        action="store_true",
        help="Experimental: allow a narrow deterministic text continuation merge across layout-band boundaries.",
    )
    parser.add_argument(
        "--enable-list-continuation-merge",
        action="store_true",
        help="Experimental: allow a narrow deterministic split-list-item continuation merge.",
    )
    parser.add_argument(
        "--enable-family-aware-merge-policy",
        action="store_true",
        help="Experimental: use family-specific MERGE thresholds/gates without changing the trained model.",
    )
    parser.add_argument("--family-body-list-merge-threshold", type=float, default=0.05)
    parser.add_argument("--family-reference-merge-threshold", type=float, default=0.82)
    parser.add_argument(
        "--enable-family-aware-missing-candidate-merge",
        action="store_true",
        help="Experimental: add only very narrow deterministic BODY/LIST continuation merges for missing candidate edges.",
    )
    parser.add_argument(
        "--heading-skeleton-mode",
        choices=["stack"],
        default="stack",
        help="Canonical decoder mode: global heading stack owns section scope, GNN refines local relations.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pdflatex", default="pdflatex")
    parser.add_argument("--compile-runs", type=int, default=2)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--clean-output-dir", action="store_true")
    parser.add_argument(
        "--renderer",
        choices=["ir"],
        default="ir",
        help="Production renderer. Only DocumentIR + RenderTreeIR + OriginalLikeIRLatexRenderer is supported.",
    )
    parser.add_argument(
        "--render-crops",
        dest="render_table_crops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate table and figure crop images in each QA output assets/ "
            "directory. Enabled by default so visual QA PDFs include all "
            "recoverable floats; use --no-render-crops to disable."
        ),
    )
    parser.add_argument(
        "--render-table-crops",
        dest="render_table_crops",
        action=argparse.BooleanOptionalAction,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    import torch

    if args.clean_output_dir and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    docs = select_documents(args)
    if not docs:
        raise ValueError("No documents selected for visual QA")

    device = resolve_device(args.device, torch=torch)
    model = load_model(args.checkpoint, device=device, torch=torch)
    decoder = TreeDecoder(
        TreeDecoderConfig(
            merge_threshold=args.merge_threshold,
            parent_threshold=args.parent_threshold,
            require_merge_argmax=args.require_merge_argmax,
            require_parent_argmax=args.require_parent_argmax,
            enable_layout_scope_continuation_merge=args.enable_layout_scope_continuation_merge,
            enable_list_continuation_merge=args.enable_list_continuation_merge,
            enable_family_aware_merge_policy=args.enable_family_aware_merge_policy,
            family_body_list_merge_threshold=args.family_body_list_merge_threshold,
            family_reference_merge_threshold=args.family_reference_merge_threshold,
            enable_family_aware_missing_candidate_merge=args.enable_family_aware_missing_candidate_merge,
            heading_skeleton_mode=args.heading_skeleton_mode,
        )
    )

    summary: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        try:
            row = run_one_document(
                doc,
                index=index,
                output_dir=args.output_dir,
                model=model,
                decoder=decoder,
                device=device,
                torch=torch,
                pdflatex=args.pdflatex,
                compile_runs=args.compile_runs,
                compile_timeout=args.compile_timeout,
                skip_compile=args.skip_compile,
                renderer=args.renderer,
                render_table_crops=args.render_table_crops,
                model_version=str(args.checkpoint),
            )
        except Exception as exc:  # noqa: BLE001 - QA must keep the batch moving.
            doc_dir = args.output_dir / f"{index:02d}_{safe_filename(str(doc.get('document_id', 'unknown')))}"
            doc_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "document_id": str(doc.get("document_id", "")),
                "doc_dir": str(doc_dir),
                "generated_pdf_exists": False,
                "error": repr(exc),
            }
            write_json(doc_dir / "qa_record.json", row)
        summary.append(row)
        status = "ok" if row.get("generated_pdf_exists") else "no_pdf"
        print(f"[{index:02d}/{len(docs):02d}] {doc['document_id']} {status} -> {row['doc_dir']}")

    payload = {
        "schema_version": "batch_visual_qa_v1",
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "limit": args.limit,
        "merge_threshold": args.merge_threshold,
        "parent_threshold": args.parent_threshold,
        "require_merge_argmax": args.require_merge_argmax,
        "require_parent_argmax": args.require_parent_argmax,
        "heading_skeleton_mode": args.heading_skeleton_mode,
        "renderer": args.renderer,
        "render_crops": args.render_table_crops,
        "documents": summary,
    }
    write_json(args.output_dir / "qa_manifest.json", payload)
    return 0


def select_documents(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    docs = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(docs, list):
        raise ValueError(f"Expected manifest list or documents list: {args.manifest}")
    docs = [doc for doc in docs if isinstance(doc, dict)]

    explicit_ids = set(args.document_id or [])
    if explicit_ids:
        selected = [doc for doc in docs if str(doc.get("document_id")) in explicit_ids]
        missing = sorted(explicit_ids - {str(doc.get("document_id")) for doc in selected})
        if missing:
            raise ValueError(f"Requested document ids not found in manifest: {missing}")
        return selected[: args.limit]

    if args.split == "all":
        split_docs = docs
    else:
        splits = split_indices(
            len(docs),
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            seed=args.seed,
        )
        split_docs = [docs[index] for index in splits[args.split]]

    selected: list[dict[str, Any]] = []
    for doc in split_docs:
        if has_required_paths(doc):
            selected.append(doc)
        if len(selected) >= args.limit:
            break
    return selected


def has_required_paths(doc: dict[str, Any]) -> bool:
    for key in ("pdf_path", "content_json", "graph_path"):
        value = doc.get(key)
        if not value or not Path(str(value)).exists():
            return False
    return True


def load_model(checkpoint_path: Path, *, device: Any, torch: Any) -> Any:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    model_config = checkpoint_compatible_config(config if isinstance(config, EdgeGATConfig) else EdgeGATConfig(), state_dict)
    model = EdgeRelationGAT(model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_one_document(
    doc: dict[str, Any],
    *,
    index: int,
    output_dir: Path,
    model: Any,
    decoder: TreeDecoder,
    device: Any,
    torch: Any,
    pdflatex: str,
    compile_runs: int,
    compile_timeout: int,
    skip_compile: bool,
    renderer: str,
    render_table_crops: bool,
    model_version: str,
) -> dict[str, Any]:
    document_id = str(doc["document_id"])
    safe_id = safe_filename(document_id)
    doc_dir = output_dir / f"{index:02d}_{safe_id}"
    doc_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(str(doc["pdf_path"]))
    content_json = Path(str(doc["content_json"]))
    graph_path = Path(str(doc["graph_path"]))
    assert_v7_content_json(content_json, require_styles=True)

    original_pdf = doc_dir / "original.pdf"
    shutil.copy2(pdf_path, original_pdf)

    data = torch.load(graph_path, map_location=device, weights_only=False)
    assert_v7_graph_data(data, graph_path)
    with torch.no_grad():
        logits = model(data.to(device)).detach().cpu()
    logits_path = doc_dir / "edge_logits.pt"
    torch.save(logits, logits_path)
    predicted_relations_path = doc_dir / "predicted_relations.json"
    write_predicted_relations(
        predicted_relations_path,
        doc_id=document_id,
        graph_path=str(graph_path),
        edge_index=data.edge_index.detach().cpu(),
        scores=logits,
        threshold_config={
            "merge": float(decoder.config.merge_threshold),
            "parent_child": float(decoder.config.parent_threshold),
            "sibling": float(decoder.config.sibling_threshold),
        },
        model_version=model_version,
        include_logits=False,
    )

    node_records = load_node_records(content_json, data)
    local_decoder = TreeDecoder(
        TreeDecoderConfig(
            merge_threshold=decoder.config.merge_threshold,
            parent_threshold=decoder.config.parent_threshold,
            require_merge_argmax=decoder.config.require_merge_argmax,
            require_parent_argmax=decoder.config.require_parent_argmax,
            enable_layout_scope_continuation_merge=decoder.config.enable_layout_scope_continuation_merge,
            enable_list_continuation_merge=decoder.config.enable_list_continuation_merge,
            enable_family_aware_merge_policy=decoder.config.enable_family_aware_merge_policy,
            family_body_list_merge_threshold=decoder.config.family_body_list_merge_threshold,
            family_reference_merge_threshold=decoder.config.family_reference_merge_threshold,
            enable_family_aware_missing_candidate_merge=decoder.config.enable_family_aware_missing_candidate_merge,
            heading_skeleton_mode=decoder.config.heading_skeleton_mode,
            source_pdf=str(pdf_path) if render_table_crops else None,
            table_asset_output_dir=str(doc_dir / "assets") if render_table_crops else None,
            figure_asset_output_dir=str(doc_dir / "assets") if render_table_crops else None,
            table_asset_latex_prefix="assets",
            figure_asset_latex_prefix="assets",
        )
    )
    root = local_decoder.decode(node_records, data.edge_index.detach().cpu(), logits)
    relation_trace_path = doc_dir / "relation_trace_report.json"
    title = infer_document_title(node_records)
    if renderer != "ir":
        raise ValueError("Production E2E rendering only supports --renderer ir")
    tex = render_decoded_tree_with_ir_backend(
        root,
        node_records=node_records,
        content_json=content_json,
        pdf_path=pdf_path,
        source_tex_path=Path(str(doc["tex_path"])) if doc.get("tex_path") else None,
        document_id=document_id,
        title=title,
        document_metadata=getattr(data, "document_metadata", None),
        predicted_relations_path=predicted_relations_path,
        decoder_trace=local_decoder.last_trace,
        attribution_output_path=relation_trace_path,
        table_asset_output_dir=doc_dir / "assets" if render_table_crops else None,
        figure_asset_output_dir=doc_dir / "assets" if render_table_crops else None,
        asset_latex_prefix="assets",
    )
    tex_path = doc_dir / "generated.tex"
    tex_path.write_text(tex, encoding="utf-8")

    compile_info: dict[str, Any] = {"skipped": bool(skip_compile)}
    generated_pdf = doc_dir / "generated.pdf"
    if not skip_compile:
        compile_info = compile_tex(
            tex_path,
            pdflatex=pdflatex,
            runs=compile_runs,
            timeout=compile_timeout,
        )

    source_pdf = doc_dir / f"{safe_id}_original.pdf"
    source_generated_pdf = doc_dir / f"{safe_id}_generated.pdf"
    shutil.copy2(original_pdf, source_pdf)
    if generated_pdf.exists():
        shutil.copy2(generated_pdf, source_generated_pdf)

    row = {
        "document_id": document_id,
        "doc_dir": str(doc_dir),
        "source_pdf": str(pdf_path),
        "source_graph": str(graph_path),
        "source_content_json": str(content_json),
        "edge_logits": str(logits_path),
        "predicted_relations": str(predicted_relations_path),
        "relation_trace_report": str(relation_trace_path),
        "original_pdf": str(original_pdf),
        "generated_tex": str(tex_path),
        "generated_pdf": str(generated_pdf),
        "paired_original_pdf": str(source_pdf),
        "paired_generated_pdf": str(source_generated_pdf) if source_generated_pdf.exists() else None,
        "generated_pdf_exists": generated_pdf.exists(),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "compile": compile_info,
        "renderer": renderer,
    }
    write_json(doc_dir / "qa_record.json", row)
    return row


def render_decoded_tree_with_ir_backend(
    root: ResolvedNode,
    *,
    node_records: list[dict[str, Any]],
    content_json: Path,
    pdf_path: Path | None,
    source_tex_path: Path | None,
    document_id: str,
    title: str | None,
    document_metadata: dict[str, Any] | None,
    predicted_relations_path: Path | None,
    table_asset_output_dir: Path | None,
    figure_asset_output_dir: Path | None,
    asset_latex_prefix: str,
    decoder_trace: dict[str, Any] | None = None,
    attribution_output_path: Path | None = None,
) -> str:
    """Render a TreeDecoder result through the canonical decoupled IR backend."""

    document = build_document_ir_from_full_v7(
        content_json=content_json,
        pdf_path=pdf_path,
        document_id=document_id,
    )
    document_title = infer_document_title_from_document(document) or title
    body_root = root_without_redundant_document_title(root, document_title) if document_title else root
    body_root = root_with_document_toc(body_root, document_metadata)
    apply_heading_render_policy(body_root)
    graph_index_to_node_ids = {
        index: v7_source_node_ids_for_graph_record(record, fallback_position=index)
        for index, record in enumerate(node_records)
    }
    tree = render_tree_from_resolved_root(
        body_root,
        document_ir_path=str(content_json),
        document_id=document.doc_id,
        graph_index_to_node_ids=graph_index_to_node_ids,
        predicted_relations_path=str(predicted_relations_path) if predicted_relations_path else None,
    )
    tree = inject_full_v7_front_matter(tree, document)
    if attribution_output_path is not None:
        write_json(
            attribution_output_path,
            build_relation_trace_report(
                document_id=document.doc_id,
                decoder_trace=decoder_trace or {},
                tree=tree,
            ),
        )
    from src.generation.ir_renderer import IRLatexRenderConfig

    config = IRLatexRenderConfig(
        title=document_title,
        front_matter_mode="original_like",
        include_maketitle=False,
        table_asset_output_dir=table_asset_output_dir,
        figure_asset_output_dir=figure_asset_output_dir,
        table_asset_latex_prefix=asset_latex_prefix,
        figure_asset_latex_prefix=asset_latex_prefix,
    )
    return render_original_like_document(document, tree, config=config, source_tex_path=source_tex_path)


def build_document_ir_from_full_v7(
    *,
    content_json: Path,
    pdf_path: Path | None,
    document_id: str,
) -> DocumentIR:
    payload = json.loads(content_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected v7 content object: {content_json}")
    return convert_v7_payload_to_document_ir(
        payload,
        source_path=content_json,
        pdf_path=pdf_path,
        doc_id=document_id,
        config=MinerUV7DocumentIRAdapterConfig(require_styles=False),
    )


def build_document_ir_from_graph_records(
    node_records: list[dict[str, Any]],
    *,
    content_json: Path,
    pdf_path: Path,
    document_id: str,
) -> Any:
    """Disabled graph-record helper kept only to make old imports fail loudly.

    The production renderer must build DocumentIR from the complete v7 JSON and
    bridge GNN predictions back through graph-side ``gnn_to_v7_ids``.  Rendering
    from graph-visible records drops metadata, notes, floats and header/footer
    evidence, so this path is intentionally blocked.
    """

    raise RuntimeError(
        "Graph-record-only DocumentIR rendering is disabled. Production rendering "
        "must use build_document_ir_from_full_v7() plus the exact graph "
        "gnn_to_v7_ids bridge."
    )

    payload = {
        "schema_version": "content_v7_graph_visible",
        "source_format": "graph_node_records",
        "document_id": document_id,
        "items": node_records,
        "style_source_pdf": str(pdf_path),
    }
    return convert_v7_payload_to_document_ir(
        payload,
        source_path=content_json,
        pdf_path=pdf_path,
        doc_id=document_id,
        config=MinerUV7DocumentIRAdapterConfig(require_styles=False),
    )


def render_tree_from_resolved_root(
    root: ResolvedNode,
    *,
    document_ir_path: str,
    document_id: str,
    graph_index_to_node_ids: dict[int, list[str]],
    predicted_relations_path: str | None = None,
) -> RenderTreeIR:
    nodes: list[RenderTreeNode] = []
    hoisted_front_matter_ids: list[str] = []

    def visit(node: ResolvedNode, *, is_root: bool = False, parent_role: RenderRole | None = None) -> str | None:
        render_id = "root" if is_root else f"r_{node.node_id}"
        if is_root:
            children = [
                child_id
                for child in sorted_render_children(node.children)
                for child_id in [visit(child, parent_role=RenderRole.ROOT)]
                if child_id is not None
            ]
            nodes.append(RenderTreeNode(render_id=render_id, role=RenderRole.ROOT, children=children))
            return render_id
        if is_noise_resolved_node(node):
            return None
        source_node_ids = [
            node_id
            for index in source_indexes_for_resolved_node(node)
            for node_id in graph_index_to_node_ids.get(index, [])
        ]
        source_node_ids = list(dict.fromkeys(source_node_ids))
        role = render_role_for_resolved_node(node, parent_role=parent_role)
        children = [
            child_id
            for child in sorted_render_children(node.children)
            for child_id in [visit(child, parent_role=role)]
            if child_id is not None
        ]
        latex = latex_override_for_resolved_node(node, role)
        text = None if latex else node.text
        attributes = render_attributes_for_resolved_node(node)
        source_graph_indexes = source_indexes_for_resolved_node(node)
        if source_graph_indexes:
            attributes["source_graph_indexes"] = source_graph_indexes
        if len(node.merged_node_ids or []) > 1:
            attributes["merged_source_graph_indexes"] = list(node.merged_node_ids)
            attributes["merge_source"] = str(node.record.get("_merge_source") or "unknown")
        nodes.append(
            RenderTreeNode(
                render_id=render_id,
                role=role,
                source_node_ids=source_node_ids,
                text=text,
                latex=latex,
                children=children,
                attributes=attributes,
            )
        )
        if parent_role not in {None, RenderRole.ROOT} and role in {
            RenderRole.DOCUMENT_TITLE,
            RenderRole.AUTHOR_BLOCK,
            RenderRole.ABSTRACT,
        }:
            hoisted_front_matter_ids.append(render_id)
            return None
        return render_id

    root_id = visit(root, is_root=True)
    if hoisted_front_matter_ids:
        for index, render_node in enumerate(nodes):
            if render_node.render_id != root_id:
                continue
            merged = list(dict.fromkeys([*hoisted_front_matter_ids, *render_node.children]))
            nodes[index] = RenderTreeNode(
                render_id=render_node.render_id,
                role=render_node.role,
                source_node_ids=render_node.source_node_ids,
                text=render_node.text,
                latex=render_node.latex,
                children=merged,
                attributes=render_node.attributes,
            )
            break
    return RenderTreeIR(
        doc_id=document_id,
        root_id=root_id,
        nodes=nodes,
        document_ir_path=document_ir_path,
        predicted_relations_path=predicted_relations_path,
        metadata={"adapter": "TreeDecoderResolvedNodeToRenderTreeIR"},
    )


def v7_source_node_ids_for_graph_record(record: dict[str, Any], *, fallback_position: int) -> list[str]:
    values = record.get("_v7_source_node_ids")
    if isinstance(values, list):
        ids = [value for value in values if isinstance(value, str) and value]
        if ids:
            return list(dict.fromkeys(ids))
    value = record.get("_v7_node_id")
    if isinstance(value, str) and value:
        return [value]
    return [stable_node_id(record, fallback_position=fallback_position)]


def build_relation_trace_report(
    *,
    document_id: str,
    decoder_trace: dict[str, Any],
    tree: RenderTreeIR,
) -> dict[str, Any]:
    """Join decoder attribution with the concrete RenderTreeIR surface.

    The decoder knows which graph indexes were merged or parented.  The render
    tree knows which resolved nodes survived as render nodes.  This read-only
    report answers the main audit question: did accepted relations actually
    become render-tree structure?
    """

    render_nodes = {node.render_id: node for node in tree.nodes}
    render_merged_nodes = []
    render_parent_sources: dict[str, int] = {}
    source_index_to_render_ids: dict[int, list[str]] = {}
    for node in tree.nodes:
        attrs = node.attributes or {}
        parent_source = str(attrs.get("parent_source") or ("root" if node.render_id == tree.root_id else "unknown"))
        render_parent_sources[parent_source] = render_parent_sources.get(parent_source, 0) + 1
        for raw_index in attrs.get("source_graph_indexes") or []:
            if isinstance(raw_index, int):
                source_index_to_render_ids.setdefault(raw_index, []).append(node.render_id)
        merged_indexes = attrs.get("merged_source_graph_indexes") or []
        if isinstance(merged_indexes, list) and len(merged_indexes) > 1:
            render_merged_nodes.append(
                {
                    "render_id": node.render_id,
                    "role": node.role.value if hasattr(node.role, "value") else str(node.role),
                    "source_graph_indexes": attrs.get("source_graph_indexes") or [],
                    "merged_source_graph_indexes": merged_indexes,
                    "source_node_ids": node.source_node_ids,
                    "merge_source": attrs.get("merge_source"),
                    "text_preview": str(node.text or "")[:240],
                }
            )

    merge_components = list(decoder_trace.get("merge_components") or [])
    accepted_not_rendered = []
    rendered_components = []
    duplicate_rendered_merge_members = []
    for component in merge_components:
        members = component.get("member_node_ids") or []
        if not isinstance(members, list):
            continue
        member_set = {int(value) for value in members if isinstance(value, int)}
        matching_render_ids = []
        for node in tree.nodes:
            attrs = node.attributes or {}
            render_members = attrs.get("merged_source_graph_indexes") or attrs.get("source_graph_indexes") or []
            if not isinstance(render_members, list):
                continue
            render_set = {int(value) for value in render_members if isinstance(value, int)}
            if member_set and member_set.issubset(render_set):
                matching_render_ids.append(node.render_id)
        updated = dict(component)
        updated["render_tree_node_ids"] = matching_render_ids
        updated["rendered_once"] = len(matching_render_ids) == 1
        if matching_render_ids:
            rendered_components.append(updated)
        else:
            accepted_not_rendered.append(updated)
        member_render_hits = {
            member: source_index_to_render_ids.get(member, [])
            for member in sorted(member_set)
            if source_index_to_render_ids.get(member)
        }
        component_render_id_set = set(matching_render_ids)
        duplicate_member_hits = {
            member: [render_id for render_id in render_ids if render_id not in component_render_id_set]
            for member, render_ids in member_render_hits.items()
            if any(render_id not in component_render_id_set for render_id in render_ids)
        }
        if duplicate_member_hits:
            duplicate_rendered_merge_members.append(
                {
                    "component": component,
                    "component_render_ids": matching_render_ids,
                    "member_render_hits_outside_component": duplicate_member_hits,
                }
            )

    node_parent_map: dict[str, str] = {}
    for parent in tree.nodes:
        for child_id in parent.children:
            node_parent_map[child_id] = parent.render_id

    return {
        "schema_version": "relation_trace_report_v1",
        "doc_id": document_id,
        "predicted_relations_path": tree.predicted_relations_path,
        "render_tree": {
            "node_count": len(tree.nodes),
            "root_id": tree.root_id,
            "merged_node_count": len(render_merged_nodes),
            "parent_source_distribution": render_parent_sources,
            "source_index_to_render_ids": {
                str(index): render_ids
                for index, render_ids in sorted(source_index_to_render_ids.items())
            },
        },
        "decoder_trace": decoder_trace,
        "merge_rendering": {
            "accepted_merge_component_count": len(merge_components),
            "render_tree_merged_node_count": len(render_merged_nodes),
            "rendered_merge_components": rendered_components,
            "accepted_merges_not_in_render_tree": accepted_not_rendered,
            "duplicate_rendered_merge_members": duplicate_rendered_merge_members,
            "render_tree_merged_nodes": render_merged_nodes,
        },
        "parent_rendering": {
            "render_tree_parent_source_distribution": render_parent_sources,
            "render_tree_parent_map_sample": dict(list(node_parent_map.items())[:200]),
            "gnn_parent_render_node_count": render_parent_sources.get("gnn_parent", 0),
            "stack_parent_render_node_count": (
                render_parent_sources.get("active_heading_stack", 0)
                + render_parent_sources.get("heading_stack_scope", 0)
                + render_parent_sources.get("heading_stack_outline", 0)
            ),
        },
    }


def inject_full_v7_front_matter(tree: RenderTreeIR, document: DocumentIR) -> RenderTreeIR:
    """Prepend complete v7 front matter that is intentionally absent from GNN view."""

    render_nodes = list(tree.nodes)
    existing_source_ids = {
        source_id
        for node in render_nodes
        for source_id in node.source_node_ids
    }
    metadata_nodes = [
        node
        for node in sorted(document.nodes, key=lambda item: item.reading_index)
        if str(node.metadata.get("layout_layer") or "").casefold() == "metadata_layer"
        and node.node_id not in existing_source_ids
    ]
    if not metadata_nodes:
        metadata_nodes = [
            node
            for node in infer_fallback_front_matter_nodes(document)
            if node.node_id not in existing_source_ids
        ]
    if not metadata_nodes:
        return tree

    abstract_label = next((node for node in metadata_nodes if _document_node_is_abstract_label(node)), None)
    title_nodes = [
        node
        for node in metadata_nodes
        if _document_node_is_title(node) or _fallback_document_title_node(node, abstract_label=abstract_label)
    ]
    author_nodes = [
        node
        for node in metadata_nodes
        if _document_node_is_author_like(node)
        or _metadata_front_matter_before_abstract(node, abstract_label)
        or _fallback_author_node(node, title_nodes=title_nodes, abstract_label=abstract_label)
    ]
    abstract_body_nodes = _front_matter_abstract_body_nodes(metadata_nodes, abstract_label=abstract_label)

    injected: list[RenderTreeNode] = []
    if title_nodes and not _render_tree_has_role(tree, RenderRole.DOCUMENT_TITLE):
        injected.append(
            RenderTreeNode(
                render_id="full_v7_document_title",
                role=RenderRole.DOCUMENT_TITLE,
                source_node_ids=[node.node_id for node in title_nodes],
            )
        )
    if author_nodes:
        # Keep the complete visual author/affiliation grid as one render node.
        # If we inject one RenderTree node per author box, the IR renderer may
        # sort those nodes by their full-v7 reading indexes and split a right
        # column author block into the body.  One source-id bundle preserves the
        # front-matter atomicity while still letting the author renderer recover
        # rows/columns from each source node's bbox.
        injected.append(
            RenderTreeNode(
                render_id="full_v7_author_block",
                role=RenderRole.AUTHOR_BLOCK,
                source_node_ids=[
                    node.node_id
                    for node in sorted(author_nodes, key=_document_node_visual_key)
                ],
            )
        )
    if abstract_label is not None or abstract_body_nodes:
        child_ids: list[str] = []
        for index, node in enumerate(abstract_body_nodes):
            render_id = f"full_v7_abstract_body_{index}"
            child_ids.append(render_id)
            injected.append(
                RenderTreeNode(
                    render_id=render_id,
                    role=RenderRole.PARAGRAPH,
                    source_node_ids=[node.node_id],
                )
            )
        injected.append(
            RenderTreeNode(
                render_id="full_v7_abstract",
                role=RenderRole.ABSTRACT,
                source_node_ids=[abstract_label.node_id] if abstract_label is not None else [],
                children=child_ids,
            )
        )
    if not injected:
        return tree

    injected_ids = [node.render_id for node in injected if node.role != RenderRole.PARAGRAPH]
    new_nodes = [*render_nodes, *injected]
    updated_nodes: list[RenderTreeNode] = []
    for node in new_nodes:
        if node.render_id != tree.root_id:
            updated_nodes.append(node)
            continue
        updated_nodes.append(
            RenderTreeNode(
                render_id=node.render_id,
                role=node.role,
                source_node_ids=node.source_node_ids,
                text=node.text,
                latex=node.latex,
                children=list(dict.fromkeys([*injected_ids, *node.children])),
                attributes=node.attributes,
            )
        )
    return RenderTreeIR(
        doc_id=tree.doc_id,
        root_id=tree.root_id,
        nodes=updated_nodes,
        document_ir_path=tree.document_ir_path,
        predicted_relations_path=tree.predicted_relations_path,
        style_profile_path=tree.style_profile_path,
        metadata={**tree.metadata, "front_matter_source": "full_v7_ir"},
    )


def infer_document_title_from_document(document: DocumentIR) -> str | None:
    for node in sorted(document.nodes, key=lambda item: item.reading_index):
        if _document_node_is_title(node) and node.text:
            return node.text
    for node in sorted(document.nodes, key=lambda item: item.reading_index):
        if str(node.metadata.get("layout_layer") or "").casefold() == "metadata_layer" and node.text:
            return node.text
    return None


def _render_tree_has_role(tree: RenderTreeIR, role: RenderRole) -> bool:
    return any(node.role == role for node in tree.nodes)


def _document_node_is_title(node: DocumentNode) -> bool:
    role = str(node.metadata.get("layout_role") or "").casefold()
    return role in {"document_title", "front_matter_title"}


def _document_node_is_author_like(node: DocumentNode) -> bool:
    role = str(node.metadata.get("layout_role") or "").casefold()
    return role in {"affiliation", "author", "authors", "date", "email", "correspondence"}


def infer_fallback_front_matter_nodes(document: DocumentIR) -> list[DocumentNode]:
    """Recover front matter for v7 files without layout_layer metadata.

    Some older v7 artifacts retain all full-v7 facts but do not annotate
    ``layout_layer=metadata_layer``.  The GNN view intentionally starts at the
    first body heading in those files, so rendering must infer the title,
    authors and abstract from the complete v7 reading order instead of dropping
    them.
    """

    nodes = sorted(document.nodes, key=lambda item: item.reading_index)
    abstract_label = next((node for node in nodes if _document_node_is_abstract_label(node)), None)
    if abstract_label is None:
        return []
    first_body_heading = next(
        (
            node
            for node in nodes
            if node.reading_index > abstract_label.reading_index
            and node.node_type == BlockType.TITLE
            and normalize_render_text(node.text) not in {"abstract", "keywords", "index terms"}
        ),
        None,
    )
    limit = first_body_heading.reading_index if first_body_heading is not None else abstract_label.reading_index + 1
    return [
        node
        for node in nodes
        if node.reading_index < limit
        and str(node.metadata.get("layout_layer") or "").casefold() not in {"noise_layer", "annotation_layer"}
    ]


def _fallback_document_title_node(node: DocumentNode, *, abstract_label: DocumentNode | None) -> bool:
    if abstract_label is None or node.reading_index >= abstract_label.reading_index:
        return False
    if node.node_type != BlockType.TITLE:
        return False
    text_key = normalize_render_text(node.text)
    return bool(text_key and text_key not in {"abstract", "keywords", "index terms"})


def _fallback_author_node(
    node: DocumentNode,
    *,
    title_nodes: list[DocumentNode],
    abstract_label: DocumentNode | None,
) -> bool:
    if abstract_label is None or node.reading_index >= abstract_label.reading_index:
        return False
    if node.node_type == BlockType.TITLE:
        return False
    if any(node.node_id == title.node_id for title in title_nodes):
        return False
    if title_nodes and node.reading_index <= max(title.reading_index for title in title_nodes):
        return False
    return bool((node.text or "").strip())


def _metadata_front_matter_before_abstract(node: DocumentNode, abstract_label: DocumentNode | None) -> bool:
    """Treat unclassified pre-abstract front matter as author/affiliation grid.

    MinerU sometimes labels a complete visual author cell as generic
    ``front_matter`` instead of ``author``/``affiliation``.  If it appears after
    the document title but before the abstract label, it belongs to the front
    matter grid, not the abstract body.
    """

    if abstract_label is None:
        return False
    role = str(node.metadata.get("layout_role") or "").casefold()
    layer = str(node.metadata.get("layout_layer") or "").casefold()
    if layer != "metadata_layer" or role not in {"front_matter", ""}:
        return False
    if node.reading_index >= abstract_label.reading_index:
        return False
    return bool((node.text or "").strip())


def _document_node_is_abstract_label(node: DocumentNode) -> bool:
    role = str(node.metadata.get("layout_role") or "").casefold()
    text = normalize_render_text(node.text)
    return role in {"abstract", "abstract_title"} or text == "abstract"


def _front_matter_abstract_body_nodes(
    metadata_nodes: list[DocumentNode],
    *,
    abstract_label: DocumentNode | None,
) -> list[DocumentNode]:
    if abstract_label is None:
        return []
    body: list[DocumentNode] = []
    for node in metadata_nodes:
        if node.reading_index <= abstract_label.reading_index:
            continue
        if _document_node_is_title(node) or _document_node_is_author_like(node) or _document_node_is_abstract_label(node):
            continue
        role = str(node.metadata.get("layout_role") or "").casefold()
        if role in {"front_matter", "abstract_body", ""} and node.text:
            body.append(node)
    return sorted(body, key=lambda item: item.reading_index)


def _document_node_visual_key(node: DocumentNode) -> tuple[int, float, float, int]:
    if node.bboxes:
        bbox = node.bboxes[0]
        return (node.page_idx, bbox.y0, bbox.x0, node.reading_index)
    return (node.page_idx, 0.0, 0.0, node.reading_index)


def source_indexes_for_resolved_node(node: ResolvedNode) -> list[int]:
    indexes: list[int] = []
    for key in (
        "float_group_source_node_ids",
        "figure_group_source_node_ids",
        "image_group_source_node_ids",
        "table_group_source_node_ids",
    ):
        value = node.record.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, int) and item >= 0:
                    indexes.append(item)
                elif isinstance(item, str) and item.isdigit():
                    indexes.append(int(item))
    for value in node.merged_node_ids or [node.node_id]:
        if isinstance(value, int) and value >= 0:
            indexes.append(value)
    for record in node.record.get("merged_records", []):
        if not isinstance(record, dict):
            continue
        gnn_index = record.get("_gnn_view_index")
        if isinstance(gnn_index, int) and gnn_index >= 0:
            indexes.append(gnn_index)
            continue

        # Legacy graph records used ``global_order`` as the graph node index.
        # In the decoupled v7 architecture, however, ``global_order`` belongs
        # to the complete fact layer and can point at metadata/front-matter
        # nodes that are intentionally absent from the GNN view.  Falling back
        # to it only for records that do not carry v7 mapping metadata keeps old
        # debug artifacts readable without corrupting the new relation bridge.
        has_v7_mapping = any(
            key in record
            for key in ("_v7_source_index", "_v7_node_id", "_v7_source_indexes", "_v7_source_node_ids")
        )
        if not has_v7_mapping:
            value = record.get("global_order")
            if isinstance(value, int) and value >= 0:
                indexes.append(value)
    return list(dict.fromkeys(indexes))


def render_role_for_resolved_node(node: ResolvedNode, parent_role: RenderRole | None = None) -> RenderRole:
    record = node.record
    layout_layer = str(record.get("layout_layer") or "").casefold()
    layout_role = str(record.get("layout_role") or record.get("role") or record.get("semantic_role") or "").casefold()
    text_key = normalize_render_text(node.text)
    if parent_role == RenderRole.ABSTRACT:
        return RenderRole.PARAGRAPH
    if layout_layer == "metadata_layer":
        if text_key == "abstract" or layout_role in {"abstract", "abstract_title"}:
            return RenderRole.ABSTRACT
        if layout_role in {"affiliation", "author", "authors", "date", "email", "correspondence"}:
            return RenderRole.AUTHOR_BLOCK
        if layout_role in {"front_matter_title", "document_title"}:
            return RenderRole.DOCUMENT_TITLE
    block_type = canonical_render_type(record)
    if block_type == "toc":
        return RenderRole.TOC_PLACEHOLDER
    if block_type == "title":
        if text_key == "abstract":
            return RenderRole.ABSTRACT
        if text_key in {"references", "bibliography"}:
            return RenderRole.REFERENCES
        if layout_layer == "metadata_layer":
            return RenderRole.DOCUMENT_TITLE
        if record.get("_appendix_heading"):
            return RenderRole.SECTION
        level = record.get("_skeleton_heading_level")
        try:
            level_int = int(level)
        except Exception:
            level_int = 1
        if level_int <= 1:
            return RenderRole.SECTION
        if level_int == 2:
            return RenderRole.SUBSECTION
        return RenderRole.SUBSUBSECTION
    if block_type == "equation":
        return RenderRole.DISPLAY_EQUATION
    if block_type == "inline_math":
        return RenderRole.INLINE_MATH
    if block_type == "table":
        return RenderRole.TABLE
    if block_type == "figure":
        return RenderRole.FIGURE
    if block_type == "algorithm":
        return RenderRole.ALGORITHM
    if block_type == "code":
        return RenderRole.CODE
    if block_type == "footnote":
        return RenderRole.FOOTNOTE
    if block_type == "margin_note":
        return RenderRole.MARGIN_NOTE
    if block_type == "reference":
        return RenderRole.REFERENCE_ITEM
    if block_type == "list":
        return RenderRole.LIST_ITEM
    return RenderRole.PARAGRAPH


def is_noise_resolved_node(node: ResolvedNode) -> bool:
    record = node.record
    layout_layer = str(record.get("layout_layer") or "").casefold()
    layout_role = str(record.get("layout_role") or record.get("role") or record.get("semantic_role") or "").casefold()
    raw = str(record.get("canonical_type") or record.get("type") or record.get("raw_type") or record.get("block_type") or "").casefold()
    if layout_layer == "noise_layer" or layout_role == "noise":
        return True
    return raw in {"page_header", "page_footer", "page_number", "header", "footer", "watermark"}


def normalize_render_text(text: str) -> str:
    value = " ".join(str(text or "").split()).casefold()
    value = value.strip(" .:;,-_")
    return value


def latex_override_for_resolved_node(node: ResolvedNode, role: RenderRole) -> str | None:
    if role not in {RenderRole.PARAGRAPH, RenderRole.SECTION, RenderRole.SUBSECTION, RenderRole.SUBSUBSECTION}:
        return None
    if not node.record.get("_render_as_paragraph_heading"):
        return None
    from src.generation.latex_helpers import escape_latex

    text = node.text.strip()
    return rf"\paragraph*{{{escape_latex(text)}}}" if text else None


def render_attributes_for_resolved_node(node: ResolvedNode) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    role = render_role_for_resolved_node(node)
    order_bias = render_order_bias_for_resolved_node(node, role)
    if order_bias is not None:
        attrs["render_order_bias"] = order_bias
    if node.record.get("_parent_source"):
        attrs["parent_source"] = str(node.record.get("_parent_source"))
    if node.record.get("_parent_node_id") is not None:
        attrs["parent_node_id"] = node.record.get("_parent_node_id")
    if node.record.get("_parent_edge_score") is not None:
        attrs["parent_edge_score"] = node.record.get("_parent_edge_score")
    if str(node.record.get("list_type") or "").casefold() in {"ordered", "enumerate", "numbered"}:
        attrs["ordered"] = True
    if node.record.get("_render_as_paragraph_heading"):
        attrs["paragraph_heading"] = True
    if node.record.get("_appendix_heading"):
        attrs["appendix_heading"] = True
    if node.record.get("_heading_unnumbered"):
        attrs["heading_unnumbered"] = True
    numbering = node.record.get("title_numbering")
    if isinstance(numbering, dict):
        attrs["title_numbering_style"] = str(numbering.get("style") or "none")
        attrs["title_numbering_token"] = str(numbering.get("token") or "")
        if numbering.get("level") is not None:
            attrs["title_numbering_level"] = numbering.get("level")
    for source_key, attr_key in (
        ("title_numbering_style", "title_numbering_style"),
        ("title_numbering_token", "title_numbering_token"),
        ("title_numbering_level", "title_numbering_level"),
    ):
        if source_key in node.record and attr_key not in attrs:
            attrs[attr_key] = node.record[source_key]
    return attrs


def render_order_bias_for_resolved_node(node: ResolvedNode, role: RenderRole) -> float | None:
    """Keep front matter before body even when graph order is noisy.

    MinerU's main reading flow can interleave right-column author blocks with
    the first body paragraph.  Rendering front matter from its visual bbox
    avoids producing ``author -> abstract -> body -> author`` layouts.
    """

    if role == RenderRole.DOCUMENT_TITLE:
        return -1_000_000.0
    if role == RenderRole.AUTHOR_BLOCK:
        x0, y0 = _node_bbox_origin(node.record)
        page = _numeric_attr(node.record.get("page_idx")) or 0.0
        return -900_000.0 + page * 10_000.0 + y0 * 10.0 + x0 / 10_000.0
    if role == RenderRole.ABSTRACT:
        page = _numeric_attr(node.record.get("page_idx")) or 0.0
        return -800_000.0 + page * 10_000.0
    return None


def _node_bbox_origin(record: dict[str, Any]) -> tuple[float, float]:
    bbox = record.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 2:
        try:
            return float(bbox[0]), float(bbox[1])
        except (TypeError, ValueError):
            pass
    return 0.0, 0.0


def _numeric_attr(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def compile_tex(tex_path: Path, *, pdflatex: str, runs: int, timeout: int) -> dict[str, Any]:
    log_path = tex_path.with_name("compile.log")
    outputs: list[dict[str, Any]] = []
    for run_index in range(1, max(1, runs) + 1):
        cmd = [
            pdflatex,
            "-interaction=nonstopmode",
            "-file-line-error",
            tex_path.name,
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=tex_path.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            output = decode_process_output(completed.stdout)
            outputs.append({"run": run_index, "returncode": completed.returncode})
        except subprocess.TimeoutExpired as exc:
            output = decode_process_output(exc.stdout)
            outputs.append({"run": run_index, "returncode": None, "timeout": True})
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== pdflatex run {run_index} =====\n")
            handle.write(output)
            if not output.endswith("\n"):
                handle.write("\n")
    return {
        "pdflatex": pdflatex,
        "runs": outputs,
        "log_path": str(log_path),
        "pdf_exists": tex_path.with_suffix(".pdf").exists(),
    }


def decode_process_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def resolve_device(value: str, *, torch: Any) -> Any:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value).strip("_") or "doc"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
