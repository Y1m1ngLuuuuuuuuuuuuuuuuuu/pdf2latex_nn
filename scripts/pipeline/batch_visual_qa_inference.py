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
from src.pipeline.v7_contract import assert_v7_content_json, assert_v7_graph_data  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402
from src.reasoning.postprocess import TreeDecoder, TreeDecoderConfig  # noqa: E402


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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pdflatex", default="pdflatex")
    parser.add_argument("--compile-runs", type=int, default=2)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--clean-output-dir", action="store_true")
    parser.add_argument(
        "--render-table-crops",
        action="store_true",
        help="Generate table and figure crop images in each QA output assets/ directory. Disabled by default to save disk.",
    )
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
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
                render_table_crops=args.render_table_crops,
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
    render_table_crops: bool,
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

    node_records = load_node_records(content_json, data)
    local_decoder = TreeDecoder(
        TreeDecoderConfig(
            merge_threshold=decoder.config.merge_threshold,
            parent_threshold=decoder.config.parent_threshold,
            require_merge_argmax=decoder.config.require_merge_argmax,
            require_parent_argmax=decoder.config.require_parent_argmax,
            source_pdf=str(pdf_path) if render_table_crops else None,
            table_asset_output_dir=str(doc_dir / "assets") if render_table_crops else None,
            figure_asset_output_dir=str(doc_dir / "assets") if render_table_crops else None,
            table_asset_latex_prefix="assets",
            figure_asset_latex_prefix="assets",
        )
    )
    root = local_decoder.decode(node_records, data.edge_index.detach().cpu(), logits)
    title = infer_document_title(node_records)
    tex = local_decoder.render_document(root, title=title, document_metadata=getattr(data, "document_metadata", None))
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
        "original_pdf": str(original_pdf),
        "generated_tex": str(tex_path),
        "generated_pdf": str(generated_pdf),
        "paired_original_pdf": str(source_pdf),
        "paired_generated_pdf": str(source_generated_pdf) if source_generated_pdf.exists() else None,
        "generated_pdf_exists": generated_pdf.exists(),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "compile": compile_info,
    }
    write_json(doc_dir / "qa_record.json", row)
    return row


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
