#!/usr/bin/env python3
"""Run a minimal PDF-to-LaTeX E2E inference batch for visual QA.

The script accepts either raw PDFs or a v7 manifest.  When cached v7 content
and graph artifacts are present it reuses them; otherwise it runs the front-end
stages:

PDF -> MinerU -> content_v7 -> PyMuPDF styles -> SciBERT graph -> GNN logits
-> TreeDecoder -> LaTeX -> pdflatex/xelatex.
"""

from __future__ import annotations

import argparse
import json
import shutil
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.batch_visual_qa_inference import (  # noqa: E402
    load_model,
    resolve_device,
    run_one_document,
    safe_filename,
    write_json,
)
from scripts.pipeline.build_mini_dataset import (  # noqa: E402
    default_mineru_command,
    find_mineru_content_source,
    normalize_mineru_content_to_v2,
    run_command_with_process_group_timeout,
)
from scripts.pipeline.train_edge_gnn_full import split_indices  # noqa: E402
from src.perception.reading_order import build_content_v7, load_content_list_v2, write_json as write_content_json  # noqa: E402
from src.perception.style_spans import StyleConfig, enrich_content_with_styles  # noqa: E402
from src.pipeline.v7_contract import assert_v7_content_json, assert_v7_graph_data  # noqa: E402
from src.reasoning.graph_builder import GraphBuildConfig, build_graph_from_content_v7  # noqa: E402
from src.reasoning.postprocess import TreeDecoder, TreeDecoderConfig  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Manifest containing pdf_path/content_json/graph_path records")
    source.add_argument("--pdf", action="append", type=Path, default=[], help="Raw PDF path; repeatable")

    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained GNN checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Visual QA output directory")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--document-id", action="append", default=[], help="Explicit manifest document id; repeatable")
    parser.add_argument(
        "--prefer-complex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer held-out docs with tables/figures/equations/headings when content JSON is available.",
    )

    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models/huggingface/allenai/scibert_scivocab_uncased")
    parser.add_argument("--embedding-device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--mineru-output-dir", type=Path, default=PROJECT_ROOT / "data/02_mineru_outputs/mineru_output_e2e")
    parser.add_argument("--graph-output-dir", type=Path, default=PROJECT_ROOT / "data/06_graph_features_e2e")
    parser.add_argument("--frontend-work-dir", type=Path, default=PROJECT_ROOT / "data/09_eval_reports/e2e_frontend_work")
    parser.add_argument("--mineru-command", default=default_mineru_command())
    parser.add_argument("--mineru-timeout", type=int, default=1800)
    parser.add_argument("--force-front-end", action="store_true", help="Rebuild content and graph even if manifest paths exist")
    parser.add_argument("--force-mineru", action="store_true")
    parser.add_argument("--force-json", action="store_true")
    parser.add_argument("--force-graph", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--merge-threshold", type=float, default=0.42)
    parser.add_argument("--parent-threshold", type=float, default=0.53)
    parser.add_argument("--require-merge-argmax", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-parent-argmax", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pdflatex", default="pdflatex", help="pdflatex or xelatex executable")
    parser.add_argument("--compile-runs", type=int, default=2)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--clean-output-dir", action="store_true")
    parser.add_argument(
        "--render-table-crops",
        action="store_true",
        help="Generate table and figure crop images in generated assets/ directories. Disabled by default to save disk.",
    )
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
    if args.clean_output_dir and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.frontend_work_dir.mkdir(parents=True, exist_ok=True)
    args.graph_output_dir.mkdir(parents=True, exist_ok=True)

    docs = select_documents(args)
    if not docs:
        raise ValueError("No documents selected for E2E inference")

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
    for index, source_doc in enumerate(docs, start=1):
        doc_id = str(source_doc.get("document_id") or Path(str(source_doc.get("pdf_path", ""))).stem)
        try:
            doc = ensure_frontend_artifacts(source_doc, args)
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
            row["frontend"] = doc.get("frontend", {})
        except Exception as exc:  # noqa: BLE001 - keep the batch useful.
            doc_dir = args.output_dir / f"{index:02d}_{safe_filename(doc_id)}"
            doc_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "document_id": doc_id,
                "doc_dir": str(doc_dir),
                "generated_pdf_exists": False,
                "error": repr(exc),
            }
            write_json(doc_dir / "qa_record.json", row)
        summary.append(row)
        status = "ok" if row.get("generated_pdf_exists") else "no_pdf"
        print(f"[{index:02d}/{len(docs):02d}] {doc_id} {status} -> {row['doc_dir']}", flush=True)

    payload = {
        "schema_version": "e2e_inference_qa_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest) if args.manifest else None,
        "checkpoint": str(args.checkpoint),
        "limit": args.limit,
        "split": args.split,
        "merge_threshold": args.merge_threshold,
        "parent_threshold": args.parent_threshold,
        "require_merge_argmax": args.require_merge_argmax,
        "require_parent_argmax": args.require_parent_argmax,
        "documents": summary,
    }
    write_json(args.output_dir / "e2e_manifest.json", payload)
    return 0 if any(row.get("generated_pdf_exists") or row.get("generated_tex") for row in summary) else 2


def select_documents(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.pdf:
        return [
            {
                "document_id": path.stem,
                "pdf_path": str(path.resolve()),
            }
            for path in args.pdf[: args.limit]
        ]

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    docs = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(docs, list):
        raise ValueError(f"Expected manifest with documents list: {args.manifest}")
    docs = [doc for doc in docs if isinstance(doc, dict) and doc.get("pdf_path")]

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
        splits = split_indices(len(docs), args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)
        split_docs = [docs[index] for index in splits[args.split]]

    if args.prefer_complex:
        split_docs = sorted(split_docs, key=complexity_score, reverse=True)
    return split_docs[: args.limit]


def complexity_score(doc: dict[str, Any]) -> float:
    score = 0.0
    counts = doc.get("label_counts", {})
    if isinstance(counts, dict):
        score += 0.05 * float(counts.get("merge", counts.get("0", 0)) or 0)
        score += 0.01 * float(counts.get("parent_child", counts.get("1", 0)) or 0)
    path = doc.get("content_json")
    if not path or not Path(str(path)).exists():
        return score
    try:
        payload = json.loads(Path(str(path)).read_text(encoding="utf-8"))
        items = payload.get("items", payload if isinstance(payload, list) else [])
    except Exception:
        return score
    if not isinstance(items, list):
        return score
    types = [str(item.get("type") or item.get("canonical_type") or "").lower() for item in items if isinstance(item, dict)]
    score += 4.0 * sum(1 for value in types if value in {"table"})
    score += 2.0 * sum(1 for value in types if value in {"figure", "image", "chart"})
    score += 1.5 * sum(1 for value in types if "equation" in value or "formula" in value)
    score += 0.5 * sum(1 for value in types if value in {"title", "section", "subsection"})
    return score


def ensure_frontend_artifacts(source_doc: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    pdf_path = Path(str(source_doc["pdf_path"])).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc_id = str(source_doc.get("document_id") or pdf_path.stem)
    safe_id = safe_filename(doc_id)
    frontend: dict[str, Any] = {"document_id": doc_id}

    content_json = existing_path(source_doc.get("content_json"))
    if args.force_front_end or args.force_json or content_json is None:
        content_json = build_content_json_from_pdf(doc_id, pdf_path, args)
        frontend["content_rebuilt"] = True
    else:
        assert_v7_content_json(content_json, require_styles=True)
        frontend["content_rebuilt"] = False

    graph_path = existing_path(source_doc.get("graph_path"))
    if args.force_front_end or args.force_graph or graph_path is None:
        graph_path = args.graph_output_dir / f"{safe_id}_e2e_graph.pt"
        graph_config = GraphBuildConfig(
            model_path=args.model_path,
            max_length=args.max_length,
            stride=args.stride,
            batch_size=args.batch_size,
            embedding_device=args.embedding_device,
        )
        build_graph_from_content_v7(content_json, graph_path, graph_config)
        frontend["graph_rebuilt"] = True
    else:
        import torch

        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        assert_v7_graph_data(graph, graph_path)
        frontend["graph_rebuilt"] = False

    return {
        **source_doc,
        "document_id": doc_id,
        "pdf_path": str(pdf_path),
        "content_json": str(content_json.resolve()),
        "graph_path": str(graph_path.resolve()),
        "frontend": frontend,
    }


def build_content_json_from_pdf(doc_id: str, pdf_path: Path, args: argparse.Namespace) -> Path:
    safe_id = safe_filename(doc_id)
    auto_dir = args.mineru_output_dir / safe_id / "auto"
    v2_path = auto_dir / f"{safe_id}_content_list_v2.json"
    v7_path = auto_dir / f"{safe_id}_content_list_v7.json"
    styles_path = auto_dir / f"{safe_id}_content_list_v7_styles.json"
    if styles_path.exists() and not (args.force_front_end or args.force_json):
        assert_v7_content_json(styles_path, require_styles=True)
        return styles_path

    content_source = find_mineru_content_source(safe_id, args.mineru_output_dir)
    if content_source is None or args.force_front_end or args.force_mineru:
        run_mineru(doc_id=safe_id, pdf_path=pdf_path, args=args)
        content_source = find_mineru_content_source(safe_id, args.mineru_output_dir)
    if content_source is None:
        raise FileNotFoundError(f"MinerU did not produce content_list for {doc_id} under {args.mineru_output_dir}")

    content_v2 = normalize_mineru_content_to_v2(content_source, v2_path)
    v7_payload = build_content_v7(load_content_list_v2(content_v2))
    v7_payload["source_path"] = str(content_v2)
    write_content_json(v7_path, v7_payload)
    enrich_content_with_styles(v7_path, pdf_path, styles_path, StyleConfig())
    assert_v7_content_json(styles_path, require_styles=True)
    return styles_path


def run_mineru(*, doc_id: str, pdf_path: Path, args: argparse.Namespace) -> None:
    args.mineru_output_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "pdf": shlex.quote(str(pdf_path)),
        "pdf_path": shlex.quote(str(pdf_path)),
        "pdf_parent": shlex.quote(str(pdf_path.parent)),
        "doc_id": shlex.quote(doc_id),
        "document_id": shlex.quote(doc_id),
        "mineru_output_dir": shlex.quote(str(args.mineru_output_dir)),
        "output_dir": shlex.quote(str(args.mineru_output_dir)),
    }
    command = args.mineru_command.format(**values)
    log_path = args.frontend_work_dir / safe_filename(doc_id) / "mineru_command.log"
    print(f"[e2e] mineru id={doc_id} cmd={command}", flush=True)
    completed = run_command_with_process_group_timeout(
        command,
        cwd=PROJECT_ROOT,
        timeout=args.mineru_timeout,
        log_path=log_path,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"MinerU failed for {doc_id}: returncode={completed.returncode} "
            f"log={log_path} tail={completed.stdout[-2000:]}"
        )


def existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path if path.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
