#!/usr/bin/env python3
"""Run Nougat and evaluate its output through the shared comparison IR.

The goal of this script is not to compare OCR fidelity.  It compares document
structure after both systems have been converted into ``comparison_structure_v1``:
heading hierarchy, reading order, paragraph/list grouping, references, and
float/caption attachment.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comparison_structure import (  # noqa: E402
    latex_file_to_comparison,
    markdown_file_to_comparison,
    write_comparison_json,
)
from src.evaluation.structure_metrics import evaluate_comparison_structures  # noqa: E402


DEFAULT_MANIFEST = Path("data/00_manifests/v7_scope320_floatfix_mergeclean_biofilter_recall98_1976_20260512.json")
DEFAULT_NOUGAT_BIN = Path("baselines/nougat/.conda_env/bin/nougat")
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/nougat_comparison")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nougat-bin", type=Path, default=DEFAULT_NOUGAT_BIN)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--document-id", action="append", default=[], help="Optional explicit document id; repeatable.")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--model", default=None, help="Optional Nougat model tag.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional local Nougat checkpoint directory.")
    parser.add_argument("--pages", default=None, help="Optional Nougat page range for smoke tests, e.g. '1-2'.")
    parser.add_argument("--recompute", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-nougat", action="store_true", help="Reuse existing .mmd files in each output directory.")
    parser.add_argument("--clean-output-dir", action="store_true")
    parser.add_argument("--match-threshold", type=float, default=0.58)
    parser.add_argument("--timeout", type=int, default=1800)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.clean_output_dir and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    docs = select_documents(args)
    if not docs:
        raise ValueError("No documents selected for Nougat comparison")

    rows: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        row = run_one_document(index, len(docs), doc, args)
        rows.append(row)
        print_status(index, len(docs), row)

    payload = {
        "schema_version": "nougat_comparison_v1",
        "manifest": str(args.manifest),
        "nougat_bin": str(args.nougat_bin),
        "split": args.split,
        "limit": args.limit,
        "pages": args.pages,
        "match_threshold": args.match_threshold,
        "summary": summarize_rows(rows),
        "documents": rows,
    }
    write_json(args.output_dir / "nougat_comparison_manifest.json", payload)
    write_summary_csv(args.output_dir / "nougat_comparison_summary.csv", rows)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_one_document(index: int, total: int, doc: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    doc_id = str(doc.get("document_id") or Path(str(doc.get("pdf_path", f"doc_{index}"))).stem)
    doc_dir = args.output_dir / f"{index:02d}_{safe_filename(doc_id)}"
    doc_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = existing_path(doc.get("pdf_path") or doc.get("source_pdf"))
    tex_path = existing_path(doc.get("tex_path") or doc.get("main_tex") or doc.get("source_tex"))
    row: dict[str, Any] = {
        "document_id": doc_id,
        "doc_dir": str(doc_dir),
        "pdf_path": str(pdf_path) if pdf_path else None,
        "tex_path": str(tex_path) if tex_path else None,
    }
    if not pdf_path or not tex_path:
        row["error"] = "missing_pdf_or_tex"
        write_json(doc_dir / "nougat_record.json", row)
        return row

    try:
        mmd_path = find_existing_mmd(doc_dir, pdf_path)
        if not args.skip_nougat or not mmd_path:
            row["nougat"] = run_nougat(pdf_path, doc_dir, args)
            mmd_path = find_existing_mmd(doc_dir, pdf_path)
        if not mmd_path:
            raise FileNotFoundError(f"Nougat did not produce .mmd for {pdf_path}")
        row["nougat_mmd"] = str(mmd_path)

        gold = latex_file_to_comparison(tex_path, doc_id=doc_id)
        pred = markdown_file_to_comparison(mmd_path, doc_id=doc_id)
        gold_path = doc_dir / "gold_structure.json"
        pred_path = doc_dir / "nougat_structure.json"
        metrics_path = doc_dir / "structure_metrics.json"
        write_comparison_json(gold, gold_path)
        write_comparison_json(pred, pred_path)
        metrics = evaluate_comparison_structures(gold.to_dict(), pred.to_dict(), match_threshold=args.match_threshold)
        write_json(metrics_path, metrics)
        row.update(flatten_metrics(metrics))
        row.update(
            {
                "gold_structure": str(gold_path),
                "nougat_structure": str(pred_path),
                "structure_metrics": str(metrics_path),
            }
        )
    except Exception as exc:  # noqa: BLE001 - comparison batches should continue.
        row["error"] = repr(exc)
    write_json(doc_dir / "nougat_record.json", row)
    return row


def run_nougat(pdf_path: Path, doc_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    command = [
        str(args.nougat_bin),
        str(pdf_path),
        "--out",
        str(doc_dir),
        "--batchsize",
        str(args.batchsize),
    ]
    if args.recompute:
        command.append("--recompute")
    if args.pages:
        command.extend(["--pages", str(args.pages)])
    if args.model:
        command.extend(["--model", str(args.model)])
    if args.checkpoint:
        command.extend(["--checkpoint", str(args.checkpoint)])
    start = time.time()
    proc = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=args.timeout)
    log_path = doc_dir / "nougat.log"
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr,
        encoding="utf-8",
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "seconds": time.time() - start,
        "log_path": str(log_path),
    }


def find_existing_mmd(doc_dir: Path, pdf_path: Path) -> Path | None:
    candidates = [
        doc_dir / f"{pdf_path.stem}.mmd",
        doc_dir / f"{pdf_path.name}.mmd",
    ]
    candidates.extend(sorted(doc_dir.glob("*.mmd")))
    candidates.extend(sorted(doc_dir.glob("**/*.mmd")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


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
    if args.split != "all":
        split = split_indices(len(docs), args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)[args.split]
        docs = [docs[idx] for idx in split]
    selected = []
    for doc in docs:
        if existing_path(doc.get("pdf_path") or doc.get("source_pdf")) and existing_path(
            doc.get("tex_path") or doc.get("main_tex") or doc.get("source_tex")
        ):
            selected.append(doc)
        if len(selected) >= args.limit:
            break
    return selected


def split_indices(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    seed: int,
) -> dict[str, list[int]]:
    import random

    if total <= 0:
        return {"train": [], "val": [], "test": []}
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if train_end + int(total * val_ratio) + int(total * test_ratio) > total:
        val_end = min(val_end, total)
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_structure_score": metrics.get("macro_structure_score"),
        "heading_tree_accuracy": (metrics.get("heading_tree_accuracy") or {}).get("score"),
        "reading_order_accuracy": (metrics.get("reading_order_accuracy") or {}).get("score"),
        "paragraph_merge_f1": (metrics.get("paragraph_merge_f1") or {}).get("f1"),
        "section_attachment_f1": (metrics.get("section_attachment_f1") or {}).get("f1"),
        "reference_section_completeness": (metrics.get("reference_section_completeness") or {}).get("score"),
        "float_caption_attachment_accuracy": (metrics.get("float_caption_attachment_accuracy") or {}).get("score"),
        "generated_structure_validity": (metrics.get("generated_structure_validity") or {}).get("score"),
        "matched_blocks": ((metrics.get("matching") or {}).get("matched_blocks")),
        "gold_blocks": ((metrics.get("matching") or {}).get("gold_blocks")),
        "pred_blocks": ((metrics.get("matching") or {}).get("pred_blocks")),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "documents": len(rows),
        "completed": sum(1 for row in rows if row.get("structure_metrics")),
        "failed": sum(1 for row in rows if row.get("error")),
        "macro_structure_score": mean_value(row.get("macro_structure_score") for row in rows),
        "heading_tree_accuracy": mean_value(row.get("heading_tree_accuracy") for row in rows),
        "reading_order_accuracy": mean_value(row.get("reading_order_accuracy") for row in rows),
        "paragraph_merge_f1": mean_value(row.get("paragraph_merge_f1") for row in rows),
        "section_attachment_f1": mean_value(row.get("section_attachment_f1") for row in rows),
        "reference_section_completeness": mean_value(row.get("reference_section_completeness") for row in rows),
        "float_caption_attachment_accuracy": mean_value(row.get("float_caption_attachment_accuracy") for row in rows),
        "generated_structure_validity": mean_value(row.get("generated_structure_validity") for row in rows),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "document_id",
        "macro_structure_score",
        "heading_tree_accuracy",
        "reading_order_accuracy",
        "paragraph_merge_f1",
        "section_attachment_f1",
        "reference_section_completeness",
        "float_caption_attachment_accuracy",
        "generated_structure_validity",
        "matched_blocks",
        "gold_blocks",
        "pred_blocks",
        "nougat_mmd",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def print_status(index: int, total: int, row: dict[str, Any]) -> None:
    print(
        f"[{index:02d}/{total:02d}] {row.get('document_id')} "
        f"score={format_float(row.get('macro_structure_score'))} "
        f"heading={format_float(row.get('heading_tree_accuracy'))} "
        f"order={format_float(row.get('reading_order_accuracy'))} "
        f"err={row.get('error', '')}",
        flush=True,
    )


def existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def mean_value(values: Any) -> float | None:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def format_float(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def safe_filename(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "document"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
