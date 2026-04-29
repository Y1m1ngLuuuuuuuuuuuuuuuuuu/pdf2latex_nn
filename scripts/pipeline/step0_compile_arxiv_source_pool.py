#!/usr/bin/env python3
"""Compile a previously downloaded arXiv TeX source pool."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.step0_build_compilable_arxiv_dataset import (  # noqa: E402
    JsonlLog,
    compile_tex,
    find_main_tex,
    load_ids_from_jsonl,
    now_iso,
    safe_id,
    tail_text,
)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def iter_pool_sources(pool_dir: Path, downloaded_manifest: Path | None) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    if downloaded_manifest and downloaded_manifest.exists():
        with downloaded_manifest.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") not in {"downloaded", "already_present"}:
                    continue
                arxiv_id = str(row.get("arxiv_id") or "")
                source_dir = Path(row.get("source_dir") or pool_dir / safe_id(arxiv_id))
                if not arxiv_id or arxiv_id in seen or not source_dir.exists():
                    continue
                rows.append({**row, "source_dir": str(source_dir)})
                seen.add(arxiv_id)

    for source_dir in sorted(pool_dir.iterdir() if pool_dir.exists() else []):
        if not source_dir.is_dir():
            continue
        arxiv_id = source_dir.name
        if arxiv_id in seen:
            continue
        rows.append({"arxiv_id": arxiv_id, "source_dir": str(source_dir), "status": "scanned_pool"})
        seen.add(arxiv_id)
    return rows


def compile_one(row: dict, args: argparse.Namespace, accepted_log: JsonlLog, rejected_log: JsonlLog) -> dict:
    arxiv_id = str(row["arxiv_id"])
    safe = safe_id(arxiv_id)
    source_dir = Path(row["source_dir"])
    final_source_dir = args.final_source_dir / safe
    final_pdf_path = args.pdf_dir / f"{safe}.pdf"
    log_path = args.report_dir / "logs" / f"{safe}.log"

    if final_source_dir.exists() and final_pdf_path.exists() and not args.force:
        accepted = {**row, "status": "already_compiled", "pdf": str(final_pdf_path), "source_dir": str(final_source_dir)}
        accepted_log.append(accepted)
        return accepted

    main_tex = source_dir / row.get("main_tex", "")
    if not main_tex.exists() or not main_tex.is_file():
        main_tex = find_main_tex(source_dir)
    if main_tex is None:
        rejected = {**row, "status": "no_main_tex", "finished_at": now_iso()}
        rejected_log.append(rejected)
        return rejected

    work_dir = Path(tempfile.mkdtemp(prefix=f"{safe}_compile_", dir=args.tmp_dir))
    try:
        ok, compile_log, pdf_path = compile_tex(main_tex, work_dir, args.compile_timeout, args.engine)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(tail_text(compile_log), encoding="utf-8", errors="ignore")
        if not ok or pdf_path is None:
            rejected = {
                **row,
                "status": "compile_failed",
                "main_tex": str(main_tex.relative_to(source_dir)),
                "log": str(log_path),
                "finished_at": now_iso(),
            }
            rejected_log.append(rejected)
            return rejected

        copy_tree(source_dir, final_source_dir)
        final_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, final_pdf_path)
        accepted = {
            **row,
            "status": "accepted",
            "main_tex": str(main_tex.relative_to(source_dir)),
            "pdf": str(final_pdf_path),
            "source_dir": str(final_source_dir),
            "log": str(log_path),
            "finished_at": now_iso(),
        }
        accepted_log.append(accepted)
        return accepted
    except subprocess.TimeoutExpired as exc:
        rejected = {**row, "status": "timeout", "error": str(exc)[:500], "finished_at": now_iso()}
        rejected_log.append(rejected)
        return rejected
    except Exception as exc:
        rejected = {**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:500], "finished_at": now_iso()}
        rejected_log.append(rejected)
        return rejected
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--source-limit", type=int, default=4000)
    parser.add_argument("--target-successes", type=int, default=3000)
    parser.add_argument("--compile-slots", type=int, default=32)
    parser.add_argument("--backlog", type=int, default=128)
    parser.add_argument("--compile-timeout", type=int, default=240)
    parser.add_argument("--engine", choices=["auto", "latexmk-pdf", "xelatex", "pdflatex", "lualatex"], default="latexmk-pdf")
    parser.add_argument("--download-run-name", default="arxiv_2025_source_pool_4000")
    parser.add_argument("--run-name", default="arxiv_2025_source_pool_compile")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser


def write_progress(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    args.pool_dir = args.data_root / "03_tex_source_pool"
    args.final_source_dir = args.data_root / "03_tex_sources"
    args.pdf_dir = args.data_root / "01_raw_pdfs"
    args.tmp_dir = args.data_root / "_tmp_arxiv_source_pool_compile"
    args.report_dir = args.data_root / "09_eval_reports" / args.run_name
    download_manifest = args.data_root / "09_eval_reports" / args.download_run_name / "downloaded.jsonl"
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "logs").mkdir(parents=True, exist_ok=True)

    accepted_path = args.report_dir / "accepted.jsonl"
    rejected_path = args.report_dir / "rejected.jsonl"
    accepted_log = JsonlLog(accepted_path)
    rejected_log = JsonlLog(rejected_path)
    processed_ids = load_ids_from_jsonl(accepted_path) | load_ids_from_jsonl(rejected_path)

    candidates = [
        row
        for row in iter_pool_sources(args.pool_dir, download_manifest)
        if args.force or row["arxiv_id"] not in processed_ids
    ][: args.source_limit]
    started_at = now_iso()
    submitted = 0
    completed = 0
    accepted_total = len(load_ids_from_jsonl(accepted_path))
    status_counts: dict[str, int] = {}
    pending = set()
    cursor = 0
    last_heartbeat = 0.0
    progress_path = args.report_dir / "progress.json"

    print(
        json.dumps(
            {
                "event": "start",
                "started_at": started_at,
                "pool_dir": str(args.pool_dir),
                "candidate_sources": len(candidates),
                "target_successes": args.target_successes,
                "compile_slots": args.compile_slots,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.compile_slots) as executor:
        while accepted_total < args.target_successes and (cursor < len(candidates) or pending):
            while len(pending) < args.backlog and cursor < len(candidates) and accepted_total < args.target_successes:
                row = candidates[cursor]
                cursor += 1
                submitted += 1
                pending.add(executor.submit(compile_one, row, args, accepted_log, rejected_log))
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                completed += 1
                status = str(result.get("status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                if status in {"accepted", "already_compiled"}:
                    accepted_total += 1

            if done or time.monotonic() - last_heartbeat >= args.heartbeat_seconds:
                last_heartbeat = time.monotonic()
                payload = {
                    "started_at": started_at,
                    "updated_at": now_iso(),
                    "target_successes": args.target_successes,
                    "accepted_total": accepted_total,
                    "submitted": submitted,
                    "completed": completed,
                    "pending": len(pending),
                    "remaining_candidates": len(candidates) - cursor,
                    "status_counts": status_counts,
                }
                write_progress(progress_path, payload)
                print(json.dumps({"event": "progress", **payload}, ensure_ascii=False, sort_keys=True), flush=True)

    summary = {
        "started_at": started_at,
        "finished_at": now_iso(),
        "target_successes": args.target_successes,
        "accepted_total": accepted_total,
        "submitted": submitted,
        "completed": completed,
        "status_counts": status_counts,
        "pool_dir": str(args.pool_dir),
        "accepted_path": str(accepted_path),
        "rejected_path": str(rejected_path),
    }
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_progress(progress_path, summary)
    print(json.dumps({"event": "finished", **summary}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if accepted_total >= args.target_successes else 2


if __name__ == "__main__":
    raise SystemExit(main())
