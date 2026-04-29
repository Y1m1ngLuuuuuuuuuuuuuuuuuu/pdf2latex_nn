#!/usr/bin/env python3
"""Download and unpack an arXiv TeX source pool before compilation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.step0_build_compilable_arxiv_dataset import (  # noqa: E402
    DownloadFailed,
    JsonlLog,
    download_eprint,
    download_failure_status,
    find_main_tex,
    iter_metadata_candidates,
    load_ids_from_jsonl,
    now_iso,
    safe_id,
    unpack_source,
)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def count_ready_sources(pool_dir: Path) -> int:
    if not pool_dir.exists():
        return 0
    return sum(1 for path in pool_dir.iterdir() if path.is_dir() and any(path.rglob("*.tex")))


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_one(candidate: dict, args: argparse.Namespace, downloaded_log: JsonlLog, error_log: JsonlLog) -> dict:
    arxiv_id = candidate["arxiv_id"]
    safe = safe_id(arxiv_id)
    final_source_dir = args.pool_dir / safe

    if final_source_dir.exists() and any(final_source_dir.rglob("*.tex")) and not args.force:
        row = {
            **candidate,
            "status": "already_present",
            "source_dir": str(final_source_dir),
            "finished_at": now_iso(),
        }
        downloaded_log.append(row)
        return row

    work_dir = Path(tempfile.mkdtemp(prefix=f"{safe}_", dir=args.tmp_dir))
    archive_path = work_dir / "source.eprint"
    extracted_dir = work_dir / "source"
    try:
        try:
            size_bytes = download_eprint(arxiv_id, archive_path, args.retries, args.retry_sleep, args.download_timeout)
        except DownloadFailed as exc:
            row = {
                **candidate,
                "status": download_failure_status(exc),
                "error": str(exc)[:500],
                "finished_at": now_iso(),
            }
            error_log.append(row)
            return row

        unpack_source(archive_path, extracted_dir, args.max_unpacked_mb)
        main_tex = find_main_tex(extracted_dir)
        if main_tex is None:
            row = {**candidate, "status": "no_main_tex", "size_bytes": size_bytes, "finished_at": now_iso()}
            error_log.append(row)
            return row

        copy_tree(extracted_dir, final_source_dir)
        row = {
            **candidate,
            "status": "downloaded",
            "size_bytes": size_bytes,
            "main_tex": str(main_tex.relative_to(extracted_dir)),
            "source_dir": str(final_source_dir),
            "finished_at": now_iso(),
        }
        downloaded_log.append(row)
        return row
    except (OSError, RuntimeError, tarfile.TarError, zipfile.BadZipFile, urllib.error.URLError) as exc:
        row = {**candidate, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:500], "finished_at": now_iso()}
        error_log.append(row)
        return row
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--target-sources", type=int, default=4000)
    parser.add_argument("--candidate-limit", type=int, default=250000)
    parser.add_argument("--download-slots", type=int, default=24)
    parser.add_argument("--backlog", type=int, default=192)
    parser.add_argument("--download-timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--max-unpacked-mb", type=int, default=120)
    parser.add_argument("--run-name", default="arxiv_2025_source_pool_4000")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.pool_dir = args.data_root / "03_tex_source_pool"
    args.tmp_dir = args.data_root / "_tmp_arxiv_source_pool"
    report_dir = args.data_root / "09_eval_reports" / args.run_name
    args.pool_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = report_dir / "downloaded.jsonl"
    errors_path = report_dir / "errors.jsonl"
    progress_path = report_dir / "progress.json"
    downloaded_log = JsonlLog(downloaded_path)
    error_log = JsonlLog(errors_path)

    existing_ids = load_ids_from_jsonl(downloaded_path) | load_ids_from_jsonl(errors_path)
    ready_sources = count_ready_sources(args.pool_dir)
    status_counts: dict[str, int] = {}
    submitted = 0
    completed = 0
    started_at = now_iso()
    exhausted = False
    last_heartbeat = 0.0

    print(
        json.dumps(
            {
                "event": "start",
                "started_at": started_at,
                "metadata": str(args.metadata),
                "pool_dir": str(args.pool_dir),
                "target_sources": args.target_sources,
                "existing_ready_sources": ready_sources,
                "download_slots": args.download_slots,
                "backlog": args.backlog,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    candidates = (
        candidate
        for candidate in iter_metadata_candidates(args.metadata, args.year, args.candidate_limit)
        if args.force or candidate["arxiv_id"] not in existing_ids
    )

    pending = set()
    with ThreadPoolExecutor(max_workers=args.download_slots) as executor:
        while ready_sources < args.target_sources:
            while len(pending) < args.backlog and not exhausted and ready_sources + len(pending) < args.target_sources + args.backlog:
                try:
                    candidate = next(candidates)
                except StopIteration:
                    exhausted = True
                    break
                submitted += 1
                pending.add(executor.submit(prepare_one, candidate, args, downloaded_log, error_log))

            if not pending:
                break

            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                completed += 1
                status = str(row.get("status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                if status in {"downloaded", "already_present"}:
                    ready_sources += 1

            if done or time.monotonic() - last_heartbeat >= args.heartbeat_seconds:
                last_heartbeat = time.monotonic()
                payload = {
                    "started_at": started_at,
                    "updated_at": now_iso(),
                    "target_sources": args.target_sources,
                    "ready_sources": ready_sources,
                    "submitted": submitted,
                    "completed": completed,
                    "pending": len(pending),
                    "exhausted_candidates": exhausted,
                    "status_counts": status_counts,
                    "pool_dir": str(args.pool_dir),
                }
                write_progress(progress_path, payload)
                print(json.dumps({"event": "progress", **payload}, ensure_ascii=False, sort_keys=True), flush=True)

    summary = {
        "started_at": started_at,
        "finished_at": now_iso(),
        "target_sources": args.target_sources,
        "ready_sources": ready_sources,
        "submitted": submitted,
        "completed": completed,
        "status_counts": status_counts,
        "pool_dir": str(args.pool_dir),
        "downloaded_path": str(downloaded_path),
        "errors_path": str(errors_path),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_progress(progress_path, summary)
    print(json.dumps({"event": "finished", **summary}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if ready_sources >= args.target_sources else 2


if __name__ == "__main__":
    raise SystemExit(main())
