#!/usr/bin/env python3
"""Add compiled TeX sources back into the v7 source pool.

The v7 batch builder scans ``data/03_tex_source_pool``. Earlier compilation
steps may have accepted sources in ``data/03_tex_sources`` without mirroring
them back into that pool. This tool fills only the missing document-id
directories and never overwrites existing pool entries.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SyncCandidate:
    document_id: str
    pdf_path: Path
    source_dir: Path
    pool_dir: Path


@dataclass
class CopyStats:
    dirs: int = 0
    files: int = 0
    bytes_logical: int = 0
    hardlinked_files: int = 0
    copied_files: int = 0

    def add(self, other: "CopyStats") -> None:
        self.dirs += other.dirs
        self.files += other.files
        self.bytes_logical += other.bytes_logical
        self.hardlinked_files += other.hardlinked_files
        self.copied_files += other.copied_files


@dataclass
class SyncReport:
    started_at: str
    finished_at: str | None
    raw_pdf_dir: str
    compiled_source_dir: str
    source_pool_dir: str
    copy_mode: str
    dry_run: bool
    pdf_count: int
    compiled_source_count: int
    existing_pool_count: int
    eligible_count: int
    missing_count: int
    synced_count: int
    skipped_count: int
    error_count: int
    synced_ids: list[str]
    skipped: list[dict[str, str]]
    errors: list[dict[str, str]]
    copy_stats: dict[str, int]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-pdf-dir", type=Path, default=REPO_ROOT / "data/01_raw_pdfs")
    parser.add_argument("--compiled-source-dir", type=Path, default=REPO_ROOT / "data/03_tex_sources")
    parser.add_argument("--source-pool-dir", type=Path, default=REPO_ROOT / "data/03_tex_source_pool")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--limit", type=int, help="Sync at most this many missing source directories.")
    parser.add_argument("--report-json", type=Path, help="Optional JSON report path.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    started_at = utc_now()

    raw_pdf_dir = args.raw_pdf_dir.resolve()
    compiled_source_dir = args.compiled_source_dir.resolve()
    source_pool_dir = args.source_pool_dir.resolve()

    pdf_index = build_pdf_index(raw_pdf_dir)
    compiled_sources = build_direct_dir_index(compiled_source_dir)
    existing_pool = build_direct_dir_index(source_pool_dir)
    candidates = find_missing_candidates(pdf_index, compiled_sources, existing_pool, source_pool_dir)
    if args.limit is not None:
        candidates = candidates[: max(0, args.limit)]

    report = SyncReport(
        started_at=started_at,
        finished_at=None,
        raw_pdf_dir=str(raw_pdf_dir),
        compiled_source_dir=str(compiled_source_dir),
        source_pool_dir=str(source_pool_dir),
        copy_mode=str(args.copy_mode),
        dry_run=bool(args.dry_run),
        pdf_count=len(pdf_index),
        compiled_source_count=len(compiled_sources),
        existing_pool_count=len(existing_pool),
        eligible_count=len(set(pdf_index) & set(compiled_sources)),
        missing_count=len(candidates),
        synced_count=0,
        skipped_count=0,
        error_count=0,
        synced_ids=[],
        skipped=[],
        errors=[],
        copy_stats=asdict(CopyStats()),
    )

    log(
        args,
        "scan "
        f"pdfs={report.pdf_count} compiled_sources={report.compiled_source_count} "
        f"existing_pool={report.existing_pool_count} eligible={report.eligible_count} "
        f"missing={report.missing_count} dry_run={report.dry_run}",
    )

    if not args.dry_run:
        source_pool_dir.mkdir(parents=True, exist_ok=True)

    total_stats = CopyStats()
    started = time.monotonic()
    for idx, candidate in enumerate(candidates, start=1):
        if args.dry_run:
            report.synced_ids.append(candidate.document_id)
            report.synced_count += 1
            continue

        try:
            stats = sync_one(candidate, args.copy_mode)
        except Exception as exc:  # pragma: no cover - exercised by integration/runtime.
            report.error_count += 1
            report.errors.append(
                {
                    "document_id": candidate.document_id,
                    "source_dir": str(candidate.source_dir),
                    "pool_dir": str(candidate.pool_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            log(args, f"error id={candidate.document_id} {type(exc).__name__}: {exc}")
            continue

        total_stats.add(stats)
        report.synced_count += 1
        report.synced_ids.append(candidate.document_id)
        if idx == 1 or idx % 50 == 0 or idx == len(candidates):
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = idx / elapsed
            log(
                args,
                f"progress {idx}/{len(candidates)} synced={report.synced_count} "
                f"errors={report.error_count} rate={rate:.2f}/s",
            )

    report.copy_stats = asdict(total_stats)
    report.finished_at = utc_now()
    report.skipped_count = len(report.skipped)
    if args.report_json:
        args.report_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.report_json.resolve().write_text(
            json.dumps(asdict(report), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    log(
        args,
        "done "
        f"synced={report.synced_count} errors={report.error_count} "
        f"hardlinked_files={total_stats.hardlinked_files} copied_files={total_stats.copied_files}",
    )
    return 1 if report.error_count else 0


def build_pdf_index(raw_pdf_dir: Path) -> dict[str, Path]:
    pdfs: dict[str, Path] = {}
    if not raw_pdf_dir.exists():
        return pdfs
    for path in sorted(raw_pdf_dir.rglob("*.pdf")):
        if should_skip_path(path):
            continue
        pdfs.setdefault(path.stem, path)
    return pdfs


def build_direct_dir_index(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    return {path.name: path for path in sorted(root.iterdir()) if path.is_dir() and not should_skip_path(path)}


def find_missing_candidates(
    pdf_index: dict[str, Path],
    compiled_sources: dict[str, Path],
    existing_pool: dict[str, Path],
    source_pool_dir: Path,
) -> list[SyncCandidate]:
    candidates: list[SyncCandidate] = []
    for document_id in sorted(set(pdf_index) & set(compiled_sources)):
        if document_id in existing_pool:
            continue
        source_dir = compiled_sources[document_id]
        if not contains_tex_file(source_dir):
            continue
        candidates.append(
            SyncCandidate(
                document_id=document_id,
                pdf_path=pdf_index[document_id],
                source_dir=source_dir,
                pool_dir=source_pool_dir / document_id,
            )
        )
    return candidates


def contains_tex_file(path: Path) -> bool:
    try:
        return any(child.is_file() and child.suffix.lower() == ".tex" for child in path.rglob("*.tex"))
    except OSError:
        return False


def sync_one(candidate: SyncCandidate, copy_mode: str) -> CopyStats:
    if candidate.pool_dir.exists():
        return CopyStats()
    tmp_dir = candidate.pool_dir.parent / f".{candidate.pool_dir.name}.sync_tmp_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    try:
        stats = copy_tree(candidate.source_dir, tmp_dir, copy_mode)
        os.replace(tmp_dir, candidate.pool_dir)
        return stats
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def copy_tree(src: Path, dst: Path, copy_mode: str) -> CopyStats:
    stats = CopyStats(dirs=1)
    dst.mkdir(parents=True, exist_ok=False)
    for root, dirnames, filenames in os.walk(src):
        root_path = Path(root)
        rel_root = root_path.relative_to(src)
        dirnames[:] = [name for name in sorted(dirnames) if not should_skip_name(name)]
        filenames = [name for name in sorted(filenames) if not should_skip_name(name)]

        for dirname in dirnames:
            target_dir = dst / rel_root / dirname
            target_dir.mkdir(parents=True, exist_ok=True)
            stats.dirs += 1

        for filename in filenames:
            source_file = root_path / filename
            target_file = dst / rel_root / filename
            if source_file.is_symlink():
                copy_symlink(source_file, target_file)
                continue
            if not source_file.is_file():
                continue
            file_size = safe_file_size(source_file)
            stats.files += 1
            stats.bytes_logical += file_size
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if copy_mode == "hardlink":
                try:
                    os.link(source_file, target_file)
                    stats.hardlinked_files += 1
                    continue
                except OSError:
                    pass
            shutil.copy2(source_file, target_file)
            stats.copied_files += 1
    return stats


def copy_symlink(source_file: Path, target_file: Path) -> None:
    target_file.parent.mkdir(parents=True, exist_ok=True)
    link_target = os.readlink(source_file)
    os.symlink(link_target, target_file)


def safe_file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def should_skip_path(path: Path) -> bool:
    return any(should_skip_name(part) for part in path.parts)


def should_skip_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in {".ipynb_checkpoints", "__pycache__"} or "checkpoint" in lowered


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(args: argparse.Namespace, message: str) -> None:
    if not args.quiet:
        print(f"[source-pool-sync] {message}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
