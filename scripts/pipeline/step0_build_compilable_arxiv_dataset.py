#!/usr/bin/env python3
"""Build a compilable arXiv source dataset without interactive supervision.

The script streams arXiv metadata, downloads e-print source packages with many
workers, compiles each source in an isolated temporary directory, and keeps only
samples that successfully produce a PDF. It stops once the target number of
accepted samples is reached.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gzip
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError:  # pragma: no cover - server requirements include requests.
    requests = None


ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
ARXIV_ID_RE = re.compile(r"^(\d{4})\.\d{4,5}(v\d+)?$")
VERSION_SUFFIX_RE = re.compile(r"v\d+$")


@dataclasses.dataclass(frozen=True)
class DatasetPaths:
    data_root: Path
    source_dir: Path
    pdf_dir: Path
    report_dir: Path
    tmp_dir: Path

    @classmethod
    def from_data_root(cls, data_root: Path, run_name: str) -> "DatasetPaths":
        return cls(
            data_root=data_root,
            source_dir=data_root / "03_tex_sources",
            pdf_dir=data_root / "01_raw_pdfs",
            report_dir=data_root / "09_eval_reports" / run_name,
            tmp_dir=data_root / "_tmp_compilable_arxiv_build",
        )

    def ensure(self) -> None:
        for path in [
            self.source_dir,
            self.pdf_dir,
            self.report_dir,
            self.report_dir / "logs",
            self.tmp_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class JsonlLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


class AcceptGate:
    def __init__(self, target: int, initial: int) -> None:
        self.target = target
        self.count = initial
        self.lock = threading.Lock()

    def full(self) -> bool:
        with self.lock:
            return self.count >= self.target

    def reserve(self) -> int | None:
        with self.lock:
            if self.count >= self.target:
                return None
            self.count += 1
            return self.count


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def normalize_arxiv_id(raw_id: object) -> str | None:
    if raw_id is None:
        return None
    arxiv_id = str(raw_id).strip()
    if not arxiv_id:
        return None
    arxiv_id = arxiv_id.rstrip("/").rsplit("/", 1)[-1]
    arxiv_id = VERSION_SUFFIX_RE.sub("", arxiv_id)
    if not ARXIV_ID_RE.match(arxiv_id):
        return None
    return arxiv_id


def id_year(arxiv_id: str) -> int | None:
    match = ARXIV_ID_RE.match(arxiv_id)
    if not match:
        return None
    prefix = match.group(1)
    year = int(prefix[:2])
    return 2000 + year if year < 90 else 1900 + year


def record_year(record: dict, arxiv_id: str) -> int | None:
    for key in ["year", "published_year", "created_year"]:
        value = record.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    for key in ["created", "published", "submitted", "update_date", "updated"]:
        value = str(record.get(key) or "")
        match = re.search(r"\b(19|20)\d{2}\b", value)
        if match:
            return int(match.group(0))

    versions = record.get("versions") or []
    if isinstance(versions, list):
        for version in versions:
            if isinstance(version, dict):
                value = str(version.get("created") or "")
                match = re.search(r"\b(19|20)\d{2}\b", value)
                if match:
                    return int(match.group(0))

    return id_year(arxiv_id)


def iter_metadata_candidates(metadata_path: Path, year: int, limit: int) -> Iterable[dict]:
    emitted = 0
    seen: set[str] = set()
    with metadata_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if emitted >= limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            arxiv_id = normalize_arxiv_id(record.get("arxiv_id") or record.get("id"))
            if not arxiv_id or arxiv_id in seen:
                continue
            if record_year(record, arxiv_id) != year:
                continue
            seen.add(arxiv_id)
            emitted += 1
            yield {
                "arxiv_id": arxiv_id,
                "title": record.get("title"),
                "categories": record.get("categories"),
                "metadata_line": line_number,
            }


def load_ids_from_jsonl(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            arxiv_id = normalize_arxiv_id(row.get("arxiv_id"))
            if arxiv_id:
                ids.add(arxiv_id)
    return ids


def run_command(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def download_eprint(arxiv_id: str, output_path: Path, retries: int, retry_sleep: float, timeout: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id)
    last_error: Exception | None = None
    headers = {"User-Agent": "pdf2latex-nn-dataset-builder/0.1"}

    for attempt in range(1, retries + 1):
        part_path = output_path.with_suffix(output_path.suffix + ".part")
        part_path.unlink(missing_ok=True)
        try:
            if requests is not None:
                with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"HTTP {response.status_code}")
                    with part_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
            else:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}")
                    with part_path.open("wb") as handle:
                        shutil.copyfileobj(response, handle)
            if not part_path.exists() or part_path.stat().st_size == 0:
                raise RuntimeError("empty e-print response")
            part_path.replace(output_path)
            return output_path.stat().st_size
        except Exception as exc:  # noqa: PERF203 - retry loop clarity matters here.
            last_error = exc
            part_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(retry_sleep)

    raise RuntimeError(f"download failed: {last_error}")


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(output_dir)


def unpack_source(archive_path: Path, output_dir: Path, max_unpacked_mb: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            total_size = sum(member.size for member in archive.getmembers() if member.isfile())
        if total_size > max_unpacked_mb * 1024 * 1024:
            raise RuntimeError(f"source too large after unpack: {total_size} bytes")
        safe_extract_tar(archive_path, output_dir)
        return

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            total_size = sum(info.file_size for info in archive.infolist())
            if total_size > max_unpacked_mb * 1024 * 1024:
                raise RuntimeError(f"source too large after unpack: {total_size} bytes")
            archive.extractall(output_dir)
        return

    raw = archive_path.read_bytes()
    try:
        text = gzip.decompress(raw).decode("utf-8", errors="replace")
    except Exception:
        text = raw.decode("utf-8", errors="replace")
    if "\\documentclass" not in text and "\\begin{document}" not in text:
        raise RuntimeError("e-print response is not an archive or TeX document")
    (output_dir / "main.tex").write_text(text, encoding="utf-8")


def find_main_tex(source_dir: Path) -> Path | None:
    candidates: list[tuple[int, int, Path]] = []
    for path in source_dir.rglob("*.tex"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        score = 0
        if "\\documentclass" in text:
            score += 20
        if "\\begin{document}" in text:
            score += 20
        if "\\end{document}" in text:
            score += 10
        if path.name.lower() in {"main.tex", "paper.tex", "ms.tex", "article.tex"}:
            score += 5
        if path.parent == source_dir:
            score += 3
        if score >= 20:
            candidates.append((score, path.stat().st_size, path))
    candidates.sort(reverse=True)
    return candidates[0][2] if candidates else None


def compile_with_latexmk(main_tex: Path, output_dir: Path, timeout: int) -> tuple[bool, str, Path | None]:
    cmd = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir={output_dir}",
        str(main_tex),
    ]
    result = run_command(cmd, cwd=main_tex.parent, timeout=timeout)
    pdf_path = output_dir / f"{main_tex.stem}.pdf"
    return result.returncode == 0 and pdf_path.exists(), result.stdout, pdf_path if pdf_path.exists() else None


def compile_with_engine(main_tex: Path, output_dir: Path, timeout: int, engine: str) -> tuple[bool, str, Path | None]:
    logs: list[str] = []
    for pass_index in range(1, 3):
        cmd = [
            engine,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-output-directory={output_dir}",
            main_tex.name,
        ]
        result = run_command(cmd, cwd=main_tex.parent, timeout=timeout)
        logs.append(f"===== {engine} pass {pass_index} =====\n{result.stdout}")
        if result.returncode != 0:
            break
    pdf_path = output_dir / f"{main_tex.stem}.pdf"
    return pdf_path.exists(), "\n".join(logs), pdf_path if pdf_path.exists() else None


def compile_tex(main_tex: Path, output_dir: Path, timeout: int, engine: str) -> tuple[bool, str, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if engine == "latexmk-pdf":
        return compile_with_latexmk(main_tex, output_dir, timeout)
    if engine in {"xelatex", "pdflatex", "lualatex"}:
        return compile_with_engine(main_tex, output_dir, timeout, engine)

    logs: list[str] = []
    if shutil.which("latexmk"):
        ok, log, pdf_path = compile_with_latexmk(main_tex, output_dir, timeout)
        logs.append(log)
        if ok:
            return True, "\n".join(logs), pdf_path
    for fallback_engine in ["xelatex", "pdflatex", "lualatex"]:
        if not shutil.which(fallback_engine):
            continue
        ok, log, pdf_path = compile_with_engine(main_tex, output_dir, timeout, fallback_engine)
        logs.append(log)
        if ok:
            return True, "\n".join(logs), pdf_path
    return False, "\n".join(logs) or "no supported TeX compiler found", None


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def tail_text(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def process_candidate(
    candidate: dict,
    paths: DatasetPaths,
    args: argparse.Namespace,
    accepted_log: JsonlLog,
    rejected_log: JsonlLog,
    gate: AcceptGate,
    compile_slots: threading.Semaphore,
) -> dict:
    arxiv_id = candidate["arxiv_id"]
    safe = safe_id(arxiv_id)
    final_source_dir = paths.source_dir / safe
    final_pdf_path = paths.pdf_dir / f"{safe}.pdf"
    log_path = paths.report_dir / "logs" / f"{safe}.log"

    if gate.full():
        return {"arxiv_id": arxiv_id, "status": "skipped_target_reached"}

    if final_source_dir.exists() and final_pdf_path.exists() and not args.force:
        accepted_index = gate.reserve()
        if accepted_index is None:
            return {"arxiv_id": arxiv_id, "status": "skipped_target_reached"}
        row = {
            **candidate,
            "status": "already_present",
            "accepted_index": accepted_index,
            "pdf": str(final_pdf_path),
            "source_dir": str(final_source_dir),
            "finished_at": now_iso(),
        }
        accepted_log.append(row)
        return row

    work_dir = Path(tempfile.mkdtemp(prefix=f"{safe}_", dir=paths.tmp_dir))
    archive_path = work_dir / "source.eprint"
    extracted_dir = work_dir / "source"
    compile_dir = work_dir / "compile"
    started_at = now_iso()
    try:
        size_bytes = download_eprint(arxiv_id, archive_path, args.retries, args.retry_sleep, args.download_timeout)
        unpack_source(archive_path, extracted_dir, args.max_unpacked_mb)
        main_tex = find_main_tex(extracted_dir)
        if main_tex is None:
            row = {**candidate, "status": "no_main_tex", "size_bytes": size_bytes, "finished_at": now_iso()}
            rejected_log.append(row)
            return row

        with compile_slots:
            if gate.full():
                return {"arxiv_id": arxiv_id, "status": "skipped_target_reached"}
            ok, compile_log, pdf_path = compile_tex(main_tex, compile_dir, args.compile_timeout, args.engine)
        log_path.write_text(tail_text(compile_log), encoding="utf-8", errors="ignore")
        if not ok or pdf_path is None:
            row = {
                **candidate,
                "status": "compile_failed",
                "size_bytes": size_bytes,
                "main_tex": str(main_tex.relative_to(extracted_dir)),
                "log": str(log_path),
                "finished_at": now_iso(),
            }
            rejected_log.append(row)
            return row

        accepted_index = gate.reserve()
        if accepted_index is None:
            return {"arxiv_id": arxiv_id, "status": "skipped_target_reached"}
        copy_tree(extracted_dir, final_source_dir)
        final_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, final_pdf_path)
        row = {
            **candidate,
            "status": "accepted",
            "accepted_index": accepted_index,
            "size_bytes": size_bytes,
            "main_tex": str(main_tex.relative_to(extracted_dir)),
            "pdf": str(final_pdf_path),
            "source_dir": str(final_source_dir),
            "log": str(log_path),
            "started_at": started_at,
            "finished_at": now_iso(),
        }
        accepted_log.append(row)
        return row
    except subprocess.TimeoutExpired as exc:
        row = {**candidate, "status": "timeout", "error": str(exc)[:500], "finished_at": now_iso()}
        rejected_log.append(row)
        return row
    except (OSError, RuntimeError, tarfile.TarError, zipfile.BadZipFile, urllib.error.URLError) as exc:
        row = {**candidate, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:500], "finished_at": now_iso()}
        rejected_log.append(row)
        return row
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def write_progress(paths: DatasetPaths, payload: dict) -> None:
    progress_path = paths.report_dir / "progress.json"
    progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--target-successes", type=int, default=3000)
    parser.add_argument("--candidate-limit", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--compile-slots", type=int, default=32)
    parser.add_argument("--max-pending", type=int, default=256)
    parser.add_argument("--download-timeout", type=int, default=120)
    parser.add_argument("--compile-timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-unpacked-mb", type=int, default=120)
    parser.add_argument("--engine", choices=["auto", "latexmk-pdf", "xelatex", "pdflatex", "lualatex"], default="auto")
    parser.add_argument("--run-name", default="arxiv_2025_compilable_unattended")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    paths = DatasetPaths.from_data_root(args.data_root, args.run_name)
    paths.ensure()

    accepted_path = paths.report_dir / "accepted.jsonl"
    rejected_path = paths.report_dir / "rejected.jsonl"
    accepted_log = JsonlLog(accepted_path)
    rejected_log = JsonlLog(rejected_path)

    accepted_ids = load_ids_from_jsonl(accepted_path)
    rejected_ids = load_ids_from_jsonl(rejected_path)
    processed_ids = accepted_ids | rejected_ids
    gate = AcceptGate(args.target_successes, len(accepted_ids))
    compile_slots = threading.Semaphore(args.compile_slots)

    started_at = now_iso()
    attempted = 0
    completed = 0
    status_counts: dict[str, int] = {}

    print(
        json.dumps(
            {
                "event": "start",
                "metadata": str(args.metadata),
                "year": args.year,
                "target_successes": args.target_successes,
                "existing_successes": len(accepted_ids),
                "workers": args.workers,
                "compile_slots": args.compile_slots,
                "engine": args.engine,
                "started_at": started_at,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    candidates = (
        candidate
        for candidate in iter_metadata_candidates(args.metadata, args.year, args.candidate_limit)
        if args.force or candidate["arxiv_id"] not in processed_ids
    )

    pending = set()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while not gate.full():
            while len(pending) < args.max_pending and not gate.full():
                try:
                    candidate = next(candidates)
                except StopIteration:
                    break
                attempted += 1
                pending.add(
                    executor.submit(
                        process_candidate,
                        candidate,
                        paths,
                        args,
                        accepted_log,
                        rejected_log,
                        gate,
                        compile_slots,
                    )
                )
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                completed += 1
                status = str(row.get("status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                if completed % args.progress_every == 0 or status == "accepted":
                    progress = {
                        "started_at": started_at,
                        "updated_at": now_iso(),
                        "attempted_submitted": attempted,
                        "completed": completed,
                        "accepted_total": gate.count,
                        "target_successes": args.target_successes,
                        "pending": len(pending),
                        "status_counts": status_counts,
                    }
                    write_progress(paths, progress)
                    print(json.dumps({"event": "progress", **progress}, ensure_ascii=False, sort_keys=True), flush=True)

    summary = {
        "started_at": started_at,
        "finished_at": now_iso(),
        "metadata": str(args.metadata),
        "year": args.year,
        "target_successes": args.target_successes,
        "accepted_total": gate.count,
        "attempted_submitted": attempted,
        "completed": completed,
        "status_counts": status_counts,
        "accepted_path": str(accepted_path),
        "rejected_path": str(rejected_path),
        "source_dir": str(paths.source_dir),
        "pdf_dir": str(paths.pdf_dir),
    }
    (paths.report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_progress(paths, summary)
    print(json.dumps({"event": "finished", **summary}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if gate.count >= args.target_successes else 2


if __name__ == "__main__":
    raise SystemExit(main())
