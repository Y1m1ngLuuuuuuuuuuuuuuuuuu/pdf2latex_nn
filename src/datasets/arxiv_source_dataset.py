"""Build a TeX-source dataset from arXiv metadata and e-print downloads."""

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
import time
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterable


ARXIV_DATASET_REF = "Cornell-University/arxiv"
ARXIV_METADATA_FILE = "arxiv-metadata-oai-snapshot.json"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclasses.dataclass(frozen=True)
class DatasetPaths:
    root: Path
    metadata_dir: Path
    source_dir: Path
    pdf_dir: Path
    report_dir: Path
    tmp_dir: Path

    @classmethod
    def from_data_root(cls, data_root: Path) -> "DatasetPaths":
        return cls(
            root=data_root,
            metadata_dir=data_root / "00_manifests",
            source_dir=data_root / "03_tex_sources",
            pdf_dir=data_root / "01_raw_pdfs",
            report_dir=data_root / "09_eval_reports" / "arxiv_2025_tex_compile",
            tmp_dir=data_root / "_tmp_arxiv_source_build",
        )

    def ensure(self) -> None:
        for path in [
            self.metadata_dir,
            self.source_dir,
            self.pdf_dir,
            self.report_dir,
            self.tmp_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def ensure_kaggle_metadata(paths: DatasetPaths) -> Path:
    metadata_path = paths.metadata_dir / ARXIV_METADATA_FILE
    if metadata_path.exists() and metadata_path.stat().st_size > 0:
        return metadata_path

    result = run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            ARXIV_DATASET_REF,
            "-f",
            ARXIV_METADATA_FILE,
            "-p",
            str(paths.metadata_dir),
        ],
        timeout=60 * 60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle metadata download failed:\n{result.stdout}")

    zip_path = paths.metadata_dir / f"{ARXIV_METADATA_FILE}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extract(ARXIV_METADATA_FILE, paths.metadata_dir)
        zip_path.unlink()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing downloaded metadata: {metadata_path}")
    return metadata_path


def version_year(record: dict) -> int | None:
    versions = record.get("versions") or []
    if versions:
        created = str(versions[0].get("created") or "")
        match = re.search(r"\b(19|20)\d{2}\b", created)
        if match:
            return int(match.group(0))
    update_date = str(record.get("update_date") or "")
    if re.match(r"^\d{4}-\d{2}-\d{2}$", update_date):
        return int(update_date[:4])
    return None


def select_candidates(metadata_path: Path, year: int, limit: int, output_path: Path) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(candidates) >= limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            arxiv_id = str(record.get("id") or "").strip()
            if not arxiv_id or arxiv_id in seen or not ARXIV_ID_RE.match(arxiv_id):
                continue
            if version_year(record) != year:
                continue
            seen.add(arxiv_id)
            candidates.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": record.get("title"),
                    "categories": record.get("categories"),
                    "update_date": record.get("update_date"),
                    "versions": record.get("versions"),
                }
            )
    write_jsonl(output_path, candidates)
    return candidates


def xml_text(element: ET.Element, path: str) -> str:
    child = element.find(path, ATOM_NS)
    if child is None or child.text is None:
        return ""
    return " ".join(child.text.split())


def select_candidates_from_arxiv_api(
    year: int,
    limit: int,
    output_path: Path,
    batch_size: int,
    sleep_seconds: float,
) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    start = 0
    year_start = f"{year}01010000"
    year_end = f"{year}12312359"

    while len(candidates) < limit:
        max_results = min(batch_size, limit - len(candidates))
        query = {
            "search_query": f"submittedDate:[{year_start} TO {year_end}]",
            "start": str(start),
            "max_results": str(max_results),
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url, headers={"User-Agent": "pdf2latex-nn-dataset/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 200:
                raise RuntimeError(f"arXiv API returned HTTP {response.status}")
            root = ET.fromstring(response.read())

        entries = root.findall("atom:entry", ATOM_NS)
        if not entries:
            break

        for entry in entries:
            raw_id = xml_text(entry, "atom:id").rstrip("/").rsplit("/", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", raw_id)
            if not arxiv_id or arxiv_id in seen or not ARXIV_ID_RE.match(arxiv_id):
                continue
            published = xml_text(entry, "atom:published")
            if published and not published.startswith(f"{year}-"):
                continue
            categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", ATOM_NS)]
            seen.add(arxiv_id)
            candidates.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": xml_text(entry, "atom:title"),
                    "categories": " ".join(term for term in categories if term),
                    "update_date": xml_text(entry, "atom:updated")[:10],
                    "versions": [{"created": published}],
                }
            )
            if len(candidates) >= limit:
                break

        start += len(entries)
        if len(candidates) < limit:
            time.sleep(sleep_seconds)

    write_jsonl(output_path, candidates)
    return candidates


def download_eprint(arxiv_id: str, output_path: Path, retries: int, sleep_seconds: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "pdf2latex-nn-dataset/0.1"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                with output_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            return
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"download failed for {arxiv_id}: {last_error}")


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        base = output_dir.resolve()
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Unsafe tar member: {member.name}")
        archive.extractall(output_dir)


def unpack_source(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = archive_path.read_bytes()[:8]
    if tarfile.is_tarfile(archive_path):
        safe_extract_tar(archive_path, output_dir)
        return
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output_dir)
        return
    if data.startswith(b"\x1f\x8b"):
        text_path = output_dir / "main.tex"
        with gzip.open(archive_path, "rb") as source, text_path.open("wb") as target:
            shutil.copyfileobj(source, target)
        return
    (output_dir / "main.tex").write_bytes(archive_path.read_bytes())


def find_main_tex(source_dir: Path) -> Path | None:
    tex_files = [path for path in source_dir.rglob("*.tex") if path.is_file()]
    if not tex_files:
        return None

    scored: list[tuple[int, int, Path]] = []
    for path in tex_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        score = 0
        if "\\documentclass" in text:
            score += 10
        if "\\begin{document}" in text:
            score += 10
        if "\\end{document}" in text:
            score += 5
        if path.parent == source_dir:
            score += 2
        scored.append((score, path.stat().st_size, path))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < 10:
        return None
    return scored[0][2]


def compile_tex(main_tex: Path, output_dir: Path, timeout: int) -> tuple[bool, str, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-outdir={output_dir}",
            str(main_tex),
        ],
        cwd=main_tex.parent,
        timeout=timeout,
    )
    pdf_path = output_dir / f"{main_tex.stem}.pdf"
    return result.returncode == 0 and pdf_path.exists(), result.stdout, pdf_path if pdf_path.exists() else None


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def process_candidate(candidate: dict, paths: DatasetPaths, args: argparse.Namespace) -> bool:
    arxiv_id = candidate["arxiv_id"]
    final_source_dir = paths.source_dir / arxiv_id
    final_pdf_path = paths.pdf_dir / f"{arxiv_id}.pdf"
    if final_source_dir.exists() and final_pdf_path.exists() and not args.force:
        append_jsonl(paths.report_dir / "accepted.jsonl", {**candidate, "status": "already_present"})
        return True

    work_dir = Path(tempfile.mkdtemp(prefix=f"{arxiv_id}_", dir=paths.tmp_dir))
    archive_path = work_dir / "source.eprint"
    extracted_dir = work_dir / "source"
    compile_dir = work_dir / "compile"
    log_path = paths.report_dir / "logs" / f"{arxiv_id}.log"
    try:
        download_eprint(arxiv_id, archive_path, args.retries, args.retry_sleep)
        unpack_source(archive_path, extracted_dir)
        main_tex = find_main_tex(extracted_dir)
        if main_tex is None:
            append_jsonl(paths.report_dir / "rejected.jsonl", {**candidate, "status": "no_main_tex"})
            return False
        ok, compile_log, pdf_path = compile_tex(main_tex, compile_dir, args.compile_timeout)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(compile_log, encoding="utf-8", errors="ignore")
        if not ok or pdf_path is None:
            append_jsonl(paths.report_dir / "rejected.jsonl", {**candidate, "status": "compile_failed"})
            return False
        copy_tree(extracted_dir, final_source_dir)
        final_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, final_pdf_path)
        append_jsonl(
            paths.report_dir / "accepted.jsonl",
            {
                **candidate,
                "status": "accepted",
                "main_tex": str(main_tex.relative_to(extracted_dir)),
                "pdf": str(final_pdf_path),
                "source_dir": str(final_source_dir),
            },
        )
        return True
    except Exception as exc:
        append_jsonl(paths.report_dir / "rejected.jsonl", {**candidate, "status": "error", "error": str(exc)[:500]})
        return False
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            arxiv_id = row.get("arxiv_id")
            if arxiv_id:
                ids.add(str(arxiv_id))
    return ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--candidate-source", choices=["kaggle", "arxiv-api"], default="kaggle")
    parser.add_argument("--target-successes", type=int, default=2000)
    parser.add_argument("--candidate-limit", type=int, default=10000)
    parser.add_argument("--api-batch-size", type=int, default=100)
    parser.add_argument("--api-sleep", type=float, default=3.0)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    paths = DatasetPaths.from_data_root(args.data_root)
    paths.ensure()

    started_at = dt.datetime.now(dt.UTC).isoformat()
    metadata_path: Path | None = None
    if args.candidate_source == "kaggle":
        metadata_path = ensure_kaggle_metadata(paths)
        candidates_path = paths.metadata_dir / f"arxiv_{args.year}_candidates.jsonl"
        candidates = select_candidates(metadata_path, args.year, args.candidate_limit, candidates_path)
    else:
        candidates_path = paths.metadata_dir / f"arxiv_{args.year}_arxiv_api_candidates.jsonl"
        candidates = select_candidates_from_arxiv_api(
            args.year,
            args.candidate_limit,
            candidates_path,
            args.api_batch_size,
            args.api_sleep,
        )
    print(f"[dataset] selected {len(candidates)} candidates from {args.candidate_source}", flush=True)

    accepted_path = paths.report_dir / "accepted.jsonl"
    accepted_ids = load_existing_ids(accepted_path)
    successes = len(accepted_ids)
    attempted = 0
    if successes:
        print(f"[dataset] resuming with {successes} accepted samples already present", flush=True)

    for candidate in candidates:
        if successes >= args.target_successes:
            break
        if candidate["arxiv_id"] in accepted_ids and not args.force:
            continue
        attempted += 1
        arxiv_id = candidate["arxiv_id"]
        print(f"[dataset] attempt {attempted}: {arxiv_id}", flush=True)
        if process_candidate(candidate, paths, args):
            successes += 1
            print(f"[dataset] accepted {arxiv_id}; total={successes}/{args.target_successes}", flush=True)
        else:
            print(f"[dataset] rejected {arxiv_id}; total={successes}/{args.target_successes}", flush=True)

    summary = {
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.UTC).isoformat(),
        "year": args.year,
        "target_successes": args.target_successes,
        "candidate_limit": args.candidate_limit,
        "candidate_source": args.candidate_source,
        "attempted_this_run": attempted,
        "accepted_total": successes,
        "metadata_path": str(metadata_path) if metadata_path else None,
        "candidates_path": str(candidates_path),
    }
    (paths.report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
