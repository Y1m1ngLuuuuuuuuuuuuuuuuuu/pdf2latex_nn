#!/usr/bin/env python3
"""Build a ranked compile-accepted manifest for expansion runs.

The staged builder is intentionally conservative: it starts from PDF/TeX pairs
that compiled successfully and then checks whether MinerU output is available.
This helper keeps that contract but reorders candidates so expansion tries the
most likely successes first.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.build_mini_dataset import (  # noqa: E402
    first_existing_tex_entry,
    find_mineru_content_source,
    iter_jsonl,
    resolve_record_path,
)


SECTION_RE = re.compile(r"\\(?:section|subsection|subsubsection)\*?\s*\{")
DISPLAY_MATH_RE = re.compile(r"\\(?:begin\{(?:equation|align|gather|multline)|\\\[)")
FIGURE_RE = re.compile(r"\\begin\{figure\*?\}|\\includegraphics")
TABLE_RE = re.compile(r"\\begin\{table\*?\}|\\begin\{tabular")
POISON_RE = re.compile(r"\\(?:usetikzlibrary|begin\{tikzpicture\}|pgfplots|begin\{tcolorbox\})")


@dataclass(frozen=True)
class RankedCandidate:
    document_id: str
    score: float
    status: str
    pdf: str
    source_dir: str
    main_tex: str
    mineru_content: str
    features: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--compiled-accepted-manifest", action="append", type=Path, default=[])
    parser.add_argument("--auto-discover-compiled-manifests", action="store_true")
    parser.add_argument("--mineru-source-dir", type=Path, required=True)
    parser.add_argument("--raw-pdf-dir", type=Path, default=REPO_ROOT / "data/01_raw_pdfs")
    parser.add_argument("--tex-source-dir", type=Path, default=REPO_ROOT / "data/03_tex_sources")
    parser.add_argument("--exclude-manifest", action="append", type=Path, default=[])
    parser.add_argument("--main-tex-names", default="main.tex")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    main_tex_names = tuple(name.strip() for name in args.main_tex_names.split(",") if name.strip())
    excluded = load_excluded_document_ids(tuple(path.resolve() for path in args.exclude_manifest), project_root)
    manifest_paths = compile_manifest_paths(args, project_root)
    ranked: list[RankedCandidate] = []
    skipped: dict[str, int] = {}
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        for row in iter_jsonl(manifest_path):
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "")
            if status not in {"accepted", "already_compiled", "already_present"}:
                continue
            document_id = str(row.get("arxiv_id") or row.get("document_id") or "").strip()
            if not document_id or document_id in seen:
                continue
            seen.add(document_id)
            if document_id in excluded:
                skipped["excluded"] = skipped.get("excluded", 0) + 1
                continue
            candidate = build_candidate(
                row,
                document_id=document_id,
                project_root=project_root,
                raw_pdf_dir=args.raw_pdf_dir.resolve(),
                tex_source_dir=args.tex_source_dir.resolve(),
                mineru_source_dir=args.mineru_source_dir.resolve(),
                main_tex_names=main_tex_names,
            )
            if candidate is None:
                skipped["missing_pdf_tex_or_mineru"] = skipped.get("missing_pdf_tex_or_mineru", 0) + 1
                continue
            ranked.append(candidate)
    ranked.sort(key=lambda item: (-item.score, item.document_id))
    if args.limit is not None:
        ranked = ranked[: max(0, int(args.limit))]
    write_jsonl_manifest(args.output, ranked)
    write_report(args.report, ranked, skipped, manifest_paths, excluded)
    print(f"ranked_candidates={len(ranked)} output={args.output}")
    return 0


def compile_manifest_paths(args: argparse.Namespace, project_root: Path) -> tuple[Path, ...]:
    paths = [resolve_path(path, project_root) for path in args.compiled_accepted_manifest]
    if args.auto_discover_compiled_manifests:
        patterns = [
            "data/09_eval_reports/arxiv_2025_compilable*/accepted.jsonl",
            "data/09_eval_reports/arxiv_2025_source_pool*compile*/accepted.jsonl",
        ]
        for pattern in patterns:
            paths.extend(sorted(project_root.glob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen and resolved.exists():
            unique.append(resolved)
            seen.add(resolved)
    return tuple(unique)


def build_candidate(
    row: dict[str, Any],
    *,
    document_id: str,
    project_root: Path,
    raw_pdf_dir: Path,
    tex_source_dir: Path,
    mineru_source_dir: Path,
    main_tex_names: tuple[str, ...],
) -> RankedCandidate | None:
    pdf_path = resolve_record_path(row.get("pdf"), project_root)
    if pdf_path is None or not pdf_path.is_file():
        pdf_path = raw_pdf_dir / f"{document_id}.pdf"
    if not pdf_path.is_file():
        return None
    tex_dir = resolve_record_path(row.get("source_dir"), project_root)
    if tex_dir is None or not tex_dir.is_dir():
        tex_dir = tex_source_dir / document_id
    if not tex_dir.is_dir():
        return None
    main_tex = None
    main_tex_value = row.get("main_tex")
    if isinstance(main_tex_value, str) and main_tex_value.strip():
        candidate_tex = tex_dir / main_tex_value.strip()
        if candidate_tex.is_file():
            main_tex = candidate_tex
    if main_tex is None:
        main_tex = first_existing_tex_entry(tex_dir, main_tex_names)
    if main_tex is None:
        return None
    mineru_content = find_mineru_content_source(document_id, mineru_source_dir)
    if mineru_content is None:
        return None
    features = extract_features(pdf_path, main_tex)
    score = rank_score(features)
    return RankedCandidate(
        document_id=document_id,
        score=score,
        status=str(row.get("status") or "accepted"),
        pdf=str(pdf_path.resolve()),
        source_dir=str(tex_dir.resolve()),
        main_tex=str(main_tex.resolve().relative_to(tex_dir.resolve())),
        mineru_content=str(mineru_content.resolve()),
        features=features,
    )


def extract_features(pdf_path: Path, tex_path: Path) -> dict[str, Any]:
    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    pages = estimate_pdf_pages(pdf_path)
    tex_chars = len(text)
    sections = len(SECTION_RE.findall(text))
    equations = len(DISPLAY_MATH_RE.findall(text))
    figures = len(FIGURE_RE.findall(text))
    tables = len(TABLE_RE.findall(text))
    poison = len(POISON_RE.findall(text))
    return {
        "page_count": pages,
        "tex_chars": tex_chars,
        "section_count": sections,
        "equation_count": equations,
        "figure_count": figures,
        "table_count": tables,
        "poison_macro_count": poison,
    }


def rank_score(features: dict[str, Any]) -> float:
    pages = int(features["page_count"])
    tex_chars = int(features["tex_chars"])
    sections = int(features["section_count"])
    equations = int(features["equation_count"])
    figures = int(features["figure_count"])
    tables = int(features["table_count"])
    poison = int(features["poison_macro_count"])
    score = 0.0
    score += 30.0 if sections >= 2 else (-20.0 if sections == 0 else 10.0)
    score += 20.0 if 3 <= pages <= 14 else (8.0 if 1 <= pages <= 24 else -20.0)
    score += 15.0 if 5_000 <= tex_chars <= 180_000 else (-15.0 if tex_chars > 350_000 else 0.0)
    score -= min(35.0, equations * 0.8)
    score -= min(25.0, (figures + tables) * 1.0)
    score -= min(40.0, poison * 8.0)
    score -= max(0.0, pages - 18) * 1.5
    return round(score, 4)


def estimate_pdf_pages(path: Path) -> int:
    try:
        import fitz

        with fitz.open(path) as doc:
            return int(doc.page_count)
    except Exception:
        return 1


def load_excluded_document_ids(paths: tuple[Path, ...], project_root: Path) -> set[str]:
    excluded: set[str] = set()
    for raw_path in paths:
        path = resolve_path(raw_path, project_root)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        documents = payload.get("documents", []) if isinstance(payload, dict) else []
        for row in documents:
            if isinstance(row, dict):
                document_id = row.get("document_id") or row.get("doc_id") or row.get("arxiv_id")
                if document_id:
                    excluded.add(str(document_id))
    return excluded


def write_jsonl_manifest(path: Path, ranked: list[RankedCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in ranked:
            payload = {
                "status": item.status,
                "arxiv_id": item.document_id,
                "document_id": item.document_id,
                "pdf": item.pdf,
                "source_dir": item.source_dir,
                "main_tex": item.main_tex,
                "rank_score": item.score,
                "rank_features": item.features,
                "mineru_content": item.mineru_content,
            }
            file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(
    path: Path,
    ranked: list[RankedCandidate],
    skipped: dict[str, int],
    manifest_paths: tuple[Path, ...],
    excluded: set[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = [item.score for item in ranked]
    page_counts = [int(item.features["page_count"]) for item in ranked]
    payload = {
        "ranked_count": len(ranked),
        "excluded_count": len(excluded),
        "skipped": skipped,
        "compiled_manifests": [str(path) for path in manifest_paths],
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "page_count_min": min(page_counts) if page_counts else None,
        "page_count_max": max(page_counts) if page_counts else None,
        "top20": [asdict(item) for item in ranked[:20]],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


if __name__ == "__main__":
    raise SystemExit(main())
