#!/usr/bin/env python3
"""Build a 10-document real-PDF overfit mini dataset.

This script scans raw PDFs and TeX source folders, runs the full front-end
pipeline, injects TeX-derived edge labels, and writes a manifest that can be
loaded directly by `test_overfit.py`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.perception.reading_order import (  # noqa: E402
    build_content_v7,
    load_content_list_v2,
    write_json,
)
from src.perception.style_spans import StyleConfig, enrich_content_with_styles  # noqa: E402
from src.reasoning.graph_builder import GraphBuildConfig, build_graph_from_content_v7  # noqa: E402
from src.reasoning.label_generator import AlignmentLabeler, AlignmentLabelerConfig, AlignmentQualityError  # noqa: E402


def default_mineru_command() -> str:
    autodl_mineru = Path("/root/miniconda3/envs/mineru/bin/mineru")
    executable = str(autodl_mineru) if autodl_mineru.exists() else "mineru"
    return f"{shlex.quote(executable)} -p {{pdf}} -o {{mineru_output_dir}} -m auto -b pipeline"


@dataclass(frozen=True)
class CandidateSample:
    document_id: str
    pdf_path: Path
    tex_dir: Path
    main_tex_path: Path


@dataclass(frozen=True)
class ProcessedSample:
    document_id: str
    pdf_path: Path
    content_json: Path
    graph_path: Path
    tex_path: Path
    alignment_mapping: Path
    label_counts: dict[int, int]
    orphan_ratio: float

    def manifest_record(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pdf_path": str(self.pdf_path.resolve()),
            "content_json": str(self.content_json.resolve()),
            "graph_path": str(self.graph_path.resolve()),
            "tex_path": str(self.tex_path.resolve()),
            "alignment_mapping": str(self.alignment_mapping.resolve()),
            "label_counts": {str(key): int(value) for key, value in self.label_counts.items()},
            "orphan_ratio": self.orphan_ratio,
        }


@dataclass(frozen=True)
class MiniDatasetConfig:
    project_root: Path
    raw_pdf_dir: Path
    tex_source_dir: Path
    mineru_output_dir: Path
    graph_output_dir: Path
    ground_truth_dir: Path
    manifest_output: Path
    error_log: Path
    model_path: Path
    target: int
    max_candidates: int | None
    main_tex_names: tuple[str, ...]
    mineru_command: str
    mineru_timeout: int
    similarity_threshold: float
    max_orphan_ratio: float
    min_non_none_edges: int
    reuse_existing: bool
    force_mineru: bool
    force_json: bool
    force_graph: bool
    force_label: bool
    dry_run: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--raw-pdf-dir", type=Path, default=REPO_ROOT / "data/01_raw_pdfs")
    parser.add_argument("--tex-source-dir", type=Path, default=REPO_ROOT / "data/03_tex_sources")
    parser.add_argument(
        "--mineru-output-dir",
        type=Path,
        default=REPO_ROOT / "data/02_mineru_outputs/mineru_output",
    )
    parser.add_argument("--graph-output-dir", type=Path, default=REPO_ROOT / "data/06_graph_features")
    parser.add_argument("--ground-truth-dir", type=Path, default=REPO_ROOT / "data/04_ground_truth_ir")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=REPO_ROOT / "data/00_manifests/overfit_10_docs.json",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=REPO_ROOT / "data/00_manifests/overfit_10_docs_errors.jsonl",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models/huggingface/allenai/scibert_scivocab_uncased",
        help="Local SciBERT model directory used by graph_builder.",
    )
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, help="Stop scanning after this many candidates")
    parser.add_argument(
        "--main-tex-names",
        default="main.tex",
        help="Comma-separated TeX entry filenames. Default is strict: main.tex only.",
    )
    parser.add_argument("--mineru-command", default=default_mineru_command())
    parser.add_argument("--mineru-timeout", type=int, default=900)
    parser.add_argument("--similarity-threshold", type=float, default=65.0)
    parser.add_argument("--max-orphan-ratio", type=float, default=0.30)
    parser.add_argument("--min-non-none-edges", type=int, default=1)
    parser.add_argument("--no-reuse-existing", action="store_true")
    parser.add_argument("--force-mineru", action="store_true")
    parser.add_argument("--force-json", action="store_true")
    parser.add_argument("--force-graph", action="store_true")
    parser.add_argument("--force-label", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    candidates = scan_candidates(config)
    if config.max_candidates is not None:
        candidates = candidates[: config.max_candidates]
    print(f"candidate_count={len(candidates)} target={config.target}")
    if config.dry_run:
        for candidate in candidates[: config.target]:
            print(
                f"dry_run candidate id={candidate.document_id} "
                f"pdf={candidate.pdf_path} tex={candidate.main_tex_path}"
            )
        return 0

    processed: list[ProcessedSample] = []
    config.error_log.parent.mkdir(parents=True, exist_ok=True)
    if config.error_log.exists():
        config.error_log.unlink()

    for candidate in progress_iter(candidates):
        if len(processed) >= config.target:
            break
        status = f"{len(processed)}/{config.target}"
        print(f"[mini-dataset] start id={candidate.document_id} success={status}")
        try:
            sample = process_candidate(candidate, config)
            processed.append(sample)
            print(
                "[mini-dataset] success "
                f"id={sample.document_id} success={len(processed)}/{config.target} "
                f"labels={sample.label_counts} orphan_ratio={sample.orphan_ratio:.2%}"
            )
        except Exception as exc:
            log_processing_error(config.error_log, candidate, exc)
            print(f"[mini-dataset] skip id={candidate.document_id} error={type(exc).__name__}: {exc}")
            continue

    if len(processed) < config.target:
        print(f"[mini-dataset] failed target={config.target} success={len(processed)}")
        write_manifest(config.manifest_output, processed, config)
        return 2

    write_manifest(config.manifest_output, processed, config)
    print(f"[mini-dataset] wrote manifest={config.manifest_output} documents={len(processed)}")
    return 0


def config_from_args(args: argparse.Namespace) -> MiniDatasetConfig:
    main_tex_names = tuple(name.strip() for name in args.main_tex_names.split(",") if name.strip())
    if not main_tex_names:
        raise ValueError("--main-tex-names must contain at least one filename")
    target = int(args.target)
    if target <= 0:
        raise ValueError("--target must be positive")
    max_candidates = int(args.max_candidates) if args.max_candidates is not None else None
    return MiniDatasetConfig(
        project_root=args.project_root.resolve(),
        raw_pdf_dir=args.raw_pdf_dir.resolve(),
        tex_source_dir=args.tex_source_dir.resolve(),
        mineru_output_dir=args.mineru_output_dir.resolve(),
        graph_output_dir=args.graph_output_dir.resolve(),
        ground_truth_dir=args.ground_truth_dir.resolve(),
        manifest_output=args.manifest_output.resolve(),
        error_log=args.error_log.resolve(),
        model_path=args.model_path.resolve(),
        target=target,
        max_candidates=max_candidates,
        main_tex_names=main_tex_names,
        mineru_command=args.mineru_command,
        mineru_timeout=int(args.mineru_timeout),
        similarity_threshold=float(args.similarity_threshold),
        max_orphan_ratio=float(args.max_orphan_ratio),
        min_non_none_edges=int(args.min_non_none_edges),
        reuse_existing=not args.no_reuse_existing,
        force_mineru=bool(args.force_mineru),
        force_json=bool(args.force_json),
        force_graph=bool(args.force_graph),
        force_label=bool(args.force_label),
        dry_run=bool(args.dry_run),
    )


def scan_candidates(config: MiniDatasetConfig) -> list[CandidateSample]:
    """Select samples that have both a raw PDF and a TeX source folder with main.tex."""

    pdf_index = build_pdf_index(config.raw_pdf_dir)
    candidates: list[CandidateSample] = []
    for document_id, pdf_path in sorted(pdf_index.items()):
        tex_dir = config.tex_source_dir / document_id
        if not tex_dir.is_dir():
            continue
        main_tex = first_existing_tex_entry(tex_dir, config.main_tex_names)
        if main_tex is None:
            continue
        candidates.append(CandidateSample(document_id=document_id, pdf_path=pdf_path, tex_dir=tex_dir, main_tex_path=main_tex))
    return candidates


def build_pdf_index(raw_pdf_dir: Path) -> dict[str, Path]:
    pdfs: dict[str, Path] = {}
    for path in sorted(raw_pdf_dir.rglob("*.pdf")):
        if ".ipynb_checkpoints" in path.parts or "checkpoint" in path.name:
            continue
        document_id = path.stem
        if document_id not in pdfs:
            pdfs[document_id] = path
    return pdfs


def first_existing_tex_entry(tex_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = tex_dir / name
        if candidate.is_file():
            return candidate
    return None


def process_candidate(candidate: CandidateSample, config: MiniDatasetConfig) -> ProcessedSample:
    paths = sample_paths(candidate, config)
    if (
        config.reuse_existing
        and not config.force_label
        and paths["styles"].exists()
        and paths["mapping"].exists()
        and graph_is_valid_labeled(paths["graph"], config)
    ):
        return summarize_processed_sample(candidate, paths)

    content_json = ensure_content_v7_styles(candidate, paths, config)
    graph_path = ensure_graph(content_json, paths["graph"], config)
    label_graph(candidate, content_json, graph_path, paths["mapping"], config)
    if not graph_is_valid_labeled(graph_path, config):
        raise RuntimeError(f"labeled graph failed validation: {graph_path}")
    return summarize_processed_sample(candidate, paths)


def sample_paths(candidate: CandidateSample, config: MiniDatasetConfig) -> dict[str, Path]:
    auto_dir = config.mineru_output_dir / candidate.document_id / "auto"
    return {
        "auto_dir": auto_dir,
        "v2": auto_dir / f"{candidate.document_id}_content_list_v2.json",
        "v7": auto_dir / f"{candidate.document_id}_content_list_v7.json",
        "styles": auto_dir / f"{candidate.document_id}_content_list_v7_styles.json",
        "graph": config.graph_output_dir / f"{candidate.document_id}_graph.pt",
        "mapping": config.ground_truth_dir / f"{candidate.document_id}_alignment_mapping.json",
    }


def ensure_content_v7_styles(candidate: CandidateSample, paths: dict[str, Path], config: MiniDatasetConfig) -> Path:
    styles_path = paths["styles"]
    if config.reuse_existing and not config.force_json and styles_path.exists():
        return styles_path

    content_v2 = ensure_mineru_content_v2(candidate, paths, config)
    v7_payload = build_content_v7(load_content_list_v2(content_v2))
    v7_payload["source_path"] = str(content_v2)
    write_json(paths["v7"], v7_payload)

    enrich_content_with_styles(paths["v7"], candidate.pdf_path, styles_path, StyleConfig())
    return styles_path


def ensure_mineru_content_v2(candidate: CandidateSample, paths: dict[str, Path], config: MiniDatasetConfig) -> Path:
    existing = find_mineru_content_source(candidate.document_id, config.mineru_output_dir)
    if existing is not None and not config.force_mineru:
        return normalize_mineru_content_to_v2(existing, paths["v2"])

    config.mineru_output_dir.mkdir(parents=True, exist_ok=True)
    command = format_mineru_command(candidate, config)
    print(f"[mini-dataset] mineru id={candidate.document_id} cmd={command}")
    completed = subprocess.run(
        command,
        shell=True,
        cwd=config.project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=config.mineru_timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"MinerU failed returncode={completed.returncode} output_tail={completed.stdout[-4000:]}"
        )
    content_source = find_mineru_content_source(candidate.document_id, config.mineru_output_dir)
    if content_source is None:
        raise FileNotFoundError(f"MinerU did not produce content_list_v2 for {candidate.document_id}")
    return normalize_mineru_content_to_v2(content_source, paths["v2"])


def find_mineru_content_source(document_id: str, mineru_output_dir: Path) -> Path | None:
    doc_dir = mineru_output_dir / document_id
    preferred = [
        doc_dir / "auto" / f"{document_id}_content_list_v2.json",
    ]
    flat = [
        doc_dir / "auto" / f"{document_id}_content_list.json",
    ]
    if doc_dir.exists():
        preferred.extend(sorted(doc_dir.rglob("*content_list_v2.json")))
        flat.extend(sorted(doc_dir.rglob("*content_list.json")))
    for path in preferred + flat:
        if path.is_file() and ".ipynb_checkpoints" not in path.parts and "checkpoint" not in path.name:
            return path
    return None


def normalize_mineru_content_to_v2(source_path: Path, output_v2_path: Path) -> Path:
    data = json.loads(source_path.read_text(encoding="utf-8"))
    if is_content_v2_pages(data):
        return source_path
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        pages = group_flat_mineru_blocks_by_page(data)
        output_v2_path.parent.mkdir(parents=True, exist_ok=True)
        output_v2_path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_v2_path
    raise ValueError(f"Unsupported MinerU content format in {source_path}")


def is_content_v2_pages(data: Any) -> bool:
    return isinstance(data, list) and (not data or all(isinstance(page, list) for page in data))


def group_flat_mineru_blocks_by_page(blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    max_page = -1
    for block in blocks:
        page = block.get("page_idx", 0)
        if isinstance(page, int) and page > max_page:
            max_page = page
    pages: list[list[dict[str, Any]]] = [[] for _ in range(max_page + 1)]
    if not pages:
        pages = [[]]
    for block in blocks:
        page = block.get("page_idx", 0)
        page_idx = page if isinstance(page, int) and page >= 0 else 0
        while page_idx >= len(pages):
            pages.append([])
        pages[page_idx].append(dict(block))
    return pages


def format_mineru_command(candidate: CandidateSample, config: MiniDatasetConfig) -> str:
    values = {
        "pdf": shlex.quote(str(candidate.pdf_path)),
        "pdf_path": shlex.quote(str(candidate.pdf_path)),
        "pdf_parent": shlex.quote(str(candidate.pdf_path.parent)),
        "doc_id": shlex.quote(candidate.document_id),
        "document_id": shlex.quote(candidate.document_id),
        "mineru_output_dir": shlex.quote(str(config.mineru_output_dir)),
        "output_dir": shlex.quote(str(config.mineru_output_dir)),
    }
    return config.mineru_command.format(**values)


def ensure_graph(content_json: Path, graph_path: Path, config: MiniDatasetConfig) -> Path:
    if config.reuse_existing and not config.force_graph and graph_path.exists():
        return graph_path
    graph_config = GraphBuildConfig(model_path=config.model_path)
    build_graph_from_content_v7(content_json, graph_path, graph_config)
    return graph_path


def label_graph(
    candidate: CandidateSample,
    content_json: Path,
    graph_path: Path,
    mapping_path: Path,
    config: MiniDatasetConfig,
) -> None:
    labeler = AlignmentLabeler(
        content_json_path=content_json,
        tex_path=candidate.main_tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(
            similarity_threshold=config.similarity_threshold,
            max_orphan_ratio=config.max_orphan_ratio,
            abort_on_bad_alignment=True,
            output_mapping_json=mapping_path,
        ),
    )
    labeler.run(overwrite=True)


def graph_is_valid_labeled(graph_path: Path, config: MiniDatasetConfig) -> bool:
    if not graph_path.exists():
        return False
    try:
        import torch

        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        if not hasattr(graph, "edge_index") or not hasattr(graph, "y"):
            return False
        if graph.y.ndim != 1 or int(graph.y.shape[0]) != int(graph.edge_index.shape[1]):
            return False
        labels = torch.where(graph.y.detach().cpu().long() >= 2, torch.full_like(graph.y.detach().cpu().long(), 2), graph.y.detach().cpu().long())
        counts = torch.bincount(labels, minlength=3).tolist()
        non_none = int(sum(counts[:2]))
        return non_none >= config.min_non_none_edges
    except Exception:
        return False


def summarize_processed_sample(candidate: CandidateSample, paths: dict[str, Path]) -> ProcessedSample:
    import torch

    graph = torch.load(paths["graph"], map_location="cpu", weights_only=False)
    labels = torch.where(graph.y.detach().cpu().long() >= 2, torch.full_like(graph.y.detach().cpu().long(), 2), graph.y.detach().cpu().long())
    counts = torch.bincount(labels, minlength=3).tolist()
    pdf_to_tex = list(getattr(graph, "pdf_to_tex", []))
    orphan_ratio = (sum(1 for item in pdf_to_tex if item is None) / len(pdf_to_tex)) if pdf_to_tex else 0.0
    return ProcessedSample(
        document_id=candidate.document_id,
        pdf_path=candidate.pdf_path,
        content_json=paths["styles"],
        graph_path=paths["graph"],
        tex_path=candidate.main_tex_path,
        alignment_mapping=paths["mapping"],
        label_counts={idx: int(counts[idx]) for idx in range(3)},
        orphan_ratio=orphan_ratio,
    )


def write_manifest(path: Path, samples: list[ProcessedSample], config: MiniDatasetConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "overfit_10_docs_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": config.target,
        "success_count": len(samples),
        "documents": [sample.manifest_record() for sample in samples],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def log_processing_error(path: Path, candidate: CandidateSample, exc: BaseException) -> None:
    payload = {
        "document_id": candidate.document_id,
        "pdf_path": str(candidate.pdf_path),
        "tex_path": str(candidate.main_tex_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(limit=20),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def progress_iter(candidates: list[CandidateSample]) -> Any:
    try:
        from tqdm import tqdm

        return tqdm(candidates, desc="mini-dataset", unit="doc")
    except ModuleNotFoundError:
        return candidates


if __name__ == "__main__":
    raise SystemExit(main())
