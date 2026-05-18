#!/usr/bin/env python3
"""Build a 10-document real-PDF overfit mini dataset.

This script scans raw PDFs and TeX source folders, runs the full front-end
pipeline, injects TeX-derived edge labels, and writes a manifest that can be
loaded directly by `test_overfit.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
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
from src.pipeline.v7_contract import assert_v7_content_json, assert_v7_graph_data  # noqa: E402
from src.reasoning.graph_builder import GraphBuildConfig, build_graph_from_content_v7  # noqa: E402
from src.reasoning.label_generator import AlignmentLabeler, AlignmentLabelerConfig, AlignmentQualityError  # noqa: E402
from tools.profile_candidate_edge_recall import profile_candidate_recall  # noqa: E402


def default_mineru_command() -> str:
    autodl_mineru = Path("/root/miniconda3/envs/mineru/bin/mineru")
    executable = str(autodl_mineru) if autodl_mineru.exists() else "mineru"
    return f"{default_mineru_env_prefix()} {shlex.quote(executable)} -p {{pdf}} -o {{mineru_output_dir}} -m auto -b pipeline"


def default_mineru_env_prefix() -> str:
    """Force MinerU to use local AutoDL model/cache paths.

    MinerU 3.x starts a temporary FastAPI service and may otherwise resolve
    model/cache paths through the process environment. Keeping this prefix in
    the default command prevents accidental system-disk cache writes and avoids
    silent remote model downloads during long batch jobs.
    """
    env = {
        "MINERU_MODEL_SOURCE": "local",
        "MINERU_TOOLS_CONFIG_JSON": "/root/mineru.json",
        "XDG_CACHE_HOME": "/root/autodl-tmp/.cache",
        "HF_HOME": "/root/autodl-tmp/.cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/root/autodl-tmp/.cache/huggingface/hub",
        "MODELSCOPE_CACHE": "/root/autodl-tmp/.cache/modelscope",
        "TORCH_HOME": "/root/autodl-tmp/.cache/torch",
        "PADDLEOCR_HOME": "/root/autodl-tmp/.cache/paddleocr",
        "MINERU_PROCESSING_WINDOW_SIZE": "32",
        "MINERU_API_MAX_CONCURRENT_REQUESTS": "1",
    }
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


@dataclass(frozen=True)
class CandidateSample:
    document_id: str
    pdf_path: Path
    tex_dir: Path
    main_tex_path: Path
    pdf_origin: str = "id_matched_scan"
    compile_manifest: str | None = None
    compile_status: str | None = None


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
    candidate_edge_recall: float | None = None
    candidate_edge_missing: int | None = None
    pdf_origin: str = "unknown"
    compile_manifest: str | None = None
    compile_status: str | None = None

    def manifest_record(self) -> dict[str, Any]:
        payload = {
            "document_id": self.document_id,
            "pdf_path": str(self.pdf_path.resolve()),
            "content_json": str(self.content_json.resolve()),
            "graph_path": str(self.graph_path.resolve()),
            "tex_path": str(self.tex_path.resolve()),
            "alignment_mapping": str(self.alignment_mapping.resolve()),
            "label_counts": {str(key): int(value) for key, value in self.label_counts.items()},
            "orphan_ratio": self.orphan_ratio,
            "pdf_origin": self.pdf_origin,
        }
        if self.compile_manifest:
            payload["compile_manifest"] = self.compile_manifest
        if self.compile_status:
            payload["compile_status"] = self.compile_status
        if self.candidate_edge_recall is not None:
            payload["candidate_edge_recall"] = float(self.candidate_edge_recall)
        if self.candidate_edge_missing is not None:
            payload["candidate_edge_missing"] = int(self.candidate_edge_missing)
        return payload


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
    embedding_device: str
    target: int
    max_candidates: int | None
    main_tex_names: tuple[str, ...]
    mineru_command: str
    mineru_timeout: int
    similarity_threshold: float
    max_orphan_ratio: float
    max_unmapped_tex_ratio: float
    max_isolated_node_ratio: float
    min_non_none_edges: int
    min_candidate_recall: float
    compiled_accepted_manifests: tuple[Path, ...]
    auto_discover_compiled_manifests: bool
    require_compiled_accepted: bool
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
    parser.add_argument("--graph-output-dir", type=Path, default=REPO_ROOT / "data/06_graph_features_v7")
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
    parser.add_argument(
        "--embedding-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="Device for SciBERT feature extraction. CPU is the robust default for multiprocessing batch builds.",
    )
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, help="Stop scanning after this many candidates")
    parser.add_argument(
        "--main-tex-names",
        default="main.tex",
        help="Comma-separated TeX entry filenames. Fallback root detection is still used when these names do not exist.",
    )
    parser.add_argument(
        "--compiled-accepted-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "accepted.jsonl emitted by step0_compile_arxiv_source_pool.py or "
            "step0_build_compilable_arxiv_dataset.py. When present, candidate "
            "PDF/TeX pairs are read from these compile records instead of only "
            "matching directories by arXiv id."
        ),
    )
    parser.add_argument(
        "--no-auto-compiled-accepted-manifests",
        action="store_true",
        help="Disable auto-discovery of data/09_eval_reports/*compile*/accepted.jsonl manifests.",
    )
    parser.add_argument(
        "--require-compiled-accepted",
        action="store_true",
        help="Fail closed: do not fall back to id-matched PDF/TeX scanning if no compile accepted manifest yields candidates.",
    )
    parser.add_argument("--mineru-command", default=default_mineru_command())
    parser.add_argument("--mineru-timeout", type=int, default=900)
    parser.add_argument("--similarity-threshold", type=float, default=65.0)
    parser.add_argument("--max-orphan-ratio", type=float, default=0.15)
    parser.add_argument("--max-unmapped-tex-ratio", type=float, default=0.30)
    parser.add_argument("--max-isolated-node-ratio", type=float, default=0.85)
    parser.add_argument("--min-non-none-edges", type=int, default=1)
    parser.add_argument(
        "--min-candidate-recall",
        type=float,
        default=1.0,
        help="Required oracle positive-edge recall in graph.edge_index. Default demands zero missing true edges.",
    )
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
            f"pdf={candidate.pdf_path} tex={candidate.main_tex_path} "
            f"origin={candidate.pdf_origin} manifest={candidate.compile_manifest}"
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
        embedding_device=str(args.embedding_device),
        target=target,
        max_candidates=max_candidates,
        main_tex_names=main_tex_names,
        mineru_command=args.mineru_command,
        mineru_timeout=int(args.mineru_timeout),
        similarity_threshold=float(args.similarity_threshold),
        max_orphan_ratio=float(args.max_orphan_ratio),
        max_unmapped_tex_ratio=float(args.max_unmapped_tex_ratio),
        max_isolated_node_ratio=float(args.max_isolated_node_ratio),
        min_non_none_edges=int(args.min_non_none_edges),
        min_candidate_recall=float(args.min_candidate_recall),
        compiled_accepted_manifests=tuple(path.resolve() for path in args.compiled_accepted_manifest),
        auto_discover_compiled_manifests=not bool(args.no_auto_compiled_accepted_manifests),
        require_compiled_accepted=bool(args.require_compiled_accepted),
        reuse_existing=not args.no_reuse_existing,
        force_mineru=bool(args.force_mineru),
        force_json=bool(args.force_json),
        force_graph=bool(args.force_graph),
        force_label=bool(args.force_label),
        dry_run=bool(args.dry_run),
    )


def scan_candidates(config: MiniDatasetConfig) -> list[CandidateSample]:
    """Select PDF/TeX pairs.

    The preferred path is an explicit compile `accepted.jsonl`, because it
    proves the PDF in `data/01_raw_pdfs` was produced from the same TeX source
    tree. If no compile manifest is available, the legacy id-matched scan is
    kept as a fallback for local smoke tests and older datasets.
    """

    manifest_candidates = scan_candidates_from_compile_manifests(config)
    if manifest_candidates or config.require_compiled_accepted:
        return manifest_candidates

    return scan_candidates_by_id(config)


def scan_candidates_by_id(config: MiniDatasetConfig) -> list[CandidateSample]:
    """Legacy fallback: select samples with same-id PDF and TeX source folder."""

    pdf_index = build_pdf_index(config.raw_pdf_dir)
    candidates: list[CandidateSample] = []
    for document_id, pdf_path in sorted(pdf_index.items()):
        tex_dir = config.tex_source_dir / document_id
        if not tex_dir.is_dir():
            continue
        main_tex = first_existing_tex_entry(tex_dir, config.main_tex_names)
        if main_tex is None:
            continue
        candidates.append(
            CandidateSample(
                document_id=document_id,
                pdf_path=pdf_path,
                tex_dir=tex_dir,
                main_tex_path=main_tex,
                pdf_origin="id_matched_scan",
            )
        )
    return candidates


def scan_candidates_from_compile_manifests(config: MiniDatasetConfig) -> list[CandidateSample]:
    manifests = compile_manifest_paths(config)
    if not manifests:
        return []

    candidates: list[CandidateSample] = []
    seen: set[str] = set()
    for manifest_path in manifests:
        if not manifest_path.exists():
            continue
        for row in iter_jsonl(manifest_path):
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "")
            if status not in {"accepted", "already_compiled", "already_present"}:
                continue
            document_id = str(row.get("arxiv_id") or row.get("document_id") or "").strip()
            if not document_id or document_id in seen:
                continue
            candidate = candidate_from_compile_record(row, manifest_path, config)
            if candidate is None:
                continue
            candidates.append(candidate)
            seen.add(document_id)
    return sorted(candidates, key=lambda item: item.document_id)


def compile_manifest_paths(config: MiniDatasetConfig) -> tuple[Path, ...]:
    paths: list[Path] = list(config.compiled_accepted_manifests)
    if config.auto_discover_compiled_manifests:
        patterns = [
            "data/09_eval_reports/arxiv_2025_compilable*/accepted.jsonl",
            "data/09_eval_reports/arxiv_2025_source_pool*compile*/accepted.jsonl",
        ]
        for pattern in patterns:
            paths.extend(sorted(config.project_root.glob(pattern)))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path if path.is_absolute() else config.project_root / path
        resolved = resolved.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return tuple(unique)


def candidate_from_compile_record(row: dict[str, Any], manifest_path: Path, config: MiniDatasetConfig) -> CandidateSample | None:
    document_id = str(row.get("arxiv_id") or row.get("document_id") or "").strip()
    if not document_id:
        return None

    pdf_path = resolve_record_path(row.get("pdf"), config.project_root)
    if pdf_path is None or not pdf_path.is_file():
        pdf_path = config.raw_pdf_dir / f"{document_id}.pdf"
    if not pdf_path.is_file():
        return None

    tex_dir = resolve_record_path(row.get("source_dir"), config.project_root)
    if tex_dir is None or not tex_dir.is_dir():
        tex_dir = config.tex_source_dir / document_id
    if not tex_dir.is_dir():
        return None

    main_tex = None
    main_tex_value = row.get("main_tex")
    if isinstance(main_tex_value, str) and main_tex_value.strip():
        candidate_tex = tex_dir / main_tex_value.strip()
        if candidate_tex.is_file():
            main_tex = candidate_tex
    if main_tex is None:
        main_tex = first_existing_tex_entry(tex_dir, config.main_tex_names)
    if main_tex is None:
        return None

    return CandidateSample(
        document_id=document_id,
        pdf_path=pdf_path.resolve(),
        tex_dir=tex_dir.resolve(),
        main_tex_path=main_tex.resolve(),
        pdf_origin="compiled_from_tex",
        compile_manifest=str(manifest_path),
        compile_status=str(row.get("status") or ""),
    )


def resolve_record_path(value: Any, project_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip())
    if not path.is_absolute():
        path = project_root / path
    return path


def iter_jsonl(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


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
    tex_files = [
        path
        for path in sorted(tex_dir.rglob("*.tex"))
        if ".ipynb_checkpoints" not in path.parts and "checkpoint" not in path.name
    ]
    for name in names:
        candidate = tex_dir / name
        if candidate.is_file():
            return candidate
        for path in tex_files:
            if path.name.lower() == name.lower():
                return path
    root_like = [path for path in tex_files if looks_like_latex_root(path)]
    if root_like:
        return sorted(root_like, key=tex_entry_sort_key(tex_dir))[0]
    root_level = [path for path in tex_files if path.parent == tex_dir and not is_auxiliary_tex_file(path)]
    if len(root_level) == 1:
        return root_level[0]
    return None


ROOT_TEX_RE = re.compile(r"\\(?:documentclass|documentstyle)\b")
AUXILIARY_TEX_NAMES = {
    "abstract.tex",
    "appendix.tex",
    "conclusion.tex",
    "intro.tex",
    "introduction.tex",
    "macros.tex",
    "math_commands.tex",
    "preamble.tex",
    "references.tex",
    "related.tex",
    "related_work.tex",
}


def looks_like_latex_root(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return bool(ROOT_TEX_RE.search(text))


def is_auxiliary_tex_file(path: Path) -> bool:
    name = path.name.lower()
    if name in AUXILIARY_TEX_NAMES:
        return True
    return any(token in name for token in ("macro", "preamble", "reference", "bibliograph", "command"))


def tex_entry_sort_key(tex_dir: Path) -> Any:
    def key(path: Path) -> tuple[int, int, str]:
        try:
            rel_depth = len(path.relative_to(tex_dir).parts)
        except ValueError:
            rel_depth = 99
        name = path.name.lower()
        preferred_rank = 0 if name in {"main.tex", "paper.tex", "article.tex"} else 1
        if name == f"{tex_dir.name.lower()}.tex":
            preferred_rank = 0
        return (preferred_rank, rel_depth, str(path).lower())

    return key


def process_candidate(candidate: CandidateSample, config: MiniDatasetConfig) -> ProcessedSample:
    paths = sample_paths(candidate, config)
    if (
        config.reuse_existing
        and not config.force_label
        and paths["styles"].exists()
        and paths["mapping"].exists()
        and graph_is_valid_labeled(paths["labeled_graph"], config)
    ):
        return summarize_processed_sample(candidate, paths)

    content_json = ensure_content_v7_styles(candidate, paths, config)
    graph_path = ensure_graph(content_json, paths["unlabeled_graph"], config)
    labeled_graph, labeler = label_graph(candidate, content_json, graph_path, paths["labeled_graph"], paths["mapping"], config)
    recall_report = assert_candidate_edge_recall(
        labeled_graph,
        labeler,
        config,
        output_graph_path=paths["labeled_graph"],
    )
    if not graph_is_valid_labeled(paths["labeled_graph"], config):
        raise RuntimeError(f"labeled graph failed validation: {paths['labeled_graph']}")
    return summarize_processed_sample(candidate, paths, recall_report=recall_report)


def sample_paths(candidate: CandidateSample, config: MiniDatasetConfig) -> dict[str, Path]:
    auto_dir = config.mineru_output_dir / candidate.document_id / "auto"
    return {
        "auto_dir": auto_dir,
        "v2": auto_dir / f"{candidate.document_id}_content_list_v2.json",
        "v7": auto_dir / f"{candidate.document_id}_content_list_v7.json",
        "styles": auto_dir / f"{candidate.document_id}_content_list_v7_styles.json",
        "unlabeled_graph": config.graph_output_dir / f"{candidate.document_id}_v7_graph.pt",
        "labeled_graph": config.graph_output_dir / f"{candidate.document_id}_v7_truthgen_labeled_graph.pt",
        "mapping": config.ground_truth_dir / f"{candidate.document_id}_v7_alignment_mapping.json",
    }


def ensure_content_v7_styles(candidate: CandidateSample, paths: dict[str, Path], config: MiniDatasetConfig) -> Path:
    styles_path = paths["styles"]
    if config.reuse_existing and not config.force_json and styles_path.exists():
        assert_v7_content_json(styles_path, require_styles=True)
        return styles_path

    content_v2 = ensure_mineru_content_v2(candidate, paths, config)
    v7_payload = build_content_v7(load_content_list_v2(content_v2))
    v7_payload["source_path"] = str(content_v2)
    write_json(paths["v7"], v7_payload)

    enrich_content_with_styles(paths["v7"], candidate.pdf_path, styles_path, StyleConfig())
    assert_v7_content_json(styles_path, require_styles=True)
    return styles_path


def ensure_mineru_content_v2(candidate: CandidateSample, paths: dict[str, Path], config: MiniDatasetConfig) -> Path:
    existing = find_mineru_content_source(candidate.document_id, config.mineru_output_dir)
    if existing is not None and not config.force_mineru:
        return normalize_mineru_content_to_v2(existing, paths["v2"])

    config.mineru_output_dir.mkdir(parents=True, exist_ok=True)
    command = format_mineru_command(candidate, config)
    print(f"[mini-dataset] mineru id={candidate.document_id} cmd={command}")
    mineru_log_path = paths["auto_dir"].parent / "mineru_command.log"
    completed = run_command_with_process_group_timeout(
        command,
        cwd=config.project_root,
        timeout=config.mineru_timeout,
        log_path=mineru_log_path,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"MinerU failed returncode={completed.returncode} log={mineru_log_path} output_tail={completed.stdout[-4000:]}"
        )
    content_source = find_mineru_content_source(candidate.document_id, config.mineru_output_dir)
    if content_source is None:
        raise FileNotFoundError(f"MinerU did not produce content_list_v2 for {candidate.document_id}")
    return normalize_mineru_content_to_v2(content_source, paths["v2"])


def run_command_with_process_group_timeout(
    command: str,
    *,
    cwd: Path,
    timeout: int,
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Run a shell command and kill the whole process group on timeout.

    MinerU can spawn a local fast_api process. Killing only the shell leaves that
    child alive and holding GPU memory. Each sample gets its own process group,
    and stdout is redirected to a per-sample log file so a detached child cannot
    keep Python blocked on a pipe.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        terminate_process_group(process)
        cleanup_mineru_fast_api_processes()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            kill_process_group(process)
            cleanup_mineru_fast_api_processes(kill=True)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        log_file.close()
        stdout = read_text_tail(log_path)
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=f"{stdout}\n[MinerU timed out after {timeout}s; process group cleaned]\n",
            stderr=None,
        )
    finally:
        if not log_file.closed:
            log_file.close()
    return subprocess.CompletedProcess(command, returncode, stdout=read_text_tail(log_path), stderr=None)


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def cleanup_mineru_fast_api_processes(*, kill: bool = False) -> None:
    """Best-effort cleanup for MinerU's detached local API worker."""

    signal_name = "-KILL" if kill else "-TERM"
    subprocess.run(
        ["pkill", signal_name, "-f", "mineru.cli.fast_api"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def read_text_tail(path: Path, *, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]


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
        import torch

        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        assert_v7_graph_data(graph, graph_path)
        return graph_path
    assert_v7_content_json(content_json, require_styles=True)
    graph_config = GraphBuildConfig(model_path=config.model_path, embedding_device=config.embedding_device)
    build_graph_from_content_v7(content_json, graph_path, graph_config)
    return graph_path


def label_graph(
    candidate: CandidateSample,
    content_json: Path,
    graph_path: Path,
    output_graph_path: Path,
    mapping_path: Path,
    config: MiniDatasetConfig,
) -> tuple[Any, AlignmentLabeler]:
    labeler = AlignmentLabeler(
        content_json_path=content_json,
        tex_path=candidate.main_tex_path,
        graph_path=graph_path,
        config=AlignmentLabelerConfig(
            similarity_threshold=config.similarity_threshold,
            max_orphan_ratio=config.max_orphan_ratio,
            max_unmapped_tex_ratio=config.max_unmapped_tex_ratio,
            max_isolated_node_ratio=config.max_isolated_node_ratio,
            abort_on_bad_alignment=True,
            output_mapping_json=mapping_path,
        ),
    )
    graph = labeler.run(output_graph_path=output_graph_path, overwrite=False)
    return graph, labeler


def assert_candidate_edge_recall(
    graph: Any,
    labeler: AlignmentLabeler,
    config: MiniDatasetConfig,
    *,
    output_graph_path: Path,
) -> dict[str, Any]:
    report = profile_candidate_recall(graph, labeler, max_examples=5)
    recall = float(report["overall"]["recall"])
    if recall < config.min_candidate_recall:
        missing = int(report["overall"]["missing_edges"])
        raise RuntimeError(
            "candidate edge recall below threshold: "
            f"recall={recall:.2%} < {config.min_candidate_recall:.2%}; missing={missing}"
        )
    graph.candidate_edge_recall = recall
    graph.candidate_edge_recall_report = report
    import torch

    torch.save(graph, output_graph_path)
    return report


def graph_is_valid_labeled(graph_path: Path, config: MiniDatasetConfig) -> bool:
    if not graph_path.exists():
        return False
    try:
        import torch

        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        assert_v7_graph_data(graph, graph_path)
        if not hasattr(graph, "edge_index") or not hasattr(graph, "y"):
            return False
        if graph.y.ndim != 1 or int(graph.y.shape[0]) != int(graph.edge_index.shape[1]):
            return False
        labels = torch.where(graph.y.detach().cpu().long() >= 2, torch.full_like(graph.y.detach().cpu().long(), 2), graph.y.detach().cpu().long())
        counts = torch.bincount(labels, minlength=3).tolist()
        non_none = int(sum(counts[:2]))
        if config.min_candidate_recall > 0:
            recall = getattr(graph, "candidate_edge_recall", None)
            if recall is None or float(recall) < config.min_candidate_recall:
                return False
        return non_none >= config.min_non_none_edges
    except Exception:
        return False


def summarize_processed_sample(
    candidate: CandidateSample,
    paths: dict[str, Path],
    *,
    recall_report: dict[str, Any] | None = None,
) -> ProcessedSample:
    import torch

    graph = torch.load(paths["labeled_graph"], map_location="cpu", weights_only=False)
    labels = torch.where(graph.y.detach().cpu().long() >= 2, torch.full_like(graph.y.detach().cpu().long(), 2), graph.y.detach().cpu().long())
    counts = torch.bincount(labels, minlength=3).tolist()
    pdf_to_tex = list(getattr(graph, "pdf_to_tex", []))
    orphan_ratio = (sum(1 for item in pdf_to_tex if item is None) / len(pdf_to_tex)) if pdf_to_tex else 0.0
    return ProcessedSample(
        document_id=candidate.document_id,
        pdf_path=candidate.pdf_path,
        content_json=paths["styles"],
        graph_path=paths["labeled_graph"],
        tex_path=candidate.main_tex_path,
        alignment_mapping=paths["mapping"],
        label_counts={idx: int(counts[idx]) for idx in range(3)},
        orphan_ratio=orphan_ratio,
        pdf_origin=candidate.pdf_origin,
        compile_manifest=candidate.compile_manifest,
        compile_status=candidate.compile_status,
        candidate_edge_recall=(
            float(recall_report["overall"]["recall"]) if recall_report is not None else getattr(graph, "candidate_edge_recall", None)
        ),
        candidate_edge_missing=(
            int(recall_report["overall"]["missing_edges"])
            if recall_report is not None
            else int(graph.candidate_edge_recall_report["overall"]["missing_edges"])
            if hasattr(graph, "candidate_edge_recall_report")
            else None
        ),
    )


def write_manifest(path: Path, samples: list[ProcessedSample], config: MiniDatasetConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "overfit_10_docs_manifest_v7",
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
        "pdf_origin": candidate.pdf_origin,
        "compile_manifest": candidate.compile_manifest,
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
