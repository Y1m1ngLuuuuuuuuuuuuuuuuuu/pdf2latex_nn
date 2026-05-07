#!/usr/bin/env python3
"""Staged v7 dataset builder optimized for throughput.

This entrypoint keeps the graph/label output contract from
``build_mini_dataset.py`` but changes the scheduling:

1. Preflight TeX sources before spending MinerU time.
2. Run MinerU on batches of PDFs so the local service/model startup cost is
   amortized across many documents.
3. Process v7 JSON, SciBERT graph features, labels, and quality gates with a
   bounded worker pool.

The resulting manifest records are compatible with the existing training
entrypoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.build_mini_dataset import (  # noqa: E402
    CandidateSample,
    MiniDatasetConfig,
    ProcessedSample,
    default_mineru_command,
    find_mineru_content_source,
    format_mineru_command,
    graph_is_valid_labeled,
    log_processing_error,
    process_candidate,
    progress_iter,
    run_command_with_process_group_timeout,
    sample_paths,
    scan_candidates,
    write_manifest,
)
from src.reasoning.label_generator import (  # noqa: E402
    ALIGNABLE_TEX_NODE_TYPES,
    AlignmentLabeler,
    AlignmentLabelerConfig,
    LayoutBreakerException,
)


@dataclass(frozen=True)
class StagedConfig:
    mini: MiniDatasetConfig
    run_name: str
    stage_root: Path
    mineru_batch_size: int
    mineru_batch_timeout: int
    mineru_batch_command: str
    mineru_fallback_single: bool
    process_workers: int
    process_backlog: int
    preflight_tex: bool
    skip_mineru_stage: bool
    process_existing_mineru_only: bool
    exclude_manifests: tuple[Path, ...]
    max_preflight_fail_ratio: float


@dataclass(frozen=True)
class PreflightResult:
    candidate: CandidateSample
    ok: bool
    error_type: str | None = None
    error: str | None = None
    tex_node_count: int = 0
    section_count: int = 0
    alignable_count: int = 0


@dataclass(frozen=True)
class WorkerResult:
    ok: bool
    candidate: CandidateSample
    sample: ProcessedSample | None = None
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--raw-pdf-dir", type=Path, default=REPO_ROOT / "data/01_raw_pdfs")
    parser.add_argument("--tex-source-dir", type=Path, default=REPO_ROOT / "data/03_tex_source_pool")
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
        default=REPO_ROOT / "data/00_manifests/v7_staged_dataset.json",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=REPO_ROOT / "data/00_manifests/v7_staged_dataset_errors.jsonl",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models/huggingface/allenai/scibert_scivocab_uncased",
    )
    parser.add_argument("--run-name", default=lambda_run_name())
    parser.add_argument("--stage-root", type=Path, default=REPO_ROOT / "data/_tmp_v7_staged_builder")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--main-tex-names", default="main.tex")
    parser.add_argument("--similarity-threshold", type=float, default=65.0)
    parser.add_argument("--max-orphan-ratio", type=float, default=0.15)
    parser.add_argument("--max-unmapped-tex-ratio", type=float, default=0.30)
    parser.add_argument("--max-isolated-node-ratio", type=float, default=0.85)
    parser.add_argument("--min-non-none-edges", type=int, default=1)
    parser.add_argument("--min-candidate-recall", type=float, default=1.0)
    parser.add_argument("--mineru-command", default=default_mineru_command())
    parser.add_argument(
        "--mineru-batch-command",
        default=default_mineru_batch_command(),
        help="Template fields: {pdf_batch_dir}, {mineru_output_dir}",
    )
    parser.add_argument("--mineru-timeout", type=int, default=900, help="Timeout for single-PDF fallback.")
    parser.add_argument("--mineru-batch-size", type=int, default=64)
    parser.add_argument("--mineru-batch-timeout", type=int, default=7200)
    parser.add_argument("--no-mineru-fallback-single", action="store_true")
    parser.add_argument("--process-workers", type=int, default=2)
    parser.add_argument("--process-backlog", type=int, default=8)
    parser.add_argument("--max-preflight-fail-ratio", type=float, default=0.60)
    parser.add_argument("--exclude-manifest", action="append", type=Path, default=[])
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-mineru-stage", action="store_true")
    parser.add_argument("--process-all-candidates", action="store_true")
    parser.add_argument("--no-reuse-existing", action="store_true")
    parser.add_argument("--force-json", action="store_true")
    parser.add_argument("--force-graph", action="store_true")
    parser.add_argument("--force-label", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    config = config_from_args(build_arg_parser().parse_args())
    candidates = prepare_candidates(config)
    print(
        "staged_builder "
        f"candidates={len(candidates)} target={config.mini.target} "
        f"preflight={config.preflight_tex} mineru_stage={not config.skip_mineru_stage} "
        f"workers={config.process_workers}",
        flush=True,
    )
    if config.mini.dry_run:
        print_dry_run(config, candidates)
        return 0

    preflighted = run_preflight_stage(candidates, config)
    if not config.skip_mineru_stage:
        run_mineru_stage(preflighted, config)
    processed = run_processing_stage(preflighted, config)
    if len(processed) < config.mini.target:
        print(f"[staged] failed target={config.mini.target} success={len(processed)}", flush=True)
        write_manifest(config.mini.manifest_output, processed, config.mini)
        return 2
    write_manifest(config.mini.manifest_output, processed, config.mini)
    print(f"[staged] wrote manifest={config.mini.manifest_output} documents={len(processed)}", flush=True)
    return 0


def config_from_args(args: argparse.Namespace) -> StagedConfig:
    main_tex_names = tuple(name.strip() for name in args.main_tex_names.split(",") if name.strip())
    if not main_tex_names:
        raise ValueError("--main-tex-names must contain at least one filename")
    target = int(args.target)
    if target <= 0:
        raise ValueError("--target must be positive")
    mini = MiniDatasetConfig(
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
        max_candidates=int(args.max_candidates) if args.max_candidates is not None else None,
        main_tex_names=main_tex_names,
        mineru_command=args.mineru_command,
        mineru_timeout=int(args.mineru_timeout),
        similarity_threshold=float(args.similarity_threshold),
        max_orphan_ratio=float(args.max_orphan_ratio),
        max_unmapped_tex_ratio=float(args.max_unmapped_tex_ratio),
        max_isolated_node_ratio=float(args.max_isolated_node_ratio),
        min_non_none_edges=int(args.min_non_none_edges),
        min_candidate_recall=float(args.min_candidate_recall),
        reuse_existing=not args.no_reuse_existing,
        force_mineru=False,
        force_json=bool(args.force_json),
        force_graph=bool(args.force_graph),
        force_label=bool(args.force_label),
        dry_run=bool(args.dry_run),
    )
    return StagedConfig(
        mini=mini,
        run_name=str(args.run_name),
        stage_root=args.stage_root.resolve() / str(args.run_name),
        mineru_batch_size=max(1, int(args.mineru_batch_size)),
        mineru_batch_timeout=max(1, int(args.mineru_batch_timeout)),
        mineru_batch_command=str(args.mineru_batch_command),
        mineru_fallback_single=not bool(args.no_mineru_fallback_single),
        process_workers=max(1, int(args.process_workers)),
        process_backlog=max(1, int(args.process_backlog)),
        preflight_tex=not bool(args.skip_preflight),
        skip_mineru_stage=bool(args.skip_mineru_stage),
        process_existing_mineru_only=not bool(args.process_all_candidates),
        exclude_manifests=tuple(path.resolve() for path in args.exclude_manifest),
        max_preflight_fail_ratio=float(args.max_preflight_fail_ratio),
    )


def prepare_candidates(config: StagedConfig) -> list[CandidateSample]:
    candidates = scan_candidates(config.mini)
    excluded = load_excluded_document_ids(config.exclude_manifests)
    if excluded:
        candidates = [candidate for candidate in candidates if candidate.document_id not in excluded]
    if config.mini.max_candidates is not None:
        candidates = candidates[: config.mini.max_candidates]
    return candidates


def print_dry_run(config: StagedConfig, candidates: list[CandidateSample]) -> None:
    excluded = load_excluded_document_ids(config.exclude_manifests)
    mineru_existing = sum(1 for candidate in candidates if has_mineru_output(candidate, config))
    labeled_existing = sum(1 for candidate in candidates if has_valid_labeled_graph(candidate, config))
    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "target": config.mini.target,
                "excluded_success_ids": len(excluded),
                "existing_mineru_outputs": mineru_existing,
                "existing_valid_labeled_graphs": labeled_existing,
                "mineru_missing": len(candidates) - mineru_existing,
                "process_workers": config.process_workers,
                "mineru_batch_size": config.mineru_batch_size,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    for candidate in candidates[: min(len(candidates), config.mini.target)]:
        print(f"dry_run candidate id={candidate.document_id} pdf={candidate.pdf_path} tex={candidate.main_tex_path}")


def run_preflight_stage(candidates: list[CandidateSample], config: StagedConfig) -> list[CandidateSample]:
    if not config.preflight_tex:
        return candidates
    ok: list[CandidateSample] = []
    failed = 0
    stage_log = config.stage_root / "preflight_errors.jsonl"
    stage_log.parent.mkdir(parents=True, exist_ok=True)
    if stage_log.exists():
        stage_log.unlink()
    for candidate in progress_iter(candidates):
        result = preflight_tex(candidate, config)
        if result.ok:
            ok.append(candidate)
        else:
            failed += 1
            append_jsonl(stage_log, preflight_result_payload(result))
    fail_ratio = failed / max(1, len(candidates))
    print(f"[staged] preflight ok={len(ok)} failed={failed} fail_ratio={fail_ratio:.2%}", flush=True)
    if fail_ratio > config.max_preflight_fail_ratio:
        raise RuntimeError(
            f"preflight failure ratio too high: {fail_ratio:.2%} > {config.max_preflight_fail_ratio:.2%}"
        )
    return ok


def preflight_tex(candidate: CandidateSample, config: StagedConfig) -> PreflightResult:
    try:
        labeler = AlignmentLabeler(
            content_json_path=Path("__preflight_content__.json"),
            tex_path=candidate.main_tex_path,
            graph_path=Path("__preflight_graph__.pt"),
            config=AlignmentLabelerConfig(
                similarity_threshold=config.mini.similarity_threshold,
                max_orphan_ratio=config.mini.max_orphan_ratio,
                max_unmapped_tex_ratio=config.mini.max_unmapped_tex_ratio,
                max_isolated_node_ratio=config.mini.max_isolated_node_ratio,
                min_section_nodes=1,
                abort_on_bad_alignment=False,
            ),
        )
        tex_nodes = labeler.parse_tex_nodes()
        alignable = sum(1 for node in tex_nodes if node.node_type in ALIGNABLE_TEX_NODE_TYPES)
        sections = sum(1 for node in tex_nodes if node.node_type == "section")
        if alignable <= 0:
            raise ValueError("no alignable TeX nodes")
        return PreflightResult(
            candidate=candidate,
            ok=True,
            tex_node_count=len(tex_nodes),
            section_count=sections,
            alignable_count=alignable,
        )
    except LayoutBreakerException as exc:
        return PreflightResult(candidate=candidate, ok=False, error_type=type(exc).__name__, error=str(exc))
    except Exception as exc:
        return PreflightResult(candidate=candidate, ok=False, error_type=type(exc).__name__, error=str(exc))


def run_mineru_stage(candidates: list[CandidateSample], config: StagedConfig) -> None:
    missing = [candidate for candidate in candidates if not has_mineru_output(candidate, config)]
    print(
        f"[staged] mineru existing={len(candidates) - len(missing)} missing={len(missing)} batch_size={config.mineru_batch_size}",
        flush=True,
    )
    if not missing:
        return
    batch_root = config.stage_root / "mineru_batches"
    log_root = config.stage_root / "mineru_logs"
    shutil.rmtree(batch_root, ignore_errors=True)
    batch_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    for batch_index, batch in enumerate(chunked(missing, config.mineru_batch_size)):
        batch_dir = batch_root / f"batch_{batch_index:05d}"
        prepare_pdf_batch_dir(batch, batch_dir)
        log_path = log_root / f"batch_{batch_index:05d}.log"
        command = format_mineru_batch_command(batch_dir, config)
        print(f"[staged] mineru batch={batch_index} size={len(batch)} cmd={command}", flush=True)
        completed = run_command_with_process_group_timeout(
            command,
            cwd=config.mini.project_root,
            timeout=config.mineru_batch_timeout,
            log_path=log_path,
        )
        if completed.returncode != 0:
            print(
                f"[staged] mineru batch failed batch={batch_index} returncode={completed.returncode} log={log_path}",
                flush=True,
            )
            if config.mineru_fallback_single:
                run_single_mineru_fallback(batch, config, log_root)
        missing_after = [candidate for candidate in batch if not has_mineru_output(candidate, config)]
        if missing_after:
            for candidate in missing_after:
                append_processing_error(config.mini.error_log, candidate, "MinerUMissingOutput", "MinerU did not produce content output")
            print(f"[staged] mineru missing_after batch={batch_index} count={len(missing_after)}", flush=True)


def run_single_mineru_fallback(candidates: list[CandidateSample], config: StagedConfig, log_root: Path) -> None:
    for candidate in candidates:
        if has_mineru_output(candidate, config):
            continue
        command = format_mineru_command(candidate, config.mini)
        log_path = log_root / f"{candidate.document_id}_single_fallback.log"
        print(f"[staged] mineru fallback id={candidate.document_id}", flush=True)
        completed = run_command_with_process_group_timeout(
            command,
            cwd=config.mini.project_root,
            timeout=config.mini.mineru_timeout,
            log_path=log_path,
        )
        if completed.returncode != 0:
            append_processing_error(
                config.mini.error_log,
                candidate,
                "MinerUFallbackFailed",
                f"returncode={completed.returncode} log={log_path}",
            )


def run_processing_stage(candidates: list[CandidateSample], config: StagedConfig) -> list[ProcessedSample]:
    processed: list[ProcessedSample] = load_existing_processed_samples(candidates, config)
    if len(processed) >= config.mini.target:
        return processed[: config.mini.target]

    pending_candidates = [
        candidate
        for candidate in candidates
        if candidate.document_id not in {sample.document_id for sample in processed}
        and (not config.process_existing_mineru_only or has_mineru_output(candidate, config))
    ]
    print(
        f"[staged] process pending={len(pending_candidates)} existing_success={len(processed)} "
        f"workers={config.process_workers}",
        flush=True,
    )

    config.mini.error_log.parent.mkdir(parents=True, exist_ok=True)
    if not config.mini.error_log.exists():
        config.mini.error_log.write_text("", encoding="utf-8")

    with ProcessPoolExecutor(max_workers=config.process_workers) as executor:
        futures: set[Future[WorkerResult]] = set()
        cursor = 0
        while len(processed) < config.mini.target and (cursor < len(pending_candidates) or futures):
            while (
                cursor < len(pending_candidates)
                and len(futures) < config.process_backlog
                and len(processed) + len(futures) < config.mini.target + config.process_backlog
            ):
                candidate = pending_candidates[cursor]
                cursor += 1
                futures.add(executor.submit(process_candidate_worker, candidate, config.mini))
            done, futures = wait(futures, timeout=2.0, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                if result.ok and result.sample is not None:
                    processed.append(result.sample)
                    print(
                        f"[staged] success id={result.sample.document_id} "
                        f"success={len(processed)}/{config.mini.target} "
                        f"labels={result.sample.label_counts} "
                        f"orphan_ratio={result.sample.orphan_ratio:.2%}",
                        flush=True,
                    )
                    write_manifest(config.mini.manifest_output, processed, config.mini)
                else:
                    append_worker_error(config.mini.error_log, result)
                    print(
                        f"[staged] skip id={result.candidate.document_id} "
                        f"error={result.error_type}: {result.error}",
                        flush=True,
                    )
    return processed[: config.mini.target]


def process_candidate_worker(candidate: CandidateSample, config: MiniDatasetConfig) -> WorkerResult:
    try:
        sample = process_candidate(candidate, config)
        return WorkerResult(ok=True, candidate=candidate, sample=sample)
    except Exception as exc:
        return WorkerResult(
            ok=False,
            candidate=candidate,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(limit=20),
        )


def load_existing_processed_samples(candidates: list[CandidateSample], config: StagedConfig) -> list[ProcessedSample]:
    samples: list[ProcessedSample] = []
    for candidate in candidates:
        if len(samples) >= config.mini.target:
            break
        paths = sample_paths(candidate, config.mini)
        if not graph_is_valid_labeled(paths["labeled_graph"], config.mini):
            continue
        try:
            from scripts.pipeline.build_mini_dataset import summarize_processed_sample

            samples.append(summarize_processed_sample(candidate, paths))
        except Exception:
            continue
    if samples:
        print(f"[staged] reuse valid labeled graphs={len(samples)}", flush=True)
        write_manifest(config.mini.manifest_output, samples, config.mini)
    return samples


def has_valid_labeled_graph(candidate: CandidateSample, config: StagedConfig) -> bool:
    return graph_is_valid_labeled(sample_paths(candidate, config.mini)["labeled_graph"], config.mini)


def has_mineru_output(candidate: CandidateSample, config: StagedConfig) -> bool:
    return find_mineru_content_source(candidate.document_id, config.mini.mineru_output_dir) is not None


def prepare_pdf_batch_dir(candidates: list[CandidateSample], batch_dir: Path) -> None:
    shutil.rmtree(batch_dir, ignore_errors=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        target = batch_dir / f"{candidate.document_id}.pdf"
        try:
            os.link(candidate.pdf_path, target)
        except OSError:
            try:
                target.symlink_to(candidate.pdf_path)
            except OSError:
                shutil.copy2(candidate.pdf_path, target)


def format_mineru_batch_command(batch_dir: Path, config: StagedConfig) -> str:
    values = {
        "pdf_batch_dir": shlex.quote(str(batch_dir)),
        "pdf_dir": shlex.quote(str(batch_dir)),
        "mineru_output_dir": shlex.quote(str(config.mini.mineru_output_dir)),
        "output_dir": shlex.quote(str(config.mini.mineru_output_dir)),
    }
    return config.mineru_batch_command.format(**values)


def default_mineru_batch_command() -> str:
    autodl_mineru = Path("/root/miniconda3/envs/mineru/bin/mineru")
    executable = str(autodl_mineru) if autodl_mineru.exists() else "mineru"
    return f"{shlex.quote(executable)} -p {{pdf_batch_dir}} -o {{mineru_output_dir}} -m auto -b pipeline"


def chunked(items: list[CandidateSample], size: int) -> list[list[CandidateSample]]:
    return [items[index : index + size] for index in range(0, len(items), max(1, size))]


def load_excluded_document_ids(paths: tuple[Path, ...]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        documents = payload.get("documents", []) if isinstance(payload, dict) else []
        for row in documents:
            if isinstance(row, dict) and row.get("document_id"):
                excluded.add(str(row["document_id"]))
    return excluded


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def append_processing_error(path: Path, candidate: CandidateSample, error_type: str, error: str) -> None:
    append_jsonl(
        path,
        {
            "document_id": candidate.document_id,
            "pdf_path": str(candidate.pdf_path),
            "tex_path": str(candidate.main_tex_path),
            "error_type": error_type,
            "error": error,
        },
    )


def append_worker_error(path: Path, result: WorkerResult) -> None:
    append_jsonl(
        path,
        {
            "document_id": result.candidate.document_id,
            "pdf_path": str(result.candidate.pdf_path),
            "tex_path": str(result.candidate.main_tex_path),
            "error_type": result.error_type,
            "error": result.error,
            "traceback": result.traceback,
        },
    )


def preflight_result_payload(result: PreflightResult) -> dict[str, Any]:
    return {
        "document_id": result.candidate.document_id,
        "pdf_path": str(result.candidate.pdf_path),
        "tex_path": str(result.candidate.main_tex_path),
        "error_type": result.error_type,
        "error": result.error,
        "tex_node_count": result.tex_node_count,
        "section_count": result.section_count,
        "alignable_count": result.alignable_count,
    }


def lambda_run_name() -> str:
    return "v7_staged_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
