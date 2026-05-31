#!/usr/bin/env python3
"""Run or normalize one canonical PDF2LaTeX E2E case.

This wrapper stabilizes the framework output contract.  It can reuse existing
artifacts and records skipped stages explicitly instead of silently failing.
It does not run MinerU and does not use source TeX for inference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.comparison_structure import latex_file_to_comparison, write_comparison_json  # noqa: E402
from src.evaluation.compile_eval import compile_latex  # noqa: E402
from src.evaluation.structure_metrics import evaluate_comparison_structures, load_comparison_json  # noqa: E402
from src.evaluation.visual_qa import compare_pdf_layouts  # noqa: E402
from src.generation.ir_renderer import IRLatexRenderConfig  # noqa: E402
from src.generation.render_surface import render_original_like_document  # noqa: E402
from src.ir import DocumentIR, RenderTreeIR  # noqa: E402
from src.ir.serialization import read_dataclass_json  # noqa: E402
from src.pipeline.e2e_contract import E2ECaseConfig  # noqa: E402
from src.pipeline.e2e_outputs import copy_if_exists, ensure_e2e_layout, write_case_summary, write_json, write_stage_skipped  # noqa: E402
from src.pipeline.failure_taxonomy import E2EFailure, classify_compile_failure, exception_failure, make_failure, write_failure_taxonomy  # noqa: E402
from src.reasoning.float_caption_layout import apply_float_caption_layout  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--mineru-output", type=Path)
    parser.add_argument("--existing-artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--renderer", default="ir", choices=("ir",))
    parser.add_argument("--use-existing-mineru", action="store_true")
    parser.add_argument("--enable-frontmatter-ir-renderer-experimental", action="store_true")
    parser.add_argument("--enable-float-caption-materialization-experimental", action="store_true")
    parser.add_argument("--enable-table-safe-fallback-experimental", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--visual-qa", action="store_true")
    parser.add_argument("--no-tex-source-inference", action="store_true")
    parser.add_argument("--gold-comparison", type=Path)
    parser.add_argument("--existing-facts-path", type=Path)
    parser.add_argument("--document-ir-path", type=Path)
    parser.add_argument("--render-tree-ir-path", type=Path)
    parser.add_argument("--generated-tex-path", type=Path)
    parser.add_argument("--generated-pdf-path", type=Path)
    parser.add_argument("--stratum", default="unknown")
    parser.add_argument("--compile-engine", default="auto")
    parser.add_argument("--compile-timeout", type=int, default=120)
    parser.add_argument("--metrics-timeout", type=int, default=60)
    parser.add_argument("--visual-qa-timeout", type=int, default=45)
    return parser


def config_from_args(args: argparse.Namespace) -> E2ECaseConfig:
    return E2ECaseConfig(
        doc_id=args.doc_id,
        output_dir=args.output_dir,
        stratum=args.stratum,
        pdf=args.pdf,
        mineru_output=args.mineru_output,
        existing_artifact_root=args.existing_artifact_root,
        gold_comparison=args.gold_comparison,
        existing_facts_path=args.existing_facts_path,
        document_ir_path=args.document_ir_path,
        render_tree_ir_path=args.render_tree_ir_path,
        generated_tex_path=args.generated_tex_path,
        generated_pdf_path=args.generated_pdf_path,
        renderer=args.renderer,
        use_existing_mineru=args.use_existing_mineru,
        enable_frontmatter_ir_renderer_experimental=args.enable_frontmatter_ir_renderer_experimental,
        enable_float_caption_materialization_experimental=args.enable_float_caption_materialization_experimental,
        enable_table_safe_fallback_experimental=args.enable_table_safe_fallback_experimental,
        compile=args.compile,
        evaluate=args.evaluate,
        visual_qa=args.visual_qa,
        no_tex_source_inference=args.no_tex_source_inference,
    )


def run_case(
    config: E2ECaseConfig,
    *,
    compile_engine: str = "auto",
    compile_timeout: int = 120,
    metrics_timeout: int = 60,
    visual_qa_timeout: int = 45,
) -> dict[str, Any]:
    layout = ensure_e2e_layout(config.output_dir)
    failures: list[E2EFailure] = []
    stages: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}

    discovered = discover_artifacts(config)
    write_json(layout.stage_dir("input") / "input_manifest.json", {"config": config.to_dict(), "discovered": stringify(discovered)})
    if discovered.get("pdf"):
        if copy_if_exists(discovered["pdf"], layout.stage_dir("input") / "original.pdf"):
            outputs["original_pdf"] = str(layout.stage_dir("input") / "original.pdf")
        (layout.stage_dir("input") / "ORIGINAL_PATH.txt").write_text(str(discovered["pdf"]) + "\n", encoding="utf-8")
        stages.append(stage_ok("input_discovery", "input artifacts discovered"))
    else:
        write_stage_skipped(layout.stage_dir("input"), stage="input_discovery", reason="missing_original_pdf")
        failures.append(
            make_failure(
                stage="input_discovery",
                failure_type="missing_original_pdf",
                severity="recoverable",
                message="Original PDF was not found in explicit paths or artifact root.",
                recommended_next_action="provide_pdf_or_artifact_root",
            )
        )
        stages.append(stage_skip("input_discovery", "missing_original_pdf"))

    facts_path = materialize_facts(layout, discovered, failures, stages)
    document_ir_path = materialize_optional_json(
        layout,
        discovered.get("document_ir"),
        stage_key="ir",
        output_name="document_ir.json",
        stage_name="document_ir",
        missing_type="document_ir_build_error",
        failures=failures,
        stages=stages,
        required=False,
    )
    render_tree_path = materialize_optional_json(
        layout,
        discovered.get("render_tree_ir"),
        stage_key="ir",
        output_name="render_tree_ir.json",
        stage_name="render_tree_ir",
        missing_type="render_tree_build_error",
        failures=failures,
        stages=stages,
        required=False,
    )
    ir_validation = {
        "schema_version": "pdf2latex_e2e_ir_validation_v1",
        "document_ir_exists": bool(document_ir_path and document_ir_path.exists()),
        "render_tree_ir_exists": bool(render_tree_path and render_tree_path.exists()),
        "facts_exists": bool(facts_path and facts_path.exists()),
    }
    write_json(layout.stage_dir("ir") / "ir_validation.json", ir_validation)

    generated_tex = materialize_generated_tex(layout, discovered, document_ir_path, render_tree_path, config, failures, stages)
    if generated_tex:
        outputs["generated_tex"] = str(generated_tex)
    generation_report = {
        "schema_version": "pdf2latex_e2e_generation_report_v1",
        "generated_tex": str(generated_tex) if generated_tex else None,
        "source": str(discovered.get("generated_tex")) if discovered.get("generated_tex") else None,
        "renderer": config.renderer,
        "frontmatter_renderer_experimental": config.enable_frontmatter_ir_renderer_experimental,
        "float_caption_materialization_experimental": config.enable_float_caption_materialization_experimental,
        "float_caption_materialization": getattr(materialize_generated_tex, "_last_float_caption_diag", None),
        "table_safe_fallback_experimental": config.enable_table_safe_fallback_experimental,
        "note": "Generated TeX was rendered from DocumentIR/RenderTreeIR when available; no TeX source inference was used.",
    }
    write_json(layout.stage_dir("generation") / "generation_report.json", generation_report)

    generated_pdf = materialize_compile(layout, generated_tex, discovered, config, failures, stages, compile_engine, compile_timeout)
    if generated_pdf:
        outputs["generated_pdf"] = str(generated_pdf)

    comparison_path, metrics_path = materialize_comparison(layout, generated_tex, discovered, config, failures, stages, metrics_timeout)
    if comparison_path:
        outputs["comparison_structure"] = str(comparison_path)
    if metrics_path:
        outputs["metrics"] = str(metrics_path)

    visual_path = materialize_visual_qa(layout, discovered.get("pdf"), generated_pdf, config, failures, stages, visual_qa_timeout)
    outputs["visual_qa"] = str(visual_path)

    status = "completed" if not any(f.severity == "blocking" for f in failures) else "completed_with_blocking_failures"
    taxonomy = write_failure_taxonomy(layout.stage_dir("failure") / "failure_taxonomy.json", failures, doc_id=config.doc_id, status=status)
    write_failure_trace(layout.stage_dir("failure") / "failure_trace.md", failures)
    outputs["failure_taxonomy"] = str(layout.stage_dir("failure") / "failure_taxonomy.json")

    summary = {
        "schema_version": "pdf2latex_e2e_case_summary_v1",
        "doc_id": config.doc_id,
        "stratum": config.stratum,
        "artifact_root": str(config.existing_artifact_root) if config.existing_artifact_root else None,
        "status": status,
        "stages": stages,
        "failures": taxonomy["failures"],
        "outputs": outputs,
        "stage_success": stage_success_map(stages),
        "main_failure_type": first_failure_type(failures),
        "compile_success": compile_success(layout.stage_dir("compile") / "compile_report.json"),
        "comparison_metrics": bool(metrics_path),
        "visual_qa_status": read_status_field(visual_path),
    }
    write_json(config.output_dir / "case_summary.json", summary)
    write_case_summary(layout.case_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def discover_artifacts(config: E2ECaseConfig) -> dict[str, Path | None]:
    root = config.existing_artifact_root
    candidates: dict[str, Path | None] = {
        "pdf": first_existing(
            config.pdf,
            root / "original.pdf" if root else None,
            root / f"{config.doc_id}_original.pdf" if root else None,
            root / "01_input" / "original.pdf" if root else None,
        ),
        "facts": first_existing(
            config.existing_facts_path,
            root / "observable_facts.json" if root else None,
            *list(root.glob("*content_list_v8*.json")) if root and root.exists() else [],
            root / "01_facts" / "observable_facts.json" if root else None,
        ),
        "document_ir": first_existing(
            config.document_ir_path,
            root / "document_ir.json" if root else None,
            root / "02_ir" / "document_ir.json" if root else None,
        ),
        "render_tree_ir": first_existing(
            config.render_tree_ir_path,
            root / "render_tree_ir.json" if root else None,
            root / "02_ir" / "render_tree_ir.json" if root else None,
        ),
        "generated_tex": first_existing(
            config.generated_tex_path,
            root / "generated.tex" if root else None,
            root / "03_generation" / "generated.tex" if root else None,
            root / "07_generation" / "generated.tex" if root else None,
        ),
        "generated_pdf": first_existing(
            config.generated_pdf_path,
            root / "generated.pdf" if root else None,
            root / "04_compile" / "generated.pdf" if root else None,
            root / "07_generation" / "generated.pdf" if root else None,
            root / f"{config.doc_id}_generated.pdf" if root else None,
        ),
        "gold_comparison": first_existing(
            config.gold_comparison,
            root / "gold_comparison_structure.json" if root else None,
            root / "08_evaluation" / "gold_comparison_structure.json" if root else None,
        ),
        "existing_metrics": first_existing(
            root / "metrics.json" if root else None,
            root / "08_evaluation" / "ours_metrics_current.json" if root else None,
            root / "ours_metrics_current.json" if root else None,
        ),
        "existing_comparison": first_existing(
            root / "comparison_structure.json" if root else None,
            root / "08_evaluation" / "ours_comparison_structure_current.json" if root else None,
            root / "ours_comparison_structure_current.json" if root else None,
        ),
    }
    return candidates


def first_existing(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and path.exists() and path.is_file():
            return path
    return None


def materialize_facts(layout, discovered, failures, stages) -> Path | None:
    facts = discovered.get("facts")
    stage_dir = layout.stage_dir("facts")
    if facts and copy_if_exists(facts, stage_dir / "observable_facts.json"):
        summary = summarize_json_file(stage_dir / "observable_facts.json") | {"source": str(facts)}
        write_json(stage_dir / "facts_summary.json", summary)
        stages.append(stage_ok("fact_layer", "observable facts available"))
        return stage_dir / "observable_facts.json"
    write_stage_skipped(stage_dir, stage="fact_layer", reason="missing_observable_facts")
    write_json(stage_dir / "facts_summary.json", {"status": "missing", "source": None})
    failures.append(
        make_failure(
            stage="fact_layer",
            failure_type="missing_observable_facts",
            severity="recoverable",
            message="No existing observable facts/internal v8 artifact family file was found.",
            recommended_next_action="provide_v8_or_mineru_artifacts",
        )
    )
    stages.append(stage_skip("fact_layer", "missing_observable_facts"))
    return None


def materialize_optional_json(layout, source, *, stage_key, output_name, stage_name, missing_type, failures, stages, required) -> Path | None:
    stage_dir = layout.stage_dir(stage_key)
    if source and copy_if_exists(source, stage_dir / output_name):
        stages.append(stage_ok(stage_name, f"{output_name} available"))
        return stage_dir / output_name
    if required:
        severity = "blocking"
    else:
        severity = "warning"
    write_stage_skipped(stage_dir, stage=stage_name, reason=f"missing_{output_name}")
    failures.append(
        make_failure(
            stage=stage_name if stage_name in {"document_ir", "render_tree_ir"} else "artifact_missing",
            failure_type=missing_type,
            severity=severity,
            message=f"{output_name} not found; stage recorded as skipped.",
            recommended_next_action=f"generate_or_provide_{output_name}",
        )
    )
    stages.append(stage_skip(stage_name, f"missing_{output_name}"))
    return None


def materialize_generated_tex(layout, discovered, document_ir_path, render_tree_path, config, failures, stages) -> Path | None:
    stage_dir = layout.stage_dir("generation")
    materialize_generated_tex._last_float_caption_diag = None
    if document_ir_path and render_tree_path and document_ir_path.exists() and render_tree_path.exists():
        try:
            document = read_dataclass_json(document_ir_path, DocumentIR)
            render_tree = read_dataclass_json(render_tree_path, RenderTreeIR)
            if config.enable_float_caption_materialization_experimental:
                render_tree, float_caption_result = apply_float_caption_layout(document, render_tree, enabled=True)
                materialize_generated_tex._last_float_caption_diag = float_caption_result.to_diagnostic()
            else:
                materialize_generated_tex._last_float_caption_diag = None
            asset_dir = stage_dir / "assets"
            asset_dir.mkdir(parents=True, exist_ok=True)
            latex = render_original_like_document(
                document,
                render_tree,
                config=IRLatexRenderConfig(
                    front_matter_renderer_experimental=config.enable_frontmatter_ir_renderer_experimental,
                    table_safe_fallback_experimental=config.enable_table_safe_fallback_experimental,
                    table_asset_output_dir=asset_dir,
                    figure_asset_output_dir=asset_dir,
                ),
                resolve_citations=False,
                source_tex_path=None,
            )
            target = stage_dir / "generated.tex"
            target.write_text(latex, encoding="utf-8")
            stages.append(stage_ok("generation", "generated.tex rendered from DocumentIR/RenderTreeIR"))
            return target
        except Exception as exc:
            failures.append(exception_failure(exc, stage="generation", failure_type="generation_error", traceback_path=stage_dir / "generation_traceback.txt"))
            write_stage_skipped(stage_dir, stage="generation", reason="ir_renderer_exception")
            stages.append(stage_fail("generation", "generation_error"))
            return None

    source = discovered.get("generated_tex")
    if source and copy_if_exists(source, stage_dir / "generated.tex"):
        copy_asset_dir_for_generated_tex(source, stage_dir)
        stages.append(stage_ok("generation", "generated.tex available"))
        return stage_dir / "generated.tex"
    write_stage_skipped(stage_dir, stage="generation", reason="generated_tex_missing")
    failures.append(
        make_failure(
            stage="generation",
            failure_type="generated_tex_missing",
            severity="blocking",
            message="No generated.tex was found and this wrapper does not run MinerU or complete renderer patches.",
            recommended_next_action="run_canonical_generation_or_provide_existing_generated_tex",
        )
    )
    stages.append(stage_skip("generation", "generated_tex_missing"))
    return None


def copy_asset_dir_for_generated_tex(source_tex: Path, target_generation_dir: Path) -> None:
    """Copy colocated asset directory so reused TeX can compile from its new cwd."""

    source_assets = source_tex.parent / "assets"
    if not source_assets.exists() or not source_assets.is_dir():
        return
    target_assets = target_generation_dir / "assets"
    shutil.copytree(source_assets, target_assets, dirs_exist_ok=True)


def materialize_compile(layout, generated_tex, discovered, config, failures, stages, engine, timeout) -> Path | None:
    stage_dir = layout.stage_dir("compile")
    existing_pdf = discovered.get("generated_pdf")
    if config.compile and generated_tex:
        try:
            report = compile_latex(generated_tex, output_dir=stage_dir, engine=engine, timeout=timeout, passes=2)
            write_json(stage_dir / "compile_report.json", report)
            (stage_dir / "compile.log").write_text(str(report.get("log_tail") or report.get("error_summary") or ""), encoding="utf-8")
            if report.get("success"):
                stages.append(stage_ok("compile", "compile succeeded"))
                return Path(report["output_pdf"]) if report.get("output_pdf") else stage_dir / "generated.pdf"
            failure_type = classify_compile_failure(report)
            failures.append(
                make_failure(
                    stage="compile",
                    failure_type=failure_type,
                    severity="recoverable",
                    message=str(report.get("error_summary") or "LaTeX compile failed."),
                    recommended_next_action="inspect_compile_log_and_renderer_contract",
                )
            )
            stages.append(stage_fail("compile", failure_type))
            return None
        except Exception as exc:  # pragma: no cover - depends on local TeX tools.
            failures.append(exception_failure(exc, stage="compile", failure_type="latex_compile_error", traceback_path=stage_dir / "compile_traceback.txt"))
            write_stage_skipped(stage_dir, stage="compile", reason="compile_exception")
            stages.append(stage_fail("compile", "compile_exception"))
            return None
    if existing_pdf and copy_if_exists(existing_pdf, stage_dir / "generated.pdf"):
        report = {"success": "existing_artifact", "skipped": True, "output_pdf": str(stage_dir / "generated.pdf")}
        write_json(stage_dir / "compile_report.json", report)
        (stage_dir / "compile.log").write_text("compile not rerun; existing generated.pdf reused\n", encoding="utf-8")
        stages.append(stage_ok("compile", "existing generated.pdf reused"))
        return stage_dir / "generated.pdf"
    write_stage_skipped(stage_dir, stage="compile", reason="compile_not_requested_or_pdf_missing")
    write_json(stage_dir / "compile_report.json", {"success": "not_run", "skipped": True, "output_pdf": None})
    (stage_dir / "compile.log").write_text("compile skipped or generated.pdf missing\n", encoding="utf-8")
    stages.append(stage_skip("compile", "compile_not_requested_or_pdf_missing"))
    return None


class StageTimeoutError(TimeoutError):
    """Raised when an optional E2E stage exceeds its local guard timeout."""


def run_with_alarm_timeout(callback, *, timeout: int, label: str):
    """Run a callback with a Unix alarm timeout when running in the main thread."""

    if timeout <= 0 or threading.current_thread() is not threading.main_thread():
        return callback()

    def _handle_timeout(_signum, _frame):
        raise StageTimeoutError(f"{label} timed out after {timeout}s")

    old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout)
    try:
        return callback()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def materialize_comparison(layout, generated_tex, discovered, config, failures, stages, metrics_timeout: int) -> tuple[Path | None, Path | None]:
    stage_dir = layout.stage_dir("comparison")
    if generated_tex:
        try:
            document = latex_file_to_comparison(generated_tex, doc_id=config.doc_id)
            comparison_path = stage_dir / "comparison_structure.json"
            write_comparison_json(document, comparison_path)
            stages.append(stage_ok("comparison_conversion", "ComparisonStructureV1 written"))
        except Exception as exc:
            failures.append(exception_failure(exc, stage="comparison_conversion", failure_type="comparison_conversion_error", traceback_path=stage_dir / "comparison_traceback.txt"))
            write_stage_skipped(stage_dir, stage="comparison_conversion", reason="conversion_exception")
            stages.append(stage_fail("comparison_conversion", "comparison_conversion_error"))
            return None, None
    elif discovered.get("existing_comparison") and copy_if_exists(discovered["existing_comparison"], stage_dir / "comparison_structure.json"):
        comparison_path = stage_dir / "comparison_structure.json"
        stages.append(stage_ok("comparison_conversion", "existing comparison structure reused"))
    else:
        write_stage_skipped(stage_dir, stage="comparison_conversion", reason="generated_tex_missing")
        stages.append(stage_skip("comparison_conversion", "generated_tex_missing"))
        return None, None

    if not config.evaluate:
        write_stage_skipped(stage_dir, stage="structure_metrics", reason="evaluate_not_requested")
        stages.append(stage_skip("structure_metrics", "evaluate_not_requested"))
        return comparison_path, None
    gold = discovered.get("gold_comparison")
    if not gold:
        write_stage_skipped(stage_dir, stage="structure_metrics", reason="gold_comparison_missing")
        failures.append(
            make_failure(
                stage="structure_metrics",
                failure_type="gold_comparison_missing",
                severity="warning",
                message="Gold ComparisonStructureV1 is missing; metrics skipped.",
                recommended_next_action="provide_gold_comparison_for_evaluation_only",
            )
        )
        stages.append(stage_skip("structure_metrics", "gold_comparison_missing"))
        return comparison_path, None
    try:
        metrics = run_with_alarm_timeout(
            lambda: evaluate_comparison_structures(load_comparison_json(gold), load_comparison_json(comparison_path)),
            timeout=metrics_timeout,
            label="structure_metrics",
        )
        metrics_path = stage_dir / "metrics.json"
        write_json(metrics_path, metrics)
        stages.append(stage_ok("structure_metrics", "metrics written"))
        return comparison_path, metrics_path
    except StageTimeoutError as exc:
        failures.append(
            make_failure(
                stage="structure_metrics",
                failure_type="comparison_conversion_error",
                severity="warning",
                message=str(exc),
                recommended_next_action="inspect_metrics_runtime_or_skip_for_smoke",
            )
        )
        write_stage_skipped(stage_dir, stage="structure_metrics", reason="metrics_timeout")
        stages.append(stage_skip("structure_metrics", "metrics_timeout"))
        return comparison_path, None
    except Exception as exc:
        failures.append(exception_failure(exc, stage="structure_metrics", failure_type="comparison_conversion_error", traceback_path=stage_dir / "metrics_traceback.txt"))
        stages.append(stage_fail("structure_metrics", "metrics_error"))
        return comparison_path, None


def materialize_visual_qa(layout, original_pdf, generated_pdf, config, failures, stages, visual_qa_timeout: int) -> Path:
    stage_dir = layout.stage_dir("visual_qa")
    visual_path = stage_dir / "visual_qa.json"
    if not config.visual_qa:
        write_stage_skipped(stage_dir, stage="visual_qa", reason="visual_qa_not_requested")
        payload = {"schema_version": "pdf2latex_e2e_visual_qa_v1", "status": "skipped", "reason": "visual_qa_not_requested"}
        write_json(visual_path, payload)
        stages.append(stage_skip("visual_qa", "visual_qa_not_requested"))
        return visual_path
    if not original_pdf or not generated_pdf or not Path(generated_pdf).exists():
        write_stage_skipped(stage_dir, stage="visual_qa", reason="pdf_pair_missing")
        payload = {
            "schema_version": "pdf2latex_e2e_visual_qa_v1",
            "status": "skipped_unavailable",
            "reason": "original or generated PDF missing",
            "screenshots_available": False,
        }
        write_json(visual_path, payload)
        failures.append(
            make_failure(
                stage="visual_qa",
                failure_type="visual_render_unavailable",
                severity="warning",
                message="Visual QA skipped because original/generated PDF pair is unavailable.",
                recommended_next_action="run_compile_or_provide_generated_pdf",
            )
        )
        stages.append(stage_skip("visual_qa", "pdf_pair_missing"))
        return visual_path
    try:
        result = run_with_alarm_timeout(
            lambda: compare_pdf_layouts(Path(original_pdf), Path(generated_pdf), max_pages=1),
            timeout=visual_qa_timeout,
            label="visual_qa",
        )
        result["schema_version"] = "pdf2latex_e2e_visual_qa_v1"
        result["status"] = "completed"
        result["screenshots_available"] = False
        write_json(visual_path, result)
        stages.append(stage_ok("visual_qa", "layout comparison completed"))
    except StageTimeoutError as exc:
        payload = {
            "schema_version": "pdf2latex_e2e_visual_qa_v1",
            "status": "skipped_unavailable",
            "reason": str(exc),
            "screenshots_available": False,
        }
        write_json(visual_path, payload)
        failures.append(
            make_failure(
                stage="visual_qa",
                failure_type="visual_render_unavailable",
                severity="warning",
                message=payload["reason"],
                recommended_next_action="increase_visual_qa_timeout_or_skip_for_smoke",
            )
        )
        stages.append(stage_skip("visual_qa", "visual_qa_timeout"))
    except Exception as exc:  # pragma: no cover - depends on optional render tools.
        payload = {
            "schema_version": "pdf2latex_e2e_visual_qa_v1",
            "status": "skipped_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "screenshots_available": False,
        }
        write_json(visual_path, payload)
        failures.append(
            make_failure(
                stage="visual_qa",
                failure_type="visual_render_unavailable",
                severity="warning",
                message=payload["reason"],
                recommended_next_action="install_or_configure_pdf_rendering_tools",
            )
        )
        stages.append(stage_skip("visual_qa", "visual_render_unavailable"))
    return visual_path


def summarize_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "error": str(exc)}
    if isinstance(payload, dict):
        return {
            "status": "available",
            "top_level_type": "dict",
            "top_level_keys": sorted(str(key) for key in payload.keys())[:30],
            "item_count": len(payload.get("items") or payload.get("blocks") or []),
        }
    if isinstance(payload, list):
        return {"status": "available", "top_level_type": "list", "item_count": len(payload)}
    return {"status": "available", "top_level_type": type(payload).__name__}


def write_failure_trace(path: Path, failures: list[E2EFailure]) -> None:
    lines = ["# Failure Trace", ""]
    if not failures:
        lines.append("- no failures")
    for failure in failures:
        lines.append(f"- {failure.severity} / {failure.stage} / {failure.failure_type}: {failure.message}")
        if failure.traceback_path:
            lines.append(f"  - traceback: `{failure.traceback_path}`")
        lines.append(f"  - next: {failure.recommended_next_action}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_ok(stage: str, message: str) -> dict[str, Any]:
    return {"stage": stage, "status": "ok", "message": message}


def stage_skip(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "status": "skipped", "reason": reason}


def stage_fail(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "status": "failed", "reason": reason}


def stage_success_map(stages: list[dict[str, Any]]) -> dict[str, bool]:
    return {str(stage.get("stage")): stage.get("status") == "ok" for stage in stages}


def first_failure_type(failures: list[E2EFailure]) -> str | None:
    if not failures:
        return None
    blocking = [failure for failure in failures if failure.severity == "blocking"]
    return (blocking[0] if blocking else failures[0]).failure_type


def compile_success(path: Path) -> bool | str | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("success")
    except Exception:
        return None


def read_status_field(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text()).get("status"))
    except Exception:
        return None


def stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: stringify(val) for key, val in value.items()}
    if isinstance(value, list):
        return [stringify(item) for item in value]
    return value


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.no_tex_source_inference:
        raise SystemExit("--no-tex-source-inference is required for the canonical E2E wrapper")
    config = config_from_args(args)
    run_case(
        config,
        compile_engine=args.compile_engine,
        compile_timeout=args.compile_timeout,
        metrics_timeout=args.metrics_timeout,
        visual_qa_timeout=args.visual_qa_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
