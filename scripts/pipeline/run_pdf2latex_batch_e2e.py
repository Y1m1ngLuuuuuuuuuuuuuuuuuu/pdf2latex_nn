#!/usr/bin/env python3
"""Run canonical PDF2LaTeX E2E cases from a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.run_pdf2latex_e2e import run_case  # noqa: E402
from src.pipeline.e2e_contract import case_config_from_manifest_item, load_manifest  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--renderer", default="ir", choices=("ir",))
    parser.add_argument("--use-existing-mineru", action="store_true")
    parser.add_argument("--enable-frontmatter-ir-renderer-experimental", action="store_true")
    parser.add_argument("--enable-float-caption-materialization-experimental", action="store_true")
    parser.add_argument("--enable-table-safe-fallback-experimental", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--visual-qa", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--compile-engine", default="auto")
    parser.add_argument("--compile-timeout", type=int, default=120)
    parser.add_argument("--metrics-timeout", type=int, default=60)
    parser.add_argument("--visual-qa-timeout", type=int, default=45)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    items = load_manifest(args.manifest)
    defaults = {
        "renderer": args.renderer,
        "use_existing_mineru": args.use_existing_mineru,
        "enable_frontmatter_ir_renderer_experimental": args.enable_frontmatter_ir_renderer_experimental,
        "enable_float_caption_materialization_experimental": args.enable_float_caption_materialization_experimental,
        "enable_table_safe_fallback_experimental": args.enable_table_safe_fallback_experimental,
        "compile": args.compile,
        "evaluate": args.evaluate,
        "visual_qa": args.visual_qa,
        "no_tex_source_inference": True,
    }
    configs = [case_config_from_manifest_item(item, output_root=args.output_root, defaults=defaults) for item in items]

    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for config in configs:
            results.append(run_or_skip(config, args))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_doc = {pool.submit(run_or_skip, config, args): config.doc_id for config in configs}
            for future in as_completed(future_to_doc):
                results.append(future.result())
    results.sort(key=lambda row: row.get("doc_id", ""))
    write_rollups(args.output_root, results)
    return 0


def run_or_skip(config, args) -> dict[str, Any]:
    if args.skip_existing and (config.output_dir / "CASE_SUMMARY.md").exists() and (config.output_dir / "07_failure" / "failure_taxonomy.json").exists():
        summary_path = config.output_dir / "case_summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {"doc_id": config.doc_id, "status": "skipped_existing", "stages": [], "failures": []}
    try:
        return run_case(
            config,
            compile_engine=args.compile_engine,
            compile_timeout=args.compile_timeout,
            metrics_timeout=args.metrics_timeout,
            visual_qa_timeout=args.visual_qa_timeout,
        )
    except Exception as exc:  # pragma: no cover - defensive batch isolation.
        config.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "stage": "unknown",
            "failure_type": "unknown",
            "severity": "blocking",
            "message": f"{type(exc).__name__}: {exc}",
            "recommended_next_action": "inspect_batch_exception",
        }
        summary = {
            "schema_version": "pdf2latex_e2e_case_summary_v1",
            "doc_id": config.doc_id,
            "stratum": config.stratum,
            "status": "batch_exception",
            "stages": [],
            "failures": [failure],
            "outputs": {},
            "main_failure_type": "unknown",
        }
        (config.output_dir / "case_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (config.output_dir / "CASE_SUMMARY.md").write_text(f"# Case Summary: {config.doc_id}\n\n- status: batch_exception\n- error: {exc}\n", encoding="utf-8")
        return summary


def write_rollups(output_root: Path, results: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    stage_counter: Counter[str] = Counter()
    stage_ok: Counter[str] = Counter()
    failure_counter: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for result in results:
        for stage in result.get("stages", []):
            name = str(stage.get("stage"))
            stage_counter[name] += 1
            if stage.get("status") == "ok":
                stage_ok[name] += 1
        for failure in result.get("failures", []):
            failure_counter[str(failure.get("failure_type") or "unknown")] += 1
        rows.append(flatten_result(result))

    rollup = {
        "schema_version": "pdf2latex_e2e_batch_rollup_v1",
        "doc_count": len(results),
        "status_counts": dict(Counter(str(result.get("status")) for result in results)),
        "stage_success_rates": {
            stage: {"ok": stage_ok[stage], "total": total, "rate": stage_ok[stage] / total if total else None}
            for stage, total in sorted(stage_counter.items())
        },
        "failure_type_counts": dict(failure_counter),
        "results": results,
    }
    (output_root / "batch_rollup.json").write_text(json.dumps(rollup, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        with (output_root / "batch_rollup.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    write_rollup_md(output_root / "batch_rollup.md", rows, rollup)


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    stage_success = result.get("stage_success") or {}
    return {
        "doc_id": result.get("doc_id"),
        "stratum": result.get("stratum"),
        "status": result.get("status"),
        "generated_tex": bool((result.get("outputs") or {}).get("generated_tex")),
        "generated_pdf": bool((result.get("outputs") or {}).get("generated_pdf")),
        "compile_success": result.get("compile_success"),
        "comparison_metrics": result.get("comparison_metrics"),
        "visual_qa": result.get("visual_qa_status"),
        "main_failure_type": result.get("main_failure_type"),
        "input_discovery": stage_success.get("input_discovery"),
        "fact_layer": stage_success.get("fact_layer"),
        "document_ir": stage_success.get("document_ir"),
        "render_tree_ir": stage_success.get("render_tree_ir"),
        "generation": stage_success.get("generation"),
        "compile": stage_success.get("compile"),
        "comparison_conversion": stage_success.get("comparison_conversion"),
        "structure_metrics": stage_success.get("structure_metrics"),
        "visual_qa_stage": stage_success.get("visual_qa"),
    }


def write_rollup_md(path: Path, rows: list[dict[str, Any]], rollup: dict[str, Any]) -> None:
    lines = ["# Smoke Batch Rollup", "", f"- docs: {rollup['doc_count']}", ""]
    lines.append("## Stage Success Rates")
    for stage, payload in rollup.get("stage_success_rates", {}).items():
        lines.append(f"- {stage}: {payload['ok']}/{payload['total']} ({payload['rate']:.3f})")
    lines.extend(["", "## Failure Types"])
    for failure_type, count in rollup.get("failure_type_counts", {}).items():
        lines.append(f"- {failure_type}: {count}")
    lines.extend(["", "## Cases", "", "| doc_id | stratum | status | generated_tex | generated_pdf | compile_success | comparison_metrics | visual_qa | main_failure_type |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for row in rows:
        lines.append(
            f"| {row['doc_id']} | {row['stratum']} | {row['status']} | {row['generated_tex']} | "
            f"{row['generated_pdf']} | {row['compile_success']} | {row['comparison_metrics']} | "
            f"{row['visual_qa']} | {row['main_failure_type']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
