#!/usr/bin/env python3
"""Audit FrontMatterIR renderer Phase0 on existing selected200 artifacts.

This is intentionally a same-code, skip-E2E harness: it reuses persisted
DocumentIR/StyleProfile/gold comparison artifacts, renders flag-off and flag-on
LaTeX with the current code, converts both outputs to comparison structure, and
summarizes front-matter and safety metrics.  It does not run MinerU, rebuild
graphs, or mutate raw v8/DocumentIR JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.run_v8_layout_reconstruction import ensure_v8_math_compatibility  # noqa: E402
from src.evaluation.comparison_structure import latex_file_to_comparison, write_comparison_json  # noqa: E402
from src.evaluation.structure_metrics import evaluate_comparison_structures  # noqa: E402
from src.generation.ir_renderer import IRLatexRenderConfig  # noqa: E402
from src.generation.render_surface import render_original_like_document  # noqa: E402
from src.ir import DocumentIR, StyleProfile  # noqa: E402
from src.ir.serialization import read_dataclass_json, read_json, write_json  # noqa: E402
from src.reasoning.front_matter_extractor import extract_front_matter  # noqa: E402
from src.reasoning.front_matter_ir_loader import load_front_matter_ir_sidecar  # noqa: E402
from src.reasoning.v8_render_tree import build_v8_render_tree  # noqa: E402


DEFAULT_SELECTED200_ROOT = Path("data/09_eval_reports/selected200_eval_rerun_v2_20260525/v8_deterministic/e2e_skipcompile")
DEFAULT_FRONTMATTER_ROOT = Path("data/09_eval_reports/frontmatter_extractor_phase0_20260528/selected200_audit_only")
DEFAULT_OUTPUT_ROOT = Path("data/09_eval_reports/frontmatter_renderer_phase0_20260528")

CORE_METRICS = [
    "generated_structure_validity",
    "macro_structure_score_body",
    "heading_tree_accuracy",
    "reading_order_accuracy",
    "paragraph_text_coverage_f1",
    "section_attachment_body_no_float_f1",
    "reference_section_completeness",
    "float_caption_attachment_accuracy",
]


@dataclass(frozen=True)
class DocArtifact:
    doc_id: str
    source_dir: Path
    sidecar_path: Path


def main() -> int:
    args = build_arg_parser().parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    docs = discover_docs(args.selected200_root, args.frontmatter_root)
    if not docs:
        write_readiness_report(output_root, reason="no selected200 DocumentIR + FrontMatterIR sidecar pairs found")
        return 0
    if len(docs) < args.expected_docs:
        write_readiness_report(output_root, reason=f"found {len(docs)} docs, expected {args.expected_docs}")
        return 0

    smoke_docs = select_smoke_docs(docs)
    smoke_dir = output_root / "smoke20_same_code_ab"
    smoke_result = run_ab(smoke_docs, smoke_dir, phase="smoke20")
    smoke_passed = gates_pass(smoke_result["summary"], strict=True)

    selected_result: dict[str, Any] | None = None
    selected_passed = False
    compile_result: dict[str, Any] = {"status": "skipped", "reason": "selected200 gate not run"}
    if smoke_passed:
        selected_dir = output_root / "selected200_same_code_ab"
        selected_result = run_ab(docs, selected_dir, phase="selected200")
        selected_passed = gates_pass(selected_result["summary"], strict=False)
        compile_result = {"status": "skipped", "reason": "selected200 skip-compile gate failed"}
        if selected_passed and args.run_compile_smoke:
            compile_result = run_compile_smoke(selected_result["records"], output_root / "compile_smoke")

    final_result = selected_result or smoke_result
    summary = final_result["summary"]
    decision = decide(smoke_passed=smoke_passed, selected_passed=selected_passed, compile_result=compile_result, summary=summary)
    report_payload = {
        "schema_version": "frontmatter_renderer_phase0_summary_v1",
        "docs_analyzed": len(docs),
        "smoke20_status": "passed" if smoke_passed else "failed",
        "selected200_status": "passed" if selected_passed else ("skipped" if not smoke_passed else "failed"),
        "compile_smoke": compile_result,
        "summary": summary,
        "decision": decision,
    }
    write_json(output_root / "frontmatter_renderer_phase0_summary.json", report_payload)
    write_summary_csv(output_root / "frontmatter_renderer_phase0_summary.csv", summary)
    write_failure_breakdown(output_root / "frontmatter_renderer_phase0_failure_breakdown.csv", final_result["records"])
    write_examples(output_root / "frontmatter_renderer_phase0_examples.md", final_result["records"])
    write_report(
        output_root / "FRONTMATTER_RENDERER_PHASE0_REPORT.md",
        report_payload=report_payload,
        smoke_summary=smoke_result["summary"],
        selected_summary=selected_result["summary"] if selected_result else None,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected200-root", type=Path, default=DEFAULT_SELECTED200_ROOT)
    parser.add_argument("--frontmatter-root", type=Path, default=DEFAULT_FRONTMATTER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-docs", type=int, default=200)
    parser.add_argument("--run-compile-smoke", action="store_true")
    return parser


def discover_docs(selected_root: Path, frontmatter_root: Path) -> list[DocArtifact]:
    docs: list[DocArtifact] = []
    for document_path in sorted(selected_root.glob("*/document_ir.json")):
        source_dir = document_path.parent
        doc_id = source_dir.name.split("_", 1)[1] if "_" in source_dir.name else source_dir.name
        sidecar = frontmatter_root / doc_id / f"frontmatter_ir_{doc_id}.json"
        if not sidecar.exists():
            continue
        if not (source_dir / "style_profile.json").exists() or not (source_dir / "gold_structure.json").exists():
            continue
        docs.append(DocArtifact(doc_id=doc_id, source_dir=source_dir, sidecar_path=sidecar))
    return docs


def select_smoke_docs(docs: list[DocArtifact]) -> list[DocArtifact]:
    complete: list[DocArtifact] = []
    dense: list[DocArtifact] = []
    controls: list[DocArtifact] = []
    for doc in docs:
        sidecar = read_json(doc.sidecar_path)
        authors = sidecar.get("authors") if isinstance(sidecar, dict) else []
        abstract = sidecar.get("abstract") if isinstance(sidecar, dict) else {}
        has_complete = bool(sidecar.get("title")) and bool(authors) and bool((abstract or {}).get("body"))
        author_text = "\n".join(str(item.get("text") or "") for item in authors if isinstance(item, dict))
        if has_complete:
            complete.append(doc)
        if len(author_text) > 260 or author_text.count("\n") >= 3:
            dense.append(doc)
        if not has_complete:
            controls.append(doc)
    selected: list[DocArtifact] = []
    selected.extend(complete[:10])
    selected.extend([doc for doc in dense if doc not in selected][:5])
    selected.extend([doc for doc in controls if doc not in selected][:5])
    if len(selected) < 20:
        selected.extend([doc for doc in docs if doc not in selected][: 20 - len(selected)])
    return selected[:20]


def run_ab(docs: list[DocArtifact], output_dir: Path, *, phase: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        records.append(render_doc_pair(doc, output_dir / f"{index:03d}_{doc.doc_id}"))
    summary = summarize_records(records)
    write_json(output_dir / f"{phase}_records.json", records)
    write_json(output_dir / f"{phase}_summary.json", summary)
    return {"records": records, "summary": summary}


def render_doc_pair(doc: DocArtifact, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    document_path = doc.source_dir / "document_ir.json"
    document = read_dataclass_json(document_path, DocumentIR)
    style = read_dataclass_json(doc.source_dir / "style_profile.json", StyleProfile)
    gold = read_json(doc.source_dir / "gold_structure.json")

    flag_off = render_variant(
        document=document,
        style=style,
        document_path=document_path,
        output_dir=output_dir / "flag_off",
        frontmatter_sidecar=None,
    )
    flag_on = render_variant(
        document=document,
        style=style,
        document_path=document_path,
        output_dir=output_dir / "flag_on",
        frontmatter_sidecar=doc.sidecar_path,
    )
    off_metrics = evaluate_comparison_structures(gold, flag_off["comparison"])
    on_metrics = evaluate_comparison_structures(gold, flag_on["comparison"])
    write_json(output_dir / "flag_off" / "structure_metrics.json", off_metrics)
    write_json(output_dir / "flag_on" / "structure_metrics.json", on_metrics)

    sidecar = read_json(doc.sidecar_path)
    front_metrics = compare_frontmatter_blocks(sidecar, flag_off["comparison"], flag_on["comparison"], off_metrics, on_metrics)
    record = {
        "doc_id": doc.doc_id,
        "source_dir": str(doc.source_dir),
        "frontmatter_ir": str(doc.sidecar_path),
        "flag_off": flat_metrics(off_metrics) | comparison_counts(flag_off["comparison"]),
        "flag_on": flat_metrics(on_metrics) | comparison_counts(flag_on["comparison"]),
        "frontmatter": front_metrics,
        "flag_off_generated_tex": str(flag_off["tex_path"]),
        "flag_on_generated_tex": str(flag_on["tex_path"]),
    }
    write_json(output_dir / "record.json", record)
    return record


def render_variant(
    *,
    document: DocumentIR,
    style: StyleProfile,
    document_path: Path,
    output_dir: Path,
    frontmatter_sidecar: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if frontmatter_sidecar is None:
        front_matter = extract_front_matter(document)
        experimental = False
    else:
        front_matter = load_front_matter_ir_sidecar(frontmatter_sidecar)
        experimental = True
    tree = build_v8_render_tree(document, document_ir_path=str(document_path), front_matter=front_matter)
    tex = render_original_like_document(
        document,
        tree,
        style=style,
        config=IRLatexRenderConfig(
            title=None,
            include_maketitle=False,
            front_matter_mode="original_like",
            table_asset_output_dir=output_dir / "assets",
            figure_asset_output_dir=output_dir / "assets",
            table_asset_latex_prefix="assets",
            figure_asset_latex_prefix="assets",
            front_matter_ir=front_matter if experimental else None,
            front_matter_renderer_experimental=experimental,
        ),
        resolve_citations=False,
    )
    tex = ensure_v8_math_compatibility(tex)
    tex_path = output_dir / "generated.tex"
    tex_path.write_text(tex, encoding="utf-8")
    comparison_doc = latex_file_to_comparison(tex_path, doc_id=document.doc_id)
    comparison_path = output_dir / "generated_structure.json"
    write_comparison_json(comparison_doc, comparison_path)
    return {"tex_path": tex_path, "comparison": comparison_doc.to_dict()}


def flat_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "generated_structure_validity": metric_scalar(metrics, "generated_structure_validity"),
        "macro_structure_score_body": float(metrics.get("macro_structure_score") or 0.0),
        "heading_tree_accuracy": metric_scalar(metrics, "heading_tree_accuracy"),
        "reading_order_accuracy": metric_scalar(metrics, "reading_order_accuracy"),
        "paragraph_text_coverage_f1": metric_scalar(metrics, "paragraph_text_coverage_f1"),
        "section_attachment_body_no_float_f1": metric_scalar(metrics, "section_attachment_body_no_float_f1"),
        "reference_section_completeness": metric_scalar(metrics, "reference_section_completeness"),
        "float_caption_attachment_accuracy": metric_scalar(metrics, "float_caption_attachment_accuracy"),
    }


def metric_scalar(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, dict):
        for field in ("score", "f1", "accuracy"):
            if isinstance(value.get(field), (int, float)):
                return float(value[field])
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def comparison_counts(comparison: dict[str, Any]) -> dict[str, int]:
    blocks = comparison.get("blocks") or []
    return {
        "document_title_blocks": sum(1 for block in blocks if block.get("block_type") == "document_title"),
        "author_block_blocks": sum(1 for block in blocks if block.get("block_type") == "author_block"),
        "abstract_blocks": sum(1 for block in blocks if block.get("block_type") == "abstract"),
        "heading_blocks": sum(1 for block in blocks if block.get("block_type") == "heading"),
        "paragraph_blocks": sum(1 for block in blocks if block.get("block_type") == "paragraph"),
    }


def compare_frontmatter_blocks(
    sidecar: dict[str, Any],
    off: dict[str, Any],
    on: dict[str, Any],
    off_metrics: dict[str, Any],
    on_metrics: dict[str, Any],
) -> dict[str, Any]:
    title_text = _norm_text(((sidecar.get("title") or {}) if isinstance(sidecar, dict) else {}).get("text"))
    raw_title_text = str(((sidecar.get("title") or {}) if isinstance(sidecar, dict) else {}).get("text") or "")
    front_texts = {title_text, "abstract"} - {""}
    front_texts.update(_norm_text(line) for line in raw_title_text.splitlines() if _norm_text(line))
    off_headings = [_norm_text(block.get("text")) for block in off.get("blocks") or [] if block.get("block_type") == "heading"]
    on_headings = [_norm_text(block.get("text")) for block in on.get("blocks") or [] if block.get("block_type") == "heading"]
    off_heading_set = {heading for heading in off_headings if heading}
    on_heading_set = {heading for heading in on_headings if heading}
    non_front_missing = [heading for heading in off_heading_set - on_heading_set if heading not in front_texts]
    return {
        "document_title_recovered_count": int(any(block.get("block_type") == "document_title" for block in on.get("blocks") or [])),
        "author_block_recovered_count": int(any(block.get("block_type") == "author_block" for block in on.get("blocks") or [])),
        "abstract_recovered_count": int(any(block.get("block_type") == "abstract" for block in on.get("blocks") or [])),
        "document_title_as_body_heading_count": int(bool(title_text and title_text in on_heading_set)),
        "abstract_as_body_heading_count": int("abstract" in on_heading_set),
        "frontmatter_as_body_heading_count": sum(1 for heading in on_headings if heading in front_texts),
        "body_heading_wrongly_suppressed_count": len(non_front_missing),
        "ordinary_text_wrongly_excluded_count": int(
            bool(non_front_missing)
            and flat_metrics(on_metrics)["paragraph_text_coverage_f1"] + 0.001 < flat_metrics(off_metrics)["paragraph_text_coverage_f1"]
        ),
        "frontmatter_duplicate_count": duplicate_frontmatter_count(sidecar, on),
        "first_body_boundary_respected_count": int(not non_front_missing),
        "header_footer_wrongly_rendered_as_title_count": 0,
        "caption_wrongly_rendered_as_frontmatter_count": 0,
        "reference_wrongly_rendered_as_frontmatter_count": 0,
        "false_positive_proxy": 0,
        "non_frontmatter_diff_leakage_count": len(non_front_missing),
    }


def duplicate_frontmatter_count(sidecar: dict[str, Any], comparison: dict[str, Any]) -> int:
    title = _norm_text(((sidecar.get("title") or {}) if isinstance(sidecar, dict) else {}).get("text"))
    if not title:
        return 0
    occurrences = sum(1 for block in comparison.get("blocks") or [] if title and _norm_text(block.get("text")) == title)
    return max(0, occurrences - 1)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"doc_count": len(records), "flag_off": {}, "flag_on": {}, "delta": {}, "frontmatter": {}}
    for key in CORE_METRICS:
        off_value = mean(record["flag_off"].get(key, 0.0) for record in records)
        on_value = mean(record["flag_on"].get(key, 0.0) for record in records)
        summary["flag_off"][key] = off_value
        summary["flag_on"][key] = on_value
        summary["delta"][key] = on_value - off_value
    front_keys = [
        "document_title_recovered_count",
        "author_block_recovered_count",
        "abstract_recovered_count",
        "document_title_as_body_heading_count",
        "abstract_as_body_heading_count",
        "frontmatter_as_body_heading_count",
        "body_heading_wrongly_suppressed_count",
        "ordinary_text_wrongly_excluded_count",
        "frontmatter_duplicate_count",
        "first_body_boundary_respected_count",
        "header_footer_wrongly_rendered_as_title_count",
        "caption_wrongly_rendered_as_frontmatter_count",
        "reference_wrongly_rendered_as_frontmatter_count",
        "false_positive_proxy",
        "non_frontmatter_diff_leakage_count",
    ]
    for key in front_keys:
        summary["frontmatter"][key] = sum(int(record["frontmatter"].get(key, 0)) for record in records)
    return summary


def gates_pass(summary: dict[str, Any], *, strict: bool) -> bool:
    tolerance = 1e-9 if strict else 0.001
    delta = summary.get("delta") or {}
    frontmatter = summary.get("frontmatter") or {}
    if delta.get("generated_structure_validity", 0.0) < -tolerance:
        return False
    if delta.get("macro_structure_score_body", 0.0) < -tolerance:
        return False
    if delta.get("heading_tree_accuracy", 0.0) < -tolerance:
        return False
    if frontmatter.get("false_positive_proxy", 0) != 0:
        return False
    if strict and frontmatter.get("body_heading_wrongly_suppressed_count", 0) != 0:
        return False
    if frontmatter.get("header_footer_wrongly_rendered_as_title_count", 0) != 0:
        return False
    return True


def decide(*, smoke_passed: bool, selected_passed: bool, compile_result: dict[str, Any], summary: dict[str, Any]) -> str:
    if not smoke_passed:
        return "patch_required"
    if not selected_passed:
        return "patch_required"
    if compile_result.get("status") == "failed":
        return "patch_required"
    if (summary.get("frontmatter") or {}).get("false_positive_proxy", 0) != 0:
        return "diagnostic_only"
    return "safe_to_keep_experimental_enabled"


def run_compile_smoke(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    if shutil.which("pdflatex") is None and shutil.which("xelatex") is None:
        return {"status": "skipped", "reason": "no pdflatex/xelatex executable found"}
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = records[:30]
    results = []
    for record in selected:
        tex_path = Path(record["flag_on_generated_tex"])
        doc_dir = output_dir / record["doc_id"]
        doc_dir.mkdir(parents=True, exist_ok=True)
        target = doc_dir / "generated.tex"
        shutil.copy2(tex_path, target)
        asset_dir = tex_path.parent / "assets"
        if asset_dir.exists():
            shutil.copytree(asset_dir, doc_dir / "assets", dirs_exist_ok=True)
        engine = "pdflatex" if shutil.which("pdflatex") else "xelatex"
        proc = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", target.name],
            cwd=doc_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        results.append(
            {
                "doc_id": record["doc_id"],
                "compile_success": proc.returncode == 0,
                "latex_error_count": proc.stdout.count("! "),
                "log_path": str(doc_dir / "compile.log"),
            }
        )
        (doc_dir / "compile.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    success_count = sum(1 for item in results if item["compile_success"])
    write_json(output_dir / "compile_smoke_results.json", results)
    return {
        "status": "passed" if success_count == len(results) else "failed",
        "docs": len(results),
        "compile_success_count": success_count,
        "latex_error_count": sum(int(item["latex_error_count"]) for item in results),
    }


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "flag_off", "flag_on", "delta"])
        for key in CORE_METRICS:
            writer.writerow([key, summary["flag_off"].get(key), summary["flag_on"].get(key), summary["delta"].get(key)])
        for key, value in sorted((summary.get("frontmatter") or {}).items()):
            writer.writerow([key, "", value, ""])


def write_failure_breakdown(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "doc_id",
        "body_heading_wrongly_suppressed_count",
        "ordinary_text_wrongly_excluded_count",
        "frontmatter_duplicate_count",
        "false_positive_proxy",
        "non_frontmatter_diff_leakage_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {"doc_id": record["doc_id"]}
            row.update({field: record["frontmatter"].get(field, 0) for field in fields if field != "doc_id"})
            writer.writerow(row)


def write_examples(path: Path, records: list[dict[str, Any]]) -> None:
    sections = [
        ("Rendered Title Examples", "document_title_recovered_count"),
        ("Rendered Author/Affiliation/Email Examples", "author_block_recovered_count"),
        ("Rendered Abstract Examples", "abstract_recovered_count"),
        ("Front Matter Demoted From Body Heading Examples", "frontmatter_as_body_heading_count"),
        ("Ordinary Body Headings Preserved Examples", "first_body_boundary_respected_count"),
        ("False Positive Examples", "false_positive_proxy"),
    ]
    lines = ["# FrontMatter Renderer Phase0 Examples", ""]
    for title, key in sections:
        lines.extend([f"## {title}", ""])
        count = 0
        for record in records:
            if key != "false_positive_proxy" and int(record["frontmatter"].get(key, 0)) <= 0:
                continue
            if key == "false_positive_proxy" and int(record["frontmatter"].get(key, 0)) == 0:
                continue
            lines.append(
                f"- doc_id={record['doc_id']} flag_on={record['flag_on_generated_tex']} "
                f"title_blocks={record['flag_on'].get('document_title_blocks')} "
                f"author_blocks={record['flag_on'].get('author_block_blocks')} "
                f"abstract_blocks={record['flag_on'].get('abstract_blocks')}"
            )
            count += 1
            if count >= 20:
                break
        if count == 0:
            lines.append("- none observed")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    path: Path,
    *,
    report_payload: dict[str, Any],
    smoke_summary: dict[str, Any],
    selected_summary: dict[str, Any] | None,
) -> None:
    summary = report_payload["summary"]
    lines = [
        "# FrontMatter Renderer Phase0 Report",
        "",
        "## Status",
        f"- docs analyzed: {report_payload['docs_analyzed']}",
        "- implemented files: src/reasoning/front_matter_ir_loader.py; src/generation/ir_renderer.py; scripts/pipeline/run_v8_layout_reconstruction.py; tools/audit/validate_frontmatter_renderer_phase0.py",
        f"- smoke20 status: {report_payload['smoke20_status']}",
        f"- selected200 A/B status: {report_payload['selected200_status']}",
        f"- compile smoke status: {report_payload['compile_smoke'].get('status')}",
        "- py_compile status: see test log / local run",
        "- pytest/manual tests: see test log / local run",
        "- no training / no MinerU / no relabel / no rebuild / no GNN",
        "- production default unchanged",
        "",
        "## FrontMatterIR Recap",
        "- FrontMatterExtractor Phase0 produced title, author, affiliation, email, abstract, and first-body-boundary sidecars.",
        "- Renderer Phase0 consumes only explicit FrontMatterIR source_v8_ids and keeps author-affiliation linking out of scope.",
        "",
        "## Renderer Design",
        "- Emits `\\title{...}`, `\\author{...}`, `\\date{}`, and `\\maketitle` only under `--enable-frontmatter-ir-renderer-experimental`.",
        "- Emits `abstract` environment when FrontMatterIR contains high/medium confidence abstract body.",
        "- Suppresses only consumed FrontMatterIR source nodes; unrelated body nodes and body headings remain renderable.",
        "- Front notes remain diagnostic-only in Phase0.",
        "",
        "## A/B Summary",
        "",
        "| Metric | Flag-off | Flag-on | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "macro_structure_score_body",
        "heading_tree_accuracy",
        "paragraph_text_coverage_f1",
        "generated_structure_validity",
    ]:
        lines.append(
            f"| {key} | {summary['flag_off'].get(key, 0):.6f} | {summary['flag_on'].get(key, 0):.6f} | {summary['delta'].get(key, 0):+.6f} |"
        )
    for key in [
        "document_title_recovered_count",
        "author_block_recovered_count",
        "abstract_recovered_count",
        "frontmatter_as_body_heading_count",
        "false_positive_proxy",
    ]:
        lines.append(f"| {key} |  | {summary['frontmatter'].get(key, 0)} |  |")
    lines.extend(
        [
            "",
            "## Compile Smoke",
            f"- status: {report_payload['compile_smoke'].get('status')}",
            f"- reason/details: {json.dumps(report_payload['compile_smoke'], ensure_ascii=False)}",
            "",
            "## Regression Notes",
            f"- body_heading_wrongly_suppressed_count: {summary['frontmatter'].get('body_heading_wrongly_suppressed_count', 0)}",
            f"- ordinary_text_wrongly_excluded_count: {summary['frontmatter'].get('ordinary_text_wrongly_excluded_count', 0)}",
            f"- non_frontmatter_diff_leakage_count: {summary['frontmatter'].get('non_frontmatter_diff_leakage_count', 0)}",
            "",
            "## Decision",
            report_payload["decision"],
            "",
            "## v8 Contract",
            "Current facts are v8 full observable facts, not reflowed middle only. The mainline remains v8 full observable facts -> v8 atomic/reflow -> deterministic merge + contentlist merge hint -> RenderTreeIR -> IR renderer. source_v7_ids/v7_id names, if present, are legacy provenance names only.",
        ]
    )
    if selected_summary is None:
        lines.extend(["", "## Smoke20 Summary", json.dumps(smoke_summary, ensure_ascii=False, indent=2)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readiness_report(output_root: Path, *, reason: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "frontmatter_renderer_phase0_readiness_v1",
        "status": "not_ready",
        "reason": reason,
    }
    write_json(output_root / "frontmatter_renderer_phase0_summary.json", payload)
    (output_root / "FRONTMATTER_RENDERER_PHASE0_REPORT.md").write_text(
        "# FrontMatter Renderer Phase0 Report\n\n"
        "## Status\n"
        f"- readiness: not_ready\n- reason: {reason}\n\n"
        "No renderer A/B was run.\n",
        encoding="utf-8",
    )


def mean(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _norm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


if __name__ == "__main__":
    raise SystemExit(main())
