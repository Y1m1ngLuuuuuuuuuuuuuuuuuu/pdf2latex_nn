#!/usr/bin/env python3
"""Collect current-model evaluation outputs into paper-ready tables.

The script is intentionally read-only.  It does not train, relabel, or mutate
model outputs.  It collects:

* GNN ablation summary
* current E2E generator comparison
* Nougat paired comparison

and writes a compact JSON/CSV/Markdown rollup.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ABLATION = Path("data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current_summary.json")
DEFAULT_E2E = Path(
    "data/09_eval_reports/current_e2e_comparison_hard20_floatcaption_rerun_20260518_132615/"
    "e2e_comparison_manifest.json"
)
DEFAULT_NOUGAT = Path(
    "data/09_eval_reports/nougat_current_paired_hard20_floatcaption_rerun_20260518_132615/"
    "nougat_comparison_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics")


CORE_STRUCTURE_METRICS = [
    "macro_structure_score",
    "heading_tree_accuracy",
    "reading_order_accuracy",
    "paragraph_boundary_f1",
    "paragraph_text_coverage_f1",
    "section_attachment_body_no_float_f1",
    "reference_section_completeness",
    "float_caption_attachment_accuracy",
    "generated_structure_validity",
]

DEPRECATED_METRIC_ALIASES = {"paragraph_merge_f1"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-summary", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--e2e-manifest", type=Path, default=DEFAULT_E2E)
    parser.add_argument("--nougat-manifest", type=Path, default=DEFAULT_NOUGAT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strict", action="store_true", help="Fail if any expected input file is missing.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ablation_payload = load_optional_json(args.ablation_summary, strict=args.strict)
    e2e_payload = load_optional_json(args.e2e_manifest, strict=args.strict)
    nougat_payload = load_optional_json(args.nougat_manifest, strict=args.strict)

    ablation_rows = normalize_ablation_rows(ablation_payload)
    e2e_rows = normalize_document_rows(e2e_payload)
    nougat_rows = normalize_document_rows(nougat_payload)

    overview = {
        "schema_version": "current_eval_rollup_v1",
        "inputs": {
            "ablation_summary": str(args.ablation_summary),
            "e2e_manifest": str(args.e2e_manifest),
            "nougat_manifest": str(args.nougat_manifest),
        },
        "status": {
            "ablation_available": bool(ablation_payload),
            "e2e_available": bool(e2e_payload),
            "nougat_available": bool(nougat_payload),
        },
        "ablation": summarize_ablation(ablation_payload, ablation_rows),
        "e2e": summarize_manifest(e2e_payload, prefix=""),
        "nougat_paired": summarize_manifest(nougat_payload, prefix=""),
        "ours_vs_nougat": summarize_ours_vs_nougat(nougat_payload),
    }

    write_json(args.output_dir / "current_eval_rollup.json", overview)
    write_csv(args.output_dir / "ablation_summary.csv", ablation_rows)
    write_csv(args.output_dir / "e2e_documents.csv", e2e_rows)
    write_csv(args.output_dir / "nougat_paired_documents.csv", nougat_rows)
    (args.output_dir / "current_eval_rollup.md").write_text(render_markdown(overview), encoding="utf-8")

    print(f"wrote {args.output_dir / 'current_eval_rollup.json'}")
    print(f"wrote {args.output_dir / 'current_eval_rollup.md'}")
    return 0


def load_optional_json(path: Path, *, strict: bool) -> dict[str, Any] | None:
    if not path.exists():
        if strict:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_ablation_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    summary = payload.get("summary")
    if isinstance(summary, list):
        return [row for row in summary if isinstance(row, dict)]
    rows = payload.get("runs")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def normalize_document_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    docs = payload.get("documents", [])
    return [drop_deprecated_metric_aliases(row) for row in docs if isinstance(row, dict)]


def summarize_ablation(payload: dict[str, Any] | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    best = None
    for row in rows:
        value = first_number(
            row.get("calibrated_test_positive_macro_f1_mean"),
            row.get("calibrated_test_positive_macro_f1"),
            row.get("final_test_positive_macro_f1_mean"),
            row.get("final_test_positive_macro_f1"),
        )
        if value is None:
            continue
        if best is None or value > best["score"]:
            best = {"experiment": row.get("experiment"), "score": value, "row": row}
    return {
        "available": True,
        "num_runs": payload.get("num_runs", len(payload.get("runs", []))),
        "num_summary_rows": len(rows),
        "best_by_calibrated_positive_macro_f1": best,
    }


def summarize_manifest(payload: dict[str, Any] | None, *, prefix: str) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    summary = drop_deprecated_metric_aliases(dict(payload.get("summary", {})))
    docs = normalize_document_rows(payload)
    errors = [row for row in docs if row.get("error")]
    return {
        "available": True,
        "summary": summary,
        "documents": len(docs),
        "errors": len(errors),
        "error_examples": [
            {"document_id": row.get("document_id"), "error": row.get("error"), "doc_dir": row.get("doc_dir")}
            for row in errors[:10]
        ],
    }


def summarize_ours_vs_nougat(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    docs = normalize_document_rows(payload)
    paired = [row for row in docs if row.get("ours_available")]
    summary: dict[str, Any] = {
        "available": bool(paired),
        "paired_documents": len(paired),
        "documents": len(docs),
    }
    if not paired:
        return summary
    for metric in CORE_STRUCTURE_METRICS:
        ours_key = f"ours_{metric}"
        delta_key = f"delta_ours_minus_nougat_{metric}"
        summary[f"nougat_{metric}"] = mean_number(row.get(metric) for row in paired)
        summary[ours_key] = mean_number(row.get(ours_key) for row in paired)
        summary[delta_key] = mean_number(row.get(delta_key) for row in paired)
    return summary


def is_deprecated_metric_alias(key: str) -> bool:
    if key in DEPRECATED_METRIC_ALIASES:
        return True
    return any(key.endswith(f"_{alias}") for alias in DEPRECATED_METRIC_ALIASES)


def drop_deprecated_metric_aliases(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not is_deprecated_metric_alias(str(key))}


def render_markdown(overview: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Current Evaluation Rollup")
    lines.append("")
    lines.append("## Inputs")
    for key, value in overview["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## GNN Ablation")
    ablation = overview["ablation"]
    if not ablation.get("available"):
        lines.append("- Not available yet.")
    else:
        best = ablation.get("best_by_calibrated_positive_macro_f1")
        lines.append(f"- Runs: `{ablation.get('num_runs')}`")
        lines.append(f"- Summary rows: `{ablation.get('num_summary_rows')}`")
        if best:
            lines.append(f"- Best calibrated positive macro F1: `{best['experiment']}` = `{fmt(best['score'])}`")
            best_row = best.get("row") or {}
            for metric in [
                "calibrated_test_merge_f1_mean",
                "calibrated_test_parent_f1_mean",
                "calibrated_test_merge_precision_mean",
                "calibrated_test_merge_recall_mean",
                "tau_merge_mean",
                "tau_parent_mean",
            ]:
                if metric in best_row:
                    lines.append(f"  - `{metric}`: `{fmt(best_row.get(metric))}`")
    lines.append("")
    lines.append("## Current E2E Generator")
    append_manifest_section(lines, overview.get("e2e") or {}, label="E2E")
    lines.append("")
    lines.append("## Nougat Paired Comparison")
    append_manifest_section(lines, overview.get("nougat_paired") or {}, label="Nougat")
    paired = overview.get("ours_vs_nougat") or {}
    if paired.get("available"):
        lines.append("")
        lines.append("### Ours vs Nougat")
        lines.append(f"- Paired documents: `{paired.get('paired_documents')}` / `{paired.get('documents')}`")
        lines.append("")
        lines.append("| Metric | Ours | Nougat | Delta |")
        lines.append("|---|---:|---:|---:|")
        for metric in CORE_STRUCTURE_METRICS:
            lines.append(
                "| "
                + metric
                + " | "
                + fmt(paired.get(f"ours_{metric}"))
                + " | "
                + fmt(paired.get(f"nougat_{metric}"))
                + " | "
                + fmt(paired.get(f"delta_ours_minus_nougat_{metric}"))
                + " |"
            )
    lines.append("")
    lines.append("## Analysis Notes")
    lines.extend(render_analysis_notes(overview))
    lines.append("")
    return "\n".join(lines)


def append_manifest_section(lines: list[str], section: dict[str, Any], *, label: str) -> None:
    if not section.get("available"):
        lines.append("- Not available yet.")
        return
    summary = section.get("summary") or {}
    lines.append(f"- Documents: `{section.get('documents')}`")
    lines.append(f"- Errors: `{section.get('errors')}`")
    for metric in [
        "compile_success_rate",
        "macro_structure_score",
        "heading_tree_accuracy",
        "reading_order_accuracy",
        "paragraph_text_coverage_f1",
        "paragraph_boundary_f1",
        "section_attachment_body_no_float_f1",
        "reference_section_completeness",
        "float_caption_attachment_accuracy",
        "generated_structure_validity",
        "layout_similarity",
    ]:
        if metric in summary:
            lines.append(f"- `{metric}`: `{fmt(summary.get(metric))}`")
    if section.get("error_examples"):
        lines.append(f"- {label} error examples:")
        for item in section["error_examples"][:5]:
            lines.append(f"  - `{item.get('document_id')}`: `{item.get('error')}`")


def render_analysis_notes(overview: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    paired = overview.get("ours_vs_nougat") or {}
    e2e = (overview.get("e2e") or {}).get("summary") or {}
    ablation = overview.get("ablation") or {}
    if not ablation.get("available"):
        notes.append("- Ablation has not completed yet; wait for the current full-eval job or run the ablation summary step.")
    if not e2e:
        notes.append("- E2E generator results are not available yet; generator-level claims should wait for compiled outputs.")
    else:
        compile_rate = first_number(e2e.get("compile_success_rate"))
        if compile_rate is not None and compile_rate < 0.95:
            notes.append("- Compile success is below 95%; inspect LaTeX syntax, math rendering, and float crop paths before using visual metrics.")
        layout = first_number(e2e.get("layout_similarity"))
        if layout is not None and layout < 0.80:
            notes.append("- Layout similarity is below 0.80; focus generator work on column detection, float sizing, and page-style profiling.")
    if paired.get("available"):
        delta = first_number(paired.get("delta_ours_minus_nougat_macro_structure_score"))
        if delta is not None:
            if delta >= 0:
                notes.append("- Current paired comparison shows Ours is ahead of Nougat on macro structure for this sample.")
            else:
                notes.append("- Current paired comparison shows Nougat ahead on macro structure; inspect heading/caption/reference submetrics before drawing conclusions.")
    else:
        notes.append("- Nougat paired comparison is not available yet; run with `--ours-e2e-manifest` to get direct deltas.")
    return notes or ["- No blocking warnings detected in the available summaries."]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def mean_number(values: Iterable[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def fmt(value: Any) -> str:
    number = first_number(value)
    if number is None:
        return "NA"
    return f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
