#!/usr/bin/env python3
"""Run direct parser baselines on a held-out manifest."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comparison_structure import write_comparison_json  # noqa: E402
from src.evaluation.structure_metrics import evaluate_comparison_structures, load_comparison_json  # noqa: E402
from tools.baselines.convert_contentlist_direct_to_comparison import contentlist_to_comparison  # noqa: E402
from tools.baselines.convert_mineru_direct_to_comparison import mineru_middle_to_comparison  # noqa: E402


@dataclass
class CaseResult:
    doc_id: str
    conversion_success: bool
    metrics_success: bool
    failure_type: str
    message: str
    pred_path: str | None = None
    metrics_path: str | None = None


class TimeoutErrorForMetrics(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def with_timeout(seconds: int, func: Callable[[], Any]) -> Any:
    if seconds <= 0:
        return func()

    def handler(_signum: int, _frame: Any) -> None:
        raise TimeoutErrorForMetrics(f"metrics timed out after {seconds}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        for key in ("items", "docs", "documents", "manifest"):
            if isinstance(payload.get(key), list):
                return [dict(item) for item in payload[key] if isinstance(item, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            return [dict(value) for value in payload.values()]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    raise ValueError(f"Unsupported manifest payload: {type(payload).__name__}")


def field_path(row: dict[str, Any], *names: str) -> Path | None:
    for name in names:
        value = row.get(name)
        if value:
            return Path(str(value))
    return None


def convert_case(row: dict[str, Any], baseline: str, output_root: Path, metrics_timeout: int) -> CaseResult:
    doc_id = str(row.get("doc_id") or row.get("paper_id") or "unknown")
    case_dir = output_root / doc_id
    case_dir.mkdir(parents=True, exist_ok=True)
    pred_name = f"{baseline}_direct_comparison_structure.json"
    metrics_name = f"{baseline}_direct_metrics.json"
    pred_path = case_dir / pred_name
    metrics_path = case_dir / metrics_name
    report_path = case_dir / "conversion_report.json"
    taxonomy_path = case_dir / "failure_taxonomy.json"
    summary_path = case_dir / "case_summary.json"

    try:
        if baseline == "contentlist":
            input_path = field_path(row, "content_list_v2_json", "content_list_json")
            if input_path is None:
                raise FileNotFoundError("content_list_v2_json/content_list_json missing")
            document = contentlist_to_comparison(input_path, doc_id=doc_id)
        elif baseline == "mineru":
            middle_path = field_path(row, "middle_json")
            input_path = middle_path or field_path(row, "content_list_v2_json", "content_list_json")
            if input_path is None:
                raise FileNotFoundError("middle_json/content_list path missing")
            document = mineru_middle_to_comparison(input_path, doc_id=doc_id) if middle_path else contentlist_to_comparison(input_path, doc_id=doc_id)
        else:
            raise ValueError(f"Unknown baseline: {baseline}")
        write_comparison_json(document, pred_path)
        write_json(
            report_path,
            {
                "doc_id": doc_id,
                "baseline": baseline,
                "input_path": str(input_path),
                "prediction_path": str(pred_path),
                "block_count": len(document.blocks),
                "source_format": document.source_format,
                "conversion_success": True,
            },
        )
    except Exception as exc:  # noqa: BLE001 - report conversion failure, keep batch running.
        message = f"{type(exc).__name__}: {exc}"
        write_json(taxonomy_path, {"doc_id": doc_id, "baseline": baseline, "failure_type": "conversion_failure", "message": message})
        write_json(summary_path, {"doc_id": doc_id, "conversion_success": False, "metrics_success": False, "failure_type": "conversion_failure", "message": message})
        return CaseResult(doc_id, False, False, "conversion_failure", message)

    gold_path = field_path(row, "gold_comparison_path", "gold_comparison", "existing_gold_comparison")
    if gold_path is None:
        message = "gold comparison path missing"
        write_json(taxonomy_path, {"doc_id": doc_id, "baseline": baseline, "failure_type": "gold_missing", "message": message})
        write_json(summary_path, {"doc_id": doc_id, "conversion_success": True, "metrics_success": False, "failure_type": "gold_missing", "message": message})
        return CaseResult(doc_id, True, False, "gold_missing", message, str(pred_path), None)
    try:
        def evaluate() -> dict[str, Any]:
            return evaluate_comparison_structures(load_comparison_json(gold_path), load_comparison_json(pred_path))

        metrics = with_timeout(metrics_timeout, evaluate)
        write_json(metrics_path, metrics)
        write_json(taxonomy_path, {"doc_id": doc_id, "baseline": baseline, "failure_type": None, "message": ""})
        write_json(summary_path, {"doc_id": doc_id, "conversion_success": True, "metrics_success": True, "failure_type": None, "message": ""})
        return CaseResult(doc_id, True, True, "", "", str(pred_path), str(metrics_path))
    except TimeoutErrorForMetrics as exc:
        message = str(exc)
        write_json(taxonomy_path, {"doc_id": doc_id, "baseline": baseline, "failure_type": "metrics_timeout", "message": message})
        write_json(summary_path, {"doc_id": doc_id, "conversion_success": True, "metrics_success": False, "failure_type": "metrics_timeout", "message": message})
        return CaseResult(doc_id, True, False, "metrics_timeout", message, str(pred_path), None)
    except Exception as exc:  # noqa: BLE001 - report metric failure, keep batch running.
        message = f"{type(exc).__name__}: {exc}"
        write_json(
            taxonomy_path,
            {
                "doc_id": doc_id,
                "baseline": baseline,
                "failure_type": "metrics_failure",
                "message": message,
                "traceback": traceback.format_exc(limit=6),
            },
        )
        write_json(summary_path, {"doc_id": doc_id, "conversion_success": True, "metrics_success": False, "failure_type": "metrics_failure", "message": message})
        return CaseResult(doc_id, True, False, "metrics_failure", message, str(pred_path), None)


def summarize_metrics(results: list[CaseResult]) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {
        "macro_structure_score_body": [],
        "heading_tree_accuracy": [],
        "reading_order_accuracy": [],
        "paragraph_text_coverage_f1": [],
        "paragraph_boundary_f1": [],
        "section_attachment_body_no_float_f1": [],
        "reference_section_completeness": [],
        "float_caption_attachment_accuracy": [],
        "generated_structure_validity": [],
    }
    for result in results:
        if not result.metrics_success or not result.metrics_path:
            continue
        metrics = read_json(Path(result.metrics_path))
        macro = metrics.get("macro_structure_score")
        if isinstance(macro, dict):
            macro_value = macro.get("body_no_float", macro.get("score", 0.0))
        else:
            macro_value = macro
        metric_values["macro_structure_score_body"].append(float(macro_value or 0.0))
        metric_values["heading_tree_accuracy"].append(float((metrics.get("heading_tree_accuracy") or {}).get("score", 0.0) or 0.0))
        metric_values["reading_order_accuracy"].append(float((metrics.get("reading_order_accuracy") or {}).get("score", 0.0) or 0.0))
        metric_values["paragraph_text_coverage_f1"].append(float((metrics.get("paragraph_text_coverage_f1") or {}).get("f1", 0.0) or 0.0))
        metric_values["paragraph_boundary_f1"].append(float((metrics.get("paragraph_boundary_f1") or {}).get("f1", 0.0) or 0.0))
        metric_values["section_attachment_body_no_float_f1"].append(float((metrics.get("section_attachment_body_no_float_f1") or {}).get("f1", 0.0) or 0.0))
        metric_values["reference_section_completeness"].append(float((metrics.get("reference_section_completeness") or {}).get("score", 0.0) or 0.0))
        metric_values["float_caption_attachment_accuracy"].append(float((metrics.get("float_caption_attachment_accuracy") or {}).get("score", 0.0) or 0.0))
        metric_values["generated_structure_validity"].append(float((metrics.get("generated_structure_validity") or {}).get("score", 0.0) or 0.0))

    summary = {
        "doc_count": len(results),
        "conversion_success_count": sum(result.conversion_success for result in results),
        "metrics_success_count": sum(result.metrics_success for result in results),
        "failure_types": dict(Counter(result.failure_type or "none" for result in results if result.failure_type)),
    }
    for key, values in metric_values.items():
        summary[key] = round(sum(values) / len(values), 6) if values else None
    return summary


def write_rollups(output_root: Path, baseline: str, results: list[CaseResult]) -> None:
    summary = summarize_metrics(results)
    summary["baseline"] = baseline
    write_json(output_root / "batch_rollup.json", summary)
    with (output_root / "batch_rollup.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["doc_id", "conversion_success", "metrics_success", "failure_type", "message", "pred_path", "metrics_path"])
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)
    lines = [
        f"# {baseline.title()} Direct Batch Rollup",
        "",
        f"- doc_count: {summary['doc_count']}",
        f"- conversion_success: {summary['conversion_success_count']}/{summary['doc_count']}",
        f"- structure_metrics: {summary['metrics_success_count']}/{summary['doc_count']}",
        f"- failure_types: {summary['failure_types']}",
    ]
    for key in [
        "macro_structure_score_body",
        "heading_tree_accuracy",
        "reading_order_accuracy",
        "paragraph_text_coverage_f1",
        "paragraph_boundary_f1",
        "section_attachment_body_no_float_f1",
        "reference_section_completeness",
        "float_caption_attachment_accuracy",
        "generated_structure_validity",
    ]:
        lines.append(f"- {key}: {summary.get(key)}")
    (output_root / "batch_rollup.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", choices=["contentlist", "mineru"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="Accepted for pipeline parity; this runner is intentionally single-process per shard.")
    parser.add_argument("--metrics-timeout", type=int, default=60)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows(args.manifest)
    results = [convert_case(row, args.baseline, args.output_root, args.metrics_timeout) for row in rows]
    write_rollups(args.output_root, args.baseline, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
