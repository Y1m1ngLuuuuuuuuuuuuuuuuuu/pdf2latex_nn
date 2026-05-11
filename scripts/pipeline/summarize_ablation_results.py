#!/usr/bin/env python3
"""Summarize repeated ablation runs from training and threshold-calibration reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Ablation output root, e.g. data/09_eval_reports/ablations_v2")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = collect_rows(args.root)
    summary = summarize_rows(rows)
    payload = {
        "schema_version": "ablation_summary_v1",
        "root": str(args.root),
        "num_runs": len(rows),
        "runs": rows,
        "summary": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows, summary)
    print(f"runs={len(rows)} experiments={len(summary)}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    return 0


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for train_report in sorted(root.glob("*/seed_*/training_report.json")):
        run_dir = train_report.parent
        experiment = run_dir.parent.name
        seed = seed_from_name(run_dir.name)
        report = load_json(train_report)
        calibration = load_json(run_dir / "threshold_calibration.json") if (run_dir / "threshold_calibration.json").exists() else {}
        final = dict(report.get("final", {}))
        calibrated_test = dict(calibration.get("calibrated", {}).get("test", {}))
        argmax_test = dict(calibration.get("argmax", {}).get("test", {}))
        best_thresholds = dict(calibration.get("best_thresholds", {}))
        row = {
            "experiment": experiment,
            "seed": seed,
            "run_dir": str(run_dir),
            "best_epoch": report.get("best_epoch"),
            "best_metric": report.get("best_metric"),
            "final_val_positive_macro_f1": final.get("val_positive_macro_f1"),
            "final_test_positive_macro_f1": final.get("test_positive_macro_f1"),
            "final_test_macro_f1": final.get("test_macro_f1"),
            "final_test_merge_f1": per_class_value(final, "test_per_class", 0, "f1"),
            "final_test_merge_precision": per_class_value(final, "test_per_class", 0, "precision"),
            "final_test_merge_recall": per_class_value(final, "test_per_class", 0, "recall"),
            "final_test_parent_f1": per_class_value(final, "test_per_class", 1, "f1"),
            "final_test_parent_precision": per_class_value(final, "test_per_class", 1, "precision"),
            "final_test_parent_recall": per_class_value(final, "test_per_class", 1, "recall"),
            "calibrated_test_positive_macro_f1": calibrated_test.get("positive_macro_f1"),
            "calibrated_test_macro_f1": calibrated_test.get("macro_f1"),
            "calibrated_test_merge_f1": per_class_value(calibrated_test, "per_class", 0, "f1"),
            "calibrated_test_merge_precision": per_class_value(calibrated_test, "per_class", 0, "precision"),
            "calibrated_test_merge_recall": per_class_value(calibrated_test, "per_class", 0, "recall"),
            "calibrated_test_parent_f1": per_class_value(calibrated_test, "per_class", 1, "f1"),
            "calibrated_test_parent_precision": per_class_value(calibrated_test, "per_class", 1, "precision"),
            "calibrated_test_parent_recall": per_class_value(calibrated_test, "per_class", 1, "recall"),
            "argmax_test_positive_macro_f1": argmax_test.get("positive_macro_f1"),
            "tau_merge": best_thresholds.get("tau_merge"),
            "tau_parent": best_thresholds.get("tau_parent"),
        }
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["experiment"])].append(row)
    metrics = [
        "calibrated_test_positive_macro_f1",
        "calibrated_test_merge_f1",
        "calibrated_test_merge_precision",
        "calibrated_test_merge_recall",
        "calibrated_test_parent_f1",
        "calibrated_test_parent_precision",
        "calibrated_test_parent_recall",
        "final_test_positive_macro_f1",
        "final_test_macro_f1",
        "tau_merge",
        "tau_parent",
    ]
    summary: list[dict[str, Any]] = []
    for experiment, exp_rows in sorted(groups.items()):
        item: dict[str, Any] = {"experiment": experiment, "num_seeds": len(exp_rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in exp_rows if row.get(metric) is not None]
            if values:
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    summary.sort(key=lambda row: float(row.get("calibrated_test_positive_macro_f1_mean", -1.0)), reverse=True)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    run_columns = sorted({key for row in rows for key in row})
    summary_columns = sorted({key for row in summary for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section"] + summary_columns)
        writer.writeheader()
        for row in summary:
            writer.writerow({"section": "summary", **row})
        handle.write("\n")
        writer = csv.DictWriter(handle, fieldnames=["section"] + run_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({"section": "run", **row})


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_from_name(value: str) -> int | None:
    if value.startswith("seed_"):
        try:
            return int(value.split("_", 1)[1])
        except ValueError:
            return None
    return None


def per_class_value(payload: dict[str, Any], key: str, class_idx: int, metric: str) -> Any:
    table = payload.get(key, {})
    if not isinstance(table, dict):
        return None
    row = table.get(class_idx, table.get(str(class_idx), {}))
    if not isinstance(row, dict):
        return None
    return row.get(metric)


if __name__ == "__main__":
    raise SystemExit(main())
