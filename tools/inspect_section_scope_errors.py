#!/usr/bin/env python3
"""Inspect section-scope errors for an E2E comparison output directory.

The regular metrics intentionally compress many failure modes into a few F1
numbers.  This diagnostic expands section attachment into human-readable rows:

- gold heading and predicted heading for each matched body block
- active heading by predicted reading order
- missing gold headings
- missing body blocks
- float/caption cases that contaminate all-content section scores
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.structure_metrics import (  # noqa: E402
    BODY_SECTION_ATTACHMENT_TYPES,
    FLOAT_SECTION_ATTACHMENT_TYPES,
    StructureMetricsEvaluator,
    block_id,
    block_text,
    block_type,
    int_or_none,
    load_comparison_json,
    nearest_heading_ancestor,
    normalize_for_eval,
    section_scope_kind,
)


BODY_TYPES = BODY_SECTION_ATTACHMENT_TYPES
FLOAT_TYPES = FLOAT_SECTION_ATTACHMENT_TYPES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-dir", type=Path, required=True, help="Directory containing per-document E2E folders.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-threshold", type=float, default=0.58)
    parser.add_argument("--max-text", type=int, default=220)
    parser.add_argument("--include-correct", action="store_true", help="Also write correct body-scope rows.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    doc_dirs = sorted(
        path for path in args.e2e_dir.iterdir()
        if path.is_dir() and (path / "gold_structure.json").exists() and (path / "generated_structure.json").exists()
    )
    if not doc_dirs:
        raise FileNotFoundError(f"No E2E document folders found under {args.e2e_dir}")

    summary_rows: list[dict[str, Any]] = []
    heading_rows: list[dict[str, Any]] = []
    body_rows: list[dict[str, Any]] = []
    float_rows: list[dict[str, Any]] = []

    for doc_dir in doc_dirs:
        payload = inspect_document(doc_dir, args)
        summary_rows.append(payload["summary"])
        heading_rows.extend(payload["heading_rows"])
        body_rows.extend(payload["body_rows"])
        float_rows.extend(payload["float_rows"])

    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "heading_errors.csv", heading_rows)
    write_csv(args.output_dir / "body_scope_rows.csv", body_rows)
    write_csv(args.output_dir / "float_caption_scope_rows.csv", float_rows)
    write_json(
        args.output_dir / "section_scope_inspection_summary.json",
        {
            "schema_version": "section_scope_inspection_v1",
            "e2e_dir": str(args.e2e_dir),
            "documents": len(summary_rows),
            "aggregate": aggregate_summary(summary_rows),
            "summary_csv": str(args.output_dir / "summary.csv"),
            "heading_errors_csv": str(args.output_dir / "heading_errors.csv"),
            "body_scope_rows_csv": str(args.output_dir / "body_scope_rows.csv"),
            "float_caption_scope_rows_csv": str(args.output_dir / "float_caption_scope_rows.csv"),
        },
    )
    print(json.dumps(aggregate_summary(summary_rows), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def inspect_document(doc_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gold = load_comparison_json(doc_dir / "gold_structure.json")
    pred = load_comparison_json(doc_dir / "generated_structure.json")
    metrics = StructureMetricsEvaluator(gold, pred, match_threshold=args.match_threshold).evaluate()
    evaluator = StructureMetricsEvaluator(gold, pred, match_threshold=args.match_threshold)

    doc_id = str(gold.get("doc_id") or doc_dir.name)
    pred_active_by_id = active_heading_by_predicted_order(evaluator)
    oracle_active_gold_by_id = oracle_gold_heading_by_predicted_order(evaluator)

    heading_rows = inspect_headings(doc_id, evaluator, args.max_text)
    all_body_rows = inspect_body_rows(
        doc_id,
        evaluator,
        pred_active_by_id,
        oracle_active_gold_by_id,
        args.max_text,
        include_correct=True,
    )
    body_rows = all_body_rows if args.include_correct else [
        row for row in all_body_rows if row["is_correct_section"] == "false"
    ]
    float_rows = inspect_float_rows(doc_id, evaluator, pred_active_by_id, args.max_text)

    body_errors = [row for row in all_body_rows if row["is_correct_section"] == "false"]
    body_missing = [row for row in all_body_rows if row["body_missing"] == "true"]
    heading_missing = [row for row in heading_rows if row["error_type"] == "missing_heading"]
    heading_mismatch = [row for row in heading_rows if row["error_type"] != "ok"]
    float_errors = [row for row in float_rows if row["is_correct_section"] == "false"]

    metric = lambda name, key="f1": ((metrics.get(name) or {}).get(key))
    summary = {
        "document_id": doc_id,
        "doc_dir": str(doc_dir),
        "macro_structure_score": metrics.get("macro_structure_score"),
        "heading_tree_accuracy": (metrics.get("heading_tree_accuracy") or {}).get("score"),
        "section_attachment_f1": metric("section_attachment_f1"),
        "section_attachment_body_no_float_f1": metric("section_attachment_body_no_float_f1"),
        "section_attachment_oracle_heading_flow_f1": metric("section_attachment_oracle_heading_flow_f1"),
        "float_caption_attachment_accuracy": (metrics.get("float_caption_attachment_accuracy") or {}).get("score"),
        "gold_headings": len([b for b in evaluator.gold_blocks if block_type(b) == "heading"]),
        "missing_headings": len(heading_missing),
        "heading_errors": len(heading_mismatch),
        "body_scope_total": len(all_body_rows),
        "body_scope_error_rows_written": len(body_rows),
        "body_scope_errors": len(body_errors),
        "body_missing": len(body_missing),
        "float_scope_rows": len(float_rows),
        "float_scope_errors": len(float_errors),
    }
    return {
        "summary": summary,
        "heading_rows": heading_rows,
        "body_rows": body_rows,
        "float_rows": float_rows,
    }


def inspect_headings(doc_id: str, evaluator: StructureMetricsEvaluator, max_text: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gold_heading in [b for b in evaluator.gold_blocks if block_type(b) == "heading"]:
        gid = block_id(gold_heading)
        pred_id = evaluator.gold_to_pred.get(gid)
        if not pred_id:
            rows.append(
                {
                    "document_id": doc_id,
                    "error_type": "missing_heading",
                    "gold_heading_id": gid,
                    "pred_heading_id": "",
                    "gold_level": gold_heading.get("level"),
                    "pred_level": "",
                    "gold_heading_text": truncate(block_text(gold_heading), max_text),
                    "pred_heading_text": "",
                    "gold_parent_heading": heading_text(nearest_heading_ancestor(gid, evaluator.gold_by_id), evaluator.gold_by_id, max_text),
                    "pred_parent_heading": "",
                }
            )
            continue
        pred_heading = evaluator.pred_by_id[pred_id]
        level_ok = int_or_none(gold_heading.get("level")) == int_or_none(pred_heading.get("level"))
        parent_ok = evaluator.parent_heading_matches(gold_heading, pred_heading)
        error_type = "ok" if level_ok and parent_ok else "level_or_parent_mismatch"
        if error_type == "ok":
            continue
        rows.append(
            {
                "document_id": doc_id,
                "error_type": error_type,
                "gold_heading_id": gid,
                "pred_heading_id": pred_id,
                "gold_level": gold_heading.get("level"),
                "pred_level": pred_heading.get("level"),
                "gold_heading_text": truncate(block_text(gold_heading), max_text),
                "pred_heading_text": truncate(block_text(pred_heading), max_text),
                "gold_parent_heading": heading_text(nearest_heading_ancestor(gid, evaluator.gold_by_id), evaluator.gold_by_id, max_text),
                "pred_parent_heading": heading_text(nearest_heading_ancestor(pred_id, evaluator.pred_by_id), evaluator.pred_by_id, max_text),
            }
        )
    return rows


def inspect_body_rows(
    doc_id: str,
    evaluator: StructureMetricsEvaluator,
    pred_active_by_id: dict[str, str | None],
    oracle_active_gold_by_id: dict[str, str | None],
    max_text: int,
    *,
    include_correct: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gold_block in evaluator.gold_blocks:
        if block_type(gold_block) not in BODY_TYPES:
            continue
        gid = block_id(gold_block)
        gold_heading = nearest_heading_ancestor(gid, evaluator.gold_by_id)
        if not gold_heading or section_scope_kind(gold_heading, evaluator.gold_by_id) != "body":
            continue
        pred_id = evaluator.gold_to_pred.get(gid)
        pred_heading = nearest_heading_ancestor(pred_id, evaluator.pred_by_id) if pred_id else None
        mapped_gold_heading = evaluator.gold_to_pred.get(gold_heading)
        is_correct = bool(pred_id and mapped_gold_heading and pred_heading == mapped_gold_heading)
        if is_correct and not include_correct:
            continue
        active_pred_heading = pred_active_by_id.get(pred_id) if pred_id else None
        oracle_active_gold = oracle_active_gold_by_id.get(gid)
        rows.append(
            {
                "document_id": doc_id,
                "gold_block_id": gid,
                "pred_block_id": pred_id or "",
                "block_type": block_type(gold_block),
                "body_missing": str(pred_id is None).lower(),
                "is_correct_section": str(is_correct).lower(),
                "gold_heading_id": gold_heading or "",
                "gold_heading_text": heading_text(gold_heading, evaluator.gold_by_id, max_text),
                "pred_parent_heading_id": pred_heading or "",
                "pred_parent_heading_text": heading_text(pred_heading, evaluator.pred_by_id, max_text),
                "current_active_heading_id": active_pred_heading or "",
                "current_active_heading_text": heading_text(active_pred_heading, evaluator.pred_by_id, max_text),
                "oracle_active_gold_heading_id": oracle_active_gold or "",
                "oracle_active_gold_heading_text": heading_text(oracle_active_gold, evaluator.gold_by_id, max_text),
                "mapped_gold_heading_in_pred": mapped_gold_heading or "",
                "gold_text": truncate(block_text(gold_block), max_text),
                "pred_text": truncate(block_text(evaluator.pred_by_id[pred_id]), max_text) if pred_id else "",
                "float_caption_interference": "false",
                "error_hint": body_error_hint(pred_id, pred_heading, mapped_gold_heading, active_pred_heading, oracle_active_gold, gold_heading),
            }
        )
    return rows


def inspect_float_rows(
    doc_id: str,
    evaluator: StructureMetricsEvaluator,
    pred_active_by_id: dict[str, str | None],
    max_text: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gold_block in evaluator.gold_blocks:
        if block_type(gold_block) not in FLOAT_TYPES:
            continue
        gid = block_id(gold_block)
        gold_heading = nearest_heading_ancestor(gid, evaluator.gold_by_id)
        pred_id = evaluator.gold_to_pred.get(gid)
        pred_heading = nearest_heading_ancestor(pred_id, evaluator.pred_by_id) if pred_id else None
        mapped_gold_heading = evaluator.gold_to_pred.get(gold_heading) if gold_heading else None
        is_correct = bool(pred_id and mapped_gold_heading and pred_heading == mapped_gold_heading)
        rows.append(
            {
                "document_id": doc_id,
                "gold_block_id": gid,
                "pred_block_id": pred_id or "",
                "block_type": block_type(gold_block),
                "body_missing": str(pred_id is None).lower(),
                "is_correct_section": str(is_correct).lower(),
                "gold_heading_id": gold_heading or "",
                "gold_heading_text": heading_text(gold_heading, evaluator.gold_by_id, max_text),
                "pred_parent_heading_id": pred_heading or "",
                "pred_parent_heading_text": heading_text(pred_heading, evaluator.pred_by_id, max_text),
                "current_active_heading_id": (pred_active_by_id.get(pred_id) if pred_id else None) or "",
                "current_active_heading_text": heading_text(pred_active_by_id.get(pred_id) if pred_id else None, evaluator.pred_by_id, max_text),
                "mapped_gold_heading_in_pred": mapped_gold_heading or "",
                "gold_text": truncate(block_text(gold_block), max_text),
                "pred_text": truncate(block_text(evaluator.pred_by_id[pred_id]), max_text) if pred_id else "",
                "float_caption_interference": "true",
                "error_hint": "float_caption_case" if not is_correct else "ok",
            }
        )
    return rows


def active_heading_by_predicted_order(evaluator: StructureMetricsEvaluator) -> dict[str, str | None]:
    active: str | None = None
    result: dict[str, str | None] = {}
    for block in sorted(evaluator.pred_blocks, key=lambda item: int(item.get("order") or 0)):
        bid = block_id(block)
        result[bid] = active
        if block_type(block) == "heading":
            active = bid
            result[bid] = active
    return result


def oracle_gold_heading_by_predicted_order(evaluator: StructureMetricsEvaluator) -> dict[str, str | None]:
    items: list[tuple[int, str, str]] = []
    for match in evaluator.matches:
        pred_block = evaluator.pred_by_id[match.pred_id]
        items.append((int(pred_block.get("order") or 0), match.gold_id, match.pred_id))
    items.sort(key=lambda item: item[0])
    active_gold_heading: str | None = None
    result: dict[str, str | None] = {}
    for _, gold_id, _ in items:
        result[gold_id] = active_gold_heading
        if block_type(evaluator.gold_by_id[gold_id]) == "heading":
            active_gold_heading = gold_id
            result[gold_id] = active_gold_heading
    return result


def body_error_hint(
    pred_id: str | None,
    pred_heading: str | None,
    mapped_gold_heading: str | None,
    active_pred_heading: str | None,
    oracle_active_gold: str | None,
    gold_heading: str | None,
) -> str:
    if pred_id is None:
        return "body_block_unmatched"
    if not mapped_gold_heading:
        return "gold_heading_unmatched_in_prediction"
    if pred_heading == mapped_gold_heading:
        return "ok"
    if active_pred_heading == mapped_gold_heading:
        return "parent_tree_differs_from_reading_active_heading"
    if oracle_active_gold != gold_heading:
        return "reading_flow_active_heading_differs_from_gold_heading"
    return "predicted_heading_scope_mismatch"


def aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "documents": len(rows),
        "gold_headings": sum_int(rows, "gold_headings"),
        "missing_headings": sum_int(rows, "missing_headings"),
        "heading_errors": sum_int(rows, "heading_errors"),
        "body_scope_total": sum_int(rows, "body_scope_total"),
        "body_scope_error_rows_written": sum_int(rows, "body_scope_error_rows_written"),
        "body_scope_errors": sum_int(rows, "body_scope_errors"),
        "body_missing": sum_int(rows, "body_missing"),
        "float_scope_rows": sum_int(rows, "float_scope_rows"),
        "float_scope_errors": sum_int(rows, "float_scope_errors"),
        "macro_structure_score": mean_float(rows, "macro_structure_score"),
        "heading_tree_accuracy": mean_float(rows, "heading_tree_accuracy"),
        "section_attachment_f1": mean_float(rows, "section_attachment_f1"),
        "section_attachment_body_no_float_f1": mean_float(rows, "section_attachment_body_no_float_f1"),
        "section_attachment_oracle_heading_flow_f1": mean_float(rows, "section_attachment_oracle_heading_flow_f1"),
        "float_caption_attachment_accuracy": mean_float(rows, "float_caption_attachment_accuracy"),
    }
    return totals


def sum_int(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key) or 0) for row in rows)


def mean_float(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def heading_text(heading_id: str | None, by_id: dict[str, dict[str, Any]], max_text: int) -> str:
    if not heading_id:
        return ""
    block = by_id.get(str(heading_id))
    if not block:
        return ""
    return truncate(block_text(block), max_text)


def truncate(text: str, max_text: int) -> str:
    value = " ".join(str(text or "").split())
    return value[: max_text - 1] + "…" if len(value) > max_text else value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
