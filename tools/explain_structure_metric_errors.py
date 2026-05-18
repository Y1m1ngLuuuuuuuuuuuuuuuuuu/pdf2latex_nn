#!/usr/bin/env python3
"""Explain comparison-structure metric failures in human-readable rows.

The core evaluator intentionally compresses many structural differences into a
small set of scores.  This tool expands those scores back into concrete cases:

- missing / extra / wrong-level headings
- body text whose coverage-aligned span is attached to the wrong heading
- paragraph text coverage gaps versus paragraph boundary splits
- float/caption pairing mismatches
- reference item coverage gaps

It accepts either a single gold/pred pair or a directory containing many E2E
document folders.  Outputs are designed for quick paper/debug triage rather
than training.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.structure_metrics import (  # noqa: E402
    BODY_SECTION_ATTACHMENT_TYPES,
    FLOAT_SECTION_ATTACHMENT_TYPES,
    TEXT_LIKE_TYPES,
    StructureMetricsEvaluator,
    block_id,
    block_text,
    block_type,
    caption_parent_kind,
    load_comparison_json,
    majority_heading,
    nearest_heading_ancestor,
    normalized_text,
    numeric_order,
    section_scope_kind,
)


DEFAULT_MAX_ROWS = 500


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--e2e-dir", type=Path, help="Directory containing per-document E2E folders.")
    input_group.add_argument("--doc-dir", type=Path, help="One E2E document folder with gold/generated structures.")
    input_group.add_argument("--gold", type=Path, help="Gold comparison-structure JSON.")
    parser.add_argument("--pred", type=Path, help="Predicted comparison-structure JSON. Required with --gold.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-threshold", type=float, default=0.58)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--max-text", type=int, default=220)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.gold and not args.pred:
        raise SystemExit("--pred is required when --gold is used")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    doc_specs = discover_document_specs(args)
    if not doc_specs:
        raise FileNotFoundError("No gold/pred comparison-structure pairs found.")

    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    for spec in doc_specs:
        payload = explain_document(spec, args)
        summaries.append(payload["summary"])
        rows.extend(payload["rows"])
        details[payload["summary"]["document_id"]] = payload["detail"]

    rows.sort(key=sort_row)
    limited_rows = rows[: args.max_rows]

    report = {
        "schema_version": "structure_metric_error_explainer_v1",
        "input_count": len(doc_specs),
        "match_threshold": args.match_threshold,
        "aggregate": aggregate_summaries(summaries, rows),
        "documents": summaries,
        "details": details,
        "rows_truncated_to": args.max_rows,
    }
    write_json(args.output_dir / "summary.json", report)
    write_csv(args.output_dir / "errors.csv", limited_rows)
    write_html(args.output_dir / "report.html", report, limited_rows)

    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {args.output_dir}")
    return 0


def discover_document_specs(args: argparse.Namespace) -> list[dict[str, Path]]:
    if args.gold:
        return [{"doc_dir": args.gold.parent, "gold": args.gold, "pred": args.pred}]
    if args.doc_dir:
        return [doc_spec_from_dir(args.doc_dir)]
    specs: list[dict[str, Path]] = []
    for doc_dir in sorted(path for path in args.e2e_dir.iterdir() if path.is_dir()):
        try:
            specs.append(doc_spec_from_dir(doc_dir))
        except FileNotFoundError:
            continue
    return specs


def doc_spec_from_dir(doc_dir: Path) -> dict[str, Path]:
    gold = doc_dir / "gold_structure.json"
    pred = doc_dir / "generated_structure.json"
    if not gold.exists() or not pred.exists():
        raise FileNotFoundError(f"{doc_dir} does not contain gold_structure.json and generated_structure.json")
    return {"doc_dir": doc_dir, "gold": gold, "pred": pred}


def explain_document(spec: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    gold = load_comparison_json(spec["gold"])
    pred = load_comparison_json(spec["pred"])
    evaluator = StructureMetricsEvaluator(gold, pred, match_threshold=args.match_threshold)
    metrics = evaluator.evaluate()
    doc_id = str(gold.get("doc_id") or spec["doc_dir"].name)

    rows: list[dict[str, Any]] = []
    rows.extend(explain_heading_errors(doc_id, evaluator, args.max_text))
    rows.extend(explain_section_attachment_errors(doc_id, evaluator, args.max_text))
    rows.extend(explain_text_coverage_errors(doc_id, evaluator, args.max_text))
    rows.extend(explain_float_caption_errors(doc_id, evaluator, args.max_text))
    rows.extend(explain_reference_errors(doc_id, evaluator, args.max_text))

    reason_counts = Counter(row["reason"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    summary = {
        "document_id": doc_id,
        "doc_dir": str(spec["doc_dir"]),
        "gold_path": str(spec["gold"]),
        "pred_path": str(spec["pred"]),
        "macro_structure_score": metrics.get("macro_structure_score"),
        "heading_tree_accuracy": (metrics.get("heading_tree_accuracy") or {}).get("score"),
        "section_attachment_f1": (metrics.get("section_attachment_f1") or {}).get("f1"),
        "section_attachment_body_no_float_f1": (metrics.get("section_attachment_body_no_float_f1") or {}).get("f1"),
        "paragraph_text_coverage_f1": (metrics.get("paragraph_text_coverage_f1") or {}).get("f1"),
        "paragraph_boundary_f1": (metrics.get("paragraph_boundary_f1") or {}).get("f1"),
        "float_caption_attachment_accuracy": (metrics.get("float_caption_attachment_accuracy") or {}).get("score"),
        "reference_section_completeness": (metrics.get("reference_section_completeness") or {}).get("score"),
        "strict_block_match_gold_coverage": (metrics.get("strict_block_match") or {}).get("coverage_gold"),
        "window_match_gold_coverage": (metrics.get("window_matching") or {}).get("coverage_gold_blocks"),
        "error_rows": len(rows),
        "category_counts": dict(category_counts),
        "top_reasons": dict(reason_counts.most_common(12)),
    }
    detail = {
        "metrics": metrics,
        "heading_errors": [row for row in rows if row["category"] == "heading"][:100],
        "section_attachment_errors": [row for row in rows if row["category"] == "section_attachment"][:100],
        "text_coverage_errors": [row for row in rows if row["category"] == "text_coverage"][:100],
        "float_caption_errors": [row for row in rows if row["category"] == "float_caption"][:100],
        "reference_errors": [row for row in rows if row["category"] == "reference"][:100],
    }
    return {"summary": summary, "rows": rows, "detail": detail}


def explain_heading_errors(doc_id: str, evaluator: StructureMetricsEvaluator, max_text: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matched_pred_headings: set[str] = set()
    for gold_heading in headings(evaluator.gold_blocks):
        gid = block_id(gold_heading)
        pred_id = evaluator.gold_to_pred.get(gid)
        if not pred_id:
            rows.append(error_row(doc_id, "heading", "missing_gold_heading", gold_heading, None, max_text))
            continue
        pred_heading = evaluator.pred_by_id[pred_id]
        matched_pred_headings.add(pred_id)
        level_ok = str(gold_heading.get("level")) == str(pred_heading.get("level"))
        gold_parent = nearest_heading_ancestor(gid, evaluator.gold_by_id)
        pred_parent = nearest_heading_ancestor(pred_id, evaluator.pred_by_id)
        mapped_gold_parent = evaluator.gold_to_pred.get(gold_parent or "")
        parent_ok = mapped_gold_parent == pred_parent
        if not level_ok:
            rows.append(
                error_row(
                    doc_id,
                    "heading",
                    "heading_level_mismatch",
                    gold_heading,
                    pred_heading,
                    max_text,
                    gold_heading=heading_descriptor(gold_parent, evaluator.gold_by_id, max_text),
                    pred_heading=heading_descriptor(pred_parent, evaluator.pred_by_id, max_text),
                    score=str(evaluator.match_score(gid, pred_id) if hasattr(evaluator, "match_score") else ""),
                )
            )
        if not parent_ok:
            rows.append(
                error_row(
                    doc_id,
                    "heading",
                    "heading_parent_mismatch",
                    gold_heading,
                    pred_heading,
                    max_text,
                    gold_heading=heading_descriptor(gold_parent, evaluator.gold_by_id, max_text),
                    pred_heading=heading_descriptor(pred_parent, evaluator.pred_by_id, max_text),
                )
            )
    for pred_heading in headings(evaluator.pred_blocks):
        pid = block_id(pred_heading)
        if pid not in matched_pred_headings and pid not in evaluator.pred_to_gold:
            rows.append(error_row(doc_id, "heading", "extra_pred_heading", None, pred_heading, max_text))
    return rows


def explain_section_attachment_errors(
    doc_id: str,
    evaluator: StructureMetricsEvaluator,
    max_text: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_gold: set[str] = set()

    for match in evaluator.text_window_matches:
        gold_blocks = [evaluator.gold_by_id[gid] for gid in match.gold_ids if gid in evaluator.gold_by_id]
        pred_blocks = [evaluator.pred_by_id[pid] for pid in match.pred_ids if pid in evaluator.pred_by_id]
        scoped_gold = [
            block for block in gold_blocks
            if block_type(block) in BODY_SECTION_ATTACHMENT_TYPES
            and section_scope_kind(nearest_heading_ancestor(block_id(block), evaluator.gold_by_id), evaluator.gold_by_id) == "body"
        ]
        if not scoped_gold:
            continue
        seen_gold.update(block_id(block) for block in scoped_gold)
        gold_heading = majority_heading(scoped_gold, evaluator.gold_by_id)
        pred_heading = majority_heading(pred_blocks, evaluator.pred_by_id)
        mapped_gold_heading = evaluator.gold_to_pred.get(gold_heading or "")
        if mapped_gold_heading and pred_heading == mapped_gold_heading:
            continue
        rows.append(
            window_error_row(
                doc_id,
                "section_attachment",
                section_error_reason(mapped_gold_heading, pred_heading),
                scoped_gold,
                pred_blocks,
                evaluator,
                max_text,
                score=match.score,
                extra={
                    "gold_heading_id": gold_heading or "",
                    "gold_heading_text": heading_descriptor(gold_heading, evaluator.gold_by_id, max_text),
                    "mapped_gold_heading_pred_id": mapped_gold_heading or "",
                    "pred_heading_id": pred_heading or "",
                    "pred_heading_text": heading_descriptor(pred_heading, evaluator.pred_by_id, max_text),
                    "match_kind": "window",
                },
            )
        )

    for gold_block in evaluator.gold_blocks:
        gid = block_id(gold_block)
        if gid in seen_gold or block_type(gold_block) not in BODY_SECTION_ATTACHMENT_TYPES:
            continue
        gold_heading = nearest_heading_ancestor(gid, evaluator.gold_by_id)
        if not gold_heading or section_scope_kind(gold_heading, evaluator.gold_by_id) != "body":
            continue
        pred_id = evaluator.gold_to_pred.get(gid)
        if not pred_id:
            rows.append(
                error_row(
                    doc_id,
                    "section_attachment",
                    "body_block_missing_strict_match",
                    gold_block,
                    None,
                    max_text,
                    gold_heading=heading_descriptor(gold_heading, evaluator.gold_by_id, max_text),
                    pred_heading="",
                )
            )
            continue
        pred_heading = nearest_heading_ancestor(pred_id, evaluator.pred_by_id)
        mapped_gold_heading = evaluator.gold_to_pred.get(gold_heading)
        if mapped_gold_heading and pred_heading == mapped_gold_heading:
            continue
        rows.append(
            error_row(
                doc_id,
                "section_attachment",
                section_error_reason(mapped_gold_heading, pred_heading),
                gold_block,
                evaluator.pred_by_id[pred_id],
                max_text,
                gold_heading=heading_descriptor(gold_heading, evaluator.gold_by_id, max_text),
                pred_heading=heading_descriptor(pred_heading, evaluator.pred_by_id, max_text),
            )
        )
    return rows


def explain_text_coverage_errors(
    doc_id: str,
    evaluator: StructureMetricsEvaluator,
    max_text: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gold_block in evaluator.gold_blocks:
        if block_type(gold_block) not in TEXT_LIKE_TYPES:
            continue
        gid = block_id(gold_block)
        if gid not in evaluator.gold_to_text_window:
            strict = evaluator.gold_to_pred.get(gid)
            rows.append(
                error_row(
                    doc_id,
                    "text_coverage",
                    "gold_text_not_covered_by_window",
                    gold_block,
                    evaluator.pred_by_id.get(strict) if strict else None,
                    max_text,
                    match_kind="strict_only" if strict else "unmatched",
                )
            )
    for pred_block in evaluator.pred_blocks:
        if block_type(pred_block) not in TEXT_LIKE_TYPES:
            continue
        pid = block_id(pred_block)
        if pid not in evaluator.pred_to_text_window:
            strict = evaluator.pred_to_gold.get(pid)
            rows.append(
                error_row(
                    doc_id,
                    "text_coverage",
                    "pred_text_not_explained_by_window",
                    evaluator.gold_by_id.get(strict) if strict else None,
                    pred_block,
                    max_text,
                    match_kind="strict_only" if strict else "extra",
                )
            )
    return rows


def explain_float_caption_errors(doc_id: str, evaluator: StructureMetricsEvaluator, max_text: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matched_pred_captions: set[str] = set()
    for gold_caption in [block for block in evaluator.gold_blocks if block_type(block) == "caption"]:
        gid = block_id(gold_caption)
        pred_id = evaluator.gold_to_pred.get(gid)
        if not pred_id:
            rows.append(error_row(doc_id, "float_caption", "missing_caption", gold_caption, None, max_text))
            continue
        pred_caption = evaluator.pred_by_id[pred_id]
        matched_pred_captions.add(pred_id)
        gold_kind = caption_parent_kind(gold_caption, evaluator.gold_by_id)
        pred_kind = caption_parent_kind(pred_caption, evaluator.pred_by_id)
        if gold_kind != pred_kind:
            rows.append(
                error_row(
                    doc_id,
                    "float_caption",
                    "caption_parent_kind_mismatch",
                    gold_caption,
                    pred_caption,
                    max_text,
                    extra={"gold_float_kind": gold_kind or "", "pred_float_kind": pred_kind or ""},
                )
            )
    for pred_caption in [block for block in evaluator.pred_blocks if block_type(block) == "caption"]:
        pid = block_id(pred_caption)
        if pid not in matched_pred_captions and pid not in evaluator.pred_to_gold:
            rows.append(error_row(doc_id, "float_caption", "extra_caption", None, pred_caption, max_text))
    return rows


def explain_reference_errors(doc_id: str, evaluator: StructureMetricsEvaluator, max_text: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gold_ref in [block for block in evaluator.gold_blocks if block_type(block) == "reference_item"]:
        gid = block_id(gold_ref)
        pred_id = evaluator.gold_to_pred.get(gid)
        if not pred_id:
            rows.append(error_row(doc_id, "reference", "missing_reference_item", gold_ref, None, max_text))
    for pred_ref in [block for block in evaluator.pred_blocks if block_type(block) == "reference_item"]:
        pid = block_id(pred_ref)
        if pid not in evaluator.pred_to_gold:
            rows.append(error_row(doc_id, "reference", "extra_reference_item", None, pred_ref, max_text))
    return rows


def section_error_reason(mapped_gold_heading: str | None, pred_heading: str | None) -> str:
    if not mapped_gold_heading and not pred_heading:
        return "both_headings_unmatched"
    if not mapped_gold_heading:
        return "gold_heading_not_matched_in_pred"
    if not pred_heading:
        return "pred_block_has_no_heading_parent"
    return "wrong_heading_parent"


def error_row(
    doc_id: str,
    category: str,
    reason: str,
    gold_block: dict[str, Any] | None,
    pred_block: dict[str, Any] | None,
    max_text: int,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "document_id": doc_id,
        "category": category,
        "reason": reason,
        "gold_id": block_id(gold_block) if gold_block else "",
        "pred_id": block_id(pred_block) if pred_block else "",
        "gold_type": block_type(gold_block) if gold_block else "",
        "pred_type": block_type(pred_block) if pred_block else "",
        "gold_order": numeric_order(gold_block) if gold_block else "",
        "pred_order": numeric_order(pred_block) if pred_block else "",
        "gold_level": gold_block.get("level", "") if gold_block else "",
        "pred_level": pred_block.get("level", "") if pred_block else "",
        "gold_heading": extra.pop("gold_heading", ""),
        "pred_heading": extra.pop("pred_heading", ""),
        "match_kind": extra.pop("match_kind", "strict"),
        "score": extra.pop("score", ""),
        "gold_text": truncate(block_text(gold_block), max_text) if gold_block else "",
        "pred_text": truncate(block_text(pred_block), max_text) if pred_block else "",
    }
    for key, value in extra.items():
        if key == "extra" and isinstance(value, dict):
            row.update(value)
        else:
            row[key] = value
    return row


def window_error_row(
    doc_id: str,
    category: str,
    reason: str,
    gold_blocks: list[dict[str, Any]],
    pred_blocks: list[dict[str, Any]],
    evaluator: StructureMetricsEvaluator,
    max_text: int,
    *,
    score: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    gold_text = " / ".join(truncate(block_text(block), max_text // 2) for block in gold_blocks[:4])
    pred_text = " / ".join(truncate(block_text(block), max_text // 2) for block in pred_blocks[:4])
    row = {
        "document_id": doc_id,
        "category": category,
        "reason": reason,
        "gold_id": "|".join(block_id(block) for block in gold_blocks),
        "pred_id": "|".join(block_id(block) for block in pred_blocks),
        "gold_type": "+".join(block_type(block) for block in gold_blocks),
        "pred_type": "+".join(block_type(block) for block in pred_blocks),
        "gold_order": numeric_order(gold_blocks[0]) if gold_blocks else "",
        "pred_order": numeric_order(pred_blocks[0]) if pred_blocks else "",
        "gold_level": "",
        "pred_level": "",
        "gold_heading": extra.get("gold_heading_text", ""),
        "pred_heading": extra.get("pred_heading_text", ""),
        "match_kind": extra.get("match_kind", "window"),
        "score": f"{score:.4f}",
        "gold_text": gold_text,
        "pred_text": pred_text,
    }
    row.update(extra)
    return row


def headings(blocks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [block for block in blocks if block_type(block) == "heading"]


def heading_descriptor(heading_id: str | None, by_id: dict[str, dict[str, Any]], max_text: int) -> str:
    if not heading_id:
        return ""
    block = by_id.get(str(heading_id))
    if not block:
        return str(heading_id)
    level = block.get("level")
    text = truncate(block_text(block), max_text)
    return f"{heading_id} L{level}: {text}"


def truncate(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def sort_row(row: dict[str, Any]) -> tuple[Any, ...]:
    severity = {
        "missing_gold_heading": 0,
        "heading_parent_mismatch": 1,
        "heading_level_mismatch": 2,
        "wrong_heading_parent": 3,
        "pred_block_has_no_heading_parent": 4,
        "gold_heading_not_matched_in_pred": 5,
        "gold_text_not_covered_by_window": 6,
        "missing_caption": 7,
        "caption_parent_kind_mismatch": 8,
        "missing_reference_item": 9,
    }.get(str(row.get("reason")), 50)
    try:
        gold_order = int(row.get("gold_order") or 10**9)
    except (TypeError, ValueError):
        gold_order = 10**9
    return (str(row.get("document_id")), severity, gold_order, str(row.get("category")), str(row.get("reason")))


def aggregate_summaries(summaries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(row["category"] for row in rows)
    reason_counts = Counter(row["reason"] for row in rows)
    return {
        "documents": len(summaries),
        "error_rows": len(rows),
        "category_counts": dict(category_counts.most_common()),
        "top_reasons": dict(reason_counts.most_common(20)),
        "mean_macro_structure_score": mean_metric(summaries, "macro_structure_score"),
        "mean_heading_tree_accuracy": mean_metric(summaries, "heading_tree_accuracy"),
        "mean_section_attachment_f1": mean_metric(summaries, "section_attachment_f1"),
        "mean_body_no_float_section_attachment_f1": mean_metric(summaries, "section_attachment_body_no_float_f1"),
        "mean_paragraph_text_coverage_f1": mean_metric(summaries, "paragraph_text_coverage_f1"),
        "mean_paragraph_boundary_f1": mean_metric(summaries, "paragraph_boundary_f1"),
        "mean_float_caption_attachment_accuracy": mean_metric(summaries, "float_caption_attachment_accuracy"),
        "mean_reference_section_completeness": mean_metric(summaries, "reference_section_completeness"),
    }


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return float(sum(values) / len(values))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "document_id",
        "category",
        "reason",
        "gold_id",
        "pred_id",
        "gold_type",
        "pred_type",
        "gold_order",
        "pred_order",
        "gold_level",
        "pred_level",
        "gold_heading",
        "pred_heading",
        "match_kind",
        "score",
        "gold_text",
        "pred_text",
    ]
    fieldnames = [field for field in preferred if field in fieldnames] + [field for field in fieldnames if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = report["aggregate"]
    table_headers = [
        "document_id",
        "category",
        "reason",
        "gold_id",
        "pred_id",
        "gold_heading",
        "pred_heading",
        "gold_text",
        "pred_text",
    ]
    html_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in table_headers)
        html_rows.append(f"<tr>{cells}</tr>")
    metric_items = "".join(
        f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>"
        for key, value in aggregate.items()
    )
    header_cells = "".join(f"<th>{html.escape(key)}</th>" for key in table_headers)
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Structure Metric Error Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f5f5f5; }}
    td:nth-child(8), td:nth-child(9) {{ max-width: 360px; }}
  </style>
</head>
<body>
  <h1>Structure Metric Error Report</h1>
  <h2>Aggregate</h2>
  <ul>{metric_items}</ul>
  <h2>Top Error Rows</h2>
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
