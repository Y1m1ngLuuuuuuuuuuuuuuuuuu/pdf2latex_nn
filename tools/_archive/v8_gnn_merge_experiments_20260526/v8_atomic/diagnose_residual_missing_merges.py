#!/usr/bin/env python3
"""Diagnose MERGE cases missed by both deterministic v8 and learned overlay.

This tool is intentionally read-only. It consumes existing projection outputs
from ``project_predictions_to_v8.py`` and the paragraph-preservation audits
already generated for:

  - deterministic v8
  - learned_plus_deterministic overlay

The target residual is:

  source TeX paragraph is split in deterministic output
  AND the same source TeX paragraph is still split after learned overlay.

That is the concrete set of merge failures that neither hard-coded v8 merge nor
the current GNN overlay recovered.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OPEN_TAIL_RE = re.compile(
    r"(?:-$|[,;:([{]$|\\b(?:and|or|of|to|for|with|by|in|on|as|that|which|where|while|from|into|between|among|using|via)$)",
    re.IGNORECASE,
)
LOWER_HEAD_RE = re.compile(r"^[a-z]")
PAREN_HEAD_RE = re.compile(r"^[([{]")
MATHISH_RE = re.compile(r"(?:\\mathbf|\\mathcal|\\frac|\\sum|\\prod|\\alpha|\\beta|\\gamma|[_^=<>]|[∑∏≤≥≈])")
HEADINGISH_RE = re.compile(
    r"^(?:[0-9]+(?:\\.[0-9]+){0,2}\\s+)?"
    r"(?:abstract|introduction|related work|method|methods|experiments?|results?|discussion|conclusion|references?)\\b",
    re.IGNORECASE,
)
CITATIONISH_RE = re.compile(r"(?:\([A-Z][A-Za-z-]+\s+et\s+al\.?[, ]+\d{4}\)|\[[0-9,\-\s]+\])")


@dataclass
class GapDiagnosis:
    left_generated_id: str
    right_generated_id: str
    left_index: int | None
    right_index: int | None
    index_gap: int | None
    left_line: int | None
    right_line: int | None
    left_tail: str
    right_head: str
    left_open_ended: bool
    right_continuation_like: bool
    suspected_reason: str
    matched_accepted_overlay_edge: dict[str, Any] | None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--doc-ids", nargs="*")
    parser.add_argument("--max-examples-per-doc", type=int, default=50)
    parser.add_argument("--max-md-examples", type=int, default=80)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    doc_dirs = sorted(p for p in args.projection_dir.iterdir() if p.is_dir())
    if args.doc_ids:
        wanted = set(args.doc_ids)
        doc_dirs = [p for p in doc_dirs if p.name in wanted]
    if args.limit is not None:
        doc_dirs = doc_dirs[: args.limit]

    rows: list[dict[str, Any]] = []
    all_residuals: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    reason_counter: Counter[str] = Counter()
    gate_counter: Counter[str] = Counter()

    for doc_dir in doc_dirs:
        try:
            result = diagnose_doc(doc_dir, args.output_dir / doc_dir.name, args.max_examples_per_doc)
        except Exception as exc:  # keep batch moving
            failures.append({"doc_id": doc_dir.name, "error": type(exc).__name__, "message": str(exc)})
            continue
        rows.append(result["summary"])
        all_residuals.extend(result["residual_missing_merges"])
        reason_counter.update(result["reason_counts"])
        gate_counter.update(result["overlay_rejection_reason_counts"])

    summary = build_summary(args, rows, failures, reason_counter, gate_counter, all_residuals)
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "summary.csv", rows)
    write_markdown(args.output_dir / "RESIDUAL_MISSING_MERGE_REPORT.md", summary, all_residuals, args.max_md_examples)
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'RESIDUAL_MISSING_MERGE_REPORT.md'}")
    return 0


def diagnose_doc(doc_dir: Path, output_dir: Path, max_examples: int) -> dict[str, Any]:
    det_path = doc_dir / "deterministic" / "paragraph_audit" / "paragraph_preservation_against_tex.json"
    learned_path = doc_dir / "learned_plus_deterministic" / "paragraph_audit" / "paragraph_preservation_against_tex.json"
    projection_path = doc_dir / "projection_report.json"
    if not det_path.exists():
        raise FileNotFoundError(det_path)
    if not learned_path.exists():
        raise FileNotFoundError(learned_path)
    if not projection_path.exists():
        raise FileNotFoundError(projection_path)

    det = load_json(det_path)
    learned = load_json(learned_path)
    projection = load_json(projection_path)

    det_missing = {ex["source"]["block_id"]: ex for ex in det.get("missing_merge_examples", [])}
    learned_missing = {ex["source"]["block_id"]: ex for ex in learned.get("missing_merge_examples", [])}
    residual_ids = sorted(set(det_missing) & set(learned_missing), key=source_sort_key)
    fixed_ids = sorted(set(det_missing) - set(learned_missing), key=source_sort_key)
    new_missing_ids = sorted(set(learned_missing) - set(det_missing), key=source_sort_key)

    accepted_edges = projection.get("projection", {}).get("model_added_owner_merges", []) or []
    rejection_counts = Counter(projection.get("summary", {}).get("rejection_reason_counts") or {})
    reason_counts: Counter[str] = Counter()
    residuals: list[dict[str, Any]] = []
    for source_id in residual_ids:
        det_ex = det_missing[source_id]
        learned_ex = learned_missing[source_id]
        gaps = diagnose_gaps(learned_ex.get("generated_parts", []), accepted_edges)
        reason_counts.update(gap.suspected_reason for gap in gaps)
        residuals.append(
            {
                "doc_id": doc_dir.name,
                "source": learned_ex.get("source"),
                "deterministic": summarize_missing_example(det_ex),
                "learned_plus_deterministic": summarize_missing_example(learned_ex),
                "gap_count": len(gaps),
                "gaps": [gap.__dict__ for gap in gaps],
                "doc_overlay": {
                    "model_added_owner_merge_count": projection.get("summary", {}).get("model_added_owner_merge_count"),
                    "model_predicted_cross_owner_atomic_merge_count": projection.get("summary", {}).get(
                        "model_predicted_cross_owner_atomic_merge_count"
                    ),
                    "strict_overlay_candidate_count": projection.get("summary", {}).get("strict_overlay_candidate_count"),
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_payload = {
        "schema_version": "v8_residual_missing_merge_doc_v1",
        "doc_id": doc_dir.name,
        "inputs": {
            "projection_report": str(projection_path),
            "deterministic_audit": str(det_path),
            "learned_audit": str(learned_path),
        },
        "summary": {
            "deterministic_missing_source_count": len(det_missing),
            "learned_missing_source_count": len(learned_missing),
            "residual_missing_source_count": len(residual_ids),
            "fixed_by_learned_count": len(fixed_ids),
            "newly_missing_after_learned_count": len(new_missing_ids),
            "residual_gap_count": sum(item["gap_count"] for item in residuals),
            "model_added_owner_merge_count": projection.get("summary", {}).get("model_added_owner_merge_count"),
            "generated_tex_changed": projection.get("summary", {}).get("generated_tex_changed"),
        },
        "reason_counts": dict(reason_counts),
        "overlay_rejection_reason_counts": dict(rejection_counts),
        "fixed_by_learned_source_ids": fixed_ids,
        "newly_missing_after_learned_source_ids": new_missing_ids,
        "residual_missing_merges": residuals[:max_examples],
    }
    write_json(output_dir / "residual_missing_merges.json", doc_payload)
    write_doc_markdown(output_dir / "RESIDUAL_MISSING_MERGES.md", doc_payload)

    summary = {
        "doc_id": doc_dir.name,
        **doc_payload["summary"],
        "top_residual_reason": reason_counts.most_common(1)[0][0] if reason_counts else "",
        "top_overlay_rejection_reason": rejection_counts.most_common(1)[0][0] if rejection_counts else "",
    }
    return {
        "summary": summary,
        "residual_missing_merges": residuals,
        "reason_counts": dict(reason_counts),
        "overlay_rejection_reason_counts": dict(rejection_counts),
    }


def diagnose_gaps(generated_parts: list[dict[str, Any]], accepted_edges: list[dict[str, Any]]) -> list[GapDiagnosis]:
    parts = sorted(generated_parts, key=lambda part: part.get("generated", {}).get("index", 0))
    gaps: list[GapDiagnosis] = []
    for left, right in zip(parts, parts[1:]):
        left_gen = left.get("generated", {})
        right_gen = right.get("generated", {})
        left_preview = str(left_gen.get("preview") or "")
        right_preview = str(right_gen.get("preview") or "")
        left_tail = tail_text(left_preview)
        right_head = head_text(right_preview)
        left_open = is_open_ended(left_tail)
        right_cont = is_continuation_like(right_head)
        index_gap = safe_index_gap(left_gen.get("index"), right_gen.get("index"))
        matched_edge = find_matching_accepted_edge(left_preview, right_preview, accepted_edges)
        reason = classify_gap(left_preview, right_preview, left_open, right_cont, index_gap, matched_edge)
        gaps.append(
            GapDiagnosis(
                left_generated_id=str(left_gen.get("block_id") or left.get("generated_id") or ""),
                right_generated_id=str(right_gen.get("block_id") or right.get("generated_id") or ""),
                left_index=left_gen.get("index"),
                right_index=right_gen.get("index"),
                index_gap=index_gap,
                left_line=left_gen.get("line"),
                right_line=right_gen.get("line"),
                left_tail=left_tail,
                right_head=right_head,
                left_open_ended=left_open,
                right_continuation_like=right_cont,
                suspected_reason=reason,
                matched_accepted_overlay_edge=matched_edge,
            )
        )
    return gaps


def classify_gap(
    left_preview: str,
    right_preview: str,
    left_open: bool,
    right_cont: bool,
    index_gap: int | None,
    matched_edge: dict[str, Any] | None,
) -> str:
    joined = f"{left_preview} {right_preview}"
    if matched_edge:
        return "accepted_overlay_edge_text_overlap_but_still_split"
    if HEADINGISH_RE.search(right_preview.strip()):
        return "right_part_looks_like_heading_or_run_in"
    if MATHISH_RE.search(joined):
        return "math_or_formula_granularity"
    if CITATIONISH_RE.search(joined):
        return "citation_reference_heavy"
    if index_gap is not None and index_gap > 1:
        return "large_generated_gap_or_intervening_blocks"
    if not left_open and not right_cont:
        return "not_visually_continuation_like"
    if not left_open:
        return "left_tail_not_open_ended"
    if not right_cont:
        return "right_head_not_continuation_like"
    return "residual_even_though_tail_head_looks_mergeable"


def find_matching_accepted_edge(left_preview: str, right_preview: str, accepted_edges: list[dict[str, Any]]) -> dict[str, Any] | None:
    left_tokens = set(tokenize(tail_text(left_preview, words=14)))
    right_tokens = set(tokenize(head_text(right_preview, words=14)))
    best: tuple[int, dict[str, Any]] | None = None
    for edge in accepted_edges:
        src_tokens = set(tokenize(str(edge.get("src_text") or "")))
        dst_tokens = set(tokenize(str(edge.get("dst_text") or "")))
        score = len(left_tokens & src_tokens) + len(right_tokens & dst_tokens)
        if score >= 5 and (best is None or score > best[0]):
            best = (score, edge)
    if best is None:
        return None
    edge = best[1]
    return {
        "edge_id": edge.get("edge_id"),
        "probability": edge.get("probability"),
        "candidate_family": edge.get("candidate_family"),
        "layout_scope": edge.get("layout_scope"),
        "reading_order_gap": edge.get("reading_order_gap"),
        "src_text": str(edge.get("src_text") or "")[:180],
        "dst_text": str(edge.get("dst_text") or "")[:180],
    }


def summarize_missing_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "combined_recall": example.get("combined_recall"),
        "best_source_recall": example.get("best_source_recall"),
        "generated_part_count": len(example.get("generated_parts", []) or []),
        "generated_part_ids": [
            (part.get("generated") or {}).get("block_id") or part.get("generated_id")
            for part in example.get("generated_parts", [])
        ],
    }


def build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    reason_counter: Counter[str],
    gate_counter: Counter[str],
    residuals: list[dict[str, Any]],
) -> dict[str, Any]:
    total_residual = sum(row.get("residual_missing_source_count", 0) or 0 for row in rows)
    total_fixed = sum(row.get("fixed_by_learned_count", 0) or 0 for row in rows)
    total_new = sum(row.get("newly_missing_after_learned_count", 0) or 0 for row in rows)
    return {
        "schema_version": "v8_residual_missing_merge_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "projection_dir": str(args.projection_dir),
        "doc_count": len(rows),
        "failure_count": len(failures),
        "total_residual_missing_source_count": total_residual,
        "total_fixed_by_learned_count": total_fixed,
        "total_newly_missing_after_learned_count": total_new,
        "total_residual_gap_count": sum(row.get("residual_gap_count", 0) or 0 for row in rows),
        "docs_with_residual_missing": sum(1 for row in rows if (row.get("residual_missing_source_count") or 0) > 0),
        "top_residual_reasons": reason_counter.most_common(20),
        "top_overlay_rejection_reasons": gate_counter.most_common(20),
        "failures": failures,
        "rows": rows,
        "sample_residuals": residuals[: min(100, len(residuals))],
    }


def write_markdown(path: Path, summary: dict[str, Any], residuals: list[dict[str, Any]], max_examples: int) -> None:
    lines = [
        "# Residual Missing Merge Report",
        "",
        "This report lists source TeX paragraphs that are split in deterministic v8 output and remain split after the learned overlay.",
        "",
        "## Summary",
        "",
        f"- docs: {summary['doc_count']}",
        f"- residual missing source paragraphs: {summary['total_residual_missing_source_count']}",
        f"- residual gaps: {summary['total_residual_gap_count']}",
        f"- fixed by learned overlay: {summary['total_fixed_by_learned_count']}",
        f"- newly missing after learned overlay: {summary['total_newly_missing_after_learned_count']}",
        f"- docs with residuals: {summary['docs_with_residual_missing']}",
        "",
        "## Top Residual Reasons",
        "",
    ]
    for reason, count in summary["top_residual_reasons"][:12]:
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Top Overlay Rejection Reasons", ""])
    for reason, count in summary["top_overlay_rejection_reasons"][:12]:
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Sample Residuals", ""])
    for item in residuals[:max_examples]:
        source = item.get("source") or {}
        lines.append(f"### {item['doc_id']} / {source.get('block_id')}")
        lines.append("")
        lines.append(f"- source line: {source.get('line')}")
        lines.append(f"- source preview: {source.get('preview')}")
        lines.append(f"- deterministic parts: {item['deterministic']['generated_part_count']}")
        lines.append(f"- learned parts: {item['learned_plus_deterministic']['generated_part_count']}")
        for gap in item.get("gaps", [])[:4]:
            lines.append(
                f"- gap {gap['left_generated_id']} -> {gap['right_generated_id']}: "
                f"{gap['suspected_reason']} | tail=`{gap['left_tail']}` head=`{gap['right_head']}`"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        f"# Residual Missing Merges: {payload['doc_id']}",
        "",
        f"- deterministic missing source count: {summary['deterministic_missing_source_count']}",
        f"- learned missing source count: {summary['learned_missing_source_count']}",
        f"- residual missing source count: {summary['residual_missing_source_count']}",
        f"- fixed by learned count: {summary['fixed_by_learned_count']}",
        f"- newly missing after learned count: {summary['newly_missing_after_learned_count']}",
        f"- residual gap count: {summary['residual_gap_count']}",
        "",
        "## Reason Counts",
        "",
    ]
    for reason, count in payload.get("reason_counts", {}).items():
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Residuals", ""])
    for item in payload.get("residual_missing_merges", []):
        source = item.get("source") or {}
        lines.append(f"### {source.get('block_id')} line {source.get('line')}")
        lines.append("")
        lines.append(str(source.get("preview") or ""))
        for gap in item.get("gaps", [])[:6]:
            lines.append(
                f"- {gap['left_generated_id']} -> {gap['right_generated_id']}: "
                f"{gap['suspected_reason']} | tail=`{gap['left_tail']}` head=`{gap['right_head']}`"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "doc_id",
        "deterministic_missing_source_count",
        "learned_missing_source_count",
        "residual_missing_source_count",
        "fixed_by_learned_count",
        "newly_missing_after_learned_count",
        "residual_gap_count",
        "model_added_owner_merge_count",
        "generated_tex_changed",
        "top_residual_reason",
        "top_overlay_rejection_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_sort_key(block_id: str) -> tuple[int, str]:
    match = re.search(r"p(\d+)$", block_id)
    if match:
        return (int(match.group(1)), block_id)
    return (10**9, block_id)


def safe_index_gap(left: Any, right: Any) -> int | None:
    if isinstance(left, int) and isinstance(right, int):
        return right - left
    return None


def tail_text(text: str, *, words: int = 16) -> str:
    tokens = text.strip().split()
    return " ".join(tokens[-words:])[:240]


def head_text(text: str, *, words: int = 16) -> str:
    tokens = text.strip().split()
    return " ".join(tokens[:words])[:240]


def is_open_ended(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("-"):
        return True
    if stripped[-1:] in ",;:([{":
        return True
    if stripped[-1:] in ".!?。！？":
        return False
    if OPEN_TAIL_RE.search(stripped):
        return True
    # A long tail without sentence-ending punctuation is usually open-ended.
    return bool(re.search(r"[A-Za-z0-9)]$", stripped))


def is_continuation_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(LOWER_HEAD_RE.search(stripped) or PAREN_HEAD_RE.search(stripped))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


if __name__ == "__main__":
    raise SystemExit(main())
