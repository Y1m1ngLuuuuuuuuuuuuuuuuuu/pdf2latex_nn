#!/usr/bin/env python3
"""Run v8 AlgorithmRegion candidate extraction Phase 0 on selected200.

This audit generates sidecars only. It does not mutate v8 facts, RenderTreeIR,
renderer behavior, graph schema, or production defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reasoning.algorithm_region_detector import compact, detect_algorithm_candidates, normalize_text


DEFAULT_BASELINE_AUDIT = Path("data/09_eval_reports/algorithm_region_20260526/selected200_baseline_audit")
DEFAULT_SELECTED200_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/algorithm_region_20260526/candidate_extraction_phase0")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-audit-dir", type=Path, default=DEFAULT_BASELINE_AUDIT)
    parser.add_argument("--selected200-root", type=Path, default=DEFAULT_SELECTED200_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-ids", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def load_json(path: Path | None, default: Any = None) -> Any:
    if path is None or not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_doc_dirs(root: Path) -> dict[str, Path]:
    docs: dict[str, Path] = {}
    if not root.exists():
        return docs
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        doc_id = path.name.split("_", 1)[-1]
        if (path / "document_ir.json").exists() and list(path.glob("*_content_list_v8_contentlist_merge_hint.json")):
            docs[doc_id] = path
    return docs


def read_baseline_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    csv_path = path / "algorithm_region_baseline_summary.csv"
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[str(row.get("doc_id") or "")] = row
    return rows


def as_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def get_block_type(block: dict[str, Any] | None) -> str:
    if not block:
        return ""
    return str(block.get("block_type") or block.get("type") or block.get("role") or "").lower()


def blocks_by_id(blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(block.get("block_id")): block for block in blocks if block.get("block_id")}


def extract_algorithm_blocks(structure: dict[str, Any], source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = structure.get("blocks") or []
    by_id = blocks_by_id(blocks)
    algorithms: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    for block in blocks:
        if get_block_type(block) == "algorithm":
            parent = by_id.get(str(block.get("parent_id") or ""))
            parent_type = get_block_type(parent)
            marker = str(block.get("marker") or "").casefold()
            if parent_type != "algorithm" or marker in {"algorithm", "algorithm2e", "lstlisting", "verbatim"}:
                algorithms.append(
                    {
                        "source": source,
                        "block_id": block.get("block_id"),
                        "text": block.get("text") or "",
                        "normalized_text": normalize_text(block.get("text") or ""),
                        "order": block.get("order"),
                        "marker": block.get("marker"),
                    }
                )
    for block in blocks:
        if get_block_type(block) != "caption":
            continue
        parent = by_id.get(str(block.get("parent_id") or block.get("label") or ""))
        text = block.get("text") or ""
        if get_block_type(parent) == "algorithm" or normalize_text(text).startswith(("algorithm ", "alg ", "procedure ", "pseudocode ")):
            captions.append(
                {
                    "source": source,
                    "caption_id": block.get("block_id"),
                    "text": text,
                    "normalized_text": normalize_text(text),
                    "order": block.get("order"),
                    "parent_id": block.get("parent_id") or block.get("label"),
                }
            )
    return algorithms, captions


def match_texts(left: list[dict[str, Any]], right: list[dict[str, Any]], *, threshold: float) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    pairs: list[tuple[float, int, int]] = []
    for i, item in enumerate(left):
        lt = item.get("normalized_text") or normalize_text(item.get("text") or "")
        if not lt:
            continue
        for j, other in enumerate(right):
            rt = other.get("normalized_text") or normalize_text(other.get("text") or other.get("text_preview") or "")
            if not rt:
                continue
            score = SequenceMatcher(None, lt, rt).ratio()
            if lt in rt or rt in lt:
                score = max(score, 0.82)
            if score >= threshold:
                pairs.append((score, i, j))
    pairs.sort(reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, i, j in pairs:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matches.append({"left_index": i, "right_index": j, "score": round(score, 4)})
    return matches, used_left, used_right


def major_failure_after(row: dict[str, Any]) -> str:
    if as_int(row["no_v8_candidate_match_count_after"]):
        return "NO_V8_ALGORITHM_CANDIDATE"
    if as_int(row["new_algorithm_caption_candidate_count"]) and not as_int(row["new_algorithm_body_candidate_count"]):
        return "CAPTION_EXISTS_BODY_MISSING"
    if as_int(row["new_algorithm_body_candidate_count"]) and not as_int(row["new_algorithm_caption_candidate_count"]):
        return "BODY_EXISTS_CAPTION_MISSING"
    if as_int(row["compile_risk_pseudocode_count_after"]):
        return "COMPILE_RISK_PSEUDOCODE"
    if as_int(row["candidate_exists_but_not_rendered_count_after"]):
        return "CANDIDATE_EXISTS_BUT_NOT_RENDERED"
    return "NONE"


def audit_doc(args: tuple[str, str, str, str, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    doc_id, doc_dir_s, baseline_dir_s, output_dir_s, max_examples = args
    doc_dir = Path(doc_dir_s)
    output_dir = Path(output_dir_s)
    baseline_dir = Path(baseline_dir_s)

    content_paths = sorted(doc_dir.glob("*_content_list_v8_contentlist_merge_hint.json"))
    content_payload = load_json(content_paths[0], {}) if content_paths else {}
    document_ir = load_json(doc_dir / "document_ir.json", {})
    gold_structure = load_json(doc_dir / "gold_structure.json", {})
    pred_structure = load_json(doc_dir / "generated_structure.json", {})
    metrics = load_json(doc_dir / "structure_metrics.json", {})

    detected = detect_algorithm_candidates(content_payload, document_ir, doc_id=doc_id)
    regions = detected["algorithm_region_candidates"]
    caption_candidates = detected["algorithm_caption_candidates"]
    body_candidates = detected["algorithm_body_candidates"]
    body_match_candidates = [
        candidate
        for candidate in body_candidates
        if candidate.get("candidate_type") != "ALGORITHM_AS_PARAGRAPH"
    ]
    risks = detected["pseudocode_compile_risk"]

    gold_algorithms, gold_captions = extract_algorithm_blocks(gold_structure, "gold")
    pred_algorithms, pred_captions = extract_algorithm_blocks(pred_structure, "pred")
    caption_matches, used_gold_caption, _ = match_texts(gold_captions, caption_candidates, threshold=0.62)
    body_matches, used_gold_body, _ = match_texts(gold_algorithms, body_match_candidates, threshold=0.43)
    pred_caption_matches, used_gold_pred_caption, _ = match_texts(gold_captions, pred_captions, threshold=0.68)
    pred_body_matches, used_gold_pred_body, _ = match_texts(gold_algorithms, pred_algorithms, threshold=0.45)

    baseline_audit = load_json(baseline_dir / f"algorithm_region_audit_{doc_id}.json", {})
    baseline_candidates = load_json(baseline_dir / f"algorithm_candidates_{doc_id}.json", {})

    safe_id = doc_id.replace("/", "_")
    write_json(output_dir / f"algorithm_region_candidates_{safe_id}.json", {"schema_version": "algorithm_region_candidates_phase0_v1", "doc_id": doc_id, "algorithm_region_candidates": regions})
    write_json(output_dir / f"algorithm_caption_candidates_{safe_id}.json", {"schema_version": "algorithm_caption_candidates_phase0_v1", "doc_id": doc_id, "algorithm_caption_candidates": caption_candidates})
    write_json(output_dir / f"algorithm_body_candidates_{safe_id}.json", {"schema_version": "algorithm_body_candidates_phase0_v1", "doc_id": doc_id, "algorithm_body_candidates": body_candidates})
    write_json(output_dir / f"pseudocode_compile_risk_{safe_id}.json", {"schema_version": "pseudocode_compile_risk_phase0_v1", "doc_id": doc_id, "pseudocode_compile_risk": risks})

    unmatched_gold_caption_after = [gold_captions[i] for i in range(len(gold_captions)) if i not in used_gold_caption]
    unmatched_gold_body_after = [gold_algorithms[i] for i in range(len(gold_algorithms)) if i not in used_gold_body and (gold_algorithms[i].get("normalized_text") or "").strip()]
    unmatched_gold_caption_pred = [gold_captions[i] for i in range(len(gold_captions)) if i not in used_gold_pred_caption]
    unmatched_gold_body_pred = [gold_algorithms[i] for i in range(len(gold_algorithms)) if i not in used_gold_pred_body and (gold_algorithms[i].get("normalized_text") or "").strip()]
    false_refs = [candidate for candidate in detected["all_candidates"] if candidate.get("candidate_type") == "FALSE_ALGORITHM_REFERENCE"]
    as_para = [candidate for candidate in body_candidates if candidate.get("candidate_type") == "ALGORITHM_AS_PARAGRAPH"]
    as_table = [candidate for candidate in body_candidates if candidate.get("candidate_type") == "ALGORITHM_AS_TABLE_LIKE"]

    metric_value = metrics.get("float_caption_attachment_accuracy") or {}
    row = {
        "doc_id": doc_id,
        "gold_algorithm_count": max(len(gold_algorithms), len(gold_captions)),
        "pred_algorithm_count_existing": max(len(pred_algorithms), len(pred_captions)),
        "old_v8_algorithm_candidate_count": len(baseline_candidates.get("algorithm_candidates") or []),
        "new_algorithm_region_candidate_count": len(regions),
        "new_algorithm_caption_candidate_count": len(caption_candidates),
        "new_algorithm_body_candidate_count": len(body_match_candidates),
        "no_v8_candidate_match_count_before": as_int((load_json(baseline_dir / "algorithm_region_baseline_summary.json", {}) or {}).get("unused", 0)),
        "no_v8_candidate_match_count_after": len(unmatched_gold_caption_after) + len(unmatched_gold_body_after),
        "algorithm_caption_missing_count_after": len(unmatched_gold_caption_after),
        "algorithm_body_missing_count_after": len(unmatched_gold_body_after),
        "algorithm_as_paragraph_count_after": len(as_para),
        "algorithm_as_table_like_count_after": len(as_table),
        "compile_risk_pseudocode_count_after": len(risks),
        "candidate_exists_but_not_rendered_count_after": (len(unmatched_gold_caption_pred) + len(unmatched_gold_body_pred)) if regions else 0,
        "false_algorithm_candidate_count_after": len(false_refs),
        "float_caption_attachment_accuracy": _metric_score(metric_value),
        "generated_structure_validity": _metric_score(metrics.get("generated_structure_validity")),
        "macro_structure_score_body": _metric_score(metrics.get("macro_structure_score_body")),
    }
    examples = {
        "recovered_no_v8_candidate": _recovered_examples(doc_id, baseline_audit, regions, max_examples),
        "body_exists_caption_missing": _region_examples(doc_id, [r for r in regions if r.get("failure_hint") == "BODY_EXISTS_CAPTION_MISSING"], max_examples),
        "caption_exists_body_missing": _region_examples(doc_id, [r for r in regions if r.get("failure_hint") == "CAPTION_EXISTS_BODY_MISSING"], max_examples),
        "compile_risk_pseudocode": _risk_examples(doc_id, risks, max_examples),
        "algorithm_as_paragraph": _candidate_examples(doc_id, as_para, max_examples),
        "top_problem_seed": _region_examples(doc_id, regions, max_examples),
        "candidate_caption_matches": caption_matches[:max_examples],
        "candidate_body_matches": body_matches[:max_examples],
    }
    return row, examples


def _metric_score(value: Any) -> float | None:
    if isinstance(value, dict):
        return as_float(value.get("score"))
    return as_float(value)


def _recovered_examples(doc_id: str, baseline_audit: dict[str, Any], regions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    failure_cases = baseline_audit.get("failure_cases") or []
    if not any(case.get("failure_type") == "NO_V8_ALGORITHM_CANDIDATE" for case in failure_cases):
        return []
    return _region_examples(doc_id, regions, limit)


def _region_examples(doc_id: str, regions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows = []
    for region in regions[:limit]:
        rows.append(
            {
                "doc_id": doc_id,
                "page_idx": region.get("page_idx"),
                "source_v8_ids": region.get("source_v8_ids"),
                "bbox": region.get("bbox_union"),
                "candidate_type": region.get("region_type"),
                "confidence": region.get("confidence"),
                "evidence": region.get("evidence"),
                "compile_risk_flags": region.get("compile_risk_flags"),
                "recommended_render_policy": region.get("recommended_render_policy"),
                "text_preview": region.get("text_preview") or "",
            }
        )
    return rows


def _candidate_examples(doc_id: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": doc_id,
            "page_idx": candidate.get("page_idx"),
            "source_v8_ids": candidate.get("source_v8_ids"),
            "bbox": candidate.get("bbox"),
            "candidate_type": candidate.get("candidate_type"),
            "confidence": candidate.get("confidence"),
            "evidence": candidate.get("evidence"),
            "compile_risk_flags": candidate.get("compile_risk_flags"),
            "recommended_render_policy": "diagnostic_only",
            "text_preview": candidate.get("text_preview"),
        }
        for candidate in candidates[:limit]
    ]


def _risk_examples(doc_id: str, risks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": doc_id,
            "page_idx": risk.get("page_idx"),
            "source_v8_ids": risk.get("source_v8_ids"),
            "bbox": None,
            "candidate_type": risk.get("candidate_type"),
            "confidence": None,
            "evidence": risk.get("risk_reasons"),
            "compile_risk_flags": risk.get("risk_reasons"),
            "recommended_render_policy": "verbatim_fallback",
            "text_preview": risk.get("text"),
        }
        for risk in risks[:limit]
    ]


def aggregate(rows: list[dict[str, Any]], baseline_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {"docs": len(rows)}
    sum_keys = [
        "gold_algorithm_count",
        "pred_algorithm_count_existing",
        "old_v8_algorithm_candidate_count",
        "new_algorithm_region_candidate_count",
        "new_algorithm_caption_candidate_count",
        "new_algorithm_body_candidate_count",
        "no_v8_candidate_match_count_before",
        "no_v8_candidate_match_count_after",
        "algorithm_caption_missing_count_before",
        "algorithm_caption_missing_count_after",
        "algorithm_body_missing_count_before",
        "algorithm_body_missing_count_after",
        "algorithm_as_paragraph_count_after",
        "algorithm_as_table_like_count_after",
        "compile_risk_pseudocode_count_after",
        "candidate_exists_but_not_rendered_count_after",
        "false_algorithm_candidate_count_after",
    ]
    for row in rows:
        base = baseline_rows.get(row["doc_id"], {})
        row["no_v8_candidate_match_count_before"] = as_int(base.get("no_v8_candidate_match_count"))
        row["algorithm_caption_missing_count_before"] = as_int(base.get("algorithm_caption_missing_count"))
        row["algorithm_body_missing_count_before"] = as_int(base.get("algorithm_body_missing_count"))
        row["compile_risk_pseudocode_count_before"] = as_int(base.get("pseudocode_compile_risk_count"))
        row["major_failure_type_before"] = base.get("major_failure_type") or "UNKNOWN"
        row["major_failure_type_after"] = major_failure_after(row)
    for key in sum_keys + ["compile_risk_pseudocode_count_before"]:
        total[key] = sum(as_int(row.get(key)) for row in rows)
    total["failure_type_counts_before"] = dict(Counter(row.get("major_failure_type_before") for row in rows))
    total["failure_type_counts_after"] = dict(Counter(row.get("major_failure_type_after") for row in rows))
    for key in ("float_caption_attachment_accuracy", "generated_structure_validity", "macro_structure_score_body"):
        values = [as_float(row.get(key)) for row in rows if as_float(row.get(key)) is not None]
        total[f"mean_{key}"] = sum(values) / len(values) if values else None
    return total


def merge_examples(target: dict[str, list[dict[str, Any]]], source: dict[str, Any], limit: int) -> None:
    for key, values in source.items():
        if not isinstance(values, list):
            continue
        bucket = target.setdefault(key, [])
        for value in values:
            if len(bucket) >= limit:
                break
            bucket.append(value)


def top_problem_docs(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        score = (
            as_int(row.get("no_v8_candidate_match_count_after"))
            + as_int(row.get("algorithm_caption_missing_count_after"))
            + as_int(row.get("algorithm_body_missing_count_after"))
            + as_int(row.get("compile_risk_pseudocode_count_after"))
            + as_int(row.get("algorithm_as_paragraph_count_after"))
        )
        if score:
            scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1].get("doc_id") or ""))
    return [row for _, row in scored[:limit]]


def decide(summary: dict[str, Any]) -> str:
    no_after = as_int(summary.get("no_v8_candidate_match_count_after"))
    no_before = as_int(summary.get("no_v8_candidate_match_count_before"))
    new_regions = as_int(summary.get("new_algorithm_region_candidate_count"))
    gold = as_int(summary.get("gold_algorithm_count"))
    false_count = as_int(summary.get("false_algorithm_candidate_count_after"))
    risk = as_int(summary.get("compile_risk_pseudocode_count_after"))
    if new_regions == 0:
        return "diagnostic_only"
    if no_after <= max(10, int(no_before * 0.55)) and new_regions >= max(20, int(gold * 0.8)) and false_count <= max(5, int(new_regions * 0.08)):
        return "ready_for_algorithm_region_phase0_renderer"
    if no_after >= max(20, int(no_before * 0.75)):
        return "need_lower_level_roi_or_model"
    if risk > new_regions:
        return "diagnostic_only"
    return "ready_for_algorithm_region_phase0_renderer"


def write_manual_review_pack(output_dir: Path, examples: dict[str, list[dict[str, Any]]], problem_docs: list[dict[str, Any]]) -> None:
    pack = {
        "schema_version": "algorithm_region_candidate_review_pack_v1",
        "recovered_no_v8_candidate": examples.get("recovered_no_v8_candidate", [])[:20],
        "body_exists_caption_missing": examples.get("body_exists_caption_missing", [])[:20],
        "caption_exists_body_missing": examples.get("caption_exists_body_missing", [])[:20],
        "compile_risk_pseudocode": examples.get("compile_risk_pseudocode", [])[:20],
        "algorithm_as_paragraph": examples.get("algorithm_as_paragraph", [])[:20],
        "top_problem_docs": problem_docs[:20],
    }
    write_json(output_dir / "manual_review_pack.json", pack)
    lines = ["# AlgorithmRegion Candidate Extraction Manual Review Pack", ""]
    for key, values in pack.items():
        if key == "schema_version":
            continue
        lines.append(f"## {key}")
        lines.append("")
        if not values:
            lines.append("- none")
        else:
            for item in values[:20]:
                if key == "top_problem_docs":
                    lines.append(f"- `{item.get('doc_id')}` after_no_v8={item.get('no_v8_candidate_match_count_after')} regions={item.get('new_algorithm_region_candidate_count')} failure={item.get('major_failure_type_after')}")
                else:
                    lines.append(f"- `{item.get('doc_id')}` page={item.get('page_idx')} type={item.get('candidate_type')} conf={item.get('confidence')} policy={item.get('recommended_render_policy')} text={compact(item.get('text_preview'), 180)}")
        lines.append("")
    (output_dir / "manual_review_pack.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]], examples: dict[str, list[dict[str, Any]]], decision: str) -> None:
    problem_docs = top_problem_docs(rows, 20)

    def line(values: list[Any]) -> str:
        return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"

    lines: list[str] = []
    lines.append("# AlgorithmRegion Candidate Extraction Phase 0 Report")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- docs analyzed: {summary.get('docs')}")
    lines.append("- no training / no MinerU / no relabel / no rebuild / no GNN")
    lines.append("- no renderer changes")
    lines.append("- v8 facts used: `*_content_list_v8_contentlist_merge_hint.json` + `document_ir.json`")
    lines.append("- no fallback to old v7; legacy names such as `source_v7_ids` / `v7_id` are provenance names only")
    lines.append("")
    lines.append("Current mainline remains:")
    lines.append("")
    lines.append("```text")
    lines.append("v8 full observable facts")
    lines.append("  -> v8 atomic/reflow")
    lines.append("  -> deterministic merge + contentlist merge hint")
    lines.append("  -> RenderTreeIR")
    lines.append("  -> IR renderer")
    lines.append("```")
    lines.append("")
    lines.append("## Baseline Recap")
    lines.append("")
    lines.append(line(["Metric", "Baseline"]))
    lines.append(line(["---", "---:"]))
    for key in [
        "gold_algorithm_count",
        "pred_algorithm_count_existing",
        "old_v8_algorithm_candidate_count",
        "algorithm_caption_missing_count_before",
        "algorithm_body_missing_count_before",
        "no_v8_candidate_match_count_before",
        "compile_risk_pseudocode_count_before",
    ]:
        lines.append(line([key, summary.get(key)]))
    lines.append("")
    lines.append("## Candidate Extraction Summary")
    lines.append("")
    lines.append(line(["Metric", "Before", "After", "Delta"]))
    lines.append(line(["---", "---:", "---:", "---:"]))
    comparisons = [
        ("algorithm candidate count", "old_v8_algorithm_candidate_count", "new_algorithm_region_candidate_count"),
        ("caption candidate count", None, "new_algorithm_caption_candidate_count"),
        ("body candidate count", None, "new_algorithm_body_candidate_count"),
        ("no_v8_candidate_match", "no_v8_candidate_match_count_before", "no_v8_candidate_match_count_after"),
        ("caption missing proxy", "algorithm_caption_missing_count_before", "algorithm_caption_missing_count_after"),
        ("body missing proxy", "algorithm_body_missing_count_before", "algorithm_body_missing_count_after"),
        ("compile risk count", "compile_risk_pseudocode_count_before", "compile_risk_pseudocode_count_after"),
    ]
    for label, before_key, after_key in comparisons:
        before = summary.get(before_key) if before_key else "n/a"
        after = summary.get(after_key)
        delta = "n/a" if before_key is None else as_int(after) - as_int(before)
        lines.append(line([label, before, after, delta]))
    lines.append("")
    lines.append("## Failure Breakdown")
    lines.append("")
    lines.append(line(["failure_type", "before_docs", "after_docs"]))
    lines.append(line(["---", "---:", "---:"]))
    keys = sorted(set(summary.get("failure_type_counts_before", {})) | set(summary.get("failure_type_counts_after", {})))
    for key in keys:
        lines.append(line([key, summary.get("failure_type_counts_before", {}).get(key, 0), summary.get("failure_type_counts_after", {}).get(key, 0)]))
    lines.append("")
    lines.append("## Top Problem Docs")
    lines.append("")
    lines.append(line(["doc_id", "gold_alg", "new_regions", "after_no_v8", "caption_proxy", "body_proxy", "risk", "major_failure_after"]))
    lines.append(line(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---"]))
    for row in problem_docs:
        lines.append(line([row.get("doc_id"), row.get("gold_algorithm_count"), row.get("new_algorithm_region_candidate_count"), row.get("no_v8_candidate_match_count_after"), row.get("algorithm_caption_missing_count_after"), row.get("algorithm_body_missing_count_after"), row.get("compile_risk_pseudocode_count_after"), row.get("major_failure_type_after")]))
    lines.append("")
    lines.append("## Manual Review Examples")
    lines.append("")
    for key in [
        "recovered_no_v8_candidate",
        "body_exists_caption_missing",
        "caption_exists_body_missing",
        "compile_risk_pseudocode",
        "algorithm_as_paragraph",
    ]:
        lines.append(f"### {key}")
        bucket = examples.get(key, [])[:10]
        if not bucket:
            lines.append("- none")
        else:
            for item in bucket:
                lines.append(f"- `{item.get('doc_id')}` page={item.get('page_idx')} type={item.get('candidate_type')} conf={item.get('confidence')} policy={item.get('recommended_render_policy')} text={compact(item.get('text_preview'), 220)}")
        lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    before_no = as_int(summary.get("no_v8_candidate_match_count_before"))
    after_no = as_int(summary.get("no_v8_candidate_match_count_after"))
    reduction = before_no - after_no
    lines.append(f"1. New detector reduced NO_V8_ALGORITHM_CANDIDATE proxy from {before_no} to {after_no} (delta {reduction}).")
    lines.append(f"2. Caption proxy after extraction: {summary.get('algorithm_caption_missing_count_after')}; body proxy after extraction: {summary.get('algorithm_body_missing_count_after')}.")
    lines.append(f"3. Pseudocode compile-risk candidates after extraction: {summary.get('compile_risk_pseudocode_count_after')}.")
    if after_no > 0:
        lines.append("4. Some remaining misses are still candidate/fact-layer misses, so renderer work should be gated on review of these candidates.")
    else:
        lines.append("4. Candidate coverage is now sufficient enough to consider grouping/rendering validation.")
    lines.append("5. Recommended render policy remains conservative: compile-risk pseudocode should prefer verbatim/plain-text or crop fallback; algorithm environment should be later, not default.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"**{decision}**")
    if decision == "ready_for_algorithm_region_phase0_renderer":
        lines.append("")
        lines.append("New detector coverage is sufficient for an AlgorithmRegion Phase 0 renderer/materialization experiment, with conservative fallback policies.")
    elif decision == "need_lower_level_roi_or_model":
        lines.append("")
        lines.append("Candidate misses remain too high; next work should be ROI role classifier / lower-level extraction, not renderer.")
    else:
        lines.append("")
        lines.append("Candidate noise or compile risk is too high; keep diagnostic-only.")
    (output_dir / "ALGORITHM_REGION_CANDIDATE_EXTRACTION_PHASE0_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def readiness_report(output_dir: Path, missing: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ALGORITHM_REGION_CANDIDATE_EXTRACTION_PHASE0_REPORT.md").write_text(
        "# AlgorithmRegion Candidate Extraction Phase 0 Readiness Report\n\n"
        "Required artifacts were missing, so the pass stopped without guessing.\n\n"
        + "\n".join(f"- {item}" for item in missing)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = read_baseline_rows(args.baseline_audit_dir)
    doc_dirs = collect_doc_dirs(args.selected200_root)
    missing = []
    if not baseline_rows:
        missing.append(str(args.baseline_audit_dir / "algorithm_region_baseline_summary.csv"))
    if not doc_dirs:
        missing.append(str(args.selected200_root))
    if missing:
        readiness_report(args.output_dir, missing)
        return 2

    if args.doc_ids:
        selected_ids = [doc_id for doc_id in args.doc_ids if doc_id in doc_dirs]
    else:
        selected_ids = [doc_id for doc_id in baseline_rows if doc_id in doc_dirs]
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]

    tasks = [
        (doc_id, str(doc_dirs[doc_id]), str(args.baseline_audit_dir), str(args.output_dir), args.max_examples)
        for doc_id in selected_ids
    ]
    rows: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row, doc_examples in pool.map(audit_doc, tasks):
                rows.append(row)
                merge_examples(examples, doc_examples, args.max_examples)
    else:
        for task in tasks:
            row, doc_examples = audit_doc(task)
            rows.append(row)
            merge_examples(examples, doc_examples, args.max_examples)

    summary = aggregate(rows, baseline_rows)
    decision = decide(summary)
    problem_docs = top_problem_docs(rows, 20)
    write_csv(args.output_dir / "algorithm_region_candidate_extraction_summary.csv", rows)
    write_json(
        args.output_dir / "algorithm_region_candidate_extraction_summary.json",
        {
            "schema_version": "algorithm_region_candidate_extraction_phase0_summary_v1",
            "selected200_root": str(args.selected200_root),
            "baseline_audit_dir": str(args.baseline_audit_dir),
            "output_dir": str(args.output_dir),
            "summary": summary,
            "decision": decision,
            "top_problem_docs": problem_docs,
            "v8_only_confirmation": {
                "current_fact_layer": "v8 full observable facts",
                "no_fallback_to_old_v7": True,
                "legacy_names_are_provenance_only": True,
            },
        },
    )
    write_manual_review_pack(args.output_dir, examples, problem_docs)
    write_report(args.output_dir, summary, rows, examples, decision)
    print(json.dumps({"docs": len(rows), "output_dir": str(args.output_dir), "decision": decision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
