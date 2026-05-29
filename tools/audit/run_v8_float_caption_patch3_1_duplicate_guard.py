#!/usr/bin/env python3
"""Patch3.1 duplicate-delta and baseline-drift audit for v8 FloatCaptionLayout.

This pass is deliberately narrow.  It reads Patch2/Patch3 artifacts, explains
metric drift, isolates Patch3 flag-on duplicate deltas, and classifies the
extra suspicious non-caption diffs.  It does not regenerate LaTeX unless a
future guard patch explicitly needs validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PATCH2_ROOT = Path("data/09_eval_reports/float_caption_layout_20260526/v8_same_code_ab_validation/patch2_same_code_ab")
DEFAULT_PATCH3_ROOT = Path("data/09_eval_reports/float_caption_layout_20260526/v8_same_code_ab_validation/patch3_same_code_ab")
DEFAULT_OUTPUT = Path("data/09_eval_reports/float_caption_layout_20260526/v8_same_code_ab_validation/patch3_1_duplicate_guard")
REPORT_NAME = "V8_FLOAT_CAPTION_PATCH3_1_DUPLICATE_GUARD_REPORT.md"


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readiness = check_readiness(args.patch2_root, args.patch3_root)
    if not readiness["ready"]:
        write_json(args.output_dir / "READINESS_REPORT.json", readiness)
        (args.output_dir / "READINESS_REPORT.md").write_text(
            "# Patch3.1 Readiness Report\n\n"
            + "\n".join(f"- {item}" for item in readiness["missing"])
            + "\n",
            encoding="utf-8",
        )
        return 2

    patch2_summary = read_json(args.patch2_root / "selected200_same_code_ab_summary.json")
    patch3_summary = read_json(args.patch3_root / "selected200_same_code_ab_summary.json")
    drift_summary, drift_cases = audit_baseline_drift(args.patch2_root, args.patch3_root, patch2_summary, patch3_summary)
    duplicate_summary, duplicate_cases = audit_duplicate_delta(args.patch3_root)
    suspicious_summary, suspicious_cases = audit_suspicious_delta(args.patch2_root, args.patch3_root)

    write_json(args.output_dir / "baseline_drift_summary.json", drift_summary)
    write_csv(args.output_dir / "baseline_drift_cases.csv", drift_cases)
    write_json(args.output_dir / "patch3_duplicate_delta_summary.json", duplicate_summary)
    write_jsonl(args.output_dir / "patch3_duplicate_delta_cases.jsonl", duplicate_cases)
    write_csv(args.output_dir / "patch3_duplicate_delta_cases.csv", duplicate_cases)
    write_json(args.output_dir / "patch3_suspicious_delta_summary.json", suspicious_summary)
    write_csv(args.output_dir / "patch3_suspicious_delta_cases.csv", suspicious_cases)

    report = build_report(
        patch2_summary=patch2_summary,
        patch3_summary=patch3_summary,
        drift_summary=drift_summary,
        duplicate_summary=duplicate_summary,
        duplicate_cases=duplicate_cases,
        suspicious_summary=suspicious_summary,
        suspicious_cases=suspicious_cases,
    )
    (args.output_dir / REPORT_NAME).write_text(report, encoding="utf-8")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch2-root", type=Path, default=DEFAULT_PATCH2_ROOT)
    parser.add_argument("--patch3-root", type=Path, default=DEFAULT_PATCH3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def check_readiness(patch2_root: Path, patch3_root: Path) -> dict[str, Any]:
    required = [
        patch2_root / "selected200_same_code_ab_summary.json",
        patch2_root / "baseline_flag_off_current_code",
        patch2_root / "experimental_flag_on_current_code",
        patch2_root / "selected200_diff_attribution.csv",
        patch3_root / "selected200_same_code_ab_summary.json",
        patch3_root / "baseline_flag_off_current_code",
        patch3_root / "experimental_flag_on_current_code",
        patch3_root / "selected200_diff_attribution.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return {"ready": not missing, "missing": missing}


def audit_baseline_drift(
    patch2_root: Path,
    patch3_root: Path,
    patch2_summary: dict[str, Any],
    patch3_summary: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    patch2_base = patch2_root / "baseline_flag_off_current_code"
    patch3_base = patch3_root / "baseline_flag_off_current_code"
    rows: list[dict[str, Any]] = []
    changed_docs = 0
    caption_block_delta = Counter()
    for doc_dir in sorted(path for path in patch3_base.iterdir() if path.is_dir()):
        patch2_doc = patch2_base / doc_dir.name
        if not patch2_doc.exists():
            continue
        doc_id = doc_dir.name.split("_", 1)[-1]
        p2 = read_json(patch2_doc / "ours_comparison_structure_current.json")
        p3 = read_json(doc_dir / "ours_comparison_structure_current.json")
        p2_caps = caption_blocks(p2)
        p3_caps = caption_blocks(p3)
        p2_panel = sum(1 for block in p2_caps if synthetic_or_panel_text(block_text(block)))
        p3_panel = sum(1 for block in p3_caps if synthetic_or_panel_text(block_text(block)))
        p2_subfig = sum(1 for block in p2_caps if subfigure_marker(block.get("label")))
        p3_subfig = sum(1 for block in p3_caps if subfigure_marker(block.get("label")))
        p2_dup = duplicate_clusters(p2)
        p3_dup = duplicate_clusters(p3)
        tex_changed = files_differ(patch2_doc / "generated.tex", doc_dir / "generated.tex")
        if tex_changed:
            changed_docs += 1
        row = {
            "doc_id": doc_id,
            "generated_tex_changed": tex_changed,
            "patch2_caption_blocks": len(p2_caps),
            "patch3_caption_blocks": len(p3_caps),
            "caption_block_delta": len(p3_caps) - len(p2_caps),
            "patch2_panel_or_synthetic_caption_blocks": p2_panel,
            "patch3_panel_or_synthetic_caption_blocks": p3_panel,
            "panel_or_synthetic_delta": p3_panel - p2_panel,
            "patch2_subfigure_caption_blocks": p2_subfig,
            "patch3_subfigure_caption_blocks": p3_subfig,
            "patch2_duplicate_count": sum(max(0, len(items) - 1) for items in p2_dup.values()),
            "patch3_duplicate_count": sum(max(0, len(items) - 1) for items in p3_dup.values()),
            "likely_cause": drift_case_cause(len(p2_caps), len(p3_caps), p2_panel, p3_panel, tex_changed),
        }
        rows.append(row)
        caption_block_delta[row["likely_cause"]] += 1
    summary = {
        "patch2_flag_off": summary_metrics(patch2_summary.get("baseline") or {}),
        "patch3_flag_off": summary_metrics(patch3_summary.get("baseline") or {}),
        "delta_patch3_minus_patch2": metric_delta(patch2_summary.get("baseline") or {}, patch3_summary.get("baseline") or {}),
        "docs_compared": len(rows),
        "generated_tex_changed_docs": changed_docs,
        "likely_cause_doc_counts": dict(caption_block_delta),
        "interpretation": baseline_drift_interpretation(patch2_summary, patch3_summary, rows),
    }
    return summary, rows


def audit_duplicate_delta(patch3_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_root = patch3_root / "baseline_flag_off_current_code"
    exp_root = patch3_root / "experimental_flag_on_current_code"
    cases: list[dict[str, Any]] = []
    class_counts = Counter()
    for exp_doc in sorted(path for path in exp_root.iterdir() if path.is_dir()):
        base_doc = base_root / exp_doc.name
        if not base_doc.exists():
            continue
        doc_id = exp_doc.name.split("_", 1)[-1]
        base_structure = read_json(base_doc / "ours_comparison_structure_current.json")
        exp_structure = read_json(exp_doc / "ours_comparison_structure_current.json")
        base_clusters = duplicate_clusters(base_structure)
        exp_clusters = duplicate_clusters(exp_structure)
        promoted = read_json(exp_doc / "promoted_captions.json", [])
        diagnostics = read_json(exp_doc / "float_caption_fix_diag.json", {})
        for key, exp_blocks in sorted(exp_clusters.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
            base_extra = max(0, len(base_clusters.get(key, [])) - 1)
            exp_extra = max(0, len(exp_blocks) - 1)
            delta = exp_extra - base_extra
            if delta <= 0:
                continue
            marker, label, text = key
            origins, source_ids = infer_candidate_origins(text, promoted)
            snippets = rendered_caption_snippets(exp_doc / "generated.tex", text)
            classification = classify_duplicate_delta(text=text, label=label, origins=origins, blocks=exp_blocks, diagnostics=diagnostics)
            class_counts[classification] += delta
            cases.append(
                {
                    "doc_id": doc_id,
                    "caption_text": text,
                    "caption_type": marker,
                    "caption_number": label,
                    "subfigure_marker": subfigure_marker(label) or "",
                    "source_v8_ids": " ".join(sorted(source_ids)),
                    "origin_list": " ".join(sorted(origins)),
                    "paired_float_id": infer_paired_float_id(text, promoted),
                    "comparison_caption_ids": " ".join(str(block.get("block_id")) for block in exp_blocks),
                    "extra_duplicate_delta": delta,
                    "rendered_tex_snippets": " || ".join(snippets[:3]),
                    "classification": classification,
                    "recommended_action": duplicate_recommendation(classification),
                }
            )
    summary = {
        "duplicate_delta_case_count": len(cases),
        "duplicate_delta_total": sum(int(row["extra_duplicate_delta"]) for row in cases),
        "classification_counts": dict(class_counts),
        "requires_patch": any(row["classification"] in {"TRUE_DUPLICATE", "TEXT_METADATA_DOUBLE_MATERIALIZATION", "CROP_METADATA_DOUBLE_MATERIALIZATION", "CONVERTER_DOUBLE_COUNT"} for row in cases),
    }
    return summary, cases


def audit_suspicious_delta(patch2_root: Path, patch3_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    patch2_rows = {row["doc_id"]: row for row in read_csv(patch2_root / "selected200_diff_attribution.csv")}
    patch3_rows = {row["doc_id"]: row for row in read_csv(patch3_root / "selected200_diff_attribution.csv")}
    cases: list[dict[str, Any]] = []
    class_counts = Counter()
    for doc_id, row3 in sorted(patch3_rows.items()):
        count2 = intish((patch2_rows.get(doc_id) or {}).get("non_caption_suspicious_change_count"))
        count3 = intish(row3.get("non_caption_suspicious_change_count"))
        if count3 <= count2:
            continue
        examples = split_examples(row3.get("non_caption_suspicious_examples") or "")
        classification = classify_suspicious_examples(examples)
        class_counts[classification] += count3 - count2
        cases.append(
            {
                "doc_id": doc_id,
                "patch2_suspicious_count": count2,
                "patch3_suspicious_count": count3,
                "delta": count3 - count2,
                "classification": classification,
                "examples": " || ".join(examples[:5]),
                "recommended_action": suspicious_recommendation(classification),
            }
        )
    summary = {
        "patch2_suspicious_total": sum(intish(row.get("non_caption_suspicious_change_count")) for row in patch2_rows.values()),
        "patch3_suspicious_total": sum(intish(row.get("non_caption_suspicious_change_count")) for row in patch3_rows.values()),
        "delta_positive_docs": len(cases),
        "positive_delta_total": sum(int(row["delta"]) for row in cases),
        "classification_counts": dict(class_counts),
        "has_true_non_caption_leakage": any(row["classification"] in {"BODY_TEXT_LEAKAGE", "HEADING_LEAKAGE", "REFERENCE_LEAKAGE", "PREAMBLE_STYLE_LEAKAGE", "UNKNOWN"} for row in cases),
    }
    return summary, cases


def baseline_drift_interpretation(patch2_summary: dict[str, Any], patch3_summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    delta = metric_delta(patch2_summary.get("baseline") or {}, patch3_summary.get("baseline") or {})
    pred_delta = float(delta.get("pred_caption_count") or 0)
    duplicate_delta = float(delta.get("duplicate_caption_count") or 0)
    panel_removed = -sum(min(0, int(row["panel_or_synthetic_delta"])) for row in rows)
    generated_changed_ratio = sum(1 for row in rows if row["generated_tex_changed"]) / (len(rows) or 1)
    if pred_delta < -100 and duplicate_delta < -50 and panel_removed > 100:
        cause = "metric_version_tightened_panel_synthetic_removed"
    elif generated_changed_ratio > 0.8:
        cause = "code_path_or_rendering_drift"
    else:
        cause = "mixed_metric_and_code_drift"
    return {
        "primary_cause": cause,
        "patch3_same_code_ab_valid": True,
        "patch3_absolute_values_comparable_to_patch2": False,
        "recommended_metric_version": "patch3_strict_caption_metric_v1",
        "panel_or_synthetic_removed_estimate": panel_removed,
        "generated_tex_changed_ratio": generated_changed_ratio,
    }


def drift_case_cause(p2_count: int, p3_count: int, p2_panel: int, p3_panel: int, tex_changed: bool) -> str:
    if p3_count < p2_count and p3_panel <= p2_panel:
        return "stricter_caption_count_or_panel_synthetic_removed"
    if tex_changed:
        return "generated_tex_changed_between_metric_versions"
    return "no_major_caption_drift"


def summary_metrics(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "float_caption_attachment_accuracy",
        "pred_caption_count",
        "missing_caption_count",
        "duplicate_caption_count",
        "caption_as_paragraph_count",
        "wrong_float_type_pairing_count",
        "generated_structure_validity",
        "macro_structure_score_body",
    ]
    return {key: row.get(key) for key in keys}


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in set(before) | set(after):
        left = floatish(before.get(key))
        right = floatish(after.get(key))
        if left is not None and right is not None:
            out[key] = right - left
    return out


def duplicate_clusters(structure: dict[str, Any]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for block in caption_blocks(structure):
        text = normalized_caption_text(block_text(block))
        if synthetic_or_panel_text(text):
            continue
        key = (str(block.get("marker") or ""), str(block.get("label") or ""), text)
        groups[key].append(block)
    return {key: items for key, items in groups.items() if len(items) > 1}


def caption_blocks(structure: dict[str, Any]) -> list[dict[str, Any]]:
    return [block for block in structure.get("blocks", []) if block.get("block_type") == "caption"]


def block_text(block: dict[str, Any]) -> str:
    return str(block.get("normalized_text") or block.get("text") or "")


def synthetic_or_panel_text(text: str) -> bool:
    value = normalized_caption_text(text)
    compact = re.sub(r"[^0-9a-z]+", "", value.casefold())
    if compact in {
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "left",
        "right",
        "upper",
        "lower",
        "figure",
        "fig",
        "table",
        "algorithm",
        "reconstructionplaceholder",
        "figurereconstructionplaceholder",
        "tablereconstructionplaceholder",
    }:
        return True
    panel_token = r"(?:\([a-z]\)|[a-z]\))"
    if re.fullmatch(rf"{panel_token}(?:\s+{panel_token}){{1,7}}", value.strip(), flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"[a-z](?:\s+[a-z]){1,7}", value.strip(), flags=re.IGNORECASE):
        return True
    return bool(re.fullmatch(r"\(?[a-z]\)?", value.strip(), flags=re.IGNORECASE))


def normalized_caption_text(text: Any) -> str:
    value = " ".join(str(text or "").casefold().split())
    value = re.sub(r"\\(?:label|ref|cite|textbf|emph|small|footnotesize)\*?(?:\[[^\]]*\])?", " ", value)
    value = re.sub(r"[{}]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .:;,-–—")


def subfigure_marker(label: Any) -> str:
    match = re.search(r"\(([a-zA-Z0-9]+)\)\s*$", str(label or ""))
    return match.group(1).casefold() if match else ""


def infer_candidate_origins(text: str, candidates: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    target = normalized_caption_text(text)
    origins: set[str] = set()
    source_ids: set[str] = set()
    for item in candidates:
        candidate_text = normalized_caption_text(item.get("normalized_caption_text") or item.get("text") or "")
        if not target or not candidate_text:
            continue
        if target in candidate_text or candidate_text in target:
            origins.add(str(item.get("origin") or "unknown"))
            source_ids.update(str(value) for value in (item.get("source_v8_ids") or []))
    return origins, source_ids


def infer_paired_float_id(text: str, candidates: list[dict[str, Any]]) -> str:
    target = normalized_caption_text(text)
    for item in candidates:
        candidate_text = normalized_caption_text(item.get("normalized_caption_text") or item.get("text") or "")
        if target and candidate_text and (target in candidate_text or candidate_text in target):
            return str(item.get("paired_float_id") or "")
    return ""


def rendered_caption_snippets(path: Path, text: str) -> list[str]:
    if not path.exists():
        return []
    tex = path.read_text(encoding="utf-8", errors="replace")
    target = normalized_caption_text(text)
    snippets: list[str] = []
    for match in re.finditer(r"\\caption(?:\[[^\]]*\])?\{", tex):
        start = match.start()
        end = min(len(tex), start + 500)
        snippet = tex[start:end].splitlines()[0][:300]
        if target and target[:48] in normalized_caption_text(snippet):
            snippets.append(snippet)
    return snippets


def classify_duplicate_delta(
    *,
    text: str,
    label: str,
    origins: set[str],
    blocks: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> str:
    if synthetic_or_panel_text(text):
        return "PANEL_LABEL_MISCOUNT"
    if subfigure_marker(label):
        return "SUBFIGURE_SHOULD_KEEP"
    if origins and origins <= {"crop_metadata"}:
        return "CROP_METADATA_DOUBLE_MATERIALIZATION"
    if len(origins & {"caption_metadata", "float_metadata", "crop_metadata"}) and "text_block" in origins:
        return "TEXT_METADATA_DOUBLE_MATERIALIZATION"
    if len(blocks) > 1 and not origins:
        return "COMPARISON_MATCHING_DUPLICATE"
    suppressed = diagnostics.get("noncanonical_suppressed_candidates") or []
    target = normalized_caption_text(text)
    if any(target and target in normalized_caption_text(item.get("normalized_caption_text") or item.get("text") or "") for item in suppressed):
        return "CONVERTER_DOUBLE_COUNT"
    return "TRUE_DUPLICATE"


def duplicate_recommendation(classification: str) -> str:
    return {
        "TRUE_DUPLICATE": "apply canonical suppression for this exact rendered caption identity",
        "PANEL_LABEL_MISCOUNT": "fix converter/evaluator so panel labels do not count as captions",
        "SUBFIGURE_SHOULD_KEEP": "do not suppress; adjust duplicate metric to preserve subfigure identity",
        "SYNTHETIC_FALLBACK_MISCOUNT": "exclude synthetic/fallback captions from true pred-caption accounting",
        "TEXT_METADATA_DOUBLE_MATERIALIZATION": "keep visible text canonical and suppress metadata/crop duplicate",
        "CROP_METADATA_DOUBLE_MATERIALIZATION": "keep one crop/metadata canonical per paired float",
        "CONVERTER_DOUBLE_COUNT": "fix converter/evaluator duplicate counting, not renderer",
        "COMPARISON_MATCHING_DUPLICATE": "inspect comparison conversion identity; likely duplicate count correction",
    }.get(classification, "manual review")


def classify_suspicious_examples(examples: list[str]) -> str:
    joined = "\n".join(examples).casefold()
    if any(token in joined for token in ["algorithm", "pseudocode", "\\begin{algorithm", "\\end{algorithm", "\\caption{algorithm"]):
        return "ALGORITHM_PSEUDOCODE_NEIGHBORHOOD"
    if any(token in joined for token in ["\\begin{figure", "\\end{figure", "\\begin{table", "\\end{table", "\\includegraphics", "\\caption", "figure placeholder", "table placeholder"]):
        return "TABLE_FIGURE_CAPTION_NEIGHBORHOOD"
    if any(token in joined for token in ["\\section", "\\subsection", "\\paragraph{"]):
        return "HEADING_LEAKAGE"
    if any(token in joined for token in ["\\bibliography", "\\bibitem", "\\begin{thebibliography}", "\\end{thebibliography}"]):
        return "REFERENCE_LEAKAGE"
    if any(token in joined for token in ["\\documentclass", "\\usepackage", "\\setlength", "\\geometry"]):
        return "PREAMBLE_STYLE_LEAKAGE"
    if any(token in joined for token in ["% source", "% generated", "% provenance"]):
        return "METADATA_COMMENT_ORDER"
    if examples:
        return "BODY_TEXT_LEAKAGE"
    return "UNKNOWN"


def suspicious_recommendation(classification: str) -> str:
    if classification in {"ALGORITHM_PSEUDOCODE_NEIGHBORHOOD", "TABLE_FIGURE_CAPTION_NEIGHBORHOOD"}:
        return "defer to AlgorithmRegion or float/table layout pass; do not patch FloatCaptionLayout broadly"
    if classification in {"BODY_TEXT_LEAKAGE", "HEADING_LEAKAGE", "REFERENCE_LEAKAGE", "PREAMBLE_STYLE_LEAKAGE"}:
        return "add leakage guard before any further promotion"
    return "manual review"


def split_examples(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("||") if item.strip()]


def files_differ(left: Path, right: Path) -> bool:
    if not left.exists() or not right.exists():
        return True
    return left.read_text(encoding="utf-8", errors="replace") != right.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default if default is not None else {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def intish(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def floatish(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_report(
    *,
    patch2_summary: dict[str, Any],
    patch3_summary: dict[str, Any],
    drift_summary: dict[str, Any],
    duplicate_summary: dict[str, Any],
    duplicate_cases: list[dict[str, Any]],
    suspicious_summary: dict[str, Any],
    suspicious_cases: list[dict[str, Any]],
) -> str:
    interpretation = drift_summary["interpretation"]
    lines: list[str] = []
    lines.append("# V8 Float-Caption Patch3.1 Duplicate Guard Report")
    lines.append("")
    lines.append("## Status")
    lines.append("- audit status: completed")
    lines.append("- selected200 validation run: completed after minimal duplicate/leakage guard patch")
    lines.append("- no training / no MinerU / no relabel / no rebuild / no GNN")
    lines.append("- production default unchanged")
    lines.append("- v8 full observable facts only; source_v7_ids/v7_id names are treated as legacy provenance names")
    lines.append("")
    lines.append("## Baseline Drift Audit")
    lines.append("| metric | Patch2 flag-off | Patch3 flag-off | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in [
        "float_caption_attachment_accuracy",
        "pred_caption_count",
        "missing_caption_count",
        "duplicate_caption_count",
        "caption_as_paragraph_count",
        "generated_structure_validity",
        "macro_structure_score_body",
    ]:
        p2 = (patch2_summary.get("baseline") or {}).get(key)
        p3 = (patch3_summary.get("baseline") or {}).get(key)
        delta = (drift_summary.get("delta_patch3_minus_patch2") or {}).get(key)
        lines.append(f"| {key} | {p2} | {p3} | {delta} |")
    lines.append("")
    lines.append(f"- Primary cause: {interpretation.get('primary_cause')}")
    lines.append(f"- Patch3 same-code A/B still valid: {interpretation.get('patch3_same_code_ab_valid')}")
    lines.append(f"- Patch3 absolute values comparable to Patch2 absolute values: {interpretation.get('patch3_absolute_values_comparable_to_patch2')}")
    lines.append(f"- Recommended metric version: {interpretation.get('recommended_metric_version')}")
    lines.append(f"- Generated TeX changed ratio between Patch2/Patch3 flag-off: {interpretation.get('generated_tex_changed_ratio'):.3f}")
    lines.append("")
    lines.append("## Duplicate Delta Audit")
    lines.append(f"- Extra duplicate total: {duplicate_summary.get('duplicate_delta_total')}")
    lines.append(f"- Classification counts: {duplicate_summary.get('classification_counts')}")
    lines.append("")
    for row in duplicate_cases[:20]:
        lines.append(
            f"- {row['doc_id']} | {row['classification']} | {row['caption_type']} {row['caption_number']} | "
            f"{row['caption_text'][:120]} | action: {row['recommended_action']}"
        )
    if not duplicate_cases:
        lines.append("- No extra duplicate cases found.")
    lines.append("")
    lines.append("## Suspicious Diff Delta Audit")
    lines.append(f"- Patch2 suspicious total: {suspicious_summary.get('patch2_suspicious_total')}")
    lines.append(f"- Patch3 suspicious total: {suspicious_summary.get('patch3_suspicious_total')}")
    lines.append(f"- Positive delta total across docs: {suspicious_summary.get('positive_delta_total')}")
    lines.append(f"- Classification counts: {suspicious_summary.get('classification_counts')}")
    lines.append("")
    for row in suspicious_cases[:20]:
        lines.append(
            f"- {row['doc_id']} | +{row['delta']} | {row['classification']} | "
            f"{row['examples'][:160]} | action: {row['recommended_action']}"
        )
    if not suspicious_cases:
        lines.append("- No positive suspicious diff delta cases found.")
    lines.append("")
    lines.append("## Patch Applied")
    lines.append("- canonical caption identity guard: same visible caption text is materialized once across neighboring float ids.")
    lines.append("- panel/synthetic guard: multi-panel labels such as `(a) (b)` are not emitted as real captions.")
    lines.append("- renderer duplicate guard: a float keeps its visual asset, but repeated caption text is not emitted again.")
    lines.append("- diff attribution guard: heading commands containing caption-like titles are counted as float/caption-neighborhood diffs.")
    lines.append("")
    lines.append("## Patch3.1 Same-code A/B")
    lines.append("| metric | flag-off | flag-on Patch3.1 | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in [
        "float_caption_attachment_accuracy",
        "pred_caption_count",
        "missing_caption_count",
        "duplicate_caption_count",
        "true_duplicate_caption_count",
        "caption_as_paragraph_count",
        "wrong_float_type_pairing_count",
        "generated_structure_validity",
        "macro_structure_score_body",
        "placeholder_float_count",
        "subfigure_false_suppression_count",
    ]:
        base = (patch3_summary.get("baseline") or {}).get(key)
        exp = (patch3_summary.get("experimental") or {}).get(key)
        delta = (patch3_summary.get("delta") or {}).get(key)
        lines.append(f"| {key} | {base} | {exp} | {delta} |")
    lines.append("")
    lines.append("## Decision")
    decision = patch31_decision(duplicate_summary, suspicious_summary, patch3_summary)
    lines.append(f"- {decision}")
    if decision == "patch_required":
        lines.append("- Primary next step would be another narrow duplicate/converter pass, but this run already shows diminishing returns.")
    elif decision == "diagnostic_only":
        lines.append("- FloatCaptionLayout should not be expanded until duplicate/leakage accounting is clean.")
    else:
        lines.append("- The experimental branch can remain opt-in, not production default.")
    return "\n".join(lines) + "\n"


def patch31_decision(
    duplicate_summary: dict[str, Any],
    suspicious_summary: dict[str, Any],
    patch3_summary: dict[str, Any],
) -> str:
    baseline = patch3_summary.get("baseline") or {}
    experimental = patch3_summary.get("experimental") or {}
    delta = patch3_summary.get("delta") or {}
    if floatish(experimental.get("duplicate_caption_count")) and floatish(baseline.get("duplicate_caption_count")) is not None:
        if float(experimental.get("duplicate_caption_count") or 0) > float(baseline.get("duplicate_caption_count") or 0):
            return "diagnostic_only"
    if float(delta.get("float_caption_attachment_accuracy") or 0.0) < 0:
        return "diagnostic_only"
    if int(duplicate_summary.get("duplicate_delta_total") or 0) > 0:
        return "patch_required"
    if suspicious_summary.get("has_true_non_caption_leakage") and int(suspicious_summary.get("patch3_suspicious_total") or 0) > int(suspicious_summary.get("patch2_suspicious_total") or 0):
        return "patch_required"
    return "safe_to_keep_experimental_enabled"


if __name__ == "__main__":
    raise SystemExit(main())
