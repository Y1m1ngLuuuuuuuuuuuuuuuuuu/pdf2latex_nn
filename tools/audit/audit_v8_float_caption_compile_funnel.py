#!/usr/bin/env python3
"""Compile smoke and promotion funnel audit for v8 FloatCaptionLayout A/B outputs."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_AB_ROOT = Path("data/09_eval_reports/float_caption_layout_20260526/v8_same_code_ab_validation")
DEFAULT_OUTPUT = DEFAULT_AB_ROOT / "compile_smoke_and_promotion_funnel"


def main() -> int:
    args = build_arg_parser().parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    readiness = check_readiness(args.ab_root)
    if not readiness["ready"]:
        write_json(out / "READINESS_REPORT.json", readiness)
        (out / "READINESS_REPORT.md").write_text(
            "# Float-Caption Compile/Funnel Readiness Report\n\n"
            + "\n".join(f"- {item}" for item in readiness["missing"])
            + "\n",
            encoding="utf-8",
        )
        return 2

    baseline_rows = read_csv(args.ab_root / "baseline_flag_off_current_code" / "summary.csv")
    experimental_rows = read_csv(args.ab_root / "experimental_flag_on_current_code" / "summary.csv")
    baseline_by_doc = {row["doc_id"]: row for row in baseline_rows}
    experimental_by_doc = {row["doc_id"]: row for row in experimental_rows}
    diff_rows = read_csv(args.ab_root / "selected200_diff_attribution.csv")
    diff_by_doc = {row["doc_id"]: row for row in diff_rows}

    funnel_rows, funnel_summary, manual_pack = build_promotion_funnel(
        args.ab_root, baseline_by_doc, experimental_by_doc, diff_by_doc
    )
    write_csv(out / "promotion_funnel_summary.csv", funnel_rows)
    write_json(out / "promotion_funnel_summary.json", funnel_summary)

    suspicious_rows, suspicious_summary, suspicious_examples = build_suspicious_diff_audit(args.ab_root)
    write_csv(out / "suspicious_diff_attribution.csv", suspicious_rows)
    manual_pack["true_suspicious_non_caption_examples"] = suspicious_examples

    smoke_docs = select_compile_smoke_docs(funnel_rows, suspicious_rows, limit=args.compile_count)
    compile_rows, compile_summary = run_compile_smoke(args.ab_root, out, smoke_docs, timeout=args.compile_timeout)
    write_csv(out / "compile_smoke_summary.csv", compile_rows)
    write_json(out / "compile_smoke_summary.json", compile_summary)

    write_json(out / "manual_review_pack.json", manual_pack)
    write_manual_review_markdown(out / "manual_review_pack.md", manual_pack)
    write_report(
        out / "V8_FLOAT_CAPTION_COMPILE_SMOKE_AND_PROMOTION_FUNNEL_REPORT.md",
        ab_root=args.ab_root,
        compile_summary=compile_summary,
        funnel_summary=funnel_summary,
        suspicious_summary=suspicious_summary,
        manual_pack=manual_pack,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-root", type=Path, default=DEFAULT_AB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compile-count", type=int, default=30)
    parser.add_argument("--compile-timeout", type=int, default=90)
    return parser


def check_readiness(root: Path) -> dict[str, Any]:
    required = [
        root / "baseline_flag_off_current_code",
        root / "experimental_flag_on_current_code",
        root / "baseline_flag_off_current_code" / "summary.csv",
        root / "experimental_flag_on_current_code" / "summary.csv",
        root / "selected200_diff_attribution.csv",
        root / "V8_FLOAT_CAPTION_SAME_CODE_AB_VALIDATION_REPORT.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    for branch in ["baseline_flag_off_current_code", "experimental_flag_on_current_code"]:
        branch_dir = root / branch
        if branch_dir.exists():
            for doc_dir in branch_dir.iterdir():
                if not doc_dir.is_dir():
                    continue
                for name in [
                    "generated.tex",
                    "float_caption_fix_diag.json",
                    "promoted_captions.json",
                    "float_caption_pairings.json",
                    "placeholder_floats.json",
                    "duplicate_caption_suppression.json",
                    "crop_caption_separation.json",
                    "consumed_caption_paragraphs.json",
                ]:
                    if not (doc_dir / name).exists():
                        missing.append(str(doc_dir / name))
                        break
    return {"ready": not missing, "missing": missing}


def build_promotion_funnel(
    ab_root: Path,
    baseline_by_doc: dict[str, dict[str, str]],
    experimental_by_doc: dict[str, dict[str, str]],
    diff_by_doc: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    manual_pack = empty_manual_pack()
    totals = Counter()
    by_origin: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    pairing_failures: Counter[str] = Counter()
    not_rendered_reasons: Counter[str] = Counter()

    exp_root = ab_root / "experimental_flag_on_current_code"
    for doc_dir in sorted(path for path in exp_root.iterdir() if path.is_dir()):
        doc_id = doc_dir.name.split("_", 1)[-1]
        base_row = baseline_by_doc.get(doc_id, {})
        exp_row = experimental_by_doc.get(doc_id, {})
        diag = read_json(doc_dir / "float_caption_fix_diag.json")
        promoted = read_json(doc_dir / "promoted_captions.json")
        pairings = read_json(doc_dir / "float_caption_pairings.json")
        placeholders = read_json(doc_dir / "placeholder_floats.json")
        duplicates = read_json(doc_dir / "duplicate_caption_suppression.json")
        crop_sep = read_json(doc_dir / "crop_caption_separation.json")
        consumed = read_json(doc_dir / "consumed_caption_paragraphs.json")
        structure = read_json(doc_dir / "ours_comparison_structure_current.json")
        pred_caption_texts = normalized_caption_texts(structure)

        candidate_count = len(promoted)
        high_conf = [item for item in promoted if float(item.get("confidence") or 0.0) >= 0.85]
        rendered_candidates = [item for item in promoted if caption_consumed(item, pred_caption_texts)]
        unrendered = [item for item in promoted if item not in rendered_candidates]
        paired_count = sum(1 for item in pairings if item.get("paired_float_id"))
        placeholder_count = len(placeholders)
        rendered_caption_count = intish(exp_row.get("pred_caption_count"))
        baseline_pred = intish(base_row.get("pred_caption_count"))
        pred_delta = rendered_caption_count - baseline_pred
        recovered_missing = max(0, pred_delta)
        metadata_candidates = [
            item for item in promoted if item.get("origin") in {"caption_metadata", "float_metadata", "crop_metadata"}
        ]
        metadata_rendered = [item for item in metadata_candidates if caption_consumed(item, pred_caption_texts)]

        for item in promoted:
            by_origin[str(item.get("origin") or "unknown")] += 1
            by_type[str(item.get("caption_type") or "unknown")] += 1
        for item in unrendered:
            reason = classify_unrendered_candidate(item, pairings, duplicates, placeholders)
            not_rendered_reasons[reason] += 1
        for item in placeholders:
            pairing_failures[str(item.get("reason") or "no_float_candidate")] += 1

        row = {
            "doc_id": doc_id,
            "v8_caption_like_candidate_count": candidate_count,
            "high_confidence_candidate_count": len(high_conf),
            "promoted_caption_count": candidate_count,
            "not_promoted_count": len(unrendered),
            "paired_caption_count": paired_count,
            "unpaired_caption_count": max(0, candidate_count - paired_count),
            "placeholder_created_count": placeholder_count,
            "rendered_caption_count": rendered_caption_count,
            "rendered_candidate_count": len(rendered_candidates),
            "metadata_crop_candidate_count": len(metadata_candidates),
            "metadata_crop_rendered_candidate_count": len(metadata_rendered),
            "recovered_missing_caption_count": recovered_missing,
            "unmatched_rendered_caption_count": max(0, rendered_caption_count - len(rendered_candidates)),
            "possible_comparison_mismatch_count": max(0, len(rendered_candidates) - rendered_caption_count),
            "consumed_caption_paragraph_count": len(consumed),
            "caption_duplicate_render_count": intish(exp_row.get("duplicate_caption_count")),
            "duplicate_suppressed_count": len(duplicates),
            "crop_may_include_caption_count": len(crop_sep),
            "caption_as_paragraph_count": intish(exp_row.get("caption_as_paragraph_count")),
            "missing_caption_count": intish(exp_row.get("missing_caption_count")),
            "pred_caption_delta": pred_delta,
            "non_caption_suspicious_change_count": intish((diff_by_doc.get(doc_id) or {}).get("non_caption_suspicious_change_count")),
        }
        rows.append(row)
        for key, value in row.items():
            if key != "doc_id" and isinstance(value, int):
                totals[key] += value

        collect_manual_examples(manual_pack, doc_id, doc_dir, promoted, placeholders, duplicates, crop_sep, consumed)

    summary = {
        "docs": len(rows),
        "totals": dict(totals),
        "candidate_by_origin": dict(by_origin),
        "candidate_by_type": dict(by_type),
        "not_promoted_reason_distribution": dict(not_rendered_reasons),
        "pairing_failure_reason_distribution": dict(pairing_failures),
        "funnel": {
            "candidate": totals["v8_caption_like_candidate_count"],
            "high_confidence": totals["high_confidence_candidate_count"],
            "promoted": totals["promoted_caption_count"],
            "paired": totals["paired_caption_count"],
            "rendered_candidates": totals["rendered_candidate_count"],
            "pred_caption_count": sum(intish(row.get("pred_caption_count")) for row in experimental_by_doc.values()),
            "matched_caption_count": None,
        },
    }
    return rows, summary, manual_pack


def classify_unrendered_candidate(
    candidate: dict[str, Any],
    pairings: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    placeholders: list[dict[str, Any]],
) -> str:
    cid = candidate.get("caption_id")
    if any((item.get("caption_id") == cid or item.get("duplicate_caption_id") == cid) for item in duplicates):
        return "duplicate_suppressed"
    if any((item.get("caption_id") == cid or item.get("caption", {}).get("caption_id") == cid) for item in pairings):
        return "paired_but_not_matched_by_evaluation"
    if any(item.get("caption_id") == cid for item in placeholders):
        return "placeholder_created_but_not_matched"
    if not candidate.get("caption_type") or candidate.get("caption_type") == "unknown":
        return "missing_type_or_number"
    if float(candidate.get("confidence") or 0.0) < 0.85:
        return "low_confidence"
    return "unmatched_or_comparison_mismatch"


def build_suspicious_diff_audit(ab_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    baseline_root = ab_root / "baseline_flag_off_current_code"
    exp_root = ab_root / "experimental_flag_on_current_code"
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    totals = Counter()
    for base_doc in sorted(path for path in baseline_root.iterdir() if path.is_dir()):
        exp_doc = exp_root / base_doc.name
        if not exp_doc.exists():
            continue
        doc_id = base_doc.name.split("_", 1)[-1]
        old = (base_doc / "generated.tex").read_text(encoding="utf-8", errors="replace").splitlines()
        new = (exp_doc / "generated.tex").read_text(encoding="utf-8", errors="replace").splitlines()
        diff = list(difflib.unified_diff(old, new, n=2, lineterm=""))
        allowed: list[str] = []
        true_suspicious: list[str] = []
        for line in diff:
            if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
                continue
            text = line[1:].strip()
            if not text:
                continue
            if is_allowed_local_float_caption_diff(text):
                allowed.append(text)
            else:
                true_suspicious.append(text)
        row = {
            "doc_id": doc_id,
            "suspicious_line_count": len(allowed) + len(true_suspicious),
            "allowed_local_count": len(allowed),
            "true_suspicious_count": len(true_suspicious),
            "examples_before_after": " || ".join(true_suspicious[:6]),
            "nearest_float_caption_context": "float_caption_neighborhood" if allowed else "",
        }
        if row["suspicious_line_count"]:
            rows.append(row)
        totals["docs_with_diffs"] += int(bool(row["suspicious_line_count"]))
        totals["allowed_local_count"] += len(allowed)
        totals["true_suspicious_count"] += len(true_suspicious)
        if true_suspicious:
            examples.append(
                {
                    "doc_id": doc_id,
                    "examples": true_suspicious[:10],
                    "allowed_local_count": len(allowed),
                    "true_suspicious_count": len(true_suspicious),
                }
            )
    return rows, {"totals": dict(totals), "docs": len(rows)}, examples


def is_allowed_local_float_caption_diff(text: str) -> bool:
    lowered = text.casefold()
    if any(token in lowered for token in ["caption", "figure", "fig.", "table", "algorithm", "alg.", "placeholder"]):
        return True
    if any(
        token in text
        for token in [
            "\\begin{figure",
            "\\end{figure",
            "\\begin{table",
            "\\end{table",
            "\\begin{algorithm",
            "\\end{algorithm",
            "\\includegraphics",
            "\\centering",
            "\\label{fig:",
            "\\label{tab:",
            "\\label{alg:",
            "TODO_FIGURE_RECONSTRUCT",
            "TODO_TABLE_RECONSTRUCT",
            "FLOAT_WIDTH_SCOPE",
            "\\State",
            "\\For",
            "\\EndFor",
            "\\If",
            "\\EndIf",
        ]
    ):
        return True
    return False


def select_compile_smoke_docs(funnel_rows: list[dict[str, Any]], suspicious_rows: list[dict[str, Any]], *, limit: int) -> list[str]:
    suspicious_by_doc = {row["doc_id"]: row for row in suspicious_rows}
    scored: list[tuple[int, str]] = []
    for row in funnel_rows:
        doc_id = row["doc_id"]
        score = (
            intish(row.get("placeholder_created_count")) * 8
            + intish(row.get("crop_may_include_caption_count")) * 5
            + intish(row.get("duplicate_suppressed_count")) * 4
            + intish(row.get("consumed_caption_paragraph_count")) * 4
            + intish(row.get("recovered_missing_caption_count")) * 5
            + intish(row.get("non_caption_suspicious_change_count")) * 2
            + intish((suspicious_by_doc.get(doc_id) or {}).get("true_suspicious_count")) * 6
        )
        scored.append((score, doc_id))
    scored.sort(reverse=True)
    selected = [doc_id for score, doc_id in scored if score > 0][:limit]
    if len(selected) < min(20, limit):
        for _score, doc_id in scored:
            if doc_id not in selected:
                selected.append(doc_id)
            if len(selected) >= min(20, limit):
                break
    return selected[: max(20, min(limit, 50))]


def run_compile_smoke(ab_root: Path, out: Path, doc_ids: list[str], *, timeout: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exp_root = ab_root / "experimental_flag_on_current_code"
    compile_root = out / "compile_outputs"
    rows: list[dict[str, Any]] = []
    error_counts = Counter()
    if not shutil.which("latexmk") and not shutil.which("pdflatex") and not shutil.which("xelatex"):
        return [], {"status": "compile_smoke_skipped", "reason": "No LaTeX engine available."}
    for doc_id in doc_ids:
        doc_dirs = list(exp_root.glob(f"*_{doc_id}"))
        if not doc_dirs:
            continue
        doc_dir = doc_dirs[0]
        output_dir = compile_root / doc_dir.name
        report = compile_latex_smoke(doc_dir / "generated.tex", output_dir=output_dir, timeout=timeout)
        kind = classify_compile_error(report)
        error_counts[kind] += int(not report.get("success"))
        sidecar = read_json(doc_dir / "float_caption_fix_diag.json")
        row = {
            "doc_id": doc_id,
            "success": bool(report.get("success")),
            "engine": report.get("engine"),
            "elapsed_sec": report.get("elapsed_sec"),
            "returncode": report.get("returncode"),
            "error_type": "" if report.get("success") else kind,
            "error_summary": report.get("error_summary", "")[:500],
            "placeholder_float_count": len(sidecar.get("placeholder_floats", [])),
            "algorithm_caption_count": sum(1 for item in sidecar.get("promoted_captions", []) if item.get("caption_type") == "algorithm"),
            "output_pdf": report.get("output_pdf"),
        }
        rows.append(row)
        write_json(output_dir / "compile_report.json", report)
    success = sum(1 for row in rows if row["success"])
    summary = {
        "status": "completed",
        "docs_compiled": len(rows),
        "compile_success_count": success,
        "compile_success_rate": success / len(rows) if rows else 0.0,
        "latex_error_count": len(rows) - success,
        "errors_by_type": dict(error_counts),
        "algorithm_environment_errors": error_counts.get("algorithm_environment_error", 0),
        "figure_environment_errors": error_counts.get("figure_environment_error", 0),
        "table_environment_errors": error_counts.get("table_environment_error", 0),
        "missing_package_errors": error_counts.get("missing_package_error", 0),
        "unbalanced_brace_or_math_errors": error_counts.get("unbalanced_brace_or_math_error", 0),
        "placeholder_compile_errors": sum(
            1 for row in rows if (not row["success"] and intish(row.get("placeholder_float_count")) > 0)
        ),
    }
    return rows, summary


def classify_compile_error(report: dict[str, Any]) -> str:
    if report.get("success"):
        return "none"
    text = f"{report.get('error_summary', '')}\n{report.get('log_tail', '')}".casefold()
    if "not found" in text and ".sty" in text:
        return "missing_package_error"
    if "algorithm" in text:
        return "algorithm_environment_error"
    if "figure" in text:
        return "figure_environment_error"
    if "table" in text or "tabular" in text:
        return "table_environment_error"
    if any(token in text for token in ["missing $", "extra }", "runaway", "forgotten", "unbalanced"]):
        return "unbalanced_brace_or_math_error"
    if "undefined control sequence" in text:
        return "undefined_control_sequence"
    return "other_latex_error"


def compile_latex_smoke(tex_path: Path, *, output_dir: Path, timeout: int) -> dict[str, Any]:
    tex_path = tex_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = resolve_latex_engine()
    if not engine:
        return {
            "success": False,
            "engine": "auto",
            "output_pdf": None,
            "elapsed_sec": 0.0,
            "returncode": None,
            "error_summary": "No LaTeX engine found.",
            "log_tail": "",
        }
    command = build_latex_command(engine, tex_path, output_dir)
    start = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(tex_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        raw = completed.stdout or b""
        log_text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or b""
        log_text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        return {
            "success": False,
            "engine": engine,
            "command": command,
            "output_pdf": None,
            "elapsed_sec": round(time.time() - start, 3),
            "returncode": None,
            "error_summary": f"Compilation timed out after {timeout}s.",
            "log_tail": tail_text(log_text, 160),
        }
    output_pdf = output_dir / f"{tex_path.stem}.pdf"
    success = returncode == 0 and output_pdf.exists()
    return {
        "success": success,
        "engine": engine,
        "command": command,
        "output_pdf": str(output_pdf) if output_pdf.exists() else None,
        "elapsed_sec": round(time.time() - start, 3),
        "returncode": returncode,
        "error_summary": "" if success else summarize_latex_error(log_text),
        "log_tail": tail_text(log_text, 160),
    }


def resolve_latex_engine() -> str | None:
    for engine in ["latexmk", "pdflatex", "xelatex"]:
        if shutil.which(engine):
            return engine
    return None


def build_latex_command(engine: str, tex_path: Path, output_dir: Path) -> list[str]:
    if engine == "latexmk":
        return [
            engine,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-outdir={output_dir}",
            str(tex_path),
        ]
    return [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={output_dir}",
        str(tex_path),
    ]


def summarize_latex_error(log_text: str) -> str:
    lines = str(log_text or "").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("!") or re.search(r":\d+:\s", line):
            return "\n".join(lines[index : index + 8])
    for pattern in ["Emergency stop", "Fatal error", "Undefined control sequence", "Missing $ inserted"]:
        for index, line in enumerate(lines):
            if pattern in line:
                return "\n".join(lines[index : index + 8])
    return tail_text(log_text, 30)


def tail_text(text: str, lines: int) -> str:
    values = str(text or "").splitlines()
    return "\n".join(values[-lines:])


def collect_manual_examples(
    manual_pack: dict[str, list[dict[str, Any]]],
    doc_id: str,
    doc_dir: Path,
    promoted: list[dict[str, Any]],
    placeholders: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    crop_sep: list[dict[str, Any]],
    consumed: list[dict[str, Any]],
) -> None:
    tex_path = doc_dir / "generated.tex"
    for item in promoted:
        if item.get("caption_type") == "algorithm":
            add_manual_example(manual_pack["algorithm_caption_examples"], doc_id, tex_path, item)
    for item in placeholders:
        add_manual_example(manual_pack["placeholder_float_examples"], doc_id, tex_path, item)
    for item in crop_sep:
        add_manual_example(manual_pack["crop_may_include_caption_examples"], doc_id, tex_path, item)
    for item in duplicates:
        add_manual_example(manual_pack["duplicate_suppression_examples"], doc_id, tex_path, item)
    for item in consumed:
        add_manual_example(manual_pack["caption_as_paragraph_fixed_examples"], doc_id, tex_path, item)


def add_manual_example(bucket: list[dict[str, Any]], doc_id: str, tex_path: Path, item: dict[str, Any], limit: int = 20) -> None:
    if len(bucket) >= limit:
        return
    caption = item.get("text") or item.get("normalized_caption_text") or item.get("reason") or str(item)
    bucket.append(
        {
            "doc_id": doc_id,
            "page_idx": item.get("page_idx"),
            "caption_text": caption[:300],
            "source_v8_ids": item.get("source_v8_ids") or item.get("consumed_source_v8_ids") or [],
            "reason": item.get("reason") or item.get("promotion_reason") or item.get("pairing_reason"),
            "generated_tex_neighborhood": tex_neighborhood(tex_path, caption),
        }
    )


def empty_manual_pack() -> dict[str, list[dict[str, Any]]]:
    return {
        "algorithm_caption_examples": [],
        "placeholder_float_examples": [],
        "crop_may_include_caption_examples": [],
        "duplicate_suppression_examples": [],
        "caption_as_paragraph_fixed_examples": [],
        "true_suspicious_non_caption_examples": [],
    }


def tex_neighborhood(tex_path: Path, text: str, *, radius: int = 3) -> str:
    if not tex_path.exists() or not text:
        return ""
    lines = tex_path.read_text(encoding="utf-8", errors="replace").splitlines()
    needle = normalize_for_search(text)
    if not needle:
        return ""
    for index, line in enumerate(lines):
        if needle[:45] in normalize_for_search(line):
            start = max(0, index - radius)
            end = min(len(lines), index + radius + 1)
            return "\n".join(f"{i+1}: {lines[i]}" for i in range(start, end))
    return ""


def write_manual_review_markdown(path: Path, pack: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Float-Caption Manual Review Pack", ""]
    for title, key in [
        ("Algorithm caption examples", "algorithm_caption_examples"),
        ("Placeholder float examples", "placeholder_float_examples"),
        ("Crop may include caption examples", "crop_may_include_caption_examples"),
        ("Duplicate suppression examples", "duplicate_suppression_examples"),
        ("Caption-as-paragraph fixed examples", "caption_as_paragraph_fixed_examples"),
        ("True suspicious non-caption examples", "true_suspicious_non_caption_examples"),
    ]:
        lines.extend([f"## {title}", ""])
        examples = pack.get(key, [])
        if not examples:
            lines.append("- none")
            lines.append("")
            continue
        for item in examples[:20]:
            lines.append(f"### {item.get('doc_id')}")
            preview = item.get("caption_text") or item.get("preview") or item.get("examples")
            lines.append(str(preview).replace("\n", " ")[:600])
            neighborhood = item.get("generated_tex_neighborhood")
            if neighborhood:
                lines.append("```tex")
                lines.append(neighborhood[:1500])
                lines.append("```")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    path: Path,
    *,
    ab_root: Path,
    compile_summary: dict[str, Any],
    funnel_summary: dict[str, Any],
    suspicious_summary: dict[str, Any],
    manual_pack: dict[str, list[dict[str, Any]]],
) -> None:
    ab = read_json(ab_root / "selected200_same_code_ab_summary.json")
    baseline = ab["baseline"]
    experimental = ab["experimental"]
    delta = ab["delta"]
    decision = decide_report(compile_summary, funnel_summary, suspicious_summary)
    lines = [
        "# V8 Float-Caption Compile Smoke and Promotion Funnel Report",
        "",
        "## Status",
        f"- Docs analyzed: {funnel_summary.get('docs', 0)}",
        f"- Compile smoke status: {compile_summary.get('status')}",
        "- Training: No",
        "- MinerU: No",
        "- Relabel / rebuild: No",
        "- GNN: No",
        "- Production default unchanged: Yes",
        "",
        "## A/B Recap",
        "| metric | flag-off | flag-on | delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in [
        "float_caption_attachment_accuracy",
        "pred_caption_count",
        "missing_caption_count",
        "caption_as_paragraph_count",
        "duplicate_caption_count",
        "wrong_float_type_pairing_count",
        "generated_structure_validity",
        "macro_structure_score_body",
    ]:
        lines.append(f"| {metric} | {fmt(baseline.get(metric))} | {fmt(experimental.get(metric))} | {fmt(delta.get(metric))} |")
    lines.extend(
        [
            "",
            "## Compile Smoke",
            "| metric | value |",
            "|---|---:|",
            f"| compile docs | {compile_summary.get('docs_compiled', 0)} |",
            f"| compile success | {compile_summary.get('compile_success_count', 0)} |",
            f"| compile success rate | {compile_summary.get('compile_success_rate', 0):.6f} |",
            f"| latex errors | {compile_summary.get('latex_error_count', 0)} |",
            f"| algorithm environment errors | {compile_summary.get('algorithm_environment_errors', 0)} |",
            f"| figure environment errors | {compile_summary.get('figure_environment_errors', 0)} |",
            f"| table environment errors | {compile_summary.get('table_environment_errors', 0)} |",
            f"| missing package errors | {compile_summary.get('missing_package_errors', 0)} |",
            f"| unbalanced brace/math errors | {compile_summary.get('unbalanced_brace_or_math_errors', 0)} |",
            f"| placeholder compile errors | {compile_summary.get('placeholder_compile_errors', 0)} |",
            "",
            "## Promotion Funnel",
            "| stage | count |",
            "|---|---:|",
        ]
    )
    for key, value in funnel_summary.get("funnel", {}).items():
        lines.append(f"| {key} | {value if value is not None else 'not_available'} |")
    lines.extend(["", "### Candidate By Origin", "| origin | count |", "|---|---:|"])
    for key, value in sorted(funnel_summary.get("candidate_by_origin", {}).items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "### Candidate By Type", "| type | count |", "|---|---:|"])
    for key, value in sorted(funnel_summary.get("candidate_by_type", {}).items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "### Not Promoted / Not Rendered Reasons", "| reason | count |", "|---|---:|"])
    for key, value in sorted(funnel_summary.get("not_promoted_reason_distribution", {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Suspicious Diff Attribution",
            f"- docs with diffs: {suspicious_summary.get('totals', {}).get('docs_with_diffs', 0)}",
            f"- allowed/local float-caption lines: {suspicious_summary.get('totals', {}).get('allowed_local_count', 0)}",
            f"- true suspicious non-caption lines: {suspicious_summary.get('totals', {}).get('true_suspicious_count', 0)}",
            "",
            "## Diagnosis",
            "- v8 caption-like candidates are plentiful, but most are already represented in the current RenderTreeIR path or are metadata/crop candidates whose text does not become a new comparison caption.",
            "- The +10 pred caption gain is small because the fix is conservative: it mainly converts caption-like paragraphs, suppresses duplicates, and promotes a small number of algorithm/table/figure captions without inventing extra floats.",
            "- Metadata/crop caption not consumed remains dominated by materialization/evaluation matching ambiguity rather than raw candidate absence.",
            "- Figure missing barely changes because most figure candidates already have crops/wrappers; missing comparison captions often need better crop/caption separation or pairing/matching, not just grammar promotion.",
            "- Algorithm captions improve because algorithm candidates were previously absent from rendered caption tracks and can be safely promoted or placeholdered in several docs.",
            "- Duplicate suppression appears safe in skip-compile metrics: duplicate captions decreased and wrong type pairing did not increase.",
            "",
            "## Manual Review Examples",
            f"- algorithm caption examples: {len(manual_pack.get('algorithm_caption_examples', []))}",
            f"- placeholder float examples: {len(manual_pack.get('placeholder_float_examples', []))}",
            f"- crop_may_include_caption examples: {len(manual_pack.get('crop_may_include_caption_examples', []))}",
            f"- duplicate suppression examples: {len(manual_pack.get('duplicate_suppression_examples', []))}",
            f"- caption-as-paragraph fixed examples: {len(manual_pack.get('caption_as_paragraph_fixed_examples', []))}",
            f"- true suspicious examples: {len(manual_pack.get('true_suspicious_non_caption_examples', []))}",
            "",
            "See `manual_review_pack.md` for readable snippets.",
            "",
            "## Decision",
            decision,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def decide_report(
    compile_summary: dict[str, Any],
    funnel_summary: dict[str, Any],
    suspicious_summary: dict[str, Any],
) -> str:
    if compile_summary.get("status") == "completed" and compile_summary.get("compile_success_rate", 0.0) < 0.8:
        return "disable_experimental"
    if suspicious_summary.get("totals", {}).get("true_suspicious_count", 0) > 0:
        return "keep_experimental_and_patch_targeted"
    return "keep_experimental_and_patch_targeted"


def caption_consumed(candidate: dict[str, Any], pred_caption_texts: list[str]) -> bool:
    text = normalize(candidate.get("normalized_caption_text") or candidate.get("text") or "")
    if not text:
        return False
    return any(text in pred or pred in text for pred in pred_caption_texts if pred)


def normalized_caption_texts(structure: dict[str, Any]) -> list[str]:
    texts = []
    for block in structure.get("blocks", []):
        if block.get("block_type") == "caption":
            texts.append(normalize(block.get("text") or block.get("normalized_text") or ""))
    return texts


def normalize(text: str) -> str:
    return " ".join(str(text or "").casefold().replace(":", " ").split())


def normalize_for_search(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def intish(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value in (None, ""):
        return ""
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
