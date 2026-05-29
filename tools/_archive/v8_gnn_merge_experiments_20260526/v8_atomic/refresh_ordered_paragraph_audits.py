#!/usr/bin/env python3
"""Refresh paragraph-preservation audits after adding order-sensitive metrics.

The script is intentionally narrow: it only finds existing ``generated.tex``
files, reruns ``check_paragraph_preservation_against_tex.py`` next to each
output, and writes a compact summary.  It does not regenerate LaTeX, rerun
MinerU, train, or delete prior artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DOC_ID_RE = re.compile(r"\d{4}\.\d{5}")


@dataclass(frozen=True)
class AuditTask:
    doc_id: str
    root_label: str
    group: str
    variant: str
    generated_tex: str
    source_tex: str
    output_dir: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True, help="Existing experiment root to scan")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-list-items", action="store_true")
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = collect_tasks(args.root)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    (args.output_dir / "refresh_manifest.json").write_text(
        json.dumps({"schema_version": "ordered_paragraph_audit_refresh_v1", "tasks": [task.__dict__ for task in tasks]}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    max_workers = max(1, min(args.workers, 12))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(run_task, task, args.include_list_items, args.max_examples, env): task
            for task in tasks
        }
        for idx, future in enumerate(as_completed(future_map), start=1):
            task = future_map[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001 - report and continue batch.
                errors.append({**task.__dict__, "error": str(exc)})
            if idx % 100 == 0 or idx == len(tasks):
                print(f"[{idx}/{len(tasks)}] refreshed")

    rows.sort(key=lambda row: (row["root_label"], row["group"], row["doc_id"], row["variant"]))
    write_csv(args.output_dir / "ordered_paragraph_audit_summary.csv", rows)
    (args.output_dir / "ordered_paragraph_audit_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n")
    write_report(args.output_dir / "ORDERED_PARAGRAPH_AUDIT_REFRESH_REPORT.md", rows, errors, max_workers)
    return 0 if not errors else 1


def collect_tasks(roots: list[Path]) -> list[AuditTask]:
    tasks: list[AuditTask] = []
    seen: set[str] = set()
    for root in roots:
        root = root.resolve()
        root_label = root.name
        for generated_tex in sorted(root.glob("**/generated.tex")):
            if "/paragraph_audit/" in str(generated_tex):
                continue
            doc_id = infer_doc_id(generated_tex)
            if not doc_id:
                continue
            source_tex = find_source_tex(doc_id)
            if not source_tex:
                continue
            group, variant = infer_group_variant(root, generated_tex, doc_id)
            key = str(generated_tex)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(
                AuditTask(
                    doc_id=doc_id,
                    root_label=root_label,
                    group=group,
                    variant=variant,
                    generated_tex=str(generated_tex),
                    source_tex=str(source_tex),
                    output_dir=str(generated_tex.parent / "paragraph_audit"),
                )
            )
    return tasks


def infer_doc_id(path: Path) -> str | None:
    matches = DOC_ID_RE.findall(str(path))
    return matches[-1] if matches else None


def find_source_tex(doc_id: str) -> Path | None:
    tex_dir = ROOT / "data/03_tex_sources" / doc_id
    if not tex_dir.exists():
        return None
    tex_files = sorted(tex_dir.glob("*.tex"))
    if not tex_files:
        return None
    return max(tex_files, key=lambda path: path.stat().st_size)


def infer_group_variant(root: Path, generated_tex: Path, doc_id: str) -> tuple[str, str]:
    rel_parts = generated_tex.relative_to(root).parts
    variant = generated_tex.parent.name
    doc_pos = next((idx for idx, part in enumerate(rel_parts) if doc_id in part), None)
    if doc_pos is None:
        return ("/".join(rel_parts[:-1]), variant)
    if doc_pos > 0 and doc_pos + 1 < len(rel_parts):
        group = "/".join(rel_parts[:doc_pos])
        variant = rel_parts[doc_pos + 1]
    elif doc_pos > 0:
        group = "/".join(rel_parts[:doc_pos])
        variant = rel_parts[doc_pos]
    else:
        group = root.name
        variant = rel_parts[doc_pos]
    return group or root.name, variant


def run_task(task: AuditTask, include_list_items: bool, max_examples: int, env: dict[str, str]) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "tools/audit/check_paragraph_preservation_against_tex.py"),
        "--source-tex",
        task.source_tex,
        "--generated-tex",
        task.generated_tex,
        "--doc-id",
        task.doc_id,
        "--output-dir",
        task.output_dir,
        "--max-examples",
        str(max_examples),
    ]
    if include_list_items:
        cmd.append("--include-list-items")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    payload = json.loads((Path(task.output_dir) / "paragraph_preservation_against_tex.json").read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    return {
        **task.__dict__,
        "source_coverage_rate_raw": summary.get("source_coverage_rate_raw", summary.get("source_coverage_rate")),
        "ordered_source_coverage_rate_raw": summary.get("ordered_source_coverage_rate_raw"),
        "source_order_inversion_rate_raw": summary.get("source_order_inversion_rate_raw"),
        "source_order_kendall_tau_raw": summary.get("source_order_kendall_tau_raw"),
        "body_source_coverage_rate": summary.get("body_source_coverage_rate"),
        "body_ordered_source_coverage_rate": summary.get("body_ordered_source_coverage_rate"),
        "body_source_order_inversion_rate": summary.get("body_source_order_inversion_rate"),
        "body_source_order_kendall_tau": summary.get("body_source_order_kendall_tau"),
        "body_missing_merge_rate_among_covered": summary.get("body_missing_merge_rate_among_covered"),
        "body_wrong_merge_rate_among_generated": summary.get("body_wrong_merge_rate_among_generated"),
        "body_paragraph_count_delta": summary.get("body_paragraph_count_delta"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], errors: list[dict[str, Any]], workers: int) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["root_label"], row["group"], row["variant"]), []).append(row)
    lines = [
        "# Ordered Paragraph Audit Refresh",
        "",
        f"- created_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- refreshed_outputs: {len(rows)}",
        f"- errors: {len(errors)}",
        f"- workers: {workers}",
        "- MinerU/training/generation rerun: No",
        "",
        "| root | group | variant | docs | body cov | ordered body cov | order inversion | Kendall tau | missing | wrong |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (root, group, variant), group_rows in sorted(grouped.items()):
        lines.append(
            "| {root} | {group} | {variant} | {docs} | {body_cov} | {ordered_cov} | {inv} | {tau} | {missing} | {wrong} |".format(
                root=f"`{root}`",
                group=f"`{group}`",
                variant=f"`{variant}`",
                docs=len(group_rows),
                body_cov=fmt(mean(row.get("body_source_coverage_rate") for row in group_rows)),
                ordered_cov=fmt(mean(row.get("body_ordered_source_coverage_rate") for row in group_rows)),
                inv=fmt(mean(row.get("body_source_order_inversion_rate") for row in group_rows)),
                tau=fmt(mean(row.get("body_source_order_kendall_tau") for row in group_rows)),
                missing=fmt(mean(row.get("body_missing_merge_rate_among_covered") for row in group_rows)),
                wrong=fmt(mean(row.get("body_wrong_merge_rate_among_generated") for row in group_rows)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(values: Any) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def fmt(value: Any) -> str:
    return "N/A" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
