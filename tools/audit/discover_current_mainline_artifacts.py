#!/usr/bin/env python3
"""Discover current-mainline PDF2LaTeX artifacts for clean E2E smoke runs.

This is a read-only scanner. It classifies discovered files by document id and
separates current observable-fact-family artifacts from hardcase intermediates
and historical generated-output-only artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DOC_ID_RE = re.compile(r"(?<!\d)(?:\d{4}\.\d{5}|[a-z]+_\d{4}_\d{5})(?!\d)")

ARTIFACT_FIELDS = {
    "original_pdf": ("original.pdf",),
    "middle_json": ("middle.json",),
    "content_list_json": ("content_list.json",),
    "content_list_v2_json": ("content_list_v2.json",),
    "model_json": ("model.json",),
    "document_ir_json": ("document_ir.json",),
    "render_tree_ir_json": ("render_tree_ir.json",),
    "generated_tex": ("generated.tex",),
    "generated_pdf": ("generated.pdf",),
    "gold_comparison": ("gold_comparison_structure.json",),
    "comparison_structure": ("comparison_structure.json",),
    "metrics": ("metrics.json",),
}


@dataclass
class DocArtifacts:
    doc_id: str
    paths: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    source_roots: set[str] = field(default_factory=set)

    def add(self, field: str, path: Path) -> None:
        value = str(path)
        if value not in self.paths[field]:
            self.paths[field].append(value)
        self.source_roots.add(str(infer_artifact_root(path, self.doc_id)))

    def first(self, field: str) -> str:
        values = self.paths.get(field) or []
        return values[0] if values else ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True, help="Root directory to scan. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-unavailable", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs = discover(args.root)
    rows = [row_for_doc(doc, remote_unavailable=args.remote_unavailable) for doc in sorted(docs.values(), key=lambda d: d.doc_id)]

    write_json(args.output_dir / "artifact_discovery.json", rows)
    write_csv(args.output_dir / "artifact_discovery.csv", rows)
    write_report(args.output_dir / "ARTIFACT_DISCOVERY_REPORT.md", rows, roots=args.root, remote_unavailable=args.remote_unavailable)
    print(json.dumps(summary_counts(rows), indent=2, sort_keys=True))
    return 0


def discover(roots: list[Path]) -> dict[str, DocArtifacts]:
    docs: dict[str, DocArtifacts] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or should_skip(path):
                continue
            doc_id = infer_doc_id(path)
            if not doc_id:
                continue
            field = classify_file(path)
            if not field:
                continue
            docs.setdefault(doc_id, DocArtifacts(doc_id=doc_id)).add(field, path)
    return docs


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if "_obsolete" in parts or "_archive" in parts:
        return True
    if "current_mainline_clean_smoke10_20260530" in parts:
        return True
    if "e2e_pipeline_stabilization_20260530" in parts:
        # Previous smoke outputs mix current and historical artifacts; do not
        # use them as clean current-mainline evidence.
        return True
    if path.name.startswith("."):
        return True
    return False


def infer_doc_id(path: Path) -> str | None:
    match = DOC_ID_RE.search(str(path))
    return match.group(0) if match else None


def classify_file(path: Path) -> str | None:
    name = path.name
    if name.endswith("_content_list_v8_contentlist_merge_hint.json") or "content_list_v8" in name:
        return "observable_facts_json"
    if name == "gold_structure.json":
        return "gold_comparison"
    if name == "generated_structure.json":
        return "comparison_structure"
    if name == "structure_metrics.json":
        return "metrics"
    if name.endswith("_content_list_v2.json"):
        return "content_list_v2_json"
    if name.endswith("_content_list.json"):
        return "content_list_json"
    if name.endswith("_middle.json"):
        return "middle_json"
    if name.endswith("_model.json"):
        return "model_json"
    for field, names in ARTIFACT_FIELDS.items():
        if name in names:
            return field
    if name.endswith(".pdf") and infer_doc_id(path) and "generated" not in name.lower():
        return "original_pdf"
    return None


def infer_artifact_root(path: Path, doc_id: str) -> Path:
    for parent in [path.parent, *path.parents]:
        if parent.name == doc_id or parent.name.startswith(doc_id):
            return parent
    return path.parent


def row_for_doc(doc: DocArtifacts, *, remote_unavailable: bool) -> dict[str, Any]:
    roots = sorted(doc.source_roots)
    artifact_root = select_best_artifact_root(doc, roots)
    selected = fields_for_root(doc, artifact_root)
    source_family = classify_source_family_for_root(doc, artifact_root)
    missing = missing_reasons(doc, selected=selected, source_family=source_family, remote_unavailable=remote_unavailable)
    clean_current_candidate = source_family == "current_observable_fact_family" and not any(
        reason
        for reason in missing
        if reason in {"missing_original_pdf", "missing_observable_facts", "missing_document_ir", "missing_render_tree_ir", "missing_generated_tex"}
    )
    return {
        "doc_id": doc.doc_id,
        "original_pdf": selected.get("original_pdf", ""),
        "mineru_raw_root": infer_mineru_raw_root(doc),
        "middle_json": doc.first("middle_json"),
        "content_list_json": doc.first("content_list_json"),
        "content_list_v2_json": doc.first("content_list_v2_json"),
        "model_json": doc.first("model_json"),
        "observable_facts_json": selected.get("observable_facts_json", ""),
        "document_ir_json": selected.get("document_ir_json", ""),
        "render_tree_ir_json": selected.get("render_tree_ir_json", ""),
        "generated_tex": selected.get("generated_tex", ""),
        "generated_pdf": selected.get("generated_pdf", ""),
        "gold_comparison": selected.get("gold_comparison", "") or doc.first("gold_comparison"),
        "comparison_structure": selected.get("comparison_structure", ""),
        "metrics": selected.get("metrics", ""),
        "artifact_root": artifact_root,
        "source_family": source_family,
        "clean_current_candidate": clean_current_candidate,
        "missing_reasons": ";".join(missing),
    }


def classify_source_family_for_root(doc: DocArtifacts, root: str) -> str:
    selected = fields_for_root(doc, root)
    joined = "\n".join([root] + list(selected.values()))
    if "post_audit_20260519/hardcase_intermediates" in joined:
        return "hardcase_intermediate"
    if "local_outputs/" in joined or "_obsolete" in joined or "_archive" in joined:
        return "historical_local_output"
    has_observable = bool(selected.get("observable_facts_json"))
    has_doc_ir = bool(selected.get("document_ir_json"))
    has_render_tree = bool(selected.get("render_tree_ir_json"))
    if has_observable or has_doc_ir or has_render_tree:
        return "current_observable_fact_family"
    return "incomplete"


def infer_mineru_raw_root(doc: DocArtifacts) -> str:
    candidates = []
    for field in ("middle_json", "content_list_json", "content_list_v2_json", "model_json"):
        if doc.first(field):
            candidates.append(str(Path(doc.first(field)).parent))
    for values in doc.paths.values():
        for value in values:
            if "data/02_mineru_outputs" in value:
                candidates.append(str(Path(value).parent))
    return sorted(set(candidates))[0] if candidates else ""


def select_best_artifact_root(doc: DocArtifacts, roots: list[str]) -> str:
    if not roots:
        return ""
    scored = []
    for root in roots:
        fields = fields_for_root(doc, root)
        score = 0
        for field in ("observable_facts_json", "document_ir_json", "render_tree_ir_json", "generated_tex", "original_pdf"):
            score += 10 if fields.get(field) else 0
        if "v8_reflow" in root or "current" in root:
            score += 5
        if "selected200_eval_rerun_v2_20260525/v8_deterministic/e2e_skipcompile" in root:
            score += 30
        if "algorithm_renderer_phase0" in root:
            score -= 8
        if "textfix" in root:
            score += 3
        if "floatskip" in root:
            score += 3
        if "post_audit_20260519/hardcase_intermediates" in root or "local_outputs/" in root:
            score -= 20
        scored.append((score, len(root), root))
    return sorted(scored, key=lambda item: (-item[0], item[1], item[2]))[0][2]


def fields_for_root(doc: DocArtifacts, root: str) -> dict[str, str]:
    selected: dict[str, str] = {}
    for field, values in doc.paths.items():
        in_root = [value for value in values if value.startswith(root.rstrip("/") + "/") or value == root]
        if in_root:
            selected[field] = sorted(in_root, key=len)[0]
    return selected


def missing_reasons(doc: DocArtifacts, *, selected: dict[str, str], source_family: str, remote_unavailable: bool) -> list[str]:
    reasons: list[str] = []
    if remote_unavailable:
        reasons.append("remote_unavailable")
    if not selected.get("original_pdf"):
        reasons.append("missing_original_pdf")
    if not selected.get("observable_facts_json"):
        reasons.append("missing_observable_facts")
    if not selected.get("document_ir_json"):
        reasons.append("missing_document_ir")
    if not selected.get("render_tree_ir_json"):
        reasons.append("missing_render_tree_ir")
    if not selected.get("generated_tex"):
        reasons.append("missing_generated_tex")
    if not doc.first("gold_comparison"):
        reasons.append("missing_gold")
    if not infer_mineru_raw_root(doc):
        reasons.append("missing_mineru_raw")
    if source_family == "historical_local_output":
        reasons.append("historical_artifact_only")
    if source_family == "hardcase_intermediate":
        reasons.append("hardcase_intermediate_not_clean_current")
    return reasons


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "doc_id",
        "original_pdf",
        "mineru_raw_root",
        "middle_json",
        "content_list_json",
        "content_list_v2_json",
        "model_json",
        "observable_facts_json",
        "document_ir_json",
        "render_tree_ir_json",
        "generated_tex",
        "generated_pdf",
        "gold_comparison",
        "comparison_structure",
        "metrics",
        "artifact_root",
        "source_family",
        "clean_current_candidate",
        "missing_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summary_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, int] = defaultdict(int)
    missing: dict[str, int] = defaultdict(int)
    for row in rows:
        by_family[str(row["source_family"])] += 1
        for reason in str(row.get("missing_reasons") or "").split(";"):
            if reason:
                missing[reason] += 1
    return {
        "total_docs_scanned": len(rows),
        "source_family_counts": dict(sorted(by_family.items())),
        "clean_current_candidate": sum(1 for row in rows if row.get("clean_current_candidate")),
        "missing_reason_counts": dict(sorted(missing.items())),
    }


def write_report(path: Path, rows: list[dict[str, Any]], *, roots: list[Path], remote_unavailable: bool) -> None:
    summary = summary_counts(rows)
    lines = [
        "# Artifact Discovery Report",
        "",
        "## Scope",
        *[f"- scanned root: `{root}` ({'exists' if root.exists() else 'missing'})" for root in roots],
        f"- remote_unavailable: {remote_unavailable}",
        "",
        "## Summary",
        f"- total docs scanned: {summary['total_docs_scanned']}",
        f"- clean_current_candidate: {summary['clean_current_candidate']}",
        "",
        "## Source Families",
        *[f"- {key}: {value}" for key, value in summary["source_family_counts"].items()],
        "",
        "## Missing Reasons",
        *[f"- {key}: {value}" for key, value in summary["missing_reason_counts"].items()],
        "",
        "## Clean Candidates",
    ]
    candidates = [row for row in rows if row.get("clean_current_candidate")]
    if not candidates:
        lines.append("- none")
    else:
        lines.extend(
            f"- {row['doc_id']}: `{row['artifact_root']}`"
            for row in candidates
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
