#!/usr/bin/env python3
"""Lightweight audit for how much GNN relations change E2E outputs.

This script is intentionally read-only. It compares an existing GNN relation
source E2E run with rules-only runs and summarizes:

* raw relation probabilities available in ``predicted_relations.json``;
* generated TeX / comparison-structure equality across relation sources;
* metric deltas between GNN and rules-only outputs.

It does not train, rebuild graphs, relabel data, or run inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


METRIC_KEYS = [
    "macro_structure_score",
    "heading_tree_accuracy",
    "reading_order_accuracy",
    "paragraph_boundary_f1",
    "paragraph_text_coverage_f1",
    "paragraph_merge_f1",
    "section_attachment_f1",
    "section_attachment_body_no_float_f1",
    "reference_section_completeness",
    "float_caption_attachment_accuracy",
    "generated_structure_validity",
]

VOLATILE_JSON_KEYS = {
    "source_path",
    "bbl_path",
    "doc_dir",
    "generated_tex",
    "generated_pdf",
    "paired_original_pdf",
    "paired_generated_pdf",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "f1", "accuracy", "value"):
            if key in value:
                return _safe_float(value[key])
    return None


def _metric_value(metrics: dict[str, Any] | None, key: str) -> float | None:
    if not metrics:
        return None
    return _safe_float(metrics.get(key))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _normalize_tex(text: str) -> str:
    # Preserve commands/content but remove formatting-only whitespace differences.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"%[^\n]*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _stable_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: _stable_json(value)
            for key, value in sorted(obj.items())
            if key not in VOLATILE_JSON_KEYS
        }
    if isinstance(obj, list):
        return [_stable_json(value) for value in obj]
    return obj


def _hash_json(path: Path) -> str | None:
    obj = _load_json(path)
    if obj is None:
        return None
    payload = json.dumps(_stable_json(obj), sort_keys=True, ensure_ascii=False)
    return _hash_text(payload)


def _read_tex_hashes(path: Path) -> tuple[str | None, str | None, int]:
    if not path.exists():
        return None, None, 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return _hash_text(text), _hash_text(_normalize_tex(text)), len(text)


def _doc_id_from_dir(path: Path) -> str:
    record = _load_json(path / "e2e_record.json")
    if record and record.get("document_id"):
        return str(record["document_id"])
    name = path.name
    if "_" in name and name.split("_", 1)[0].isdigit():
        return name.split("_", 1)[1]
    return name


def _list_doc_dirs(root: Path) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    if not root.exists():
        return dirs
    for child in sorted(root.iterdir()):
        if child.is_dir():
            dirs[_doc_id_from_dir(child)] = child
    return dirs


def _label_indices(label_names: list[Any]) -> tuple[int | None, int | None]:
    merge_idx = None
    parent_idx = None
    for idx, raw in enumerate(label_names):
        name = str(raw).upper()
        if name == "MERGE":
            merge_idx = idx
        if "PARENT" in name or "ATTACH" in name:
            parent_idx = idx
    return merge_idx, parent_idx


def _bucket_count(values: Iterable[float]) -> dict[str, int]:
    buckets = {
        "0.00-0.25": 0,
        "0.25-0.35": 0,
        "0.35-0.50": 0,
        "0.50-0.75": 0,
        "0.75-1.00": 0,
    }
    for value in values:
        if value < 0.25:
            buckets["0.00-0.25"] += 1
        elif value < 0.35:
            buckets["0.25-0.35"] += 1
        elif value < 0.50:
            buckets["0.35-0.50"] += 1
        elif value < 0.75:
            buckets["0.50-0.75"] += 1
        else:
            buckets["0.75-1.00"] += 1
    return buckets


def _relation_summary(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    if not data:
        return {
            "has_predicted_relations": False,
            "edge_count": 0,
            "raw_merge_argmax": None,
            "raw_parent_argmax": None,
            "threshold_merge_edges": None,
            "threshold_parent_edges": None,
        }
    metadata = data.get("metadata") or {}
    label_names = metadata.get("label_names") or []
    merge_idx, parent_idx = _label_indices(label_names)
    predicted_labels = data.get("predicted_labels") or []
    probabilities = data.get("probabilities") or []
    thresholds = data.get("threshold_config") or {}
    edge_count = len(data.get("edge_ids") or probabilities or predicted_labels)

    def class_probs(index: int | None) -> list[float]:
        if index is None:
            return []
        vals = []
        for row in probabilities:
            if isinstance(row, list) and len(row) > index:
                vals.append(float(row[index]))
        return vals

    merge_probs = class_probs(merge_idx)
    parent_probs = class_probs(parent_idx)
    tau_merge = _safe_float(thresholds.get("merge"))
    tau_parent = _safe_float(thresholds.get("parent_child"))

    raw_merge = None if merge_idx is None else sum(1 for label in predicted_labels if label == merge_idx)
    raw_parent = None if parent_idx is None else sum(1 for label in predicted_labels if label == parent_idx)
    threshold_merge = (
        None
        if merge_idx is None or tau_merge is None
        else sum(1 for value in merge_probs if value >= tau_merge)
    )
    threshold_parent = (
        None
        if parent_idx is None or tau_parent is None
        else sum(1 for value in parent_probs if value >= tau_parent)
    )
    return {
        "has_predicted_relations": True,
        "edge_count": edge_count,
        "label_names": label_names,
        "merge_label_index": merge_idx,
        "parent_label_index": parent_idx,
        "tau_merge": tau_merge,
        "tau_parent": tau_parent,
        "raw_merge_argmax": raw_merge,
        "raw_parent_argmax": raw_parent,
        "threshold_merge_edges": threshold_merge,
        "threshold_parent_edges": threshold_parent,
        "merge_prob_buckets": _bucket_count(merge_probs),
        "parent_prob_buckets": _bucket_count(parent_probs),
        "model_version": data.get("model_version"),
        "graph_input_path": data.get("graph_input_path"),
    }


@dataclass
class RunDoc:
    doc_id: str
    path: Path
    metrics: dict[str, Any] | None
    tex_raw_hash: str | None
    tex_norm_hash: str | None
    tex_chars: int
    structure_hash: str | None
    relation_summary: dict[str, Any] | None


def _load_run_doc(doc_id: str, path: Path, include_relations: bool) -> RunDoc:
    raw_hash, norm_hash, tex_chars = _read_tex_hashes(path / "generated.tex")
    return RunDoc(
        doc_id=doc_id,
        path=path,
        metrics=_load_json(path / "structure_metrics.json"),
        tex_raw_hash=raw_hash,
        tex_norm_hash=norm_hash,
        tex_chars=tex_chars,
        structure_hash=_hash_json(path / "generated_structure.json"),
        relation_summary=_relation_summary(path / "predicted_relations.json") if include_relations else None,
    )


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _compare_doc(
    doc_id: str,
    gnn: RunDoc,
    rules: RunDoc | None,
    deterministic: RunDoc | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "doc_id": doc_id,
        "gnn_tex_chars": gnn.tex_chars,
        "raw_merge_argmax": None,
        "raw_parent_argmax": None,
        "threshold_merge_edges": None,
        "threshold_parent_edges": None,
    }
    rel = gnn.relation_summary or {}
    for key in [
        "edge_count",
        "raw_merge_argmax",
        "raw_parent_argmax",
        "threshold_merge_edges",
        "threshold_parent_edges",
        "tau_merge",
        "tau_parent",
    ]:
        row[key] = rel.get(key)

    for key in METRIC_KEYS:
        gnn_val = _metric_value(gnn.metrics, key)
        row[f"gnn_{key}"] = gnn_val
        if rules:
            rules_val = _metric_value(rules.metrics, key)
            row[f"rules_{key}"] = rules_val
            row[f"delta_gnn_minus_rules_{key}"] = _delta(gnn_val, rules_val)
        if deterministic:
            det_val = _metric_value(deterministic.metrics, key)
            row[f"deterministic_{key}"] = det_val
            row[f"delta_gnn_minus_deterministic_{key}"] = _delta(gnn_val, det_val)

    if rules:
        row["gnn_rules_tex_raw_identical"] = gnn.tex_raw_hash == rules.tex_raw_hash
        row["gnn_rules_tex_normalized_identical"] = gnn.tex_norm_hash == rules.tex_norm_hash
        row["gnn_rules_structure_identical"] = gnn.structure_hash == rules.structure_hash
    if deterministic:
        row["gnn_deterministic_tex_raw_identical"] = gnn.tex_raw_hash == deterministic.tex_raw_hash
        row["gnn_deterministic_tex_normalized_identical"] = gnn.tex_norm_hash == deterministic.tex_norm_hash
        row["gnn_deterministic_structure_identical"] = gnn.structure_hash == deterministic.structure_hash
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _sort_by_delta(rows: list[dict[str, Any]], key: str, limit: int = 8) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: abs(row.get(key) or 0.0),
        reverse=True,
    )[:limit]


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    gnn_dir: Path,
    rules_dir: Path | None,
    deterministic_dir: Path | None,
) -> None:
    doc_count = len(rows)
    raw_identical = sum(1 for r in rows if r.get("gnn_rules_tex_raw_identical") is True)
    norm_identical = sum(1 for r in rows if r.get("gnn_rules_tex_normalized_identical") is True)
    structure_identical = sum(1 for r in rows if r.get("gnn_rules_structure_identical") is True)
    det_norm_identical = sum(
        1 for r in rows if r.get("gnn_deterministic_tex_normalized_identical") is True
    )
    det_structure_identical = sum(
        1 for r in rows if r.get("gnn_deterministic_structure_identical") is True
    )

    metric_rows = []
    for key in [
        "macro_structure_score",
        "heading_tree_accuracy",
        "paragraph_text_coverage_f1",
        "section_attachment_body_no_float_f1",
        "float_caption_attachment_accuracy",
        "generated_structure_validity",
    ]:
        metric_rows.append(
            [
                key,
                _mean(r.get(f"gnn_{key}") for r in rows),
                _mean(r.get(f"rules_{key}") for r in rows),
                _mean(r.get(f"delta_gnn_minus_rules_{key}") for r in rows),
                _mean(r.get(f"deterministic_{key}") for r in rows),
                _mean(r.get(f"delta_gnn_minus_deterministic_{key}") for r in rows),
            ]
        )

    relation_rows = [
        [
            "edge_count",
            _mean(r.get("edge_count") for r in rows),
            max((r.get("edge_count") or 0) for r in rows) if rows else None,
        ],
        [
            "raw_merge_argmax",
            _mean(r.get("raw_merge_argmax") for r in rows),
            sum((r.get("raw_merge_argmax") or 0) for r in rows),
        ],
        [
            "threshold_merge_edges",
            _mean(r.get("threshold_merge_edges") for r in rows),
            sum((r.get("threshold_merge_edges") or 0) for r in rows),
        ],
        [
            "raw_parent_argmax",
            _mean(r.get("raw_parent_argmax") for r in rows),
            sum((r.get("raw_parent_argmax") or 0) for r in rows),
        ],
        [
            "threshold_parent_edges",
            _mean(r.get("threshold_parent_edges") for r in rows),
            sum((r.get("threshold_parent_edges") or 0) for r in rows),
        ],
    ]

    top_macro = _sort_by_delta(rows, "delta_gnn_minus_rules_macro_structure_score", 6)
    top_heading = _sort_by_delta(rows, "delta_gnn_minus_rules_heading_tree_accuracy", 6)
    top_rows = []
    for row in top_macro:
        top_rows.append(
            [
                row["doc_id"],
                row.get("delta_gnn_minus_rules_macro_structure_score"),
                row.get("delta_gnn_minus_rules_heading_tree_accuracy"),
                row.get("threshold_merge_edges"),
                row.get("threshold_parent_edges"),
                row.get("gnn_rules_tex_normalized_identical"),
                row.get("gnn_rules_structure_identical"),
            ]
        )

    heading_rows = []
    for row in top_heading:
        heading_rows.append(
            [
                row["doc_id"],
                row.get("delta_gnn_minus_rules_heading_tree_accuracy"),
                row.get("gnn_heading_tree_accuracy"),
                row.get("rules_heading_tree_accuracy"),
                row.get("threshold_merge_edges"),
                row.get("threshold_parent_edges"),
            ]
        )

    text = f"""# GNN E2E Influence Audit Report

## Status
- Documents compared: {doc_count}
- GNN output dir: `{gnn_dir}`
- Rules-only output dir: `{rules_dir or 'N/A'}`
- Deterministic-merge rules output dir: `{deterministic_dir or 'N/A'}`
- Training / MinerU / relabel / rebuild / API / CompHRDoc: No
- This is a lightweight read-only summary over existing E2E outputs.

## Relation Prediction Availability

{_markdown_table(['quantity', 'mean per doc', 'total / max'], relation_rows)}

Notes:
- `raw_*_argmax` counts argmax labels before decoder constraints.
- `threshold_*_edges` counts class probabilities above the run threshold.
- The existing artifacts do not contain full hard-gate rejection reasons or final RenderTreeIR attribution, so this report cannot yet distinguish every blocked edge from every consumed edge.

## E2E Metric Deltas

{_markdown_table(['metric', 'M06 GNN', 'R00 rules', 'delta GNN-R00', 'R01 rules+det merge', 'delta GNN-R01'], metric_rows)}

## Output Equality

- M06 vs R00 raw TeX identical: {raw_identical}/{doc_count}
- M06 vs R00 normalized TeX identical: {norm_identical}/{doc_count}
- M06 vs R00 comparison-structure identical: {structure_identical}/{doc_count}
- M06 vs R01 normalized TeX identical: {det_norm_identical}/{doc_count}
- M06 vs R01 comparison-structure identical: {det_structure_identical}/{doc_count}

## Largest Macro Deltas: M06 vs R00

{_markdown_table(['doc_id', 'macro delta', 'heading delta', 'threshold MERGE', 'threshold PARENT', 'same normalized TeX', 'same structure'], top_rows)}

## Largest Heading Deltas: M06 vs R00

{_markdown_table(['doc_id', 'heading delta', 'M06 heading', 'R00 heading', 'threshold MERGE', 'threshold PARENT'], heading_rows)}

## Interpretation

The current 20-doc E2E relation-source comparison is almost rules-dominated.
M06 has many raw/threshold relation predictions, but the final document-level
metrics move only slightly. In this lightweight audit, the main evidence to look
at is whether normalized TeX / comparison structures differ. If they are mostly
identical, the GNN relations are either agreeing with deterministic logic or are
being absorbed by decoder safety constraints. If structures differ but metrics
barely move, the current metrics are not very sensitive to those relation-level
changes.

This does not mean the GNN is useless: the edge-level ablation still shows that
geometry/layout, message passing, reading-flow, SciBERT, and the merge gate are
valuable. It means the current E2E stack has a strong full-v7 + deterministic
decoder backbone, and the GNN contribution should be audited on harder,
GNN-sensitive documents before making broad claims.

## Suggested Next Step

Do not add more broad scripts yet. If we need a deeper pass, add one trace hook
inside decoder/postprocess to record accepted MERGE components and final parent
source attribution, then rerun only the same 20 docs.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-dir", required=True, type=Path)
    parser.add_argument("--rules-dir", type=Path)
    parser.add_argument("--deterministic-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--doc-ids", nargs="*")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gnn_docs = _list_doc_dirs(args.gnn_dir)
    rules_docs = _list_doc_dirs(args.rules_dir) if args.rules_dir else {}
    deterministic_docs = _list_doc_dirs(args.deterministic_dir) if args.deterministic_dir else {}

    doc_ids = sorted(gnn_docs)
    if args.doc_ids:
        wanted = set(args.doc_ids)
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in wanted]
    if args.limit is not None:
        doc_ids = doc_ids[: args.limit]

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "gnn_dir": str(args.gnn_dir),
        "rules_dir": str(args.rules_dir) if args.rules_dir else None,
        "deterministic_dir": str(args.deterministic_dir) if args.deterministic_dir else None,
        "documents": {},
    }
    for doc_id in doc_ids:
        gnn = _load_run_doc(doc_id, gnn_docs[doc_id], include_relations=True)
        rules = (
            _load_run_doc(doc_id, rules_docs[doc_id], include_relations=False)
            if doc_id in rules_docs
            else None
        )
        deterministic = (
            _load_run_doc(doc_id, deterministic_docs[doc_id], include_relations=False)
            if doc_id in deterministic_docs
            else None
        )
        row = _compare_doc(doc_id, gnn, rules, deterministic)
        rows.append(row)
        details["documents"][doc_id] = {
            "row": row,
            "relation_summary": gnn.relation_summary,
            "paths": {
                "gnn": str(gnn.path),
                "rules": str(rules.path) if rules else None,
                "deterministic": str(deterministic.path) if deterministic else None,
            },
        }

    _write_csv(args.output_dir / "gnn_e2e_influence_summary.csv", rows)
    (args.output_dir / "gnn_e2e_influence_summary.json").write_text(
        json.dumps({"rows": rows, "details": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        args.output_dir / "GNN_E2E_INFLUENCE_AUDIT_REPORT.md",
        rows,
        args.gnn_dir,
        args.rules_dir,
        args.deterministic_dir,
    )
    print(f"Wrote {len(rows)} doc summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
