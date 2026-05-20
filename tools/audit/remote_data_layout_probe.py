#!/usr/bin/env python3
"""Probe remote AutoDL data layout and v7 content schema coverage."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/root/autodl-tmp/pdf2latex_nn")
MINERU_ROOT = PROJECT_ROOT / "data" / "02_mineru_outputs"


def main() -> int:
    rows = []
    for root in sorted([p for p in MINERU_ROOT.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime):
        doc_dirs = [p for p in root.iterdir() if p.is_dir()]
        content_paths = find_content_paths(root, limit=80)
        metrics = [content_metrics(path) for path in content_paths[:50]]
        rows.append(
            {
                "dir": root.name,
                "mtime": root.stat().st_mtime,
                "doc_dir_count": len(doc_dirs),
                "sample_content_count": len(metrics),
                "layout_layer_coverage_median": median([m["layout_layer_coverage"] for m in metrics]),
                "layout_role_coverage_median": median([m["layout_role_coverage"] for m in metrics]),
                "canonical_type_coverage_median": median([m["canonical_type_coverage"] for m in metrics]),
                "style_spans_coverage_median": median([m["style_spans_coverage"] for m in metrics]),
                "item_count_median": median([m["item_count"] for m in metrics]),
                "sample_paths": [str(path.relative_to(PROJECT_ROOT)) for path in content_paths[:3]],
            }
        )
    report = {
        "project_root": str(PROJECT_ROOT),
        "mineru_root": str(MINERU_ROOT),
        "mineru_output_dirs": rows,
        "manifests": list_dir(PROJECT_ROOT / "data" / "00_manifests", limit=80),
        "graph_feature_dirs": list_dir(PROJECT_ROOT / "data" / "06_graph_features", limit=80),
        "ground_truth_ir_dirs": list_dir(PROJECT_ROOT / "data" / "04_ground_truth_ir", limit=40),
        "recent_eval_reports": list_dir(PROJECT_ROOT / "data" / "09_eval_reports", limit=80),
    }
    out_dir = PROJECT_ROOT / "data" / "09_eval_reports" / "expansion_path_audit_20260519"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REMOTE_DATA_LAYOUT_PROBE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REMOTE_DATA_LAYOUT_PROBE.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"dirs": len(rows), "output": str(out_dir / "REMOTE_DATA_LAYOUT_PROBE.md")}, ensure_ascii=False, indent=2))
    return 0


def find_content_paths(root: Path, *, limit: int) -> list[Path]:
    paths: list[Path] = []
    for doc_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        for candidate in (
            doc_dir / "auto" / f"{doc_dir.name}_content_list_v7_styles.json",
            doc_dir / "auto" / f"{doc_dir.name}_content_list_v7.json",
            doc_dir / f"{doc_dir.name}_content_list_v7_styles.json",
            doc_dir / f"{doc_dir.name}_content_list_v7.json",
        ):
            if candidate.exists():
                paths.append(candidate)
                break
        if len(paths) >= limit:
            break
    return paths


def content_metrics(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return empty_metrics(path)
    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        items = []
    return {
        "path": str(path),
        "item_count": len(items),
        "layout_layer_coverage": coverage(items, "layout_layer"),
        "layout_role_coverage": coverage(items, "layout_role"),
        "canonical_type_coverage": coverage(items, "canonical_type"),
        "style_spans_coverage": style_coverage(items),
        "type_counts": dict(Counter(str(item.get("type")) for item in items if isinstance(item, dict)).most_common(8)),
    }


def empty_metrics(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "item_count": 0,
        "layout_layer_coverage": 0.0,
        "layout_role_coverage": 0.0,
        "canonical_type_coverage": 0.0,
        "style_spans_coverage": 0.0,
        "type_counts": {},
    }


def coverage(items: list[Any], key: str) -> float:
    dict_items = [item for item in items if isinstance(item, dict)]
    if not dict_items:
        return 0.0
    return sum(1 for item in dict_items if item.get(key) not in (None, "")) / len(dict_items)


def style_coverage(items: list[Any]) -> float:
    dict_items = [item for item in items if isinstance(item, dict)]
    if not dict_items:
        return 0.0
    return sum(1 for item in dict_items if item.get("style_spans") or item.get("spans")) / len(dict_items)


def median(values: list[float | int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def list_dir(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = sorted(path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return [
        {
            "name": entry.name,
            "type": "dir" if entry.is_dir() else "file",
            "mtime": entry.stat().st_mtime,
        }
        for entry in entries
    ]


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Remote Data Layout Probe",
        "",
        f"- project_root: `{report['project_root']}`",
        f"- mineru_root: `{report['mineru_root']}`",
        "",
        "## MinerU Output Directories",
        "",
        "| dir | doc dirs | sample content | layer | role | canonical | style | median items | examples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["mineru_output_dirs"]:
        examples = "<br>".join(row["sample_paths"])
        lines.append(
            "| {dir} | {doc_dir_count} | {sample_content_count} | {layer:.3f} | {role:.3f} | {canonical:.3f} | {style:.3f} | {items} | {examples} |".format(
                dir=row["dir"],
                doc_dir_count=row["doc_dir_count"],
                sample_content_count=row["sample_content_count"],
                layer=row["layout_layer_coverage_median"] or 0.0,
                role=row["layout_role_coverage_median"] or 0.0,
                canonical=row["canonical_type_coverage_median"] or 0.0,
                style=row["style_spans_coverage_median"] or 0.0,
                items=row["item_count_median"],
                examples=examples,
            )
        )
    lines.extend(["", "## Recent Manifests", ""])
    for item in report["manifests"][:30]:
        lines.append(f"- `{item['name']}` ({item['type']})")
    lines.extend(["", "## Recent Eval Reports", ""])
    for item in report["recent_eval_reports"][:30]:
        lines.append(f"- `{item['name']}` ({item['type']})")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
