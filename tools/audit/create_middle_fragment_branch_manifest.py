#!/usr/bin/env python3
"""Create a graph-build manifest for the middle-fragment branch.

The manifest points graph construction at pseudo-v7 fragment JSON files while
preserving links back to the source v7/middle artifacts and existing TeX/PDF
paths when a base manifest provides them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment-output-dir", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, help="Optional existing manifest with tex/pdf/source paths.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", default="middlefrag_selected200_20260523")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    fragment_dir = resolve(args.fragment_output_dir)
    base_by_id = load_base_manifest(args.base_manifest)
    docs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for doc_dir in sorted((fragment_dir / "per_doc").iterdir() if (fragment_dir / "per_doc").exists() else []):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        content_json = doc_dir / f"{doc_id}_middle_fragment_content_v7_styles.json"
        labels_json = doc_dir / "middle_fragment_merge_labels.json"
        summary_json = doc_dir / "middle_fragment_summary.json"
        view_json = doc_dir / "middle_fragment_view.json"
        if not content_json.exists() or not labels_json.exists():
            errors.append({"doc_id": doc_id, "error": "missing_fragment_content_or_labels", "doc_dir": str(doc_dir)})
            continue
        base = dict(base_by_id.get(doc_id, {}))
        summary = load_json(summary_json) if summary_json.exists() else {}
        record = {
            **base,
            "document_id": doc_id,
            "id": doc_id,
            "content_json": str(content_json.resolve()),
            "tex_path": base.get("tex_path") or "__middle_fragment_branch_no_tex_required__",
            "pdf_path": base.get("pdf_path"),
            "source_content_json": base.get("content_json") or summary.get("v7_json"),
            "source_graph_path": base.get("graph_path"),
            "middle_json": summary.get("middle_json"),
            "middle_fragment_view": str(view_json.resolve()) if view_json.exists() else None,
            "middle_fragment_merge_labels": str(labels_json.resolve()),
            "middle_fragment_summary": str(summary_json.resolve()) if summary_json.exists() else None,
            "middle_fragment_run_name": args.run_name,
            "fragment_count": summary.get("fragment_count"),
            "positive_merge_edge_count": summary.get("positive_merge_edge_count"),
        }
        docs.append(record)
    payload = {
        "schema_version": "middle_fragment_branch_manifest_v1",
        "run_name": args.run_name,
        "fragment_output_dir": str(fragment_dir),
        "base_manifest": str(args.base_manifest) if args.base_manifest else None,
        "num_documents": len(docs),
        "num_errors": len(errors),
        "documents": docs,
        "errors": errors,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("schema_version", "run_name", "num_documents", "num_errors")}, indent=2))
    return 0 if docs else 2


def load_base_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    actual = resolve(path)
    if not actual.exists():
        return {}
    payload = load_json(actual)
    records = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        doc_id = str(record.get("document_id") or record.get("doc_id") or record.get("id") or "")
        if not doc_id:
            content = str(record.get("content_json") or record.get("graph_path") or "")
            doc_id = infer_doc_id_from_path(content)
        if doc_id:
            by_id[doc_id] = record
    return by_id


def infer_doc_id_from_path(value: str) -> str:
    name = Path(value).name
    return re.sub(r"_(?:content_list.*|middle_fragment.*|v7.*|graph.*|relabel.*).*", "", name)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
