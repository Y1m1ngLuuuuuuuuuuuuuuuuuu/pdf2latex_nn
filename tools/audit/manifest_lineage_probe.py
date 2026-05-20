#!/usr/bin/env python3
"""Print manifest lineage: content roots, graph roots, and first sample."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path, PurePosixPath


FILES = [
    "data/00_manifests/v7_floatproxy_adapter_20260516_205926_trainable_recall98.json",
    "data/00_manifests/v7_floatproxy_adapter_20260516_205926_labeled.json",
    "data/00_manifests/v7_floatproxy_adapter_20260516_205926_rebuilt.json",
    "data/00_manifests/v7_clean_expand5000_localmodels_20260518_153355.json",
    "data/00_manifests/v7_clean_expand5000_from_rawmineru_20260520_smoke100.json",
]


def main() -> int:
    for filename in FILES:
        path = Path(filename)
        print("---", filename, "exists", path.exists())
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        docs = payload.get("documents", payload if isinstance(payload, list) else [])
        print("docs", len(docs), "schema", payload.get("schema_version") if isinstance(payload, dict) else None)
        print("success_count", payload.get("success_count") if isinstance(payload, dict) else None)
        content_roots: Counter[str] = Counter()
        graph_roots: Counter[str] = Counter()
        for doc in docs:
            content_roots[root_after(doc.get("content_json") or doc.get("content_json_path") or doc.get("v7_json") or "", "02_mineru_outputs")] += 1
            graph_roots[root_after(doc.get("graph_path") or "", "06_graph_features")] += 1
        print("content_roots", content_roots.most_common(10))
        print("graph_roots", graph_roots.most_common(10))
        if docs:
            first = docs[0]
            keys = ("document_id", "content_json", "graph_path", "orphan_ratio", "candidate_edge_recall")
            print("first", json.dumps({key: first.get(key) for key in keys}, ensure_ascii=False, indent=2))
    return 0


def root_after(path_text: str, marker: str) -> str:
    if not path_text:
        return "missing"
    parts = PurePosixPath(path_text).parts
    if marker not in parts:
        return "other"
    index = parts.index(marker)
    if index + 1 >= len(parts):
        return "missing_after_marker"
    return parts[index + 1]


if __name__ == "__main__":
    raise SystemExit(main())
