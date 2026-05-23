#!/usr/bin/env python3
"""Audit MinerU middle.json continuation evidence without changing main data.

This tool reads MinerU ``*_middle.json`` plus optional v7/graph artifacts and
emits a sidecar describing paragraph/line fragments that MinerU had before
``content_list`` collapsed them into higher-level blocks.  It is intentionally
audit-only: it never rewrites v7 JSON, graphs, labels, or generator outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapters.mineru_v7_document_ir import stable_node_id  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/09_eval_reports/mineru_middle_continuation_audit_20260523"


@dataclass(frozen=True)
class DocInput:
    doc_id: str
    middle_json: Path | None = None
    v7_json: Path | None = None
    graph_path: Path | None = None
    mapping_path: Path | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Manifest with doc_id and optional artifact paths")
    parser.add_argument("--doc-ids", nargs="+", default=[], help="Explicit doc ids")
    parser.add_argument("--doc-ids-file", type=Path, help="One doc id per line")
    parser.add_argument("--mineru-root", type=Path, help="Root containing <doc_id>/auto/*_middle.json")
    parser.add_argument("--v7-root", type=Path, help="Root containing <doc_id>/auto/*_content_list_v7_styles.json")
    parser.add_argument("--graph-root", type=Path, help="Optional root containing graph .pt files")
    parser.add_argument("--mapping-root", type=Path, help="Optional root containing alignment mapping JSON files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--middle-block-source",
        choices=("para_blocks", "preproc_blocks"),
        default="para_blocks",
        help="Which middle.json page block list to audit.",
    )
    return parser


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_path(value: Any, *, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def doc_id_from_item(item: dict[str, Any]) -> str:
    for key in ("doc_id", "document_id", "paper_id", "arxiv_id", "id"):
        value = item.get(key)
        if value:
            return str(value)
    for key in ("content_json", "content_json_path", "middle_json", "pdf_path", "graph_path"):
        value = item.get(key)
        if value:
            name = Path(str(value)).name
            return re.sub(r"_(?:content_list.*|middle|v7.*|graph).*", "", name)
    raise ValueError(f"manifest item has no document id: {item}")


def manifest_items(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "documents", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Unsupported manifest format: {path}")


def read_doc_ids_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def discover_doc_ids(mineru_root: Path | None) -> list[str]:
    if mineru_root is None or not mineru_root.exists():
        return []
    ids: set[str] = set()
    for middle in mineru_root.glob("*/auto/*_middle.json"):
        ids.add(middle.parent.parent.name)
    return sorted(ids)


def make_doc_inputs(args: argparse.Namespace) -> list[DocInput]:
    root = Path.cwd()
    by_id: dict[str, DocInput] = {}
    manifest_records = manifest_items(args.manifest)
    explicit_doc_ids = [*args.doc_ids, *read_doc_ids_file(args.doc_ids_file)]
    for item in manifest_records:
        doc_id = doc_id_from_item(item)
        by_id[doc_id] = DocInput(
            doc_id=doc_id,
            middle_json=first_existing(
                resolve_path(item.get("middle_json") or item.get("middle_path"), root=root),
                find_middle_json(doc_id, args.mineru_root, root=root),
            ),
            v7_json=first_existing(
                resolve_path(item.get("content_json") or item.get("content_json_path") or item.get("v7_json"), root=root),
                find_v7_json(doc_id, args.v7_root or args.mineru_root, root=root),
            ),
            graph_path=first_existing(
                resolve_path(item.get("graph_path") or item.get("graph_pt"), root=root),
                find_graph_path(doc_id, args.graph_root, root=root),
            ),
            mapping_path=first_existing(
                resolve_path(item.get("mapping_path") or item.get("alignment_mapping"), root=root),
                find_mapping_path(doc_id, args.mapping_root, root=root),
            ),
        )
    auto_discovered = [] if (manifest_records or explicit_doc_ids) else discover_doc_ids(args.mineru_root)
    for doc_id in [*explicit_doc_ids, *auto_discovered]:
        by_id.setdefault(
            doc_id,
            DocInput(
                doc_id=doc_id,
                middle_json=find_middle_json(doc_id, args.mineru_root, root=root),
                v7_json=find_v7_json(doc_id, args.v7_root or args.mineru_root, root=root),
                graph_path=find_graph_path(doc_id, args.graph_root, root=root),
                mapping_path=find_mapping_path(doc_id, args.mapping_root, root=root),
            ),
        )
    docs = [by_id[key] for key in sorted(by_id)]
    if args.limit is not None:
        docs = docs[: max(0, int(args.limit))]
    return docs


def first_existing(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def find_middle_json(doc_id: str, root_path: Path | None, *, root: Path) -> Path | None:
    if root_path is None:
        return None
    base = root_path if root_path.is_absolute() else root / root_path
    candidates = [
        base / doc_id / "auto" / f"{doc_id}_middle.json",
        base / doc_id / f"{doc_id}_middle.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted((base / doc_id).glob("**/*_middle.json")) if (base / doc_id).exists() else []
    return matches[0] if matches else None


def find_v7_json(doc_id: str, root_path: Path | None, *, root: Path) -> Path | None:
    if root_path is None:
        return None
    base = root_path if root_path.is_absolute() else root / root_path
    candidates = [
        base / doc_id / "auto" / f"{doc_id}_content_list_v7_styles.json",
        base / doc_id / "auto" / f"{doc_id}_content_list_v7.json",
        base / doc_id / f"{doc_id}_content_list_v7_styles.json",
        base / doc_id / f"{doc_id}_content_list_v7.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted((base / doc_id).glob("**/*_content_list_v7_styles.json")) if (base / doc_id).exists() else []
    return matches[0] if matches else None


def find_graph_path(doc_id: str, root_path: Path | None, *, root: Path) -> Path | None:
    if root_path is None:
        return None
    base = root_path if root_path.is_absolute() else root / root_path
    patterns = [
        f"{doc_id}_v7_truthgen_labeled_graph.pt",
        f"{doc_id}_v7_graph.pt",
        f"{doc_id}*.pt",
    ]
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_mapping_path(doc_id: str, root_path: Path | None, *, root: Path) -> Path | None:
    if root_path is None:
        return None
    base = root_path if root_path.is_absolute() else root / root_path
    patterns = [
        f"{doc_id}_v7_alignment_mapping.json",
        f"{doc_id}*mapping*.json",
        f"{doc_id}*.json",
    ]
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def middle_blocks(payload: Any, *, source: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("middle.json must be an object with pdf_info")
    pdf_info = payload.get("pdf_info")
    if not isinstance(pdf_info, list):
        raise ValueError("middle.json missing pdf_info list")
    blocks: list[dict[str, Any]] = []
    for page_idx, page in enumerate(pdf_info):
        if not isinstance(page, dict):
            continue
        page_blocks = page.get(source)
        if not isinstance(page_blocks, list):
            continue
        for local_index, block in enumerate(page_blocks):
            if not isinstance(block, dict):
                continue
            record = dict(block)
            record["_middle_page_idx"] = int(record.get("page_idx") if isinstance(record.get("page_idx"), int) else page_idx)
            record["_middle_local_index"] = local_index
            blocks.append(record)
    return blocks


def extract_middle_logical_blocks(payload: Any, *, doc_id: str, source: str) -> list[dict[str, Any]]:
    logical: list[dict[str, Any]] = []
    for block in middle_blocks(payload, source=source):
        page_idx = int(block["_middle_page_idx"])
        block_index = block.get("index")
        if not isinstance(block_index, int):
            block_index = int(block["_middle_local_index"])
        fragments = extract_fragments(block, page_idx=page_idx)
        text = normalize_space(" ".join(fragment["text"] for fragment in fragments if fragment["text"]))
        has_cross_page = bool(block.get("cross_page")) or any(bool(fragment.get("cross_page")) for fragment in fragments)
        has_cross_column = bool(block.get("cross_column")) or any(bool(fragment.get("cross_column")) for fragment in fragments)
        logical.append(
            {
                "middle_block_id": f"p{page_idx:04d}_b{block_index:06d}",
                "doc_id": doc_id,
                "source": source,
                "source_page_idx": page_idx,
                "middle_index": block_index,
                "type": str(block.get("type") or ""),
                "score": block.get("score"),
                "bbox": list(block.get("bbox") or []),
                "line_count": len(block.get("lines") or []),
                "span_count": len(fragments),
                "text": text,
                "has_cross_page": has_cross_page,
                "has_cross_column": has_cross_column,
                "fragments": fragments,
            }
        )
    return logical


def extract_fragments(block: dict[str, Any], *, page_idx: int) -> list[dict[str, Any]]:
    lines = block.get("lines")
    fragments: list[dict[str, Any]] = []
    if isinstance(lines, list):
        for line_idx, line in enumerate(lines):
            if not isinstance(line, dict):
                continue
            spans = line.get("spans")
            if not isinstance(spans, list):
                spans = []
            if not spans:
                value = text_from_any(line)
                if value:
                    fragments.append(
                        {
                            "page_idx": page_idx,
                            "line_idx": line_idx,
                            "span_idx": None,
                            "type": str(line.get("type") or ""),
                            "text": value,
                            "bbox": list(line.get("bbox") or []),
                            "cross_page": bool(line.get("cross_page")),
                            "cross_column": bool(line.get("cross_column")),
                            "score": line.get("score"),
                        }
                    )
                continue
            for span_idx, span in enumerate(spans):
                if not isinstance(span, dict):
                    continue
                value = text_from_any(span)
                if not value:
                    continue
                fragments.append(
                    {
                        "page_idx": page_idx,
                        "line_idx": line_idx,
                        "span_idx": span_idx,
                        "type": str(span.get("type") or line.get("type") or ""),
                        "text": value,
                        "bbox": list(span.get("bbox") or line.get("bbox") or []),
                        "line_bbox": list(line.get("bbox") or []),
                        "cross_page": bool(span.get("cross_page") or line.get("cross_page")),
                        "cross_column": bool(span.get("cross_column") or line.get("cross_column")),
                        "score": span.get("score", line.get("score")),
                    }
                )
    if fragments:
        return fragments
    value = text_from_any(block)
    return [
        {
            "page_idx": page_idx,
            "line_idx": None,
            "span_idx": None,
            "type": str(block.get("type") or ""),
            "text": value,
            "bbox": list(block.get("bbox") or []),
            "cross_page": bool(block.get("cross_page")),
            "cross_column": bool(block.get("cross_column")),
            "score": block.get("score"),
        }
    ] if value else []


def text_from_any(value: Any) -> str:
    if isinstance(value, str):
        return normalize_space(value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("content", "text", "latex", "html"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                parts.append(inner)
        if parts:
            return normalize_space(" ".join(parts))
        for key in ("spans", "lines", "children"):
            inner = value.get(key)
            if isinstance(inner, list):
                nested = [text_from_any(item) for item in inner]
                nested = [item for item in nested if item]
                if nested:
                    return normalize_space(" ".join(nested))
    if isinstance(value, list):
        nested = [text_from_any(item) for item in value]
        return normalize_space(" ".join(item for item in nested if item))
    return ""


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def compact_text(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(text or "").casefold())


def v7_items(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = load_json(path)
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        record = dict(item)
        record["_v7_node_id"] = stable_node_id(record, fallback_position=index)
        record["_v7_index"] = index
        normalized.append(record)
    return normalized


def v7_text(item: dict[str, Any]) -> str:
    for key in ("text_for_embedding", "text", "content", "latex", "text_preview"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_space(value)
    return ""


def map_middle_to_v7(block: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"mapped_v7_ids": [], "mapped_v7_indices": [], "method": "missing_v7", "score": 0.0}
    page_idx = block["source_page_idx"]
    middle_index = block["middle_index"]
    exact = [
        item
        for item in items
        if int_or_none(item.get("mineru_page_idx"), item.get("page_idx")) == page_idx
        and int_or_none(item.get("mineru_block_idx"), item.get("original_index")) == middle_index
    ]
    if exact:
        return mapping_payload(exact, method="page_block_index", score=1.0)

    block_compact = compact_text(block.get("text") or "")
    if len(block_compact) < 8:
        return {"mapped_v7_ids": [], "mapped_v7_indices": [], "method": "empty_or_short_text", "score": 0.0}
    candidates = [
        item for item in items if int_or_none(item.get("page_idx"), item.get("mineru_page_idx")) in {page_idx, page_idx + 1}
    ]
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        item_compact = compact_text(v7_text(item))
        if len(item_compact) < 8:
            continue
        score = containment_similarity(block_compact, item_compact)
        if score < 0.55:
            score = SequenceMatcher(None, block_compact[:500], item_compact[:500]).ratio()
        scored.append((score, item))
    if not scored:
        return {"mapped_v7_ids": [], "mapped_v7_indices": [], "method": "no_text_candidate", "score": 0.0}
    scored.sort(key=lambda entry: entry[0], reverse=True)
    best_score, best = scored[0]
    if best_score < 0.65:
        return {"mapped_v7_ids": [], "mapped_v7_indices": [], "method": "low_text_similarity", "score": round(float(best_score), 4)}
    return mapping_payload([best], method="text_similarity", score=best_score)


def containment_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left in right:
        return min(1.0, len(left) / max(len(right), 1) + 0.25)
    if right in left:
        return min(1.0, len(right) / max(len(left), 1) + 0.25)
    prefix = min(len(left), len(right), 400)
    return SequenceMatcher(None, left[:prefix], right[:prefix]).ratio()


def mapping_payload(items: list[dict[str, Any]], *, method: str, score: float) -> dict[str, Any]:
    return {
        "mapped_v7_ids": [str(item["_v7_node_id"]) for item in items],
        "mapped_v7_indices": [int(item["_v7_index"]) for item in items],
        "method": method,
        "score": round(float(score), 4),
    }


def int_or_none(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int):
            return value
        try:
            if value is not None and str(value).strip():
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def load_graph_bridge(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False, "v7_id_to_gnn_idx": {}, "edge_pairs": set()}
    try:
        import torch

        graph = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "v7_id_to_gnn_idx": {}, "edge_pairs": set()}

    v7_id_to_gnn_idx: dict[str, int] = {}
    existing = getattr(graph, "v7_id_to_gnn_idx", None)
    if isinstance(existing, dict):
        v7_id_to_gnn_idx = {str(key): int(value) for key, value in existing.items()}
    else:
        gnn_to_v7_ids = getattr(graph, "gnn_to_v7_ids", None)
        if isinstance(gnn_to_v7_ids, list):
            for gnn_idx, ids in enumerate(gnn_to_v7_ids):
                if isinstance(ids, list):
                    for value in ids:
                        v7_id_to_gnn_idx[str(value)] = int(gnn_idx)
                elif ids is not None:
                    v7_id_to_gnn_idx[str(ids)] = int(gnn_idx)
        else:
            gnn_to_v7_id = getattr(graph, "gnn_to_v7_id", None)
            if isinstance(gnn_to_v7_id, list):
                v7_id_to_gnn_idx = {str(value): int(index) for index, value in enumerate(gnn_to_v7_id)}
    edge_pairs: set[tuple[int, int]] = set()
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None:
        matrix = edge_index.detach().cpu().tolist() if hasattr(edge_index, "detach") else edge_index
        if isinstance(matrix, list) and len(matrix) == 2:
            for src, dst in zip(matrix[0], matrix[1]):
                edge_pairs.add((int(src), int(dst)))
    return {
        "available": True,
        "node_count": int(getattr(graph, "num_nodes", 0) or 0),
        "edge_count": len(edge_pairs),
        "v7_id_to_gnn_idx": v7_id_to_gnn_idx,
        "edge_pairs": edge_pairs,
        "graph_path": str(path),
    }


def graph_has_candidate_edge(v7_ids: list[str], bridge: dict[str, Any]) -> bool | None:
    if not bridge.get("available"):
        return None
    mapping: dict[str, int] = bridge.get("v7_id_to_gnn_idx", {})
    edge_pairs: set[tuple[int, int]] = bridge.get("edge_pairs", set())
    gnn_ids = [mapping[value] for value in v7_ids if value in mapping]
    if len(gnn_ids) < 2:
        return False
    for i, src in enumerate(gnn_ids):
        for dst in gnn_ids[i + 1 :]:
            if (src, dst) in edge_pairs or (dst, src) in edge_pairs:
                return True
    return False


def classify_continuation(block: dict[str, Any], mapped_v7_ids: list[str], *, graph_edge: bool | None) -> str:
    mapped = len(mapped_v7_ids)
    if mapped == 0:
        return "uncertain_middle_mapping"
    if mapped == 1:
        if block.get("has_cross_page"):
            return "cross_page_premerged"
        if block.get("has_cross_column"):
            return "cross_column_premerged"
        if int(block.get("span_count") or 0) > 1 or int(block.get("line_count") or 0) > 1:
            return "premerged_single_v7"
        return "single_v7_no_continuation"
    if graph_edge is True:
        return "split_v7_merge"
    if graph_edge is False:
        return "split_v7_no_candidate_edge"
    return "split_v7_multiple_no_graph"


def audit_doc(doc: DocInput, *, output_dir: Path, middle_block_source: str) -> dict[str, Any]:
    if doc.middle_json is None or not doc.middle_json.exists():
        raise FileNotFoundError(f"missing middle.json for {doc.doc_id}")
    middle_payload = load_json(doc.middle_json)
    logical_blocks = extract_middle_logical_blocks(middle_payload, doc_id=doc.doc_id, source=middle_block_source)
    items = v7_items(doc.v7_json)
    bridge = load_graph_bridge(doc.graph_path)

    enriched_blocks: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    cross_page_blocks = 0
    cross_column_blocks = 0
    for block in logical_blocks:
        mapping = map_middle_to_v7(block, items)
        mapped_ids = list(mapping["mapped_v7_ids"])
        graph_edge = graph_has_candidate_edge(mapped_ids, bridge)
        kind = classify_continuation(block, mapped_ids, graph_edge=graph_edge)
        class_counts[kind] += 1
        cross_page_blocks += int(bool(block.get("has_cross_page")))
        cross_column_blocks += int(bool(block.get("has_cross_column")))
        enriched = {
            **block,
            **mapping,
            "continuation_kind": kind,
            "graph_candidate_edge_between_mapped_v7": graph_edge,
        }
        enriched_blocks.append(enriched)

    sidecar = {
        "schema_version": "mineru_middle_continuation_sidecar_v1",
        "doc_id": doc.doc_id,
        "middle_json": str(doc.middle_json),
        "v7_json": str(doc.v7_json) if doc.v7_json else None,
        "graph_path": str(doc.graph_path) if doc.graph_path else None,
        "middle_block_source": middle_block_source,
        "logical_blocks": enriched_blocks,
    }
    summary = {
        "doc_id": doc.doc_id,
        "middle_json": str(doc.middle_json),
        "v7_json": str(doc.v7_json) if doc.v7_json else None,
        "graph_path": str(doc.graph_path) if doc.graph_path else None,
        "middle_block_source": middle_block_source,
        "middle_block_count": len(logical_blocks),
        "v7_item_count": len(items),
        "graph_available": bool(bridge.get("available")),
        "graph_node_count": int(bridge.get("node_count") or 0),
        "graph_edge_count": int(bridge.get("edge_count") or 0),
        "cross_page_block_count": cross_page_blocks,
        "cross_column_block_count": cross_column_blocks,
        **{f"kind_{key}": int(value) for key, value in sorted(class_counts.items())},
    }
    doc_dir = output_dir / "per_doc" / safe_name(doc.doc_id)
    dump_json(doc_dir / "middle_continuation_sidecar.json", sidecar)
    dump_json(doc_dir / "middle_continuation_audit.json", summary)
    return summary


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, *, rows: list[dict[str, Any]], errors: list[dict[str, Any]], output_dir: Path) -> None:
    totals: Counter[str] = Counter()
    for row in rows:
        for key, value in row.items():
            if key.startswith("kind_") or key in {"middle_block_count", "cross_page_block_count", "cross_column_block_count"}:
                totals[key] += int(value or 0)
    lines = [
        "# MinerU Middle Continuation Audit",
        "",
        "## Status",
        f"- docs attempted: {len(rows) + len(errors)}",
        f"- docs succeeded: {len(rows)}",
        f"- docs failed: {len(errors)}",
        f"- output dir: `{output_dir}`",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in ["middle_block_count", "cross_page_block_count", "cross_column_block_count"]:
        lines.append(f"| {key} | {totals.get(key, 0)} |")
    for key, value in sorted(totals.items()):
        if key.startswith("kind_"):
            lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- `cross_page_premerged` / `cross_column_premerged`: MinerU middle evidence says the logical block spans page/column regions, but v7 has a single logical item. These should explain sparse MERGE labels rather than create new MERGE positives.",
            "- `split_v7_merge`: middle indicates one logical block and graph has candidate edges between mapped v7 nodes. These are candidates for strong MERGE supervision.",
            "- `split_v7_no_candidate_edge`: middle indicates one logical block, but graph did not generate a candidate edge. These point to candidate-edge generation, not label policy.",
            "- `uncertain_middle_mapping`: middle block could not be mapped to v7 reliably; do not train on it until mapping is improved.",
            "",
            "## Artifacts",
            "- per-doc sidecars: `per_doc/<doc_id>/middle_continuation_sidecar.json`",
            "- per-doc summaries: `per_doc/<doc_id>/middle_continuation_audit.json`",
            "- summary JSON: `summary.json`",
            "- summary CSV: `summary.csv`",
        ]
    )
    if errors:
        lines.extend(["", "## Errors", ""])
        for error in errors[:20]:
            lines.append(f"- `{error.get('doc_id')}`: {error.get('error')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    docs = make_doc_inputs(args)
    if not docs:
        raise SystemExit("No documents selected. Provide --manifest, --doc-ids, --doc-ids-file, or --mineru-root.")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        try:
            rows.append(audit_doc(doc, output_dir=output_dir, middle_block_source=args.middle_block_source))
        except Exception as exc:
            errors.append({"doc_id": doc.doc_id, "error": f"{type(exc).__name__}: {exc}"})
        if index % 25 == 0:
            print(f"[middle-audit] processed={index}/{len(docs)} ok={len(rows)} errors={len(errors)}", flush=True)

    payload = {
        "schema_version": "mineru_middle_continuation_audit_summary_v1",
        "output_dir": str(output_dir),
        "middle_block_source": args.middle_block_source,
        "doc_count": len(rows),
        "error_count": len(errors),
        "rows": rows,
        "errors": errors,
    }
    dump_json(output_dir / "summary.json", payload)
    write_summary_csv(output_dir / "summary.csv", rows)
    write_report(output_dir / "MINERU_MIDDLE_CONTINUATION_AUDIT_REPORT.md", rows=rows, errors=errors, output_dir=output_dir)
    print(f"[middle-audit] wrote {output_dir}")
    print(f"[middle-audit] docs_ok={len(rows)} errors={len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
