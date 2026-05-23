#!/usr/bin/env python3
"""Build middle.json-derived fragment views without changing the main v7 path.

Outputs are intentionally sidecar artifacts.  They can be inspected directly or
used later as an explicit branch input for SciBERT/GNN graph construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.perception.middle_fragment_view import build_middle_fragment_view  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/09_eval_reports/middle_fragment_view_20260523"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Manifest with doc_id and optional artifact paths.")
    parser.add_argument("--doc-ids", nargs="+", default=[], help="Explicit doc ids.")
    parser.add_argument("--doc-ids-file", type=Path, help="One doc id per line.")
    parser.add_argument("--mineru-root", type=Path, help="Root containing <doc_id>/auto/*_middle.json.")
    parser.add_argument("--v7-root", type=Path, help="Root containing <doc_id>/auto/*_content_list_v7_styles.json.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--middle-block-source",
        choices=("para_blocks", "preproc_blocks"),
        default="para_blocks",
        help="Which middle.json block list to use.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip docs with an existing per-doc middle_fragment_summary.json.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    docs = make_doc_inputs(args)
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    per_doc_root = output_dir / "per_doc"
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for doc in docs:
        doc_dir = per_doc_root / sanitize_doc_id(doc["doc_id"])
        summary_path = doc_dir / "middle_fragment_summary.json"
        if args.skip_existing and summary_path.exists():
            try:
                rows.append(load_json(summary_path))
                continue
            except Exception:
                pass
        if doc.get("middle_json") is None or not Path(doc["middle_json"]).exists():
            error = {"doc_id": doc["doc_id"], "error": "missing_middle_json", "middle_json": str(doc.get("middle_json"))}
            errors.append(error)
            write_doc_error(doc_dir, error)
            continue
        try:
            result = build_middle_fragment_view(
                doc_id=doc["doc_id"],
                middle_json_path=Path(doc["middle_json"]),
                v7_json_path=Path(doc["v7_json"]) if doc.get("v7_json") else None,
                middle_block_source=args.middle_block_source,
            )
            dump_json(doc_dir / "middle_fragment_view.json", result.fragment_view)
            dump_json(
                doc_dir / f"{sanitize_doc_id(doc['doc_id'])}_middle_fragment_content_v7_styles.json",
                result.fragment_v7_payload,
            )
            dump_json(doc_dir / "middle_fragment_merge_labels.json", result.merge_labels)
            dump_json(summary_path, result.summary)
            rows.append(result.summary)
        except Exception as exc:  # pragma: no cover - batch safety
            error = {
                "doc_id": doc["doc_id"],
                "error": f"{type(exc).__name__}: {exc}",
                "middle_json": str(doc.get("middle_json")),
                "v7_json": str(doc.get("v7_json")),
            }
            errors.append(error)
            write_doc_error(doc_dir, error)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_summary(rows, errors=errors, args=args)
    dump_json(output_dir / "summary.json", summary)
    write_summary_csv(output_dir / "summary.csv", rows)
    dump_json(output_dir / "errors.json", errors)
    write_report(output_dir / "MIDDLE_FRAGMENT_VIEW_BUILD_REPORT.md", summary, rows, errors)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def make_doc_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    repo_root = REPO_ROOT
    by_id: dict[str, dict[str, Any]] = {}
    for item in manifest_items(args.manifest):
        doc_id = doc_id_from_item(item)
        middle = first_existing(
            resolve_path(item.get("middle_json") or item.get("middle_path"), root=repo_root),
            find_middle_json(doc_id, args.mineru_root, root=repo_root),
        )
        v7 = first_existing(
            resolve_path(
                item.get("content_json")
                or item.get("content_json_path")
                or item.get("v7_json")
                or item.get("content_list_v7_styles"),
                root=repo_root,
            ),
            find_v7_json(doc_id, args.v7_root or args.mineru_root, root=repo_root),
        )
        by_id[doc_id] = {"doc_id": doc_id, "middle_json": middle, "v7_json": v7}

    explicit_doc_ids = [*args.doc_ids, *read_doc_ids_file(args.doc_ids_file)]
    auto_doc_ids = [] if by_id or explicit_doc_ids else discover_doc_ids(args.mineru_root, root=repo_root)
    for doc_id in [*explicit_doc_ids, *auto_doc_ids]:
        by_id.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "middle_json": find_middle_json(doc_id, args.mineru_root, root=repo_root),
                "v7_json": find_v7_json(doc_id, args.v7_root or args.mineru_root, root=repo_root),
            },
        )

    docs = [by_id[key] for key in sorted(by_id)]
    offset = max(0, int(args.offset or 0))
    if offset:
        docs = docs[offset:]
    if args.limit is not None:
        docs = docs[: max(0, int(args.limit))]
    return docs


def manifest_items(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = load_json(path if path.is_absolute() else REPO_ROOT / path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "documents", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Unsupported manifest format: {path}")


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


def read_doc_ids_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    actual = path if path.is_absolute() else REPO_ROOT / path
    return [
        line.strip()
        for line in actual.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def discover_doc_ids(root_path: Path | None, *, root: Path) -> list[str]:
    if root_path is None:
        return []
    base = root_path if root_path.is_absolute() else root / root_path
    if not base.exists():
        return []
    return sorted({path.parent.parent.name for path in base.glob("*/auto/*_middle.json")})


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
    doc_dir = base / doc_id
    matches = sorted(doc_dir.glob("**/*_middle.json")) if doc_dir.exists() else []
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
    doc_dir = base / doc_id
    matches = sorted(doc_dir.glob("**/*_content_list_v7_styles.json")) if doc_dir.exists() else []
    return matches[0] if matches else None


def first_existing(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def resolve_path(value: Any, *, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def aggregate_summary(rows: list[dict[str, Any]], *, errors: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    totals = {
        "doc_count": len(rows),
        "error_count": len(errors),
        "logical_block_count": sum_int(rows, "logical_block_count"),
        "fragment_count": sum_int(rows, "fragment_count"),
        "positive_merge_edge_count": sum_int(rows, "positive_merge_edge_count"),
        "cross_page_fragment_count": sum_int(rows, "cross_page_fragment_count"),
        "cross_column_fragment_count": sum_int(rows, "cross_column_fragment_count"),
    }
    channel_counts: dict[str, int] = {}
    mapping_method_counts: dict[str, int] = {}
    for row in rows:
        merge_counts(channel_counts, row.get("channel_counts"))
        merge_counts(mapping_method_counts, row.get("mapping_method_counts"))
    return {
        "schema_version": "middle_fragment_view_build_summary_v1",
        **totals,
        "channel_counts": dict(sorted(channel_counts.items())),
        "mapping_method_counts": dict(sorted(mapping_method_counts.items())),
        "middle_block_source": args.middle_block_source,
        "output_dir": str(args.output_dir),
        "errors": errors[:20],
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "doc_id",
        "logical_block_count",
        "fragment_count",
        "positive_merge_edge_count",
        "cross_page_fragment_count",
        "cross_column_fragment_count",
        "channel_counts",
        "mapping_method_counts",
        "middle_json",
        "v7_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), dict)
                    else row.get(key)
                    for key in fieldnames
                }
            )


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    top_rows = sorted(rows, key=lambda row: int(row.get("positive_merge_edge_count") or 0), reverse=True)[:20]
    lines = [
        "# Middle Fragment View Build Report",
        "",
        "## Status",
        f"- docs processed: {summary['doc_count']}",
        f"- errors: {summary['error_count']}",
        f"- fragments: {summary['fragment_count']}",
        f"- positive fragment MERGE edges: {summary['positive_merge_edge_count']}",
        f"- cross-page fragments: {summary['cross_page_fragment_count']}",
        f"- cross-column fragments: {summary['cross_column_fragment_count']}",
        "",
        "## Channel Counts",
        "",
    ]
    for key, value in summary.get("channel_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Mapping Methods", ""])
    for key, value in summary.get("mapping_method_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Docs By Fragment MERGE Edges", ""])
    lines.append("| doc_id | fragments | merge_edges | cross_page_fragments |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in top_rows:
        lines.append(
            f"| {row.get('doc_id')} | {row.get('fragment_count', 0)} | "
            f"{row.get('positive_merge_edge_count', 0)} | {row.get('cross_page_fragment_count', 0)} |"
        )
    if errors:
        lines.extend(["", "## Errors", ""])
        for error in errors[:20]:
            lines.append(f"- {error.get('doc_id')}: {error.get('error')}")
    lines.extend(
        [
            "",
            "## Notes",
            "- This is a sidecar branch view; it does not mutate main v7 JSON.",
            "- Fragment nodes project back to v7 logical owners through owner_v7_ids.",
            "- The pseudo-v7 payload is only for explicit branch experiments, not for the main renderer.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc_error(doc_dir: Path, error: dict[str, Any]) -> None:
    dump_json(doc_dir / "error.json", error)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_counts(target: dict[str, int], value: Any) -> None:
    if not isinstance(value, dict):
        return
    for key, count in value.items():
        target[str(key)] = target.get(str(key), 0) + int(count or 0)


def sum_int(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key) or 0) for row in rows)


def sanitize_doc_id(doc_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(doc_id))


if __name__ == "__main__":
    raise SystemExit(main())
