#!/usr/bin/env python3
"""Audit and validate v8 preservation of MinerU algorithm subtype evidence.

This pass is intentionally sidecar-only: it reads raw MinerU/content_list/middle
artifacts plus current v8 outputs, runs the v8 adapter in-memory, and writes
diagnostic reports.  It does not mutate v8 facts, RenderTreeIR, renderer output,
graphs, labels, or generated TeX.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.mineru_v8_document_ir import convert_v8_payload_to_document_ir, normalize_v8_items_for_adapter


DEFAULT_BASELINE_AUDIT = Path("data/09_eval_reports/algorithm_region_20260526/selected200_baseline_audit")
DEFAULT_PHASE0_DIR = Path("data/09_eval_reports/algorithm_region_20260526/candidate_extraction_phase0")
DEFAULT_SELECTED200_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/algorithm_region_20260526/algorithm_subtype_preservation")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-audit-dir", type=Path, default=DEFAULT_BASELINE_AUDIT)
    parser.add_argument("--phase0-dir", type=Path, default=DEFAULT_PHASE0_DIR)
    parser.add_argument("--selected200-root", type=Path, default=DEFAULT_SELECTED200_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-ids", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1, help="Accepted for consistency; this audit is I/O light and runs serially.")
    return parser


def load_json(path: Path | None, default: Any = None) -> Any:
    if path is None or not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def marker(value: Any) -> str:
    return str(value or "").casefold().strip()


def nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(nonempty(part) for part in value)
    if isinstance(value, dict):
        return any(nonempty(part) for part in value.values())
    return value not in (None, "", [], {})


def iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def raw_items(payload: Any) -> list[dict[str, Any]]:
    items = payload if isinstance(payload, list) else payload.get("items") if isinstance(payload, dict) else []
    return [item for item in items if isinstance(item, dict)]


def is_algorithm_subtype(item: dict[str, Any]) -> bool:
    return any(marker(item.get(key)) == "algorithm" for key in ("type", "canonical_type", "content_list_type", "sub_type", "subtype", "raw_sub_type", "mineru_subtype"))


def has_code_body(item: dict[str, Any]) -> bool:
    return nonempty(item.get("code_body"))


def has_code_caption(item: dict[str, Any]) -> bool:
    return nonempty(item.get("code_caption"))


def has_algorithm_content(item: dict[str, Any]) -> bool:
    return nonempty(item.get("algorithm_content"))


def has_algorithm_caption(item: dict[str, Any]) -> bool:
    return nonempty(item.get("algorithm_caption"))


def algorithm_signal(item: dict[str, Any]) -> bool:
    return is_algorithm_subtype(item) or has_algorithm_caption(item) or has_algorithm_content(item) or (
        marker(item.get("type")) == "code" and marker(item.get("sub_type") or item.get("raw_sub_type") or item.get("mineru_subtype")) == "algorithm"
    )


def count_raw_content_list(payload: Any) -> dict[str, int]:
    items = raw_items(payload)
    return {
        "raw_contentlist_algorithm_subtype_count": sum(1 for item in items if is_algorithm_subtype(item)),
        "raw_contentlist_code_subtype_algorithm_count": sum(1 for item in items if marker(item.get("type")) == "code" and marker(item.get("sub_type")) == "algorithm"),
        "raw_contentlist_code_body_count": sum(1 for item in items if has_code_body(item)),
        "raw_contentlist_code_caption_count": sum(1 for item in items if has_code_caption(item)),
    }


def count_content_list_v2(payload: Any) -> dict[str, int]:
    dicts = list(iter_dicts(payload))
    return {
        "contentlist_v2_type_algorithm_count": sum(1 for item in dicts if marker(item.get("type")) == "algorithm"),
        "contentlist_v2_algorithm_content_count": sum(1 for item in dicts if has_algorithm_content(item)),
        "contentlist_v2_algorithm_caption_count": sum(1 for item in dicts if has_algorithm_caption(item)),
    }


def count_middle(payload: Any) -> dict[str, int]:
    dicts = list(iter_dicts(payload))
    return {
        "middle_subtype_algorithm_count": sum(1 for item in dicts if marker(item.get("sub_type") or item.get("subtype")) == "algorithm"),
        "middle_code_body_count": sum(1 for item in dicts if marker(item.get("type")) == "code_body" or has_code_body(item)),
        "middle_code_caption_count": sum(1 for item in dicts if marker(item.get("type")) == "code_caption" or has_code_caption(item)),
        "middle_algorithm_content_count": sum(1 for item in dicts if has_algorithm_content(item)),
        "middle_algorithm_caption_count": sum(1 for item in dicts if has_algorithm_caption(item)),
    }


def count_v8_items(items: list[dict[str, Any]], *, prefix: str = "v8") -> dict[str, int]:
    return {
        f"{prefix}_type_algorithm_count": sum(1 for item in items if marker(item.get("type")) == "algorithm"),
        f"{prefix}_canonical_type_algorithm_count": sum(1 for item in items if marker(item.get("canonical_type")) == "algorithm"),
        f"{prefix}_content_list_type_algorithm_count": sum(1 for item in items if marker(item.get("content_list_type")) == "algorithm"),
        f"{prefix}_subtype_algorithm_count": sum(1 for item in items if marker(item.get("sub_type") or item.get("subtype") or item.get("raw_sub_type") or item.get("mineru_subtype")) == "algorithm"),
        f"{prefix}_algorithm_content_count": sum(1 for item in items if has_algorithm_content(item)),
        f"{prefix}_algorithm_caption_count": sum(1 for item in items if has_algorithm_caption(item)),
        f"{prefix}_code_body_preserved_count": sum(1 for item in items if has_code_body(item)),
        f"{prefix}_code_caption_preserved_count": sum(1 for item in items if has_code_caption(item)),
    }


def document_nodes(payload: Any) -> list[dict[str, Any]]:
    nodes = payload.get("nodes") if isinstance(payload, dict) else []
    return [node for node in nodes if isinstance(node, dict)]


def node_metadata(node: dict[str, Any]) -> dict[str, Any]:
    value = node.get("metadata")
    return value if isinstance(value, dict) else {}


def count_document_ir_nodes(nodes: list[dict[str, Any]], *, prefix: str = "document_ir") -> dict[str, int]:
    return {
        f"{prefix}_node_type_algorithm_count": sum(1 for node in nodes if marker(node.get("node_type")) == "algorithm"),
        f"{prefix}_canonical_type_algorithm_count": sum(1 for node in nodes if marker(node_metadata(node).get("canonical_type")) == "algorithm"),
        f"{prefix}_algorithm_metadata_count": sum(1 for node in nodes if any(key in node_metadata(node) for key in ("is_algorithm_subtype", "algorithm_origin", "algorithm_confidence", "algorithm_content", "algorithm_caption", "code_body", "code_caption"))),
        f"{prefix}_code_but_algorithm_subtype_count": sum(1 for node in nodes if marker(node.get("node_type")) == "code" and marker(node_metadata(node).get("raw_sub_type") or node_metadata(node).get("mineru_subtype")) == "algorithm"),
    }


def dataclass_document_to_nodes(document: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in getattr(document, "nodes", []) or []:
        rows.append(
            {
                "node_id": getattr(node, "node_id", None),
                "node_type": str(getattr(node, "node_type", "")),
                "text": getattr(node, "text", ""),
                "page_idx": getattr(node, "page_idx", None),
                "metadata": dict(getattr(node, "metadata", {}) or {}),
            }
        )
    return rows


def collect_doc_dirs(root: Path) -> dict[str, Path]:
    docs: dict[str, Path] = {}
    if not root.exists():
        return docs
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        doc_id = path.name.split("_", 1)[-1]
        if (path / "document_ir.json").exists() and list(path.glob("*_content_list_v8_contentlist_merge_hint.json")):
            docs[doc_id] = path
    return docs


def sibling_v2(path: Path | None) -> Path | None:
    if path is None:
        return None
    name = path.name
    if name.endswith("_content_list.json"):
        return path.with_name(name.replace("_content_list.json", "_content_list_v2.json"))
    return None


def sibling_middle(path: Path | None) -> Path | None:
    if path is None:
        return None
    name = path.name
    if name.endswith("_content_list.json"):
        return path.with_name(name.replace("_content_list.json", "_middle.json"))
    return None


def path_from_value(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def source_paths(v8_payload: dict[str, Any], v8_path: Path) -> dict[str, Path | None]:
    source = v8_payload.get("source") if isinstance(v8_payload.get("source"), dict) else {}
    content_path = path_from_value(source.get("content_list_json"))
    middle_path = path_from_value(source.get("middle_json"))
    if content_path is None:
        for item in v8_payload.get("items") or []:
            if isinstance(item, dict):
                content_path = path_from_value(item.get("content_list_json"))
                if content_path is not None:
                    break
    if middle_path is None:
        for item in v8_payload.get("items") or []:
            if isinstance(item, dict):
                middle_path = path_from_value(item.get("middle_json"))
                if middle_path is not None:
                    break
    return {
        "v8": v8_path,
        "content_list": content_path,
        "content_list_v2": sibling_v2(content_path),
        "middle": middle_path or sibling_middle(content_path),
    }


def status_for_counts(row: dict[str, Any], *, after: bool) -> str:
    raw = row.get("raw_algorithm_signal_count", 0)
    middle = row.get("middle_algorithm_signal_count", 0)
    v8 = row.get("v8_after_algorithm_preserved_count" if after else "v8_algorithm_preserved_count", 0)
    ir = row.get("document_ir_after_algorithm_preserved_count" if after else "document_ir_algorithm_preserved_count", 0)
    if raw == 0 and middle == 0:
        return "RAW_MISSING"
    if raw > 0 and middle == 0:
        return "LOST_RAW_TO_MIDDLE"
    if middle > 0 and v8 == 0:
        return "LOST_MIDDLE_TO_V8"
    if v8 > 0 and ir == 0:
        return "LOST_V8_TO_DOCUMENT_IR"
    if v8 > 0 and ir > 0:
        return "PRESERVED"
    return "UNKNOWN"


def raw_algorithm_rows(doc_id: str, payload: Any, layer: str) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(raw_items(payload)):
        if not (algorithm_signal(item) or marker(item.get("type")) == "code" and (has_code_body(item) or has_code_caption(item))):
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "source_layer": layer,
                "raw_id": index,
                "page_idx": item.get("page_idx"),
                "bbox": item.get("bbox"),
                "text_preview": compact(item.get("algorithm_content") or item.get("code_body") or item.get("algorithm_caption") or item.get("code_caption") or item.get("text")),
                "raw_type": item.get("type"),
                "raw_sub_type": item.get("sub_type") or item.get("subtype"),
                "has_algorithm_caption": has_algorithm_caption(item),
                "has_code_caption": has_code_caption(item),
                "has_algorithm_content": has_algorithm_content(item),
                "has_code_body": has_code_body(item),
                "has_code_caption": has_code_caption(item),
            }
        )
    return rows


def build_loss_rows(doc_id: str, raw_payload: Any, v8_items: list[dict[str, Any]], normalized_items: list[dict[str, Any]], after_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    v8_by_source_index = {
        int(item.get("source_content_list_index")): item
        for item in v8_items
        if str(item.get("source_content_list_index") or "").isdigit()
    }
    norm_by_source_index = {
        int(item.get("source_content_list_index")): item
        for item in normalized_items
        if str(item.get("source_content_list_index") or "").isdigit()
    }
    ir_by_source_index: dict[int, dict[str, Any]] = {}
    for node in after_nodes:
        meta = node_metadata(node)
        value = meta.get("source_content_list_index")
        if str(value or "").isdigit():
            ir_by_source_index[int(value)] = node
    rows = []
    for raw_row in raw_algorithm_rows(doc_id, raw_payload, "raw_content_list"):
        index = int(raw_row["raw_id"])
        v8_item = v8_by_source_index.get(index, {})
        norm_item = norm_by_source_index.get(index, {})
        ir_node = ir_by_source_index.get(index, {})
        preserved_v8 = marker(norm_item.get("canonical_type")) == "algorithm"
        preserved_ir = marker(ir_node.get("node_type")) == "algorithm"
        if not v8_item:
            loss_stage = "LOST_RAW_TO_V8_MATCH"
        elif not preserved_v8:
            loss_stage = "LOST_MIDDLE_TO_V8"
        elif not preserved_ir:
            loss_stage = "LOST_V8_TO_DOCUMENT_IR"
        else:
            loss_stage = "PRESERVED"
        rows.append(
            {
                **raw_row,
                "source_v8_id": v8_item.get("id"),
                "content_list_type": v8_item.get("content_list_type"),
                "middle_type": None,
                "v8_type": v8_item.get("type"),
                "canonical_type": norm_item.get("canonical_type") or v8_item.get("canonical_type"),
                "document_ir_node_type": ir_node.get("node_type"),
                "preserved_to_v8": preserved_v8,
                "preserved_to_document_ir": preserved_ir,
                "loss_stage": loss_stage,
                "evidence": "; ".join(
                    part
                    for part in [
                        f"raw_sub_type={raw_row.get('raw_sub_type')}" if raw_row.get("raw_sub_type") else "",
                        f"v8_type={v8_item.get('type')}" if v8_item else "no v8 source index match",
                        f"norm_canonical={norm_item.get('canonical_type')}" if norm_item else "",
                        f"ir_node_type={ir_node.get('node_type')}" if ir_node else "",
                    ]
                    if part
                ),
            }
        )
    return rows


def as_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def audit_doc(doc_id: str, doc_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    v8_paths = sorted(doc_dir.glob("*_content_list_v8_contentlist_merge_hint.json"))
    v8_path = v8_paths[0] if v8_paths else None
    v8_payload = load_json(v8_path, {}) if v8_path else {}
    paths = source_paths(v8_payload, v8_path) if v8_path else {"v8": None, "content_list": None, "content_list_v2": None, "middle": None}
    raw_payload = load_json(paths.get("content_list"), [])
    v2_payload = load_json(paths.get("content_list_v2"), [])
    middle_payload = load_json(paths.get("middle"), {})
    document_ir_payload = load_json(doc_dir / "document_ir.json", {})
    v8_items = [item for item in v8_payload.get("items") or [] if isinstance(item, dict)]
    normalized_items = normalize_v8_items_for_adapter(v8_payload) if isinstance(v8_payload, dict) else []
    try:
        after_document = convert_v8_payload_to_document_ir(v8_payload, source_path=v8_path, doc_id=doc_id)
        after_nodes = dataclass_document_to_nodes(after_document)
    except Exception as exc:
        after_nodes = []
        conversion_error = str(exc)
    else:
        conversion_error = ""
    before_nodes = document_nodes(document_ir_payload)

    row: dict[str, Any] = {
        "doc_id": doc_id,
        "raw_content_list_path": str(paths.get("content_list") or ""),
        "content_list_v2_path": str(paths.get("content_list_v2") or ""),
        "middle_path": str(paths.get("middle") or ""),
        "v8_path": str(v8_path or ""),
        "document_ir_path": str(doc_dir / "document_ir.json"),
        "raw_content_list_found": bool(paths.get("content_list")),
        "content_list_v2_found": bool(paths.get("content_list_v2")),
        "middle_found": bool(paths.get("middle")),
        "conversion_error": conversion_error,
    }
    row.update(count_raw_content_list(raw_payload))
    row.update(count_content_list_v2(v2_payload))
    row.update(count_middle(middle_payload))
    row.update(count_v8_items(v8_items, prefix="v8"))
    row.update(count_v8_items(normalized_items, prefix="v8_after"))
    row.update(count_document_ir_nodes(before_nodes, prefix="document_ir"))
    row.update(count_document_ir_nodes(after_nodes, prefix="document_ir_after"))
    row["raw_algorithm_signal_count"] = (
        as_int(row.get("raw_contentlist_algorithm_subtype_count"))
        + as_int(row.get("raw_contentlist_code_subtype_algorithm_count"))
        + as_int(row.get("contentlist_v2_type_algorithm_count"))
    )
    row["middle_algorithm_signal_count"] = as_int(row.get("middle_subtype_algorithm_count")) + as_int(row.get("middle_code_body_count")) + as_int(row.get("middle_code_caption_count"))
    row["v8_algorithm_preserved_count"] = as_int(row.get("v8_type_algorithm_count")) + as_int(row.get("v8_canonical_type_algorithm_count")) + as_int(row.get("v8_subtype_algorithm_count")) + as_int(row.get("v8_algorithm_content_count")) + as_int(row.get("v8_algorithm_caption_count")) + as_int(row.get("v8_code_body_preserved_count")) + as_int(row.get("v8_code_caption_preserved_count"))
    row["v8_after_algorithm_preserved_count"] = as_int(row.get("v8_after_type_algorithm_count")) + as_int(row.get("v8_after_canonical_type_algorithm_count")) + as_int(row.get("v8_after_subtype_algorithm_count")) + as_int(row.get("v8_after_algorithm_content_count")) + as_int(row.get("v8_after_algorithm_caption_count")) + as_int(row.get("v8_after_code_body_preserved_count")) + as_int(row.get("v8_after_code_caption_preserved_count"))
    row["document_ir_algorithm_preserved_count"] = as_int(row.get("document_ir_node_type_algorithm_count")) + as_int(row.get("document_ir_algorithm_metadata_count"))
    row["document_ir_after_algorithm_preserved_count"] = as_int(row.get("document_ir_after_node_type_algorithm_count")) + as_int(row.get("document_ir_after_algorithm_metadata_count"))
    row["subtype_preservation_status_before"] = status_for_counts(row, after=False)
    row["subtype_preservation_status_after"] = status_for_counts(row, after=True)
    row["algorithm_candidate_count_from_strong_subtype"] = sum(1 for item in normalized_items if marker(item.get("canonical_type")) == "algorithm" and marker(item.get("algorithm_confidence")) == "strong_subtype")
    row["algorithm_caption_candidate_count_after"] = sum(1 for item in normalized_items if marker(item.get("canonical_type")) == "algorithm" and (has_algorithm_caption(item) or has_code_caption(item)))
    row["algorithm_body_candidate_count_after"] = sum(1 for item in normalized_items if marker(item.get("canonical_type")) == "algorithm" and (has_algorithm_content(item) or has_code_body(item) or compact(item.get("text"))))
    row["no_v8_candidate_match_proxy_after"] = max(0, as_int(row.get("raw_contentlist_code_subtype_algorithm_count")) - as_int(row.get("v8_after_canonical_type_algorithm_count")))
    row["caption_missing_proxy_after"] = max(0, as_int(row.get("raw_contentlist_code_caption_count")) - as_int(row.get("algorithm_caption_candidate_count_after")))
    row["body_missing_proxy_after"] = max(0, as_int(row.get("raw_contentlist_code_body_count")) - as_int(row.get("algorithm_body_candidate_count_after")))

    examples = raw_algorithm_rows(doc_id, raw_payload, "raw_content_list")[:8]
    return row, build_loss_rows(doc_id, raw_payload, v8_items, normalized_items, after_nodes), examples


def aggregate(rows: list[dict[str, Any]], phase0_summary: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"docs": len(rows)}
    sum_keys = [
        "raw_algorithm_signal_count",
        "middle_algorithm_signal_count",
        "v8_algorithm_preserved_count",
        "v8_after_algorithm_preserved_count",
        "document_ir_algorithm_preserved_count",
        "document_ir_after_algorithm_preserved_count",
        "algorithm_candidate_count_from_strong_subtype",
        "algorithm_caption_candidate_count_after",
        "algorithm_body_candidate_count_after",
        "no_v8_candidate_match_proxy_after",
        "caption_missing_proxy_after",
        "body_missing_proxy_after",
        "raw_contentlist_code_subtype_algorithm_count",
        "raw_contentlist_code_body_count",
        "raw_contentlist_code_caption_count",
        "contentlist_v2_type_algorithm_count",
        "middle_subtype_algorithm_count",
    ]
    for key in sum_keys:
        summary[key] = sum(as_int(row.get(key)) for row in rows)
    summary["status_counts_before"] = dict(Counter(row.get("subtype_preservation_status_before") for row in rows))
    summary["status_counts_after"] = dict(Counter(row.get("subtype_preservation_status_after") for row in rows))
    summary["raw_artifacts_found"] = {
        "content_list": sum(1 for row in rows if row.get("raw_content_list_found")),
        "content_list_v2": sum(1 for row in rows if row.get("content_list_v2_found")),
        "middle": sum(1 for row in rows if row.get("middle_found")),
        "v8": sum(1 for row in rows if row.get("v8_path")),
        "document_ir": sum(1 for row in rows if row.get("document_ir_path")),
    }
    phase0_values = phase0_summary.get("summary") if isinstance(phase0_summary.get("summary"), dict) else phase0_summary
    summary["phase0_keyword_detector"] = {
        "algorithm_region_candidate_count": phase0_values.get("new_algorithm_region_candidate_count"),
        "algorithm_caption_candidate_count": phase0_values.get("new_algorithm_caption_candidate_count"),
        "algorithm_body_candidate_count": phase0_values.get("new_algorithm_body_candidate_count"),
        "compile_risk_count": phase0_values.get("compile_risk_pseudocode_count_after"),
        "no_v8_candidate_match": phase0_values.get("no_v8_candidate_match_count_after"),
        "decision": phase0_summary.get("decision"),
    }
    false_proxy_docs = [
        row
        for row in rows
        if as_int(row.get("raw_contentlist_code_subtype_algorithm_count")) == 0
        and as_int(row.get("v8_after_canonical_type_algorithm_count")) > 0
    ]
    summary["false_positive_proxy_on_raw_alg0_docs"] = len(false_proxy_docs)
    return summary


def decide(summary: dict[str, Any]) -> str:
    preserved = as_int(summary.get("document_ir_after_algorithm_preserved_count"))
    raw = as_int(summary.get("raw_contentlist_code_subtype_algorithm_count"))
    no_proxy = as_int(summary.get("no_v8_candidate_match_proxy_after"))
    false_proxy = as_int(summary.get("false_positive_proxy_on_raw_alg0_docs"))
    if raw == 0 or preserved == 0:
        return "diagnostic_only"
    if no_proxy <= max(10, int(raw * 0.25)) and false_proxy <= 3:
        return "ready_for_algorithm_renderer_phase0"
    return "need_roi_audit_after_subtype_preservation"


def write_examples(path: Path, examples: list[dict[str, Any]], loss_rows: list[dict[str, Any]]) -> None:
    lines = ["# Algorithm Subtype Preservation Examples", ""]
    lines.append("## Raw Algorithm Signals")
    lines.append("")
    if not examples:
        lines.append("- none")
    for item in examples[:40]:
        lines.append(f"- `{item.get('doc_id')}` page={item.get('page_idx')} type={item.get('raw_type')} sub_type={item.get('raw_sub_type')} text={compact(item.get('text_preview'), 220)}")
    lines.append("")
    lines.append("## Loss Matrix Examples")
    lines.append("")
    for item in loss_rows[:40]:
        lines.append(f"- `{item.get('doc_id')}` idx={item.get('raw_id')} stage={item.get('loss_stage')} raw={item.get('raw_type')}/{item.get('raw_sub_type')} v8={item.get('v8_type')} canon={item.get('canonical_type')} ir={item.get('document_ir_node_type')} text={compact(item.get('text_preview'), 180)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]], loss_rows: list[dict[str, Any]], decision: str) -> None:
    def line(values: list[Any]) -> str:
        return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"

    loss_counts = Counter(row.get("loss_stage") for row in loss_rows)
    lines = ["# Algorithm Subtype Preservation Audit And Patch Report", ""]
    lines.append("## Status")
    lines.append("")
    lines.append(f"- docs analyzed: {summary.get('docs')}")
    artifacts = summary.get("raw_artifacts_found", {})
    lines.append(f"- raw/middle/v8/document_ir artifacts found: content_list={artifacts.get('content_list')}, content_list_v2={artifacts.get('content_list_v2')}, middle={artifacts.get('middle')}, v8={artifacts.get('v8')}, document_ir={artifacts.get('document_ir')}")
    lines.append("- no training / no MinerU / no relabel / no rebuild / no GNN")
    lines.append("- no renderer changes")
    lines.append("- v8 facts used; no fallback to old v7")
    lines.append("- legacy names such as `source_v7_ids` / `v7_id` are provenance names only")
    lines.append("")
    lines.append("```text")
    lines.append("v8 full observable facts")
    lines.append("  -> v8 atomic/reflow")
    lines.append("  -> deterministic merge + contentlist merge hint")
    lines.append("  -> RenderTreeIR")
    lines.append("  -> IR renderer")
    lines.append("```")
    lines.append("")
    lines.append("## Key Finding")
    lines.append("")
    if as_int(summary.get("raw_contentlist_code_subtype_algorithm_count")):
        lines.append("- raw content_list/content_list_v2/middle already contain algorithm subtype evidence.")
    else:
        lines.append("- raw algorithm subtype evidence is sparse in the located selected200 artifacts.")
    lines.append("- before the patch, current serialized v8+hint often kept only `type=code` / `content_list_type=code` and lost explicit subtype/body fields.")
    lines.append("- the loss is primarily at middle/raw-content-list -> v8/DocumentIR preservation, not a renderer problem.")
    lines.append("")
    lines.append("## Before / After Summary")
    lines.append("")
    lines.append(line(["Metric", "Before", "After"]))
    lines.append(line(["---", "---:", "---:"]))
    comparisons = [
        ("raw algorithm signals", "raw_algorithm_signal_count", "raw_algorithm_signal_count"),
        ("middle algorithm signals", "middle_algorithm_signal_count", "middle_algorithm_signal_count"),
        ("v8 algorithm preserved", "v8_algorithm_preserved_count", "v8_after_algorithm_preserved_count"),
        ("document_ir algorithm preserved", "document_ir_algorithm_preserved_count", "document_ir_after_algorithm_preserved_count"),
        ("no_v8_candidate_match proxy", "", "no_v8_candidate_match_proxy_after"),
        ("algorithm caption candidate count", "", "algorithm_caption_candidate_count_after"),
        ("algorithm body candidate count", "", "algorithm_body_candidate_count_after"),
        ("compile risk proxy", "", "not_run_in_subtype_pass"),
    ]
    for label, before_key, after_key in comparisons:
        before = summary.get(before_key) if before_key else "n/a"
        after = summary.get(after_key)
        lines.append(line([label, before, after]))
    lines.append("")
    lines.append("## Preservation Status")
    lines.append("")
    lines.append(line(["status", "before_docs", "after_docs"]))
    lines.append(line(["---", "---:", "---:"]))
    status_keys = sorted(set(summary.get("status_counts_before", {})) | set(summary.get("status_counts_after", {})))
    for key in status_keys:
        lines.append(line([key, summary.get("status_counts_before", {}).get(key, 0), summary.get("status_counts_after", {}).get(key, 0)]))
    lines.append("")
    lines.append("## Loss Matrix")
    lines.append("")
    lines.append(line(["loss_stage", "count"]))
    lines.append(line(["---", "---:"]))
    for key, value in loss_counts.most_common():
        lines.append(line([key, value]))
    lines.append("")
    lines.append("## Adapter Patch")
    lines.append("")
    lines.append("- `mineru_v8_document_ir.py` now recognizes strong algorithm signals: `type=algorithm`, `canonical_type=algorithm`, `sub_type/raw_sub_type/mineru_subtype=algorithm`, `content_list_type=algorithm`, and metadata-backed `algorithm_content/algorithm_caption/code_body/code_caption` when attached to explicit algorithm/code subtype evidence.")
    lines.append("- Existing v8 items that only carry `content_list_json + source_content_list_index` are enriched in-memory from the raw content_list during DocumentIR conversion.")
    lines.append("- `mineru_v8_reflow.py` now preserves `sub_type`, `code_body`, `algorithm_content`, `code_caption`, and `algorithm_caption` when future v8 facts are built.")
    lines.append("- Text-only references such as `Algorithm 1 shows ...` are not promoted because the adapter does not use keyword-only inference.")
    lines.append("- Renderer and graph/GNN behavior are unchanged.")
    lines.append("")
    lines.append("## Comparison With Keyword Detector")
    lines.append("")
    phase0 = summary.get("phase0_keyword_detector", {})
    lines.append(line(["Source", "region/candidate count", "caption", "body", "compile risk", "no_v8_candidate_match"]))
    lines.append(line(["---", "---:", "---:", "---:", "---:", "---:"]))
    lines.append(line(["Phase0 keyword detector", phase0.get("algorithm_region_candidate_count"), phase0.get("algorithm_caption_candidate_count"), phase0.get("algorithm_body_candidate_count"), phase0.get("compile_risk_count"), phase0.get("no_v8_candidate_match")]))
    lines.append(line(["Subtype preservation", summary.get("algorithm_candidate_count_from_strong_subtype"), summary.get("algorithm_caption_candidate_count_after"), summary.get("algorithm_body_candidate_count_after"), "not expanded", summary.get("no_v8_candidate_match_proxy_after")]))
    lines.append("")
    lines.append("Subtype preservation is a high-confidence upstream fact inheritance path. The Phase0 keyword detector remains diagnostic and should not be mixed into production candidates.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"**{decision}**")
    if decision == "ready_for_algorithm_renderer_phase0":
        lines.append("")
        lines.append("Subtype preservation gives enough precise algorithm candidates to consider a renderer-safe AlgorithmRegion Phase 0.")
    elif decision == "need_roi_audit_after_subtype_preservation":
        lines.append("")
        lines.append("Subtype preservation improves the high-confidence path, but remaining no-candidate cases should go to ROI/fact audit before renderer work.")
    else:
        lines.append("")
        lines.append("Keep this as diagnostic because raw/middle subtype evidence is insufficient or noisy.")
    (output_dir / "ALGORITHM_SUBTYPE_PRESERVATION_AUDIT_AND_PATCH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def readiness_report(output_dir: Path, missing: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ALGORITHM_SUBTYPE_PRESERVATION_AUDIT_AND_PATCH_REPORT.md").write_text(
        "# Algorithm Subtype Preservation Readiness Report\n\n"
        "Required artifacts were missing; the pass stopped without guessing.\n\n"
        + "\n".join(f"- {item}" for item in missing)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    doc_dirs = collect_doc_dirs(args.selected200_root)
    if args.doc_ids:
        doc_dirs = {doc_id: doc_dirs[doc_id] for doc_id in args.doc_ids if doc_id in doc_dirs}
    if args.limit is not None:
        doc_dirs = dict(list(doc_dirs.items())[: args.limit])
    missing = []
    if not doc_dirs:
        missing.append(str(args.selected200_root))
    if not args.baseline_audit_dir.exists():
        missing.append(str(args.baseline_audit_dir))
    if not args.phase0_dir.exists():
        missing.append(str(args.phase0_dir))
    if missing:
        readiness_report(args.output_dir, missing)
        return 2

    rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for doc_id, doc_dir in doc_dirs.items():
        row, doc_loss_rows, doc_examples = audit_doc(doc_id, doc_dir)
        rows.append(row)
        loss_rows.extend(doc_loss_rows)
        examples.extend(doc_examples)

    phase0_summary = load_json(args.phase0_dir / "algorithm_region_candidate_extraction_summary.json", {})
    summary = aggregate(rows, phase0_summary if isinstance(phase0_summary, dict) else {})
    decision = decide(summary)
    summary["decision"] = decision

    write_csv(args.output_dir / "algorithm_subtype_preservation_summary.csv", rows)
    write_json(args.output_dir / "algorithm_subtype_preservation_summary.json", summary)
    write_csv(args.output_dir / "algorithm_subtype_loss_matrix.csv", loss_rows)
    write_jsonl(args.output_dir / "algorithm_subtype_loss_matrix.jsonl", loss_rows)
    write_examples(args.output_dir / "algorithm_subtype_preservation_examples.md", examples, loss_rows)
    write_report(args.output_dir, summary, rows, loss_rows, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
