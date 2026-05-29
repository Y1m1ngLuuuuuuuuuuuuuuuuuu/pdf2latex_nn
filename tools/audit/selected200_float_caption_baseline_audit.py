#!/usr/bin/env python3
"""Audit selected200 float/caption baseline without rerunning generation.

The pass is read-only over the current v8+contentlist-merge-hint selected200
outputs.  It compares gold/pred comparison structures, scans DocumentIR/full
fact nodes for caption-like candidates, and produces caption-float pairing
diagnostics for figure/table/algorithm captions.

It does not modify the renderer, v8 merge path, GNN code, graph schema, labels,
or generated outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_BREAKDOWN_DIR = Path("data/09_eval_reports/v8_visible_prose_failure_breakdown_20260526/v8_contentlist_merge_hint")
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/float_caption_layout_20260526/selected200_baseline_audit")

FLOAT_TYPES = {"figure", "table", "algorithm"}

CAPTION_PREFIX_RE = re.compile(
    r"^\s*(?P<kind>fig(?:ure)?|table|tab(?:le)?|algorithm|alg)\s*\.?\s*"
    r"(?P<number>(?:s)?\d+(?:\.\d+)*(?:\([a-z]\))?|[ivxlcdm]+)?"
    r"\s*(?P<punct>[:.\-–—])?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
CAPTION_FALSE_START_RE = re.compile(
    r"^\s*(?:as shown in|shown in|see|we use|we compare|in)\s+"
    r"(?:fig(?:ure)?|table|tab(?:le)?|algorithm|alg)\b",
    re.IGNORECASE,
)
CAPTION_BODY_VERB_RE = re.compile(
    r"^(?:shows?|illustrates?|depicts?|reports?|presents?|summari[sz]es?|lists?|"
    r"contains?|is|are|was|were|can|will|should|may|we|this|that)\b",
    re.IGNORECASE,
)


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breakdown-dir", type=Path, default=DEFAULT_BREAKDOWN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-ids", nargs="*", default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path | None, default: Any = None) -> Any:
    if path is None:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, BBox):
            return bbox_to_list(value)
        return str(value)

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=default) + "\n", encoding="utf-8")


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def as_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def compact(text: Any, limit: int = 220) -> str:
    return " ".join(str(text or "").split())[:limit]


def md(text: Any) -> str:
    return compact(text).replace("|", "\\|").replace("\n", " ")


def normalize(text: Any) -> str:
    value = str(text or "").lower()
    value = re.sub(r"\[math\]|\$[^$]*\$|\\[a-zA-Z]+", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def infer_caption_type(text: str, fallback: str = "unknown") -> tuple[str, str | None, bool]:
    """Return (type, number, is_caption_like)."""
    stripped = str(text or "").strip()
    if not stripped:
        return fallback, None, False
    if CAPTION_FALSE_START_RE.match(stripped):
        return fallback, None, False
    match = CAPTION_PREFIX_RE.match(stripped)
    if not match:
        return fallback, None, False
    kind_raw = (match.group("kind") or "").lower()
    rest = (match.group("rest") or "").strip()
    punct = match.group("punct")
    number = match.group("number")
    if rest and CAPTION_BODY_VERB_RE.match(rest) and not punct:
        return fallback, number, False
    if kind_raw.startswith("fig"):
        kind = "figure"
    elif kind_raw.startswith("tab"):
        kind = "table"
    elif kind_raw.startswith("alg"):
        kind = "algorithm"
    else:
        kind = fallback
    # Caption prefixes without a number are allowed only if punctuation and text
    # make them caption-like.  This keeps "Figure shows ..." out.
    if not number and not punct:
        return fallback, None, False
    return kind, number.upper() if number and number.isalpha() else number, True


def parse_bbox(value: Any) -> BBox | None:
    if value is None:
        return None
    if isinstance(value, dict):
        keys = ("x0", "y0", "x1", "y1")
        if all(key in value for key in keys):
            return BBox(float(value["x0"]), float(value["y0"]), float(value["x1"]), float(value["y1"]))
    if isinstance(value, list):
        if not value:
            return None
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            x0, y0, x1, y1 = value
            return BBox(float(x0), float(y0), float(x1), float(y1))
        parsed = [parse_bbox(item) for item in value]
        boxes = [box for box in parsed if box is not None]
        if boxes:
            return union_bbox(boxes)
    return None


def union_bbox(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    return BBox(
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def bbox_to_list(box: BBox | None) -> list[float] | None:
    if box is None:
        return None
    return [box.x0, box.y0, box.x1, box.y1]


def x_overlap_ratio(a: BBox | None, b: BBox | None) -> float | None:
    if a is None or b is None:
        return None
    overlap = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    denom = max(1e-6, min(a.width, b.width))
    return overlap / denom


def vertical_relation(caption: BBox | None, flt: BBox | None) -> str:
    if caption is None or flt is None:
        return "unknown"
    if caption.cy < flt.cy:
        return "above"
    if caption.cy > flt.cy:
        return "below"
    return "overlap"


def proximity_score(caption: dict[str, Any], flt: dict[str, Any]) -> float:
    cbox = caption.get("bbox_obj")
    fbox = flt.get("bbox_obj")
    page_delta = abs(as_int(caption.get("page_idx")) - as_int(flt.get("page_idx")))
    if cbox is None or fbox is None:
        return 10000 + page_delta * 1000
    dx = abs(cbox.cx - fbox.cx)
    if cbox.cy < fbox.cy:
        dy = max(0.0, fbox.y0 - cbox.y1)
    else:
        dy = max(0.0, cbox.y0 - fbox.y1)
    return page_delta * 1000.0 + dy + dx * 0.1


def blocks_by_id(blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(block.get("block_id")): block for block in blocks if block.get("block_id")}


def get_block_type(block: dict[str, Any]) -> str:
    return str(block.get("block_type") or block.get("type") or block.get("role") or "").lower()


def extract_comparison_captions(structure: dict[str, Any], source: str) -> list[dict[str, Any]]:
    blocks = structure.get("blocks") or []
    by_id = blocks_by_id(blocks)
    captions = []
    for block in blocks:
        block_type = get_block_type(block)
        if block_type != "caption":
            continue
        parent = by_id.get(str(block.get("parent_id") or block.get("label") or ""))
        parent_type = get_block_type(parent or {})
        cap_type, number, _ = infer_caption_type(str(block.get("text") or ""), parent_type if parent_type in FLOAT_TYPES else "unknown")
        captions.append(
            {
                "source": source,
                "caption_id": block.get("block_id"),
                "text": block.get("text") or "",
                "normalized_text": normalize(block.get("text") or ""),
                "caption_type": cap_type,
                "caption_number": number,
                "parent_id": block.get("parent_id") or block.get("label"),
                "parent_type": parent_type or "unknown",
                "order": block.get("order"),
                "block": block,
            }
        )
    return captions


def extract_comparison_floats(structure: dict[str, Any], source: str) -> list[dict[str, Any]]:
    blocks = structure.get("blocks") or []
    floats = []
    for block in blocks:
        block_type = get_block_type(block)
        if block_type not in FLOAT_TYPES:
            continue
        floats.append(
            {
                "source": source,
                "float_id": block.get("block_id"),
                "float_type": block_type,
                "order": block.get("order"),
                "block": block,
            }
        )
    return floats


def extract_caption_like_paragraphs(structure: dict[str, Any], source: str) -> list[dict[str, Any]]:
    rows = []
    for block in structure.get("blocks") or []:
        block_type = get_block_type(block)
        if block_type == "caption":
            continue
        cap_type, number, is_cap = infer_caption_type(str(block.get("text") or ""))
        if is_cap and block_type in {"paragraph", "abstract", "list_item", "text"}:
            rows.append(
                {
                    "source": source,
                    "block_id": block.get("block_id"),
                    "block_type": block_type,
                    "caption_type": cap_type,
                    "caption_number": number,
                    "text": block.get("text") or "",
                    "order": block.get("order"),
                    "failure_type": "caption_as_paragraph",
                }
            )
    return rows


def node_text(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") or {}
    pieces = [node.get("text") or ""]
    for key in ("figure_caption", "table_caption", "algorithm_caption"):
        value = metadata.get(key)
        if isinstance(value, str):
            pieces.append(value)
    for key in ("image_caption", "table_caption", "caption"):
        value = metadata.get(key)
        if isinstance(value, list):
            pieces.extend(str(item) for item in value)
        elif isinstance(value, str):
            pieces.append(value)
    return " ".join(piece for piece in pieces if piece)


def node_float_type(node: dict[str, Any]) -> str | None:
    metadata = node.get("metadata") or {}
    values = [
        node.get("node_type"),
        node.get("raw_type"),
        metadata.get("canonical_type"),
        metadata.get("layout_role"),
        metadata.get("type"),
        metadata.get("raw_type"),
    ]
    lowered = " ".join(str(value or "").lower() for value in values)
    if "algorithm" in lowered:
        return "algorithm"
    if "table" in lowered:
        return "table"
    if "figure" in lowered or "image" in lowered:
        return "figure"
    return None


def extract_document_ir_floats(document_ir: dict[str, Any]) -> list[dict[str, Any]]:
    floats: list[dict[str, Any]] = []
    for node in document_ir.get("nodes") or []:
        ftype = node_float_type(node)
        if ftype not in FLOAT_TYPES:
            continue
        box = parse_bbox(node.get("bboxes") or node.get("bbox"))
        metadata = node.get("metadata") or {}
        captions = []
        for key in ("figure_caption", "table_caption", "algorithm_caption", "image_caption", "caption"):
            value = metadata.get(key)
            if isinstance(value, list):
                captions.extend(str(item) for item in value if item)
            elif isinstance(value, str) and value:
                captions.append(value)
        floats.append(
            {
                "float_id": node.get("node_id"),
                "float_type": ftype,
                "page_idx": node.get("page_idx"),
                "bbox": bbox_to_list(box),
                "bbox_obj": box,
                "reading_index": node.get("reading_index"),
                "caption_metadata": captions,
                "metadata": metadata,
            }
        )
    return floats


def extract_v7_caption_candidates(document_ir: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for node in document_ir.get("nodes") or []:
        metadata = node.get("metadata") or {}
        source_node = str(node.get("node_id") or "")
        box = parse_bbox(node.get("bboxes") or node.get("bbox"))
        base = {
            "source_node_id": source_node,
            "page_idx": node.get("page_idx"),
            "bbox": bbox_to_list(box),
            "bbox_obj": box,
            "reading_index": node.get("reading_index"),
            "layout_role": metadata.get("layout_role"),
            "layout_layer": metadata.get("layout_layer"),
            "node_type": node.get("node_type"),
        }

        for key in ("figure_caption", "table_caption", "algorithm_caption", "image_caption", "caption"):
            value = metadata.get(key)
            values = value if isinstance(value, list) else [value] if isinstance(value, str) else []
            for text in values:
                cap_type, number, is_cap = infer_caption_type(str(text or ""), node_float_type(node) or "unknown")
                if not text:
                    continue
                ident = (source_node, normalize(text))
                if ident in seen:
                    continue
                seen.add(ident)
                candidates.append(
                    {
                        **base,
                        "candidate_id": f"{source_node}:metadata_caption:{len(candidates):04d}",
                        "candidate_source": "metadata_caption",
                        "text": str(text),
                        "normalized_text": normalize(text),
                        "caption_type": cap_type,
                        "caption_number": number,
                        "is_caption_like": True,
                        "embedded_in_float_metadata": node_float_type(node) in FLOAT_TYPES,
                    }
                )

        text = node.get("text") or ""
        cap_type, number, is_cap = infer_caption_type(text)
        if is_cap:
            ident = (source_node, normalize(text))
            if ident not in seen:
                seen.add(ident)
                candidates.append(
                    {
                        **base,
                        "candidate_id": f"{source_node}:text_caption:{len(candidates):04d}",
                        "candidate_source": "text_block",
                        "text": text,
                        "normalized_text": normalize(text),
                        "caption_type": cap_type,
                        "caption_number": number,
                        "is_caption_like": True,
                        "embedded_in_float_metadata": False,
                    }
                )
    return candidates


def similarity(a: str, b: str) -> float:
    a_norm = normalize(a)
    b_norm = normalize(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def match_captions(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            score = similarity(g.get("text", ""), p.get("text", ""))
            if g.get("caption_type") == p.get("caption_type") and g.get("caption_type") != "unknown":
                score += 0.15
            if g.get("caption_number") and g.get("caption_number") == p.get("caption_number"):
                score += 0.1
            if score >= 0.58:
                pairs.append((score, gi, pi))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append({"gold_index": gi, "pred_index": pi, "score": score, "gold": gold[gi], "pred": pred[pi]})
    return matches, used_g, used_p


def nearest_float_rows(caption: dict[str, Any], floats: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    rows = []
    for flt in floats:
        cbox = caption.get("bbox_obj")
        fbox = flt.get("bbox_obj")
        rows.append(
            {
                "float_id": flt.get("float_id"),
                "float_type": flt.get("float_type"),
                "page_idx": flt.get("page_idx"),
                "same_page": caption.get("page_idx") == flt.get("page_idx"),
                "page_delta": abs(as_int(caption.get("page_idx")) - as_int(flt.get("page_idx"))),
                "x_overlap": x_overlap_ratio(cbox, fbox),
                "vertical_relation": vertical_relation(cbox, fbox),
                "distance_score": proximity_score(caption, flt),
                "bbox": flt.get("bbox"),
            }
        )
    rows.sort(key=lambda row: row["distance_score"])
    return rows[:limit]


def render_tree_caption_texts(render_tree: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for node in render_tree.get("nodes") or []:
        text = node.get("text") or ""
        role = str(node.get("role") or node.get("node_type") or node.get("type") or "").lower()
        if "caption" in role or "\\caption" in text:
            texts.append(text)
    return texts


def doc_dir_for_row(row: dict[str, str]) -> Path | None:
    generated = Path(row.get("generated_tex") or "")
    if generated.exists():
        return generated.parent
    return None


def audit_doc(doc_id: str, row: dict[str, str], out_dir: Path, max_examples: int) -> dict[str, Any]:
    doc_dir = doc_dir_for_row(row)
    if doc_dir is None:
        raise FileNotFoundError(f"generated_tex parent not found for {doc_id}")
    gold_structure = load_json(doc_dir / "gold_structure.json", {})
    pred_structure = load_json(doc_dir / "generated_structure.json", {})
    metrics = load_json(doc_dir / "structure_metrics.json", {})
    document_ir = load_json(doc_dir / "document_ir.json", {})
    render_tree = load_json(doc_dir / "render_tree_ir.json", {})

    gold_caps = extract_comparison_captions(gold_structure, "gold")
    pred_caps = extract_comparison_captions(pred_structure, "pred")
    gold_floats = extract_comparison_floats(gold_structure, "gold")
    pred_floats = extract_comparison_floats(pred_structure, "pred")
    caption_like_paragraphs = extract_caption_like_paragraphs(pred_structure, "pred")
    v7_floats = extract_document_ir_floats(document_ir)
    v7_candidates = extract_v7_caption_candidates(document_ir)
    matches, used_gold, used_pred = match_captions(gold_caps, pred_caps)

    pred_by_id = blocks_by_id(pred_structure.get("blocks") or [])
    pred_caption_parent_ids = {str(cap.get("parent_id")) for cap in pred_caps if cap.get("parent_id")}
    pred_float_ids = {str(flt.get("float_id")) for flt in pred_floats if flt.get("float_id")}
    pred_float_with_caption = pred_caption_parent_ids & pred_float_ids
    pred_float_without_caption = [flt for flt in pred_floats if str(flt.get("float_id")) not in pred_float_with_caption]

    duplicate_counter = Counter(cap["normalized_text"] for cap in pred_caps if cap["normalized_text"])
    duplicate_caps = [cap for cap in pred_caps if duplicate_counter.get(cap["normalized_text"], 0) > 1]

    caption_without_float = []
    wrong_type = []
    for cap in pred_caps:
        parent_id = str(cap.get("parent_id") or "")
        parent = pred_by_id.get(parent_id)
        parent_type = get_block_type(parent or {})
        if parent_type not in FLOAT_TYPES:
            caption_without_float.append(cap)
        elif cap.get("caption_type") != "unknown" and cap.get("caption_type") != parent_type:
            wrong_type.append({**cap, "expected_parent_type": cap.get("caption_type"), "actual_parent_type": parent_type})

    unmatched_gold = [gold_caps[idx] for idx in range(len(gold_caps)) if idx not in used_gold]
    unmatched_pred = [pred_caps[idx] for idx in range(len(pred_caps)) if idx not in used_pred]
    type_counts_gold = Counter(cap.get("caption_type") or "unknown" for cap in gold_caps)
    type_counts_pred = Counter(cap.get("caption_type") or "unknown" for cap in pred_caps)
    missing_by_type = {
        "figure": max(0, type_counts_gold.get("figure", 0) - type_counts_pred.get("figure", 0)),
        "table": max(0, type_counts_gold.get("table", 0) - type_counts_pred.get("table", 0)),
        "algorithm": max(0, type_counts_gold.get("algorithm", 0) - type_counts_pred.get("algorithm", 0)),
    }

    pred_norms = [cap["normalized_text"] for cap in pred_caps]
    candidate_not_consumed = [
        candidate
        for candidate in v7_candidates
        if not any(SequenceMatcher(None, candidate["normalized_text"], pred_norm).ratio() >= 0.72 for pred_norm in pred_norms)
    ]
    metadata_not_consumed = [candidate for candidate in candidate_not_consumed if candidate.get("embedded_in_float_metadata")]

    pairing_rows = []
    for candidate in v7_candidates:
        nearest = nearest_float_rows(candidate, v7_floats)
        pairing_rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "text": candidate.get("text"),
                "caption_type": candidate.get("caption_type"),
                "caption_number": candidate.get("caption_number"),
                "page_idx": candidate.get("page_idx"),
                "bbox": candidate.get("bbox"),
                "candidate_source": candidate.get("candidate_source"),
                "nearest_floats": nearest,
                "current_paired_float": candidate.get("source_node_id") if candidate.get("embedded_in_float_metadata") else None,
                "consumed_as_pred_caption": candidate not in candidate_not_consumed,
            }
        )

    rendered_caption_texts = render_tree_caption_texts(render_tree)
    crop_swallowed = len(metadata_not_consumed)
    placeholder_needed = sum(1 for candidate in candidate_not_consumed if not nearest_float_rows(candidate, v7_floats, 1))
    if caption_like_paragraphs:
        placeholder_needed += sum(1 for _ in caption_like_paragraphs)

    float_metric = metrics.get("float_caption_attachment_accuracy") or {}
    row_out = {
        "doc_id": doc_id,
        "gold_caption_count": len(gold_caps),
        "pred_caption_count": len(pred_caps),
        "caption_candidate_count_v7": len(v7_candidates),
        "caption_like_paragraph_count": len(caption_like_paragraphs),
        "missing_caption_count": len(unmatched_gold),
        "caption_as_paragraph_count": len(caption_like_paragraphs),
        "duplicate_caption_count": len(duplicate_caps),
        "caption_without_float_count": len(caption_without_float),
        "float_without_caption_count": len(pred_float_without_caption),
        "wrong_float_type_count": len(wrong_type),
        "figure_caption_count": type_counts_pred.get("figure", 0),
        "table_caption_count": type_counts_pred.get("table", 0),
        "algorithm_caption_count": type_counts_pred.get("algorithm", 0),
        "unknown_caption_count": type_counts_pred.get("unknown", 0),
        "algorithm_caption_missing": missing_by_type["algorithm"],
        "table_caption_missing": missing_by_type["table"],
        "figure_caption_missing": missing_by_type["figure"],
        "subfigure_caption_issue": sum(1 for cap in unmatched_gold if re.search(r"\([a-z]\)", cap.get("text", ""), re.I)),
        "placeholder_needed_count": placeholder_needed,
        "crop_swallowed_caption_count": crop_swallowed,
        "caption_rendered_inside_crop_only_count": crop_swallowed,
        "float_caption_attachment_accuracy": as_float(float_metric.get("score") if isinstance(float_metric, dict) else float_metric),
        "macro_structure_score_body": as_float((metrics.get("macro_structure_score_body") or {}).get("score") if isinstance(metrics.get("macro_structure_score_body"), dict) else metrics.get("macro_structure_score_body")),
        "macro_structure_score": as_float((metrics.get("macro_structure_score") or {}).get("score") if isinstance(metrics.get("macro_structure_score"), dict) else metrics.get("macro_structure_score")),
        "generated_structure_validity": as_float((metrics.get("generated_structure_validity") or {}).get("score") if isinstance(metrics.get("generated_structure_validity"), dict) else metrics.get("generated_structure_validity")),
        "render_tree_caption_text_count": len(rendered_caption_texts),
        "v7_metadata_caption_not_consumed_count": len(metadata_not_consumed),
    }

    audit = {
        "schema_version": "selected200_float_caption_audit_v1",
        "doc_id": doc_id,
        "doc_dir": str(doc_dir),
        "summary": row_out,
        "gold_captions": gold_caps,
        "pred_captions": pred_caps,
        "matched_captions": matches,
        "unmatched_gold_captions": unmatched_gold,
        "unmatched_pred_captions": unmatched_pred,
        "caption_like_paragraphs": caption_like_paragraphs,
        "duplicate_captions": duplicate_caps,
        "caption_without_float": caption_without_float,
        "float_without_caption": pred_float_without_caption,
        "wrong_float_type_pairing": wrong_type,
        "crop_swallowed_caption_candidates": metadata_not_consumed,
        "placeholder_float_needed_candidates": candidate_not_consumed[:max_examples],
        "render_tree_caption_texts": rendered_caption_texts[:max_examples],
    }
    candidates_payload = {
        "schema_version": "selected200_caption_candidates_v1",
        "doc_id": doc_id,
        "caption_candidates": [
            {key: value for key, value in candidate.items() if key != "bbox_obj"}
            for candidate in v7_candidates
        ],
    }
    pairing_payload = {
        "schema_version": "selected200_caption_pairing_candidates_v1",
        "doc_id": doc_id,
        "pairing_candidates": pairing_rows,
    }
    safe_id = doc_id.replace("/", "_")
    write_json(out_dir / f"float_caption_audit_{safe_id}.json", audit)
    write_json(out_dir / f"caption_candidates_{safe_id}.json", candidates_payload)
    write_json(out_dir / f"caption_pairing_candidates_{safe_id}.json", pairing_payload)
    return row_out


def readiness_check(args: argparse.Namespace) -> list[str]:
    required = [args.breakdown_dir / "doc_failure_breakdown.csv"]
    return [str(path) for path in required if not path.exists()]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sum_keys = [
        "gold_caption_count",
        "pred_caption_count",
        "caption_candidate_count_v7",
        "caption_like_paragraph_count",
        "missing_caption_count",
        "caption_as_paragraph_count",
        "duplicate_caption_count",
        "caption_without_float_count",
        "float_without_caption_count",
        "wrong_float_type_count",
        "figure_caption_missing",
        "table_caption_missing",
        "algorithm_caption_missing",
        "placeholder_needed_count",
        "crop_swallowed_caption_count",
        "caption_rendered_inside_crop_only_count",
        "v7_metadata_caption_not_consumed_count",
    ]
    mean_keys = ["float_caption_attachment_accuracy", "macro_structure_score_body", "generated_structure_validity"]
    payload = {key: sum(as_int(row.get(key)) for row in rows) for key in sum_keys}
    payload.update({key: mean([as_float(row.get(key)) for row in rows]) for key in mean_keys})
    payload["docs"] = len(rows)
    payload["caption_recall_proxy"] = payload["pred_caption_count"] / payload["gold_caption_count"] if payload["gold_caption_count"] else None
    payload["caption_candidate_recall_proxy"] = payload["caption_candidate_count_v7"] / payload["gold_caption_count"] if payload["gold_caption_count"] else None
    return payload


def collect_examples(out_dir: Path, rows: list[dict[str, Any]], max_examples: int) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: as_int(item.get("missing_caption_count")), reverse=True):
        audit = load_json(out_dir / f"float_caption_audit_{row['doc_id']}.json", {})
        mapping = {
            "missing captions": audit.get("unmatched_gold_captions") or [],
            "caption-like paragraph": audit.get("caption_like_paragraphs") or [],
            "wrong pairing": audit.get("wrong_float_type_pairing") or [],
            "duplicate caption": audit.get("duplicate_captions") or [],
            "algorithm caption": [cap for cap in audit.get("unmatched_gold_captions") or [] if cap.get("caption_type") == "algorithm"],
            "table caption": [cap for cap in audit.get("unmatched_gold_captions") or [] if cap.get("caption_type") == "table"],
            "placeholder float needed": audit.get("placeholder_float_needed_candidates") or [],
            "crop swallowed caption": audit.get("crop_swallowed_caption_candidates") or [],
        }
        for key, values in mapping.items():
            for value in values:
                if len(buckets[key]) >= max_examples:
                    break
                buckets[key].append({"doc_id": row["doc_id"], **value})
    return dict(buckets)


def major_failure_type(row: dict[str, Any]) -> str:
    candidates = {
        "missing_caption": as_int(row.get("missing_caption_count")),
        "caption_as_paragraph": as_int(row.get("caption_as_paragraph_count")),
        "duplicate_caption": as_int(row.get("duplicate_caption_count")),
        "caption_without_float": as_int(row.get("caption_without_float_count")),
        "float_without_caption": as_int(row.get("float_without_caption_count")),
        "wrong_float_type_pairing": as_int(row.get("wrong_float_type_count")),
        "crop_swallowed_caption": as_int(row.get("crop_swallowed_caption_count")),
        "placeholder_float_needed": as_int(row.get("placeholder_needed_count")),
    }
    return max(candidates.items(), key=lambda item: item[1])[0]


def render_report(payload: dict[str, Any], rows: list[dict[str, Any]], examples: dict[str, list[dict[str, Any]]]) -> str:
    summary = payload["summary"]
    top_docs = sorted(
        rows,
        key=lambda row: (
            as_int(row.get("missing_caption_count"))
            + as_int(row.get("caption_as_paragraph_count"))
            + as_int(row.get("float_without_caption_count"))
            + as_int(row.get("crop_swallowed_caption_count"))
        ),
        reverse=True,
    )[:20]
    lines = [
        "# Float-Caption Baseline Audit Report",
        "",
        "## Status",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- selected200 docs analyzed: `{summary['docs']}`",
        "- no training / no MinerU / no relabel / no GNN / no generation rerun",
        "- current mainline: `v8 + contentlist merge hint`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| total gold captions | {summary['gold_caption_count']} |",
        f"| total pred captions | {summary['pred_caption_count']} |",
        f"| total v7 caption-like candidates | {summary['caption_candidate_count_v7']} |",
        f"| caption recall proxy pred/gold | {fmt(summary.get('caption_recall_proxy'))} |",
        f"| caption candidate recall proxy v7/gold | {fmt(summary.get('caption_candidate_recall_proxy'))} |",
        f"| mean float_caption_attachment_accuracy | {fmt(summary.get('float_caption_attachment_accuracy'))} |",
        f"| missing caption count | {summary['missing_caption_count']} |",
        f"| caption-as-paragraph count | {summary['caption_as_paragraph_count']} |",
        f"| duplicate caption count | {summary['duplicate_caption_count']} |",
        f"| caption without float count | {summary['caption_without_float_count']} |",
        f"| float without caption count | {summary['float_without_caption_count']} |",
        f"| algorithm caption missing count | {summary['algorithm_caption_missing']} |",
        f"| table caption missing count | {summary['table_caption_missing']} |",
        f"| figure caption missing count | {summary['figure_caption_missing']} |",
        "",
        "## Failure Breakdown",
        "",
        "| failure type | count |",
        "| --- | ---: |",
        f"| missing_caption | {summary['missing_caption_count']} |",
        f"| caption_as_paragraph | {summary['caption_as_paragraph_count']} |",
        f"| caption_without_float | {summary['caption_without_float_count']} |",
        f"| float_without_caption | {summary['float_without_caption_count']} |",
        f"| wrong_float_type_pairing | {summary['wrong_float_type_count']} |",
        f"| duplicate_caption | {summary['duplicate_caption_count']} |",
        f"| algorithm_caption_missing | {summary['algorithm_caption_missing']} |",
        f"| table_caption_missing | {summary['table_caption_missing']} |",
        f"| subfigure_caption_issue | {sum(as_int(row.get('subfigure_caption_issue')) for row in rows)} |",
        f"| crop_swallowed_caption | {summary['crop_swallowed_caption_count']} |",
        f"| placeholder_float_needed | {summary['placeholder_needed_count']} |",
        "",
        "## Top Problem Docs",
        "",
        "| doc_id | gold captions | pred captions | v7 candidates | major failure type |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in top_docs:
        lines.append(
            f"| `{row['doc_id']}` | {row['gold_caption_count']} | {row['pred_caption_count']} | "
            f"{row['caption_candidate_count_v7']} | `{major_failure_type(row)}` |"
        )

    lines += ["", "## Examples", ""]
    for title, bucket_key in [
        ("Missing Captions", "missing captions"),
        ("Caption-Like Paragraphs", "caption-like paragraph"),
        ("Wrong Pairing", "wrong pairing"),
        ("Duplicate Captions", "duplicate caption"),
        ("Algorithm Captions", "algorithm caption"),
        ("Table Captions", "table caption"),
        ("Placeholder Float Needed", "placeholder float needed"),
        ("Crop Swallowed Caption", "crop swallowed caption"),
    ]:
        lines += [f"### {title}", "", "| doc_id | type | number | text |", "| --- | --- | --- | --- |"]
        bucket = examples.get(bucket_key, [])
        if not bucket:
            lines.append("| N/A |  |  |  |")
        for item in bucket[:10]:
            lines.append(
                f"| `{item.get('doc_id')}` | `{item.get('caption_type') or item.get('block_type') or ''}` | "
                f"`{item.get('caption_number') or ''}` | {md(item.get('text') or item.get('normalized_text') or item.get('candidate_source'))} |"
            )
        lines.append("")

    diagnosis = diagnose(summary)
    lines += [
        "## Diagnosis",
        "",
        f"1. {diagnosis['candidate_vs_decoder']}",
        f"2. {diagnosis['recall_vs_pairing']}",
        f"3. {diagnosis['worst_type']}",
        f"4. {diagnosis['body_order_pollution']}",
        f"5. {diagnosis['placeholder']}",
        f"6. {diagnosis['crop']}",
        "",
        "## Next Recommendation",
        "",
        "- FloatCaptionMatcher grammar patch",
        "- caption-float pairing ranker/rules",
        "- placeholder float promotion",
        "- duplicate suppression",
        "- crop bbox excluding caption",
        "- selected200 A/B validation after fix",
    ]
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    number = as_float(value)
    return "N/A" if number is None else f"{number:.6f}"


def diagnose(summary: dict[str, Any]) -> dict[str, str]:
    gold = as_int(summary.get("gold_caption_count"))
    candidates = as_int(summary.get("caption_candidate_count_v7"))
    pred = as_int(summary.get("pred_caption_count"))
    missing = as_int(summary.get("missing_caption_count"))
    as_para = as_int(summary.get("caption_as_paragraph_count"))
    without_float = as_int(summary.get("caption_without_float_count"))
    floats_without = as_int(summary.get("float_without_caption_count"))
    crop = as_int(summary.get("crop_swallowed_caption_count"))
    placeholder = as_int(summary.get("placeholder_needed_count"))
    by_type = {
        "figure": as_int(summary.get("figure_caption_missing")),
        "table": as_int(summary.get("table_caption_missing")),
        "algorithm": as_int(summary.get("algorithm_caption_missing")),
    }
    worst = max(by_type.items(), key=lambda item: item[1])[0]
    if candidates >= gold and pred < candidates:
        candidate_vs_decoder = "v7/full fact layer contains enough caption-like candidates, but decoder/renderer/comparison path does not consume all of them as structural captions."
    elif candidates < gold:
        candidate_vs_decoder = "caption recall is already limited in v7/full fact candidate extraction; some gold captions have no detected v7 candidate."
    else:
        candidate_vs_decoder = "caption candidates and pred captions are close; remaining issues are mostly pairing/type/duplication."
    if missing > (without_float + floats_without):
        recall_vs_pairing = "Main issue is caption recall: many gold captions are absent from pred caption blocks."
    else:
        recall_vs_pairing = "Pairing/attachment is a major issue: captions/floats exist but are not reliably attached."
    return {
        "candidate_vs_decoder": candidate_vs_decoder,
        "recall_vs_pairing": recall_vs_pairing,
        "worst_type": f"Most severe missing type is `{worst}` captions.",
        "body_order_pollution": f"caption-as-paragraph count is `{as_para}`, so caption-like text can still pollute body/order metrics.",
        "placeholder": f"placeholder_float_needed candidates total `{placeholder}`; placeholder float promotion should be audited after matcher fixes.",
        "crop": f"crop_swallowed_caption count is `{crop}`, indicating metadata/crop-contained captions may need crop/caption separation or explicit materialization.",
    }


def render_readiness_report(args: argparse.Namespace, missing: list[str]) -> str:
    lines = [
        "# Float-Caption Baseline Audit Readiness Report",
        "",
        f"- created_at: `{datetime.now(timezone.utc).isoformat()}`",
        "- status: blocked",
        "",
        "## Missing Inputs",
        "",
    ]
    lines.extend(f"- `{item}`" for item in missing)
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing = readiness_check(args)
    if missing:
        report = render_readiness_report(args, missing)
        (args.output_dir / "FLOAT_CAPTION_BASELINE_AUDIT_READINESS_REPORT.md").write_text(report, encoding="utf-8")
        return {"status": "blocked", "missing": missing}

    rows = read_csv(args.breakdown_dir / "doc_failure_breakdown.csv")
    if args.doc_ids:
        wanted = set(args.doc_ids)
        rows = [row for row in rows if row.get("doc_id") in wanted]
    rows.sort(key=lambda row: row.get("doc_id", ""))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        report = render_readiness_report(args, ["No selected200 doc ids found."])
        (args.output_dir / "FLOAT_CAPTION_BASELINE_AUDIT_READINESS_REPORT.md").write_text(report, encoding="utf-8")
        return {"status": "blocked", "missing": ["No selected200 doc ids found."]}

    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in rows:
        doc_id = row.get("doc_id") or ""
        try:
            summary_rows.append(audit_doc(doc_id, row, args.output_dir, args.max_examples))
        except Exception as exc:
            errors.append({"doc_id": doc_id, "error": repr(exc)})

    summary = aggregate(summary_rows)
    examples = collect_examples(args.output_dir, summary_rows, args.max_examples)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "selected200_float_caption_baseline_audit_v1",
        "status": "completed" if not errors else "completed_with_errors",
        "docs": len(summary_rows),
        "errors": errors,
        "no_training": True,
        "no_mineru": True,
        "no_relabel": True,
        "no_gnn": True,
        "no_generation_rerun": True,
        "mainline": "v8 + contentlist merge hint",
        "summary": summary,
        "top_problem_docs": sorted(
            summary_rows,
            key=lambda row: (
                as_int(row.get("missing_caption_count"))
                + as_int(row.get("caption_as_paragraph_count"))
                + as_int(row.get("float_without_caption_count"))
                + as_int(row.get("crop_swallowed_caption_count"))
            ),
            reverse=True,
        )[:20],
    }
    write_csv(args.output_dir / "float_caption_baseline_summary.csv", summary_rows)
    write_json(args.output_dir / "float_caption_baseline_summary.json", payload)
    (args.output_dir / "FLOAT_CAPTION_BASELINE_AUDIT_REPORT.md").write_text(
        render_report(payload, summary_rows, examples),
        encoding="utf-8",
    )
    if errors:
        write_json(args.output_dir / "errors.json", errors)
    return payload


def main() -> int:
    payload = run(build_arg_parser().parse_args())
    if payload.get("status") == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
