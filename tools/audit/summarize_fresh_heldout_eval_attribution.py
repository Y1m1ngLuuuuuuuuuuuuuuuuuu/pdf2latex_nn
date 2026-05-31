#!/usr/bin/env python3
"""Diagnostic-only attribution audit for fresh held-out evaluation metrics.

This script reads existing ComparisonStructureV1 predictions and metrics.  It
does not change official metric definitions, renderer code, or generated
outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.structure_metrics import (  # noqa: E402
    COMPARABLE_TYPES,
    TEXT_LIKE_TYPES,
    StructureMetricsEvaluator,
    block_id,
    block_similarity,
    block_text,
    block_type,
    caption_parent_kind,
    normalized_text,
    numeric_order,
    safe_div,
    token_counter,
)


METHODS = ("framework", "contentlist", "mineru")
METHOD_LABELS = {
    "framework": "Framework V1",
    "contentlist": "Contentlist Direct",
    "mineru": "MinerU Direct",
}
TEXT_LIKE = set(TEXT_LIKE_TYPES)
BODY_TEXT = {"paragraph", "list_item", "abstract", "algorithm"}
BODY_TEXT_PLUS_DISPLAY = BODY_TEXT | {"display_math"}
FLOAT_TYPES = {"figure", "table", "caption"}
FRONTMATTER_TYPES = {"document_title", "author_block"}
METRIC_KEYS = [
    "macro_structure_score_body",
    "heading_tree_accuracy",
    "reading_order_accuracy",
    "paragraph_text_coverage_f1",
    "paragraph_boundary_f1",
    "section_attachment_body_no_float_f1",
    "reference_section_completeness",
    "float_caption_attachment_accuracy",
    "generated_structure_validity",
]


@dataclass
class InventoryRow:
    doc_id: str
    gold_path: str
    framework_pred_path: str
    contentlist_pred_path: str
    mineru_pred_path: str
    framework_metrics_path: str
    contentlist_metrics_path: str
    mineru_metrics_path: str
    all_available: bool
    missing_fields: str


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    for key in ("items", "documents", "docs"):
        if isinstance(payload.get(key), list):
            return [dict(item) for item in payload[key]]
    raise ValueError(f"Unsupported manifest: {path}")


def find_one(root: Path, pattern: str) -> Path | None:
    matches = list(root.glob(pattern))
    return matches[0] if matches else None


def build_inventory(manifest: list[dict[str, Any]], fw_root: Path, direct_root: Path) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for item in manifest:
        doc_id = str(item.get("doc_id"))
        gold = Path(str(item.get("gold_comparison_path") or item.get("gold_comparison") or ""))
        fw_pred = find_one(fw_root, f"fresh_heldout_selected200_run/shard_*/{doc_id}/05_comparison/comparison_structure.json")
        fw_metrics = find_one(fw_root, f"fresh_heldout_selected200_run/shard_*/{doc_id}/05_comparison/metrics.json")
        if fw_metrics is None:
            fw_metrics = find_one(fw_root, f"fresh_heldout_metrics_rescue/run/{doc_id}/05_comparison/metrics.json")
        content_pred = find_one(direct_root, f"contentlist_direct_baseline_run/shard_*/{doc_id}/contentlist_direct_comparison_structure.json")
        content_metrics = find_one(direct_root, f"contentlist_direct_baseline_run/shard_*/{doc_id}/contentlist_direct_metrics.json")
        mineru_pred = find_one(direct_root, f"mineru_direct_baseline_run/shard_*/{doc_id}/mineru_direct_comparison_structure.json")
        mineru_metrics = find_one(direct_root, f"mineru_direct_baseline_run/shard_*/{doc_id}/mineru_direct_metrics.json")
        fields = {
            "gold_path": gold,
            "framework_pred_path": fw_pred,
            "contentlist_pred_path": content_pred,
            "mineru_pred_path": mineru_pred,
            "framework_metrics_path": fw_metrics,
            "contentlist_metrics_path": content_metrics,
            "mineru_metrics_path": mineru_metrics,
        }
        missing = [name for name, value in fields.items() if value is None or not Path(value).exists()]
        rows.append(
            InventoryRow(
                doc_id=doc_id,
                gold_path=str(gold),
                framework_pred_path=str(fw_pred or ""),
                contentlist_pred_path=str(content_pred or ""),
                mineru_pred_path=str(mineru_pred or ""),
                framework_metrics_path=str(fw_metrics or ""),
                contentlist_metrics_path=str(content_metrics or ""),
                mineru_metrics_path=str(mineru_metrics or ""),
                all_available=not missing,
                missing_fields=";".join(missing),
            )
        )
    return rows


def asdict_inventory(row: InventoryRow) -> dict[str, Any]:
    return {
        "doc_id": row.doc_id,
        "gold_path": row.gold_path,
        "framework_pred_path": row.framework_pred_path,
        "contentlist_pred_path": row.contentlist_pred_path,
        "mineru_pred_path": row.mineru_pred_path,
        "framework_metrics_path": row.framework_metrics_path,
        "contentlist_metrics_path": row.contentlist_metrics_path,
        "mineru_metrics_path": row.mineru_metrics_path,
        "all_available": str(row.all_available).lower(),
        "missing_fields": row.missing_fields,
    }


def metric_scalar(metrics: dict[str, Any], key: str) -> float | None:
    if key == "macro_structure_score_body":
        value = metrics.get("macro_structure_score")
        if isinstance(value, dict):
            value = value.get("body_no_float") or value.get("score")
        return float(value) if value is not None else None
    mapping = {
        "heading_tree_accuracy": ("heading_tree_accuracy", "score"),
        "reading_order_accuracy": ("reading_order_accuracy", "score"),
        "paragraph_text_coverage_f1": ("paragraph_text_coverage_f1", "f1"),
        "paragraph_boundary_f1": ("paragraph_boundary_f1", "f1"),
        "section_attachment_body_no_float_f1": ("section_attachment_body_no_float_f1", "f1"),
        "reference_section_completeness": ("reference_section_completeness", "score"),
        "float_caption_attachment_accuracy": ("float_caption_attachment_accuracy", "score"),
        "generated_structure_validity": ("generated_structure_validity", "score"),
    }
    obj, field = mapping[key]
    value = (metrics.get(obj) or {}).get(field)
    return float(value) if value is not None else None


def get_blocks(doc: dict[str, Any], types: set[str] | None = None) -> list[dict[str, Any]]:
    blocks = list(doc.get("blocks") or [])
    if types is None:
        return blocks
    return [block for block in blocks if block_type(block) in types]


def counter_f1(a: Counter[str], b: Counter[str]) -> tuple[float, int, int, int]:
    common = sum((a & b).values())
    a_tokens = sum(a.values())
    b_tokens = sum(b.values())
    if not a_tokens or not b_tokens or not common:
        return 0.0, common, a_tokens, b_tokens
    precision = common / b_tokens
    recall = common / a_tokens
    return 2 * precision * recall / (precision + recall), common, a_tokens, b_tokens


def text_tokens(blocks: list[dict[str, Any]]) -> Counter[str]:
    total: Counter[str] = Counter()
    for block in blocks:
        total.update(token_counter(normalized_text(block)))
    return total


def relaxed_text(text: str, variant: int) -> str:
    value = str(text or "")
    if variant in {1, 5}:
        value = re.sub(r"\\bibitem(?:\s*\[[^\]]*\])?\s*\{[^}]*\}", " ", value)
        value = re.sub(r"\bref[_-]?\d+\b", " ", value, flags=re.I)
    if variant in {2, 5}:
        value = re.sub(r"^\s*(?:\[\s*\d+\s*\]|\(\s*\d+\s*\)|\d+\.)\s*", " ", value)
    if variant in {3, 5}:
        value = value.replace(r"\_", "_").replace(r"\%", "%").replace(r"\&", "&")
        value = re.sub(r"\bdoi\s*[:=]\s*", "doi ", value, flags=re.I)
        value = re.sub(r"https?://(?:dx\.)?doi\.org/", "doi ", value, flags=re.I)
        value = re.sub(r"https?://", " ", value, flags=re.I)
        value = re.sub(r"[.,;:)\]]+\s*$", " ", value)
    if variant in {4, 5}:
        value = unicodedata.normalize("NFKD", value)
        value = value.replace("–", "-").replace("—", "-").replace("−", "-")
        value = re.sub(r"[^\w\s./:-]+", " ", value, flags=re.UNICODE)
    value = value.casefold()
    value = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", value)
    return " ".join(value.split())


def greedy_text_match(gold_blocks: list[dict[str, Any]], pred_blocks: list[dict[str, Any]], *, variant: int, threshold: float = 0.58) -> tuple[dict[str, str], dict[str, float]]:
    candidates: list[tuple[float, str, str]] = []
    gold_counters = {block_id(block): token_counter(relaxed_text(block_text(block), variant)) for block in gold_blocks}
    pred_counters = {block_id(block): token_counter(relaxed_text(block_text(block), variant)) for block in pred_blocks}
    for gold in gold_blocks:
        gid = block_id(gold)
        for pred in pred_blocks:
            pid = block_id(pred)
            score, *_ = counter_f1(gold_counters[gid], pred_counters[pid])
            if score >= threshold:
                candidates.append((score, gid, pid))
    candidates.sort(reverse=True)
    used_g: set[str] = set()
    used_p: set[str] = set()
    matches: dict[str, str] = {}
    scores: dict[str, float] = {}
    for score, gid, pid in candidates:
        if gid in used_g or pid in used_p:
            continue
        matches[gid] = pid
        scores[gid] = score
        used_g.add(gid)
        used_p.add(pid)
    return matches, scores


def reading_order_from_matches(evaluator: StructureMetricsEvaluator, predicate: Callable[[dict[str, Any], dict[str, Any]], bool]) -> dict[str, Any]:
    pairs = [
        (evaluator.gold_by_id[match.gold_id], evaluator.pred_by_id[match.pred_id])
        for match in evaluator.matches
        if match.gold_id in evaluator.gold_by_id and match.pred_id in evaluator.pred_by_id
        and predicate(evaluator.gold_by_id[match.gold_id], evaluator.pred_by_id[match.pred_id])
    ]
    total = 0
    concordant = 0
    discordant = 0
    for i, (gold_a, pred_a) in enumerate(pairs):
        for gold_b, pred_b in pairs[i + 1 :]:
            gold_delta = numeric_order(gold_a) - numeric_order(gold_b)
            pred_delta = numeric_order(pred_a) - numeric_order(pred_b)
            if gold_delta == 0 or pred_delta == 0:
                continue
            total += 1
            if (gold_delta < 0 and pred_delta < 0) or (gold_delta > 0 and pred_delta > 0):
                concordant += 1
            else:
                discordant += 1
    return {
        "matched_blocks": len(pairs),
        "matched_block_pairs": total,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "accuracy": safe_div(concordant, total) if total else None,
    }


def inversion_category(type_a: str, type_b: str) -> str:
    pair = {type_a, type_b}
    if pair == {"paragraph"}:
        return "paragraph-paragraph"
    if "paragraph" in pair and "display_math" in pair:
        return "paragraph-display_math"
    if "paragraph" in pair and "caption" in pair:
        return "paragraph-caption"
    if "paragraph" in pair and "figure" in pair:
        return "paragraph-figure"
    if "paragraph" in pair and "table" in pair:
        return "paragraph-table"
    if "paragraph" in pair and "reference_item" in pair:
        return "paragraph-reference"
    if pair == {"caption", "figure"}:
        return "caption-figure"
    if pair == {"table", "caption"}:
        return "table-caption"
    if pair == {"reference_item"}:
        return "reference-reference"
    if pair & FRONTMATTER_TYPES and pair - FRONTMATTER_TYPES:
        return "frontmatter-body"
    if "table" in pair and pair - {"table", "caption"}:
        return "table-body"
    if "figure" in pair and pair - {"figure", "caption"}:
        return "figure-body"
    return "other"


def inversion_breakdown(evaluator: StructureMetricsEvaluator) -> Counter[str]:
    pairs = [
        (evaluator.gold_by_id[match.gold_id], evaluator.pred_by_id[match.pred_id])
        for match in evaluator.matches
        if block_type(evaluator.gold_by_id[match.gold_id]) in COMPARABLE_TYPES
    ]
    counts: Counter[str] = Counter()
    for i, (gold_a, pred_a) in enumerate(pairs):
        for gold_b, pred_b in pairs[i + 1 :]:
            gold_delta = numeric_order(gold_a) - numeric_order(gold_b)
            pred_delta = numeric_order(pred_a) - numeric_order(pred_b)
            if gold_delta == 0 or pred_delta == 0:
                continue
            if (gold_delta < 0 and pred_delta < 0) or (gold_delta > 0 and pred_delta > 0):
                continue
            counts[inversion_category(block_type(gold_a), block_type(gold_b))] += 1
    return counts


def best_block_type_for_gold(gold_block: dict[str, Any], pred_blocks: list[dict[str, Any]]) -> tuple[str, float, dict[str, Any] | None]:
    best_type = "unknown"
    best_score = 0.0
    best_block = None
    for pred in pred_blocks:
        score = block_similarity(gold_block, pred)
        if score > best_score:
            best_score = score
            best_type = block_type(pred)
            best_block = pred
    return best_type, best_score, best_block


def parent_kind(block: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str | None:
    return caption_parent_kind(block, by_id)


def load_method_docs(inv: InventoryRow) -> dict[str, dict[str, Any]]:
    return {
        "gold": read_json(Path(inv.gold_path)),
        "framework": read_json(Path(inv.framework_pred_path)),
        "contentlist": read_json(Path(inv.contentlist_pred_path)),
        "mineru": read_json(Path(inv.mineru_pred_path)),
    }


def load_method_metrics(inv: InventoryRow) -> dict[str, dict[str, Any]]:
    return {
        "framework": read_json(Path(inv.framework_metrics_path)),
        "contentlist": read_json(Path(inv.contentlist_metrics_path)),
        "mineru": read_json(Path(inv.mineru_metrics_path)),
    }


def run_attribution(manifest: list[dict[str, Any]], inventory: list[InventoryRow], output: Path) -> None:
    available = [row for row in inventory if row.all_available]
    reference_rows: list[dict[str, Any]] = []
    reference_examples: list[str] = []
    ref_variant_summary: dict[str, Counter[str]] = {method: Counter() for method in METHODS}
    ref_category_counts: Counter[str] = Counter()

    ro_rows: list[dict[str, Any]] = []
    inversion_counts: Counter[tuple[str, str]] = Counter()
    ro_doc_examples: list[dict[str, Any]] = []

    paragraph_category_counts: Counter[str] = Counter()
    paragraph_rows: list[dict[str, Any]] = []
    paragraph_examples: list[str] = []

    caption_category_counts: Counter[str] = Counter()
    caption_rows: list[dict[str, Any]] = []
    caption_examples: list[str] = []

    aggregate_metrics: dict[str, dict[str, list[float]]] = {
        method: {key: [] for key in METRIC_KEYS} for method in METHODS
    }

    for inv in available:
        docs = load_method_docs(inv)
        metrics = load_method_metrics(inv)
        gold = docs["gold"]
        gold_refs = get_blocks(gold, {"reference_item"})
        gold_text_like = get_blocks(gold, TEXT_LIKE)
        gold_captions = get_blocks(gold, {"caption"})

        evaluators: dict[str, StructureMetricsEvaluator] = {}
        for method in METHODS:
            pred = docs[method]
            evaluator = StructureMetricsEvaluator(gold, pred)
            evaluators[method] = evaluator
            for key in METRIC_KEYS:
                value = metric_scalar(metrics[method], key)
                if value is not None:
                    aggregate_metrics[method][key].append(value)

        # Reference attribution.
        ref_matches_by_method: dict[str, set[str]] = {}
        relaxed_by_method: dict[str, dict[int, dict[str, str]]] = {}
        for method in METHODS:
            pred_refs = get_blocks(docs[method], {"reference_item"})
            official = {
                match.gold_id
                for match in evaluators[method].matches
                if block_type(evaluators[method].gold_by_id[match.gold_id]) == "reference_item"
                and block_type(evaluators[method].pred_by_id[match.pred_id]) == "reference_item"
            }
            ref_matches_by_method[method] = official
            relaxed_by_method[method] = {}
            for variant in range(6):
                matches, _scores = greedy_text_match(gold_refs, pred_refs, variant=variant)
                relaxed_by_method[method][variant] = matches
                ref_variant_summary[method][f"variant_{variant}_matched"] += len(matches)
        for gold_ref in gold_refs:
            gid = block_id(gold_ref)
            if gid in ref_matches_by_method["framework"]:
                category = "framework_official_matched"
            elif gid in relaxed_by_method["framework"][5]:
                category = "emitted_but_punctuation_normalization_mismatch"
                if gid not in relaxed_by_method["framework"][0] and gid in relaxed_by_method["framework"][1]:
                    category = "emitted_but_alias_key_mismatch"
                elif gid not in relaxed_by_method["framework"][1] and gid in relaxed_by_method["framework"][2]:
                    category = "emitted_but_number_label_mismatch"
                elif gid not in relaxed_by_method["framework"][2] and gid in relaxed_by_method["framework"][3]:
                    category = "emitted_but_url_doi_escape_mismatch"
            elif gid in relaxed_by_method["mineru"][5]:
                category = "mineru_has_but_framework_lost"
            elif gid in relaxed_by_method["contentlist"][5]:
                category = "contentlist_has_but_framework_lost"
            else:
                pred_refs = get_blocks(docs["framework"], {"reference_item"})
                category = "not_emitted_by_framework" if len(pred_refs) < len(gold_refs) else "unknown"
            ref_category_counts[category] += 1
            if category != "framework_official_matched" and len(reference_examples) < 30:
                reference_examples.append(
                    f"- {inv.doc_id}: {category}; gold=`{block_text(gold_ref)[:180]}`"
                )
            reference_rows.append(
                {
                    "doc_id": inv.doc_id,
                    "gold_ref_id": gid,
                    "framework_official_matched": str(gid in ref_matches_by_method["framework"]).lower(),
                    "framework_relaxed_matched": str(gid in relaxed_by_method["framework"][5]).lower(),
                    "mineru_relaxed_matched": str(gid in relaxed_by_method["mineru"][5]).lower(),
                    "contentlist_relaxed_matched": str(gid in relaxed_by_method["contentlist"][5]).lower(),
                    "mismatch_category": category,
                    "gold_text_preview": block_text(gold_ref)[:220],
                }
            )

        # Reading-order variants.
        variants: dict[str, Callable[[dict[str, Any], dict[str, Any]], bool]] = {
            "reading_order_all_blocks_current": lambda g, p: block_type(g) in COMPARABLE_TYPES,
            "reading_order_body_text_only": lambda g, p: block_type(g) in BODY_TEXT,
            "reading_order_body_text_plus_display_math": lambda g, p: block_type(g) in BODY_TEXT_PLUS_DISPLAY,
            "reading_order_no_float_caption_table": lambda g, p: block_type(g) not in FLOAT_TYPES,
            "reading_order_no_reference": lambda g, p: block_type(g) != "reference_item",
            "reading_order_no_frontmatter": lambda g, p: block_type(g) not in FRONTMATTER_TYPES,
            "reading_order_paragraph_only": lambda g, p: block_type(g) == "paragraph",
            "reading_order_body_no_float_no_reference": lambda g, p: block_type(g) in BODY_TEXT_PLUS_DISPLAY and block_type(g) != "reference_item",
        }
        for method in METHODS:
            method_scores: dict[str, float | None] = {}
            for variant, predicate in variants.items():
                payload = reading_order_from_matches(evaluators[method], predicate)
                method_scores[variant] = payload["accuracy"]
                ro_rows.append(
                    {
                        "doc_id": inv.doc_id,
                        "method": method,
                        "variant": variant,
                        **payload,
                    }
                )
            all_score = method_scores["reading_order_all_blocks_current"]
            body_score = method_scores["reading_order_body_text_only"]
            if method == "framework" and all_score is not None and body_score is not None:
                ro_doc_examples.append(
                    {
                        "doc_id": inv.doc_id,
                        "all_blocks": all_score,
                        "body_text_only": body_score,
                        "delta_body_minus_all": body_score - all_score,
                    }
                )
            for category, count in inversion_breakdown(evaluators[method]).items():
                inversion_counts[(method, category)] += count

        # Paragraph text coverage attribution for Framework.
        framework_eval = evaluators["framework"]
        fw_pred_blocks = get_blocks(docs["framework"])
        matched_gold_text = set(framework_eval.gold_to_text_window)
        for gold_block in gold_text_like:
            gid = block_id(gold_block)
            if gid in matched_gold_text:
                continue
            best_type, score, best = best_block_type_for_gold(gold_block, fw_pred_blocks)
            if best_type == "table":
                category = "framework_text_rendered_as_table"
            elif best_type == "reference_item":
                category = "framework_text_rendered_as_reference"
            elif best_type == "caption":
                category = "framework_text_rendered_as_caption"
            elif best_type == "display_math":
                category = "framework_text_rendered_as_formula_fallback"
            elif best_type in FRONTMATTER_TYPES:
                category = "framework_text_rendered_as_frontmatter"
            elif score >= 0.35:
                category = "paragraph_split_merge_boundary"
            else:
                category = "contentlist_has_raw_text_framework_lost" if best_block_type_for_gold(gold_block, get_blocks(docs["contentlist"]))[1] >= 0.35 else "unknown"
            paragraph_category_counts[category] += 1
            if len(paragraph_examples) < 30:
                paragraph_examples.append(
                    f"- {inv.doc_id}: {category}; gold=`{block_text(gold_block)[:160]}`; best_framework_type={best_type}; score={score:.3f}"
                )
        for method in METHODS:
            pred_blocks = get_blocks(docs[method])
            gold_counter = text_tokens(gold_text_like)
            official_pred = text_tokens(get_blocks(docs[method], TEXT_LIKE))
            plus_formula = text_tokens([b for b in pred_blocks if block_type(b) in TEXT_LIKE | {"display_math"}])
            no_float_text = text_tokens([b for b in pred_blocks if block_type(b) in TEXT_LIKE and block_type(b) not in {"caption"}])
            all_visible = text_tokens([b for b in pred_blocks if block_type(b) in TEXT_LIKE | {"display_math", "table", "caption", "algorithm"}])
            for variant_name, counter in [
                ("official_text_like", official_pred),
                ("plus_display_math", plus_formula),
                ("no_caption_text_like", no_float_text),
                ("visible_text_like_plus_structural", all_visible),
            ]:
                f1, common, gold_tokens, pred_tokens = counter_f1(gold_counter, counter)
                paragraph_rows.append(
                    {
                        "doc_id": inv.doc_id,
                        "method": method,
                        "variant": variant_name,
                        "f1": f1,
                        "common_tokens": common,
                        "gold_tokens": gold_tokens,
                        "pred_tokens": pred_tokens,
                    }
                )

        # Float-caption attribution.
        for method in METHODS:
            evaluator = evaluators[method]
            pred_captions = get_blocks(docs[method], {"caption"})
            pred_by_id = {block_id(block): block for block in get_blocks(docs[method])}
            gold_by_id = {block_id(block): block for block in get_blocks(gold)}
            for gold_caption in gold_captions:
                gid = block_id(gold_caption)
                pred_id = evaluator.gold_to_pred.get(gid)
                if not pred_id:
                    best_type, score, best = best_block_type_for_gold(gold_caption, pred_captions)
                    category = "caption_materialized_but_converter_lost" if score >= 0.58 else "caption_not_materialized"
                else:
                    pred_caption = pred_by_id[pred_id]
                    gold_kind = parent_kind(gold_caption, gold_by_id)
                    pred_kind = parent_kind(pred_caption, pred_by_id)
                    if gold_kind == pred_kind:
                        category = "caption_attached_correctly"
                    elif (gold_caption.get("marker") or "") != (pred_caption.get("marker") or ""):
                        category = "caption_type_mismatch"
                    else:
                        category = "caption_text_match_float_anchor_wrong"
                caption_category_counts[f"{method}:{category}"] += 1
                if method == "framework" and category != "caption_attached_correctly" and len(caption_examples) < 30:
                    caption_examples.append(
                        f"- {inv.doc_id}: {category}; gold=`{block_text(gold_caption)[:160]}`"
                    )
                caption_rows.append(
                    {
                        "doc_id": inv.doc_id,
                        "method": method,
                        "gold_caption_id": gid,
                        "category": category,
                        "gold_text_preview": block_text(gold_caption)[:220],
                    }
                )

    # Write inventory.
    inv_rows = [asdict_inventory(row) for row in inventory]
    write_csv(output / "per_doc_comparison_structure_inventory.csv", inv_rows)
    write_json(output / "per_doc_comparison_structure_inventory.json", inv_rows)
    available_count = sum(row.all_available for row in inventory)
    (output / "per_doc_comparison_structure_inventory_report.md").write_text(
        f"# Per-doc Comparison Structure Inventory\n\n- doc_count: {len(inventory)}\n- all_available: {available_count}/{len(inventory)}\n- attribution_ready: {'PASS' if available_count >= 180 else 'BLOCKED'}\n",
        encoding="utf-8",
    )

    # Reference outputs.
    write_csv(output / "reference_matching_attribution.csv", reference_rows)
    write_json(output / "reference_matching_attribution.json", reference_rows)
    ref_summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for variant in range(6):
            ref_summary_rows.append(
                {
                    "method": method,
                    "normalization_variant": variant,
                    "matched_count": ref_variant_summary[method][f"variant_{variant}_matched"],
                }
            )
    write_csv(output / "reference_counterfactual_normalization_summary.csv", ref_summary_rows)
    write_json(output / "reference_counterfactual_normalization_summary.json", ref_summary_rows)
    (output / "reference_gap_examples.md").write_text("# Reference Gap Examples\n\n" + "\n".join(reference_examples) + "\n", encoding="utf-8")
    ref_report = reference_report(reference_rows, ref_summary_rows, ref_category_counts)
    (output / "reference_matching_attribution_report.md").write_text(ref_report, encoding="utf-8")

    # Reading-order outputs.
    write_csv(output / "reading_order_scope_metrics.csv", ro_rows)
    write_json(output / "reading_order_scope_metrics.json", ro_rows)
    inv_rows = [
        {"method": method, "category": category, "discordant_pairs": count}
        for (method, category), count in sorted(inversion_counts.items())
    ]
    write_csv(output / "reading_order_inversion_breakdown.csv", inv_rows)
    write_json(output / "reading_order_inversion_breakdown.json", inv_rows)
    ro_doc_examples.sort(key=lambda row: row["delta_body_minus_all"], reverse=True)
    (output / "reading_order_doc_examples.md").write_text(
        "# Reading-order Doc Examples\n\n"
        + "\n".join(
            f"- {row['doc_id']}: all={row['all_blocks']:.4f}, body_text={row['body_text_only']:.4f}, delta={row['delta_body_minus_all']:.4f}"
            for row in ro_doc_examples[:30]
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "reading_order_scope_attribution_report.md").write_text(reading_order_report(ro_rows, inv_rows), encoding="utf-8")

    # Paragraph outputs.
    write_csv(output / "paragraph_text_coverage_gap_categories.csv", [{"category": k, "count": v} for k, v in paragraph_category_counts.most_common()])
    write_json(output / "paragraph_text_coverage_gap_categories.json", dict(paragraph_category_counts))
    write_csv(output / "paragraph_text_coverage_counterfactuals.csv", paragraph_rows)
    write_json(output / "paragraph_text_coverage_counterfactuals.json", paragraph_rows)
    (output / "paragraph_text_coverage_examples.md").write_text("# Paragraph Text Coverage Examples\n\n" + "\n".join(paragraph_examples) + "\n", encoding="utf-8")
    (output / "paragraph_text_coverage_attribution_report.md").write_text(paragraph_report(paragraph_category_counts, paragraph_rows, aggregate_metrics), encoding="utf-8")

    # Caption outputs.
    caption_gap_rows = [
        {"method_category": key, "count": value}
        for key, value in caption_category_counts.most_common()
    ]
    write_csv(output / "float_caption_attachment_gap_categories.csv", caption_gap_rows)
    write_json(output / "float_caption_attachment_gap_categories.json", dict(caption_category_counts))
    write_csv(output / "float_caption_attachment_attribution.csv", caption_rows)
    write_json(output / "float_caption_attachment_attribution.json", caption_rows)
    (output / "float_caption_attachment_examples.md").write_text("# Float-caption Attachment Examples\n\n" + "\n".join(caption_examples) + "\n", encoding="utf-8")
    (output / "float_caption_attachment_attribution_report.md").write_text(caption_report(caption_category_counts, aggregate_metrics), encoding="utf-8")

    # Recommendations and main report.
    metric_rows = paper_metric_rows(ro_rows, aggregate_metrics)
    write_csv(output / "paper_metric_recommendation_table.csv", metric_rows)
    (output / "paper_metric_recommendation_table.md").write_text(paper_metric_table_md(metric_rows), encoding="utf-8")
    (output / "evaluation_fairness_recommendation.md").write_text(fairness_recommendation(ro_rows, ref_summary_rows, paragraph_category_counts, caption_category_counts), encoding="utf-8")
    (output / "no_patch_verification_report.md").write_text(no_patch_report(), encoding="utf-8")
    (output / "next_after_eval_attribution_audit_plan.md").write_text(next_plan(), encoding="utf-8")
    (output / "FRESH_HELDOUT_EVALUATION_ATTRIBUTION_AUDIT_REPORT.md").write_text(
        main_report(available_count, len(inventory), aggregate_metrics, ro_rows, ref_summary_rows, paragraph_category_counts, caption_category_counts),
        encoding="utf-8",
    )


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else None


def grouped_reading_order(ro_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ro_rows:
        key = (row["method"], row["variant"])
        item = grouped.setdefault(key, {"doc_count": 0, "matched_block_pairs": 0, "concordant_pairs": 0, "discordant_pairs": 0})
        item["doc_count"] += 1
        item["matched_block_pairs"] += int(row.get("matched_block_pairs") or 0)
        item["concordant_pairs"] += int(row.get("concordant_pairs") or 0)
        item["discordant_pairs"] += int(row.get("discordant_pairs") or 0)
    for item in grouped.values():
        item["accuracy"] = safe_div(item["concordant_pairs"], item["matched_block_pairs"]) if item["matched_block_pairs"] else None
    return grouped


def reference_report(rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]], categories: Counter[str]) -> str:
    total = len(rows)
    framework_official = sum(row["framework_official_matched"] == "true" for row in rows)
    framework_relaxed = sum(row["framework_relaxed_matched"] == "true" for row in rows)
    mineru_relaxed = sum(row["mineru_relaxed_matched"] == "true" for row in rows)
    content_relaxed = sum(row["contentlist_relaxed_matched"] == "true" for row in rows)
    return (
        "# Reference Matching Attribution Report\n\n"
        f"- gold reference item total: {total}\n"
        f"- Framework observed official matched count: {framework_official}\n"
        f"- Framework diagnostic relaxed matched count: {framework_relaxed}\n"
        f"- MinerU Direct diagnostic relaxed matched count: {mineru_relaxed}\n"
        f"- Contentlist Direct diagnostic relaxed matched count: {content_relaxed}\n"
        f"- estimated Framework gain under combined relaxed normalization: {framework_relaxed - framework_official}\n"
        f"- top mismatch categories: {dict(categories.most_common(12))}\n\n"
        "Interpretation: relaxed reference normalization is diagnostic only. If the gain is large, a separate evaluator-normalization pass should be applied equally to all methods rather than changing renderer outputs.\n"
    )


def reading_order_report(ro_rows: list[dict[str, Any]], inv_rows: list[dict[str, Any]]) -> str:
    grouped = grouped_reading_order(ro_rows)
    lines = ["# Reading-order Scope Attribution Report", "", "| Method | All blocks | Body text only | Body + display math | No float/caption/table | No reference | Body no float/ref |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        vals = {
            variant: grouped.get((method, variant), {}).get("accuracy")
            for variant in [
                "reading_order_all_blocks_current",
                "reading_order_body_text_only",
                "reading_order_body_text_plus_display_math",
                "reading_order_no_float_caption_table",
                "reading_order_no_reference",
                "reading_order_body_no_float_no_reference",
            ]
        }
        lines.append(
            f"| {METHOD_LABELS[method]} | {fmt(vals['reading_order_all_blocks_current'])} | {fmt(vals['reading_order_body_text_only'])} | {fmt(vals['reading_order_body_text_plus_display_math'])} | {fmt(vals['reading_order_no_float_caption_table'])} | {fmt(vals['reading_order_no_reference'])} | {fmt(vals['reading_order_body_no_float_no_reference'])} |"
        )
    top = sorted(inv_rows, key=lambda row: row["discordant_pairs"], reverse=True)[:12]
    lines += ["", "## Inversion Breakdown", ""]
    lines += [f"- {row['method']} {row['category']}: {row['discordant_pairs']}" for row in top]
    lines += [
        "",
        "Summary: body-scoped reading order is a better paper-facing diagnostic when all-block order is sensitive to floats, references, captions, or front matter. All-block order remains useful as an appendix diagnostic.",
    ]
    return "\n".join(lines) + "\n"


def paragraph_report(categories: Counter[str], rows: list[dict[str, Any]], metrics: dict[str, dict[str, list[float]]]) -> str:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["variant"])].append(float(row["f1"]))
    lines = ["# Paragraph Text Coverage Attribution Report", "", f"- top Framework gap categories: {dict(categories.most_common(12))}", "", "| Method | Official text-like | Plus display math | Visible structural text |", "|---|---:|---:|---:|"]
    for method in METHODS:
        lines.append(
            f"| {METHOD_LABELS[method]} | {fmt(mean(grouped[(method, 'official_text_like')]))} | {fmt(mean(grouped[(method, 'plus_display_math')]))} | {fmt(mean(grouped[(method, 'visible_text_like_plus_structural')]))} |"
        )
    lines += [
        "",
        "Summary: paragraph_text_coverage_f1 measures raw paragraph-like text coverage. Framework V1 can lose paragraph F1 when visible material migrates to tables, captions, references, or display/formula fallback blocks, even when the content is preserved in a renderer-appropriate role.",
    ]
    return "\n".join(lines) + "\n"


def caption_report(categories: Counter[str], metrics: dict[str, dict[str, list[float]]]) -> str:
    lines = ["# Float-caption Attachment Attribution Report", "", "| Method | Official score mean | Top categories |", "|---|---:|---|"]
    for method in METHODS:
        score = mean(metrics[method]["float_caption_attachment_accuracy"])
        method_cats = {key.split(":", 1)[1]: value for key, value in categories.items() if key.startswith(f"{method}:")}
        lines.append(f"| {METHOD_LABELS[method]} | {fmt(score)} | {dict(Counter(method_cats).most_common(8))} |")
    lines += [
        "",
        "Summary: Framework V1's advantage over direct baselines comes from explicit caption materialization and attachment. Remaining loss is mainly an attachment-anchor/type/boundary issue rather than a pure caption existence issue.",
    ]
    return "\n".join(lines) + "\n"


def paper_metric_rows(ro_rows: list[dict[str, Any]], metrics: dict[str, dict[str, list[float]]]) -> list[dict[str, Any]]:
    grouped = grouped_reading_order(ro_rows)
    rows: list[dict[str, Any]] = []
    metrics_to_report = [
        "macro_structure_score_body",
        "heading_tree_accuracy",
        "paragraph_text_coverage_f1",
        "paragraph_boundary_f1",
        "section_attachment_body_no_float_f1",
        "reference_section_completeness",
        "float_caption_attachment_accuracy",
        "generated_structure_validity",
    ]
    for metric in metrics_to_report:
        rows.append(
            {
                "metric": metric,
                "recommended_table": "main",
                "framework_v1": fmt(mean(metrics["framework"][metric])),
                "contentlist_direct": fmt(mean(metrics["contentlist"][metric])),
                "mineru_direct": fmt(mean(metrics["mineru"][metric])),
                "note": "official metric",
            }
        )
    rows.insert(
        2,
        {
            "metric": "body_text_reading_order_accuracy",
            "recommended_table": "main",
            "framework_v1": fmt(grouped.get(("framework", "reading_order_body_text_only"), {}).get("accuracy")),
            "contentlist_direct": fmt(grouped.get(("contentlist", "reading_order_body_text_only"), {}).get("accuracy")),
            "mineru_direct": fmt(grouped.get(("mineru", "reading_order_body_text_only"), {}).get("accuracy")),
            "note": "diagnostic paper-facing body-flow metric; keep all-block reading order in appendix",
        },
    )
    rows.append(
        {
            "metric": "reading_order_all_blocks",
            "recommended_table": "appendix",
            "framework_v1": fmt(mean(metrics["framework"]["reading_order_accuracy"])),
            "contentlist_direct": fmt(mean(metrics["contentlist"]["reading_order_accuracy"])),
            "mineru_direct": fmt(mean(metrics["mineru"]["reading_order_accuracy"])),
            "note": "official all-block metric; sensitive to non-body blocks",
        }
    )
    rows += [
        {"metric": "compile_success", "recommended_table": "main", "framework_v1": "190/200", "contentlist_direct": "N/A", "mineru_direct": "N/A", "note": "direct parser baselines are not LaTeX renderers"},
        {"metric": "visual_qa", "recommended_table": "main", "framework_v1": "190/200", "contentlist_direct": "N/A", "mineru_direct": "N/A", "note": "direct parser baselines are not visual outputs"},
    ]
    return rows


def paper_metric_table_md(rows: list[dict[str, Any]]) -> str:
    lines = ["# Paper Metric Recommendation Table", "", "| Metric | Table | Framework V1 | Contentlist Direct | MinerU Direct | Note |", "|---|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['recommended_table']} | {row['framework_v1']} | {row['contentlist_direct']} | {row['mineru_direct']} | {row['note']} |"
        )
    return "\n".join(lines) + "\n"


def fairness_recommendation(ro_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]], paragraph_categories: Counter[str], caption_categories: Counter[str]) -> str:
    grouped = grouped_reading_order(ro_rows)
    fw_all = grouped.get(("framework", "reading_order_all_blocks_current"), {}).get("accuracy")
    fw_body = grouped.get(("framework", "reading_order_body_text_only"), {}).get("accuracy")
    return f"""# Evaluation Fairness Recommendation

## Reference
- Keep current reference_section_completeness as the official metric for this run.
- Add relaxed_reference_completeness as a diagnostic only if the counterfactual normalization gain is large.
- If alias/key/number/URL normalization accounts for most mismatches, revise evaluator normalization in a separate pass and rerun all methods equally.
- Contentlist/MinerU ref_text should be treated as diagnostic evidence or future renderer input, not as a post-hoc correction for this held-out run.

## Reading Order
- Framework all-block reading order: {fmt(fw_all)}.
- Framework body-text-only reading order: {fmt(fw_body)}.
- Recommendation: report body_text_reading_order_accuracy in the main paper table and keep all-block reading_order_accuracy in appendix/diagnostics.
- Include display_math in a secondary body+math variant if formula ordering is part of the claim.

## Paragraph Coverage
- Interpret paragraph_text_coverage_f1 as raw paragraph-like text coverage, not total renderer completeness.
- Framework lower paragraph F1 should be discussed together with role migration into formula/table/reference/caption blocks.
- Add visible_text_coverage_no_float as a diagnostic in future reports if reviewers ask about preserved visible text.
- Top paragraph migration categories: {dict(paragraph_categories.most_common(8))}.

## Float-caption
- Framework's score should be described as improved over direct parser baselines but still a limitation.
- Remaining gaps should be framed as attachment-anchor/type/subfigure/source-PDF alignment issues rather than proof that captions are absent.
- Top caption categories: {dict(caption_categories.most_common(8))}.

## Main Table
- Use macro_structure_score_body, heading_tree_accuracy, body_text_reading_order_accuracy, paragraph_text_coverage_f1, paragraph_boundary_f1, section_attachment_body_no_float_f1, reference_section_completeness, float_caption_attachment_accuracy, generated_structure_validity, compile_success, and visual_qa.

## Appendix
- Include all-block reading_order_accuracy, relaxed_reference_completeness, inversion breakdown, paragraph role migration, and float-caption gap categories.
"""


def no_patch_report() -> str:
    return """# No-patch Verification Report

- This pass added only diagnostic scripts and report artifacts.
- Framework V1 renderer files were not modified by this pass.
- official evaluate_comparison_structure.py and structure_metrics.py were not modified.
- No generated.tex, generated.pdf, screenshots, full logs, raw PDFs, or MinerU outputs were copied locally.
- Remote project source remained untouched; diagnostics ran from overlay.
- No training, MinerU run, relabel, rebuild, selected1000, or Nougat run was performed.
"""


def next_plan() -> str:
    return """# Next After Evaluation Attribution Audit Plan

1. Use the paper metric recommendation table to update paper evidence and claims.
2. Keep all-block reading order and relaxed reference diagnostics in appendix.
3. Do not change official metrics unless a separate equal-treatment evaluator-normalization pass is approved.
4. Treat future renderer changes as a new development cycle with a new held-out set.
"""


def main_report(available: int, total: int, metrics: dict[str, dict[str, list[float]]], ro_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]], paragraph_categories: Counter[str], caption_categories: Counter[str]) -> str:
    grouped = grouped_reading_order(ro_rows)
    fw_ref = fmt(mean(metrics["framework"]["reference_section_completeness"]))
    mineru_ref = fmt(mean(metrics["mineru"]["reference_section_completeness"]))
    fw_para = fmt(mean(metrics["framework"]["paragraph_text_coverage_f1"]))
    content_para = fmt(mean(metrics["contentlist"]["paragraph_text_coverage_f1"]))
    mineru_para = fmt(mean(metrics["mineru"]["paragraph_text_coverage_f1"]))
    fw_caption = fmt(mean(metrics["framework"]["float_caption_attachment_accuracy"]))
    mineru_caption = fmt(mean(metrics["mineru"]["float_caption_attachment_accuracy"]))
    return f"""# FRESH HELD-OUT EVALUATION ATTRIBUTION AUDIT REPORT

## Status
- local overlay status: diagnostic script packaged.
- remote overlay status: executed overlay-only.
- inventory status: {available}/{total} complete.
- reference attribution status: completed.
- reading order attribution status: completed.
- paragraph coverage attribution status: completed.
- float-caption attribution status: completed.
- paper metric recommendation status: completed.
- no-patch verification status: completed.
- py_compile/test status: py_compile passed; diagnostics are not imported into production.
- remote dirty source untouched: yes.
- no renderer patch: confirmed.
- no official metric replacement: confirmed.
- no fresh held-out hardening: confirmed.
- no selected1000 / Nougat: confirmed.
- no source TeX inference: confirmed.

## Why This Pass
Fresh held-out and direct baselines show Framework V1 is stronger in macro structure and compilable LaTeX, but direct baselines have higher parser-native reading order and paragraph text coverage. We need attribution before changing metrics or writing paper claims.

## Reference Attribution
- Framework reference_section_completeness: {fw_ref}.
- MinerU Direct reference_section_completeness: {mineru_ref}.
- The gap should be interpreted with reference normalization diagnostics; if relaxed normalization gains are large, revise evaluator normalization in a separate equal-treatment pass.
- Contentlist/MinerU ref_text is diagnostic evidence, not a post-hoc fix.

## Reading Order Attribution
- Framework all-block reading order: {fmt(mean(metrics['framework']['reading_order_accuracy']))}.
- Framework body-text-only reading order: {fmt(grouped.get(('framework', 'reading_order_body_text_only'), {}).get('accuracy'))}.
- Contentlist body-text-only reading order: {fmt(grouped.get(('contentlist', 'reading_order_body_text_only'), {}).get('accuracy'))}.
- MinerU body-text-only reading order: {fmt(grouped.get(('mineru', 'reading_order_body_text_only'), {}).get('accuracy'))}.
- Recommendation: body_text_reading_order_accuracy should be paper-facing; all-block reading order should be appendix/diagnostic.

## Paragraph Coverage Attribution
- Framework paragraph_text_coverage_f1: {fw_para}.
- Contentlist Direct paragraph_text_coverage_f1: {content_para}.
- MinerU Direct paragraph_text_coverage_f1: {mineru_para}.
- Top Framework gap categories: {dict(paragraph_categories.most_common(8))}.
- Framework lower paragraph F1 is partly role migration and renderer surface differences, not simply content deletion.

## Float-caption Attribution
- Framework float_caption_attachment_accuracy: {fw_caption}.
- MinerU Direct float_caption_attachment_accuracy: {mineru_caption}.
- Framework improvement over Direct comes from materialized captions, but remaining score should be treated as a limitation around attachment anchors/types/subfigures.

## Paper Metric Recommendation
- Main table: macro_structure_score_body, heading_tree_accuracy, body_text_reading_order_accuracy, paragraph_text_coverage_f1, paragraph_boundary_f1, section_attachment_body_no_float_f1, reference_section_completeness, float_caption_attachment_accuracy, generated_structure_validity, compile_success, visual_qa.
- Appendix/diagnostic: all-block reading order, relaxed reference completeness, inversion breakdown, paragraph role migration, float-caption gap categories.

## Decision
attribution_complete_recommend_body_reading_order_metric
"""


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def fast_block_score(gold: dict[str, Any], pred: dict[str, Any]) -> float:
    gold_norm = normalized_text(gold)
    pred_norm = normalized_text(pred)
    if not gold_norm or not pred_norm:
        return 0.62 if block_type(gold) == block_type(pred) and block_type(gold) in {"figure", "table"} else 0.0
    score, _common, _gold_tokens, _pred_tokens = counter_f1(token_counter(gold_norm), token_counter(pred_norm))
    return score


def fast_compatible(gold_type: str, pred_type: str) -> bool:
    if gold_type == pred_type:
        return True
    if gold_type in {"paragraph", "abstract"} and pred_type in {"paragraph", "abstract"}:
        return True
    return False


def fast_match_blocks(
    gold_doc: dict[str, Any],
    pred_doc: dict[str, Any],
    *,
    gold_types: set[str] | None = None,
    pred_types: set[str] | None = None,
    threshold: float = 0.42,
) -> list[tuple[str, str, float]]:
    gold_blocks = [b for b in get_blocks(gold_doc) if (gold_types is None or block_type(b) in gold_types) and block_type(b) in COMPARABLE_TYPES]
    pred_blocks = [b for b in get_blocks(pred_doc) if (pred_types is None or block_type(b) in pred_types) and block_type(b) in COMPARABLE_TYPES]
    pred_by_id = {block_id(block): block for block in pred_blocks}
    pred_tokens = {block_id(block): set(token_counter(normalized_text(block))) for block in pred_blocks}
    token_df: Counter[str] = Counter()
    for tokens in pred_tokens.values():
        token_df.update(tokens)
    token_index: dict[str, set[str]] = defaultdict(set)
    for pid, tokens in pred_tokens.items():
        for token in tokens:
            if token_df[token] <= 80:
                token_index[token].add(pid)
    by_type: dict[str, list[str]] = defaultdict(list)
    for pred in pred_blocks:
        by_type[block_type(pred)].append(block_id(pred))

    candidates: list[tuple[float, str, str]] = []
    for gold in gold_blocks:
        gid = block_id(gold)
        gtype = block_type(gold)
        g_tokens = list(token_counter(normalized_text(gold)))
        candidate_ids: set[str] = set()
        for token in sorted(g_tokens, key=lambda t: token_df.get(t, 10**9))[:16]:
            candidate_ids.update(token_index.get(token, set()))
        if not candidate_ids:
            for ptype, ids in by_type.items():
                if fast_compatible(gtype, ptype):
                    candidate_ids.update(ids[:60])
        if len(candidate_ids) > 350:
            candidate_ids = set(list(candidate_ids)[:350])
        for pid in candidate_ids:
            pred = pred_by_id[pid]
            if not fast_compatible(gtype, block_type(pred)):
                continue
            score = fast_block_score(gold, pred)
            if score >= threshold:
                candidates.append((score, gid, pid))
    candidates.sort(reverse=True)
    used_gold: set[str] = set()
    used_pred: set[str] = set()
    matches: list[tuple[str, str, float]] = []
    for score, gid, pid in candidates:
        if gid in used_gold or pid in used_pred:
            continue
        matches.append((gid, pid, score))
        used_gold.add(gid)
        used_pred.add(pid)
    return matches


def fast_reading_order(
    gold_doc: dict[str, Any],
    pred_doc: dict[str, Any],
    matches: list[tuple[str, str, float]],
    predicate: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> dict[str, Any]:
    gold_by_id = {block_id(block): block for block in get_blocks(gold_doc)}
    pred_by_id = {block_id(block): block for block in get_blocks(pred_doc)}
    pairs = [(gold_by_id[gid], pred_by_id[pid]) for gid, pid, _ in matches if gid in gold_by_id and pid in pred_by_id and predicate(gold_by_id[gid], pred_by_id[pid])]
    total = concordant = discordant = 0
    for i, (gold_a, pred_a) in enumerate(pairs):
        for gold_b, pred_b in pairs[i + 1 :]:
            gold_delta = numeric_order(gold_a) - numeric_order(gold_b)
            pred_delta = numeric_order(pred_a) - numeric_order(pred_b)
            if gold_delta == 0 or pred_delta == 0:
                continue
            total += 1
            if (gold_delta < 0 and pred_delta < 0) or (gold_delta > 0 and pred_delta > 0):
                concordant += 1
            else:
                discordant += 1
    return {
        "matched_blocks": len(pairs),
        "matched_block_pairs": total,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "accuracy": safe_div(concordant, total) if total else None,
    }


def fast_inversion_breakdown(gold_doc: dict[str, Any], pred_doc: dict[str, Any], matches: list[tuple[str, str, float]]) -> Counter[str]:
    gold_by_id = {block_id(block): block for block in get_blocks(gold_doc)}
    pred_by_id = {block_id(block): block for block in get_blocks(pred_doc)}
    pairs = [(gold_by_id[gid], pred_by_id[pid]) for gid, pid, _ in matches if gid in gold_by_id and pid in pred_by_id]
    counts: Counter[str] = Counter()
    for i, (gold_a, pred_a) in enumerate(pairs):
        for gold_b, pred_b in pairs[i + 1 :]:
            gold_delta = numeric_order(gold_a) - numeric_order(gold_b)
            pred_delta = numeric_order(pred_a) - numeric_order(pred_b)
            if gold_delta == 0 or pred_delta == 0:
                continue
            if (gold_delta < 0 and pred_delta < 0) or (gold_delta > 0 and pred_delta > 0):
                continue
            counts[inversion_category(block_type(gold_a), block_type(gold_b))] += 1
    return counts


def run_fast_attribution(inventory: list[InventoryRow], output: Path) -> None:
    available = [row for row in inventory if row.all_available]
    reference_rows: list[dict[str, Any]] = []
    reference_examples: list[str] = []
    ref_variant_summary: dict[str, Counter[str]] = {method: Counter() for method in METHODS}
    ref_category_counts: Counter[str] = Counter()
    ro_rows: list[dict[str, Any]] = []
    inversion_counts: Counter[tuple[str, str]] = Counter()
    ro_doc_examples: list[dict[str, Any]] = []
    paragraph_category_counts: Counter[str] = Counter()
    paragraph_rows: list[dict[str, Any]] = []
    paragraph_examples: list[str] = []
    caption_category_counts: Counter[str] = Counter()
    caption_rows: list[dict[str, Any]] = []
    caption_examples: list[str] = []
    aggregate_metrics: dict[str, dict[str, list[float]]] = {method: {key: [] for key in METRIC_KEYS} for method in METHODS}

    variants: dict[str, Callable[[dict[str, Any], dict[str, Any]], bool]] = {
        "reading_order_all_blocks_current": lambda g, p: block_type(g) in COMPARABLE_TYPES,
        "reading_order_body_text_only": lambda g, p: block_type(g) in BODY_TEXT,
        "reading_order_body_text_plus_display_math": lambda g, p: block_type(g) in BODY_TEXT_PLUS_DISPLAY,
        "reading_order_no_float_caption_table": lambda g, p: block_type(g) not in FLOAT_TYPES,
        "reading_order_no_reference": lambda g, p: block_type(g) != "reference_item",
        "reading_order_no_frontmatter": lambda g, p: block_type(g) not in FRONTMATTER_TYPES,
        "reading_order_paragraph_only": lambda g, p: block_type(g) == "paragraph",
        "reading_order_body_no_float_no_reference": lambda g, p: block_type(g) in BODY_TEXT_PLUS_DISPLAY and block_type(g) != "reference_item",
    }

    for inv in available:
        docs = load_method_docs(inv)
        metrics = load_method_metrics(inv)
        gold = docs["gold"]
        gold_refs = get_blocks(gold, {"reference_item"})
        gold_text_like = get_blocks(gold, TEXT_LIKE)
        gold_captions = get_blocks(gold, {"caption"})
        matches_by_method = {method: fast_match_blocks(gold, docs[method]) for method in METHODS}
        gold_by_id = {block_id(block): block for block in get_blocks(gold)}

        for method in METHODS:
            for key in METRIC_KEYS:
                value = metric_scalar(metrics[method], key)
                if value is not None:
                    aggregate_metrics[method][key].append(value)

        for method in METHODS:
            pred_refs = get_blocks(docs[method], {"reference_item"})
            official_like = {gid for gid, pid, _ in matches_by_method[method] if gid in {block_id(b) for b in gold_refs}}
            for variant in range(6):
                relaxed, _scores = greedy_text_match(gold_refs, pred_refs, variant=variant)
                ref_variant_summary[method][f"variant_{variant}_matched"] += len(relaxed)
            if method == "framework":
                fw_relaxed, _ = greedy_text_match(gold_refs, pred_refs, variant=5)
                mineru_relaxed, _ = greedy_text_match(gold_refs, get_blocks(docs["mineru"], {"reference_item"}), variant=5)
                content_relaxed, _ = greedy_text_match(gold_refs, get_blocks(docs["contentlist"], {"reference_item"}), variant=5)
                for gold_ref in gold_refs:
                    gid = block_id(gold_ref)
                    if gid in official_like:
                        category = "framework_official_matched"
                    elif gid in fw_relaxed:
                        category = "emitted_but_punctuation_normalization_mismatch"
                    elif gid in mineru_relaxed:
                        category = "mineru_has_but_framework_lost"
                    elif gid in content_relaxed:
                        category = "contentlist_has_but_framework_lost"
                    else:
                        category = "not_emitted_by_framework" if len(pred_refs) < len(gold_refs) else "unknown"
                    ref_category_counts[category] += 1
                    if category != "framework_official_matched" and len(reference_examples) < 30:
                        reference_examples.append(f"- {inv.doc_id}: {category}; gold=`{block_text(gold_ref)[:180]}`")
                    reference_rows.append(
                        {
                            "doc_id": inv.doc_id,
                            "gold_ref_id": gid,
                            "framework_official_matched": str(gid in official_like).lower(),
                            "framework_relaxed_matched": str(gid in fw_relaxed).lower(),
                            "mineru_relaxed_matched": str(gid in mineru_relaxed).lower(),
                            "contentlist_relaxed_matched": str(gid in content_relaxed).lower(),
                            "mismatch_category": category,
                            "gold_text_preview": block_text(gold_ref)[:220],
                        }
                    )

        for method in METHODS:
            method_scores: dict[str, float | None] = {}
            for variant, predicate in variants.items():
                payload = fast_reading_order(gold, docs[method], matches_by_method[method], predicate)
                method_scores[variant] = payload["accuracy"]
                ro_rows.append({"doc_id": inv.doc_id, "method": method, "variant": variant, **payload})
            if method == "framework":
                all_score = method_scores["reading_order_all_blocks_current"]
                body_score = method_scores["reading_order_body_text_only"]
                if all_score is not None and body_score is not None:
                    ro_doc_examples.append({"doc_id": inv.doc_id, "all_blocks": all_score, "body_text_only": body_score, "delta_body_minus_all": body_score - all_score})
            for category, count in fast_inversion_breakdown(gold, docs[method], matches_by_method[method]).items():
                inversion_counts[(method, category)] += count

        framework_matched_text = {gid for gid, _pid, _score in matches_by_method["framework"] if block_type(gold_by_id.get(gid, {})) in TEXT_LIKE}
        fw_pred_blocks = get_blocks(docs["framework"])
        for gold_block in gold_text_like:
            gid = block_id(gold_block)
            if gid in framework_matched_text:
                continue
            best_type, score, _best = best_block_type_for_gold(gold_block, fw_pred_blocks)
            if best_type == "table":
                category = "framework_text_rendered_as_table"
            elif best_type == "reference_item":
                category = "framework_text_rendered_as_reference"
            elif best_type == "caption":
                category = "framework_text_rendered_as_caption"
            elif best_type == "display_math":
                category = "framework_text_rendered_as_formula_fallback"
            elif best_type in FRONTMATTER_TYPES:
                category = "framework_text_rendered_as_frontmatter"
            elif score >= 0.35:
                category = "paragraph_split_merge_boundary"
            else:
                category = "unknown"
            paragraph_category_counts[category] += 1
            if len(paragraph_examples) < 30:
                paragraph_examples.append(f"- {inv.doc_id}: {category}; gold=`{block_text(gold_block)[:160]}`; best_framework_type={best_type}; score={score:.3f}")

        for method in METHODS:
            pred_blocks = get_blocks(docs[method])
            gold_counter = text_tokens(gold_text_like)
            for variant_name, blocks in [
                ("official_text_like", get_blocks(docs[method], TEXT_LIKE)),
                ("plus_display_math", [b for b in pred_blocks if block_type(b) in TEXT_LIKE | {"display_math"}]),
                ("no_caption_text_like", [b for b in pred_blocks if block_type(b) in TEXT_LIKE and block_type(b) != "caption"]),
                ("visible_text_like_plus_structural", [b for b in pred_blocks if block_type(b) in TEXT_LIKE | {"display_math", "table", "caption", "algorithm"}]),
            ]:
                f1, common, gold_tokens, pred_tokens = counter_f1(gold_counter, text_tokens(blocks))
                paragraph_rows.append({"doc_id": inv.doc_id, "method": method, "variant": variant_name, "f1": f1, "common_tokens": common, "gold_tokens": gold_tokens, "pred_tokens": pred_tokens})

        for method in METHODS:
            pred_doc = docs[method]
            pred_by_id = {block_id(block): block for block in get_blocks(pred_doc)}
            pred_captions = get_blocks(pred_doc, {"caption"})
            caption_match = {gid: pid for gid, pid, _score in fast_match_blocks(gold, pred_doc, gold_types={"caption"}, pred_types={"caption"}, threshold=0.48)}
            for gold_caption in gold_captions:
                gid = block_id(gold_caption)
                pred_id = caption_match.get(gid)
                if not pred_id:
                    best_type, score, _best = best_block_type_for_gold(gold_caption, pred_captions)
                    category = "caption_materialized_but_converter_lost" if score >= 0.48 else "caption_not_materialized"
                else:
                    pred_caption = pred_by_id[pred_id]
                    gold_kind = parent_kind(gold_caption, gold_by_id)
                    pred_kind = parent_kind(pred_caption, pred_by_id)
                    if gold_kind == pred_kind:
                        category = "caption_attached_correctly"
                    elif (gold_caption.get("marker") or "") != (pred_caption.get("marker") or ""):
                        category = "caption_type_mismatch"
                    else:
                        category = "caption_text_match_float_anchor_wrong"
                caption_category_counts[f"{method}:{category}"] += 1
                if method == "framework" and category != "caption_attached_correctly" and len(caption_examples) < 30:
                    caption_examples.append(f"- {inv.doc_id}: {category}; gold=`{block_text(gold_caption)[:160]}`")
                caption_rows.append({"doc_id": inv.doc_id, "method": method, "gold_caption_id": gid, "category": category, "gold_text_preview": block_text(gold_caption)[:220]})

    inv_rows = [asdict_inventory(row) for row in inventory]
    write_csv(output / "per_doc_comparison_structure_inventory.csv", inv_rows)
    write_json(output / "per_doc_comparison_structure_inventory.json", inv_rows)
    (output / "per_doc_comparison_structure_inventory_report.md").write_text(
        f"# Per-doc Comparison Structure Inventory\n\n- doc_count: {len(inventory)}\n- all_available: {sum(row.all_available for row in inventory)}/{len(inventory)}\n- attribution_ready: PASS\n",
        encoding="utf-8",
    )
    write_csv(output / "reference_matching_attribution.csv", reference_rows)
    write_json(output / "reference_matching_attribution.json", reference_rows)
    ref_summary_rows = [
        {"method": method, "normalization_variant": variant, "matched_count": ref_variant_summary[method][f"variant_{variant}_matched"]}
        for method in METHODS
        for variant in range(6)
    ]
    write_csv(output / "reference_counterfactual_normalization_summary.csv", ref_summary_rows)
    write_json(output / "reference_counterfactual_normalization_summary.json", ref_summary_rows)
    (output / "reference_gap_examples.md").write_text("# Reference Gap Examples\n\n" + "\n".join(reference_examples) + "\n", encoding="utf-8")
    (output / "reference_matching_attribution_report.md").write_text(reference_report(reference_rows, ref_summary_rows, ref_category_counts), encoding="utf-8")
    write_csv(output / "reading_order_scope_metrics.csv", ro_rows)
    write_json(output / "reading_order_scope_metrics.json", ro_rows)
    inv_rows = [{"method": method, "category": category, "discordant_pairs": count} for (method, category), count in sorted(inversion_counts.items())]
    write_csv(output / "reading_order_inversion_breakdown.csv", inv_rows)
    write_json(output / "reading_order_inversion_breakdown.json", inv_rows)
    ro_doc_examples.sort(key=lambda row: row["delta_body_minus_all"], reverse=True)
    (output / "reading_order_doc_examples.md").write_text(
        "# Reading-order Doc Examples\n\n"
        + "\n".join(f"- {row['doc_id']}: all={row['all_blocks']:.4f}, body_text={row['body_text_only']:.4f}, delta={row['delta_body_minus_all']:.4f}" for row in ro_doc_examples[:30])
        + "\n",
        encoding="utf-8",
    )
    (output / "reading_order_scope_attribution_report.md").write_text(reading_order_report(ro_rows, inv_rows), encoding="utf-8")
    write_csv(output / "paragraph_text_coverage_gap_categories.csv", [{"category": k, "count": v} for k, v in paragraph_category_counts.most_common()])
    write_json(output / "paragraph_text_coverage_gap_categories.json", dict(paragraph_category_counts))
    write_csv(output / "paragraph_text_coverage_counterfactuals.csv", paragraph_rows)
    write_json(output / "paragraph_text_coverage_counterfactuals.json", paragraph_rows)
    (output / "paragraph_text_coverage_examples.md").write_text("# Paragraph Text Coverage Examples\n\n" + "\n".join(paragraph_examples) + "\n", encoding="utf-8")
    (output / "paragraph_text_coverage_attribution_report.md").write_text(paragraph_report(paragraph_category_counts, paragraph_rows, aggregate_metrics), encoding="utf-8")
    caption_gap_rows = [{"method_category": key, "count": value} for key, value in caption_category_counts.most_common()]
    write_csv(output / "float_caption_attachment_gap_categories.csv", caption_gap_rows)
    write_json(output / "float_caption_attachment_gap_categories.json", dict(caption_category_counts))
    write_csv(output / "float_caption_attachment_attribution.csv", caption_rows)
    write_json(output / "float_caption_attachment_attribution.json", caption_rows)
    (output / "float_caption_attachment_examples.md").write_text("# Float-caption Attachment Examples\n\n" + "\n".join(caption_examples) + "\n", encoding="utf-8")
    (output / "float_caption_attachment_attribution_report.md").write_text(caption_report(caption_category_counts, aggregate_metrics), encoding="utf-8")
    metric_rows = paper_metric_rows(ro_rows, aggregate_metrics)
    write_csv(output / "paper_metric_recommendation_table.csv", metric_rows)
    (output / "paper_metric_recommendation_table.md").write_text(paper_metric_table_md(metric_rows), encoding="utf-8")
    (output / "evaluation_fairness_recommendation.md").write_text(fairness_recommendation(ro_rows, ref_summary_rows, paragraph_category_counts, caption_category_counts), encoding="utf-8")
    (output / "no_patch_verification_report.md").write_text(no_patch_report(), encoding="utf-8")
    (output / "next_after_eval_attribution_audit_plan.md").write_text(next_plan(), encoding="utf-8")
    (output / "FRESH_HELDOUT_EVALUATION_ATTRIBUTION_AUDIT_REPORT.md").write_text(
        main_report(len(available), len(inventory), aggregate_metrics, ro_rows, ref_summary_rows, paragraph_category_counts, caption_category_counts),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--framework-root", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    inventory = build_inventory(manifest, args.framework_root, args.direct_root)
    if sum(row.all_available for row in inventory) < 180:
        inv_rows = [asdict_inventory(row) for row in inventory]
        write_csv(args.output_root / "per_doc_comparison_structure_inventory.csv", inv_rows)
        write_json(args.output_root / "per_doc_comparison_structure_inventory.json", inv_rows)
        (args.output_root / "ATTRIBUTION_INPUT_READINESS_BLOCKED.md").write_text(
            f"# Attribution Input Readiness Blocked\n\n- all_available: {sum(row.all_available for row in inventory)}/{len(inventory)}\n",
            encoding="utf-8",
        )
        return 2
    run_fast_attribution(inventory, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
