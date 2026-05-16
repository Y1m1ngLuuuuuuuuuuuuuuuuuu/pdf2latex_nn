"""Metrics for the shared comparison-structure JSON.

The evaluator compares a gold structure, usually converted from the source
LaTeX, with a predicted structure, such as our generated LaTeX or Nougat MMD.
It deliberately evaluates structure rather than low-level OCR quality.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT_ID = "ROOT"
TEXT_LIKE_TYPES = {"paragraph", "abstract", "list_item", "reference_item", "caption"}
CONTENT_TYPES = {"paragraph", "abstract", "list_item", "display_math", "figure", "table", "caption", "algorithm"}
COMPARABLE_TYPES = {
    "document_title",
    "author_block",
    "abstract",
    "heading",
    "paragraph",
    "list_item",
    "display_math",
    "figure",
    "table",
    "caption",
    "reference_item",
    "algorithm",
}
STRICT_TYPE_GROUPS = [
    {"heading"},
    {"reference_item"},
    {"caption"},
    {"display_math"},
    {"figure"},
    {"table"},
    {"list_item"},
    {"document_title"},
    {"author_block"},
    {"paragraph", "abstract"},
    {"algorithm"},
]


@dataclass(frozen=True)
class BlockMatch:
    gold_id: str
    pred_id: str
    score: float


def load_comparison_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_comparison_structures(
    gold: dict[str, Any],
    pred: dict[str, Any],
    *,
    match_threshold: float = 0.58,
) -> dict[str, Any]:
    evaluator = StructureMetricsEvaluator(gold, pred, match_threshold=match_threshold)
    return evaluator.evaluate()


class StructureMetricsEvaluator:
    def __init__(self, gold: dict[str, Any], pred: dict[str, Any], *, match_threshold: float = 0.58) -> None:
        self.gold = gold
        self.pred = pred
        self.match_threshold = match_threshold
        self.gold_blocks = list(gold.get("blocks") or [])
        self.pred_blocks = list(pred.get("blocks") or [])
        self.gold_by_id = {str(block.get("block_id")): block for block in self.gold_blocks}
        self.pred_by_id = {str(block.get("block_id")): block for block in self.pred_blocks}
        self.matches = self.match_blocks()
        self.gold_to_pred = {match.gold_id: match.pred_id for match in self.matches}
        self.pred_to_gold = {match.pred_id: match.gold_id for match in self.matches}

    def evaluate(self) -> dict[str, Any]:
        metrics = {
            "schema_version": "comparison_metrics_v1",
            "gold_doc_id": self.gold.get("doc_id"),
            "pred_doc_id": self.pred.get("doc_id"),
            "pred_source_format": self.pred.get("source_format"),
            "matching": self.matching_summary(),
            "heading_tree_accuracy": self.heading_tree_accuracy(),
            "reading_order_accuracy": self.reading_order_accuracy(),
            "paragraph_merge_f1": self.paragraph_merge_f1(),
            "section_attachment_f1": self.section_attachment_f1(),
            "reference_section_completeness": self.reference_section_completeness(),
            "float_caption_attachment_accuracy": self.float_caption_attachment_accuracy(),
            "generated_structure_validity": self.generated_structure_validity(),
        }
        metrics["macro_structure_score"] = macro_score(
            [
                metrics["heading_tree_accuracy"].get("score"),
                metrics["reading_order_accuracy"].get("score"),
                metrics["paragraph_merge_f1"].get("f1"),
                metrics["section_attachment_f1"].get("f1"),
                metrics["reference_section_completeness"].get("score"),
                metrics["float_caption_attachment_accuracy"].get("score"),
                metrics["generated_structure_validity"].get("score"),
            ]
        )
        return metrics

    def match_blocks(self) -> list[BlockMatch]:
        candidates: list[BlockMatch] = []
        for gold_block in self.gold_blocks:
            if block_type(gold_block) not in COMPARABLE_TYPES:
                continue
            for pred_block in self.pred_blocks:
                if block_type(pred_block) not in COMPARABLE_TYPES:
                    continue
                if not compatible_types(block_type(gold_block), block_type(pred_block)):
                    continue
                score = block_similarity(gold_block, pred_block)
                if score >= self.match_threshold:
                    candidates.append(BlockMatch(block_id(gold_block), block_id(pred_block), score))
        candidates.sort(key=lambda item: item.score, reverse=True)
        used_gold: set[str] = set()
        used_pred: set[str] = set()
        matches: list[BlockMatch] = []
        for candidate in candidates:
            if candidate.gold_id in used_gold or candidate.pred_id in used_pred:
                continue
            matches.append(candidate)
            used_gold.add(candidate.gold_id)
            used_pred.add(candidate.pred_id)
        return sorted(matches, key=lambda item: self.gold_by_id[item.gold_id].get("order", 0))

    def matching_summary(self) -> dict[str, Any]:
        return {
            "matched_blocks": len(self.matches),
            "gold_blocks": len(self.gold_blocks),
            "pred_blocks": len(self.pred_blocks),
            "coverage_gold": safe_div(len(self.matches), len([b for b in self.gold_blocks if block_type(b) in COMPARABLE_TYPES])),
            "coverage_pred": safe_div(len(self.matches), len([b for b in self.pred_blocks if block_type(b) in COMPARABLE_TYPES])),
            "threshold": self.match_threshold,
        }

    def heading_tree_accuracy(self) -> dict[str, Any]:
        gold_headings = [block for block in self.gold_blocks if block_type(block) == "heading"]
        pred_headings = [block for block in self.pred_blocks if block_type(block) == "heading"]
        correct = 0
        matched = 0
        errors: list[dict[str, Any]] = []
        for gold_heading in gold_headings:
            pred_id = self.gold_to_pred.get(block_id(gold_heading))
            if not pred_id:
                errors.append({"gold_id": block_id(gold_heading), "reason": "missing_heading", "text": block_text(gold_heading)})
                continue
            matched += 1
            pred_heading = self.pred_by_id[pred_id]
            level_ok = int_or_none(gold_heading.get("level")) == int_or_none(pred_heading.get("level"))
            parent_ok = self.parent_heading_matches(gold_heading, pred_heading)
            if level_ok and parent_ok:
                correct += 1
            else:
                errors.append(
                    {
                        "gold_id": block_id(gold_heading),
                        "pred_id": pred_id,
                        "reason": "level_or_parent_mismatch",
                        "gold_level": gold_heading.get("level"),
                        "pred_level": pred_heading.get("level"),
                        "gold_text": block_text(gold_heading),
                        "pred_text": block_text(pred_heading),
                    }
                )
        return {
            "score": safe_div(correct, len(gold_headings)),
            "correct": correct,
            "matched": matched,
            "gold_headings": len(gold_headings),
            "pred_headings": len(pred_headings),
            "errors": errors[:50],
        }

    def parent_heading_matches(self, gold_block: dict[str, Any], pred_block: dict[str, Any]) -> bool:
        gold_parent = nearest_heading_ancestor(block_id(gold_block), self.gold_by_id)
        pred_parent = nearest_heading_ancestor(block_id(pred_block), self.pred_by_id)
        if gold_parent is None and pred_parent is None:
            return True
        if gold_parent is None or pred_parent is None:
            return False
        mapped_gold_parent = self.gold_to_pred.get(gold_parent)
        return mapped_gold_parent == pred_parent

    def reading_order_accuracy(self) -> dict[str, Any]:
        matched_pairs = [
            (self.gold_by_id[match.gold_id], self.pred_by_id[match.pred_id])
            for match in self.matches
            if block_type(self.gold_by_id[match.gold_id]) in COMPARABLE_TYPES
        ]
        total_pairs = 0
        concordant = 0
        inversions: list[dict[str, Any]] = []
        for i in range(len(matched_pairs)):
            gold_a, pred_a = matched_pairs[i]
            for j in range(i + 1, len(matched_pairs)):
                gold_b, pred_b = matched_pairs[j]
                gold_delta = numeric_order(gold_a) - numeric_order(gold_b)
                pred_delta = numeric_order(pred_a) - numeric_order(pred_b)
                if gold_delta == 0 or pred_delta == 0:
                    continue
                total_pairs += 1
                if (gold_delta < 0 and pred_delta < 0) or (gold_delta > 0 and pred_delta > 0):
                    concordant += 1
                elif len(inversions) < 50:
                    inversions.append(
                        {
                            "gold_a": short_block(gold_a),
                            "gold_b": short_block(gold_b),
                            "pred_a": short_block(pred_a),
                            "pred_b": short_block(pred_b),
                        }
                    )
        return {
            "score": safe_div(concordant, total_pairs) if total_pairs else None,
            "concordant_pairs": concordant,
            "total_pairs": total_pairs,
            "sample_inversions": inversions,
        }

    def paragraph_merge_f1(self) -> dict[str, Any]:
        gold_paragraphs = [block for block in self.gold_blocks if block_type(block) in TEXT_LIKE_TYPES]
        pred_paragraphs = [block for block in self.pred_blocks if block_type(block) in TEXT_LIKE_TYPES]
        matched_gold = {match.gold_id for match in self.matches if block_type(self.gold_by_id[match.gold_id]) in TEXT_LIKE_TYPES}
        matched_pred = {match.pred_id for match in self.matches if block_type(self.pred_by_id[match.pred_id]) in TEXT_LIKE_TYPES}
        return prf_payload(
            true_positive=len(matched_gold),
            predicted=len(pred_paragraphs),
            gold=len(gold_paragraphs),
            label="paragraph_like_block_match",
        ) | {"matched_predicted": len(matched_pred)}

    def section_attachment_f1(self) -> dict[str, Any]:
        gold_edges: set[tuple[str, str]] = set()
        pred_edges: set[tuple[str, str]] = set()
        for gold_block in self.gold_blocks:
            if block_type(gold_block) not in CONTENT_TYPES:
                continue
            gold_heading = nearest_heading_ancestor(block_id(gold_block), self.gold_by_id)
            pred_block_id = self.gold_to_pred.get(block_id(gold_block))
            if not gold_heading or not pred_block_id:
                continue
            pred_heading = nearest_heading_ancestor(pred_block_id, self.pred_by_id)
            mapped_heading = self.gold_to_pred.get(gold_heading)
            if mapped_heading:
                gold_edges.add((mapped_heading, pred_block_id))
            if pred_heading:
                pred_edges.add((pred_heading, pred_block_id))
        tp = len(gold_edges & pred_edges)
        return prf_payload(tp, predicted=len(pred_edges), gold=len(gold_edges), label="content_to_section_attachment")

    def reference_section_completeness(self) -> dict[str, Any]:
        gold_refs = [block for block in self.gold_blocks if block_type(block) == "reference_item"]
        pred_refs = [block for block in self.pred_blocks if block_type(block) == "reference_item"]
        matched_refs = [
            match for match in self.matches
            if block_type(self.gold_by_id[match.gold_id]) == "reference_item"
            and block_type(self.pred_by_id[match.pred_id]) == "reference_item"
        ]
        heading_present = bool(reference_heading(self.pred_blocks))
        completeness = safe_div(len(matched_refs), len(gold_refs)) if gold_refs else (1.0 if not pred_refs else 0.0)
        return {
            "score": completeness,
            "matched_references": len(matched_refs),
            "gold_references": len(gold_refs),
            "pred_references": len(pred_refs),
            "reference_heading_present": heading_present,
        }

    def float_caption_attachment_accuracy(self) -> dict[str, Any]:
        gold_captions = [block for block in self.gold_blocks if block_type(block) == "caption"]
        correct = 0
        matched = 0
        errors: list[dict[str, Any]] = []
        for gold_caption in gold_captions:
            pred_id = self.gold_to_pred.get(block_id(gold_caption))
            if not pred_id:
                errors.append({"gold_id": block_id(gold_caption), "reason": "missing_caption", "text": block_text(gold_caption)})
                continue
            matched += 1
            pred_caption = self.pred_by_id[pred_id]
            gold_kind = caption_parent_kind(gold_caption, self.gold_by_id)
            pred_kind = caption_parent_kind(pred_caption, self.pred_by_id)
            if gold_kind == pred_kind:
                correct += 1
            else:
                errors.append(
                    {
                        "gold_id": block_id(gold_caption),
                        "pred_id": pred_id,
                        "reason": "caption_parent_kind_mismatch",
                        "gold_kind": gold_kind,
                        "pred_kind": pred_kind,
                    }
                )
        return {
            "score": safe_div(correct, len(gold_captions)),
            "correct": correct,
            "matched": matched,
            "gold_captions": len(gold_captions),
            "errors": errors[:50],
        }

    def generated_structure_validity(self) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        pred_ids = [block_id(block) for block in self.pred_blocks]
        pred_id_set = set(pred_ids)
        if len(pred_ids) != len(pred_id_set):
            violations.append({"type": "duplicate_block_id"})
        if set(self.pred.get("reading_order") or pred_ids) != pred_id_set:
            violations.append({"type": "reading_order_mismatch"})
        for block in self.pred_blocks:
            parent_id = block.get("parent_id")
            if parent_id and parent_id not in pred_id_set:
                violations.append({"type": "missing_parent", "block": block_id(block), "parent": parent_id})
            if block_type(block) == "list_item":
                parent = self.pred_by_id.get(str(parent_id))
                if not parent or block_type(parent) != "list":
                    violations.append({"type": "list_item_without_list_parent", "block": block_id(block)})
            if block_type(block) == "caption":
                parent = self.pred_by_id.get(str(parent_id))
                if parent and block_type(parent) not in {"figure", "table", "algorithm"}:
                    violations.append({"type": "caption_parent_not_float", "block": block_id(block), "parent": str(parent_id)})
        violations.extend(parent_cycle_violations(self.pred_blocks)[:50])
        violations.extend(heading_jump_violations(self.pred_blocks)[:50])
        denominator = max(1, len(self.pred_blocks))
        score = max(0.0, 1.0 - len(violations) / denominator)
        return {
            "score": score,
            "is_valid": not violations,
            "violation_count": len(violations),
            "violations": violations[:100],
        }


def compatible_types(gold_type: str, pred_type: str) -> bool:
    for group in STRICT_TYPE_GROUPS:
        if gold_type in group and pred_type in group:
            return True
    return False


def block_similarity(gold_block: dict[str, Any], pred_block: dict[str, Any]) -> float:
    gold_text = normalized_text(gold_block)
    pred_text = normalized_text(pred_block)
    if not gold_text or not pred_text:
        if block_type(gold_block) == block_type(pred_block):
            if (gold_block.get("marker") or gold_block.get("label")) and (
                gold_block.get("marker") == pred_block.get("marker") or gold_block.get("label") == pred_block.get("label")
            ):
                return 0.75
            return 0.62 if block_type(gold_block) in {"figure", "table"} else 0.0
        return 0.0
    ratio = SequenceMatcher(None, gold_text, pred_text).ratio()
    jaccard = token_jaccard(gold_text, pred_text)
    containment = min(len(gold_text), len(pred_text)) / max(len(gold_text), len(pred_text), 1)
    if gold_text in pred_text or pred_text in gold_text:
        return max(ratio, 0.7 * containment + 0.3)
    return max(ratio, jaccard)


def token_jaccard(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def normalized_text(block: dict[str, Any]) -> str:
    value = str(block.get("normalized_text") or "")
    if value:
        return value
    return normalize_for_eval(str(block.get("text") or ""))


def normalize_for_eval(text: str) -> str:
    value = str(text or "").casefold()
    value = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", value)
    return " ".join(value.split())


def nearest_heading_ancestor(block_id_value: str, by_id: dict[str, dict[str, Any]]) -> str | None:
    seen: set[str] = set()
    current = by_id.get(block_id_value)
    while current:
        parent_id = current.get("parent_id")
        if not parent_id or parent_id in seen:
            return None
        seen.add(str(parent_id))
        parent = by_id.get(str(parent_id))
        if not parent:
            return None
        if block_type(parent) == "heading":
            return block_id(parent)
        current = parent
    return None


def parent_cycle_violations(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {block_id(block): block for block in blocks}
    violations: list[dict[str, Any]] = []
    for block in blocks:
        start = block_id(block)
        seen: set[str] = set()
        current = block
        while current.get("parent_id"):
            parent_id = str(current["parent_id"])
            if parent_id in seen:
                violations.append({"type": "parent_cycle", "block": start, "cycle_at": parent_id})
                break
            seen.add(parent_id)
            current = by_id.get(parent_id)
            if current is None:
                break
    return violations


def heading_jump_violations(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    previous_level: int | None = None
    for block in blocks:
        if block_type(block) != "heading":
            continue
        level = int_or_none(block.get("level"))
        if level is None:
            continue
        if previous_level is not None and level > previous_level + 1:
            violations.append({"type": "heading_level_jump", "block": block_id(block), "previous_level": previous_level, "level": level})
        previous_level = level
    return violations


def caption_parent_kind(block: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str | None:
    marker = block.get("marker")
    if marker in {"figure", "table", "algorithm"}:
        return str(marker)
    parent = by_id.get(str(block.get("parent_id")))
    return block_type(parent) if parent else None


def reference_heading(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for block in blocks:
        if block_type(block) == "heading" and normalize_for_eval(block_text(block)) in {"references", "bibliography", "参考文献"}:
            return block
    return None


def prf_payload(true_positive: int, predicted: int, gold: int, *, label: str) -> dict[str, Any]:
    precision = safe_div(true_positive, predicted)
    recall = safe_div(true_positive, gold)
    return {
        "label": label,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "true_positive": true_positive,
        "predicted": predicted,
        "gold": gold,
    }


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def macro_score(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None and not math.isnan(float(value))]
    if not valid:
        return None
    return sum(float(value) for value in valid) / len(valid)


def int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def numeric_order(block: dict[str, Any]) -> int:
    return int(block.get("order") or 0)


def block_id(block: dict[str, Any]) -> str:
    return str(block.get("block_id"))


def block_type(block: dict[str, Any] | None) -> str:
    return str((block or {}).get("block_type") or "")


def block_text(block: dict[str, Any]) -> str:
    return str(block.get("text") or "")


def short_block(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": block_id(block),
        "type": block_type(block),
        "order": numeric_order(block),
        "text": block_text(block)[:120],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate two comparison-structure JSON files.")
    parser.add_argument("--gold", type=Path, required=True, help="Gold comparison JSON, usually converted from source LaTeX.")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted comparison JSON, converted from generated LaTeX or Markdown.")
    parser.add_argument("--output", type=Path, help="Optional path for metrics JSON.")
    parser.add_argument("--match-threshold", type=float, default=0.58)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    metrics = evaluate_comparison_structures(
        load_comparison_json(args.gold),
        load_comparison_json(args.pred),
        match_threshold=args.match_threshold,
    )
    text = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
