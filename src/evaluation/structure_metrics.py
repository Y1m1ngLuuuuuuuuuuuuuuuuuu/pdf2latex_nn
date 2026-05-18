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
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT_ID = "ROOT"
TEXT_LIKE_TYPES = {"paragraph", "abstract", "list_item", "reference_item", "caption"}
CONTENT_TYPES = {"paragraph", "abstract", "list_item", "display_math", "figure", "table", "caption", "algorithm"}
BODY_SECTION_ATTACHMENT_TYPES = {"paragraph", "abstract", "list_item", "display_math", "algorithm"}
FLOAT_SECTION_ATTACHMENT_TYPES = {"figure", "table", "caption"}
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


@dataclass(frozen=True)
class WindowMatch:
    gold_ids: tuple[str, ...]
    pred_ids: tuple[str, ...]
    score: float
    common_tokens: int
    gold_tokens: int
    pred_tokens: int


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
        self.text_window_matches = self.match_text_windows()
        self.gold_to_text_window: dict[str, WindowMatch] = {}
        self.pred_to_text_window: dict[str, WindowMatch] = {}
        for match in self.text_window_matches:
            for gold_id in match.gold_ids:
                self.gold_to_text_window[gold_id] = match
            for pred_id in match.pred_ids:
                self.pred_to_text_window[pred_id] = match

    def evaluate(self) -> dict[str, Any]:
        paragraph_boundary = self.paragraph_boundary_f1()
        paragraph_text_coverage = self.paragraph_text_coverage_f1()
        metrics = {
            "schema_version": "comparison_metrics_v1",
            "gold_doc_id": self.gold.get("doc_id"),
            "pred_doc_id": self.pred.get("doc_id"),
            "pred_source_format": self.pred.get("source_format"),
            "matching": self.matching_summary(),
            "strict_block_match": self.matching_summary(),
            "window_matching": self.window_matching_summary(),
            "heading_tree_accuracy": self.heading_tree_accuracy(),
            "reading_order_accuracy": self.reading_order_accuracy(),
            "paragraph_boundary_f1": paragraph_boundary,
            "paragraph_text_coverage_f1": paragraph_text_coverage,
            "paragraph_merge_f1": paragraph_boundary
            | {
                "deprecated": True,
                "deprecated_alias_of": "paragraph_boundary_f1",
                "note": "Kept only for backward compatibility; do not report as an independent metric.",
            },
            "section_attachment_f1": self.section_attachment_f1(),
            "section_attachment_body_no_float_f1": self.section_attachment_body_no_float_f1(),
            "section_attachment_oracle_heading_flow_f1": self.section_attachment_oracle_heading_flow_f1(),
            "section_attachment_breakdown": self.section_attachment_breakdown(),
            "reference_section_completeness": self.reference_section_completeness(),
            "float_caption_attachment_accuracy": self.float_caption_attachment_accuracy(),
            "generated_structure_validity": self.generated_structure_validity(),
        }
        metrics["macro_structure_score"] = macro_score(
            [
                metrics["heading_tree_accuracy"].get("score"),
                metrics["reading_order_accuracy"].get("score"),
                metrics["paragraph_text_coverage_f1"].get("f1"),
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

    def match_text_windows(self) -> list[WindowMatch]:
        """Match paragraph-like blocks while tolerating split/merged output.

        The strict matcher is intentionally one-to-one.  This matcher is used
        for text-coverage and section-scope metrics where one source paragraph
        may legitimately become several generated paragraphs, or vice versa.
        It only considers consecutive paragraph-like blocks and greedily keeps
        non-overlapping windows.
        """

        gold_text_blocks = [block for block in self.gold_blocks if block_type(block) in TEXT_LIKE_TYPES]
        pred_text_blocks = [block for block in self.pred_blocks if block_type(block) in TEXT_LIKE_TYPES]
        candidates: list[WindowMatch] = []
        max_window = 8
        threshold = min(0.52, self.match_threshold)

        gold_ids = [block_id(block) for block in gold_text_blocks]
        pred_ids = [block_id(block) for block in pred_text_blocks]
        gold_counters = [token_counter(normalized_text(block)) for block in gold_text_blocks]
        pred_counters = [token_counter(normalized_text(block)) for block in pred_text_blocks]
        gold_token_counts = [sum(counter.values()) for counter in gold_counters]
        pred_token_counts = [sum(counter.values()) for counter in pred_counters]

        def add_candidate(
            gold_window_ids: tuple[str, ...],
            pred_window_ids: tuple[str, ...],
            gold_counter: Counter[str],
            pred_counter: Counter[str],
            gold_tokens: int,
            pred_tokens: int,
        ) -> None:
            score, common = counter_similarity(gold_counter, pred_counter, gold_tokens, pred_tokens)
            if score < threshold or common <= 0:
                return
            candidates.append(
                WindowMatch(
                    gold_ids=gold_window_ids,
                    pred_ids=pred_window_ids,
                    score=score,
                    common_tokens=common,
                    gold_tokens=gold_tokens,
                    pred_tokens=pred_tokens,
                )
            )

        for gold_idx, gold_block in enumerate(gold_text_blocks):
            for pred_start in range(len(pred_text_blocks)):
                max_width = min(max_window, len(pred_text_blocks) - pred_start)
                pred_counter: Counter[str] = Counter()
                pred_tokens = 0
                for width in range(1, max_width + 1):
                    pred_idx = pred_start + width - 1
                    pred_counter.update(pred_counters[pred_idx])
                    pred_tokens += pred_token_counts[pred_idx]
                    add_candidate(
                        (gold_ids[gold_idx],),
                        tuple(pred_ids[pred_start : pred_start + width]),
                        gold_counters[gold_idx],
                        pred_counter,
                        gold_token_counts[gold_idx],
                        pred_tokens,
                    )

        for gold_start in range(len(gold_text_blocks)):
            max_width = min(max_window, len(gold_text_blocks) - gold_start)
            gold_counter: Counter[str] = Counter()
            gold_tokens = 0
            for width in range(2, max_width + 1):
                gold_idx = gold_start + width - 1
                if width == 2:
                    gold_counter = Counter()
                    gold_counter.update(gold_counters[gold_start])
                    gold_tokens = gold_token_counts[gold_start]
                gold_counter.update(gold_counters[gold_idx])
                gold_tokens += gold_token_counts[gold_idx]
                for pred_idx, _pred_block in enumerate(pred_text_blocks):
                    add_candidate(
                        tuple(gold_ids[gold_start : gold_start + width]),
                        (pred_ids[pred_idx],),
                        gold_counter,
                        pred_counters[pred_idx],
                        gold_tokens,
                        pred_token_counts[pred_idx],
                    )

        candidates.sort(key=lambda item: (item.score, item.common_tokens), reverse=True)
        used_gold: set[str] = set()
        used_pred: set[str] = set()
        matches: list[WindowMatch] = []
        for candidate in candidates:
            if any(gold_id in used_gold for gold_id in candidate.gold_ids):
                continue
            if any(pred_id in used_pred for pred_id in candidate.pred_ids):
                continue
            matches.append(candidate)
            used_gold.update(candidate.gold_ids)
            used_pred.update(candidate.pred_ids)
        return sorted(matches, key=lambda item: numeric_order(self.gold_by_id[item.gold_ids[0]]))

    def window_matching_summary(self) -> dict[str, Any]:
        gold_ids = {block_id(block) for block in self.gold_blocks if block_type(block) in TEXT_LIKE_TYPES}
        pred_ids = {block_id(block) for block in self.pred_blocks if block_type(block) in TEXT_LIKE_TYPES}
        matched_gold = {gold_id for match in self.text_window_matches for gold_id in match.gold_ids}
        matched_pred = {pred_id for match in self.text_window_matches for pred_id in match.pred_ids}
        return {
            "matched_windows": len(self.text_window_matches),
            "matched_gold_blocks": len(matched_gold),
            "matched_pred_blocks": len(matched_pred),
            "gold_text_blocks": len(gold_ids),
            "pred_text_blocks": len(pred_ids),
            "coverage_gold_blocks": safe_div(len(matched_gold), len(gold_ids)),
            "coverage_pred_blocks": safe_div(len(matched_pred), len(pred_ids)),
            "common_tokens": sum(match.common_tokens for match in self.text_window_matches),
            "gold_tokens_in_matches": sum(match.gold_tokens for match in self.text_window_matches),
            "pred_tokens_in_matches": sum(match.pred_tokens for match in self.text_window_matches),
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

    def paragraph_boundary_f1(self) -> dict[str, Any]:
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

    def paragraph_merge_f1(self) -> dict[str, Any]:
        return self.paragraph_boundary_f1() | {
            "deprecated": True,
            "deprecated_alias_of": "paragraph_boundary_f1",
            "note": "Kept only for backward compatibility; do not report as an independent metric.",
        }

    def paragraph_text_coverage_f1(self) -> dict[str, Any]:
        common = sum(match.common_tokens for match in self.text_window_matches)
        gold_tokens = sum(token_count(normalized_text(block)) for block in self.gold_blocks if block_type(block) in TEXT_LIKE_TYPES)
        pred_tokens = sum(token_count(normalized_text(block)) for block in self.pred_blocks if block_type(block) in TEXT_LIKE_TYPES)
        return prf_payload(
            true_positive=common,
            predicted=pred_tokens,
            gold=gold_tokens,
            label="paragraph_like_text_coverage_window_match",
        ) | {
            "matched_windows": len(self.text_window_matches),
            "gold_text_blocks": len([block for block in self.gold_blocks if block_type(block) in TEXT_LIKE_TYPES]),
            "pred_text_blocks": len([block for block in self.pred_blocks if block_type(block) in TEXT_LIKE_TYPES]),
        }

    def section_attachment_f1(self) -> dict[str, Any]:
        return self.section_attachment_f1_for(
            lambda block, heading: block_type(block) in CONTENT_TYPES,
            label="content_to_section_attachment",
        )

    def section_attachment_body_no_float_f1(self) -> dict[str, Any]:
        return self.section_attachment_f1_for(
            lambda block, heading: block_type(block) in BODY_SECTION_ATTACHMENT_TYPES
            and section_scope_kind(heading, self.gold_by_id) == "body",
            label="body_text_to_section_attachment_no_float",
        )

    def section_attachment_breakdown(self) -> dict[str, Any]:
        groups = {
            "body": lambda block, heading: block_type(block) in BODY_SECTION_ATTACHMENT_TYPES
            and section_scope_kind(heading, self.gold_by_id) == "body",
            "float": lambda block, heading: block_type(block) in FLOAT_SECTION_ATTACHMENT_TYPES,
            "references": lambda block, heading: block_type(block) == "reference_item"
            or section_scope_kind(heading, self.gold_by_id) == "references",
            "appendix": lambda block, heading: section_scope_kind(heading, self.gold_by_id) == "appendix",
        }
        return {
            name: self.section_attachment_f1_for(predicate, label=f"{name}_section_attachment")
            for name, predicate in groups.items()
        }

    def section_attachment_oracle_heading_flow_f1(self) -> dict[str, Any]:
        """Upper-bound diagnostic for section scope with oracle heading identity.

        This ignores predicted parent edges.  It walks the predicted reading
        order over matched blocks; whenever a matched predicted heading appears,
        the active scope is set to that heading's matched gold heading.  Matched
        gold body blocks are then scored against their true gold heading.  The
        metric answers: if heading identities were known and only reading flow
        drove attachment, how much section attachment could be recovered?
        """

        pred_items: list[tuple[int, str, str]] = []
        for match in self.matches:
            gold_block = self.gold_by_id[match.gold_id]
            pred_block = self.pred_by_id[match.pred_id]
            pred_items.append((numeric_order(pred_block), match.gold_id, match.pred_id))
        pred_items.sort(key=lambda item: item[0])

        predicted: set[tuple[str, str]] = set()
        gold: set[tuple[str, str]] = set()
        active_gold_heading: str | None = None
        for _, gold_id, pred_id in pred_items:
            gold_block = self.gold_by_id[gold_id]
            if block_type(gold_block) == "heading":
                active_gold_heading = gold_id
                continue
            if block_type(gold_block) not in BODY_SECTION_ATTACHMENT_TYPES:
                continue
            gold_heading = nearest_heading_ancestor(gold_id, self.gold_by_id)
            if not gold_heading or section_scope_kind(gold_heading, self.gold_by_id) != "body":
                continue
            gold.add((gold_heading, gold_id))
            if active_gold_heading:
                predicted.add((active_gold_heading, gold_id))
        tp = len(gold & predicted)
        return prf_payload(
            tp,
            predicted=len(predicted),
            gold=len(gold),
            label="oracle_heading_identity_plus_predicted_reading_flow",
        )

    def section_attachment_f1_for(self, predicate: Any, *, label: str) -> dict[str, Any]:
        gold_edges: set[tuple[str, str]] = set()
        pred_edges: set[tuple[str, str]] = set()
        true_positive = 0.0
        handled_gold: set[str] = set()
        handled_pred: set[str] = set()

        for match in self.text_window_matches:
            gold_blocks = [self.gold_by_id[gold_id] for gold_id in match.gold_ids if gold_id in self.gold_by_id]
            pred_blocks = [self.pred_by_id[pred_id] for pred_id in match.pred_ids if pred_id in self.pred_by_id]
            scoped_gold_blocks = [
                block
                for block in gold_blocks
                if predicate(block, nearest_heading_ancestor(block_id(block), self.gold_by_id))
            ]
            if not scoped_gold_blocks or not pred_blocks:
                continue
            gold_heading = majority_heading(scoped_gold_blocks, self.gold_by_id)
            pred_heading = majority_heading(pred_blocks, self.pred_by_id)
            if not gold_heading:
                continue
            handled_gold.update(block_id(block) for block in scoped_gold_blocks)
            handled_pred.update(block_id(block) for block in pred_blocks)
            edge_id = "window:" + "|".join(match.pred_ids)
            mapped_heading = self.gold_to_pred.get(gold_heading)
            if mapped_heading:
                gold_edges.add((mapped_heading, edge_id))
            if pred_heading:
                pred_edges.add((pred_heading, edge_id))
            if mapped_heading and pred_heading and mapped_heading == pred_heading:
                true_positive += 1.0

        for gold_block in self.gold_blocks:
            if block_id(gold_block) in handled_gold:
                continue
            gold_heading = nearest_heading_ancestor(block_id(gold_block), self.gold_by_id)
            if not predicate(gold_block, gold_heading):
                continue
            pred_block_id = self.gold_to_pred.get(block_id(gold_block))
            if not gold_heading or not pred_block_id:
                continue
            if pred_block_id in handled_pred:
                continue
            pred_heading = nearest_heading_ancestor(pred_block_id, self.pred_by_id)
            mapped_heading = self.gold_to_pred.get(gold_heading)
            if mapped_heading:
                gold_edges.add((mapped_heading, pred_block_id))
            if pred_heading:
                pred_edges.add((pred_heading, pred_block_id))
            if mapped_heading and pred_heading and mapped_heading == pred_heading:
                true_positive += 1.0
        return prf_payload(true_positive, predicted=len(pred_edges), gold=len(gold_edges), label=label)

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

    gold_counter = token_counter(gold_text)
    pred_counter = token_counter(pred_text)
    gold_tokens = sum(gold_counter.values())
    pred_tokens = sum(pred_counter.values())
    token_score, common = counter_similarity(gold_counter, pred_counter, gold_tokens, pred_tokens)
    jaccard = token_jaccard_from_counters(gold_counter, pred_counter)
    containment = min(len(gold_text), len(pred_text)) / max(len(gold_text), len(pred_text), 1)
    if gold_text in pred_text or pred_text in gold_text:
        return max(token_score, 0.7 * containment + 0.3)
    if common == 0:
        return 0.0

    # Character-level SequenceMatcher is useful for short headings/captions but
    # becomes painfully slow on long generated paragraphs.  For long body text,
    # token overlap is both faster and a better proxy for structural matching.
    if max(len(gold_text), len(pred_text)) > 1000:
        return max(token_score, jaccard)
    if token_score < 0.08 and jaccard < 0.05:
        return max(token_score, jaccard)
    ratio = SequenceMatcher(None, gold_text, pred_text).ratio()
    return max(ratio, token_score, jaccard)


def window_similarity(gold_blocks: list[dict[str, Any]], pred_blocks: list[dict[str, Any]]) -> tuple[float, int, int, int]:
    gold_text = " ".join(normalized_text(block) for block in gold_blocks if normalized_text(block))
    pred_text = " ".join(normalized_text(block) for block in pred_blocks if normalized_text(block))
    gold_counter = token_counter(gold_text)
    pred_counter = token_counter(pred_text)
    gold_tokens = sum(gold_counter.values())
    pred_tokens = sum(pred_counter.values())
    common = sum((gold_counter & pred_counter).values())
    if not gold_tokens or not pred_tokens or not common:
        return 0.0, common, gold_tokens, pred_tokens
    precision = common / pred_tokens
    recall = common / gold_tokens
    token_f1 = f1_score(precision, recall)
    containment = min(gold_tokens, pred_tokens) / max(gold_tokens, pred_tokens)
    score = token_f1 * 0.9 + containment * 0.1
    return score, common, gold_tokens, pred_tokens


def counter_similarity(
    gold_counter: Counter[str],
    pred_counter: Counter[str],
    gold_tokens: int,
    pred_tokens: int,
) -> tuple[float, int]:
    common = sum((gold_counter & pred_counter).values())
    if not gold_tokens or not pred_tokens or not common:
        return 0.0, common
    precision = common / pred_tokens
    recall = common / gold_tokens
    token_f1 = f1_score(precision, recall)
    containment = min(gold_tokens, pred_tokens) / max(gold_tokens, pred_tokens)
    score = token_f1 * 0.9 + containment * 0.1
    return score, common


def token_jaccard(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def token_jaccard_from_counters(a: Counter[str], b: Counter[str]) -> float:
    a_tokens = set(a)
    b_tokens = set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def token_counter(text: str) -> Counter[str]:
    return Counter(str(text or "").split())


def token_count(text: str) -> int:
    return sum(token_counter(text).values())


def normalized_text(block: dict[str, Any]) -> str:
    value = str(block.get("normalized_text") or "")
    if value:
        return value
    return normalize_for_eval(str(block.get("text") or ""))


def majority_heading(blocks: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> str | None:
    votes: Counter[str] = Counter()
    for block in blocks:
        heading = nearest_heading_ancestor(block_id(block), by_id)
        if not heading:
            continue
        votes[heading] += max(1, token_count(normalized_text(block)))
    if not votes:
        return None
    return votes.most_common(1)[0][0]


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


def section_scope_kind(heading_id: str | None, by_id: dict[str, dict[str, Any]]) -> str:
    if not heading_id:
        return "body"
    heading = by_id.get(str(heading_id))
    text = normalize_for_eval(block_text(heading or {}))
    if text in {"references", "bibliography", "参考文献"} or text.startswith("references ") or text.startswith("bibliography "):
        return "references"
    if text.startswith("appendix") or text.startswith("appendices") or text.startswith("附录"):
        return "appendix"
    return "body"


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
