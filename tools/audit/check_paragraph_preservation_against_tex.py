#!/usr/bin/env python3
"""Audit paragraph preservation between source TeX and generated TeX.

This answers a generator-facing question that edge-level MERGE metrics do not:
if the source TeX has one paragraph, did the generated TeX keep it as one
paragraph, or was it split into multiple paragraphs? Conversely, did the
generated TeX accidentally merge multiple source paragraphs into one paragraph?
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.check_reading_order_against_tex import (
    begin_document_offset,
    clean_anchor_phrase,
    drop_environment_blocks,
    is_commented_position,
    strip_comments,
)

try:  # rapidfuzz is available in the project env and makes batch audits tractable.
    from rapidfuzz.distance import LCSseq
except Exception:  # pragma: no cover - optional acceleration.
    LCSseq = None


DROP_PARAGRAPH_ENVIRONMENTS = (
    "figure",
    "figure*",
    "table",
    "table*",
    "algorithm",
    "algorithm*",
    "tabular",
    "tabular*",
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
)


@dataclass
class ParagraphBlock:
    block_id: str
    source: str
    index: int
    raw_start: int
    line: int | None
    raw_text: str
    text: str
    normalized_text: str
    tokens: list[str]
    semantic_channel: str = "ordinary_prose"


@dataclass
class PairScore:
    source_id: str
    generated_id: str
    common_tokens: int
    source_token_count: int
    generated_token_count: int
    source_recall: float
    generated_precision: float
    f1: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tex", required=True, type=Path)
    parser.add_argument("--generated-tex", required=True, type=Path)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--min-common-tokens", type=int, default=5)
    parser.add_argument("--candidate-source-recall", type=float, default=0.12)
    parser.add_argument("--candidate-generated-precision", type=float, default=0.30)
    parser.add_argument("--covered-source-recall", type=float, default=0.55)
    parser.add_argument("--covered-combined-recall", type=float, default=0.65)
    parser.add_argument("--split-combined-recall", type=float, default=0.60)
    parser.add_argument("--split-best-source-recall-max", type=float, default=0.85)
    parser.add_argument("--overmerge-generated-coverage", type=float, default=0.60)
    parser.add_argument("--include-list-items", action="store_true")
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_paragraph_preservation(args)
    return 0


def run_paragraph_preservation(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_text = args.source_tex.read_text(encoding="utf-8", errors="ignore")
    generated_text = args.generated_tex.read_text(encoding="utf-8", errors="ignore")
    doc_id = args.doc_id or args.source_tex.parent.name or args.generated_tex.stem

    source_blocks = extract_paragraph_blocks(
        source_text,
        source="source",
        min_tokens=args.min_tokens,
        include_list_items=args.include_list_items,
    )
    generated_blocks = extract_paragraph_blocks(
        generated_text,
        source="generated",
        min_tokens=args.min_tokens,
        include_list_items=args.include_list_items,
    )
    scores = pair_scores(
        source_blocks,
        generated_blocks,
        min_common_tokens=args.min_common_tokens,
        candidate_source_recall=args.candidate_source_recall,
        candidate_generated_precision=args.candidate_generated_precision,
    )
    candidate_scores = [
        score
        for score in scores
        if score.common_tokens >= args.min_common_tokens
        and (
            score.source_recall >= args.candidate_source_recall
            or score.generated_precision >= args.candidate_generated_precision
        )
    ]
    analysis = analyze_preservation(source_blocks, generated_blocks, candidate_scores, args)

    body_source_blocks, excluded_source_blocks = split_body_source_blocks(source_blocks)
    body_scores = pair_scores(
        body_source_blocks,
        generated_blocks,
        min_common_tokens=args.min_common_tokens,
        candidate_source_recall=args.candidate_source_recall,
        candidate_generated_precision=args.candidate_generated_precision,
    )
    body_candidate_scores = [
        score
        for score in body_scores
        if score.common_tokens >= args.min_common_tokens
        and (
            score.source_recall >= args.candidate_source_recall
            or score.generated_precision >= args.candidate_generated_precision
        )
    ]
    body_analysis = analyze_preservation(body_source_blocks, generated_blocks, body_candidate_scores, args)

    visible_source_blocks = reindex_blocks(
        [block for block in body_source_blocks if is_visible_prose_channel(block.semantic_channel)]
    )
    visible_generated_blocks = reindex_blocks(
        [block for block in generated_blocks if is_visible_prose_channel(block.semantic_channel)]
    )
    visible_scores = pair_scores(
        visible_source_blocks,
        visible_generated_blocks,
        min_common_tokens=args.min_common_tokens,
        candidate_source_recall=args.candidate_source_recall,
        candidate_generated_precision=args.candidate_generated_precision,
        type_aware=True,
    )
    visible_candidate_scores = [
        score
        for score in visible_scores
        if score.common_tokens >= args.min_common_tokens
        and (
            score.source_recall >= args.candidate_source_recall
            or score.generated_precision >= args.candidate_generated_precision
        )
    ]
    visible_analysis = analyze_preservation(visible_source_blocks, visible_generated_blocks, visible_candidate_scores, args)

    visible_plus_list_source_blocks = reindex_blocks(
        [block for block in body_source_blocks if is_visible_prose_or_list_channel(block.semantic_channel)]
    )
    visible_plus_list_generated_blocks = reindex_blocks(
        [block for block in generated_blocks if is_visible_prose_or_list_channel(block.semantic_channel)]
    )
    visible_plus_list_scores = pair_scores(
        visible_plus_list_source_blocks,
        visible_plus_list_generated_blocks,
        min_common_tokens=args.min_common_tokens,
        candidate_source_recall=args.candidate_source_recall,
        candidate_generated_precision=args.candidate_generated_precision,
        type_aware=True,
    )
    visible_plus_list_candidate_scores = [
        score
        for score in visible_plus_list_scores
        if score.common_tokens >= args.min_common_tokens
        and (
            score.source_recall >= args.candidate_source_recall
            or score.generated_precision >= args.candidate_generated_precision
        )
    ]
    visible_plus_list_analysis = analyze_preservation(
        visible_plus_list_source_blocks,
        visible_plus_list_generated_blocks,
        visible_plus_list_candidate_scores,
        args,
    )
    raw_summary = analysis["summary"]
    body_summary = body_analysis["summary"]
    visible_summary = visible_analysis["summary"]
    visible_plus_list_summary = visible_plus_list_analysis["summary"]
    exclusion_reasons = Counter(reason for _, reason in excluded_source_blocks)
    body_source_count = body_summary["source_paragraph_count"]
    body_covered_count = body_summary["covered_source_paragraph_count"]
    generated_count = body_summary["generated_paragraph_count"]
    summary = {
        **raw_summary,
        "source_coverage_rate_raw": raw_summary["source_coverage_rate"],
        "raw_source_paragraph_count": raw_summary["source_paragraph_count"],
        "raw_covered_source_paragraph_count": raw_summary["covered_source_paragraph_count"],
        "raw_uncovered_source_paragraph_count": raw_summary["uncovered_source_paragraph_count"],
        "ordered_source_coverage_rate_raw": raw_summary.get("ordered_source_coverage_rate"),
        "source_order_inversion_rate_raw": raw_summary.get("source_order_inversion_rate"),
        "source_order_kendall_tau_raw": raw_summary.get("source_order_kendall_tau"),
        "body_source_paragraph_count": body_summary["source_paragraph_count"],
        "body_covered_source_paragraph_count": body_summary["covered_source_paragraph_count"],
        "body_uncovered_source_paragraph_count": body_summary["uncovered_source_paragraph_count"],
        "body_source_coverage_rate": body_summary["source_coverage_rate"] if body_source_count else None,
        "body_ordered_source_coverage_rate": body_summary.get("ordered_source_coverage_rate") if body_source_count else None,
        "body_source_order_inversion_rate": body_summary.get("source_order_inversion_rate") if body_source_count else None,
        "body_source_order_kendall_tau": body_summary.get("source_order_kendall_tau") if body_source_count else None,
        "body_source_order_adjacent_inversion_rate": body_summary.get("source_order_adjacent_inversion_rate") if body_source_count else None,
        "visible_prose_source_paragraph_count": visible_summary["source_paragraph_count"],
        "visible_prose_generated_paragraph_count": visible_summary["generated_paragraph_count"],
        "visible_prose_source_coverage_rate": visible_summary["source_coverage_rate"] if visible_summary["source_paragraph_count"] else None,
        "visible_prose_ordered_coverage_rate": visible_summary.get("ordered_source_coverage_rate") if visible_summary["source_paragraph_count"] else None,
        "visible_prose_order_inversion_rate": visible_summary.get("source_order_inversion_rate") if visible_summary["source_paragraph_count"] else None,
        "adjacent_prose_inversion_rate": visible_summary.get("source_order_adjacent_inversion_rate") if visible_summary["source_paragraph_count"] else None,
        "displaced_prose_paragraph_rate_010": visible_summary.get("source_order_displaced_rate_010") if visible_summary["source_paragraph_count"] else None,
        "displaced_prose_paragraph_rate_015": visible_summary.get("source_order_displaced_rate_015") if visible_summary["source_paragraph_count"] else None,
        "visible_prose_lis_disorder_rate": visible_summary.get("source_order_lis_disorder_rate") if visible_summary["source_paragraph_count"] else None,
        "visible_prose_body_plus_list_source_coverage_rate": visible_plus_list_summary["source_coverage_rate"] if visible_plus_list_summary["source_paragraph_count"] else None,
        "visible_prose_body_plus_list_ordered_coverage_rate": visible_plus_list_summary.get("ordered_source_coverage_rate") if visible_plus_list_summary["source_paragraph_count"] else None,
        "visible_prose_body_plus_list_order_inversion_rate": visible_plus_list_summary.get("source_order_inversion_rate") if visible_plus_list_summary["source_paragraph_count"] else None,
        "visible_prose_body_plus_list_lis_disorder_rate": visible_plus_list_summary.get("source_order_lis_disorder_rate") if visible_plus_list_summary["source_paragraph_count"] else None,
        "body_missing_merge_source_paragraph_count": body_summary["missing_merge_source_paragraph_count"],
        "body_missing_merge_rate_among_covered": body_summary["missing_merge_rate_among_covered"] if body_covered_count else None,
        "body_wrong_merge_generated_paragraph_count": body_summary["wrong_merge_generated_paragraph_count"],
        "body_wrong_merge_rate_among_generated": body_summary["wrong_merge_rate_among_generated"] if generated_count else None,
        "body_paragraph_count_delta": body_summary["paragraph_count_delta"],
        "excluded_source_paragraph_count": len(excluded_source_blocks),
        "source_body_exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "source_semantic_channel_counts": dict(sorted(Counter(block.semantic_channel for block in source_blocks).items())),
        "generated_semantic_channel_counts": dict(sorted(Counter(block.semantic_channel for block in generated_blocks).items())),
    }
    payload = {
        "schema_version": "paragraph_preservation_against_tex_v3",
        "doc_id": doc_id,
        "inputs": {
            "source_tex": str(args.source_tex),
            "generated_tex": str(args.generated_tex),
        },
        "config": {
            "min_tokens": args.min_tokens,
            "min_common_tokens": args.min_common_tokens,
            "candidate_source_recall": args.candidate_source_recall,
            "candidate_generated_precision": args.candidate_generated_precision,
            "covered_source_recall": args.covered_source_recall,
            "covered_combined_recall": args.covered_combined_recall,
            "split_combined_recall": args.split_combined_recall,
            "split_best_source_recall_max": args.split_best_source_recall_max,
            "overmerge_generated_coverage": args.overmerge_generated_coverage,
            "include_list_items": args.include_list_items,
        },
        "summary": summary,
        "raw_summary": raw_summary,
        "body_summary": body_summary,
        "missing_merge_examples": analysis["missing_merge_examples"],
        "wrong_merge_examples": analysis["wrong_merge_examples"],
        "uncovered_source_examples": analysis["uncovered_source_examples"],
        "body_missing_merge_examples": body_analysis["missing_merge_examples"],
        "body_wrong_merge_examples": body_analysis["wrong_merge_examples"],
        "body_uncovered_source_examples": body_analysis["uncovered_source_examples"],
        "source_order_inversion_examples": analysis.get("source_order_inversion_examples", []),
        "body_source_order_inversion_examples": body_analysis.get("source_order_inversion_examples", []),
        "visible_prose_order_inversion_examples": visible_analysis.get("source_order_inversion_examples", []),
        "visible_prose_displaced_examples": visible_analysis.get("source_order_displaced_examples", []),
        "source_paragraphs": [block_summary(block) for block in source_blocks],
        "body_source_paragraphs": [block_summary(block) for block in body_source_blocks],
        "excluded_source_paragraphs": [
            {**block_summary(block), "exclude_reason": reason}
            for block, reason in excluded_source_blocks[: args.max_examples]
        ],
        "generated_paragraphs": [block_summary(block) for block in generated_blocks],
    }

    json_path = args.output_dir / "paragraph_preservation_against_tex.json"
    md_path = args.output_dir / "PARAGRAPH_PRESERVATION_AGAINST_TEX.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return payload


def extract_paragraph_blocks(
    tex: str,
    *,
    source: str,
    min_tokens: int,
    include_list_items: bool,
) -> list[ParagraphBlock]:
    body_start = begin_document_offset(tex)
    body = tex[body_start:]
    body = strip_comments(body)
    body = drop_environment_blocks(body, DROP_PARAGRAPH_ENVIRONMENTS)
    body = remove_heading_commands(body)
    body = remove_captions_and_references(body)
    body = normalize_list_boundaries(body, include_list_items=include_list_items)
    body = re.sub(r"\\begin\{multicols\}\{[^{}]+\}|\\end\{multicols\}", "\n\n", body)
    body = re.sub(r"\\begin\{[^{}]+\}|\\end\{[^{}]+\}", "\n\n", body)
    body = re.sub(r"\$\$.*?\$\$", "\n\n", body, flags=re.DOTALL)
    body = re.sub(r"\\\[.*?\\\]", "\n\n", body, flags=re.DOTALL)

    blocks: list[ParagraphBlock] = []
    for idx, match in enumerate(re.finditer(r"\S.*?(?=\n\s*\n|$)", body, flags=re.DOTALL)):
        raw = match.group(0)
        if not raw.strip():
            continue
        if is_probably_layout_only(raw):
            continue
        text = clean_paragraph_text(raw)
        tokens = tokenize(text)
        if len(tokens) < min_tokens:
            continue
        raw_start = body_start + match.start()
        blocks.append(
            ParagraphBlock(
                block_id=f"{source}_p{len(blocks):04d}",
                source=source,
                index=len(blocks),
                raw_start=raw_start,
                line=tex.count("\n", 0, raw_start) + 1,
                raw_text=raw,
                text=text,
                normalized_text=" ".join(tokens),
                tokens=tokens,
                semantic_channel=classify_paragraph_channel(raw, text),
            )
        )
    return blocks


def reindex_blocks(blocks: list[ParagraphBlock]) -> list[ParagraphBlock]:
    """Return shallow copies with contiguous metric indexes.

    Raw/body legacy metrics keep extractor indexes.  Visible-prose metrics need
    indexes over the filtered visible-prose sequence so adjacent inversion and
    normalized displacement answer the question users actually inspect: how many
    ordinary prose paragraphs are locally or globally out of order?
    """
    return [
        ParagraphBlock(
            block_id=block.block_id,
            source=block.source,
            index=idx,
            raw_start=block.raw_start,
            line=block.line,
            raw_text=block.raw_text,
            text=block.text,
            normalized_text=block.normalized_text,
            tokens=block.tokens,
            semantic_channel=block.semantic_channel,
        )
        for idx, block in enumerate(blocks)
    ]


def classify_paragraph_channel(raw: str, text: str) -> str:
    raw_l = raw.lower()
    text_l = text.lower().strip()
    compact = re.sub(r"\s+", " ", text_l)

    if not compact:
        return "artifact"
    if re.search(r"\b(?:figure|fig\.|table|tab\.|algorithm|alg\.)\s*(?:s?\d+|[ivxlcdm]+|[a-z])[:.]", compact, re.I):
        return "caption"
    if re.match(r"^\s*(?:\[\d+\]|\d+\.|[a-z][a-z ,.-]{1,80}\(\d{4}[a-z]?\))", compact) and (
        re.search(r"\b(?:proceedings|journal|conference|arxiv|doi|vol\.|pp\.|pages?|transactions|springer|ieee|acm)\b", compact)
        or len(compact) > 120
    ):
        return "reference_item"
    if re.search(r"\\begin\{(?:equation|align|gather|multline|split)\*?\}", raw_l):
        return "display_math"
    if formula_like_text(compact):
        return "math_context"
    if re.search(r"https?://|www\.|github\.com|doi\.org|arxiv\.org", compact):
        return "url_or_metadata"
    if re.search(r"\b(?:corresponding author|equal contribution|funded by|project page|code is available|data and models)\b", compact):
        return "front_note"
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        return "front_matter"
    if re.search(r"\b(?:university|institute|department|laboratory|lab|school|college|faculty|academy)\b", compact):
        if len(tokenize(text)) < 35:
            return "front_matter"
    if re.match(r"^\s*abstract\b[:.\-]?", compact):
        return "abstract"
    if re.match(r"^\s*(?:keywords?|index terms)\b[:.\-]?", compact):
        return "metadata"
    if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", text):
        return "body_list"
    if len(tokenize(text)) < 4:
        return "artifact"
    return "ordinary_prose"


def formula_like_text(text: str) -> bool:
    if re.search(r"\\(?:frac|sum|int|prod|alpha|beta|gamma|lambda|theta|sigma|mathbb|mathbf|mathrm)\b", text):
        return True
    symbol_count = sum(text.count(ch) for ch in "=<>^_{}[]()+*/|")
    token_count = max(len(tokenize(text)), 1)
    if symbol_count >= 8 and symbol_count > token_count:
        return True
    if re.match(r"^\s*(?:where|with|s\.t\.|subject to)\b", text) and symbol_count >= 2:
        return True
    return False


def is_visible_prose_channel(channel: str) -> bool:
    return channel == "ordinary_prose"


def is_visible_prose_or_list_channel(channel: str) -> bool:
    return channel in {"ordinary_prose", "body_list"}


def split_body_source_blocks(
    source_blocks: list[ParagraphBlock],
) -> tuple[list[ParagraphBlock], list[tuple[ParagraphBlock, str]]]:
    body_blocks: list[ParagraphBlock] = []
    excluded: list[tuple[ParagraphBlock, str]] = []
    for block in source_blocks:
        reason = source_body_exclusion_reason(block)
        if reason:
            excluded.append((block, reason))
        else:
            body_blocks.append(block)
    return body_blocks, excluded


def source_body_exclusion_reason(block: ParagraphBlock) -> str | None:
    """Return why a source paragraph should not count as visible body text.

    The raw metric intentionally keeps legacy behavior.  This filter is only
    used for body-facing source coverage so TeX implementation details such as
    TikZ node styles do not masquerade as missing prose.
    """
    raw = block.raw_text.strip()
    text = block.text.strip()
    raw_l = raw.lower()
    text_l = text.lower()

    if re.search(r"\\(?:newcommand|renewcommand|providecommand|def|let|setlength|addtolength|usepackage|usetikzlibrary|tikzset|pgfplotsset)\b", raw):
        return "tex_definition_or_setup"
    if re.search(r"\\(?:node|path|draw|coordinate|matrix|addplot|addlegendentry|foreach)\b", raw):
        return "diagram_source"
    if re.search(r"(?:/\\.style|node distance|minimum (?:height|width)|rounded corners|shorten [<>]?=|>=latex|align=center|color=gray|font=\\)", raw_l):
        return "diagram_style_options"
    if re.search(r"\\begin\{(?:tikzpicture|pgfpicture|axis|scope|forest|picture|circuitikz|lstlisting|minted|verbatim)\}", raw_l):
        return "non_body_environment_source"
    if text_l.startswith("[ node ") or text_l.startswith("node distance"):
        return "diagram_style_options"
    if re.search(r"\([a-z0-9_-]+\)\s+[a-z0-9]", text_l) and text.count(";") >= 2 and raw.count("\\\\") >= 2:
        return "diagram_node_listing"
    if text.count(";") >= 4 and re.search(r"\b(?:block|dense|flatten|transformer|arrow|style|node)\b", text_l):
        return "diagram_node_listing"

    # Macro-dominated paragraphs are usually source implementation rather than
    # visible prose.  Keep normal math-heavy prose unless command density is
    # clearly extreme.
    command_count = len(re.findall(r"\\[a-zA-Z]+", raw))
    word_count = len(re.findall(r"[A-Za-z]{3,}", text))
    symbol_count = sum(raw.count(ch) for ch in "\\{}[];=<>")
    if command_count >= 8 and command_count > max(4, word_count // 3):
        return "macro_dominated_source"
    if symbol_count > max(80, len(raw) * 0.28) and word_count < 30:
        return "symbol_dominated_source"
    return None


def remove_heading_commands(text: str) -> str:
    return re.sub(
        r"\\(?:section|subsection|subsubsection|paragraph|subparagraph)\*?(?:\[[^\]]*\])?\{[^{}]*\}",
        "\n\n",
        text,
    )


def remove_captions_and_references(text: str) -> str:
    text = re.sub(r"\\caption(?:\[[^\]]*\])?\{[^{}]*\}", "\n\n", text)
    text = re.sub(r"\\(?:cite|ref|label|url|href)\*?(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\bibliographystyle\{[^{}]*\}|\\bibliography\{[^{}]*\}", "\n\n", text)
    return text


def normalize_list_boundaries(text: str, *, include_list_items: bool) -> str:
    if include_list_items:
        text = re.sub(r"\\item(?:\[[^\]]*\])?", "\n\n", text)
        return text
    text = re.sub(r"\\begin\{(?:itemize|enumerate|description)\}.*?\\end\{(?:itemize|enumerate|description)\}", "\n\n", text, flags=re.DOTALL)
    text = re.sub(r"\\item(?:\[[^\]]*\])?.*?(?=\\item|\\end\{|$)", "\n\n", text, flags=re.DOTALL)
    return text


def is_probably_layout_only(raw: str) -> bool:
    stripped = raw.strip()
    if is_commented_position(raw, 0):
        return True
    if re.fullmatch(r"(?:\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?\s*)+", stripped):
        return True
    if re.search(r"\\(?:toprule|midrule|bottomrule|cmidrule|hline)\b", stripped):
        return True
    return False


def clean_paragraph_text(raw: str) -> str:
    text = clean_anchor_phrase(raw)
    text = re.sub(r"\\(?:textbf|textit|emph|underline)\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\ensuremath\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"[$^_{}&]", " ", text)
    text = text.replace("~", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    # Remove hyphenation line-break scars but keep ordinary word separation.
    text = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", text)
    return re.findall(r"[a-z0-9]+", text)


def pair_scores(
    source_blocks: list[ParagraphBlock],
    generated_blocks: list[ParagraphBlock],
    *,
    min_common_tokens: int = 1,
    candidate_source_recall: float = 0.0,
    candidate_generated_precision: float = 0.0,
    type_aware: bool = False,
) -> list[PairScore]:
    rows: list[PairScore] = []
    generated_token_counts = [Counter(block.tokens) for block in generated_blocks]
    for source_block in source_blocks:
        source_token_counts = Counter(source_block.tokens)
        for generated_block, generated_token_count in zip(generated_blocks, generated_token_counts):
            if type_aware and not channels_can_match(source_block.semantic_channel, generated_block.semantic_channel):
                continue
            upper_common = token_overlap_upper_bound(source_token_counts, generated_token_count)
            if upper_common <= 0:
                continue
            if (
                upper_common < min_common_tokens
                and safe_div(upper_common, len(source_block.tokens)) < candidate_source_recall
                and safe_div(upper_common, len(generated_block.tokens)) < candidate_generated_precision
            ):
                continue
            common = ordered_common_token_count(source_block.tokens, generated_block.tokens)
            if common <= 0:
                continue
            source_recall = safe_div(common, len(source_block.tokens))
            generated_precision = safe_div(common, len(generated_block.tokens))
            f1 = safe_div(2 * source_recall * generated_precision, source_recall + generated_precision)
            rows.append(
                PairScore(
                    source_id=source_block.block_id,
                    generated_id=generated_block.block_id,
                    common_tokens=common,
                    source_token_count=len(source_block.tokens),
                    generated_token_count=len(generated_block.tokens),
                    source_recall=source_recall,
                    generated_precision=generated_precision,
                    f1=f1,
                )
            )
    return rows


def channels_can_match(source_channel: str, generated_channel: str) -> bool:
    if source_channel in {"ordinary_prose", "body_list"}:
        return generated_channel in {"ordinary_prose", "body_list"}
    if source_channel == "abstract":
        return generated_channel == "abstract"
    if source_channel == "caption":
        return generated_channel == "caption"
    if source_channel == "reference_item":
        return generated_channel == "reference_item"
    if source_channel in {"display_math", "math_context"}:
        return generated_channel in {"display_math", "math_context"}
    return source_channel == generated_channel


def token_overlap_upper_bound(left: Counter[str], right: Counter[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    return sum(min(count, right.get(token, 0)) for token, count in left.items())


def ordered_common_token_count(left: list[str], right: list[str]) -> int:
    if LCSseq is not None:
        return int(LCSseq.similarity(left, right))
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks())


def analyze_preservation(
    source_blocks: list[ParagraphBlock],
    generated_blocks: list[ParagraphBlock],
    candidate_scores: list[PairScore],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_by_id = {block.block_id: block for block in source_blocks}
    generated_by_id = {block.block_id: block for block in generated_blocks}
    by_source: dict[str, list[PairScore]] = {block.block_id: [] for block in source_blocks}
    by_generated: dict[str, list[PairScore]] = {block.block_id: [] for block in generated_blocks}
    for score in candidate_scores:
        by_source.setdefault(score.source_id, []).append(score)
        by_generated.setdefault(score.generated_id, []).append(score)
    for scores in by_source.values():
        scores.sort(key=lambda score: (-score.source_recall, generated_by_id[score.generated_id].index))
    for scores in by_generated.values():
        scores.sort(key=lambda score: (-score.generated_precision, source_by_id[score.source_id].index))

    covered_sources = 0
    one_to_one_sources = 0
    missing_merge_examples: list[dict[str, Any]] = []
    uncovered_source_examples: list[dict[str, Any]] = []
    order_anchors: list[dict[str, Any]] = []
    for source_block in source_blocks:
        scores = by_source.get(source_block.block_id, [])
        best = scores[0] if scores else None
        high_precision_scores = [
            score
            for score in scores
            if score.source_recall >= 0.08 and score.generated_precision >= 0.50
        ]
        combined_common = min(len(source_block.tokens), sum(score.common_tokens for score in high_precision_scores[:5]))
        combined_recall = safe_div(combined_common, len(source_block.tokens))
        covered = bool(
            best
            and (
                best.source_recall >= args.covered_source_recall
                or combined_recall >= args.covered_combined_recall
            )
        )
        if covered:
            covered_sources += 1
            order_anchors.append(
                source_order_anchor(
                    source_block=source_block,
                    best=best,
                    high_precision_scores=high_precision_scores,
                    source_by_id=source_by_id,
                    generated_by_id=generated_by_id,
                )
            )
        else:
            uncovered_source_examples.append(
                {
                    "source": block_payload(source_block),
                    "best_matches": [score_payload(score, source_by_id, generated_by_id) for score in scores[:3]],
                    "combined_recall": combined_recall,
                }
            )
            continue

        split_scores = high_precision_scores
        if (
            len(split_scores) >= 2
            and combined_recall >= args.split_combined_recall
            and (best is None or best.source_recall < args.split_best_source_recall_max)
        ):
            missing_merge_examples.append(
                {
                    "source": block_payload(source_block),
                    "combined_recall": combined_recall,
                    "best_source_recall": best.source_recall if best else None,
                    "generated_parts": [
                        score_payload(score, source_by_id, generated_by_id)
                        for score in sorted(split_scores, key=lambda score: generated_by_id[score.generated_id].index)[:8]
                    ],
                }
            )
        else:
            one_to_one_sources += 1

    wrong_merge_examples: list[dict[str, Any]] = []
    for generated_block in generated_blocks:
        scores = by_generated.get(generated_block.block_id, [])
        source_like_scores = [
            score
            for score in scores
            if score.generated_precision >= 0.45 and score.source_recall >= 0.45
        ]
        combined_common = min(len(generated_block.tokens), sum(score.common_tokens for score in source_like_scores[:6]))
        generated_coverage = safe_div(combined_common, len(generated_block.tokens))
        if len(source_like_scores) >= 2 and generated_coverage >= args.overmerge_generated_coverage:
            wrong_merge_examples.append(
                {
                    "generated": block_payload(generated_block),
                    "generated_coverage": generated_coverage,
                    "source_parts": [
                        score_payload(score, source_by_id, generated_by_id)
                        for score in sorted(source_like_scores, key=lambda score: source_by_id[score.source_id].index)[:8]
                    ],
                }
            )

    summary = {
        "source_paragraph_count": len(source_blocks),
        "generated_paragraph_count": len(generated_blocks),
        "covered_source_paragraph_count": covered_sources,
        "one_to_one_or_preserved_source_count": one_to_one_sources,
        "missing_merge_source_paragraph_count": len(missing_merge_examples),
        "wrong_merge_generated_paragraph_count": len(wrong_merge_examples),
        "uncovered_source_paragraph_count": len(source_blocks) - covered_sources,
        "source_coverage_rate": safe_div(covered_sources, len(source_blocks)),
        "missing_merge_rate_among_covered": safe_div(len(missing_merge_examples), covered_sources),
        "wrong_merge_rate_among_generated": safe_div(len(wrong_merge_examples), len(generated_blocks)),
        "paragraph_count_delta": len(generated_blocks) - len(source_blocks),
    }
    order_summary, order_examples = source_order_metrics(
        order_anchors,
        source_count=len(source_blocks),
        generated_count=len(generated_blocks),
    )
    summary.update(order_summary)
    return {
        "summary": summary,
        "missing_merge_examples": missing_merge_examples[: args.max_examples],
        "wrong_merge_examples": wrong_merge_examples[: args.max_examples],
        "uncovered_source_examples": uncovered_source_examples[: args.max_examples],
        "source_order_inversion_examples": order_examples[: args.max_examples],
    }


def source_order_anchor(
    *,
    source_block: ParagraphBlock,
    best: PairScore | None,
    high_precision_scores: list[PairScore],
    source_by_id: dict[str, ParagraphBlock],
    generated_by_id: dict[str, ParagraphBlock],
) -> dict[str, Any]:
    """Return the generated paragraph position used for source-order checks.

    Coverage is intentionally content-oriented.  This anchor adds a second,
    order-sensitive view: after a source paragraph is covered somewhere in the
    generated text, where did that match appear in generated paragraph order?
    """
    anchor_scores = high_precision_scores or ([best] if best else [])
    anchor_scores = [score for score in anchor_scores if score is not None]
    if not anchor_scores:
        generated_indexes: list[int] = []
    else:
        generated_indexes = sorted({generated_by_id[score.generated_id].index for score in anchor_scores})
    anchor_index = generated_indexes[0] if generated_indexes else None
    anchor_end_index = generated_indexes[-1] if generated_indexes else None
    return {
        "source": block_payload(source_block),
        "source_index": source_block.index,
        "generated_index": anchor_index,
        "generated_end_index": anchor_end_index,
        "generated_indexes": generated_indexes,
        "best_match": score_payload(best, source_by_id, generated_by_id) if best else None,
    }


def source_order_metrics(
    order_anchors: list[dict[str, Any]],
    *,
    source_count: int,
    generated_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchors = [anchor for anchor in sorted(order_anchors, key=lambda item: item["source_index"]) if anchor.get("generated_index") is not None]
    n = len(anchors)
    pair_count = n * (n - 1) // 2
    inversions = 0
    ties = 0
    examples: list[dict[str, Any]] = []
    for i in range(n):
        left = anchors[i]
        left_gen = int(left["generated_index"])
        for j in range(i + 1, n):
            right = anchors[j]
            right_gen = int(right["generated_index"])
            if left_gen > right_gen:
                inversions += 1
                if len(examples) < 100:
                    examples.append(
                        {
                            "source_order": [left["source_index"], right["source_index"]],
                            "generated_order": [left_gen, right_gen],
                            "left_source": left["source"],
                            "right_source": right["source"],
                            "left_best_match": left.get("best_match"),
                            "right_best_match": right.get("best_match"),
                        }
                    )
            elif left_gen == right_gen:
                ties += 1
    matched_by_source = {int(anchor["source_index"]): anchor for anchor in anchors}
    adjacent_source_pair_count = 0
    adjacent_inversions = 0
    adjacent_examples: list[dict[str, Any]] = []
    for source_index in range(max(source_count - 1, 0)):
        left = matched_by_source.get(source_index)
        right = matched_by_source.get(source_index + 1)
        if not left or not right:
            continue
        adjacent_source_pair_count += 1
        left_gen = int(left["generated_index"])
        right_gen = int(right["generated_index"])
        if left_gen > right_gen:
            adjacent_inversions += 1
            if len(adjacent_examples) < 20:
                adjacent_examples.append(
                    {
                        "source_order": [left["source_index"], right["source_index"]],
                        "generated_order": [left_gen, right_gen],
                        "left_source": left["source"],
                        "right_source": right["source"],
                    }
                )
    displaced_010 = 0
    displaced_015 = 0
    displaced_examples: list[dict[str, Any]] = []
    for anchor in anchors:
        source_norm = safe_div(int(anchor["source_index"]), max(source_count - 1, 1))
        generated_norm = safe_div(int(anchor["generated_index"]), max(generated_count - 1, 1))
        displacement = abs(source_norm - generated_norm)
        if displacement > 0.10:
            displaced_010 += 1
        if displacement > 0.15:
            displaced_015 += 1
            if len(displaced_examples) < 40:
                displaced_examples.append(
                    {
                        "source_index": anchor["source_index"],
                        "generated_index": anchor["generated_index"],
                        "source_rank_norm": source_norm,
                        "generated_rank_norm": generated_norm,
                        "displacement": displacement,
                        "source": anchor["source"],
                        "best_match": anchor.get("best_match"),
                    }
                )
    generated_sequence = [int(anchor["generated_index"]) for anchor in anchors]
    lis_len = longest_increasing_subsequence_length(generated_sequence)
    comparable_pairs = pair_count - ties
    inversion_rate = safe_div(inversions, pair_count)
    tau = 1.0 if comparable_pairs == 0 else safe_div((comparable_pairs - inversions) - inversions, comparable_pairs)
    coverage_rate = safe_div(n, source_count)
    summary = {
        "source_order_matched_count": n,
        "source_order_pair_count": pair_count,
        "source_order_inversion_count": inversions,
        "source_order_tie_count": ties,
        "source_order_inversion_rate": inversion_rate,
        "source_order_kendall_tau": tau,
        "source_order_adjacent_pair_count": adjacent_source_pair_count,
        "source_order_adjacent_inversion_count": adjacent_inversions,
        "source_order_adjacent_inversion_rate": safe_div(adjacent_inversions, adjacent_source_pair_count),
        "source_order_displaced_count_010": displaced_010,
        "source_order_displaced_rate_010": safe_div(displaced_010, n),
        "source_order_displaced_count_015": displaced_015,
        "source_order_displaced_rate_015": safe_div(displaced_015, n),
        "source_order_lis_length": lis_len,
        "source_order_lis_disorder_rate": 1.0 - safe_div(lis_len, n),
        "ordered_source_coverage_rate": coverage_rate * (1.0 - inversion_rate),
    }
    return summary, examples + adjacent_examples + displaced_examples


def longest_increasing_subsequence_length(values: list[int]) -> int:
    """Return LIS length for generated positions in source order."""
    import bisect

    piles: list[int] = []
    for value in values:
        pos = bisect.bisect_left(piles, value)
        if pos == len(piles):
            piles.append(value)
        else:
            piles[pos] = value
    return len(piles)


def score_payload(
    score: PairScore,
    source_by_id: dict[str, ParagraphBlock],
    generated_by_id: dict[str, ParagraphBlock],
) -> dict[str, Any]:
    return {
        **asdict(score),
        "source": block_payload(source_by_id[score.source_id]),
        "generated": block_payload(generated_by_id[score.generated_id]),
    }


def block_payload(block: ParagraphBlock) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "index": block.index,
        "line": block.line,
        "semantic_channel": block.semantic_channel,
        "token_count": len(block.tokens),
        "preview": block.text[:240],
    }


def block_summary(block: ParagraphBlock) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "index": block.index,
        "line": block.line,
        "semantic_channel": block.semantic_channel,
        "token_count": len(block.tokens),
        "preview": block.text[:180],
    }


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Paragraph Preservation Against TeX",
        "",
        f"- doc_id: `{payload['doc_id']}`",
        f"- source paragraphs: {summary['source_paragraph_count']}",
        f"- body source paragraphs: {summary.get('body_source_paragraph_count', 0)}",
        f"- excluded source paragraphs from body metric: {summary.get('excluded_source_paragraph_count', 0)}",
        f"- generated paragraphs: {summary['generated_paragraph_count']}",
        f"- covered source paragraphs: {summary['covered_source_paragraph_count']}",
        f"- covered body source paragraphs: {summary.get('body_covered_source_paragraph_count', 0)}",
        f"- missing-merge source paragraphs: {summary['missing_merge_source_paragraph_count']}",
        f"- body missing-merge source paragraphs: {summary.get('body_missing_merge_source_paragraph_count', 0)}",
        f"- wrong-merge generated paragraphs: {summary['wrong_merge_generated_paragraph_count']}",
        f"- source coverage rate raw/legacy: {summary['source_coverage_rate']:.4f}",
        f"- ordered source coverage rate raw/legacy: {fmt_metric(summary.get('ordered_source_coverage_rate_raw'))}",
        f"- source order inversion rate raw/legacy: {fmt_metric(summary.get('source_order_inversion_rate_raw'))}",
        f"- body source coverage rate: {fmt_metric(summary.get('body_source_coverage_rate'))}",
        f"- body ordered source coverage rate: {fmt_metric(summary.get('body_ordered_source_coverage_rate'))}",
        f"- body source order inversion rate: {fmt_metric(summary.get('body_source_order_inversion_rate'))}",
        f"- body source order Kendall tau: {fmt_metric(summary.get('body_source_order_kendall_tau'))}",
        f"- visible prose source coverage rate: {fmt_metric(summary.get('visible_prose_source_coverage_rate'))}",
        f"- visible prose ordered coverage rate: {fmt_metric(summary.get('visible_prose_ordered_coverage_rate'))}",
        f"- visible prose order inversion rate: {fmt_metric(summary.get('visible_prose_order_inversion_rate'))}",
        f"- adjacent prose inversion rate: {fmt_metric(summary.get('adjacent_prose_inversion_rate'))}",
        f"- displaced prose paragraph rate >0.10: {fmt_metric(summary.get('displaced_prose_paragraph_rate_010'))}",
        f"- displaced prose paragraph rate >0.15: {fmt_metric(summary.get('displaced_prose_paragraph_rate_015'))}",
        f"- visible prose LIS disorder rate: {fmt_metric(summary.get('visible_prose_lis_disorder_rate'))}",
        f"- missing-merge rate among covered: {summary['missing_merge_rate_among_covered']:.4f}",
        f"- body missing-merge rate among covered: {fmt_metric(summary.get('body_missing_merge_rate_among_covered'))}",
        f"- wrong-merge rate among generated: {summary['wrong_merge_rate_among_generated']:.4f}",
        f"- body wrong-merge rate among generated: {fmt_metric(summary.get('body_wrong_merge_rate_among_generated'))}",
        f"- paragraph count delta: {summary['paragraph_count_delta']}",
        f"- body paragraph count delta: {summary.get('body_paragraph_count_delta', 0)}",
        "",
        "## Interpretation",
        "",
        "- `source coverage rate raw/legacy` keeps the historical extractor behavior.",
        "- `body source coverage rate` excludes source-only TeX implementation fragments such as TikZ styles, diagram node listings, macro setup, and non-body environments.",
        "- `ordered source coverage rate` discounts pairwise source/generated order inversions, so text that is present but shuffled is visible to this audit.",
        "- `visible prose` metrics add type-aware matching and only count ordinary body prose, excluding front matter, captions, references, display math, formula-only blocks, URLs/metadata, and OCR artifacts.",
        "- `adjacent prose inversion` measures source-adjacent visible prose paragraph pairs only, avoiding pairwise amplification from one misplaced paragraph.",
        "- `visible prose LIS disorder` estimates the minimum fraction of matched prose paragraphs that must move to make generated order monotonic.",
        "- `missing-merge source paragraphs` means one source paragraph appears split across multiple generated paragraphs.",
        "- `wrong-merge generated paragraphs` means one generated paragraph appears to combine multiple source paragraphs.",
        "- This is a generator-facing paragraph preservation audit, not an edge-level GNN MERGE metric.",
        "",
        "## Missing-Merge Examples",
        "",
    ]
    append_missing_merge_table(lines, payload.get("missing_merge_examples", []))
    lines.extend(["", "## Wrong-Merge Examples", ""])
    append_wrong_merge_table(lines, payload.get("wrong_merge_examples", []))
    lines.extend(["", "## Uncovered Source Examples", ""])
    append_uncovered_table(lines, payload.get("uncovered_source_examples", []))
    lines.extend(["", "## Body Uncovered Source Examples", ""])
    append_uncovered_table(lines, payload.get("body_uncovered_source_examples", []))
    lines.extend(["", "## Visible Prose Order Inversion Examples", ""])
    append_order_inversion_table(lines, payload.get("visible_prose_order_inversion_examples", []))
    lines.extend(["", "## Visible Prose Displaced Examples", ""])
    append_displaced_table(lines, payload.get("visible_prose_displaced_examples", []))
    lines.extend(["", "## Excluded Source Paragraphs From Body Metric", ""])
    append_excluded_table(lines, payload.get("excluded_source_paragraphs", []))
    lines.append("")
    return "\n".join(lines)


def append_missing_merge_table(lines: list[str], examples: list[dict[str, Any]]) -> None:
    if not examples:
        lines.append("No missing-merge examples detected under the current thresholds.")
        return
    lines.append("| source line | combined recall | best recall | source preview | generated parts |")
    lines.append("| ---: | ---: | ---: | --- | --- |")
    for example in examples:
        source = example["source"]
        parts = " / ".join(
            f"L{part['generated']['line']} {part['generated']['preview'][:80]}"
            for part in example.get("generated_parts", [])
        )
        lines.append(
            f"| {source.get('line')} | {float(example.get('combined_recall') or 0):.3f} | "
            f"{float(example.get('best_source_recall') or 0):.3f} | {md(source.get('preview'))} | {md(parts)} |"
        )


def append_wrong_merge_table(lines: list[str], examples: list[dict[str, Any]]) -> None:
    if not examples:
        lines.append("No wrong-merge examples detected under the current thresholds.")
        return
    lines.append("| generated line | generated coverage | generated preview | source parts |")
    lines.append("| ---: | ---: | --- | --- |")
    for example in examples:
        generated = example["generated"]
        parts = " / ".join(
            f"L{part['source']['line']} {part['source']['preview'][:80]}"
            for part in example.get("source_parts", [])
        )
        lines.append(
            f"| {generated.get('line')} | {float(example.get('generated_coverage') or 0):.3f} | "
            f"{md(generated.get('preview'))} | {md(parts)} |"
        )


def append_uncovered_table(lines: list[str], examples: list[dict[str, Any]]) -> None:
    if not examples:
        lines.append("No uncovered source examples detected under the current thresholds.")
        return
    lines.append("| source line | combined recall | source preview | best matches |")
    lines.append("| ---: | ---: | --- | --- |")
    for example in examples:
        source = example["source"]
        matches = " / ".join(
            f"L{match['generated']['line']} r={match['source_recall']:.2f} {match['generated']['preview'][:60]}"
            for match in example.get("best_matches", [])
        )
        lines.append(
            f"| {source.get('line')} | {float(example.get('combined_recall') or 0):.3f} | "
            f"{md(source.get('preview'))} | {md(matches)} |"
        )


def append_excluded_table(lines: list[str], examples: list[dict[str, Any]]) -> None:
    if not examples:
        lines.append("No source paragraphs were excluded from the body metric.")
        return
    lines.append("| source line | reason | source preview |")
    lines.append("| ---: | --- | --- |")
    for example in examples:
        lines.append(
            f"| {example.get('line')} | {md(example.get('exclude_reason'))} | {md(example.get('preview'))} |"
        )


def append_order_inversion_table(lines: list[str], examples: list[dict[str, Any]]) -> None:
    if not examples:
        lines.append("No visible-prose order inversion examples detected under the current thresholds.")
        return
    lines.append("| source order | generated order | left source | right source |")
    lines.append("| --- | --- | --- | --- |")
    for example in examples[:20]:
        left = example.get("left_source") or {}
        right = example.get("right_source") or {}
        lines.append(
            f"| {example.get('source_order')} | {example.get('generated_order')} | {md(left.get('preview'))} | {md(right.get('preview'))} |"
        )


def append_displaced_table(lines: list[str], examples: list[dict[str, Any]]) -> None:
    if not examples:
        lines.append("No displaced visible-prose examples detected under the current thresholds.")
        return
    lines.append("| source idx | generated idx | displacement | source preview |")
    lines.append("| ---: | ---: | ---: | --- |")
    for example in examples[:20]:
        source = example.get("source") or {}
        lines.append(
            f"| {example.get('source_index')} | {example.get('generated_index')} | "
            f"{float(example.get('displacement') or 0):.3f} | {md(source.get('preview'))} |"
        )


def md(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return text[:260]


def fmt_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
