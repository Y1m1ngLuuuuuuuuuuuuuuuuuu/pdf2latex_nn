#!/usr/bin/env python3
"""Audit generated reading order against source TeX anchors.

This is a lightweight sanity checker, not a replacement for the project
comparison metrics. It extracts heading/body anchors from source TeX, searches
for them in generated TeX after conservative normalization, and reports anchor
order inversions. When a v8 diagnostics file is provided it also summarizes
page-order and continuation-merge evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BODY_ENVIRONMENTS_TO_DROP = (
    "figure",
    "figure*",
    "table",
    "table*",
    "algorithm",
    "algorithm*",
    "tabular",
    "tabular*",
)

TEXT_COMMANDS_TO_KEEP = (
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "subparagraph",
    "title",
    "author",
    "caption",
    "textbf",
    "textit",
    "emph",
    "underline",
)


@dataclass
class Anchor:
    anchor_id: str
    kind: str
    source_index: int
    source_line: int | None
    phrase: str
    normalized_key: str
    generated_index: int
    generated_line: int | None
    source_occurrence_count: int
    generated_occurrence_count: int
    ambiguous: bool
    status: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tex", required=True, type=Path)
    parser.add_argument("--generated-tex", required=True, type=Path)
    parser.add_argument("--v8-diagnostics", type=Path)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-anchor-chars", type=int, default=48)
    parser.add_argument("--body-anchor-tokens", type=int, default=12)
    parser.add_argument("--max-body-anchors", type=int, default=80)
    parser.add_argument("--include-caption-anchors", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_text = args.source_tex.read_text(encoding="utf-8", errors="ignore")
    generated_text = args.generated_tex.read_text(encoding="utf-8", errors="ignore")
    doc_id = args.doc_id or args.source_tex.parent.name or args.generated_tex.stem

    generated_norm = normalize_for_search(generated_text)
    generated_line_offsets = normalized_line_offsets(generated_text)

    anchors = extract_anchors(
        source_text,
        source_text,
        generated_norm=generated_norm,
        generated_line_offsets=generated_line_offsets,
        min_anchor_chars=args.min_anchor_chars,
        body_anchor_tokens=args.body_anchor_tokens,
        max_body_anchors=args.max_body_anchors,
        include_caption_anchors=args.include_caption_anchors,
    )
    inversions = find_order_inversions(anchors)
    diagnostics = load_v8_diagnostics(args.v8_diagnostics)
    summary = summarize(doc_id, args, anchors, inversions, diagnostics)

    payload = {
        "schema_version": "reading_order_tex_anchor_audit_v1",
        "doc_id": doc_id,
        "inputs": {
            "source_tex": str(args.source_tex),
            "generated_tex": str(args.generated_tex),
            "v8_diagnostics": str(args.v8_diagnostics) if args.v8_diagnostics else None,
        },
        "config": {
            "min_anchor_chars": args.min_anchor_chars,
            "body_anchor_tokens": args.body_anchor_tokens,
            "max_body_anchors": args.max_body_anchors,
            "include_caption_anchors": args.include_caption_anchors,
        },
        "summary": summary,
        "anchors": [asdict(anchor) for anchor in anchors],
        "order_inversions": inversions,
        "v8_diagnostics_summary": diagnostics,
    }

    json_path = args.output_dir / "reading_order_tex_anchor_audit.json"
    md_path = args.output_dir / "READING_ORDER_TEX_ANCHOR_AUDIT.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


def strip_comments(text: str) -> str:
    # Remove unescaped percent comments line-by-line.
    lines = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        lines.append(line[: match.start()] if match else line)
    return "\n".join(lines)


def drop_environment_blocks(text: str, envs: tuple[str, ...] = BODY_ENVIRONMENTS_TO_DROP) -> str:
    result = text
    for env in envs:
        escaped = re.escape(env)
        result = re.sub(
            rf"\\begin\{{{escaped}\}}.*?\\end\{{{escaped}\}}",
            " ",
            result,
            flags=re.DOTALL,
        )
    return result


def preserve_simple_command_arguments(text: str) -> str:
    result = text
    command_pattern = "|".join(re.escape(cmd) for cmd in TEXT_COMMANDS_TO_KEEP)
    pattern = re.compile(rf"\\(?:{command_pattern})\*?(?:\[[^\]]*\])?\{{([^{{}}]*)\}}")
    # Iterate a few times to unwrap nested simple commands.
    for _ in range(5):
        new = pattern.sub(r" \1 ", result)
        if new == result:
            break
        result = new
    return result


def tex_to_plainish(text: str, *, drop_floats: bool) -> str:
    text = strip_comments(text)
    if drop_floats:
        text = drop_environment_blocks(text)
    text = preserve_simple_command_arguments(text)
    text = re.sub(r"\\(?:cite|ref|label|url|href)\*?(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\begin\{[^{}]+\}|\\end\{[^{}]+\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("~", " ")
    text = text.replace("--", " ")
    text = text.replace("–", " ")
    text = text.replace("—", " ")
    return text


def normalize_for_search(text: str) -> str:
    text = tex_to_plainish(text, drop_floats=False)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "", text).lower()
    return text


def normalize_phrase(text: str) -> str:
    return normalize_for_search(text)


def normalized_line_offsets(text: str) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    normalized_cursor = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        line_with_newline = line + "\n"
        normalized = normalize_for_search(line_with_newline)
        offsets.append((normalized_cursor, line_no))
        normalized_cursor += len(normalized)
        cursor += len(line_with_newline)
    return offsets


def line_for_normalized_index(offsets: list[tuple[int, int]], index: int) -> int | None:
    if index < 0:
        return None
    current_line = None
    for start, line_no in offsets:
        if start <= index:
            current_line = line_no
        else:
            break
    return current_line


def source_line_for_raw_index(text: str, raw_index: int) -> int:
    return text.count("\n", 0, max(raw_index, 0)) + 1


def extract_anchors(
    full_source_text: str,
    source_text: str,
    *,
    generated_norm: str,
    generated_line_offsets: list[tuple[int, int]],
    min_anchor_chars: int,
    body_anchor_tokens: int,
    max_body_anchors: int,
    include_caption_anchors: bool,
) -> list[Anchor]:
    anchors: list[tuple[str, int, int | None, str]] = []
    seen_keys: set[str] = set()
    body_start = begin_document_offset(source_text)

    heading_re = re.compile(
        r"\\(?P<cmd>section|subsection|subsubsection)\*?(?:\[[^\]]*\])?\{(?P<text>[^{}]+)\}"
    )
    for match in heading_re.finditer(source_text):
        if match.start() < body_start or is_commented_position(source_text, match.start()):
            continue
        phrase = clean_anchor_phrase(match.group("text"))
        add_anchor_candidate(anchors, seen_keys, f"heading:{match.group('cmd')}", match.start(), source_line_for_raw_index(source_text, match.start()), phrase, min_anchor_chars=8)

    if include_caption_anchors:
        caption_re = re.compile(r"\\caption(?:\[[^\]]*\])?\{(?P<text>[^{}]+)\}")
        for match in caption_re.finditer(source_text):
            if match.start() < body_start or is_commented_position(source_text, match.start()):
                continue
            phrase = clean_anchor_phrase(match.group("text"))
            add_anchor_candidate(anchors, seen_keys, "caption", match.start(), source_line_for_raw_index(source_text, match.start()), phrase, min_anchor_chars=24)

    body_added = 0
    for body_offset, paragraph in iter_source_paragraphs(source_text, start=body_start):
        if paragraph_is_float_or_layout(paragraph):
            continue
        raw_phrase = clean_anchor_phrase(tex_to_plainish(paragraph, drop_floats=True))
        phrase = first_tokens(raw_phrase, body_anchor_tokens)
        if len(normalize_phrase(phrase)) < min_anchor_chars:
            continue
        if looks_like_low_value_anchor(phrase):
            continue
        if add_anchor_candidate(
            anchors,
            seen_keys,
            "body",
            body_offset,
            source_line_for_raw_index(source_text, body_offset),
            phrase,
            min_anchor_chars=min_anchor_chars,
        ):
            body_added += 1
        if body_added >= max_body_anchors:
            break

    anchors.sort(key=lambda row: row[1])
    materialized: list[Anchor] = []
    for idx, (kind, source_index, source_line, phrase) in enumerate(anchors):
        key = normalize_phrase(phrase)
        source_occurrences = occurrence_count(normalize_for_search(full_source_text), key)
        generated_occurrences = occurrence_count(generated_norm, key)
        generated_index = generated_norm.find(key)
        ambiguous = is_ambiguous_anchor(key, source_occurrences, generated_occurrences)
        if generated_index < 0:
            status = "missing_generated"
        elif ambiguous:
            status = "found_ambiguous"
        else:
            status = "found"
        materialized.append(
            Anchor(
                anchor_id=f"a{idx:04d}",
                kind=kind,
                source_index=source_index,
                source_line=source_line,
                phrase=phrase,
                normalized_key=key,
                generated_index=generated_index,
                generated_line=line_for_normalized_index(generated_line_offsets, generated_index),
                source_occurrence_count=source_occurrences,
                generated_occurrence_count=generated_occurrences,
                ambiguous=ambiguous,
                status=status,
            )
        )
    materialized.sort(key=lambda anchor: (anchor.source_index if anchor.source_index >= 0 else 10**12, anchor.anchor_id))
    return materialized


def begin_document_offset(text: str) -> int:
    match = re.search(r"\\begin\{document\}", text)
    return match.end() if match else 0


def is_commented_position(text: str, index: int) -> bool:
    line_start = text.rfind("\n", 0, index) + 1
    prefix = text[line_start:index]
    match = re.search(r"(?<!\\)%", prefix)
    return match is not None


def iter_source_paragraphs(text: str, *, start: int) -> list[tuple[int, str]]:
    body = text[start:]
    paragraphs: list[tuple[int, str]] = []
    cursor = 0
    for match in re.finditer(r"\S.*?(?=\n\s*\n|$)", body, flags=re.DOTALL):
        paragraph = match.group(0)
        raw_start = start + match.start()
        # Advance defensively to avoid repeated zero-length surprises.
        if raw_start < cursor:
            continue
        cursor = raw_start + len(paragraph)
        paragraphs.append((raw_start, paragraph))
    return paragraphs


def paragraph_is_float_or_layout(paragraph: str) -> bool:
    if re.search(r"\\begin\{(?:figure\*?|table\*?|tabular\*?|algorithm\*?)\}", paragraph):
        return True
    if re.search(r"\\(?:maketitle|bibliography|bibliographystyle)\b", paragraph):
        return True
    if re.fullmatch(r"\s*\\(?:label|vspace|hspace|centering|small|normalsize|clearpage|newpage)\b.*", paragraph, flags=re.DOTALL):
        return True
    return False


def occurrence_count(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    return sum(1 for _ in re.finditer(re.escape(needle), haystack))


def is_ambiguous_anchor(key: str, source_occurrences: int, generated_occurrences: int) -> bool:
    if generated_occurrences <= 1 and source_occurrences <= 1:
        return False
    # Short heading labels such as "Methodology" or dataset names often appear
    # in prose before their real heading; using them for order inversions creates
    # false positives. Long body anchors are usually specific enough.
    return len(key) < 48


def add_anchor_candidate(
    anchors: list[tuple[str, int, int | None, str]],
    seen_keys: set[str],
    kind: str,
    source_index: int,
    source_line: int | None,
    phrase: str,
    *,
    min_anchor_chars: int,
) -> bool:
    key = normalize_phrase(phrase)
    if len(key) < min_anchor_chars or key in seen_keys:
        return False
    seen_keys.add(key)
    anchors.append((kind, source_index, source_line, phrase))
    return True


def clean_anchor_phrase(text: str) -> str:
    text = preserve_simple_command_arguments(text)
    text = re.sub(r"\\(?:cite|ref|label|url|href)\*?(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"[$^_{}&]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_tokens(text: str, count: int) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", text)
    return " ".join(tokens[:count])


def looks_like_low_value_anchor(phrase: str) -> bool:
    lowered = phrase.lower()
    if lowered.startswith(("figure ", "table ", "algorithm ")):
        return True
    if len(re.findall(r"[A-Za-z]", phrase)) < 20:
        return True
    return False


def find_order_inversions(anchors: list[Anchor]) -> list[dict[str, Any]]:
    found = [
        anchor
        for anchor in anchors
        if anchor.generated_index >= 0 and anchor.source_index >= 0 and not anchor.ambiguous
    ]
    inversions: list[dict[str, Any]] = []
    for i, left in enumerate(found):
        for right in found[i + 1 :]:
            if left.source_index < right.source_index and left.generated_index > right.generated_index:
                inversions.append(
                    {
                        "before_anchor_id": left.anchor_id,
                        "after_anchor_id": right.anchor_id,
                        "before_phrase": left.phrase,
                        "after_phrase": right.phrase,
                        "before_source_line": left.source_line,
                        "after_source_line": right.source_line,
                        "before_generated_line": left.generated_line,
                        "after_generated_line": right.generated_line,
                    }
                )
    return inversions


def load_v8_diagnostics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    page_orders = diagnostics.get("page_orders") or {}
    page_summaries: dict[str, Any] = {}
    for page, rows in page_orders.items():
        if not isinstance(rows, list):
            continue
        reading_orders = [row.get("reading_order") for row in rows if isinstance(row, dict)]
        page_summaries[str(page)] = {
            "item_count": len(rows),
            "reading_order_min": min(reading_orders) if reading_orders else None,
            "reading_order_max": max(reading_orders) if reading_orders else None,
            "non_monotonic_count": count_non_monotonic(reading_orders),
            "items": [
                {
                    "reading_order": row.get("reading_order"),
                    "middle_index": row.get("middle_index"),
                    "type": row.get("type"),
                    "text_preview": row.get("text_preview"),
                }
                for row in rows
                if isinstance(row, dict)
            ],
        }
    return {
        "schema_version": diagnostics.get("schema_version"),
        "doc_id": diagnostics.get("doc_id"),
        "block_count": diagnostics.get("block_count"),
        "item_count": diagnostics.get("item_count"),
        "merge_count": diagnostics.get("merge_count"),
        "merge_reason_counts": diagnostics.get("merge_reason_counts"),
        "merge_decisions": diagnostics.get("merge_decisions", []),
        "page_order_pages": sorted(page_summaries.keys()),
        "page_order_summaries": page_summaries,
    }


def count_non_monotonic(values: list[Any]) -> int:
    cleaned = [value for value in values if isinstance(value, (int, float))]
    return sum(1 for left, right in zip(cleaned, cleaned[1:]) if right < left)


def summarize(
    doc_id: str,
    args: argparse.Namespace,
    anchors: list[Anchor],
    inversions: list[dict[str, Any]],
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    found = [anchor for anchor in anchors if anchor.generated_index >= 0]
    unambiguous_found = [anchor for anchor in anchors if anchor.status == "found"]
    missing = [anchor for anchor in anchors if anchor.status == "missing_generated"]
    found_count = len(found)
    total = len(anchors)
    max_pairs = len(unambiguous_found) * (len(unambiguous_found) - 1) // 2
    inversion_count = len(inversions)
    order_score = None if max_pairs == 0 else max(0.0, 1.0 - inversion_count / max_pairs)
    found_rate = None if total == 0 else found_count / total
    status = "pass"
    notes: list[str] = []
    if total == 0:
        status = "fail"
        notes.append("no anchors extracted from source TeX")
    elif found_rate is not None and found_rate < 0.5:
        status = "fail"
        notes.append("less than half of anchors found in generated TeX")
    elif inversion_count > 10:
        status = "fail"
        notes.append("many source-order inversions detected")
    elif inversion_count > 0 or (found_rate is not None and found_rate < 0.75):
        status = "warn"
        if inversion_count > 0:
            notes.append("some anchor order inversions detected")
        if found_rate is not None and found_rate < 0.75:
            notes.append("anchor found rate is moderate")
    if diagnostics:
        merge_count = diagnostics.get("merge_count") or 0
        notes.append(f"v8 diagnostics merge_count={merge_count}")
    return {
        "doc_id": doc_id,
        "status": status,
        "notes": notes,
        "anchor_count": total,
        "found_anchor_count": found_count,
        "unambiguous_found_anchor_count": len(unambiguous_found),
        "missing_anchor_count": len([anchor for anchor in anchors if anchor.status == "missing_generated"]),
        "ambiguous_anchor_count": len([anchor for anchor in anchors if anchor.ambiguous]),
        "found_rate": found_rate,
        "order_inversion_count": inversion_count,
        "source_order_score": order_score,
        "missing_anchor_ids": [anchor.anchor_id for anchor in missing],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    diagnostics = payload.get("v8_diagnostics_summary")
    lines = [
        "# Reading Order TeX Anchor Audit",
        "",
        f"- doc_id: `{payload['doc_id']}`",
        f"- status: **{summary['status']}**",
        f"- anchors: {summary['found_anchor_count']} / {summary['anchor_count']} found",
        f"- found_rate: {format_float(summary['found_rate'])}",
        f"- order_inversions: {summary['order_inversion_count']}",
        f"- source_order_score: {format_float(summary['source_order_score'])}",
        "",
        "## Inputs",
        "",
        f"- source_tex: `{payload['inputs']['source_tex']}`",
        f"- generated_tex: `{payload['inputs']['generated_tex']}`",
        f"- v8_diagnostics: `{payload['inputs']['v8_diagnostics']}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in summary.get("notes", []))
    lines.extend(["", "## Inversions", ""])
    inversions = payload.get("order_inversions", [])
    if not inversions:
        lines.append("No source-order inversions were detected among found anchors.")
    else:
        lines.append("| before source line | after source line | before generated line | after generated line | before | after |")
        lines.append("| ---: | ---: | ---: | ---: | --- | --- |")
        for inv in inversions[:30]:
            lines.append(
                "| {before_source_line} | {after_source_line} | {before_generated_line} | {after_generated_line} | {before_phrase} | {after_phrase} |".format(
                    before_source_line=inv.get("before_source_line"),
                    after_source_line=inv.get("after_source_line"),
                    before_generated_line=inv.get("before_generated_line"),
                    after_generated_line=inv.get("after_generated_line"),
                    before_phrase=md_cell(inv.get("before_phrase")),
                    after_phrase=md_cell(inv.get("after_phrase")),
                )
            )
        if len(inversions) > 30:
            lines.append(f"\nOnly the first 30 of {len(inversions)} inversions are shown.")
    lines.extend(["", "## Anchor Summary", ""])
    lines.append("| id | kind | source line | generated line | status | phrase |")
    lines.append("| --- | --- | ---: | ---: | --- | --- |")
    for anchor in payload.get("anchors", [])[:120]:
        lines.append(
            "| {anchor_id} | {kind} | {source_line} | {generated_line} | {status} | {phrase} |".format(
                anchor_id=anchor.get("anchor_id"),
                kind=anchor.get("kind"),
                source_line=anchor.get("source_line"),
                generated_line=anchor.get("generated_line"),
                status=anchor.get("status"),
                phrase=md_cell(anchor.get("phrase")),
            )
        )
    if diagnostics:
        lines.extend(["", "## V8 Diagnostics", ""])
        lines.append(f"- merge_count: {diagnostics.get('merge_count')}")
        lines.append(f"- merge_reason_counts: `{json.dumps(diagnostics.get('merge_reason_counts'), ensure_ascii=False)}`")
        lines.append(f"- page_order_pages: `{', '.join(diagnostics.get('page_order_pages') or [])}`")
        merge_decisions = diagnostics.get("merge_decisions") or []
        if merge_decisions:
            lines.extend(["", "### Merge Decisions", ""])
            lines.append("| src | dst | reason | confidence | tail -> head |")
            lines.append("| --- | --- | --- | ---: | --- |")
            for decision in merge_decisions[:30]:
                evidence = decision.get("evidence") or {}
                tail_head = f"{evidence.get('prev_text_tail', '')} -> {evidence.get('curr_text_head', '')}"
                lines.append(
                    "| {src} | {dst} | {reason} | {confidence} | {tail_head} |".format(
                        src=md_cell(decision.get("src_block_id")),
                        dst=md_cell(decision.get("dst_block_id")),
                        reason=decision.get("reason"),
                        confidence=decision.get("confidence"),
                        tail_head=md_cell(tail_head),
                    )
                )
    lines.append("")
    return "\n".join(lines)


def format_float(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ")
    text = text.replace("|", "\\|")
    return text[:180]


if __name__ == "__main__":
    raise SystemExit(main())
