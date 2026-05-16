"""Stepwise layout parser for reconstructing document structure.

The parser intentionally reasons over layout signals only: type, role, page,
order, bbox, heading/list prefixes, and local band information.  Text content is
used only for shallow prefix probes, not semantic understanding.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.perception.title_features import is_front_matter_date_text, strip_title_numbering, title_numbering_level


NUMERIC_ID_RE = re.compile(r"\d+")
ORDERED_NUMERIC_DOT_PREFIX_RE = re.compile(r"^\s*(\d+)\.\s+")
NUMERIC_PAREN_PREFIX_RE = re.compile(r"^\s*\d+\)\s+")
NUMERIC_PREFIX_RE = re.compile(r"^\s*\d+[\.\)]\s+")
DOTTED_NUMERIC_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)+\.?\s+")
APPENDIX_DOTTED_PREFIX_RE = re.compile(r"^\s*[A-Z](?:\.\d+)+\.?\s+")
ALPHA_PREFIX_RE = re.compile(r"^\s*[A-Za-z][\.\)]\s+")
ROMAN_PREFIX_RE = re.compile(r"^\s*[IVXLCDM]+[\.\)]\s+", re.IGNORECASE)
CUSTOM_COLON_PREFIX_RE = re.compile(r"^\s*[^\s:]{1,24}(?:\s+[\w\-\.]+){0,3}\s*:\s+")
LIST_MARKER_RE = re.compile(r"^\s*([\u2022\u25E6\u25CB\u25AA\-\*]|\d+\.|[a-zA-Z]\.)\s+")


@dataclass(frozen=True)
class LayoutToken:
    node_id: int
    record: dict[str, Any]
    text: str
    block_type: str
    role: str
    order_key: tuple[int, float, float, str]
    is_noise: bool
    prefix_kind: str
    prefix_level: int | None
    numeric_item: int | None
    heading_candidate: bool


@dataclass
class ListState:
    parent_id: int | None
    environment: str
    next_number: int | None = None
    last_item_id: int | None = None
    last_effective_pos: int = 0


@dataclass
class LayoutParseResult:
    heading_ids: set[int] = field(default_factory=set)
    heading_levels: dict[int, int] = field(default_factory=dict)
    heading_parent: dict[int, int | None] = field(default_factory=dict)
    scope_by_node: dict[int, int | None] = field(default_factory=dict)
    parent_by_node: dict[int, int] = field(default_factory=dict)
    render_hints: dict[int, dict[str, Any]] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)


def parse_layout_state_machine(
    records_by_id: dict[int, dict[str, Any]],
    *,
    text_by_id: dict[int, str] | None = None,
) -> LayoutParseResult:
    """Parse a v7 node stream into a conservative layout AST skeleton."""

    text_by_id = text_by_id or {}
    ordered_ids = sorted(records_by_id, key=lambda node_id: node_order_key(records_by_id[node_id], node_id))
    tokens = {
        node_id: make_token(node_id, records_by_id[node_id], text=text_by_id.get(node_id))
        for node_id in ordered_ids
    }
    body_font_size = infer_body_font_size(records_by_id.values())
    result = LayoutParseResult()
    heading_stack: list[tuple[int, int]] = []
    active_list: ListState | None = None
    effective_pos = 0

    for node_id in ordered_ids:
        token = tokens[node_id]
        if token.is_noise:
            result.events.append(f"{node_id}:skip-noise")
            continue
        effective_pos += 1

        if token_layout_layer(token) == "main_text_flow":
            while heading_stack and token_layout_layer(tokens[heading_stack[-1][1]]) != "main_text_flow":
                popped = heading_stack.pop()
                result.events.append(f"{node_id}:close-frontmatter-heading {popped[1]}")

        current_scope = heading_stack[-1][1] if heading_stack else None
        current_heading_level = heading_stack[-1][0] if heading_stack else 0

        if is_numbered_list_continuation(token, active_list, effective_pos=effective_pos):
            parent = active_list.parent_id if active_list is not None else current_scope
            set_parent(result, node_id, parent)
            mark_list_item(result, node_id)
            if active_list is None:
                active_list = ListState(parent_id=parent, environment="enumerate")
            active_list.last_item_id = node_id
            active_list.last_effective_pos = effective_pos
            active_list.next_number = (token.numeric_item or 0) + 1
            result.scope_by_node[node_id] = current_scope
            result.events.append(f"{node_id}:continue-enumerate parent={parent}")
            continue

        decision = classify_heading_token(
            token,
            body_font_size=body_font_size,
            current_heading_level=current_heading_level,
            effective_pos=effective_pos,
        )
        if decision is not None:
            active_list = None
            level, hints, reason = decision
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            parent = heading_stack[-1][1] if heading_stack else None
            result.heading_ids.add(node_id)
            result.heading_levels[node_id] = level
            result.heading_parent[node_id] = parent
            result.scope_by_node[node_id] = node_id
            if parent is not None:
                result.parent_by_node[node_id] = parent
            if hints:
                result.render_hints[node_id] = hints
            heading_stack.append((level, node_id))
            result.events.append(f"{node_id}:heading level={level} parent={parent} reason={reason}")
            continue

        if is_list_item_token(token):
            parent = current_scope
            set_parent(result, node_id, parent)
            if token.numeric_item is not None:
                environment = "enumerate"
                next_number = token.numeric_item + 1
            else:
                environment = "itemize"
                next_number = None
            active_list = ListState(
                parent_id=parent,
                environment=environment,
                next_number=next_number,
                last_item_id=node_id,
                last_effective_pos=effective_pos,
            )
            result.scope_by_node[node_id] = current_scope
            result.events.append(f"{node_id}:start-list parent={parent} env={environment}")
            continue

        set_parent(result, node_id, current_scope)
        result.scope_by_node[node_id] = current_scope
        if token.block_type in {"title", "table", "figure"}:
            active_list = None
        result.events.append(f"{node_id}:attach scope={current_scope}")

    return result


def classify_heading_token(
    token: LayoutToken,
    *,
    body_font_size: float,
    current_heading_level: int,
    effective_pos: int,
) -> tuple[int, dict[str, Any], str] | None:
    if not token.heading_candidate:
        return None
    if token.record.get("run_in_heading"):
        level = int(numeric_value(token.record.get("run_in_heading_level")) or token.prefix_level or 2)
        level = max(1, min(level, 3))
        return (
            level,
            {
                "_layout_state_locked": True,
                "_heading_render_level": level,
                "_run_in_heading": True,
            },
            "run-in-heading",
        )
    if special_title(token.text):
        return (1, {"_heading_unnumbered": True, "_layout_state_locked": True}, "special-title")
    if token.prefix_kind in {"custom_colon", "numeric_paren"}:
        return None
    if token.prefix_level is not None and token.block_type == "title" and looks_like_standalone_heading(token.text):
        return (token.prefix_level, {}, "numbered-title")
    if token.role in {"list_item", "list"}:
        return None
    if token.prefix_level is not None:
        return (token.prefix_level, {}, "numbered-heading")
    if token.block_type != "title" and not looks_like_standalone_heading(token.text):
        return None
    if token.role == "heading" and is_local_subheading_layout(token.record):
        # Local column/band headings are usually lower-level headings only
        # after a section is already open.  If the heading stack is empty, this
        # is the first body heading after front matter; rendering it as a
        # subsection makes LaTeX number it as "0.1".  Open a real section first.
        level = 1 if current_heading_level <= 0 else max(2, min(current_heading_level + 1, 3))
        return (level, {"_heading_unnumbered": True, "_layout_state_locked": True}, "local-heading")
    if token.role == "heading" or token.block_type == "title":
        level = heading_level_from_style(token.record, body_font_size=body_font_size, effective_pos=effective_pos, text=token.text)
        return (level, {}, "visual-heading")
    return None


def is_numbered_list_continuation(token: LayoutToken, active_list: ListState | None, *, effective_pos: int) -> bool:
    if token.numeric_item is None or token.numeric_item <= 1:
        return False
    if token.role not in {"list_item", "list"}:
        return False
    if active_list is None or active_list.environment != "enumerate":
        return False
    if active_list.next_number != token.numeric_item:
        return False
    return effective_pos - active_list.last_effective_pos <= 16


def is_list_item_token(token: LayoutToken) -> bool:
    if token.block_type == "list":
        return True
    if token.role in {"list_item", "list"} and LIST_MARKER_RE.match(token.text):
        return True
    return False


def mark_list_item(result: LayoutParseResult, node_id: int) -> None:
    hints = result.render_hints.setdefault(node_id, {})
    hints["_render_as_list_item"] = True
    hints["_layout_state_locked"] = True


def set_parent(result: LayoutParseResult, node_id: int, parent: int | None) -> None:
    if parent is not None and parent != node_id:
        result.parent_by_node[node_id] = parent


def make_token(node_id: int, record: dict[str, Any], *, text: str | None = None) -> LayoutToken:
    value = text if text is not None else record_text(record)
    block_type = canonical_type(record)
    role = layout_role(record)
    prefix_kind, prefix_level = heading_prefix(value)
    return LayoutToken(
        node_id=node_id,
        record=record,
        text=value,
        block_type=block_type,
        role=role,
        order_key=node_order_key(record, node_id),
        is_noise=is_noise_record(record, value),
        prefix_kind=prefix_kind,
        prefix_level=prefix_level,
        numeric_item=ordered_numeric_dot_number(value),
        heading_candidate=is_heading_candidate(record, value, block_type=block_type, role=role),
    )


def record_text(record: dict[str, Any]) -> str:
    return str(
        record.get("merged_text")
        or record.get("text_for_embedding")
        or record.get("text")
        or record.get("text_preview")
        or record.get("latex")
        or ""
    )


def canonical_type(record: dict[str, Any]) -> str:
    if str(record.get("list_type") or "").lower() == "reference_list":
        return "reference"
    raw = str(record.get("canonical_type") or record.get("type") or record.get("raw_type") or record.get("block_type") or "").lower()
    if raw in {"paragraph", "text", "paragraph_text", "body"}:
        return "text"
    if raw in {"title", "section", "subsection", "subsubsection", "heading"}:
        return "title"
    if raw in {"equation", "equation_interline", "interline_equation", "display_formula", "formula"}:
        return "equation"
    if raw in {"inline_math", "inline_formula", "math_inline"}:
        return "inline_math"
    if raw in {"table"}:
        return "table"
    if raw in {"figure", "image", "chart"}:
        return "figure"
    if raw in {"list", "item", "itemize", "enumerate"}:
        return "list"
    if raw in {"algorithm"}:
        return "algorithm"
    if raw in {"code"}:
        return "code"
    if raw in {"reference", "references", "bibliography"}:
        return "reference"
    return "text"


def layout_role(record: dict[str, Any]) -> str:
    return str(record.get("layout_role") or record.get("role") or record.get("semantic_role") or "").casefold()


def token_layout_layer(token: LayoutToken) -> str:
    return str(token.record.get("layout_layer") or "").casefold()


def node_order_key(record: dict[str, Any], node_id: int) -> tuple[int, float, float, str]:
    for key in ("regime_reading_order", "dag_reading_order", "xycut_reading_order", "global_order", "reading_order", "original_order", "original_index", "index"):
        value = numeric_value(record.get(key))
        if value is not None:
            return (0, value, 0.0, "")
    for key in ("id", "node_id", "block_id"):
        value = numeric_value(record.get(key))
        if value is not None:
            return (1, value, 0.0, "")
    page = numeric_value(record.get("page_idx"))
    visual = numeric_value(record.get("visual_order"))
    if page is not None or visual is not None:
        return (2, page or 0.0, visual or 0.0, "")
    return (3, float(node_id), 0.0, "")


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = NUMERIC_ID_RE.search(value)
        if match:
            return float(match.group(0))
    return None


def is_noise_record(record: dict[str, Any], text: str) -> bool:
    if layout_role(record) == "noise" or str(record.get("layout_layer") or "") == "noise_layer":
        return True
    raw = str(record.get("type") or record.get("raw_type") or record.get("block_type") or record.get("canonical_type") or "").casefold()
    if raw in {"header", "footer", "page_header", "page_footer", "page_number", "page_num", "page_no", "pagenum", "noise"}:
        return True
    normalized = normalize_heading_text(text).replace(" ", "")
    return bool(normalized) and normalized.isdigit() and len(normalized) <= 4


def is_heading_candidate(record: dict[str, Any], text: str, *, block_type: str, role: str) -> bool:
    if not text.strip():
        return False
    if record.get("run_in_heading"):
        return True
    if role in {"toc_title", "toc_entry"} or is_toc_title_text(text):
        return False
    if is_front_matter_date_text(text):
        return False
    if block_type == "title":
        return True
    if role == "heading":
        return True
    if block_type not in {"text", "reference"}:
        return False
    font_size = node_font_size(record)
    body_like = font_size <= 0
    return not body_like and looks_like_standalone_heading(text)


def heading_prefix(text: str) -> tuple[str, int | None]:
    value = " ".join(str(text or "").split())
    if not value:
        return ("empty", None)
    if APPENDIX_DOTTED_PREFIX_RE.match(value):
        token = value.split(maxsplit=1)[0].rstrip(".")
        return ("appendix_dotted", max(2, token.count(".") + 1))
    if DOTTED_NUMERIC_PREFIX_RE.match(value):
        token = value.split(maxsplit=1)[0].rstrip(".")
        return ("dotted_numeric", max(2, token.count(".") + 1))
    if NUMERIC_PAREN_PREFIX_RE.match(value):
        return ("numeric_paren", 2)
    if NUMERIC_PREFIX_RE.match(value):
        return ("numeric", 1)
    alpha_match = ALPHA_PREFIX_RE.match(value)
    if alpha_match:
        token = value.split(maxsplit=1)[0].rstrip(".").rstrip(")")
        tail = value[alpha_match.end() :].strip()
        if ROMAN_PREFIX_RE.match(value) and (len(token) > 1 or looks_like_all_caps_heading(tail)):
            return ("roman", 1)
        return ("alpha", 2)
    numbered_level = title_numbering_level(value)
    if numbered_level is not None:
        return ("bare_numbered", numbered_level)
    if ROMAN_PREFIX_RE.match(value):
        return ("roman", 1)
    if CUSTOM_COLON_PREFIX_RE.match(value):
        return ("custom_colon", None)
    return ("freeform", None)


def looks_like_all_caps_heading(text: str) -> bool:
    letters = [char for char in str(text or "") if char.isalpha()]
    if len(letters) < 4:
        return False
    uppercase = sum(1 for char in letters if char.isupper())
    return uppercase / max(1, len(letters)) >= 0.75


def ordered_numeric_dot_number(text: str) -> int | None:
    match = ORDERED_NUMERIC_DOT_PREFIX_RE.match(" ".join(str(text or "").split()))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def special_title(text: str) -> bool:
    normalized = normalize_heading_text(text)
    return normalized in {"abstract", "references", "bibliography"} or normalized.startswith("appendix")


def is_toc_title_text(text: str) -> bool:
    normalized = re.sub(r"[^a-z]+", "", str(text or "").casefold())
    return normalized in {"contents", "tableofcontents"}


def normalize_heading_text(text: str) -> str:
    lowered = str(text or "").casefold().strip()
    without_punctuation = "".join(char for char in lowered if not unicodedata.category(char).startswith("P"))
    return " ".join(without_punctuation.split())


def looks_like_standalone_heading(text: str) -> bool:
    value = " ".join(str(text or "").split())
    if not value or len(value) > 180:
        return False
    if "@" in value or "\\@" in value or value.count(",") >= 2:
        return False
    if ":" in value and not value.rstrip().endswith(":"):
        return False
    if value.endswith((".", "。", "?", "!", "？", "！")):
        return False
    return True


def is_local_subheading_layout(record: dict[str, Any]) -> bool:
    band_type = str(record.get("layout_band_type") or "").casefold()
    band_column = str(record.get("layout_band_column") or "").casefold()
    column_id = numeric_value(record.get("layout_band_column_id"))
    boundary = bool(record.get("layout_is_band_boundary"))
    if not band_type and not band_column and column_id is None:
        return False
    if band_type == "double_column":
        return True
    if band_column in {"left", "right"}:
        return True
    if column_id is not None and int(column_id) in {0, 1}:
        return True
    return not boundary and band_type not in {"full_span", "single_column"}


def heading_level_from_style(record: dict[str, Any], *, body_font_size: float, effective_pos: int, text: str) -> int:
    explicit = title_numbering_level(text)
    if explicit is not None:
        return explicit
    raw = str(record.get("type") or record.get("raw_type") or record.get("block_type") or "").casefold()
    if raw == "section":
        return 1
    if raw == "subsection":
        return 2
    if raw == "subsubsection":
        return 3
    font_size = node_font_size(record)
    if effective_pos == 1 and body_font_size > 0 and font_size >= body_font_size * 1.25 and len(text) >= 25:
        return 0
    if body_font_size > 0 and font_size >= body_font_size * 1.15:
        return 1
    if body_font_size > 0 and font_size >= body_font_size * 1.03:
        return 2
    return 1


def infer_body_font_size(records: Any) -> float:
    weighted: dict[float, int] = {}
    fallback: dict[float, int] = {}
    for record in records:
        size = node_font_size(record)
        if size <= 0:
            continue
        text_len = max(1, len(record_text(record)))
        fallback[size] = fallback.get(size, 0) + text_len
        if canonical_type(record) == "text":
            weighted[size] = weighted.get(size, 0) + text_len
    source = weighted or fallback
    if not source:
        return 0.0
    return max(source.items(), key=lambda item: item[1])[0]


def node_font_size(record: dict[str, Any]) -> float:
    for key in ("style_baseline_size", "font_size", "baseline_font_size"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    spans = record.get("style_spans")
    if not isinstance(spans, list):
        return 0.0
    weighted: dict[float, int] = {}
    for span in spans:
        if not isinstance(span, dict):
            continue
        size = span.get("font_size")
        if not isinstance(size, (int, float)):
            continue
        weight = int(span.get("char_count") or len(str(span.get("text") or "")) or 1)
        weighted[float(size)] = weighted.get(float(size), 0) + max(1, weight)
    if not weighted:
        return 0.0
    return max(weighted.items(), key=lambda item: item[1])[0]
