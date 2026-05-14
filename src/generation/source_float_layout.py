"""Recover float placement hints from the original TeX source.

The generated document should not blindly copy arbitrary source table code: our
current table reconstruction is image-crop based.  Still, the source TeX tells
us a lot about *placement intent*: column-local ``table`` versus page-wide
``table*``, explicit placement options, and whether the author wrapped the
tabular in ``resizebox{\\textwidth}`` or similar commands.

This module extracts only those stable layout hints and matches them back to
DocumentIR table nodes by caption text.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from src.reasoning.latex_flattener import flatten_inputs, read_text_lossy, strip_comments


TABLE_ENV_RE = re.compile(
    r"\\begin\{(?P<env>table\*?)\}(?:\[(?P<placement>[^\]]+)\])?(?P<body>.*?)\\end\{(?P=env)\}",
    re.DOTALL,
)
CAPTION_COMMAND_RE = re.compile(r"\\caption\*?(?:\s*\[[^\]]*\])?\s*\{", re.DOTALL)
TABLE_LABEL_RE = re.compile(r"\btable\s*(?P<label>[0-9]+(?:\.[0-9]+)*[A-Za-z]?|[IVXLCDM]+)\b", re.IGNORECASE)
COMMAND_RE = re.compile(r"\\[A-Za-z@]+\*?(?:\s*\[[^\]]*\])?")
BRACE_RE = re.compile(r"[{}]")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SourceTableLayout:
    index: int
    environment: str
    placement: str | None
    caption: str
    normalized_caption: str
    label: str | None
    width_scope: str | None
    uses_resizebox: bool = False
    uses_adjustbox: bool = False
    uses_tabularx: bool = False
    uses_minipage: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "source_table_index": self.index,
            "source_environment": self.environment,
            "source_placement": self.placement,
            "source_caption": self.caption,
            "source_label": self.label,
            "source_width_scope": self.width_scope,
            "source_uses_resizebox": self.uses_resizebox,
            "source_uses_adjustbox": self.uses_adjustbox,
            "source_uses_tabularx": self.uses_tabularx,
            "source_uses_minipage": self.uses_minipage,
        }


@dataclass(frozen=True)
class SourceFloatLayout:
    source_tex_path: str
    tables: tuple[SourceTableLayout, ...]

    def match_table(self, caption: str, *, min_score: float = 0.45) -> SourceTableLayout | None:
        normalized = normalize_caption(caption)
        if not normalized:
            return None
        label = extract_table_label(caption)
        best: tuple[float, SourceTableLayout] | None = None
        for layout in self.tables:
            score = caption_similarity(normalized, layout.normalized_caption)
            if label and layout.label and label.casefold() == layout.label.casefold():
                score += 0.35
            if normalized and layout.normalized_caption and normalized in layout.normalized_caption:
                score += 0.15
            if layout.normalized_caption and normalized and layout.normalized_caption in normalized:
                score += 0.15
            if best is None or score > best[0]:
                best = (score, layout)
        if best is None or best[0] < min_score:
            return None
        return best[1]


@lru_cache(maxsize=64)
def resolve_source_float_layout(source_tex_path: str | Path | None) -> SourceFloatLayout | None:
    if source_tex_path is None:
        return None
    path = Path(source_tex_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        raw = strip_comments(read_text_lossy(path))
        flat = flatten_inputs(raw, path.parent)
    except Exception:
        return None
    tables = tuple(extract_source_table_layouts(flat))
    if not tables:
        return None
    return SourceFloatLayout(source_tex_path=str(path), tables=tables)


def extract_source_table_layouts(tex: str) -> list[SourceTableLayout]:
    layouts: list[SourceTableLayout] = []
    for index, match in enumerate(TABLE_ENV_RE.finditer(tex)):
        env = match.group("env")
        placement = (match.group("placement") or "").strip() or None
        body = match.group("body") or ""
        caption = extract_caption_text(body)
        normalized = normalize_caption(caption)
        layouts.append(
            SourceTableLayout(
                index=index,
                environment=env,
                placement=placement,
                caption=caption,
                normalized_caption=normalized,
                label=extract_table_label(caption),
                width_scope=infer_table_width_scope(env, body),
                uses_resizebox=bool(re.search(r"\\resizebox\b", body)),
                uses_adjustbox=bool(re.search(r"\\begin\{adjustbox\}|\\adjustbox\b", body)),
                uses_tabularx=bool(re.search(r"\\begin\{tabularx\}|\\begin\{tabular\*\}", body)),
                uses_minipage=bool(re.search(r"\\begin\{minipage\}", body)),
            )
        )
    return layouts


def extract_caption_text(body: str) -> str:
    match = CAPTION_COMMAND_RE.search(body)
    if not match:
        return ""
    start = match.end() - 1
    group, _end = read_balanced_group(body, start)
    return latex_to_text(group)


def read_balanced_group(text: str, open_index: int) -> tuple[str, int]:
    if open_index >= len(text) or text[open_index] != "{":
        return "", open_index
    depth = 0
    cursor = open_index
    chars: list[str] = []
    while cursor < len(text):
        char = text[cursor]
        if char == "\\" and cursor + 1 < len(text):
            chars.append(char)
            cursor += 1
            chars.append(text[cursor])
            cursor += 1
            continue
        if char == "{":
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars), cursor + 1
            chars.append(char)
        else:
            chars.append(char)
        cursor += 1
    return "".join(chars), cursor


def infer_table_width_scope(environment: str, body: str) -> str | None:
    body_lc = body.casefold()
    if environment == "table*":
        return "page"
    if re.search(r"\\(?:resizebox|makebox)\s*\{\s*\\(?:textwidth|linewidth)\s*\}", body):
        macro = re.search(r"\\(?:resizebox|makebox)\s*\{\s*\\(?P<width>textwidth|linewidth|columnwidth)\s*\}", body)
        if macro and macro.group("width") == "textwidth":
            return "page"
        return "column"
    if re.search(r"\\begin\{adjustbox\}\s*\[[^\]]*(?:width|max width)\s*=\s*\\textwidth", body_lc):
        return "page"
    if re.search(r"\\begin\{adjustbox\}\s*\[[^\]]*(?:width|max width)\s*=\s*\\(?:linewidth|columnwidth)", body_lc):
        return "column"
    if re.search(r"\\begin\{tabularx\}\s*\{\s*\\textwidth\s*\}", body):
        return "page"
    if re.search(r"\\begin\{tabularx\}\s*\{\s*\\(?:linewidth|columnwidth)\s*\}", body):
        return "column"
    return None


def latex_to_text(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"\\(?:cite|citep|citet|ref|label)\*?(?:\[[^\]]*\])?\{[^{}]*\}", " ", value)
    value = COMMAND_RE.sub(" ", value)
    value = BRACE_RE.sub(" ", value)
    value = value.replace("~", " ")
    value = SPACE_RE.sub(" ", value)
    return value.strip()


def normalize_caption(text: str) -> str:
    value = latex_to_text(text)
    value = TABLE_LABEL_RE.sub(" ", value)
    value = re.sub(r"[^0-9A-Za-z]+", " ", value).casefold()
    return SPACE_RE.sub(" ", value).strip()


def extract_table_label(text: str) -> str | None:
    match = TABLE_LABEL_RE.search(str(text or ""))
    return match.group("label") if match else None


def caption_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    ratio = SequenceMatcher(None, left, right).ratio()
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return ratio
    jaccard = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    return 0.65 * ratio + 0.35 * jaccard
