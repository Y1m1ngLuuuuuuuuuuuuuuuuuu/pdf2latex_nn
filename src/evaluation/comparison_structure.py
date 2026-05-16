"""Converters from LaTeX/Markdown outputs into a shared evaluation structure.

The comparison structure is intentionally coarser than our production
``DocumentIR``.  It is the neutral layer used to compare systems with different
output languages, such as our LaTeX generator and Nougat's Markdown-like output.
It focuses on the parts we actually claim to improve: reading order, heading
hierarchy, paragraph/list structure, float/caption placement, references, and
cross references.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.reasoning.latex_flattener import LatexFlattenerConfig, flatten_latex_file, strip_comments


COMPARISON_SCHEMA_VERSION = "comparison_structure_v1"
ROOT_ID = "ROOT"

SECTION_LEVELS = {
    "part": 0,
    "chapter": 1,
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
    "paragraph": 4,
    "subparagraph": 5,
}
SECTION_COMMAND_RE = re.compile(
    r"\\(?P<name>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?"
)
BEGIN_END_RE = re.compile(r"\\(?P<kind>begin|end)\s*\{\s*(?P<name>[^}]+?)\s*\}")
ITEM_RE = re.compile(r"\\item\b(?:\s*\[[^\]]*\])?")
CAPTION_RE = re.compile(r"\\caption\*?")
DISPLAY_MATH_DELIM_RE = re.compile(r"\\\[.*?\\\]|\$\$.*?\$\$", re.DOTALL)
INLINE_MATH_RE = re.compile(r"(?<!\\)\$.*?(?<!\\)\$|\\\(.*?\\\)", re.DOTALL)
DISPLAY_MATH_ENVS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "flalign",
    "flalign*",
    "eqnarray",
    "eqnarray*",
    "displaymath",
}
LIST_ENVS = {"itemize", "enumerate", "description"}
FIGURE_ENVS = {"figure", "figure*"}
TABLE_ENVS = {"table", "table*"}
ALGORITHM_ENVS = {"algorithm", "algorithmic", "algorithm2e", "lstlisting", "verbatim"}
REFERENCE_ENVS = {"thebibliography"}
SKIP_ENVS = {"document", "center", "flushleft", "flushright", "minipage", "multicols"}

LATEX_TOKEN_RE = re.compile(
    r"\\begin\s*\{\s*(?:equation\*?|align\*?|gather\*?|multline\*?|flalign\*?|eqnarray\*?|displaymath)\s*\}"
    r".*?\\end\s*\{\s*(?:equation\*?|align\*?|gather\*?|multline\*?|flalign\*?|eqnarray\*?|displaymath)\s*\}"
    r"|\\(?:begin|end)\s*\{\s*[^}]+?\s*\}"
    r"|\\(?:part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?"
    r"|\\(?:title|author)\*?"
    r"|\\caption\*?"
    r"|\\item\b(?:\s*\[[^\]]*\])?"
    r"|\\bibitem\b(?:\s*\[[^\]]*\])?\s*\{[^}]*\}"
    r"|\\\[.*?\\\]"
    r"|\$\$.*?\$\$",
    re.DOTALL,
)

MARKDOWN_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
MARKDOWN_LIST_RE = re.compile(r"^\s*(?P<marker>(?:[-*+])|(?:\d+\.))\s+(?P<text>.+?)\s*$")
CAPTION_TEXT_RE = re.compile(
    r"^\s*(?P<kind>fig(?:ure)?|table|alg(?:orithm)?)\s*\.?\s*"
    r"(?P<number>[A-Za-z0-9.:-]+)?\s*[:.\-]?\s*(?P<text>.*)$",
    re.I,
)
MARKDOWN_LATEX_SECTION_RE = re.compile(
    r"^\s*\\(?P<name>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?"
    r"\s*\{(?P<title>.*)\}\s*$"
)
MARKDOWN_LATEX_CAPTION_RE = re.compile(r"\\caption\*?\s*\{(?P<caption>.*?)\}", re.DOTALL)
MARKDOWN_LATEX_BEGIN_RE = re.compile(r"^\s*\\begin\s*\{\s*(?P<name>[^}]+?)\s*\}")
MARKDOWN_LATEX_END_TEMPLATE = r"\\end\s*\{\s*%s\s*\}"
REFERENCE_HEADING_RE = re.compile(r"^(references|bibliography|参考文献)\s*$", re.I)
ABSTRACT_HEADING_RE = re.compile(r"^abstract\s*$", re.I)
REFERENCE_ITEM_RE = re.compile(r"^\s*(?:\[(?P<bracket>[A-Za-z0-9]+)\]|(?P<num>\d+)\.)\s*(?P<text>.+)")
CITATION_RE = re.compile(
    r"\\(?:cite|citet|citep|citealp|citeauthor|citeyear|autocite|parencite|textcite)\*?"
    r"(?:\s*\[[^\]]*\]){0,2}\s*\{(?P<keys>[^}]+)\}"
)
CROSS_REF_RE = re.compile(
    r"\b(?P<kind>fig(?:ure)?|table|tab\.|equation|eq\.|algorithm|alg\.)\s*"
    r"(?P<label>[A-Za-z]?\d+(?:\.\d+)*[A-Za-z]?)",
    re.I,
)
MARKDOWN_CITATION_RE = re.compile(
    r"\[(?P<key>(?:[A-Za-z0-9_:-]+)(?:\s*,\s*[A-Za-z0-9_:-]+)*)\]"
)
SILENT_LATEX_ARG_COMMAND_RE = re.compile(
    r"\\(?:includegraphics|label|vspace|hspace|rule|resizebox|scalebox|input|include)"
    r"\*?(?:\s*\[[^\]]*\])*\s*\{[^{}]*\}",
    re.DOTALL,
)


@dataclass
class ComparisonBlock:
    block_id: str
    block_type: str
    order: int
    text: str = ""
    normalized_text: str = ""
    level: int | None = None
    parent_id: str | None = None
    marker: str | None = None
    label: str | None = None
    citations: list[str] = field(default_factory=list)
    cross_refs: list[dict[str, str]] = field(default_factory=list)
    inline_math_count: int = 0
    display_math_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonDocument:
    doc_id: str
    source_format: str
    blocks: list[ComparisonBlock]
    schema_version: str = COMPARISON_SCHEMA_VERSION
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reading_order"] = [block.block_id for block in self.blocks]
        payload["heading_tree"] = [
            {
                "block_id": block.block_id,
                "level": block.level,
                "parent_id": block.parent_id,
                "text": block.text,
                "normalized_text": block.normalized_text,
            }
            for block in self.blocks
            if block.block_type == "heading"
        ]
        payload["parent_edges"] = [
            {"parent_id": block.parent_id, "child_id": block.block_id}
            for block in self.blocks
            if block.parent_id and block.parent_id != ROOT_ID
        ]
        payload["test_items"] = build_test_items(self.blocks)
        return payload


def latex_file_to_comparison(path: Path, *, doc_id: str | None = None) -> ComparisonDocument:
    flattened = flatten_latex_file(
        path,
        config=LatexFlattenerConfig(mask_math=False, inject_bbl=True, expand_zero_arg_macros=True),
    )
    return latex_to_comparison(flattened.content, doc_id=doc_id or path.stem, source_path=path, metadata=flattened.summary())


def latex_to_comparison(
    tex: str,
    *,
    doc_id: str = "document",
    source_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> ComparisonDocument:
    parser = _LatexComparisonParser(tex, doc_id=doc_id, source_path=source_path, metadata=metadata or {})
    return parser.parse()


def markdown_file_to_comparison(path: Path, *, doc_id: str | None = None) -> ComparisonDocument:
    return markdown_to_comparison(path.read_text(encoding="utf-8", errors="replace"), doc_id=doc_id or path.stem, source_path=path)


def markdown_to_comparison(
    markdown: str,
    *,
    doc_id: str = "document",
    source_path: Path | None = None,
) -> ComparisonDocument:
    parser = _MarkdownComparisonParser(markdown, doc_id=doc_id, source_path=source_path)
    return parser.parse()


class _LatexComparisonParser:
    def __init__(self, tex: str, *, doc_id: str, source_path: Path | None, metadata: dict[str, Any]) -> None:
        self.tex = document_body(strip_comments(tex))
        self.doc_id = doc_id
        self.source_path = str(source_path) if source_path else None
        self.metadata = metadata
        self.blocks: list[ComparisonBlock] = []
        self.section_stack: dict[int, str] = {}
        self.context_stack: list[dict[str, str]] = []
        self.next_id = 1

    def parse(self) -> ComparisonDocument:
        cursor = 0
        for match in LATEX_TOKEN_RE.finditer(self.tex):
            self.flush_text(self.tex[cursor : match.start()])
            token = match.group(0)
            if token.startswith("\\begin"):
                cursor = self.handle_begin(token, match.end())
                continue
            elif token.startswith("\\end"):
                self.handle_end(token)
            elif SECTION_COMMAND_RE.match(token):
                end = self.handle_section(match.start(), match.end(), token)
                cursor = end
                continue
            elif re.match(r"\\(?:title|author)\*?", token):
                end = self.handle_front_matter_command(match.end(), token)
                cursor = end
                continue
            elif CAPTION_RE.match(token):
                end = self.handle_caption(match.end())
                cursor = end
                continue
            elif ITEM_RE.match(token):
                self.handle_item(token)
            elif token.startswith("\\bibitem"):
                self.handle_bibitem(token)
            else:
                self.handle_display_math(token)
            cursor = match.end()
        self.flush_text(self.tex[cursor:])
        return ComparisonDocument(
            doc_id=self.doc_id,
            source_format="latex",
            source_path=self.source_path,
            blocks=self.blocks,
            metadata=self.metadata,
        )

    def flush_text(self, raw: str) -> None:
        text = latex_text(raw)
        for paragraph in split_paragraphs(text):
            if not paragraph:
                continue
            block_type = self.default_text_type()
            self.add_block(block_type, paragraph)

    def handle_begin(self, token: str, token_end: int) -> int:
        env = env_name_from_token(token)
        if env in DISPLAY_MATH_ENVS and "\\end" in token:
            self.handle_display_math(token)
            return token_end
        if env in SKIP_ENVS:
            return token_end
        if env in LIST_ENVS:
            self.context_stack.append({"type": "list", "env": env, "id": self.add_block("list", env, marker=env).block_id})
            return token_end
        if env in FIGURE_ENVS:
            self.context_stack.append({"type": "figure", "env": env, "id": self.add_block("figure", "", marker=env).block_id})
            return token_end
        if env in TABLE_ENVS:
            self.context_stack.append({"type": "table", "env": env, "id": self.add_block("table", "", marker=env).block_id})
            return token_end
        if env in REFERENCE_ENVS:
            self.context_stack.append({"type": "references", "env": env, "id": self.ensure_references_heading()})
            after = skip_optional_args(self.tex, token_end)
            _, close = read_braced(self.tex, after)
            return close if close > after else token_end
        if env == "abstract":
            self.context_stack.append({"type": "abstract", "env": env, "id": self.add_block("abstract", "Abstract", level=1).block_id})
            return token_end
        if env in DISPLAY_MATH_ENVS:
            self.context_stack.append({"type": "math", "env": env, "id": ""})
            return token_end
        if env in ALGORITHM_ENVS:
            self.context_stack.append({"type": "algorithm", "env": env, "id": self.add_block("algorithm", "", marker=env).block_id})
        return token_end

    def handle_end(self, token: str) -> None:
        env = env_name_from_token(token)
        while self.context_stack:
            ctx = self.context_stack.pop()
            if ctx.get("env") == env:
                break

    def handle_section(self, start: int, end: int, token: str) -> int:
        name_match = SECTION_COMMAND_RE.match(token)
        if not name_match:
            return end
        name = name_match.group("name")
        after = skip_optional_args(self.tex, end)
        title, close = read_braced(self.tex, after)
        if close <= after:
            return end
        level = SECTION_LEVELS[name]
        parent_id = self.section_parent(level)
        block = self.add_block("heading", latex_text(title), level=level, parent_id=parent_id, marker=name)
        self.section_stack = {key: value for key, value in self.section_stack.items() if key < level}
        self.section_stack[level] = block.block_id
        return close

    def handle_front_matter_command(self, start: int, token: str) -> int:
        after = skip_optional_args(self.tex, start)
        text, close = read_braced(self.tex, after)
        if close <= after:
            return start
        name = token.lstrip("\\").rstrip("*")
        block_type = "author_block" if name == "author" else "document_title"
        self.add_block(block_type, latex_text(text), parent_id=ROOT_ID, marker=name)
        return close

    def handle_caption(self, start: int) -> int:
        after = skip_optional_args(self.tex, start)
        text, close = read_braced(self.tex, after)
        if close <= after:
            return start
        parent = self.current_float_parent()
        caption_text = latex_text(text)
        kind, label = caption_kind_and_label(caption_text, parent)
        self.add_block("caption", caption_text, parent_id=parent, marker=kind, label=label)
        return close

    def handle_item(self, token: str) -> None:
        parent_id = self.current_parent_id()
        marker_match = re.search(r"\[([^\]]+)\]", token)
        self.add_block("list_item", "", parent_id=parent_id, marker=marker_match.group(1) if marker_match else None)

    def handle_bibitem(self, token: str) -> None:
        self.ensure_references_heading()
        key_match = re.search(r"\{([^}]+)\}", token)
        self.add_block("reference_item", "", parent_id=self.current_parent_id(), marker=key_match.group(1) if key_match else None)

    def handle_display_math(self, token: str) -> None:
        math_text = normalize_math_text(token)
        self.add_block("display_math", math_text, display_math_count=1, metadata={"raw": token[:200]})

    def add_block(
        self,
        block_type: str,
        text: str,
        *,
        level: int | None = None,
        parent_id: str | None = None,
        marker: str | None = None,
        label: str | None = None,
        display_math_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ComparisonBlock:
        if block_type == "list_item" and text and self.blocks and self.blocks[-1].block_type == "list_item" and not self.blocks[-1].text:
            return self.update_last_block(text)
        if block_type == "reference_item" and text and self.blocks and self.blocks[-1].block_type == "reference_item" and not self.blocks[-1].text:
            return self.update_last_block(text)
        block_id = f"B_{self.next_id:06d}"
        self.next_id += 1
        parent = parent_id if parent_id is not None else self.current_parent_id()
        clean_text_value = " ".join(text.split())
        block = ComparisonBlock(
            block_id=block_id,
            block_type=block_type,
            order=len(self.blocks),
            text=clean_text_value,
            normalized_text=normalize_for_compare(clean_text_value),
            level=level,
            parent_id=None if parent == ROOT_ID else parent,
            marker=marker,
            label=label,
            citations=extract_latex_citations(text),
            cross_refs=extract_cross_refs(clean_text_value),
            inline_math_count=count_inline_math(text),
            display_math_count=display_math_count,
            metadata=metadata or {},
        )
        self.blocks.append(block)
        return block

    def update_last_block(self, text: str) -> ComparisonBlock:
        previous = self.blocks[-1]
        clean_text_value = " ".join(text.split())
        updated = ComparisonBlock(
            block_id=previous.block_id,
            block_type=previous.block_type,
            order=previous.order,
            text=clean_text_value,
            normalized_text=normalize_for_compare(clean_text_value),
            level=previous.level,
            parent_id=previous.parent_id,
            marker=previous.marker,
            label=previous.label,
            citations=sorted(set([*previous.citations, *extract_latex_citations(text)])),
            cross_refs=[*previous.cross_refs, *extract_cross_refs(clean_text_value)],
            inline_math_count=previous.inline_math_count + count_inline_math(text),
            display_math_count=previous.display_math_count,
            metadata=previous.metadata,
        )
        self.blocks[-1] = updated
        return updated

    def default_text_type(self) -> str:
        ctx = self.current_context()
        if ctx and ctx.get("type") == "abstract":
            return "abstract"
        if ctx and ctx.get("type") == "references":
            return "reference_item"
        if ctx and ctx.get("type") == "algorithm":
            return "algorithm"
        last = self.blocks[-1] if self.blocks else None
        if last and last.block_type == "list_item" and not last.text:
            return "list_item"
        return "paragraph"

    def current_context(self) -> dict[str, str] | None:
        return self.context_stack[-1] if self.context_stack else None

    def current_float_parent(self) -> str | None:
        for ctx in reversed(self.context_stack):
            if ctx.get("type") in {"figure", "table", "algorithm"}:
                return ctx.get("id")
        return None

    def current_parent_id(self) -> str:
        for ctx in reversed(self.context_stack):
            if ctx.get("id"):
                return ctx["id"]
        if self.section_stack:
            return self.section_stack[max(self.section_stack)]
        return ROOT_ID

    def section_parent(self, level: int) -> str:
        lower = [key for key in self.section_stack if key < level]
        if not lower:
            return ROOT_ID
        return self.section_stack[max(lower)]

    def ensure_references_heading(self) -> str:
        for block in self.blocks:
            if block.block_type == "heading" and REFERENCE_HEADING_RE.match(block.text):
                return block.block_id
        block = self.add_block("heading", "References", level=1, parent_id=ROOT_ID, marker="references")
        self.section_stack = {1: block.block_id}
        return block.block_id


class _MarkdownComparisonParser:
    def __init__(self, markdown: str, *, doc_id: str, source_path: Path | None) -> None:
        self.lines = markdown.splitlines()
        self.doc_id = doc_id
        self.source_path = str(source_path) if source_path else None
        self.blocks: list[ComparisonBlock] = []
        self.section_stack: dict[int, str] = {}
        self.next_id = 1
        self.paragraph: list[str] = []
        self.in_references = False
        self.in_abstract = False
        self.active_abstract_id: str | None = None
        self.active_list_id: str | None = None
        self.active_list_marker: str | None = None
        self.markdown_heading_base: int | None = None
        self.seen_document_title = False
        self.has_nougat_abstract_heading = self.detect_nougat_abstract_heading()

    def parse(self) -> ComparisonDocument:
        index = 0
        while index < len(self.lines):
            line = self.lines[index].rstrip()
            stripped = line.strip()
            if not stripped:
                self.flush_paragraph()
                self.active_list_id = None
                self.active_list_marker = None
                index += 1
                continue
            latex_heading = MARKDOWN_LATEX_SECTION_RE.match(line)
            if latex_heading:
                self.flush_paragraph()
                self.active_list_id = None
                self.active_list_marker = None
                name = latex_heading.group("name")
                level = SECTION_LEVELS[name]
                text = latex_text(latex_heading.group("title"))
                self.add_heading(text, level)
                self.in_references = bool(REFERENCE_HEADING_RE.match(text))
                self.in_abstract = False
                self.active_abstract_id = None
                index += 1
                continue
            heading = MARKDOWN_HEADING_RE.match(line)
            if heading:
                self.flush_paragraph()
                self.active_list_id = None
                self.active_list_marker = None
                raw_level = len(heading.group(1))
                text = strip_markdown_inline(heading.group(2))
                if self.should_treat_as_document_title(raw_level, text):
                    self.add_block("document_title", text, parent_id=ROOT_ID, marker="markdown_title")
                    self.seen_document_title = True
                    self.in_references = False
                    self.in_abstract = False
                    self.active_abstract_id = None
                    index += 1
                    continue
                if ABSTRACT_HEADING_RE.match(text):
                    block = self.add_block("abstract", "Abstract", level=1, parent_id=ROOT_ID, marker="markdown_abstract")
                    self.in_abstract = True
                    self.active_abstract_id = block.block_id
                    self.in_references = False
                    index += 1
                    continue
                level = self.normalized_markdown_heading_level(raw_level)
                self.add_heading(text, level)
                self.in_references = bool(REFERENCE_HEADING_RE.match(text))
                self.in_abstract = False
                self.active_abstract_id = None
                index += 1
                continue
            if starts_markdown_display_math(stripped):
                self.flush_paragraph()
                self.active_list_id = None
                self.active_list_marker = None
                math_lines, index = collect_markdown_display_math(self.lines, index)
                self.add_block("display_math", normalize_math_text("\n".join(math_lines)), display_math_count=1)
                continue
            latex_env = MARKDOWN_LATEX_BEGIN_RE.match(stripped)
            if latex_env and latex_env.group("name").strip() in FIGURE_ENVS | TABLE_ENVS | ALGORITHM_ENVS:
                self.flush_paragraph()
                self.active_list_id = None
                self.active_list_marker = None
                index = self.handle_latex_environment(index, latex_env.group("name").strip())
                continue
            if markdown_image_line(stripped):
                self.flush_paragraph()
                self.active_list_id = None
                self.active_list_marker = None
                alt = markdown_image_alt(stripped)
                self.add_block("figure", alt, marker="image")
                index += 1
                continue
            if is_markdown_table_start(self.lines, index):
                self.flush_paragraph()
                self.active_list_id = None
                self.active_list_marker = None
                table_lines = [self.lines[index], self.lines[index + 1]]
                index += 2
                while index < len(self.lines) and "|" in self.lines[index]:
                    table_lines.append(self.lines[index])
                    index += 1
                self.add_block("table", "\n".join(table_lines), marker="markdown_table")
                continue
            list_match = MARKDOWN_LIST_RE.match(line)
            if list_match:
                self.flush_paragraph()
                if self.in_references:
                    marker = list_match.group("marker")
                    reference_text = strip_markdown_inline(list_match.group("text"))
                    ref_item = REFERENCE_ITEM_RE.match(reference_text)
                    if ref_item:
                        reference_text = strip_markdown_inline(ref_item.group("text"))
                        marker = ref_item.group("bracket") or ref_item.group("num") or marker
                    self.add_block("reference_item", reference_text, marker=marker)
                    index += 1
                    continue
                marker = list_match.group("marker")
                list_id = self.ensure_list_container("enumerate" if marker[0].isdigit() else "itemize")
                self.add_block("list_item", strip_markdown_inline(list_match.group("text")), parent_id=list_id, marker=marker)
                index += 1
                continue
            caption = CAPTION_TEXT_RE.match(stripped)
            if caption and len(stripped) > 8:
                self.flush_paragraph()
                self.active_list_id = None
                self.active_list_marker = None
                raw_kind = caption.group("kind").lower()
                if raw_kind.startswith("fig"):
                    kind = "figure"
                elif raw_kind.startswith("alg"):
                    kind = "algorithm"
                else:
                    kind = "table"
                label = caption.group("number")
                self.add_block("caption", strip_markdown_inline(stripped), marker=kind, label=label)
                index += 1
                continue
            ref_item = REFERENCE_ITEM_RE.match(stripped)
            if self.in_references and ref_item:
                self.flush_paragraph()
                self.active_list_id = None
                self.active_list_marker = None
                self.add_block("reference_item", strip_markdown_inline(ref_item.group("text")), marker=ref_item.group("bracket") or ref_item.group("num"))
                index += 1
                continue
            self.paragraph.append(line)
            index += 1
        self.flush_paragraph()
        return ComparisonDocument(
            doc_id=self.doc_id,
            source_format="markdown",
            source_path=self.source_path,
            blocks=self.blocks,
        )

    def flush_paragraph(self) -> None:
        text = strip_markdown_inline(" ".join(line.strip() for line in self.paragraph if line.strip()))
        self.paragraph.clear()
        if not text:
            return
        if REFERENCE_HEADING_RE.match(text):
            self.add_heading(text, 1)
            self.in_references = True
            self.in_abstract = False
            self.active_abstract_id = None
            self.active_list_id = None
            self.active_list_marker = None
            return
        if ABSTRACT_HEADING_RE.match(text):
            block = self.add_block("abstract", "Abstract", level=1, parent_id=ROOT_ID, marker="markdown_abstract")
            self.in_abstract = True
            self.active_abstract_id = block.block_id
            return
        if self.in_abstract:
            self.add_block("abstract", text, parent_id=self.active_abstract_id or ROOT_ID)
            return
        self.add_block("reference_item" if self.in_references else "paragraph", text)

    def detect_nougat_abstract_heading(self) -> bool:
        for line in self.lines[:80]:
            heading = MARKDOWN_HEADING_RE.match(line)
            if not heading:
                continue
            raw_level = len(heading.group(1))
            text = strip_markdown_inline(heading.group(2))
            if raw_level >= 5 and ABSTRACT_HEADING_RE.match(text):
                return True
        return False

    def should_treat_as_document_title(self, raw_level: int, text: str) -> bool:
        if self.seen_document_title or raw_level != 1 or not self.has_nougat_abstract_heading:
            return False
        if self.blocks:
            return False
        return not REFERENCE_HEADING_RE.match(text) and not ABSTRACT_HEADING_RE.match(text)

    def normalized_markdown_heading_level(self, raw_level: int) -> int:
        if self.markdown_heading_base is None:
            self.markdown_heading_base = raw_level
        return max(1, min(5, raw_level - self.markdown_heading_base + 1))

    def add_heading(self, text: str, level: int) -> ComparisonBlock:
        parent = self.section_parent(level)
        block = self.add_block("heading", text, level=level, parent_id=parent)
        self.section_stack = {key: value for key, value in self.section_stack.items() if key < level}
        self.section_stack[level] = block.block_id
        return block

    def ensure_list_container(self, marker: str) -> str:
        if self.active_list_id is not None and self.active_list_marker == marker:
            return self.active_list_id
        block = self.add_block("list", marker, marker=marker)
        self.active_list_id = block.block_id
        self.active_list_marker = marker
        return block.block_id

    def add_block(
        self,
        block_type: str,
        text: str,
        *,
        level: int | None = None,
        parent_id: str | None = None,
        marker: str | None = None,
        label: str | None = None,
        display_math_count: int = 0,
    ) -> ComparisonBlock:
        block_id = f"B_{self.next_id:06d}"
        self.next_id += 1
        clean_text_value = " ".join(str(text or "").split())
        parent = parent_id if parent_id is not None else self.current_parent_id()
        block = ComparisonBlock(
            block_id=block_id,
            block_type=block_type,
            order=len(self.blocks),
            text=clean_text_value,
            normalized_text=normalize_for_compare(clean_text_value),
            level=level,
            parent_id=None if parent == ROOT_ID else parent,
            marker=marker,
            label=label,
            citations=extract_markdown_citations(clean_text_value),
            cross_refs=extract_cross_refs(clean_text_value),
            inline_math_count=count_markdown_inline_math(clean_text_value),
            display_math_count=display_math_count,
        )
        self.blocks.append(block)
        return block

    def handle_latex_environment(self, index: int, env_name: str) -> int:
        env_lines = [self.lines[index]]
        index += 1
        end_pattern = re.compile(MARKDOWN_LATEX_END_TEMPLATE % re.escape(env_name))
        while index < len(self.lines):
            env_lines.append(self.lines[index])
            if end_pattern.search(self.lines[index]):
                index += 1
                break
            index += 1
        raw = "\n".join(env_lines)
        if env_name in FIGURE_ENVS:
            float_block = self.add_block("figure", "", marker=env_name)
        elif env_name in TABLE_ENVS:
            float_block = self.add_block("table", latex_text(raw), marker=env_name)
        else:
            float_block = self.add_block("algorithm", latex_text(raw), marker=env_name)
        caption_match = MARKDOWN_LATEX_CAPTION_RE.search(raw)
        if caption_match:
            caption_text = latex_text(caption_match.group("caption"))
            kind, label = caption_kind_and_label(caption_text, float_block.block_type)
            self.add_block("caption", caption_text, parent_id=float_block.block_id, marker=kind, label=label)
        return index

    def current_parent_id(self) -> str:
        if self.section_stack:
            return self.section_stack[max(self.section_stack)]
        return ROOT_ID

    def section_parent(self, level: int) -> str:
        lower = [key for key in self.section_stack if key < level]
        if not lower:
            return ROOT_ID
        return self.section_stack[max(lower)]


def build_test_items(blocks: list[ComparisonBlock]) -> dict[str, Any]:
    return {
        "document_titles": [block.block_id for block in blocks if block.block_type == "document_title"],
        "author_blocks": [block.block_id for block in blocks if block.block_type == "author_block"],
        "text_blocks": [block.block_id for block in blocks if block.block_type in {"paragraph", "abstract", "list_item"}],
        "headings": [block.block_id for block in blocks if block.block_type == "heading"],
        "paragraphs": [block.block_id for block in blocks if block.block_type == "paragraph"],
        "lists": [block.block_id for block in blocks if block.block_type == "list"],
        "list_items": [block.block_id for block in blocks if block.block_type == "list_item"],
        "figures": [block.block_id for block in blocks if block.block_type == "figure"],
        "tables": [block.block_id for block in blocks if block.block_type == "table"],
        "algorithms": [block.block_id for block in blocks if block.block_type == "algorithm"],
        "captions": [block.block_id for block in blocks if block.block_type == "caption"],
        "references": [block.block_id for block in blocks if block.block_type == "reference_item"],
        "display_math": [block.block_id for block in blocks if block.block_type == "display_math"],
        "citations": sorted({citation for block in blocks for citation in block.citations}),
        "cross_refs": [ref for block in blocks for ref in block.cross_refs],
        "counts": {
            "blocks": len(blocks),
            "document_titles": sum(1 for block in blocks if block.block_type == "document_title"),
            "author_blocks": sum(1 for block in blocks if block.block_type == "author_block"),
            "headings": sum(1 for block in blocks if block.block_type == "heading"),
            "paragraphs": sum(1 for block in blocks if block.block_type == "paragraph"),
            "list_items": sum(1 for block in blocks if block.block_type == "list_item"),
            "figures": sum(1 for block in blocks if block.block_type == "figure"),
            "tables": sum(1 for block in blocks if block.block_type == "table"),
            "algorithms": sum(1 for block in blocks if block.block_type == "algorithm"),
            "captions": sum(1 for block in blocks if block.block_type == "caption"),
            "references": sum(1 for block in blocks if block.block_type == "reference_item"),
            "display_math": sum(block.display_math_count for block in blocks),
            "inline_math": sum(block.inline_math_count for block in blocks),
        },
    }


def document_body(tex: str) -> str:
    match = re.search(r"\\begin\s*\{\s*document\s*\}(?P<body>.*)\\end\s*\{\s*document\s*\}", tex, re.DOTALL)
    return match.group("body") if match else tex


def env_name_from_token(token: str) -> str:
    match = BEGIN_END_RE.match(token)
    return match.group("name").strip() if match else ""


def read_braced(text: str, start: int) -> tuple[str, int]:
    cursor = skip_space(text, start)
    if cursor >= len(text) or text[cursor] != "{":
        return "", start
    depth = 0
    content_start = cursor + 1
    cursor += 1
    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return text[content_start:cursor], cursor + 1
            depth -= 1
        cursor += 1
    return "", start


def skip_optional_args(text: str, start: int) -> int:
    cursor = skip_space(text, start)
    while cursor < len(text) and text[cursor] == "[":
        depth = 0
        cursor += 1
        while cursor < len(text):
            char = text[cursor]
            if char == "\\":
                cursor += 2
                continue
            if char == "[":
                depth += 1
            elif char == "]":
                if depth == 0:
                    cursor += 1
                    break
                depth -= 1
            cursor += 1
        cursor = skip_space(text, cursor)
    return cursor


def skip_space(text: str, start: int) -> int:
    cursor = start
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def latex_text(value: str) -> str:
    text = str(value or "")
    text = SILENT_LATEX_ARG_COMMAND_RE.sub(" ", text)
    text = CITATION_RE.sub(lambda match: " [" + ",".join(key.strip() for key in match.group("keys").split(",")) + "] ", text)
    text = re.sub(r"\\(?:ref|autoref|cref|Cref|eqref)\s*\{([^}]+)\}", r" \1 ", text)
    text = DISPLAY_MATH_DELIM_RE.sub(" [DISPLAY_MATH] ", text)
    text = INLINE_MATH_RE.sub(" [MATH] ", text)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])?\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])?", " ", text)
    text = re.sub(r"\\([#$%&_{}])", r"\1", text)
    replacements = {
        "~": " ",
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\#": "#",
        "---": "-",
        "--": "-",
        "``": '"',
        "''": '"',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return " ".join(text.split())


def normalize_math_text(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\\begin\s*\{[^}]+\}(.*)\\end\s*\{[^}]+\}$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^\\\[(.*)\\\]$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^\$\$(.*)\$\$$", r"\1", text, flags=re.DOTALL)
    return " ".join(text.split())


def split_paragraphs(text: str) -> list[str]:
    return [" ".join(part.split()) for part in re.split(r"(?:\r?\n\s*){2,}|\\par\b", text) if part.strip()]


def normalize_for_compare(text: str) -> str:
    value = str(text or "").casefold()
    value = re.sub(r"\[(?:display_)?math\]", " math ", value)
    value = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", value)
    return " ".join(value.split())


def extract_latex_citations(text: str) -> list[str]:
    keys: list[str] = []
    for match in CITATION_RE.finditer(str(text or "")):
        keys.extend(key.strip() for key in match.group("keys").split(",") if key.strip())
    for match in MARKDOWN_CITATION_RE.finditer(str(text or "")):
        raw = match.group("key")
        keys.extend(part.strip() for part in raw.split(",") if part.strip())
    return sorted({key for key in keys if key.casefold() not in {"math", "display_math"}})


def extract_markdown_citations(text: str) -> list[str]:
    keys: list[str] = []
    for match in MARKDOWN_CITATION_RE.finditer(str(text or "")):
        raw = match.group("key")
        keys.extend(part.strip() for part in raw.split(",") if part.strip())
    return sorted({key for key in keys if key.casefold() not in {"math", "display_math"}})


def extract_cross_refs(text: str) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for match in CROSS_REF_RE.finditer(str(text or "")):
        kind = match.group("kind").lower().rstrip(".")
        if kind == "fig":
            kind = "figure"
        if kind == "tab":
            kind = "table"
        if kind == "eq":
            kind = "equation"
        if kind == "alg":
            kind = "algorithm"
        refs.append({"kind": kind, "label": match.group("label")})
    return refs


def count_inline_math(text: str) -> int:
    return len(INLINE_MATH_RE.findall(str(text or "")))


def count_markdown_inline_math(text: str) -> int:
    return len(re.findall(r"(?<!\\)\$[^$]+(?<!\\)\$", str(text or "")))


def caption_kind_and_label(text: str, parent: str | None) -> tuple[str | None, str | None]:
    match = CAPTION_TEXT_RE.match(text)
    if match:
        raw_kind = match.group("kind").lower()
        if raw_kind.startswith("fig"):
            kind = "figure"
        elif raw_kind.startswith("alg"):
            kind = "algorithm"
        else:
            kind = "table"
        return kind, match.group("number")
    if parent in {"figure", "table", "algorithm"}:
        return parent, None
    return None, parent


def strip_markdown_inline(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", value)
    return " ".join(value.split())


def markdown_image_line(text: str) -> bool:
    return bool(re.match(r"!\[[^\]]*\]\([^)]+\)", text))


def markdown_image_alt(text: str) -> str:
    match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", text)
    return match.group(1) if match else ""


def is_markdown_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index] and bool(re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[index + 1]))


def starts_markdown_display_math(line: str) -> bool:
    stripped = str(line or "").strip()
    if stripped.startswith("$$") or stripped.startswith(r"\["):
        return True
    begin = MARKDOWN_LATEX_BEGIN_RE.match(stripped)
    return bool(begin and begin.group("name").strip() in DISPLAY_MATH_ENVS)


def collect_markdown_display_math(lines: list[str], index: int) -> tuple[list[str], int]:
    first = lines[index].strip()
    if first.startswith("$$"):
        if first.count("$$") >= 2 and first != "$$":
            return [first], index + 1
        return collect_until_token(lines, index, "$$")
    if first.startswith(r"\["):
        if r"\]" in first and first != r"\[":
            return [first], index + 1
        return collect_until_token(lines, index, r"\]")
    begin = MARKDOWN_LATEX_BEGIN_RE.match(first)
    env_name = begin.group("name").strip() if begin else ""
    if not env_name:
        return [first], index + 1
    end_pattern = re.compile(MARKDOWN_LATEX_END_TEMPLATE % re.escape(env_name))
    collected = [lines[index].strip()]
    index += 1
    while index < len(lines):
        collected.append(lines[index].strip())
        if end_pattern.search(lines[index]):
            index += 1
            break
        index += 1
    return collected, index


def collect_until_token(lines: list[str], index: int, token: str) -> tuple[list[str], int]:
    collected = [lines[index].strip()]
    index += 1
    while index < len(lines):
        collected.append(lines[index].strip())
        if token in lines[index].strip():
            index += 1
            break
        index += 1
    return collected, index


def write_comparison_json(document: ComparisonDocument, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert LaTeX or Markdown into the shared comparison structure.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=["latex", "markdown", "auto"], default="auto")
    parser.add_argument("--doc-id")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    fmt = args.format
    if fmt == "auto":
        suffix = args.input.suffix.lower()
        fmt = "latex" if suffix in {".tex", ".ltx"} else "markdown"
    if fmt == "latex":
        document = latex_file_to_comparison(args.input, doc_id=args.doc_id)
    else:
        document = markdown_file_to_comparison(args.input, doc_id=args.doc_id)
    write_comparison_json(document, args.output)
    print(f"wrote {args.output} blocks={len(document.blocks)} format={fmt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
