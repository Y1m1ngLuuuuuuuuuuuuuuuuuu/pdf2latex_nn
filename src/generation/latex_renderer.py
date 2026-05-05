"""Render resolved document trees back into compilable LaTeX."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


DEFAULT_PACKAGES = ["graphicx", "amsmath", "amssymb", "booktabs", "hyperref"]
SECTION_COMMANDS = ["section", "subsection", "subsubsection", "paragraph", "subparagraph"]
DISPLAY_MATH_ENVS = {"equation", "align", "gather", "eqnarray", "flalign", "multline"}
DEFAULT_PREAMBLE_COMMANDS = [r"\providecommand{\mathbfcal}[1]{\mathbf{\mathcal{#1}}}"]


@dataclass
class RenderConfig:
    document_class: str = "article"
    packages: list[str] = field(default_factory=lambda: list(DEFAULT_PACKAGES))
    title: str | None = None


def render_latex_document(root: Any, config: RenderConfig | None = None) -> str:
    cfg = config or RenderConfig()
    lines = [rf"\documentclass{{{cfg.document_class}}}"]
    for package in cfg.packages:
        lines.append(rf"\usepackage{{{package}}}")
    lines.extend(DEFAULT_PREAMBLE_COMMANDS)
    lines.append("")
    if cfg.title:
        lines.extend([rf"\title{{{escape_latex(cfg.title)}}}", r"\date{}", ""])
    lines.append(r"\begin{document}")
    if cfg.title:
        lines.append(r"\maketitle")
        lines.append("")
    for child in getattr(root, "children", []):
        rendered = render_node(child, depth=0)
        if rendered:
            lines.append(rendered)
            lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines).rstrip() + "\n"


def render_node(node: Any, *, depth: int = 0) -> str:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    block_type = canonical_render_type(record)
    text = node_text(node)
    children = list(getattr(node, "children", record.get("children", [])))

    if block_type == "title":
        body = [render_title(text, depth=depth)] if text else []
        body.extend(render_child_block(child, depth=depth + 1) for child in children)
        return "\n\n".join(part for part in body if part)
    if block_type == "equation":
        return render_equation(text)
    if block_type == "inline_math":
        return render_inline_math(text)
    if block_type == "table":
        return render_verbatim_like(text, "table")
    if block_type == "figure":
        caption = escape_latex(text) if text else "Figure"
        return "\\begin{figure}[htbp]\n\\centering\n% image placeholder\n" + rf"\caption{{{caption}}}" + "\n\\end{figure}"
    if block_type in {"algorithm", "code"}:
        return render_verbatim_like(text, block_type)
    if block_type == "reference":
        return render_references(record, text)
    if block_type == "list":
        return render_list_node(node, depth=depth)

    paragraph = render_textual_content(record, text) if text else ""
    rendered_children = [render_child_block(child, depth=depth + 1) for child in children]
    parts = [paragraph] if paragraph else []
    parts.extend(part for part in rendered_children if part)
    return "\n\n".join(parts)


def render_child_block(child: Any, *, depth: int) -> str:
    rendered = render_node(child, depth=depth)
    return rendered.strip()


def render_list_node(node: Any, *, depth: int) -> str:
    children = list(getattr(node, "children", []))
    if not children:
        text = node_text(node)
        return "\\begin{itemize}\n" + rf"\item {escape_latex(text)}" + "\n\\end{itemize}"
    lines = [r"\begin{itemize}"]
    for child in children:
        item_body = render_list_item(child, depth=depth + 1)
        lines.append(rf"\item {item_body}".rstrip())
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def render_list_item(node: Any, *, depth: int) -> str:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    block_type = canonical_render_type(record)
    text = node_text(node)
    if block_type == "equation":
        item_body = render_equation(text)
    elif block_type == "inline_math":
        item_body = render_inline_math(text)
    else:
        item_body = render_textual_content(record, text) if text else ""
    nested = [render_node(grandchild, depth=depth + 1) for grandchild in getattr(node, "children", [])]
    if nested:
        item_body = (item_body + "\n" + "\n".join(part for part in nested if part)).strip()
    return item_body


def render_references(record: dict[str, Any], fallback_text: str) -> str:
    references = record.get("reference_items")
    if isinstance(references, list) and references:
        items = [str(item.get("text") if isinstance(item, dict) else item).strip() for item in references]
    else:
        items = [line.strip() for line in re.split(r"\n+|\s{2,}", fallback_text) if line.strip()]
    if not items:
        return ""
    lines = [r"\begin{thebibliography}{99}"]
    for idx, item in enumerate(items, start=1):
        lines.append(rf"\bibitem{{ref{idx}}} {escape_latex(item)}")
    lines.append(r"\end{thebibliography}")
    return "\n".join(lines)


def render_verbatim_like(text: str, label: str) -> str:
    if not text:
        return f"% empty {label} block"
    return "\\begin{verbatim}\n" + safe_verbatim_text(text.strip()) + "\n\\end{verbatim}"


def render_title(text: str, *, depth: int) -> str:
    command = SECTION_COMMANDS[min(depth, len(SECTION_COMMANDS) - 1)]
    return rf"\{command}{{{escape_latex(text)}}}"


def render_equation(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return "\\[\n\n\\]"
    if stripped.startswith("\\[") or stripped.startswith("$$"):
        return stripped
    begin_match = re.match(r"\\begin\{([^}]+)\}", stripped)
    if begin_match and begin_match.group(1).rstrip("*") in DISPLAY_MATH_ENVS:
        return stripped
    return "\\[\n" + stripped + "\n\\]"


def render_inline_math(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return "$$"
    if stripped.startswith("$") or stripped.startswith(r"\("):
        return stripped
    return "$" + stripped + "$"


def render_textual_content(record: dict[str, Any], fallback_text: str) -> str:
    segments = extract_content_segments(record)
    if not segments:
        return escape_latex(fallback_text)
    rendered: list[str] = []
    for segment in segments:
        segment_type = str(segment.get("type") or "").lower()
        content = str(segment.get("content") or segment.get("text") or "")
        if not content:
            continue
        if segment_type in {"equation_inline", "inline_equation", "inline_math", "inline_formula"}:
            rendered.append(render_inline_math(content))
        elif segment_type in {"equation_interline", "interline_equation", "display_formula", "formula", "equation"}:
            rendered.append("\n\n" + render_equation(content) + "\n\n")
        else:
            rendered.append(escape_latex(content))
    return normalize_latex_text("".join(rendered))


def extract_content_segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    block = record.get("block")
    if not isinstance(block, dict):
        return []
    content = block.get("content")
    if isinstance(content, dict):
        for key in ("paragraph_content", "title_content", "content"):
            value = content.get(key)
            if isinstance(value, list):
                return [segment for segment in value if isinstance(segment, dict)]
    if isinstance(content, list):
        return [segment for segment in content if isinstance(segment, dict)]
    return []


def normalize_latex_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def node_text(node: Any) -> str:
    if hasattr(node, "text"):
        return str(node.text).strip()
    if isinstance(node, dict):
        return str(node.get("text") or node.get("text_for_embedding") or node.get("text_preview") or "").strip()
    return ""


def canonical_render_type(record: dict[str, Any]) -> str:
    raw = str(record.get("canonical_type") or record.get("type") or record.get("raw_type") or record.get("block_type") or "").lower()
    if raw in {"paragraph", "text", "paragraph_text"}:
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
    if raw in {"algorithm"}:
        return "algorithm"
    if raw in {"list", "item", "itemize", "enumerate"}:
        return "list"
    if raw in {"code"}:
        return "code"
    if raw in {"reference", "references", "bibliography"}:
        return "reference"
    return "text"


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(_escape_latex_char(char, replacements) for char in str(text))


def _escape_latex_char(char: str, replacements: dict[str, str]) -> str:
    if char in UNICODE_LATEX_REPLACEMENTS:
        return UNICODE_LATEX_REPLACEMENTS[char]
    if char in replacements:
        return replacements[char]
    if ord(char) < 128:
        return char
    ascii_fallback = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode("ascii")
    if ascii_fallback:
        return "".join(replacements.get(fallback_char, fallback_char) for fallback_char in ascii_fallback)
    return "?"


def safe_verbatim_text(text: str) -> str:
    return "".join(_safe_verbatim_char(char) for char in str(text))


def _safe_verbatim_char(char: str) -> str:
    if ord(char) < 128:
        return char
    if char in UNICODE_LATEX_REPLACEMENTS:
        return UNICODE_LATEX_REPLACEMENTS[char]
    ascii_fallback = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode("ascii")
    return ascii_fallback or "?"


UNICODE_LATEX_REPLACEMENTS = {
    "α": r"\ensuremath{\alpha}",
    "β": r"\ensuremath{\beta}",
    "γ": r"\ensuremath{\gamma}",
    "δ": r"\ensuremath{\delta}",
    "ϵ": r"\ensuremath{\epsilon}",
    "ε": r"\ensuremath{\epsilon}",
    "θ": r"\ensuremath{\theta}",
    "λ": r"\ensuremath{\lambda}",
    "μ": r"\ensuremath{\mu}",
    "π": r"\ensuremath{\pi}",
    "σ": r"\ensuremath{\sigma}",
    "φ": r"\ensuremath{\phi}",
    "ω": r"\ensuremath{\omega}",
    "Δ": r"\ensuremath{\Delta}",
    "Σ": r"\ensuremath{\Sigma}",
    "Ω": r"\ensuremath{\Omega}",
    "≤": r"\ensuremath{\leq}",
    "≥": r"\ensuremath{\geq}",
    "≠": r"\ensuremath{\neq}",
    "≈": r"\ensuremath{\approx}",
    "±": r"\ensuremath{\pm}",
    "×": r"\ensuremath{\times}",
    "∞": r"\ensuremath{\infty}",
    "∈": r"\ensuremath{\in}",
    "∑": r"\ensuremath{\sum}",
    "∫": r"\ensuremath{\int}",
    "→": r"\ensuremath{\rightarrow}",
    "←": r"\ensuremath{\leftarrow}",
    "–": "--",
    "—": "---",
    "−": r"\ensuremath{-}",
    "•": r"\textbullet{}",
    "“": "``",
    "”": "''",
    "‘": "`",
    "’": "'",
}
