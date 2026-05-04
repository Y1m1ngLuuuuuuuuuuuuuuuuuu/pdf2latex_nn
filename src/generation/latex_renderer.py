"""Render resolved document trees back into compilable LaTeX."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


DEFAULT_PACKAGES = ["graphicx", "amsmath", "amssymb", "booktabs", "hyperref"]
SECTION_COMMANDS = ["section", "subsection", "subsubsection", "paragraph", "subparagraph"]


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
        command = SECTION_COMMANDS[min(depth, len(SECTION_COMMANDS) - 1)]
        body = [rf"\{command}{{{escape_latex(text)}}}"]
        body.extend(render_child_block(child, depth=depth + 1) for child in children)
        return "\n\n".join(part for part in body if part)
    if block_type == "equation":
        return "\\[\n" + text.strip() + "\n\\]"
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

    paragraph = escape_latex(text) if text else ""
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
        child_text = node_text(child)
        nested = [render_node(grandchild, depth=depth + 1) for grandchild in getattr(child, "children", [])]
        item_body = escape_latex(child_text) if child_text else ""
        if nested:
            item_body = (item_body + "\n" + "\n".join(part for part in nested if part)).strip()
        lines.append(rf"\item {item_body}".rstrip())
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


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
    return "\\begin{verbatim}\n" + text.strip() + "\n\\end{verbatim}"


def node_text(node: Any) -> str:
    if hasattr(node, "text"):
        return str(node.text).strip()
    if isinstance(node, dict):
        return str(node.get("text") or node.get("text_for_embedding") or node.get("text_preview") or "").strip()
    return ""


def canonical_render_type(record: dict[str, Any]) -> str:
    raw = str(record.get("canonical_type") or record.get("type") or record.get("block_type") or "").lower()
    if raw in {"paragraph", "text", "paragraph_text"}:
        return "text"
    if raw in {"title", "section", "subsection", "subsubsection"}:
        return "title"
    if raw in {"equation", "equation_interline", "display_formula"}:
        return "equation"
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
    return "".join(replacements.get(char, char) for char in text)
