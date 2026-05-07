"""Render resolved document trees back into compilable LaTeX."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.perception.xy_cut import sort_nodes_by_reading_order
from src.perception.title_features import strip_title_numbering, title_numbering_level


DEFAULT_PACKAGES = ["graphicx", "amsmath", "amssymb", "booktabs", "hyperref", "float", "algorithm", "algpseudocode"]
SECTION_COMMANDS = ["section", "subsection", "subsubsection", "paragraph", "subparagraph"]
DISPLAY_MATH_ENVS = {"equation", "align", "gather", "eqnarray", "flalign", "multline"}
DEFAULT_PREAMBLE_COMMANDS = [r"\providecommand{\mathbfcal}[1]{\mathbf{\mathcal{#1}}}"]
INLINE_MATH_COMMANDS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Phi",
    "Psi",
    "Omega",
    "mathrm",
    "mathbf",
    "mathit",
    "mathsf",
    "mathtt",
    "mathcal",
    "mathbfcal",
    "operatorname",
    "frac",
    "dfrac",
    "tfrac",
    "sqrt",
    "left",
    "right",
    "leq",
    "geq",
    "neq",
    "approx",
    "sim",
    "simeq",
    "times",
    "cdot",
    "pm",
    "mp",
    "prime",
    "partial",
    "nabla",
    "sum",
    "prod",
    "int",
    "in",
    "notin",
}
LIST_MARKER_RE = re.compile(r"^\s*(?P<marker>[\u2022\u25E6\u25CB\u25AA\-\*]|\d+\.|[a-zA-Z]\.)\s+")
ORDERED_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\.|[a-zA-Z]\.)\s+")
NUMERIC_ID_RE = re.compile(r"\d+")
PSEUDOCODE_START_RE = re.compile(
    r"^\s*(?:Algorithm\s*\d+\b|Input\s*:|Output\s*:|Require\s*:|Ensure\s*:)",
    re.IGNORECASE,
)
PSEUDOCODE_BREAK_RE = re.compile(
    r"\s+(?=(?:Input|Output|Require|Ensure)\s*:|Algorithm\s*\d+\b|(?:for|while|if|else|elif|return|end)\b)",
    re.IGNORECASE,
)
VERBATIM_END_RE = re.compile(r"\\end\s*\{\s*verbatim\s*\}", re.IGNORECASE)
ALGORITHM_CAPTION_RE = re.compile(r"^\s*Algorithm\s*(?:\d+)?\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
PSEUDOCODE_IO_RE = re.compile(r"^\s*(Input|Require|Output|Ensure)\s*:\s*(.*)$", re.IGNORECASE)
PSEUDOCODE_FOR_RE = re.compile(r"^\s*for\s+(.+?)(?:\s+do)?\s*$", re.IGNORECASE)
PSEUDOCODE_WHILE_RE = re.compile(r"^\s*while\s+(.+?)(?:\s+do)?\s*$", re.IGNORECASE)
PSEUDOCODE_IF_RE = re.compile(r"^\s*if\s+(.+?)(?:\s+then)?\s*$", re.IGNORECASE)
PSEUDOCODE_RETURN_RE = re.compile(r"^\s*return\s+(.+)$", re.IGNORECASE)
PSEUDOCODE_END_RE = re.compile(r"^\s*end(?:\s+(for|if|while))?\s*$", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^\s*(Table\s*\d*[:.\-]?\s*[^\n]+)", re.IGNORECASE)
LATEX_MATH_MARKER_RE = re.compile(r"(\\[A-Za-z]+|[_^{}]|[<>=+\-*/]|\\\(|\\\[)")
MATH_COMMAND_RE = re.compile(r"\\([A-Za-z]+)\*?")


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
    for rendered in render_child_blocks_with_dynamic_lists(getattr(root, "children", []), depth=0):
        if rendered:
            lines.append(rendered)
            lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines).rstrip() + "\n"


def render_node(node: Any, *, depth: int = 0) -> str:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    block_type = canonical_render_type(record)
    text = node_text(node)
    children = sorted_render_children(getattr(node, "children", record.get("children", [])))

    if is_algorithm_like_node(record, text):
        return render_algorithm_block(text)
    if block_type == "title":
        body = [render_title(text, depth=depth)] if text else []
        body.extend(render_child_blocks_with_dynamic_lists(children, depth=depth + 1))
        return "\n\n".join(part for part in body if part)
    if block_type == "equation":
        body = [render_equation(text)]
        body.extend(render_child_blocks_with_dynamic_lists(children, depth=depth + 1))
        return "\n\n".join(part for part in body if part)
    if block_type == "inline_math":
        body = [render_inline_math(text)]
        body.extend(render_child_blocks_with_dynamic_lists(children, depth=depth + 1))
        return "\n\n".join(part for part in body if part)
    if block_type == "table":
        return render_table_placeholder(record, text)
    if block_type == "figure":
        caption = render_text_with_inline_latex(text) if text else "Figure"
        return "\\begin{figure}[htbp]\n\\centering\n% image placeholder\n" + rf"\caption{{{caption}}}" + "\n\\end{figure}"
    if block_type == "reference":
        return render_references(record, text)
    if block_type == "list":
        return render_list_node(node, depth=depth)

    paragraph = render_textual_content(record, text) if text else ""
    rendered_children = render_child_blocks_with_dynamic_lists(children, depth=depth + 1)
    parts = [paragraph] if paragraph else []
    parts.extend(part for part in rendered_children if part)
    return "\n\n".join(parts)


def render_child_blocks_with_dynamic_lists(children: Any, *, depth: int) -> list[str]:
    child_list = sorted_render_children(children)
    rendered: list[str] = []
    index = 0
    while index < len(child_list):
        child = child_list[index]
        if canonical_render_type(node_record(child)) == "reference":
            run: list[Any] = []
            while index < len(child_list) and canonical_render_type(node_record(child_list[index])) == "reference":
                run.append(child_list[index])
                index += 1
            rendered.append(render_reference_run(run))
            continue
        list_environment = list_environment_for_node(child)
        if list_environment is not None:
            run: list[Any] = []
            while index < len(child_list) and list_environment_for_node(child_list[index]) is not None:
                run.append(child_list[index])
                index += 1
            rendered.append(render_dynamic_list_group(run, environment=list_environment, depth=depth))
            continue
        block = render_child_block(child, depth=depth)
        if block:
            rendered.append(block)
        index += 1
    return rendered


def render_child_block(child: Any, *, depth: int) -> str:
    rendered = render_node(child, depth=depth)
    return rendered.strip()


def is_bullet_list_candidate(node: Any) -> bool:
    return list_environment_for_node(node) is not None


def list_environment_for_node(node: Any) -> str | None:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    block_type = canonical_render_type(record)
    text = node_text(node)
    children = getattr(node, "children", record.get("children", []))
    if block_type == "list" and not children:
        return list_environment_for_record(record, fallback_text=text)
    if block_type != "text":
        return None
    return list_environment_for_text(text)


def list_environment_for_record(record: dict[str, Any], *, fallback_text: str = "") -> str:
    explicit = str(record.get("list_type") or record.get("list_style") or record.get("enum_type") or "").casefold()
    if explicit in {"ordered", "enumerate", "numbered", "number", "alpha", "roman"}:
        return "enumerate"
    text = fallback_text or node_text(record)
    return list_environment_for_text(text) or "itemize"


def list_environment_for_text(text: str) -> str | None:
    value = str(text or "")
    if not LIST_MARKER_RE.match(value):
        return None
    return "enumerate" if ORDERED_LIST_MARKER_RE.match(value) else "itemize"


def render_dynamic_list_group(items: list[Any], *, environment: str, depth: int) -> str:
    lines = [rf"\begin{{{environment}}}"]
    for item in items:
        item_body = render_textual_node_without_list_marker(item) if node_text(item) else ""
        nested = render_child_blocks_with_dynamic_lists(getattr(item, "children", []), depth=depth + 1)
        if nested:
            item_body = (item_body + "\n" + "\n".join(part for part in nested if part)).strip()
        lines.append(rf"\item {item_body}".rstrip())
    lines.append(rf"\end{{{environment}}}")
    return "\n".join(lines)


def render_dynamic_itemize(items: list[Any], *, depth: int) -> str:
    return render_dynamic_list_group(items, environment="itemize", depth=depth)


def strip_list_marker(text: str) -> str:
    return LIST_MARKER_RE.sub("", str(text or ""), count=1).strip()


def render_list_node(node: Any, *, depth: int) -> str:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    children = sorted_render_children(getattr(node, "children", record.get("children", [])))
    environment = list_environment_for_record(record, fallback_text=node_text(node))
    if not children:
        text = node_text(node)
        item_body = render_textual_node_without_list_marker(node) if text else ""
        return rf"\begin{{{environment}}}" + "\n" + rf"\item {item_body}".rstrip() + "\n" + rf"\end{{{environment}}}"
    first_child_environment = list_environment_for_node(children[0])
    if first_child_environment is not None:
        environment = first_child_environment
    lines = [rf"\begin{{{environment}}}"]
    for child in children:
        item_body = render_list_item(child, depth=depth + 1)
        lines.append(rf"\item {item_body}".rstrip())
    lines.append(rf"\end{{{environment}}}")
    return "\n".join(lines)


def render_list_item(node: Any, *, depth: int) -> str:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    block_type = canonical_render_type(record)
    text = node_text(node)
    if is_algorithm_like_node(record, text):
        item_body = render_algorithm_block(text)
    elif block_type == "equation":
        item_body = render_equation(text)
    elif block_type == "inline_math":
        item_body = render_inline_math(text)
    else:
        item_body = render_textual_node_without_list_marker(node) if text else ""
    nested = [render_node(grandchild, depth=depth + 1) for grandchild in sorted_render_children(getattr(node, "children", []))]
    if nested:
        item_body = (item_body + "\n" + "\n".join(part for part in nested if part)).strip()
    return item_body


def render_references(record: dict[str, Any], fallback_text: str) -> str:
    items = collect_reference_items(record)
    if not items:
        items = [line.strip() for line in re.split(r"\n+|\s{2,}", fallback_text) if line.strip()]
    if not items:
        return ""
    lines = [r"\begin{thebibliography}{99}"]
    for idx, item in enumerate(items, start=1):
        lines.append(rf"\bibitem{{ref{idx}}} {escape_latex(item)}")
    lines.append(r"\end{thebibliography}")
    return "\n".join(lines)


def render_reference_run(nodes: list[Any]) -> str:
    if not nodes:
        return ""
    primary = dict(node_record(nodes[0]))
    merged_records = list(primary.get("merged_records") or [])
    for node in nodes[1:]:
        record = node_record(node)
        merged_records.append(record)
        merged_records.extend(record.get("merged_records") or [])
    primary["merged_records"] = merged_records
    fallback = "\n".join(node_text(node) for node in nodes if node_text(node))
    return render_references(primary, fallback)


def collect_reference_items(record: dict[str, Any]) -> list[str]:
    items = normalize_reference_items(record.get("reference_items"))
    for merged_record in record.get("merged_records", []):
        if isinstance(merged_record, dict):
            items.extend(normalize_reference_items(merged_record.get("reference_items")))
    return [item for item in items if item]


def normalize_reference_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        text = item.get("text") if isinstance(item, dict) else item
        text = str(text or "").strip()
        if text:
            items.append(text)
    return items


def render_verbatim_like(text: str, label: str) -> str:
    if not text:
        return f"% empty {label} block"
    return "\\begin{verbatim}\n" + safe_verbatim_text(text.strip()) + "\n\\end{verbatim}"


def is_algorithm_like_node(record: dict[str, Any], text: str) -> bool:
    if canonical_render_type(record) in {"algorithm", "code"}:
        return True
    return bool(PSEUDOCODE_START_RE.match(str(text or "")))


def render_algorithm_block(text: str) -> str:
    caption, commands = parse_pseudo_code(text)
    lines = [r"\begin{algorithm}[H]"]
    if caption:
        lines.append(rf"\caption{{{escape_latex(caption)}}}")
    lines.append(r"\begin{algorithmic}[1]")
    lines.extend(commands or [r"\State " + format_algorithmic_text(text)])
    lines.append(r"\end{algorithmic}")
    lines.append(r"\end{algorithm}")
    return "\n".join(lines)


def parse_pseudo_code(text: str) -> tuple[str | None, list[str]]:
    body = restore_algorithm_line_breaks(text)
    raw_lines = [line.strip() for line in body.split("\n") if line.strip()]
    caption: str | None = None
    commands: list[str] = []
    block_stack: list[str] = []

    for raw_line in raw_lines:
        line = strip_pseudocode_line_number(raw_line)
        caption_match = ALGORITHM_CAPTION_RE.match(line)
        if caption_match and caption is None:
            caption = caption_match.group(1).strip() or "Algorithm"
            continue

        io_match = PSEUDOCODE_IO_RE.match(line)
        if io_match:
            kind, content = io_match.group(1).casefold(), io_match.group(2).strip()
            command = r"\Require" if kind in {"input", "require"} else r"\Ensure"
            commands.append(rf"{command} {format_algorithmic_text(content)}")
            continue

        end_match = PSEUDOCODE_END_RE.match(line)
        if end_match:
            close_kind = end_match.group(1)
            commands.append(close_algorithmic_block(block_stack, close_kind))
            continue

        for_match = PSEUDOCODE_FOR_RE.match(line)
        if for_match:
            commands.append(rf"\For{{{format_algorithmic_text(for_match.group(1).strip())}}}")
            block_stack.append("for")
            continue

        while_match = PSEUDOCODE_WHILE_RE.match(line)
        if while_match:
            commands.append(rf"\While{{{format_algorithmic_text(while_match.group(1).strip())}}}")
            block_stack.append("while")
            continue

        if_match = PSEUDOCODE_IF_RE.match(line)
        if if_match:
            commands.append(rf"\If{{{format_algorithmic_text(if_match.group(1).strip())}}}")
            block_stack.append("if")
            continue

        return_match = PSEUDOCODE_RETURN_RE.match(line)
        if return_match:
            commands.append(rf"\State \Return {format_algorithmic_text(return_match.group(1).strip())}")
            continue

        commands.append(rf"\State {format_algorithmic_text(line)}")

    while block_stack:
        commands.append(close_algorithmic_block(block_stack, None))
    return caption, commands


def strip_pseudocode_line_number(line: str) -> str:
    return re.sub(r"^\s*\d+\s*[:.)]\s*", "", line).strip()


def close_algorithmic_block(block_stack: list[str], close_kind: str | None) -> str:
    normalized = str(close_kind or "").casefold()
    if normalized in block_stack:
        block_stack.pop(len(block_stack) - 1 - block_stack[::-1].index(normalized))
        kind = normalized
    elif block_stack:
        kind = block_stack.pop()
    else:
        kind = normalized or "for"
    if kind == "if":
        return r"\EndIf"
    if kind == "while":
        return r"\EndWhile"
    return r"\EndFor"


def format_algorithmic_text(text: str) -> str:
    prepared = normalize_algorithm_math_text(text)
    if not prepared:
        return ""
    if LATEX_MATH_MARKER_RE.search(prepared):
        return r"\(\displaystyle " + escape_algorithm_math_text(prepared) + r"\)"
    return escape_latex(prepared)


def normalize_algorithm_math_text(text: str) -> str:
    normalized = "".join(ALGORITHM_MATH_UNICODE_REPLACEMENTS.get(char, char) for char in str(text or ""))
    normalized = normalized.replace("<-", r"\gets").replace("->", r"\to")
    return " ".join(normalized.split())


def escape_algorithm_math_text(text: str) -> str:
    return (
        str(text)
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def restore_algorithm_line_breaks(text: str) -> str:
    body = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" in body:
        return body
    body = PSEUDOCODE_BREAK_RE.sub("\n", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def sanitize_verbatim_body(text: str) -> str:
    sanitized = VERBATIM_END_RE.sub(r"\\end {verbatim}", str(text or ""))
    return "".join(_safe_code_verbatim_char(char) for char in sanitized)


def render_table_placeholder(record: dict[str, Any], text: str) -> str:
    table_id = table_node_identifier(record)
    bbox = format_table_bbox(record.get("bbox"))
    caption = extract_table_caption(text) or "Table reconstruction placeholder"
    todo = f"% [TODO_TABLE_RECONSTRUCT: BBOX={bbox}, ID={table_id}]"
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            todo,
            rf"\caption{{{escape_latex(caption)}}}",
            r"\end{table}",
        ]
    )


def table_node_identifier(record: dict[str, Any]) -> str:
    for key in ("id", "block_id", "table_id"):
        value = record.get(key)
        if value:
            return str(value)
    value = record.get("global_order")
    if value is not None:
        return f"table_{value}"
    return "table_unknown"


def format_table_bbox(value: Any) -> str:
    if not isinstance(value, list) or len(value) < 4:
        return "UNKNOWN"
    try:
        coords = [float(coord) for coord in value[:4]]
    except (TypeError, ValueError):
        return "UNKNOWN"
    return "(" + ", ".join(format_bbox_number(coord) for coord in coords) + ")"


def format_bbox_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def extract_table_caption(text: str) -> str | None:
    match = TABLE_CAPTION_RE.search(str(text or ""))
    if not match:
        return None
    return " ".join(match.group(1).split())


def _safe_code_verbatim_char(char: str) -> str:
    if ord(char) < 128:
        return char
    if char in CODE_UNICODE_REPLACEMENTS:
        return CODE_UNICODE_REPLACEMENTS[char]
    ascii_fallback = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode("ascii")
    return ascii_fallback or "?"


def render_title(text: str, *, depth: int) -> str:
    command = title_command(text, depth=depth)
    title_text = strip_title_numbering(text)
    return rf"\{command}{{{escape_latex(title_text)}}}"


def title_command(text: str, *, depth: int) -> str:
    numbered_level = title_numbering_level(text)
    if numbered_level is not None:
        return SECTION_COMMANDS[min(numbered_level - 1, len(SECTION_COMMANDS) - 1)]
    return SECTION_COMMANDS[min(max(0, depth), len(SECTION_COMMANDS) - 1)]


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
        return render_text_with_inline_latex(fallback_text)
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
            rendered.append(render_text_with_inline_latex(content, strip=False))
    return normalize_latex_text("".join(rendered))


def render_textual_node_without_list_marker(node: Any) -> str:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    records = [record] + [item for item in record.get("merged_records", []) if isinstance(item, dict)]
    rendered_parts: list[str] = []
    used_structured_content = False
    marker_stripped = False
    for current_record in records:
        prepared_record = strip_list_marker_from_record(current_record) if not marker_stripped else current_record
        if prepared_record is not current_record:
            marker_stripped = True
        rendered = render_textual_content(prepared_record, node_text(prepared_record))
        if extract_content_segments(prepared_record):
            used_structured_content = True
        if rendered:
            rendered_parts.append(rendered)
    if rendered_parts:
        if used_structured_content:
            return merge_latex_fragments(rendered_parts)
        return normalize_latex_text(" ".join(rendered_parts))
    return escape_latex(strip_list_marker(node_text(node)))


def strip_list_marker_from_record(record: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(record)
    changed = False
    for key in ("merged_text", "text_for_embedding", "text", "text_preview"):
        value = prepared.get(key)
        if isinstance(value, str) and LIST_MARKER_RE.match(value):
            prepared[key] = strip_list_marker(value)
            changed = True
            break

    block = prepared.get("block")
    if isinstance(block, dict):
        block_copy = dict(block)
        content = block_copy.get("content")
        content_copy: Any = content
        segments = extract_content_segments(prepared)
        if segments:
            stripped_segments = []
            stripped = False
            for segment in segments:
                segment_copy = dict(segment)
                if not stripped and str(segment_copy.get("type") or "").lower() == "text":
                    content_text = str(segment_copy.get("content") or segment_copy.get("text") or "")
                    if LIST_MARKER_RE.match(content_text):
                        replacement = strip_list_marker(content_text)
                        if "content" in segment_copy:
                            segment_copy["content"] = replacement
                        else:
                            segment_copy["text"] = replacement
                        stripped = True
                        changed = True
                stripped_segments.append(segment_copy)
            if isinstance(content, dict):
                content_copy = dict(content)
                for key in ("paragraph_content", "title_content", "content"):
                    if isinstance(content_copy.get(key), list):
                        content_copy[key] = stripped_segments
                        break
            elif isinstance(content, list):
                content_copy = stripped_segments
            block_copy["content"] = content_copy
            prepared["block"] = block_copy
    return prepared if changed else record


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


def merge_latex_fragments(parts: list[str]) -> str:
    text = ""
    for part in parts:
        part = str(part or "").strip()
        if not part:
            continue
        if not text:
            text = part
        elif part.startswith((",", ".", ";", ":", ")", "]", "}")):
            text += part
        else:
            text += " " + part
    return normalize_latex_text(text)


def node_text(node: Any) -> str:
    if hasattr(node, "text"):
        return str(node.text).strip()
    if isinstance(node, dict):
        return str(node.get("text") or node.get("text_for_embedding") or node.get("text_preview") or "").strip()
    return ""


def node_record(node: Any) -> dict[str, Any]:
    return getattr(node, "record", node if isinstance(node, dict) else {})


def sorted_render_children(children: Any) -> list[Any]:
    child_list = list(children or [])
    if any(has_explicit_reading_order(getattr(child, "record", child if isinstance(child, dict) else {})) for child in child_list):
        return sorted(child_list, key=node_reading_order_key)
    if any(record_has_bbox(getattr(child, "record", child if isinstance(child, dict) else {})) for child in child_list):
        return sort_nodes_by_reading_order(child_list, fallback_key=node_reading_order_key)
    return sorted(child_list, key=node_reading_order_key)


def has_explicit_reading_order(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    for key in ("regime_reading_order", "dag_reading_order", "global_order", "reading_order", "original_order", "original_index", "index"):
        if numeric_value(record.get(key)) is not None:
            return True
    return False


def node_reading_order_key(node: Any) -> tuple[int, float, float, str]:
    record = getattr(node, "record", node if isinstance(node, dict) else {})
    for key in ("regime_reading_order", "dag_reading_order", "xycut_reading_order", "global_order", "reading_order", "original_order", "original_index", "index"):
        value = numeric_value(record.get(key))
        if value is not None:
            return (0, value, 0.0, "")

    source_id = min_numeric_sequence(record.get("source_node_ids"))
    if source_id is not None:
        return (1, source_id, 0.0, "")

    merged_ids = getattr(node, "merged_node_ids", None)
    merged_id = min_numeric_sequence(merged_ids)
    if merged_id is not None:
        return (1, merged_id, 0.0, "")

    node_id = numeric_value(getattr(node, "node_id", None))
    if node_id is not None and node_id >= 0:
        return (1, node_id, 0.0, "")

    page = numeric_value(record.get("page_idx"))
    visual = numeric_value(record.get("visual_order"))
    if page is not None or visual is not None:
        return (2, page if page is not None else 0.0, visual if visual is not None else 0.0, "")

    for key in ("id", "node_id", "block_id"):
        value = numeric_value(record.get(key))
        if value is not None:
            return (3, value, 0.0, "")

    return (4, 0.0, 0.0, "")


def record_has_bbox(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    value = record.get("bbox")
    return isinstance(value, (list, tuple)) and len(value) >= 4


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


def min_numeric_sequence(value: Any) -> float | None:
    if not isinstance(value, (list, tuple)):
        return None
    numbers = [number for number in (numeric_value(item) for item in value) if number is not None]
    return min(numbers) if numbers else None


def canonical_render_type(record: dict[str, Any]) -> str:
    if str(record.get("list_type") or "").lower() == "reference_list":
        return "reference"
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


def render_text_with_inline_latex(text: str, *, strip: bool = True) -> str:
    value = str(text or "")
    if not value:
        return ""
    rendered: list[str] = []
    cursor = 0
    while cursor < len(value):
        span = find_next_inline_latex_span(value, cursor)
        if span is None:
            rendered.append(escape_latex(value[cursor:]))
            break
        start, end = span
        if start > cursor:
            rendered.append(escape_latex(value[cursor:start]))
        raw_math = value[start:end].strip()
        if raw_math:
            rendered.append(render_inline_math(raw_math))
        cursor = end
    output = re.sub(r"\n{3,}", "\n\n", "".join(rendered))
    return output.strip() if strip else output


def find_next_inline_latex_span(text: str, start_index: int) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    dollar = text.find("$", start_index)
    if dollar >= 0 and not text.startswith("$$", dollar):
        end = find_unescaped(text, "$", dollar + 1)
        if end is not None:
            candidates.append((dollar, end + 1))
    paren = text.find(r"\(", start_index)
    if paren >= 0:
        end = text.find(r"\)", paren + 2)
        if end >= 0:
            candidates.append((paren, end + 2))
    command_match = find_next_math_command(text, start_index)
    if command_match is not None:
        command_start, _command_name = command_match
        end = consume_bare_latex_math(text, command_start)
        if end > command_start:
            candidates.append((command_start, end))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])


def find_unescaped(text: str, needle: str, start_index: int) -> int | None:
    index = text.find(needle, start_index)
    while index >= 0:
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return index
        index = text.find(needle, index + 1)
    return None


def find_next_math_command(text: str, start_index: int) -> tuple[int, str] | None:
    for match in MATH_COMMAND_RE.finditer(text, start_index):
        command_name = match.group(1)
        if command_name in INLINE_MATH_COMMANDS:
            return match.start(), command_name
    return None


def consume_bare_latex_math(text: str, start_index: int) -> int:
    index = start_index
    brace_depth = 0
    saw_command = False
    while index < len(text):
        char = text[index]
        if char == "\\":
            command = MATH_COMMAND_RE.match(text, index)
            if command:
                saw_command = True
                index = command.end()
                continue
            if index + 1 < len(text):
                saw_command = True
                index += 2
                continue
            break
        if char == "{":
            brace_depth += 1
            index += 1
            continue
        if char == "}":
            if brace_depth <= 0:
                break
            brace_depth -= 1
            index += 1
            continue
        if brace_depth > 0:
            index += 1
            continue
        if char.isspace():
            next_index = index + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if next_index < len(text) and (text[next_index] == "\\" or text[next_index] in "{}_^+-=*/<>,.()[]|"):
                index = next_index
                continue
            break
        if char in "_^+-=*/<>,.()[]|" or char.isdigit():
            index += 1
            continue
        break
    return index if saw_command else start_index


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


ALGORITHM_MATH_UNICODE_REPLACEMENTS = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ϵ": r"\epsilon",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "±": r"\pm",
    "×": r"\times",
    "÷": r"\div",
    "∞": r"\infty",
    "∂": r"\partial",
    "∇": r"\nabla",
    "∑": r"\sum",
    "∫": r"\int",
    "∈": r"\in",
    "∉": r"\notin",
    "∋": r"\ni",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "⊃": r"\supset",
    "⊇": r"\supseteq",
    "∪": r"\cup",
    "∩": r"\cap",
    "∧": r"\wedge",
    "∨": r"\vee",
    "¬": r"\neg",
    "∀": r"\forall",
    "∃": r"\exists",
    "∅": r"\emptyset",
    "∝": r"\propto",
    "∼": r"\sim",
    "≃": r"\simeq",
    "≅": r"\cong",
    "≡": r"\equiv",
    "≪": r"\ll",
    "≫": r"\gg",
    "⋅": r"\cdot",
    "·": r"\cdot",
    "∗": r"*",
    "√": r"\sqrt{}",
    "→": r"\to",
    "←": r"\gets",
    "↔": r"\leftrightarrow",
    "⟶": r"\longrightarrow",
    "⟵": r"\longleftarrow",
    "⇔": r"\Leftrightarrow",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
}


CODE_UNICODE_REPLACEMENTS = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ϵ": "epsilon",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Γ": "Gamma",
    "Δ": "Delta",
    "Θ": "Theta",
    "Λ": "Lambda",
    "Ξ": "Xi",
    "Π": "Pi",
    "Σ": "Sigma",
    "Φ": "Phi",
    "Ψ": "Psi",
    "Ω": "Omega",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "≈": "~=",
    "±": "+/-",
    "×": "x",
    "÷": "/",
    "∞": "inf",
    "∂": "partial",
    "∇": "nabla",
    "∑": "sum",
    "∫": "int",
    "∈": " in ",
    "∉": " notin ",
    "∋": " contains ",
    "⊂": " subset ",
    "⊆": " subseteq ",
    "⊃": " superset ",
    "⊇": " superseteq ",
    "∪": " union ",
    "∩": " inter ",
    "∧": " and ",
    "∨": " or ",
    "¬": "not ",
    "∀": "forall ",
    "∃": "exists ",
    "∅": "empty",
    "∝": "propto",
    "∼": "~",
    "≃": "~=",
    "≅": "~=",
    "≡": "==",
    "≪": "<<",
    "≫": ">>",
    "⋅": "*",
    "·": "*",
    "∗": "*",
    "√": "sqrt",
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "⟶": "->",
    "⟵": "<-",
    "⇔": "<=>",
    "⇒": "=>",
    "⇐": "<=",
    "′": "'",
    "″": "''",
    "°": "deg",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "•": "*",
    "–": "-",
    "—": "---",
    "−": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}


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
