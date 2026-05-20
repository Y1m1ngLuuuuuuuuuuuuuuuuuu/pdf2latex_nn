"""Shared LaTeX rendering helpers for the canonical IR renderer.

This module owns escaping, inline/display math repair, list marker handling,
algorithm formatting, and figure/table crop placeholder helpers.  It deliberately
does not expose the deprecated tree renderer entrypoint; production generation
should import helpers from here instead of ``src.generation.latex_renderer``.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from src.generation.citations import strip_reference_label
from src.generation.table_assets import (
    ensure_figure_asset,
    ensure_table_pdf_crop,
    first_record_bbox,
    is_wide_visual_record,
    record_bbox,
    table_caption_text,
    union_bbox,
)
from src.perception.xy_cut import sort_nodes_by_reading_order
from src.perception.title_features import is_front_matter_date_text, strip_title_numbering, title_numbering_level


DISPLAY_MATH_ENVS = {"equation", "align", "gather", "eqnarray", "flalign", "multline"}


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
    "ell",
    "pmb",
    "mathbb",
    "mathscr",
    "boldsymbol",
    "hat",
    "widehat",
    "bar",
    "overline",
    "tilde",
    "widetilde",
    "vec",
    "dot",
    "ddot",
    "check",
    "breve",
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
    "subset",
    "subseteq",
    "supset",
    "supseteq",
    "cup",
    "cap",
    "forall",
    "exists",
    "emptyset",
    "infty",
    "parallel",
    "lVert",
    "rVert",
    "langle",
    "rangle",
}


ORDERED_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\.|[a-zA-Z]\.)\s+")


LIST_MARKER_RE = re.compile(
    r"^(?:\s*[\)\]\}）】、,.;:：;]*\s*[\u2022\u25E6\u25CB\u25AA\-\*]\s*|\s*(?:\d+\.|[a-zA-Z]\.)\s+)"
)


DECIMAL_HEADING_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)+\.?\s+\S")


PSEUDOCODE_BREAK_RE = re.compile(
    r"\s+(?=(?:Input|Output|Require|Ensure)\s*:|Algorithm\s*\d+\b|(?:for|while|if|else|elif|return|end)\b)",
    re.IGNORECASE,
)


ALGORITHM_CAPTION_RE = re.compile(r"^\s*Algorithm\s*(?:\d+)?\s*[:.\-]?\s*(.*)$", re.IGNORECASE)


PSEUDOCODE_IO_RE = re.compile(r"^\s*(Input|Require|Output|Ensure)\s*:\s*(.*)$", re.IGNORECASE)


PSEUDOCODE_FOR_RE = re.compile(r"^\s*for\s+(.+?)(?:\s+do)?\s*$", re.IGNORECASE)


PSEUDOCODE_WHILE_RE = re.compile(r"^\s*while\s+(.+?)(?:\s+do)?\s*$", re.IGNORECASE)


PSEUDOCODE_IF_RE = re.compile(r"^\s*if\s+(.+?)(?:\s+then)?\s*$", re.IGNORECASE)


PSEUDOCODE_RETURN_RE = re.compile(r"^\s*return\s+(.+)$", re.IGNORECASE)


PSEUDOCODE_END_RE = re.compile(r"^\s*end(?:\s+(for|if|while))?\s*$", re.IGNORECASE)


ALGORITHM_CODE_MARKER_RE = re.compile(r"([{};]|(?:\+\+|--|==|!=|&&|\|\|))")


TABLE_CAPTION_RE = re.compile(r"^\s*(Table\s*\d*[:.\-]?\s*[^\n]+)", re.IGNORECASE)


FLOAT_CAPTION_LABEL_RE = {
    "table": re.compile(
        r"^\s*(?:Table|Tab\.?)\s+(?:\d+(?:\.\d+)*[A-Za-z]?|[IVXLCDM]+)\s*[:.\-–—]?\s*",
        re.IGNORECASE,
    ),
    "figure": re.compile(
        r"^\s*(?:Figure|Fig\.?)\s+(?:\d+(?:\.\d+)*[A-Za-z]?|[IVXLCDM]+)\s*[:.\-–—]?\s*",
        re.IGNORECASE,
    ),
}


FLOAT_CAPTION_ANY_LABEL_RE = {
    "table": re.compile(
        r"\b(?:Table|Tab\.?)\s+(?:\d+(?:\.\d+)*[A-Za-z]?|[IVXLCDM]+)\s*[:.\-–—]\s*",
        re.IGNORECASE,
    ),
    "figure": re.compile(
        r"\b(?:Figure|Fig\.?)\s+(?:\d+(?:\.\d+)*[A-Za-z]?|[IVXLCDM]+)\s*[:.\-–—]\s*",
        re.IGNORECASE,
    ),
}


CAPTION_TEXT_MATH_COMMAND_RE = re.compile(r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt)\s*\{\s*([^{}]+?)\s*\}")


LATEX_MATH_MARKER_RE = re.compile(r"(\\[A-Za-z]+|[_^{}]|[<>=+\-*/]|\\\(|\\\[)")


MATH_COMMAND_RE = re.compile(r"\\([A-Za-z]+)\*?")


LATEX_FONTSIZE_WRAPPER_RE = re.compile(
    r"\{\\fontsize\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\\selectfont\s*(?P<body>[^{}]*)\}",
    re.DOTALL,
)


LATEX_FONTSIZE_COMMAND_RE = re.compile(
    r"\\fontsize\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\\selectfont\s*",
    re.DOTALL,
)


STRUCTURAL_LATEX_COMMAND_RE = re.compile(
    r"\\(?:"
    r"(?:sub)*section|paragraph|subparagraph|caption|captionof|label|ref|cite|"
    r"begin\s*\{\s*(?:figure|table|algorithm|multicols|multicols\*|abstract|thebibliography)\s*\}|"
    r"end\s*\{\s*(?:figure|table|algorithm|multicols|multicols\*|abstract|thebibliography)\s*\}"
    r")",
    re.IGNORECASE,
)


GREEK_CONTEXT_RE = re.compile(r"[αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ]")


def list_environment_for_record(record: dict[str, Any], *, fallback_text: str = "") -> str:
    explicit = str(record.get("list_type") or record.get("list_style") or record.get("enum_type") or "").casefold()
    if explicit in {"ordered", "enumerate", "numbered", "number", "alpha", "roman"}:
        return "enumerate"
    text = fallback_text or node_text(record)
    return list_environment_for_text(text) or "itemize"


def list_environment_for_text(text: str) -> str | None:
    value = str(text or "")
    if DECIMAL_HEADING_PREFIX_RE.match(value):
        return None
    if not LIST_MARKER_RE.match(value):
        return None
    return "enumerate" if ORDERED_LIST_MARKER_RE.match(value) else "itemize"


def strip_list_marker(text: str) -> str:
    return LIST_MARKER_RE.sub("", str(text or ""), count=1).strip()


def render_algorithm_block(text: str, *, label: str | None = None) -> str:
    caption, commands = parse_pseudo_code(text)
    lines = [r"\begin{algorithm}[H]"]
    if caption:
        lines.append(rf"\caption{{{escape_latex(caption)}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
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
            commands.append(rf"\For{{{format_algorithmic_text(for_match.group(1).strip(), allow_math=False)}}}")
            block_stack.append("for")
            continue

        while_match = PSEUDOCODE_WHILE_RE.match(line)
        if while_match:
            commands.append(rf"\While{{{format_algorithmic_text(while_match.group(1).strip(), allow_math=False)}}}")
            block_stack.append("while")
            continue

        if_match = PSEUDOCODE_IF_RE.match(line)
        if if_match:
            commands.append(rf"\If{{{format_algorithmic_text(if_match.group(1).strip(), allow_math=False)}}}")
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


def format_algorithmic_text(text: str, *, allow_math: bool = True) -> str:
    prepared = normalize_algorithm_math_text(text)
    if not prepared:
        return ""
    if is_algorithm_code_like(prepared) or not allow_math:
        return r"\texttt{" + escape_algorithm_code_text(prepared) + r"}"
    if allow_math and LATEX_MATH_MARKER_RE.search(prepared):
        return r"\(\displaystyle " + escape_algorithm_math_text(prepared) + r"\)"
    return escape_latex(prepared)


def is_algorithm_code_like(text: str) -> bool:
    return bool(ALGORITHM_CODE_MARKER_RE.search(str(text or "")))


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


def escape_algorithm_code_text(text: str) -> str:
    safe = "".join(_safe_code_verbatim_char(char) for char in str(text or ""))
    return escape_latex(safe)


def restore_algorithm_line_breaks(text: str) -> str:
    body = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" in body:
        return body
    body = PSEUDOCODE_BREAK_RE.sub("\n", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def render_table_placeholder(
    record: dict[str, Any],
    text: str,
    *,
    source_pdf: str | Path | None = None,
    asset_output_dir: str | Path | None = None,
    asset_latex_prefix: str = "assets",
    as_nonfloat: bool = False,
    label: str | None = None,
) -> str:
    if int(record.get("table_group_size") or 1) > 1 and record.get("table_group_primary") is False:
        return ""
    table_id = table_node_identifier(record)
    bbox = format_table_bbox(record.get("table_group_bbox") or record.get("bbox"))
    caption = table_caption_text(record) or extract_table_caption(text) or "Table reconstruction placeholder"
    caption = clean_float_caption_text(caption, "table") or "Table reconstruction placeholder"
    graphic = ensure_table_pdf_crop(
        record,
        source_pdf=source_pdf or cfg_source_pdf(record),
        asset_output_dir=asset_output_dir,
        asset_latex_prefix=asset_latex_prefix,
    )
    source_layout = record.get("source_table_layout") if isinstance(record.get("source_table_layout"), dict) else {}
    source_width_scope = str(source_layout.get("source_width_scope") or "").casefold()
    source_environment = str(source_layout.get("source_environment") or "").casefold()
    if source_width_scope == "page" or source_environment == "table*":
        wide = True
    elif source_width_scope == "column":
        wide = False
    else:
        wide = is_wide_visual_record(record, bbox_keys=("table_group_bbox", "bbox"))
    todo = f"% [TODO_TABLE_RECONSTRUCT: BBOX={bbox}, ID={table_id}]"
    include_width = r"\linewidth" if as_nonfloat else (r"\textwidth" if wide else r"\linewidth")
    graphic_line = rf"\includegraphics[width={include_width}]{{{graphic}}}" if graphic else todo
    # ``table*`` is allowed to float away from the source position in
    # two-column layouts, and ``[H]`` does not reliably pin double-column
    # floats.  Keep every reconstructed table in a normal float with [H]; wide
    # tables still use ``\textwidth`` so the crop can span the available page
    # width when it is emitted outside a multicol band.
    environment = "table"
    placement = "H"
    lines = [r"\begin{center}"] if as_nonfloat else [rf"\begin{{{environment}}}[{placement}]", r"\centering"]
    if source_layout:
        lines.append(
            "% [SOURCE_TABLE_LAYOUT: "
            f"env={source_layout.get('source_environment')}, "
            f"placement={source_layout.get('source_placement')}, "
            f"width={source_layout.get('source_width_scope')}]"
        )
    lines.append(graphic_line)
    lines.append(rf"\captionof{{table}}{{{render_text_with_inline_latex(caption)}}}" if as_nonfloat else rf"\caption{{{render_text_with_inline_latex(caption)}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{center}" if as_nonfloat else rf"\end{{{environment}}}")
    return "\n".join(lines)


def figure_placeholder(record: dict[str, Any]) -> str:
    figure_id = figure_node_identifier(record)
    bbox = format_table_bbox(record.get("figure_group_bbox") or record.get("image_group_bbox") or record.get("bbox"))
    return f"% [TODO_FIGURE_RECONSTRUCT: BBOX={bbox}, ID={figure_id}]"


def render_figure_block(
    record: dict[str, Any],
    text: str = "",
    *,
    source_pdf: str | Path | None = None,
    asset_output_dir: str | Path | None = None,
    asset_latex_prefix: str = "assets",
    rendered_caption: str | None = None,
    as_nonfloat: bool = False,
    label: str | None = None,
) -> str:
    caption = rendered_caption
    if caption is None:
        caption_text = str(record.get("figure_group_caption") or record.get("image_group_caption") or record.get("figure_caption") or record.get("caption") or text or "Figure")
        caption_text = clean_float_caption_text(caption_text, "figure") or "Figure"
        caption = render_text_with_inline_latex(caption_text)
    asset_path = ensure_figure_asset(
        record,
        source_pdf=source_pdf or cfg_source_pdf(record),
        asset_output_dir=asset_output_dir,
        asset_latex_prefix=asset_latex_prefix,
    )
    graphic_line = (
        rf"\includegraphics[width={'1.000' if as_nonfloat else figure_include_width(record)}\linewidth]{{{asset_path}}}"
        if asset_path
        else figure_placeholder(record)
    )
    if as_nonfloat:
        lines = [
            r"\begin{center}",
            graphic_line,
            rf"\captionof{{figure}}{{{caption}}}",
        ]
        if label:
            lines.append(rf"\label{{{label}}}")
        lines.append(r"\end{center}")
        return "\n".join(lines)
    lines = [
        r"\begin{figure}[H]",
        r"\centering",
        graphic_line,
        rf"\caption{{{caption}}}",
    ]
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def render_figure_minipage_group(
    records: Sequence[dict[str, Any]],
    text: str = "",
    *,
    source_pdf: str | Path | None = None,
    asset_output_dir: str | Path | None = None,
    asset_latex_prefix: str = "assets",
    rendered_caption: str | None = None,
    as_nonfloat: bool = False,
    label: str | None = None,
) -> str:
    members = [_figure_member_record(record) for record in records]
    members = [record for record in members if record_bbox(record) is not None or any(record.get(key) for key in ("img_path", "image_path", "figure_path", "figure_asset_path", "image_asset_path"))]
    if not members:
        return render_figure_block(
            records[0] if records else {"type": "figure"},
            text,
            source_pdf=source_pdf,
            asset_output_dir=asset_output_dir,
            asset_latex_prefix=asset_latex_prefix,
            rendered_caption=rendered_caption,
            as_nonfloat=as_nonfloat,
            label=label,
        )
    members = sorted(members, key=_figure_member_sort_key)
    caption = rendered_caption
    if caption is None:
        caption_text = _figure_group_caption(records, text)
        caption_text = clean_float_caption_text(caption_text, "figure") or "Figure"
        caption = render_text_with_inline_latex(caption_text or "Figure")
    group_width = 0.96 if as_nonfloat else _figure_group_width_fraction(records, members)
    widths = _figure_minipage_widths(members)
    lines = [r"\begin{center}"] if as_nonfloat else [r"\begin{figure}[H]", r"\centering"]
    chunks: list[str] = []
    for member, width in zip(members, widths):
        asset_path = ensure_figure_asset(
            member,
            source_pdf=source_pdf or cfg_source_pdf(member),
            asset_output_dir=asset_output_dir,
            asset_latex_prefix=asset_latex_prefix,
        )
        graphic_line = (
            rf"\includegraphics[width=\linewidth]{{{asset_path}}}"
            if asset_path
            else figure_placeholder(member)
        )
        chunks.append(
            "\n".join(
                [
                    rf"\begin{{minipage}}[t]{{{width:.3f}\linewidth}}",
                    r"\centering",
                    graphic_line,
                    r"\end{minipage}",
                ]
            )
        )
    image_row = r"\hfill".join(chunks)
    if group_width < 0.90:
        lines.extend(
            [
                rf"\begin{{minipage}}[t]{{{group_width:.3f}\linewidth}}",
                r"\centering",
                image_row,
                r"\end{minipage}",
            ]
        )
    else:
        lines.append(image_row)
    if caption:
        lines.append(rf"\captionof{{figure}}{{{caption}}}" if as_nonfloat else rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{center}" if as_nonfloat else r"\end{figure}")
    return "\n".join(lines)


def figure_include_width(record: dict[str, Any]) -> str:
    bbox = first_record_bbox(record, ("bbox", "figure_group_bbox", "image_group_bbox"))
    if bbox is None:
        return "0.95"
    ratio = _visual_width_fraction_from_bbox(bbox, [record])
    return f"{ratio:.3f}"


def figure_node_identifier(record: dict[str, Any]) -> str:
    for key in ("figure_group_id", "image_group_id", "id", "block_id", "figure_id", "image_id"):
        value = record.get(key)
        if value:
            return str(value)
    value = record.get("global_order")
    if value is not None:
        return f"figure_{value}"
    return "figure_unknown"


def _figure_member_record(record: dict[str, Any]) -> dict[str, Any]:
    member = dict(record)
    for key in (
        "figure_group_bbox",
        "image_group_bbox",
        "figure_group_id",
        "image_group_id",
        "figure_group_caption",
        "image_group_caption",
        "figure_group_member_ids",
        "image_group_member_ids",
        "figure_group_member_node_ids",
        "image_group_member_node_ids",
        "figure_group_primary",
        "image_group_primary",
        "figure_group_size",
        "image_group_size",
        "figure_group_render_strategy",
        "image_group_render_strategy",
    ):
        member.pop(key, None)
    return member


def _figure_member_sort_key(record: dict[str, Any]) -> tuple[int, float, float, int]:
    member_index = record.get("figure_group_member_index")
    if member_index is None:
        member_index = record.get("image_group_member_index")
    box = record_bbox(record) or (0.0, 0.0, 0.0, 0.0)
    if member_index is not None:
        try:
            return (0, float(member_index), box[0], int(record.get("global_order") or 0))
        except (TypeError, ValueError):
            pass
    return (1, box[1], box[0], int(record.get("global_order") or 0))


def _figure_group_caption(records: Sequence[dict[str, Any]], fallback: str) -> str:
    for record in records:
        value = (
            record.get("figure_group_caption")
            or record.get("image_group_caption")
            or record.get("figure_caption")
            or record.get("image_caption")
            or record.get("caption")
        )
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return fallback


def _figure_minipage_widths(records: Sequence[dict[str, Any]]) -> list[float]:
    boxes = [record_bbox(record) for record in records]
    raw_widths = [max((box[2] - box[0]) if box else 1.0, 1.0) for box in boxes]
    total = sum(raw_widths)
    if total <= 0:
        return [0.96 / max(len(records), 1)] * len(records)
    widths = [0.96 * width / total for width in raw_widths]
    if len(widths) == 1:
        return [min(max(widths[0], 0.35), 0.96)]
    min_width = 0.16 if len(widths) > 3 else 0.20
    max_width = 0.78 if len(widths) == 2 else 0.55
    widths = [min(max(width, min_width), max_width) for width in widths]
    total = sum(widths)
    if total > 0.96:
        scale = 0.96 / total
        widths = [width * scale for width in widths]
    return widths


def _figure_group_width_fraction(
    records: Sequence[dict[str, Any]],
    members: Sequence[dict[str, Any]],
) -> float:
    member_boxes = [box for box in (record_bbox(member) for member in members) if box is not None]
    if member_boxes:
        return _visual_width_fraction_from_bbox(union_bbox(member_boxes), list(records) + list(members))
    for record in records:
        group_box = first_record_bbox(record, ("figure_group_bbox", "image_group_bbox"))
        if group_box is not None:
            return _visual_width_fraction_from_bbox(group_box, records)
    return 0.96


def _visual_width_fraction_from_bbox(
    bbox: tuple[float, float, float, float],
    records: Sequence[dict[str, Any]],
) -> float:
    width = max(float(bbox[2]) - float(bbox[0]), 0.0)
    if width <= 0:
        return 0.95
    page_width = _page_width_for_visual_records(records, bbox)
    page_ratio = width / max(page_width, 1.0)
    if page_ratio >= 0.62:
        return 0.98
    # For figures rendered outside the main multicol flow, preserve their
    # physical width instead of expanding every crop to the full text width.
    return min(max(page_ratio, 0.22), 0.95)


def _page_width_for_visual_records(
    records: Sequence[dict[str, Any]],
    fallback_bbox: tuple[float, float, float, float],
) -> float:
    for record in records:
        value = _number_or_none(record.get("page_width"))
        if value and value > 0:
            return float(value)
    return max(float(fallback_bbox[2]), 1000.0)


def cfg_source_pdf(record: dict[str, Any]) -> str | None:
    for key in ("source_pdf", "pdf_path", "style_source_pdf"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    source_refs = record.get("source_refs")
    if isinstance(source_refs, list):
        for ref in source_refs:
            if isinstance(ref, dict):
                metadata = ref.get("metadata")
                if isinstance(metadata, dict) and isinstance(metadata.get("pdf_path"), str):
                    return metadata["pdf_path"]
    return None


def table_node_identifier(record: dict[str, Any]) -> str:
    for key in ("table_group_id", "id", "block_id", "table_id"):
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


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def extract_table_caption(text: str) -> str | None:
    match = TABLE_CAPTION_RE.search(str(text or ""))
    if not match:
        return None
    return " ".join(match.group(1).split())


def clean_float_caption_text(text: str, kind: str) -> str:
    """Normalize OCR float captions before feeding them to ``\\caption``.

    The source PDF visibly contains labels such as ``Table 3:`` and
    ``Figure 2:``, but LaTeX generates those labels itself.  Keeping the
    recognized label inside ``\\caption{...}`` renders duplicate prefixes.
    Caption OCR also often turns short prose tokens such as ``N/A`` into
    spaced math commands (``\\mathrm { N } / \\mathrm { A }``), which should
    be plain text in the generated caption.
    """

    value = " ".join(str(text or "").split())
    if not value:
        return ""
    value = normalize_caption_ocr_math(value)
    value = select_float_caption_segment(value, kind)
    value = strip_float_caption_label(value, kind)
    return value.strip(" \t\n\r:.-–—")


def select_float_caption_segment(text: str, kind: str) -> str:
    """Pick the caption segment belonging to this float kind.

    MinerU can occasionally attach adjacent float captions to one visual group,
    e.g. ``Table 5: ... Figure 3: ...``.  A table should not inherit the
    following figure caption, and a figure should start from its own label.
    """

    value = str(text or "")
    normalized_kind = str(kind or "").casefold()
    table_label = FLOAT_CAPTION_ANY_LABEL_RE["table"]
    figure_label = FLOAT_CAPTION_ANY_LABEL_RE["figure"]
    if normalized_kind == "table":
        figure_match = figure_label.search(value, 1)
        if figure_match:
            return value[: figure_match.start()].rstrip(" ;,")
        return value
    if normalized_kind == "figure":
        figure_match = figure_label.search(value)
        if figure_match:
            return value[figure_match.start() :]
        table_match = table_label.search(value)
        if table_match and table_match.start() == 0:
            return ""
    return value


def strip_float_caption_label(text: str, kind: str) -> str:
    value = str(text or "")
    pattern = FLOAT_CAPTION_LABEL_RE.get(str(kind or "").casefold())
    if pattern is None:
        return value
    # OCR and metadata can both carry the visible label, producing
    # ``Table 3: Table 3: ...``.  Strip all leading labels and let LaTeX own
    # the counter.
    previous = None
    while value and value != previous:
        previous = value
        value = pattern.sub("", value, count=1).lstrip()
    return value


def normalize_caption_ocr_math(text: str) -> str:
    value = str(text or "")
    value = CAPTION_TEXT_MATH_COMMAND_RE.sub(_caption_math_command_to_text, value)
    value = re.sub(r"\b([A-Za-z])\s*/\s*([A-Za-z])\b", r"\1/\2", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value


def _caption_math_command_to_text(match: re.Match[str]) -> str:
    body = " ".join(match.group(1).split())
    compact = "".join(body.split())
    if compact and len(compact) <= 12 and re.fullmatch(r"[A-Za-z0-9/+_.-]+", compact):
        return compact
    return body


def _safe_code_verbatim_char(char: str) -> str:
    if ord(char) < 128:
        return char
    if char in CODE_UNICODE_REPLACEMENTS:
        return CODE_UNICODE_REPLACEMENTS[char]
    ascii_fallback = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode("ascii")
    return ascii_fallback or "?"


def render_equation(text: str, *, label: str | None = None) -> str:
    stripped = normalize_display_math_text(str(text or "").strip())
    if not stripped:
        return "\\[\n\n\\]"
    if contains_structural_latex_command(stripped):
        return render_display_math_fallback(stripped)
    if not is_safe_display_math_latex(stripped):
        return render_display_math_fallback(stripped)
    multi_tag_render = render_multi_tag_equation(stripped)
    if multi_tag_render:
        return inject_display_math_label(multi_tag_render, label)
    if stripped.startswith("\\[") or stripped.startswith("$$"):
        return inject_display_math_label(stripped, label)
    begin_match = re.match(r"\\begin\{([^}]+)\}", stripped)
    if begin_match and begin_match.group(1).rstrip("*") in DISPLAY_MATH_ENVS:
        return inject_display_math_label(stripped, label)
    body_with_explicit_tag, explicit_tag = split_single_explicit_equation_tag(stripped)
    if explicit_tag is not None:
        label_line = f"\\label{{{label}}}\n" if label else ""
        body = scale_oversized_display_math_body(body_with_explicit_tag)
        return "\\begin{equation}\n" + label_line + body + rf" \tag{{{explicit_tag}}}" + "\n\\end{equation}"
    body, tag = split_trailing_equation_number(stripped)
    if tag is not None:
        label_line = f"\\label{{{label}}}\n" if label else ""
        body = scale_oversized_display_math_body(body)
        return "\\begin{equation}\n" + label_line + body + rf" \tag{{{tag}}}" + "\n\\end{equation}"
    if TAG_RE.search(stripped):
        label_line = f"\\label{{{label}}}\n" if label else ""
        return "\\begin{equation}\n" + label_line + stripped + "\n\\end{equation}"
    if should_render_as_align(stripped):
        label_line = f"\\label{{{label}}}\n" if label else ""
        return "\\begin{align}\n" + label_line + stripped + "\n\\end{align}"
    scaled = scale_oversized_display_math_body(stripped)
    if label:
        return "\\begin{equation}\n" + rf"\label{{{label}}}" + "\n" + scaled + "\n\\end{equation}"
    return "\\[\n" + scaled + "\n\\]"


def render_display_math_fallback(text: str) -> str:
    r"""Render malformed OCR formula text without breaking the whole document.

    MinerU can occasionally emit display-equation LaTeX with unbalanced braces
    or unmatched ``\left`` / ``\right`` delimiters.  Passing those fragments
    into ``align`` causes a fatal compile error, so the generator degrades that
    single formula into a compact escaped text box.  It is deliberately ugly
    but safe; preserving the rest of the paper is more important than trusting
    broken OCR math.
    """

    compact = re.sub(r"\s+", " ", str(text or "").strip())
    escaped = escape_latex(compact)
    return (
        "\\begin{center}\n"
        "\\begin{minipage}{0.95\\linewidth}\n"
        "\\footnotesize\\ttfamily\\raggedright\n"
        f"{escaped}\n"
        "\\end{minipage}\n"
        "\\end{center}"
    )


def is_safe_display_math_latex(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    if contains_structural_latex_command(value):
        return False
    if _contains_unescaped_dollar(value):
        return False
    if re.search(r"\\(?:frac|dfrac|tfrac)\s+(?!\{)", value):
        return False
    if re.search(r"\^\s*[-+]\s*[A-Za-z0-9]", value):
        return False
    if not _has_balanced_unescaped_braces(value):
        return False
    if len(re.findall(r"\\left\b|\\left(?=[\\.()\\[\\]{}|])", value)) != len(
        re.findall(r"\\right\b|\\right(?=[\\.()\\[\\]{}|])", value)
    ):
        return False
    begins = re.findall(r"\\begin\s*\{\s*([^{}]+?)\s*\}", value)
    ends = re.findall(r"\\end\s*\{\s*([^{}]+?)\s*\}", value)
    if begins or ends:
        stack: list[str] = []
        for match in re.finditer(r"\\(begin|end)\s*\{\s*([^{}]+?)\s*\}", value):
            kind, env = match.group(1), match.group(2).strip()
            if kind == "begin":
                stack.append(env)
            elif not stack or stack.pop() != env:
                return False
        if stack:
            return False
    return True


def _has_balanced_unescaped_braces(text: str) -> bool:
    depth = 0
    escaped = False
    for char in str(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def inject_display_math_label(latex: str, label: str | None) -> str:
    if not label or r"\label{" in str(latex or ""):
        return latex
    value = str(latex or "")
    begin_match = re.search(r"\\begin\{(?:equation|align|gather|eqnarray|flalign|multline)\*?\}", value)
    if begin_match:
        insert_at = begin_match.end()
        return value[:insert_at] + "\n" + rf"\label{{{label}}}" + value[insert_at:]
    if value.startswith("\\[") and value.endswith("\\]"):
        return "\\begin{equation}\n" + rf"\label{{{label}}}" + "\n" + value[2:-2].strip() + "\n\\end{equation}"
    if value.startswith("$$") and value.endswith("$$"):
        return "\\begin{equation}\n" + rf"\label{{{label}}}" + "\n" + value[2:-2].strip() + "\n\\end{equation}"
    return value


TAG_RE = re.compile(r"\\tag\s*\{([^{}]+)\}")


TRAILING_EQUATION_NUMBER_RE = re.compile(r"^(?P<body>.+?)\s*(?:\((?P<tag>[A-Za-z]?\d+(?:\.\d+)*)\))\s*$", re.DOTALL)


SINGLE_EXPLICIT_TAG_RE = re.compile(r"^(?P<body>.+?)\s*\\tag\s*\{\s*(?P<tag>[^{}]+?)\s*\}\s*$", re.DOTALL)


WIDE_DISPLAY_MATH_ENV_RE = re.compile(
    r"\\begin\s*\{\s*(?:array|aligned|alignedat|split|matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|smallmatrix)\s*\}",
    re.DOTALL,
)


DISPLAY_MATH_SCALE_MIN_COMPACT_CHARS = 180


DISPLAY_MATH_SCALE_MIN_ROW_CHARS = 150


def render_multi_tag_equation(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("\\[") and stripped.endswith("\\]"):
        stripped = stripped[2:-2].strip()
    elif stripped.startswith("$$") and stripped.endswith("$$"):
        stripped = stripped[2:-2].strip()

    matches = list(TAG_RE.finditer(stripped))
    if len(matches) <= 1:
        return None

    rows: list[tuple[str, str | None]] = []
    cursor = 0
    for match in matches:
        expr = stripped[cursor : match.start()].strip()
        tag = match.group(1).strip()
        if expr:
            rows.append((expr, tag))
        elif rows:
            prev_expr, _ = rows[-1]
            rows[-1] = (prev_expr, tag)
        cursor = match.end()

    tail = stripped[cursor:].strip()
    if tail:
        rows.append((tail, None))

    if len(rows) <= 1:
        return None

    rendered_rows = []
    for expr, tag in rows:
        rendered_rows.append(f"{expr} \\tag{{{tag}}}" if tag else expr)
    return "\\begin{align}\n" + "\\\\\n".join(rendered_rows) + "\n\\end{align}"


def split_trailing_equation_number(text: str) -> tuple[str, str | None]:
    stripped = text.strip()
    if TAG_RE.search(stripped):
        return stripped, None
    match = TRAILING_EQUATION_NUMBER_RE.match(stripped)
    if not match:
        return stripped, None
    body = match.group("body").strip()
    tag = match.group("tag").strip()
    if not body or not tag:
        return stripped, None
    return body, tag


def split_single_explicit_equation_tag(text: str) -> tuple[str, str | None]:
    stripped = text.strip()
    matches = list(TAG_RE.finditer(stripped))
    if len(matches) != 1:
        return stripped, None
    match = SINGLE_EXPLICIT_TAG_RE.match(stripped)
    if not match:
        return stripped, None
    body = match.group("body").strip()
    tag = match.group("tag").strip()
    if not body or not tag:
        return stripped, None
    return body, tag


def scale_oversized_display_math_body(text: str) -> str:
    body = str(text or "").strip()
    if not should_scale_display_math_body(body):
        return body
    return rf"\resizebox{{\ifdim\width>\linewidth\linewidth\else\width\fi}}{{!}}{{$\displaystyle {body}$}}"


def should_scale_display_math_body(text: str) -> bool:
    body = str(text or "").strip()
    if not body or r"\resizebox" in body:
        return False
    compact = re.sub(r"\s+", "", body)
    rows = [row.strip() for row in re.split(r"\\\\|\n", body) if row.strip()]
    max_row_len = max((len(re.sub(r"\s+", "", row)) for row in rows), default=len(compact))
    if WIDE_DISPLAY_MATH_ENV_RE.search(body):
        return len(compact) >= DISPLAY_MATH_SCALE_MIN_COMPACT_CHARS or max_row_len >= DISPLAY_MATH_SCALE_MIN_ROW_CHARS
    return max_row_len >= 260


def should_render_as_align(text: str) -> bool:
    stripped = text.strip()
    if "\\\\" in stripped and ("&" in stripped or "\n" in stripped):
        return True
    rows = [row.strip() for row in stripped.splitlines() if row.strip()]
    return len(rows) > 1 and any("&" in row for row in rows)


def render_inline_math(text: str) -> str:
    stripped = sanitize_latex_render_artifacts(str(text or "")).strip()
    if not stripped:
        return "$$"
    if contains_structural_latex_command(stripped) or _contains_unescaped_dollar(stripped[1:-1] if stripped.startswith("$") and stripped.endswith("$") else stripped):
        return escape_latex(stripped)
    if not _has_balanced_unescaped_braces(stripped):
        return escape_latex(stripped)
    stripped = strip_unbalanced_left_right_delimiters(stripped)
    if not is_plausible_inline_math_payload(stripped):
        return escape_latex(stripped)
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        return "$" + normalize_inline_math_unicode(stripped[1:-1].strip()) + "$"
    if stripped.startswith(r"\(") and stripped.endswith(r"\)") and len(stripped) >= 4:
        return r"\(" + normalize_inline_math_unicode(stripped[2:-2].strip()) + r"\)"
    return "$" + normalize_inline_math_unicode(stripped) + "$"


def strip_unbalanced_left_right_delimiters(text: str) -> str:
    value = str(text or "")
    left_count = len(re.findall(r"\\left\b|\\left(?=[\\.()\[\]{}|])", value))
    right_count = len(re.findall(r"\\right\b|\\right(?=[\\.()\[\]{}|])", value))
    if left_count == right_count:
        return value
    return re.sub(r"\\(?:left|right)\s*", "", value)


def normalize_inline_math_unicode(text: str) -> str:
    normalized = "".join(ALGORITHM_MATH_UNICODE_REPLACEMENTS.get(char, char) for char in str(text or ""))
    normalized = normalize_duplicate_math_command_slashes(normalized)
    return normalize_math_ocr_spacing(separate_glued_math_commands(normalized))


def normalize_display_math_text(text: str) -> str:
    """Repair MinerU/OCR LaTeX fragments before emitting display math.

    The OCR path often returns valid-looking LaTeX with command spacing such as
    ``\\operatorname* { m i n }`` and ``\\mathcal { L }``.  LaTeX compiles it,
    but the rendered equation becomes visually sparse and equation tags appear
    separated by a huge blank.  This cleanup keeps the original formula
    semantics while compacting only known math command patterns.
    """

    normalized = sanitize_latex_render_artifacts(text)
    normalized = "".join(ALGORITHM_MATH_UNICODE_REPLACEMENTS.get(char, char) for char in str(normalized or ""))
    normalized = normalize_duplicate_math_command_slashes(normalized)
    normalized = separate_glued_math_commands(normalized)
    normalized = normalize_math_ocr_spacing(normalized)
    return normalized.strip()


def sanitize_latex_render_artifacts(text: object) -> str:
    r"""Remove renderer-only style commands before math/text safety checks.

    Span-level font preservation is useful in ordinary text, but OCR/math
    fragments sometimes get emitted as ``{\fontsize{...}\selectfont x}``.
    Those commands are illegal inside many reconstructed math fragments and can
    also make the evaluation parser swallow following ``\section`` or
    ``\caption`` commands.  Keep the payload and drop only the style wrapper.
    """

    value = str(text or "")
    previous = None
    while previous != value:
        previous = value
        value = LATEX_FONTSIZE_WRAPPER_RE.sub(lambda match: match.group("body"), value)
    value = LATEX_FONTSIZE_COMMAND_RE.sub("", value)
    value = value.replace("\\selectfont", "")
    return value


def contains_structural_latex_command(text: object) -> bool:
    return bool(STRUCTURAL_LATEX_COMMAND_RE.search(str(text or "")))


def _contains_unescaped_dollar(text: str) -> bool:
    return find_unescaped(str(text or ""), "$", 0) is not None


MATH_TEXT_BRACE_COMMANDS = {
    "mathrm",
    "mathbf",
    "mathit",
    "mathsf",
    "mathtt",
    "mathcal",
    "mathbfcal",
    "mathbb",
    "mathscr",
    "boldsymbol",
    "pmb",
}


SPACED_OPERATOR_COMMANDS = {
    "min": r"\min",
    "max": r"\max",
    "lim": r"\lim",
    "sup": r"\sup",
    "inf": r"\inf",
    "argmin": r"\arg\min",
    "argmax": r"\arg\max",
}


MATH_COMMAND_BRACE_RE = re.compile(r"\\(?P<cmd>[A-Za-z]+)\s*\{\s*(?P<body>[^{}]+?)\s*\}")


SPACED_OPERATORNAME_RE = re.compile(r"\\operatorname\*?\s*\{\s*(?P<body>[A-Za-z](?:\s+[A-Za-z]){1,16})\s*\}")


def normalize_math_ocr_spacing(text: str) -> str:
    value = str(text or "")
    value = SPACED_OPERATORNAME_RE.sub(_replace_spaced_operatorname, value)
    value = MATH_COMMAND_BRACE_RE.sub(_replace_spaced_math_command, value)
    value = re.sub(r"\\operatorname\s*\{\s*([^{}]+?)\s*\}", lambda m: r"\operatorname{" + " ".join(m.group(1).split()) + "}", value)
    value = re.sub(r"\\operatorname\*\s*\{\s*([^{}]+?)\s*\}", lambda m: r"\operatorname*{" + " ".join(m.group(1).split()) + "}", value)
    value = re.sub(r"\s+([_^])\s*", r"\1", value)
    value = re.sub(r"([_^])\s+\{", r"\1{", value)
    value = re.sub(r"([_^])\{\s*(\\[A-Za-z]+\{[^{}]+\})\s*\}", r"\1{\2}", value)
    value = re.sub(r"([_^])\{\s*([^{}]+?)\s*\}", lambda m: m.group(1) + "{" + " ".join(m.group(2).split()) + "}", value)
    value = re.sub(r"\\left\s*\\\|", lambda _: r"\left\|", value)
    value = re.sub(r"\\right\s*\\\|", lambda _: r"\right\|", value)
    value = re.sub(r"\s+,", ",", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value


def _replace_spaced_operatorname(match: re.Match[str]) -> str:
    compact = "".join(match.group("body").split()).casefold()
    return SPACED_OPERATOR_COMMANDS.get(compact, r"\operatorname{" + compact + "}")


def _replace_spaced_math_command(match: re.Match[str]) -> str:
    cmd = match.group("cmd")
    body = match.group("body")
    if cmd not in MATH_TEXT_BRACE_COMMANDS:
        return match.group(0)
    compact_body = "".join(body.split()) if _looks_like_spaced_math_token(body) else " ".join(body.split())
    return rf"\{cmd}{{{compact_body}}}"


def _looks_like_spaced_math_token(text: str) -> bool:
    pieces = str(text or "").split()
    if not pieces:
        return False
    if len(pieces) == 1:
        return True
    return len(pieces) <= 12 and all(re.fullmatch(r"[A-Za-z0-9]+", piece) for piece in pieces)


def normalize_duplicate_math_command_slashes(text: str) -> str:
    return re.sub(r"\\\\([A-Za-z]+)", r"\\\1", str(text or ""))


GLUED_MATH_COMMAND_RE = re.compile(
    r"\\(times|sigma|sim|ell|lambda|Phi|phi|theta|rho|mu|alpha|beta|gamma|delta|epsilon|in)(?=[A-Z])"
)


def separate_glued_math_commands(text: str) -> str:
    """Repair OCR-style command/variable glue such as ``\\timesY``."""

    return GLUED_MATH_COMMAND_RE.sub(lambda match: f"\\{match.group(1)} ", str(text or ""))


def is_plausible_inline_math_payload(text: str) -> bool:
    """Reject OCR/style-span artifacts that are not real inline formulae.

    PyMuPDF often marks lone braces from email addresses or affiliation blocks
    as math because they use a symbol font.  Rendering ``{`` as ``${$`` creates
    uncompilable LaTeX, so inline math must contain at least one substantive
    math token: a letter/number, a Greek glyph, or a LaTeX command.
    """

    value = str(text or "").strip()
    if value.startswith("$") and value.endswith("$") and len(value) >= 2:
        value = value[1:-1].strip()
    if value.startswith(r"\(") and value.endswith(r"\)") and len(value) >= 4:
        value = value[2:-2].strip()
    if not value:
        return False
    return any(char.isalnum() for char in value) or bool(GREEK_CONTEXT_RE.search(value)) or "\\" in value


def node_text(node: Any) -> str:
    if hasattr(node, "text"):
        return str(node.text).strip()
    if isinstance(node, dict):
        return str(node.get("text") or node.get("text_for_embedding") or node.get("text_preview") or "").strip()
    return ""


def render_text_with_inline_latex(text: str, *, strip: bool = True) -> str:
    value = sanitize_latex_render_artifacts(text)
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
            raw_math, trailing_punctuation = split_trailing_inline_math_punctuation(raw_math)
            rendered.append(render_inline_math(raw_math))
            if trailing_punctuation:
                rendered.append(escape_latex(trailing_punctuation))
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


def split_trailing_inline_math_punctuation(text: str) -> tuple[str, str]:
    value = str(text or "").strip()
    if len(value) < 2 or value.startswith("$") or value.startswith(r"\("):
        return value, ""
    if value[-1] not in ".,;:":
        return value, ""
    if len(value) >= 2 and value[-2].isdigit() and value[-1] == ".":
        return value, ""
    return value[:-1].rstrip(), value[-1]


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
            command_start = match.start()
            if command_start > start_index and text[command_start - 1] == "\\":
                command_start -= 1
            return command_start, command_name
    return None


def consume_bare_latex_math(text: str, start_index: int) -> int:
    index = start_index
    brace_depth = 0
    saw_command = False
    while index < len(text):
        char = text[index]
        if char == "\\":
            if index + 2 < len(text) and text[index + 1] == "\\" and text[index + 2].isalpha():
                command = MATH_COMMAND_RE.match(text, index + 1)
                if command:
                    saw_command = True
                    index = command.end()
                    continue
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
    if char in {"\n", "\t"}:
        return char
    if ord(char) < 32 or ord(char) == 127:
        return ""
    if char in UNICODE_LATEX_REPLACEMENTS:
        return UNICODE_LATEX_REPLACEMENTS[char]
    if char in replacements:
        return replacements[char]
    if ord(char) < 128:
        return char
    ascii_fallback = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode("ascii")
    if ascii_fallback:
        return "".join(replacements.get(fallback_char, fallback_char) for fallback_char in ascii_fallback)
    # Do not inject literal question marks for glyphs that pdflatex cannot
    # represent.  Most such glyphs are OCR debris or unsupported math symbols;
    # a visible ``?`` is more misleading than a silent drop.  Common math/text
    # symbols are covered by UNICODE_LATEX_REPLACEMENTS above.
    return ""


def safe_verbatim_text(text: str) -> str:
    return "".join(_safe_verbatim_char(char) for char in str(text))


def _safe_verbatim_char(char: str) -> str:
    if ord(char) < 128:
        return char
    if char in UNICODE_LATEX_REPLACEMENTS:
        return UNICODE_LATEX_REPLACEMENTS[char]
    ascii_fallback = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode("ascii")
    return ascii_fallback or ""


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
    "ℓ": r"\ell",
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
    "η": r"\ensuremath{\eta}",
    "ζ": r"\ensuremath{\zeta}",
    "θ": r"\ensuremath{\theta}",
    "ι": r"\ensuremath{\iota}",
    "κ": r"\ensuremath{\kappa}",
    "ℓ": r"\ensuremath{\ell}",
    "λ": r"\ensuremath{\lambda}",
    "μ": r"\ensuremath{\mu}",
    "ν": r"\ensuremath{\nu}",
    "ξ": r"\ensuremath{\xi}",
    "π": r"\ensuremath{\pi}",
    "ρ": r"\ensuremath{\rho}",
    "σ": r"\ensuremath{\sigma}",
    "τ": r"\ensuremath{\tau}",
    "υ": r"\ensuremath{\upsilon}",
    "φ": r"\ensuremath{\phi}",
    "χ": r"\ensuremath{\chi}",
    "ψ": r"\ensuremath{\psi}",
    "ω": r"\ensuremath{\omega}",
    "Γ": r"\ensuremath{\Gamma}",
    "Δ": r"\ensuremath{\Delta}",
    "Θ": r"\ensuremath{\Theta}",
    "Λ": r"\ensuremath{\Lambda}",
    "Ξ": r"\ensuremath{\Xi}",
    "Π": r"\ensuremath{\Pi}",
    "Σ": r"\ensuremath{\Sigma}",
    "Φ": r"\ensuremath{\Phi}",
    "Ψ": r"\ensuremath{\Psi}",
    "Ω": r"\ensuremath{\Omega}",
    "≤": r"\ensuremath{\leq}",
    "≥": r"\ensuremath{\geq}",
    "≠": r"\ensuremath{\neq}",
    "≈": r"\ensuremath{\approx}",
    "±": r"\ensuremath{\pm}",
    "×": r"\ensuremath{\times}",
    "÷": r"\ensuremath{\div}",
    "∞": r"\ensuremath{\infty}",
    "∂": r"\ensuremath{\partial}",
    "∇": r"\ensuremath{\nabla}",
    "∈": r"\ensuremath{\in}",
    "∉": r"\ensuremath{\notin}",
    "∋": r"\ensuremath{\ni}",
    "⊂": r"\ensuremath{\subset}",
    "⊆": r"\ensuremath{\subseteq}",
    "⊃": r"\ensuremath{\supset}",
    "⊇": r"\ensuremath{\supseteq}",
    "∪": r"\ensuremath{\cup}",
    "∩": r"\ensuremath{\cap}",
    "∧": r"\ensuremath{\wedge}",
    "∨": r"\ensuremath{\vee}",
    "¬": r"\ensuremath{\neg}",
    "∀": r"\ensuremath{\forall}",
    "∃": r"\ensuremath{\exists}",
    "∅": r"\ensuremath{\emptyset}",
    "∑": r"\ensuremath{\sum}",
    "∫": r"\ensuremath{\int}",
    "∝": r"\ensuremath{\propto}",
    "∼": r"\ensuremath{\sim}",
    "≃": r"\ensuremath{\simeq}",
    "≅": r"\ensuremath{\cong}",
    "≡": r"\ensuremath{\equiv}",
    "≪": r"\ensuremath{\ll}",
    "≫": r"\ensuremath{\gg}",
    "⋅": r"\ensuremath{\cdot}",
    "·": r"\ensuremath{\cdot}",
    "∗": r"\ensuremath{*}",
    "∥": r"\ensuremath{\parallel}",
    "√": r"\ensuremath{\sqrt{}}",
    "→": r"\ensuremath{\rightarrow}",
    "←": r"\ensuremath{\leftarrow}",
    "↔": r"\ensuremath{\leftrightarrow}",
    "⟶": r"\ensuremath{\longrightarrow}",
    "⟵": r"\ensuremath{\longleftarrow}",
    "⇔": r"\ensuremath{\Leftrightarrow}",
    "⇒": r"\ensuremath{\Rightarrow}",
    "⇐": r"\ensuremath{\Leftarrow}",
    "′": r"\ensuremath{'}",
    "″": r"\ensuremath{''}",
    "°": r"\ensuremath{^\circ}",
    "¹": r"\ensuremath{^1}",
    "²": r"\ensuremath{^2}",
    "³": r"\ensuremath{^3}",
    "–": "--",
    "—": "---",
    "−": r"\ensuremath{-}",
    "•": r"\textbullet{}",
    "“": "``",
    "”": "''",
    "‘": "`",
    "’": "'",
}
