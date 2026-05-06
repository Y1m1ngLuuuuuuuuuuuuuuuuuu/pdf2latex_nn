#!/usr/bin/env python3
"""Compare the TexSoup alignment AST against source-level TeX structure cues."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.reasoning.label_generator import AlignmentLabeler, clean_equation_text, clean_text  # noqa: E402
from src.reasoning.label_generator import DISPLAY_MATH_ENV_NAMES, LIST_ENV_NAMES  # noqa: E402
from src.reasoning.latex_flattener import LatexFlattenerConfig, flatten_latex_file  # noqa: E402


SECTION_COMMANDS = {"section", "subsection", "subsubsection", "paragraph", "subparagraph"}
STRUCTURAL_TYPES = {
    "section",
    "list_container",
    "list_item",
    "equation_display",
    "figure_caption",
    "table_caption",
    "algorithm",
}
CAPTION_ENVS = {"figure", "figure*", "table", "table*"}


@dataclass(frozen=True)
class FrameworkNode:
    node_type: str
    clean_text: str
    text: str
    source_name: str | None
    position: int


def main() -> int:
    args = parse_args()
    source_nodes = extract_source_framework(args.tex)
    parser_nodes = extract_parser_framework(args)
    report = compare_frameworks(source_nodes, parser_nodes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["section_mismatches"] or report["sequence_mismatches"]:
        print(json.dumps({k: report[k] for k in ("section_mismatches", "sequence_mismatches")}, ensure_ascii=False, indent=2))
    return 1 if report["summary"]["has_mismatch"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--content-json", type=Path, default=Path("__unused_content.json"))
    parser.add_argument("--graph", type=Path, default=Path("__unused_graph.pt"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-mismatches", type=int, default=20)
    return parser.parse_args()


def extract_parser_framework(args: argparse.Namespace) -> list[FrameworkNode]:
    labeler = AlignmentLabeler(content_json_path=args.content_json, tex_path=args.tex, graph_path=args.graph)
    nodes = labeler.parse_tex_nodes()
    framework: list[FrameworkNode] = []
    for position, node in enumerate(nodes):
        if node.node_type not in STRUCTURAL_TYPES:
            continue
        framework.append(
            FrameworkNode(
                node_type=node.node_type,
                clean_text=node.clean_text,
                text=collapse_ws(node.text),
                source_name=node.source_name,
                position=position,
            )
        )
    return framework


def extract_source_framework(tex_path: Path) -> list[FrameworkNode]:
    flattened = flatten_latex_file(tex_path, config=LatexFlattenerConfig(mask_math=False))
    tex = document_body(flattened.content)
    nodes: list[FrameworkNode] = []
    env_stack: list[str] = []
    index = 0
    while index < len(tex):
        if tex.startswith(r"\[", index):
            end = tex.find(r"\]", index + 2)
            payload = tex[index + 2 : end if end >= 0 else len(tex)]
            nodes.append(make_node("equation_display", payload, "[", index, equation=True))
            index = (end + 2) if end >= 0 else len(tex)
            continue
        if tex.startswith("$$", index):
            end = tex.find("$$", index + 2)
            payload = tex[index + 2 : end if end >= 0 else len(tex)]
            nodes.append(make_node("equation_display", payload, "$$", index, equation=True))
            index = (end + 2) if end >= 0 else len(tex)
            continue
        if tex[index] != "\\":
            index += 1
            continue
        command_start = index
        command, index = read_command(tex, index)
        if not command:
            index = command_start + 1
            continue
        if command == "begin":
            env, index = read_required_argument(tex, index)
            if not env:
                continue
            env_stack.append(env)
            if env in LIST_ENV_NAMES:
                nodes.append(make_node("list_container", env, env, command_start))
            elif env in DISPLAY_MATH_ENV_NAMES:
                nodes.append(make_node("equation_display", env, env, command_start, equation=True))
            elif env in {"algorithm", "algorithm2e"} or (env == "algorithmic" and not in_algorithm_env(env_stack)):
                nodes.append(make_node("algorithm", env, env, command_start))
            continue
        if command == "end":
            env, index = read_required_argument(tex, index)
            if env:
                pop_env(env_stack, env)
            continue
        if command in SECTION_COMMANDS:
            index = skip_optional_arguments(tex, skip_ws(tex, index))
            title, index = read_required_argument(tex, index)
            if title:
                nodes.append(make_node("section", title, command, command_start))
            continue
        if command == "item":
            nodes.append(make_node("list_item", "", "item", command_start))
            index = skip_optional_arguments(tex, skip_ws(tex, index))
            continue
        if command == "caption":
            index = skip_optional_arguments(tex, skip_ws(tex, index))
            caption, index = read_required_argument(tex, index)
            caption_env = current_caption_env(env_stack)
            if caption_env in CAPTION_ENVS:
                caption_type = "table_caption" if caption_env in {"table", "table*"} else "figure_caption"
                nodes.append(make_node(caption_type, caption or "", "caption", command_start))
            continue
    return nodes


def compare_frameworks(source_nodes: list[FrameworkNode], parser_nodes: list[FrameworkNode]) -> dict[str, Any]:
    source_sections = [node for node in source_nodes if node.node_type == "section"]
    parser_sections = [node for node in parser_nodes if node.node_type == "section"]
    section_mismatches = []
    for idx in range(max(len(source_sections), len(parser_sections))):
        source = source_sections[idx] if idx < len(source_sections) else None
        parser = parser_sections[idx] if idx < len(parser_sections) else None
        if source is None or parser is None or source.clean_text != parser.clean_text or source.source_name != parser.source_name:
            section_mismatches.append({"index": idx, "source": node_payload(source), "parser": node_payload(parser)})

    sequence_mismatches = []
    max_len = max(len(source_nodes), len(parser_nodes))
    for idx in range(max_len):
        source = source_nodes[idx] if idx < len(source_nodes) else None
        parser = parser_nodes[idx] if idx < len(parser_nodes) else None
        if equivalent_framework_node(source, parser):
            continue
        sequence_mismatches.append({"index": idx, "source": node_payload(source), "parser": node_payload(parser)})
        if len(sequence_mismatches) >= 20:
            break

    summary = {
        "source_count": len(source_nodes),
        "parser_count": len(parser_nodes),
        "source_counts": dict(Counter(node.node_type for node in source_nodes)),
        "parser_counts": dict(Counter(node.node_type for node in parser_nodes)),
        "source_section_count": len(source_sections),
        "parser_section_count": len(parser_sections),
        "section_mismatch_count": len(section_mismatches),
        "sequence_mismatch_count_prefix": len(sequence_mismatches),
    }
    summary["has_mismatch"] = bool(section_mismatches or sequence_mismatches or summary["source_counts"] != summary["parser_counts"])
    return {
        "summary": summary,
        "section_mismatches": section_mismatches,
        "sequence_mismatches": sequence_mismatches,
        "source_nodes": [asdict(node) for node in source_nodes],
        "parser_nodes": [asdict(node) for node in parser_nodes],
    }


def equivalent_framework_node(left: FrameworkNode | None, right: FrameworkNode | None) -> bool:
    if left is None or right is None:
        return left is right
    if left.node_type != right.node_type:
        return False
    if left.node_type == "equation_display":
        return True
    if left.node_type in {"section", "figure_caption", "table_caption"}:
        return left.clean_text == right.clean_text and left.source_name == right.source_name
    return left.source_name == right.source_name


def make_node(node_type: str, text: str, source_name: str, position: int, *, equation: bool = False) -> FrameworkNode:
    return FrameworkNode(
        node_type=node_type,
        clean_text=clean_equation_text(text) if equation else clean_text(text),
        text=collapse_ws(text),
        source_name=source_name,
        position=position,
    )


def node_payload(node: FrameworkNode | None) -> dict[str, Any] | None:
    return asdict(node) if node is not None else None


def document_body(tex: str) -> str:
    begin = re.search(r"\\begin\s*\{document\}", tex)
    if not begin:
        return tex
    end = re.search(r"\\end\s*\{document\}", tex[begin.end() :])
    if not end:
        return tex[begin.end() :]
    return tex[begin.end() : begin.end() + end.start()]


def read_command(tex: str, index: int) -> tuple[str | None, int]:
    if index >= len(tex) or tex[index] != "\\":
        return None, index
    index += 1
    start = index
    while index < len(tex) and tex[index].isalpha():
        index += 1
    if start == index:
        command = tex[index] if index < len(tex) else None
        return command, index + 1
    command = tex[start:index]
    if index < len(tex) and tex[index] == "*":
        index += 1
    return command, index


def read_required_argument(tex: str, index: int) -> tuple[str | None, int]:
    index = skip_ws(tex, index)
    if index >= len(tex) or tex[index] != "{":
        return None, index
    depth = 0
    start = index + 1
    while index < len(tex):
        char = tex[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return tex[start:index], index + 1
        index += 1
    return tex[start:], len(tex)


def skip_optional_arguments(tex: str, index: int) -> int:
    while True:
        index = skip_ws(tex, index)
        if index >= len(tex) or tex[index] != "[":
            return index
        depth = 1
        index += 1
        while index < len(tex) and depth:
            if tex[index] == "\\":
                index += 2
                continue
            if tex[index] == "[":
                depth += 1
            elif tex[index] == "]":
                depth -= 1
            index += 1


def skip_ws(tex: str, index: int) -> int:
    while index < len(tex) and tex[index].isspace():
        index += 1
    return index


def pop_env(env_stack: list[str], env: str) -> None:
    while env_stack:
        current = env_stack.pop()
        if current == env:
            return


def current_caption_env(env_stack: list[str]) -> str | None:
    for env in reversed(env_stack):
        if env in CAPTION_ENVS:
            return env
    return None


def in_algorithm_env(env_stack: list[str]) -> bool:
    return any(env in {"algorithm", "algorithm2e"} for env in env_stack)


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


if __name__ == "__main__":
    raise SystemExit(main())
