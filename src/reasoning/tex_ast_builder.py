"""Build a path-encoded TeX structural AST for relation labeling."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TEX_AST_SCHEMA_VERSION = "tex_ast_path_v0"
ROOT_ID = "ROOT"
SECTION_LEVELS = {
    "part": 0,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
    "paragraph": 5,
    "subparagraph": 6,
}
STRUCTURAL_RE = re.compile(
    r"\\(?P<begin>begin)\s*\{(?P<begin_env>[^}]+)\}"
    r"|\\(?P<end>end)\s*\{(?P<end_env>[^}]+)\}"
    r"|\\(?P<section>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\s*(?:\[[^\]]*\])?\s*\{"
    r"|\\(?P<item>item)\b(?:\s*\[[^\]]*\])?",
    re.DOTALL,
)
COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])?(?:\s*\{([^{}]*)\})?")


@dataclass(frozen=True)
class TexNode:
    tex_id: str
    node_type: str
    parent_id: str | None
    path: tuple[str, ...]
    child_index: int
    text: str
    source_span: tuple[int, int]
    command: str | None = None
    env_name: str | None = None


def build_tex_ast(tex: str, *, document_id: str | None = None) -> dict[str, Any]:
    """Parse TeX into path-encoded structural nodes.

    The parser is intentionally lightweight. It captures section commands,
    environments, list items, and paragraph text while preserving source spans
    and absolute paths for O(depth) relation labeling.
    """

    masked = mask_comments(tex)
    builder = _TexAstBuilder(masked, document_id=document_id)
    builder.build()
    return builder.to_payload()


def build_tex_ast_from_file(path: Path) -> dict[str, Any]:
    return build_tex_ast(path.read_text(encoding="utf-8"), document_id=path.stem)


def write_tex_ast(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = build_tex_ast_from_file(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def tex_nodes_by_id(payload_or_nodes: dict[str, Any] | list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if isinstance(payload_or_nodes, dict):
        nodes = payload_or_nodes.get("nodes", [])
    else:
        nodes = payload_or_nodes
    return {str(node["tex_id"]): node for node in nodes if isinstance(node, dict) and node.get("tex_id")}


class _TexAstBuilder:
    def __init__(self, tex: str, *, document_id: str | None) -> None:
        self.tex = tex
        self.document_id = document_id
        self.nodes: list[TexNode] = []
        self.next_id = 1
        self.child_counts: dict[str, int] = {ROOT_ID: 0}
        self.section_by_level: dict[int, str] = {}
        self.context_stack: list[tuple[str, str]] = []

    def build(self) -> None:
        previous = 0
        for token in iter_structural_tokens(self.tex):
            self.emit_paragraphs(previous, token["start"])
            kind = token["kind"]
            if kind == "section":
                self.handle_section(token)
            elif kind == "begin":
                self.handle_begin(token)
            elif kind == "end":
                self.handle_end(token)
            elif kind == "item":
                self.handle_item(token)
            previous = int(token["end"])
        self.emit_paragraphs(previous, len(self.tex))

    def to_payload(self) -> dict[str, Any]:
        nodes = [asdict(node) for node in self.nodes]
        for node in nodes:
            node["path"] = list(node["path"])
            node["source_span"] = list(node["source_span"])
        return {
            "schema_version": TEX_AST_SCHEMA_VERSION,
            "document_id": self.document_id,
            "root_id": ROOT_ID,
            "nodes": nodes,
            "node_count": len(nodes),
        }

    def handle_section(self, token: dict[str, Any]) -> None:
        command = str(token["command"])
        level = SECTION_LEVELS[command]
        parent_id = self.find_section_parent(level)
        self.context_stack.clear()
        node_id = self.new_node(
            node_type=command,
            parent_id=parent_id,
            text=str(token.get("text") or ""),
            source_span=(int(token["start"]), int(token["end"])),
            command=command,
        )
        self.section_by_level = {key: value for key, value in self.section_by_level.items() if key < level}
        self.section_by_level[level] = node_id

    def handle_begin(self, token: dict[str, Any]) -> None:
        env_name = str(token["env_name"])
        parent_id = self.current_parent_id()
        node_id = self.new_node(
            node_type="environment",
            parent_id=parent_id,
            text=env_name,
            source_span=(int(token["start"]), int(token["end"])),
            env_name=env_name,
        )
        self.context_stack.append(("environment", node_id))

    def handle_end(self, token: dict[str, Any]) -> None:
        env_name = str(token["env_name"])
        while self.context_stack:
            kind, node_id = self.context_stack.pop()
            if kind == "environment" and self.node_by_id(node_id).env_name == env_name:
                break

    def handle_item(self, token: dict[str, Any]) -> None:
        while self.context_stack and self.context_stack[-1][0] == "item":
            self.context_stack.pop()
        parent_id = self.current_parent_id()
        node_id = self.new_node(
            node_type="item",
            parent_id=parent_id,
            text="",
            source_span=(int(token["start"]), int(token["end"])),
            command="item",
        )
        self.context_stack.append(("item", node_id))

    def emit_paragraphs(self, start: int, end: int) -> None:
        if end <= start:
            return
        raw = self.tex[start:end]
        cursor = start
        for match in re.finditer(r"\n\s*\n+", raw):
            self.emit_paragraph(cursor, start + match.start())
            cursor = start + match.end()
        self.emit_paragraph(cursor, end)

    def emit_paragraph(self, start: int, end: int) -> None:
        text = clean_latex_text(self.tex[start:end])
        if not text:
            return
        self.new_node(
            node_type="paragraph_text",
            parent_id=self.current_parent_id(),
            text=text,
            source_span=(start, end),
        )

    def current_parent_id(self) -> str:
        if self.context_stack:
            return self.context_stack[-1][1]
        if self.section_by_level:
            return self.section_by_level[max(self.section_by_level)]
        return ROOT_ID

    def find_section_parent(self, level: int) -> str:
        lower = [key for key in self.section_by_level if key < level]
        if not lower:
            return ROOT_ID
        return self.section_by_level[max(lower)]

    def new_node(
        self,
        *,
        node_type: str,
        parent_id: str,
        text: str,
        source_span: tuple[int, int],
        command: str | None = None,
        env_name: str | None = None,
    ) -> str:
        tex_id = f"T_{self.next_id}"
        self.next_id += 1
        child_index = self.child_counts.get(parent_id, 0)
        self.child_counts[parent_id] = child_index + 1
        parent_path = (ROOT_ID,) if parent_id == ROOT_ID else self.node_by_id(parent_id).path
        path = (*parent_path, tex_id)
        self.nodes.append(
            TexNode(
                tex_id=tex_id,
                node_type=node_type,
                parent_id=None if parent_id == ROOT_ID else parent_id,
                path=path,
                child_index=child_index,
                text=text,
                source_span=source_span,
                command=command,
                env_name=env_name,
            )
        )
        self.child_counts[tex_id] = 0
        return tex_id

    def node_by_id(self, tex_id: str) -> TexNode:
        for node in reversed(self.nodes):
            if node.tex_id == tex_id:
                return node
        raise KeyError(tex_id)


def iter_structural_tokens(tex: str) -> list[dict[str, Any]]:
    tokens = []
    for match in STRUCTURAL_RE.finditer(tex):
        if match.group("begin"):
            tokens.append(
                {
                    "kind": "begin",
                    "env_name": match.group("begin_env"),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        elif match.group("end"):
            tokens.append(
                {
                    "kind": "end",
                    "env_name": match.group("end_env"),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        elif match.group("section"):
            brace_start = match.end() - 1
            brace_end = find_matching_brace(tex, brace_start)
            if brace_end is None:
                continue
            tokens.append(
                {
                    "kind": "section",
                    "command": match.group("section"),
                    "text": clean_latex_text(tex[brace_start + 1 : brace_end]),
                    "start": match.start(),
                    "end": brace_end + 1,
                }
            )
        elif match.group("item"):
            tokens.append(
                {
                    "kind": "item",
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return sorted(tokens, key=lambda token: (int(token["start"]), int(token["end"])))


def find_matching_brace(tex: str, open_index: int) -> int | None:
    if open_index >= len(tex) or tex[open_index] != "{":
        return None
    depth = 0
    escaped = False
    for index in range(open_index, len(tex)):
        char = tex[index]
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
            if depth == 0:
                return index
    return None


def clean_latex_text(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", " ", text)
    text = COMMAND_RE.sub(lambda match: match.group(1) or " ", text)
    text = text.replace("~", " ")
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mask_comments(tex: str) -> str:
    lines = []
    for line in tex.splitlines(keepends=True):
        comment_start = find_comment_start(line)
        if comment_start is None:
            lines.append(line)
            continue
        newline = ""
        body_end = len(line)
        if line.endswith("\r\n"):
            newline = "\r\n"
            body_end -= 2
        elif line.endswith("\n"):
            newline = "\n"
            body_end -= 1
        lines.append(line[:comment_start] + " " * (body_end - comment_start) + newline)
    return "".join(lines)


def find_comment_start(line: str) -> int | None:
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "%":
            return index
    return None
