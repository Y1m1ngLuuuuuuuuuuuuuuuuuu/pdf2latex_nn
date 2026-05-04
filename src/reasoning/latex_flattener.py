"""Defensive LaTeX flattening before TexSoup parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MATH_PLACEHOLDER = "mathplaceholder"
MATH_SENTINEL = "[MATH]"
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{\s*([^}]+?)\s*\}")
BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\s*\{\s*[^}]+\s*\}")
NEWCOMMAND_RE = re.compile(
    r"\\(?:re)?newcommand\*?\s*(?:\{\s*\\([a-zA-Z@]+)\s*\}|\\([a-zA-Z@]+))\s*"
    r"(?!\[[^\]]*\]\s*)\{\s*([^{}]+?)\s*\}",
    re.DOTALL,
)
DEF_RE = re.compile(r"\\def\s*\\([a-zA-Z@]+)\s*\{\s*([^{}]+?)\s*\}", re.DOTALL)
MATH_ENV_RE = re.compile(
    r"\\begin\{(?P<env>"
    r"equation\*?|align\*?|gather\*?|multline\*?|flalign\*?|eqnarray\*?|"
    r"displaymath|math|split|aligned|cases|array|[bBpvV]?matrix"
    r")\}.*?\\end\{(?P=env)\}",
    re.DOTALL,
)
DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.DOTALL)
INLINE_MATH_RE = re.compile(r"\$.*?\$|\\\(.*?\\\)", re.DOTALL)


@dataclass(frozen=True)
class LatexFlattenerConfig:
    max_input_depth: int = 32
    inject_bbl: bool = True
    expand_zero_arg_macros: bool = True
    mask_math: bool = True


@dataclass
class FlattenedLatex:
    content: str
    source_path: Path
    included_files: list[Path] = field(default_factory=list)
    missing_files: list[Path] = field(default_factory=list)
    expanded_macros: dict[str, str] = field(default_factory=dict)
    bbl_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "included_file_count": len(self.included_files),
            "missing_file_count": len(self.missing_files),
            "expanded_macro_count": len(self.expanded_macros),
            "bbl_path": str(self.bbl_path) if self.bbl_path is not None else None,
            "warnings": list(self.warnings),
        }


@dataclass
class _FlattenState:
    config: LatexFlattenerConfig
    included_files: list[Path] = field(default_factory=list)
    missing_files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    visited: set[Path] = field(default_factory=set)


def flatten_latex_file(path: Path, config: LatexFlattenerConfig | None = None) -> FlattenedLatex:
    """Run the full five-step TeX cleanup pipeline for a main TeX file."""

    cfg = config or LatexFlattenerConfig()
    main_path = Path(path).resolve()
    state = _FlattenState(config=cfg)
    state.visited.add(main_path)
    raw = read_text_lossy(main_path)
    tex_1 = strip_comments(raw)
    tex_2 = flatten_inputs(tex_1, main_path.parent, state=state, depth=0)
    tex_3, bbl_path = inject_bbl(tex_2, main_path.parent, main_path.stem) if cfg.inject_bbl else (tex_2, None)
    if bbl_path is not None:
        state.included_files.append(bbl_path)
    if cfg.expand_zero_arg_macros:
        tex_4, macros = expand_simple_macros(tex_3)
    else:
        tex_4, macros = tex_3, {}
    clean_flat_tex = mask_math_environments(tex_4) if cfg.mask_math else tex_4
    return FlattenedLatex(
        content=clean_flat_tex,
        source_path=main_path,
        included_files=state.included_files,
        missing_files=state.missing_files,
        expanded_macros=macros,
        bbl_path=bbl_path,
        warnings=state.warnings,
    )


def create_flat_tex_ast(path: Path, config: LatexFlattenerConfig | None = None) -> Any:
    """Flatten a TeX file and parse the result with TexSoup."""

    from TexSoup import TexSoup

    return TexSoup(flatten_latex_file(path, config=config).content)


def strip_comments(tex_string: str) -> str:
    """Remove unescaped TeX comments while preserving escaped percent signs."""

    return "\n".join(strip_comment_from_line(line) for line in tex_string.splitlines())


def strip_comment_from_line(line: str) -> str:
    for index, char in enumerate(line):
        if char != "%":
            continue
        backslash_count = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            backslash_count += 1
            cursor -= 1
        if backslash_count % 2 == 0:
            return line[:index]
    return line


def flatten_inputs(tex_string: str, current_dir: Path, *, state: _FlattenState | None = None, depth: int = 0) -> str:
    """Recursively replace local input/include commands with file content."""

    cfg = state.config if state is not None else LatexFlattenerConfig()
    if depth >= cfg.max_input_depth:
        if state is not None:
            state.warnings.append(f"max input depth reached at {current_dir}")
        return tex_string

    def replace_input(match: re.Match[str]) -> str:
        candidate = resolve_tex_input_path(match.group(1), current_dir)
        if not candidate.exists():
            if state is not None:
                state.missing_files.append(candidate)
                state.warnings.append(f"missing input file: {candidate}")
            return " "
        resolved = candidate.resolve()
        if state is not None:
            if resolved in state.visited:
                state.warnings.append(f"skipped recursive input loop: {resolved}")
                return " "
            state.visited.add(resolved)
            state.included_files.append(resolved)
        child = strip_comments(read_text_lossy(resolved))
        return flatten_inputs(child, resolved.parent, state=state, depth=depth + 1)

    return INPUT_RE.sub(replace_input, tex_string)


def resolve_tex_input_path(raw_name: str, current_dir: Path) -> Path:
    name = raw_name.strip().strip("\"'")
    candidate = Path(name)
    if not candidate.suffix:
        candidate = candidate.with_suffix(".tex")
    if not candidate.is_absolute():
        candidate = Path(current_dir) / candidate
    return candidate


def inject_bbl(tex_string: str, root_dir: Path, main_filename: str = "main") -> tuple[str, Path | None]:
    """Replace bibliography commands with the compiled .bbl text when available."""

    if not BIBLIOGRAPHY_RE.search(tex_string):
        return tex_string, None
    bbl_path = Path(root_dir) / f"{main_filename}.bbl"
    if not bbl_path.exists():
        return tex_string, None
    bbl_content = strip_comments(read_text_lossy(bbl_path))
    return BIBLIOGRAPHY_RE.sub(lambda _match: bbl_content, tex_string), bbl_path.resolve()


def expand_simple_macros(tex_string: str) -> tuple[str, dict[str, str]]:
    """Expand zero-argument textual newcommand/renewcommand/def aliases only."""

    macros: dict[str, str] = {}

    def collect_newcommand(match: re.Match[str]) -> str:
        alias = match.group(1) or match.group(2)
        value = match.group(3)
        if alias and is_safe_macro_value(value):
            macros[alias] = value.strip()
            return " "
        return match.group(0)

    def collect_def(match: re.Match[str]) -> str:
        alias = match.group(1)
        value = match.group(2)
        if alias and is_safe_macro_value(value):
            macros[alias] = value.strip()
            return " "
        return match.group(0)

    tex_without_defs = NEWCOMMAND_RE.sub(collect_newcommand, tex_string)
    tex_without_defs = DEF_RE.sub(collect_def, tex_without_defs)
    for alias, value in macros.items():
        tex_without_defs = re.sub(rf"\\{re.escape(alias)}(?![a-zA-Z@])", lambda _match, replacement=value: replacement, tex_without_defs)
    return tex_without_defs, macros


def is_safe_macro_value(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if any(token in stripped for token in ("\\", "#", "{", "}", "\n\n")):
        return False
    return True


def mask_math_environments(tex_string: str) -> str:
    """Replace math spans/environments with a stable placeholder token."""

    masked = MATH_ENV_RE.sub(f" {MATH_SENTINEL} ", tex_string)
    masked = DISPLAY_MATH_RE.sub(f" {MATH_SENTINEL} ", masked)
    return INLINE_MATH_RE.sub(f" {MATH_SENTINEL} ", masked)


def read_text_lossy(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")
