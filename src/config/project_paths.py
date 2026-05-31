"""Portable project path resolution.

Resolution order:
1. Explicit argument passed to a getter.
2. Environment variable.
3. ``config/paths.local.yaml`` or ``PDF2LATEX_CONFIG``.
4. ``config/paths.example.yaml``.
5. Repository-relative default.

The parser intentionally supports only simple ``key: value`` YAML so this
module remains safe in minimal WSL/AutoDL environments without requiring
PyYAML.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


ENV_KEYS = {
    "project_root": "PDF2LATEX_PROJECT_ROOT",
    "data_root": "PDF2LATEX_DATA_ROOT",
    "output_root": "PDF2LATEX_OUTPUT_ROOT",
    "report_root": "PDF2LATEX_REPORT_ROOT",
    "runtime_root": "PDF2LATEX_RUNTIME_ROOT",
    "paper_root": "PDF2LATEX_PAPER_ROOT",
    "autodl_root": "PDF2LATEX_AUTODL_ROOT",
}

DEFAULTS = {
    "project_root": ".",
    "data_root": "data",
    "output_root": "outputs",
    "report_root": "data/09_eval_reports",
    "runtime_root": "data/runtime",
    "paper_root": "../paper",
    "autodl_root": "/root/autodl-tmp/pdf2latex_nn",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def _config_values() -> dict[str, str]:
    repo = _repo_root()
    candidates = []
    if os.environ.get("PDF2LATEX_CONFIG"):
        candidates.append(Path(os.environ["PDF2LATEX_CONFIG"]).expanduser())
    candidates.extend([repo / "config/paths.local.yaml", repo / "config/paths.example.yaml"])
    merged: dict[str, str] = {}
    for candidate in candidates:
        merged.update(_parse_simple_yaml(candidate))
    return merged


def _resolve(key: str, explicit: str | Path | None = None) -> Path:
    repo = _repo_root()
    if explicit is not None:
        raw = str(explicit)
    else:
        env_name = ENV_KEYS[key]
        raw = os.environ.get(env_name) or _config_values().get(key) or DEFAULTS[key]
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo / path
    return path.resolve()


def get_project_root(explicit: str | Path | None = None) -> Path:
    return _resolve("project_root", explicit)


def get_data_root(explicit: str | Path | None = None) -> Path:
    return _resolve("data_root", explicit)


def get_output_root(explicit: str | Path | None = None) -> Path:
    return _resolve("output_root", explicit)


def get_report_root(explicit: str | Path | None = None) -> Path:
    return _resolve("report_root", explicit)


def get_runtime_root(explicit: str | Path | None = None) -> Path:
    return _resolve("runtime_root", explicit)


def get_paper_root(explicit: str | Path | None = None) -> Path:
    return _resolve("paper_root", explicit)


def get_autodl_root(explicit: str | Path | None = None) -> Path:
    return _resolve("autodl_root", explicit)


def resolve_project_path(*parts: str | Path, root: str | Path | None = None) -> Path:
    base = get_project_root(root)
    return base.joinpath(*map(Path, parts)).resolve()


def describe_paths() -> Mapping[str, str]:
    return {
        "project_root": str(get_project_root()),
        "data_root": str(get_data_root()),
        "output_root": str(get_output_root()),
        "report_root": str(get_report_root()),
        "runtime_root": str(get_runtime_root()),
        "paper_root": str(get_paper_root()),
        "autodl_root": str(get_autodl_root()),
    }

