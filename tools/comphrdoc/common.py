"""Shared helpers for the CompHRDoc bridge scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


CLASS_MAP_TO_OFFICIAL = {
    "para": "paraline",
    "sec1": "section",
    "sec2": "section",
    "sec3": "section",
    "secx": "section",
    "tab": "table",
    "fig": "figure",
    "tabcap": "caption",
    "figcap": "caption",
    "equ": "equation",
    "alg": "paraline",
    "foot": "footer",
    "fnote": "footnote",
    "background": "table",
}

OFFICIAL_CLASSES = {
    "title",
    "author",
    "mail",
    "affili",
    "section",
    "fstline",
    "paraline",
    "table",
    "figure",
    "caption",
    "equation",
    "footer",
    "header",
    "footnote",
}


def load_config(path: Path) -> dict[str, Any]:
    """Load a small YAML config without making PyYAML mandatory."""

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping config: {path}")
        return data
    except ModuleNotFoundError:
        return _load_simple_yaml(path)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, sep, value = raw_line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = parse_scalar(value.strip())
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def config_path(config: dict[str, Any], section: str, key: str) -> Path:
    value = config.get(section, {}).get(key)
    if not value:
        raise KeyError(f"Missing config path {section}.{key}")
    return Path(str(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_class(value: str | None) -> str:
    raw = str(value or "paraline").strip()
    mapped = CLASS_MAP_TO_OFFICIAL.get(raw, raw)
    return mapped if mapped in OFFICIAL_CLASSES else "paraline"


def doc_id_from_json(path: Path) -> str:
    return path.stem


def natural_page_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**9, path.name)


def safe_doc_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "document"


def bbox_iou(a: list[float] | list[int], b: list[float] | list[int]) -> float:
    ax0, ay0, ax1, ay1 = map(float, a[:4])
    bx0, by0, bx1, by1 = map(float, b[:4])
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(area_a + area_b - inter, 1e-6)


def bbox_center(box: list[float] | list[int]) -> tuple[float, float]:
    x0, y0, x1, y1 = map(float, box[:4])
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def text_similarity(a: str, b: str) -> float:
    import difflib

    clean_a = re.sub(r"\W+", "", str(a or "").casefold())
    clean_b = re.sub(r"\W+", "", str(b or "").casefold())
    if not clean_a and not clean_b:
        return 1.0
    if not clean_a or not clean_b:
        return 0.0
    return difflib.SequenceMatcher(a=clean_a, b=clean_b).ratio()
