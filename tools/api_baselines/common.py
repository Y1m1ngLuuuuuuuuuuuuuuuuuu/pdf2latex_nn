#!/usr/bin/env python3
"""Shared helpers for API/VLM baseline evaluation tools."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_doc_id(record: dict[str, Any]) -> str:
    for key in ("doc_id", "document_id", "paper_id", "arxiv_id", "id"):
        value = record.get(key)
        if value:
            return str(value)
    for key in ("pdf_path", "original_pdf", "source_pdf"):
        value = record.get(key)
        if value:
            return Path(str(value)).stem
    raise ValueError(f"Cannot infer doc_id from record keys={sorted(record)}")


def normalize_item_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("items", "documents", "records", "samples"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def load_manifest_items(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    items = normalize_item_list(data)
    for item in items:
        item.setdefault("doc_id", stable_doc_id(item))
    return items


def parse_doc_ids(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    parts: list[str] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            parts.append(piece)
    return set(parts) if parts else None


def slice_items(
    items: Iterable[dict[str, Any]],
    *,
    offset: int = 0,
    limit: int | None = None,
    doc_ids: set[str] | None = None,
    sort_by_doc_id: bool = True,
) -> list[dict[str, Any]]:
    selected = list(items)
    if sort_by_doc_id:
        selected.sort(key=lambda item: str(item.get("doc_id") or stable_doc_id(item)))
    if doc_ids is not None:
        selected = [item for item in selected if str(item.get("doc_id") or stable_doc_id(item)) in doc_ids]
    end = None if limit is None else offset + limit
    return selected[offset:end]


def resolve_path(value: str | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def existing_path_from(record: dict[str, Any], keys: tuple[str, ...]) -> Path | None:
    for key in keys:
        path = resolve_path(str(record[key])) if record.get(key) else None
        if path and path.exists():
            return path
    return None


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "doc"


def infer_source_format(path: Path, explicit: str = "auto") -> str:
    if explicit != "auto":
        return explicit
    suffix = path.suffix.lower()
    if suffix in {".md", ".mmd", ".markdown"}:
        return "markdown"
    if suffix in {".tex", ".latex"}:
        return "latex"
    text = path.read_text(encoding="utf-8", errors="replace")[:4096]
    if "\\section" in text or "\\begin{" in text or "\\[" in text:
        return "latex"
    return "markdown"


def require_api_enabled(provider: str, dry_run: bool) -> None:
    if provider == "mock" or dry_run:
        return
    if os.environ.get("ALLOW_API_CALLS") != "1":
        raise RuntimeError("Real API calls are disabled. Set ALLOW_API_CALLS=1 explicitly to run.")

