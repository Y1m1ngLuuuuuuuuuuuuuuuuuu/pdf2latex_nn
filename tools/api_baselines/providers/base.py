"""Provider protocol for API/VLM baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class APIRequest:
    prompt: str
    image_paths: list[Path] = field(default_factory=list)
    pdf_path: Path | None = None
    model: str = "mock"
    temperature: float = 0.0
    max_output_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


class BaseProvider:
    name = "base"

    def generate(self, request: APIRequest) -> APIResponse:  # pragma: no cover - interface
        raise NotImplementedError

