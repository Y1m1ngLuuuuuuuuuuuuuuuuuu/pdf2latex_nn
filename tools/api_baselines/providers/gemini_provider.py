"""Gemini provider placeholder with lazy dependency error."""

from __future__ import annotations

from .base import APIRequest, APIResponse, BaseProvider


class GeminiProvider(BaseProvider):
    name = "gemini"

    def generate(self, request: APIRequest) -> APIResponse:  # pragma: no cover - optional
        raise RuntimeError(
            "Gemini provider is intentionally a lazy stub in this pipeline. "
            "Install/configure the Gemini SDK and extend GeminiProvider.generate before real calls."
        )

