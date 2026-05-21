"""Anthropic provider placeholder with lazy dependency error."""

from __future__ import annotations

from .base import APIRequest, APIResponse, BaseProvider


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def generate(self, request: APIRequest) -> APIResponse:  # pragma: no cover - optional
        raise RuntimeError(
            "Anthropic provider is intentionally a lazy stub in this pipeline. "
            "Install/configure the Anthropic SDK and extend AnthropicProvider.generate before real calls."
        )

