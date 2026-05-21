"""DashScope provider placeholder with lazy dependency error."""

from __future__ import annotations

from .base import APIRequest, APIResponse, BaseProvider


class DashscopeProvider(BaseProvider):
    name = "dashscope"

    def generate(self, request: APIRequest) -> APIResponse:  # pragma: no cover - optional
        raise RuntimeError(
            "DashScope provider is intentionally a lazy stub in this pipeline. "
            "Install/configure DashScope and extend DashscopeProvider.generate before real calls."
        )

