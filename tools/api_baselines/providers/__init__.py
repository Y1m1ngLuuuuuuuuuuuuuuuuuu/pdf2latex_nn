"""Provider factory for API/VLM baselines."""

from __future__ import annotations

from .base import APIRequest, APIResponse, BaseProvider
from .mock_provider import MockProvider


def get_provider(name: str) -> BaseProvider:
    normalized = name.lower().strip()
    if normalized == "mock":
        return MockProvider()
    if normalized == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider()
    if normalized == "openai_compatible":
        from .openai_compatible_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider()
    if normalized == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider()
    if normalized == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if normalized == "dashscope":
        from .dashscope_provider import DashscopeProvider

        return DashscopeProvider()
    raise ValueError(f"Unknown provider: {name}")


__all__ = ["APIRequest", "APIResponse", "BaseProvider", "get_provider"]

