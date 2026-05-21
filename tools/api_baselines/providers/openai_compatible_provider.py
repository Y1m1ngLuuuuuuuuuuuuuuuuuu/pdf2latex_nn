"""OpenAI-compatible provider wrapper."""

from __future__ import annotations

import os

from .base import APIResponse
from .openai_provider import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    name = "openai_compatible"

    def generate(self, request):  # type: ignore[override]
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("OpenAI SDK is not installed. Install `openai` to use provider=openai_compatible.") from exc
        # Reuse OpenAIProvider encoding logic by temporarily constructing a client here.
        import base64
        import mimetypes

        def image_data_url(path):
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")

        client = OpenAI(
            api_key=os.environ.get("OPENAI_COMPATIBLE_API_KEY"),
            base_url=os.environ.get("OPENAI_COMPATIBLE_BASE_URL"),
        )
        content = [{"type": "input_text", "text": request.prompt}]
        for image_path in request.image_paths:
            content.append({"type": "input_image", "image_url": image_data_url(image_path)})
        kwargs = {"model": request.model, "input": [{"role": "user", "content": content}], "temperature": request.temperature}
        if request.max_output_tokens:
            kwargs["max_output_tokens"] = request.max_output_tokens
        response = client.responses.create(**kwargs)
        return APIResponse(
            text=getattr(response, "output_text", "") or "",
            raw=response.model_dump(mode="json"),
            usage=getattr(response, "usage", {}) or {},
        )
