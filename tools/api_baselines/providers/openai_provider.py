"""OpenAI provider wrapper.

Real calls are guarded by ALLOW_API_CALLS in the caller.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from .base import APIRequest, APIResponse, BaseProvider


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class OpenAIProvider(BaseProvider):
    name = "openai"

    def generate(self, request: APIRequest) -> APIResponse:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("OpenAI SDK is not installed. Install `openai` to use provider=openai.") from exc
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        content: list[dict[str, object]] = [{"type": "input_text", "text": request.prompt}]
        for image_path in request.image_paths:
            content.append({"type": "input_image", "image_url": _image_data_url(image_path)})
        if request.pdf_path:
            content.append({"type": "input_text", "text": f"[PDF input path available to runner: {request.pdf_path.name}]"})
        kwargs = {"model": request.model, "input": [{"role": "user", "content": content}], "temperature": request.temperature}
        if request.max_output_tokens:
            kwargs["max_output_tokens"] = request.max_output_tokens
        response = client.responses.create(**kwargs)
        text = getattr(response, "output_text", "") or ""
        return APIResponse(text=text, raw=response.model_dump(mode="json"), usage=getattr(response, "usage", {}) or {})

