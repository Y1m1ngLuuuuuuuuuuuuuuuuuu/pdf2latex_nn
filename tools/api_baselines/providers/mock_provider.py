"""Deterministic mock provider used for dry-run and CI."""

from __future__ import annotations

from .base import APIRequest, APIResponse, BaseProvider


class MockProvider(BaseProvider):
    name = "mock"

    def generate(self, request: APIRequest) -> APIResponse:
        doc_id = request.metadata.get("doc_id", "mock_doc")
        window_id = request.metadata.get("window_id", "full")
        pages = request.metadata.get("pages", [])
        body = [
            f"% MOCK_PROVIDER_OUTPUT doc_id={doc_id} window_id={window_id}",
            "\\section{Mock Section}",
            f"This is deterministic mock content for pages {pages}.",
            "\\begin{figure}[H]",
            "\\centering",
            "% MOCK_FIGURE_PLACEHOLDER",
            "\\caption{Mock figure caption}",
            "\\label{fig:figure_1}",
            "\\end{figure}",
        ]
        return APIResponse(
            text="\n".join(body) + "\n",
            raw={"provider": self.name, "model": request.model, "doc_id": doc_id, "window_id": window_id},
            usage={"mock_calls": 1, "input_images": len(request.image_paths), "input_pdf": bool(request.pdf_path)},
        )

