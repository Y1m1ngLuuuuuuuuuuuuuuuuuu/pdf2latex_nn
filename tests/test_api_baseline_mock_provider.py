from __future__ import annotations

from pathlib import Path

from tools.api_baselines.providers import APIRequest, get_provider


def test_mock_provider_deterministic_output():
    provider = get_provider("mock")
    response = provider.generate(
        APIRequest(
            prompt="Reconstruct this window.",
            model="mock",
            metadata={"doc_id": "doc_a", "window_id": "doc_a_p0001_p0002", "pages": [1, 2]},
        )
    )
    assert "\\section{Mock Section}" in response.text
    assert response.usage["mock_calls"] == 1
    assert response.raw["doc_id"] == "doc_a"
