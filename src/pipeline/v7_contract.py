"""Shared v7 pipeline guards for feature extraction, labeling, and rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


V7_PIPELINE_VERSION = "v7"
V7_CONTENT_SCHEMA_PREFIX = "content_v7"
V7_CONTENT_STYLED_SUFFIX = "_with_styles"
V7_GRAPH_SCHEMA_VERSION = "graph_v7"
DEFAULT_V7_GRAPH_DIR_NAME = "data/06_graph_features_v7"


class V7ContractError(ValueError):
    """Raised when an input belongs to an old or ambiguous pipeline contract."""


def read_json_payload(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_v7_content_json(path: Path, *, require_styles: bool = True) -> dict[str, Any]:
    """Load and validate a MinerU-derived v7 content JSON payload.

    v7 is the only active visual-side contract after MinerU extraction. It must
    preserve MinerU block granularity, carry the v7 reading-order metadata, and
    for graph/label/render stages include PyMuPDF style enrichment.
    """

    payload = read_json_payload(path)
    if not isinstance(payload, dict):
        raise V7ContractError(f"Expected v7 content JSON object with an items list: {path}")
    items = payload.get("items")
    if not isinstance(items, list):
        raise V7ContractError(f"Expected v7 content JSON object with an items list: {path}")

    schema_version = str(payload.get("schema_version") or "")
    if not schema_version.startswith(V7_CONTENT_SCHEMA_PREFIX):
        raise V7ContractError(
            f"Expected schema_version starting with {V7_CONTENT_SCHEMA_PREFIX!r}, "
            f"got {schema_version!r}: {path}"
        )
    if require_styles and not is_v7_styles_payload(payload, path=path):
        raise V7ContractError(
            f"Expected styled v7 content JSON (*_content_list_v7_styles.json or "
            f"schema ending in {V7_CONTENT_STYLED_SUFFIX!r}): {path}"
        )
    return payload


def is_v7_styles_payload(payload: dict[str, Any], *, path: Path | None = None) -> bool:
    schema_version = str(payload.get("schema_version") or "")
    if schema_version.startswith(V7_CONTENT_SCHEMA_PREFIX) and schema_version.endswith(V7_CONTENT_STYLED_SUFFIX):
        return True
    if "style_source_pdf" in payload:
        return True
    if path is not None and path.name.endswith("_content_list_v7_styles.json"):
        return True
    return False


def assert_v7_graph_data(data: Any, graph_path: Path | None = None) -> None:
    """Validate that a PyG Data object was built from the v7 graph contract."""

    graph_schema = str(getattr(data, "graph_schema_version", "") or "")
    pipeline_version = str(getattr(data, "pipeline_version", "") or "")
    source_path = str(getattr(data, "source_path", "") or "")
    if graph_schema == V7_GRAPH_SCHEMA_VERSION and pipeline_version == V7_PIPELINE_VERSION:
        return
    if "_content_list_v7" in source_path:
        return
    where = f" ({graph_path})" if graph_path is not None else ""
    raise V7ContractError(
        f"Expected graph built by {V7_GRAPH_SCHEMA_VERSION}{where}; "
        f"got graph_schema_version={graph_schema!r}, pipeline_version={pipeline_version!r}, "
        f"source_path={source_path!r}"
    )
