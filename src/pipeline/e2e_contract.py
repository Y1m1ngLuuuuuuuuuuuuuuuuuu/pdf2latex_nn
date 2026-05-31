"""Canonical PDF2LaTeX E2E contract types and manifest helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class E2ECaseConfig:
    doc_id: str
    output_dir: Path
    stratum: str = "unknown"
    pdf: Path | None = None
    mineru_output: Path | None = None
    existing_artifact_root: Path | None = None
    gold_comparison: Path | None = None
    existing_facts_path: Path | None = None
    document_ir_path: Path | None = None
    render_tree_ir_path: Path | None = None
    generated_tex_path: Path | None = None
    generated_pdf_path: Path | None = None
    renderer: str = "ir"
    use_existing_mineru: bool = True
    enable_frontmatter_ir_renderer_experimental: bool = False
    enable_float_caption_materialization_experimental: bool = False
    enable_table_safe_fallback_experimental: bool = False
    compile: bool = False
    evaluate: bool = False
    visual_qa: bool = False
    no_tex_source_inference: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def path_or_none(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def case_config_from_manifest_item(item: dict[str, Any], *, output_root: Path, defaults: dict[str, Any]) -> E2ECaseConfig:
    doc_id = str(item["doc_id"])
    output_dir = output_root / doc_id
    return E2ECaseConfig(
        doc_id=doc_id,
        output_dir=output_dir,
        stratum=str(item.get("stratum") or "unknown"),
        pdf=path_or_none(item.get("original_pdf_path") or item.get("pdf")),
        mineru_output=path_or_none(item.get("mineru_output")),
        existing_artifact_root=path_or_none(item.get("artifact_root") or item.get("existing_artifact_root")),
        gold_comparison=path_or_none(item.get("gold_comparison_path") or item.get("gold_comparison")),
        existing_facts_path=path_or_none(item.get("existing_facts_path")),
        document_ir_path=path_or_none(item.get("document_ir_path")),
        render_tree_ir_path=path_or_none(item.get("render_tree_ir_path")),
        generated_tex_path=path_or_none(item.get("generated_tex_path")),
        generated_pdf_path=path_or_none(item.get("generated_pdf_path")),
        renderer=str(defaults.get("renderer") or "ir"),
        use_existing_mineru=bool(defaults.get("use_existing_mineru", True)),
        enable_frontmatter_ir_renderer_experimental=bool(defaults.get("enable_frontmatter_ir_renderer_experimental", False)),
        enable_float_caption_materialization_experimental=bool(defaults.get("enable_float_caption_materialization_experimental", False)),
        enable_table_safe_fallback_experimental=bool(defaults.get("enable_table_safe_fallback_experimental", False)),
        compile=bool(defaults.get("compile", False)),
        evaluate=bool(defaults.get("evaluate", False)),
        visual_qa=bool(defaults.get("visual_qa", False)),
        no_tex_source_inference=bool(defaults.get("no_tex_source_inference", True)),
    )


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("cases", "items", "documents"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported E2E manifest shape: {path}")
