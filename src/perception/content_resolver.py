"""Resolve the best v7 content JSON for a document across multiple roots."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class V7SchemaThresholds:
    min_layout_layer_coverage: float = 0.90
    min_layout_role_coverage: float = 0.90
    min_canonical_type_coverage: float = 0.00
    require_v7_schema_fields: bool = True
    allow_stale_v7_content: bool = False


@dataclass(frozen=True)
class V7ContentMetrics:
    path: str
    exists: bool
    non_empty_items: bool
    content_page_count: int | None
    raw_pdf_page_count: int | None
    page_count_match: bool | None
    item_count: int
    layout_layer_coverage: float
    layout_role_coverage: float
    canonical_type_coverage: float
    style_spans_coverage: float
    has_v7_node_ids: bool
    stale_schema_flag: bool
    failed_reasons: tuple[str, ...]
    mtime: float | None = None
    parent_dir: str | None = None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContentResolverConfig:
    mineru_output_dirs: tuple[Path, ...]
    thresholds: V7SchemaThresholds = field(default_factory=V7SchemaThresholds)
    force_refresh_v7_conversion: bool = False


@dataclass(frozen=True)
class ContentResolution:
    selected_path: Path | None
    selected_metrics: V7ContentMetrics | None
    candidates: tuple[V7ContentMetrics, ...]
    skip_reason: str | None

    def to_report(self) -> dict[str, Any]:
        return {
            "selected_content_json_path": str(self.selected_path) if self.selected_path else None,
            "selected_metrics": self.selected_metrics.to_dict() if self.selected_metrics else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "skip_reason": self.skip_reason,
        }


def resolve_v7_content_for_doc(
    document_id: str,
    *,
    raw_pdf_path: Path | None,
    config: ContentResolverConfig,
) -> ContentResolution:
    candidates = tuple(
        sorted(
            (
                score_candidate(path, raw_pdf_path=raw_pdf_path, thresholds=config.thresholds)
                for path in enumerate_v7_content_candidates(document_id, config.mineru_output_dirs)
            ),
            key=lambda metrics: metrics.score,
            reverse=True,
        )
    )
    valid = [candidate for candidate in candidates if not candidate.failed_reasons]
    if valid:
        selected = valid[0]
        return ContentResolution(Path(selected.path), selected, candidates, None)
    if config.force_refresh_v7_conversion:
        return ContentResolution(None, None, candidates, "force_refresh_v7_conversion_required")
    return ContentResolution(None, None, candidates, "stale_or_missing_v7_schema")


def enumerate_v7_content_candidates(document_id: str, roots: tuple[Path, ...]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    names = (
        f"{document_id}_content_list_v7_styles.json",
        f"{document_id}_content_list_v7.json",
    )
    for root in roots:
        if not root:
            continue
        root = Path(root)
        # MinerU/v7 roots in this project are organized as:
        #   <root>/<doc_id>/auto/<doc_id>_content_list_v7_styles.json
        # Keep lookup bounded per document. Recursive scans over
        # data/02_mineru_outputs are too expensive and can accidentally prefer
        # stale mixed directories.
        direct_candidates = [
            root / document_id / "auto" / name for name in names
        ] + [
            root / document_id / name for name in names
        ]
        for path in direct_candidates:
            if path.exists() and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def score_candidate(path: Path, *, raw_pdf_path: Path | None, thresholds: V7SchemaThresholds) -> V7ContentMetrics:
    metrics = validate_content_v7_schema(path, raw_pdf_path=raw_pdf_path, thresholds=thresholds)
    directory_priority = directory_priority_score(path)
    recency_bonus = min(1.0, (metrics.mtime or 0.0) / 2_000_000_000.0)
    page_bonus = 1.0 if metrics.page_count_match is True else 0.0
    score = (
        3.0 * metrics.layout_layer_coverage
        + 3.0 * metrics.layout_role_coverage
        + 1.0 * metrics.canonical_type_coverage
        + 1.0 * metrics.style_spans_coverage
        + page_bonus
        + directory_priority
        + recency_bonus
    )
    return V7ContentMetrics(**{**metrics.to_dict(), "score": score})


def validate_content_v7_schema(
    content_json: Path,
    *,
    raw_pdf_path: Path | None = None,
    thresholds: V7SchemaThresholds | None = None,
) -> V7ContentMetrics:
    thresholds = thresholds or V7SchemaThresholds()
    failed: list[str] = []
    if not content_json.exists():
        return V7ContentMetrics(
            path=str(content_json),
            exists=False,
            non_empty_items=False,
            content_page_count=None,
            raw_pdf_page_count=pdf_page_count(raw_pdf_path),
            page_count_match=None,
            item_count=0,
            layout_layer_coverage=0.0,
            layout_role_coverage=0.0,
            canonical_type_coverage=0.0,
            style_spans_coverage=0.0,
            has_v7_node_ids=False,
            stale_schema_flag=True,
            failed_reasons=("content_json_missing",),
            mtime=None,
            parent_dir=str(content_json.parent),
        )
    try:
        payload = json.loads(content_json.read_text(encoding="utf-8"))
    except Exception:
        failed.append("content_json_invalid")
        payload = {}
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        items = []
    item_count = len(items)
    if item_count == 0:
        failed.append("content_json_empty")
    content_pages = content_page_count(items, payload if isinstance(payload, dict) else {})
    pdf_pages = pdf_page_count(raw_pdf_path)
    page_match = None if pdf_pages is None or content_pages is None else content_pages == pdf_pages
    if page_match is False:
        failed.append("content_page_count_mismatch")

    layer_cov = coverage(items, "layout_layer")
    role_cov = coverage(items, "layout_role")
    canonical_cov = coverage(items, "canonical_type")
    style_cov = style_coverage(items)
    has_ids = any(bool(item.get("_v7_node_id") or item.get("id")) for item in items if isinstance(item, dict))
    if thresholds.require_v7_schema_fields:
        if layer_cov < thresholds.min_layout_layer_coverage:
            failed.append("stale_v7_schema_missing_layout_layer")
        if role_cov < thresholds.min_layout_role_coverage:
            failed.append("stale_v7_schema_missing_layout_role")
        if canonical_cov < thresholds.min_canonical_type_coverage:
            failed.append("stale_v7_schema_missing_canonical_type")
    stale = any(reason.startswith("stale_v7_schema") for reason in failed)
    if thresholds.allow_stale_v7_content:
        failed = [reason for reason in failed if not reason.startswith("stale_v7_schema")]
    return V7ContentMetrics(
        path=str(content_json),
        exists=True,
        non_empty_items=item_count > 0,
        content_page_count=content_pages,
        raw_pdf_page_count=pdf_pages,
        page_count_match=page_match,
        item_count=item_count,
        layout_layer_coverage=layer_cov,
        layout_role_coverage=role_cov,
        canonical_type_coverage=canonical_cov,
        style_spans_coverage=style_cov,
        has_v7_node_ids=has_ids,
        stale_schema_flag=stale,
        failed_reasons=tuple(dict.fromkeys(failed)),
        mtime=content_json.stat().st_mtime,
        parent_dir=str(content_json.parent),
    )


def coverage(items: list[Any], key: str) -> float:
    dict_items = [item for item in items if isinstance(item, dict)]
    if not dict_items:
        return 0.0
    return sum(1 for item in dict_items if item.get(key) not in (None, "")) / len(dict_items)


def style_coverage(items: list[Any]) -> float:
    dict_items = [item for item in items if isinstance(item, dict)]
    if not dict_items:
        return 0.0
    return sum(1 for item in dict_items if item.get("style_spans") or item.get("spans")) / len(dict_items)


def content_page_count(items: list[Any], payload: dict[str, Any]) -> int | None:
    for key in ("page_count", "num_pages", "pages"):
        value = payload.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, list) and value:
            return len(value)
    pages = [item.get("page_idx") for item in items if isinstance(item, dict) and isinstance(item.get("page_idx"), int)]
    if not pages:
        return None
    return max(pages) + 1


def pdf_page_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        import fitz

        with fitz.open(path) as doc:
            return int(doc.page_count)
    except Exception:
        return None


def directory_priority_score(path: Path) -> float:
    text = str(path).casefold()
    score = 0.0
    if "tocfilter" in text:
        score += 3.0
    if "frontmatterfix" in text:
        score += 2.5
    if "layerband" in text:
        score += 2.0
    if "v7" in text:
        score += 1.0
    if "/mineru_output/" in text or text.endswith("/mineru_output"):
        score -= 1.0
    return score
