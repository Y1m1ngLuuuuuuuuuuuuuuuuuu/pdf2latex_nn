"""Canonical public render surface for backend generation.

Production generation should enter here (or directly call
``OriginalLikeIRLatexRenderer``). Shared escaping/math/list/float helpers live
in ``latex_helpers``; the deprecated tree renderer is not part of production.
"""

from __future__ import annotations

from pathlib import Path

from src.generation.citations import CitationResolution, CitationResolver
from src.generation.ir_renderer import IRLatexRenderConfig, OriginalLikeIRLatexRenderer
from src.generation.source_float_layout import resolve_source_float_layout
from src.generation.style_profile import StyleProfileExtractor
from src.ir import DocumentIR, RenderTreeIR, StyleProfile


def render_original_like_document(
    document: DocumentIR,
    tree: RenderTreeIR,
    *,
    style: StyleProfile | None = None,
    citations: CitationResolution | None = None,
    config: IRLatexRenderConfig | None = None,
    resolve_citations: bool = True,
    source_tex_path: str | Path | None = None,
) -> str:
    """Render a document through the stable IR-only backend.

    This is the preferred one-call API for scripts: it computes missing style and
    citation sidecars, then delegates to ``OriginalLikeIRLatexRenderer``.
    """

    active_style = style or StyleProfileExtractor().extract(document)
    active_citations = citations
    if active_citations is None and resolve_citations:
        active_citations = CitationResolver().resolve_document(document, source_tex_path=source_tex_path)
    source_float_layout = resolve_source_float_layout(source_tex_path)
    return OriginalLikeIRLatexRenderer(config).render(
        document,
        tree,
        active_style,
        active_citations,
        source_float_layout=source_float_layout,
    )
