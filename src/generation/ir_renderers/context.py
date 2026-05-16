from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.generation.citations import CitationResolution
from src.ir import DocumentNode, RenderTreeNode


@dataclass(frozen=True)
class RenderContext:
    """Shared context passed to role-level renderers.

    ``owner`` is the active ``OriginalLikeIRLatexRenderer`` instance.  The
    first refactor stage keeps mature helper logic on the owner and moves
    dispatch decisions into small renderers.
    """

    owner: Any
    node: RenderTreeNode
    render_nodes: dict[str, RenderTreeNode]
    document_nodes: dict[str, DocumentNode]
    citations: CitationResolution | None
    depth: int
    source_nodes: list[DocumentNode]
    text: str


@dataclass(frozen=True)
class DocumentNodeRenderContext:
    """Shared context passed to block-type renderers."""

    owner: Any
    node: DocumentNode
    citations: CitationResolution | None
    text: str
    strip_leading_list_marker: bool = False
