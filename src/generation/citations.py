"""Citation and bibliography repair for LaTeX generation.

The frontend OCR sees citation labels as visible text, but LaTeX should emit
semantic citations and bibliography items.  This module keeps that repair
separate from tree decoding and low-level rendering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.ir import BlockType, DocumentIR, DocumentNode


REFERENCE_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"[\[\(【]\s*(?P<bracket>\d{1,4})\s*[\]\)】]"
    r"|(?P<number>\d{1,4})[\.\)]"
    r")\s*"
)
CITATION_MARKER_RE = re.compile(r"(?<![A-Za-z])(?:\[(?P<bracket>[0-9,\-\s]+)\]|【(?P<cjk>[0-9,\-\s]+)】)")
CITATION_TOKEN_RE = re.compile(r"\d+|\-")


@dataclass(frozen=True)
class BibliographyEntry:
    key: str
    label: str
    text: str
    source_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CitationOccurrence:
    node_id: str
    raw_marker: str
    keys: list[str]
    start: int
    end: int


@dataclass(frozen=True)
class CitationResolution:
    entries: list[BibliographyEntry]
    occurrences: list[CitationOccurrence]
    text_by_node_id: dict[str, str]
    unresolved_markers: list[str] = field(default_factory=list)

    @property
    def entries_by_label(self) -> dict[str, BibliographyEntry]:
        return {entry.label: entry for entry in self.entries}

    @property
    def entries_by_key(self) -> dict[str, BibliographyEntry]:
        return {entry.key: entry for entry in self.entries}


@dataclass(frozen=True)
class CitationResolverConfig:
    key_prefix: str = "ref"
    strip_reference_labels: bool = True
    replace_body_citations: bool = True


class CitationResolver:
    """Build citation keys and repair OCR citation markers."""

    def __init__(self, config: CitationResolverConfig | None = None) -> None:
        self.config = config or CitationResolverConfig()

    def resolve_document(self, document: DocumentIR) -> CitationResolution:
        entries = self._extract_bibliography_entries(document)
        label_to_key = {entry.label: entry.key for entry in entries}
        text_by_node_id: dict[str, str] = {}
        occurrences: list[CitationOccurrence] = []
        unresolved_markers: list[str] = []

        for node in document.nodes:
            if node.node_type == BlockType.REFERENCE:
                text_by_node_id[node.node_id] = strip_reference_label(node.text) if self.config.strip_reference_labels else node.text
                continue
            repaired, node_occurrences, node_unresolved = replace_citation_markers(
                node.text,
                label_to_key,
                node_id=node.node_id,
                enabled=self.config.replace_body_citations,
            )
            text_by_node_id[node.node_id] = repaired
            occurrences.extend(node_occurrences)
            unresolved_markers.extend(node_unresolved)

        return CitationResolution(
            entries=entries,
            occurrences=occurrences,
            text_by_node_id=text_by_node_id,
            unresolved_markers=unresolved_markers,
        )

    def _extract_bibliography_entries(self, document: DocumentIR) -> list[BibliographyEntry]:
        entries: list[BibliographyEntry] = []
        next_index = 1
        for node in document.nodes:
            if node.node_type != BlockType.REFERENCE:
                continue
            for raw_text, metadata in iter_reference_texts(node):
                label, text = split_reference_label(raw_text)
                if not text:
                    continue
                if not label:
                    label = _metadata_label(metadata) or str(next_index)
                key = _metadata_key(metadata) or reference_key(label, prefix=self.config.key_prefix)
                entries.append(
                    BibliographyEntry(
                        key=key,
                        label=label,
                        text=text if self.config.strip_reference_labels else raw_text.strip(),
                        source_node_id=node.node_id,
                        metadata=metadata,
                    )
                )
                next_index += 1
        return _dedupe_entries(entries)


def iter_reference_texts(node: DocumentNode) -> Iterable[tuple[str, dict[str, Any]]]:
    items = node.metadata.get("reference_items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("raw_text") or item.get("content") or "").strip()
                if text:
                    yield text, dict(item)
            else:
                text = str(item or "").strip()
                if text:
                    yield text, {}
        return

    for line in split_reference_block(node.text):
        yield line, {}


def split_reference_block(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    lines = [line.strip() for line in re.split(r"\n+", value) if line.strip()]
    if len(lines) > 1:
        return lines
    starts = list(REFERENCE_LABEL_RE.finditer(value))
    if len(starts) <= 1:
        return [value]
    parts: list[str] = []
    for index, match in enumerate(starts):
        start = match.start()
        end = starts[index + 1].start() if index + 1 < len(starts) else len(value)
        part = value[start:end].strip()
        if part:
            parts.append(part)
    return parts or [value]


def split_reference_label(text: str) -> tuple[str | None, str]:
    value = str(text or "").strip()
    match = REFERENCE_LABEL_RE.match(value)
    if not match:
        return None, value
    label = match.group("bracket") or match.group("number")
    return label, value[match.end() :].strip()


def strip_reference_label(text: str) -> str:
    return split_reference_label(text)[1]


def replace_citation_markers(
    text: str,
    label_to_key: dict[str, str],
    *,
    node_id: str,
    enabled: bool = True,
) -> tuple[str, list[CitationOccurrence], list[str]]:
    value = str(text or "")
    if not enabled or not value or not label_to_key:
        return value, [], []

    occurrences: list[CitationOccurrence] = []
    unresolved: list[str] = []

    def replacer(match: re.Match[str]) -> str:
        raw_inner = match.group("bracket") or match.group("cjk") or ""
        labels = expand_citation_labels(raw_inner)
        keys = [label_to_key[label] for label in labels if label in label_to_key]
        if not keys:
            unresolved.append(match.group(0))
            return match.group(0)
        occurrences.append(
            CitationOccurrence(
                node_id=node_id,
                raw_marker=match.group(0),
                keys=keys,
                start=match.start(),
                end=match.end(),
            )
        )
        return r"\cite{" + ",".join(keys) + "}"

    return CITATION_MARKER_RE.sub(replacer, value), occurrences, unresolved


def expand_citation_labels(raw_inner: str) -> list[str]:
    tokens = CITATION_TOKEN_RE.findall(str(raw_inner or ""))
    labels: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-":
            index += 1
            continue
        if index + 2 < len(tokens) and tokens[index + 1] == "-" and tokens[index + 2].isdigit():
            start = int(token)
            end = int(tokens[index + 2])
            if start <= end and end - start <= 100:
                labels.extend(str(value) for value in range(start, end + 1))
                index += 3
                continue
        labels.append(str(int(token)) if token.isdigit() else token)
        index += 1
    return _dedupe_preserve_order(labels)


def reference_key(label: str, *, prefix: str = "ref") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(label or "").strip()).strip("_")
    return f"{prefix}_{cleaned or 'unknown'}"


def apply_citation_resolution_to_record(record: dict[str, Any], resolution: CitationResolution) -> dict[str, Any]:
    """Return a shallow-copy record with repaired text when a matching node id exists."""

    node_id = str(record.get("node_id") or record.get("id") or record.get("block_id") or "")
    if not node_id or node_id not in resolution.text_by_node_id:
        return record
    repaired = dict(record)
    repaired["text"] = resolution.text_by_node_id[node_id]
    repaired["text_for_embedding"] = resolution.text_by_node_id[node_id]
    return repaired


def _dedupe_entries(entries: list[BibliographyEntry]) -> list[BibliographyEntry]:
    seen: set[str] = set()
    deduped: list[BibliographyEntry] = []
    for entry in entries:
        key = entry.key
        if key in seen:
            suffix = 2
            while f"{key}_{suffix}" in seen:
                suffix += 1
            entry = BibliographyEntry(
                key=f"{key}_{suffix}",
                label=entry.label,
                text=entry.text,
                source_node_id=entry.source_node_id,
                metadata=entry.metadata,
            )
        seen.add(entry.key)
        deduped.append(entry)
    return deduped


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _metadata_key(metadata: dict[str, Any]) -> str | None:
    for key in ("citation_key", "bib_key", "bibkey", "key", "tex_key"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return sanitize_citation_key(value)
    return None


def _metadata_label(metadata: dict[str, Any]) -> str | None:
    for key in ("label", "ref_label", "number", "index"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(int(value))
        if isinstance(value, str) and value.strip():
            stripped = value.strip()
            match = REFERENCE_LABEL_RE.match(stripped)
            if match:
                return match.group("bracket") or match.group("number")
            if stripped.isdigit():
                return str(int(stripped))
    return None


def sanitize_citation_key(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z:._-]+", "_", str(value or "").strip()).strip("_")
    if not cleaned:
        return "ref_unknown"
    if cleaned[0].isdigit():
        return f"ref_{cleaned}"
    return cleaned
