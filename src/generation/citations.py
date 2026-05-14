"""Citation and bibliography repair for LaTeX generation.

The frontend OCR sees citation labels as visible text, but LaTeX should emit
semantic citations and bibliography items.  This module keeps that repair
separate from tree decoding and low-level rendering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
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
YEAR_RE = re.compile(r"\b(?P<year>(?:19|20)\d{2}[a-z]?)\b", re.IGNORECASE)
AUTHOR_YEAR_PAREN_RE = re.compile(r"(?<![A-Za-z])\((?P<body>[^()]*\b(?:19|20)\d{2}[a-z]?(?:[^()]*)?)\)")
AUTHOR_YEAR_TEXTUAL_RE = re.compile(
    r"(?P<author>[A-Z][A-Za-z'’\-]+(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-z'’\-]+)?)\s*"
    r"\((?P<year>(?:19|20)\d{2}[a-z]?)\)"
)
AUTHOR_YEAR_SPLIT_RE = re.compile(r"\s*;\s*")
BBL_BIBITEM_RE = re.compile(
    r"\\bibitem(?:\[(?P<display>(?:[^\]\\]|\\.)*)\])?\{(?P<key>[^{}]+)\}",
    re.DOTALL,
)
BIBLIOGRAPHY_ENV_RE = re.compile(
    r"\\begin\{thebibliography\}(?:\{[^{}]*\})?.*?\\end\{thebibliography\}",
    re.DOTALL,
)


@dataclass(frozen=True)
class BibliographyEntry:
    key: str
    label: str
    text: str
    source_node_id: str | None = None
    display_label: str | None = None
    authors: str | None = None
    year: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthorYearLabel:
    authors: str
    surname: str
    year: str
    display_label: str


@dataclass(frozen=True)
class CitationOccurrence:
    node_id: str
    raw_marker: str
    keys: list[str]
    start: int
    end: int
    citation_style: str = "numeric"


@dataclass(frozen=True)
class CitationResolution:
    entries: list[BibliographyEntry]
    occurrences: list[CitationOccurrence]
    text_by_node_id: dict[str, str]
    unresolved_markers: list[str] = field(default_factory=list)
    citation_style: str = "numeric"
    raw_bibliography_latex: str | None = None
    source_bbl_path: str | None = None

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
    infer_author_year_keys: bool = True
    author_year_command: str = "cite"


class CitationResolver:
    """Build citation keys and repair OCR citation markers."""

    def __init__(self, config: CitationResolverConfig | None = None) -> None:
        self.config = config or CitationResolverConfig()

    def resolve_document(self, document: DocumentIR, *, source_tex_path: str | Path | None = None) -> CitationResolution:
        source_bibliography = self._resolve_source_bibliography(source_tex_path)
        entries = source_bibliography.entries if source_bibliography is not None else self._extract_bibliography_entries(document)
        label_to_key = {entry.label: entry.key for entry in entries}
        author_year_to_key = author_year_lookup(entries)
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
                author_year_to_key,
                node_id=node.node_id,
                enabled=self.config.replace_body_citations,
                author_year_command=self.config.author_year_command,
            )
            text_by_node_id[node.node_id] = repaired
            occurrences.extend(node_occurrences)
            unresolved_markers.extend(node_unresolved)

        return CitationResolution(
            entries=entries,
            occurrences=occurrences,
            text_by_node_id=text_by_node_id,
            unresolved_markers=unresolved_markers,
            citation_style=infer_citation_style(entries, occurrences),
            raw_bibliography_latex=source_bibliography.raw_latex if source_bibliography is not None else None,
            source_bbl_path=str(source_bibliography.path) if source_bibliography is not None else None,
        )

    def _resolve_source_bibliography(self, source_tex_path: str | Path | None) -> "_SourceBibliography | None":
        bbl_path = find_bbl_for_tex(source_tex_path)
        if bbl_path is None:
            return None
        raw_latex = bbl_path.read_text(encoding="utf-8", errors="replace")
        entries = parse_bbl_entries(raw_latex)
        if not entries:
            return None
        return _SourceBibliography(path=bbl_path, raw_latex=normalize_bbl_latex(raw_latex), entries=entries)

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
                author_year = infer_author_year(raw_text, metadata) if self.config.infer_author_year_keys else None
                key = _metadata_key(metadata) or (
                    author_year_key(author_year, fallback_label=label, prefix=self.config.key_prefix)
                    if author_year
                    else reference_key(label, prefix=self.config.key_prefix)
                )
                entries.append(
                    BibliographyEntry(
                        key=key,
                        label=label,
                        text=text if self.config.strip_reference_labels else raw_text.strip(),
                        source_node_id=node.node_id,
                        display_label=_metadata_display_label(metadata) or (author_year.display_label if author_year else None),
                        authors=author_year.authors if author_year else None,
                        year=author_year.year if author_year else None,
                        metadata=metadata,
                    )
                )
                next_index += 1
        return _dedupe_entries(entries)


@dataclass(frozen=True)
class _SourceBibliography:
    path: Path
    raw_latex: str
    entries: list[BibliographyEntry]


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


def find_bbl_for_tex(source_tex_path: str | Path | None) -> Path | None:
    """Return the compiled bibliography for a TeX source when one exists.

    We prefer the exact sibling ``main.tex`` -> ``main.bbl``.  If that is not
    available, a single ``*.bbl`` in the source directory is a safe fallback for
    arXiv packages where the main file may be named differently by the scanner.
    """

    if source_tex_path is None:
        return None
    tex_path = Path(source_tex_path)
    if not tex_path.exists():
        return None
    exact = tex_path.with_suffix(".bbl")
    if exact.exists():
        return exact
    candidates = sorted(path for path in tex_path.parent.glob("*.bbl") if path.is_file())
    if len(candidates) == 1:
        return candidates[0]
    for preferred_name in ("main.bbl", "arxiv.bbl", "paper.bbl", "ms.bbl"):
        preferred = tex_path.parent / preferred_name
        if preferred.exists():
            return preferred
    return None


def normalize_bbl_latex(raw_latex: str) -> str:
    """Keep the source bibliography as LaTeX.

    Compiled ``.bbl`` files sometimes define small helper macros before
    ``thebibliography`` (for example ``\natexlab`` or URL fallbacks).  Preserve
    those definitions instead of trimming to the environment; otherwise the
    generated document may compile differently from the source package.
    """

    value = str(raw_latex or "").strip()
    if not value:
        return ""
    # Keep source LaTeX commands such as \newblock, {\em ...}, accents and URLs.
    # Only remove comments so they do not leak into generated source.
    lines = []
    for line in value.splitlines():
        lines.append(re.sub(r"(?<!\\)%.*$", "", line).rstrip())
    return "\n".join(lines).strip()


def parse_bbl_entries(raw_latex: str) -> list[BibliographyEntry]:
    """Parse ``\\bibitem`` metadata while preserving the raw BBL for rendering."""

    value = normalize_bbl_latex(raw_latex)
    matches = list(BBL_BIBITEM_RE.finditer(value))
    entries: list[BibliographyEntry] = []
    for index, match in enumerate(matches, start=1):
        next_start = matches[index].start() if index < len(matches) else _bibliography_end_start(value, len(value))
        body = value[match.end() : next_start].strip()
        key = sanitize_citation_key(match.group("key"))
        display_label = normalize_author_year_display_label(match.group("display") or "") or None
        label = str(index)
        author_year = infer_author_year_from_bbl_item(display_label, body)
        entries.append(
            BibliographyEntry(
                key=key,
                label=label,
                text=body,
                display_label=display_label,
                authors=author_year.authors if author_year else None,
                year=author_year.year if author_year else None,
                metadata={"source": "bbl", "original_key": match.group("key"), "ordinal": index},
            )
        )
    return entries


def _bibliography_end_start(value: str, fallback: int) -> int:
    match = re.search(r"\\end\{thebibliography\}", value)
    return match.start() if match else fallback


def infer_author_year_from_bbl_item(display_label: str | None, body: str) -> AuthorYearLabel | None:
    if display_label:
        parsed = parse_author_year_part(display_label)
        if parsed:
            surname, year = parsed
            return AuthorYearLabel(
                authors=display_label,
                surname=surname,
                year=year,
                display_label=display_label,
            )
    return infer_author_year(body, {})


def strip_reference_label(text: str) -> str:
    return split_reference_label(text)[1]


def replace_citation_markers(
    text: str,
    label_to_key: dict[str, str],
    author_year_to_key: dict[tuple[str, str], str] | None = None,
    *,
    node_id: str,
    enabled: bool = True,
    author_year_command: str = "cite",
) -> tuple[str, list[CitationOccurrence], list[str]]:
    value = str(text or "")
    author_year_to_key = author_year_to_key or {}
    if not enabled or not value or (not label_to_key and not author_year_to_key):
        return value, [], []

    occurrences: list[CitationOccurrence] = []
    unresolved: list[str] = []

    def numeric_replacer(match: re.Match[str]) -> str:
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
                citation_style="numeric",
            )
        )
        return r"\cite{" + ",".join(keys) + "}"

    repaired = CITATION_MARKER_RE.sub(numeric_replacer, value)
    if author_year_to_key:
        repaired = replace_author_year_citations(
            repaired,
            author_year_to_key,
            node_id=node_id,
            command=author_year_command,
            occurrences=occurrences,
            unresolved=unresolved,
        )
    return repaired, occurrences, unresolved


def replace_author_year_citations(
    text: str,
    author_year_to_key: dict[tuple[str, str], str],
    *,
    node_id: str,
    command: str,
    occurrences: list[CitationOccurrence],
    unresolved: list[str],
) -> str:
    command = sanitize_cite_command(command)

    def paren_replacer(match: re.Match[str]) -> str:
        raw_body = match.group("body")
        keys = keys_for_author_year_marker(raw_body, author_year_to_key)
        if not keys:
            if looks_like_author_year_marker(raw_body):
                unresolved.append(match.group(0))
            return match.group(0)
        occurrences.append(
            CitationOccurrence(
                node_id=node_id,
                raw_marker=match.group(0),
                keys=keys,
                start=match.start(),
                end=match.end(),
                citation_style="author_year",
            )
        )
        return rf"\{command}{{{','.join(keys)}}}"

    repaired = AUTHOR_YEAR_PAREN_RE.sub(paren_replacer, text)

    def textual_replacer(match: re.Match[str]) -> str:
        marker = f"{match.group('author')} {match.group('year')}"
        keys = keys_for_author_year_marker(marker, author_year_to_key)
        if not keys:
            return match.group(0)
        occurrences.append(
            CitationOccurrence(
                node_id=node_id,
                raw_marker=match.group(0),
                keys=keys,
                start=match.start(),
                end=match.end(),
                citation_style="author_year_textual",
            )
        )
        return rf"\{command}{{{','.join(keys)}}}"

    return AUTHOR_YEAR_TEXTUAL_RE.sub(textual_replacer, repaired)


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
                display_label=entry.display_label,
                authors=entry.authors,
                year=entry.year,
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
    for key in ("citation_key", "cite_key", "bib_key", "bibkey", "bibtex_key", "entry_key", "key", "tex_key"):
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


def _metadata_display_label(metadata: dict[str, Any]) -> str | None:
    for key in ("display_label", "citation_label", "author_year_label", "optional_label"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_author_year_display_label(value.strip())
    return None


def sanitize_citation_key(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z:._-]+", "_", str(value or "").strip()).strip("_")
    if not cleaned:
        return "ref_unknown"
    if cleaned[0].isdigit():
        return f"ref_{cleaned}"
    return cleaned


def sanitize_cite_command(value: str) -> str:
    command = re.sub(r"[^A-Za-z*]+", "", str(value or "cite")).strip()
    return command or "cite"


def infer_author_year(text: str, metadata: dict[str, Any] | None = None) -> AuthorYearLabel | None:
    metadata = metadata or {}
    year = _metadata_year(metadata) or first_year(text)
    authors = _metadata_authors(metadata) or leading_author_text(text)
    if not year or not authors:
        return None
    surname = first_author_surname(authors)
    if not surname:
        return None
    display = _metadata_display_label(metadata) or f"{display_author_label(authors)}, {year}"
    return AuthorYearLabel(authors=authors, surname=surname, year=year, display_label=display)


def _metadata_year(metadata: dict[str, Any]) -> str | None:
    for key in ("year", "date", "publication_year"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(int(value))
        if isinstance(value, str):
            match = YEAR_RE.search(value)
            if match:
                return match.group("year")
    return None


def _metadata_authors(metadata: dict[str, Any]) -> str | None:
    for key in ("authors", "author", "creator"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            if parts:
                return ", ".join(parts)
    return None


def first_year(text: str) -> str | None:
    match = YEAR_RE.search(str(text or ""))
    return match.group("year") if match else None


def leading_author_text(text: str) -> str | None:
    value = strip_reference_label(text).strip()
    if not value:
        return None
    year_match = YEAR_RE.search(value)
    if not year_match:
        return None
    prefix = value[: year_match.start()].strip(" .,(;:")
    if not prefix:
        return None
    sentence_head = re.split(r"\.\s+", prefix, maxsplit=1)[0].strip()
    return sentence_head or prefix


def first_author_surname(authors: str) -> str | None:
    value = str(authors or "").strip()
    if not value:
        return None
    first = re.split(r"\s*(?:,|;|\band\b|&)\s*", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    first = re.sub(r"\bet\s+al\.?$", "", first, flags=re.IGNORECASE).strip()
    tokens = re.findall(r"[A-Za-z][A-Za-z'’\-]*", first)
    if not tokens:
        return None
    if len(tokens) == 1:
        return tokens[0]
    # References commonly use "Surname, Initials"; metadata may use "Given Surname".
    return tokens[0] if "," in value[: max(value.find(first), 0) + len(first) + 1] else tokens[-1]


def display_author_label(authors: str) -> str:
    value = str(authors or "").strip()
    surnames = [name for name in (first_author_surname(part) for part in re.split(r"\s*(?:;|\band\b|&)\s*", value, flags=re.IGNORECASE)) if name]
    if len(surnames) >= 3:
        return f"{surnames[0]} et al."
    if len(surnames) == 2:
        return f"{surnames[0]} and {surnames[1]}"
    return surnames[0] if surnames else value


def normalize_author_year_display_label(value: str) -> str:
    return " ".join(str(value or "").replace(";", ",").split())


def author_year_key(author_year: AuthorYearLabel, *, fallback_label: str, prefix: str) -> str:
    if not author_year:
        return reference_key(fallback_label, prefix=prefix)
    key = f"{author_year.surname}{author_year.year}"
    return sanitize_citation_key(key)


def author_year_lookup(entries: list[BibliographyEntry]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for entry in entries:
        if not entry.year:
            continue
        keys = set()
        if entry.authors:
            surname = first_author_surname(entry.authors)
            if surname:
                keys.add((normalize_author_token(surname), normalize_year_token(entry.year)))
        if entry.display_label:
            parsed = parse_author_year_part(entry.display_label)
            if parsed:
                keys.add((normalize_author_token(parsed[0]), normalize_year_token(parsed[1])))
        for key in keys:
            lookup.setdefault(key, entry.key)
    return lookup


def keys_for_author_year_marker(raw_body: str, lookup: dict[tuple[str, str], str]) -> list[str]:
    keys: list[str] = []
    for part in AUTHOR_YEAR_SPLIT_RE.split(str(raw_body or "")):
        parsed = parse_author_year_part(part)
        if not parsed:
            continue
        author, year = parsed
        key = lookup.get((normalize_author_token(author), normalize_year_token(year)))
        if key:
            keys.append(key)
    return _dedupe_preserve_order(keys)


def parse_author_year_part(value: str) -> tuple[str, str] | None:
    text = str(value or "").strip()
    year_match = YEAR_RE.search(text)
    if not year_match:
        return None
    year = year_match.group("year")
    author_part = text[: year_match.start()].strip(" ,;()")
    if not author_part:
        author_part = text[year_match.end() :].strip(" ,;()")
    surname = first_author_surname(author_part)
    if not surname:
        return None
    return surname, year


def normalize_author_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def normalize_year_token(value: str) -> str:
    return str(value or "").casefold().strip()


def looks_like_author_year_marker(raw_body: str) -> bool:
    return parse_author_year_part(raw_body) is not None


def infer_citation_style(entries: list[BibliographyEntry], occurrences: list[CitationOccurrence]) -> str:
    if any(occurrence.citation_style.startswith("author_year") for occurrence in occurrences):
        return "author_year"
    if entries and all(entry.display_label and entry.year for entry in entries):
        return "author_year"
    return "numeric"
