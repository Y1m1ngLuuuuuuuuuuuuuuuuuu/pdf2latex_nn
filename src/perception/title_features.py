"""Title numbering probes shared by graph features and LaTeX rendering."""

from __future__ import annotations

import re


ROMAN_NUMERAL_PATTERN = r"(?=[MDCLXVI])M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
H3_TITLE_RE = re.compile(r"^\s*(?P<number>\d+(?:\.\d+){2,})\.?\s+")
H2_TITLE_RE = re.compile(r"^\s*(?P<number>\d+\.\d+)\.?\s+")
APPENDIX_H3_TITLE_RE = re.compile(r"^\s*(?P<number>[A-Z](?:\.\d+){2,})\.?\s+")
APPENDIX_H2_TITLE_RE = re.compile(r"^\s*(?P<number>[A-Z]\.\d+)\.?\s+")
H1_TITLE_RE = re.compile(rf"^\s*(?P<number>(?:\d+|{ROMAN_NUMERAL_PATTERN}))\.?\s+", re.IGNORECASE)


def title_numbering_level(text: str) -> int | None:
    """Return 1/2/3 for numbered headings, or None for unnumbered headings."""

    value = str(text or "")
    if H3_TITLE_RE.match(value):
        return 3
    if H2_TITLE_RE.match(value):
        return 2
    if APPENDIX_H3_TITLE_RE.match(value):
        return 3
    if APPENDIX_H2_TITLE_RE.match(value):
        return 2
    if H1_TITLE_RE.match(value):
        return 1
    return None


def title_pattern_flags(text: str) -> tuple[float, float]:
    """Return [is_h1_pattern, is_h2_or_deeper_pattern] as float flags."""

    level = title_numbering_level(text)
    return (float(level == 1), float(level is not None and level >= 2))


def strip_title_numbering(text: str) -> str:
    """Remove leading section numbers so LaTeX owns numbering."""

    value = str(text or "").strip()
    for pattern in (H3_TITLE_RE, H2_TITLE_RE, H1_TITLE_RE):
        match = pattern.match(value)
        if match:
            return value[match.end() :].strip()
    return value
