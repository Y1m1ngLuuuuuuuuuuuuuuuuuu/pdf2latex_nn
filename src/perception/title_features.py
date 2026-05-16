"""Title numbering probes shared by graph features and LaTeX rendering."""

from __future__ import annotations

import re


ROMAN_NUMERAL_PATTERN = r"(?=[MDCLXVI])M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
MONTH_NAME_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
H3_TITLE_RE = re.compile(r"^\s*(?P<number>\d+(?:\.\d+){2,})\.?\s+")
H2_TITLE_RE = re.compile(r"^\s*(?P<number>\d+\.\d+)\.?\s+")
APPENDIX_H3_TITLE_RE = re.compile(r"^\s*(?P<number>[A-Z](?:\.\d+){2,})\.?\s+")
APPENDIX_H2_TITLE_RE = re.compile(r"^\s*(?P<number>[A-Z]\.\d+)\.?\s+")
H1_TITLE_RE = re.compile(rf"^\s*(?P<number>(?:\d+|{ROMAN_NUMERAL_PATTERN}))\.?\s+", re.IGNORECASE)
ALPHA_TITLE_RE = re.compile(r"^\s*(?P<number>[A-Za-z])[\.\)]\s+")
CHINESE_NUMERAL_PATTERN = r"[一二三四五六七八九十百千零〇两壹贰叁肆伍陆柒捌玖拾]+"
CHINESE_TITLE_RE = re.compile(
    rf"^\s*(?P<number>(?:第\s*)?{CHINESE_NUMERAL_PATTERN}(?:\s*[章节篇部分])?)(?:[、\.．\)]\s*|\s+)"
)
FRONT_MATTER_DATE_RE = re.compile(
    rf"""
    ^\s*
    (?:(?:received|accepted|revised|published|available\s+online|date)\s*[:,-]?\s*)?
    (?:
        \d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTH_NAME_PATTERN})(?:\s+\d{{4}})?
        |
        (?:{MONTH_NAME_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?(?:\s+\d{{4}})?
        |
        (?:{MONTH_NAME_PATTERN})\s+\d{{4}}
    )
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_front_matter_date_text(text: str) -> bool:
    """Return True for date-only front matter lines such as ``25 March 2025``."""

    value = " ".join(str(text or "").strip().split())
    if not value:
        return False
    return FRONT_MATTER_DATE_RE.match(value) is not None


def title_numbering_level(text: str) -> int | None:
    """Return 1/2/3 for numbered headings, or None for unnumbered headings."""

    value = str(text or "")
    if is_front_matter_date_text(value):
        return None
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


def title_numbering_path(text: str) -> tuple[str, ...] | None:
    """Return the explicit heading number path, if one is present.

    The path is used by layout/label code to keep numbered heading stacks
    coherent. For example, ``"2.3 Method"`` returns ``("2", "3")`` while
    ``"I. Introduction"`` returns ``("I",)``.  Single alphabetic prefixes are
    included here for already-identified heading candidates; callers must still
    avoid applying this to arbitrary list items.
    """

    value = str(text or "")
    if is_front_matter_date_text(value):
        return None
    for pattern in (H3_TITLE_RE, H2_TITLE_RE, APPENDIX_H3_TITLE_RE, APPENDIX_H2_TITLE_RE, H1_TITLE_RE, ALPHA_TITLE_RE, CHINESE_TITLE_RE):
        match = pattern.match(value)
        if match:
            token = match.group("number").rstrip(".")
            return tuple(part.upper() if part.isalpha() else part for part in token.split(".") if part)
    return None


def title_numbering_info(text: str) -> dict[str, object]:
    """Return explicit numbering features for a heading-like string.

    The renderer and graph features use this as a probe, not as the sole title
    detector.  It records the numbering scheme separately from the inferred
    hierarchy because formats such as ``C. Method`` are visually alphabetic but
    can be confused with Roman numerals if only a regex level is kept.
    """

    value = str(text or "")
    if is_front_matter_date_text(value):
        return {"has_numbering": False, "style": "none", "level": None, "path": (), "token": ""}
    patterns: tuple[tuple[re.Pattern[str], int, str], ...] = (
        (H3_TITLE_RE, 3, "decimal"),
        (H2_TITLE_RE, 2, "decimal"),
        (APPENDIX_H3_TITLE_RE, 3, "appendix_decimal"),
        (APPENDIX_H2_TITLE_RE, 2, "appendix_decimal"),
        (CHINESE_TITLE_RE, 1, "chinese"),
        (H1_TITLE_RE, 1, "roman_or_arabic"),
        (ALPHA_TITLE_RE, 1, "alpha"),
    )
    for pattern, level, style in patterns:
        match = pattern.match(value)
        if not match:
            continue
        token = match.group("number").strip().rstrip(".")
        if style == "roman_or_arabic":
            if token.isdigit():
                style = "arabic"
            elif len(token) == 1:
                style = "roman_or_alpha"
            else:
                style = "roman"
        path = tuple(part.upper() if part.isalpha() else part for part in token.split(".") if part)
        return {
            "has_numbering": True,
            "style": style,
            "level": level,
            "path": path,
            "token": token,
        }
    return {"has_numbering": False, "style": "none", "level": None, "path": (), "token": ""}


def title_pattern_flags(text: str) -> tuple[float, float]:
    """Return [is_h1_pattern, is_h2_or_deeper_pattern] as float flags."""

    level = title_numbering_level(text)
    return (float(level == 1), float(level is not None and level >= 2))


def strip_title_numbering(text: str) -> str:
    """Remove leading section numbers so LaTeX owns numbering."""

    value = str(text or "").strip()
    if is_front_matter_date_text(value):
        return value
    for pattern in (H3_TITLE_RE, H2_TITLE_RE, APPENDIX_H3_TITLE_RE, APPENDIX_H2_TITLE_RE, CHINESE_TITLE_RE, ALPHA_TITLE_RE, H1_TITLE_RE):
        match = pattern.match(value)
        if match:
            return value[match.end() :].strip()
    return value
