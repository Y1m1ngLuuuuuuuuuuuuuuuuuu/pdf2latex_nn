"""Conservative font inference and LaTeX fallback mapping.

PDF font names are useful evidence, but they are not necessarily installable
system font names.  This module keeps the two ideas separate:

1. canonical PDF font identity, used for style statistics;
2. safe LaTeX font fallback, used only when a renderer opts into fontspec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")
STYLE_TOKEN_RE = re.compile(r"\b(bold|italic|oblique|regular|roman|medium|light|black|semibold|demibold|mono|courier)\b", re.I)


@dataclass(frozen=True)
class FontInfo:
    raw_name: str
    canonical_name: str
    family_name: str
    font_class: str
    is_bold: bool
    is_italic: bool
    latex_family: str
    latex_package_safe: bool = True


def resolve_pdf_font(font_name: str | None) -> FontInfo | None:
    raw = str(font_name or "").strip()
    if not raw:
        return None
    canonical = canonicalize_pdf_font_name(raw)
    lower = canonical.lower()
    family = family_from_canonical(canonical)
    font_class = classify_font(canonical)
    latex_family = latex_fallback_for_font(canonical, font_class)
    return FontInfo(
        raw_name=raw,
        canonical_name=canonical,
        family_name=family,
        font_class=font_class,
        is_bold=any(token in lower for token in ("bold", "black", "semibold", "demibold", "bd")),
        is_italic=any(token in lower for token in ("italic", "oblique", "ital")),
        latex_family=latex_family,
    )


def canonicalize_pdf_font_name(font_name: str) -> str:
    value = SUBSET_PREFIX_RE.sub("", str(font_name or "").strip())
    value = value.replace(",", " ").replace("_", " ").replace("-", " ").replace("+", " ")
    value = value.replace("PSMT", "").replace("MT", "")
    value = re.sub(r"\s+", " ", value).strip()
    aliases = {
        "Times Roman": "Times",
        "TimesNewRoman": "Times New Roman",
        "TimesNewRomanPS": "Times New Roman",
        "NimbusRomNo9L": "Nimbus Roman",
        "Helvetica": "Helvetica",
        "Arial": "Arial",
        "Courier": "Courier",
        "CourierNew": "Courier New",
        "CMR": "Computer Modern Roman",
        "CMMI": "Computer Modern Math Italic",
        "CMSY": "Computer Modern Symbols",
        "CMEX": "Computer Modern Extensions",
        "LMRoman": "Latin Modern Roman",
    }
    compact = value.replace(" ", "")
    for key, replacement in aliases.items():
        if compact.startswith(key.replace(" ", "")):
            suffix = compact[len(key.replace(" ", "")) :]
            suffix = STYLE_TOKEN_RE.sub("", suffix).strip()
            return replacement
    return value


def family_from_canonical(canonical_name: str) -> str:
    value = STYLE_TOKEN_RE.sub("", canonical_name).strip()
    value = re.sub(r"\s+", " ", value)
    return value or canonical_name


def classify_font(canonical_name: str) -> str:
    lower = canonical_name.lower()
    if any(token in lower for token in ("courier", "mono", "typewriter", "consolas")):
        return "mono"
    if any(token in lower for token in ("helvetica", "arial", "sans", "heros")):
        return "sans"
    if any(token in lower for token in ("math", "cmmi", "cmsy", "cmex", "symbol")):
        return "math"
    return "serif"


def latex_fallback_for_font(canonical_name: str, font_class: str) -> str:
    lower = canonical_name.lower()
    if font_class == "mono":
        return "TeX Gyre Cursor"
    if font_class == "sans":
        return "TeX Gyre Heros"
    if "times" in lower or "nimbus roman" in lower:
        return "TeX Gyre Termes"
    if "computer modern" in lower or "latin modern" in lower:
        return "Latin Modern Roman"
    return "TeX Gyre Termes" if font_class == "serif" else "Latin Modern Roman"


def build_latex_font_setup(body_font: str | None, role_fonts: dict[str, str | None]) -> dict[str, object]:
    body = resolve_pdf_font(body_font)
    role_infos = {
        role: info
        for role, font in role_fonts.items()
        if (info := resolve_pdf_font(font)) is not None
    }
    serif_candidates = [info.latex_family for info in [body, *role_infos.values()] if info and info.font_class == "serif"]
    sans_candidates = [info.latex_family for info in [body, *role_infos.values()] if info and info.font_class == "sans"]
    mono_candidates = [info.latex_family for info in [body, *role_infos.values()] if info and info.font_class == "mono"]
    return {
        "enabled": False,
        "requires_engine": "xelatex_or_lualatex",
        "body_pdf_font": body.canonical_name if body else None,
        "body_font_class": body.font_class if body else None,
        "main_font": _first_or_default(serif_candidates, body.latex_family if body else "TeX Gyre Termes"),
        "sans_font": _first_or_default(sans_candidates, "TeX Gyre Heros"),
        "mono_font": _first_or_default(mono_candidates, "TeX Gyre Cursor"),
        "role_pdf_fonts": {role: info.canonical_name for role, info in role_infos.items()},
        "role_font_classes": {role: info.font_class for role, info in role_infos.items()},
        "note": "Font names are inferred from PDF spans. Enable fontspec only when compiling with XeLaTeX/LuaLaTeX.",
    }


def _first_or_default(values: list[str], default: str) -> str:
    return values[0] if values else default
