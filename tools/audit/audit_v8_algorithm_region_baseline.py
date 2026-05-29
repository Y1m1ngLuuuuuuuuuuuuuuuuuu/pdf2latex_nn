#!/usr/bin/env python3
"""Audit selected200 v8 AlgorithmRegion / pseudocode candidates.

This pass is read-only. It scans current v8 full observable fact artifacts and
existing selected200 generated/comparison outputs. It does not regenerate LaTeX,
compile, train, run MinerU, rebuild graphs, or modify renderer behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_ROOT = Path(
    "data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/"
    "v8_contentlist_merge_hint_valid_manifest/e2e_skipcompile"
)
DEFAULT_FLOAT_CAPTION_ROOT = Path("data/09_eval_reports/float_caption_layout_20260526/v8_same_code_ab_validation")
DEFAULT_OUTPUT_DIR = Path("data/09_eval_reports/algorithm_region_20260526/selected200_baseline_audit")


ALGORITHM_CAPTION_RE = re.compile(
    r"^\s*(?P<label>algorithm|alg\.?|procedure|pseudocode)\s*"
    r"(?P<number>(?:\d+(?:\.\d+)*(?:\([a-z]\))?|[ivxlcdm]+))?"
    r"\s*(?P<punct>[:.\-–—])?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
ALGORITHM_BODY_KEYWORD_RE = re.compile(
    r"\b(input|output|require|ensure|initiali[sz]ation|initialize|return|for|while|if|else|"
    r"repeat|until|break|continue|function|procedure|end\s+if|end\s+for|end\s+while)\b",
    re.IGNORECASE,
)
PSEUDOCODE_ANCHOR_RE = re.compile(
    r"^\s*(input|output|require|ensure|initiali[sz]ation|return)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
FALSE_ALGORITHM_REFERENCE_RE = re.compile(
    r"^\s*(?:as\s+shown\s+in|shown\s+in|see|according\s+to|we\s+use|in)\s+"
    r"(?:algorithm|alg\.?)\b",
    re.IGNORECASE,
)
LINE_NUMBER_RE = re.compile(r"(?m)^\s*(?:\d+[.)]|\(\d+\))\s+\S+")
RISKY_UNICODE_RE = re.compile(r"[✓✗✘✔×□■●▲▶→⇒≤≥∈∞−∑∏∂∇ϵηρΓ]")
UNESCAPED_SPECIAL_RE = re.compile(r"(?<!\\)[#%&_]")
MATH_GLYPH_RE = re.compile(r"[α-ωΑ-ΩϵηρΓ∑∏∂∇∞≤≥≈≠]")

FLOAT_TYPES = {"figure", "table", "algorithm"}


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--float-caption-root", type=Path, default=DEFAULT_FLOAT_CAPTION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-ids", nargs="*", default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def load_json(path: Path | None, default: Any = None) -> Any:
    if path is None or not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compact(text: Any, limit: int = 240) -> str:
    return " ".join(str(text or "").split())[:limit]


def normalize(text: Any) -> str:
    value = str(text or "").casefold()
    value = re.sub(r"\[math\]|\$[^$]*\$|\\[a-zA-Z]+", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def as_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def parse_bbox(value: Any) -> BBox | None:
    if isinstance(value, dict):
        keys = ("x0", "y0", "x1", "y1")
        if all(key in value for key in keys):
            return BBox(float(value["x0"]), float(value["y0"]), float(value["x1"]), float(value["y1"]))
    if isinstance(value, list):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            return BBox(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        boxes = [box for item in value if (box := parse_bbox(item)) is not None]
        if boxes:
            return BBox(
                min(box.x0 for box in boxes),
                min(box.y0 for box in boxes),
                max(box.x1 for box in boxes),
                max(box.y1 for box in boxes),
            )
    return None


def bbox_to_list(box: BBox | None) -> list[float] | None:
    if box is None:
        return None
    return [box.x0, box.y0, box.x1, box.y1]


def collect_doc_dirs(root: Path) -> dict[str, Path]:
    doc_dirs: dict[str, Path] = {}
    if not root.exists():
        return doc_dirs
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        doc_id = path.name.split("_", 1)[-1]
        if (path / "document_ir.json").exists() and (path / "generated.tex").exists():
            doc_dirs[doc_id] = path
    return doc_dirs


def get_block_type(block: dict[str, Any] | None) -> str:
    if not block:
        return ""
    return str(block.get("block_type") or block.get("type") or block.get("role") or "").lower()


def blocks_by_id(blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(block.get("block_id")): block for block in blocks if block.get("block_id")}


def is_algorithm_caption_text(text: Any) -> tuple[bool, str | None, str | None]:
    value = str(text or "").strip()
    if not value or FALSE_ALGORITHM_REFERENCE_RE.match(value):
        return False, None, None
    match = ALGORITHM_CAPTION_RE.match(value)
    if not match:
        return False, None, None
    label = (match.group("label") or "").lower().rstrip(".")
    number = match.group("number")
    punct = match.group("punct")
    rest = (match.group("rest") or "").strip()
    if label in {"method", "pseudocode"} and not punct:
        return False, None, None
    if label in {"algorithm", "alg", "procedure"} and not number and not punct:
        return False, None, None
    if rest.lower().startswith(("is ", "are ", "shows ", "show ", "uses ", "used ")):
        return False, None, None
    return True, "algorithm", number


def algorithm_body_score(text: str, *, raw_type: str = "", style_spans: list[dict[str, Any]] | None = None) -> tuple[int, list[str]]:
    signals: list[str] = []
    value = text or ""
    lowered_raw = raw_type.casefold()
    keyword_hits = len(ALGORITHM_BODY_KEYWORD_RE.findall(value))
    has_anchor = bool(PSEUDOCODE_ANCHOR_RE.search(value))
    has_line_numbers = bool(LINE_NUMBER_RE.search(value))
    if lowered_raw in {"code", "algorithm"}:
        signals.append("raw_type_code_or_algorithm")
    if has_anchor:
        signals.append("input_output_require_anchor")
    # Prose often contains "algorithm", "for", and "if". Count keyword density
    # only when the block has algorithm-like layout evidence.
    if keyword_hits >= 3 and (has_anchor or lowered_raw in {"code", "algorithm"}):
        signals.append(f"algorithm_keyword_hits={keyword_hits}")
    if has_line_numbers and (has_anchor or lowered_raw in {"code", "algorithm"}):
        signals.append("line_number_pattern")
    if style_spans:
        font_names = " ".join(str(span.get("font_name") or "").casefold() for span in style_spans)
        bold_labels = sum(
            1
            for span in style_spans
            if span.get("is_bold") and str(span.get("text") or "").strip().rstrip(":").casefold()
            in {"input", "output", "require", "ensure", "return", "initialization"}
        )
        if any(token in font_names for token in ("mono", "cour", "cmtt", "typewriter")):
            signals.append("monospace_style")
        if bold_labels >= 1:
            signals.append(f"bold_pseudocode_labels={bold_labels}")
        if len(style_spans) >= 8 and (has_anchor or lowered_raw in {"code", "algorithm"}):
            signals.append("multi_span_code_region")
    score = 0
    weights = {
        "raw_type_code_or_algorithm": 4,
        "input_output_require_anchor": 3,
        "line_number_pattern": 2,
        "monospace_style": 2,
        "multi_span_code_region": 2,
    }
    for signal in signals:
        score += weights.get(signal, 1)
        if signal.startswith("algorithm_keyword_hits"):
            score += min(4, keyword_hits)
        if signal.startswith("bold_pseudocode_labels"):
            score += 2
    return score, signals


def combined_item_text(item: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("text", "content_list_text", "algorithm_caption", "code_caption", "caption"):
        value = item.get(key)
        if isinstance(value, list):
            pieces.extend(str(part) for part in value if str(part).strip())
        elif str(value or "").strip():
            pieces.append(str(value))
    for key in ("style_spans", "source_lines"):
        values = item.get(key) or []
        if isinstance(values, list):
            for part in values:
                if isinstance(part, dict) and str(part.get("text") or "").strip():
                    pieces.append(str(part.get("text")))
    return "\n".join(pieces)


def first_nonempty_line(text: str) -> str:
    for line in str(text or "").splitlines():
        if line.strip():
            return line.strip()
    return str(text or "").strip()


def node_combined_text(node: dict[str, Any]) -> str:
    pieces = [str(node.get("text") or "")]
    metadata = node.get("metadata") or {}
    for key in ("algorithm_caption", "code_caption", "caption", "figure_caption", "table_caption"):
        value = metadata.get(key)
        if isinstance(value, list):
            pieces.extend(str(part) for part in value if str(part).strip())
        elif str(value or "").strip():
            pieces.append(str(value))
    for span in node.get("spans") or []:
        if isinstance(span, dict) and str(span.get("text") or "").strip():
            pieces.append(str(span.get("text")))
    return "\n".join(piece for piece in pieces if piece)


def detect_algorithm_candidates(content_payload: dict[str, Any], document_ir: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for idx, item in enumerate(content_payload.get("items") or []):
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get("type") or item.get("raw_type") or "")
        style_spans = item.get("style_spans") or []
        text = combined_item_text(item)
        is_caption, _, caption_number = is_algorithm_caption_text(first_nonempty_line(text))
        body_score, signals = algorithm_body_score(text, raw_type=raw_type, style_spans=style_spans)
        metadata_caption = any(str(item.get(key) or "").strip() for key in ("algorithm_caption", "code_caption"))
        if not (is_caption or metadata_caption or body_score >= 5):
            continue
        cid = str(item.get("id") or f"item_{idx:06d}")
        if cid in seen:
            continue
        seen.add(cid)
        candidate_kind = []
        if is_caption or metadata_caption:
            candidate_kind.append("caption")
        if body_score >= 5 or raw_type.casefold() in {"code", "algorithm"}:
            candidate_kind.append("body")
        box = parse_bbox(item.get("bbox"))
        candidates.append(
            {
                "candidate_id": cid,
                "source": "v8_content_item",
                "source_v8_ids": [cid],
                "page_idx": item.get("page_idx"),
                "bbox": bbox_to_list(box),
                "text_preview": compact(text, 500),
                "normalized_text": normalize(text),
                "candidate_kind": candidate_kind or ["unknown"],
                "caption_number": caption_number,
                "raw_type": raw_type,
                "current_role": item.get("layout_role") or item.get("canonical_type") or raw_type,
                "current_canonical_type": item.get("canonical_type"),
                "signals": signals + (["algorithm_caption_pattern"] if is_caption else []) + (["metadata_algorithm_caption"] if metadata_caption else []),
                "body_score": body_score,
                "style_span_count": len(style_spans) if isinstance(style_spans, list) else 0,
            }
        )

    for idx, node in enumerate(document_ir.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        metadata = node.get("metadata") or {}
        raw_type = str(node.get("raw_type") or metadata.get("raw_type") or node.get("node_type") or "")
        role_blob = " ".join(
            str(value or "")
            for value in (
                node.get("node_type"),
                node.get("raw_type"),
                metadata.get("canonical_type"),
                metadata.get("layout_role"),
                metadata.get("type"),
            )
        ).casefold()
        text = node_combined_text(node)
        is_caption, _, caption_number = is_algorithm_caption_text(first_nonempty_line(text))
        metadata_caption = any(str(metadata.get(key) or "").strip() for key in ("algorithm_caption", "code_caption"))
        body_score, signals = algorithm_body_score(text, raw_type=raw_type, style_spans=node.get("spans") or [])
        role_algorithm = "algorithm" in role_blob or "code" in role_blob
        if not (is_caption or metadata_caption or body_score >= 5 or role_algorithm):
            continue
        cid = str(node.get("node_id") or f"node_{idx:06d}")
        if cid in seen:
            continue
        seen.add(cid)
        box = parse_bbox(node.get("bboxes") or node.get("bbox"))
        kind = []
        if is_caption or metadata_caption:
            kind.append("caption")
        if body_score >= 5 or role_algorithm:
            kind.append("body")
        candidates.append(
            {
                "candidate_id": cid,
                "source": "document_ir",
                "source_v8_ids": [cid],
                "page_idx": node.get("page_idx"),
                "bbox": bbox_to_list(box),
                "text_preview": compact(text, 500),
                "normalized_text": normalize(text),
                "candidate_kind": kind or ["unknown"],
                "caption_number": caption_number,
                "raw_type": raw_type,
                "current_role": metadata.get("layout_role") or metadata.get("canonical_type") or node.get("node_type"),
                "current_canonical_type": metadata.get("canonical_type"),
                "signals": signals + (["algorithm_caption_pattern"] if is_caption else []) + (["metadata_algorithm_caption"] if metadata_caption else []) + (["role_algorithm_or_code"] if role_algorithm else []),
                "body_score": body_score,
                "style_span_count": len(node.get("spans") or []),
            }
        )
    return candidates


def extract_algorithm_blocks(structure: dict[str, Any], source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = structure.get("blocks") or []
    by_id = blocks_by_id(blocks)
    algorithms: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    algorithm_ids = set()
    for block in blocks:
        block_type = get_block_type(block)
        if block_type == "algorithm":
            parent = by_id.get(str(block.get("parent_id") or ""))
            parent_type = get_block_type(parent)
            marker = str(block.get("marker") or "").casefold()
            # Count the container-like algorithm block, not every nested
            # algorithmic child, when possible.
            if parent_type != "algorithm" or marker in {"algorithm", "algorithm2e", "lstlisting", "verbatim"}:
                algorithm_ids.add(str(block.get("block_id")))
                algorithms.append(
                    {
                        "source": source,
                        "block_id": block.get("block_id"),
                        "text": block.get("text") or "",
                        "normalized_text": normalize(block.get("text") or ""),
                        "order": block.get("order"),
                        "marker": block.get("marker"),
                        "parent_id": block.get("parent_id"),
                        "block": block,
                    }
                )
    for block in blocks:
        if get_block_type(block) != "caption":
            continue
        parent = by_id.get(str(block.get("parent_id") or block.get("label") or ""))
        parent_type = get_block_type(parent)
        is_caption, _, number = is_algorithm_caption_text(block.get("text") or "")
        if parent_type == "algorithm" or is_caption:
            captions.append(
                {
                    "source": source,
                    "caption_id": block.get("block_id"),
                    "text": block.get("text") or "",
                    "normalized_text": normalize(block.get("text") or ""),
                    "caption_number": number,
                    "parent_id": block.get("parent_id") or block.get("label"),
                    "parent_type": parent_type or "unknown",
                    "order": block.get("order"),
                    "block": block,
                }
            )
    return algorithms, captions


def extract_algorithm_like_paragraphs(structure: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for block in structure.get("blocks") or []:
        block_type = get_block_type(block)
        if block_type in {"algorithm", "caption"}:
            continue
        text = str(block.get("text") or "")
        is_caption, _, number = is_algorithm_caption_text(text)
        body_score, signals = algorithm_body_score(text)
        if is_caption or body_score >= 5:
            rows.append(
                {
                    "block_id": block.get("block_id"),
                    "block_type": block_type,
                    "text": compact(text),
                    "caption_number": number,
                    "signals": signals + (["algorithm_caption_pattern"] if is_caption else []),
                    "order": block.get("order"),
                }
            )
    return rows


def match_texts(left: list[dict[str, Any]], right: list[dict[str, Any]], *, threshold: float = 0.68) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    pairs: list[tuple[float, int, int]] = []
    for i, item in enumerate(left):
        lt = item.get("normalized_text") or normalize(item.get("text") or "")
        if not lt:
            continue
        for j, other in enumerate(right):
            rt = other.get("normalized_text") or normalize(other.get("text") or other.get("text_preview") or "")
            if not rt:
                continue
            score = SequenceMatcher(None, lt, rt).ratio()
            if lt in rt or rt in lt:
                score = max(score, 0.82)
            if score >= threshold:
                pairs.append((score, i, j))
    pairs.sort(reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, i, j in pairs:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matches.append({"left_index": i, "right_index": j, "score": score})
    return matches, used_left, used_right


def classify_candidate_role(candidate: dict[str, Any]) -> str:
    role_blob = " ".join(
        str(candidate.get(key) or "")
        for key in ("raw_type", "current_role", "current_canonical_type")
    ).casefold()
    if "table" in role_blob:
        return "table"
    if "figure" in role_blob or "image" in role_blob or "chart" in role_blob:
        return "figure_or_crop"
    if "algorithm" in role_blob or "code" in role_blob:
        return "algorithm_or_code"
    if "paragraph" in role_blob or "text" in role_blob or "body" in role_blob:
        return "paragraph"
    return "unknown"


def detect_compile_risks(doc_id: str, doc_dir: Path, generated_tex: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    in_algorithm_env = False
    for line_no, line in enumerate(generated_tex.splitlines(), start=1):
        lower = line.casefold()
        if "\\begin{algorithm" in lower or "\\begin{algorithmic" in lower:
            in_algorithm_env = True
        is_pseudocode_line = bool(PSEUDOCODE_ANCHOR_RE.search(line) or ALGORITHM_CAPTION_RE.match(line))
        if not (in_algorithm_env or is_pseudocode_line):
            if "\\end{algorithm" in lower or "\\end{algorithmic" in lower:
                in_algorithm_env = False
            continue
        reasons = []
        if RISKY_UNICODE_RE.search(line):
            reasons.append("unicode_math_or_symbol")
        if UNESCAPED_SPECIAL_RE.search(line):
            reasons.append("unescaped_special_char")
        if "\\begin{algorithm}" in line and "\\usepackage{algorithm" not in generated_tex:
            reasons.append("algorithm_environment_package_risk")
        if reasons:
            risks.append({"doc_id": doc_id, "source": "generated_tex", "line_no": line_no, "text": compact(line), "risk_reasons": reasons})
        if "\\end{algorithm" in lower or "\\end{algorithmic" in lower:
            in_algorithm_env = False
    for candidate in candidates:
        text = candidate.get("text_preview") or ""
        reasons = []
        if RISKY_UNICODE_RE.search(text):
            reasons.append("unicode_math_or_symbol")
        if MATH_GLYPH_RE.search(text) and candidate.get("raw_type") in {"code", "algorithm"}:
            reasons.append("math_glyph_in_code_region")
        if UNESCAPED_SPECIAL_RE.search(text):
            reasons.append("unescaped_special_char_in_candidate")
        if reasons:
            risks.append(
                {
                    "doc_id": doc_id,
                    "source": "v8_candidate",
                    "candidate_id": candidate.get("candidate_id"),
                    "page_idx": candidate.get("page_idx"),
                    "text": compact(text),
                    "risk_reasons": reasons,
                }
            )
    compile_report = load_json(doc_dir / "compile_report.json", {})
    if isinstance(compile_report, dict) and compile_report.get("success") not in {None, "not_run", True}:
        risks.append({"doc_id": doc_id, "source": "compile_report", "text": compact(compile_report), "risk_reasons": ["existing_compile_failure"]})
    return risks


def audit_doc(doc_dir: Path, output_dir: Path, max_examples: int) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    doc_id = doc_dir.name.split("_", 1)[-1]
    content_paths = sorted(doc_dir.glob("*_content_list_v8_contentlist_merge_hint.json"))
    content_payload = load_json(content_paths[0], {}) if content_paths else {}
    document_ir = load_json(doc_dir / "document_ir.json", {})
    gold_structure = load_json(doc_dir / "gold_structure.json", {})
    pred_structure = load_json(doc_dir / "generated_structure.json", {})
    metrics = load_json(doc_dir / "structure_metrics.json", {})
    generated_tex = (doc_dir / "generated.tex").read_text(encoding="utf-8") if (doc_dir / "generated.tex").exists() else ""

    candidates = detect_algorithm_candidates(content_payload, document_ir)
    caption_candidates = [candidate for candidate in candidates if "caption" in candidate.get("candidate_kind", [])]
    body_candidates = [candidate for candidate in candidates if "body" in candidate.get("candidate_kind", [])]

    gold_algorithms, gold_captions = extract_algorithm_blocks(gold_structure, "gold")
    pred_algorithms, pred_captions = extract_algorithm_blocks(pred_structure, "pred")
    pred_algorithm_like_paragraphs = extract_algorithm_like_paragraphs(pred_structure)

    candidate_caption_matches, used_gold_caption, used_candidate_caption = match_texts(gold_captions, caption_candidates, threshold=0.62)
    pred_caption_matches, used_gold_pred_caption, used_pred_caption = match_texts(gold_captions, pred_captions, threshold=0.68)
    candidate_body_matches, used_gold_body, used_candidate_body = match_texts(gold_algorithms, body_candidates, threshold=0.45)
    pred_body_matches, used_gold_pred_body, used_pred_body = match_texts(gold_algorithms, pred_algorithms, threshold=0.45)

    unmatched_gold_caption = [gold_captions[i] for i in range(len(gold_captions)) if i not in used_gold_pred_caption]
    unmatched_gold_body = [
        gold_algorithms[i]
        for i in range(len(gold_algorithms))
        if i not in used_gold_pred_body and (gold_algorithms[i].get("normalized_text") or "").strip()
    ]
    no_v8_caption_match = [gold_captions[i] for i in range(len(gold_captions)) if i not in used_gold_caption]
    no_v8_body_match = [
        gold_algorithms[i]
        for i in range(len(gold_algorithms))
        if i not in used_gold_body and (gold_algorithms[i].get("normalized_text") or "").strip()
    ]

    role_counts = Counter(classify_candidate_role(candidate) for candidate in candidates)
    algorithm_as_table = [candidate for candidate in candidates if classify_candidate_role(candidate) == "table"]
    algorithm_as_paragraph = [candidate for candidate in candidates if classify_candidate_role(candidate) == "paragraph"] + pred_algorithm_like_paragraphs
    algorithm_as_figure = [candidate for candidate in candidates if classify_candidate_role(candidate) == "figure_or_crop"]
    caption_without_body = len(caption_candidates) if caption_candidates and not body_candidates else 0
    body_without_caption = len(body_candidates) if body_candidates and not caption_candidates else 0

    compile_risks = detect_compile_risks(doc_id, doc_dir, generated_tex, candidates)

    failure_cases: list[dict[str, Any]] = []
    if (gold_captions or gold_algorithms) and not candidates:
        failure_cases.append({"failure_type": "NO_V8_ALGORITHM_CANDIDATE", "reason": "gold algorithm/caption exists but no v8 algorithm candidate was detected"})
    for item in no_v8_caption_match:
        failure_cases.append({"failure_type": "NO_V8_ALGORITHM_CANDIDATE", "gold_text": item.get("text"), "reason": "gold algorithm caption has no v8 candidate match"})
    if caption_candidates and not body_candidates:
        failure_cases.append({"failure_type": "CAPTION_EXISTS_BODY_MISSING", "candidate_count": len(caption_candidates)})
    if body_candidates and not caption_candidates:
        failure_cases.append({"failure_type": "BODY_EXISTS_CAPTION_MISSING", "candidate_count": len(body_candidates)})
    for candidate in algorithm_as_table[:max_examples]:
        failure_cases.append({"failure_type": "ALGORITHM_AS_TABLE", "candidate": candidate})
    for candidate in algorithm_as_paragraph[:max_examples]:
        failure_cases.append({"failure_type": "ALGORITHM_AS_PARAGRAPH", "candidate": candidate})
    for candidate in algorithm_as_figure[:max_examples]:
        failure_cases.append({"failure_type": "ALGORITHM_AS_FIGURE_CROP", "candidate": candidate})
    if candidates and (unmatched_gold_caption or unmatched_gold_body):
        failure_cases.append({"failure_type": "candidate_EXISTS_BUT_NOT_RENDERED".upper(), "reason": "v8 candidate exists but gold algorithm/caption remains unmatched in generated structure"})
    for risk in compile_risks[:max_examples]:
        failure_cases.append({"failure_type": "COMPILE_RISK_PSEUDOCODE", "risk": risk})
    if not failure_cases and candidates:
        failure_cases.append({"failure_type": "UNKNOWN", "reason": "candidate detected without a specific failure classification"})

    safe_id = doc_id.replace("/", "_")
    audit_payload = {
        "schema_version": "v8_algorithm_region_baseline_audit_v1",
        "doc_id": doc_id,
        "doc_dir": str(doc_dir),
        "v8_content_path": str(content_paths[0]) if content_paths else None,
        "gold_algorithms": gold_algorithms[:max_examples],
        "gold_algorithm_captions": gold_captions[:max_examples],
        "pred_algorithms": pred_algorithms[:max_examples],
        "pred_algorithm_captions": pred_captions[:max_examples],
        "failure_cases": failure_cases,
        "candidate_caption_matches": candidate_caption_matches[:max_examples],
        "candidate_body_matches": candidate_body_matches[:max_examples],
        "pred_caption_matches": pred_caption_matches[:max_examples],
        "pred_body_matches": pred_body_matches[:max_examples],
    }
    candidate_payload = {
        "schema_version": "v8_algorithm_candidates_v1",
        "doc_id": doc_id,
        "algorithm_candidates": candidates,
        "caption_candidates": caption_candidates,
        "body_candidates": body_candidates,
        "role_counts": dict(role_counts),
    }
    compile_payload = {
        "schema_version": "v8_algorithm_compile_risk_v1",
        "doc_id": doc_id,
        "compile_risks": compile_risks,
    }
    write_json(output_dir / f"algorithm_region_audit_{safe_id}.json", audit_payload)
    write_json(output_dir / f"algorithm_candidates_{safe_id}.json", candidate_payload)
    write_json(output_dir / f"algorithm_compile_risk_{safe_id}.json", compile_payload)

    float_metric = metrics.get("float_caption_attachment_accuracy") or {}
    summary = {
        "doc_id": doc_id,
        "gold_algorithm_count": max(len(gold_algorithms), len(gold_captions)),
        "gold_algorithm_block_count": len(gold_algorithms),
        "gold_algorithm_caption_count": len(gold_captions),
        "pred_algorithm_count": max(len(pred_algorithms), len(pred_captions)),
        "pred_algorithm_block_count": len(pred_algorithms),
        "pred_algorithm_caption_count": len(pred_captions),
        "v8_algorithm_candidate_count": len(candidates),
        "v8_algorithm_caption_candidate_count": len(caption_candidates),
        "v8_algorithm_body_candidate_count": len(body_candidates),
        "algorithm_caption_missing_count": max(0, len(gold_captions) - len(used_gold_pred_caption)),
        "algorithm_body_missing_count": max(0, len(gold_algorithms) - len(used_gold_pred_body)),
        "algorithm_as_table_count": len(algorithm_as_table),
        "algorithm_as_paragraph_count": len(algorithm_as_paragraph),
        "algorithm_as_figure_or_crop_count": len(algorithm_as_figure),
        "algorithm_caption_without_body_count": caption_without_body,
        "algorithm_body_without_caption_count": body_without_caption,
        "pseudocode_compile_risk_count": len(compile_risks),
        "no_v8_candidate_match_count": len(no_v8_caption_match) + len(no_v8_body_match),
        "candidate_exists_but_not_rendered_count": (len(unmatched_gold_caption) + len(unmatched_gold_body)) if candidates else 0,
        "rendered_not_converted_count": 0,
        "converted_not_matched_count": max(0, len(pred_captions) - len(used_pred_caption)),
        "false_algorithm_candidate_count": sum(1 for candidate in candidates if "caption" in candidate.get("candidate_kind", []) and FALSE_ALGORITHM_REFERENCE_RE.match(candidate.get("text_preview", ""))),
        "float_caption_attachment_accuracy": as_float(float_metric.get("score") if isinstance(float_metric, dict) else float_metric),
        "generated_structure_validity": as_float((metrics.get("generated_structure_validity") or {}).get("score") if isinstance(metrics.get("generated_structure_validity"), dict) else metrics.get("generated_structure_validity")),
        "macro_structure_score_body": as_float((metrics.get("macro_structure_score_body") or {}).get("score") if isinstance(metrics.get("macro_structure_score_body"), dict) else metrics.get("macro_structure_score_body")),
        "major_failure_type": major_failure_type(failure_cases),
    }
    examples = defaultdict(list)
    for case in failure_cases:
        bucket = case.get("failure_type") or "UNKNOWN"
        if len(examples[bucket]) < max_examples:
            examples[bucket].append(
                {
                    "doc_id": doc_id,
                    "preview": compact(case.get("gold_text") or case.get("reason") or case.get("candidate", {}).get("text_preview") or case.get("risk", {}).get("text")),
                    "failure_type": bucket,
                }
            )
    return summary, examples


def major_failure_type(failure_cases: list[dict[str, Any]]) -> str:
    priority = [
        "NO_V8_ALGORITHM_CANDIDATE",
        "CAPTION_EXISTS_BODY_MISSING",
        "BODY_EXISTS_CAPTION_MISSING",
        "ALGORITHM_AS_TABLE",
        "ALGORITHM_AS_PARAGRAPH",
        "ALGORITHM_AS_FIGURE_CROP",
        "CANDIDATE_EXISTS_BUT_NOT_RENDERED",
        "COMPILE_RISK_PSEUDOCODE",
    ]
    counts = Counter(case.get("failure_type") or "UNKNOWN" for case in failure_cases)
    for key in priority:
        if counts.get(key):
            return key
    return counts.most_common(1)[0][0] if counts else "NONE"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sum_keys = [
        "gold_algorithm_count",
        "gold_algorithm_block_count",
        "gold_algorithm_caption_count",
        "pred_algorithm_count",
        "pred_algorithm_block_count",
        "pred_algorithm_caption_count",
        "v8_algorithm_candidate_count",
        "v8_algorithm_caption_candidate_count",
        "v8_algorithm_body_candidate_count",
        "algorithm_caption_missing_count",
        "algorithm_body_missing_count",
        "algorithm_as_table_count",
        "algorithm_as_paragraph_count",
        "algorithm_as_figure_or_crop_count",
        "algorithm_caption_without_body_count",
        "algorithm_body_without_caption_count",
        "pseudocode_compile_risk_count",
        "no_v8_candidate_match_count",
        "candidate_exists_but_not_rendered_count",
        "rendered_not_converted_count",
        "converted_not_matched_count",
        "false_algorithm_candidate_count",
    ]
    summary: dict[str, Any] = {"docs": len(rows)}
    for key in sum_keys:
        summary[key] = sum(as_int(row.get(key)) for row in rows)
    for key in ("float_caption_attachment_accuracy", "generated_structure_validity", "macro_structure_score_body"):
        values = [as_float(row.get(key)) for row in rows if as_float(row.get(key)) is not None]
        summary[f"mean_{key}"] = sum(values) / len(values) if values else None
    summary["failure_type_counts"] = dict(Counter(row.get("major_failure_type") or "UNKNOWN" for row in rows))
    return summary


def merge_examples(target: dict[str, list[dict[str, Any]]], source: dict[str, list[dict[str, Any]]], limit: int) -> None:
    for key, rows in source.items():
        bucket = target.setdefault(key, [])
        for row in rows:
            if len(bucket) >= limit:
                break
            bucket.append(row)


def top_problem_docs(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        score = (
            as_int(row.get("algorithm_caption_missing_count"))
            + as_int(row.get("algorithm_body_missing_count"))
            + as_int(row.get("no_v8_candidate_match_count"))
            + as_int(row.get("algorithm_as_paragraph_count"))
            + as_int(row.get("algorithm_as_table_count"))
            + as_int(row.get("pseudocode_compile_risk_count"))
        )
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1]["doc_id"]))
    return [row for _, row in scored[:limit]]


def decide(summary: dict[str, Any]) -> str:
    gold = as_int(summary.get("gold_algorithm_count"))
    candidates = as_int(summary.get("v8_algorithm_candidate_count"))
    gold_captions = as_int(summary.get("gold_algorithm_caption_count"))
    caption_candidates = as_int(summary.get("v8_algorithm_caption_candidate_count"))
    no_match = as_int(summary.get("no_v8_candidate_match_count"))
    missing = as_int(summary.get("algorithm_caption_missing_count")) + as_int(summary.get("algorithm_body_missing_count"))
    if gold == 0 and candidates == 0:
        return "audit_inconclusive"
    if (
        candidates < max(3, gold // 2)
        or (gold_captions and caption_candidates < int(gold_captions * 0.75))
        or no_match >= max(3, int(gold * 0.4))
        or no_match >= max(3, missing // 2)
    ):
        return "need_lower_level_algorithm_candidate_extraction"
    return "ready_for_algorithm_region_phase0"


def write_report(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any], examples: dict[str, list[dict[str, Any]]], decision: str) -> None:
    failure_counts = Counter()
    for row in rows:
        failure_counts[row.get("major_failure_type") or "UNKNOWN"] += 1
    problem_docs = top_problem_docs(rows, 20)

    def table_line(values: list[Any]) -> str:
        return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"

    lines: list[str] = []
    lines.append("# V8 AlgorithmRegion Baseline Audit")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- selected200 docs analyzed: {summary.get('docs')}")
    lines.append("- no training / no MinerU / no relabel / no rebuild / no GNN / no renderer changes")
    lines.append("- v8 facts used: `*_content_list_v8_contentlist_merge_hint.json` plus current `document_ir.json` provenance")
    lines.append("- no fallback to old v7 fact layer; legacy names such as `source_v7_ids` / `v7_id` are provenance names only")
    lines.append("")
    lines.append("Current mainline remains:")
    lines.append("")
    lines.append("```text")
    lines.append("v8 full observable facts")
    lines.append("  -> v8 atomic/reflow")
    lines.append("  -> deterministic merge + contentlist merge hint")
    lines.append("  -> RenderTreeIR")
    lines.append("  -> IR renderer")
    lines.append("```")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(table_line(["Metric", "Value"]))
    lines.append(table_line(["---", "---:"]))
    for key in [
        "gold_algorithm_count",
        "pred_algorithm_count",
        "v8_algorithm_candidate_count",
        "v8_algorithm_caption_candidate_count",
        "v8_algorithm_body_candidate_count",
        "algorithm_caption_missing_count",
        "algorithm_body_missing_count",
        "no_v8_candidate_match_count",
        "algorithm_as_table_count",
        "algorithm_as_paragraph_count",
        "algorithm_as_figure_or_crop_count",
        "pseudocode_compile_risk_count",
    ]:
        lines.append(table_line([key, summary.get(key)]))
    lines.append("")
    lines.append("## Failure Breakdown")
    lines.append("")
    lines.append(table_line(["failure_type", "doc_count"]))
    lines.append(table_line(["---", "---:"]))
    for key, count in failure_counts.most_common():
        lines.append(table_line([key, count]))
    lines.append("")
    lines.append("## Top Problem Docs")
    lines.append("")
    lines.append(table_line(["doc_id", "gold_alg", "pred_alg", "v8_candidates", "caption_missing", "body_missing", "no_v8_match", "major_failure_type"]))
    lines.append(table_line(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---"]))
    for row in problem_docs:
        lines.append(
            table_line(
                [
                    row.get("doc_id"),
                    row.get("gold_algorithm_count"),
                    row.get("pred_algorithm_count"),
                    row.get("v8_algorithm_candidate_count"),
                    row.get("algorithm_caption_missing_count"),
                    row.get("algorithm_body_missing_count"),
                    row.get("no_v8_candidate_match_count"),
                    row.get("major_failure_type"),
                ]
            )
        )
    lines.append("")
    lines.append("## Examples")
    lines.append("")
    wanted = [
        "NO_V8_ALGORITHM_CANDIDATE",
        "CAPTION_EXISTS_BODY_MISSING",
        "BODY_EXISTS_CAPTION_MISSING",
        "ALGORITHM_AS_TABLE",
        "ALGORITHM_AS_PARAGRAPH",
        "COMPILE_RISK_PSEUDOCODE",
    ]
    for key in wanted:
        lines.append(f"### {key}")
        bucket = examples.get(key, [])[:10]
        if not bucket:
            lines.append("")
            lines.append("- none observed")
        else:
            for item in bucket:
                lines.append(f"- `{item.get('doc_id')}`: {compact(item.get('preview'), 300)}")
        lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    candidate_count = as_int(summary.get("v8_algorithm_candidate_count"))
    no_v8 = as_int(summary.get("no_v8_candidate_match_count"))
    caption_missing = as_int(summary.get("algorithm_caption_missing_count"))
    body_missing = as_int(summary.get("algorithm_body_missing_count"))
    as_table = as_int(summary.get("algorithm_as_table_count"))
    as_para = as_int(summary.get("algorithm_as_paragraph_count"))
    as_crop = as_int(summary.get("algorithm_as_figure_or_crop_count"))
    lines.append(f"1. Candidate availability: v8 detected {candidate_count} algorithm/pseudocode candidates; {no_v8} gold algorithm/caption units still had no v8 candidate match.")
    lines.append(f"2. Caption vs body: caption missing = {caption_missing}; body missing = {body_missing}.")
    lines.append(f"3. Misclassification tendency: as_table = {as_table}, as_paragraph = {as_para}, as_figure_or_crop = {as_crop}.")
    lines.append("4. Rendering strategy: this audit does not choose a renderer, but compile-risk examples show pseudocode needs a safe text/verbatim-style fallback before any algorithm environment is enabled by default.")
    lines.append("5. ROI support: if no-v8-candidate remains high, the next useful work is lower-level ROI/fact extraction, not renderer patching.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"**{decision}**")
    if decision == "ready_for_algorithm_region_phase0":
        lines.append("")
        lines.append("v8 candidates are sufficient enough that an AlgorithmRegion Phase 0 grouping/materialization audit can be considered next.")
    elif decision == "need_lower_level_algorithm_candidate_extraction":
        lines.append("")
        lines.append("The dominant blocker is candidate availability / ROI extraction. Do not start by patching the renderer; first improve lower-level algorithm/code region facts.")
    else:
        lines.append("")
        lines.append("Artifacts are insufficient or too sparse to choose a safe implementation path.")
    (output_dir / "ALGORITHM_REGION_BASELINE_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def readiness_report(output_dir: Path, missing: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ALGORITHM_REGION_BASELINE_AUDIT_REPORT.md").write_text(
        "# V8 AlgorithmRegion Baseline Audit Readiness Report\n\n"
        "Required artifacts were missing, so the audit stopped without guessing.\n\n"
        + "\n".join(f"- {item}" for item in missing)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    doc_dirs = collect_doc_dirs(args.baseline_root)
    missing = []
    if not doc_dirs:
        missing.append(str(args.baseline_root))
    if missing:
        readiness_report(args.output_dir, missing)
        return 2

    if args.doc_ids:
        selected = [doc_dirs[doc_id] for doc_id in args.doc_ids if doc_id in doc_dirs]
    else:
        selected = list(doc_dirs.values())
    if args.limit is not None:
        selected = selected[: args.limit]

    rows: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = {}
    for doc_dir in selected:
        row, doc_examples = audit_doc(doc_dir, args.output_dir, args.max_examples)
        rows.append(row)
        merge_examples(examples, doc_examples, args.max_examples)

    summary = aggregate(rows)
    decision = decide(summary)
    summary_payload = {
        "schema_version": "v8_algorithm_region_baseline_summary_v1",
        "baseline_root": str(args.baseline_root),
        "output_dir": str(args.output_dir),
        "summary": summary,
        "decision": decision,
        "failure_type_counts": summary.get("failure_type_counts", {}),
        "top_problem_docs": top_problem_docs(rows, 20),
        "v8_only_confirmation": {
            "current_fact_layer": "v8 full observable facts",
            "no_fallback_to_old_v7": True,
            "legacy_field_names": "source_v7_ids / v7_id are provenance names only",
        },
    }
    write_json(args.output_dir / "algorithm_region_baseline_summary.json", summary_payload)
    write_csv(args.output_dir / "algorithm_region_baseline_summary.csv", rows)
    write_report(args.output_dir, rows, summary, examples, decision)

    print(json.dumps({"docs": len(rows), "output_dir": str(args.output_dir), "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
