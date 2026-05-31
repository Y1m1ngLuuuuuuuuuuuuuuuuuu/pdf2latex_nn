#!/usr/bin/env python3
"""Build a manual visual review pack for float-caption sidecar audits.

This is an audit-only utility. It reads existing sidecar CSV files and optional
PDF pages, then writes review manifests, lightweight overlays, and a static
HTML/Markdown review sheet. It does not modify predictions, renderer outputs,
or evaluator definitions.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REVIEWER_LABELS = [
    "converter_projection_fix",
    "renderer_propagation_fix",
    "resolver_anchor_fix_safe",
    "resolver_anchor_fix_risky",
    "subfigure_non_isomorphic",
    "source_pdf_non_isomorphic",
    "gold_unobservable",
    "evaluator_text_normalization",
    "correct_current_behavior",
    "unknown",
]

REVIEWER_DECISIONS = [
    "safe_patch_candidate",
    "unsafe_patch_candidate",
    "limitation_only",
    "needs_second_reviewer",
    "exclude_from_claim",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) >= 4:
            return [float(parsed[i]) for i in range(4)]
    except Exception:
        pass
    parts = [p for p in text.replace(",", " ").replace(";", " ").split() if p]
    if len(parts) >= 4:
        try:
            return [float(parts[i]) for i in range(4)]
        except ValueError:
            return None
    return None


def trunc(text: Any, limit: int = 180) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def stable_pick(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    if count <= 0 or not rows:
        return []
    ordered = sorted(
        rows,
        key=lambda r: (
            r.get("doc_id", ""),
            r.get("caption_candidate_id", ""),
            r.get("float_candidate_id", ""),
            r.get("gold_caption_id", ""),
        ),
    )
    if len(ordered) <= count:
        return ordered
    rnd = random.Random(seed)
    indices = list(range(len(ordered)))
    rnd.shuffle(indices)
    chosen = sorted(indices[:count])
    return [ordered[i] for i in chosen]


def load_manifest_pdfs(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        rows = obj.get("docs") or obj.get("items") or obj.get("selected") or obj.get("manifest") or []
    else:
        rows = obj
    out: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("doc_id") or row.get("id") or "")
        pdf = str(row.get("original_pdf_path") or row.get("pdf") or "")
        if doc_id and pdf:
            out[doc_id] = pdf
    return out


class ReviewBuilder:
    def __init__(
        self,
        sidecar_root: Path,
        fresh_root: Path,
        output_dir: Path,
        max_cases: int,
        generate_images: bool,
    ) -> None:
        self.sidecar_root = sidecar_root
        self.fresh_root = fresh_root
        self.output_dir = output_dir
        self.max_cases = max_cases
        self.generate_images = generate_images
        self.caption_rows = read_csv(sidecar_root / "caption_candidates_sidecar.csv")
        self.float_rows = read_csv(sidecar_root / "float_candidates_sidecar.csv")
        self.pair_rows = read_csv(sidecar_root / "caption_float_pair_features.csv")
        self.lost_rows = read_csv(sidecar_root / "materialized_converter_lost_121_audit.csv")
        self.summary_rows = read_csv(sidecar_root / "float_caption_true_counterfactual_summary.csv")
        self.caption_by_key = {(r.get("doc_id", ""), r.get("caption_candidate_id", "")): r for r in self.caption_rows}
        self.float_by_key = {(r.get("doc_id", ""), r.get("float_candidate_id", "")): r for r in self.float_rows}
        self.pairs_by_caption: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in self.pair_rows:
            self.pairs_by_caption[(row.get("doc_id", ""), row.get("caption_candidate_id", ""))].append(row)
        self.pdf_by_doc = load_manifest_pdfs(fresh_root / "fresh_heldout_selected200_manifest.json")
        self.used_keys: set[tuple[str, str, str, str]] = set()
        self.cases: list[dict[str, Any]] = []

    def pick_caption_for_lost_row(self, row: dict[str, str]) -> dict[str, str] | None:
        doc_id = row.get("doc_id", "")
        candidate_ids = [x.strip() for x in row.get("candidate_ids", "").split(";") if x.strip()]
        for cid in candidate_ids:
            cap = self.caption_by_key.get((doc_id, cid))
            if cap:
                return cap
        # Fallback: any caption in this doc whose text overlaps the gold preview.
        preview = str(row.get("gold_text_preview", "")).lower()
        preview_words = set(preview.split()[:10])
        best: tuple[int, dict[str, str]] | None = None
        for cap in self.caption_rows:
            if cap.get("doc_id") != doc_id:
                continue
            words = set(str(cap.get("normalized_text") or cap.get("text") or "").lower().split())
            score = len(preview_words & words)
            if score and (best is None or score > best[0]):
                best = (score, cap)
        return best[1] if best else None

    def best_pair_for_caption(self, cap: dict[str, str]) -> dict[str, str] | None:
        key = (cap.get("doc_id", ""), cap.get("caption_candidate_id", ""))
        rows = self.pairs_by_caption.get(key, [])
        if not rows:
            return None
        return max(rows, key=lambda r: num(r.get("proposed_total_score")))

    def add_case(
        self,
        category: str,
        cap: dict[str, str] | None,
        float_id: str = "",
        pair: dict[str, str] | None = None,
        current_status: str = "",
        proposed_status: str = "",
        risk_reason: str = "",
        why_selected: str = "",
        source_row: dict[str, str] | None = None,
    ) -> bool:
        source_row = source_row or {}
        doc_id = (cap or pair or source_row).get("doc_id", "")
        caption_id = (cap or {}).get("caption_candidate_id", "") or (pair or {}).get("caption_candidate_id", "")
        float_id = float_id or (pair or {}).get("float_candidate_id", "") or (cap or {}).get("paired_float_id", "")
        if not doc_id:
            return False
        dedupe = (category, doc_id, caption_id, float_id)
        if dedupe in self.used_keys:
            return False
        self.used_keys.add(dedupe)
        float_row = self.float_by_key.get((doc_id, float_id), {})
        if pair is None and caption_id:
            pair = self.best_pair_for_caption(cap or {})
        page_idx = ""
        if cap and cap.get("page_idx", "") != "":
            page_idx = cap.get("page_idx", "")
        elif pair and pair.get("caption_page", "") != "":
            page_idx = pair.get("caption_page", "")
        elif float_row.get("page_idx", "") != "":
            page_idx = float_row.get("page_idx", "")
        case = {
            "case_id": f"FCMR_{len(self.cases) + 1:04d}",
            "doc_id": doc_id,
            "page_idx": page_idx,
            "category": category,
            "caption_candidate_id": caption_id,
            "float_candidate_id": float_id,
            "caption_text": (cap or {}).get("text", "") or source_row.get("gold_text_preview", ""),
            "float_type": float_row.get("type", "") or (cap or {}).get("paired_float_type", ""),
            "current_status": current_status or (cap or {}).get("current_gap_category", "") or source_row.get("classification", ""),
            "proposed_status": proposed_status,
            "risk_reason": risk_reason,
            "why_selected": why_selected,
            "caption_bbox": (cap or {}).get("bbox", "") or (pair or {}).get("caption_bbox", ""),
            "float_bbox": float_row.get("bbox", "") or (pair or {}).get("float_bbox", ""),
            "pair_score": (pair or {}).get("proposed_total_score", ""),
            "pair_current_renderer_attached": (pair or {}).get("current_renderer_attached", ""),
            "pair_current_gold_attached": (pair or {}).get("current_gold_attached", ""),
            "pair_official_correct": (pair or {}).get("official_correct", ""),
        }
        self.cases.append(case)
        return True

    def add_lost_category(self, classification: str, target: int, category: str, risk_reason: str) -> int:
        rows = [r for r in self.lost_rows if r.get("classification") == classification]
        added = 0
        for row in stable_pick(rows, target, seed=1000 + len(self.cases)):
            cap = self.pick_caption_for_lost_row(row)
            pair = self.best_pair_for_caption(cap) if cap else None
            float_id = (cap or {}).get("paired_float_id", "") or (pair or {}).get("float_candidate_id", "")
            if self.add_case(
                category=category,
                cap=cap,
                float_id=float_id,
                pair=pair,
                current_status=classification,
                proposed_status="manual_review",
                risk_reason=risk_reason,
                why_selected=f"sampled from materialized_converter_lost_121: {classification}",
                source_row=row,
            ):
                added += 1
        return added

    def add_pair_category(
        self,
        rows: list[dict[str, str]],
        target: int,
        category: str,
        risk_reason: str,
        why_selected: str,
    ) -> int:
        added = 0
        sorted_rows = sorted(rows, key=lambda r: num(r.get("proposed_total_score")), reverse=True)
        for pair in stable_pick(sorted_rows[: max(target * 5, target)], target, seed=2000 + len(self.cases)):
            cap = self.caption_by_key.get((pair.get("doc_id", ""), pair.get("caption_candidate_id", "")))
            if self.add_case(
                category=category,
                cap=cap,
                float_id=pair.get("float_candidate_id", ""),
                pair=pair,
                current_status=(cap or {}).get("current_gap_category", pair.get("caption_gap_category", "")),
                proposed_status=f"candidate_pair_score={pair.get('proposed_total_score', '')}",
                risk_reason=risk_reason,
                why_selected=why_selected,
            ):
                added += 1
        return added

    def build_cases(self) -> dict[str, int]:
        # The requested raw quotas exceed 80, so this keeps all critical pools
        # while staying inside the manual-review budget.
        quota = {
            "generated_tex_contains_caption_but_converter_missed": 8,
            "comparison_matcher_failed_text_normalization": 12,
            "renderer_materialized_but_not_in_generated_tex": 10,
            "subfigure_boundary_mismatch": 9,
            "source_pdf_float_non_isomorphism": 0,
            "unknown_needs_manual": 12,
            "conservative_same_page_high_risk": 12,
            "current_materialized_only_high_risk": 6,
            "positive_control": 6,
            "negative_control": 5,
        }
        stats: dict[str, int] = {}
        stats["generated_tex_contains_caption_but_converter_missed"] = self.add_lost_category(
            "generated_tex_contains_caption_but_converter_missed",
            quota["generated_tex_contains_caption_but_converter_missed"],
            "converter_projection_clean_pool",
            "Generated TeX appears to contain the caption, but conversion did not preserve the matched caption/anchor.",
        )
        stats["comparison_matcher_failed_text_normalization"] = self.add_lost_category(
            "comparison_matcher_failed_text_normalization",
            quota["comparison_matcher_failed_text_normalization"],
            "text_normalization_pool",
            "Automated audit suggests text normalization or comparison matching may be the bottleneck.",
        )
        stats["renderer_materialized_but_not_in_generated_tex"] = self.add_lost_category(
            "renderer_materialized_but_not_in_generated_tex",
            quota["renderer_materialized_but_not_in_generated_tex"],
            "renderer_projection_pool",
            "Materialization metadata exists but generated LaTeX did not retain the caption evidence.",
        )
        stats["subfigure_boundary_mismatch"] = self.add_lost_category(
            "subfigure_boundary_mismatch",
            quota["subfigure_boundary_mismatch"],
            "subfigure_non_isomorphism_pool",
            "Subfigure/multi-panel boundary may not be isomorphic between PDF and source target.",
        )
        stats["unknown_needs_manual"] = self.add_lost_category(
            "unknown_needs_manual",
            quota["unknown_needs_manual"],
            "unknown_manual_pool",
            "Automated sidecar could not classify this gap.",
        )

        no_veto_same_page = [
            r
            for r in self.pair_rows
            if r.get("page_delta") == "0"
            and boolish(r.get("current_gold_attached"))
            and not boolish(r.get("official_correct"))
            and not boolish(r.get("veto_body_reference"))
            and not boolish(r.get("veto_regex_only"))
            and not boolish(r.get("veto_type_mismatch"))
        ]
        stats["conservative_same_page_high_risk"] = self.add_pair_category(
            no_veto_same_page,
            quota["conservative_same_page_high_risk"],
            "resolver_conservative_same_page_high_risk",
            "This family rescued many captions but also introduced many wrong matches in simulation.",
            "high-score same-page proposed rescue from high-risk resolver family",
        )

        current_materialized = [
            r
            for r in self.pair_rows
            if boolish(r.get("current_renderer_attached"))
            and boolish(r.get("current_gold_attached"))
            and not boolish(r.get("official_correct"))
        ]
        stats["current_materialized_only_high_risk"] = self.add_pair_category(
            current_materialized,
            quota["current_materialized_only_high_risk"],
            "resolver_current_materialized_high_risk",
            "Current-materialized-only simulation has nontrivial wrong-match risk.",
            "current renderer already materialized a candidate pair, but official attachment is not correct",
        )

        positives = [r for r in self.caption_rows if r.get("current_gap_category") == "caption_attached_correctly"]
        added = 0
        for cap in stable_pick(positives, quota["positive_control"], seed=3001):
            pair = self.best_pair_for_caption(cap)
            if self.add_case(
                "positive_control_caption_attached_correctly",
                cap,
                float_id=(cap.get("paired_float_id", "") or (pair or {}).get("float_candidate_id", "")),
                pair=pair,
                current_status="caption_attached_correctly",
                proposed_status="positive_control",
                risk_reason="Known correct official attachment control.",
                why_selected="clean positive control",
            ):
                added += 1
        stats["positive_control"] = added

        negative_caps = [
            r
            for r in self.caption_rows
            if boolish(r.get("veto_body_reference"))
            or boolish(r.get("regex_only"))
            or boolish(r.get("diagnostic_only"))
            or boolish(r.get("veto_regex_only"))
        ]
        negative_pairs = [r for r in self.pair_rows if boolish(r.get("veto_type_mismatch"))]
        added = 0
        for cap in stable_pick(negative_caps, math.ceil(quota["negative_control"] / 2), seed=4001):
            if self.add_case(
                "negative_control_veto_or_regex",
                cap,
                float_id=cap.get("paired_float_id", ""),
                current_status=cap.get("current_gap_category", ""),
                proposed_status="negative_control",
                risk_reason="Regex-only/body-reference/diagnostic-only evidence should not be promoted.",
                why_selected="clean negative control",
            ):
                added += 1
        for pair in stable_pick(negative_pairs, quota["negative_control"] - added, seed=4002):
            cap = self.caption_by_key.get((pair.get("doc_id", ""), pair.get("caption_candidate_id", "")))
            if self.add_case(
                "negative_control_type_mismatch_veto",
                cap,
                float_id=pair.get("float_candidate_id", ""),
                pair=pair,
                current_status=(cap or {}).get("current_gap_category", ""),
                proposed_status="negative_control",
                risk_reason="Type mismatch is a veto candidate for a safe resolver.",
                why_selected="clean negative control",
            ):
                added += 1
        stats["negative_control"] = added

        if len(self.cases) > self.max_cases:
            self.cases = self.cases[: self.max_cases]
        for i, case in enumerate(self.cases, 1):
            case["case_id"] = f"FCMR_{i:04d}"
        return stats

    def render_case_image(self, case: dict[str, Any], image_dir: Path) -> tuple[str, str]:
        if not self.generate_images:
            return "", "image_generation_disabled"
        if not shutil.which("pdftoppm"):
            return "", "pdftoppm_not_found"
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
        except Exception as exc:
            return "", f"pil_unavailable:{exc}"
        doc_id = str(case.get("doc_id", ""))
        pdf = self.pdf_by_doc.get(doc_id, "")
        if not pdf or not Path(pdf).exists():
            return "", "pdf_not_found"
        try:
            page_idx = int(float(str(case.get("page_idx", "0"))))
        except ValueError:
            page_idx = 0
        image_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "page"
            cmd = [
                "pdftoppm",
                "-f",
                str(page_idx + 1),
                "-l",
                str(page_idx + 1),
                "-r",
                "144",
                "-png",
                pdf,
                str(prefix),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            if proc.returncode != 0:
                return "", f"pdftoppm_failed:{trunc(proc.stderr, 120)}"
            pngs = sorted(Path(tmp).glob("page*.png"))
            if not pngs:
                return "", "pdftoppm_no_output"
            image = Image.open(pngs[0]).convert("RGB")
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            scale = 2.0

            def box(bbox_text: str, color: tuple[int, int, int], width: int, label: str, dashed: bool = False) -> None:
                bbox = parse_bbox(bbox_text)
                if not bbox:
                    return
                x0, y0, x1, y1 = [v * scale for v in bbox[:4]]
                if dashed:
                    dash = 12
                    for x in range(int(x0), int(x1), dash * 2):
                        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=color, width=width)
                        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=color, width=width)
                    for y in range(int(y0), int(y1), dash * 2):
                        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=color, width=width)
                        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=color, width=width)
                else:
                    for offset in range(width):
                        draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)
                tx, ty = max(0, x0), max(0, y0 - 14)
                draw.rectangle([tx, ty, tx + 8 * len(label) + 4, ty + 12], fill=(255, 255, 255))
                draw.text((tx + 2, ty), label, fill=color, font=font)

            box(str(case.get("caption_bbox", "")), (0, 150, 70), 3, "caption")
            is_high_risk = "high_risk" in str(case.get("category", ""))
            box(str(case.get("float_bbox", "")), (20, 90, 220), 2, "float/current")
            box(str(case.get("float_bbox", "")), (255, 140, 0), 2, "proposed", dashed=True)
            if is_high_risk:
                cap_bbox = parse_bbox(case.get("caption_bbox"))
                float_bbox = parse_bbox(case.get("float_bbox"))
                if cap_bbox and float_bbox:
                    c0 = ((cap_bbox[0] + cap_bbox[2]) * scale / 2, (cap_bbox[1] + cap_bbox[3]) * scale / 2)
                    c1 = ((float_bbox[0] + float_bbox[2]) * scale / 2, (float_bbox[1] + float_bbox[3]) * scale / 2)
                    draw.line([c0, c1], fill=(220, 0, 0), width=3)
            header = f"{case.get('case_id')}  {doc_id}  page {page_idx + 1}"
            draw.rectangle([0, 0, min(image.width, 10 * len(header) + 18), 24], fill=(255, 255, 255))
            draw.text((8, 6), header, fill=(0, 0, 0), font=font)
            snippet = trunc(case.get("caption_text", ""), 90)
            draw.rectangle([0, image.height - 24, min(image.width, 8 * len(snippet) + 18), image.height], fill=(255, 255, 255))
            draw.text((8, image.height - 18), snippet, fill=(0, 90, 40), font=font)
            out = image_dir / f"case_{case.get('case_id')}_{doc_id}_p{page_idx}.png"
            image.save(out)
            return str(out.relative_to(self.output_dir)), ""

    def generate_images_for_cases(self) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        image_dir = self.output_dir / "review_images"
        for case in self.cases:
            rel, failure = self.render_case_image(case, image_dir)
            case["image_path"] = rel
            if failure:
                failures.append({"case_id": case.get("case_id", ""), "doc_id": case.get("doc_id", ""), "reason": failure})
        return failures

    def write_review_sheets(self) -> None:
        sheet_fields = [
            "case_id",
            "doc_id",
            "page_idx",
            "category",
            "image_path",
            "caption_text",
            "current_classification",
            "proposed_pair",
            "automated_reason",
            "reviewer_label",
            "reviewer_decision",
            "reviewer_notes",
        ]
        rows = []
        for c in self.cases:
            rows.append(
                {
                    "case_id": c.get("case_id", ""),
                    "doc_id": c.get("doc_id", ""),
                    "page_idx": c.get("page_idx", ""),
                    "category": c.get("category", ""),
                    "image_path": c.get("image_path", ""),
                    "caption_text": c.get("caption_text", ""),
                    "current_classification": c.get("current_status", ""),
                    "proposed_pair": c.get("float_candidate_id", ""),
                    "automated_reason": c.get("risk_reason", ""),
                    "reviewer_label": "",
                    "reviewer_decision": "",
                    "reviewer_notes": "",
                }
            )
        write_csv(self.output_dir / "manual_review_sheet.csv", rows, sheet_fields)
        write_json(self.output_dir / "manual_review_sheet.json", rows)

        lines = [
            "# Float-Caption Manual Review Sheet",
            "",
            "Reviewer labels: `" + "`, `".join(REVIEWER_LABELS) + "`.",
            "",
            "Reviewer decisions: `" + "`, `".join(REVIEWER_DECISIONS) + "`.",
            "",
        ]
        for c in self.cases:
            lines.extend(
                [
                    f"## {c.get('case_id')} - {c.get('category')}",
                    "",
                    f"- doc_id: `{c.get('doc_id')}`",
                    f"- page_idx: `{c.get('page_idx')}`",
                    f"- caption_candidate_id: `{c.get('caption_candidate_id')}`",
                    f"- float_candidate_id: `{c.get('float_candidate_id')}`",
                    f"- current_status: `{c.get('current_status')}`",
                    f"- proposed_status: `{c.get('proposed_status')}`",
                    f"- automated_reason: {c.get('risk_reason')}",
                    f"- caption_text: {trunc(c.get('caption_text'), 260)}",
                    "",
                ]
            )
            if c.get("image_path"):
                lines.append(f"![{c.get('case_id')}]({c.get('image_path')})")
                lines.append("")
            lines.extend(["Reviewer label:", "", "Reviewer decision:", "", "Notes:", "", "---", ""])
        (self.output_dir / "manual_review_sheet.md").write_text("\n".join(lines), encoding="utf-8")

    def write_html(self) -> None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in self.cases:
            grouped[str(c.get("category", ""))].append(c)
        parts = [
            "<!doctype html><meta charset='utf-8'><title>Float-Caption Manual Review Pack</title>",
            "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;}"
            ".case{border:1px solid #ccc;padding:12px;margin:12px 0;}img{max-width:100%;border:1px solid #ddd;}"
            "code{background:#f5f5f5;padding:1px 3px;} .meta{color:#555;}</style>",
            "<h1>Float-Caption Manual Review Pack</h1>",
            "<p>This pack is audit-only. It does not change renderer, evaluator, or predictions.</p>",
        ]
        for category, rows in sorted(grouped.items()):
            parts.append(f"<h2>{html.escape(category)} ({len(rows)})</h2>")
            for c in rows:
                parts.append("<div class='case'>")
                parts.append(f"<h3>{html.escape(str(c.get('case_id')))}</h3>")
                parts.append(
                    "<p class='meta'>doc: <code>{}</code> page: <code>{}</code> caption: <code>{}</code> float: <code>{}</code></p>".format(
                        html.escape(str(c.get("doc_id", ""))),
                        html.escape(str(c.get("page_idx", ""))),
                        html.escape(str(c.get("caption_candidate_id", ""))),
                        html.escape(str(c.get("float_candidate_id", ""))),
                    )
                )
                parts.append(f"<p><b>Caption:</b> {html.escape(trunc(c.get('caption_text'), 320))}</p>")
                parts.append(f"<p><b>Automated reason:</b> {html.escape(str(c.get('risk_reason', '')))}</p>")
                parts.append(
                    "<p><b>Reviewer labels:</b> "
                    + ", ".join(f"<code>{html.escape(x)}</code>" for x in REVIEWER_LABELS)
                    + "</p>"
                )
                if c.get("image_path"):
                    parts.append(f"<img src='{html.escape(str(c.get('image_path')))}' alt='{html.escape(str(c.get('case_id')))}'>")
                else:
                    parts.append("<p><i>No image generated for this case.</i></p>")
                parts.append("</div>")
        (self.output_dir / "manual_review_pack.html").write_text("\n".join(parts), encoding="utf-8")

    def write_reports(self, sampling_stats: dict[str, int], image_failures: list[dict[str, str]]) -> None:
        manifest_fields = [
            "case_id",
            "doc_id",
            "page_idx",
            "category",
            "caption_candidate_id",
            "float_candidate_id",
            "caption_text",
            "float_type",
            "current_status",
            "proposed_status",
            "risk_reason",
            "why_selected",
            "image_path",
            "pair_score",
            "pair_current_renderer_attached",
            "pair_current_gold_attached",
            "pair_official_correct",
        ]
        write_csv(self.output_dir / "manual_review_case_manifest.csv", self.cases, manifest_fields)
        write_json(self.output_dir / "manual_review_case_manifest.json", self.cases)
        if image_failures:
            write_csv(self.output_dir / "review_image_failures.csv", image_failures, ["case_id", "doc_id", "reason"])

        cat_counts = Counter(str(c.get("category", "")) for c in self.cases)
        sampling_lines = [
            "# Manual Review Sampling Report",
            "",
            f"- requested max cases: {self.max_cases}",
            f"- sampled cases: {len(self.cases)}",
            f"- images generated: {sum(1 for c in self.cases if c.get('image_path'))}",
            f"- image failures: {len(image_failures)}",
            "",
            "## Category Counts",
            "",
        ]
        for k, v in sorted(cat_counts.items()):
            sampling_lines.append(f"- {k}: {v}")
        sampling_lines.extend(["", "## Source Pool Adds", ""])
        for k, v in sorted(sampling_stats.items()):
            sampling_lines.append(f"- {k}: {v}")
        sampling_lines.extend(
            [
                "",
                "Note: raw requested category quotas exceed the 60-80 human-review budget, so the pack keeps critical pools and proportionally caps high-risk pools.",
            ]
        )
        (self.output_dir / "manual_review_sampling_report.md").write_text("\n".join(sampling_lines), encoding="utf-8")

        total_gold = 1603
        converter_clean = 8 + 27
        renderer_prop = 49
        non_iso = 9
        unknown = 28
        resolver_risky_sampled = cat_counts.get("resolver_conservative_same_page_high_risk", 0) + cat_counts.get(
            "resolver_current_materialized_high_risk", 0
        )
        estimates = [
            {
                "bucket": "converter_projection_clean_candidates",
                "case_count_known_pool": converter_clean,
                "sampled_for_review": cat_counts.get("converter_projection_clean_pool", 0)
                + cat_counts.get("text_normalization_pool", 0),
                "estimated_item_gain_upper_bound": f"{converter_clean / total_gold:.6f}",
                "risk": "low_to_medium",
            },
            {
                "bucket": "renderer_propagation_clean_candidates",
                "case_count_known_pool": renderer_prop,
                "sampled_for_review": cat_counts.get("renderer_projection_pool", 0),
                "estimated_item_gain_upper_bound": f"{renderer_prop / total_gold:.6f}",
                "risk": "medium",
            },
            {
                "bucket": "resolver_safe_candidates",
                "case_count_known_pool": 0,
                "sampled_for_review": 0,
                "estimated_item_gain_upper_bound": "0.000000",
                "risk": "not_established_automatically",
            },
            {
                "bucket": "resolver_risky_candidates",
                "case_count_known_pool": 203,
                "sampled_for_review": resolver_risky_sampled,
                "estimated_item_gain_upper_bound": f"{203 / total_gold:.6f}",
                "risk": "high_wrong_match_risk",
            },
            {
                "bucket": "non_isomorphic_candidates",
                "case_count_known_pool": non_iso,
                "sampled_for_review": cat_counts.get("subfigure_non_isomorphism_pool", 0),
                "estimated_item_gain_upper_bound": f"{non_iso / total_gold:.6f}",
                "risk": "limitation_likely",
            },
            {
                "bucket": "unknown_manual_candidates",
                "case_count_known_pool": unknown,
                "sampled_for_review": cat_counts.get("unknown_manual_pool", 0),
                "estimated_item_gain_upper_bound": f"{unknown / total_gold:.6f}",
                "risk": "unknown",
            },
        ]
        fields = ["bucket", "case_count_known_pool", "sampled_for_review", "estimated_item_gain_upper_bound", "risk"]
        write_csv(self.output_dir / "provisional_patch_value_estimate.csv", estimates, fields)
        estimate_lines = [
            "# Provisional Patch Value Estimate",
            "",
            "This estimate is automatic and must not be treated as patch approval.",
            "",
            "| bucket | known pool | sampled | item-gain upper bound | risk |",
            "|---|---:|---:|---:|---|",
        ]
        for row in estimates:
            estimate_lines.append(
                f"| {row['bucket']} | {row['case_count_known_pool']} | {row['sampled_for_review']} | {row['estimated_item_gain_upper_bound']} | {row['risk']} |"
            )
        estimate_lines.extend(
            [
                "",
                "- Clean converter/projection pool is around 35 cases, but the previous converter-only diagnostic gain was about +0.0197 document mean.",
                "- Broad resolver pools remain high-risk until manual review can separate safe anchors from visually plausible false positives.",
            ]
        )
        (self.output_dir / "provisional_patch_value_estimate.md").write_text("\n".join(estimate_lines), encoding="utf-8")

        final_decision = "manual_review_pack_ready" if sum(1 for c in self.cases if c.get("image_path")) else "manual_review_pack_ready_without_images"
        report_lines = [
            "# FLOAT_CAPTION_MANUAL_REVIEW_PACK_REPORT",
            "",
            f"Decision: `{final_decision}`",
            "",
            "## Status",
            "",
            "- No renderer patch was applied.",
            "- No evaluator or official metric definition was changed.",
            "- No fresh held-out outputs were modified.",
            "- No MinerU, Nougat, training, relabel, or GNN job was started by this pass.",
            f"- Review cases: {len(self.cases)}",
            f"- Overlay images generated: {sum(1 for c in self.cases if c.get('image_path'))}",
            f"- Image failures: {len(image_failures)}",
            "",
            "## Why No Patch Yet",
            "",
            "The automatic sidecar counterfactual found that broad resolver variants can rescue captions but also introduce many wrong attachments. The clean converter/projection-only pool is safer but below the earlier +0.04 patch threshold without human confirmation.",
            "",
            "## Categories Requiring Human Review",
            "",
        ]
        for k, v in sorted(cat_counts.items()):
            report_lines.append(f"- {k}: {v}")
        report_lines.extend(
            [
                "",
                "## Likely Patch Value",
                "",
                f"- likely converter/projection pool before review: {converter_clean} known cases",
                f"- likely renderer propagation pool before review: {renderer_prop} known cases",
                f"- likely subfigure/non-isomorphic pool before review: {non_iso} known cases",
                f"- unknown/manual pool before review: {unknown} known cases",
                "- resolver-anchor opportunity is not PRCV-worthy yet because high-risk variants generated many false attachments automatically.",
                "",
                "## Patch Trigger Thresholds",
                "",
                "- narrow converter/projection patch if manual safe converter/projection cases >= 30, wrong-risk cases <= 5, duplicate risk manageable, and official metric semantics are unchanged.",
                "- resolver patch if manual safe resolver-anchor cases >= 60, risky/wrong cases <= 15, strong type/subfigure/veto rules are identifiable, and a fresh caption-heavy validation set can be prepared.",
                "- no patch if safe cases < 30, non-isomorphic/unknown dominates, or wrong-risk remains high.",
                "- paper-only limitation if recoverable gain is < +0.02 or only converter normalization.",
                "",
                "## Outputs",
                "",
                "- manual_review_case_manifest.csv/json",
                "- manual_review_sheet.csv/md/json",
                "- manual_review_pack.html",
                "- provisional_patch_value_estimate.md/csv",
                "- review_images/ if image generation succeeded",
            ]
        )
        (self.output_dir / "FLOAT_CAPTION_MANUAL_REVIEW_PACK_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

        next_lines = [
            "# Next After Float-Caption Manual Review Pack",
            "",
            "1. Fill `manual_review_sheet.csv` or review `manual_review_pack.html` case by case.",
            "2. Count safe converter/projection, renderer propagation, resolver-anchor, non-isomorphic, and unknown labels.",
            "3. Apply the trigger thresholds in `FLOAT_CAPTION_MANUAL_REVIEW_PACK_REPORT.md`.",
            "4. If a patch is justified, run a controlled patch sprint on a new validation setup rather than patching fresh selected200 in place.",
        ]
        (self.output_dir / "next_after_float_caption_manual_review_pack.md").write_text("\n".join(next_lines), encoding="utf-8")

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stats = self.build_cases()
        failures = self.generate_images_for_cases()
        self.write_review_sheets()
        self.write_html()
        self.write_reports(stats, failures)


def validate_inputs(sidecar_root: Path) -> list[str]:
    required = [
        "FLOAT_CAPTION_SIDECAR_COUNTERFACTUAL_REPORT.md",
        "caption_candidates_sidecar.csv",
        "float_candidates_sidecar.csv",
        "caption_float_pair_features.csv",
        "float_caption_true_counterfactual_summary.csv",
        "materialized_converter_lost_121_audit.csv",
        "materialized_converter_lost_examples.md",
    ]
    return [name for name in required if not (sidecar_root / name).exists()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--fresh-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=80)
    parser.add_argument("--generate-images", action="store_true")
    args = parser.parse_args()

    missing = validate_inputs(args.sidecar_root)
    if missing:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "MISSING_FLOAT_CAPTION_MANUAL_REVIEW_INPUTS.md").write_text(
            "# Missing Float-Caption Manual Review Inputs\n\n"
            + "\n".join(f"- {name}" for name in missing)
            + "\n",
            encoding="utf-8",
        )
        return 2

    builder = ReviewBuilder(
        sidecar_root=args.sidecar_root,
        fresh_root=args.fresh_root,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        generate_images=args.generate_images,
    )
    builder.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
