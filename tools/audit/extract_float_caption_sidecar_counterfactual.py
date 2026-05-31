#!/usr/bin/env python3
"""Audit-only float-caption sidecar extraction and counterfactual simulation.

This script reads existing fresh held-out artifacts and writes aggregate
sidecars/reports. It does not mutate predictions, generated TeX, renderer
source, evaluator source, or official metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CAPTION_LABEL_RE = re.compile(
    r"^\s*(?P<label>Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?)\s+"
    r"(?P<number>(?:S?\d+(?:\.\d+)*)(?:\([a-zA-Z0-9]+\))?|[IVXLCDM]+(?:\([a-zA-Z0-9]+\))?)?"
    r"\s*(?P<sep>[:.\-–—]|\s+)?(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)
BODY_REFERENCE_RE = re.compile(
    r"^\s*(?:as\s+)?(?:shown\s+in|see|according\s+to|in|from|using|the)?\s*"
    r"(?:Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?)\s+"
    r"(?:S?\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?|[IVXLCDM]+(?:\([a-zA-Z0-9]+\))?)"
    r"\s+(?:shows?|reports?|illustrates?|depicts?|summari[sz]es?|contains?|presents?|compares?)\b",
    re.IGNORECASE,
)
LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(errors="ignore"))
    except Exception:
        return default


def norm_text(value: str | None) -> str:
    text = str(value or "").casefold()
    text = LATEX_COMMAND_RE.sub(lambda m: " " + (m.group(1) or "") + " ", text)
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return " ".join(text.split())


def loose_contains(haystack: str, needle: str) -> bool:
    h = norm_text(haystack)
    n = norm_text(needle)
    if not h or not n:
        return False
    compact_h = h.replace(" ", "")
    compact_n = n.replace(" ", "")
    if len(compact_n) < 24:
        return compact_n in compact_h
    return compact_n[:120] in compact_h or compact_n[:80] in compact_h


def safe_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "")).strip("_")


def bbox_list(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except Exception:
            return None
    if isinstance(value, dict):
        try:
            return [float(value["x0"]), float(value["y0"]), float(value["x1"]), float(value["y1"])]
        except Exception:
            return None
    return None


def first_bbox(node: dict[str, Any]) -> list[float] | None:
    boxes = node.get("bboxes")
    if isinstance(boxes, list) and boxes:
        return bbox_list(boxes[0])
    return bbox_list(node.get("bbox"))


def caption_kind_from_label(label: str) -> str:
    value = str(label or "").casefold()
    if value.startswith("tab"):
        return "table"
    if value.startswith("alg"):
        return "algorithm"
    if value.startswith("fig"):
        return "figure"
    return "unknown"


def parse_caption(text: str | None) -> dict[str, str]:
    value = " ".join(str(text or "").replace("\n", " ").split())
    if BODY_REFERENCE_RE.match(value):
        return {"type": "unknown", "number": "", "body": value, "label": "", "body_reference_guard": "true"}
    match = CAPTION_LABEL_RE.match(value)
    if not match:
        return {"type": "unknown", "number": "", "body": value, "label": "", "body_reference_guard": "false"}
    label = match.group("label") or ""
    number = match.group("number") or ""
    return {
        "type": caption_kind_from_label(label),
        "number": number,
        "body": (match.group("body") or value).strip(),
        "label": f"{label} {number}".strip(),
        "body_reference_guard": "false",
    }


def caption_type_from_node(node: dict[str, Any]) -> str:
    node_type = str(node.get("node_type") or node.get("type") or "").casefold()
    metadata = node.get("metadata") or {}
    caption_type = str(metadata.get("caption_type") or "").casefold()
    if caption_type:
        if caption_type in {"image", "chart"}:
            return "figure"
        return caption_type
    if node_type in {"figure", "image", "chart"}:
        return "figure"
    if node_type == "table":
        return "table"
    if node_type in {"algorithm", "code"}:
        return node_type
    raw = " ".join(str(metadata.get(k) or "") for k in ("raw_type", "content_list_type", "model_label", "mineru_caption_role")).casefold()
    if "table" in raw:
        return "table"
    if "image" in raw or "figure" in raw or "chart" in raw:
        return "figure"
    if "algorithm" in raw or "code" in raw:
        return "algorithm"
    return "unknown"


def compatible(caption_type: str, float_type: str) -> bool:
    c = str(caption_type or "").casefold()
    f = str(float_type or "").casefold()
    if c in {"figure", "image", "chart"} and f in {"figure", "image", "chart", "crop"}:
        return True
    if c == "table" and f == "table":
        return True
    if c in {"algorithm", "code"} and f in {"algorithm", "code"}:
        return True
    return False


def x_overlap(cbox: list[float] | None, fbox: list[float] | None) -> float:
    cbox = bbox_list(cbox)
    fbox = bbox_list(fbox)
    if not cbox or not fbox:
        return 0.0
    x0 = max(cbox[0], fbox[0])
    x1 = min(cbox[2], fbox[2])
    inter = max(0.0, x1 - x0)
    denom = max(1.0, min(cbox[2] - cbox[0], fbox[2] - fbox[0]))
    return min(1.0, inter / denom)


def vertical_distance(cbox: list[float] | None, fbox: list[float] | None) -> float | None:
    cbox = bbox_list(cbox)
    fbox = bbox_list(fbox)
    if not cbox or not fbox:
        return None
    if cbox[1] >= fbox[3]:
        return cbox[1] - fbox[3]
    if fbox[1] >= cbox[3]:
        return fbox[1] - cbox[3]
    return 0.0


def center_distance(cbox: list[float] | None, fbox: list[float] | None) -> float | None:
    cbox = bbox_list(cbox)
    fbox = bbox_list(fbox)
    if not cbox or not fbox:
        return None
    cx = (cbox[0] + cbox[2]) / 2
    cy = (cbox[1] + cbox[3]) / 2
    fx = (fbox[0] + fbox[2]) / 2
    fy = (fbox[1] + fbox[3]) / 2
    return math.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)


def score_pair(caption: dict[str, Any], flt: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    components: dict[str, Any] = {}
    score = 0.0
    if compatible(caption["caption_type"], flt["type"]):
        components["type_compatibility"] = 0.35
        score += 0.35
    else:
        components["type_compatibility"] = -1.0
        score -= 1.0
    if caption.get("caption_number") and caption.get("caption_number") == flt.get("caption_number"):
        components["label_number_match"] = 0.25
        score += 0.25
    else:
        components["label_number_match"] = 0.0
    page_delta = abs(int(caption.get("page_idx", -99)) - int(flt.get("page_idx", 99))) if caption.get("page_idx") not in ("", None) and flt.get("page_idx") not in ("", None) else 99
    if page_delta == 0:
        components["same_page_prior"] = 0.15
        score += 0.15
    elif page_delta == 1:
        components["same_page_prior"] = 0.04
        score += 0.04
    else:
        components["same_page_prior"] = -0.1
        score -= 0.1
    cbox = caption.get("bbox")
    fbox = flt.get("bbox")
    vd = vertical_distance(cbox, fbox)
    xo = x_overlap(cbox, fbox)
    if vd is not None:
        if vd <= 80:
            components["visual_proximity"] = 0.15
            score += 0.15
        elif vd <= 180:
            components["visual_proximity"] = 0.07
            score += 0.07
        else:
            components["visual_proximity"] = -0.05
            score -= 0.05
        components["horizontal_overlap"] = round(0.10 * xo, 4)
        score += 0.10 * xo
    else:
        components["visual_proximity"] = -0.05
        score -= 0.05
    ro_dist = None
    if caption.get("reading_index") not in ("", None) and flt.get("reading_index") not in ("", None):
        ro_dist = abs(int(caption["reading_index"]) - int(flt["reading_index"]))
        if ro_dist <= 2:
            components["reading_order_adjacency"] = 0.10
            score += 0.10
        elif ro_dist <= 6:
            components["reading_order_adjacency"] = 0.04
            score += 0.04
        else:
            components["reading_order_adjacency"] = 0.0
    else:
        components["reading_order_adjacency"] = 0.0
    if caption.get("subfigure_marker"):
        components["subfigure_consistency"] = 0.03
        score += 0.03
    components["duplicate_canonical_penalty"] = -0.20 if caption.get("veto_duplicate_canonical") == "true" else 0.0
    score += components["duplicate_canonical_penalty"]
    components["body_reference_penalty"] = -1.0 if caption.get("veto_body_reference") == "true" else 0.0
    score += components["body_reference_penalty"]
    return max(0.0, min(1.0, score)), components


def read_manifest(path: Path) -> list[str]:
    obj = load_json(path, [])
    rows: list[Any]
    if isinstance(obj, dict):
        rows = obj.get("docs") or obj.get("items") or obj.get("manifest") or []
    else:
        rows = obj
    docs = []
    for row in rows:
        if isinstance(row, str):
            docs.append(row)
        elif isinstance(row, dict):
            doc_id = row.get("doc_id") or row.get("arxiv_id") or row.get("id")
            if doc_id:
                docs.append(str(doc_id))
    return docs


def find_run_dir(fresh_root: Path, doc_id: str) -> Path | None:
    matches = list((fresh_root / "fresh_heldout_selected200_run").glob(f"shard_*/{doc_id}"))
    return matches[0] if matches else None


def comparison_captions(cs: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    blocks = cs.get("blocks") or []
    by_id = {b.get("block_id"): b for b in blocks}
    parent_kind = {}
    caps = []
    for b in blocks:
        if b.get("block_type") == "caption":
            parent = by_id.get(b.get("parent_id")) or {}
            kind = b.get("metadata", {}).get("caption_parent_kind") or parent.get("block_type") or "unknown"
            parent_kind[b.get("block_id")] = kind
            caps.append(
                {
                    "block_id": b.get("block_id"),
                    "text": b.get("text") or "",
                    "normalized_text": b.get("normalized_text") or norm_text(b.get("text")),
                    "parent_id": b.get("parent_id"),
                    "parent_kind": kind,
                    "order": b.get("order"),
                }
            )
    return caps, parent_kind


def match_caption_to_gold(candidate_text: str, gold_caps: list[dict[str, Any]]) -> dict[str, Any] | None:
    cn = norm_text(candidate_text)
    if not cn:
        return None
    best = None
    best_score = 0.0
    for cap in gold_caps:
        gn = norm_text(cap.get("text") or cap.get("normalized_text"))
        if not gn:
            continue
        compact_c = cn.replace(" ", "")
        compact_g = gn.replace(" ", "")
        if compact_c == compact_g:
            score = 1.0
        elif compact_c and compact_g and (compact_c[:80] in compact_g or compact_g[:80] in compact_c):
            score = min(len(compact_c), len(compact_g)) / max(len(compact_c), len(compact_g))
        else:
            cset = set(cn.split())
            gset = set(gn.split())
            score = len(cset & gset) / max(1, len(cset | gset))
        if score > best_score:
            best_score = score
            best = cap
    if best is not None and best_score >= 0.58:
        result = dict(best)
        result["match_score"] = best_score
        return result
    return None


def build_rows(args: argparse.Namespace) -> dict[str, Any]:
    fresh_root = Path(args.fresh_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = fresh_root / "fresh_heldout_selected200_manifest.json"
    doc_ids = read_manifest(manifest_path)
    if not doc_ids:
        doc_ids = sorted(p.name for p in (fresh_root / "fresh_heldout_artifacts").glob("*") if p.is_dir())[:200]

    attr_rows = []
    attr_path = Path(args.attribution_root) / "float_caption_attachment_attribution.csv"
    if attr_path.exists():
        with attr_path.open(newline="") as f:
            attr_rows = [r for r in csv.DictReader(f) if r.get("method") == "framework"]
    attr_by_doc_gold = {(r["doc_id"], r["gold_caption_id"]): r for r in attr_rows}
    attr_by_doc = defaultdict(list)
    for r in attr_rows:
        attr_by_doc[r["doc_id"]].append(r)

    inventory = []
    caption_rows = []
    float_rows = []
    pair_rows = []
    materialized_lost_rows = []
    per_doc_metrics: dict[str, dict[str, Any]] = {}

    for doc_id in doc_ids:
        run_dir = find_run_dir(fresh_root, doc_id)
        artifacts_dir = fresh_root / "fresh_heldout_artifacts" / doc_id
        paths = {
            "gold_path": artifacts_dir / "gold_comparison_structure.json",
            "pred_path": (run_dir / "05_comparison/comparison_structure.json") if run_dir else Path(""),
            "metrics_path": (run_dir / "05_comparison/metrics.json") if run_dir else Path(""),
            "document_ir_path": (run_dir / "02_ir/document_ir.json") if run_dir else Path(""),
            "render_tree_ir_path": (run_dir / "02_ir/render_tree_ir.json") if run_dir else Path(""),
            "v8_fact_path": (run_dir / "01_facts/observable_facts.json") if run_dir else Path(""),
            "caption_sidecar_path": (run_dir / "03_generation/generation_report.json") if run_dir else Path(""),
            "float_candidate_path": (run_dir / "02_ir/document_ir.json") if run_dir else Path(""),
            "generated_structure_path": (run_dir / "03_generation/generated.tex") if run_dir else Path(""),
        }
        missing = [k for k, p in paths.items() if not p or not p.exists()]
        inventory.append(
            {
                "doc_id": doc_id,
                **{k: str(v) if v else "" for k, v in paths.items()},
                "all_required_for_sidecar": str(not missing).lower(),
                "missing_fields": ";".join(missing),
            }
        )
        if missing:
            continue

        document_ir = load_json(paths["document_ir_path"], {})
        render_tree = load_json(paths["render_tree_ir_path"], {})
        generation = load_json(paths["caption_sidecar_path"], {})
        gold = load_json(paths["gold_path"], {})
        pred = load_json(paths["pred_path"], {})
        metrics = load_json(paths["metrics_path"], {})
        tex = paths["generated_structure_path"].read_text(errors="ignore") if paths["generated_structure_path"].exists() else ""
        gold_caps, _gold_parent = comparison_captions(gold)
        pred_caps, _pred_parent = comparison_captions(pred)
        pred_caption_text = "\n".join(c.get("text", "") for c in pred_caps)
        fc_metric = metrics.get("float_caption_attachment_accuracy") or {}
        per_doc_metrics[doc_id] = {
            "gold": int(fc_metric.get("gold_captions") or 0),
            "correct": int(fc_metric.get("correct") or 0),
            "matched": int(fc_metric.get("matched") or 0),
            "score": float(fc_metric.get("score") or 0.0),
        }

        nodes = document_ir.get("nodes") or []
        nodes_by_id = {n.get("node_id"): n for n in nodes}
        render_nodes = render_tree.get("nodes") or []
        render_by_source = defaultdict(list)
        for rn in render_nodes:
            for sid in rn.get("source_node_ids") or []:
                render_by_source[sid].append(rn)

        # Float candidates from DocumentIR.
        for node in nodes:
            ftype = caption_type_from_node(node)
            if ftype not in {"figure", "table", "algorithm", "code"}:
                continue
            md = node.get("metadata") or {}
            source_ids = [node.get("node_id")]
            render_nodes_for_source = render_by_source.get(node.get("node_id"), [])
            render_role = ";".join(str(rn.get("role") or "") for rn in render_nodes_for_source)
            caption_text = md.get("caption_text") or md.get("figure_caption") or md.get("table_caption") or md.get("crop_caption") or ""
            parsed = parse_caption(caption_text)
            asset = md.get("img_path") or md.get("crop_asset_path") or md.get("asset_path") or ""
            float_rows.append(
                {
                    "doc_id": doc_id,
                    "float_candidate_id": f"float_{safe_id(node.get('node_id'))}",
                    "source_layer": "DocumentIR",
                    "source_ids": json.dumps(source_ids, ensure_ascii=False),
                    "page_idx": node.get("page_idx"),
                    "bbox": json.dumps(first_bbox(node)),
                    "bbox_obj": first_bbox(node),
                    "type": ftype,
                    "visual_asset_path_exists": str(bool(asset and Path(str(asset)).exists())).lower(),
                    "crop_asset_path": asset,
                    "rendered_in_latex": str(any(rn.get("role") in {"figure", "table", "algorithm"} for rn in render_nodes_for_source)).lower(),
                    "render_role": render_role,
                    "has_caption_child": str(bool(caption_text)).lower(),
                    "caption_text_if_any": caption_text,
                    "caption_number": parsed.get("number", ""),
                    "table_fallback_type": "crop" if ftype == "table" and asset else ("tabular" if md.get("table_body") else "none"),
                    "figure_fallback_type": "crop" if ftype == "figure" and asset else "none",
                    "comparison_structure_float_id": "",
                    "gold_match_status": "infer_by_caption_text" if match_caption_to_gold(caption_text, gold_caps) else "unknown",
                    "reading_index": node.get("reading_index"),
                }
            )

        gen_fc = generation.get("float_caption_materialization") or {}
        promoted = {item.get("caption_id"): item for item in gen_fc.get("promoted_captions") or []}
        consumed = {item.get("caption_id"): item for item in gen_fc.get("consumed_caption_paragraphs") or []}
        suppressed = {item.get("caption_id"): item for item in (gen_fc.get("noncanonical_suppressed_candidates") or [])}
        pairings = gen_fc.get("float_caption_pairings") or []
        seen_candidates = set()

        def add_candidate(caption: dict[str, Any], pair: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> None:
            cid = caption.get("caption_id") or f"cap_unknown_{len(caption_rows)}"
            if (doc_id, cid) in seen_candidates:
                return
            seen_candidates.add((doc_id, cid))
            source_ids = caption.get("source_v8_ids") or []
            text = caption.get("text") or ""
            parsed = parse_caption(text)
            origin = caption.get("origin") or "unknown"
            evidence = caption.get("evidence") or {}
            mineru_backed = origin != "text_block" or bool(evidence.get("metadata_key")) or "metadata" in origin
            regex_only = origin == "text_block" and not mineru_backed
            pred_match = match_caption_to_gold(text, pred_caps)
            gold_match = match_caption_to_gold(text, gold_caps)
            attr = attr_by_doc_gold.get((doc_id, gold_match["block_id"])) if gold_match else None
            category = attr.get("category") if attr else ""
            materialized = cid in promoted
            row = {
                "doc_id": doc_id,
                "caption_candidate_id": cid,
                "source_layer": "generation_report.float_caption_materialization",
                "source_ids": json.dumps(source_ids, ensure_ascii=False),
                "page_idx": caption.get("page_idx"),
                "bbox": json.dumps(caption.get("bbox")),
                "bbox_obj": bbox_list(caption.get("bbox")),
                "text": text,
                "normalized_text": caption.get("normalized_caption_text") or norm_text(text),
                "canonical_label": parsed.get("label", ""),
                "caption_number": caption.get("caption_number") or parsed.get("number", ""),
                "subfigure_marker": (re.search(r"\(([a-zA-Z0-9]+)\)", str(caption.get("caption_number") or "")) or [None, ""])[1],
                "caption_type": caption.get("caption_type") or parsed.get("type", "unknown"),
                "mineru_backed": str(mineru_backed).lower(),
                "regex_only": str(regex_only).lower(),
                "diagnostic_only": str(regex_only or parsed.get("body_reference_guard") == "true").lower(),
                "rendered_in_latex": str(loose_contains(tex, text)).lower(),
                "materialized_by_renderer": str(materialized).lower(),
                "consumed_paragraph": str(cid in consumed).lower(),
                "converter_detected_caption": str(bool(pred_match) or loose_contains(pred_caption_text, text)).lower(),
                "gold_match_status": gold_match.get("block_id") if gold_match else "",
                "gold_parent_kind": gold_match.get("parent_kind") if gold_match else "",
                "official_match_status": "correct" if category == "caption_attached_correctly" else ("gap" if category else "unknown"),
                "current_gap_category": category,
                "paired_float_id": pair.get("paired_float_id") if pair else "",
                "paired_float_type": pair.get("paired_float_type") if pair else "",
                "pairing_confidence": pair.get("pairing_confidence") if pair else "",
                "veto_body_reference": str(parsed.get("body_reference_guard") == "true").lower(),
                "veto_regex_only": str(regex_only).lower(),
                "veto_duplicate_canonical": str(cid in suppressed).lower(),
                "veto_low_confidence": str(float(caption.get("confidence") or 0.0) < 0.78).lower(),
                "reading_index": nodes_by_id.get(source_ids[0], {}).get("reading_index") if source_ids else "",
            }
            if extra:
                row.update(extra)
            caption_rows.append(row)

        for pair in pairings:
            add_candidate(pair.get("caption") or {}, pair)
        for item in gen_fc.get("noncanonical_suppressed_candidates") or []:
            add_candidate(item, None, {"suppression_reason": item.get("reason", "")})

        # Additional DocumentIR metadata/text candidates missed by generation sidecar.
        for node in nodes:
            md = node.get("metadata") or {}
            texts = []
            for key in ("caption_text", "figure_caption", "table_caption", "crop_caption", "detected_caption"):
                val = md.get(key)
                if isinstance(val, str) and val.strip():
                    texts.append((key, val))
            if node.get("text") and parse_caption(node.get("text")).get("label"):
                texts.append(("text_regex", node.get("text")))
            for key, text in texts:
                cid = f"docir_{safe_id(node.get('node_id'))}_{key}"
                if (doc_id, cid) in seen_candidates:
                    continue
                parsed = parse_caption(text)
                gold_match = match_caption_to_gold(text, gold_caps)
                attr = attr_by_doc_gold.get((doc_id, gold_match["block_id"])) if gold_match else None
                regex_only = key == "text_regex"
                caption_rows.append(
                    {
                        "doc_id": doc_id,
                        "caption_candidate_id": cid,
                        "source_layer": "DocumentIR",
                        "source_ids": json.dumps([node.get("node_id")], ensure_ascii=False),
                        "page_idx": node.get("page_idx"),
                        "bbox": json.dumps(first_bbox(node)),
                        "bbox_obj": first_bbox(node),
                        "text": text,
                        "normalized_text": norm_text(text),
                        "canonical_label": parsed.get("label", ""),
                        "caption_number": parsed.get("number", ""),
                        "subfigure_marker": (re.search(r"\(([a-zA-Z0-9]+)\)", str(parsed.get("number") or "")) or [None, ""])[1],
                        "caption_type": parsed.get("type") if parsed.get("type") != "unknown" else caption_type_from_node(node),
                        "mineru_backed": str(not regex_only).lower(),
                        "regex_only": str(regex_only).lower(),
                        "diagnostic_only": str(regex_only or parsed.get("body_reference_guard") == "true").lower(),
                        "rendered_in_latex": str(loose_contains(tex, text)).lower(),
                        "materialized_by_renderer": "false",
                        "consumed_paragraph": "false",
                        "converter_detected_caption": str(loose_contains(pred_caption_text, text)).lower(),
                        "gold_match_status": gold_match.get("block_id") if gold_match else "",
                        "gold_parent_kind": gold_match.get("parent_kind") if gold_match else "",
                        "official_match_status": "correct" if attr and attr.get("category") == "caption_attached_correctly" else ("gap" if attr else "unknown"),
                        "current_gap_category": attr.get("category") if attr else "",
                        "paired_float_id": "",
                        "paired_float_type": "",
                        "pairing_confidence": "",
                        "veto_body_reference": str(parsed.get("body_reference_guard") == "true").lower(),
                        "veto_regex_only": str(regex_only).lower(),
                        "veto_duplicate_canonical": "false",
                        "veto_low_confidence": "false",
                        "reading_index": node.get("reading_index"),
                    }
                )

        # Pair features for this doc.
        doc_caps = [r for r in caption_rows if r["doc_id"] == doc_id]
        doc_floats = [r for r in float_rows if r["doc_id"] == doc_id]
        for cap in doc_caps:
            for flt in doc_floats:
                page_delta = 99
                if cap.get("page_idx") not in ("", None) and flt.get("page_idx") not in ("", None):
                    page_delta = abs(int(cap["page_idx"]) - int(flt["page_idx"]))
                current_pair = cap.get("paired_float_id") and cap.get("paired_float_id") == flt.get("float_candidate_id")
                if page_delta > 1 and not current_pair:
                    continue
                if not compatible(cap.get("caption_type"), flt.get("type")) and not current_pair:
                    continue
                score, comps = score_pair(cap, flt)
                cbox = cap.get("bbox_obj")
                fbox = flt.get("bbox_obj")
                current_gold = cap.get("gold_match_status") and compatible(cap.get("gold_parent_kind"), flt.get("type"))
                pair_rows.append(
                    {
                        "doc_id": doc_id,
                        "caption_candidate_id": cap["caption_candidate_id"],
                        "float_candidate_id": flt["float_candidate_id"],
                        "caption_page": cap.get("page_idx"),
                        "float_page": flt.get("page_idx"),
                        "page_delta": page_delta,
                        "caption_bbox": json.dumps(cbox),
                        "float_bbox": json.dumps(fbox),
                        "vertical_distance": vertical_distance(cbox, fbox),
                        "horizontal_overlap": round(x_overlap(cbox, fbox), 4),
                        "center_distance": center_distance(cbox, fbox),
                        "reading_order_distance": abs(int(cap["reading_index"]) - int(flt["reading_index"])) if cap.get("reading_index") not in ("", None) and flt.get("reading_index") not in ("", None) else "",
                        "caption_before_after_float": "after" if cap.get("reading_index") not in ("", None) and flt.get("reading_index") not in ("", None) and int(cap["reading_index"]) > int(flt["reading_index"]) else "before_or_same",
                        "same_column_or_region": "unknown",
                        "type_compatibility": str(compatible(cap.get("caption_type"), flt.get("type"))).lower(),
                        "label_type_compatibility": str(compatible(cap.get("caption_type"), flt.get("type"))).lower(),
                        "label_number_match_if_available": str(bool(cap.get("caption_number") and cap.get("caption_number") == flt.get("caption_number"))).lower(),
                        "subfigure_consistency": "true" if cap.get("subfigure_marker") else "not_applicable",
                        "current_renderer_attached": str(bool(current_pair)).lower(),
                        "current_converter_attached": cap.get("converter_detected_caption"),
                        "current_gold_attached": str(bool(current_gold)).lower(),
                        "official_correct": str(cap.get("official_match_status") == "correct").lower(),
                        "veto_body_reference": cap.get("veto_body_reference"),
                        "veto_regex_only": cap.get("veto_regex_only"),
                        "veto_type_mismatch": str(not compatible(cap.get("caption_type"), flt.get("type"))).lower(),
                        "veto_duplicate_canonical": cap.get("veto_duplicate_canonical"),
                        "veto_no_visual_anchor": str(not bool(cbox and fbox)).lower(),
                        "veto_low_confidence": cap.get("veto_low_confidence"),
                        "proposed_score_components": json.dumps(comps, sort_keys=True),
                        "proposed_total_score": round(score, 4),
                        "caption_gold_match_status": cap.get("gold_match_status"),
                        "caption_gap_category": cap.get("current_gap_category"),
                    }
                )

        # 121 materialized/converter-lost audit rows.
        for attr in attr_by_doc.get(doc_id, []):
            if attr.get("category") != "caption_materialized_but_converter_lost":
                continue
            gold_id = attr.get("gold_caption_id")
            gold_text = attr.get("gold_text_preview") or ""
            candidates = [c for c in doc_caps if c.get("gold_match_status") == gold_id or loose_contains(c.get("text"), gold_text)]
            pred_matches = [p for p in pred_caps if loose_contains(p.get("text"), gold_text) or loose_contains(gold_text, p.get("text"))]
            generated_contains = loose_contains(tex, gold_text)
            materialized = any(c.get("materialized_by_renderer") == "true" for c in candidates)
            rendered = any(c.get("rendered_in_latex") == "true" for c in candidates) or generated_contains
            paired_wrong_type = any(c.get("paired_float_type") and c.get("gold_parent_kind") and not compatible(c.get("gold_parent_kind"), c.get("paired_float_type")) for c in candidates)
            subfig = bool(re.search(r"\([a-zA-Z0-9]\)", gold_text))
            classification = "unknown_needs_manual"
            if materialized and not rendered:
                classification = "renderer_materialized_but_not_in_generated_tex"
            elif generated_contains and not pred_matches:
                classification = "generated_tex_contains_caption_but_converter_missed"
            elif pred_matches:
                pred_kind = pred_matches[0].get("parent_kind") or "unknown"
                gold_kind = next((g.get("parent_kind") for g in gold_caps if g.get("block_id") == gold_id), "unknown")
                if pred_kind == "unknown":
                    classification = "converter_detected_caption_but_no_float_anchor"
                elif gold_kind != "unknown" and pred_kind != gold_kind:
                    classification = "converter_detected_caption_but_wrong_type"
                else:
                    classification = "comparison_matcher_failed_text_normalization"
            elif paired_wrong_type:
                classification = "current_pair_wrong_anchor"
            elif subfig:
                classification = "subfigure_boundary_mismatch"
            elif candidates and not materialized:
                classification = "renderer_materialized_but_not_in_generated_tex"
            materialized_lost_rows.append(
                {
                    "doc_id": doc_id,
                    "gold_caption_id": gold_id,
                    "gold_text_preview": gold_text,
                    "candidate_count": len(candidates),
                    "candidate_ids": ";".join(c.get("caption_candidate_id", "") for c in candidates[:5]),
                    "generated_tex_contains_caption": str(generated_contains).lower(),
                    "renderer_candidate_materialized": str(materialized).lower(),
                    "candidate_rendered_in_latex": str(rendered).lower(),
                    "converter_caption_match_count": len(pred_matches),
                    "predicted_parent_kinds": ";".join(sorted({str(p.get("parent_kind")) for p in pred_matches})),
                    "paired_float_types": ";".join(sorted({str(c.get("paired_float_type")) for c in candidates if c.get("paired_float_type")})),
                    "classification": classification,
                }
            )

    return {
        "doc_ids": doc_ids,
        "inventory": inventory,
        "caption_rows": caption_rows,
        "float_rows": float_rows,
        "pair_rows": pair_rows,
        "materialized_lost_rows": materialized_lost_rows,
        "per_doc_metrics": per_doc_metrics,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], drop_keys: set[str] | None = None) -> None:
    drop_keys = drop_keys or set()
    public_rows = [{k: v for k, v in row.items() if k not in drop_keys} for row in rows]
    keys: list[str] = []
    for row in public_rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(public_rows)


def run_counterfactual(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_rows = data["pair_rows"]
    captions_by_id = {(r["doc_id"], r["caption_candidate_id"]): r for r in data["caption_rows"]}
    current_correct = sum(m["correct"] for m in data["per_doc_metrics"].values())
    current_gold = sum(m["gold"] for m in data["per_doc_metrics"].values())
    current_all_doc_mean = sum(m["score"] for m in data["per_doc_metrics"].values()) / max(1, len(data["per_doc_metrics"]))
    caption_docs = [m for m in data["per_doc_metrics"].values() if m["gold"] > 0]
    current_caption_doc_mean = sum(m["score"] for m in caption_docs) / max(1, len(caption_docs))

    variants = {
        "conservative_same_page": lambda p, c: p["page_delta"] == 0 and c["mineru_backed"] == "true" and c["diagnostic_only"] != "true" and p["veto_type_mismatch"] != "true" and p["veto_no_visual_anchor"] != "true" and float(p["proposed_total_score"]) >= 0.55,
        "conservative_same_or_adjacent_page": lambda p, c: p["page_delta"] in (0, 1) and c["mineru_backed"] == "true" and c["diagnostic_only"] != "true" and p["veto_type_mismatch"] != "true" and p["veto_no_visual_anchor"] != "true" and float(p["proposed_total_score"]) >= 0.50,
        "current_materialized_only": lambda p, c: p["current_renderer_attached"] == "true" and c["materialized_by_renderer"] == "true",
        "type_mismatch_fix_only": lambda p, c: c["current_gap_category"] == "caption_type_mismatch" and c["mineru_backed"] == "true" and p["veto_type_mismatch"] != "true",
        "converter_projection_recovery_only": lambda p, c: c["current_gap_category"] == "caption_materialized_but_converter_lost" and (c["materialized_by_renderer"] == "true" or c["rendered_in_latex"] == "true"),
        "risky_upper_bound": lambda p, c: c["current_gap_category"] != "caption_attached_correctly" and p["veto_body_reference"] != "true" and p["veto_type_mismatch"] != "true" and float(p["proposed_total_score"]) >= 0.35,
    }

    rows = []
    details = {}
    for name, predicate in variants.items():
        selected_by_doc_float: dict[tuple[str, str], dict[str, Any]] = {}
        for p in sorted(pair_rows, key=lambda r: float(r["proposed_total_score"]), reverse=True):
            c = captions_by_id.get((p["doc_id"], p["caption_candidate_id"]))
            if not c or not predicate(p, c):
                continue
            cap_key = (p["doc_id"], p["caption_candidate_id"])
            flt_key = (p["doc_id"], p["float_candidate_id"])
            if cap_key in {(v["doc_id"], v["caption_candidate_id"]) for v in selected_by_doc_float.values()}:
                continue
            if flt_key in selected_by_doc_float:
                continue
            selected_by_doc_float[flt_key] = p
        rescued = set()
        wrong = 0
        duplicate_risk = 0
        ambiguous = 0
        for p in selected_by_doc_float.values():
            c = captions_by_id[(p["doc_id"], p["caption_candidate_id"])]
            if c["official_match_status"] == "correct":
                continue
            if c.get("gold_match_status") and p["current_gold_attached"] == "true":
                rescued.add((p["doc_id"], c["gold_match_status"]))
            elif c.get("gold_match_status"):
                ambiguous += 1
            else:
                wrong += 1
            if c.get("veto_duplicate_canonical") == "true":
                duplicate_risk += 1
        rescued_by_doc = Counter(doc for doc, _ in rescued)
        estimated_doc_scores = []
        for doc, m in data["per_doc_metrics"].items():
            gold = m["gold"]
            if gold <= 0:
                continue
            correct = min(gold, m["correct"] + rescued_by_doc.get(doc, 0))
            estimated_doc_scores.append(correct / gold)
        estimated_mean = sum(estimated_doc_scores) / max(1, len(estimated_doc_scores))
        rows.append(
            {
                "variant": name,
                "current_correct_count": current_correct,
                "current_gold_count": current_gold,
                "selected_pair_count": len(selected_by_doc_float),
                "potential_rescued_correct_items": len(rescued),
                "potential_new_wrong_matches": wrong,
                "potential_duplicate_risk": duplicate_risk,
                "ambiguous_cases": ambiguous,
                "current_all_doc_mean_float_caption_attachment": round(current_all_doc_mean, 6),
                "current_caption_doc_mean_float_caption_attachment": round(current_caption_doc_mean, 6),
                "estimated_doc_mean_float_caption_attachment": round(estimated_mean, 6),
                "item_level_gain": round(len(rescued) / max(1, current_gold), 6),
                "doc_mean_gain": round(estimated_mean - current_caption_doc_mean, 6),
                "risk_category": "low" if wrong == 0 and duplicate_risk == 0 and ambiguous < 20 else ("medium" if wrong < 20 else "high"),
            }
        )
        details[name] = {
            "rescued": sorted([{"doc_id": d, "gold_caption_id": g} for d, g in rescued], key=lambda x: (x["doc_id"], x["gold_caption_id"]))[:100],
            "wrong": wrong,
            "ambiguous": ambiguous,
        }
    return rows, details


def write_reports(args: argparse.Namespace, data: dict[str, Any]) -> None:
    out = Path(args.output_dir)
    inv = data["inventory"]
    caps = data["caption_rows"]
    floats = data["float_rows"]
    pairs = data["pair_rows"]
    lost = data["materialized_lost_rows"]
    counter_rows, counter_details = run_counterfactual(data)

    write_csv(out / "float_caption_sidecar_inventory.csv", inv)
    write_csv(out / "caption_candidates_sidecar.csv", caps, drop_keys={"bbox_obj"})
    write_csv(out / "float_candidates_sidecar.csv", floats, drop_keys={"bbox_obj"})
    write_csv(out / "caption_float_pair_features.csv", pairs)
    write_csv(out / "materialized_converter_lost_121_audit.csv", lost)
    write_csv(out / "float_caption_true_counterfactual_summary.csv", counter_rows)

    (out / "float_caption_sidecar_inventory.json").write_text(json.dumps(inv, indent=2, ensure_ascii=False))
    (out / "caption_candidates_sidecar.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "bbox_obj"} for r in caps], indent=2, ensure_ascii=False))
    (out / "float_candidates_sidecar.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "bbox_obj"} for r in floats], indent=2, ensure_ascii=False))
    (out / "caption_float_pair_features.json").write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
    (out / "float_caption_true_counterfactual_summary.json").write_text(json.dumps({"summary": counter_rows, "examples": counter_details}, indent=2, ensure_ascii=False))

    inv_ready = sum(1 for r in inv if r["all_required_for_sidecar"] == "true")
    lost_counts = Counter(r["classification"] for r in lost)
    cap_counts = Counter(r.get("current_gap_category") or "no_gold_gap_category" for r in caps)
    pair_veto_counts = Counter()
    for p in pairs:
        for key in ("veto_body_reference", "veto_regex_only", "veto_type_mismatch", "veto_duplicate_canonical", "veto_no_visual_anchor", "veto_low_confidence"):
            if p.get(key) == "true":
                pair_veto_counts[key] += 1
    best_safe = next((r for r in counter_rows if r["variant"] == "conservative_same_page"), {})
    best_adj = next((r for r in counter_rows if r["variant"] == "conservative_same_or_adjacent_page"), {})
    converter = next((r for r in counter_rows if r["variant"] == "converter_projection_recovery_only"), {})
    safe_gain = max(float(best_safe.get("doc_mean_gain") or 0), float(best_adj.get("doc_mean_gain") or 0))
    converter_gain = float(converter.get("doc_mean_gain") or 0)
    risky = next((r for r in counter_rows if r["variant"] == "risky_upper_bound"), {})
    risky_gain = float(risky.get("doc_mean_gain") or 0)
    if inv_ready < 180:
        decision = "sidecar_still_blocked"
    elif safe_gain >= 0.04 and best_adj.get("risk_category") == "low":
        decision = "resolver_patch_recommended"
    elif converter_gain >= 0.04 or lost_counts.get("generated_tex_contains_caption_but_converter_missed", 0) + lost_counts.get("comparison_matcher_failed_text_normalization", 0) > len(lost) / 2:
        decision = "converter_projection_patch_recommended"
    elif risky_gain >= 0.04:
        decision = "needs_manual_visual_review"
    else:
        decision = "resolver_patch_not_worth_it"

    inventory_md = f"""# Float-caption Sidecar Inventory

- docs in manifest: {len(inv)}
- all_required_for_sidecar: {inv_ready}/{len(inv)}
- caption candidate rows: {len(caps)}
- float candidate rows: {len(floats)}
- pair feature rows: {len(pairs)}

Gate: {'pass' if inv_ready >= 180 else 'blocked'}.
"""
    (out / "float_caption_sidecar_inventory_report.md").write_text(inventory_md)

    counter_table = "\n".join(
        "| {variant} | {potential_rescued_correct_items} | {potential_new_wrong_matches} | {estimated_doc_mean_float_caption_attachment} | {doc_mean_gain} | {risk_category} |".format(**r)
        for r in counter_rows
    )
    counter_md = f"""# Float-caption True Counterfactual Summary

This is a diagnostic simulation over existing sidecars. It does not modify predictions or official metrics.

| Variant | Rescued | New wrong | Estimated doc-mean score | Doc-mean gain | Risk |
|---|---:|---:|---:|---:|---|
{counter_table}

Notes:
- Conservative variants require MinerU-backed, non-diagnostic captions and compatible existing visual anchors.
- `converter_projection_recovery_only` isolates captions already materialized/rendered but lost during conversion/matching.
- `risky_upper_bound` is not a production policy.
"""
    (out / "float_caption_true_counterfactual_summary.md").write_text(counter_md)

    lost_table = "\n".join(f"| {k} | {v} |" for k, v in lost_counts.most_common())
    (out / "materialized_converter_lost_121_audit.md").write_text(
        f"""# Materialized-but-converter-lost Audit

Rows audited: {len(lost)}

| Classification | Count |
|---|---:|
{lost_table}

Estimated recoverable_by_converter_fix: {lost_counts.get('generated_tex_contains_caption_but_converter_missed', 0) + lost_counts.get('comparison_matcher_failed_text_normalization', 0)}

Estimated recoverable_by_resolver_fix: {lost_counts.get('current_pair_wrong_anchor', 0) + lost_counts.get('converter_detected_caption_but_wrong_type', 0)}

Estimated nonrecoverable_due_to_non_isomorphism: {lost_counts.get('source_pdf_float_non_isomorphism', 0)}

Estimated risky_cases: {lost_counts.get('unknown_needs_manual', 0) + lost_counts.get('subfigure_boundary_mismatch', 0)}
"""
    )
    examples = ["# Materialized Converter Lost Examples\n"]
    for row in lost[:20]:
        examples.append(
            f"- {row['doc_id']} {row['gold_caption_id']}: {row['classification']}; "
            f"generated_tex_contains={row['generated_tex_contains_caption']}; "
            f"converter_matches={row['converter_caption_match_count']}; "
            f"text=`{row['gold_text_preview'][:220]}`"
        )
    (out / "materialized_converter_lost_examples.md").write_text("\n".join(examples) + "\n")

    main_report = f"""# FLOAT CAPTION SIDECAR COUNTERFACTUAL REPORT

## Status

- Sidecar extraction: completed.
- Existing per-doc artifacts modified: no.
- Renderer/evaluator/official metrics modified: no.
- Fresh held-out outputs modified: no.
- MinerU/training/relabel/GNN: no.
- Source TeX inference: no.

## Coverage

- Docs inventoried: {len(inv)}
- All required sidecar inputs: {inv_ready}/{len(inv)}
- Caption candidates: {len(caps)}
- Float candidates: {len(floats)}
- Candidate pairs: {len(pairs)}

## What Causes the Float-caption Weakness?

Current sidecars show a mixed picture:
- Missing/materialization evidence remains substantial in the original attribution.
- The 121 `materialized_but_converter_lost` rows classify as: {dict(lost_counts)}.
- Pair-level veto pressure is: {dict(pair_veto_counts)}.

## Is There a PRCV-worthy Anchor Resolver Opportunity?

The resolver is technically well-formed, but the true counterfactual must be judged by the conservative variants:

| Variant | Rescued | New wrong | Estimated score | Gain | Risk |
|---|---:|---:|---:|---:|---|
{counter_table}

If conservative gain is below +0.04, the anchor resolver should not be promoted as the next patch. If converter/projection recovery dominates the 121-row pool, the next technical sprint should target representation/projection rather than anchor matching.

## Decision

**{decision}**

## Required Validation if Any Patch Is Attempted

- Implement behind an explicit experimental flag.
- No regex-only production promotion.
- No broad placeholder creation.
- No Algorithm broad promotion.
- Rerun fresh selected200 no-patch-equivalent controlled evaluation with identical official metrics.
- Compare Framework V1 old vs patched, and keep Direct/Nougat claims unchanged unless rerun under the same denominator.
"""
    (out / "FLOAT_CAPTION_SIDECAR_COUNTERFACTUAL_REPORT.md").write_text(main_report)
    (out / "next_after_float_caption_sidecar_counterfactual.md").write_text(
        f"""# Next After Float-caption Sidecar Counterfactual

Decision: `{decision}`.

Recommended next action:
- If `resolver_patch_recommended`: implement the constrained matcher behind an experimental flag and rerun selected200.
- If `converter_projection_patch_recommended`: inspect converter/projection path for generated captions that are lost before ComparisonStructureV1, without changing official metrics.
- If `resolver_patch_not_worth_it`: keep the resolver as a paper design/limitation note and focus on a stronger contribution candidate.
- If `needs_manual_visual_review`: sample the ambiguous examples before patching.
- If `sidecar_still_blocked`: repair artifact availability first.
"""
    )
    (out / "sidecar_counterfactual_manifest.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "inventory_ready": inv_ready,
                "doc_count": len(inv),
                "caption_candidate_count": len(caps),
                "float_candidate_count": len(floats),
                "pair_feature_count": len(pairs),
                "lost_classification_counts": dict(lost_counts),
                "counterfactual": counter_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-root", required=True)
    parser.add_argument("--attribution-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = build_rows(args)
    write_reports(args, data)


if __name__ == "__main__":
    main()
