"""Weak-supervision labels for the v8 atomic merge JSON route.

The label sidecar is intentionally separate from atomic node/edge JSON.  It may
use deterministic v8 merge evidence and optional TeX alignment, but none of that
truth information is written back into graph-input nodes or candidate edges.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any


MERGE = "MERGE"
NONE = "NONE"
UNKNOWN = "UNKNOWN"

TRAINABLE_MERGE_FAMILIES = {
    "BODY_TEXT_CONTINUATION",
    "LIST_CONTINUATION",
    "REFERENCE_CONTINUATION",
    "FLOAT_SKIP_CONTINUATION",
}
MASKED_FAMILIES = {"FORMULA_CONTEXT", "MASKED_UNKNOWN"}
HARD_NEGATIVE_FAMILIES = {"LAYOUT_SCOPE_MISMATCH"}


@dataclass
class TexParagraph:
    tex_id: str
    index: int
    text: str
    tokens: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "tex_id": self.tex_id,
            "index": self.index,
            "text_preview": self.text[:240],
            "token_count": len(self.tokens),
        }


def build_v8_atomic_merge_labels(
    atomic_payload: dict[str, Any],
    *,
    source_tex_path: Path | None = None,
    min_tex_alignment_confidence: float = 0.55,
) -> dict[str, Any]:
    doc_id = str(atomic_payload.get("doc_id") or "document")
    nodes = atomic_payload.get("nodes") or []
    edges = atomic_payload.get("candidate_edges") or []
    node_by_id = {str(node.get("atomic_id")): node for node in nodes if isinstance(node, dict)}
    deterministic_pairs = deterministic_merge_atomic_pairs(atomic_payload, node_by_id)
    same_block_pairs = same_middle_block_atomic_pairs(nodes)

    tex_paragraphs: list[TexParagraph] = []
    alignments: dict[str, dict[str, Any]] = {}
    if source_tex_path is not None:
        tex_paragraphs = extract_tex_paragraphs(source_tex_path)
        alignments = align_nodes_to_tex(nodes, tex_paragraphs)

    labels: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        labels.append(
            label_edge(
                edge,
                node_by_id=node_by_id,
                deterministic_pairs=deterministic_pairs,
                same_block_pairs=same_block_pairs,
                tex_alignments=alignments,
                min_tex_alignment_confidence=min_tex_alignment_confidence,
            )
        )

    return {
        "schema_version": "v8_atomic_merge_labels_v1",
        "doc_id": doc_id,
        "source": {
            "atomic_graph_view_schema": atomic_payload.get("schema_version"),
            "source_tex": str(source_tex_path) if source_tex_path is not None else None,
        },
        "policy": {
            "no_truth_in_graph_inputs": True,
            "labels_are_sidecar_only": True,
            "min_tex_alignment_confidence": min_tex_alignment_confidence,
            "parent_label_policy": "not_generated_in_v8_atomic_merge_route",
        },
        "edge_labels": labels,
        "tex_paragraphs": [paragraph.to_json() for paragraph in tex_paragraphs],
        "node_tex_alignments": list(alignments.values()),
        "summary": summarize_labels(labels, alignments),
    }


def label_edge(
    edge: dict[str, Any],
    *,
    node_by_id: dict[str, dict[str, Any]],
    deterministic_pairs: set[tuple[str, str]],
    same_block_pairs: set[tuple[str, str]],
    tex_alignments: dict[str, dict[str, Any]],
    min_tex_alignment_confidence: float,
) -> dict[str, Any]:
    src_id = str(edge.get("src_atomic_id"))
    dst_id = str(edge.get("dst_atomic_id"))
    pair = (src_id, dst_id)
    family = str(edge.get("candidate_family") or "MASKED_UNKNOWN")
    src = node_by_id.get(src_id, {})
    dst = node_by_id.get(dst_id, {})

    label = UNKNOWN
    train_mask = False
    strength = "masked"
    weight = 0.0
    source = "unlabeled"
    confidence = 0.0
    reasons: list[str] = []

    if family in HARD_NEGATIVE_FAMILIES:
        label, train_mask, strength, weight = NONE, True, "hard_negative", 1.0
        source, confidence = "geometry_hard_negative", 0.90
        reasons.append(family)
    elif pair in deterministic_pairs:
        label, train_mask, strength, weight = MERGE, True, "strong", 1.0
        source, confidence = "frontend_v8_deterministic_merge", 0.90
    elif pair in same_block_pairs and family in TRAINABLE_MERGE_FAMILIES:
        label, train_mask, strength, weight = MERGE, True, "strong", 1.0
        source, confidence = "frontend_same_middle_block_lines", 0.88
    elif family in MASKED_FAMILIES:
        label, train_mask, strength, weight = UNKNOWN, False, "masked", 0.0
        source, confidence = "masked_family", 0.0
        reasons.append(family)
    else:
        src_align = tex_alignments.get(src_id)
        dst_align = tex_alignments.get(dst_id)
        if src_align and dst_align:
            src_conf = float(src_align.get("confidence") or 0.0)
            dst_conf = float(dst_align.get("confidence") or 0.0)
            same_tex = src_align.get("tex_id") == dst_align.get("tex_id")
            strong_enough = min(src_conf, dst_conf) >= min_tex_alignment_confidence
            if same_tex and strong_enough and family in TRAINABLE_MERGE_FAMILIES:
                label, train_mask = MERGE, True
                strength = "strong" if backend_edge_has_precision_continuation_evidence(edge) else "weak"
                weight = 1.0 if strength == "strong" else 0.2
                source, confidence = "backend_tex_same_paragraph", min(src_conf, dst_conf)
            elif (not same_tex) and strong_enough:
                label, train_mask, strength, weight = NONE, True, "hard_negative", 1.0
                source, confidence = "backend_tex_different_paragraph", min(src_conf, dst_conf)
        if label == UNKNOWN:
            label, train_mask, strength, weight = UNKNOWN, False, "masked", 0.0
            source, confidence = "insufficient_truth", 0.0

    return {
        "edge_id": edge.get("edge_id"),
        "src_atomic_id": src_id,
        "dst_atomic_id": dst_id,
        "src_channel": edge.get("src_channel"),
        "dst_channel": edge.get("dst_channel"),
        "candidate_family": family,
        "label": label,
        "train_mask": train_mask,
        "label_strength": strength,
        "proposed_loss_weight": weight,
        "label_source": source,
        "confidence": round(confidence, 4),
        "reasons": reasons,
        "source_middle_block_pair": [
            src.get("source_middle_block_id"),
            dst.get("source_middle_block_id"),
        ],
        "tex_alignment": {
            "src": tex_alignments.get(src_id),
            "dst": tex_alignments.get(dst_id),
        },
    }


def backend_edge_has_precision_continuation_evidence(edge: dict[str, Any]) -> bool:
    family = str(edge.get("candidate_family") or "")
    if family not in {"BODY_TEXT_CONTINUATION", "LIST_CONTINUATION"}:
        return False
    features = edge.get("features") if isinstance(edge.get("features"), dict) else {}
    if int(features.get("skipped_barrier_count") or 0) > 0:
        return False
    if not bool(features.get("src_open_ended")):
        return False
    return bool(
        features.get("src_hyphen_ended")
        or features.get("dst_lowercase_start")
        or features.get("dst_continuation_word_start")
        or int(edge.get("reading_order_gap") or 0) == 1
    )


def deterministic_merge_atomic_pairs(
    atomic_payload: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
) -> set[tuple[str, str]]:
    by_block: dict[str, list[dict[str, Any]]] = {}
    for node in node_by_id.values():
        by_block.setdefault(str(node.get("source_middle_block_id")), []).append(node)
    for members in by_block.values():
        members.sort(key=lambda item: int(item.get("reading_order") or 0))

    pairs: set[tuple[str, str]] = set()
    for decision in atomic_payload.get("deterministic_merge_sidecar") or []:
        if not isinstance(decision, dict):
            continue
        src_members = by_block.get(str(decision.get("src_block_id"))) or []
        dst_members = by_block.get(str(decision.get("dst_block_id"))) or []
        src_text = [node for node in src_members if node.get("channel") in TRAINABLE_CHANNELS]
        dst_text = [node for node in dst_members if node.get("channel") in TRAINABLE_CHANNELS]
        if src_text and dst_text:
            pairs.add((str(src_text[-1].get("atomic_id")), str(dst_text[0].get("atomic_id"))))
    return pairs


TRAINABLE_CHANNELS = {"BODY_TEXT", "LIST_ITEM", "REFERENCE_ITEM"}


def same_middle_block_atomic_pairs(nodes: list[Any]) -> set[tuple[str, str]]:
    by_block: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("channel") not in TRAINABLE_CHANNELS:
            continue
        by_block.setdefault(str(node.get("source_middle_block_id")), []).append(node)
    pairs: set[tuple[str, str]] = set()
    for members in by_block.values():
        members.sort(key=lambda item: int(item.get("reading_order") or 0))
        for src, dst in zip(members, members[1:], strict=False):
            pairs.add((str(src.get("atomic_id")), str(dst.get("atomic_id"))))
    return pairs


def extract_tex_paragraphs(source_tex_path: Path) -> list[TexParagraph]:
    text = source_tex_path.read_text(encoding="utf-8", errors="ignore")
    body = strip_tex_comments(text)
    body = drop_tex_environments(
        body,
        (
            "figure",
            "figure*",
            "table",
            "table*",
            "algorithm",
            "algorithm*",
            "equation",
            "equation*",
            "align",
            "align*",
            "tabular",
            "tabular*",
        ),
    )
    body = re.sub(
        r"\\(?:section|subsection|subsubsection|paragraph|subparagraph)\*?(?:\[[^\]]*\])?\{([^{}]*)\}",
        lambda match: f"\n\nHEADING {match.group(1)}\n\n",
        body,
    )
    body = re.sub(r"\\caption(?:\[[^\]]*\])?\{[^{}]*\}", "\n\n", body)
    body = re.sub(r"\\(?:cite|ref|label|url|href)\*?(?:\[[^\]]*\])?\{[^{}]*\}", " ", body)
    body = re.sub(r"\\item(?:\[[^\]]*\])?", "\n\n", body)
    paragraphs: list[TexParagraph] = []
    for raw in re.split(r"\n\s*\n+", body):
        clean = clean_tex_text(raw)
        tokens = tokenize(clean)
        if len(tokens) < 4:
            continue
        paragraphs.append(TexParagraph(tex_id=f"tex_p{len(paragraphs):05d}", index=len(paragraphs), text=clean, tokens=tokens))
    return paragraphs


def align_nodes_to_tex(nodes: list[Any], paragraphs: list[TexParagraph]) -> dict[str, dict[str, Any]]:
    alignments: dict[str, dict[str, Any]] = {}
    if not paragraphs:
        return alignments
    paragraph_token_sets = [(paragraph, set(paragraph.tokens)) for paragraph in paragraphs]
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("channel") not in TRAINABLE_CHANNELS:
            continue
        tokens = tokenize(str(node.get("text") or ""))
        if len(tokens) < 2:
            continue
        token_set = set(tokens)
        best: TexParagraph | None = None
        best_score = 0.0
        best_common = 0
        for paragraph, paragraph_tokens in paragraph_token_sets:
            common = len(token_set & paragraph_tokens)
            if common == 0:
                continue
            score = common / max(1, min(len(token_set), 12))
            if score > best_score:
                best = paragraph
                best_score = score
                best_common = common
        if best is not None:
            alignments[str(node.get("atomic_id"))] = {
                "atomic_id": node.get("atomic_id"),
                "tex_id": best.tex_id,
                "tex_index": best.index,
                "confidence": round(min(1.0, best_score), 4),
                "common_token_count": best_common,
                "node_token_count": len(tokens),
            }
    return alignments


def summarize_labels(labels: list[dict[str, Any]], alignments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "edge_label_count": len(labels),
        "trainable_edge_count": sum(1 for label in labels if label.get("train_mask")),
        "merge_positive_count": sum(1 for label in labels if label.get("label") == MERGE),
        "none_negative_count": sum(1 for label in labels if label.get("label") == NONE),
        "masked_unknown_count": sum(1 for label in labels if label.get("label") == UNKNOWN),
        "node_tex_alignment_count": len(alignments),
        "label_source_counts": count_values(label.get("label_source") for label in labels),
        "candidate_family_counts": count_values(label.get("candidate_family") for label in labels),
        "label_strength_counts": count_values(label.get("label_strength") for label in labels),
    }


def strip_tex_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        lines.append(line[: match.start()] if match else line)
    return "\n".join(lines)


def drop_tex_environments(text: str, envs: tuple[str, ...]) -> str:
    result = text
    for env in envs:
        escaped = re.escape(env)
        result = re.sub(rf"\\begin\{{{escaped}\}}.*?\\end\{{{escaped}\}}", "\n\n", result, flags=re.DOTALL)
    return result


def clean_tex_text(text: str) -> str:
    text = re.sub(r"\\(?:textbf|textit|emph|underline)\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r" \1 ", text)
    text = re.sub(r"[$^_{}&#]", " ", text)
    text = text.replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", text)
    return re.findall(r"[a-z0-9][a-z0-9-]{1,}", text)


def count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
