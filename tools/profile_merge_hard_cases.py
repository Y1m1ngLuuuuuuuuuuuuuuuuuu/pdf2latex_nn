#!/usr/bin/env python3
"""Profile MERGE false positives from an edge-relation checkpoint.

The model-level MERGE precision failure mode is not visible from aggregate F1.
This utility runs a trained checkpoint over a document-level split and writes:

* an aggregate JSON report grouped by visual/layout features;
* a JSONL file containing the highest-confidence MERGE false positives.

It intentionally does not mutate graphs, labels, manifests, or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.step5_generate_tex import checkpoint_compatible_config  # noqa: E402
from scripts.pipeline.train_edge_gnn_full import split_indices  # noqa: E402
from src.reasoning.gnn_model import EdgeGATConfig, EdgeRelationGAT  # noqa: E402


LABEL_NAMES = {0: "merge", 1: "parent_child", 2: "none"}
LIST_MARKER_RE = re.compile(r"^\s*(?:[\u2022\-\*\u25cb\u25aa]|\d+\.|[a-zA-Z]\.)\s+")
TERMINAL_RE = re.compile(r"[.!?。！？]\s*$")
HYPHEN_RE = re.compile(r"(?:-|‐|‑|‒|–|—)\s*$")
UPPERCASE_RE = re.compile(r"^\s*[A-Z]")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument(
        "--prediction-mode",
        choices=["argmax", "threshold"],
        default="argmax",
        help="argmax profiles raw model mistakes; threshold profiles deployment-like decisions.",
    )
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--parent-threshold", type=float, default=0.5)
    return parser


def main() -> int:
    import torch

    args = build_arg_parser().parse_args()
    device = resolve_device(args.device, torch=torch)
    docs = load_manifest_documents(args.manifest)
    selected = select_split_documents(docs, args)
    if args.max_docs > 0:
        selected = selected[: args.max_docs]
    if not selected:
        raise ValueError(f"No documents selected from {args.manifest}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    model_config = checkpoint_compatible_config(config if isinstance(config, EdgeGATConfig) else EdgeGATConfig(), state_dict)
    model = EdgeRelationGAT(model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    aggregate = Aggregate()
    top_cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for doc_pos, doc in enumerate(selected, start=1):
            graph_path = Path(doc["graph_path"])
            data = torch.load(graph_path, map_location="cpu", weights_only=False)
            logits = model(data.to(device)).detach().cpu()
            probs = torch.softmax(logits, dim=-1)
            y = torch.where(data.y.detach().cpu().long() >= 2, torch.full_like(data.y.detach().cpu().long(), 2), data.y.detach().cpu().long())
            pred = predict_edges(probs, args, torch=torch)
            aggregate.add_document(doc, data, y, pred)
            false_positions = torch.nonzero((pred == 0) & (y != 0), as_tuple=False).flatten().tolist()
            for edge_pos in false_positions:
                case = build_case(doc, data, probs, y, pred, edge_pos)
                aggregate.add_false_positive_case(case)
                top_cases.append(case)
            if doc_pos == 1 or doc_pos % 25 == 0:
                print(
                    f"[profile-merge] docs={doc_pos}/{len(selected)} "
                    f"fp_merge={aggregate.false_positive_count} "
                    f"tp_merge={aggregate.confusion[(0, 0)]}",
                    flush=True,
                )

    top_cases.sort(key=lambda item: (float(item["p_merge"]), float(item["margin_merge_none"])), reverse=True)
    top_cases = top_cases[: max(0, args.top_k)]
    payload = aggregate.to_json(args=args, selected_docs=len(selected), top_cases=top_cases[:20])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for case in top_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_jsonl}")
    print_summary(payload)
    return 0


def predict_edges(prob: Any, args: argparse.Namespace, *, torch: Any) -> Any:
    if args.prediction_mode == "argmax":
        return prob.argmax(dim=-1)
    pred = torch.full((int(prob.shape[0]),), 2, dtype=torch.long)
    merge_mask = prob[:, 0] >= float(args.merge_threshold)
    pred[merge_mask] = 0
    parent_mask = (~merge_mask) & (prob[:, 1] >= float(args.parent_threshold))
    pred[parent_mask] = 1
    return pred


def load_manifest_documents(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = payload.get("documents", payload if isinstance(payload, list) else [])
    if not isinstance(docs, list):
        raise ValueError(f"Malformed manifest: {path}")
    return [doc for doc in docs if isinstance(doc, dict) and doc.get("graph_path")]


def select_split_documents(docs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.split == "all":
        return docs
    splits = split_indices(len(docs), args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)
    return [docs[index] for index in splits[args.split]]


def resolve_device(value: str, *, torch: Any) -> Any:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Aggregate:
    def __init__(self) -> None:
        self.documents = 0
        self.edges = 0
        self.confusion: Counter[tuple[int, int]] = Counter()
        self.false_positive_count = 0
        self.false_positive_groups: dict[str, Counter[str]] = defaultdict(Counter)
        self.true_positive_groups: dict[str, Counter[str]] = defaultdict(Counter)
        self.by_doc: Counter[str] = Counter()

    def add_document(self, doc: dict[str, Any], data: Any, target: Any, pred: Any) -> None:
        self.documents += 1
        self.edges += int(target.numel())
        for p, y in zip(pred.tolist(), target.tolist()):
            self.confusion[(int(p), int(y))] += 1

    def add_false_positive_case(self, case: dict[str, Any]) -> None:
        self.false_positive_count += 1
        self.by_doc[str(case["document_id"])] += 1
        for group_name, group_value in case["groups"].items():
            self.false_positive_groups[group_name][str(group_value)] += 1

    def add_true_positive_case(self, case: dict[str, Any]) -> None:
        for group_name, group_value in case["groups"].items():
            self.true_positive_groups[group_name][str(group_value)] += 1

    def to_json(self, *, args: argparse.Namespace, selected_docs: int, top_cases: list[dict[str, Any]]) -> dict[str, Any]:
        per_class = metrics_from_confusion(self.confusion)
        return {
            "schema_version": "merge_hard_case_profile_v1",
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "selected_documents": selected_docs,
            "prediction_mode": args.prediction_mode,
            "thresholds": {
                "merge": args.merge_threshold,
                "parent": args.parent_threshold,
            },
            "edges": self.edges,
            "confusion": confusion_to_json(self.confusion),
            "per_class": per_class,
            "merge_false_positive_count": self.false_positive_count,
            "merge_false_positive_groups": {
                key: counter.most_common(50) for key, counter in sorted(self.false_positive_groups.items())
            },
            "merge_false_positive_docs": self.by_doc.most_common(50),
            "top_cases_preview": top_cases,
        }


def metrics_from_confusion(confusion: Counter[tuple[int, int]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for cls in (0, 1, 2):
        tp = confusion[(cls, cls)]
        fp = sum(count for (pred, target), count in confusion.items() if pred == cls and target != cls)
        fn = sum(count for (pred, target), count in confusion.items() if pred != cls and target == cls)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        support = sum(count for (_pred, target), count in confusion.items() if target == cls)
        output[LABEL_NAMES[cls]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return output


def confusion_to_json(confusion: Counter[tuple[int, int]]) -> dict[str, int]:
    return {f"pred_{LABEL_NAMES[pred]}__true_{LABEL_NAMES[target]}": count for (pred, target), count in sorted(confusion.items())}


def build_case(doc: dict[str, Any], data: Any, probs: Any, target: Any, pred: Any, edge_pos: int) -> dict[str, Any]:
    source = int(data.edge_index[0, edge_pos].item())
    dest = int(data.edge_index[1, edge_pos].item())
    records = list(getattr(data, "node_records", []))
    source_record = dict(records[source]) if 0 <= source < len(records) and isinstance(records[source], dict) else {}
    target_record = dict(records[dest]) if 0 <= dest < len(records) and isinstance(records[dest], dict) else {}
    edge_fields = edge_attr_fields(data)
    edge_values = {}
    if getattr(data, "edge_attr", None) is not None:
        for name, index in edge_fields.items():
            if index < int(data.edge_attr.shape[1]):
                edge_values[name] = round(float(data.edge_attr[edge_pos, index].item()), 6)
    p = probs[edge_pos].tolist()
    groups = classify_case(source, dest, source_record, target_record, edge_values, data, edge_pos)
    return {
        "document_id": str(doc.get("document_id") or ""),
        "graph_path": str(doc.get("graph_path") or ""),
        "content_json": str(doc.get("content_json") or ""),
        "tex_path": str(doc.get("tex_path") or ""),
        "edge_pos": edge_pos,
        "source": source,
        "target": dest,
        "true_label": int(target[edge_pos].item()),
        "pred_label": int(pred[edge_pos].item()),
        "p_merge": round(float(p[0]), 8),
        "p_parent": round(float(p[1]), 8),
        "p_none": round(float(p[2]), 8),
        "margin_merge_none": round(float(p[0] - p[2]), 8),
        "edge_source_type": edge_source_type(data, edge_pos),
        "groups": groups,
        "edge_attr": edge_values,
        "source_record": summarize_record(source_record),
        "target_record": summarize_record(target_record),
    }


def classify_case(
    source: int,
    target: int,
    source_record: dict[str, Any],
    target_record: dict[str, Any],
    edge_values: dict[str, float],
    data: Any,
    edge_pos: int,
) -> dict[str, str]:
    source_type = record_type(source_record)
    target_type = record_type(target_record)
    source_text = record_text(source_record)
    target_text = record_text(target_record)
    source_layer = str(source_record.get("layout_layer") or "unknown")
    target_layer = str(target_record.get("layout_layer") or "unknown")
    source_band = source_record.get("layout_band_global_id", source_record.get("layout_band_id"))
    target_band = target_record.get("layout_band_global_id", target_record.get("layout_band_id"))
    source_page = source_record.get("page_idx")
    target_page = target_record.get("page_idx")
    page_delta = numeric(target_page) - numeric(source_page)
    index_delta = target - source
    edge_type = edge_source_type(data, edge_pos)
    main_reason = first_true_reason(
        [
            ("reverse_edge", index_delta <= 0),
            ("target_list_marker", bool(LIST_MARKER_RE.match(target_text))),
            ("type_mismatch", source_type != target_type),
            ("structural_type", source_type in STRUCTURAL_TYPES or target_type in STRUCTURAL_TYPES),
            ("layout_layer_mismatch", source_layer != target_layer),
            ("layout_band_mismatch", source_band is not None and target_band is not None and source_band != target_band),
            ("gutter_overlap", edge_values.get("has_x_gutter", 0.0) >= 0.5 and edge_values.get("y_overlap_ratio", 0.0) > 0.3),
            ("far_index_non_hyphen", edge_values.get("index_delta_bin_far", 0.0) >= 0.5 and not bool(HYPHEN_RE.search(source_text))),
            ("source_terminal_target_upper", bool(TERMINAL_RE.search(source_text)) and bool(UPPERCASE_RE.match(target_text))),
            ("cross_page", page_delta != 0),
            ("sequential_adjacent", edge_values.get("index_delta_bin_adjacent", 0.0) >= 0.5 or abs(index_delta) == 1),
            ("other", True),
        ]
    )
    return {
        "main_reason": main_reason,
        "type_pair": f"{source_type}->{target_type}",
        "layout_layer_pair": f"{source_layer}->{target_layer}",
        "same_band": str(source_band == target_band),
        "same_page": str(source_page == target_page),
        "page_delta": str(int(page_delta)),
        "index_delta_bin": index_delta_bin(index_delta),
        "edge_source_type": edge_type,
        "source_terminal": str(bool(TERMINAL_RE.search(source_text))),
        "source_hyphen": str(bool(HYPHEN_RE.search(source_text))),
        "target_uppercase": str(bool(UPPERCASE_RE.match(target_text))),
    }


STRUCTURAL_TYPES = {
    "title",
    "equation",
    "inline_math",
    "table",
    "figure",
    "algorithm",
    "code",
    "toc",
    "list",
}


def first_true_reason(pairs: list[tuple[str, bool]]) -> str:
    for name, ok in pairs:
        if ok:
            return name
    return "other"


def index_delta_bin(delta: int) -> str:
    if delta <= 0:
        return "reverse"
    if delta == 1:
        return "adjacent"
    if delta == 2:
        return "skip_one"
    if delta <= 5:
        return "near"
    return "far"


def record_type(record: dict[str, Any]) -> str:
    raw = str(
        record.get("canonical_type")
        or record.get("type")
        or record.get("raw_type")
        or record.get("block_type")
        or ""
    ).casefold()
    if raw in {"paragraph", "paragraph_text", "body"}:
        return "text"
    if raw in {"section", "subsection", "subsubsection", "heading"}:
        return "title"
    if raw in {"display_formula", "formula", "interline_equation", "equation_interline"}:
        return "equation"
    if raw in {"inline_formula", "math_inline"}:
        return "inline_math"
    if raw in {"item", "itemize", "enumerate"}:
        return "list"
    return raw or "unknown"


def record_text(record: dict[str, Any]) -> str:
    for key in ("text_for_embedding", "merged_text", "text", "text_preview"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    text = record_text(record)
    return {
        "type": record_type(record),
        "layout_layer": record.get("layout_layer"),
        "layout_band_id": record.get("layout_band_global_id", record.get("layout_band_id")),
        "layout_band_column": record.get("layout_band_column"),
        "page_idx": record.get("page_idx"),
        "global_order": record.get("global_order"),
        "regime_reading_order": record.get("regime_reading_order"),
        "bbox": record.get("bbox"),
        "text": preview(text, 220),
    }


def edge_attr_fields(data: Any) -> dict[str, int]:
    schema = getattr(data, "edge_attr_schema", None)
    if isinstance(schema, dict) and isinstance(schema.get("fields"), list):
        return {str(name): idx for idx, name in enumerate(schema["fields"])}
    return {}


def edge_source_type(data: Any, edge_pos: int) -> str:
    values = getattr(data, "edge_source_types", None)
    if isinstance(values, list) and 0 <= edge_pos < len(values):
        return str(values[edge_pos])
    return "unknown"


def numeric(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def preview(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def print_summary(payload: dict[str, Any]) -> None:
    print("merge hard-case profile")
    print(f"documents={payload['selected_documents']} edges={payload['edges']}")
    print(f"mode={payload['prediction_mode']} fp_merge={payload['merge_false_positive_count']}")
    print("per_class=", json.dumps(payload["per_class"], ensure_ascii=False))
    for group_name, rows in payload["merge_false_positive_groups"].items():
        print(f"{group_name}: {rows[:8]}")


if __name__ == "__main__":
    raise SystemExit(main())
