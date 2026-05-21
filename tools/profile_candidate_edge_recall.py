#!/usr/bin/env python3
"""Profile whether graph candidate edges cover oracle positive relations.

This is the pre-training safety probe for the Dual-View sampler.  It builds
the same PDF-to-TeX alignment state as ``label_generator.py``, then asks:

    of all positive MERGE/PARENT_CHILD pairs that the current truth generator
    can infer, how many are present in graph.edge_index?

If recall is below the requested threshold, the graph should be rebuilt with a
larger window or more recall patches before training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.label_generator import AlignmentLabeler, AlignmentLabelerConfig, build_visual_hierarchy  # noqa: E402
from src.reasoning.tex_relation_labeler import TexRelationLabel  # noqa: E402


LABEL_NAMES = {
    int(TexRelationLabel.MERGE): "MERGE",
    int(TexRelationLabel.PARENT_CHILD): "PARENT_CHILD",
    int(TexRelationLabel.NONE): "NONE",
}


@dataclass(frozen=True)
class OracleEdge:
    source: int
    target: int
    label: int
    reason: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.source, self.target)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-json", type=Path, required=True, help="*_content_list_v7_styles.json path")
    parser.add_argument("--tex", type=Path, required=True, help="Flattenable TeX entrypoint, usually main.tex")
    parser.add_argument("--graph", type=Path, required=True, help="Graph .pt built from the same content JSON")
    parser.add_argument("--similarity-threshold", type=float, default=65.0)
    parser.add_argument(
        "--merge-label-policy",
        choices=("strict", "skip_over_continuation"),
        default="strict",
        help="MERGE supervision policy used when building oracle positive edges.",
    )
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--output-json", type=Path, help="Optional JSON report path")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=1.0,
        help="Exit non-zero if overall recall is below this value. Default demands 100%% recall.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    graph, labeler = prepare_labeler(
        args.content_json,
        args.tex,
        args.graph,
        args.similarity_threshold,
        merge_label_policy=args.merge_label_policy,
    )
    report = profile_candidate_recall(graph, labeler, max_examples=args.max_examples)
    print_report(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.output_json}")
    recall = float(report["overall"]["recall"])
    if args.fail_under is not None and recall < args.fail_under:
        return 2
    return 0


def prepare_labeler(
    content_json_path: Path,
    tex_path: Path,
    graph_path: Path,
    similarity_threshold: float,
    *,
    merge_label_policy: str = "strict",
) -> tuple[Any, AlignmentLabeler]:
    config = AlignmentLabelerConfig(
        similarity_threshold=similarity_threshold,
        merge_label_policy=merge_label_policy,
        abort_on_bad_alignment=False,
    )
    labeler = AlignmentLabeler(
        content_json_path=content_json_path,
        tex_path=tex_path,
        graph_path=graph_path,
        config=config,
    )
    graph = labeler.load_graph()
    labeler.pdf_nodes = labeler.parse_pdf_nodes(
        expected_node_count=int(graph.num_nodes),
        force_micro_fusion=bool(getattr(graph, "micro_fusion_applied", False)),
    )
    if len(labeler.pdf_nodes) != int(graph.num_nodes):
        raise ValueError(
            f"content node count ({len(labeler.pdf_nodes)}) does not match graph.num_nodes ({int(graph.num_nodes)})"
        )
    labeler.tex_nodes = {node.tex_id: node for node in labeler.parse_tex_nodes()}
    labeler.matches = labeler.align_pdf_to_tex()
    labeler.visual_hierarchy = build_visual_hierarchy(labeler.pdf_nodes, config=labeler.config)
    return graph, labeler


def profile_candidate_recall(graph: Any, labeler: AlignmentLabeler, *, max_examples: int = 30) -> dict[str, Any]:
    candidates = candidate_pair_set(graph)
    oracle_edges = build_oracle_positive_edges(labeler)
    by_label: dict[str, dict[str, Any]] = {}
    missing_examples: list[dict[str, Any]] = []

    for label in (int(TexRelationLabel.MERGE), int(TexRelationLabel.PARENT_CHILD)):
        edges = [edge for edge in oracle_edges if edge.label == label]
        matched = [edge for edge in edges if edge.key in candidates]
        missing = [edge for edge in edges if edge.key not in candidates]
        by_label[LABEL_NAMES[label]] = {
            "oracle_edges": len(edges),
            "matched_edges": len(matched),
            "missing_edges": len(missing),
            "recall": safe_ratio(len(matched), len(edges)),
        }
        for edge in missing[:max(0, max_examples - len(missing_examples))]:
            missing_examples.append(edge_payload(edge, labeler))

    matched_total = sum(1 for edge in oracle_edges if edge.key in candidates)
    report = {
        "schema_version": "candidate_edge_recall_v1",
        "content_json_path": str(labeler.content_json_path),
        "tex_path": str(labeler.tex_path),
        "graph_path": str(labeler.graph_path),
        "num_nodes": len(labeler.pdf_nodes),
        "candidate_edges": len(candidates),
        "candidate_edge_sources": edge_source_counts(graph),
        "overall": {
            "oracle_edges": len(oracle_edges),
            "matched_edges": matched_total,
            "missing_edges": len(oracle_edges) - matched_total,
            "recall": safe_ratio(matched_total, len(oracle_edges)),
        },
        "by_label": by_label,
        "missing_examples": missing_examples,
        "alignment_quality": {
            "matched_pdf_nodes": sum(1 for match in labeler.matches if match.tex_id),
            "unmatched_pdf_nodes": sum(1 for match in labeler.matches if not match.tex_id),
        },
    }
    return report


def candidate_pair_set(graph: Any) -> set[tuple[int, int]]:
    edge_index = graph.edge_index.detach().cpu()
    return {(int(edge_index[0, pos].item()), int(edge_index[1, pos].item())) for pos in range(edge_index.shape[1])}


def build_oracle_positive_edges(labeler: AlignmentLabeler) -> list[OracleEdge]:
    """Return critical positive edges independent of graph.edge_index.

    MERGE edges are consecutive pieces inside the same TeX node; requiring all
    pairwise combinations would overstate the graph's needs.  Parent-child
    edges are evaluated over all ordered PDF-node anchors because those are the
    logical edges most likely to be dropped by too-small windows.
    """

    oracle: dict[tuple[int, int, int], OracleEdge] = {}

    for tex_id, pdf_indices in labeler.tex_to_pdf_indices.items():
        ordered = sorted(set(pdf_indices))
        for source, target in zip(ordered, ordered[1:]):
            relation = labeler.infer_relation(source, target)
            if relation == int(TexRelationLabel.MERGE):
                oracle[(source, target, relation)] = OracleEdge(
                    source=source,
                    target=target,
                    label=relation,
                    reason=f"same_tex_consecutive:{tex_id}",
                )

    node_count = len(labeler.pdf_nodes)
    for source in range(node_count):
        for target in range(node_count):
            if source == target:
                continue
            relation = labeler.infer_relation(source, target)
            if relation == int(TexRelationLabel.PARENT_CHILD):
                oracle[(source, target, relation)] = OracleEdge(
                    source=source,
                    target=target,
                    label=relation,
                    reason="inferred_parent_child",
                )

    return sorted(oracle.values(), key=lambda edge: (edge.label, edge.source, edge.target, edge.reason))


def edge_payload(edge: OracleEdge, labeler: AlignmentLabeler) -> dict[str, Any]:
    source_node = labeler.pdf_nodes[edge.source]
    target_node = labeler.pdf_nodes[edge.target]
    return {
        **asdict(edge),
        "label_name": LABEL_NAMES.get(edge.label, str(edge.label)),
        "source": node_payload(source_node),
        "target": node_payload(target_node),
        "index_delta": edge.target - edge.source,
    }


def node_payload(node: Any) -> dict[str, Any]:
    bbox = node.item.get("bbox") if isinstance(node.item, dict) else None
    return {
        "index": node.node_index,
        "type": node.item.get("type") if isinstance(node.item, dict) else None,
        "page_idx": node.item.get("page_idx") if isinstance(node.item, dict) else None,
        "bbox": bbox,
        "text_preview": " ".join(str(node.text or "").split())[:160],
    }


def edge_source_counts(graph: Any) -> dict[str, int]:
    source_types = getattr(graph, "edge_source_types", None)
    if not isinstance(source_types, list):
        return {}
    return dict(sorted(Counter(str(value) for value in source_types).items()))


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def print_report(report: dict[str, Any]) -> None:
    overall = report["overall"]
    print(
        "Candidate Edge Recall: "
        f"{overall['recall']:.2%} "
        f"({overall['matched_edges']}/{overall['oracle_edges']} positive oracle edges)"
    )
    for label_name, stats in report["by_label"].items():
        print(
            f"  {label_name}: {stats['recall']:.2%} "
            f"({stats['matched_edges']}/{stats['oracle_edges']}), "
            f"missing={stats['missing_edges']}"
        )
    if report["missing_examples"]:
        print("Missing examples:")
        for example in report["missing_examples"]:
            source = example["source"]
            target = example["target"]
            print(
                f"  {example['label_name']} {example['source']['index']} -> {example['target']['index']} "
                f"(delta={example['index_delta']}, reason={example['reason']})"
            )
            print(f"    source[{source['type']} p{source['page_idx']}]: {source['text_preview']}")
            print(f"    target[{target['type']} p{target['page_idx']}]: {target['text_preview']}")


if __name__ == "__main__":
    raise SystemExit(main())
