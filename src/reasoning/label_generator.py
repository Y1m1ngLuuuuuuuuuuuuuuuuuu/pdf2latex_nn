"""Generate supervised edge labels from PDF-to-TeX alignments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.reasoning.tex_ast_builder import build_tex_ast_from_file, tex_nodes_by_id
from src.reasoning.tex_relation_labeler import TexRelationLabel, label_tex_relation


@dataclass(frozen=True)
class LabelGeneratorConfig:
    similarity_threshold: float = 0.55
    adjacent_siblings_only: bool = True
    directed_parent_child: bool = False
    orphan_label: int = int(TexRelationLabel.NONE)


@dataclass(frozen=True)
class OrphanAlignment:
    node_index: int
    node_key: str
    reason: str
    score: float | None = None


@dataclass(frozen=True)
class LabelGenerationResult:
    data: Any
    label_counts: dict[int, int]
    orphan_alignments: list[OrphanAlignment]


def load_pdf_to_tex_mapping(path: Path) -> dict[str, Any]:
    """Load a PDF-block to TeX-node alignment mapping from JSON or JSONL."""

    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if path.suffix.lower() == ".jsonl":
        mapping: dict[str, Any] = {}
        for line in text.splitlines():
            record = json.loads(line)
            pdf_id = record.get("pdf_id") or record.get("block_id") or record.get("source")
            tex_id = record.get("tex_id") or record.get("target")
            if pdf_id and tex_id:
                mapping[str(pdf_id)] = {"tex_id": str(tex_id), "score": record.get("score")}
        return mapping
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("alignments"), list):
        mapping: dict[str, Any] = {}
        for record in data["alignments"]:
            pdf_id = record.get("pdf_id") or record.get("block_id") or record.get("source")
            tex_id = record.get("tex_id") or record.get("target")
            if pdf_id and tex_id:
                mapping[str(pdf_id)] = {"tex_id": str(tex_id), "score": record.get("score")}
        return mapping
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain an object mapping or an alignments list")
    return data


def label_graph_edges_from_paths(
    data: Any,
    *,
    tex_path: Path,
    pdf_to_tex_path: Path,
    config: LabelGeneratorConfig | None = None,
    orphan_log_path: Path | None = None,
) -> LabelGenerationResult:
    tex_ast = build_tex_ast_from_file(tex_path)
    pdf_to_tex = load_pdf_to_tex_mapping(pdf_to_tex_path)
    return label_graph_edges(
        data,
        tex_ast=tex_ast,
        pdf_to_tex=pdf_to_tex,
        config=config,
        orphan_log_path=orphan_log_path,
    )


def label_graph_edges(
    data: Any,
    *,
    tex_ast: dict[str, Any] | list[dict[str, Any]] | dict[str, dict[str, Any]],
    pdf_to_tex: dict[str, Any],
    config: LabelGeneratorConfig | None = None,
    orphan_log_path: Path | None = None,
) -> LabelGenerationResult:
    """Attach `data.y` edge labels, falling back to None for orphan nodes."""

    import torch

    cfg = config or LabelGeneratorConfig()
    ast_nodes = tex_nodes_by_id(tex_ast)
    node_records = getattr(data, "node_records", None)
    if not isinstance(node_records, list):
        node_records = [{} for _ in range(int(data.num_nodes))]

    node_tex_ids: dict[int, str | None] = {}
    orphans: dict[int, OrphanAlignment] = {}
    for node_index in range(int(data.num_nodes)):
        tex_id, orphan = resolve_node_tex_id(node_index, node_records[node_index], pdf_to_tex, cfg)
        node_tex_ids[node_index] = tex_id
        if orphan is not None:
            orphans[node_index] = orphan

    labels: list[int] = []
    edge_index = data.edge_index.detach().cpu()
    for edge_pos in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        tex_source = node_tex_ids.get(source)
        tex_target = node_tex_ids.get(target)
        if not tex_source or not tex_target:
            labels.append(cfg.orphan_label)
            continue
        label = label_tex_relation(
            tex_source,
            tex_target,
            ast_nodes,
            adjacent_siblings_only=cfg.adjacent_siblings_only,
            directed_parent_child=cfg.directed_parent_child,
        )
        labels.append(int(label))

    y = torch.tensor(labels, dtype=torch.long)
    data.y = y
    data.edge_label = y
    data.label_schema = {
        "task": "edge_relation_classification",
        "labels": {
            int(TexRelationLabel.MERGE): "merge",
            int(TexRelationLabel.PARENT_CHILD): "parent_child",
            int(TexRelationLabel.SIBLING): "sibling",
            int(TexRelationLabel.NONE): "none",
        },
        "orphan_label": cfg.orphan_label,
        "similarity_threshold": cfg.similarity_threshold,
    }
    data.pdf_to_tex = [node_tex_ids.get(idx) for idx in range(int(data.num_nodes))]

    label_counts = {label: labels.count(label) for label in range(4)}
    orphan_list = list(orphans.values())
    if orphan_log_path is not None:
        write_orphan_log(orphan_log_path, orphan_list)
    return LabelGenerationResult(data=data, label_counts=label_counts, orphan_alignments=orphan_list)


def resolve_node_tex_id(
    node_index: int,
    node_record: dict[str, Any],
    pdf_to_tex: dict[str, Any],
    config: LabelGeneratorConfig,
) -> tuple[str | None, OrphanAlignment | None]:
    node_key = resolve_node_key(node_index, node_record, pdf_to_tex)
    if node_key is None:
        return None, OrphanAlignment(node_index=node_index, node_key=str(node_index), reason="missing_alignment")

    raw_alignment = pdf_to_tex.get(node_key)
    tex_id, score = parse_alignment_value(raw_alignment)
    if not tex_id:
        return None, OrphanAlignment(node_index=node_index, node_key=node_key, reason="missing_tex_id", score=score)
    if score is not None and score < config.similarity_threshold:
        return None, OrphanAlignment(node_index=node_index, node_key=node_key, reason="low_similarity", score=score)
    return tex_id, None


def resolve_node_key(node_index: int, node_record: dict[str, Any], pdf_to_tex: dict[str, Any]) -> str | None:
    candidates: list[str] = []
    for key in ("block_id", "id", "pdf_id", "global_order", "visual_order"):
        value = node_record.get(key)
        if value is not None:
            candidates.append(str(value))
    candidates.extend([str(node_index), f"P_{node_index}", f"P_{node_index + 1}", f"B_{node_index}", f"B_{node_index + 1}"])
    for candidate in candidates:
        if candidate in pdf_to_tex:
            return candidate
    return None


def parse_alignment_value(value: Any) -> tuple[str | None, float | None]:
    if isinstance(value, str):
        return value, None
    if isinstance(value, dict):
        tex_id = value.get("tex_id") or value.get("target") or value.get("id")
        score = value.get("score")
        if score is None:
            score = value.get("similarity")
        if score is None:
            score = value.get("confidence")
        parsed_score = float(score) if isinstance(score, (int, float)) else None
        return str(tex_id) if tex_id else None, parsed_score
    return None, None


def write_orphan_log(path: Path, orphans: list[OrphanAlignment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for orphan in orphans:
            file.write(json.dumps(asdict(orphan), ensure_ascii=False) + "\n")
