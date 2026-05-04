"""PyG dataset wrapper for document graph training samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.reasoning.graph_builder import GraphBuildConfig, build_graph_from_content_v3
from src.reasoning.label_generator import LabelGeneratorConfig, label_graph_edges_from_paths

try:
    import torch
    from torch_geometric.data import Dataset
    from torch_geometric.loader import DataLoader
except ModuleNotFoundError:  # pragma: no cover - local lightweight env may omit torch/PyG.
    torch = None
    Dataset = object
    DataLoader = None


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    content_json: Path | None = None
    graph_path: Path | None = None
    tex_path: Path | None = None
    pdf_to_tex_path: Path | None = None


@dataclass(frozen=True)
class DocumentDatasetConfig:
    root: Path
    manifest_path: Path | None = None
    model_path: Path | None = None
    processed_dir_name: str = "processed"
    max_length: int = 512
    stride: int = 384
    batch_size: int = 16
    sequential_window: int = 3
    spatial_k: int = 3
    bidirectional_edges: bool = True
    alignment_threshold: float = 0.55

    def graph_config(self) -> GraphBuildConfig:
        if self.model_path is None:
            raise ValueError("model_path is required when a record does not provide graph_path")
        return GraphBuildConfig(
            model_path=self.model_path,
            max_length=self.max_length,
            stride=self.stride,
            batch_size=self.batch_size,
            sequential_window=self.sequential_window,
            spatial_k=self.spatial_k,
            bidirectional_edges=self.bidirectional_edges,
        )


class DocumentDataset(Dataset):  # type: ignore[misc]
    """Standard PyG Dataset that stores one processed `.pt` per document."""

    def __init__(self, config: DocumentDatasetConfig, records: list[DocumentRecord] | None = None):
        if torch is None:
            raise ModuleNotFoundError("DocumentDataset requires torch and torch-geometric to be installed")
        self.config = config
        self.records = records if records is not None else load_document_records(config.manifest_path, root=config.root)
        super().__init__(root=str(config.root))

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / self.config.processed_dir_name)

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{record.document_id}.pt" for record in self.records]

    def len(self) -> int:
        return len(self.records)

    def get(self, idx: int) -> Any:
        return torch.load(self.processed_paths[idx], map_location="cpu", weights_only=False)

    def process(self) -> None:
        label_config = LabelGeneratorConfig(similarity_threshold=self.config.alignment_threshold)
        for record, output_path in zip(self.records, self.processed_paths):
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            data = self._load_or_build_graph(record, output)
            if record.tex_path is not None and record.pdf_to_tex_path is not None:
                orphan_log_path = output.with_suffix(".orphans.jsonl")
                result = label_graph_edges_from_paths(
                    data,
                    tex_path=record.tex_path,
                    pdf_to_tex_path=record.pdf_to_tex_path,
                    config=label_config,
                    orphan_log_path=orphan_log_path,
                )
                data = result.data
                data.label_counts = result.label_counts
                data.orphan_count = len(result.orphan_alignments)
            else:
                data = attach_default_none_labels(data)
                data.label_counts = {0: 0, 1: 0, 2: 0, 3: int(data.edge_index.shape[1])}
                data.orphan_count = int(data.num_nodes)
            data.document_id = record.document_id
            torch.save(data, output)

    def _load_or_build_graph(self, record: DocumentRecord, output_path: Path) -> Any:
        if record.graph_path is not None:
            return torch.load(record.graph_path, map_location="cpu", weights_only=False)
        if record.content_json is None:
            raise ValueError(f"Record {record.document_id} must provide graph_path or content_json")
        return build_graph_from_content_v3(record.content_json, output_path, self.config.graph_config())


def attach_default_none_labels(data: Any) -> Any:
    if torch is None:
        raise ModuleNotFoundError("attach_default_none_labels requires torch")
    edge_count = int(data.edge_index.shape[1])
    y = torch.full((edge_count,), 3, dtype=torch.long)
    data.y = y
    data.edge_label = y
    data.label_schema = {
        "task": "edge_relation_classification",
        "labels": {0: "merge", 1: "parent_child", 2: "sibling", 3: "none"},
        "orphan_label": 3,
    }
    return data


def build_document_dataloader(dataset: DocumentDataset, *, batch_size: int = 8, shuffle: bool = True, **kwargs: Any) -> Any:
    if DataLoader is None:
        raise ModuleNotFoundError("PyG DataLoader requires torch-geometric to be installed")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def load_document_records(manifest_path: Path | None, *, root: Path) -> list[DocumentRecord]:
    if manifest_path is None:
        raise ValueError("manifest_path is required when records are not provided directly")
    base = manifest_path.parent
    text = manifest_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if manifest_path.suffix.lower() == ".jsonl":
        raw_records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        raw_records = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected manifest {manifest_path} to contain a list or documents list")
    return [document_record_from_mapping(record, base=base, root=root) for record in raw_records]


def document_record_from_mapping(record: dict[str, Any], *, base: Path, root: Path) -> DocumentRecord:
    document_id = str(record.get("document_id") or record.get("id") or record.get("stem") or "")
    if not document_id:
        raise ValueError(f"Document manifest record is missing document_id: {record}")
    return DocumentRecord(
        document_id=document_id,
        content_json=resolve_optional_path(record.get("content_json") or record.get("json"), base=base, root=root),
        graph_path=resolve_optional_path(record.get("graph_path") or record.get("graph"), base=base, root=root),
        tex_path=resolve_optional_path(record.get("tex_path") or record.get("tex"), base=base, root=root),
        pdf_to_tex_path=resolve_optional_path(record.get("pdf_to_tex_path") or record.get("alignment"), base=base, root=root),
    )


def resolve_optional_path(value: Any, *, base: Path, root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    base_candidate = base / path
    if base_candidate.exists():
        return base_candidate
    return root / path
