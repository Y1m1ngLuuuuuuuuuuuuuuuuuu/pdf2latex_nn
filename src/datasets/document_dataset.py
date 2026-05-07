"""PyG dataset wrapper for document graph training samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.perception.schema import FeatureTensorSchema
from src.pipeline.v7_contract import assert_v7_content_json, assert_v7_graph_data, V7ContractError
from src.reasoning.graph_builder import GraphBuildConfig, build_graph_from_content_v7
from src.reasoning.label_generator import AlignmentQualityError, LabelGeneratorConfig, label_graph_edges_from_paths

try:
    import torch
    from torch_geometric.data import Dataset
    from torch_geometric.loader import DataLoader
except ModuleNotFoundError:  # pragma: no cover - local lightweight env may omit torch/PyG.
    torch = None
    Dataset = object
    DataLoader = None


FEATURE_SCHEMA = FeatureTensorSchema()
PROCESSED_INDEX_NAME = "index.json"
SKIPPED_RECORDS_NAME = "skipped_records.jsonl"
PYG_EXCLUDE_KEYS = [
    "edge_attr_schema",
    "edge_source_types",
    "feature_schema",
    "label_counts",
    "label_schema",
    "model_path",
    "node_records",
    "pdf_to_tex",
    "pdf_to_tex_scores",
    "alignment_schema",
    "source_path",
]


class GraphFilterError(ValueError):
    """Raised when a graph is structurally invalid for training."""


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
    expected_node_dim: int = FEATURE_SCHEMA.node_feature_dim
    expected_edge_dim: int = FEATURE_SCHEMA.edge_attr_dim
    drop_empty_edge_graphs: bool = True
    drop_all_orphan_graphs: bool = True
    require_v7_contract: bool = True

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
        return [PROCESSED_INDEX_NAME]

    def len(self) -> int:
        return len(self._processed_index())

    def get(self, idx: int) -> Any:
        entries = self._processed_index()
        entry = entries[idx]
        return torch.load(Path(self.processed_dir) / entry["path"], map_location="cpu", weights_only=False)

    def process(self) -> None:
        label_config = LabelGeneratorConfig(similarity_threshold=self.config.alignment_threshold)
        processed_dir = Path(self.processed_dir)
        graph_dir = processed_dir / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        skipped_path = processed_dir / SKIPPED_RECORDS_NAME
        if skipped_path.exists():
            skipped_path.unlink()

        index_entries: list[dict[str, Any]] = []
        for record in self.records:
            output = graph_dir / f"{safe_filename(record.document_id)}.pt"
            try:
                data = self._load_or_build_graph(record, output)
                data = sanitize_graph_data(data, config=self.config, require_labels=False)
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
                elif has_valid_edge_labels(data):
                    data.y = normalize_edge_labels(data.y)
                    data.edge_label = data.y
                    data.label_counts = count_edge_labels(data.y)
                    data.orphan_count = int(getattr(data, "orphan_count", 0))
                else:
                    data = attach_default_none_labels(data)
                    data.label_counts = {0: 0, 1: 0, 2: int(data.edge_index.shape[1])}
                    data.orphan_count = int(data.num_nodes)
                data.document_id = record.document_id
                data = sanitize_graph_data(data, config=self.config, require_labels=True)
                assert_graph_is_trainable(data, config=self.config)
            except (GraphFilterError, AlignmentQualityError, V7ContractError) as exc:
                append_skip_log(skipped_path, record, reason=str(exc))
                continue
            torch.save(data, output)
            index_entries.append(
                {
                    "document_id": record.document_id,
                    "path": str(output.relative_to(processed_dir)),
                    "num_nodes": int(data.num_nodes),
                    "num_edges": int(data.edge_index.shape[1]),
                    "orphan_count": int(getattr(data, "orphan_count", 0)),
                }
            )
        self._write_processed_index(index_entries)

    def _load_or_build_graph(self, record: DocumentRecord, output_path: Path) -> Any:
        if record.graph_path is not None:
            data = torch.load(record.graph_path, map_location="cpu", weights_only=False)
            if self.config.require_v7_contract:
                assert_v7_graph_data(data, record.graph_path)
            return data
        if record.content_json is None:
            raise ValueError(f"Record {record.document_id} must provide graph_path or content_json")
        if self.config.require_v7_contract:
            assert_v7_content_json(record.content_json, require_styles=True)
        return build_graph_from_content_v7(record.content_json, output_path, self.config.graph_config())

    def _processed_index(self) -> list[dict[str, Any]]:
        index_path = Path(self.processed_dir) / PROCESSED_INDEX_NAME
        if not index_path.exists():
            return []
        data = json.loads(index_path.read_text(encoding="utf-8"))
        entries = data.get("graphs", []) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError(f"Malformed processed index: {index_path}")
        return [entry for entry in entries if isinstance(entry, dict)]

    def _write_processed_index(self, entries: list[dict[str, Any]]) -> None:
        index_path = Path(self.processed_dir) / PROCESSED_INDEX_NAME
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "document_dataset_index_v1",
            "graphs": entries,
            "skipped_log": SKIPPED_RECORDS_NAME,
        }
        index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def attach_default_none_labels(data: Any) -> Any:
    if torch is None:
        raise ModuleNotFoundError("attach_default_none_labels requires torch")
    edge_count = int(data.edge_index.shape[1])
    y = torch.full((edge_count,), 2, dtype=torch.long)
    data.y = y
    data.edge_label = y
    data.label_schema = {
        "task": "edge_relation_classification",
        "labels": {0: "merge", 1: "parent_child", 2: "none"},
        "orphan_label": 2,
    }
    return data


def has_valid_edge_labels(data: Any) -> bool:
    if not hasattr(data, "y"):
        return False
    y = data.y
    return y is not None and y.ndim == 1 and int(y.shape[0]) == int(data.edge_index.shape[1])


def count_edge_labels(labels: Any) -> dict[int, int]:
    labels = normalize_edge_labels(labels).detach().cpu().long()
    counts = torch.bincount(labels, minlength=3).tolist()
    return {label: int(counts[label]) for label in range(3)}


def normalize_edge_labels(labels: Any) -> Any:
    """Fold legacy 4-class labels into the current 3-class target space.

    Old graphs used 2=sibling and 3=none. Sibling is now a derived relation,
    so both old sibling and old none become 2=none.
    """

    if torch is None:
        raise ModuleNotFoundError("normalize_edge_labels requires torch")
    labels = labels.to(dtype=torch.long)
    return torch.where(labels >= 2, torch.full_like(labels, 2), labels)


def sanitize_graph_data(
    data: Any,
    *,
    config: DocumentDatasetConfig | None = None,
    require_labels: bool = False,
) -> Any:
    """Normalize tensor dtypes, clamp non-finite values, and validate shapes."""

    if torch is None:
        raise ModuleNotFoundError("sanitize_graph_data requires torch")
    cfg = config or DocumentDatasetConfig(root=Path("."))
    if not hasattr(data, "x") or not hasattr(data, "edge_index") or not hasattr(data, "edge_attr"):
        raise GraphFilterError("missing required graph tensors")

    data.x = torch.nan_to_num(data.x.to(dtype=torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    data.edge_attr = torch.nan_to_num(data.edge_attr.to(dtype=torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    data.edge_index = data.edge_index.to(dtype=torch.long)

    if data.x.ndim != 2 or int(data.x.shape[1]) != cfg.expected_node_dim:
        raise GraphFilterError(f"bad node feature shape: {tuple(data.x.shape)}")
    if data.edge_index.ndim != 2 or int(data.edge_index.shape[0]) != 2:
        raise GraphFilterError(f"bad edge_index shape: {tuple(data.edge_index.shape)}")
    if data.edge_attr.ndim != 2 or int(data.edge_attr.shape[1]) != cfg.expected_edge_dim:
        raise GraphFilterError(f"bad edge_attr shape: {tuple(data.edge_attr.shape)}")
    if int(data.edge_attr.shape[0]) != int(data.edge_index.shape[1]):
        raise GraphFilterError("edge_attr rows must match edge_index columns")
    if int(data.x.shape[0]) == 0:
        raise GraphFilterError("empty node graph")
    if cfg.drop_empty_edge_graphs and int(data.edge_index.shape[1]) == 0:
        raise GraphFilterError("empty edge graph")
    if int(data.edge_index.shape[1]) > 0:
        max_node_index = int(data.x.shape[0]) - 1
        if int(data.edge_index.min().item()) < 0 or int(data.edge_index.max().item()) > max_node_index:
            raise GraphFilterError("edge_index contains out-of-range node ids")
    if require_labels:
        if not hasattr(data, "y"):
            raise GraphFilterError("missing edge labels")
        data.y = normalize_edge_labels(data.y)
        if data.y.ndim != 1 or int(data.y.shape[0]) != int(data.edge_index.shape[1]):
            raise GraphFilterError(f"bad edge label shape: {tuple(data.y.shape)}")
        data.edge_label = data.y
    return data


def assert_graph_is_trainable(data: Any, *, config: DocumentDatasetConfig) -> None:
    if config.drop_all_orphan_graphs and int(getattr(data, "orphan_count", 0)) >= int(data.num_nodes):
        raise GraphFilterError("all-orphan graph")
    if config.drop_empty_edge_graphs and int(data.edge_index.shape[1]) == 0:
        raise GraphFilterError("empty edge graph")


def append_skip_log(path: Path, record: DocumentRecord, *, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"document_id": record.document_id, "reason": reason}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def build_document_dataloader(dataset: DocumentDataset, *, batch_size: int = 8, shuffle: bool = True, **kwargs: Any) -> Any:
    if DataLoader is None:
        raise ModuleNotFoundError("PyG DataLoader requires torch-geometric to be installed")
    kwargs.setdefault("exclude_keys", list(PYG_EXCLUDE_KEYS))
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
