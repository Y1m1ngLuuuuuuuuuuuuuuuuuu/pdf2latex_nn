"""Neural modules for structure reasoning over PDF block graphs."""

from __future__ import annotations

from dataclasses import dataclass

from src.perception.schema import (
    COLUMN_FEATURE_FIELDS,
    DERIVED_STAT_FIELDS,
    EDGE_ATTR_FIELDS,
    GEOMETRY_FIELDS,
    SCIBERT_DIM,
    SEQUENCE_POSITION_FIELDS,
    STYLE_STAT_FIELDS,
    TITLE_STRUCTURE_FIELDS,
)

try:
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - local lightweight env may omit torch.
    torch = None
    F = None
    nn = None

try:
    from torch_geometric.nn import GATv2Conv
except ModuleNotFoundError:  # pragma: no cover - local lightweight env may omit PyG.
    GATv2Conv = None


@dataclass(frozen=True)
class FeatureProjectorConfig:
    semantic_dim: int = SCIBERT_DIM
    semantic_hidden_dim: int = 64
    layout_hidden_dim: int = 32
    type_dim: int = 10
    dropout: float = 0.1
    layout_input_dim_override: int | None = None

    @property
    def layout_input_dim(self) -> int:
        override = getattr(self, "layout_input_dim_override", None)
        if override is not None:
            return int(override)
        return (
            self.type_dim
            + len(GEOMETRY_FIELDS)
            + len(DERIVED_STAT_FIELDS)
            + len(STYLE_STAT_FIELDS)
            + len(SEQUENCE_POSITION_FIELDS)
            + len(COLUMN_FEATURE_FIELDS)
            + len(TITLE_STRUCTURE_FIELDS)
        )

    @property
    def output_dim(self) -> int:
        return self.semantic_hidden_dim + self.layout_hidden_dim


_MODULE_BASE = nn.Module if nn is not None else object


class FeatureProjector(_MODULE_BASE):
    """Project raw node features into a balanced model input space.

    The raw `.pt` graph keeps the full 768-dimensional SciBERT vector. This
    module is the model-side bottleneck: it compresses semantics, applies L2
    normalization, and gives layout/type/stat features their own projection
    before concatenation.
    """

    def __init__(self, config: FeatureProjectorConfig | None = None):
        if nn is None:
            raise ModuleNotFoundError("FeatureProjector requires torch to be installed")
        super().__init__()
        self.config = config or FeatureProjectorConfig()
        self.semantic_projection = nn.Linear(self.config.semantic_dim, self.config.semantic_hidden_dim)
        self.semantic_dropout = nn.Dropout(self.config.dropout)
        self.layout = nn.Sequential(
            nn.Linear(self.config.layout_input_dim, self.config.layout_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.config.layout_hidden_dim),
            nn.Dropout(self.config.dropout),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        semantic = x[:, : self.config.semantic_dim]
        layout = x[:, self.config.semantic_dim :]
        expected_layout_dim = self.config.layout_input_dim
        if layout.shape[1] > expected_layout_dim:
            layout = layout[:, :expected_layout_dim]
        elif layout.shape[1] < expected_layout_dim:
            layout = F.pad(layout, (0, expected_layout_dim - layout.shape[1]))
        semantic_projected = F.relu(self.semantic_projection(semantic))
        semantic_projected = F.normalize(semantic_projected, p=2, dim=-1)
        semantic_projected = self.semantic_dropout(semantic_projected)
        return torch.cat([semantic_projected, self.layout(layout)], dim=-1)


@dataclass(frozen=True)
class EdgeGATConfig:
    node_projector: FeatureProjectorConfig = FeatureProjectorConfig()
    edge_dim: int = len(EDGE_ATTR_FIELDS)
    hidden_dim: int = 64
    heads: int = 4
    num_layers: int = 2
    num_classes: int = 4
    dropout: float = 0.1

    @property
    def projected_node_dim(self) -> int:
        return self.node_projector.output_dim


class EdgeRelationGAT(_MODULE_BASE):
    """GATv2 edge classifier for Merge/Parent-Child/Sibling/None relations."""

    def __init__(self, config: EdgeGATConfig | None = None):
        if nn is None or GATv2Conv is None:
            raise ModuleNotFoundError("EdgeRelationGAT requires torch and torch-geometric to be installed")
        super().__init__()
        self.config = config or EdgeGATConfig()
        self.projector = FeatureProjector(self.config.node_projector)

        convs = []
        in_dim = self.config.projected_node_dim
        for _ in range(max(1, self.config.num_layers)):
            convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=self.config.hidden_dim,
                    heads=self.config.heads,
                    edge_dim=self.config.edge_dim,
                    dropout=self.config.dropout,
                    concat=True,
                )
            )
            in_dim = self.config.hidden_dim * self.config.heads
        self.convs = nn.ModuleList(convs)
        self.edge_head = nn.Sequential(
            nn.Linear(in_dim * 4 + self.config.edge_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_classes),
        )

    def forward(self, data=None, *, x=None, edge_index=None, edge_attr=None):  # type: ignore[no-untyped-def]
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
        if x is None or edge_index is None or edge_attr is None:
            raise ValueError("EdgeRelationGAT.forward requires data or x/edge_index/edge_attr")

        h = self.projector(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.config.dropout, training=self.training)

        source, target = edge_index
        edge_features = self._build_edge_features(h, source, target, edge_attr)
        return self.edge_head(edge_features)

    @staticmethod
    def _build_edge_features(h, source, target, edge_attr):  # type: ignore[no-untyped-def]
        """Build directional edge features with symmetry-breaking terms."""

        h_source = h[source]
        h_target = h[target]
        return torch.cat(
            [
                h_source,
                h_target,
                h_target - h_source,
                h_source * h_target,
                edge_attr,
            ],
            dim=-1,
        )
