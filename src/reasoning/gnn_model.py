"""Neural modules for structure reasoning over PDF block graphs."""

from __future__ import annotations

from dataclasses import dataclass

from src.perception.schema import (
    COLUMN_FEATURE_FIELDS,
    DERIVED_STAT_FIELDS,
    EDGE_ATTR_FIELDS,
    FLOW_CONTEXT_FIELDS,
    GEOMETRY_FIELDS,
    LAYOUT_LAYER_FIELDS,
    SCIBERT_DIM,
    SCROLL_GEOMETRY_FIELDS,
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
    dropout: float = 0.2
    semantic_norm_eps: float = 1e-12
    layout_input_dim_override: int | None = None

    @property
    def layout_input_dim(self) -> int:
        override = getattr(self, "layout_input_dim_override", None)
        if override is not None:
            return int(override)
        return (
            self.type_dim
            + len(GEOMETRY_FIELDS)
            + len(SCROLL_GEOMETRY_FIELDS)
            + len(DERIVED_STAT_FIELDS)
            + len(STYLE_STAT_FIELDS)
            + len(SEQUENCE_POSITION_FIELDS)
            + len(COLUMN_FEATURE_FIELDS)
            + len(TITLE_STRUCTURE_FIELDS)
            + len(LAYOUT_LAYER_FIELDS)
            + len(FLOW_CONTEXT_FIELDS)
        )

    @property
    def output_dim(self) -> int:
        return self.semantic_hidden_dim + self.layout_hidden_dim


_MODULE_BASE = nn.Module if nn is not None else object


class FeatureProjector(_MODULE_BASE):
    """Project raw node features into a balanced model input space.

    The raw `.pt` graph keeps the full 768-dimensional SciBERT vector. This
    module is the model-side bottleneck: it L2-normalizes raw SciBERT vectors,
    compresses semantics, applies a second L2 stabilization after the ReLU
    bottleneck, and gives layout/type/stat features their own projection before
    concatenation.
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
        semantic = F.normalize(semantic, p=2, dim=-1, eps=self.config.semantic_norm_eps)
        semantic_projected = F.relu(self.semantic_projection(semantic))
        semantic_projected = self.semantic_dropout(semantic_projected)
        semantic_projected = F.normalize(semantic_projected, p=2, dim=-1, eps=self.config.semantic_norm_eps)
        return torch.cat([semantic_projected, self.layout(layout)], dim=-1)


@dataclass(frozen=True)
class EdgeGATConfig:
    node_projector: FeatureProjectorConfig = FeatureProjectorConfig()
    edge_dim: int = len(EDGE_ATTR_FIELDS)
    hidden_dim: int = 64
    heads: int = 4
    num_layers: int = 2
    num_classes: int = 3
    dropout: float = 0.1
    predictor_hidden_dims: tuple[int, ...] = (1024, 512, 128)
    predictor_layer_norm: bool = True
    edge_feature_mode: str = "full"
    disabled_node_feature_ranges: tuple[tuple[int, int], ...] = ()
    disabled_edge_attr_indices: tuple[int, ...] = ()

    @property
    def projected_node_dim(self) -> int:
        return self.node_projector.output_dim


class EdgeRelationGAT(_MODULE_BASE):
    """GATv2 edge classifier for Merge/Parent-Child/None relations."""

    def __init__(self, config: EdgeGATConfig | None = None):
        if nn is None or GATv2Conv is None:
            raise ModuleNotFoundError("EdgeRelationGAT requires torch and torch-geometric to be installed")
        super().__init__()
        self.config = config or EdgeGATConfig()
        self.projector = FeatureProjector(self.config.node_projector)

        convs = []
        in_dim = self.config.projected_node_dim
        for _ in range(max(0, self.config.num_layers)):
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
        self.edge_feature_dim = self._edge_feature_dim(in_dim)
        self.edge_head = build_edge_predictor_head(
            input_dim=self.edge_feature_dim,
            hidden_dims=getattr(self.config, "predictor_hidden_dims", (1024, 512, 128)),
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
            layer_norm=getattr(self.config, "predictor_layer_norm", True),
        )

    def forward(self, data=None, *, x=None, edge_index=None, edge_attr=None):  # type: ignore[no-untyped-def]
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
        if x is None or edge_index is None or edge_attr is None:
            raise ValueError("EdgeRelationGAT.forward requires data or x/edge_index/edge_attr")

        x, edge_attr = self._mask_inputs(x, edge_attr)
        edge_attr = self._align_edge_attr(edge_attr)
        h = self.projector(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.config.dropout, training=self.training)

        source, target = edge_index
        edge_features = self._build_edge_features(h, source, target, edge_attr)
        return self.edge_head(edge_features)

    def _mask_inputs(self, x, edge_attr):  # type: ignore[no-untyped-def]
        """Apply checkpoint-stored feature ablations without mutating graph data."""

        node_ranges = tuple(getattr(self.config, "disabled_node_feature_ranges", ()) or ())
        edge_indices = tuple(getattr(self.config, "disabled_edge_attr_indices", ()) or ())
        if node_ranges:
            x = x.clone()
            width = int(x.shape[1])
            for start, end in node_ranges:
                start_i = max(0, min(width, int(start)))
                end_i = max(start_i, min(width, int(end)))
                if end_i > start_i:
                    x[:, start_i:end_i] = 0.0
        if edge_indices:
            edge_attr = edge_attr.clone()
            width = int(edge_attr.shape[1])
            for idx in edge_indices:
                idx_i = int(idx)
                if 0 <= idx_i < width:
                    edge_attr[:, idx_i] = 0.0
        return x, edge_attr

    def _build_edge_features(self, h, source, target, edge_attr):  # type: ignore[no-untyped-def]
        """Build directional edge features with symmetry-breaking terms."""

        h_source = h[source]
        h_target = h[target]
        mode = getattr(self.config, "edge_feature_mode", "full")
        if mode == "simple_concat":
            return torch.cat([h_source, h_target, edge_attr], dim=-1)
        return torch.cat(
            [
                h_source,
                h_target,
                h_source - h_target,
                h_source * h_target,
                edge_attr,
            ],
            dim=-1,
        )

    def _edge_feature_dim(self, node_dim: int) -> int:
        mode = getattr(self.config, "edge_feature_mode", "full")
        if mode == "full":
            return int(node_dim) * 4 + self.config.edge_dim
        if mode == "simple_concat":
            return int(node_dim) * 2 + self.config.edge_dim
        raise ValueError(f"Unknown edge_feature_mode: {mode}")

    def _align_edge_attr(self, edge_attr):  # type: ignore[no-untyped-def]
        """Pad or truncate runtime edge attributes to the checkpoint config."""

        expected_dim = int(self.config.edge_dim)
        if edge_attr.shape[1] > expected_dim:
            return edge_attr[:, :expected_dim]
        if edge_attr.shape[1] < expected_dim:
            return F.pad(edge_attr, (0, expected_dim - edge_attr.shape[1]))
        return edge_attr


def build_edge_predictor_head(
    *,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    num_classes: int,
    dropout: float,
    layer_norm: bool = True,
):
    """Build the edge-classification MLP.

    The first Linear layer intentionally consumes the full directional relation
    vector: concat([Hu, Hv, Hu-Hv, Hu*Hv, Euv]).  Hidden dimensions default to
    1024 -> 512 -> 128 so the predictor has enough capacity to model rare
    MERGE edges without changing the graph encoder itself.
    """

    if nn is None:
        raise ModuleNotFoundError("build_edge_predictor_head requires torch")
    layers = []
    current_dim = int(input_dim)
    for hidden_dim in tuple(int(dim) for dim in hidden_dims):
        if hidden_dim <= 0:
            continue
        layers.append(nn.Linear(current_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, num_classes))
    return nn.Sequential(*layers)
