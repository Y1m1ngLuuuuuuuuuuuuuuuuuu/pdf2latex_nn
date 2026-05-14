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
    prediction_architecture: str = "shared"
    message_edge_mode: str = "all"
    merge_gate_mode: str = "none"
    merge_gate_logit: float = -20.0
    gaussian_edge_feature_mode: str = "none"
    gaussian_sigma: float = 0.10
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
        self.raw_edge_dim = int(self.config.edge_dim)
        self.gaussian_extra_dim = self._gaussian_extra_dim()
        self.effective_edge_dim = self.raw_edge_dim + self.gaussian_extra_dim
        self.projector = FeatureProjector(self.config.node_projector)

        convs = []
        in_dim = self.config.projected_node_dim
        for _ in range(max(0, self.config.num_layers)):
            convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=self.config.hidden_dim,
                    heads=self.config.heads,
                    edge_dim=self.effective_edge_dim,
                    dropout=self.config.dropout,
                    concat=True,
                )
            )
            in_dim = self.config.hidden_dim * self.config.heads
        self.convs = nn.ModuleList(convs)
        self.gnn_node_dim = in_dim
        self.raw_node_dim = self.config.projected_node_dim
        architecture = getattr(self.config, "prediction_architecture", "shared")
        if architecture == "shared":
            self.edge_feature_dim = self._edge_feature_dim(self.gnn_node_dim)
            self.edge_head = build_edge_predictor_head(
                input_dim=self.edge_feature_dim,
                hidden_dims=getattr(self.config, "predictor_hidden_dims", (1024, 512, 128)),
                num_classes=self.config.num_classes,
                dropout=self.config.dropout,
                layer_norm=getattr(self.config, "predictor_layer_norm", True),
            )
        elif architecture == "y_network":
            self.merge_edge_feature_dim = self._edge_feature_dim(self.raw_node_dim)
            self.parent_edge_feature_dim = self._edge_feature_dim(self.gnn_node_dim)
            self.merge_head = build_edge_predictor_head(
                input_dim=self.merge_edge_feature_dim,
                hidden_dims=getattr(self.config, "predictor_hidden_dims", (1024, 512, 128)),
                num_classes=1,
                dropout=self.config.dropout,
                layer_norm=getattr(self.config, "predictor_layer_norm", True),
            )
            self.parent_none_head = build_edge_predictor_head(
                input_dim=self.parent_edge_feature_dim,
                hidden_dims=getattr(self.config, "predictor_hidden_dims", (1024, 512, 128)),
                num_classes=max(1, self.config.num_classes - 1),
                dropout=self.config.dropout,
                layer_norm=getattr(self.config, "predictor_layer_norm", True),
            )
            # Keep this attribute for tooling/tests that inspect the relation
            # feature width. It refers to the propagated parent/none tower.
            self.edge_feature_dim = self.parent_edge_feature_dim
        else:
            raise ValueError(f"Unknown prediction_architecture: {architecture}")

    def forward(
        self,
        data=None,
        *,
        x=None,
        edge_index=None,
        edge_attr=None,
        message_edge_mask=None,
        merge_candidate_mask=None,
    ):  # type: ignore[no-untyped-def]
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            message_edge_mask = getattr(data, "message_edge_mask", message_edge_mask)
            merge_candidate_mask = getattr(data, "merge_candidate_mask", merge_candidate_mask)
        if x is None or edge_index is None or edge_attr is None:
            raise ValueError("EdgeRelationGAT.forward requires data or x/edge_index/edge_attr")

        x, edge_attr = self._mask_inputs(x, edge_attr)
        edge_attr = self._prepare_edge_attr(edge_attr)
        h_raw = self.projector(x)
        h = h_raw
        conv_edge_index, conv_edge_attr = self._message_passing_edges(edge_index, edge_attr, message_edge_mask)
        for conv in self.convs:
            h = conv(h, conv_edge_index, edge_attr=conv_edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.config.dropout, training=self.training)

        source, target = edge_index
        architecture = getattr(self.config, "prediction_architecture", "shared")
        if architecture == "shared":
            edge_features = self._build_edge_features(h, source, target, edge_attr)
            logits = self.edge_head(edge_features)
        elif architecture == "y_network":
            merge_features = self._build_edge_features(h_raw, source, target, edge_attr)
            parent_features = self._build_edge_features(h, source, target, edge_attr)
            merge_logit = self.merge_head(merge_features)
            parent_none_logits = self.parent_none_head(parent_features)
            logits = torch.cat([merge_logit, parent_none_logits], dim=-1)
        else:  # pragma: no cover - guarded during initialization.
            raise ValueError(f"Unknown prediction_architecture: {architecture}")
        return self._apply_merge_gate(logits, merge_candidate_mask)

    def _message_passing_edges(self, edge_index, edge_attr, message_edge_mask):  # type: ignore[no-untyped-def]
        """Return the propagation-only graph.

        Edge classification still uses the full candidate graph.  When
        ``message_edge_mode=type_aware``, GAT node updates are restricted to
        ``data.message_edge_mask`` so structural/float/noise nodes do not
        pollute ordinary text states.
        """

        mode = getattr(self.config, "message_edge_mode", "all")
        if mode == "all":
            return edge_index, edge_attr
        if mode != "type_aware":
            raise ValueError(f"Unknown message_edge_mode: {mode}")
        if message_edge_mask is None:
            return edge_index, edge_attr
        mask = message_edge_mask.to(device=edge_index.device, dtype=torch.bool).flatten()
        edge_count = int(edge_index.shape[1])
        if int(mask.numel()) != edge_count:
            raise ValueError(f"message_edge_mask length {int(mask.numel())} does not match edge_count {edge_count}")
        if int(mask.sum().item()) == 0:
            return edge_index[:, :0], edge_attr[:0]
        return edge_index[:, mask], edge_attr[mask]

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
            return int(node_dim) * 4 + self.effective_edge_dim
        if mode == "simple_concat":
            return int(node_dim) * 2 + self.effective_edge_dim
        raise ValueError(f"Unknown edge_feature_mode: {mode}")

    def _apply_merge_gate(self, logits, merge_candidate_mask):  # type: ignore[no-untyped-def]
        """Optionally suppress MERGE for physically impossible candidate edges.

        This is class-specific gating, not edge pruning: the edge remains
        available for PARENT_CHILD/NONE classification, but the MERGE logit is
        clamped to a very low value when the upstream layout gate says this pair
        cannot be safely contracted.
        """

        mode = getattr(self.config, "merge_gate_mode", "none")
        if mode == "none" or merge_candidate_mask is None or int(logits.shape[1]) == 0:
            return logits
        if mode != "hard":
            raise ValueError(f"Unknown merge_gate_mode: {mode}")
        mask = merge_candidate_mask.to(device=logits.device, dtype=torch.bool).flatten()
        edge_count = int(logits.shape[0])
        if int(mask.numel()) != edge_count:
            raise ValueError(f"merge_candidate_mask length {int(mask.numel())} does not match edge_count {edge_count}")
        if int(logits.shape[1]) < 1:
            return logits
        gated = logits.clone()
        gated[:, 0] = torch.where(
            mask,
            gated[:, 0],
            torch.full_like(gated[:, 0], float(getattr(self.config, "merge_gate_logit", -20.0))),
        )
        return gated

    def _prepare_edge_attr(self, edge_attr):  # type: ignore[no-untyped-def]
        """Align raw edge attributes and append optional derived model features."""

        edge_attr = self._align_edge_attr(edge_attr)
        return self._append_gaussian_edge_features(edge_attr)

    def _align_edge_attr(self, edge_attr):  # type: ignore[no-untyped-def]
        """Pad or truncate runtime raw edge attributes to the checkpoint config."""

        expected_dim = self.raw_edge_dim
        if edge_attr.shape[1] > expected_dim:
            return edge_attr[:, :expected_dim]
        if edge_attr.shape[1] < expected_dim:
            return F.pad(edge_attr, (0, expected_dim - edge_attr.shape[1]))
        return edge_attr

    def _gaussian_extra_dim(self) -> int:
        mode = getattr(self.config, "gaussian_edge_feature_mode", "none")
        if mode in (None, "", "none"):
            return 0
        if mode == "center":
            return 1
        raise ValueError(f"Unknown gaussian_edge_feature_mode: {mode}")

    def _append_gaussian_edge_features(self, edge_attr):  # type: ignore[no-untyped-def]
        """Append Gaussian proximity hints derived from existing edge features.

        M07 keeps graph `.pt` tensors immutable.  The stored edge_attr remains
        the 22-dimensional v7 contract; this model-side feature turns the
        normalized center distance into a proximity cue:

            exp(-d^2 / (2 * sigma^2))

        The feature is a hint for the learned edge heads and GATv2 edge MLP. It
        is not a hard attention bias.
        """

        mode = getattr(self.config, "gaussian_edge_feature_mode", "none")
        if mode in (None, "", "none"):
            return edge_attr
        if mode != "center":
            raise ValueError(f"Unknown gaussian_edge_feature_mode: {mode}")
        try:
            distance_idx = EDGE_ATTR_FIELDS.index("center_distance")
        except ValueError as exc:  # pragma: no cover - schema constant regression guard.
            raise ValueError("EDGE_ATTR_FIELDS must contain center_distance for gaussian edge features") from exc
        if distance_idx >= int(edge_attr.shape[1]):
            distance = torch.zeros((int(edge_attr.shape[0]),), dtype=edge_attr.dtype, device=edge_attr.device)
        else:
            distance = edge_attr[:, distance_idx].clamp_min(0.0)
        sigma = max(float(getattr(self.config, "gaussian_sigma", 0.10)), 1e-6)
        gaussian = torch.exp(-((distance ** 2) / (2.0 * sigma * sigma))).unsqueeze(-1)
        return torch.cat([edge_attr, gaussian], dim=-1)


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
