"""Neural modules for structure reasoning over PDF block graphs."""

from __future__ import annotations

from dataclasses import dataclass

from src.perception.schema import DERIVED_STAT_FIELDS, GEOMETRY_FIELDS, SCIBERT_DIM

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - local lightweight env may omit torch.
    torch = None
    nn = None


@dataclass(frozen=True)
class FeatureProjectorConfig:
    semantic_dim: int = SCIBERT_DIM
    semantic_hidden_dim: int = 64
    layout_hidden_dim: int = 32
    type_dim: int = 10
    dropout: float = 0.1

    @property
    def layout_input_dim(self) -> int:
        return self.type_dim + len(GEOMETRY_FIELDS) + len(DERIVED_STAT_FIELDS)

    @property
    def output_dim(self) -> int:
        return self.semantic_hidden_dim + self.layout_hidden_dim


_MODULE_BASE = nn.Module if nn is not None else object


class FeatureProjector(_MODULE_BASE):
    """Project raw node features into a balanced model input space.

    The raw `.pt` graph keeps the full 768-dimensional SciBERT vector. This
    module is the model-side bottleneck: it compresses semantics, normalizes
    them, and gives layout/type/stat features their own projection before
    concatenation.
    """

    def __init__(self, config: FeatureProjectorConfig | None = None):
        if nn is None:
            raise ModuleNotFoundError("FeatureProjector requires torch to be installed")
        super().__init__()
        self.config = config or FeatureProjectorConfig()
        self.semantic = nn.Sequential(
            nn.Linear(self.config.semantic_dim, self.config.semantic_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.config.semantic_hidden_dim),
            nn.Dropout(self.config.dropout),
        )
        self.layout = nn.Sequential(
            nn.Linear(self.config.layout_input_dim, self.config.layout_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.config.layout_hidden_dim),
            nn.Dropout(self.config.dropout),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        semantic = x[:, : self.config.semantic_dim]
        layout = x[:, self.config.semantic_dim :]
        return torch.cat([self.semantic(semantic), self.layout(layout)], dim=-1)
