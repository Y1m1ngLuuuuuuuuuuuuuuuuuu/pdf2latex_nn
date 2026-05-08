"""Training losses and metrics for edge relation classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - local lightweight env may omit torch.
    torch = None
    nn = None
    F = None

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover - Lightning is optional in this repo.
    pl = None


DEFAULT_EDGE_CLASS_WEIGHTS = (4.0, 5.0, 1.0)


@dataclass(frozen=True)
class EdgeMetrics:
    per_class: dict[int, dict[str, float]]
    macro_f1: float


_LOSS_BASE = nn.Module if nn is not None else object


class FocalLoss(_LOSS_BASE):
    """Multi-class focal loss with optional class weights."""

    def __init__(self, gamma: float = 2.0, weight: Any | None = None, reduction: str = "mean"):
        if nn is None:
            raise ModuleNotFoundError("FocalLoss requires torch to be installed")
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits: Any, target: Any) -> Any:
        if int(logits.shape[-1]) == 3:
            target = torch.where(target.long() >= 2, torch.full_like(target.long(), 2), target.long())
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.weight is not None:
            alpha = self.weight.to(device=logits.device, dtype=logits.dtype)
            loss = loss * alpha[target.long()]
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def default_class_weight_tensor(device: Any | None = None) -> Any:
    if torch is None:
        raise ModuleNotFoundError("default_class_weight_tensor requires torch")
    return torch.tensor(DEFAULT_EDGE_CLASS_WEIGHTS, dtype=torch.float32, device=device)


def compute_inverse_frequency_weights(labels: Any, *, num_classes: int = 3, smoothing: float = 1.0) -> Any:
    if torch is None:
        raise ModuleNotFoundError("compute_inverse_frequency_weights requires torch")
    labels = labels.detach().cpu().long()
    if num_classes == 3:
        labels = torch.where(labels >= 2, torch.full_like(labels, 2), labels)
    counts = torch.bincount(labels, minlength=num_classes).float() + smoothing
    counts = counts[:num_classes]
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.mean()


def build_edge_loss(
    *,
    loss_name: str = "focal",
    class_weights: Any | None = None,
    gamma: float = 2.0,
    device: Any | None = None,
) -> Any:
    if nn is None:
        raise ModuleNotFoundError("build_edge_loss requires torch")
    weight = class_weights if class_weights is not None else default_class_weight_tensor(device=device)
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight)
    if loss_name == "focal":
        return FocalLoss(gamma=gamma, weight=weight)
    raise ValueError(f"Unknown edge loss: {loss_name}")


def edge_precision_recall_f1(pred: Any, target: Any, *, num_classes: int = 3) -> EdgeMetrics:
    if torch is None:
        raise ModuleNotFoundError("edge_precision_recall_f1 requires torch")
    if pred.ndim == 2:
        pred = pred.argmax(dim=-1)
    pred = pred.detach().cpu().long()
    target = target.detach().cpu().long()
    if num_classes == 3:
        pred = torch.where(pred >= 2, torch.full_like(pred, 2), pred)
        target = torch.where(target >= 2, torch.full_like(target, 2), target)
    per_class: dict[int, dict[str, float]] = {}
    f1_values = []
    for cls in range(num_classes):
        true_positive = int(((pred == cls) & (target == cls)).sum().item())
        false_positive = int(((pred == cls) & (target != cls)).sum().item())
        false_negative = int(((pred != cls) & (target == cls)).sum().item())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": int((target == cls).sum().item())}
        f1_values.append(f1)
    return EdgeMetrics(per_class=per_class, macro_f1=sum(f1_values) / max(1, len(f1_values)))


def train_one_epoch(model: Any, dataloader: Any, optimizer: Any, loss_fn: Any, *, device: Any) -> dict[str, float]:
    if torch is None:
        raise ModuleNotFoundError("train_one_epoch requires torch")
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        batches += 1
    return {"loss": total_loss / max(1, batches)}


def evaluate_edge_model(model: Any, dataloader: Any, *, device: Any) -> dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("evaluate_edge_model requires torch")
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            preds.append(model(batch).detach().cpu())
            targets.append(batch.y.detach().cpu())
    metrics = edge_precision_recall_f1(torch.cat(preds, dim=0), torch.cat(targets, dim=0))
    return {"macro_f1": metrics.macro_f1, "per_class": metrics.per_class}


if pl is not None and nn is not None:

    class EdgeRelationLightningModule(pl.LightningModule):  # type: ignore[misc]
        """Optional Lightning wrapper for the edge-classification GAT."""

        def __init__(self, model: Any, lr: float = 1e-3, loss_name: str = "focal", gamma: float = 2.0):
            super().__init__()
            self.model = model
            self.lr = lr
            self.loss_name = loss_name
            self.gamma = gamma
            self.loss_fn = build_edge_loss(loss_name=loss_name, gamma=gamma)

        def forward(self, batch: Any) -> Any:
            return self.model(batch)

        def training_step(self, batch: Any, batch_idx: int) -> Any:
            logits = self(batch)
            loss = self.loss_fn(logits, batch.y)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch: Any, batch_idx: int) -> None:
            logits = self(batch)
            loss = self.loss_fn(logits, batch.y)
            metrics = edge_precision_recall_f1(logits.detach(), batch.y.detach())
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_macro_f1", metrics.macro_f1, prog_bar=True)

        def configure_optimizers(self) -> Any:
            return torch.optim.AdamW(self.parameters(), lr=self.lr)

else:

    class EdgeRelationLightningModule:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError("EdgeRelationLightningModule requires lightning to be installed")
