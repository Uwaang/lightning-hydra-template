from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multiclass focal loss built on top of cross entropy."""

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        pt = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt).pow(self.gamma) * ce

        if self.reduction == "mean":
            valid = targets != self.ignore_index
            return loss[valid].mean() if torch.any(valid) else loss.sum() * 0.0
        if self.reduction == "sum":
            return loss.sum()
        return loss
