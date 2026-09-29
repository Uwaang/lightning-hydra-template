from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


_DEFAULTS = {
    "arcface": (64.0, 0.5),
    "sphereface": (64.0, 1.35),
    "cosface": (30.0, 0.4),
}


class AngularMarginSoftmaxLoss(nn.Module):
    """ArcFace, SphereFace, or CosFace classification loss."""

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        loss_type: str = "cosface",
        scale: float | None = None,
        margin: float | None = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        loss_type = loss_type.lower()
        if loss_type not in _DEFAULTS:
            raise ValueError(f"loss_type must be one of {tuple(_DEFAULTS)}")

        default_scale, default_margin = _DEFAULTS[loss_type]
        self.loss_type = loss_type
        self.scale = default_scale if scale is None else float(scale)
        self.margin = default_margin if margin is None else float(margin)
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [batch, embedding_size].")

        labels = labels.long()
        cosine = F.linear(
            F.normalize(embeddings, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )
        row = torch.arange(labels.shape[0], device=labels.device)
        target = cosine[row, labels]

        if self.loss_type == "cosface":
            target = target - self.margin
        else:
            theta = torch.acos(target.clamp(-1.0 + self.eps, 1.0 - self.eps))
            if self.loss_type == "arcface":
                target = torch.cos(theta + self.margin)
            else:
                target = torch.cos(theta * self.margin)

        logits = cosine * self.scale
        logits = logits.clone()
        logits[row, labels] = target * self.scale

        loss = F.cross_entropy(logits, labels)
        return loss, cosine
