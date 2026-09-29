from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class MultiHeadClassifier(nn.Module):
    """Shared feature extractor with one linear classification head per task."""

    def __init__(
        self,
        backbone: nn.Module,
        heads: Mapping[str, int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "num_features"):
            raise ValueError("backbone must expose a num_features attribute.")
        if not heads:
            raise ValueError("At least one head is required.")

        self.backbone = backbone
        self.num_features = int(backbone.num_features)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(self.num_features, int(num_classes)),
                )
                for name, num_classes in heads.items()
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        head: str | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)

        if head is not None:
            if head not in self.heads:
                raise KeyError(f"Unknown head: {head}")
            return self.heads[head](features)

        return {name: layer(features) for name, layer in self.heads.items()}
