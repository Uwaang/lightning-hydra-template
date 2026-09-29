from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as tv_models
from torchvision.models.feature_extraction import create_feature_extractor


class TorchvisionFeatureMapBackbone(nn.Module):
    """Return one explicitly named intermediate feature map from torchvision."""

    def __init__(
        self,
        model_name: str,
        return_node: str,
        weights: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        model = tv_models.get_model(model_name, weights=weights, **model_kwargs)
        self.extractor = create_feature_extractor(
            model,
            return_nodes={return_node: "features"},
        )
        self.return_node = return_node

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)["features"]


class GeM(nn.Module):
    """Generalized mean pooling over spatial dimensions."""

    def __init__(
        self,
        p: float = 3.0,
        eps: float = 1e-6,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if p <= 0:
            raise ValueError("GeM p must be positive.")
        self.eps = eps
        initial = torch.tensor(float(p))
        if trainable:
            self.p = nn.Parameter(initial)
        else:
            self.register_buffer("p", initial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("GeM expects a feature map with shape [B, C, H, W].")
        p = self.p.clamp(min=self.eps)
        pooled = x.clamp(min=self.eps).pow(p).mean(dim=(-2, -1)).pow(1.0 / p)
        return pooled


class EmbeddingModel(nn.Module):
    """Feature-map backbone followed by pooling, projection, and normalization."""

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        embedding_dim: int | None = 128,
        pool: nn.Module | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = pool if pool is not None else GeM()
        self.normalize = normalize

        if embedding_dim is None:
            self.projection: nn.Module = nn.Identity()
            self.embedding_dim = in_features
        else:
            self.projection = nn.Sequential(
                nn.Linear(in_features, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
            )
            self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim == 4:
            features = self.pool(features)
        elif features.ndim > 2:
            features = torch.flatten(features, start_dim=1)

        embeddings = self.projection(features)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
