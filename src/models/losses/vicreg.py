from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Covariance matrix must be square.")
    return matrix.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    """VICReg invariance, variance, and covariance regularization."""

    def __init__(
        self,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape or z1.ndim != 2:
            raise ValueError("VICReg inputs must have matching shape [batch, features].")
        if z1.shape[0] < 2:
            raise ValueError("VICReg requires a batch size of at least 2.")

        invariance = F.mse_loss(z1, z2)

        std_z1 = torch.sqrt(z1.var(dim=0, unbiased=True) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0, unbiased=True) + self.eps)
        variance = F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean()

        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        divisor = z1.shape[0] - 1
        cov_z1 = z1_centered.T @ z1_centered / divisor
        cov_z2 = z2_centered.T @ z2_centered / divisor
        feature_dim = z1.shape[1]
        covariance = (
            _off_diagonal(cov_z1).pow(2).sum() / feature_dim
            + _off_diagonal(cov_z2).pow(2).sum() / feature_dim
        )

        return (
            self.invariance_weight * invariance
            + self.variance_weight * variance
            + self.covariance_weight * covariance
        )
