from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric


class VICRegLitModule(LightningModule):
    """Self-supervised representation learning with VICReg."""

    def __init__(
        self,
        net: nn.Module,
        optimizer: Any,
        scheduler: Any | None,
        loss: nn.Module,
        projector_hidden_dim: int = 2048,
        projector_output_dim: int = 512,
        feature_dim: int | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__()

        if feature_dim is None:
            feature_dim = getattr(net, "num_features", None)
        if feature_dim is None:
            raise ValueError("feature_dim is required when net does not expose num_features.")

        self.save_hyperparameters(logger=False, ignore=["net", "loss"])
        self.net = net
        self.criterion = loss
        self.feature_dim = int(feature_dim)

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, projector_output_dim),
        )

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encode(x))

    def model_step(self, batch: Mapping[str, Any]) -> torch.Tensor:
        z1 = self.forward(batch["view1"])
        z2 = self.forward(batch["view2"])
        return self.criterion(z1, z2)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        loss = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        loss = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        image = batch.get("image")
        if image is None:
            image = batch["view1"]

        result: dict[str, Any] = {"features": self.encode(image)}
        if "label" in batch:
            result["targets"] = batch["label"]
        if "name" in batch:
            result["names"] = batch["name"]
        return result

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
