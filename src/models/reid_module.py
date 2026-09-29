from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy


class ReIdentificationLitModule(LightningModule):
    """Embedding training with an angular-margin classification objective."""

    def __init__(
        self,
        net: nn.Module,
        optimizer: Any,
        scheduler: Any | None,
        loss: nn.Module,
        num_classes: int,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "loss"])
        self.net = net
        self.criterion = loss

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def model_step(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.forward(batch["image"])
        targets = batch["label"].long()
        loss, cosine = self.criterion(embeddings, targets)
        preds = torch.argmax(cosine, dim=1)
        return loss, embeddings, preds, targets

    def on_train_start(self) -> None:
        self.val_acc_best.reset()

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        loss, _, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        accuracy = self.val_acc.compute()
        self.val_acc_best(accuracy)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        loss, _, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"embeddings": self.forward(batch["image"])}
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
