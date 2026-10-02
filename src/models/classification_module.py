from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy


class ClassificationLitModule(LightningModule):
    """Generic multiclass classification module for image models."""

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
        if num_classes < 2:
            raise ValueError("ClassificationLitModule requires num_classes >= 2.")

        self.save_hyperparameters(
            logger=False,
            ignore=["net", "optimizer", "scheduler", "loss"],
        )
        self.net = net
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
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

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, Mapping):
            return batch["image"], batch["label"]
        x, y = batch
        return x, y

    def model_step(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, targets = self._unpack_batch(batch)
        logits = self.forward(x)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        return loss, logits, preds, targets

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, _, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, _, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        accuracy = self.val_acc.compute()
        self.val_acc_best(accuracy)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, _, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        if isinstance(batch, Mapping):
            x = batch["image"]
            targets = batch.get("label")
            names = batch.get("name")
        else:
            x, targets = batch
            names = None

        logits = self.forward(x)
        result: dict[str, Any] = {
            "logits": logits,
            "preds": torch.argmax(logits, dim=1),
        }
        if targets is not None:
            result["targets"] = targets
        if names is not None:
            result["names"] = names
        return result

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.optimizer_factory(params=self.parameters())
        if self.scheduler_factory is None:
            return {"optimizer": optimizer}

        scheduler = self.scheduler_factory(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
