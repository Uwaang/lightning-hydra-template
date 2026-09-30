from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy


class MultiHeadClassificationLitModule(LightningModule):
    """Train named classification heads from a CombinedLoader batch."""

    def __init__(
        self,
        net: nn.Module,
        optimizer: Any,
        scheduler: Any | None,
        loss: nn.Module,
        heads: Mapping[str, int],
        compile: bool = False,
    ) -> None:
        super().__init__()
        if not heads:
            raise ValueError("At least one classification head is required.")

        self.save_hyperparameters(logger=False, ignore=["net", "loss"])
        self.net = net
        self.criterion = loss
        self.head_names = list(heads)

        self.train_loss = nn.ModuleDict({name: MeanMetric() for name in heads})
        self.val_loss = nn.ModuleDict({name: MeanMetric() for name in heads})
        self.test_loss = nn.ModuleDict({name: MeanMetric() for name in heads})

        self.train_acc = nn.ModuleDict(
            {
                name: MulticlassAccuracy(num_classes=int(num_classes))
                for name, num_classes in heads.items()
            }
        )
        self.val_acc = nn.ModuleDict(
            {
                name: MulticlassAccuracy(num_classes=int(num_classes))
                for name, num_classes in heads.items()
            }
        )
        self.test_acc = nn.ModuleDict(
            {
                name: MulticlassAccuracy(num_classes=int(num_classes))
                for name, num_classes in heads.items()
            }
        )

        self.total_val_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def forward(
        self,
        x: torch.Tensor,
        head: str | None = None,
    ) -> Any:
        return self.net(x, head=head)

    def _task_step(
        self,
        batch: Mapping[str, Any],
        head: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(batch["image"], head=head)
        targets = batch["label"]
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        return loss, logits, preds, targets

    def on_train_start(self) -> None:
        self.val_acc_best.reset()

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        losses = []
        for head in self.head_names:
            task_batch = batch.get(head)
            if task_batch is None:
                continue

            loss, _, preds, targets = self._task_step(task_batch, head)
            self.train_loss[head](loss)
            self.train_acc[head](preds, targets)
            self.log(
                f"train/{head}/loss",
                self.train_loss[head],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"train/{head}/acc",
                self.train_acc[head],
                on_step=False,
                on_epoch=True,
            )
            losses.append(loss)

        if not losses:
            raise RuntimeError("CombinedLoader batch did not contain any configured heads.")

        total_loss = torch.stack(losses).sum()
        self.log("train/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def on_validation_epoch_start(self) -> None:
        self.total_val_loss.reset()

    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        head = self.head_names[dataloader_idx]
        loss, _, preds, targets = self._task_step(batch, head)
        self.val_loss[head](loss)
        self.val_acc[head](preds, targets)
        self.total_val_loss(loss)
        self.log(
            f"val/{head}/loss",
            self.val_loss[head],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"val/{head}/acc",
            self.val_acc[head],
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self) -> None:
        mean_accuracy = torch.stack(
            [self.val_acc[head].compute() for head in self.head_names]
        ).mean()
        self.val_acc_best(mean_accuracy)
        self.log("val/loss", self.total_val_loss.compute(), prog_bar=True, sync_dist=True)
        self.log("val/acc", mean_accuracy, prog_bar=True, sync_dist=True)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)

    def test_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        head = self.head_names[dataloader_idx]
        loss, _, preds, targets = self._task_step(batch, head)
        self.test_loss[head](loss)
        self.test_acc[head](preds, targets)
        self.log(
            f"test/{head}/loss",
            self.test_loss[head],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"test/{head}/acc",
            self.test_acc[head],
            on_step=False,
            on_epoch=True,
        )

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        logits = self.forward(batch["image"])
        result: dict[str, Any] = {
            "logits": logits,
            "preds": {
                head: torch.argmax(head_logits, dim=1) for head, head_logits in logits.items()
            },
        }
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
