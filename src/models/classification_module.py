from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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
        teacher: nn.Module | None = None,
        teacher_state_dict_path: str | None = None,
        distillation_alpha: float = 0.0,
        distillation_temperature: float = 4.0,
        compile: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("ClassificationLitModule requires num_classes >= 2.")
        if not 0.0 <= distillation_alpha <= 1.0:
            raise ValueError("distillation_alpha must be between 0 and 1.")
        if distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be greater than zero.")
        if teacher is None and distillation_alpha > 0:
            raise ValueError("A teacher model is required when distillation_alpha is positive.")
        if teacher is None and teacher_state_dict_path is not None:
            raise ValueError("teacher_state_dict_path requires a teacher model.")

        self.save_hyperparameters(
            logger=False,
            ignore=["net", "optimizer", "scheduler", "loss", "teacher"],
        )
        self.net = net
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.criterion = loss
        self.teacher = teacher
        if self.teacher is not None:
            if teacher_state_dict_path is not None:
                teacher_state = torch.load(
                    Path(teacher_state_dict_path),
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(teacher_state, Mapping):
                    raise ValueError("Teacher state dict must be a mapping of parameter names.")
                self.teacher.load_state_dict(teacher_state)
            self.teacher.requires_grad_(False)
            self.teacher.eval()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc_best = MaxMetric()

    def train(self, mode: bool = True) -> ClassificationLitModule:
        super().train(mode)
        teacher = getattr(self, "teacher", None)
        if teacher is not None:
            teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _training_loss(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        hard_loss = self.criterion(logits, targets)
        if self.teacher is None or self.hparams.distillation_alpha == 0:
            return hard_loss

        with torch.no_grad():
            teacher_logits = self.teacher(x)

        if teacher_logits.shape != logits.shape:
            raise ValueError(
                "Teacher and student logits must have the same shape for logit distillation."
            )

        temperature = float(self.hparams.distillation_temperature)
        soft_loss = F.kl_div(
            F.log_softmax(logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature**2)
        alpha = float(self.hparams.distillation_alpha)
        return (1.0 - alpha) * hard_loss + alpha * soft_loss

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch, Mapping):
            targets = batch["label"]
            metric_targets = batch.get("hard_label", targets)
            return batch["image"], targets, metric_targets
        x, y = batch
        return x, y, y

    def model_step(
        self,
        batch: Any,
        *,
        distill: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, targets, metric_targets = self._unpack_batch(batch)
        logits = self.forward(x)
        loss = self._training_loss(x, logits, targets) if distill else self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        return loss, logits, preds, metric_targets

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, _, preds, targets = self.model_step(batch, distill=True)
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
        if self.teacher is None:
            optimizer_parameters = self.parameters()
        else:
            teacher_parameter_ids = {id(parameter) for parameter in self.teacher.parameters()}
            optimizer_parameters = (
                parameter
                for parameter in self.parameters()
                if id(parameter) not in teacher_parameter_ids
            )
        optimizer = self.optimizer_factory(params=optimizer_parameters)
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
