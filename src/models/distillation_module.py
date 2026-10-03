from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import MeanMetric

from src.models.classification_module import ClassificationLitModule
from src.utils.saving_utils import load_module_weights


class DistillationClassificationLitModule(ClassificationLitModule):
    """Classification module with optional teacher-logit knowledge distillation during training."""

    def __init__(
        self,
        teacher: nn.Module,
        *,
        alpha: float = 0.5,
        temperature: float = 4.0,
        teacher_checkpoint: str | None = None,
        teacher_strip_prefix: str = "",
        teacher_ignore_shape_mismatch: bool = False,
        **kwargs: Any,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        super().__init__(**kwargs)
        self.teacher = teacher
        if teacher_checkpoint is not None:
            load_module_weights(
                self.teacher,
                teacher_checkpoint,
                strip_prefix=teacher_strip_prefix,
                ignore_shape_mismatch=teacher_ignore_shape_mismatch,
            )

        self.alpha = float(alpha)
        self.temperature = float(temperature)
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()

        self.train_supervised_loss = MeanMetric()
        self.train_distillation_loss = MeanMetric()
        self.save_hyperparameters(
            {
                "distillation_alpha": self.alpha,
                "distillation_temperature": self.temperature,
                "teacher_checkpoint": teacher_checkpoint,
                "teacher_strip_prefix": teacher_strip_prefix,
                "teacher_ignore_shape_mismatch": teacher_ignore_shape_mismatch,
            }
        )

    def train(self, mode: bool = True) -> DistillationClassificationLitModule:
        module = super().train(mode)
        if hasattr(self, "teacher"):
            self.teacher.eval()
        return module

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "Student and teacher logits must have the same shape for logit distillation: "
                f"{tuple(student_logits.shape)} != {tuple(teacher_logits.shape)}"
            )

        temperature = self.temperature
        return (
            F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            )
            * temperature**2
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, loss_targets, metric_targets = self._unpack_batch(batch)
        student_logits = self.forward(inputs)
        supervised_loss = self.criterion(student_logits, loss_targets)

        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        distillation_loss = self._distillation_loss(student_logits, teacher_logits)
        loss = (1.0 - self.alpha) * supervised_loss + self.alpha * distillation_loss

        predictions = torch.argmax(student_logits, dim=1)
        self.train_loss(loss)
        self.train_supervised_loss(supervised_loss)
        self.train_distillation_loss(distillation_loss)
        self.train_acc(predictions, metric_targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/supervised_loss",
            self.train_supervised_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/distillation_loss",
            self.train_distillation_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
