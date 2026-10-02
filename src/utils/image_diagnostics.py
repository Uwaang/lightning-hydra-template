from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from torch.utils.data import Dataset


def _extract_images_targets(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        images = batch["image"]
        targets = batch["label"]
    else:
        images, targets = batch[:2]
    if not isinstance(images, torch.Tensor):
        raise TypeError("Image diagnostics require tensor images.")
    return images, torch.as_tensor(targets)


def _denormalize(
    images: torch.Tensor,
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
) -> torch.Tensor:
    images = images.detach().cpu().float()
    if mean is not None and std is not None:
        channels = images.shape[1]
        if len(mean) != channels or len(std) != channels:
            raise ValueError(
                f"Normalization metadata has {len(mean)}/{len(std)} channels, expected {channels}."
            )
        mean_tensor = torch.tensor(mean, dtype=images.dtype).view(1, channels, 1, 1)
        std_tensor = torch.tensor(std, dtype=images.dtype).view(1, channels, 1, 1)
        images = images * std_tensor + mean_tensor
    return images.clamp(0.0, 1.0)


def _label_name(value: int, class_names: Sequence[str] | None) -> str:
    if class_names is not None and 0 <= value < len(class_names):
        return class_names[value]
    return str(value)


def save_classification_image_grid(
    images: torch.Tensor,
    targets: torch.Tensor,
    path: str | Path,
    *,
    class_names: Sequence[str] | None = None,
    predictions: torch.Tensor | None = None,
    confidences: torch.Tensor | None = None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    max_images: int = 16,
    title: str | None = None,
) -> Path | None:
    """Save a compact classification image grid for qualitative inspection."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    count = min(max_images, int(images.shape[0]))
    if count < 1:
        return None

    display_images = _denormalize(images[:count], mean, std)
    targets = targets[:count].detach().cpu()
    predictions = predictions[:count].detach().cpu() if predictions is not None else None
    confidences = confidences[:count].detach().cpu() if confidences is not None else None

    columns = min(4, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.3 * rows), squeeze=False)

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue

        image = display_images[index]
        if image.shape[0] == 1:
            axis.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axis.imshow(image.permute(1, 2, 0).numpy())

        target = int(targets[index])
        label = f"gt: {_label_name(target, class_names)}"
        if predictions is not None:
            pred = int(predictions[index])
            label += f"\npred: {_label_name(pred, class_names)}"
            if confidences is not None:
                label += f" ({float(confidences[index]):.3f})"
        axis.set_title(label, fontsize=9)

    if title:
        figure.suptitle(title, y=0.995)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        figure.tight_layout()

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def select_confident_errors(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_images: int,
) -> list[Mapping[str, Any]]:
    """Select bounded high-confidence failures for human error analysis."""
    if max_images < 1:
        return []

    errors = [
        row
        for row in rows
        if row.get("correct") is False and isinstance(row.get("confidence"), (int, float))
    ]
    errors.sort(key=lambda row: float(row["confidence"]), reverse=True)
    return errors[:max_images]


def save_confident_error_gallery(
    rows: Sequence[Mapping[str, Any]],
    dataset: Dataset[Any],
    path: str | Path,
    *,
    class_names: Sequence[str] | None = None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    max_images: int = 32,
) -> Path | None:
    """Load only selected test samples and save a high-confidence error gallery."""
    selected = select_confident_errors(rows, max_images=max_images)
    if not selected:
        return None

    images: list[torch.Tensor] = []
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []

    for row in selected:
        sample = dataset[int(row["sample_index"])]
        image, target_tensor = _extract_images_targets(sample)
        if image.ndim == 3:
            images.append(image)
        else:
            raise ValueError("Dataset samples must expose individual CHW image tensors.")

        target = int(target_tensor.item()) if target_tensor.ndim == 0 else int(target_tensor)
        targets.append(target)
        predictions.append(int(row["pred"]))
        confidences.append(float(row["confidence"]))

    return save_classification_image_grid(
        torch.stack(images),
        torch.tensor(targets),
        path,
        class_names=class_names,
        predictions=torch.tensor(predictions),
        confidences=torch.tensor(confidences),
        mean=mean,
        std=std,
        max_images=max_images,
        title="Highest-confidence test errors",
    )


class ClassificationImageDiagnosticsCallback(Callback):
    """Bounded image diagnostics: one train preview and fixed validation first/final views."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        class_names: Sequence[str] | None = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        train_preview_images: int = 16,
        fixed_val_images: int = 16,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir) / "reports" / "images"
        self.class_names = list(class_names) if class_names is not None else None
        self.mean = list(mean) if mean is not None else None
        self.std = list(std) if std is not None else None
        self.train_preview_images = train_preview_images
        self.fixed_val_images = fixed_val_images
        self._train_saved = False
        self._first_val_saved = False
        self._fixed_val_images: torch.Tensor | None = None
        self._fixed_val_targets: torch.Tensor | None = None

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (
            self._train_saved
            or not trainer.is_global_zero
            or trainer.current_epoch != 0
            or batch_idx != 0
        ):
            return
        images, targets = _extract_images_targets(batch)
        save_classification_image_grid(
            images,
            targets,
            self.output_dir / "train_batch.png",
            class_names=self.class_names,
            mean=self.mean,
            std=self.std,
            max_images=self.train_preview_images,
            title="Train mini-batch after transforms",
        )
        self._train_saved = True

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if (
            self._fixed_val_images is not None
            or trainer.sanity_checking
            or not trainer.is_global_zero
            or dataloader_idx != 0
            or batch_idx != 0
        ):
            return
        images, targets = _extract_images_targets(batch)
        count = min(self.fixed_val_images, int(images.shape[0]))
        self._fixed_val_images = images[:count].detach().cpu()
        self._fixed_val_targets = targets[:count].detach().cpu()

    def _save_fixed_validation(
        self,
        pl_module: LightningModule,
        filename: str,
        title: str,
    ) -> None:
        if self._fixed_val_images is None or self._fixed_val_targets is None:
            return

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            images = self._fixed_val_images.to(pl_module.device)
            logits = pl_module(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
        if was_training:
            pl_module.train()

        save_classification_image_grid(
            self._fixed_val_images,
            self._fixed_val_targets,
            self.output_dir / "validation_fixed" / filename,
            class_names=self.class_names,
            predictions=predictions,
            confidences=confidences,
            mean=self.mean,
            std=self.std,
            max_images=self.fixed_val_images,
            title=title,
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (
            trainer.sanity_checking
            or not trainer.is_global_zero
            or self._first_val_saved
            or trainer.current_epoch != 0
        ):
            return
        self._save_fixed_validation(pl_module, "first.png", "Fixed validation samples: epoch 0")
        self._first_val_saved = True

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self._save_fixed_validation(pl_module, "final.png", "Fixed validation samples: final")
