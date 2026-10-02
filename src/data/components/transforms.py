from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import default_collate
from torchvision.transforms import v2


class TorchvisionV2Transform:
    """Compose torchvision v2 sample transforms behind the dataset's dict interface."""

    def __init__(
        self,
        operations: Sequence[Any],
        *,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be provided together.")

        pipeline: list[Any] = [
            v2.ToImage(),
            *operations,
            v2.ToDtype(torch.float32, scale=True),
        ]
        if mean is not None and std is not None:
            pipeline.append(v2.Normalize(mean=list(mean), std=list(std)))
        self.transform = v2.Compose(pipeline)

    def __call__(self, image: Any, **kwargs: Any) -> dict[str, Any]:
        return {"image": self.transform(image)}


class ClassificationBatchCollate:
    """Default-collate a classification batch and optionally apply MixUp/CutMix."""

    _SUPPORTED_MODES = {"none", "mixup", "cutmix", "mixup_cutmix"}

    def __init__(
        self,
        num_classes: int,
        mode: str = "none",
        *,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        if mode not in self._SUPPORTED_MODES:
            supported = ", ".join(sorted(self._SUPPORTED_MODES))
            raise ValueError(f"mode must be one of: {supported}.")
        if mixup_alpha <= 0 or cutmix_alpha <= 0:
            raise ValueError("mixup_alpha and cutmix_alpha must be positive.")

        mixup = v2.MixUp(alpha=mixup_alpha, num_classes=num_classes)
        cutmix = v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes)
        transforms = {
            "none": None,
            "mixup": mixup,
            "cutmix": cutmix,
            "mixup_cutmix": v2.RandomChoice([mixup, cutmix]),
        }
        self.num_classes = num_classes
        self.mode = mode
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.transform = transforms[mode]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"mode={self.mode!r}, "
            f"mixup_alpha={self.mixup_alpha}, "
            f"cutmix_alpha={self.cutmix_alpha})"
        )

    def __call__(self, samples: list[Any]) -> Any:
        batch = default_collate(samples)
        if self.transform is None:
            return batch

        if isinstance(batch, Mapping):
            images = batch["image"]
            hard_labels = batch["label"]
            mixed_images, mixed_labels = self.transform(images, hard_labels)
            result = dict(batch)
            result["image"] = mixed_images
            result["label"] = mixed_labels
            result["hard_label"] = hard_labels
            return result

        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, hard_labels = batch
            mixed_images, mixed_labels = self.transform(images, hard_labels)
            return {
                "image": mixed_images,
                "label": mixed_labels,
                "hard_label": hard_labels,
            }

        raise TypeError(
            "ClassificationBatchCollate expects mapping samples or two-item (image, label) samples."
        )
