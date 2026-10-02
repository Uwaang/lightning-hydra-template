import pickle

import numpy as np
import pytest
import torch
from torchvision.transforms import v2

from src.data.components.transforms import (
    ClassificationBatchCollate,
    TorchvisionV2Transform,
)


def test_torchvision_v2_transform_returns_normalized_tensor() -> None:
    transform = TorchvisionV2Transform(
        operations=[v2.Resize((8, 8), antialias=True)],
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    image = np.full((4, 6, 3), 255, dtype=np.uint8)

    output = transform(image=image)["image"]

    assert output.shape == (3, 8, 8)
    assert output.dtype == torch.float32
    assert torch.allclose(output, torch.ones_like(output))


@pytest.mark.parametrize("mode", ["mixup", "cutmix", "mixup_cutmix"])
def test_classification_batch_collate_preserves_hard_labels(mode: str) -> None:
    collate = ClassificationBatchCollate(
        num_classes=3,
        mode=mode,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
    )
    samples = [
        {"image": torch.rand(3, 8, 8), "label": torch.tensor(index % 3)} for index in range(4)
    ]

    batch = collate(samples)

    assert batch["image"].shape == (4, 3, 8, 8)
    assert batch["label"].shape == (4, 3)
    assert batch["label"].dtype == torch.float32
    assert torch.allclose(batch["label"].sum(dim=1), torch.ones(4))
    assert torch.equal(batch["hard_label"], torch.tensor([0, 1, 2, 0]))


def test_classification_batch_collate_repr_is_stable() -> None:
    collate = ClassificationBatchCollate(
        num_classes=10,
        mode="mixup_cutmix",
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
    )

    assert repr(collate) == (
        "ClassificationBatchCollate("
        "num_classes=10, mode='mixup_cutmix', mixup_alpha=0.2, cutmix_alpha=1.0)"
    )


def test_classification_batch_collate_is_picklable_for_worker_processes() -> None:
    collate = ClassificationBatchCollate(num_classes=3, mode="mixup_cutmix")

    restored = pickle.loads(pickle.dumps(collate))

    assert restored.mode == "mixup_cutmix"


def test_classification_batch_collate_none_keeps_default_batch() -> None:
    collate = ClassificationBatchCollate(num_classes=3, mode="none")
    samples = [
        {"image": torch.rand(3, 8, 8), "label": torch.tensor(index % 3)} for index in range(4)
    ]

    batch = collate(samples)

    assert "hard_label" not in batch
    assert batch["label"].shape == (4,)
