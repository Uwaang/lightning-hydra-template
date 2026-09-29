import pytest
import torch
from torch import nn

from src.utils.gradcam import resolve_module


class NestedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3),
            nn.Sequential(nn.ReLU(), nn.Conv2d(4, 8, 3)),
        )


def test_resolve_module_path() -> None:
    model = NestedModel()

    assert resolve_module(model, "features.0") is model.features[0]
    assert resolve_module(model, "features.1.1") is model.features[1][1]


def test_resolve_module_rejects_unknown_path() -> None:
    with pytest.raises(KeyError):
        resolve_module(NestedModel(), "features.4")


def test_compute_gradcam_optional_dependency() -> None:
    pytest.importorskip("pytorch_grad_cam")

    from src.utils.gradcam import compute_gradcam

    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    masks = compute_gradcam(
        model=model,
        input_tensor=torch.randn(2, 3, 16, 16),
        target_layer="0",
        categories=[0, 1],
    )

    assert masks.shape == (2, 16, 16)
