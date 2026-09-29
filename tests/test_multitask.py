from functools import partial

import torch
from torch import nn

from src.models.components.multihead import MultiHeadClassifier
from src.models.multitask_module import MultiHeadClassificationLitModule


class DummyBackbone(nn.Module):
    num_features = 8

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, self.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def _network() -> MultiHeadClassifier:
    return MultiHeadClassifier(
        backbone=DummyBackbone(),
        heads={"a": 3, "b": 2},
    )


def test_multihead_classifier() -> None:
    model = _network()
    x = torch.randn(5, 4)

    outputs = model(x)
    assert outputs["a"].shape == (5, 3)
    assert outputs["b"].shape == (5, 2)
    assert model(x, head="a").shape == (5, 3)


def test_multitask_task_step() -> None:
    module = MultiHeadClassificationLitModule(
        net=_network(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        loss=nn.CrossEntropyLoss(),
        heads={"a": 3, "b": 2},
    )
    batch = {
        "image": torch.randn(4, 4),
        "label": torch.tensor([0, 1, 2, 1]),
    }

    loss, logits, preds, targets = module._task_step(batch, head="a")

    assert loss.ndim == 0
    assert logits.shape == (4, 3)
    assert preds.shape == (4,)
    assert torch.equal(targets, batch["label"])
