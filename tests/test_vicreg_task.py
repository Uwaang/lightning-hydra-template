from functools import partial

import torch
from torch import nn

from src.models.losses import VICRegLoss
from src.models.vicreg_module import VICRegLitModule


class DummyBackbone(nn.Module):
    num_features = 16

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, self.num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _module() -> VICRegLitModule:
    return VICRegLitModule(
        net=DummyBackbone(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        loss=VICRegLoss(),
        projector_hidden_dim=32,
        projector_output_dim=8,
    )


def test_vicreg_task_backward() -> None:
    module = _module()
    batch = {
        "view1": torch.randn(6, 3, 8, 8),
        "view2": torch.randn(6, 3, 8, 8),
    }

    loss = module.model_step(batch)
    loss.backward()

    assert loss.ndim == 0
    assert any(parameter.grad is not None for parameter in module.projector.parameters())


def test_vicreg_predicts_backbone_features() -> None:
    module = _module()
    batch = {
        "image": torch.randn(4, 3, 8, 8),
        "name": ["a", "b", "c", "d"],
    }

    output = module.predict_step(batch, batch_idx=0)

    assert output["features"].shape == (4, 16)
    assert output["names"] == ["a", "b", "c", "d"]
