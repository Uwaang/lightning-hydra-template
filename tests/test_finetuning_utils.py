from functools import partial

import pytest
import torch
from torch import nn

from src.models.classification_module import ClassificationLitModule
from src.utils.finetuning_utils import ParameterPatternFinetuning


class TinyTransferNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.backbone(x)))


def _module() -> ClassificationLitModule:
    return ClassificationLitModule(
        net=TinyTransferNet(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        scheduler=None,
        loss=nn.CrossEntropyLoss(),
        num_classes=2,
    )


def test_pattern_finetuning_freezes_then_unfreezes() -> None:
    module = _module()
    callback = ParameterPatternFinetuning(
        trainable_patterns=["net.head.*"],
        unfreeze_at_epoch=1,
        initial_denom_lr=10.0,
        train_bn=False,
    )

    callback.freeze_before_training(module)

    assert not module.net.backbone.weight.requires_grad
    assert not module.net.backbone.bias.requires_grad
    assert module.net.head.weight.requires_grad
    assert module.net.head.bias.requires_grad

    optimizer = module.configure_optimizers()["optimizer"]
    assert len(optimizer.param_groups) == 1
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == {
        id(module.net.head.weight),
        id(module.net.head.bias),
    }

    callback.finetune_function(module, current_epoch=0, optimizer=optimizer)
    assert len(optimizer.param_groups) == 1

    callback.finetune_function(module, current_epoch=1, optimizer=optimizer)

    assert all(parameter.requires_grad for parameter in module.parameters())
    assert len(optimizer.param_groups) == 2
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized_ids == {id(parameter) for parameter in module.parameters()}
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.01)


def test_pattern_finetuning_rejects_unmatched_patterns() -> None:
    module = _module()
    callback = ParameterPatternFinetuning(
        trainable_patterns=["net.not_a_real_head.*"],
        train_bn=False,
    )

    with pytest.raises(ValueError, match="No parameters matched"):
        callback.freeze_before_training(module)
