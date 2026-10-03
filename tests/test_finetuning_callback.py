from types import SimpleNamespace

import torch
from lightning import LightningModule
from torch import nn

from src.callbacks import BackboneUnfreezingCallback


class TinyTransferNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)


class TinyTransferModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = TinyTransferNet()


def test_backbone_unfreezing_callback_freezes_then_unfreezes() -> None:
    module = TinyTransferModule()
    callback = BackboneUnfreezingCallback(unfreeze_at_epoch=2)
    trainer = SimpleNamespace(current_epoch=0)

    callback.on_fit_start(trainer, module)  # type: ignore[arg-type]

    assert all(not parameter.requires_grad for parameter in module.net.backbone.parameters())
    assert all(parameter.requires_grad for parameter in module.net.head.parameters())

    trainer.current_epoch = 1
    callback.on_train_epoch_start(trainer, module)  # type: ignore[arg-type]
    assert all(not parameter.requires_grad for parameter in module.net.backbone.parameters())

    trainer.current_epoch = 2
    callback.on_train_epoch_start(trainer, module)  # type: ignore[arg-type]
    assert all(parameter.requires_grad for parameter in module.net.backbone.parameters())


def test_backbone_unfreezing_callback_rejects_negative_epoch() -> None:
    try:
        BackboneUnfreezingCallback(unfreeze_at_epoch=-1)
    except ValueError as exc:
        assert "zero or greater" in str(exc)
    else:
        raise AssertionError("negative unfreeze epoch should be rejected")


def test_backbone_unfreezing_callback_requires_adapter_contract() -> None:
    module = LightningModule()
    module.net = nn.Linear(2, 2)  # type: ignore[attr-defined]
    callback = BackboneUnfreezingCallback()

    try:
        callback.on_fit_start(SimpleNamespace(current_epoch=0), module)  # type: ignore[arg-type]
    except TypeError as exc:
        assert "set_backbone_trainable" in str(exc)
    else:
        raise AssertionError("missing fine-tuning adapter contract should be rejected")
