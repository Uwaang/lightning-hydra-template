from __future__ import annotations

from typing import Any

from lightning import Callback, LightningModule, Trainer


class BackboneUnfreezingCallback(Callback):
    """Freeze a classifier backbone first, then unfreeze it at a chosen epoch.

    The wrapped network must expose ``set_backbone_trainable(bool)``. Optimizers in this
    template are constructed from all module parameters, so no optimizer param group needs to
    be added when the backbone is unfrozen; frozen parameters simply have no gradients until
    their ``requires_grad`` flag is restored.
    """

    def __init__(self, unfreeze_at_epoch: int = 1) -> None:
        super().__init__()
        if unfreeze_at_epoch < 0:
            raise ValueError("unfreeze_at_epoch must be zero or greater.")
        self.unfreeze_at_epoch = unfreeze_at_epoch

    @staticmethod
    def _set_backbone_trainable(pl_module: LightningModule, trainable: bool) -> None:
        net: Any = getattr(pl_module, "net", None)
        setter = getattr(net, "set_backbone_trainable", None)
        if not callable(setter):
            raise TypeError(
                "BackboneUnfreezingCallback requires pl_module.net to expose "
                "set_backbone_trainable(bool)."
            )
        setter(trainable)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._set_backbone_trainable(pl_module, False)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch >= self.unfreeze_at_epoch:
            self._set_backbone_trainable(pl_module, True)
