from __future__ import annotations

from fnmatch import fnmatchcase

from lightning import LightningModule
from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim import Optimizer


class ParameterPatternFinetuning(BaseFinetuning):
    """Freeze everything except matching parameters, then unfreeze the rest later.

    Parameter names are matched with ``fnmatch`` patterns against
    ``LightningModule.named_parameters()``. This avoids requiring a specific ``backbone``/``head``
    attribute layout from custom, torchvision, or timm models.
    """

    def __init__(
        self,
        trainable_patterns: list[str],
        unfreeze_at_epoch: int = 1,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
    ) -> None:
        super().__init__()
        if not trainable_patterns:
            raise ValueError("trainable_patterns must contain at least one pattern.")
        if unfreeze_at_epoch < 0:
            raise ValueError("unfreeze_at_epoch must be zero or greater.")
        if initial_denom_lr <= 0:
            raise ValueError("initial_denom_lr must be positive.")

        self.trainable_patterns = tuple(trainable_patterns)
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.initial_denom_lr = initial_denom_lr
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        self.freeze(pl_module, train_bn=self.train_bn)

        matched = []
        for name, parameter in pl_module.named_parameters():
            if any(fnmatchcase(name, pattern) for pattern in self.trainable_patterns):
                parameter.requires_grad = True
                matched.append(name)

        if not matched:
            patterns = ", ".join(self.trainable_patterns)
            raise ValueError(f"No parameters matched fine-tuning patterns: {patterns}")

    def finetune_function(
        self,
        pl_module: LightningModule,
        current_epoch: int,
        optimizer: Optimizer,
    ) -> None:
        if current_epoch != self.unfreeze_at_epoch:
            return

        self.unfreeze_and_add_param_group(
            modules=pl_module,
            optimizer=optimizer,
            initial_denom_lr=self.initial_denom_lr,
            train_bn=self.train_bn,
        )
