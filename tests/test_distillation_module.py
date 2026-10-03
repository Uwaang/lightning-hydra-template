from functools import partial

import pytest
import torch
from torch import nn

from src.models.distillation_module import DistillationClassificationLitModule


def _network(num_classes: int = 3) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes))


def _module(
    *,
    alpha: float = 0.5,
    temperature: float = 2.0,
    teacher_classes: int = 3,
) -> DistillationClassificationLitModule:
    return DistillationClassificationLitModule(
        net=_network(),
        teacher=_network(teacher_classes),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        loss=nn.CrossEntropyLoss(),
        num_classes=3,
        alpha=alpha,
        temperature=temperature,
    )


def test_teacher_is_frozen_and_stays_in_eval_mode() -> None:
    module = _module()

    module.train()

    assert module.training
    assert not module.teacher.training
    assert all(not parameter.requires_grad for parameter in module.teacher.parameters())


def test_distillation_step_backpropagates_only_through_student() -> None:
    module = _module()
    batch = {
        "image": torch.randn(4, 3, 8, 8),
        "label": torch.tensor([0, 1, 2, 1]),
    }

    loss, supervised, distillation, logits, predictions, metric_targets = (
        module.distillation_model_step(batch)
    )
    loss.backward()

    assert loss.ndim == supervised.ndim == distillation.ndim == 0
    assert logits.shape == (4, 3)
    assert predictions.shape == (4,)
    assert torch.equal(metric_targets, batch["label"])
    assert any(parameter.grad is not None for parameter in module.net.parameters())
    assert all(parameter.grad is None for parameter in module.teacher.parameters())


def test_distillation_supports_mixup_soft_targets() -> None:
    module = _module()
    hard_labels = torch.tensor([0, 1, 2, 1])
    soft_labels = torch.nn.functional.one_hot(hard_labels, num_classes=3).float()
    batch = {
        "image": torch.randn(4, 3, 8, 8),
        "label": soft_labels,
        "hard_label": hard_labels,
    }

    loss, _, _, _, _, metric_targets = module.distillation_model_step(batch)

    assert torch.isfinite(loss)
    assert torch.equal(metric_targets, hard_labels)


@pytest.mark.parametrize(
    ("alpha", "temperature", "message"),
    [
        (-0.1, 2.0, "alpha"),
        (1.1, 2.0, "alpha"),
        (0.5, 0.0, "temperature"),
    ],
)
def test_distillation_validates_hyperparameters(
    alpha: float,
    temperature: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _module(alpha=alpha, temperature=temperature)


def test_distillation_rejects_teacher_shape_mismatch() -> None:
    module = _module(teacher_classes=4)
    batch = {
        "image": torch.randn(2, 3, 8, 8),
        "label": torch.tensor([0, 1]),
    }

    with pytest.raises(ValueError, match="same shape"):
        module.distillation_model_step(batch)


def test_distillation_optimizer_excludes_teacher_parameters() -> None:
    module = _module()
    optimizer = module.configure_optimizers()["optimizer"]
    optimized_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }

    assert optimized_ids == {id(parameter) for parameter in module.net.parameters()}
