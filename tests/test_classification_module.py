from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.models.classification_module import ClassificationLitModule


def _net() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 3),
    )


def _module() -> ClassificationLitModule:
    return ClassificationLitModule(
        net=_net(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        loss=nn.CrossEntropyLoss(),
        num_classes=3,
    )


def test_model_step_dict_batch() -> None:
    module = _module()
    batch = {
        "image": torch.randn(4, 3, 8, 8),
        "label": torch.tensor([0, 1, 2, 1]),
    }

    loss, logits, preds, targets = module.model_step(batch)

    assert loss.ndim == 0
    assert logits.shape == (4, 3)
    assert preds.shape == (4,)
    assert torch.equal(targets, batch["label"])


def test_model_step_tuple_batch() -> None:
    module = _module()
    batch = (
        torch.randn(4, 3, 8, 8),
        torch.tensor([0, 1, 2, 1]),
    )

    _, logits, _, _ = module.model_step(batch)
    assert logits.shape == (4, 3)


def test_model_step_soft_targets_uses_hard_labels_for_metrics() -> None:
    module = _module()
    hard_labels = torch.tensor([0, 1, 2, 1])
    soft_labels = torch.nn.functional.one_hot(hard_labels, num_classes=3).float()
    soft_labels = 0.8 * soft_labels + 0.2 / 3
    batch = {
        "image": torch.randn(4, 3, 8, 8),
        "label": soft_labels,
        "hard_label": hard_labels,
    }

    loss, logits, preds, metric_targets = module.model_step(batch)

    assert loss.ndim == 0
    assert logits.shape == (4, 3)
    assert preds.shape == (4,)
    assert torch.equal(metric_targets, hard_labels)


def test_predict_step_unlabeled_batch() -> None:
    module = _module()
    batch = {
        "image": torch.randn(2, 3, 8, 8),
        "name": ["a.jpg", "b.jpg"],
    }

    output = module.predict_step(batch, batch_idx=0)

    assert output["logits"].shape == (2, 3)
    assert output["preds"].shape == (2,)
    assert output["names"] == ["a.jpg", "b.jpg"]


def test_configure_optimizer_without_scheduler() -> None:
    module = _module()
    configured = module.configure_optimizers()

    assert isinstance(configured["optimizer"], torch.optim.Adam)


def test_checkpoint_hyperparameters_are_weights_only_safe(tmp_path) -> None:
    module = _module()
    checkpoint_path = tmp_path / "classification_hparams.ckpt"
    torch.save({"hyper_parameters": dict(module.hparams)}, checkpoint_path)

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert loaded["hyper_parameters"] == {
        "num_classes": 3,
        "teacher_state_dict_path": None,
        "distillation_alpha": 0.0,
        "distillation_temperature": 4.0,
        "compile": False,
    }


def test_logit_distillation_applies_only_to_training_loss() -> None:
    teacher = _net()
    module = ClassificationLitModule(
        net=_net(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        loss=nn.CrossEntropyLoss(),
        num_classes=3,
        teacher=teacher,
        distillation_alpha=0.4,
        distillation_temperature=2.0,
    )
    module.train()

    batch = (
        torch.randn(4, 3, 8, 8),
        torch.tensor([0, 1, 2, 1]),
    )
    x, targets = batch
    student_logits = module(x)
    with torch.no_grad():
        teacher_logits = teacher(x)

    hard_loss = F.cross_entropy(student_logits, targets)
    temperature = 2.0
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
    expected = 0.6 * hard_loss + 0.4 * soft_loss

    train_loss, _, _, _ = module.model_step(batch, distill=True)
    eval_loss, _, _, _ = module.model_step(batch)

    assert torch.allclose(train_loss, expected)
    assert torch.allclose(eval_loss, hard_loss)
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_distillation_optimizer_excludes_teacher_parameters() -> None:
    teacher = _net()
    module = ClassificationLitModule(
        net=_net(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        loss=nn.CrossEntropyLoss(),
        num_classes=3,
        teacher=teacher,
        distillation_alpha=0.5,
    )

    optimizer = module.configure_optimizers()["optimizer"]
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    teacher_ids = {id(parameter) for parameter in teacher.parameters()}
    student_ids = {id(parameter) for parameter in module.net.parameters()}

    assert teacher_ids.isdisjoint(optimized_ids)
    assert student_ids <= optimized_ids


def test_teacher_plain_state_dict_can_be_loaded(tmp_path: Path) -> None:
    source_teacher = _net()
    with torch.no_grad():
        for parameter in source_teacher.parameters():
            parameter.fill_(0.25)
    teacher_path = tmp_path / "teacher_state_dict.pt"
    torch.save(source_teacher.state_dict(), teacher_path)

    teacher = _net()
    module = ClassificationLitModule(
        net=_net(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        loss=nn.CrossEntropyLoss(),
        num_classes=3,
        teacher=teacher,
        teacher_state_dict_path=str(teacher_path),
        distillation_alpha=0.5,
    )

    for loaded, expected in zip(
        module.teacher.parameters(),  # type: ignore[union-attr]
        source_teacher.parameters(),
        strict=True,
    ):
        assert torch.equal(loaded, expected)


def test_distillation_requires_teacher_for_positive_alpha() -> None:
    try:
        ClassificationLitModule(
            net=_net(),
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            scheduler=None,
            loss=nn.CrossEntropyLoss(),
            num_classes=3,
            distillation_alpha=0.5,
        )
    except ValueError as exc:
        assert "teacher" in str(exc).lower()
    else:
        raise AssertionError("positive distillation alpha should require a teacher")
