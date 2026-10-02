from functools import partial

import torch
from torch import nn

from src.models.classification_module import ClassificationLitModule


def _module() -> ClassificationLitModule:
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 3),
    )
    return ClassificationLitModule(
        net=net,
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

    assert loaded["hyper_parameters"] == {"num_classes": 3, "compile": False}
