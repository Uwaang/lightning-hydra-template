import json
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from src.utils.saving_utils import (
    _load_checkpoint_state_dict,
    load_module_weights,
    process_state_dict,
    save_predictions,
)


def test_process_state_dict() -> None:
    state = OrderedDict(
        {
            "net.layer.weight": torch.tensor([1.0]),
            "net.layer.bias": torch.tensor([2.0]),
            "criterion.weight": torch.tensor([3.0]),
        }
    )

    processed = process_state_dict(
        state,
        strip_prefix="net.",
        exclude_prefixes=["criterion."],
    )

    assert list(processed) == ["layer.weight", "layer.bias"]


def test_load_checkpoint_state_dict_uses_safe_weights_only(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    state = OrderedDict({"net.weight": torch.tensor([1.0, 2.0])})
    torch.save({"state_dict": state, "epoch": 1}, checkpoint_path)

    loaded = _load_checkpoint_state_dict(checkpoint_path)

    assert list(loaded) == ["net.weight"]
    assert torch.equal(loaded["net.weight"], state["net.weight"])



def test_load_module_weights_skips_changed_head_shape(tmp_path: Path) -> None:
    source = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    target = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 4))
    state = OrderedDict((f"net.{key}", value) for key, value in source.state_dict().items())
    path = tmp_path / "pretrained.pt"
    torch.save(state, path)

    report = load_module_weights(
        target,
        path,
        strip_prefix="net.",
        ignore_shape_mismatch=True,
    )

    assert "0.weight" in report.loaded_keys
    assert set(report.skipped_shape_mismatch) == {"1.weight", "1.bias"}
    assert torch.equal(target[0].weight, source[0].weight)
    assert torch.equal(target[0].bias, source[0].bias)


def test_load_module_weights_accepts_lightning_checkpoint(tmp_path: Path) -> None:
    source = nn.Linear(4, 2)
    target = nn.Linear(4, 2)
    path = tmp_path / "model.ckpt"
    torch.save(
        {"state_dict": OrderedDict((f"net.{key}", value) for key, value in source.state_dict().items())},
        path,
    )

    report = load_module_weights(target, path, strip_prefix="net.")

    assert not report.missing_keys
    assert not report.unexpected_keys
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_save_predictions_json(tmp_path: Path) -> None:
    predictions = [
        {
            "preds": torch.tensor([1, 2]),
            "targets": torch.tensor([1, 0]),
        }
    ]

    paths = save_predictions(predictions, tmp_path, output_format="json")

    assert len(paths) == 1
    content = json.loads(paths[0].read_text(encoding="utf-8"))
    assert content == [
        {"preds": 1, "targets": 1},
        {"preds": 2, "targets": 0},
    ]


def test_save_predictions_nested_multihead_json(tmp_path: Path) -> None:
    predictions = [
        {
            "logits": {
                "a": torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
                "b": torch.tensor([[0.7, 0.3], [0.4, 0.6]]),
            },
            "preds": {
                "a": torch.tensor([1, 0]),
                "b": torch.tensor([0, 1]),
            },
        }
    ]

    paths = save_predictions(predictions, tmp_path, output_format="json")
    content = json.loads(paths[0].read_text(encoding="utf-8"))

    assert len(content) == 2
    assert content[0]["preds"] == {"a": 1, "b": 0}
    assert content[1]["preds"] == {"a": 0, "b": 1}
    assert len(content[0]["logits"]["a"]) == 2


def test_save_predictions_multiple_dataloaders(tmp_path: Path) -> None:
    predictions = [
        [{"preds": torch.tensor([1])}],
        [{"preds": torch.tensor([2])}],
    ]

    paths = save_predictions(predictions, tmp_path, output_format="csv")

    assert [path.name for path in paths] == ["predictions_0.csv", "predictions_1.csv"]
    assert all(path.exists() for path in paths)
