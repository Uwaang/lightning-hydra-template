import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.utils.model_profiling import (
    ParetoObjective,
    pareto_front,
    profile_model_complexity,
    save_model_complexity_report,
)


class TinyProfileModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_profile_model_complexity_counts_standard_layers_and_restores_mode() -> None:
    model = TinyProfileModel()
    model.train()
    example = (torch.randn(2, 3, 8, 8),)

    report = profile_model_complexity(model, example)

    assert model.training
    assert report.input_shapes == [[2, 3, 8, 8]]
    assert report.total_params == 118
    assert report.trainable_params == 118
    assert report.macs == 13_840
    assert report.flops == 27_680
    assert report.state_dict_bytes > 118 * 4
    assert report.peak_cuda_memory_bytes is None


def test_profile_model_complexity_is_repeatable_without_duplicate_hooks() -> None:
    model = TinyProfileModel()
    example = (torch.randn(1, 3, 8, 8),)

    first = profile_model_complexity(model, example)
    second = profile_model_complexity(model, example)

    assert first.macs == second.macs == 6_920


def test_save_model_complexity_report(tmp_path: Path) -> None:
    report = profile_model_complexity(TinyProfileModel(), (torch.randn(1, 3, 8, 8),))
    path = save_model_complexity_report(report, tmp_path / "complexity.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total_params"] == 118
    assert payload["macs"] == 6_920
    assert payload["flops"] == 13_840


def test_pareto_front_mixed_min_max_objectives() -> None:
    records = [
        {"name": "a", "accuracy": 0.90, "params": 100},
        {"name": "b", "accuracy": 0.91, "params": 120},
        {"name": "c", "accuracy": 0.88, "params": 150},
        {"name": "d", "accuracy": 0.89, "params": 90},
    ]
    objectives = [
        ParetoObjective("accuracy", "max"),
        ParetoObjective("params", "min"),
    ]

    frontier = pareto_front(records, objectives)

    assert [record["name"] for record in frontier] == ["a", "b", "d"]


def test_pareto_front_requires_objectives() -> None:
    with pytest.raises(ValueError, match="At least one Pareto objective"):
        pareto_front([{"accuracy": 1.0}], [])


def test_pareto_front_rejects_missing_metric() -> None:
    with pytest.raises(KeyError, match="params"):
        pareto_front(
            [{"accuracy": 1.0}, {"accuracy": 0.9, "params": 2}],
            [ParetoObjective("params", "min")],
        )


def test_pareto_objective_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        ParetoObjective("accuracy", "sideways")  # type: ignore[arg-type]
