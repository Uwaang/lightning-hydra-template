from __future__ import annotations

import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn


@dataclass(frozen=True)
class ModelComplexityReport:
    """Serializable model-complexity summary for one representative forward pass."""

    model_class: str
    input_shapes: list[list[int]]
    total_params: int
    trainable_params: int
    state_dict_bytes: int
    macs: int
    flops: int
    macs_counted_ops: tuple[str, ...]
    peak_cuda_memory_bytes: int | None


@dataclass(frozen=True)
class ParetoObjective:
    """One optimization direction used when constructing a Pareto frontier."""

    key: str
    direction: Literal["min", "max"]


def _state_dict_serialized_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def _conv_macs(module: nn.Module, output: Any) -> int:
    if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return 0
    if not isinstance(output, torch.Tensor):
        return 0
    kernel_ops = math.prod(module.kernel_size) * (module.in_channels // module.groups)
    return int(output.numel() * kernel_ops)


def _linear_macs(module: nn.Module, output: Any) -> int:
    if not isinstance(module, nn.Linear):
        return 0
    if not isinstance(output, torch.Tensor):
        return 0
    return int(output.numel() * module.in_features)


def profile_model_complexity(
    model: nn.Module,
    example_args: tuple[Any, ...],
) -> ModelComplexityReport:
    """Profile parameters, size, standard-layer MACs/FLOPs and CUDA peak memory.

    MAC counting covers Conv1d/2d/3d and Linear modules. FLOPs use the explicit convention
    2 * MACs. Functional/custom operators are intentionally not guessed, so models that rely
    heavily on them should treat the MAC/FLOP fields as partial coverage.
    """
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    state_dict_bytes = _state_dict_serialized_bytes(model)
    input_shapes = [
        list(value.shape) for value in example_args if isinstance(value, torch.Tensor)
    ]

    macs = 0
    handles: list[Any] = []

    def count_conv(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal macs
        macs += _conv_macs(module, output)

    def count_linear(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal macs
        macs += _linear_macs(module, output)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(count_conv))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(count_linear))

    cuda_devices = {
        value.device
        for value in example_args
        if isinstance(value, torch.Tensor) and value.device.type == "cuda"
    }
    peak_cuda_memory_bytes: int | None = None
    if cuda_devices:
        if len(cuda_devices) != 1:
            raise ValueError("Complexity profiling expects CUDA inputs on one device.")
        cuda_device = next(iter(cuda_devices))
        torch.cuda.reset_peak_memory_stats(cuda_device)
        baseline_cuda_memory = torch.cuda.memory_allocated(cuda_device)
    else:
        cuda_device = None
        baseline_cuda_memory = 0

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            model(*example_args)
        if cuda_device is not None:
            peak_cuda_memory_bytes = max(
                0,
                int(torch.cuda.max_memory_allocated(cuda_device) - baseline_cuda_memory),
            )
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return ModelComplexityReport(
        model_class=f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        input_shapes=input_shapes,
        total_params=int(total_params),
        trainable_params=int(trainable_params),
        state_dict_bytes=int(state_dict_bytes),
        macs=int(macs),
        flops=int(2 * macs),
        macs_counted_ops=("Conv1d", "Conv2d", "Conv3d", "Linear"),
        peak_cuda_memory_bytes=peak_cuda_memory_bytes,
    )


def save_model_complexity_report(
    report: ModelComplexityReport,
    output_path: str | Path,
) -> Path:
    """Save a model-complexity report as stable JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _objective_value(record: Mapping[str, Any], objective: ParetoObjective) -> float:
    if objective.key not in record:
        raise KeyError(f"Pareto record is missing objective {objective.key!r}.")
    value = float(record[objective.key])
    if not math.isfinite(value):
        raise ValueError(f"Pareto objective {objective.key!r} must be finite.")
    return value


def _dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    objectives: Sequence[ParetoObjective],
) -> bool:
    no_worse = True
    strictly_better = False

    for objective in objectives:
        left_value = _objective_value(left, objective)
        right_value = _objective_value(right, objective)

        if objective.direction == "min":
            no_worse &= left_value <= right_value
            strictly_better |= left_value < right_value
        else:
            no_worse &= left_value >= right_value
            strictly_better |= left_value > right_value

        if not no_worse:
            return False

    return strictly_better


def pareto_front(
    records: Sequence[Mapping[str, Any]],
    objectives: Sequence[ParetoObjective],
) -> list[Mapping[str, Any]]:
    """Return nondominated records while preserving original record order."""
    if not objectives:
        raise ValueError("At least one Pareto objective is required.")

    frontier: list[Mapping[str, Any]] = []
    for index, candidate in enumerate(records):
        dominated = any(
            _dominates(other, candidate, objectives)
            for other_index, other in enumerate(records)
            if other_index != index
        )
        if not dominated:
            frontier.append(candidate)
    return frontier
