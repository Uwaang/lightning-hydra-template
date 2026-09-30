"""Adapted from the PyTorch Lightning test helper RunIf pattern."""

import sys
from importlib.metadata import version
from typing import Any, Dict, Optional

import pytest
import torch
from packaging.version import Version
from pytest import MarkDecorator

from tests.helpers.package_available import (
    _COMET_AVAILABLE,
    _DEEPSPEED_AVAILABLE,
    _FAIRSCALE_AVAILABLE,
    _IS_WINDOWS,
    _MLFLOW_AVAILABLE,
    _NEPTUNE_AVAILABLE,
    _OPTUNA_SWEEPER_AVAILABLE,
    _SH_AVAILABLE,
    _TPU_AVAILABLE,
    _WANDB_AVAILABLE,
)


class RunIf:
    """RunIf wrapper for conditional skipping of tests."""

    def __new__(
        cls,
        min_gpus: int = 0,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        skip_windows: bool = False,
        sh: bool = False,
        tpu: bool = False,
        fairscale: bool = False,
        deepspeed: bool = False,
        wandb: bool = False,
        neptune: bool = False,
        comet: bool = False,
        mlflow: bool = False,
        optuna_sweeper: bool = False,
        **kwargs: Dict[Any, Any],
    ) -> MarkDecorator:
        """Create a new pytest skip marker from runtime requirements."""
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if min_torch:
            torch_version = version("torch")
            conditions.append(Version(torch_version) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            torch_version = version("torch")
            conditions.append(Version(torch_version) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if skip_windows:
            conditions.append(_IS_WINDOWS)
            reasons.append("does not run on Windows")

        if tpu:
            conditions.append(not _TPU_AVAILABLE)
            reasons.append("TPU")

        if sh:
            conditions.append(not _SH_AVAILABLE)
            reasons.append("sh")

        if fairscale:
            conditions.append(not _FAIRSCALE_AVAILABLE)
            reasons.append("fairscale")

        if deepspeed:
            conditions.append(not _DEEPSPEED_AVAILABLE)
            reasons.append("deepspeed")

        if wandb:
            conditions.append(not _WANDB_AVAILABLE)
            reasons.append("wandb")

        if neptune:
            conditions.append(not _NEPTUNE_AVAILABLE)
            reasons.append("neptune")

        if comet:
            conditions.append(not _COMET_AVAILABLE)
            reasons.append("comet")

        if mlflow:
            conditions.append(not _MLFLOW_AVAILABLE)
            reasons.append("mlflow")

        if optuna_sweeper:
            conditions.append(not _OPTUNA_SWEEPER_AVAILABLE)
            reasons.append("hydra-optuna-sweeper")

        reasons = [reason for condition, reason in zip(conditions, reasons) if condition]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )
