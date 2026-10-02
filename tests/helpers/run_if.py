"""Minimal conditional-test helper for optional project test paths."""

from typing import Any

import pytest
import torch
from pytest import MarkDecorator

from tests.helpers.package_available import (
    _OPTUNA_SWEEPER_AVAILABLE,
    _SH_AVAILABLE,
)


class RunIf:
    """Create skip markers for runtime requirements used by this repository."""

    def __new__(
        cls,
        min_gpus: int = 0,
        sh: bool = False,
        optuna_sweeper: bool = False,
        **kwargs: dict[Any, Any],
    ) -> MarkDecorator:
        conditions: list[bool] = []
        reasons: list[str] = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if sh:
            conditions.append(not _SH_AVAILABLE)
            reasons.append("sh")

        if optuna_sweeper:
            conditions.append(not _OPTUNA_SWEEPER_AVAILABLE)
            reasons.append("hydra-optuna-sweeper")

        missing = [
            reason for condition, reason in zip(conditions, reasons, strict=True) if condition
        ]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(missing)}]",
            **kwargs,
        )
