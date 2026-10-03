import pytest
import torch

from src.utils import get_metric_value, get_metric_values


def test_get_metric_value_accepts_tensor_and_plain_scalar() -> None:
    assert get_metric_value({"tensor": torch.tensor(1.25)}, "tensor") == pytest.approx(1.25)
    assert get_metric_value({"plain": 42}, "plain") == 42.0


def test_get_metric_values_preserves_objective_order() -> None:
    metrics = {
        "accuracy": torch.tensor(0.9),
        "params": 1234,
    }

    values = get_metric_values(metrics, ["accuracy", "params"])

    assert values == pytest.approx((0.9, 1234.0))


def test_get_metric_values_requires_at_least_one_objective() -> None:
    with pytest.raises(ValueError, match="At least one optimized metric"):
        get_metric_values({"accuracy": 0.9}, [])


def test_get_metric_value_rejects_non_scalar_value() -> None:
    with pytest.raises((TypeError, RuntimeError, ValueError)):
        get_metric_value({"bad": torch.tensor([1.0, 2.0])}, "bad")
