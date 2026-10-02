import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.utils.benchmark_utils import (
    available_onnxruntime_providers,
    benchmark_callable,
    benchmark_onnx,
    benchmark_runtime_stack,
    save_benchmark_report,
)
from src.utils.export_utils import export_onnx, export_pt2


class TinyBenchmarkModel(nn.Module):
    """Small tensor-only model for runtime benchmark coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def test_benchmark_callable_reports_latency_distribution() -> None:
    value = torch.randn(8, 8)

    stats = benchmark_callable(
        lambda: value @ value,
        backend="test",
        batch_size=8,
        warmup_iterations=1,
        iterations_per_repeat=2,
        repeats=3,
        num_threads=1,
    )

    assert stats.backend == "test"
    assert stats.mean_ms > 0
    assert stats.min_ms <= stats.median_ms <= stats.max_ms
    assert stats.p90_ms >= stats.median_ms
    assert stats.p95_ms >= stats.p90_ms
    assert stats.throughput_items_per_second > 0
    assert len(stats.samples_ms) == 3


def test_runtime_stack_pt2_and_json_report(tmp_path: Path) -> None:
    model = TinyBenchmarkModel()
    model.train()
    inputs = (torch.randn(2, 4),)
    pt2_path = tmp_path / "tiny.pt2"
    report_path = tmp_path / "benchmark.json"

    export_pt2(model, inputs, pt2_path)
    report = benchmark_runtime_stack(
        model,
        inputs,
        pt2_path=pt2_path,
        batch_size=2,
        warmup_iterations=1,
        iterations_per_repeat=2,
        repeats=3,
        num_threads=1,
    )
    saved_path = save_benchmark_report(report, report_path)

    assert model.training
    assert [result.backend for result in report.results] == [
        "pytorch_eager",
        "pytorch_pt2_graph",
    ]
    assert report.input_shapes == [[2, 4]]
    assert report.model_class.endswith(".TinyBenchmarkModel")
    assert saved_path == report_path

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["device"] == "cpu"
    assert len(payload["results"]) == 2


def test_runtime_stack_with_onnxruntime(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    model = TinyBenchmarkModel()
    inputs = (torch.randn(2, 4),)
    pt2_path = tmp_path / "tiny.pt2"
    onnx_path = tmp_path / "tiny.onnx"

    export_pt2(model, inputs, pt2_path)
    export_onnx(model, inputs, onnx_path, verify=True)
    report = benchmark_runtime_stack(
        model,
        inputs,
        pt2_path=pt2_path,
        onnx_path=onnx_path,
        batch_size=2,
        warmup_iterations=1,
        iterations_per_repeat=2,
        repeats=3,
        num_threads=1,
    )

    assert [result.backend for result in report.results] == [
        "pytorch_eager",
        "pytorch_pt2_graph",
        "onnxruntime_cpu",
    ]
    assert report.environment["onnxruntime"] != "not-installed"


def test_runtime_stack_rejects_cuda_inputs_when_available(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    model = TinyBenchmarkModel().cuda()
    inputs = (torch.randn(2, 4, device="cuda"),)

    with pytest.raises(ValueError, match="CPU host tensors"):
        benchmark_runtime_stack(model, inputs, pt2_path=tmp_path / "unused.pt2")


def test_available_onnxruntime_providers_reports_cpu() -> None:
    pytest.importorskip("onnxruntime")
    assert "CPUExecutionProvider" in available_onnxruntime_providers()


def test_onnx_provider_fails_when_primary_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

    with pytest.raises(RuntimeError, match="XnnpackExecutionProvider is not available"):
        benchmark_onnx(
            "unused.onnx",
            (torch.randn(1, 4),),
            provider="xnnpack",
        )


def test_onnx_provider_rejects_runtime_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    class CpuFallbackSession:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(ort, "InferenceSession", CpuFallbackSession)

    with pytest.raises(RuntimeError, match="requested but is not active"):
        benchmark_onnx(
            "unused.onnx",
            (torch.randn(1, 4),),
            provider="cuda",
        )
