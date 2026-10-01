from __future__ import annotations

import json
import platform
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class BenchmarkStats:
    """Latency distribution for one inference backend."""

    backend: str
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    throughput_items_per_second: float
    batch_size: int
    warmup_iterations: int
    iterations_per_repeat: int
    repeats: int
    num_threads: int
    samples_ms: list[float]


@dataclass(frozen=True)
class BenchmarkReport:
    """Serializable benchmark report for equivalent inference paths."""

    schema_version: int
    created_at_utc: str
    device: str
    model_class: str
    input_shapes: list[list[int]]
    environment: dict[str, str]
    results: list[BenchmarkStats]


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("Cannot compute a percentile from an empty sample.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1.")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    backend: str,
    batch_size: int = 1,
    warmup_iterations: int = 10,
    iterations_per_repeat: int = 50,
    repeats: int = 20,
    num_threads: int = 1,
) -> BenchmarkStats:
    """Measure repeated CPU inference latency without setup or export time."""
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be zero or greater.")

    for name, value in {
        "batch_size": batch_size,
        "iterations_per_repeat": iterations_per_repeat,
        "repeats": repeats,
        "num_threads": num_threads,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")

    previous_threads = torch.get_num_threads()
    samples_ms: list[float] = []

    try:
        torch.set_num_threads(num_threads)
        with torch.inference_mode():
            for _ in range(warmup_iterations):
                fn()

            for _ in range(repeats):
                started = perf_counter_ns()
                for _ in range(iterations_per_repeat):
                    fn()
                elapsed_ns = perf_counter_ns() - started
                samples_ms.append(elapsed_ns / 1_000_000 / iterations_per_repeat)
    finally:
        torch.set_num_threads(previous_threads)

    median_ms = statistics.median(samples_ms)
    return BenchmarkStats(
        backend=backend,
        mean_ms=statistics.mean(samples_ms),
        median_ms=median_ms,
        p90_ms=_percentile(samples_ms, 0.90),
        p95_ms=_percentile(samples_ms, 0.95),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
        throughput_items_per_second=batch_size * 1000.0 / median_ms,
        batch_size=batch_size,
        warmup_iterations=warmup_iterations,
        iterations_per_repeat=iterations_per_repeat,
        repeats=repeats,
        num_threads=num_threads,
        samples_ms=samples_ms,
    )


def _require_cpu_tensors(args: tuple[Any, ...]) -> None:
    """Reject accelerator inputs until all compared backends share one device path."""
    for value in args:
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            raise ValueError(
                "Runtime stack benchmarks are CPU-only for now. "
                "GPU timing will be added with an ONNX Runtime GPU provider path."
            )


def benchmark_eager(
    model: nn.Module,
    example_args: tuple[Any, ...],
    *,
    batch_size: int = 1,
    warmup_iterations: int = 10,
    iterations_per_repeat: int = 50,
    repeats: int = 20,
    num_threads: int = 1,
) -> BenchmarkStats:
    """Benchmark a plain PyTorch module in eager inference mode on CPU."""
    _require_cpu_tensors(example_args)
    was_training = model.training
    model.eval()
    try:
        return benchmark_callable(
            lambda: model(*example_args),
            backend="pytorch_eager",
            batch_size=batch_size,
            warmup_iterations=warmup_iterations,
            iterations_per_repeat=iterations_per_repeat,
            repeats=repeats,
            num_threads=num_threads,
        )
    finally:
        model.train(was_training)


def benchmark_pt2(
    artifact_path: str | Path,
    example_args: tuple[Any, ...],
    *,
    batch_size: int = 1,
    warmup_iterations: int = 10,
    iterations_per_repeat: int = 50,
    repeats: int = 20,
    num_threads: int = 1,
) -> BenchmarkStats:
    """Benchmark a loaded PT2 ExportedProgram graph on CPU."""
    _require_cpu_tensors(example_args)
    module = torch.export.load(Path(artifact_path)).module()
    return benchmark_callable(
        lambda: module(*example_args),
        backend="pytorch_pt2_graph",
        batch_size=batch_size,
        warmup_iterations=warmup_iterations,
        iterations_per_repeat=iterations_per_repeat,
        repeats=repeats,
        num_threads=num_threads,
    )


def benchmark_onnx(
    artifact_path: str | Path,
    example_args: tuple[Any, ...],
    *,
    batch_size: int = 1,
    warmup_iterations: int = 10,
    iterations_per_repeat: int = 50,
    repeats: int = 20,
    num_threads: int = 1,
) -> BenchmarkStats:
    """Benchmark an ONNX artifact with ONNX Runtime's CPU execution provider."""
    _require_cpu_tensors(example_args)

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "ONNX Runtime benchmarking requires the optional ONNX stack. "
            "Install it with `uv sync --extra onnx`."
        ) from exc

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = num_threads
    session = ort.InferenceSession(
        str(artifact_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    if len(inputs) != len(example_args):
        raise ValueError(
            f"ONNX model expects {len(inputs)} positional inputs, "
            f"but {len(example_args)} example inputs were provided."
        )

    feed: dict[str, Any] = {}
    for input_metadata, value in zip(inputs, example_args, strict=True):
        if not isinstance(value, torch.Tensor):
            raise TypeError("ONNX benchmarking currently supports positional tensor inputs only.")
        feed[input_metadata.name] = value.detach().cpu().numpy()

    return benchmark_callable(
        lambda: session.run(None, feed),
        backend="onnxruntime_cpu",
        batch_size=batch_size,
        warmup_iterations=warmup_iterations,
        iterations_per_repeat=iterations_per_repeat,
        repeats=repeats,
        num_threads=num_threads,
    )


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def benchmark_runtime_stack(
    model: nn.Module,
    example_args: tuple[Any, ...],
    *,
    pt2_path: str | Path,
    onnx_path: str | Path | None = None,
    batch_size: int = 1,
    warmup_iterations: int = 10,
    iterations_per_repeat: int = 50,
    repeats: int = 20,
    num_threads: int = 1,
) -> BenchmarkReport:
    """Benchmark eager, PT2 graph, and optional ONNX Runtime inference paths."""
    _require_cpu_tensors(example_args)

    common = {
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "iterations_per_repeat": iterations_per_repeat,
        "repeats": repeats,
        "num_threads": num_threads,
    }
    results = [
        benchmark_eager(model, example_args, **common),
        benchmark_pt2(pt2_path, example_args, **common),
    ]
    if onnx_path is not None:
        results.append(benchmark_onnx(onnx_path, example_args, **common))

    input_shapes = [
        list(value.shape) for value in example_args if isinstance(value, torch.Tensor)
    ]
    return BenchmarkReport(
        schema_version=1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        device="cpu",
        model_class=f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        input_shapes=input_shapes,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "logical_cpu_count": str(__import__("os").cpu_count() or "unknown"),
            "torch": str(torch.__version__),
            "onnxruntime": _package_version("onnxruntime"),
        },
        results=results,
    )


def save_benchmark_report(report: BenchmarkReport, output_path: str | Path) -> Path:
    """Serialize a benchmark report as stable, human-readable JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
