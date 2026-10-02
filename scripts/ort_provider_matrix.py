from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.utils.benchmark_utils import (
    available_onnxruntime_providers,
    benchmark_onnx,
)


def _parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be comma-separated integers") from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one ONNX model across selected ONNX Runtime execution providers."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 32, 32))
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["cpu", "xnnpack", "cuda", "tensorrt"],
    )
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--iterations-per-repeat", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _stats_payload(stats: Any) -> dict[str, Any]:
    payload = asdict(stats)
    payload.pop("samples_ms", None)
    return payload


def main() -> None:
    args = _parse_args()
    sample = torch.randn(*args.input_shape)
    available = available_onnxruntime_providers()
    results: dict[str, Any] = {}

    for provider in args.providers:
        try:
            stats = benchmark_onnx(
                args.model,
                (sample,),
                provider=provider,
                batch_size=args.input_shape[0],
                warmup_iterations=args.warmup_iterations,
                iterations_per_repeat=args.iterations_per_repeat,
                repeats=args.repeats,
                num_threads=args.num_threads,
            )
        except (RuntimeError, ValueError) as exc:
            results[provider] = {
                "status": "unavailable",
                "error": str(exc),
            }
        except Exception as exc:
            results[provider] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            results[provider] = {
                "status": "ok",
                "benchmark": _stats_payload(stats),
            }

    payload = {
        "model": str(args.model),
        "input_shape": list(args.input_shape),
        "available_onnxruntime_providers": available,
        "num_threads": args.num_threads,
        "results": results,
    }

    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
