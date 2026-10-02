from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from src.data.cifar10_datamodule import CIFAR10DataModule
from src.models.components.adapters import TorchvisionClassifier
from src.utils.benchmark_utils import benchmark_onnx
from src.utils.export_utils import export_onnx
from src.utils.quantization_utils import (
    quantize_onnx_static,
    save_quantization_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 ResNet-18 ONNX Runtime FP32 vs static-INT8 spike."
    )
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cifar10-ort-spike"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--data-batch-size", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--iterations-per-repeat", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def _load_model(state_dict_path: Path) -> TorchvisionClassifier:
    model = TorchvisionClassifier(
        model_name="resnet18",
        num_classes=10,
        weights=None,
    )
    state_dict = torch.load(
        state_dict_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_datamodule(data_dir: Path, batch_size: int) -> CIFAR10DataModule:
    datamodule = CIFAR10DataModule(
        data_dir=str(data_dir),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def _input_name(model_path: Path) -> str:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return session.get_inputs()[0].name


def _calibration_samples(
    datamodule: CIFAR10DataModule,
    input_name: str,
    max_batches: int,
) -> list[dict[str, torch.Tensor]]:
    samples: list[dict[str, torch.Tensor]] = []
    for batch_index, (images, _) in enumerate(datamodule.val_dataloader()):
        if batch_index >= max_batches:
            break
        samples.append({input_name: images})
    if not samples:
        raise RuntimeError("No calibration batches were produced.")
    return samples


def _accuracy(
    model_path: Path,
    datamodule: CIFAR10DataModule,
) -> tuple[float, float]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0
    started = perf_counter()

    for images, targets in datamodule.test_dataloader():
        logits = session.run(None, {input_name: images.numpy()})[0]
        predictions = torch.from_numpy(logits).argmax(dim=1)
        correct += int((predictions == targets).sum())
        total += int(targets.numel())

    return correct / total, perf_counter() - started


def _stats_payload(stats: Any) -> dict[str, Any]:
    payload = asdict(stats)
    payload.pop("samples_ms", None)
    return payload


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = args.output_dir / "resnet18-cifar10-fp32.onnx"
    int8_path = args.output_dir / "resnet18-cifar10-int8.onnx"

    model = _load_model(args.state_dict)
    datamodule = _build_datamodule(args.data_dir, args.data_batch_size)

    latency_image, _ = datamodule.test_dataloader().dataset[0]
    latency_input = latency_image.unsqueeze(0)
    batch_dim = torch.export.Dim("batch")
    export_onnx(
        model,
        (latency_input,),
        fp32_path,
        dynamic_shapes=({0: batch_dim},),
        verify=True,
    )

    input_name = _input_name(fp32_path)
    calibration = _calibration_samples(
        datamodule,
        input_name,
        args.calibration_batches,
    )
    quant_report = quantize_onnx_static(
        fp32_path,
        int8_path,
        calibration,
        per_channel=True,
    )
    save_quantization_report(
        quant_report,
        args.output_dir / "quantization.json",
    )

    benchmark_kwargs = {
        "provider": "cpu",
        "batch_size": 1,
        "warmup_iterations": args.warmup_iterations,
        "iterations_per_repeat": args.iterations_per_repeat,
        "repeats": args.repeats,
        "num_threads": args.num_threads,
    }
    fp32_stats = benchmark_onnx(
        fp32_path,
        (latency_input,),
        **benchmark_kwargs,
    )
    int8_stats = benchmark_onnx(
        int8_path,
        (latency_input,),
        **benchmark_kwargs,
    )

    fp32_accuracy, fp32_eval_seconds = _accuracy(fp32_path, datamodule)
    int8_accuracy, int8_eval_seconds = _accuracy(int8_path, datamodule)

    results = {
        "model": "torchvision/resnet18",
        "dataset": "CIFAR-10",
        "state_dict": str(args.state_dict),
        "calibration_batches": len(calibration),
        "calibration_samples": sum(int(sample[input_name].shape[0]) for sample in calibration),
        "num_threads": args.num_threads,
        "fp32": {
            "accuracy": fp32_accuracy,
            "evaluation_seconds": fp32_eval_seconds,
            "model_size_bytes": fp32_path.stat().st_size,
            "benchmark": _stats_payload(fp32_stats),
        },
        "int8": {
            "accuracy": int8_accuracy,
            "evaluation_seconds": int8_eval_seconds,
            "model_size_bytes": int8_path.stat().st_size,
            "benchmark": _stats_payload(int8_stats),
        },
        "comparison": {
            "accuracy_delta": int8_accuracy - fp32_accuracy,
            "size_ratio": int8_path.stat().st_size / fp32_path.stat().st_size,
            "median_latency_speedup": fp32_stats.median_ms / int8_stats.median_ms,
        },
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
