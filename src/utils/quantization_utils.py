from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch


@dataclass(frozen=True)
class QuantizationReport:
    """Serializable metadata for one static ONNX quantization run."""

    input_path: str
    output_path: str
    input_size_bytes: int
    output_size_bytes: int
    quant_format: str
    activation_type: str
    weight_type: str
    per_channel: bool
    calibration_method: str
    calibration_batches: int
    preprocessed: bool

    @property
    def size_ratio(self) -> float:
        return self.output_size_bytes / self.input_size_bytes


class StaticCalibrationDataReader:
    """Bounded in-memory calibration reader compatible with ONNX Runtime."""

    def __init__(self, samples: Sequence[Mapping[str, Any]]) -> None:
        if not samples:
            raise ValueError("Static quantization requires at least one calibration batch.")
        self._samples = [self._normalize(sample) for sample in samples]
        self.rewind()

    @staticmethod
    def _normalize(sample: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for name, value in sample.items():
            if isinstance(value, torch.Tensor):
                normalized[name] = value.detach().cpu().numpy()
            else:
                normalized[name] = value
        return normalized

    def get_next(self) -> dict[str, Any] | None:
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = iter(self._samples)


def quantize_onnx_static(
    input_path: str | Path,
    output_path: str | Path,
    calibration_samples: Sequence[Mapping[str, Any]],
    *,
    per_channel: bool = True,
    calibration_method: str = "minmax",
    preprocess: bool = True,
) -> QuantizationReport:
    """Create a QDQ S8/S8 static-PTQ ONNX artifact with ONNX Runtime."""
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as exc:
        raise ImportError(
            "ONNX quantization requires the optional ONNX stack. "
            "Install it with `uv sync --extra onnx`."
        ) from exc

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    if calibration_method not in methods:
        supported = ", ".join(sorted(methods))
        raise ValueError(
            f"Unknown calibration method {calibration_method!r}; expected one of: {supported}."
        )

    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    reader = StaticCalibrationDataReader(calibration_samples)

    with TemporaryDirectory(prefix="ort-quant-") as temp_dir:
        quantization_source = source
        if preprocess:
            quantization_source = Path(temp_dir) / "preprocessed.onnx"
            quant_pre_process(
                source,
                quantization_source,
            )

        quantize_static(
            quantization_source,
            target,
            reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            calibrate_method=methods[calibration_method],
        )

    return QuantizationReport(
        input_path=str(source),
        output_path=str(target),
        input_size_bytes=source.stat().st_size,
        output_size_bytes=target.stat().st_size,
        quant_format="QDQ",
        activation_type="QInt8",
        weight_type="QInt8",
        per_channel=per_channel,
        calibration_method=calibration_method,
        calibration_batches=len(calibration_samples),
        preprocessed=preprocess,
    )


def save_quantization_report(
    report: QuantizationReport,
    output_path: str | Path,
) -> Path:
    """Save quantization metadata as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["size_ratio"] = report.size_ratio
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
