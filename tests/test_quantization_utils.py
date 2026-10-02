import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.utils.export_utils import export_onnx
from src.utils.quantization_utils import (
    StaticCalibrationDataReader,
    quantize_onnx_static,
    save_quantization_report,
)


class TinyQuantModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def test_calibration_reader_rewinds() -> None:
    reader = StaticCalibrationDataReader([{"x": torch.ones(2, 4)}, {"x": torch.zeros(2, 4)}])
    first = reader.get_next()
    second = reader.get_next()
    assert first is not None
    assert second is not None
    assert reader.get_next() is None

    reader.rewind()
    rewound = reader.get_next()
    assert rewound is not None
    assert rewound["x"].shape == (2, 4)


def test_static_qdq_quantization_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    model = TinyQuantModel().eval()
    example = torch.randn(2, 4)
    fp32_path = tmp_path / "tiny.onnx"
    int8_path = tmp_path / "tiny.int8.onnx"
    report_path = tmp_path / "quantization.json"

    export_onnx(model, (example,), fp32_path, verify=True)
    fp32_session = ort.InferenceSession(
        str(fp32_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = fp32_session.get_inputs()[0].name
    calibration = [{input_name: torch.randn(2, 4)} for _ in range(4)]

    report = quantize_onnx_static(
        fp32_path,
        int8_path,
        calibration,
        per_channel=True,
    )
    saved = save_quantization_report(report, report_path)

    int8_session = ort.InferenceSession(
        str(int8_path),
        providers=["CPUExecutionProvider"],
    )
    output = int8_session.run(None, {input_name: example.numpy()})[0]

    assert int8_path.is_file()
    assert output.shape == (2, 3)
    assert report.calibration_batches == 4
    assert report.quant_format == "QDQ"
    assert report.activation_type == "QInt8"
    assert report.weight_type == "QInt8"
    assert saved == report_path

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["size_ratio"] > 0
