from pathlib import Path

import pytest
import torch
from torch import nn

from src.utils.qat_utils import (
    convert_x86_qat_pt2e,
    export_qat_onnx,
    prepare_x86_qat_pt2e,
)


class TinyQATModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        return self.fc(self.pool(x).flatten(1))


def _prepared_model(example: torch.Tensor) -> nn.Module:
    pytest.importorskip("torchao")
    return prepare_x86_qat_pt2e(TinyQATModel(), (example,))


def test_qat_prepare_step_reload_and_convert() -> None:
    example = torch.randn(4, 3, 16, 16)
    targets = torch.tensor([0, 1, 2, 3])
    qat_model = _prepared_model(example)
    optimizer = torch.optim.SGD(qat_model.parameters(), lr=1e-3)

    output = qat_model(example)
    loss = nn.functional.cross_entropy(output, targets)
    loss.backward()
    optimizer.step()

    saved_state = {name: value.detach().clone() for name, value in qat_model.state_dict().items()}
    reloaded = _prepared_model(example)
    reloaded.load_state_dict(saved_state)
    converted = convert_x86_qat_pt2e(reloaded)

    with torch.no_grad():
        converted_output = converted(example)

    quantized_targets = {
        str(node.target)
        for node in converted.graph.nodes
        if "quantized_decomposed" in str(node.target)
    }
    assert converted_output.shape == (4, 4)
    assert torch.isfinite(converted_output).all()
    assert any("quantize_per_tensor" in target for target in quantized_targets)
    assert any("dequantize_per_channel" in target for target in quantized_targets)


def test_qat_qdq_onnx_runs_in_ort(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    pytest.importorskip("torchao")

    example = torch.randn(1, 3, 16, 16)
    qat_model = prepare_x86_qat_pt2e(TinyQATModel(), (example,))
    _ = qat_model(example)
    converted = convert_x86_qat_pt2e(qat_model)
    output_path = tmp_path / "tiny-qat.onnx"

    export_qat_onnx(converted, (example,), output_path)

    onnx_model = onnx.load(output_path)
    op_types = [node.op_type for node in onnx_model.graph.node]
    assert op_types.count("QuantizeLinear") > 0
    assert op_types.count("DequantizeLinear") > 0

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: example.numpy()})[0]
    assert output.shape == (1, 4)
    assert torch.isfinite(torch.from_numpy(output)).all()
