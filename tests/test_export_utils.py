from pathlib import Path

import pytest
import torch
from torch import nn

from src.utils.export_utils import export_onnx, export_pt2


class TinyExportModel(nn.Module):
    """Small model with nested outputs for export round-trip coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.linear(x)
        return {"logits": logits, "preds": torch.argmax(logits, dim=-1)}


def test_export_pt2_roundtrip_and_restore_mode(tmp_path: Path) -> None:
    model = TinyExportModel()
    model.train()
    inputs = (torch.randn(2, 4),)
    output_path = tmp_path / "tiny_model.pt2"

    exported_program = export_pt2(model, inputs, output_path)

    assert output_path.exists()
    assert model.training

    with torch.inference_mode():
        output = exported_program.module()(*inputs)

    assert set(output) == {"logits", "preds"}
    assert output["logits"].shape == (2, 3)
    assert output["preds"].shape == (2,)


class TinyOnnxModel(nn.Module):
    """Small tensor-only model for ONNX Runtime parity coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def test_export_onnx_roundtrip_and_restore_mode(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    model = TinyOnnxModel()
    model.train()
    inputs = (torch.randn(2, 4),)
    output_path = tmp_path / "tiny_model.onnx"

    program = export_onnx(model, inputs, output_path, verify=True)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert program.exported_program is not None
    assert model.training
