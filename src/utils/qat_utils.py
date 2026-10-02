from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.utils.console_utils import configure_windows_stdio


def _require_torchao() -> tuple[Any, Any, Any, Any]:
    try:
        from torchao.quantization.pt2e import (
            move_exported_model_to_eval,
            move_exported_model_to_train,
        )
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e,
            prepare_qat_pt2e,
        )
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer,
            get_default_x86_inductor_quantization_config,
        )
    except ImportError as exc:
        raise ImportError(
            "QAT requires the optional torchao stack. Install it with 'uv sync --extra qat'."
        ) from exc

    return (
        prepare_qat_pt2e,
        convert_pt2e,
        (move_exported_model_to_train, move_exported_model_to_eval),
        (X86InductorQuantizer, get_default_x86_inductor_quantization_config),
    )


def prepare_x86_qat_pt2e(
    model: nn.Module,
    example_args: tuple[Any, ...],
    *,
    example_kwargs: Mapping[str, Any] | None = None,
    dynamic_shapes: Any | None = None,
    strict: bool = False,
) -> nn.Module:
    """Export a module and insert x86 PT2E fake-quant nodes for QAT.

    The source module is captured in training mode so training-sensitive operators such as
    BatchNorm keep their training semantics. The source module's original mode is restored
    before returning. The returned GraphModule is ready for normal forward/backward/optimizer
    steps.
    """
    prepare_qat_pt2e, _, exported_mode_api, quantizer_api = _require_torchao()
    X86InductorQuantizer, get_default_x86_inductor_quantization_config = quantizer_api

    kwargs = dict(example_kwargs or {})
    was_training = model.training
    model.train()
    try:
        exported = torch.export.export(
            model,
            args=example_args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        ).module()
    finally:
        model.train(was_training)

    quantizer = X86InductorQuantizer().set_global(
        get_default_x86_inductor_quantization_config(is_qat=True)
    )
    qat_model = prepare_qat_pt2e(exported, quantizer)
    move_exported_model_to_train, _ = exported_mode_api
    move_exported_model_to_train(qat_model)
    return qat_model


def convert_x86_qat_pt2e(qat_model: nn.Module) -> nn.Module:
    """Convert a prepared QAT graph to quantized-decomposed PT2E inference form."""
    _, convert_pt2e, exported_mode_api, _ = _require_torchao()
    _, move_exported_model_to_eval = exported_mode_api
    converted = convert_pt2e(qat_model)
    move_exported_model_to_eval(converted)
    return converted


def export_qat_onnx(
    converted_model: nn.Module,
    example_args: tuple[Any, ...],
    output_path: str | Path,
    *,
    example_kwargs: Mapping[str, Any] | None = None,
    dynamic_shapes: Any | None = None,
    opset_version: int | None = None,
) -> torch.onnx.ONNXProgram:
    """Export a converted PT2E-QAT graph without calling the normal Module.eval API.

    Exported PT2E GraphModules intentionally do not support the regular eval() path.
    convert_x86_qat_pt2e already switches exported-model semantics to inference mode,
    so this helper directly invokes the dynamo ONNX exporter.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(example_kwargs or {})
    configure_windows_stdio()

    program = torch.onnx.export(
        converted_model,
        args=example_args,
        kwargs=kwargs,
        f=None,
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        opset_version=opset_version,
        verify=False,
    )
    if program is None:
        raise RuntimeError("The dynamo ONNX exporter did not return an ONNXProgram.")

    program.save(path)
    return program
