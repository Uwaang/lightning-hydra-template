from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.utils.console_utils import configure_windows_stdio


def _assert_outputs_close(
    eager_output: Any,
    exported_output: Any,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Recursively compare eager and exported inference outputs."""
    if isinstance(eager_output, torch.Tensor) and isinstance(exported_output, torch.Tensor):
        torch.testing.assert_close(eager_output, exported_output, rtol=rtol, atol=atol)
        return

    if isinstance(eager_output, Mapping) and isinstance(exported_output, Mapping):
        if set(eager_output) != set(exported_output):
            raise AssertionError("Eager and exported outputs have different mapping keys.")
        for key in eager_output:
            _assert_outputs_close(
                eager_output[key],
                exported_output[key],
                rtol=rtol,
                atol=atol,
            )
        return

    if (
        isinstance(eager_output, Sequence)
        and not isinstance(eager_output, (str, bytes))
        and isinstance(exported_output, Sequence)
        and not isinstance(exported_output, (str, bytes))
    ):
        if len(eager_output) != len(exported_output):
            raise AssertionError("Eager and exported outputs have different sequence lengths.")
        for eager_item, exported_item in zip(eager_output, exported_output, strict=True):
            _assert_outputs_close(eager_item, exported_item, rtol=rtol, atol=atol)
        return

    if eager_output != exported_output:
        raise AssertionError(
            f"Eager and exported outputs differ: {eager_output!r} != {exported_output!r}"
        )


def export_pt2(
    model: nn.Module,
    example_args: tuple[Any, ...],
    output_path: str | Path,
    *,
    example_kwargs: Mapping[str, Any] | None = None,
    dynamic_shapes: Any | None = None,
    strict: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> torch.export.ExportedProgram:
    """Export an inference module to PT2 and verify eager/runtime parity.

    The model is exported in eval mode, serialized with torch.export.save, loaded back with
    torch.export.load, and executed through ExportedProgram.module() using the same example inputs.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(example_kwargs or {})

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            eager_output = model(*example_args, **kwargs)

        exported_program = torch.export.export(
            model,
            args=example_args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )
        torch.export.save(exported_program, path)

        loaded_program = torch.export.load(path)
        with torch.inference_mode():
            exported_output = loaded_program.module()(*example_args, **kwargs)

        _assert_outputs_close(
            eager_output,
            exported_output,
            rtol=rtol,
            atol=atol,
        )
        return loaded_program
    finally:
        model.train(was_training)


def export_onnx(
    model: nn.Module,
    example_args: tuple[Any, ...],
    output_path: str | Path,
    *,
    example_kwargs: Mapping[str, Any] | None = None,
    dynamic_shapes: Any | None = None,
    opset_version: int | None = None,
    verify: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> torch.onnx.ONNXProgram:
    """Export a plain inference module and optionally assert ONNX Runtime parity.

    The modern dynamo exporter captures the model through torch.export. When verification is
    enabled, torch.onnx.testing.assert_onnx_program compares the ONNX Runtime result against the
    exported PyTorch program before the ONNXProgram is serialized.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(example_kwargs or {})
    configure_windows_stdio()

    was_training = model.training
    model.eval()
    try:
        program = torch.onnx.export(
            model,
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

        if verify:
            from torch.onnx.testing import assert_onnx_program

            assert_onnx_program(
                program,
                args=example_args,
                kwargs=kwargs,
                rtol=rtol,
                atol=atol,
                strategy=None,
                backend="onnxruntime",
            )

        program.save(path)
        return program
    finally:
        model.train(was_training)
