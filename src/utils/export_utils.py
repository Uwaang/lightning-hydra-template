from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn


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

    The model is exported in eval mode, serialized with torch.export.save,
    loaded back with torch.export.load, and executed through
    ExportedProgram.module() using the same example inputs.
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
