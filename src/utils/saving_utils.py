from __future__ import annotations

import csv
import json
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def process_state_dict(
    state_dict: Mapping[str, Any],
    strip_prefix: str = "",
    exclude_prefixes: Sequence[str] | None = None,
) -> OrderedDict[str, Any]:
    """Return a filtered state dict with an optional leading prefix removed."""
    excluded = tuple(exclude_prefixes or ())
    processed: OrderedDict[str, Any] = OrderedDict()

    for key, value in state_dict.items():
        if excluded and key.startswith(excluded):
            continue

        mapped_key = key
        if strip_prefix and mapped_key.startswith(strip_prefix):
            mapped_key = mapped_key[len(strip_prefix) :]
        processed[mapped_key] = value

    return processed


def _load_checkpoint_state_dict(path: str | Path) -> Mapping[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain a Lightning state_dict: {path}")
    return checkpoint["state_dict"]


def save_state_dicts(
    trainer: Trainer,
    model: LightningModule,
    dirname: str | Path,
    strip_prefix: str = "",
    exclude_prefixes: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Export current and best Lightning weights as plain PyTorch state dicts."""
    output_dir = Path(dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    current_state = process_state_dict(
        model.state_dict(),
        strip_prefix=strip_prefix,
        exclude_prefixes=exclude_prefixes,
    )
    last_path = output_dir / "last_state_dict.pt"
    torch.save(current_state, last_path)
    written["last"] = last_path
    log.info(f"Saved current state dict to: {last_path}")

    checkpoint_callback = trainer.checkpoint_callback
    best_ckpt_path = getattr(checkpoint_callback, "best_model_path", "")
    if not best_ckpt_path:
        log.warning("Best checkpoint not found; skipping best state-dict export.")
        return written

    best_state = process_state_dict(
        _load_checkpoint_state_dict(best_ckpt_path),
        strip_prefix=strip_prefix,
        exclude_prefixes=exclude_prefixes,
    )
    best_path = output_dir / "best_state_dict.pt"
    torch.save(best_state, best_path)
    written["best"] = best_path
    log.info(f"Saved best state dict to: {best_path}")

    return written


def _to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    return value


def _batch_lengths(value: Any) -> list[int]:
    """Collect candidate leading batch dimensions from nested prediction values."""
    if isinstance(value, torch.Tensor):
        return [len(value)] if value.ndim > 0 else []
    if isinstance(value, Mapping):
        lengths: list[int] = []
        for item in value.values():
            lengths.extend(_batch_lengths(item))
        return lengths
    if isinstance(value, (list, tuple)):
        return [len(value)]
    return []


def _slice_batch_value(value: Any, index: int, batch_size: int) -> Any:
    """Take one sample from a nested prediction value when it carries batch dimension."""
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and len(value) == batch_size:
            return _to_python(value[index])
        return _to_python(value)
    if isinstance(value, Mapping):
        return {
            str(key): _slice_batch_value(item, index, batch_size) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return _to_python(value[index])
        return _to_python(value)
    return _to_python(value)


def prediction_rows(predictions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for batch in predictions:
        if not batch:
            continue

        lengths: list[int] = []
        for value in batch.values():
            lengths.extend(_batch_lengths(value))

        batch_size = lengths[0] if lengths else 1
        if any(length != batch_size for length in lengths):
            raise ValueError("Prediction batch values must have matching leading dimensions.")

        for index in range(batch_size):
            rows.append(
                {
                    str(key): _slice_batch_value(value, index, batch_size)
                    for key, value in batch.items()
                }
            )

    return rows


def _save_prediction_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    if path.suffix == ".csv":
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                        for key, value in row.items()
                    }
                )
        return

    raise ValueError(f"Unsupported prediction format: {path.suffix}")


def save_predictions(
    predictions: list[Any],
    dirname: str | Path,
    output_format: str = "json",
) -> list[Path]:
    """Save Lightning predict outputs for one or multiple dataloaders."""
    if output_format not in {"json", "csv"}:
        raise ValueError("output_format must be either 'json' or 'csv'.")

    if not predictions:
        log.warning("Predictions are empty; nothing to save.")
        return []

    output_dir = Path(dirname) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: list[list[Mapping[str, Any]]]
    if isinstance(predictions[0], Mapping):
        groups = [predictions]
    elif isinstance(predictions[0], list):
        groups = predictions
    else:
        raise TypeError("Predictions must be a list of mappings or a list of dataloader results.")

    written = []
    for index, group in enumerate(groups):
        if not group:
            continue
        filename = (
            f"predictions.{output_format}"
            if len(groups) == 1
            else f"predictions_{index}.{output_format}"
        )
        path = output_dir / filename
        _save_prediction_rows(prediction_rows(group), path)
        written.append(path)
        log.info(f"Saved predictions to: {path}")

    return written
