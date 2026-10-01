from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lightning import LightningDataModule


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _split_sizes(datamodule: LightningDataModule) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for split_name, attribute in (
        ("train", "data_train"),
        ("val", "data_val"),
        ("test", "data_test"),
    ):
        dataset = getattr(datamodule, attribute, None)
        if dataset is None:
            continue
        try:
            sizes[split_name] = len(dataset)
        except TypeError:
            continue
    return sizes


def _fallback_fingerprint_payload(
    datamodule: LightningDataModule,
    hparams: dict[str, Any],
    split_sizes: dict[str, int],
) -> dict[str, Any]:
    excluded = {"data_dir", "root", "path", "batch_size", "num_workers", "pin_memory"}
    fingerprint_hparams = {
        key: value for key, value in hparams.items() if key not in excluded
    }
    return {
        "datamodule": f"{datamodule.__class__.__module__}.{datamodule.__class__.__qualname__}",
        "hparams": fingerprint_hparams,
        "split_sizes": split_sizes,
    }


def build_dataset_provenance(datamodule: LightningDataModule) -> dict[str, Any]:
    """Build a stable dataset/configuration provenance record.

    The default fingerprint covers dataset identity/configuration and split policy, not
    the raw bytes of every sample. Datamodules can provide a dataset_provenance method
    to define a more precise domain-specific identity payload.
    """
    hparams = _json_safe(dict(getattr(datamodule, "hparams", {})))
    if not isinstance(hparams, dict):
        hparams = {}

    split_sizes = _split_sizes(datamodule)
    custom_method = getattr(datamodule, "dataset_provenance", None)
    custom_payload = _json_safe(custom_method()) if callable(custom_method) else None

    if isinstance(custom_payload, dict):
        fingerprint_payload = custom_payload
    else:
        fingerprint_payload = _fallback_fingerprint_payload(
            datamodule,
            hparams,
            split_sizes,
        )

    canonical = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return {
        "datamodule": f"{datamodule.__class__.__module__}.{datamodule.__class__.__qualname__}",
        "hparams": hparams,
        "split_sizes": split_sizes,
        "dataset_identity": custom_payload,
        "fingerprint": fingerprint,
        "fingerprint_algorithm": "sha256",
        "fingerprint_scope": (
            "dataset identity/configuration and split policy; raw sample bytes are not hashed"
        ),
    }


def save_dataset_provenance(
    datamodule: LightningDataModule,
    dirname: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Save dataset provenance as a JSON artifact and return its path and payload."""
    output_dir = Path(dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_dataset_provenance(datamodule)
    path = output_dir / "dataset.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path, payload
