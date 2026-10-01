from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "HDF5 support requires the optional vision dependencies. "
            "Install them with 'uv sync --extra vision'."
        ) from exc
    return h5py


class HDF5ImageStore:
    """Process-aware lazy HDF5 reader suitable for DataLoader workers."""

    def __init__(self, path: str | Path, swmr: bool = False) -> None:
        self.path = Path(path)
        self.swmr = swmr
        self._handle: Any | None = None
        self._pid: int | None = None

    def _ensure_open(self) -> Any:
        current_pid = os.getpid()
        if self._handle is not None and self._pid == current_pid:
            return self._handle

        self.close()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        h5py = _import_h5py()
        self._handle = h5py.File(self.path, mode="r", swmr=self.swmr)
        self._pid = current_pid
        return self._handle

    def __contains__(self, key: str) -> bool:
        return key in self._ensure_open()

    def read_bytes(self, key: str) -> bytes:
        value = self._ensure_open()[key][()]
        if isinstance(value, bytes):
            return value
        if isinstance(value, np.ndarray):
            return value.tobytes()
        return bytes(value)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()


def write_hdf5_images(
    output_path: str | Path,
    image_paths: list[tuple[str, Path]],
) -> Path:
    """Write encoded image bytes to an HDF5 file using relative paths as keys."""
    h5py = _import_h5py()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, mode="w", libver="latest") as handle:
        for key, path in image_paths:
            normalized = key.replace("\\", "/").lstrip("/")
            if not normalized:
                raise ValueError("HDF5 image keys must not be empty.")

            parent, _, dataset_name = normalized.rpartition("/")
            group = handle.require_group(parent) if parent else handle
            encoded: np.ndarray = np.fromfile(path, dtype=np.uint8)
            group.create_dataset(dataset_name, data=encoded)

    return output_path
