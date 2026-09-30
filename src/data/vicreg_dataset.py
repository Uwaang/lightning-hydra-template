from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.data.components.hdf5 import HDF5ImageStore
from src.data.components.paths import load_manifest
from src.data.image_dataset import BaseImageDataset


class TwoViewImageDataset(BaseImageDataset):
    """Image dataset that returns two independent augmented views.

    Labels are optional. This keeps VICReg usable with genuinely unlabeled pretraining manifests
    while preserving labels when they are available.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path = "",
        hdf5_path: str | Path | None = None,
        transforms: Callable[..., dict[str, Any]] | None = None,
        read_backend: str = "pillow",
        grayscale: bool = False,
        include_name: bool = False,
        label_dtype: str = "long",
    ) -> None:
        super().__init__(
            transforms=transforms,
            read_backend=read_backend,
            grayscale=grayscale,
        )
        self.samples = load_manifest(manifest_path)
        self.root_dir = Path(root_dir)
        self.store = HDF5ImageStore(hdf5_path) if hdf5_path else None
        self.include_name = include_name

        dtypes = {"long": torch.long, "float": torch.float32}
        if label_dtype not in dtypes:
            raise ValueError("label_dtype must be either 'long' or 'float'.")
        self.label_dtype = dtypes[label_dtype]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        key = str(sample["path"]).replace("\\", "/")

        if self.store is None:
            source: Path | bytes = self.root_dir / key
        else:
            source = self.store.read_bytes(key.lstrip("/"))

        image = self._read_image(source)
        item: dict[str, Any] = {
            "view1": self._process_image(np.array(image, copy=True)),
            "view2": self._process_image(np.array(image, copy=True)),
        }

        if "label" in sample:
            item["label"] = torch.as_tensor(sample["label"], dtype=self.label_dtype)
        if self.include_name:
            item["name"] = key
        return item


# Compatibility alias for users of the first integration draft.
TwoViewClassificationImageDataset = TwoViewImageDataset
