from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.components.hdf5 import HDF5ImageStore
from src.data.components.paths import discover_image_paths, load_manifest


class BaseImageDataset(Dataset[dict[str, Any]]):
    """Shared image decoding and transform logic for image datasets."""

    def __init__(
        self,
        transforms: Callable[..., dict[str, Any]] | None = None,
        read_backend: str = "pillow",
        grayscale: bool = False,
    ) -> None:
        self.transforms = transforms
        self.read_backend = read_backend
        self.grayscale = grayscale

    def _read_image(self, source: Path | bytes) -> np.ndarray:
        if self.read_backend == "pillow":
            image_source: Any = source if isinstance(source, Path) else io.BytesIO(source)
            with Image.open(image_source) as image:
                mode = "L" if self.grayscale else "RGB"
                array = np.asarray(image.convert(mode))
        elif self.read_backend == "opencv":
            import cv2

            if isinstance(source, Path):
                flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
                array = cv2.imread(str(source), flag)
            else:
                buffer = np.frombuffer(source, dtype=np.uint8)
                flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
                array = cv2.imdecode(buffer, flag)
            if array is None:
                raise ValueError(f"Unable to decode image: {source}")
            if not self.grayscale:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("read_backend must be 'pillow' or 'opencv'.")

        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        return array

    def _to_tensor(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image.float()

        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image)
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div(255.0)
        else:
            tensor = tensor.float()
        return tensor

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        return self._to_tensor(image)


class ClassificationImageDataset(BaseImageDataset):
    """Classification dataset backed by image files or encoded images in HDF5."""

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
        shuffle_seed: int | None = None,
    ) -> None:
        super().__init__(transforms=transforms, read_backend=read_backend, grayscale=grayscale)
        self.samples = load_manifest(manifest_path)
        if not all("label" in sample for sample in self.samples):
            raise ValueError("Classification manifests require a 'label' for every sample.")

        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.samples)

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

        image = self._process_image(self._read_image(source))
        label = torch.as_tensor(sample["label"], dtype=self.label_dtype)

        item: dict[str, Any] = {"image": image, "label": label}
        if self.include_name:
            item["name"] = key
        return item


class UnlabeledImageDataset(BaseImageDataset):
    """Prediction dataset that discovers image files or reads a manifest."""

    def __init__(
        self,
        paths: list[str] | None = None,
        directories: list[str] | None = None,
        manifest_path: str | Path | None = None,
        root_dir: str | Path = "",
        transforms: Callable[..., dict[str, Any]] | None = None,
        read_backend: str = "pillow",
        grayscale: bool = False,
        include_name: bool = True,
        recursive: bool = True,
    ) -> None:
        super().__init__(transforms=transforms, read_backend=read_backend, grayscale=grayscale)
        root = Path(root_dir)

        collected: list[Path] = []
        if paths:
            collected.extend(root / path for path in paths)
        if directories:
            collected.extend(discover_image_paths(directories, recursive=recursive))
        if manifest_path:
            collected.extend(root / str(item["path"]) for item in load_manifest(manifest_path))

        if not collected:
            raise ValueError("Provide paths, directories, or manifest_path.")

        self.paths = collected
        self.include_name = include_name

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        item: dict[str, Any] = {"image": self._process_image(self._read_image(path))}
        if self.include_name:
            item["name"] = str(path)
        return item
