from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.data.image_dataset import ClassificationImageDataset


class TwoViewClassificationImageDataset(ClassificationImageDataset):
    """Classification image dataset that returns two independent augmented views."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        key = str(sample["path"]).replace("\\", "/")

        if self.store is None:
            source = self.root_dir / key
        else:
            source = self.store.read_bytes(key.lstrip("/"))

        image = self._read_image(source)
        view1 = self._process_image(np.array(image, copy=True))
        view2 = self._process_image(np.array(image, copy=True))
        label = torch.as_tensor(sample["label"], dtype=self.label_dtype)

        item: dict[str, Any] = {
            "view1": view1,
            "view2": view2,
            "label": label,
        }
        if self.include_name:
            item["name"] = key
        return item
