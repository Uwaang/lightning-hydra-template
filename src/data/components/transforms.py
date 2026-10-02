from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image


class AlbumentationsTransform:
    """Compose Hydra-instantiated Albumentations operations."""

    def __init__(self, operations: Sequence[Any]) -> None:
        if not operations:
            raise ValueError("At least one Albumentations operation is required.")

        try:
            import albumentations as A
        except ImportError as exc:
            raise ImportError(
                "AlbumentationsTransform requires the optional vision dependencies. "
                "Install them with 'uv sync --extra vision'."
            ) from exc

        self.transform = A.Compose(list(operations))

    def __call__(self, image: Any, **kwargs: Any) -> dict[str, Any]:
        if isinstance(image, Image.Image):
            image = np.asarray(image)
        return self.transform(image=image, **kwargs)
