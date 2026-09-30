from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted module path, including numeric Sequential/ModuleList indices."""
    module: nn.Module = root
    for token in path.split("."):
        if token in module._modules:
            module = module._modules[token]
            continue
        if token.isdigit() and hasattr(module, "__getitem__"):
            try:
                candidate = module[int(token)]
            except (IndexError, KeyError, TypeError) as exc:
                raise KeyError(f"Unable to resolve module path '{path}' at '{token}'.") from exc
            if not isinstance(candidate, nn.Module):
                raise TypeError(f"Resolved object at '{token}' is not a torch module.")
            module = candidate
            continue
        if hasattr(module, token):
            candidate = getattr(module, token)
            if not isinstance(candidate, nn.Module):
                raise TypeError(f"Resolved object at '{token}' is not a torch module.")
            module = candidate
            continue
        raise KeyError(f"Unable to resolve module path '{path}' at '{token}'.")
    return module


def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str | nn.Module,
    categories: Sequence[int] | None = None,
    method: str = "gradcam",
    reshape_transform: Any | None = None,
    eigen_smooth: bool = False,
    aug_smooth: bool = False,
) -> np.ndarray:
    """Compute class activation maps for a batch using pytorch-grad-cam."""
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    methods = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "gradcamplusplus": GradCAMPlusPlus,
    }
    if method not in methods:
        raise ValueError(f"Unsupported CAM method: {method}")

    layer = resolve_module(model, target_layer) if isinstance(target_layer, str) else target_layer
    targets = None
    if categories is not None:
        if len(categories) != input_tensor.shape[0]:
            raise ValueError("categories length must match the input batch size.")
        targets = [ClassifierOutputTarget(int(category)) for category in categories]

    was_training = model.training
    model.eval()
    try:
        with methods[method](
            model=model,
            target_layers=[layer],
            reshape_transform=reshape_transform,
        ) as cam:
            return cam(
                input_tensor=input_tensor,
                targets=targets,
                eigen_smooth=eigen_smooth,
                aug_smooth=aug_smooth,
            )
    finally:
        model.train(was_training)


def overlay_gradcam(
    rgb_images: Sequence[np.ndarray],
    masks: np.ndarray,
) -> list[np.ndarray]:
    """Overlay CAM masks on RGB float images in the [0, 1] range."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    if len(rgb_images) != len(masks):
        raise ValueError("rgb_images and masks must have the same batch length.")

    visualizations = []
    for image, mask in zip(rgb_images, masks):
        image = np.asarray(image, dtype=np.float32)
        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("RGB images must be float arrays in the [0, 1] range.")
        visualizations.append(show_cam_on_image(image, mask, use_rgb=True))
    return visualizations