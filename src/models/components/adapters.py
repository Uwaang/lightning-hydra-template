from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models as tv_models


_HEAD_ATTRIBUTES = ("fc", "classifier", "heads", "head")


def _replace_last_linear(module: nn.Module, num_classes: int | None) -> int:
    """Replace the last Linear layer inside a known classifier container."""
    children = list(module.named_children())

    for name, child in reversed(children):
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            replacement: nn.Module
            if num_classes is None:
                replacement = nn.Identity()
            else:
                replacement = nn.Linear(in_features, num_classes)
            setattr(module, name, replacement)
            return in_features

        try:
            return _replace_last_linear(child, num_classes)
        except ValueError:
            continue

    raise ValueError("No Linear layer was found in the classifier container.")


def replace_torchvision_classifier(
    model: nn.Module,
    num_classes: int | None,
) -> int:
    """Replace a torchvision classification head without guessing the last model child."""
    for attribute in _HEAD_ATTRIBUTES:
        if not hasattr(model, attribute):
            continue

        head = getattr(model, attribute)
        if isinstance(head, nn.Linear):
            in_features = head.in_features
            replacement: nn.Module
            if num_classes is None:
                replacement = nn.Identity()
            else:
                replacement = nn.Linear(in_features, num_classes)
            setattr(model, attribute, replacement)
            return in_features

        if isinstance(head, nn.Module):
            try:
                return _replace_last_linear(head, num_classes)
            except ValueError:
                continue

    raise ValueError(
        f"Unsupported torchvision classifier layout for {model.__class__.__name__}. "
        f"Expected one of {_HEAD_ATTRIBUTES} containing a Linear layer."
    )


class TorchvisionClassifier(nn.Module):
    """Torchvision classifier with an explicit, replaceable classification head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        weights: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = tv_models.get_model(model_name, weights=weights, **model_kwargs)
        self.num_features = replace_torchvision_classifier(self.model, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if self.num_classes == 1 and output.ndim > 1:
            output = output.squeeze(-1)
        return output


class TorchvisionBackbone(nn.Module):
    """Torchvision model with its classification head replaced by Identity."""

    def __init__(
        self,
        model_name: str,
        weights: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = tv_models.get_model(model_name, weights=weights, **model_kwargs)
        self.num_features = replace_torchvision_classifier(self.model, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TimmClassifier(nn.Module):
    """Classifier backed by timm's native classifier-reset API."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = False,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            **model_kwargs,
        )
        self.num_features = int(self.model.num_features)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if self.num_classes == 1 and output.ndim > 1:
            output = output.squeeze(-1)
        return output


class TimmBackbone(nn.Module):
    """Feature extractor backed by timm with the classifier disabled."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **model_kwargs,
        )
        self.num_features = int(self.model.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SegmentationModel(nn.Module):
    """Thin adapter over segmentation-models-pytorch architectures."""

    def __init__(
        self,
        architecture: str,
        num_classes: int,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        if not hasattr(smp, architecture):
            raise ValueError(f"Unknown segmentation architecture: {architecture}")
        factory = getattr(smp, architecture)
        self.model = factory(classes=num_classes, **model_kwargs)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
