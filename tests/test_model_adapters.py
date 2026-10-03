import pytest
import torch

from src.models.components.adapters import TorchvisionBackbone, TorchvisionClassifier


def test_torchvision_classifier() -> None:
    model = TorchvisionClassifier(
        model_name="resnet18",
        num_classes=3,
        weights=None,
    )
    output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 3)
    assert model.num_features == 512


def test_torchvision_classifier_can_freeze_backbone_only() -> None:
    model = TorchvisionClassifier(
        model_name="resnet18",
        num_classes=3,
        weights=None,
    )

    model.set_backbone_trainable(False)

    assert all(not parameter.requires_grad for parameter in model.backbone_parameters())
    assert all(parameter.requires_grad for parameter in model.head_parameters())

    model.set_backbone_trainable(True)

    assert all(parameter.requires_grad for parameter in model.backbone_parameters())
    assert all(parameter.requires_grad for parameter in model.head_parameters())


def test_torchvision_backbone() -> None:
    model = TorchvisionBackbone(model_name="resnet18", weights=None)
    output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 512)
    assert model.num_features == 512


def test_timm_classifier() -> None:
    pytest.importorskip("timm")
    from src.models.components.adapters import TimmClassifier

    model = TimmClassifier(
        model_name="resnet18",
        num_classes=4,
        pretrained=False,
    )
    output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 4)


def test_timm_classifier_can_freeze_backbone_only() -> None:
    pytest.importorskip("timm")
    from src.models.components.adapters import TimmClassifier

    model = TimmClassifier(
        model_name="resnet18",
        num_classes=4,
        pretrained=False,
    )

    model.set_backbone_trainable(False)

    assert all(not parameter.requires_grad for parameter in model.backbone_parameters())
    assert all(parameter.requires_grad for parameter in model.head_parameters())

    model.set_backbone_trainable(True)

    assert all(parameter.requires_grad for parameter in model.backbone_parameters())


def test_segmentation_model() -> None:
    pytest.importorskip("segmentation_models_pytorch")
    from src.models.components.adapters import SegmentationModel

    model = SegmentationModel(
        architecture="Unet",
        num_classes=2,
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
    )
    output = model(torch.randn(1, 3, 64, 64))

    assert output.shape == (1, 2, 64, 64)
