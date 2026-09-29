import pytest
import torch

from src.models.losses import AngularMarginSoftmaxLoss, FocalLoss, VICRegLoss


def test_focal_loss_backward() -> None:
    logits = torch.randn(6, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 1, 0])

    loss = FocalLoss()(logits, targets)
    loss.backward()

    assert loss.ndim == 0
    assert logits.grad is not None


@pytest.mark.parametrize("loss_type", ["arcface", "sphereface", "cosface"])
def test_angular_margin_softmax(loss_type: str) -> None:
    embeddings = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    criterion = AngularMarginSoftmaxLoss(
        embedding_size=16,
        num_classes=4,
        loss_type=loss_type,
    )

    loss, cosine = criterion(embeddings, labels)
    loss.backward()

    assert cosine.shape == (8, 4)
    assert embeddings.grad is not None
    assert criterion.weight.grad is not None


def test_vicreg_loss_backward() -> None:
    z1 = torch.randn(8, 32, requires_grad=True)
    z2 = torch.randn(8, 32, requires_grad=True)

    loss = VICRegLoss()(z1, z2)
    loss.backward()

    assert loss.ndim == 0
    assert z1.grad is not None
    assert z2.grad is not None


def test_vicreg_rejects_batch_size_one() -> None:
    with pytest.raises(ValueError, match="batch size"):
        VICRegLoss()(torch.randn(1, 8), torch.randn(1, 8))
