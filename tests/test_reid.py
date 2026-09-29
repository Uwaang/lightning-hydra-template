from functools import partial

import torch
from torch import nn

from src.models.components.reid import EmbeddingModel, GeM, TorchvisionFeatureMapBackbone
from src.models.losses import AngularMarginSoftmaxLoss
from src.models.reid_module import ReIdentificationLitModule


def test_gem_shape_and_trainable_p() -> None:
    pool = GeM(p=3.0, trainable=True)
    output = pool(torch.rand(2, 8, 4, 4))

    assert output.shape == (2, 8)
    assert isinstance(pool.p, nn.Parameter)


def test_torchvision_feature_map_embedding() -> None:
    backbone = TorchvisionFeatureMapBackbone(
        model_name="resnet18",
        return_node="layer4",
        weights=None,
    )
    model = EmbeddingModel(
        backbone=backbone,
        in_features=512,
        embedding_dim=32,
        pool=GeM(),
    )
    model.eval()

    output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 32)
    assert torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5)


def test_reid_model_step_backward() -> None:
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16))
    module = ReIdentificationLitModule(
        net=net,
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        loss=AngularMarginSoftmaxLoss(
            embedding_size=16,
            num_classes=4,
            loss_type="cosface",
        ),
        num_classes=4,
    )
    batch = {
        "image": torch.randn(6, 3, 8, 8),
        "label": torch.tensor([0, 1, 2, 3, 1, 0]),
    }

    loss, embeddings, preds, targets = module.model_step(batch)
    loss.backward()

    assert embeddings.shape == (6, 16)
    assert preds.shape == (6,)
    assert torch.equal(targets, batch["label"])
