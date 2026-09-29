# Model adapters

The reference repository loads torchvision, timm, and
segmentation-models-pytorch models through one string-based wrapper. This fork
keeps provider-specific adapters so each provider can use its supported model
construction and classifier APIs.

## Optional dependencies

```bash
python -m pip install -r requirements/model-zoo.txt
```

Torchvision remains part of the core dependency set. timm and
segmentation-models-pytorch are optional.

## Torchvision

```yaml
_target_: src.models.components.adapters.TorchvisionClassifier
model_name: resnet18
num_classes: 10
weights: null
```

The adapter does not assume that the final child of the whole model is the
classifier. It only inspects known torchvision classifier attributes such as
`fc`, `classifier`, `heads`, and `head`.

For representation learning:

```yaml
_target_: src.models.components.adapters.TorchvisionBackbone
model_name: resnet18
weights: null
```

The classification head is replaced with `nn.Identity`, and
`num_features` records the feature dimension.

## timm

timm already exposes a stable classifier-reset interface, so the adapter uses
`timm.create_model(..., num_classes=N)` directly. A headless backbone uses
`num_classes=0`.

## segmentation-models-pytorch

Segmentation models are instantiated through the architecture name and native
SMP constructor arguments. No classifier introspection is performed.
