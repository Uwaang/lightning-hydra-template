# VICReg

The VICReg path uses the generic image pipeline and a headless vision backbone.

## Data flow

Each sample is decoded once and then the configured random augmentation
pipeline is called twice independently:

```text
encoded image
  -> decode
  -> augmentation #1 -> view1
  -> augmentation #2 -> view2
```

This avoids performing duplicate image I/O while preserving independent random
views.

## Model flow

```text
view
  -> backbone representation
  -> projector
  -> VICReg loss
```

Prediction exports the backbone representation rather than projector output,
which is generally the useful representation for downstream tasks.

## Run

```bash
python src/train.py experiment=image_vicreg
```

The default example uses a headless torchvision ResNet-18, a 2048-hidden /
512-output projector, and the common VICReg weights 25 / 25 / 1.

VICReg covariance estimation requires at least two samples per batch. The
custom loss explicitly rejects batch size one.
