# Losses and metrics

This integration keeps custom implementations only where they add real value.
Metrics that are already maintained by TorchMetrics are configured directly
instead of carrying local copies.

## Custom losses

- `FocalLoss`
- `AngularMarginSoftmaxLoss` with ArcFace, SphereFace, and CosFace modes
- `VICRegLoss`

The angular-margin implementation differs from the reference project in two
important ways:

1. scale and margin remain ordinary scalar hyperparameters, so there is no
   CPU/GPU tensor-device mismatch;
2. target logits are replaced vectorially and optimized with standard
   `torch.nn.functional.cross_entropy`.

## Standard metrics

Use TorchMetrics directly for common metrics:

- classification accuracy and AUROC
- segmentation Jaccard/IoU
- retrieval MRR
- retrieval normalized DCG

Example configs live under `configs/model/metrics/`.
