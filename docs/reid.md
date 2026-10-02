# Re-identification and embedding learning

The reference template modifies a model's last children by position to install
GeM and an embedding head. This implementation separates those concerns.

## Architecture

```text
torchvision model
  -> explicit intermediate node (for example layer4)
  -> GeM
  -> Linear + BatchNorm
  -> L2-normalized embedding
  -> ArcFace / SphereFace / CosFace training loss
```

Torchvision's feature-extraction API is used to request the feature node
explicitly. The default ResNet-18 example uses `layer4`, which produces 512
channels.

## Run

Use the same manifest-driven image dataset as ordinary classification, but set
identity labels to contiguous integer class IDs:

```bash
uv run train-command experiment=image_reid model.num_classes=1000
```

The default embedding dimension is 128 and the default objective is CosFace.

## GeM

`GeM` supports fixed or trainable `p`. A trainable value is represented as
a model parameter and clamped to remain positive during the forward pass.

## Prediction

Prediction returns embeddings directly. Labels and image names are retained
when the input batch contains them, making exported embeddings traceable to
their source samples.
