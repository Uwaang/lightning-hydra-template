# Image classification

The generic image classification path combines the image data pipeline,
provider-specific model adapters, and a Lightning 2.x classification module.

## Minimal layout

```text
data/
├── images/
├── train.json
├── val.json
└── test.json
```

Each manifest maps image paths to integer class labels.

## Run

Install the optional image pipeline dependencies:

```bash
uv sync --extra vision
```

Then launch the provided experiment:

```bash
uv run train-command experiment=image_classification
```

The default network is torchvision ResNet-18 with two classes. Override the
class count in one place:

```bash
uv run train-command experiment=image_classification model.num_classes=5
```

To use a timm model, install `uv sync --extra model-zoo`, then override the
network target and model name through a project-specific model config.

The module accepts both the original tuple-style batches used by the MNIST
example and dictionary batches with `image` and `label` keys.

## CIFAR-10 validation experiment

A self-contained CIFAR-10 experiment is included for end-to-end validation:

```bash
uv run train-command experiment=cifar10
```

It downloads `torchvision.datasets.CIFAR10`, creates a deterministic 45k/5k
train/validation split, and evaluates on the 10k test split. The default model is
torchvision ResNet-18 trained from scratch with SGD, cosine annealing, random crop,
horizontal flip, and CIFAR-10 normalization.

The experiment also enables bounded image diagnostics:

- one transformed training mini-batch;
- fixed validation samples at the first and final checkpoints;
- a high-confidence test-error gallery.

The default dataloader uses `num_workers=0` for Windows-safe validation. Override it
for Linux or Docker when multiprocessing has been validated on the target system.
