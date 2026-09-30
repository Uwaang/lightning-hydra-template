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
python -m pip install -r requirements/vision.txt
```

Then launch the provided experiment:

```bash
python src/train.py experiment=image_classification
```

The default network is torchvision ResNet-18 with two classes. Override the
class count in one place:

```bash
python src/train.py experiment=image_classification model.num_classes=5
```

To use a timm model after installing `requirements/model-zoo.txt`, override
the network target and model name through a project-specific model config.

The module accepts both the original tuple-style batches used by the MNIST
example and dictionary batches with `image` and `label` keys.
