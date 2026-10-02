# Image data pipeline

The optional vision data pipeline adds manifest-driven image datasets,
torchvision transforms v2, HDF5-backed image reads, and multiple prediction
dataloaders without changing the minimal MNIST dependency set.

## Install

```bash
uv sync --extra vision
```

## Manifest format

A classification manifest can be a mapping:

```json
{
  "class_a/0001.jpg": 0,
  "class_b/0002.jpg": 1
}
```

or a list of records:

```json
[
  {"path": "class_a/0001.jpg", "label": 0},
  {"path": "class_b/0002.jpg", "label": 1}
]
```

The list form can carry extra metadata for project-specific dataset subclasses.

## Train from image files

```bash
uv run train-command data=image_classification
```

The default config expects `train.json`, `val.json`, and `test.json` under
`data/`, with image paths relative to `data/images/`.

## HDF5

Pack encoded image bytes while keeping manifest keys unchanged:

```bash
python scripts/create_image_hdf5.py data/train.json data/train.h5 \
  --root-dir data/images
```

Then override the dataset storage path:

```bash
uv run train-command data=image_classification \
  data.datasets.train.hdf5_path=data/train.h5
```

The HDF5 reader opens files lazily per process. DataLoader workers therefore do
not inherit a live `h5py.File` handle from the parent process.

## Multiple prediction datasets

`datasets.predict` may contain multiple named dataset configs. Lightning will
receive one prediction dataloader per entry. Multi-dataset training is
intentionally not handled here; that requires an explicit Lightning 2.x
`CombinedLoader` policy and is implemented separately.

## Batch augmentation

Classification configs expose torchvision v2 MixUp/CutMix at the DataLoader collate boundary.
The default is disabled:

```bash
uv run train-command experiment=image_classification data.batch_augmentation.mode=none
```

Enable one policy with:

```bash
uv run train-command experiment=image_classification data.batch_augmentation.mode=mixup
uv run train-command experiment=image_classification data.batch_augmentation.mode=cutmix
uv run train-command experiment=image_classification data.batch_augmentation.mode=mixup_cutmix
```

`mixup_cutmix` randomly chooses `torchvision.transforms.v2.MixUp` or `CutMix` for each
training batch. Validation, test, and prediction data are never mixed. The mixed soft
labels are passed to cross-entropy loss while the original hard labels are preserved for
training accuracy and qualitative image diagnostics. With mixing enabled, that train accuracy
is a source-label diagnostic rather than a clean-sample accuracy metric; validation/test
accuracy remains the model-selection metric.

Sample-level augmentation is also torchvision v2. The shipped configs use basic resize/
crop/flip/color transforms, and additional v2 policies such as `RandAugment`,
`TrivialAugmentWide`, or `AugMix` can be added as Hydra-instantiated operations when a
specific experiment needs them.
