# Image data pipeline

The optional vision data pipeline adds manifest-driven image datasets,
Albumentations transforms, HDF5-backed image reads, and multiple prediction
dataloaders without changing the minimal MNIST dependency set.

## Install

```bash
python -m pip install -r requirements/vision.txt
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
python src/train.py data=image_classification
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
python src/train.py data=image_classification \
  data.datasets.train.hdf5_path=data/train.h5
```

The HDF5 reader opens files lazily per process. DataLoader workers therefore do
not inherit a live `h5py.File` handle from the parent process.

## Multiple prediction datasets

`datasets.predict` may contain multiple named dataset configs. Lightning will
receive one prediction dataloader per entry. Multi-dataset training is
intentionally not handled here; that requires an explicit Lightning 2.x
`CombinedLoader` policy and is implemented separately.
