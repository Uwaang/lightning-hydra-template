import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

pytest.importorskip("albumentations")
pytest.importorskip("h5py")

from src.data.components.hdf5 import write_hdf5_images
from src.data.image_datamodule import ImageDataModule
from src.data.image_dataset import ClassificationImageDataset


def _make_images(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest = {}

    for index in range(4):
        filename = f"{index}.png"
        Image.new("RGB", (12, 10), color=(index * 20, 10, 30)).save(image_dir / filename)
        manifest[filename] = index % 2

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return image_dir, manifest_path


def test_classification_dataset_from_files(tmp_path: Path) -> None:
    image_dir, manifest_path = _make_images(tmp_path)
    dataset = ClassificationImageDataset(
        manifest_path=manifest_path,
        root_dir=image_dir,
        include_name=True,
    )

    item = dataset[0]
    assert item["image"].shape == (3, 10, 12)
    assert item["image"].dtype == torch.float32
    assert item["label"].dtype == torch.int64
    assert item["name"].endswith(".png")


def test_classification_dataset_from_hdf5(tmp_path: Path) -> None:
    image_dir, manifest_path = _make_images(tmp_path)
    hdf5_path = tmp_path / "images.h5"
    write_hdf5_images(
        hdf5_path,
        [(f"{index}.png", image_dir / f"{index}.png") for index in range(4)],
    )

    dataset = ClassificationImageDataset(
        manifest_path=manifest_path,
        hdf5_path=hdf5_path,
    )

    assert dataset[1]["image"].shape == (3, 10, 12)


def test_datamodule_multiple_predict_dataloaders(tmp_path: Path) -> None:
    image_dir, manifest_path = _make_images(tmp_path)

    datasets = OmegaConf.create(
        {
            "predict": {
                "first": {
                    "_target_": "src.data.image_dataset.ClassificationImageDataset",
                    "manifest_path": str(manifest_path),
                    "root_dir": str(image_dir),
                },
                "second": {
                    "_target_": "src.data.image_dataset.ClassificationImageDataset",
                    "manifest_path": str(manifest_path),
                    "root_dir": str(image_dir),
                },
            }
        }
    )
    loaders = OmegaConf.create(
        {
            "predict": {
                "batch_size": 2,
                "shuffle": False,
                "num_workers": 0,
                "pin_memory": False,
            }
        }
    )

    datamodule = ImageDataModule(datasets=datasets, loaders=loaders)
    datamodule.setup(stage="predict")
    dataloaders = datamodule.predict_dataloader()

    assert isinstance(dataloaders, list)
    assert len(dataloaders) == 2
    assert next(iter(dataloaders[0]))["image"].shape == (2, 3, 10, 12)
