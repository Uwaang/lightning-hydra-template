from __future__ import annotations

from typing import Any, cast

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


class CIFAR10DataModule(LightningDataModule):
    """Deterministic CIFAR-10 train/validation/test data module."""

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    visualization_mean = (0.4914, 0.4822, 0.4465)
    visualization_std = (0.2470, 0.2435, 0.2616)

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_split: tuple[int, int] = (45_000, 5_000),
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        split_seed: int = 42,
    ) -> None:
        super().__init__()
        train_val_split = cast(tuple[int, int], tuple(train_val_split))
        if sum(train_val_split) != 50_000:
            raise ValueError("CIFAR-10 train_val_split must sum to 50,000.")
        self.save_hyperparameters(logger=False)

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.visualization_mean, self.visualization_std),
            ]
        )
        self.eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.visualization_mean, self.visualization_std),
            ]
        )

        self.data_train: Dataset[Any] | None = None
        self.data_val: Dataset[Any] | None = None
        self.data_test: Dataset[Any] | None = None
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 10

    def visualization_metadata(self) -> dict[str, Any]:
        return {
            "class_names": list(self.class_names),
            "mean": list(self.visualization_mean),
            "std": list(self.visualization_std),
        }

    def dataset_provenance(self) -> dict[str, Any]:
        return {
            "dataset": "torchvision.datasets.CIFAR10",
            "split_seed": int(self.hparams.split_seed),
            "split_lengths": list(self.hparams.train_val_split) + [10_000],
            "train_transform": repr(self.train_transforms),
            "eval_transform": repr(self.eval_transforms),
            "class_names": list(self.class_names),
        }

    def prepare_data(self) -> None:
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"world size ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is None:
            augmented_train = CIFAR10(
                self.hparams.data_dir,
                train=True,
                transform=self.train_transforms,
            )
            evaluation_train = CIFAR10(
                self.hparams.data_dir,
                train=True,
                transform=self.eval_transforms,
            )
            generator = torch.Generator().manual_seed(int(self.hparams.split_seed))
            indices = torch.randperm(len(augmented_train), generator=generator).tolist()
            train_count = int(self.hparams.train_val_split[0])
            self.data_train = Subset(augmented_train, indices[:train_count])
            self.data_val = Subset(evaluation_train, indices[train_count:])
            self.data_test = CIFAR10(
                self.hparams.data_dir,
                train=False,
                transform=self.eval_transforms,
            )

    def _loader(self, dataset: Dataset[Any] | None, *, shuffle: bool) -> DataLoader[Any]:
        if dataset is None:
            raise RuntimeError("CIFAR10DataModule.setup() must run before requesting a dataloader.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=shuffle,
            num_workers=int(self.hparams.num_workers),
            pin_memory=bool(self.hparams.pin_memory),
            persistent_workers=bool(self.hparams.num_workers),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.data_test, shuffle=False)
