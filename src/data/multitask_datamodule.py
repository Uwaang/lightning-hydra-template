from __future__ import annotations

from collections import OrderedDict
from typing import Any

from lightning.pytorch.utilities.combined_loader import CombinedLoader
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.data.image_datamodule import ImageDataModule


class MultiTaskImageDataModule(ImageDataModule):
    """Image datamodule for named tasks with explicit CombinedLoader semantics."""

    def __init__(
        self,
        datasets: DictConfig,
        loaders: DictConfig,
        transforms: DictConfig | None = None,
        train_mode: str = "max_size_cycle",
    ) -> None:
        super().__init__(datasets=datasets, loaders=loaders, transforms=transforms)
        if train_mode not in {"min_size", "max_size_cycle", "max_size"}:
            raise ValueError("train_mode must be min_size, max_size_cycle, or max_size.")

        self.train_mode = train_mode
        self.task_names = list(self.datasets_cfg.get("train", {}).keys())
        if not self.task_names:
            raise ValueError("MultiTaskImageDataModule requires named train datasets.")

        self.train_tasks: OrderedDict[str, Dataset[Any]] = OrderedDict()
        self.val_tasks: OrderedDict[str, Dataset[Any]] = OrderedDict()
        self.test_tasks: OrderedDict[str, Dataset[Any]] = OrderedDict()
        self.predict_tasks: OrderedDict[str, Dataset[Any]] = OrderedDict()

    def _setup_group(self, stage: str) -> OrderedDict[str, Dataset[Any]]:
        cfg = self.datasets_cfg.get(stage)
        datasets: OrderedDict[str, Dataset[Any]] = OrderedDict()
        if cfg is None:
            return datasets

        for name, dataset_cfg in cfg.items():
            datasets[name] = self._dataset(stage, dataset_cfg)

        if stage in {"val", "test"} and datasets and list(datasets) != self.task_names:
            raise ValueError(
                f"{stage} datasets must use the same ordered task names as train: "
                f"{self.task_names}"
            )
        return datasets

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if not self.train_tasks:
                self.train_tasks = self._setup_group("train")
            if not self.val_tasks:
                self.val_tasks = self._setup_group("val")

        if stage in (None, "validate") and not self.val_tasks:
            self.val_tasks = self._setup_group("val")

        if stage in (None, "test") and not self.test_tasks:
            self.test_tasks = self._setup_group("test")

        if stage in (None, "predict") and not self.predict_tasks:
            self.predict_tasks = self._setup_group("predict")

    def train_dataloader(self) -> CombinedLoader:
        loaders = {
            name: self._loader("train", dataset) for name, dataset in self.train_tasks.items()
        }
        return CombinedLoader(loaders, mode=self.train_mode)

    def val_dataloader(self) -> list[DataLoader[Any]]:
        return [self._loader("val", self.val_tasks[name]) for name in self.task_names]

    def test_dataloader(self) -> list[DataLoader[Any]]:
        return [self._loader("test", self.test_tasks[name]) for name in self.task_names]

    def predict_dataloader(self) -> list[DataLoader[Any]]:
        return [self._loader("predict", dataset) for dataset in self.predict_tasks.values()]
