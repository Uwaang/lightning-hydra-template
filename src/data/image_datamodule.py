from __future__ import annotations

from typing import Any

import hydra
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


class ImageDataModule(LightningDataModule):
    """Hydra-configurable image datamodule with optional multi-dataset prediction."""

    def __init__(
        self,
        datasets: DictConfig,
        loaders: DictConfig,
        transforms: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.datasets_cfg = datasets
        self.loaders_cfg = loaders
        self.transforms_cfg = transforms

        self.data_train: Dataset[Any] | None = None
        self.data_val: Dataset[Any] | None = None
        self.data_test: Dataset[Any] | None = None
        self.data_predict: Dataset[Any] | list[Dataset[Any]] | None = None

    def _transform(self, stage: str) -> Any:
        if self.transforms_cfg is None or not self.transforms_cfg.get(stage):
            return None
        return hydra.utils.instantiate(self.transforms_cfg[stage])

    def _dataset(self, stage: str, cfg: DictConfig) -> Dataset[Any]:
        return hydra.utils.instantiate(cfg, transforms=self._transform(stage))

    def _setup_single(self, stage: str) -> Dataset[Any] | None:
        cfg = self.datasets_cfg.get(stage)
        if cfg is None:
            return None
        return self._dataset(stage, cfg)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if self.data_train is None:
                self.data_train = self._setup_single("train")
            if self.data_val is None:
                self.data_val = self._setup_single("val")

        if stage in (None, "validate") and self.data_val is None:
            self.data_val = self._setup_single("val")

        if stage in (None, "test") and self.data_test is None:
            self.data_test = self._setup_single("test")

        if stage in (None, "predict") and self.data_predict is None:
            predict_cfg = self.datasets_cfg.get("predict")
            if predict_cfg is None:
                return
            if "_target_" in predict_cfg:
                self.data_predict = self._dataset("predict", predict_cfg)
            else:
                self.data_predict = [
                    self._dataset("predict", dataset_cfg)
                    for dataset_cfg in predict_cfg.values()
                ]

    def _loader(self, stage: str, dataset: Dataset[Any]) -> DataLoader[Any]:
        loader_cfg = OmegaConf.to_container(self.loaders_cfg[stage], resolve=True)
        if not isinstance(loader_cfg, dict):
            raise TypeError(f"Loader config for '{stage}' must be a mapping.")

        batch_size = int(loader_cfg.get("batch_size", 1))
        if self.trainer is not None and self.trainer.world_size > 1:
            if batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({batch_size}) must be divisible by world size "
                    f"({self.trainer.world_size})."
                )
            loader_cfg["batch_size"] = batch_size // self.trainer.world_size

        if int(loader_cfg.get("num_workers", 0)) == 0:
            loader_cfg.pop("persistent_workers", None)

        return DataLoader(dataset=dataset, **loader_cfg)

    def train_dataloader(self) -> DataLoader[Any]:
        if self.data_train is None:
            raise RuntimeError("Train dataset is not configured.")
        return self._loader("train", self.data_train)

    def val_dataloader(self) -> DataLoader[Any]:
        if self.data_val is None:
            raise RuntimeError("Validation dataset is not configured.")
        return self._loader("val", self.data_val)

    def test_dataloader(self) -> DataLoader[Any]:
        if self.data_test is None:
            raise RuntimeError("Test dataset is not configured.")
        return self._loader("test", self.data_test)

    def predict_dataloader(self) -> DataLoader[Any] | list[DataLoader[Any]]:
        if self.data_predict is None:
            raise RuntimeError("Predict dataset is not configured.")
        if isinstance(self.data_predict, list):
            return [self._loader("predict", dataset) for dataset in self.data_predict]
        return self._loader("predict", self.data_predict)
