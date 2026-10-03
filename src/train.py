from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.config_schema import register_configs
from src.utils.console_utils import configure_windows_stdio

configure_windows_stdio()
register_configs()
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    ClassificationImageDiagnosticsCallback,
    RankedLogger,
    ResearchMonitorCallback,
    extras,
    get_metric_value,
    get_metric_values,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    log_metrics_to_loggers,
    log_prediction_table_to_mlflow,
    log_run_metadata,
    publish_mlflow_artifacts,
    save_classification_report,
    save_confident_error_gallery,
    save_dataset_provenance,
    save_state_dicts,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    optimization_target = getattr(model, "net", model)
    optimization_metrics = {
        "model/params": float(sum(parameter.numel() for parameter in optimization_target.parameters()))
    }

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    reporting_cfg = cfg.extras.get("reporting", {}) if cfg.get("extras") else {}
    reporting_requested = bool(logger and reporting_cfg.get("enabled", False))

    visualization_metadata: dict[str, Any] = {}
    visualization_metadata_fn = getattr(datamodule, "visualization_metadata", None)
    if callable(visualization_metadata_fn):
        visualization_metadata = dict(visualization_metadata_fn())

    reporting_class_names: list[str] | None
    class_names_cfg = reporting_cfg.get("class_names")
    if class_names_cfg is not None:
        reporting_class_names = list(class_names_cfg)
    else:
        metadata_class_names = visualization_metadata.get("class_names")
        reporting_class_names = list(metadata_class_names) if metadata_class_names else None

    image_logging_cfg = reporting_cfg.get("image_logging", {})
    image_logging_requested = bool(reporting_requested and image_logging_cfg.get("enabled", False))
    image_mean_cfg = image_logging_cfg.get("mean")
    image_std_cfg = image_logging_cfg.get("std")
    image_mean = (
        list(image_mean_cfg) if image_mean_cfg is not None else visualization_metadata.get("mean")
    )
    image_std = (
        list(image_std_cfg) if image_std_cfg is not None else visualization_metadata.get("std")
    )
    if reporting_requested:
        if reporting_cfg.get("log_learning_rate", True) and not any(
            isinstance(callback, LearningRateMonitor) for callback in callbacks
        ):
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        if reporting_cfg.get("log_epoch_time", True) or reporting_cfg.get(
            "log_system_metrics", True
        ):
            callbacks.append(
                ResearchMonitorCallback(
                    log_epoch_time=bool(reporting_cfg.get("log_epoch_time", True)),
                    log_system_metrics=bool(reporting_cfg.get("log_system_metrics", True)),
                )
            )
        if image_logging_requested:
            callbacks.append(
                ClassificationImageDiagnosticsCallback(
                    output_dir=cfg.paths.output_dir,
                    class_names=reporting_class_names,
                    mean=image_mean,
                    std=image_std,
                    train_preview_images=int(image_logging_cfg.get("train_preview_images", 16)),
                    fixed_val_images=int(image_logging_cfg.get("fixed_val_images", 16)),
                )
            )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    reporting_enabled = bool(reporting_requested and not trainer.fast_dev_run)
    artifact_groups: dict[str, list[Path]] = {}
    metadata_dir: Path | None = None

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    should_log_metadata = bool(cfg.get("extras") and cfg.extras.get("log_metadata")) or (
        reporting_enabled and reporting_cfg.get("log_metadata", True)
    )
    if should_log_metadata:
        metadata_dir = log_run_metadata(cfg)
        if reporting_enabled:
            artifact_groups.setdefault("metadata", []).append(metadata_dir)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = dict(trainer.callback_metrics)

    should_save_state_dicts = bool(cfg.get("save_state_dict")) or (
        reporting_enabled and reporting_cfg.get("log_state_dicts", True)
    )
    if should_save_state_dicts and cfg.get("train"):
        state_dict_cfg = cfg.extras.get("state_dict", {})
        exclude_prefixes = list(state_dict_cfg.get("exclude_prefixes", []))
        if getattr(model, "teacher", None) is not None and "teacher." not in exclude_prefixes:
            exclude_prefixes.append("teacher.")
        state_dict_paths = save_state_dicts(
            trainer=trainer,
            model=model,
            dirname=cfg.paths.output_dir,
            strip_prefix=state_dict_cfg.get("strip_prefix", ""),
            exclude_prefixes=exclude_prefixes,
        )
        if reporting_enabled:
            artifact_groups.setdefault("weights", []).extend(state_dict_paths.values())

    ckpt_path: str | None = None
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = dict(trainer.callback_metrics)
    report_metrics: dict[str, float] = {}

    if reporting_enabled and reporting_cfg.get("log_dataset_provenance", True):
        provenance_path, provenance = save_dataset_provenance(
            datamodule,
            Path(cfg.paths.output_dir) / "metadata",
        )
        for logger_instance in trainer.loggers:
            logger_instance.log_hyperparams(
                {
                    "data/fingerprint": provenance["fingerprint"],
                    "data/fingerprint_scope": provenance["fingerprint_scope"],
                }
            )
        if metadata_dir is None:
            artifact_groups.setdefault("metadata", []).append(provenance_path)

    if reporting_enabled and cfg.get("test"):
        log.info("Generating classification research report!")
        predictions = trainer.predict(
            model=model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=ckpt_path,
        )
        try:
            class_names = reporting_class_names
            report = save_classification_report(
                predictions=predictions,
                dirname=cfg.paths.output_dir,
                split="test",
                top_k=int(reporting_cfg.get("top_k", 3)),
                class_names=class_names,
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning(f"Classification reporting skipped: {exc}")
        else:
            report_metrics = report.metrics
            log_metrics_to_loggers(
                trainer.loggers,
                report.metrics,
                step=trainer.global_step,
            )
            artifact_groups.setdefault("reports/test", []).extend(report.artifacts.values())
            if image_logging_requested:
                test_loader = datamodule.test_dataloader()
                test_dataset = getattr(test_loader, "dataset", None)
                if test_dataset is not None:
                    try:
                        error_gallery = save_confident_error_gallery(
                            report.rows,
                            test_dataset,
                            Path(cfg.paths.output_dir)
                            / "reports"
                            / "test"
                            / "confident_errors.png",
                            class_names=reporting_class_names,
                            mean=image_mean,
                            std=image_std,
                            max_images=int(image_logging_cfg.get("confident_error_images", 32)),
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        log.warning(f"Test error gallery skipped: {exc}")
                    else:
                        if error_gallery is not None:
                            artifact_groups.setdefault("reports/test", []).append(error_gallery)
            if reporting_cfg.get("log_prediction_table", True):
                log_prediction_table_to_mlflow(
                    trainer.loggers,
                    report.rows,
                    artifact_file="reports/test/predictions_table.json",
                    max_rows=int(reporting_cfg.get("prediction_table_max_rows", 500)),
                )

    if reporting_enabled:
        if reporting_cfg.get("log_best_checkpoint", True) and ckpt_path:
            artifact_groups.setdefault("checkpoints", []).append(Path(ckpt_path))

        output_dir = Path(cfg.paths.output_dir)
        image_reports_dir = output_dir / "reports" / "images"
        if image_reports_dir.is_dir():
            artifact_groups.setdefault("reports/images", []).append(image_reports_dir)

        hydra_dir = output_dir / ".hydra"
        if hydra_dir.is_dir():
            artifact_groups.setdefault("config", []).append(hydra_dir)
        train_log = output_dir / "train.log"
        if train_log.is_file():
            artifact_groups.setdefault("logs", []).append(train_log)

        if reporting_cfg.get("log_existing_exports", True):
            for directory_name in ("artifacts", "exports"):
                export_dir = output_dir / directory_name
                if export_dir.is_dir():
                    artifact_groups.setdefault("exports", []).append(export_dir)

        publish_mlflow_artifacts(trainer.loggers, artifact_groups)

    # merge train, test, and report metrics
    metric_dict = {**train_metrics, **test_metrics, **report_metrics, **optimization_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | tuple[float, ...] | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: One optimized metric, an ordered tuple of objectives, or None.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve objective value(s) for hydra-based hyperparameter optimization
    optimized_metrics = list(cfg.get("optimized_metrics") or [])
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metrics:
        if optimized_metric:
            raise ValueError("Set either optimized_metric or optimized_metrics, not both.")
        return get_metric_values(metric_dict=metric_dict, metric_names=optimized_metrics)

    return get_metric_value(metric_dict=metric_dict, metric_name=optimized_metric)


if __name__ == "__main__":
    main()
