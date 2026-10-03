import hydra
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.errors import ConfigCompositionException
from lightning.pytorch.callbacks import WeightAveraging
from omegaconf import DictConfig, OmegaConf

from src.callbacks import BackboneUnfreezingCallback


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_mlflow_config_uses_sqlite_and_log_artifact_root() -> None:
    """Local MLflow defaults should avoid the deprecated filesystem tracking backend."""
    cfg = OmegaConf.load("configs/logger/mlflow.yaml")
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(raw, dict)
    assert raw["mlflow"]["tracking_uri"] == "sqlite:///${paths.log_dir}/mlflow/mlflow.db"
    assert raw["mlflow"]["artifact_location"] == "${paths.log_dir}/mlflow/artifacts"


def test_train_schema_rejects_invalid_seed_type() -> None:
    """Structured config should reject invalid values for stable runtime options."""
    with initialize(version_base="1.3", config_path="../configs"):
        with pytest.raises(ConfigCompositionException):
            compose(config_name="train.yaml", overrides=["seed=not-an-int"])


@pytest.mark.parametrize("callbacks_name", ["ema", "finetune", "finetune_ema"])
def test_training_recipe_callback_configs(callbacks_name: str) -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=[f"callbacks={callbacks_name}"])

    if "ema" in cfg.callbacks:
        ema = hydra.utils.instantiate(cfg.callbacks.ema)
        assert isinstance(ema, WeightAveraging)

    if "backbone_unfreezing" in cfg.callbacks:
        finetuning = hydra.utils.instantiate(cfg.callbacks.backbone_unfreezing)
        assert isinstance(finetuning, BackboneUnfreezingCallback)


def test_classification_label_smoothing_defaults_off() -> None:
    cfg = OmegaConf.load("configs/model/image_classification.yaml")
    assert cfg.loss.label_smoothing == 0.0


def test_multi_objective_optuna_config_composes() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=example", "hparams_search=mnist_optuna_pareto"],
            return_hydra_config=True,
        )

    assert list(cfg.optimized_metrics) == ["val/acc_best", "model/params"]
    assert list(cfg.hydra.sweeper.direction) == ["maximize", "minimize"]


def test_image_classification_optuna_config_composes() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=image_classification",
                "hparams_search=image_classification_optuna",
            ],
            return_hydra_config=True,
        )

    assert cfg.optimized_metric == "val/acc_best"
    assert cfg.hydra.sweeper.params["model.loss.label_smoothing"] == "interval(0.0, 0.2)"


def test_kd_model_recipe_requires_teacher_state_dict() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=image_classification", "model=image_classification_kd"],
        )

    assert OmegaConf.is_missing(cfg.model, "teacher_state_dict_path")
    assert cfg.model.distillation_alpha == 0.5
    assert cfg.model.teacher.model_name == "resnet50"
