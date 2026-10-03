import hydra
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig, OmegaConf


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


def test_ema_callback_preset_instantiates() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=["callbacks=ema"])

    callback = hydra.utils.instantiate(cfg.callbacks.ema)

    assert callback.__class__.__name__ == "WeightAveraging"


def test_finetuning_callback_preset_instantiates() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=["callbacks=finetuning"])

    callback = hydra.utils.instantiate(cfg.callbacks.finetuning)

    assert callback.__class__.__name__ == "ParameterPatternFinetuning"


@pytest.mark.parametrize(
    ("data_name", "policy_target"),
    [
        ("image_classification_randaugment", "torchvision.transforms.v2.RandAugment"),
        ("image_classification_trivialaugment", "torchvision.transforms.v2.TrivialAugmentWide"),
        ("image_classification_augmix", "torchvision.transforms.v2.AugMix"),
    ],
)
def test_image_augmentation_policy_presets_compose(
    data_name: str,
    policy_target: str,
) -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[f"data={data_name}", "model=image_classification"],
        )

    assert cfg.data.transforms.train.policy._target_ == policy_target


def test_pretrained_classification_preset_composes() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=["model=image_classification_pretrained"])

    assert cfg.model.net.weights == "DEFAULT"
    assert cfg.model.loss.label_smoothing == 0.0
