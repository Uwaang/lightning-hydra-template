import pytest
from hydra import compose, initialize


@pytest.mark.parametrize(
    ("experiment", "monitor"),
    [
        ("example", "val/acc"),
        ("cifar10", "val/acc"),
        ("image_classification", "val/acc"),
        ("image_classification_finetune", "val/acc"),
        ("image_multitask", "val/acc"),
        ("image_reid", "val/acc"),
        ("image_vicreg", "val/loss"),
    ],
)
def test_experiment_config_composes(experiment: str, monitor: str) -> None:
    """All shipped experiments must compose without requiring their datasets."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[f"experiment={experiment}"],
        )

    assert cfg.data
    assert cfg.model
    assert cfg.trainer
    assert cfg.callbacks.model_checkpoint.monitor == monitor
    assert cfg.callbacks.early_stopping.monitor == monitor
