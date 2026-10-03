from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TrainConfig:
    """Typed schema for stable top-level training options.

    Dynamic Hydra component groups stay typed as ``Any`` so existing ``_target_`` composition
    remains flexible while the stable runtime flags are validated.
    """

    task_name: str = "train"
    tags: list[str] = field(default_factory=lambda: ["dev"])
    train: bool = True
    test: bool = True
    save_state_dict: bool = False
    ckpt_path: str | None = None
    seed: int | None = None
    optimized_metric: str | None = None
    optimized_metrics: list[str] = field(default_factory=list)

    data: Any = MISSING
    model: Any = MISSING
    callbacks: Any = None
    logger: Any = None
    trainer: Any = MISSING
    paths: Any = MISSING
    extras: Any = MISSING


@dataclass
class EvalConfig:
    """Typed schema for stable top-level evaluation options."""

    task_name: str = "eval"
    tags: list[str] = field(default_factory=lambda: ["dev"])
    predict: bool = False
    ckpt_path: str = MISSING

    data: Any = MISSING
    model: Any = MISSING
    logger: Any = None
    trainer: Any = MISSING
    paths: Any = MISSING
    extras: Any = MISSING


_REGISTERED = False


def register_configs() -> None:
    """Register structured schemas used by the file-based Hydra configs."""
    global _REGISTERED
    if _REGISTERED:
        return

    config_store = ConfigStore.instance()
    config_store.store(group="schema", name="train", node=TrainConfig, package="_global_")
    config_store.store(group="schema", name="eval", node=EvalConfig, package="_global_")
    _REGISTERED = True
