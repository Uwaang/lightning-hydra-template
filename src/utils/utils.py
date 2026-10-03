import warnings
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def _metric_as_float(metric_dict: dict[str, Any], metric_name: str) -> float:
    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure the metric name logged by the task is correct and the HPO config "
            "references the same key."
        )

    raw_value = metric_dict[metric_name]
    if hasattr(raw_value, "item"):
        raw_value = raw_value.item()
    try:
        metric_value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Metric {metric_name!r} is not scalar-convertible: {raw_value!r}") from exc

    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")
    return metric_value


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """Safely retrieve one scalar metric for single-objective optimization."""
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None
    return _metric_as_float(metric_dict, metric_name)


def get_metric_values(
    metric_dict: dict[str, Any],
    metric_names: list[str] | tuple[str, ...] | None,
) -> tuple[float, ...] | None:
    """Safely retrieve ordered scalar metrics for multi-objective optimization."""
    if not metric_names:
        log.info("Metric names are empty! Skipping multi-objective metric retrieval...")
        return None
    return tuple(_metric_as_float(metric_dict, name) for name in metric_names)
