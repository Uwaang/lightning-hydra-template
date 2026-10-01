from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers import Logger

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _mlflow_handles(logger: Logger) -> tuple[Any, str] | None:
    """Return MLflow client/run id without importing the optional mlflow package."""
    if logger.__class__.__name__ != "MLFlowLogger":
        return None

    client = getattr(logger, "experiment", None)
    run_id = getattr(logger, "run_id", None)
    if client is None or not isinstance(run_id, str):
        return None
    return client, run_id


def log_metrics_to_loggers(
    loggers: Sequence[Logger],
    metrics: Mapping[str, float],
    *,
    step: int,
) -> None:
    """Log summary metrics through every configured Lightning logger."""
    for logger in loggers:
        logger.log_metrics(dict(metrics), step=step)


def publish_mlflow_artifacts(
    loggers: Sequence[Logger],
    artifact_groups: Mapping[str, Sequence[str | Path]],
) -> None:
    """Upload files/directories to any configured Lightning MLflow logger."""
    for logger in loggers:
        handles = _mlflow_handles(logger)
        if handles is None:
            continue
        client, run_id = handles

        for artifact_path, local_paths in artifact_groups.items():
            for local_path in local_paths:
                path = Path(local_path)
                if not path.exists():
                    log.warning(f"Skipping missing MLflow artifact: {path}")
                    continue
                if path.is_dir():
                    client.log_artifacts(run_id, str(path), artifact_path=artifact_path)
                else:
                    client.log_artifact(run_id, str(path), artifact_path=artifact_path)


def log_prediction_table_to_mlflow(
    loggers: Sequence[Logger],
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact_file: str,
) -> None:
    """Log a directly viewable prediction table to MLflow when available."""
    if not rows:
        return

    preferred_columns = [
        "sample_index",
        "sample_id",
        "target",
        "pred",
        "confidence",
        "correct",
        "top_k",
    ]
    columns = [column for column in preferred_columns if any(column in row for row in rows)]
    data = [[row.get(column) for column in columns] for row in rows]

    for logger in loggers:
        handles = _mlflow_handles(logger)
        if handles is None:
            continue
        client, run_id = handles
        client.log_table(
            run_id,
            data={"columns": columns, "data": data},
            artifact_file=artifact_file,
        )
