from __future__ import annotations

import shutil
import subprocess  # nosec B404 - fixed nvidia-smi argv, never shell=True
from time import perf_counter
import torch
from lightning import Callback, LightningModule, Trainer

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _optional_cpu_metrics() -> dict[str, float]:
    try:
        import psutil
    except ImportError:
        return {}

    process = psutil.Process()
    return {
        "system/cpu_percent": float(psutil.cpu_percent(interval=None)),
        "system/process_cpu_percent": float(process.cpu_percent(interval=None)),
        "system/process_rss_mb": process.memory_info().rss / (1024.0**2),
    }


def _single_gpu_nvidia_smi_metrics() -> dict[str, float]:
    if not torch.cuda.is_available() or shutil.which("nvidia-smi") is None:
        return {}

    try:
        result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    if result.returncode != 0 or not result.stdout.strip():
        return {}

    first_line = result.stdout.strip().splitlines()[0]
    try:
        utilization, memory_used, memory_total = (
            float(value.strip()) for value in first_line.split(",")
        )
    except (ValueError, TypeError):
        return {}

    return {
        "system/gpu_utilization_percent": utilization,
        "system/gpu_memory_used_mb": memory_used,
        "system/gpu_memory_total_mb": memory_total,
    }


def collect_system_metrics(
    *,
    include_cuda: bool,
    include_nvidia_smi: bool,
) -> dict[str, float]:
    """Collect lightweight process/CPU/CUDA metrics without requiring extra dependencies."""
    metrics = _optional_cpu_metrics()

    if include_cuda and torch.cuda.is_available():
        device = torch.cuda.current_device()
        metrics.update(
            {
                "system/cuda_memory_allocated_mb": torch.cuda.memory_allocated(device)
                / (1024.0**2),
                "system/cuda_memory_reserved_mb": torch.cuda.memory_reserved(device)
                / (1024.0**2),
                "system/cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated(device)
                / (1024.0**2),
            }
        )
        if include_nvidia_smi:
            metrics.update(_single_gpu_nvidia_smi_metrics())

    return metrics


class ResearchMonitorCallback(Callback):
    """Log epoch durations and lightweight system metrics at epoch boundaries."""

    def __init__(
        self,
        *,
        log_epoch_time: bool = True,
        log_system_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.log_epoch_time = log_epoch_time
        self.log_system_metrics = log_system_metrics
        self._train_epoch_started_at: float | None = None
        self._validation_epoch_started_at: float | None = None

    @staticmethod
    def _log(trainer: Trainer, metrics: dict[str, float]) -> None:
        if (
            not metrics
            or trainer.fast_dev_run
            or trainer.sanity_checking
            or not trainer.is_global_zero
        ):
            return
        for logger_instance in trainer.loggers:
            logger_instance.log_metrics(metrics, step=trainer.global_step)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if self.log_system_metrics:
            # Prime psutil's non-blocking CPU counters when the optional dependency is available.
            _optional_cpu_metrics()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        if self.log_epoch_time:
            self._train_epoch_started_at = perf_counter()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        metrics: dict[str, float] = {}
        if self.log_epoch_time and self._train_epoch_started_at is not None:
            metrics["time/train_epoch_seconds"] = perf_counter() - self._train_epoch_started_at
        if self.log_system_metrics:
            metrics.update(
                collect_system_metrics(
                    include_cuda=trainer.strategy.root_device.type == "cuda",
                    include_nvidia_smi=trainer.num_devices == 1,
                )
            )
        self._log(trainer, metrics)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        del trainer, pl_module
        if self.log_epoch_time:
            self._validation_epoch_started_at = perf_counter()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not self.log_epoch_time or self._validation_epoch_started_at is None:
            return
        self._log(
            trainer,
            {
                "time/validation_epoch_seconds": perf_counter()
                - self._validation_epoch_started_at
            },
        )
