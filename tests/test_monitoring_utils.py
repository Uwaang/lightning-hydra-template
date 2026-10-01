import pytest

import src.utils.monitoring_utils as monitoring_utils
from src.utils.monitoring_utils import ResearchMonitorCallback


class FakeLogger:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float], int]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.logged.append((metrics, step))


class FakeTrainer:
    fast_dev_run = False
    sanity_checking = False
    is_global_zero = True
    global_step = 12
    num_devices = 1

    def __init__(self, logger: FakeLogger) -> None:
        self.loggers = [logger]


def test_research_monitor_logs_epoch_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.0, 12.5])
    monkeypatch.setattr(monitoring_utils, "perf_counter", lambda: next(times))
    monkeypatch.setattr(monitoring_utils, "collect_system_metrics", lambda **kwargs: {})

    logger = FakeLogger()
    trainer = FakeTrainer(logger)
    callback = ResearchMonitorCallback(log_epoch_time=True, log_system_metrics=False)

    callback.on_train_epoch_start(trainer, None)
    callback.on_train_epoch_end(trainer, None)

    assert logger.logged == [({"time/train_epoch_seconds": 2.5}, 12)]
