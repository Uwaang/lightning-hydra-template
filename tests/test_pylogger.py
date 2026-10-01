import logging

import pytest
from lightning_utilities.core.rank_zero import rank_zero_only

from src.utils.pylogger import RankedLogger


def test_ranked_logger_preserves_standard_logging_args(caplog: pytest.LogCaptureFixture) -> None:
    logger = RankedLogger("test-ranked-standard")
    rank_zero_only.rank = 0

    with caplog.at_level(logging.INFO, logger="test-ranked-standard"):
        logger.info("value=%s", 7)

    assert "value=7" in caplog.text


def test_ranked_logger_filters_custom_rank(caplog: pytest.LogCaptureFixture) -> None:
    logger = RankedLogger("test-ranked-filter")
    rank_zero_only.rank = 1

    with caplog.at_level(logging.INFO, logger="test-ranked-filter"):
        logger.info("hidden", rank=0)
        logger.info("shown", rank=1)

    assert "hidden" not in caplog.text
    assert "shown" in caplog.text


def test_ranked_logger_rejects_invalid_rank() -> None:
    logger = RankedLogger("test-ranked-invalid")
    rank_zero_only.rank = 0

    with pytest.raises(TypeError, match="rank must be an int or None"):
        logger.info("message", rank="zero")
