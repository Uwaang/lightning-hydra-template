import logging
from collections.abc import Mapping
from typing import Any

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize a multi-GPU-friendly command line logger.

        :param name: The name of the logger. Default is __name__.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process.
            Default is False.
        :param extra: Optional contextual information forwarded to logging.LoggerAdapter.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None:
        """Delegate a log call after prefixing the message with the current rank.

        A custom rank=<int> keyword remains supported and filters the record to that process. The
        keyword is removed before delegating to the standard logging implementation.
        """
        if not self.isEnabledFor(level):
            return

        target_rank = kwargs.pop("rank", None)
        if target_rank is not None and not isinstance(target_rank, int):
            raise TypeError("rank must be an int or None")

        msg, processed_kwargs = self.process(msg, kwargs)
        current_rank = getattr(rank_zero_only, "rank", None)
        if current_rank is None:
            raise RuntimeError("The rank_zero_only.rank needs to be set before use")

        if self.rank_zero_only and current_rank != 0:
            return
        if target_rank is not None and current_rank != target_rank:
            return

        prefixed_msg = rank_prefixed_message(str(msg), current_rank)
        self.logger.log(level, prefixed_msg, *args, **processed_kwargs)
