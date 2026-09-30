from __future__ import annotations

import sys
from typing import TextIO


def _make_stream_encoding_safe(stream: TextIO) -> None:
    """Keep the stream encoding but replace characters it cannot encode."""
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return

    encoding = (getattr(stream, "encoding", "") or "").lower()
    if "utf" in encoding:
        return

    reconfigure(errors="replace")


def configure_windows_stdio() -> None:
    """Prevent Rich/Lightning progress output from crashing non-UTF Windows consoles."""
    if sys.platform != "win32":
        return

    _make_stream_encoding_safe(sys.stdout)
    _make_stream_encoding_safe(sys.stderr)
