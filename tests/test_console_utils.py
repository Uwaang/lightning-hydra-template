from src.utils.console_utils import _make_stream_encoding_safe


class FakeStream:
    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self.errors = "strict"

    def reconfigure(self, **kwargs: str) -> None:
        if "errors" in kwargs:
            self.errors = kwargs["errors"]


def test_non_utf_stream_uses_replacement_errors() -> None:
    stream = FakeStream("cp949")
    _make_stream_encoding_safe(stream)
    assert stream.encoding == "cp949"
    assert stream.errors == "replace"


def test_utf_stream_is_unchanged() -> None:
    stream = FakeStream("utf-8")
    _make_stream_encoding_safe(stream)
    assert stream.errors == "strict"
