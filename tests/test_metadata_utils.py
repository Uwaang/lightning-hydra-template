from pathlib import Path

import src.utils.metadata_utils as metadata_utils


class FakeDistribution:
    def __init__(self, name: str, version: str) -> None:
        self.metadata = {"Name": name}
        self.version = version


def test_write_installed_packages_is_sorted_and_pip_independent(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        metadata_utils,
        "distributions",
        lambda: [
            FakeDistribution("z-package", "2.0"),
            FakeDistribution("A-package", "1.0"),
        ],
    )
    path = tmp_path / "packages.log"

    metadata_utils._write_installed_packages(path)

    assert path.read_text(encoding="utf-8") == "A-package==1.0\nz-package==2.0\n"
