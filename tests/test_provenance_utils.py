from pathlib import Path

from src.utils.provenance_utils import build_dataset_provenance, save_dataset_provenance


class DummyDataModule:
    def __init__(self, data_dir: str) -> None:
        self.hparams = {
            "data_dir": data_dir,
            "batch_size": 64,
            "num_workers": 4,
            "split": [8, 1, 1],
        }
        self.data_train = list(range(8))
        self.data_val = list(range(1))
        self.data_test = list(range(1))

    def dataset_provenance(self) -> dict:
        return {
            "dataset": "dummy-v1",
            "split_seed": 42,
            "split_lengths": [8, 1, 1],
            "transform": "normalize-v1",
        }


def test_dataset_fingerprint_is_machine_path_independent() -> None:
    first = build_dataset_provenance(DummyDataModule("C:/machine-a/data"))
    second = build_dataset_provenance(DummyDataModule("/machine-b/data"))

    assert first["fingerprint"] == second["fingerprint"]
    assert first["split_sizes"] == {"train": 8, "val": 1, "test": 1}
    assert "raw sample bytes are not hashed" in first["fingerprint_scope"]


def test_save_dataset_provenance(tmp_path: Path) -> None:
    path, payload = save_dataset_provenance(DummyDataModule("data"), tmp_path)

    assert path == tmp_path / "dataset.json"
    assert path.is_file()
    assert payload["dataset_identity"]["dataset"] == "dummy-v1"
