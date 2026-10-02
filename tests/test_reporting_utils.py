import csv
import json
from pathlib import Path

import pytest
import torch

from src.utils.mlflow_utils import (
    log_prediction_table_to_mlflow,
    publish_mlflow_artifacts,
    select_prediction_table_rows,
)
from src.utils.reporting_utils import save_classification_report


def _example_predictions() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "logits": torch.tensor([[4.0, 1.0], [1.0, 4.0]]),
            "preds": torch.tensor([0, 1]),
            "targets": torch.tensor([0, 0]),
        },
        {
            "logits": torch.tensor([[0.5, 3.0]]),
            "preds": torch.tensor([1]),
            "targets": torch.tensor([1]),
        },
    ]


def test_save_classification_report(tmp_path: Path) -> None:
    report = save_classification_report(
        _example_predictions(),
        tmp_path,
        split="test",
        top_k=2,
        class_names=["negative", "positive"],
    )

    assert report.metrics["test/report/accuracy"] == pytest.approx(2 / 3)
    assert report.metrics["test/report/macro_recall"] == pytest.approx(0.75)
    assert report.metrics["test/report/macro_f1"] == pytest.approx(2 / 3)

    confusion_path = report.artifacts["confusion_matrix_csv"]
    with confusion_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["target\\pred", "negative", "positive"],
        ["negative", "1", "1"],
        ["positive", "0", "1"],
    ]

    normalized_path = report.artifacts["confusion_matrix_normalized_csv"]
    with normalized_path.open(encoding="utf-8", newline="") as handle:
        normalized_rows = list(csv.reader(handle))
    assert normalized_rows == [
        ["target\\pred", "negative", "positive"],
        ["negative", "0.500000", "0.500000"],
        ["positive", "0.000000", "1.000000"],
    ]

    payload = json.loads(
        report.artifacts["classification_report_json"].read_text(encoding="utf-8")
    )
    assert payload["classes"][0]["support"] == 2
    assert payload["classes"][1]["precision"] == pytest.approx(0.5)

    assert len(report.rows) == 3
    assert report.rows[0]["correct"] is True
    assert report.rows[1]["correct"] is False
    assert report.rows[0]["confidence"] > 0.9


class FakeMlflowClient:
    def __init__(self) -> None:
        self.artifacts: list[tuple[str, str, str | None]] = []
        self.tables: list[tuple[str, dict, str]] = []

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((run_id, local_path, artifact_path))

    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((run_id, local_dir, artifact_path))

    def log_table(self, run_id: str, data: dict, artifact_file: str) -> None:
        self.tables.append((run_id, data, artifact_file))


class MLFlowLogger:
    def __init__(self, client: FakeMlflowClient) -> None:
        self.experiment = client
        self.run_id = "run-123"


def test_mlflow_artifact_and_prediction_table_publish(tmp_path: Path) -> None:
    client = FakeMlflowClient()
    logger = MLFlowLogger(client)
    artifact = tmp_path / "report.json"
    artifact.write_text("{}", encoding="utf-8")

    publish_mlflow_artifacts([logger], {"reports/test": [artifact]})
    log_prediction_table_to_mlflow(
        [logger],
        [{"sample_index": 0, "target": 1, "pred": 1, "correct": True}],
        artifact_file="reports/test/predictions_table.json",
    )

    assert client.artifacts == [("run-123", str(artifact), "reports/test")]
    assert client.tables[0][0] == "run-123"
    assert client.tables[0][2] == "reports/test/predictions_table.json"
    assert client.tables[0][1] == {
        "sample_index": [0],
        "target": [1],
        "pred": [1],
        "correct": [True],
    }


def test_select_prediction_table_rows_prioritizes_errors_then_uncertain() -> None:
    rows = [
        {"sample_index": 0, "correct": True, "confidence": 0.99},
        {"sample_index": 1, "correct": False, "confidence": 0.70},
        {"sample_index": 2, "correct": True, "confidence": 0.40},
        {"sample_index": 3, "correct": False, "confidence": 0.95},
        {"sample_index": 4, "correct": True, "confidence": 0.60},
    ]

    selected = select_prediction_table_rows(rows, max_rows=3)

    assert [row["sample_index"] for row in selected] == [3, 1, 2]
