from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.utils.saving_utils import prediction_rows


@dataclass(frozen=True)
class ClassificationReport:
    """Artifacts and summary metrics produced from classification predictions."""

    metrics: dict[str, float]
    rows: list[dict[str, Any]]
    artifacts: dict[str, Path]


def _prediction_batches(predictions: list[Any]) -> list[dict[str, Any]]:
    if not predictions:
        return []
    if isinstance(predictions[0], dict):
        return prediction_rows(predictions)
    if len(predictions) == 1 and isinstance(predictions[0], list):
        return prediction_rows(predictions[0])
    raise ValueError(
        "Classification reporting currently expects exactly one prediction dataloader."
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _class_labels(num_classes: int, class_names: list[str] | None) -> list[str]:
    if class_names is None:
        return [str(index) for index in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError(f"class_names has {len(class_names)} entries, expected {num_classes}.")
    return class_names


def _save_matrix_csv(
    path: Path,
    matrix: torch.Tensor,
    labels: list[str],
    *,
    normalized: bool = False,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target\\pred", *labels])
        for label, row in zip(labels, matrix.tolist(), strict=True):
            values = [f"{float(value):.6f}" for value in row] if normalized else row
            writer.writerow([label, *values])


def _save_confusion_matrix_figure(
    path: Path,
    matrix: torch.Tensor,
    labels: list[str],
    *,
    normalized: bool = False,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    size = max(6.0, min(14.0, 0.7 * len(labels)))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix.numpy(), interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Normalized confusion matrix" if normalized else "Confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    axis.tick_params(axis="x", rotation=45)

    threshold = float(matrix.max().item()) / 2.0 if matrix.numel() else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            raw_value = matrix[row_index, column_index].item()
            text_value = f"{float(raw_value):.2f}" if normalized else str(int(raw_value))
            axis.text(
                column_index,
                row_index,
                text_value,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def save_classification_report(
    predictions: list[Any],
    dirname: str | Path,
    *,
    split: str = "test",
    top_k: int = 3,
    class_names: list[str] | None = None,
) -> ClassificationReport:
    """Create research-friendly classification reports from Lightning predict outputs."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    raw_rows = _prediction_batches(predictions)
    if not raw_rows:
        raise ValueError("Predictions are empty; cannot create a classification report.")

    report_rows: list[dict[str, Any]] = []
    targets: list[int] = []
    predicted: list[int] = []
    inferred_num_classes = 0

    for index, row in enumerate(raw_rows):
        if "preds" not in row or "targets" not in row:
            raise KeyError("Classification predictions must include 'preds' and 'targets'.")

        pred = int(row["preds"])
        target = int(row["targets"])
        logits = row.get("logits")
        confidence: float | None = None
        top_predictions: list[dict[str, float | int]] = []

        if logits is not None:
            logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
            if logits_tensor.ndim != 1:
                raise ValueError("Per-sample logits must be one-dimensional.")
            probabilities = torch.softmax(logits_tensor, dim=0)
            inferred_num_classes = max(inferred_num_classes, int(probabilities.numel()))
            count = min(top_k, int(probabilities.numel()))
            values, indices = torch.topk(probabilities, k=count)
            confidence = float(probabilities[pred].item())
            top_predictions = [
                {"class": int(class_index), "probability": float(probability)}
                for probability, class_index in zip(values.tolist(), indices.tolist(), strict=True)
            ]

        inferred_num_classes = max(inferred_num_classes, pred + 1, target + 1)
        targets.append(target)
        predicted.append(pred)

        report_row: dict[str, Any] = {
            "sample_index": index,
            "target": target,
            "pred": pred,
            "confidence": confidence,
            "correct": pred == target,
            "top_k": json.dumps(top_predictions, ensure_ascii=False),
        }
        if "name" in row and row["name"] is not None:
            report_row["sample_id"] = str(row["name"])
        report_rows.append(report_row)

    labels = _class_labels(inferred_num_classes, class_names)
    confusion_matrix = torch.zeros(
        (inferred_num_classes, inferred_num_classes),
        dtype=torch.int64,
    )
    for target, pred in zip(targets, predicted, strict=True):
        confusion_matrix[target, pred] += 1

    class_metrics: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    supports: list[int] = []

    for class_index, class_name in enumerate(labels):
        true_positive = int(confusion_matrix[class_index, class_index].item())
        support = int(confusion_matrix[class_index].sum().item())
        predicted_count = int(confusion_matrix[:, class_index].sum().item())
        precision = _safe_divide(true_positive, predicted_count)
        recall = _safe_divide(true_positive, support)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        supports.append(support)
        class_metrics.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    total = sum(supports)
    correct = int(torch.diag(confusion_matrix).sum().item())
    weighted_f1 = _safe_divide(
        sum(f1 * support for f1, support in zip(f1_values, supports, strict=True)),
        total,
    )
    metrics = {
        f"{split}/report/accuracy": _safe_divide(correct, total),
        f"{split}/report/macro_precision": sum(precision_values) / len(precision_values),
        f"{split}/report/macro_recall": sum(recall_values) / len(recall_values),
        f"{split}/report/macro_f1": sum(f1_values) / len(f1_values),
        f"{split}/report/weighted_f1": weighted_f1,
    }

    output_dir = Path(dirname) / "reports" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.csv"
    fieldnames = ["sample_index", "sample_id", "target", "pred", "confidence", "correct", "top_k"]
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report_rows)

    class_report_csv = output_dir / "classification_report.csv"
    with class_report_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_metrics[0]))
        writer.writeheader()
        writer.writerows(class_metrics)

    class_report_json = output_dir / "classification_report.json"
    class_report_json.write_text(
        json.dumps(
            {"summary": metrics, "classes": class_metrics},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    confusion_csv = output_dir / "confusion_matrix.csv"
    _save_matrix_csv(confusion_csv, confusion_matrix, labels)

    support = confusion_matrix.sum(dim=1, keepdim=True).clamp_min(1)
    normalized_confusion = confusion_matrix.to(torch.float64) / support
    normalized_confusion_csv = output_dir / "confusion_matrix_normalized.csv"
    _save_matrix_csv(
        normalized_confusion_csv,
        normalized_confusion,
        labels,
        normalized=True,
    )

    artifacts = {
        "predictions_csv": predictions_path,
        "classification_report_csv": class_report_csv,
        "classification_report_json": class_report_json,
        "confusion_matrix_csv": confusion_csv,
        "confusion_matrix_normalized_csv": normalized_confusion_csv,
    }
    confusion_png = _save_confusion_matrix_figure(
        output_dir / "confusion_matrix.png",
        confusion_matrix,
        labels,
    )
    normalized_confusion_png = _save_confusion_matrix_figure(
        output_dir / "confusion_matrix_normalized.png",
        normalized_confusion,
        labels,
        normalized=True,
    )
    if confusion_png is not None:
        artifacts["confusion_matrix_png"] = confusion_png
    if normalized_confusion_png is not None:
        artifacts["confusion_matrix_normalized_png"] = normalized_confusion_png

    return ClassificationReport(metrics=metrics, rows=report_rows, artifacts=artifacts)
