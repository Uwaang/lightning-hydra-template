from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from src.utils.image_diagnostics import (
    save_classification_image_grid,
    save_confident_error_gallery,
    select_confident_errors,
)


def test_select_confident_errors_prioritizes_high_confidence() -> None:
    rows = [
        {"sample_index": 0, "correct": False, "confidence": 0.7},
        {"sample_index": 1, "correct": True, "confidence": 0.99},
        {"sample_index": 2, "correct": False, "confidence": 0.95},
    ]
    selected = select_confident_errors(rows, max_images=2)
    assert [row["sample_index"] for row in selected] == [2, 0]


def test_save_classification_image_grid_rgb(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    path = save_classification_image_grid(
        torch.rand(4, 3, 8, 8),
        torch.tensor([0, 1, 0, 1]),
        tmp_path / "grid.png",
        class_names=["zero", "one"],
        predictions=torch.tensor([0, 0, 0, 1]),
        confidences=torch.tensor([0.9, 0.8, 0.7, 0.6]),
        max_images=4,
    )
    assert path is not None
    assert path.is_file()
    assert path.stat().st_size > 0


def test_save_confident_error_gallery_reads_only_selected_samples(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    dataset = TensorDataset(
        torch.rand(4, 3, 8, 8),
        torch.tensor([0, 1, 0, 1]),
    )
    rows = [
        {"sample_index": 0, "target": 0, "pred": 1, "correct": False, "confidence": 0.8},
        {"sample_index": 1, "target": 1, "pred": 1, "correct": True, "confidence": 0.9},
        {"sample_index": 2, "target": 0, "pred": 1, "correct": False, "confidence": 0.95},
        {"sample_index": 3, "target": 1, "pred": 1, "correct": True, "confidence": 0.7},
    ]
    path = save_confident_error_gallery(
        rows,
        dataset,
        tmp_path / "errors.png",
        class_names=["zero", "one"],
        max_images=2,
    )
    assert path is not None
    assert path.is_file()
