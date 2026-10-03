from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def _images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _record(path: Path, root: Path, label: int) -> dict[str, object]:
    return {"path": path.relative_to(root).as_posix(), "label": label}


def prepare(source: Path, seed: int) -> None:
    train_root = source / "train"
    official_val_root = source / "val"
    if not train_root.is_dir() or not official_val_root.is_dir():
        raise FileNotFoundError(f"Expected Imagewoof train/ and val/ directories under {source}")

    classes = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if len(classes) != 10:
        raise ValueError(f"Expected 10 Imagewoof classes, found {len(classes)}")

    class_to_idx = {name: index for index, name in enumerate(classes)}
    train_records: list[dict[str, object]] = []
    val_records: list[dict[str, object]] = []
    test_records: list[dict[str, object]] = []

    for class_name, label in class_to_idx.items():
        train_records.extend(
            _record(path, source, label) for path in _images(train_root / class_name)
        )

        candidates = _images(official_val_root / class_name)
        rng = random.Random(seed + label)  # nosec B311 - deterministic dataset split
        rng.shuffle(candidates)
        split = len(candidates) // 2
        val_records.extend(_record(path, source, label) for path in candidates[:split])
        test_records.extend(_record(path, source, label) for path in candidates[split:])

    manifests = source / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    payloads = {
        "train.json": train_records,
        "val.json": val_records,
        "test.json": test_records,
        "predict.json": test_records,
        "classes.json": class_to_idx,
    }
    for name, payload in payloads.items():
        (manifests / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"source={source}")
    print(f"classes={len(classes)}")
    print(f"train={len(train_records)}")
    print(f"val={len(val_records)}")
    print(f"test={len(test_records)}")
    print(f"manifests={manifests}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic Imagewoof-160 benchmark manifests."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Extracted imagewoof2-160 directory containing train/ and val/.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.seed)


if __name__ == "__main__":
    main()
