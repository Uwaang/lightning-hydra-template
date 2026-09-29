from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a mapping or list-style JSON image manifest."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        return [{"path": key, "label": value} for key, value in data.items()]

    if isinstance(data, list):
        if not all(isinstance(item, dict) and "path" in item for item in data):
            raise ValueError("List manifests require a 'path' field for every item.")
        return data

    raise TypeError("Manifest must be a JSON object or a list of objects.")


def discover_image_paths(
    directories: Iterable[str | Path],
    recursive: bool = True,
    extensions: set[str] | None = None,
) -> list[Path]:
    """Discover image files under one or more directories."""
    allowed = {suffix.lower() for suffix in (extensions or DEFAULT_IMAGE_EXTENSIONS)}
    paths: list[Path] = []

    for directory in directories:
        directory = Path(directory)
        iterator = directory.rglob("*") if recursive else directory.glob("*")
        paths.extend(
            path for path in iterator if path.is_file() and path.suffix.lower() in allowed
        )

    return sorted(paths)
