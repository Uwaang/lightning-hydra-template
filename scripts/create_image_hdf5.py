#!/usr/bin/env python

import argparse
from pathlib import Path

from src.data.components.hdf5 import write_hdf5_images
from src.data.components.paths import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack manifest images into an HDF5 file.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    samples = load_manifest(args.manifest)
    image_paths = [
        (
            str(sample["path"]).replace("\\", "/").lstrip("/"),
            args.root_dir / str(sample["path"]),
        )
        for sample in samples
    ]
    write_hdf5_images(args.output, image_paths)
    print(args.output)


if __name__ == "__main__":
    main()
