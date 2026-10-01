from __future__ import annotations

import os
import platform
import shutil
import subprocess  # nosec B404 - fixed argv metadata commands, never shell=True
import sys
from collections.abc import Sequence
from importlib.metadata import distributions
from pathlib import Path

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def run_command(command: Sequence[str], cwd: str | Path | None = None) -> str:
    """Run a metadata command without invoking a shell."""
    try:
        result = subprocess.run(  # nosec B603 - argv list is executed without a shell
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"$ {' '.join(command)}\n{exc}\n"

    output = result.stdout
    if result.stderr:
        output += result.stderr
    return f"$ {' '.join(command)}\n{output}\n"


def _write_command_group(path: Path, commands: Sequence[Sequence[str]], cwd: Path) -> None:
    path.write_text(
        "\n".join(run_command(command, cwd=cwd) for command in commands),
        encoding="utf-8",
    )


def _write_installed_packages(path: Path) -> None:
    packages = {
        distribution.metadata.get("Name", "").strip(): distribution.version
        for distribution in distributions()
        if distribution.metadata.get("Name")
    }
    path.write_text(
        "\n".join(
            f"{name}=={version}"
            for name, version in sorted(packages.items(), key=lambda item: item[0].lower())
        )
        + "\n",
        encoding="utf-8",
    )


@rank_zero_only
def log_run_metadata(cfg: DictConfig) -> Path:
    """Capture environment, git state, GPU details, source, and configs for a run."""
    root_dir = Path(cfg.paths.root_dir)
    target_dir = Path(cfg.paths.output_dir) / "metadata"
    target_dir.mkdir(parents=True, exist_ok=True)

    _write_installed_packages(target_dir / "packages.log")
    (target_dir / "runtime.log").write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"executable={sys.executable}",
                f"platform={platform.platform()}",
                f"machine={platform.machine()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_command_group(
        target_dir / "git.log",
        [
            ["git", "describe", "--tags", "--long", "--dirty", "--always"],
            ["git", "branch", "--verbose", "--verbose", "--all"],
            ["git", "remote", "--verbose"],
            ["git", "status", "--short", "--branch"],
        ],
        cwd=root_dir,
    )

    gpu_lines = [
        f"{key}={value}"
        for key, value in sorted(os.environ.items())
        if key.startswith(("CUDA", "NVIDIA", "NCCL"))
    ]
    if shutil.which("nvidia-smi"):
        gpu_lines.append(run_command(["nvidia-smi"], cwd=root_dir))
    (target_dir / "gpu.log").write_text("\n".join(gpu_lines), encoding="utf-8")

    for directory_name in ("src", "configs"):
        source = root_dir / directory_name
        destination = target_dir / directory_name
        if source.is_dir() and not destination.exists():
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )

    log.info(f"Saved run metadata to: {target_dir}")
    return target_dir
