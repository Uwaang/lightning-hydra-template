from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404 - fixed argv metadata commands, never shell=True
import sys
from collections.abc import Sequence
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


@rank_zero_only
def log_run_metadata(cfg: DictConfig) -> Path:
    """Capture environment, git state, GPU details, source, and configs for a run."""
    root_dir = Path(cfg.paths.root_dir)
    target_dir = Path(cfg.paths.output_dir) / "metadata"
    target_dir.mkdir(parents=True, exist_ok=True)

    _write_command_group(
        target_dir / "pip.log",
        [[sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"]],
        cwd=root_dir,
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
            shutil.copytree(source, destination)

    log.info(f"Saved run metadata to: {target_dir}")
    return target_dir
