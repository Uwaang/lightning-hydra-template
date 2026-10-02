import platform
from importlib.metadata import PackageNotFoundError, version


def _package_available(package_name: str) -> bool:
    """Check whether an installed distribution is available."""
    try:
        version(package_name)
    except PackageNotFoundError:
        return False
    return True


_IS_WINDOWS = platform.system() == "Windows"
_SH_AVAILABLE = not _IS_WINDOWS and _package_available("sh")
_OPTUNA_SWEEPER_AVAILABLE = _package_available("hydra-optuna-sweeper")
