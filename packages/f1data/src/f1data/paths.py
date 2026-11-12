"""Where the archive and the package configuration live."""

import os
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def repository_root(start=None):
    """The monorepo root: the nearest ancestor holding both `packages/` and `.git`."""
    here = Path(start or __file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "packages").is_dir() and (candidate / ".git").exists():
            return candidate
    return None


def default_data_root():
    """`F1_DATA_ROOT` if set, otherwise `<repository>/data`.

    Outside a checkout there is no safe default, so the caller must say where the
    archive is instead of it being created somewhere unexpected.
    """
    configured = os.environ.get("F1_DATA_ROOT")
    if configured:
        return Path(configured)
    root = repository_root()
    if root is None:
        raise RuntimeError("Cannot locate the repository; set F1_DATA_ROOT to the archive directory.")
    return root / "data"
