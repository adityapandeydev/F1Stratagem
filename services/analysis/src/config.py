"""Analysis service settings.

Values come from the environment, with the repository's `.env` loaded first.
Relative paths resolve against this service's directory, so the service behaves
the same whichever working directory it is started from. The service keeps its
own cache under `services/analysis/data/`; it never writes the shared archive.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]

load_dotenv(REPO_ROOT / ".env", override=False)


def _service_path(name: str, default: str) -> Path:
    path = Path(os.environ.get(name) or default)
    return path if path.is_absolute() else SERVICE_ROOT / path


FASTF1_CACHE_DIR = _service_path("ANALYSIS_FASTF1_CACHE_DIR", "data/fastf1_cache")
TELEMETRY_DIR = _service_path("ANALYSIS_TELEMETRY_DIR", "data/telemetry")
