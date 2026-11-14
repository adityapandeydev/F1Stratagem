"""Fitted settings saved per model and season, reused by every later run.

A season's settings are fitted on data before 1 January of that season, so they
are the same at every cutoff and in every run until the code or the data
changes. The key covers everything a fit reads:

- the model: its name and options;
- the start: the settings the fit starts from (the previous season's fit, so
  the chain of seasons is part of the key);
- the code: the source of the modules a fit runs through, the native module
  binary, and the Python, NumPy, SciPy and pandas versions;
- the data: the SHA-256 of every reference table the archive reads, and of the
  normalized tables of every session before the season (from the manifests f1data
  writes; no session rows are read and no later season is touched).

Fingerprints are taken when the cache is created, before the run reads code or
data, and taken again before a fit is written: if either changed during the fit,
the fit is used but not saved. A cached fit is the fit itself, written as JSON
(floats round-trip exactly), so a run that reads it forecasts exactly as one
that fits. Two processes fitting the same key wait for each other (a lock file)
instead of fitting twice. Writing the cache never fails a run.
"""

import hashlib
import json
import os
import platform
import re
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

CODE_MODULES = ("pace.py", "program.py", "rules.py", "archive.py", "fitcache.py", "track.py", "circuit_layouts.csv")
REFERENCE_TABLES = ("events", "sessions", "entrants", "qualifying_history")
READ_FILES = ("normalized/", "source/car_data/", "source/pos_data/")   # session files a fit can read
LOCK_WAIT_SECONDS = 5.0
STALE_LOCK_SECONDS = 24 * 3600
WRITE_ATTEMPTS = 10


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def code_fingerprint():
    import numpy
    import pandas
    import predictor_native
    import scipy
    package = Path(__file__).resolve().parent
    parts = [(name, _sha((package / name).read_bytes())) for name in CODE_MODULES]
    native = Path(predictor_native.__file__).resolve().parent
    binaries = sorted(p for p in native.iterdir() if p.suffix in (".pyd", ".so", ".dll"))
    parts += [(p.name, _sha(p.read_bytes())) for p in binaries]
    parts += [("python", sys.version), ("machine", platform.machine()), ("numpy", numpy.__version__),
              ("scipy", scipy.__version__), ("pandas", pandas.__version__)]
    return _sha(json.dumps(parts).encode())


def data_fingerprint(root, season):
    """Reference tables and the normalized tables of every session in the seasons
    before `season`, by their SHA-256 in the f1data manifests."""
    reference = json.loads((root / "reference" / "manifest.json").read_text(encoding="utf-8")).get("files", {})
    parts = [(name, reference[name]["sha256"]) for name in REFERENCE_TABLES if name in reference]
    sessions = root / "sessions"
    if sessions.exists():
        for folder in sorted(p for p in sessions.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name) < season):
            for manifest in sorted(folder.glob("*/manifest.json")):
                files = json.loads(manifest.read_text(encoding="utf-8")).get("files", {})
                parts.append((manifest.parent.name, sorted((name, entry.get("sha256")) for name, entry in files.items()
                                                           if name.startswith(READ_FILES))))
    return _sha(json.dumps(parts).encode())


def hyper_dict(hyper):
    return None if hyper is None else asdict(hyper)


class FitCache:
    def __init__(self, folder):
        self.folder = Path(folder)
        self._code = code_fingerprint()
        self._data = {}

    def prime(self, root, seasons):
        """Take the data fingerprints of `seasons` now, before the run reads the data."""
        for season in seasons:
            self._data.setdefault(season, data_fingerprint(root, season))

    def key(self, model, season, start, qualifying_start, root, base=None):
        """`base`: in a screen, the base model's fitted settings for the season."""
        self.prime(root, [season])
        content = {"model": model.name, "options": model.options(), "season": season,
                   "start": hyper_dict(start), "qualifying_start": hyper_dict(qualifying_start),
                   "base": hyper_dict(base),
                   "code": self._code, "data": self._data[season]}
        return _sha(json.dumps(content, sort_keys=True).encode())

    def unchanged(self, root, season):
        return code_fingerprint() == self._code and data_fingerprint(root, season) == self._data[season]

    def path(self, model, season, key):
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model.name)      # a valid folder name on every system
        return self.folder / safe / f"{season}-{key[:24]}.json"

    def load(self, path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def get_or_fit(self, path, fit, still_valid=lambda: True):
        """(entry, cached): the saved entry at `path`, else `fit()` (a JSON-able dict),
        saved there if `still_valid()`. Another process fitting the same entry is waited
        for, not duplicated."""
        entry = self.load(path)
        if entry is not None:
            return entry, True
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(".lock")
        owner = f"{os.getpid()}-{time.time()}"
        announced = False
        while True:
            try:
                handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(handle, owner.encode())
                os.close(handle)
                break
            except (FileExistsError, PermissionError):    # held, or being deleted (Windows)
                entry = self.load(path)
                if entry is not None:
                    return entry, True
                try:
                    if time.time() - lock.stat().st_mtime > STALE_LOCK_SECONDS:
                        lock.unlink()
                        continue
                except OSError:
                    continue
                if not announced:
                    print(f"waiting for another process fitting {path.name} ({lock})", flush=True)
                    announced = True
                time.sleep(LOCK_WAIT_SECONDS)
        try:
            entry = self.load(path)          # written while we waited for the lock
            if entry is not None:
                return entry, True
            entry = fit()
            if still_valid():
                self._write(path, entry)
            else:
                warnings.warn(f"code or data changed during the fit; {path.name} not saved")
            return entry, False
        finally:
            try:
                if lock.read_text() == owner:
                    lock.unlink()
            except OSError:
                pass

    def _write(self, path, entry):
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(entry, indent=1), encoding="utf-8")
            for attempt in range(WRITE_ATTEMPTS):
                try:
                    os.replace(temporary, path)
                    return
                except PermissionError:      # antivirus or a reader holds the file (Windows)
                    time.sleep(0.5 * (attempt + 1))
            warnings.warn(f"could not save {path.name}; the fit is used but not cached")
        except OSError as error:
            warnings.warn(f"could not save {path.name}: {error}")
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass
