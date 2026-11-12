"""Immutable core acquisition receipts, timestamped when actually recorded locally."""

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from .sessions import CORE_FILES, verified
from .storage import atomic_json, checksum, utcnow


def receipt_valid(directory, identity):
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        return (manifest["status"] == "complete" and manifest["fingerprint"] == identity
                and all((directory / name).is_file() and checksum(directory / name) == info["sha256"]
                        for name, info in manifest["files"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def record_snapshot(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if not directory.is_relative_to(root) or not verified(directory, False):
        raise ValueError("A snapshot requires checksum-verified local core data")
    source = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    session_id = source["session_id"]
    if session_id != directory.name or Path(session_id).name != session_id:
        raise ValueError("Invalid snapshot session identity")
    files = {name: source["files"][name] for name in sorted(CORE_FILES)}
    identity = hashlib.sha256(json.dumps({name: info["sha256"] for name, info in files.items()}, sort_keys=True).encode()).hexdigest()
    output = root / "snapshots" / session_id / identity[:20]
    if (output / "manifest.json").exists() and not receipt_valid(output, identity):
        raise ValueError(f"Existing immutable snapshot is corrupt: {output}")
    if not receipt_valid(output, identity):
        observed = utcnow()
        for name, info in files.items():
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".parquet.tmp")
            shutil.copyfile(directory / name, temporary)
            if checksum(temporary) != info["sha256"]:
                raise ValueError("Source changed while recording a snapshot")
            temporary.replace(destination)
        atomic_json(output / "manifest.json", {
            "schema_version": 1, "status": "complete", "session_id": session_id,
            "fingerprint": identity, "observed_at_utc": observed, "available_at_utc": observed,
            "source_collected_at_utc": source.get("collected_at_utc"),
            "original_publication_time_verified": False, "files": files,
            "note": "Local core observation receipt; not a historical publication timestamp. Telemetry and cache are not copied.",
        })
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    index_path = root / "snapshots/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"schema_version": 1, "receipts": []}
    receipts = {(row["session_id"], row["fingerprint"]): row for row in index["receipts"]}
    receipts[(session_id, identity)] = {
        "session_id": session_id, "fingerprint": identity, "observed_at_utc": manifest["observed_at_utc"],
        "directory": output.relative_to(root).as_posix(), "manifest_sha256": checksum(output / "manifest.json"),
    }
    atomic_json(index_path, {"schema_version": 1, "receipts": sorted(receipts.values(), key=lambda row: (row["observed_at_utc"], row["session_id"]))})
    return output


def snapshot_asof(root, session_id, cutoff):
    root, cutoff = Path(root).resolve(), pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Snapshot cutoff must include a timezone")
    index_path = root / "snapshots/index.json"
    if not index_path.exists():
        return None
    receipts = json.loads(index_path.read_text(encoding="utf-8"))["receipts"]
    eligible = [row for row in receipts if row["session_id"] == session_id and pd.Timestamp(row["observed_at_utc"]) <= cutoff]
    if not eligible:
        return None
    selected = max(eligible, key=lambda row: pd.Timestamp(row["observed_at_utc"]))
    directory = root / selected["directory"]
    if checksum(directory / "manifest.json") != selected["manifest_sha256"] or not receipt_valid(directory, selected["fingerprint"]):
        raise ValueError("Snapshot failed integrity verification")
    return directory
