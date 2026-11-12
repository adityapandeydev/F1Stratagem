"""Archive file primitives shared by every source: atomic writes and checksums."""

from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_table(frame, path, metadata):
    """Keep source types (including timedeltas and nulls); never impute."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(frame).reset_index(drop=True).copy()
    for column, value in metadata.items():
        if column in result:
            raise ValueError(f"metadata would overwrite source column {column}")
        result[column] = value
    temporary = path.with_suffix(".parquet.tmp")
    result.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    # Read back before promoting; catches schema/serialization problems.
    restored = pd.read_parquet(temporary)
    if len(restored) != len(result) or list(restored.columns) != list(result.columns):
        raise ValueError(f"Parquet verification failed: {path}")
    temporary.replace(path)
    return {
        "rows": len(result), "bytes": path.stat().st_size, "sha256": checksum(path),
        "schema": {column: str(dtype) for column, dtype in result.dtypes.items()},
        "null_counts": {column: int(n) for column, n in result.isna().sum().items()},
    }


def total_bytes(directory):
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) if directory.exists() else 0
