"""Integrity audit and inventory for the external sources (Jolpica, Open-Meteo, FIA).

Every downloaded file is recorded in its source's manifest with a SHA-256.
`verify` re-hashes them all and also reports files on disk that no manifest
lists, so a run that checks nothing cannot pass. Derived tables are checked
against their table manifests where those record hashes.
"""

import json

from .storage import checksum

SOURCES = {"jolpica": "jolpica", "openmeteo": "open-meteo", "fia": "fia"}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _table_hashes(root):
    path = root / "tables" / "manifest.json"
    if not path.is_file():
        return {}
    files = _load(path).get("files", {})
    return {f"tables/{name}.parquet": info["sha256"] for name, info in files.items() if "sha256" in info}


def recorded_files(source, root):
    """Relative path -> SHA-256 for every file the source's manifests record,
    and the glob patterns that every downloaded file must fall under."""
    if source == "jolpica":
        expected = {}
        for manifest in sorted((root / "raw").glob("*/manifest.json")):
            for name, info in _load(manifest).get("pages", {}).items():
                expected[name] = info["sha256"]
        return {**expected, **_table_hashes(root)}, ["raw/*/**/*.json"]
    if source == "openmeteo":
        entries = _load(root / "manifest.json").get("entries", {})
        expected = {entry["file"]: entry["sha256"] for entry in entries.values() if entry.get("file")}
        return {**expected, **_table_hashes(root)}, ["raw/**/*.json"]
    if source == "fia":
        files = _load(root / "manifest.json").get("files", {})
        return {name: info["sha256"] for name, info in files.items()}, ["pdf/**/*"]
    raise ValueError(f"unknown source {source}")


def verify_source(source, data_root):
    root = data_root / "external" / SOURCES[source]
    expected, patterns = recorded_files(source, root)
    checked, mismatched, missing = 0, [], []
    for name, sha256 in sorted(expected.items()):
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        checked += 1
        if checksum(path) != sha256:
            mismatched.append(name)
    on_disk = {path.relative_to(root).as_posix() for pattern in patterns for path in root.glob(pattern)
               if path.is_file() and path.name != "manifest.json"}
    unlisted = sorted(on_disk - set(expected))
    return {
        "source": source, "files_recorded": len(expected), "files_checked": checked,
        "mismatched": mismatched, "missing": missing, "unlisted": unlisted,
        "ok": checked > 0 and not (mismatched or missing or unlisted),
    }


def report_source(source, data_root):
    root = data_root / "external" / SOURCES[source]
    expected, _ = recorded_files(source, root)
    tables = {}
    manifest = root / "tables" / "manifest.json"
    if manifest.is_file():
        tables = {name: info.get("rows") for name, info in _load(manifest).get("files", {}).items()}
    return {
        "source": source, "root": str(root), "files_recorded": len(expected),
        "bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "table_rows": tables,
    }


def run_source(source, data_root, command):
    """CLI handler: print the result; exit status 1 when verification fails."""
    if command == "verify":
        result = verify_source(source, data_root)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["ok"] else 1
    print(json.dumps(report_source(source, data_root), indent=2), flush=True)
    return 0
