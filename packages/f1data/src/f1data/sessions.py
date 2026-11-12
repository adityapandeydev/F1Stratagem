"""FastF1 session archive: discovery, ingestion, verification and readers (2018+).

One directory per session under `sessions/<year>/<session-id>/`, described by a
manifest that records every file with its SHA-256. Telemetry is stored once, in
the source layer, and re-joined to session and driver columns on read.
"""

import json
from pathlib import Path
import threading
import time

import fastf1
import pandas as pd
import psutil
import requests
from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError

from .paths import CONFIG_DIR
from .storage import atomic_json, checksum, total_bytes, utcnow, write_table

SCHEMA_VERSION = 1
TABLES = {
    "laps": "laps", "results": "results", "weather": "weather_data",
    "track_status": "track_status", "session_status": "session_status",
    "race_control": "race_control_messages",
}
REQUIRED = {"laps": ["DriverNumber", "LapNumber"], "results": ["DriverNumber"]}
# Testing has no classification, so an empty results table there is expected, not missing.
TESTING_NOT_APPLICABLE = {"results"}
CORE_FILES = {f"{layer}/{name}.parquet" for layer in ("source", "normalized") for name in TABLES}
TELEMETRY_STREAMS = ("car_data", "pos_data")


class ProviderAccessError(RuntimeError):
    """The source rejected the request; retries cannot supply credentials."""


class RateLimitReached(ProviderAccessError):
    """FastF1's hourly request budget is spent; the same session can be retried later."""


def validate(name, frame):
    for column in REQUIRED.get(name, []):
        if column not in frame or frame[column].isna().any():
            raise ValueError(f"{name}: missing required key {column}")
    keys = REQUIRED.get(name)
    if keys and frame.duplicated(keys).any():
        raise ValueError(f"{name}: duplicate keys {keys}")
    if name == "laps" and (frame.LapNumber <= 0).any():
        raise ValueError("laps: non-positive lap number")


def core_recorded_complete(manifest):
    return (manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("status") in ("complete", "partial")
            and CORE_FILES.issubset(manifest.get("files", {})))


def verified(directory, telemetry):
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not core_recorded_complete(manifest):
            return False
        if telemetry and (manifest["status"] != "complete" or not manifest["telemetry"]):
            return False
        files = manifest["files"] if telemetry else {name: manifest["files"][name] for name in CORE_FILES}
        return bool(manifest["files"]) and all(
            (directory / name).is_file() and checksum(directory / name) == info["sha256"]
            for name, info in files.items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


# Failures caused by our side (request budget, credentials, local network) say
# nothing about the source data, so those sessions are always attempted again.
TRANSIENT_ERRORS = {"RateLimitExceededError", "ProviderAccessError", "ConnectionError",
                    "Timeout", "ReadTimeout", "ConnectTimeout", "ChunkedEncodingError"}


def unresolved_attempt(directory, telemetry):
    """Why an earlier attempt left this session unresolved, or None.

    Unresolved means the source was reached but returned incomplete data: a failed
    load, missing tables, or missing telemetry streams when telemetry was attempted.
    Normal runs skip these so known gaps don't spend the hourly request budget on
    every run; `--retry-failed` attempts them again. A session whose telemetry was
    never requested is not unresolved: requesting it now is a first attempt.
    """
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    status = manifest.get("status")
    error = (manifest.get("error") or {}).get("type")
    if status == "failed":
        if error == "SessionUnavailable" and LIVE_PROBE_MARK not in (manifest.get("error") or {}).get("message", ""):
            return None
        return None if error in TRANSIENT_ERRORS else f"failed: {error}"
    if status == "partial" and (manifest.get("telemetry") or not telemetry):
        missing = manifest.get("missing_tables", [])
        return f"partial: {len(missing)} missing tables/streams"
    return None


def present(directory, telemetry):
    """Fast skip check: the manifest records the tier as complete and every file
    exists with its recorded size. Content checksums are left to verified(), which
    `f1data fastf1 verify` runs; a skip decision does not need them."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not core_recorded_complete(manifest):
            return False
        if telemetry and (manifest["status"] != "complete" or not manifest["telemetry"]):
            return False
        files = manifest["files"] if telemetry else {name: manifest["files"][name] for name in CORE_FILES}
        return bool(files) and all(
            (directory / name).is_file() and (directory / name).stat().st_size == info["bytes"]
            for name, info in files.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def verify_all(root):
    """Full checksum audit of every session (no network). Returns the sessions
    whose recorded files are missing or no longer match their SHA-256. Tables a
    source never had (a partial session's missing_tables) are not corruption;
    they are listed in missed-sessions.json instead."""
    problems = []
    for path in sorted((root / "sessions").glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") not in ("complete", "partial"):
            continue
        files = manifest.get("files", {})
        if not files or not all((path.parent / name).is_file() and checksum(path.parent / name) == info["sha256"]
                                for name, info in files.items()):
            problems.append(manifest.get("session_id", path.parent.name))
    return problems


def export_session(session, directory, metadata, telemetry):
    """Core tables are written as source + normalized. Telemetry streams are written
    once (source layer); session/driver metadata is re-attached on read, see
    read_telemetry(). Normalized telemetry would only repeat constant columns."""
    files, missing, not_applicable = {}, [], []
    collected_at = utcnow()
    tags = {**metadata, "collected_at_utc": collected_at, "schema_version": SCHEMA_VERSION}
    testing = bool(metadata.get("is_testing"))

    def save(name, frame, layers=("source", "normalized"), extra=None):
        if frame is None or frame.empty:
            if testing and name in TESTING_NOT_APPLICABLE:
                not_applicable.append(name)
                frame = frame if frame is not None and len(frame.columns) else pd.DataFrame(
                    {key: pd.Series(dtype=str) for key in REQUIRED.get(name, [])})
            else:
                missing.append(name)
                if name in REQUIRED:
                    raise ValueError(f"{name}: required table is empty")
                return
        validate(name, frame)
        for layer in layers:
            values = {} if layer == "source" else {**tags, **(extra or {})}
            relative = f"{layer}/{name}.parquet"
            files[relative] = write_table(frame, directory / relative, values)

    for name, attribute in TABLES.items():
        try:
            frame = getattr(session, attribute)
        except (AttributeError, DataNotLoadedError):
            frame = None
        save(name, frame)
    if telemetry:
        drivers = [str(n) for n in session.results.DriverNumber.dropna().unique()]
        if not drivers:
            drivers = [str(n) for n in session.laps.DriverNumber.dropna().unique()]
        for attribute in TELEMETRY_STREAMS:
            try:
                streams = getattr(session, attribute)
            except (AttributeError, DataNotLoadedError):
                streams = {}
            for driver in drivers:
                save(f"{attribute}/driver-{driver}", streams.get(driver), layers=("source",))
    manifest = {
        **tags, "status": "partial" if missing else "complete", "telemetry": telemetry,
        "telemetry_layout": "source_only" if telemetry else None,
        "fastf1_version": fastf1.__version__, "files": files, "missing_tables": missing,
        "not_applicable_tables": not_applicable,
        "source": "FastF1 historical session API; processed provider extracts, not wire captures",
        "availability_note": "Collection time is not original historical publication time. Results and lap accuracy flags may include retrospective corrections.",
    }
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def read_telemetry(directory, stream, driver):
    """One driver's native stream with the session metadata columns that older
    layouts stored in normalized/. Works for both source-only and legacy layouts."""
    if stream not in TELEMETRY_STREAMS:
        raise ValueError(f"unknown telemetry stream {stream}")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    relative = f"source/{stream}/driver-{driver}.parquet"
    info = manifest.get("files", {}).get(relative)
    if info is None:
        raise FileNotFoundError(f"{relative} is not recorded in {directory / 'manifest.json'}")
    frame = pd.read_parquet(directory / relative)
    for key in ("session_id", "year", "round", "slot", "event_name", "session_name",
                "scheduled_start_utc", "collected_at_utc", "schema_version"):
        if key in manifest and key not in frame:
            frame[key] = manifest[key]
    frame["driver_number"] = str(driver)
    return frame


def compact(root):
    """Remove storage that is provably redundant; never touches irreplaceable data.

    1. Legacy normalized/ telemetry: deleted only after its source/ counterpart passes
       its checksum and matches column-for-column (normalized = source + constants).
    2. FastF1 parsed cache (*.ff1pkl): rebuilt offline by FastF1 from the raw HTTP
       cache (fastf1_http_cache.sqlite), which is kept. Its content is also what
       source/ archives.
    """
    removed = {"normalized_telemetry_files": 0, "normalized_telemetry_bytes": 0,
               "parsed_cache_files": 0, "parsed_cache_bytes": 0, "sessions_updated": []}
    for path in sorted((root / "sessions").glob("*/*/manifest.json")):
        directory = path.parent
        manifest = json.loads(path.read_text(encoding="utf-8"))
        files = manifest.get("files", {})
        legacy = [name for name in files if name.startswith(tuple(f"normalized/{s}/" for s in TELEMETRY_STREAMS))]
        if not legacy:
            continue
        for name in legacy:
            source_name = "source/" + name.split("/", 1)[1]
            source_info = files.get(source_name)
            if source_info is None or checksum(directory / source_name) != source_info["sha256"]:
                raise ValueError(f"{directory / source_name} missing or corrupt; refusing to remove {name}")
            source = pd.read_parquet(directory / source_name)
            normalized = pd.read_parquet(directory / name)
            pd.testing.assert_frame_equal(normalized[list(source.columns)], source)
        for name in legacy:
            removed["normalized_telemetry_bytes"] += files[name]["bytes"]
            removed["normalized_telemetry_files"] += 1
            del files[name]
        manifest["telemetry_layout"] = "source_only"
        manifest["compacted_at_utc"] = utcnow()
        atomic_json(path, manifest)
        for name in legacy:
            (directory / name).unlink()
        for stream in TELEMETRY_STREAMS:
            folder = directory / "normalized" / stream
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        removed["sessions_updated"].append(manifest.get("session_id", directory.name))
    cache = root / "cache"
    if (cache / "fastf1_http_cache.sqlite").is_file():
        for pickle in cache.rglob("*.ff1pkl"):
            removed["parsed_cache_bytes"] += pickle.stat().st_size
            removed["parsed_cache_files"] += 1
            pickle.unlink()
    return removed


class MemorySampler:
    """Sample process RSS, not Python-only allocations. Peak is a sampled lower bound."""

    def __enter__(self):
        self.stop = threading.Event()
        self.peak = psutil.Process().memory_info().rss
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(0.1):
            self.peak = max(self.peak, psutil.Process().memory_info().rss)

    def __exit__(self, *args):
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.stop.set()
        self.thread.join()


def retry(call, attempts):
    for attempt in range(attempts):
        try:
            return call()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
            response = getattr(error, "response", None)
            if response is not None and response.status_code < 500 and response.status_code != 429:
                raise
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))


def merge_inventory(root, discovered):
    """New download configurations must not erase earlier known sessions."""
    path = root / "inventory.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    combined = {item["session_id"]: item for item in existing}
    for item in discovered:
        combined[item["session_id"]] = {**combined.get(item["session_id"], {}), **item}
    atomic_json(path, sorted(combined.values(), key=lambda item: item["session_id"]))


def write_progress(root):
    """Human-readable cumulative ledger; manifests remain the skip authority."""
    inventory_path = root / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else []
    known = {item["session_id"]: item for item in inventory}
    manifests = {}
    for path in sorted((root / "sessions").glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        session_id = manifest.get("session_id", path.parent.name)
        manifests[session_id] = (manifest, path)
        if session_id not in known:
            known[session_id] = {key: manifest[key] for key in (
                "year", "round", "slot", "session_id", "event_name", "session_name",
                "is_testing", "test_number") if key in manifest}
    sessions, weekends = [], {}
    for session_id, item in sorted(known.items()):
        manifest, path = manifests.get(session_id, ({}, None))
        status = manifest.get("status", "not_started")
        entry = {
            **item, "session_id": session_id,
            "core_status": "complete" if core_recorded_complete(manifest) else status,
            "telemetry_status": status if manifest.get("telemetry") else "not_started",
            "manifest": path.relative_to(root).as_posix() if path else None,
            "finished_at_utc": manifest.get("finished_at_utc"),
            "missing_tables": manifest.get("missing_tables", []),
            "error": manifest.get("error"),
        }
        sessions.append(entry)
        if "year" not in item or "round" not in item:
            continue
        key = (item["year"], item["round"], item.get("test_number", 0))
        weekend = weekends.setdefault(key, {
            "year": item["year"], "round": item["round"], "event_name": item.get("event_name"),
            **({"test_number": item["test_number"]} if item.get("is_testing") else {}),
            "expected_sessions": 0, "core_completed": 0, "telemetry_completed": 0,
        })
        weekend["expected_sessions"] += 1
        weekend["core_completed"] += int(entry["core_status"] == "complete")
        weekend["telemetry_completed"] += int(entry["telemetry_status"] == "complete")
    atomic_json(root / "progress.json", {
        "schema_version": 1, "updated_at_utc": utcnow(),
        "note": "Last recorded completion by tier. Files are checksum-verified before ingestion skips; this ledger alone never authorizes a skip.",
        "weekends": [weekends[key] for key in sorted(weekends)], "sessions": sessions,
    })
    missed = []
    for session_id, (manifest, path) in sorted(manifests.items()):
        reason = unresolved_attempt(path.parent, bool(manifest.get("telemetry")))
        if reason:
            missed.append({
                "session_id": session_id, "event_name": manifest.get("event_name"),
                "session_name": manifest.get("session_name"), "reason": reason,
                "error": (manifest.get("error") or {}).get("message"),
                "missing": manifest.get("missing_tables", []),
                "last_attempt_utc": manifest.get("finished_at_utc") or manifest.get("started_at_utc"),
            })
    atomic_json(root / "missed-sessions.json", {
        "updated_at_utc": utcnow(), "count": len(missed),
        "note": "Sessions whose source returned no or incomplete data. Normal runs skip them; "
                "re-attempt once with `f1data fastf1 ingest ... --retry-failed` after the full download.",
        "sessions": missed,
    })


def discover(config, root, attempts=3):
    from .catalog import select_schedule_sessions, validate_year
    explicit_years = config.get("years", [])
    weekends = config.get("weekends", [])
    if explicit_years and weekends:
        raise ValueError("Choose years or explicit weekends, not both")
    years = sorted(set(explicit_years) if explicit_years else {w["year"] for w in weekends})
    if not years:
        raise ValueError("Specify at least one season or weekend")
    for year in years:
        validate_year(year)
    testing = bool(config.get("testing", False))
    if testing and not explicit_years:
        raise ValueError("Testing sessions are discovered with whole seasons, not selected weekends")
    items = []
    for year in years:
        # The stored schedule always includes testing rows so later runs see the full season.
        schedule = retry(lambda: fastf1.get_event_schedule(year, include_testing=True), attempts)
        write_table(schedule, root / "schedules" / f"{year}.parquet", {})
        found, excluded = select_schedule_sessions(
            schedule, year, weekends=None if explicit_years else [w for w in weekends if w["year"] == year],
            purpose=config.get("purpose", "consecutive-season corpus"), testing=testing)
        items.extend(found)
        atomic_json(root / "catalog" / f"{year}.json", {
            "year": year, "discovered_at_utc": utcnow(), "selection": "whole_season" if explicit_years else "selected_weekends",
            "schedule_events": int(len(schedule)), "discovered_sessions": len(found),
            "excluded_sessions": excluded,
            "note": "A schedule inventory is not proof of successful downloads; missing/cancelled events may be absent upstream. See progress.json and session manifests.",
        })
    merge_inventory(root, items)
    write_progress(root)
    return items


def report(root):
    manifests = []
    for path in sorted((root / "sessions").glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        files = manifest.get("files", {})
        manifests.append({
            **{k: manifest.get(k) for k in ("session_id", "event_name", "session_name", "status", "telemetry", "metrics", "missing_tables", "error")},
            "source_bytes": sum(f["bytes"] for p, f in files.items() if p.startswith("source/")),
            "normalized_bytes": sum(f["bytes"] for p, f in files.items() if p.startswith("normalized/")),
            "rows_by_table": {p: f["rows"] for p, f in files.items() if p.startswith("normalized/")},
        })
    inventory_path = root / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else []
    attempted = {m["session_id"] for m in manifests}
    result = {
        "generated_at_utc": utcnow(), "data_root": str(root),
        "cache_bytes": total_bytes(root / "cache"),
        "session_storage_bytes": total_bytes(root / "sessions"),
        "sessions": manifests,
        "expected_sessions": len(inventory),
        "not_attempted": [i["session_id"] for i in inventory if i["session_id"] not in attempted],
        "note": "Cache is shared; no per-session attribution. RSS is sampled every 100ms; durations include cache reads and retry waits. Source and normalized exports intentionally duplicate data.",
    }
    atomic_json(root / "report.json", result)
    write_progress(root)
    return result


LIVETIMING = "https://livetiming.formula1.com"
LIVETIMING_MIRROR = "https://livetiming-mirror.fastf1.dev"  # fastf1._api.base_url_mirror
# A session known to exist upstream. The timing archive answers 403 both when
# access is refused and when a session has no data (e.g. 2019 Japanese GP FP3,
# cancelled for Typhoon Hagibis), so a refusal is confirmed against this path.
REFERENCE_SESSION_PATH = "/static/2023/2023-03-05_Bahrain_Grand_Prix/2023-03-05_Race/"


class SessionUnavailable(RuntimeError):
    """The archive has no data for this session although access works."""


# Present in SessionUnavailable messages whose reference probe reached the server.
# Older records lack it: their probe may have been answered from the cache, so
# they are not trusted as known gaps (see unresolved_attempt).
LIVE_PROBE_MARK = "reference checked uncached"


def uncached_get(url):
    """GET through FastF1's authenticated, rate-limited session, bypassing only the HTTP cache."""
    previous = fastf1.Cache._http_tmp_disabled
    fastf1.Cache._http_tmp_disabled = True
    try:
        return fastf1.Cache.requests_get(url, timeout=30)
    finally:
        fastf1.Cache._http_tmp_disabled = previous


def check_access(session, attempts):
    """Use FastF1's authenticated, rate-limited cache instead of bypassing it."""
    # FastF1 falls back to its mirror for sessions F1's archive lost in its 2024
    # server migration (most of 2022 from the Bahrain race on; see FastF1 #607, #615).
    # Probe both, in the same order, so those sessions are not refused here.
    urls = (LIVETIMING + session.api_path + "SessionInfo.json",
            LIVETIMING_MIRROR + session.api_path + "SessionInfo.json")
    response = retry(lambda: fastf1.Cache.requests_get(urls, timeout=30), attempts)
    if response.status_code in (401, 403, 404):
        # The reference must hit the server: its 200 is in the HTTP cache, which
        # would report access as working even after the login has expired.
        reference = retry(lambda: uncached_get(LIVETIMING + REFERENCE_SESSION_PATH + "SessionInfo.json"), attempts)
        if reference.status_code == 200:
            raise SessionUnavailable(
                f"HTTP {response.status_code} for {session.api_path} while a known session is accessible "
                f"({LIVE_PROBE_MARK}); the archive has no data for this session (cancelled or never published).")
        raise ProviderAccessError(
            f"Historical timing endpoint returned HTTP {response.status_code}. "
            "Authenticate locally with: python -m fastf1 auth f1tv --authenticate. "
            "No timing data was downloaded. Do not put credentials in the repository.")
    response.raise_for_status()


def ingest(item, root, telemetry, attempts):
    from .snapshots import record_snapshot
    directory = root / "sessions" / str(item["year"]) / item["session_id"]
    if present(directory, telemetry):
        if not (root / "snapshots" / item["session_id"]).is_dir():
            record_snapshot(root, directory)
        print(f"SKIP {item['session_id']} {item['session_name']}", flush=True)
        return True
    if verified(directory, False):
        record_snapshot(root, directory)
    atomic_json(directory / "manifest.json", {**item, "status": "running", "started_at_utc": utcnow()})
    write_progress(root)
    started = time.perf_counter()
    cache_before = total_bytes(root / "cache")
    print(f"LOAD {item['session_id']} {item['event_name']} / {item['session_name']}", flush=True)
    with MemorySampler() as memory:
        try:
            if item.get("is_testing"):
                session = fastf1.get_testing_session(item["year"], item["test_number"], item["slot"])
            else:
                session = fastf1.get_session(item["year"], item["round"], item["slot"])
            check_access(session, attempts)
            retry(lambda: session.load(laps=True, telemetry=telemetry, weather=True, messages=True), attempts)
            manifest = export_session(session, directory, item, telemetry)
        except Exception as error:
            manifest = {**item, "status": "failed", "schema_version": SCHEMA_VERSION,
                        "telemetry": telemetry, "error": {"type": type(error).__name__, "message": str(error)}}
    manifest["metrics"] = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_sampled_rss_bytes": memory.peak,
        "cache_growth_bytes": total_bytes(root / "cache") - cache_before,
    }
    manifest["finished_at_utc"] = utcnow()
    atomic_json(directory / "manifest.json", manifest)
    if verified(directory, False):
        record_snapshot(root, directory)
    write_progress(root)
    print(f"{manifest['status'].upper()} {item['session_id']} {manifest.get('error', '')}", flush=True)
    if manifest.get("error", {}).get("type") == "ProviderAccessError":
        raise ProviderAccessError(manifest["error"]["message"])
    if manifest.get("error", {}).get("type") == RateLimitExceededError.__name__:
        raise RateLimitReached("FastF1 source rate limit reached. Resume later with the same data root; completed sessions are preserved.")
    return manifest["status"] == "complete"


WEEKEND_SELECTION_EXAMPLE = CONFIG_DIR / "pilot-weekends.json"


def register(commands, common):
    parser = commands.add_parser(
        "fastf1", parents=[common], help="FastF1 sessions: timing, telemetry, weather, race control (2018+)",
        description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["discover", "ingest", "report", "compact", "verify"])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--years", type=int, nargs="+", help="discover and ingest whole seasons")
    selection.add_argument("--weekends", type=Path,
                           help=f"JSON selection of individual weekends (example: {WEEKEND_SELECTION_EXAMPLE.name})")
    parser.add_argument("--telemetry", action="store_true", help="also load native per-driver car and position streams")
    parser.add_argument("--telemetry-scope", choices=["all", "competitive"], default="all",
                        help="with --telemetry: 'competitive' limits streams to qualifying, sprint and race sessions")
    parser.add_argument("--testing", action="store_true", help="with --years: also discover pre-season testing sessions")
    parser.add_argument("--retry-failed", action="store_true",
                        help="also re-attempt sessions an earlier run left failed or partial (normally skipped)")
    parser.add_argument("--wait-on-rate-limit", type=int, metavar="MINUTES",
                        help="on FastF1's hourly rate limit, wait this long and continue instead of stopping")
    parser.add_argument("--session-id", help="limit ingestion to one discovered session ID")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-sessions", type=int, help="maximum new session attempts; verified skips do not consume this limit")
    parser.set_defaults(handler=run, parser=parser)


def run(args):
    parser = args.parser
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    if args.max_sessions is not None and args.max_sessions < 1:
        parser.error("--max-sessions must be positive")
    if args.wait_on_rate_limit is not None and args.wait_on_rate_limit < 1:
        parser.error("--wait-on-rate-limit must be at least 1 minute")
    if args.testing and not args.years:
        parser.error("--testing requires --years")
    root = args.data_root
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        print(json.dumps(report(root), indent=2))
        return 0
    if args.command == "compact":
        print(json.dumps(compact(root), indent=2))
        return 0
    if args.command == "verify":
        problems = verify_all(root)
        print(json.dumps({"checksum_failures": problems}, indent=2))
        return 1 if problems else 0
    if not (args.years or args.weekends):
        parser.error(f"{args.command} needs --years or --weekends")
    config = ({"schema_version": SCHEMA_VERSION, "years": args.years, "testing": args.testing} if args.years
              else json.loads(args.weekends.read_text(encoding="utf-8")))
    if config.get("schema_version") != SCHEMA_VERSION:
        parser.error("unsupported weekend selection schema")
    (root / "cache").mkdir(exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(root / "cache"))
    # Keep the raw HTTP cache (stage 1, the replayable provider responses) but skip
    # FastF1's parsed pickles (stage 2): they duplicate source/ and cost ~55 MB per
    # telemetry session. Re-loading a session re-parses from stage 1 without network.
    fastf1.Cache.set_disabled(disable_http_cache=False, disable_func_cache=True)
    from . import ratelimit
    restored = ratelimit.restore(root)
    if restored:
        print(f"Restored {restored} recent FastF1 calls from earlier runs; hourly limits continue from there.", flush=True)
    inventory = discover(config, root, args.attempts)
    if args.command == "discover":
        print(json.dumps(inventory, indent=2))
        return 0
    selected = [i for i in inventory if args.session_id is None or i["session_id"] == args.session_id]
    if not selected:
        parser.error("session ID is not in the selected sessions")
    from .catalog import TELEMETRY_SCOPES, ingestion_batch
    outcome = ingestion_batch(selected, root, args.telemetry, args.attempts, args.max_sessions,
                              telemetry_scope=TELEMETRY_SCOPES[args.telemetry_scope],
                              rate_limit_wait=args.wait_on_rate_limit * 60 if args.wait_on_rate_limit else None,
                              compute_wait=ratelimit.seconds_until_budget,
                              after_session=lambda: ratelimit.save(root),
                              retry_failed=args.retry_failed)
    ratelimit.save(root)
    atomic_json(root / "last-batch.json", {"finished_at_utc": utcnow(), "telemetry_requested": args.telemetry,
                                           "telemetry_scope": args.telemetry_scope if args.telemetry else None, **outcome})
    summary = report(root)
    print(json.dumps(outcome, indent=2), flush=True)
    print(f"Report: {root / 'report.json'}; cache bytes: {summary['cache_bytes']}", flush=True)
    return 1 if outcome["failed"] or outcome["stopped_reason"] else 0
