"""Open-Meteo weather for every Grand Prix weekend, 1950 onward.

Two datasets, both hourly in UTC at the circuit coordinates from the Jolpica
archive (run `scripts/jolpica.py` first):

  observed       Historical Weather API (reanalysis: ERA5 before 2017, ECMWF IFS
                 9 km after, Open-Meteo "best match"). What the weather was.
  forecast-runs  Previous Runs API. What forecasts said N days ahead
                 (`<var>_previous_day<N>`, N = 0..7). Archived from 2021 for some
                 variables (GFS temperature) and from January 2024 for most; the
                 rest come back as nulls, which the manifest counts per variable.
                 These are the values a forecast made before a cutoff could know.

    data/external/open-meteo/
      raw/<dataset>/<season>/<season>-<rr>.json   API response, unchanged
      manifest.json                               per race: file, SHA-256, params, fill counts
      failed.json                                 races the API refused; retried with --retry-failed
      rate-state.json                             weighted call log (limits across runs)
      tables/observed_hourly.parquet              one row per race-hour, wide
      tables/forecast_runs_hourly.parquet         one row per race-hour-variable-lead, long

Free tier (open-meteo.com/en/terms, checked 2026-11-10): 600 calls/min, 5,000/h,
10,000/day. A request counts as several calls when it asks for more than 10
variables or more than 2 weeks: weight = max(1, vars/10) * max(1, days/14).
This client keeps well below that. Licence: CC BY 4.0 with attribution; the free
API is non-commercial only (commercial use needs a paid plan and the customer-
endpoint).
"""

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import time

import pandas as pd
import requests

from .storage import atomic_json, checksum, utcnow, write_table

USER_AGENT = "f1data/0.1.0"
SCHEMA_VERSION = 1
FIRST_SEASON = 1950
# A weekend is fetched once its last day is this far in the past, so reanalysis
# (published with a ~5-day delay) has settled and the file never needs refetching.
SETTLE_DAYS = 7
DAYS_BEFORE, DAYS_AFTER = 3, 1  # race day -3 (Thursday) .. +1 (UTC spill of evening races)

OBSERVED_VARIABLES = (
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain", "weather_code",
    "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "soil_temperature_0_to_7cm",
)
FORECAST_VARIABLES = (
    "temperature_2m", "relative_humidity_2m", "precipitation", "rain", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "weather_code",
)
FORECAST_LEADS = range(0, 8)

DATASETS = {
    "observed": {
        "url": "https://archive-api.open-meteo.com/v1/archive", "first": FIRST_SEASON,
        "hourly": OBSERVED_VARIABLES,
    },
    "forecast-runs": {
        "url": "https://previous-runs-api.open-meteo.com/v1/forecast", "first": 2021,
        "hourly": tuple(variable if lead == 0 else f"{variable}_previous_day{lead}"
                        for variable in FORECAST_VARIABLES for lead in FORECAST_LEADS),
    },
}


def call_weight(variables, days):
    return max(1.0, variables / 10) * max(1.0, days / 14)


class WeightedLimiter:
    """Budget in Open-Meteo's weighted calls, persisted so restarts keep the windows."""

    def __init__(self, path, per_hour=2500, per_day=8000, min_interval=1.0, clock=time.time, sleep=time.sleep):
        self.path, self.per_hour, self.per_day, self.min_interval = path, per_hour, per_day, min_interval
        self.clock, self.sleep = clock, sleep
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))["calls"]
        except (OSError, ValueError, KeyError):
            saved = []
        self.calls = [(t, w) for t, w in saved if t > clock() - 86400]

    def _wait_for(self, window, budget, weight, now):
        recent = [(t, w) for t, w in self.calls if t > now - window]
        used = sum(w for _, w in recent)
        if used + weight <= budget:
            return 0.0
        for t, w in recent:  # oldest first: wait until enough weight leaves the window
            used -= w
            if used + weight <= budget:
                return t + window - now + 1
        return window

    def acquire(self, weight):
        while True:
            now = self.clock()
            self.calls = [(t, w) for t, w in self.calls if t > now - 86400]
            wait = max(self._wait_for(3600, self.per_hour, weight, now), self._wait_for(86400, self.per_day, weight, now))
            if wait <= 0:
                break
            if wait >= 60:
                print(f"OPEN-METEO budget reached; waiting {round(wait / 60)} min", flush=True)
            self.sleep(wait)
        if self.calls and now - self.calls[-1][0] < self.min_interval:
            self.sleep(self.min_interval - (now - self.calls[-1][0]))
            now = self.clock()
        self.calls.append((now, weight))
        atomic_json(self.path, {"calls": self.calls})


class RequestRefused(Exception):
    """The API answered with a client error (4xx other than 429); retrying will not help."""


class Client:
    def __init__(self, limiter, session=None, attempts=5, sleep=time.sleep):
        self.limiter, self.attempts, self.sleep = limiter, attempts, sleep
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def get(self, url, params, weight):
        for attempt in range(self.attempts):
            self.limiter.acquire(weight)
            try:
                response = self.session.get(url, params=params, timeout=120)
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):  # dropped mid-body too
                if attempt + 1 == self.attempts:
                    raise
                self.sleep(min(30 * 2 ** attempt, 600))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 == self.attempts:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                delay = int(retry_after) if retry_after and retry_after.isdigit() else min(60 * 2 ** attempt, 1800)
                print(f"OPEN-METEO HTTP {response.status_code}; retrying in {delay}s", flush=True)
                self.sleep(delay)
                continue
            if response.status_code >= 400:
                try:
                    reason = response.json().get("reason", response.text[:300])
                except ValueError:
                    reason = response.text[:300]
                raise RequestRefused(f"HTTP {response.status_code}: {reason}")
            return response.json()
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------- weekends

SESSION_COLUMNS = ("FirstPractice", "SecondPractice", "ThirdPractice", "SprintQualifying", "SprintShootout",
                   "Sprint", "Qualifying")


def weekends(jolpica_root):
    """One record per championship race with coordinates and the UTC date window."""
    path = jolpica_root / "tables" / "races.parquet"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run scripts/jolpica.py first")
    races = pd.read_parquet(path)
    records = []
    for row in races.itertuples(index=False):
        row = row._asdict()
        race_day = date.fromisoformat(row["date"])
        days = [race_day - timedelta(days=DAYS_BEFORE)]
        days += [date.fromisoformat(row[f"{name}_date"]) for name in SESSION_COLUMNS
                 if row.get(f"{name}_date") is not None and not pd.isna(row.get(f"{name}_date"))]
        records.append({
            "season": int(row["season"]), "round": int(row["round"]), "race_name": row["raceName"],
            "circuit_id": row["Circuit_circuitId"],
            "latitude": float(row["Circuit_Location_lat"]), "longitude": float(row["Circuit_Location_long"]),
            "start_date": min(days).isoformat(), "end_date": (race_day + timedelta(days=DAYS_AFTER)).isoformat(),
        })
    return sorted(records, key=lambda r: (r["season"], r["round"]))


def race_key(weekend):
    return f"{weekend['season']}-{weekend['round']:02d}"


def raw_path(dataset, weekend):
    return Path("raw") / dataset / str(weekend["season"]) / f"{race_key(weekend)}.json"


def settled(weekend, today):
    return date.fromisoformat(weekend["end_date"]) + timedelta(days=SETTLE_DAYS) <= today


def fill_counts(payload):
    hourly = payload.get("hourly", {})
    return {name: sum(value is not None for value in values) for name, values in hourly.items() if name != "time"}


def download(root, client, datasets, seasons=None, today=None, retry_failed=False, jolpica_root=None):
    today = today or datetime.now(timezone.utc).date()
    manifest_path, failed_path = root / "manifest.json", root / "failed.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {"schema_version": SCHEMA_VERSION, "licence": "CC BY 4.0 (Open-Meteo; non-commercial free API)",
                    "entries": {}}
    try:
        failed = json.loads(failed_path.read_text(encoding="utf-8"))["entries"]
    except (OSError, ValueError, KeyError):
        failed = {}
    counts = {"fetched": 0, "present": 0, "not_settled": 0, "failed": 0, "skipped_failed": 0}
    for weekend in weekends(jolpica_root or root.parent / "jolpica"):
        if seasons and weekend["season"] not in seasons:
            continue
        for dataset in datasets:
            spec = DATASETS[dataset]
            if weekend["season"] < spec["first"]:
                continue
            key, relative = f"{dataset}/{race_key(weekend)}", raw_path(dataset, weekend)
            entry = manifest["entries"].get(key)
            if entry and (root / relative).is_file() and checksum(root / relative) == entry["sha256"]:
                counts["present"] += 1
                continue
            if not settled(weekend, today):
                counts["not_settled"] += 1
                continue
            if key in failed and not retry_failed:
                counts["skipped_failed"] += 1
                continue
            params = {"latitude": weekend["latitude"], "longitude": weekend["longitude"],
                      "start_date": weekend["start_date"], "end_date": weekend["end_date"],
                      "hourly": ",".join(spec["hourly"]), "timezone": "GMT"}
            days = (date.fromisoformat(weekend["end_date"]) - date.fromisoformat(weekend["start_date"])).days + 1
            try:
                payload = client.get(spec["url"], params, call_weight(len(spec["hourly"]), days))
            except RequestRefused as error:
                failed[key] = {**weekend, "dataset": dataset, "error": str(error), "attempted_at_utc": utcnow()}
                atomic_json(failed_path, {"note": "Requests the API refused; rerun with --retry-failed.",
                                          "entries": failed})
                counts["failed"] += 1
                print(f"FAILED {key}: {error}", flush=True)
                continue
            atomic_json(root / relative, payload)
            manifest["entries"][key] = {
                **weekend, "dataset": dataset, "file": relative.as_posix(), "sha256": checksum(root / relative),
                "fetched_at_utc": utcnow(), "url": spec["url"], "params": params, "fill": fill_counts(payload),
            }
            atomic_json(manifest_path, manifest)
            if failed.pop(key, None) is not None:
                atomic_json(failed_path, {"note": "Requests the API refused; rerun with --retry-failed.",
                                          "entries": failed})
            counts["fetched"] += 1
            print(f"OK {key} {weekend['race_name']}", flush=True)
    return counts


# ---------------------------------------------------------------- tables

def build_tables(root):
    """Rebuild Parquet tables from raw responses (no network)."""
    try:
        entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["entries"]
    except OSError:
        return {}
    observed, forecasts = [], []
    for key, entry in sorted(entries.items()):
        payload = json.loads((root / entry["file"]).read_text(encoding="utf-8"))
        hourly = pd.DataFrame(payload["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"], utc=True)
        base = {"season": entry["season"], "round": entry["round"], "circuit_id": entry["circuit_id"],
                "grid_latitude": payload["latitude"], "grid_longitude": payload["longitude"],
                "grid_elevation": payload.get("elevation")}
        hourly = hourly.assign(**base)
        if entry["dataset"] == "observed":
            observed.append(hourly)
            continue
        long = hourly.melt(id_vars=["time", *base], var_name="column", value_name="value")
        split = long["column"].str.extract(r"^(?P<variable>.+?)(?:_previous_day(?P<lead_days>\d))?$")
        long["variable"] = split["variable"]
        long["lead_days"] = split["lead_days"].fillna(0).astype("int8")
        forecasts.append(long.drop(columns="column").dropna(subset=["value"]))
    written = {}
    for name, frames in (("observed_hourly", observed), ("forecast_runs_hourly", forecasts)):
        if frames:
            frame = pd.concat(frames, ignore_index=True)
            written[name] = write_table(frame, root / "tables" / f"{name}.parquet", {})
    atomic_json(root / "tables" / "manifest.json", {
        "built_at_utc": utcnow(), "files": written,
        "note": "Rebuilt from raw/ responses. Times are UTC; lead_days 0 is the latest run, N is the run N*24 h earlier.",
    })
    return written


def register(commands, common):
    parser = commands.add_parser("openmeteo", parents=[common], help="Open-Meteo observed weather and archived forecasts",
                                 description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "tables", "report", "verify"])
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--seasons", type=int, nargs="+", help="seasons to fetch (default: all)")
    parser.add_argument("--retry-failed", action="store_true", help="re-attempt requests the API refused earlier")
    parser.add_argument("--per-hour", type=int, default=2500, help="weighted calls per hour (free tier allows 5000)")
    parser.set_defaults(handler=run, parser=parser)


def run(args):
    root = args.data_root / "external" / "open-meteo"
    if args.command in ("report", "verify"):
        from . import audit
        return audit.run_source("openmeteo", args.data_root, args.command)
    if args.command == "download":
        if not 1 <= args.per_hour <= 4000:
            args.parser.error("--per-hour must be between 1 and 4000 (free tier allows 5000/h)")
        client = Client(WeightedLimiter(root / "rate-state.json", per_hour=args.per_hour))
        counts = download(root, client, args.datasets, seasons=set(args.seasons or []),
                          retry_failed=args.retry_failed, jolpica_root=args.data_root / "external" / "jolpica")
        print(json.dumps(counts), flush=True)
    written = build_tables(root)
    print(json.dumps({name: info["rows"] for name, info in written.items()}, indent=2), flush=True)
    print(f"Tables: {root / 'tables'}", flush=True)
    return 0
