"""Jolpica (Ergast-compatible) historical results archive, 1950 onward.

Raw API pages are stored unchanged; tables are rebuilt from them offline.

    data/external/jolpica/
      raw/<season>/<endpoint>/offset-<n>.json          season-wide endpoints
      raw/<season>/<endpoint>/round-<rr>/offset-<n>.json   per-round endpoints
      raw/<season>/manifest.json                       pages, totals, SHA-256, status
      tables/<table>.parquet                           flattened, rebuilt from raw
      rate-state.json                                  request timestamps (limit across runs)

Limits (docs/rate_limits.md, checked 2026-11-10): 4 req/s burst, 500 req/h
unauthenticated. FastF1 also calls Jolpica (self-limited to 200/h), so this client
defaults to 250/h and one request per second. Data licence: CC BY-NC-SA 4.0
(non-commercial; commercial use requires permission from Jolpica).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd
import requests

from .storage import atomic_json, checksum, utcnow, write_table

BASE_URL = "https://api.jolpi.ca/ergast/f1"
USER_AGENT = "f1data/0.1.0"
PAGE_LIMIT = 100
FIRST_SEASON = 1950
SCHEMA_VERSION = 1

# endpoint -> first season with data (None: all seasons). Per-round endpoints are
# fetched once per race; the rest once per season with pagination.
SEASON_ENDPOINTS = {
    "races": None, "results": None, "qualifying": None, "sprint": 2021,
    "driverstandings": None, "constructorstandings": 1958,
    "drivers": None, "constructors": None, "circuits": None, "status": None,
}
ROUND_ENDPOINTS = {"pitstops": 2011}
OPTIONAL_ROUND_ENDPOINTS = {"laps": 1996}


class RateLimiter:
    """Sliding-window limiter persisted to disk so restarts keep the hourly budget."""

    def __init__(self, path, per_hour=250, min_interval=1.0, clock=time.time, sleep=time.sleep):
        self.path, self.per_hour, self.min_interval = path, per_hour, min_interval
        self.clock, self.sleep = clock, sleep
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))["timestamps"]
        except (OSError, ValueError, KeyError):
            saved = []
        now = clock()
        self.timestamps = sorted(t for t in saved if t > now - 3600)

    def acquire(self):
        now = self.clock()
        self.timestamps = [t for t in self.timestamps if t > now - 3600]
        if len(self.timestamps) >= self.per_hour:
            wait = self.timestamps[len(self.timestamps) - self.per_hour] + 3600 - now + 1
            if wait >= 60:  # short waits are the normal pace of a full window; don't log each one
                print(f"JOLPICA hourly budget used ({self.per_hour}/h); waiting {round(wait / 60)} min", flush=True)
            self.sleep(wait)
            now = self.clock()
        if self.timestamps and now - self.timestamps[-1] < self.min_interval:
            self.sleep(self.min_interval - (now - self.timestamps[-1]))
            now = self.clock()
        self.timestamps.append(now)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"timestamps": self.timestamps}), encoding="utf-8")
        temporary.replace(self.path)


class Client:
    def __init__(self, limiter, session=None, attempts=5, sleep=time.sleep):
        self.limiter, self.attempts, self.sleep = limiter, attempts, sleep
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def get(self, path, offset):
        url = f"{BASE_URL}/{path}.json"
        for attempt in range(self.attempts):
            self.limiter.acquire()
            try:
                response = self.session.get(url, params={"limit": PAGE_LIMIT, "offset": offset}, timeout=60)
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
                print(f"JOLPICA HTTP {response.status_code} for {path}; retrying in {delay}s", flush=True)
                self.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("unreachable")


def season_final(season, today=None):
    """Past seasons are final. The current season is provisional and refetched."""
    today = today or datetime.now(timezone.utc).date()
    return season < today.year


def page_valid(path, info):
    return path.is_file() and checksum(path) == info["sha256"]


def fetch_all_pages(client, path, directory, pages):
    """Fetch every page of one endpoint; reuse verified pages. Returns the total."""
    offset, total = 0, None
    while total is None or offset < total:
        file = directory / f"offset-{offset:05d}.json"
        key = file.as_posix()
        if key in pages and page_valid(file, pages[key]):
            payload = json.loads(file.read_text(encoding="utf-8"))
        else:
            payload = client.get(path, offset)
            file.parent.mkdir(parents=True, exist_ok=True)
            temporary = file.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            temporary.replace(file)
            pages[key] = {"sha256": checksum(file), "fetched_at_utc": utcnow()}
        total = int(payload["MRData"]["total"])
        limit = int(payload["MRData"]["limit"]) or PAGE_LIMIT
        offset += limit
    return total


def race_rounds(season_dir):
    rounds = []
    for page in sorted((season_dir / "races").glob("offset-*.json")):
        rounds += [int(r["round"]) for r in json.loads(page.read_text(encoding="utf-8"))["MRData"]["RaceTable"]["Races"]]
    return sorted(set(rounds))


def download_season(root, season, client, laps=False, today=None):
    season_dir = root / "raw" / str(season)
    manifest_path = season_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    final = season_final(season, today)
    wanted = {**ROUND_ENDPOINTS, **(OPTIONAL_ROUND_ENDPOINTS if laps else {})}
    if (manifest.get("status") == "complete" and manifest.get("final") and final
            and set(wanted).issubset(manifest.get("round_endpoints", []))
            and all(page_valid(root / name, info) for name, info in manifest["pages"].items())):
        print(f"SKIP jolpica {season}", flush=True)
        return manifest
    # Provisional seasons are refetched in full; final seasons reuse verified pages.
    pages = {} if not final else {name: info for name, info in manifest.get("pages", {}).items()}
    absolute = {str((root / name).as_posix()): info for name, info in pages.items()}
    totals = {}
    print(f"FETCH jolpica {season}", flush=True)
    for endpoint, first in SEASON_ENDPOINTS.items():
        if first is not None and season < first:
            continue
        totals[endpoint] = fetch_all_pages(client, f"{season}/{endpoint}", season_dir / endpoint, absolute)
    for endpoint, first in wanted.items():
        if season < first:
            continue
        for rnd in race_rounds(season_dir):
            key = f"{endpoint}/{rnd:02d}"
            totals[key] = fetch_all_pages(client, f"{season}/{rnd}/{endpoint}",
                                          season_dir / endpoint / f"round-{rnd:02d}", absolute)
    manifest = {
        "schema_version": SCHEMA_VERSION, "season": season, "status": "complete", "final": final,
        "finished_at_utc": utcnow(), "source": BASE_URL, "licence": "CC BY-NC-SA 4.0",
        "round_endpoints": sorted(wanted), "totals": totals,
        "pages": {Path(name).relative_to(root).as_posix(): info for name, info in absolute.items()},
    }
    atomic_json(manifest_path, manifest)
    return manifest


# ---------------------------------------------------------------- tables

def _records(root, season, endpoint, table_key, list_key):
    for page in sorted((root / "raw" / str(season) / endpoint).rglob("offset-*.json")):
        table = json.loads(page.read_text(encoding="utf-8"))["MRData"].get(table_key, {})
        for record in table.get(list_key, []):
            yield table, record


def _flatten_race_rows(root, season, endpoint, row_key):
    rows = []
    for _, race in _records(root, season, endpoint, "RaceTable", "Races"):
        base = {"season": int(race["season"]), "round": int(race["round"]), "race_name": race["raceName"],
                "circuit_id": race["Circuit"]["circuitId"], "date": race.get("date")}
        for row in race.get(row_key, []):
            flat = pd.json_normalize(row, sep="_").iloc[0].to_dict()
            rows.append({**base, **flat})
    return rows


def build_tables(root):
    """Rebuild every Parquet table from the raw pages (no network)."""
    seasons = sorted(int(p.name) for p in (root / "raw").iterdir() if p.is_dir() and p.name.isdigit())
    tables = {name: [] for name in ("races", "results", "qualifying", "sprint_results", "pitstops", "laps",
                                     "driver_standings", "constructor_standings", "drivers",
                                     "constructors", "circuits", "status")}
    for season in seasons:
        for _, race in _records(root, season, "races", "RaceTable", "Races"):
            flat = pd.json_normalize({k: v for k, v in race.items()
                                      if k not in ("Results", "QualifyingResults", "SprintResults")}, sep="_")
            tables["races"].append(flat.iloc[0].to_dict())
        tables["results"] += _flatten_race_rows(root, season, "results", "Results")
        tables["qualifying"] += _flatten_race_rows(root, season, "qualifying", "QualifyingResults")
        tables["sprint_results"] += _flatten_race_rows(root, season, "sprint", "SprintResults")
        tables["pitstops"] += _flatten_race_rows(root, season, "pitstops", "PitStops")
        for _, race in _records(root, season, "laps", "RaceTable", "Races"):
            for lap in race.get("Laps", []):
                for timing in lap["Timings"]:
                    tables["laps"].append({"season": int(race["season"]), "round": int(race["round"]),
                                           "lap": int(lap["number"]), **timing})
        for kind, key, list_key in (("driverstandings", "driver_standings", "DriverStandings"),
                                    ("constructorstandings", "constructor_standings", "ConstructorStandings")):
            for _, standings in _records(root, season, kind, "StandingsTable", "StandingsLists"):
                for row in standings[list_key]:
                    flat = pd.json_normalize(row, sep="_").iloc[0].to_dict()
                    tables[key].append({"season": int(standings["season"]), "round": int(standings["round"]), **flat})
        for endpoint, table_key, list_key, name in (("drivers", "DriverTable", "Drivers", "drivers"),
                                                    ("constructors", "ConstructorTable", "Constructors", "constructors"),
                                                    ("circuits", "CircuitTable", "Circuits", "circuits"),
                                                    ("status", "StatusTable", "Status", "status")):
            for _, record in _records(root, season, endpoint, table_key, list_key):
                tables[name].append({"season": season, **pd.json_normalize(record, sep="_").iloc[0].to_dict()})
    written = {}
    for name, rows in tables.items():
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame = frame.astype({c: "string" for c in frame.columns if frame[c].dtype == object})
        written[name] = write_table(frame, root / "tables" / f"{name}.parquet", {})
    atomic_json(root / "tables" / "manifest.json", {
        "built_at_utc": utcnow(), "seasons": seasons, "files": written,
        "note": "Rebuilt from raw/ pages; values are Jolpica's strings except season/round/lap.",
    })
    return written


def register(commands, common):
    parser = commands.add_parser("jolpica", parents=[common], help="Jolpica (Ergast) results, standings and laps (1950+)",
                                 description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "tables", "report", "verify"])
    parser.add_argument("--seasons", type=int, nargs="+", help="seasons to download (default: all from 1950)")
    parser.add_argument("--laps", action="store_true", help="also download lap times (1996+, ~13 requests per race)")
    parser.add_argument("--per-hour", type=int, default=250, help="request budget per hour (Jolpica allows 500 in total)")
    parser.set_defaults(handler=run, parser=parser)


def run(args):
    root = args.data_root / "external" / "jolpica"
    if args.command in ("report", "verify"):
        from . import audit
        return audit.run_source("jolpica", args.data_root, args.command)
    if args.command == "download":
        if not 1 <= args.per_hour <= 400:
            args.parser.error("--per-hour must be between 1 and 400 (FastF1 needs part of Jolpica's 500/h)")
        today = datetime.now(timezone.utc).date()
        seasons = args.seasons or list(range(FIRST_SEASON, today.year + 1))
        client = Client(RateLimiter(root / "rate-state.json", per_hour=args.per_hour))
        for season in seasons:
            if season < FIRST_SEASON or season > today.year:
                args.parser.error(f"season {season} outside {FIRST_SEASON}-{today.year}")
            download_season(root, season, client, laps=args.laps, today=today)
    written = build_tables(root)
    print(json.dumps({name: info["rows"] for name, info in written.items()}, indent=2), flush=True)
    print(f"Tables: {root / 'tables'}", flush=True)
    return 0
