"""Canonical reference tables: events, sessions, entrants, drivers, teams, circuits.

Identifiers follow the Ergast scheme used by Jolpica, FastF1 and the platform's
database: drivers (`max_verstappen`), constructors (`red_bull`) and circuits
(`bahrain`). Built offline from the archive:

    data/reference/
      events.parquet      one row per championship round and pre-season test
      sessions.parquet    one row per archived session, with actual start and end
      entrants.parquet    one row per car in each session, with resolved IDs
      drivers.parquet     Jolpica driver records (2018+ entrants)
      teams.parquet       constructor ID, name and lineage per season (1996+)
      circuits.parquet    Jolpica circuit records
      qualifying_history.parquet
                          Jolpica qualifying times before 2018 (1996+), one row per
                          driver per event, with lineage and circuit
      manifest.json       SHA-256 per table, resolution report

FastF1 leaves driver and team IDs blank for some sessions (most testing, some
practice and sprint qualifying). They are filled from the same car in the same
event, then from the same car number and abbreviation in the same season. An ID
is filled only when exactly one candidate exists; anything else stays empty
and is listed in the manifest, never guessed. Practice-only drivers who never
started a race keep an empty driver ID; their laps still count for the team.
"""

import json
import re
import unicodedata

import pandas as pd

from .paths import CONFIG_DIR
from .storage import atomic_json, utcnow, write_table

FIRST_SEASON = 2018
LINEAGE_FIRST_SEASON = 1996
LINEAGE_CONFIG = CONFIG_DIR / "team-lineage.json"
SESSION_TYPES = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
    "Qualifying": "Q", "Sprint Shootout": "SQ", "Sprint Qualifying": "SQ",
    "Sprint": "S", "Race": "R",
}
ENTRANT_COLUMNS = ["DriverNumber", "Abbreviation", "DriverId", "TeamId", "TeamName", "FullName"]


def _blank_to_na(frame, columns):
    for column in columns:
        if column in frame:
            frame[column] = frame[column].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return frame


def _utc(value):
    """Manifest times are UTC; older ones were written without an offset."""
    if not value:
        return pd.NaT
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _manifests(root):
    for path in sorted((root / "sessions").glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") in ("complete", "partial"):
            yield path.parent, manifest


def event_key(manifest):
    if manifest.get("is_testing"):
        return f"{manifest['year']}-T{manifest['test_number']}"
    return f"{manifest['year']}-{int(manifest['round']):02d}"


def session_rows(root):
    """Sessions with their actual first lap start and last lap end (UTC)."""
    rows = []
    for directory, manifest in _manifests(root):
        start = end = pd.NaT
        laps_path = directory / "normalized" / "laps.parquet"
        if laps_path.is_file():
            laps = pd.read_parquet(laps_path, columns=["LapStartDate", "LapTime"])
            if len(laps):
                start = laps.LapStartDate.min()
                end = (laps.LapStartDate + laps.LapTime).max()
                if pd.isna(end):
                    end = laps.LapStartDate.max()
        testing = bool(manifest.get("is_testing"))
        rows.append({
            "session_id": manifest["session_id"], "event_key": event_key(manifest),
            "season": int(manifest["year"]), "round": None if testing else int(manifest["round"]),
            "test_number": manifest.get("test_number") if testing else None,
            "slot": int(manifest["slot"]), "session_name": manifest["session_name"],
            "session_type": "T" if testing else SESSION_TYPES.get(manifest["session_name"], manifest["session_name"]),
            "is_testing": testing, "event_format": manifest.get("event_format"),
            "scheduled_start_utc": _utc(manifest.get("scheduled_start_utc")),
            "first_lap_start_utc": start.tz_localize("UTC") if pd.notna(start) else pd.NaT,
            "last_lap_end_utc": end.tz_localize("UTC") if pd.notna(end) else pd.NaT,
            "archive_status": manifest["status"], "telemetry": bool(manifest.get("telemetry")),
        })
    return pd.DataFrame(rows)


def raw_entrants(root):
    """Every car seen in a session: classification rows plus cars that only set laps."""
    frames = []
    for directory, manifest in _manifests(root):
        results = pd.read_parquet(directory / "normalized" / "results.parquet")
        results = results[[c for c in ENTRANT_COLUMNS if c in results]].copy()
        laps = pd.read_parquet(directory / "normalized" / "laps.parquet", columns=["DriverNumber", "Driver", "Team"])
        lap_only = laps.drop_duplicates("DriverNumber").rename(columns={"Driver": "Abbreviation", "Team": "TeamName"})
        lap_only = lap_only[~lap_only.DriverNumber.isin(results.DriverNumber)]
        frame = pd.concat([results, lap_only], ignore_index=True)
        frame["session_id"] = manifest["session_id"]
        frame["event_key"] = event_key(manifest)
        frame["season"] = int(manifest["year"])
        frame["classified_in_results"] = frame.index < len(results)
        frames.append(frame)
    entrants = pd.concat(frames, ignore_index=True)
    return _blank_to_na(entrants, ENTRANT_COLUMNS)


def _unique_lookup(frame, keys, value):
    """Map key -> value where the key has exactly one non-empty value."""
    known = frame.dropna(subset=[value]).groupby(keys)[value].unique()
    return known[known.apply(len) == 1].apply(lambda values: values[0])


def resolve_entrants(entrants):
    entrants = entrants.copy()
    entrants["driver_id"] = entrants["DriverId"]
    entrants["team_id"] = entrants["TeamId"]
    entrants["driver_source"] = entrants["DriverId"].notna().map({True: "fastf1", False: pd.NA})
    entrants["team_source"] = entrants["TeamId"].notna().map({True: "fastf1", False: pd.NA})

    def fill(target, source_label, lookup, keys):
        missing = entrants[target].isna()
        values = [lookup.get(tuple(row) if len(keys) > 1 else row[0])
                  for row in entrants.loc[missing, keys].itertuples(index=False)]
        values = pd.Series(values, index=entrants.index[missing], dtype="string")
        entrants.loc[missing, target] = values
        entrants.loc[missing & values.reindex(entrants.index).notna(), target.replace("_id", "_source")] = source_label

    fill("driver_id", "same_event", _unique_lookup(entrants, ["event_key", "DriverNumber"], "DriverId"),
         ["event_key", "DriverNumber"])
    fill("driver_id", "same_season", _unique_lookup(entrants, ["season", "DriverNumber", "Abbreviation"], "driver_id"),
         ["season", "DriverNumber", "Abbreviation"])
    fill("driver_id", "other_season_same_name",
         _unique_lookup(entrants, ["FullName", "Abbreviation"], "driver_id"), ["FullName", "Abbreviation"])
    fill("team_id", "same_event", _unique_lookup(entrants, ["event_key", "DriverNumber"], "TeamId"),
         ["event_key", "DriverNumber"])
    fill("team_id", "same_season_team_name", _unique_lookup(entrants, ["season", "TeamName"], "team_id"),
         ["season", "TeamName"])
    # Pre-season testing: the team the driver entered at the season's first event.
    first_event = (entrants[~entrants.event_key.str.contains("-T")].dropna(subset=["driver_id", "team_id"])
                   .sort_values("event_key").drop_duplicates(["season", "driver_id"]))
    fill("team_id", "season_first_event", first_event.set_index(["season", "driver_id"]).team_id,
         ["season", "driver_id"])
    columns = ["session_id", "event_key", "season", "DriverNumber", "Abbreviation", "FullName", "TeamName",
               "driver_id", "team_id", "driver_source", "team_source", "classified_in_results"]
    resolved = entrants[columns].rename(columns={
        "DriverNumber": "driver_number", "Abbreviation": "abbreviation", "FullName": "full_name",
        "TeamName": "team_name"})
    return resolved


def unresolved(entrants):
    gaps = entrants[entrants.driver_id.isna() | entrants.team_id.isna()]
    summary = (gaps.groupby(["season", "driver_number", "abbreviation", "team_name"], dropna=False)
               .session_id.agg(sessions="count", example_session="first").reset_index())
    return [{"season": int(r.season), "driver_number": r.driver_number, "abbreviation": r.abbreviation,
             "team_name": None if pd.isna(r.team_name) else r.team_name, "sessions": int(r.sessions),
             "example_session": r.example_session} for r in summary.itertuples()]


def lineage_table(constructors, config_path=LINEAGE_CONFIG, last_season=None):
    """(season, constructor_id) -> lineage_id from the lineage config.

    Every constructor-season from 1996 must map to exactly one lineage, so a new
    constructor or a renamed team fails the build until the config covers it."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    last_season = last_season or int(constructors.season.max())
    rows = []
    for lineage_id, lineage in config["lineages"].items():
        for member in lineage["members"]:
            for season in range(member["from"], member.get("to", last_season) + 1):
                rows.append({"season": season, "constructor_id": member["constructor_id"], "lineage_id": lineage_id})
    table = pd.DataFrame(rows)
    duplicated = table[table.duplicated(["season", "constructor_id"], keep=False)]
    if not duplicated.empty:
        raise ValueError(f"constructor-seasons in more than one lineage: {duplicated.to_dict('records')}")
    needed = constructors[constructors.season >= LINEAGE_FIRST_SEASON][["season", "constructor_id"]].drop_duplicates()
    missing = needed.merge(table, how="left", on=["season", "constructor_id"])
    missing = missing[missing.lineage_id.isna()]
    if not missing.empty:
        raise ValueError(f"constructor-seasons without a lineage in {config_path.name}: "
                         f"{missing[['season', 'constructor_id']].to_dict('records')}")
    return table


def _seconds(times):
    """'1:23.456' (or '83.456') -> seconds; anything else -> NaN."""
    def parse(value):
        if pd.isna(value) or not str(value).strip():
            return float("nan")
        minutes, _, seconds = str(value).strip().rpartition(":")
        try:
            return int(minutes or 0) * 60 + float(seconds)
        except ValueError:
            return float("nan")
    return times.map(parse).astype(float)


def qualifying_history(root, races, lineage):
    """Qualifying times from Jolpica for the seasons before the FastF1 archive.

    One row per driver per event: Q1-Q3 best times in seconds (only Q1 before the
    knockout format), the constructor and its lineage, the circuit, and an estimated
    session start (qualifying date, else the day before the race, at the race's start
    time of day, else 12:00 UTC); Jolpica has no qualifying start times before 2018.
    Seasons the source covers only partly (1996-2002) are kept as they are; the
    model decides which seasons and segments it uses."""
    path = root / "external" / "jolpica" / "tables" / "qualifying.parquet"
    frame = pd.read_parquet(path)
    frame["season"], frame["round"] = frame.season.astype(int), frame["round"].astype(int)
    frame = frame[(frame.season >= LINEAGE_FIRST_SEASON) & (frame.season < FIRST_SEASON)]
    schedule = races[(races.season >= LINEAGE_FIRST_SEASON) & (races.season < FIRST_SEASON)]
    race_day = pd.to_datetime(schedule.date)
    day = pd.to_datetime(schedule.Qualifying_date).fillna(race_day - pd.Timedelta(days=1))
    time_of_day = pd.to_timedelta(schedule.time.fillna("12:00:00Z").str.rstrip("Z"))
    starts = schedule.assign(start_utc=(day + time_of_day).dt.tz_localize("UTC"))[["season", "round", "start_utc"]]
    history = pd.DataFrame({
        "event_key": [f"{s}-{r:02d}" for s, r in zip(frame.season, frame["round"])],
        "season": frame.season, "round": frame["round"], "circuit_id": frame.circuit_id,
        "driver_id": frame.Driver_driverId, "constructor_id": frame.Constructor_constructorId,
        "number": frame.number, "position": pd.to_numeric(frame.position, errors="coerce").astype("Int64"),
        "q1_seconds": _seconds(frame.Q1),
        "q2_seconds": _seconds(frame.Q2) if "Q2" in frame else float("nan"),
        "q3_seconds": _seconds(frame.Q3) if "Q3" in frame else float("nan"),
    })
    history = history.merge(starts, on=["season", "round"], how="left")
    history = history.merge(lineage, on=["season", "constructor_id"], how="left")
    if history.lineage_id.isna().any() or history.start_utc.isna().any():
        raise ValueError("qualifying history rows without a lineage or a session start")
    return history.sort_values(["season", "round", "position"]).reset_index(drop=True)


def jolpica_tables(root):
    base = root / "external" / "jolpica" / "tables"

    def load(name):
        frame = pd.read_parquet(base / f"{name}.parquet")
        for column in ("season", "round"):
            if column in frame:
                frame[column] = frame[column].astype(int)
        return frame
    return load("races"), load("drivers"), load("constructors"), load("circuits")


def event_rows(root, races, sessions):
    rows = []
    for path in sorted((root / "schedules").glob("*.parquet")):
        schedule = pd.read_parquet(path)
        season = int(path.stem)
        if season < FIRST_SEASON:
            continue
        for event in schedule[schedule.RoundNumber > 0].itertuples():
            rows.append({"event_key": f"{season}-{int(event.RoundNumber):02d}", "season": season,
                         "round": int(event.RoundNumber), "is_testing": False,
                         "event_name": event.EventName, "official_name": event.OfficialEventName,
                         "event_format": event.EventFormat, "country": event.Country, "location": event.Location})
    events = pd.DataFrame(rows)
    circuits = races[["season", "round", "Circuit_circuitId", "date", "time"]].rename(
        columns={"Circuit_circuitId": "circuit_id", "date": "race_date", "time": "race_time_utc"})
    events = events.merge(circuits, on=["season", "round"], how="left")
    tests = (sessions[sessions.is_testing].groupby("event_key")
             .agg(season=("season", "first"), test_number=("test_number", "first"),
                  first_session=("scheduled_start_utc", "min")).reset_index())
    tests = tests.drop(columns="first_session").assign(is_testing=True, event_name="Pre-season testing",
                                                       event_format="testing")
    events["round"] = events["round"].astype("Int64")
    tests["test_number"] = tests["test_number"].astype("Int64")
    events["test_number"] = pd.array([pd.NA] * len(events), dtype="Int64")
    events = pd.concat([events, tests], ignore_index=True)
    return events.sort_values(["season", "is_testing", "round", "test_number"], na_position="first").reset_index(drop=True)


def _name_tokens(text):
    """Lower-case ASCII tokens of a name ("Nico Hülkenberg" -> {"nico", "hulkenberg"})."""
    plain = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    plain = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", plain)        # "LewisHamilton" -> "Lewis Hamilton"
    return set(re.findall(r"[a-z]+", plain.lower()))


def entry_lists(root, events, driver_table, entrants, sessions):
    """FIA entry lists resolved to event keys and driver IDs: one row per driver per
    published list, with its role ("race": a car's entered driver; "practice": an
    extra driver for first practice) and publication time. A name is matched to the
    drivers of that season (FastF1 entrants), then to every known driver; it must
    contain the driver's family name and the first given name (in either order:
    "Zhou Guanyu"). Unmatched names keep an empty driver ID."""
    path = root / "external" / "fia" / "tables" / "entry_lists.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=["event_key", "season", "driver_id", "driver_name", "raw_number", "role", "team",
                                     "published_utc", "doc_number"])
    lists = pd.read_parquet(path)
    lists["season"] = lists.season.astype(int)
    # the event whose weekend (3 days before its first session to its last) contains the
    # publication: FIA and FastF1 name some events differently (Brazilian / Sao Paulo)
    windows = sessions.groupby("event_key").scheduled_start_utc.agg(["min", "max"])
    published = pd.to_datetime(lists.published_utc, utc=True)
    lists["event_key"] = [next((key for key, (first, last) in windows.iterrows()
                                if first - pd.Timedelta(days=3) <= when <= last + pd.Timedelta(hours=6)), None)
                          for when in published]
    known = [(row.driver_id, _name_tokens(row.family_name), _name_tokens(str(row.given_name).split()[0]))
             for row in driver_table.itertuples()]
    by_season = entrants.groupby("season").driver_id.apply(lambda d: set(d.dropna())).to_dict()

    def resolve(name, season):
        tokens = _name_tokens(name)
        season_drivers = by_season.get(season, set())
        matches = [d for d, family, given in known if family <= tokens and given <= tokens]
        local = [d for d in matches if d in season_drivers]
        chosen = local or matches
        if len(chosen) != 1:   # spelling of the given name differs ("Nicolas"): a unique family name that season
            chosen = [d for d, family, given in known if family <= tokens and d in season_drivers]
        return chosen[0] if len(chosen) == 1 else None

    lists["driver_id"] = [resolve(n, s) for n, s in zip(lists.driver_name, lists.season)]
    columns = ["event_key", "season", "driver_id", "driver_name", "raw_number", "role", "team", "published_utc",
               "doc_number"]
    return lists[columns].sort_values(["published_utc", "doc_number"]).reset_index(drop=True)


def starting_grids(root, sessions, entrants):
    """FIA race starting grids (provisional and final, grid penalties applied)
    resolved to event keys: one row per grid slot with the car's driver number and
    the document's publication time. A grid is kept only if every car number in it
    is an entrant of that event's race (checked, never repaired); the rest are
    listed in the manifest."""
    columns = ["event_key", "season", "grid_kind", "grid_position", "driver_number", "published_utc", "doc_number",
               "file"]
    path = root / "external" / "fia" / "tables" / "starting_grids.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=columns), []
    grids = pd.read_parquet(path)
    grids["season"] = grids.season.astype(int)
    published = pd.to_datetime(grids.published_utc, utc=True)
    races = sessions[sessions.session_type == "R"]
    windows = races.set_index("event_key").scheduled_start_utc
    # the race the grid was published for: the first race start after it, within 3 days
    grids["event_key"] = [next((key for key, start in windows.items()
                                if pd.Timedelta(0) < start - when <= pd.Timedelta(days=3)), None) for when in published]
    grids["published_utc"] = published
    grids["driver_number"] = grids.raw_number.astype(str)
    race_entrants = (entrants[entrants.session_id.isin(set(races.session_id))]
                     .assign(driver_number=lambda e: e.driver_number.astype(str))
                     .groupby("event_key").driver_number.apply(set).to_dict())
    rejected, keep = [], []
    for file, grid in grids.groupby("file", sort=False):
        event = grid.event_key.iloc[0]
        unknown = set(grid.driver_number) - race_entrants.get(event, set()) if event else set(grid.driver_number)
        if event is None or unknown:
            rejected.append({"file": file, "event_key": event, "unknown_numbers": sorted(unknown)})
        else:
            keep.append(grid)
    kept = pd.concat(keep) if keep else pd.DataFrame(columns=columns)
    return kept[columns].sort_values(["published_utc", "grid_position"]).reset_index(drop=True), rejected


def build(root):
    """Rebuild every reference table from the archive. Returns the manifest."""
    out = root / "reference"
    races, drivers, constructors, circuits = jolpica_tables(root)
    sessions = session_rows(root)
    entrants = resolve_entrants(raw_entrants(root))
    events = event_rows(root, races, sessions)
    driver_ids = set(entrants.driver_id.dropna())
    driver_table = (drivers[drivers.season >= FIRST_SEASON].sort_values("season")
                    .drop_duplicates("driverId", keep="last").drop(columns="season")
                    .rename(columns={"driverId": "driver_id", "dateOfBirth": "date_of_birth",
                                     "familyName": "family_name", "givenName": "given_name",
                                     "permanentNumber": "permanent_number"}))
    constructors = constructors.rename(columns={"constructorId": "constructor_id"})
    lineage = lineage_table(constructors)
    team_table = (constructors[constructors.season >= LINEAGE_FIRST_SEASON]
                  [["season", "constructor_id", "name", "nationality"]]
                  .merge(lineage, on=["season", "constructor_id"], how="left"))
    entrants = entrants.merge(lineage.rename(columns={"constructor_id": "team_id"}),
                              on=["season", "team_id"], how="left")
    circuit_table = (circuits.sort_values("season").drop_duplicates("circuitId", keep="last").drop(columns="season")
                     .rename(columns={"circuitId": "circuit_id", "circuitName": "circuit_name",
                                      "Location_country": "country", "Location_locality": "locality",
                                      "Location_lat": "latitude", "Location_long": "longitude"}))
    history = qualifying_history(root, races, lineage)
    entries = entry_lists(root, events, driver_table, entrants, sessions)
    grids, rejected_grids = starting_grids(root, sessions, entrants)
    files = {}
    for name, frame in (("events", events), ("sessions", sessions), ("entrants", entrants),
                        ("drivers", driver_table), ("teams", team_table), ("circuits", circuit_table),
                        ("qualifying_history", history), ("entry_lists", entries), ("starting_grids", grids)):
        files[name] = write_table(frame, out / f"{name}.parquet", {})
    gaps = unresolved(entrants)
    manifest = {
        "schema_version": 1, "built_at_utc": utcnow(), "files": files,
        "entrants": {
            "rows": len(entrants),
            "driver_id_by_source": entrants.driver_source.value_counts(dropna=False).rename(str).to_dict(),
            "team_id_by_source": entrants.team_source.value_counts(dropna=False).rename(str).to_dict(),
            "driver_ids_not_in_jolpica": sorted(driver_ids - set(driver_table.driver_id)),
            "unresolved": gaps,
        },
        "lineages": int(team_table.lineage_id.nunique()),
        "qualifying_history": {"rows": len(history),
                               "events_per_season": history.groupby("season").event_key.nunique().rename(str).to_dict()},
        "entry_lists": {"rows": len(entries), "events": int(entries.event_key.nunique()),
                        "unresolved_names": sorted(set(entries.driver_name[entries.driver_id.isna()])),
                        "unmatched_events": int(entries.event_key.isna().sum())},
        "starting_grids": {"rows": len(grids), "events": int(grids.event_key.nunique()), "rejected": rejected_grids},
        "note": "Unresolved entrants keep empty IDs.",
    }
    atomic_json(out / "manifest.json", manifest)
    return manifest


def register(commands, common):
    parser = commands.add_parser("reference", parents=[common], help="canonical events, sessions, entrants, drivers, teams",
                                 description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build", "report", "verify"])
    parser.set_defaults(handler=run, parser=parser)


def run(args):
    if args.command in ("report", "verify"):
        from . import audit
        return audit.run_source("reference", args.data_root, args.command)
    manifest = build(args.data_root)
    print(json.dumps({key: value for key, value in manifest["entrants"].items() if key != "unresolved"}
                     | {"unresolved_groups": len(manifest["entrants"]["unresolved"]),
                        "lineages": manifest["lineages"]}, indent=2, default=str))
    return 0
