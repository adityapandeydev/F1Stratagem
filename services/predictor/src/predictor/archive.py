"""Cutoff-gated access to the archive. Every read the prediction engine makes goes
through an `ArchiveView`, which only exposes data available at its cutoff.

Historical sessions have no recorded publication time, so availability is
estimated (`archive_proxy` mode): a session's data is available once its last lap
has ended plus a publication delay. Sessions without laps fall back to the
scheduled start plus a nominal duration. Results tables may contain corrections
made after the session (penalties, disqualifications); forecasts built on them
are labelled retrospective.
"""

from dataclasses import dataclass
import functools

import pandas as pd

from f1data.paths import default_data_root

PUBLICATION_DELAY = {
    "FP1": pd.Timedelta(minutes=30), "FP2": pd.Timedelta(minutes=30), "FP3": pd.Timedelta(minutes=30),
    "Q": pd.Timedelta(minutes=30), "SQ": pd.Timedelta(minutes=30), "S": pd.Timedelta(minutes=60),
    "R": pd.Timedelta(minutes=60), "T": pd.Timedelta(minutes=30),
}
NOMINAL_DURATION = {
    "FP1": pd.Timedelta(minutes=90), "FP2": pd.Timedelta(minutes=90), "FP3": pd.Timedelta(minutes=60),
    "Q": pd.Timedelta(minutes=75), "SQ": pd.Timedelta(minutes=60), "S": pd.Timedelta(minutes=60),
    "R": pd.Timedelta(hours=3), "T": pd.Timedelta(hours=9),
}
SESSION_ORDER = ("T", "FP1", "FP2", "FP3", "SQ", "S", "Q", "R")


class CutoffViolation(LookupError):
    """A read asked for data that was not available at the view's cutoff."""


@dataclass(frozen=True)
class Cutoff:
    event_key: str
    label: str
    time: pd.Timestamp
    anchor_session: str | None


def available_at(sessions):
    """Estimated time each session's data became available."""
    end = sessions["last_lap_end_utc"].fillna(
        sessions["scheduled_start_utc"] + sessions["session_type"].map(NOMINAL_DURATION))
    return end + sessions["session_type"].map(PUBLICATION_DELAY)


class Archive:
    """Reference tables and raw session tables of one data root."""

    def __init__(self, root=None):
        self.root = root or default_data_root()
        reference = self.root / "reference"
        self.events = pd.read_parquet(reference / "events.parquet")
        sessions = pd.read_parquet(reference / "sessions.parquet")
        sessions["available_at_utc"] = available_at(sessions)
        self.sessions = sessions.sort_values("available_at_utc").reset_index(drop=True)
        self.entrants = pd.read_parquet(reference / "entrants.parquet")
        history = reference / "qualifying_history.parquet"
        self.qualifying_history = pd.read_parquet(history) if history.exists() else None
        if self.qualifying_history is not None:
            self.qualifying_history["available_at_utc"] = (self.qualifying_history.start_utc + NOMINAL_DURATION["Q"]
                                                           + PUBLICATION_DELAY["Q"])

    def event_sessions(self, event_key):
        return self.sessions[self.sessions.event_key == event_key].sort_values("scheduled_start_utc")

    def cutoffs(self, event_key):
        """Forecast cutoffs for one event: before the weekend, then after each session."""
        sessions = self.event_sessions(event_key)
        if sessions.empty:
            return []
        first = sessions.scheduled_start_utc.min()
        cutoffs = [Cutoff(event_key, "pre_weekend", first - pd.Timedelta(minutes=1), None)]
        for row in sessions.itertuples():
            cutoffs.append(Cutoff(event_key, f"after_{row.session_type}", row.available_at_utc, row.session_id))
        return cutoffs

    def view(self, cutoff):
        time = cutoff.time if isinstance(cutoff, Cutoff) else pd.Timestamp(cutoff)
        return ArchiveView(self, time)

    @functools.lru_cache(maxsize=4096)
    def _table(self, session_id, table):
        season = session_id[:4]
        return pd.read_parquet(self.root / "sessions" / season / session_id / "normalized" / f"{table}.parquet")


class ArchiveView:
    """The archive as it could have been read at `cutoff`."""

    def __init__(self, archive, cutoff):
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware (UTC)")
        self.archive = archive
        self.cutoff = cutoff
        self.sessions = archive.sessions[archive.sessions.available_at_utc <= cutoff]
        self._available = set(self.sessions.session_id)

    def _check(self, session_id):
        if session_id not in self._available:
            raise CutoffViolation(f"{session_id} is not available at {self.cutoff.isoformat()}")

    def laps(self, session_id):
        self._check(session_id)
        return self.archive._table(session_id, "laps").copy()

    def results(self, session_id):
        self._check(session_id)
        return self.archive._table(session_id, "results").copy()

    def telemetry(self, session_id, driver_number, kind="car_data"):
        """One driver's raw telemetry of a session: kind "car_data" (speed, throttle,
        brake, gear, DRS) or "pos_data" (X, Y, Z); None if the archive has none."""
        self._check(session_id)
        if kind not in ("car_data", "pos_data"):
            raise ValueError(kind)
        path = self.archive.root / "sessions" / session_id[:4] / session_id / "source" / kind / f"driver-{driver_number}.parquet"
        return pd.read_parquet(path) if path.exists() else None

    def entrants(self, session_id):
        self._check(session_id)
        return self.archive.entrants[self.archive.entrants.session_id == session_id].copy()

    def qualifying_history(self):
        """Qualifying times before the FastF1 archive (Jolpica, one row per driver per
        event) available at the cutoff; empty if the reference build has none."""
        history = self.archive.qualifying_history
        if history is None:
            return pd.DataFrame(columns=["event_key", "season", "circuit_id", "driver_id", "lineage_id",
                                         "q1_seconds", "q2_seconds", "q3_seconds", "start_utc", "available_at_utc"])
        return history[history.available_at_utc <= self.cutoff].copy()

    def standings(self, season):
        """Drivers' points from races and sprints of `season` available at the cutoff.
        Includes every point actually awarded (fastest-lap bonus included)."""
        scored = self.sessions[(self.sessions.season == season) & self.sessions.session_type.isin(["R", "S"])]
        totals = {}
        for session_id in scored.session_id:
            results = self.results(session_id)
            ids = self.entrants(session_id).set_index("driver_number").driver_id
            for number, points in zip(results.DriverNumber, results.Points.fillna(0)):
                driver = ids.get(number)
                if pd.notna(driver):
                    totals[driver] = totals.get(driver, 0.0) + float(points)
        return pd.Series(totals, dtype=float).sort_values(ascending=False)
