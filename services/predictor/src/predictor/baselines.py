"""Reference models every component of the prediction engine must beat.

A baseline ranks the entrants from one piece of information available at the
cutoff and turns the ranking into probabilities with a Plackett-Luce model:
log-strength = -beta x (rank - 1). `beta` (how decisive the ranking is) is fitted
walk-forward by maximum likelihood on earlier events only. For races and sprints
each entrant also retires with the historical retirement rate; retired cars are
classified behind finishers, split mechanical / incident by the historical share.

    uniform        no information: every order equally likely
    championship   drivers' standings before the event (last season's at round 1)
    persistence    the most recent qualifying classification before the event
    practice       best lap of the weekend's practice sessions so far
    qualifying     the event's qualifying classification
"""

from dataclasses import dataclass, field
import zlib

import numpy as np
import pandas as pd

from .forecast import Forecast
from .rules import points_positions
from .simulate import log_likelihood, sample_orders
from .targets import OUTCOMES, RETIREMENTS

BETA_GRID = np.round(np.concatenate([np.arange(0.0, 0.5, 0.02), np.arange(0.5, 2.01, 0.1)]), 3)
DRAWS = 4000


def dense_ranks(values):
    """1-based ranks; ties share the average rank; missing values rank last."""
    series = pd.Series(values, dtype=float)
    ranks = series.rank(method="average", na_option="keep")
    filler = (ranks.max() if ranks.notna().any() else 0) + 1
    return ranks.fillna(filler).to_numpy()


@dataclass
class Example:
    """One event's target session seen from one cutoff. Models never read the
    actual result fields; they receive an ArchiveView at `cutoff_time`."""
    event_key: str
    season: int
    session_type: str
    cutoff_label: str
    cutoff_time: pd.Timestamp
    target_available_at: pd.Timestamp
    driver_numbers: list
    driver_ids: list
    positions: np.ndarray        # actual full-order positions, aligned with driver_numbers
    outcomes: list               # actual outcomes (races and sprints), aligned
    excluded: np.ndarray
    band_rank: np.ndarray        # championship rank at the cutoff, for reporting only
    cutoff_session: str | None = None
    team_ids: list = field(default_factory=list)
    lineage_ids: list = field(default_factory=list)


# Rankers: information available at the cutoff -> one value per entrant (lower is better).
def rank_uniform(view, example):
    return np.zeros(len(example.driver_numbers))


def rank_championship(view, example):
    standings = view.standings(example.season)
    if standings.empty:
        standings = view.standings(example.season - 1)
    order = {driver: position for position, driver in enumerate(standings.index, start=1)}
    return np.array([order.get(driver, np.nan) for driver in example.driver_ids], dtype=float)


def rank_persistence(view, example):
    earlier = view.sessions[(view.sessions.session_type == "Q") & (view.sessions.event_key != example.event_key)]
    if earlier.empty:
        return None
    session_id = earlier.sort_values("available_at_utc").session_id.iloc[-1]
    results = view.results(session_id)
    ids = view.entrants(session_id).set_index("driver_number").driver_id
    positions = {ids.get(str(n)): p for n, p in zip(results.DriverNumber.astype(str), results.Position)}
    return np.array([positions.get(driver, np.nan) for driver in example.driver_ids], dtype=float)


def rank_practice(view, example):
    sessions = view.sessions[(view.sessions.event_key == example.event_key)
                             & view.sessions.session_type.isin(["FP1", "FP2", "FP3"])]
    if sessions.empty:
        return None
    best = {}
    for session_id in sessions.session_id:
        laps = view.laps(session_id)
        laps = laps[laps.LapTime.notna() & ~laps.Deleted.fillna(False).astype(bool)]
        for number, time in laps.groupby("DriverNumber").LapTime.min().items():
            best[str(number)] = min(best.get(str(number), np.inf), time.total_seconds())
    return np.array([best.get(number, np.nan) for number in example.driver_numbers], dtype=float)


def rank_qualifying(view, example):
    sessions = view.sessions[(view.sessions.event_key == example.event_key) & (view.sessions.session_type == "Q")]
    if sessions.empty:
        return None
    results = view.results(sessions.session_id.iloc[0])
    positions = dict(zip(results.DriverNumber.astype(str), results.Position))
    return np.array([positions.get(number, np.nan) for number in example.driver_numbers], dtype=float)


RANKERS = {"uniform": rank_uniform, "championship": rank_championship, "persistence": rank_persistence,
           "practice": rank_practice, "qualifying": rank_qualifying}


def seed_for(example):
    """Random seed of an example, the same for every model: paired comparisons then
    share their random draws (common random numbers), so a difference between two
    models is not inflated by their sampling noise."""
    return zlib.crc32(f"{example.event_key}|{example.session_type}|{example.cutoff_label}".encode())


class RetirementModel:
    """Historical retirement rate and mechanical share, applied to simulated orders."""

    def __init__(self):
        self.retire_rate = 0.0
        self.mechanical_share = 0.5

    def fit(self, history):
        outcomes = [o for e in history for o, x in zip(e.outcomes, e.excluded) if o is not None and not x]
        if outcomes:
            retired = [o for o in outcomes if o in RETIREMENTS]
            self.retire_rate = len(retired) / len(outcomes)
            resolved = [o for o in retired if o in ("mechanical", "incident_or_other")]
            if resolved:
                self.mechanical_share = resolved.count("mechanical") / len(resolved)
        return self

    def apply(self, positions, example, rng):
        draws, n = positions.shape
        retired = rng.random((draws, n)) < self.retire_rate
        # Finishers keep their relative order; retirements are classified behind them.
        key = positions + retired * (n + 1)
        order = np.argsort(key, axis=1)
        final = np.empty_like(positions)
        final[np.arange(draws)[:, None], order] = np.arange(1, n + 1)
        paying = points_positions(example.season, example.session_type)
        outcomes = np.where(final == 1, OUTCOMES.index("win"),
                            np.where(final <= paying, OUTCOMES.index("other_points"), OUTCOMES.index("outside_points")))
        mechanical = rng.random((draws, n)) < self.mechanical_share
        outcomes = np.where(retired, np.where(mechanical, OUTCOMES.index("mechanical"),
                                              OUTCOMES.index("incident_or_other")), outcomes)
        return final, outcomes


def forecast_from_positions(name, example, positions, rng, retirements, seed, metadata):
    outcomes = None
    if example.session_type in ("R", "S"):
        positions, outcomes = retirements.apply(positions, example, rng)
    return Forecast.from_draws(
        positions=positions, outcomes=outcomes, season=example.season, session_type=example.session_type,
        event_key=example.event_key, cutoff_label=example.cutoff_label, cutoff_time=example.cutoff_time.isoformat(),
        driver_numbers=list(example.driver_numbers), driver_ids=list(example.driver_ids), model=name, seed=seed,
        metadata=metadata)


class RankBaseline:
    """Model interface: fit(history, view_for) then predict(view, example)."""

    def __init__(self, ranker):
        self.ranker = ranker
        self.name = f"baseline:{ranker}"
        self.beta = 0.0
        self.retirements = RetirementModel()
        self._ranks = {}

    def ranks(self, view, example):
        key = (example.event_key, example.session_type, example.cutoff_label)
        if key not in self._ranks:
            values = RANKERS[self.ranker](view, example)
            self._ranks[key] = None if values is None else dense_ranks(values)
        return self._ranks[key]

    def fit(self, history, view_for):
        usable = [(e, self.ranks(view_for(e), e)) for e in history]
        usable = [(e, r) for e, r in usable if r is not None]
        if self.ranker != "uniform" and usable:
            # excluded drivers (withdrawn, not classified) take no part in the ranking likelihood
            scores = [sum(log_likelihood(-beta * (r[~e.excluded] - 1), e.positions[~e.excluded]) for e, r in usable)
                      for beta in BETA_GRID]
            self.beta = float(BETA_GRID[int(np.argmax(scores))])
        self.retirements.fit(history)
        return self

    def predict(self, view, example, draws=DRAWS):
        ranks = self.ranks(view, example)
        if ranks is None:
            return None
        seed = seed_for(example)
        rng = np.random.default_rng(seed)
        positions = sample_orders(-self.beta * (ranks - 1), draws, rng)
        return forecast_from_positions(self.name, example, positions, rng, self.retirements, seed,
                                       {"beta": self.beta, "retire_rate": self.retirements.retire_rate,
                                        "mechanical_share": self.retirements.mechanical_share})
