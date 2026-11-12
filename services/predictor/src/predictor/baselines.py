"""Reference models every component of the prediction engine must beat.

A baseline ranks the entrants from one piece of information available at the
cutoff and turns the ranking into probabilities with a Plackett-Luce model:
log-strength = -beta x (rank - 1). `beta` (how decisive the ranking is) is fitted
walk-forward by maximum likelihood on earlier events only. For races and sprints
each entrant also retires with the historical retirement rate; retired cars are
classified behind finishers, split mechanical / incident by the historical share.

    uniform        no information: every order equally likely
    championship   drivers' standings before the event (last season's at round 1)
    practice       best lap of the weekend's practice sessions so far
    qualifying     the event's qualifying classification
"""

from dataclasses import dataclass
import zlib

import numpy as np
import pandas as pd

from .forecast import Forecast
from .rules import points_positions
from .simulate import log_likelihood, sample_orders
from .targets import OUTCOMES, RETIREMENTS

BETA_GRID = np.round(np.concatenate([np.arange(0.0, 0.5, 0.02), np.arange(0.5, 2.01, 0.1)]), 3)
DRAWS = 4000


def _dense_ranks(values):
    """1-based ranks; ties share the average rank; missing values rank last."""
    series = pd.Series(values, dtype=float)
    ranks = series.rank(method="average", na_option="keep")
    filler = (ranks.max() if ranks.notna().any() else 0) + 1
    return ranks.fillna(filler).to_numpy()


# Rankers: information available at the cutoff -> one value per entrant (lower is better).
def rank_uniform(view, example):
    return np.zeros(len(example.driver_numbers))


def rank_championship(view, example):
    standings = view.standings(example.season)
    if standings.empty:
        standings = view.standings(example.season - 1)
    order = {driver: position for position, driver in enumerate(standings.index, start=1)}
    return np.array([order.get(driver, np.nan) for driver in example.driver_ids], dtype=float)


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
            seconds = time.total_seconds()
            best[str(number)] = min(best.get(str(number), np.inf), seconds)
    return np.array([best.get(number, np.nan) for number in example.driver_numbers], dtype=float)


def rank_qualifying(view, example):
    sessions = view.sessions[(view.sessions.event_key == example.event_key) & (view.sessions.session_type == "Q")]
    if sessions.empty:
        return None
    results = view.results(sessions.session_id.iloc[0])
    positions = dict(zip(results.DriverNumber.astype(str), results.Position))
    return np.array([positions.get(number, np.nan) for number in example.driver_numbers], dtype=float)


RANKERS = {"uniform": rank_uniform, "championship": rank_championship,
           "practice": rank_practice, "qualifying": rank_qualifying}


@dataclass
class Example:
    """One event's target session seen from one cutoff."""
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
    ranks: dict                  # ranker name -> dense ranks (or None if not applicable)


class RankBaseline:
    def __init__(self, ranker):
        self.name = ranker
        self.beta = 0.0
        self.retire_rate = 0.0
        self.mechanical_share = 0.5

    def fit(self, examples):
        usable = [e for e in examples if e.ranks.get(self.name) is not None]
        if self.name != "uniform" and usable:
            scores = [sum(log_likelihood(-beta * (e.ranks[self.name] - 1), e.positions) for e in usable)
                      for beta in BETA_GRID]
            self.beta = float(BETA_GRID[int(np.argmax(scores))])
        outcomes = [o for e in examples for o, x in zip(e.outcomes, e.excluded) if o is not None and not x]
        if outcomes:
            retired = [o for o in outcomes if o in RETIREMENTS]
            self.retire_rate = len(retired) / len(outcomes)
            resolved = [o for o in retired if o in ("mechanical", "incident_or_other")]
            if resolved:
                self.mechanical_share = resolved.count("mechanical") / len(resolved)
        return self

    def predict(self, example, draws=DRAWS):
        ranks = example.ranks.get(self.name)
        if ranks is None:
            return None
        seed = zlib.crc32(f"{self.name}|{example.event_key}|{example.session_type}|{example.cutoff_label}".encode())
        rng = np.random.default_rng(seed)
        n = len(example.driver_numbers)
        positions = sample_orders(-self.beta * (ranks - 1), draws, rng)
        outcomes = None
        if example.session_type in ("R", "S"):
            positions, outcomes = self._retirements(positions, example, rng)
        return Forecast.from_draws(
            positions=positions, outcomes=outcomes, season=example.season, session_type=example.session_type,
            event_key=example.event_key, cutoff_label=example.cutoff_label,
            cutoff_time=example.cutoff_time.isoformat(), driver_numbers=list(example.driver_numbers),
            driver_ids=list(example.driver_ids), model=f"baseline:{self.name}", seed=seed,
            metadata={"beta": self.beta, "retire_rate": self.retire_rate, "mechanical_share": self.mechanical_share,
                      "entrants": n})

    def _retirements(self, positions, example, rng):
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
