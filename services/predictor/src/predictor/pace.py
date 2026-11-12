"""Pace model, stage 1: car and driver pace as drifting latent states, learned from
qualifying.

Each car lineage and each driver has a latent pace (log lap time x 100, i.e. about
percent; lower is faster). A driver's pace in a session is car + driver + noise.
Between sessions every state drifts as a random walk whose variance grows with
the days elapsed, so long gaps (winter) leave estimates less certain.

Observations come from qualifying segments (Q1, Q2, Q3). Within a segment all
drivers share track conditions, so each segment contributes only relative
information: every driver's log time minus a reference driver's. Laps slower
than 107% of the segment's best are discarded. A Kalman filter updates the joint
state; teammates share the car state, which separates car from driver.

Hyperparameters (observation noise, drift rates, prior spreads for new cars and
drivers) are fitted by maximising the one-step predictive likelihood on sessions
available before the evaluated season starts.

The forecast draws car and driver states jointly from the posterior, adds session
noise, and ranks the draws.
"""

from dataclasses import dataclass
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .forecast import Forecast

SEGMENTS = ("Q1", "Q2", "Q3")
SLOW_CUTOFF = 1.07
DRAWS = 10000
DAY_SCALE = 14.0  # drift variance is per 14 days


@dataclass
class Hyper:
    noise: float = 0.3        # per-lap observation noise (% of lap time)
    car_drift: float = 0.05   # car variance added per 14 days
    driver_drift: float = 0.01
    car_prior: float = 1.0    # prior spread of a new car lineage
    driver_prior: float = 0.3

    def vector(self):
        return np.log([self.noise, self.car_drift, self.driver_drift, self.car_prior, self.driver_prior])

    @classmethod
    def from_vector(cls, values):
        return cls(*np.exp(values))


def segment_observations(results, entrants):
    """[(car keys, driver keys, log times x 100)] per qualifying segment."""
    ids = entrants.set_index("driver_number")
    observations = []
    for segment in SEGMENTS:
        if segment not in results:
            continue
        frame = pd.DataFrame({"number": results.DriverNumber.astype(str), "time": results[segment]}).dropna()
        if frame.empty:
            continue
        seconds = frame.time.dt.total_seconds()
        frame = frame[seconds <= seconds.min() * SLOW_CUTOFF]
        cars = [ids.lineage_id.get(n) for n in frame.number]
        drivers = [ids.driver_id.get(n) for n in frame.number]
        keep = [pd.notna(c) and pd.notna(d) for c, d in zip(cars, drivers)]
        if sum(keep) < 2:
            continue
        values = 100 * np.log(frame.time.dt.total_seconds().to_numpy()[keep])
        observations.append(([c for c, k in zip(cars, keep) if k], [d for d, k in zip(drivers, keep) if k], values))
    return observations


class Filter:
    """Joint Gaussian state over car lineages and drivers."""

    def __init__(self, hyper):
        self.hyper = hyper
        self.index = {}
        self.kind = []
        self.mean = np.zeros(0)
        self.cov = np.zeros((0, 0))
        self.time = None
        self.log_likelihood = 0.0

    def _add(self, key, kind):
        prior = self.hyper.car_prior if kind == "car" else self.hyper.driver_prior
        self.index[key] = len(self.kind)
        self.kind.append(kind)
        self.mean = np.append(self.mean, 0.0)
        size = len(self.kind)
        cov = np.zeros((size, size))
        cov[:-1, :-1] = self.cov
        cov[-1, -1] = prior ** 2
        self.cov = cov

    def column(self, key, kind):
        if (kind, key) not in self.index:
            self._add((kind, key), kind)
        return self.index[(kind, key)]

    def advance(self, time):
        if self.time is not None and time > self.time:
            steps = (time - self.time).total_seconds() / 86400 / DAY_SCALE
            drift = np.where(np.array(self.kind) == "car", self.hyper.car_drift, self.hyper.driver_drift)
            self.cov = self.cov + np.diag(drift * steps)
        self.time = time if self.time is None else max(self.time, time)

    def design(self, cars, drivers):
        columns = [(self.column(c, "car"), self.column(d, "driver")) for c, d in zip(cars, drivers)]
        rows = np.zeros((len(cars), len(self.kind)))
        for row, (car, driver) in enumerate(columns):
            rows[row, car] += 1
            rows[row, driver] += 1
        return rows

    def update(self, cars, drivers, values):
        """Relative update: every driver against the first one in the segment."""
        design = self.design(cars, drivers)
        h = design[1:] - design[0]
        z = values[1:] - values[0]
        n = len(z)
        r = self.hyper.noise ** 2 * (np.eye(n) + np.ones((n, n)))
        innovation = z - h @ self.mean
        s = h @ self.cov @ h.T + r
        chol = np.linalg.cholesky(s)
        solved = np.linalg.solve(chol, innovation)
        self.log_likelihood += -0.5 * (solved @ solved) - np.log(np.diag(chol)).sum() - 0.5 * n * np.log(2 * np.pi)
        gain = np.linalg.solve(s, h @ self.cov).T
        self.mean = self.mean + gain @ innovation
        self.cov = self.cov - gain @ h @ self.cov
        self.cov = (self.cov + self.cov.T) / 2


def qualifying_sessions(view, before=None):
    sessions = view.sessions[view.sessions.session_type == "Q"]
    if before is not None:
        sessions = sessions[sessions.available_at_utc < before]
    return sessions.sort_values("scheduled_start_utc")


def run_filter(view, hyper, sessions):
    state = Filter(hyper)
    for session in sessions.itertuples():
        state.advance(session.scheduled_start_utc)
        for cars, drivers, values in segment_observations(view.results(session.session_id),
                                                          view.entrants(session.session_id)):
            state.update(cars, drivers, values)
    return state


def fit_hyper(view, sessions, start=None):
    """Maximise the one-step predictive likelihood over `sessions`."""
    start = start or Hyper()
    cache = {s.session_id: segment_observations(view.results(s.session_id), view.entrants(s.session_id))
             for s in sessions.itertuples()}

    def negative(values):
        hyper = Hyper.from_vector(values)
        state = Filter(hyper)
        for session in sessions.itertuples():
            state.advance(session.scheduled_start_utc)
            for cars, drivers, obs in cache[session.session_id]:
                state.update(cars, drivers, obs)
        return -state.log_likelihood

    result = minimize(negative, start.vector(), method="Nelder-Mead",
                      options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-2})
    return Hyper.from_vector(result.x)


class PaceModel:
    """Model interface: fit(history, view_for) then predict(view, example). Qualifying only."""

    name = "pace:v1"

    def __init__(self):
        self.hyper = None
        self.fitted_season = None

    def fit(self, history, view_for):
        return self

    def _ensure_hyper(self, view, season):
        if self.fitted_season == season:
            return
        season_start = pd.Timestamp(f"{season}-01-01", tz="UTC")
        earlier = view.archive.view(season_start)
        sessions = qualifying_sessions(earlier)
        self.hyper = fit_hyper(earlier, sessions, self.hyper) if len(sessions) >= 5 else Hyper()
        self.fitted_season = season

    def predict(self, view, example, draws=DRAWS):
        if example.session_type != "Q":
            return None
        self._ensure_hyper(view, example.season)
        state = run_filter(view, self.hyper, qualifying_sessions(view))
        target_time = example.cutoff_time
        state.advance(target_time)
        lineages = list(example.lineage_ids)
        drivers = [d if pd.notna(d) else f"unknown:{n}" for d, n in zip(example.driver_ids, example.driver_numbers)]
        cars = [c if pd.notna(c) else f"unknown:{n}" for c, n in zip(lineages, example.driver_numbers)]
        design = state.design(cars, drivers)
        mean = design @ state.mean
        cov = design @ state.cov @ design.T + self.hyper.noise ** 2 * np.eye(len(cars))
        seed = zlib.crc32(f"{self.name}|{example.event_key}|{example.session_type}|{example.cutoff_label}".encode())
        rng = np.random.default_rng(seed)
        jitter = 1e-9 * np.eye(len(cars))
        samples = rng.multivariate_normal(mean, cov + jitter, size=draws, method="cholesky")
        order = np.argsort(samples, axis=1)
        positions = np.empty_like(order)
        positions[np.arange(draws)[:, None], order] = np.arange(1, len(cars) + 1)
        return Forecast.from_draws(
            positions=positions, season=example.season, session_type=example.session_type,
            event_key=example.event_key, cutoff_label=example.cutoff_label, cutoff_time=example.cutoff_time.isoformat(),
            driver_numbers=list(example.driver_numbers), driver_ids=list(example.driver_ids), model=self.name,
            seed=seed, metadata={"hyper": vars(self.hyper), "sessions": len(qualifying_sessions(view))})
