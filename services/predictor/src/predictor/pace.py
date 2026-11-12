"""Pace model: car and driver pace as drifting latent states.

Each car lineage and each driver has a latent pace (log lap time x 100, about
percent; lower is faster). A driver's pace in a session is

    car + driver + offset(car lineage, session type) + noise(session type)

Qualifying is the reference (no offset). Every other session type (sprint
qualifying, FP1-FP3, testing) has its own noise level and, per car lineage, a
latent offset with a zero-mean prior that drifts slowly. The offsets learn
systematic patterns such as a team that is flattered by practice (engine modes,
low fuel) only as far as the data supports them. Between sessions every state
drifts as a random walk whose variance grows with the days elapsed. With winter
inflation enabled (stage 3), the start of each season adds further variance to
every car and driver, multiplied for cars in seasons that begin under new
technical regulations (2022, 2026): early-season sessions then move estimates a
lot, mid-season ones little.

Observations are relative within a group of laps run under shared conditions:
each qualifying segment (Q1-Q3, SQ1-SQ3), or each practice / testing session
(every driver's best clean lap: no in or out laps, not deleted). Each driver is
compared with a reference driver of the same group, and laps slower than 107%
of the group's best are discarded. A Kalman filter updates the joint state;
teammates share the car state, which separates car from driver.

Hyperparameters are fitted on sessions available before the evaluated season
starts. With the default objective they maximise the one-step predictive
likelihood of every observation. With the qualifying objective (v2q, v3q) every
session still updates the state, but the settings maximise only the predictive
likelihood of qualifying, at the two horizons the model forecasts: from the
state before the weekend's first session, and from the state after the
weekend's practice. Practice then informs qualifying without its own noise
patterns deciding how fast car pace may drift. The forecast draws all
states jointly from the posterior, adds qualifying noise and ranks the draws.

Weekend term (v2w, v3w): each car also has a weekend effect shared by every
session of one event (practice, sprint qualifying and qualifying alike) and reset
to its prior at the next event. Track-specific swings, setup and conditions go
there instead of moving the car's long-run pace. Before the weekend the effect is
unknown (prior spread, shared by teammates); after practice it carries what that
weekend's practice showed into the same weekend's qualifying.

Frozen shared settings (v2s): car and driver drift, priors and qualifying noise
are fitted on qualifying alone, then frozen; only practice noise and offsets are
fitted on the full evidence.
"""

from dataclasses import dataclass, field
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .forecast import Forecast
from .rules import REGULATION_CHANGES

SEGMENTS = ("Q1", "Q2", "Q3")
SEGMENTED = frozenset({"Q", "SQ"})
SLOW_CUTOFF = 1.07
DRAWS = 10000
DAY_SCALE = 14.0  # drift variances are per 14 days
QUALIFYING_ONLY = ("Q",)
ALL_SESSIONS = ("Q", "SQ", "FP1", "FP2", "FP3", "T")


@dataclass
class Hyper:
    noise: dict = field(default_factory=lambda: {"Q": 0.3, "SQ": 0.35, "FP1": 0.8, "FP2": 0.7, "FP3": 0.6,
                                                 "T": 1.2})
    car_drift: float = 0.05     # variances added per 14 days
    driver_drift: float = 0.01
    offset_drift: float = 0.005
    car_prior: float = 1.0      # prior spreads of new states
    driver_prior: float = 0.3
    offset_prior: float = 0.3
    car_winter: float = 0.0     # variances added at each season start (stage 3)
    driver_winter: float = 0.0
    regulation_multiplier: float = 1.0   # car winter variance multiplier at new regulations
    weekend_prior: float = 0.0  # spread of a car's weekend effect; 0 disables it

    SCALARS = ("car_drift", "driver_drift", "offset_drift", "car_prior", "driver_prior", "offset_prior",
               "car_winter", "driver_winter", "regulation_multiplier", "weekend_prior")
    SHARED = ("car_drift", "driver_drift", "car_prior", "driver_prior")

    @staticmethod
    def fitted(types, winter=False, regulation=False, weekend=False):
        """Scalars that affect the likelihood. Offsets exist only for non-qualifying
        sessions; winter terms only if the data crosses a season start; the regulation
        multiplier only if it crosses a season that began under new regulations."""
        offsets = any(t != "Q" for t in types)
        return ("car_drift", "driver_drift", *(("offset_drift",) if offsets else ()), "car_prior", "driver_prior",
                *(("offset_prior",) if offsets else ()), *(("car_winter", "driver_winter") if winter else ()),
                *(("regulation_multiplier",) if winter and regulation else ()),
                *(("weekend_prior",) if weekend else ()))

    def vector(self, noise_types, scalars):
        return np.log([self.noise[t] for t in noise_types] + [getattr(self, name) for name in scalars])

    @classmethod
    def from_vector(cls, values, noise_types, scalars, base=None):
        values = np.exp(values)
        base = base or cls()
        noise = {**base.noise, **dict(zip(noise_types, values[:len(noise_types)]))}
        fields = {name: getattr(base, name) for name in cls.SCALARS}
        fields.update(zip(scalars, values[len(noise_types):]))
        return cls(noise=noise, **fields)


def _lap_groups(session_type, results, laps):
    """Groups of (driver number, seconds) run under shared conditions."""
    if session_type in SEGMENTED:
        for segment in SEGMENTS:
            if segment not in results or results[segment].isna().all():
                continue
            frame = pd.DataFrame({"number": results.DriverNumber.astype(str),
                                  "seconds": pd.to_timedelta(results[segment]).dt.total_seconds()}).dropna()
            if len(frame) >= 2:
                yield frame
        return
    clean = laps[laps.LapTime.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna()
                 & ~laps.Deleted.fillna(False).astype(bool)]
    best = clean.groupby(clean.DriverNumber.astype(str)).LapTime.min().dt.total_seconds()
    if len(best) >= 2:
        yield pd.DataFrame({"number": best.index, "seconds": best.to_numpy()})


def session_observations(session_type, results, laps, entrants):
    """[(car keys, driver keys, log times x 100)] for one session."""
    ids = entrants.set_index("driver_number")
    observations = []
    for frame in _lap_groups(session_type, results, laps):
        frame = frame[frame.seconds <= frame.seconds.min() * SLOW_CUTOFF]
        cars = [ids.lineage_id.get(n) if "lineage_id" in ids else None for n in frame.number]
        drivers = [ids.driver_id.get(n) for n in frame.number]
        keep = [pd.notna(c) and pd.notna(d) for c, d in zip(cars, drivers)]
        if sum(keep) < 2:
            continue
        values = 100 * np.log(frame.seconds.to_numpy()[keep])
        observations.append(([c for c, k in zip(cars, keep) if k], [d for d, k in zip(drivers, keep) if k], values))
    return observations


class Filter:
    """Joint Gaussian state over car lineages, drivers and car x session-type offsets."""

    def __init__(self, hyper):
        self.hyper = hyper
        self.index = {}
        self.kind = []
        self.mean = np.zeros(0)
        self.cov = np.zeros((0, 0))
        self.time = None
        self.log_likelihood = 0.0
        self.priors = {"car": hyper.car_prior, "driver": hyper.driver_prior, "offset": hyper.offset_prior,
                       "weekend": hyper.weekend_prior}
        self.drifts = {"car": hyper.car_drift, "driver": hyper.driver_drift, "offset": hyper.offset_drift,
                       "weekend": 0.0}
        self.weekend = hyper.weekend_prior > 0
        self.event = None

    def column(self, key, kind):
        if (kind, key) not in self.index:
            self.index[(kind, key)] = len(self.kind)
            self.kind.append(kind)
            self.mean = np.append(self.mean, 0.0)
            size = len(self.kind)
            cov = np.zeros((size, size))
            cov[:-1, :-1] = self.cov
            cov[-1, -1] = self.priors[kind] ** 2
            self.cov = cov
        return self.index[(kind, key)]

    def copy(self):
        clone = Filter.__new__(Filter)
        clone.__dict__.update(self.__dict__)
        clone.index, clone.kind = dict(self.index), list(self.kind)
        clone.mean, clone.cov = self.mean.copy(), self.cov.copy()
        return clone

    def start_event(self, event):
        """A new event: every car's weekend effect returns to its prior, uncorrelated."""
        self.event = event
        if not self.weekend:
            return
        for index, kind in enumerate(self.kind):
            if kind == "weekend":
                self.mean[index] = 0.0
                self.cov[index, :] = 0.0
                self.cov[:, index] = 0.0
                self.cov[index, index] = self.hyper.weekend_prior ** 2

    def advance(self, time):
        if self.time is not None and time > self.time:
            steps = (time - self.time).total_seconds() / 86400 / DAY_SCALE
            added = np.array([self.drifts[k] * steps for k in self.kind])
            for season in range(self.time.year + 1, time.year + 1):
                car = self.hyper.car_winter * (self.hyper.regulation_multiplier if season in REGULATION_CHANGES else 1)
                added = added + np.array([car if k == "car" else self.hyper.driver_winter if k == "driver" else 0.0
                                          for k in self.kind])
            self.cov = self.cov + np.diag(added)
        self.time = time if self.time is None else max(self.time, time)

    def design(self, cars, drivers, session_type="Q"):
        columns = []
        for car, driver in zip(cars, drivers):
            row = [self.column(car, "car"), self.column(driver, "driver")]
            if session_type != "Q":
                row.append(self.column((car, session_type), "offset"))
            if self.weekend:
                row.append(self.column(car, "weekend"))
            columns.append(row)
        design = np.zeros((len(cars), len(self.kind)))
        for row, indices in enumerate(columns):
            design[row, indices] += 1
        return design

    def update(self, cars, drivers, values, session_type="Q"):
        """Relative update: every driver against the first one in the group."""
        design = self.design(cars, drivers, session_type)
        h = design[1:] - design[0]
        z = values[1:] - values[0]
        n = len(z)
        r = self.hyper.noise[session_type] ** 2 * (np.eye(n) + np.ones((n, n)))
        innovation = z - h @ self.mean
        s = h @ self.cov @ h.T + r
        chol = np.linalg.cholesky(s)
        solved = np.linalg.solve(chol, innovation)
        self.log_likelihood += -0.5 * (solved @ solved) - np.log(np.diag(chol)).sum() - 0.5 * n * np.log(2 * np.pi)
        gain = np.linalg.solve(s, h @ self.cov).T
        self.mean = self.mean + gain @ innovation
        self.cov = self.cov - gain @ h @ self.cov
        self.cov = (self.cov + self.cov.T) / 2


class PaceModel:
    """Model interface: fit(history, view_for) then predict(view, example). Qualifying only.

    `session_types` selects the evidence: ("Q",) is stage 1, ALL_SESSIONS stage 2."""

    def __init__(self, session_types=QUALIFYING_ONLY, name="pace:v1", winter=False, objective="all",
                 weekend=False, shared_from_qualifying=False):
        self.session_types = tuple(session_types)
        self.name = name
        self.winter = winter
        self.objective = objective
        self.weekend = weekend
        self.shared_from_qualifying = shared_from_qualifying
        self.hyper = None
        self.fitted_season = None
        self.fit_log = {}
        self._qualifying_hyper = None
        self._observations = {}

    def sessions(self, view, before=None):
        sessions = view.sessions[view.sessions.session_type.isin(self.session_types)]
        if before is not None:
            sessions = sessions[sessions.available_at_utc < before]
        return sessions.sort_values("scheduled_start_utc")

    def observations(self, view, session):
        """Cached per session; the view still decides whether the session may be read."""
        if session.session_id not in self._observations:
            laps = view.laps(session.session_id) if session.session_type not in SEGMENTED else None
            self._observations[session.session_id] = session_observations(
                session.session_type, view.results(session.session_id), laps, view.entrants(session.session_id))
        return self._observations[session.session_id]

    def run(self, view, hyper, sessions):
        """Filter every session. `state.objective` is the quantity the settings maximise."""
        state = Filter(hyper)
        objective, before_weekend, event = 0.0, None, None
        horizons = {"before_weekend": 0.0, "after_practice": 0.0}
        for session in sessions.itertuples():
            state.advance(session.scheduled_start_utc)
            if session.event_key != event:
                event = session.event_key
                state.start_event(event)
                before_weekend = state.copy()
            groups = self.observations(view, session)
            if self.objective == "qualifying" and session.session_type == "Q":
                before_weekend.advance(session.scheduled_start_utc)
                start_pre, start_post = before_weekend.log_likelihood, state.log_likelihood
                for cars, drivers, values in groups:
                    before_weekend.update(cars, drivers, values, "Q")
                    state.update(cars, drivers, values, "Q")
                horizons["before_weekend"] += before_weekend.log_likelihood - start_pre
                horizons["after_practice"] += state.log_likelihood - start_post
                objective += (before_weekend.log_likelihood - start_pre) + (state.log_likelihood - start_post)
                continue
            for cars, drivers, values in groups:
                state.update(cars, drivers, values, session.session_type)
        state.objective = objective if self.objective == "qualifying" else state.log_likelihood
        state.horizons = horizons
        return state

    def fit_hyper(self, view, sessions, start, frozen=()):
        """Fit the settings that affect the objective, except those named in `frozen`."""
        types = [t for t in self.session_types if t in set(sessions.session_type)]
        for session in sessions.itertuples():
            self.observations(view, session)
        seasons = sorted(set(sessions.scheduled_start_utc.dt.year))
        winter = self.winter and len(seasons) > 1
        regulation = winter and any(s in REGULATION_CHANGES for s in seasons[1:])
        scalars = [s for s in Hyper.fitted(types, winter, regulation, self.weekend) if s not in frozen]
        noise_types = [t for t in types if f"noise:{t}" not in frozen]

        def negative(values):
            return -self.run(view, Hyper.from_vector(values, noise_types, scalars, start), sessions).objective

        dimensions = len(noise_types) + len(scalars)
        result = minimize(negative, start.vector(noise_types, scalars), method="Nelder-Mead",
                          options={"maxiter": 80 * dimensions, "xatol": 1e-3, "fatol": 1e-2})
        self.last_fit = {"converged": bool(result.success), "iterations": int(result.nit),
                         "objective": -float(result.fun), "fitted": noise_types + scalars}
        return Hyper.from_vector(result.x, noise_types, scalars, start)

    def fit(self, history, view_for):
        return self

    def _ensure_hyper(self, view, season):
        if self.fitted_season == season:
            return
        earlier = view.archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        sessions = self.sessions(earlier)
        start = self.hyper or Hyper(**({"car_winter": 0.5, "driver_winter": 0.05, "regulation_multiplier": 3.0}
                                       if self.winter else {}), **({"weekend_prior": 0.3} if self.weekend else {}))
        self.last_fit = {"converged": None, "note": "too little data; starting values used"}
        if (sessions.session_type == "Q").sum() < 5:
            self.hyper = start
        elif self.shared_from_qualifying:
            qualifying = PaceModel(QUALIFYING_ONLY)
            self._qualifying_hyper = qualifying.fit_hyper(earlier, qualifying.sessions(earlier),
                                                          self._qualifying_hyper or Hyper())
            for name in Hyper.SHARED:
                setattr(start, name, getattr(self._qualifying_hyper, name))
            start.noise["Q"] = self._qualifying_hyper.noise["Q"]
            self.hyper = self.fit_hyper(earlier, sessions, start, frozen=(*Hyper.SHARED, "noise:Q"))
        else:
            self.hyper = self.fit_hyper(earlier, sessions, start)
        self.fit_log[season] = self.last_fit
        self.fitted_season = season

    def predict(self, view, example, draws=DRAWS):
        if example.session_type != "Q":
            return None
        self._ensure_hyper(view, example.season)
        state = self.run(view, self.hyper, self.sessions(view))
        state.advance(example.cutoff_time)
        if state.event != example.event_key:
            state.start_event(example.event_key)   # before the weekend: its effect is still unknown
        drivers = [d if pd.notna(d) else f"unknown:{n}" for d, n in zip(example.driver_ids, example.driver_numbers)]
        cars = [c if pd.notna(c) else f"unknown:{n}" for c, n in zip(example.lineage_ids, example.driver_numbers)]
        design = state.design(cars, drivers, "Q")
        mean = design @ state.mean
        cov = design @ state.cov @ design.T + self.hyper.noise["Q"] ** 2 * np.eye(len(cars))
        seed = zlib.crc32(f"{self.name}|{example.event_key}|{example.session_type}|{example.cutoff_label}".encode())
        rng = np.random.default_rng(seed)
        samples = rng.multivariate_normal(mean, cov + 1e-9 * np.eye(len(cars)), size=draws, method="cholesky")
        order = np.argsort(samples, axis=1)
        positions = np.empty_like(order)
        positions[np.arange(draws)[:, None], order] = np.arange(1, len(cars) + 1)
        return Forecast.from_draws(
            positions=positions, season=example.season, session_type=example.session_type,
            event_key=example.event_key, cutoff_label=example.cutoff_label, cutoff_time=example.cutoff_time.isoformat(),
            driver_numbers=list(example.driver_numbers), driver_ids=list(example.driver_ids), model=self.name,
            seed=seed, metadata={"hyper": {"noise": self.hyper.noise, **{k: getattr(self.hyper, k)
                                                                         for k in Hyper.SCALARS},
                                           "fit": self.fit_log.get(example.season)}})
