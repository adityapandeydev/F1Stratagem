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

The filter itself is `program.Recorder` (bookkeeping, in Python) replayed by the
native module `predictor_native` (the arithmetic, in Rust).

Weekend term (v2w, v3w): each car also has a weekend effect shared by every
session of one event (practice, sprint qualifying and qualifying alike) and reset
to its prior at the next event. Track-specific swings, setup and conditions go
there instead of moving the car's long-run pace. Before the weekend the effect is
unknown (prior spread, shared by teammates); after practice it carries what that
weekend's practice showed into the same weekend's qualifying.

Car x circuit affinity (v3wa): each car lineage also has a persistent effect
per circuit, shared by every session it runs there and kept from one visit to
the next. Its uncertainty grows with time and at every season start, multiplied
at regulation changes, but its value is kept: last year at a track weighs more
than the year before, and an old regulation era still counts for something.
Only the current circuit's affinities are active in the joint state; the others
are parked with their mean and variance (their cross-correlations are dropped)
and resume, with the variance accrued while parked, at the next visit.

Frozen shared settings (v2s): car and driver drift, priors and qualifying noise
are fitted on qualifying alone, then frozen; only practice noise and offsets are
fitted on the full evidence.
"""

from dataclasses import dataclass, field, replace
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .forecast import Forecast
from .program import Recorder
from .rules import REGULATION_CHANGES

SEGMENTS = ("Q1", "Q2", "Q3")
SEGMENTED = frozenset({"Q", "SQ"})
SLOW_CUTOFF = 1.07
DRAWS = 10000
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
    affinity_prior: float = 0.0  # spread of a car x circuit affinity; 0 disables it
    affinity_drift: float = 0.0  # affinity variance added per 14 days
    affinity_winter: float = 0.0  # affinity variance added at each season start

    SCALARS = ("car_drift", "driver_drift", "offset_drift", "car_prior", "driver_prior", "offset_prior",
               "car_winter", "driver_winter", "regulation_multiplier", "weekend_prior",
               "affinity_prior", "affinity_drift", "affinity_winter")
    SHARED = ("car_drift", "driver_drift", "car_prior", "driver_prior")

    @staticmethod
    def fitted(types, winter=False, regulation=False, weekend=False, affinity=False):
        """Scalars that affect the likelihood. Offsets exist only for non-qualifying
        sessions; winter terms only if the data crosses a season start; the regulation
        multiplier only if it crosses a season that began under new regulations."""
        offsets = any(t != "Q" for t in types)
        return ("car_drift", "driver_drift", *(("offset_drift",) if offsets else ()), "car_prior", "driver_prior",
                *(("offset_prior",) if offsets else ()), *(("car_winter", "driver_winter") if winter else ()),
                *(("regulation_multiplier",) if winter and regulation else ()),
                *(("weekend_prior",) if weekend else ()),
                *(("affinity_prior", "affinity_drift", *(("affinity_winter",) if winter else ())) if affinity else ()))

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


NESTED_AFFINITY = {"affinity_prior": 1e-3, "affinity_drift": 1e-8, "affinity_winter": 1e-8}
LOG_BOUNDS = (-20.0, 4.0)   # settings are fitted as logs: about 2e-9 to 55
FIT_TOLERANCE = 0.01        # stop once a full round improves the log-likelihood by less than this
FIT_ROUNDS = 10


def _optimise(negative, x0):
    """Minimise `negative` thoroughly; fit quality matters more than run time.

    Each round runs Powell, then L-BFGS-B (finite differences) and adaptive
    Nelder-Mead from the best point so far; rounds repeat until one improves the
    objective by less than FIT_TOLERANCE. Methods differ in where they stall, so
    alternating them escapes points that a single method takes for an optimum.
    Returns (best point, best value, converged)."""
    bounds = [LOG_BOUNDS] * len(x0)
    x = np.clip(np.asarray(x0, dtype=float), *LOG_BOUNDS)
    value = negative(x)
    for _ in range(FIT_ROUNDS):
        round_start = value
        for method, options in (("Powell", {"xtol": 1e-4, "ftol": 1e-8, "maxfev": 20000}),
                                ("L-BFGS-B", {"eps": 1e-5, "ftol": 1e-12, "gtol": 1e-5, "maxfun": 20000}),
                                ("Nelder-Mead", {"xatol": 1e-4, "fatol": 1e-4, "maxfev": 20000,
                                                 "adaptive": True})):
            result = minimize(negative, x, method=method, bounds=bounds, options=options)
            if np.isfinite(result.fun) and result.fun < value:
                x, value = np.clip(result.x, *LOG_BOUNDS), float(result.fun)
        if round_start - value < FIT_TOLERANCE:
            return x, value, True
    return x, value, False


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


class PaceModel:
    """Model interface: fit(history, view_for) then predict(view, example). Qualifying only.

    `session_types` selects the evidence: ("Q",) is stage 1, ALL_SESSIONS stage 2."""

    def __init__(self, session_types=QUALIFYING_ONLY, name="pace:v1", winter=False, objective="all",
                 weekend=False, shared_from_qualifying=False, affinity=False):
        self.session_types = tuple(session_types)
        self.name = name
        self.winter = winter
        self.objective = objective
        self.weekend = weekend
        self.shared_from_qualifying = shared_from_qualifying
        self.affinity = affinity
        self._circuits = None
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

    def circuit(self, view, event):
        """Circuit ID of an event (reference data); None for testing or unknown."""
        if self._circuits is None:
            events = view.archive.events
            self._circuits = {k: (None if pd.isna(c) else c) for k, c in zip(events.event_key, events.circuit_id)} \
                if "circuit_id" in events else {}
        return self._circuits.get(event)

    def record(self, view, sessions, weekend, affinity, objective=None):
        """The filter run over `sessions`, recorded (see `program.Recorder`). With the
        qualifying objective, track 1 is the filter as it stood before each weekend and
        sees only that weekend's qualifying; only qualifying updates count."""
        objective = objective or self.objective
        recorder = Recorder(self.session_types, weekend, affinity)
        event = None
        for session in sessions.itertuples():
            recorder.advance(session.scheduled_start_utc)
            if session.event_key != event:
                event = session.event_key
                recorder.start_event(event, self.circuit(view, event))
                if objective == "qualifying":
                    recorder.snapshot()
            groups = self.observations(view, session)
            if objective == "qualifying" and session.session_type == "Q":
                recorder.advance(session.scheduled_start_utc, track=1)
                for cars, drivers, values in groups:
                    recorder.update(cars, drivers, values, "Q", track=1)
                    recorder.update(cars, drivers, values, "Q")
                continue
            for cars, drivers, values in groups:
                recorder.update(cars, drivers, values, session.session_type, counted=objective == "all")
        return recorder

    def run(self, view, hyper, sessions):
        """Filter every session. `state.objective` is the quantity the settings maximise."""
        return self.record(view, sessions, hyper.weekend_prior > 0, hyper.affinity_prior > 0).replay(hyper)

    def objective_function(self, view, sessions, like):
        """hyper -> objective over `sessions`, for settings with the same weekend and
        affinity switches as `like`: recorded once, replayed for every candidate."""
        return self.record(view, sessions, like.weekend_prior > 0, like.affinity_prior > 0).objective

    def fit_hyper(self, view, sessions, start, frozen=()):
        """Fit the settings that affect the objective, except those named in `frozen`."""
        types = [t for t in self.session_types if t in set(sessions.session_type)]
        for session in sessions.itertuples():
            self.observations(view, session)
        seasons = sorted(set(sessions.scheduled_start_utc.dt.year))
        winter = self.winter and len(seasons) > 1
        regulation = winter and any(s in REGULATION_CHANGES for s in seasons[1:])
        # affinity needs repeat visits: with a single season of data every circuit was seen once
        affinity = self.affinity and len(seasons) > 1
        scalars = [s for s in Hyper.fitted(types, winter, regulation, self.weekend, affinity)
                   if s not in frozen]
        noise_types = [t for t in types if f"noise:{t}" not in frozen]
        default = self._default_start()
        start = replace(start, **({"affinity_prior": 0.0} if self.affinity and not affinity else {}),
                        **{s: getattr(default, s) for s in scalars if getattr(start, s) <= 0})

        evaluations = [0]
        objective = self.objective_function(view, sessions, start)

        def negative(values):
            evaluations[0] += 1
            return -objective(Hyper.from_vector(values, noise_types, scalars, start))

        starts = {"given": start}
        if start != default:
            starts["default"] = replace(default, **({"affinity_prior": 0.0} if self.affinity and not affinity else {}))
        if self.affinity and any(s.startswith("affinity") for s in scalars):
            # nested start: this model without its affinity (fitted first), the affinity barely on.
            # The full fit begins there, so it can never end below the model it extends.
            extra = [s for s in scalars if s.startswith("affinity")]
            base = [s for s in scalars if s not in extra]
            assert scalars == base + extra
            off = np.log([NESTED_AFFINITY[s] for s in extra])
            x, _, _ = _optimise(lambda v: negative(np.concatenate([v, off])), start.vector(noise_types, base))
            starts["nested"] = Hyper.from_vector(np.concatenate([x, off]), noise_types, scalars, start)
        best, results = None, {}
        for label, initial in starts.items():
            x, value, converged = _optimise(negative, initial.vector(noise_types, scalars))
            results[label] = -float(value)
            if best is None or value < best[1]:
                best = (x, value, converged, label)
        self.last_fit = {"converged": bool(best[2]), "evaluations": evaluations[0], "objective": -float(best[1]),
                         "best_start": best[3], "starts": results, "fitted": noise_types + scalars}
        return Hyper.from_vector(best[0], noise_types, scalars, start)

    def _default_start(self):
        return Hyper(**({"car_winter": 0.5, "driver_winter": 0.05, "regulation_multiplier": 3.0}
                        if self.winter else {}), **({"weekend_prior": 0.3} if self.weekend else {}),
                     **({"affinity_prior": 0.2, "affinity_drift": 0.001, "affinity_winter": 0.02}
                        if self.affinity else {}))

    def fit(self, history, view_for):
        return self

    def _ensure_hyper(self, view, season):
        if self.fitted_season == season:
            return
        earlier = view.archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        sessions = self.sessions(earlier)
        start = self.hyper or self._default_start()
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

    def distribution(self, view, example):
        """Mean and covariance of the entrants' qualifying pace at the example's cutoff."""
        recorder = self.record(view, self.sessions(view), self.hyper.weekend_prior > 0,
                               self.hyper.affinity_prior > 0, objective="all")
        recorder.advance(example.cutoff_time)
        if recorder.event != example.event_key:
            # before the weekend: its weekend effect is unknown; this circuit's affinity resumes
            recorder.start_event(example.event_key, self.circuit(view, example.event_key))
        drivers = [d if pd.notna(d) else f"unknown:{n}" for d, n in zip(example.driver_ids, example.driver_numbers)]
        cars = [c if pd.notna(c) else f"unknown:{n}" for c, n in zip(example.lineage_ids, example.driver_numbers)]
        rows = recorder.columns(cars, drivers, "Q")
        state = recorder.replay(self.hyper)
        design = state.design(rows)
        mean = design @ state.mean
        cov = design @ state.cov @ design.T + self.hyper.noise["Q"] ** 2 * np.eye(len(cars))
        return mean, cov

    def predict(self, view, example, draws=DRAWS):
        if example.session_type != "Q":
            return None
        self._ensure_hyper(view, example.season)
        mean, cov = self.distribution(view, example)
        size = len(mean)
        seed = zlib.crc32(f"{self.name}|{example.event_key}|{example.session_type}|{example.cutoff_label}".encode())
        rng = np.random.default_rng(seed)
        samples = rng.multivariate_normal(mean, cov + 1e-9 * np.eye(size), size=draws, method="cholesky")
        order = np.argsort(samples, axis=1)
        positions = np.empty_like(order)
        positions[np.arange(draws)[:, None], order] = np.arange(1, size + 1)
        return Forecast.from_draws(
            positions=positions, season=example.season, session_type=example.session_type,
            event_key=example.event_key, cutoff_label=example.cutoff_label, cutoff_time=example.cutoff_time.isoformat(),
            driver_numbers=list(example.driver_numbers), driver_ids=list(example.driver_ids), model=self.name,
            seed=seed, metadata={"hyper": {"noise": self.hyper.noise, **{k: getattr(self.hyper, k)
                                                                         for k in Hyper.SCALARS},
                                           "fit": self.fit_log.get(example.season)}})
