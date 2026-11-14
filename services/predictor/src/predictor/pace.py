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

Qualifying history (v3wh, v3wah): qualifying before the FastF1 archive, from
Jolpica (best times per segment, no laps), filtered from 2006 like every other
session, on the low-fuel segments only (`rules.low_fuel_segments`). It gives the
car x circuit affinity repeat visits to learn from and drivers a longer record.
`history` chooses how it enters: "shared" reads it as qualifying (v3wh);
"noise" as its own session type H, with its own noise, no offset and no weekend
effect (its only session would make the weekend effect absorb noise) (v3whn);
"state" as qualifying that updates the state but not the objective, so the
settings are fitted on the FastF1 era alone (v3whs); "weighted" as state, read as
type H whose noise is fitted on the FastF1 era alone (v3whw): how far history is
trusted is learned from how well it predicts 2018 on, so it can act as an
educated first guess that current-season evidence overrides (no offset or
weekend effect for H, as for "noise").

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
from . import track
from .laps import race_pace
from .program import REFERENCE_TYPES, Recorder
from .rules import REGULATION_CHANGES, low_fuel_segments

SEGMENTS = ("Q1", "Q2", "Q3")
SEGMENTED = frozenset({"Q", "SQ"})
SLOW_CUTOFF = 1.07
DRAWS = 10000
QUALIFYING_ONLY = ("Q",)
ALL_SESSIONS = ("Q", "SQ", "FP1", "FP2", "FP3", "T")
WITH_RACES = ALL_SESSIONS + ("R",)   # races as evidence: corrected race pace (laps.race_pace)
HISTORY_FIRST_SEASON = 2006      # first season with low-fuel qualifying segments
HISTORY_PREFIX = "history:"
MIN_QUALIFYING_SESSIONS = 5   # fewer before a season: the settings keep their starting values
HISTORY_MODES = ("shared", "noise", "state", "weighted")
UNCOUNTED_HISTORY = ("state", "weighted")   # history updates the state, not the objective
OWN_TYPE_HISTORY = ("noise", "weighted")    # history read as session type H


@dataclass
class Hyper:
    noise: dict = field(default_factory=lambda: {"Q": 0.3, "SQ": 0.35, "FP1": 0.8, "FP2": 0.7, "FP3": 0.6,
                                                 "T": 1.2, "H": 0.3, "R": 0.5})
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
    form_prior: float = 0.0      # spread of a driver's season form, reset each season; 0 disables it
    skill_prior: float = 0.0     # spread of a driver's ability in a part of the track (per unit of its share)
    skill_drift: float = 0.0     # its variance added per 14 days
    season_skill_prior: float = 0.0   # spread of a driver's season ability in a part of the track, reset each season

    SCALARS = ("car_drift", "driver_drift", "offset_drift", "car_prior", "driver_prior", "offset_prior",
               "car_winter", "driver_winter", "regulation_multiplier", "weekend_prior",
               "affinity_prior", "affinity_drift", "affinity_winter", "form_prior", "skill_prior", "skill_drift",
               "season_skill_prior")
    SHARED = ("car_drift", "driver_drift", "car_prior", "driver_prior")

    @staticmethod
    def fitted(types, winter=False, regulation=False, weekend=False, affinity=False, new_cars=True, new_drivers=True,
               form=False, skills=False, season_skills=False):
        """Scalars that affect the likelihood. Offsets exist only for non-qualifying
        sessions; winter terms only if the data crosses a season start; the regulation
        multiplier only if it crosses a season that began under new regulations; the
        car and driver priors only if a lineage or a driver first appears in the data."""
        offsets = any(t not in REFERENCE_TYPES for t in types)
        return ("car_drift", "driver_drift", *(("offset_drift",) if offsets else ()),
                *(("car_prior",) if new_cars else ()), *(("driver_prior",) if new_drivers else ()),
                *(("offset_prior",) if offsets else ()), *(("car_winter", "driver_winter") if winter else ()),
                *(("regulation_multiplier",) if winter and regulation else ()),
                *(("weekend_prior",) if weekend else ()),
                *(("affinity_prior", "affinity_drift", *(("affinity_winter",) if winter else ())) if affinity else ()),
                *(("form_prior",) if form else ()),
                *(("skill_prior", "skill_drift") if skills else ()), *(("season_skill_prior",) if season_skills else ()))

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


NESTED_AFFINITY = {"affinity_prior": 1e-3, "affinity_drift": 1e-8, "affinity_winter": 1e-8,
                   "skill_prior": 1e-3, "skill_drift": 1e-8, "season_skill_prior": 1e-3}
NESTED_TERMS = ("affinity", "skill", "season_skill")   # terms a model adds to the one it extends
LOG_BOUNDS = (-20.0, 4.0)   # settings are fitted as logs: about 2e-9 to 55
# Historical qualifying cannot be more precise than this (modern qualifying noise is
# about 0.45): below it the 2019 fit had a second optimum treating history as exact.
HISTORY_NOISE_FLOOR = 0.05
FIT_TOLERANCE = 0.01        # stop once a full round improves the log-likelihood by less than this
FIT_ROUNDS = 10


def _optimise(negative, x0, lower=None):
    """Minimise `negative` thoroughly; fit quality matters more than run time.

    Each round runs Powell, then L-BFGS-B (finite differences) and adaptive
    Nelder-Mead from the best point so far; rounds repeat until one improves the
    objective by less than FIT_TOLERANCE. Methods differ in where they stall, so
    alternating them escapes points that a single method takes for an optimum.
    `negative` returns +inf for candidates it rejects; only finite results are
    adopted. `lower` optionally raises the lower log bound per setting. Returns
    (best point, best value, converged)."""
    lower = np.full(len(x0), LOG_BOUNDS[0]) if lower is None else np.asarray(lower, dtype=float)
    upper = np.full(len(x0), LOG_BOUNDS[1])
    bounds = list(zip(lower, upper))
    x = np.clip(np.asarray(x0, dtype=float), lower, upper)
    value = negative(x)
    for _ in range(FIT_ROUNDS):
        round_start = value
        for method, options in (("Powell", {"xtol": 1e-4, "ftol": 1e-8, "maxfev": 20000}),
                                ("L-BFGS-B", {"eps": 1e-5, "ftol": 1e-12, "gtol": 1e-5, "maxfun": 20000}),
                                ("Nelder-Mead", {"xatol": 1e-4, "fatol": 1e-4, "maxfev": 20000,
                                                 "adaptive": True})):
            result = minimize(negative, x, method=method, bounds=bounds, options=options)
            if np.isfinite(result.fun) and np.all(np.isfinite(result.x)) and result.fun < value:
                x, value = np.clip(result.x, lower, upper), float(result.fun)
        if not np.isfinite(value):
            return x, value, False     # nothing finite found from this start
        if np.isfinite(round_start) and round_start - value < FIT_TOLERANCE:
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


def _group_observation(cars, drivers, seconds):
    """(car keys, driver keys, log times x 100) of one group, without laps beyond 107%
    of its best and drivers without IDs; None if fewer than two remain."""
    seconds = np.asarray(seconds, dtype=float)
    fast = seconds <= np.nanmin(seconds) * SLOW_CUTOFF
    keep = [bool(f) and pd.notna(c) and pd.notna(d) for f, c, d in zip(fast, cars, drivers)]
    if sum(keep) < 2:
        return None
    return ([c for c, k in zip(cars, keep) if k], [d for d, k in zip(drivers, keep) if k],
            100 * np.log(seconds[keep]))


def session_observations(session_type, results, laps, entrants):
    """[(car keys, driver keys, log times x 100)] for one session."""
    ids = entrants.set_index("driver_number")
    observations = []
    for frame in _lap_groups(session_type, results, laps):
        cars = [ids.lineage_id.get(n) if "lineage_id" in ids else None for n in frame.number]
        drivers = [ids.driver_id.get(n) for n in frame.number]
        group = _group_observation(cars, drivers, frame.seconds.to_numpy())
        if group is not None:
            observations.append(group)
    return observations


def race_observations(laps, entrants):
    """One group from a race: each driver's corrected race pace (`laps.race_pace`:
    tyres, tyre age and fuel taken out lap by lap); none for a wet race."""
    pace = race_pace(laps)
    if pace.empty:
        return []
    ids = entrants.set_index("driver_number")
    cars = [ids.lineage_id.get(n) if "lineage_id" in ids else None for n in pace.driver_number]
    drivers = [ids.driver_id.get(n) for n in pace.driver_number]
    keep = [pd.notna(c) and pd.notna(d) for c, d in zip(cars, drivers)]
    if sum(keep) < 2:
        return []
    return [([c for c, k in zip(cars, keep) if k], [d for d, k in zip(drivers, keep) if k],
             pace.pace.to_numpy()[keep])]


def history_observations(rows):
    """Observations of one historical qualifying session: one group per low-fuel segment."""
    observations = []
    for segment in low_fuel_segments(int(rows.season.iloc[0])):
        times = rows[rows[f"{segment.lower()}_seconds"].notna()]
        if len(times) >= 2:
            group = _group_observation(list(times.lineage_id), list(times.driver_id),
                                       times[f"{segment.lower()}_seconds"].to_numpy())
            if group is not None:
                observations.append(group)
    return observations


class PaceModel:
    """Model interface: fit(history, view_for) then predict(view, example). Qualifying only.

    `session_types` selects the evidence: ("Q",) is stage 1, ALL_SESSIONS stage 2."""

    def __init__(self, session_types=QUALIFYING_ONLY, name="pace:v1", winter=False, objective="all",
                 weekend=False, shared_from_qualifying=False, affinity=False, history=False, form=False,
                 skills=(), season_skills=(), driver_offsets=()):
        history = "shared" if history is True else history
        assert history in (False, *HISTORY_MODES), history
        self.session_types = tuple(session_types) + (("H",) if history in OWN_TYPE_HISTORY else ())
        self.name = name
        self.winter = winter
        self.objective = objective
        self.weekend = weekend
        self.shared_from_qualifying = shared_from_qualifying
        self.affinity = affinity
        self.form = form
        self.skills, self.season_skills = tuple(skills), tuple(season_skills)
        self.driver_offsets = tuple(driver_offsets)
        self._profiles, self._shares = {}, {}
        self._layout_index = self._event_start = self._layout_sessions = None
        self.history = history
        self._circuits = None
        self.hyper = None
        self.fit_cache = None        # a fitcache.FitCache: reuse fits across cutoffs and runs
        self.screen_base = None      # a PaceModel: screen against it (see `_fit_season`)
        self.fitted_season = None
        self.fit_log = {}
        self._qualifying_hyper = None
        self._observations = {}

    def sessions(self, view, before=None):
        sessions = view.sessions[view.sessions.session_type.isin(self.session_types)]
        if self.history:
            sessions = pd.concat([self._history_sessions(view), sessions], ignore_index=True)
        if before is not None:
            sessions = sessions[sessions.available_at_utc < before]
        return sessions.sort_values("scheduled_start_utc", kind="stable")

    def _history_sessions(self, view):
        """One pseudo-session per historical qualifying with low-fuel segments."""
        history = view.qualifying_history()
        history = history[history.season >= HISTORY_FIRST_SEASON]
        events = history.groupby("event_key").agg(season=("season", "first"), scheduled_start_utc=("start_utc", "first"),
                                                  available_at_utc=("available_at_utc", "first")).reset_index()
        return events.assign(session_id=HISTORY_PREFIX + events.event_key,
                             session_type="H" if self.history in OWN_TYPE_HISTORY else "Q")

    def observations(self, view, session):
        """Cached per session; the view still decides whether the session may be read."""
        if session.session_id.startswith(HISTORY_PREFIX):
            if session.session_id not in self._observations:
                history = view.qualifying_history()
                self._observations[session.session_id] = history_observations(
                    history[history.event_key == session.event_key])
            return self._observations[session.session_id]
        if session.session_type == "R":
            if session.session_id not in self._observations:
                self._observations[session.session_id] = race_observations(
                    view.laps(session.session_id), view.entrants(session.session_id))
            return self._observations[session.session_id]
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
            history = view.archive.qualifying_history
            if self.history and history is not None:
                self._circuits.update(zip(history.event_key, history.circuit_id))
        return self._circuits.get(event)

    def shares(self, view, event_key):
        """How an event's track differs from a typical one: its layout's profile from
        qualifying sessions of earlier visits (sessions available before the event's
        first session), minus a fixed reference profile (track.REFERENCE_SHARES).
        The same for every session of the event and for its forecasts; zeros (neutral)
        on a first visit or without skills. Centred shares sum to zero over the bands,
        so equal skills in every band cancel: a season-wide level stays with the driver
        terms."""
        if not (self.skills or self.season_skills):
            return None
        if event_key not in self._shares:
            self._shares[event_key] = track.centred_shares(self._layout_profile(view, event_key))
        return self._shares[event_key]

    def _layout_profile(self, view, event_key):
        archive = view.archive
        if self._layout_index is None:
            table = track.layouts()
            events = archive.events.set_index("event_key")
            self._layout_index = {key: track.layout_of(self.circuit(view, key), int(events.season[key]), key, table)
                                  for key in events.index}
            self._event_start = archive.sessions.groupby("event_key").scheduled_start_utc.min().to_dict()
            qualifying = archive.sessions[archive.sessions.session_type == "Q"].sort_values("available_at_utc")
            self._layout_sessions = {}
            for q in qualifying.itertuples():
                self._layout_sessions.setdefault(self._layout_index.get(q.event_key), []).append(q)
        layout = self._layout_index.get(event_key)
        start = self._event_start.get(event_key)
        if layout is None or start is None:
            return None
        profiles = []
        for q in self._layout_sessions.get(layout, []):
            if q.available_at_utc >= start or q.available_at_utc > view.cutoff:
                break
            if q.session_id not in self._profiles:
                self._profiles[q.session_id] = track.session_profile(archive.view(q.available_at_utc), q.session_id)
            profiles.append(self._profiles[q.session_id])
        return track.layout_profile(profiles)

    def record(self, view, sessions, weekend, affinity, objective=None, form=False, skills=False,
               season_skills=False):
        """The filter run over `sessions`, recorded (see `program.Recorder`). With the
        qualifying objective, track 1 is the filter as it stood before each weekend and
        sees only that weekend's qualifying; only qualifying updates count."""
        objective = objective or self.objective
        recorder = Recorder(self.session_types, weekend, affinity, form, self.skills if skills else (),
                            self.season_skills if season_skills else (), self.driver_offsets)
        event = None
        for session in sessions.itertuples():
            recorder.advance(session.scheduled_start_utc)
            if skills or season_skills:
                recorder.shares = None if session.session_id.startswith(HISTORY_PREFIX) else \
                    self.shares(view, session.event_key)
            if session.event_key != event:
                event = session.event_key
                recorder.start_event(event, self.circuit(view, event))
                if objective == "qualifying":
                    recorder.snapshot()
            groups = self.observations(view, session)
            if self.history in UNCOUNTED_HISTORY and session.session_id.startswith(HISTORY_PREFIX):
                for cars, drivers, values in groups:
                    recorder.update(cars, drivers, values, session.session_type, counted=False)
                continue
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
        return self.record(view, sessions, hyper.weekend_prior > 0, hyper.affinity_prior > 0,
                           form=hyper.form_prior > 0, skills=hyper.skill_prior > 0,
                           season_skills=hyper.season_skill_prior > 0).replay(hyper)

    def objective_function(self, view, sessions, like):
        """hyper -> objective over `sessions`, for settings with the same weekend and
        affinity switches as `like`: recorded once, replayed for every candidate."""
        return self.record(view, sessions, like.weekend_prior > 0, like.affinity_prior > 0,
                           form=like.form_prior > 0, skills=like.skill_prior > 0,
                           season_skills=like.season_skill_prior > 0).objective

    def fit_hyper(self, view, sessions, start, frozen=()):
        """Fit the settings that affect the objective, except those named in `frozen`."""
        types = [t for t in self.session_types if t in set(sessions.session_type)]
        for session in sessions.itertuples():
            self.observations(view, session)
        # A setting is fitted only if the counted sessions identify it well. Uncounted
        # history (state mode) still shapes the state entering the counted sessions, so
        # terms acting inside it (winter terms at the 2017-2018 crossing, the regulation
        # multiplier at 2009-2017 changes, the car prior at a lineage's first sight in
        # 2006) do move the counted objective, but weakly: fitted, they ran to their
        # bounds (winter terms from the single crossing too). As a regularisation choice,
        # each is held at its starting value until the counted sessions themselves
        # identify it, as for a model without history: winter terms once they cross a
        # season start, the multiplier once they cross a regulation change (so the 2023
        # fit sets it, and with it the history's changes and the affinity winter), the
        # priors once a car or driver first appears in them.
        history = sessions.session_id.str.startswith(HISTORY_PREFIX)
        state = self.history in UNCOUNTED_HISTORY
        counted = sessions[~history] if state else sessions
        seasons = sorted(set(counted.scheduled_start_utc.dt.year))
        crossed = seasons[1:]                          # season starts inside the counted sessions
        if state:
            old_cars, old_drivers = self._entrants(view, sessions[history])
            cars, drivers = self._entrants(view, counted)
            new_cars, new_drivers = bool(cars - old_cars), bool(drivers - old_drivers)
        else:
            new_cars = new_drivers = True
        winter = self.winter and bool(crossed)
        regulation = winter and any(s in REGULATION_CHANGES for s in crossed)
        # affinity needs repeat visits: with a single season of data every circuit was seen once
        affinity = self.affinity and len(seasons) > 1
        # season form needs a season start inside the counted sessions: within one season
        # it cannot be told apart from the driver's level
        form = self.form and winter
        # season skills, like season form, need a season start inside the counted sessions
        # skills are read only at revisited layouts (a first visit is neutral): without a
        # revisit in the counted data nothing depends on them, and fitted they would
        # drift to their bounds (2019 fit, 2018 data: every layout a first visit)
        modern = counted[~counted.session_id.str.startswith(HISTORY_PREFIX)]
        revisited = bool(self.skills or self.season_skills) and any(
            any(abs(v) > 0 for v in self.shares(view, event).values()) for event in modern.event_key.unique())
        skills = bool(self.skills) and revisited
        season_skills = bool(self.season_skills) and winter and revisited
        if self.affinity and state and not affinity:
            # with history in the state, a counted session at a circuit history visited is
            # also a repeat visit, so the affinity can be identified from the first counted season
            visited = {self.circuit(view, e) for e in sessions[history].event_key} - {None}
            affinity = bool(visited & {self.circuit(view, e) for e in counted.event_key})
        scalars = [s for s in Hyper.fitted(types, winter, regulation, self.weekend, affinity, new_cars, new_drivers,
                                           form, skills, season_skills)
                   if s not in frozen]
        noise_types = [t for t in types if f"noise:{t}" not in frozen]
        default = self._default_start()
        off = {**({"affinity_prior": 0.0} if self.affinity and not affinity else {}),
               **({"form_prior": 0.0} if self.form and not form else {}),
               **({"season_skill_prior": 0.0} if self.season_skills and not season_skills else {}),
               **({"skill_prior": 0.0, "skill_drift": 0.0} if self.skills and not skills else {})}
        start = replace(start, **off,
                        **{s: getattr(default, s) for s in scalars if getattr(start, s) <= 0})

        lower = [np.log(HISTORY_NOISE_FLOOR) if t == "H" else LOG_BOUNDS[0] for t in noise_types] \
            + [LOG_BOUNDS[0]] * len(scalars)
        evaluations, rejected = [0], [0]
        objective = self.objective_function(view, sessions, start)
        if not noise_types and not scalars:   # everything frozen (only in a screen): nothing to fit
            self.last_fit = {"converged": True, "evaluations": 0, "rejected": 0, "objective": float(objective(start)),
                             "best_start": "given", "starts": {}, "fitted": []}
            return start

        def negative(values):
            """Minus the objective; +inf rejects a candidate whose filter fails numerically
            (e.g. noise so small that an innovation covariance is no longer positive definite)."""
            evaluations[0] += 1
            try:
                value = -objective(Hyper.from_vector(values, noise_types, scalars, start))
            except ArithmeticError:
                value = np.inf
            if not np.isfinite(value):
                rejected[0] += 1
                return np.inf
            return value

        starts = {"given": start}
        if start != default:
            starts["default"] = replace(default, **off)
        if any(s.startswith(NESTED_TERMS) for s in scalars):
            # nested start: this model without its affinity (fitted first), the affinity barely on.
            # The full fit begins there, so it can never end below the model it extends.
            extra = [s for s in scalars if s.startswith(NESTED_TERMS)]
            base = [s for s in scalars if s not in extra]
            scalars = base + extra        # the vector order: what the model extends, then its own terms
            off = np.log([NESTED_AFFINITY[s] for s in extra])
            inner = start.vector(noise_types, base)
            x = _optimise(lambda v: negative(np.concatenate([v, off])), inner,
                          lower[:len(noise_types) + len(base)])[0] if len(inner) else inner
            starts["nested"] = Hyper.from_vector(np.concatenate([x, off]), noise_types, scalars, start)
        best, results = None, {}
        for label, initial in starts.items():
            x, value, converged = _optimise(negative, initial.vector(noise_types, scalars), lower)
            results[label] = -float(value)
            if best is None or value < best[1]:
                best = (x, value, converged, label)
        if not np.isfinite(best[1]):
            raise ArithmeticError("no start reached a finite log-likelihood")
        self.last_fit = {"converged": bool(best[2]), "evaluations": evaluations[0], "rejected": rejected[0],
                         "objective": -float(best[1]), "best_start": best[3], "starts": results,
                         "fitted": noise_types + scalars}
        return Hyper.from_vector(best[0], noise_types, scalars, start)

    def _entrants(self, view, sessions):
        """(car keys, driver keys) observed in `sessions`."""
        groups = [group for session in sessions.itertuples() for group in self.observations(view, session)]
        return {c for cars, _, _ in groups for c in cars}, {d for _, drivers, _ in groups for d in drivers}

    def _screen_season(self, view, season, earlier, sessions, start):
        """Quick screen: every setting the base model fitted (or holds) is taken from
        its fit for this season and frozen; only the settings this model adds are
        fitted. Minutes instead of hours; a screen only decides whether a full gate
        run is worth it, never whether a feature is adopted."""
        base = self.screen_base
        self._check_extends(base)
        base._ensure_hyper(view, season)
        # every setting the base model has, fitted or held, by its switches
        owned = Hyper.fitted(base.session_types, winter=base.winter, regulation=base.winter, weekend=base.weekend,
                             affinity=base.affinity, form=base.form, skills=bool(base.skills),
                             season_skills=bool(base.season_skills))
        noise = {**start.noise, **{t: base.hyper.noise[t] for t in base.session_types}}
        merged = replace(start, noise=noise, **{name: getattr(base.hyper, name) for name in owned})
        if base.fit_log[season].get("fitted") is None:     # the base kept its starting values: too little data
            self.hyper, self.last_fit = merged, dict(base.fit_log[season])
            return
        frozen = [f"noise:{t}" for t in base.session_types] + list(owned)
        self.hyper = self.fit_hyper(earlier, sessions, merged, frozen=frozen)
        self.last_fit = {**self.last_fit, "screened_against": base.name}

    def _check_extends(self, base):
        """A screen compares a model with one it extends: same objective, the base's
        evidence and terms all present. Shared-from-qualifying fits are not screened."""
        problems = []
        if not set(base.session_types) <= set(self.session_types):
            problems.append("session types")
        if base.objective != self.objective:
            problems.append("objective")
        if base.shared_from_qualifying or self.shared_from_qualifying:
            problems.append("shared-from-qualifying fit")
        for flag in ("winter", "weekend", "affinity", "form"):
            if getattr(base, flag) and not getattr(self, flag):
                problems.append(flag)
        for bands in ("skills", "season_skills"):
            if not set(getattr(base, bands)) <= set(getattr(self, bands)):
                problems.append(bands)
        if base.history and base.history != self.history:
            problems.append("history mode")
        if problems:
            raise ValueError(f"{self.name} does not extend {base.name} ({', '.join(problems)}); cannot screen")

    def _default_start(self):
        return Hyper(**({"car_winter": 0.5, "driver_winter": 0.05, "regulation_multiplier": 3.0}
                        if self.winter else {}), **({"weekend_prior": 0.3} if self.weekend else {}),
                     **({"affinity_prior": 0.2, "affinity_drift": 0.001, "affinity_winter": 0.02}
                        if self.affinity else {}), **({"form_prior": 0.1} if self.form else {}),
                     **({"skill_prior": 0.3, "skill_drift": 0.001} if self.skills else {}),
                     **({"season_skill_prior": 0.2} if self.season_skills else {}))

    def fit(self, history, view_for):
        return self

    def options(self):
        """Everything besides the name that changes what this model fits."""
        return {"session_types": list(self.session_types), "winter": self.winter, "objective": self.objective,
                "weekend": self.weekend, "shared_from_qualifying": self.shared_from_qualifying,
                "affinity": self.affinity, "history": self.history, "form": self.form,
                "skills": list(self.skills), "season_skills": list(self.season_skills),
                "driver_offsets": list(self.driver_offsets),
                "screen_base": None if self.screen_base is None else
                [self.screen_base.name, self.screen_base.options()]}

    def _ensure_hyper(self, view, season):
        if self.fitted_season == season:
            return
        start = self.hyper or self._default_start()
        if self.screen_base is not None:
            # the base's own chain of seasons, so a screen resumed from the cache starts it right
            self.screen_base._ensure_hyper(view, season)
        if self.fit_cache is None:
            self._fit_season(view, season, start)
        else:
            from dataclasses import asdict
            key = self.fit_cache.key(self, season, start, self._qualifying_hyper, view.archive.root,
                                     base=None if self.screen_base is None else self.screen_base.hyper)

            def fit():
                self._fit_season(view, season, start)
                return {"hyper": asdict(self.hyper), "fit": self.last_fit,
                        "qualifying_hyper": None if self._qualifying_hyper is None else asdict(self._qualifying_hyper)}

            entry, cached = self.fit_cache.get_or_fit(
                self.fit_cache.path(self, season, key), fit,
                still_valid=lambda: self.fit_cache.unchanged(view.archive.root, season))
            if cached:
                self.hyper, self.last_fit = Hyper(**entry["hyper"]), entry["fit"]
                if entry["qualifying_hyper"] is not None:
                    self._qualifying_hyper = Hyper(**entry["qualifying_hyper"])
        self.fit_log[season] = self.last_fit
        self.fitted_season = season

    def _fit_season(self, view, season, start):
        """Fit the settings on data before `season` (sets hyper and last_fit)."""
        earlier = view.archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        sessions = self.sessions(earlier)
        if self.screen_base is not None:
            return self._screen_season(view, season, earlier, sessions, start)
        self.last_fit = {"converged": None, "note": "too little data; starting values used"}
        counted = sessions[~sessions.session_id.str.startswith(HISTORY_PREFIX)] \
            if self.history in UNCOUNTED_HISTORY else sessions
        if (counted.session_type == "Q").sum() < MIN_QUALIFYING_SESSIONS:
            self.hyper = start
        elif self.shared_from_qualifying:
            qualifying = PaceModel(QUALIFYING_ONLY)
            self._qualifying_hyper = qualifying.fit_hyper(earlier, qualifying.sessions(earlier),
                                                          self._qualifying_hyper or Hyper())
            start = replace(start, noise=dict(start.noise))   # never edit the previous season's settings
            for name in Hyper.SHARED:
                setattr(start, name, getattr(self._qualifying_hyper, name))
            start.noise["Q"] = self._qualifying_hyper.noise["Q"]
            self.hyper = self.fit_hyper(earlier, sessions, start, frozen=(*Hyper.SHARED, "noise:Q"))
        else:
            self.hyper = self.fit_hyper(earlier, sessions, start)

    def distribution(self, view, example):
        """Mean and covariance of the entrants' qualifying pace at the example's cutoff."""
        recorder = self.record(view, self.sessions(view), self.hyper.weekend_prior > 0,
                               self.hyper.affinity_prior > 0, objective="all", form=self.hyper.form_prior > 0,
                               skills=self.hyper.skill_prior > 0, season_skills=self.hyper.season_skill_prior > 0)
        recorder.advance(example.cutoff_time)
        if recorder.event != example.event_key:
            # before the weekend: its weekend effect is unknown; this circuit's affinity resumes
            recorder.start_event(example.event_key, self.circuit(view, example.event_key))
        if recorder.skills or recorder.season_skills:
            recorder.shares = self.shares(view, example.event_key)
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
            seed=seed, metadata={"hyper": {"noise": dict(self.hyper.noise), **{k: getattr(self.hyper, k)
                                                                         for k in Hyper.SCALARS},
                                           "fit": self.fit_log.get(example.season)}})
