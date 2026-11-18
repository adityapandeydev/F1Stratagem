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

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .forecast import Forecast
from . import track
from .baselines import seed_for
from .laps import long_run_pace, push_pace, qualifying_trend, race_pace, segment_wet_shares, wet_race, wet_session
from .program import REFERENCE_TYPES, WET_SUFFIX, Recorder, base_type
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
                                                 "T": 1.2, "H": 0.3, "R": 0.5, "L": 0.5,
                                                 "FP1P": 0.5, "FP2P": 0.5, "FP3P": 0.5,
                                                 "QW": 0.8, "SQW": 0.8, "FP1W": 1.0, "FP2W": 1.0, "FP3W": 1.0,
                                                 "RW": 0.8})
    car_drift: float = 0.05     # variances added per 14 days
    driver_drift: float = 0.01
    offset_drift: float = 0.005
    car_prior: float = 1.0      # prior spreads of new states
    driver_prior: float = 0.3
    offset_prior: float = 0.3
    car_winter: float = 0.0     # variances added at each season start (stage 3)
    driver_winter: float = 0.0
    regulation_multiplier: float = 1.0   # car winter variance multiplier at new regulations
    estimate_scale: float = 0.0  # scales a race's pace estimation covariance into its noise; 0: not used
    robust_dof: float = 0.0      # Student-t degrees of freedom of observation noise; 0: Gaussian.
    # The likelihood is the multivariate t of the whole innovation; the update rescales
    # its noise part (session noise and any estimation covariance) by (nu + q)/(nu + m),
    # which can fall below 1 for a quiet session (a one-step variational approximation).
    weekend_prior: float = 0.0  # spread of a car's weekend effect; 0 disables it
    affinity_prior: float = 0.0  # spread of a car x circuit affinity; 0 disables it
    affinity_drift: float = 0.0  # affinity variance added per 14 days
    affinity_winter: float = 0.0  # affinity variance added at each season start
    form_prior: float = 0.0      # spread of a driver's season form, reset each season; 0 disables it
    skill_prior: float = 0.0     # spread of a driver's ability in a part of the track (per unit of its share)
    skill_drift: float = 0.0     # its variance added per 14 days
    season_skill_prior: float = 0.0   # spread of a driver's season ability in a part of the track, reset each season
    wet_prior: float = 0.0       # spread of a driver's wet ability (wet sessions only); 0 disables it
    wet_drift: float = 0.0       # its variance added per 14 days
    driver_affinity_prior: float = 0.0   # spread of a driver x circuit affinity; 0 disables it
    driver_affinity_drift: float = 0.0   # its variance added per 14 days
    driver_affinity_winter: float = 0.0  # its variance added at each season start

    SCALARS = ("car_drift", "driver_drift", "offset_drift", "car_prior", "driver_prior", "offset_prior",
               "car_winter", "driver_winter", "regulation_multiplier", "weekend_prior",
               "affinity_prior", "affinity_drift", "affinity_winter", "form_prior", "skill_prior", "skill_drift",
               "season_skill_prior", "estimate_scale", "robust_dof", "wet_prior", "wet_drift",
               "driver_affinity_prior", "driver_affinity_drift", "driver_affinity_winter")
    SHARED = ("car_drift", "driver_drift", "car_prior", "driver_prior")

    @staticmethod
    def fitted(types, winter=False, regulation=False, weekend=False, affinity=False, new_cars=True, new_drivers=True,
               form=False, skills=False, season_skills=False, estimates=False, robust=False, wet=False,
               driver_affinity=False):
        """Scalars that affect the likelihood. Offsets exist only for non-qualifying
        sessions; winter terms only if the data crosses a season start; the regulation
        multiplier only if it crosses a season that began under new regulations; the
        car and driver priors only if a lineage or a driver first appears in the data."""
        offsets = any(base_type(t) not in REFERENCE_TYPES for t in types)
        return ("car_drift", "driver_drift", *(("offset_drift",) if offsets else ()),
                *(("car_prior",) if new_cars else ()), *(("driver_prior",) if new_drivers else ()),
                *(("offset_prior",) if offsets else ()), *(("car_winter", "driver_winter") if winter else ()),
                *(("regulation_multiplier",) if winter and regulation else ()),
                *(("weekend_prior",) if weekend else ()),
                *(("affinity_prior", "affinity_drift", *(("affinity_winter",) if winter else ())) if affinity else ()),
                *(("form_prior",) if form else ()),
                *(("skill_prior", "skill_drift") if skills else ()), *(("season_skill_prior",) if season_skills else ()),
                *(("estimate_scale",) if estimates else ()), *(("robust_dof",) if robust else ()),
                *(("wet_prior", "wet_drift") if wet else ()),
                *(("driver_affinity_prior", "driver_affinity_drift", *(("driver_affinity_winter",) if winter else ()))
                  if driver_affinity else ()))

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
                   "skill_prior": 1e-3, "skill_drift": 1e-8, "season_skill_prior": 1e-3,
                   "driver_affinity_prior": 1e-3, "driver_affinity_drift": 1e-8, "driver_affinity_winter": 1e-8}
NESTED_TERMS = ("affinity", "skill", "season_skill", "driver_affinity")   # terms a model adds to the one it extends
LOG_BOUNDS = (-20.0, 4.0)   # settings are fitted as logs: about 2e-9 to 55
# Historical qualifying cannot be more precise than this (modern qualifying noise is
# about 0.45): below it the 2019 fit had a second optimum treating history as exact.
HISTORY_NOISE_FLOOR = 0.05
MIN_ROBUST_DOF = 2.0         # fewer degrees of freedom: forecast lap noise of infinite variance
MIN_WET_SESSIONS = 3         # a wet noise type is fitted once the data has this many of its sessions
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


def _lap_groups(session_type, results, laps, trends=None):
    """Groups of (driver number, seconds) run under shared conditions. `trends`
    (`laps.qualifying_trend`) reads each segment time at the segment's mean track."""
    if session_type in SEGMENTED:
        for segment in SEGMENTS:
            if segment not in results or results[segment].isna().all():
                continue
            frame = pd.DataFrame({"number": results.DriverNumber.astype(str),
                                  "seconds": pd.to_timedelta(results[segment]).dt.total_seconds()}).dropna()
            if trends and segment in trends and frame.number.isin(list(trends[segment])).all():
                shift = frame.number.map(trends[segment]).to_numpy(dtype=float)
                frame["seconds"] = frame.seconds.to_numpy() * np.exp(-shift / 100)
            frame.attrs["segment"] = segment
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


def session_observations(session_type, results, laps, entrants, trends=None, segments=None):
    """[(car keys, driver keys, log times x 100)] for one session. `segments`: a list
    that receives each group's segment ("Q1".., None for an unsegmented session)."""
    ids = entrants.set_index("driver_number")
    observations = []
    for frame in _lap_groups(session_type, results, laps, trends):
        cars = [ids.lineage_id.get(n) if "lineage_id" in ids else None for n in frame.number]
        drivers = [ids.driver_id.get(n) for n in frame.number]
        group = _group_observation(cars, drivers, frame.seconds.to_numpy())
        if group is not None:
            observations.append(group)
            if segments is not None:
                segments.append(frame.attrs.get("segment"))
    return observations


PRACTICE = ("FP1", "FP2", "FP3")
LONG_RUN_PREFIX = "long:"
PUSH_TYPES = ("FP1P", "FP2P", "FP3P")
PUSH_PREFIX = "push:"           # a practice session's push-pace group (practice_push "split")


Q3_CARS = 10


def knockout_cuts(size):
    """(cars out after Q1, cars out after Q2): ten reach Q3, the rest leave in two
    halves, the first the larger (20 cars: 5 and 5; 22: 6 and 6)."""
    out = max(size - Q3_CARS, 0)
    first = -(-out // 2)
    return first, out - first


FAILURE_PRIOR_SEGMENTS = 40.0   # weight of the field's failure rate in a driver's own


def failure_rates(records, drivers):
    """Probability per driver and segment (Q1, Q2, Q3) of setting no time: each
    segment's rate in the field times the driver's own tendency, his failures over
    his segments shrunk towards the field rate with FAILURE_PRIOR_SEGMENTS of weight.
    `records`: (driver, segment, failed) rows from earlier qualifying sessions."""
    if records.empty:
        return np.zeros((len(drivers), 3))
    overall = records.failed.mean()
    segment = records.groupby("segment").failed.mean().reindex(["Q1", "Q2", "Q3"]).fillna(overall).to_numpy()
    own = records.groupby("driver").failed.agg(["sum", "count"])
    rates = []
    for driver in drivers:
        fails, count = (own.loc[driver, "sum"], own.loc[driver, "count"]) if driver in own.index else (0, 0)
        tendency = (fails + FAILURE_PRIOR_SEGMENTS * overall) / (count + FAILURE_PRIOR_SEGMENTS) / max(overall, 1e-9)
        rates.append(np.clip(segment * tendency, 0.0, 0.5))
    return np.array(rates)


def qualifying_failures(results):
    """(driver number, segment, failed) of one qualifying: a driver took part in Q1,
    in Q2 if classified in its places, in Q3 if in the top ten, and failed a segment he
    took part in when he set no time in it."""
    position = pd.to_numeric(results.Position, errors="coerce")
    size = len(results)
    first, second = knockout_cuts(size)
    rows = []
    for segment, limit in (("Q1", size), ("Q2", size - first), ("Q3", Q3_CARS)):
        if segment not in results:
            continue
        times = pd.to_timedelta(results[segment])
        for number, place, time in zip(results.DriverNumber.astype(str), position, times):
            if pd.notna(place) and place <= limit:
                rows.append((number, segment, bool(pd.isna(time))))
    return rows


def knockout_positions(pace, noise, rng, failures=None, failure_rng=None, dof=0.0, scale_rng=None,
                       track=0.0, track_rng=None):
    """Positions from Q1, Q2 and Q3: per draw (rows of `pace`) every driver sets a lap
    per segment he takes part in (pace + independent noise); the slowest leave after Q1
    and Q2 and are placed by the lap that eliminated them, Q3 orders the top ten.
    `failures` (drivers x 3): probability of setting no time in each segment; a driver
    who fails is behind everyone who set a time in that segment. Failures draw from
    `failure_rng`, so the lap noise is the same with or without them (models compared
    on shared random draws stay paired). With `dof` > 0 the lap noise of a segment is
    multivariate Student-t as the robust filter fits it: one chi-squared scale per draw
    and segment shared by every driver, times independent normal draws. `track` > 0:
    each lap also meets a track condition of that standard deviation (a wet qualifying:
    which lap of the drying track a driver's best falls on), normal and drawn from
    `track_rng` apart from the lap noise."""
    draws, size = pace.shape
    if failures is not None and failure_rng is None:
        failure_rng = np.random.default_rng(0)        # never the lap-noise stream
    if dof > 0 and scale_rng is None:
        scale_rng = np.random.default_rng(1)          # nor the Student-t scales: lap normals stay paired
    if track > 0 and track_rng is None:
        track_rng = np.random.default_rng(2)
    first, second = knockout_cuts(size)
    positions = np.zeros((draws, size), dtype=int)
    alive = np.ones((draws, size), dtype=bool)
    rows = np.arange(draws)[:, None]
    def segment_lap(index):
        draw = rng.standard_normal(pace.shape)
        if dof > 0:
            draw = draw * np.sqrt(dof / scale_rng.chisquare(dof, (pace.shape[0], 1)))
        lap = pace + noise * draw
        if track > 0:
            lap = lap + track * track_rng.standard_normal(pace.shape)
        if failures is not None:
            failed = failure_rng.random(pace.shape) < failures[None, :, index]
            lap = np.where(failed, 1e6 + failure_rng.random(pace.shape), lap)     # behind every time set
        return np.where(alive, lap, np.inf)

    for index, (out, start) in enumerate(((first, size - first + 1), (second, size - first - second + 1))):
        if out == 0:
            continue
        lap = segment_lap(index)
        order = np.argsort(lap, axis=1)
        running = alive.sum(axis=1)[0]
        leaving = order[:, running - out:running]                # the slowest `out` of those still running
        positions[rows, leaving] = start + np.arange(out)[None, :]
        alive[rows, leaving] = False
    lap = segment_lap(2)
    order = np.argsort(lap, axis=1)
    finalists = alive.sum(axis=1)[0]
    positions[rows, order[:, :finalists]] = np.arange(1, finalists + 1)[None, :]
    return positions


def _pace_group(pace, entrants, slow_cutoff=None):
    """One observation group from a per-driver pace table (`laps` module), or [].
    With `slow_cutoff`, drivers slower than that ratio of the group's best are left
    out (practice: a driver who ran only on high fuel)."""
    if pace.empty:
        return []
    if slow_cutoff is not None:
        pace = pace[pace.pace <= pace.pace.min() + 100 * np.log(slow_cutoff)]
    ids = entrants.set_index("driver_number")
    cars = [ids.lineage_id.get(n) if "lineage_id" in ids else None for n in pace.driver_number]
    drivers = [ids.driver_id.get(n) for n in pace.driver_number]
    keep = [pd.notna(c) and pd.notna(d) for c, d in zip(cars, drivers)]
    if sum(keep) < 2:
        return []
    return [([c for c, k in zip(cars, keep) if k], [d for d, k in zip(drivers, keep) if k],
             pace.pace.to_numpy(dtype=float)[keep])]


def race_observations(laps, entrants, with_covariance=False, drying=False):
    """One group from a race: each driver's corrected race pace (`laps.race_pace`:
    tyres, tyre age and fuel taken out lap by lap); none for a wet race unless
    `drying` (then none for a dry one). With `with_covariance`, also [the estimation
    covariance of the group's values]."""
    pace = race_pace(laps, drying)
    groups = _pace_group(pace, entrants)
    if not with_covariance:
        return groups
    if not groups:
        return groups, []
    ids = entrants.set_index("driver_number")
    keep = [pd.notna(ids.lineage_id.get(n) if "lineage_id" in ids else None) and pd.notna(ids.driver_id.get(n))
            for n in pace.driver_number]
    covariance = np.asarray(pace.attrs["covariance"])[np.ix_(keep, keep)]
    return groups, [covariance]


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
                 skills=(), season_skills=(), driver_offsets=(), practice_push=False, long_runs=False,
                 race_estimates=False, knockout=False, failures=False, wet_noise=False, robust=False,
                 wet_forecast=False, wet_ability=False, wet_races=False, wet_evolution=False,
                 wet_segments=False, driver_affinity=False):
        history = "shared" if history is True else history
        assert history in (False, *HISTORY_MODES), history
        self.session_types = tuple(session_types) + (("H",) if history in OWN_TYPE_HISTORY else ())
        self.name = name
        self.winter = winter
        self.objective = objective
        self.weekend = weekend
        self.shared_from_qualifying = shared_from_qualifying
        self.affinity = affinity
        # each driver's own affinity for a circuit, against his teammates (the car's is `affinity`)
        self.driver_affinity = driver_affinity
        self.form = form
        self.skills, self.season_skills = tuple(skills), tuple(season_skills)
        self.driver_offsets = tuple(driver_offsets)
        # practice read as low-fuel push pace (compound-corrected) instead of each best lap,
        # and practice long runs as their own evidence (session type L)
        # practice_push "own": push-pace observations are their own session types (FP1P..),
        # with their own noise and per-team offset, apart from sessions read as best laps
        self.practice_push, self.long_runs = practice_push, long_runs
        # each race's pace estimation covariance enters its observation noise, scaled by a
        # fitted estimate_scale (laps within a stint are correlated, so the raw errors are small)
        self.race_estimates = race_estimates
        # forecasts simulate Q1, Q2 and Q3 (eliminations, a lap per segment) instead of one draw
        self.knockout = knockout
        # drivers fail to set a time in a segment with a learned, driver-specific probability
        self.failures = failures
        self._failure_rows = {}
        # sessions run in wet or mixed conditions are their own types ("QW", "FP2W") with their
        # own noise: a compromised session widens uncertainty instead of moving long-run pace
        self.wet_noise = wet_noise
        self._wet = {}
        # heavy-tailed observation noise: an unusual session moves the estimates less
        self.robust = robust
        # forecasts for a wet qualifying (example.condition "wet") use the wet qualifying
        # noise and, with wet_ability, each driver's wet ability (a state read by wet sessions)
        if (wet_forecast or wet_ability) and not wet_noise:
            raise ValueError("wet forecasts and wet ability need wet sessions as their own types")
        self.wet_forecast, self.wet_ability = wet_forecast or wet_ability, wet_ability
        # wet and mixed races as their own type "RW" (laps.race_pace with drying)
        if wet_races and not (wet_noise and "R" in session_types):
            raise ValueError("wet races need races and wet sessions as their own types")
        self.wet_races = wet_races
        # a wet qualifying read at one track per segment (laps.qualifying_trend); a wet
        # forecast adds the spread of track conditions the drivers' laps met
        if wet_evolution and not wet_noise:
            raise ValueError("track evolution in wet qualifying needs wet sessions as their own types")
        self.wet_evolution = wet_evolution
        self._evolution = {}
        # in a wet qualifying, each segment is typed by the tyres its times were set on:
        # a segment run on slicks is dry evidence with the dry noise ("Q", not "QW")
        if wet_segments and not wet_noise:
            raise ValueError("wet segments need wet sessions as their own types")
        self.wet_segments = wet_segments
        self._group_types = {}
        if wet_races:
            self.session_types = self.session_types + (f"R{WET_SUFFIX}",)
        if wet_noise:
            self.session_types = self.session_types + tuple(
                f"{t}{WET_SUFFIX}" for t in ("Q", "SQ", "FP1", "FP2", "FP3") if t in self.session_types)
        self._covariances = {}
        if long_runs and "L" not in self.session_types:
            self.session_types = self.session_types + ("L",)
        if practice_push in ("own", "split"):
            # "split": every practice session gives a push-pace group (drivers with push laps,
            # types FP1P-FP3P) and a best-lap group of the other drivers (FP1-FP3)
            self.session_types = self.session_types + tuple(f"{t}P" for t in PRACTICE if t in self.session_types)
        self._profiles, self._shares, self._push, self._push_groups = {}, {}, {}, {}
        self._layout_index = self._event_start = self._layout_sessions = None
        self.history = history
        self._circuits = None
        self.hyper = None
        self.fit_cache = None        # a fitcache.FitCache: reuse fits across cutoffs and runs
        self.fit_name = None         # fits stored under another model's name (same fit, other forecasts)
        self.screen_base = None      # a PaceModel: screen against it (see `_fit_season`)
        self.fitted_season = None
        self.fit_log = {}
        self._qualifying_hyper = None
        self._observations = {}

    def sessions(self, view, before=None):
        sessions = view.sessions[view.sessions.session_type.isin(self.session_types)]
        if self.history:
            sessions = pd.concat([self._history_sessions(view), sessions], ignore_index=True)
        push = None
        if self.practice_push == "split":
            practice = sessions[sessions.session_type.isin(PRACTICE)]
            push = practice.assign(session_id=PUSH_PREFIX + practice.session_id,
                                   session_type=practice.session_type + "P")
        if self.wet_noise:
            sessions = sessions.copy()
            dry = sessions.session_type.isin(("Q", "SQ", "FP1", "FP2", "FP3"))
            sessions.loc[dry, "session_type"] = [
                f"{row.session_type}{WET_SUFFIX}" if self._wet_session(view, row.session_id) else row.session_type
                for row in sessions[dry].itertuples()]
            if self.wet_races:
                races = sessions.session_type == "R"
                sessions.loc[races, "session_type"] = [
                    f"R{WET_SUFFIX}" if self._wet_race(view, row.session_id) else "R"
                    for row in sessions[races].itertuples()]
        long = None
        if self.long_runs:                        # from the practice sessions as they are
            practice = sessions[sessions.session_type.isin(PRACTICE)]
            long = practice.assign(session_id=LONG_RUN_PREFIX + practice.session_id, session_type="L")
        if self.practice_push == "own":
            sessions = sessions.copy()
            practice = sessions.session_type.isin(PRACTICE)
            sessions.loc[practice, "session_type"] = [
                f"{row.session_type}P" if self._push_covers(view, row) else row.session_type
                for row in sessions[practice].itertuples()]
        if long is not None:
            sessions = pd.concat([sessions, long], ignore_index=True)
        if push is not None:
            sessions = pd.concat([sessions, push], ignore_index=True)
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
        if session.session_type == "L":
            if session.session_id not in self._observations:
                practice = session.session_id[len(LONG_RUN_PREFIX):]
                self._observations[session.session_id] = _pace_group(
                    long_run_pace(view.laps(practice)), view.entrants(practice), SLOW_CUTOFF)
            return self._observations[session.session_id]
        if session.session_id.startswith(PUSH_PREFIX):
            if session.session_id not in self._observations:
                practice = session.session_id[len(PUSH_PREFIX):]
                self._observations[session.session_id] = self._push_group(view, practice)
            return self._observations[session.session_id]
        if self.practice_push == "split" and base_type(session.session_type) in PRACTICE:
            if session.session_id not in self._observations:
                pushed = {d for group in self._push_group(view, session.session_id) for d in group[1]}
                raw = session_observations(base_type(session.session_type), view.results(session.session_id),
                                           view.laps(session.session_id), view.entrants(session.session_id))
                rest = []
                for cars, drivers, values in raw:
                    keep = [d not in pushed for d in drivers]
                    if sum(keep) >= 2:
                        rest.append(([c for c, k in zip(cars, keep) if k], [d for d, k in zip(drivers, keep) if k],
                                     np.asarray(values)[keep]))
                self._observations[session.session_id] = rest
            return self._observations[session.session_id]
        if session.session_type in PUSH_TYPES:
            self._push_covers(view, session)
            return self._observations[session.session_id]
        if self.practice_push is True and session.session_type in PRACTICE:
            if session.session_id not in self._observations:
                laps = view.laps(session.session_id)
                entrants = view.entrants(session.session_id)
                raw = session_observations(session.session_type, view.results(session.session_id), laps, entrants)
                push = _pace_group(push_pace(laps), entrants, SLOW_CUTOFF)
                # push pace replaces the best laps only when it covers at least half the drivers
                covered = push and raw and len(push[0][0]) >= len(raw[0][0]) / 2
                self._observations[session.session_id] = push if covered else raw
            return self._observations[session.session_id]
        if base_type(session.session_type) == "R":
            if session.session_id not in self._observations:
                groups, covariances = race_observations(view.laps(session.session_id),
                                                        view.entrants(session.session_id), with_covariance=True,
                                                        drying=session.session_type != "R")
                self._observations[session.session_id] = groups
                self._covariances[session.session_id] = covariances
            return self._observations[session.session_id]
        if session.session_id not in self._observations:
            kind = base_type(session.session_type)
            laps = view.laps(session.session_id) if kind not in SEGMENTED else None
            results = view.results(session.session_id)
            trends = None
            if self.wet_evolution and kind in SEGMENTED and session.session_type != kind:
                trends = qualifying_trend(view.laps(session.session_id), results)
                self._evolution[session.session_id] = [list(v.values()) for v in trends.values()]
            segments = []
            self._observations[session.session_id] = session_observations(
                kind, results, laps, view.entrants(session.session_id), trends, segments)
            if self.wet_segments and kind in SEGMENTED and session.session_type != kind:
                shares = segment_wet_shares(view.laps(session.session_id), results)
                self._group_types[session.session_id] = [
                    kind if shares.get(segment, 1.0) <= 0.5 else session.session_type for segment in segments]
        return self._observations[session.session_id]

    def group_types(self, session, count):
        """The session type each of a session's `count` groups is read as."""
        return self._group_types.get(session.session_id) or [session.session_type] * count

    def evolution_variance(self, view):
        """The variance of track condition between the segment times of a wet qualifying,
        pooled over the wet qualifying sessions available at the view's cutoff (0 if none):
        which lap of the track a driver's best fell on is unknown before the session."""
        total, count = 0.0, 0
        for session_id in view.sessions.session_id:
            for values in self._evolution.get(session_id, ()):
                if len(values) >= 2:
                    total += float(np.sum(np.square(np.asarray(values) - np.mean(values))))
                    count += len(values) - 1
        return total / count if count else 0.0

    def forecast_noise(self, view, example):
        """(lap noise, track spread) of a forecast segment: the forecast type's noise
        (a Student-t scale when robust) and, for a wet forecast with wet_evolution, the
        standard deviation of the track conditions the laps meet (0 otherwise)."""
        kind = self.forecast_type(example)
        track = np.sqrt(self.evolution_variance(view)) if self.wet_evolution and kind != "Q" else 0.0
        return self.hyper.noise[kind], float(track)

    def _wet_race(self, view, session_id):
        if session_id not in self._wet:
            self._wet[session_id] = wet_race(view.laps(session_id))
        return self._wet[session_id]

    def _wet_session(self, view, session_id):
        """Wet or mixed: wet-weather tyres on more than WET_SHARE of the session's timed laps."""
        if session_id not in self._wet:
            self._wet[session_id] = wet_session(view.laps(session_id))
        return self._wet[session_id]

    def failure_rates(self, view, example):
        """Per entrant and segment, the probability of setting no time, from every
        qualifying available at the cutoff (`failure_rates`)."""
        rows = []
        for session in view.sessions[view.sessions.session_type == "Q"].itertuples():
            if session.session_id not in self._failure_rows:
                ids = view.entrants(session.session_id).set_index("driver_number").driver_id
                self._failure_rows[session.session_id] = [
                    (ids.get(number), segment, failed)
                    for number, segment, failed in qualifying_failures(view.results(session.session_id))]
            rows += self._failure_rows[session.session_id]
        records = pd.DataFrame(rows, columns=["driver", "segment", "failed"]).dropna(subset=["driver"])
        drivers = [d if pd.notna(d) else f"unknown:{n}" for d, n in zip(example.driver_ids, example.driver_numbers)]
        return failure_rates(records, drivers)

    def _push_group(self, view, session_id):
        """The push-pace group of a practice session (cached), or [] for a wet or mixed
        session (its dry laps are not dry pace) or with too few drivers."""
        key = PUSH_PREFIX + session_id
        if key not in self._push_groups:
            self._push_groups[key] = [] if self._wet_session(view, session_id) else _pace_group(
                push_pace(view.laps(session_id)), view.entrants(session_id), SLOW_CUTOFF)
        return self._push_groups[key]

    def _push_covers(self, view, session):
        """Whether a practice session's push pace covers at least half the drivers of its
        best-lap group; caches the push group as its observations when it does."""
        if session.session_id not in self._push:
            laps = view.laps(session.session_id)
            entrants = view.entrants(session.session_id)
            base = session.session_type[:-1] if session.session_type in PUSH_TYPES else session.session_type
            raw = session_observations(base, view.results(session.session_id), laps, entrants)
            push = _pace_group(push_pace(laps), entrants, SLOW_CUTOFF)
            covered = bool(push and raw and len(push[0][0]) >= len(raw[0][0]) / 2)
            self._push[session.session_id] = covered
            if covered:
                self._observations[session.session_id] = push
        return self._push[session.session_id]

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
               season_skills=False, driver_affinity=False):
        """The filter run over `sessions`, recorded (see `program.Recorder`). With the
        qualifying objective, track 1 is the filter as it stood before each weekend and
        sees only that weekend's qualifying; only qualifying updates count."""
        objective = objective or self.objective
        recorder = Recorder(self.session_types, weekend, affinity, form, self.skills if skills else (),
                            self.season_skills if season_skills else (), self.driver_offsets, self.wet_ability,
                            driver_affinity)
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
            types = self.group_types(session, len(groups))
            if objective == "qualifying" and base_type(session.session_type) == "Q":
                # wet qualifying (QW) counts as qualifying and keeps its own noise
                recorder.advance(session.scheduled_start_utc, track=1)
                for (cars, drivers, values), kind in zip(groups, types):
                    recorder.update(cars, drivers, values, kind, track=1)
                    recorder.update(cars, drivers, values, kind)
                continue
            covariances = self._covariances.get(session.session_id, []) if self.race_estimates else []
            for index, (cars, drivers, values) in enumerate(groups):
                recorder.update(cars, drivers, values, types[index], counted=objective == "all",
                                covariance=covariances[index] if index < len(covariances) else None)
        return recorder

    def run(self, view, hyper, sessions):
        """Filter every session. `state.objective` is the quantity the settings maximise."""
        return self.record(view, sessions, hyper.weekend_prior > 0, hyper.affinity_prior > 0,
                           form=hyper.form_prior > 0, skills=hyper.skill_prior > 0,
                           season_skills=hyper.season_skill_prior > 0,
                           driver_affinity=hyper.driver_affinity_prior > 0).replay(hyper)

    def objective_function(self, view, sessions, like):
        """hyper -> objective over `sessions`, for settings with the same weekend and
        affinity switches as `like`: recorded once, replayed for every candidate."""
        return self.record(view, sessions, like.weekend_prior > 0, like.affinity_prior > 0,
                           form=like.form_prior > 0, skills=like.skill_prior > 0,
                           season_skills=like.season_skill_prior > 0,
                           driver_affinity=like.driver_affinity_prior > 0).objective

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
        driver_affinity = self.driver_affinity and len(seasons) > 1
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
        if self.driver_affinity and state and not driver_affinity:
            visited = {self.circuit(view, e) for e in sessions[history].event_key} - {None}
            driver_affinity = bool(visited & {self.circuit(view, e) for e in counted.event_key})
        counts = sessions.session_type.value_counts()
        # wet ability once the data has enough wet sessions to tell it from noise
        wet = self.wet_ability and sum(n for t, n in counts.items()
                                       if t.endswith(WET_SUFFIX) and base_type(t) != t) >= MIN_WET_SESSIONS
        scalars = [s for s in Hyper.fitted(types, winter, regulation, self.weekend, affinity, new_cars, new_drivers,
                                           form, skills, season_skills, self.race_estimates and "R" in types,
                                           self.robust, wet, driver_affinity)
                   if s not in frozen]
        noise_types = [t for t in types if f"noise:{t}" not in frozen
                       and not (t.endswith(WET_SUFFIX) and base_type(t) != t and counts.get(t, 0) < MIN_WET_SESSIONS)]
        default = self._default_start()
        off = {**({"affinity_prior": 0.0} if self.affinity and not affinity else {}),
               **({"driver_affinity_prior": 0.0} if self.driver_affinity and not driver_affinity else {}),
               **({"form_prior": 0.0} if self.form and not form else {}),
               **({"season_skill_prior": 0.0} if self.season_skills and not season_skills else {}),
               **({"skill_prior": 0.0, "skill_drift": 0.0} if self.skills and not skills else {})}
        start = replace(start, **off,
                        **{s: getattr(default, s) for s in scalars if getattr(start, s) <= 0})

        lower = [np.log(HISTORY_NOISE_FLOOR) if t == "H" else LOG_BOUNDS[0] for t in noise_types] \
            + [np.log(MIN_ROBUST_DOF) if name == "robust_dof" else LOG_BOUNDS[0] for name in scalars]
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
                             season_skills=bool(base.season_skills),
                             estimates=base.race_estimates and "R" in base.session_types, robust=base.robust,
                             wet=base.wet_ability, driver_affinity=base.driver_affinity)
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
        for flag in ("winter", "weekend", "affinity", "form", "driver_affinity"):
            if getattr(base, flag) and not getattr(self, flag):
                problems.append(flag)
        for bands in ("skills", "season_skills"):
            if not set(getattr(base, bands)) <= set(getattr(self, bands)):
                problems.append(bands)
        if base.race_estimates and not self.race_estimates:
            problems.append("race estimates")
        if base.robust and not self.robust:
            problems.append("robust")
        if base.wet_ability and not self.wet_ability:
            problems.append("wet ability")
        if base.wet_races and not self.wet_races:
            problems.append("wet races")
        if base.wet_evolution and not self.wet_evolution:
            problems.append("wet evolution")
        if base.wet_segments and not self.wet_segments:
            problems.append("wet segments")
        if base.practice_push and base.practice_push != self.practice_push:
            problems.append("practice push mode")
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
                     **({"season_skill_prior": 0.2} if self.season_skills else {}),
                     **({"estimate_scale": 1.0} if self.race_estimates else {}),
                     **({"robust_dof": 5.0} if self.robust else {}),
                     **({"wet_prior": 0.2, "wet_drift": 0.001} if self.wet_ability else {}),
                     **({"driver_affinity_prior": 0.1, "driver_affinity_drift": 0.0005,
                         "driver_affinity_winter": 0.01} if self.driver_affinity else {}))

    def fit(self, history, view_for):
        return self

    def options(self):
        """Everything besides the name that changes what this model fits."""
        return {"session_types": list(self.session_types), "winter": self.winter, "objective": self.objective,
                "weekend": self.weekend, "shared_from_qualifying": self.shared_from_qualifying,
                "affinity": self.affinity, "history": self.history, "form": self.form,
                "skills": list(self.skills), "season_skills": list(self.season_skills),
                "driver_offsets": list(self.driver_offsets), "practice_push": self.practice_push,
                "long_runs": self.long_runs, "race_estimates": self.race_estimates, "knockout": self.knockout, "failures": self.failures, "wet_noise": self.wet_noise, "robust": self.robust,
                "wet_ability": self.wet_ability, "wet_races": self.wet_races,
                **({"wet_evolution": True} if self.wet_evolution else {}),
                **({"wet_segments": True} if self.wet_segments else {}),
                **({"driver_affinity": True} if self.driver_affinity else {}),
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
        if (counted.session_type.map(base_type) == "Q").sum() < MIN_QUALIFYING_SESSIONS:
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
                               skills=self.hyper.skill_prior > 0, season_skills=self.hyper.season_skill_prior > 0,
                               driver_affinity=self.hyper.driver_affinity_prior > 0)
        recorder.advance(example.cutoff_time)
        if recorder.event != example.event_key:
            # before the weekend: its weekend effect is unknown; this circuit's affinity resumes
            recorder.start_event(example.event_key, self.circuit(view, example.event_key))
        if recorder.skills or recorder.season_skills:
            recorder.shares = self.shares(view, example.event_key)
        drivers = [d if pd.notna(d) else f"unknown:{n}" for d, n in zip(example.driver_ids, example.driver_numbers)]
        cars = [c if pd.notna(c) else f"unknown:{n}" for c, n in zip(example.lineage_ids, example.driver_numbers)]
        kind = self.forecast_type(example)
        rows = recorder.columns(cars, drivers, kind)
        state = recorder.replay(self.hyper)
        design = state.design(rows)
        mean = design @ state.mean
        noise, track = self.forecast_noise(view, example)
        cov = design @ state.cov @ design.T + (noise ** 2 + track ** 2) * np.eye(len(cars))
        return mean, cov

    def forecast_type(self, example):
        """The session type a forecast is made as: "QW" for a wet qualifying when the
        model forecasts by condition, else "Q"."""
        return f"Q{WET_SUFFIX}" if self.wet_forecast and getattr(example, "condition", None) == "wet" else "Q"

    def predict(self, view, example, draws=DRAWS):
        if example.session_type != "Q":
            return None
        self._ensure_hyper(view, example.season)
        mean, cov = self.distribution(view, example)
        size = len(mean)
        seed = seed_for(example)              # shared by every model: common random numbers
        rng = np.random.default_rng(seed)
        if self.knockout:
            # weekend pace without lap noise, then each segment's laps with their own noise
            noise, track = self.forecast_noise(view, example)
            lap = noise ** 2 + track ** 2
            pace = rng.multivariate_normal(mean, cov - lap * np.eye(size) + 1e-9 * np.eye(size), size=draws,
                                           method="cholesky")
            failures = self.failure_rates(view, example) if self.failures else None
            positions = knockout_positions(pace, noise, rng, failures, np.random.default_rng([seed, 1]),
                                           dof=self.hyper.robust_dof if self.robust else 0.0,
                                           scale_rng=np.random.default_rng([seed, 2]),
                                           track=track, track_rng=np.random.default_rng([seed, 3]))
        else:
            samples = rng.multivariate_normal(mean, cov + 1e-9 * np.eye(size), size=draws, method="cholesky")
            order = np.argsort(samples, axis=1)
            positions = np.empty_like(order)
            positions[np.arange(draws)[:, None], order] = np.arange(1, size + 1)
        return Forecast.from_draws(
            positions=positions, season=example.season, session_type=example.session_type,
            event_key=example.event_key, cutoff_label=example.cutoff_label, cutoff_time=example.cutoff_time.isoformat(),
            driver_numbers=list(example.driver_numbers), driver_ids=list(example.driver_ids), model=self.name,
            seed=seed, metadata={"condition": getattr(example, "condition", None), "hyper": {"noise": dict(self.hyper.noise), **{k: getattr(self.hyper, k)
                                                                         for k in Hyper.SCALARS},
                                           "fit": self.fit_log.get(example.season)}})
