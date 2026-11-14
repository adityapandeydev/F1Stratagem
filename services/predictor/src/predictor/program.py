"""The pace filter: Python records what happens, the native module does the arithmetic.

Which states exist, when they are added, parked, reset or removed, and which
states every observation touches depend only on the sessions and on whether the
weekend and affinity terms are on, never on the settings' values. A `Recorder`
walks the sessions once, keeps that bookkeeping (state keys, parked affinities,
the current event) and emits a flat program. `Recorder.replay(hyper)` hands the
program to `predictor_native`, which runs the Kalman filter for those settings
and returns the log-likelihood and the final mean and covariance. A settings fit
records once and replays the same program for every candidate.

Two filters can run side by side: track 0 is the filter, track 1 a copy taken at
an event's start (SNAPSHOT) that sees only that weekend's qualifying, for the
qualifying objective's before-the-weekend horizon.
"""
from dataclasses import dataclass

import numpy as np

from .rules import REGULATION_CHANGES

try:
    import predictor_native
except ImportError as error:   # pragma: no cover - the build step is part of installing the service
    raise ImportError("predictor_native is not built; run `make predictor-native`") from error

KINDS = ("car", "driver", "offset", "weekend", "affinity", "form", "skill", "season_skill")
REFERENCE_TYPES = ("Q", "H")  # session types read without an offset: qualifying and historical qualifying
DAY_SCALE = 14.0  # drift variances are per 14 days

# opcodes; every op is one row of `ops` (code, track, a, b, c) plus optional slices of the shared arrays
ADVANCE = 0   # a = steps (index into floats), b = ordinary season starts crossed, c = regulation season starts crossed
ADD = 1       # a = kind, b = parked slot or -1, c = accrual record (start in ints: steps index or -1, two counts)
PARK = 2      # a = start in ints, b = count: pairs (state index, slot); the states are then removed
RESET = 3     # a = start in ints, b = count, c = kind: state indices returned to that kind's prior
UPDATE = 4    # a = group index, b = 1 if its log-likelihood counts towards the objective
SNAPSHOT = 5  # track 1 becomes a copy of track 0 (state and parked affinities)


@dataclass
class Program:
    ops: np.ndarray            # (n, 5) int64
    floats: np.ndarray         # steps for ADVANCE and accruals
    ints: np.ndarray           # index lists for PARK, RESET and accrual records
    group_rows: np.ndarray     # per group: first driver row, number of drivers, noise type index
    row_starts: np.ndarray     # per driver row: start in `entries` (rows of a group are consecutive), plus the end
    entries: np.ndarray        # state indices of every design row
    weights: np.ndarray        # one per entry: how much of that state the pace reads (1 unless weighted)
    values: np.ndarray         # observed log times x 100, one per driver row
    noise_types: tuple
    slots: int

    def arrays(self):
        return (self.ops.ravel(), self.floats, self.ints, self.group_rows.ravel(), self.row_starts, self.entries,
                self.weights, self.values, self.slots)


@dataclass
class State:
    """Track 0 after a replay."""
    keys: list                 # (kind, key) of every state, in index order
    mean: np.ndarray
    cov: np.ndarray
    objective: float           # counted log-likelihood, both tracks
    horizons: dict             # counted log-likelihood per track: before the weekend (1), the filter (0)

    @property
    def index(self):
        return {key: i for i, key in enumerate(self.keys)}

    @property
    def kind(self):
        return [kind for kind, _ in self.keys]

    def design(self, rows):
        """Design matrix of `Recorder.columns` rows (state indices, or (index, weight) pairs)."""
        design = np.zeros((len(rows), len(self.keys)))
        for row, entries in enumerate(rows):
            for index, weight in (_weighted(e) for e in entries):
                design[row, index] += weight
        return design


def _weighted(entry):
    """(state index, weight) of a design entry: an index (weight 1) or an (index, weight) tuple."""
    return entry if isinstance(entry, tuple) else (entry, 1.0)


def settings(hyper, noise_types):
    """Settings as the arrays the native filter reads."""
    priors = [hyper.car_prior, hyper.driver_prior, hyper.offset_prior, hyper.weekend_prior, hyper.affinity_prior,
              hyper.form_prior, hyper.skill_prior, hyper.season_skill_prior]
    drifts = [hyper.car_drift, hyper.driver_drift, hyper.offset_drift, 0.0, hyper.affinity_drift, 0.0,
              hyper.skill_drift, 0.0]
    winters = [hyper.car_winter, hyper.driver_winter, 0.0, 0.0, hyper.affinity_winter, 0.0, 0.0, 0.0]
    multiplied = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]   # which kinds the regulation multiplier scales
    noise = [hyper.noise.get(t, 0.0) for t in noise_types]
    return (np.array(priors, dtype=np.float64), np.array(drifts, dtype=np.float64),
            np.array(winters, dtype=np.float64), np.array(multiplied, dtype=np.float64),
            float(hyper.regulation_multiplier), np.array(noise, dtype=np.float64))


class _Track:
    """Bookkeeping of one filter: which state sits at which index."""

    def __init__(self):
        self.index, self.kind = {}, []
        self.parked = {}      # affinity key -> (slot, time parked)
        self.time = None

    def copy(self):
        clone = _Track()
        clone.index, clone.kind, clone.parked, clone.time = dict(self.index), list(self.kind), dict(self.parked), self.time
        return clone


class Recorder:
    """A pace filter run, recorded. Drive it like a filter (advance, start_event,
    update), then `replay(hyper)` for the state under any settings with the same
    weekend / affinity switches. `columns` adds the states a forecast reads."""

    def __init__(self, noise_types, weekend=False, affinity=False, form=False, skills=(), season_skills=()):
        self.weekend, self.affinity, self.form = weekend, affinity, form
        # parts of the track (track.BANDS) with a driver skill, persistent or per season; an
        # observation reads each weighted by its track's share (`shares`, set per session)
        self.skills, self.season_skills = tuple(skills), tuple(season_skills)
        self.shares = None
        self.noise_types = list(noise_types)
        self.tracks = [_Track()]
        self.slot_of = {}
        self.event, self.circuit = None, None
        self.ops, self.floats, self.ints = [], [], []
        self.groups, self.row_starts, self.entries, self.weights, self.values = [], [], [], [], []
        self._program = None

    def _float(self, value):
        self.floats.append(float(value))
        return len(self.floats) - 1

    def _op(self, *op):
        self.ops.append(op)
        self._program = None

    @staticmethod
    def _seasons(start, end):
        years = range(start.year + 1, end.year + 1)
        regulation = sum(1 for y in years if y in REGULATION_CHANGES)
        return len(years) - regulation, regulation

    def column(self, key, kind, track=0):
        t = self.tracks[track]
        if (kind, key) not in t.index:
            slot, accrual = -1, -1
            if kind == "affinity" and key in t.parked:
                slot, since = t.parked.pop(key)
                accrual = len(self.ints)
                if since is None or t.time is None or t.time <= since:
                    self.ints += [-1, 0, 0]
                else:
                    ordinary, regulation = self._seasons(since, t.time)
                    steps = (t.time - since).total_seconds() / 86400 / DAY_SCALE
                    self.ints += [self._float(steps), ordinary, regulation]
            self._op(ADD, track, KINDS.index(kind), slot, accrual)
            t.index[(kind, key)] = len(t.kind)
            t.kind.append(kind)
        return t.index[(kind, key)]

    def advance(self, time, track=0):
        t = self.tracks[track]
        if t.time is not None and time > t.time:
            ordinary, regulation = self._seasons(t.time, time)
            steps = (time - t.time).total_seconds() / 86400 / DAY_SCALE
            self._op(ADVANCE, track, self._float(steps), ordinary, regulation)
            if self.form and ordinary + regulation:     # a new season: season form returns to its prior
                self._reset("form", track)
            if self.season_skills and ordinary + regulation:
                self._reset("season_skill", track)
        t.time = time if t.time is None else max(t.time, time)

    def start_event(self, event, circuit=None):
        """A new event: weekend effects return to their prior, and affinities for other
        circuits are parked until the next visit."""
        self.event = event
        t = self.tracks[0]
        if self.affinity:
            active = [i for i, kind in enumerate(t.kind) if kind == "affinity"]
            keys = sorted(t.index, key=t.index.get)
            if active:
                start = len(self.ints)
                for i in active:
                    key = keys[i][1]
                    slot = self.slot_of.setdefault(key, len(self.slot_of))
                    t.parked[key] = (slot, t.time)
                    self.ints += [i, slot]
                self._op(PARK, 0, start, len(active), 0)
                removed = set(active)
                keep = [i for i in range(len(t.kind)) if i not in removed]
                t.index = {key: position for position, key in enumerate(keys[i] for i in keep)}
                t.kind = [t.kind[i] for i in keep]
            self.circuit = circuit
        if self.weekend:
            self._reset("weekend", 0)

    def _reset(self, kind, track):
        t = self.tracks[track]
        states = [i for i, k in enumerate(t.kind) if k == kind]
        if states:
            start = len(self.ints)
            self.ints += states
            self._op(RESET, track, start, len(states), KINDS.index(kind))

    def snapshot(self):
        """Track 1 becomes a copy of track 0."""
        if len(self.tracks) == 1:
            self.tracks.append(None)
        self.tracks[1] = self.tracks[0].copy()
        self._op(SNAPSHOT, 1, 0, 0, 0)

    def columns(self, cars, drivers, session_type="Q", track=0):
        """State indices each driver's pace reads, adding states seen for the first time."""
        rows = []
        for car, driver in zip(cars, drivers):
            row = [self.column(car, "car", track), self.column(driver, "driver", track)]
            if session_type not in REFERENCE_TYPES:
                row.append(self.column((car, session_type), "offset", track))
            if self.weekend and session_type != "H":   # a historical session is its weekend's only one
                row.append(self.column(car, "weekend", track))
            if self.form:
                row.append(self.column(driver, "form", track))
            if self.affinity and self.circuit is not None:
                row.append(self.column((car, self.circuit), "affinity", track))
            if self.shares is not None and session_type != "H":
                for band in self.skills:
                    row.append((self.column((driver, band), "skill", track), self.shares[f"share_{band}"]))
                for band in self.season_skills:
                    row.append((self.column((driver, band), "season_skill", track), self.shares[f"share_{band}"]))
            rows.append(row)
        return rows

    def update(self, cars, drivers, values, session_type="Q", track=0, counted=True):
        """Relative update: every driver against the first one in the group."""
        first_row = len(self.row_starts)
        for row in self.columns(cars, drivers, session_type, track):
            self.row_starts.append(len(self.entries))
            for index, weight in (_weighted(e) for e in row):
                self.entries.append(index)
                self.weights.append(float(weight))
        self.values += [float(v) for v in values]
        self._op(UPDATE, track, len(self.groups), int(counted), 0)
        self.groups.append((first_row, len(cars), self.noise_types.index(session_type)))

    @property
    def keys(self):
        t = self.tracks[0]
        return sorted(t.index, key=t.index.get)

    def program(self):
        if self._program is None:
            self._program = Program(
                ops=np.array(self.ops, dtype=np.int64).reshape(-1, 5), floats=np.array(self.floats, dtype=np.float64),
                ints=np.array(self.ints, dtype=np.int64), group_rows=np.array(self.groups, dtype=np.int64).reshape(-1, 3),
                row_starts=np.array(self.row_starts + [len(self.entries)], dtype=np.int64),
                entries=np.array(self.entries, dtype=np.int64), weights=np.array(self.weights, dtype=np.float64),
                values=np.array(self.values, dtype=np.float64),
                noise_types=tuple(self.noise_types), slots=len(self.slot_of))
        return self._program

    def _check(self, hyper):
        if (hyper.weekend_prior > 0, hyper.affinity_prior > 0, hyper.form_prior > 0, hyper.skill_prior > 0,
                hyper.season_skill_prior > 0) != (self.weekend, self.affinity, self.form, bool(self.skills),
                                                  bool(self.season_skills)):
            raise ValueError("settings switch a term the recorded program does not have")

    def objective(self, hyper):
        """Counted log-likelihood under `hyper` (the settings fit's objective)."""
        self._check(hyper)
        p = self.program()
        return predictor_native.log_likelihood(*p.arrays(), *settings(hyper, p.noise_types))

    def replay(self, hyper):
        """Track 0's state under `hyper`, with the counted log-likelihood of each track."""
        self._check(hyper)
        p = self.program()
        likelihoods, mean, cov = predictor_native.filter_state(*p.arrays(), *settings(hyper, p.noise_types))
        return State(keys=self.keys, mean=mean, cov=cov, objective=float(likelihoods.sum()),
                     horizons={"after_practice": float(likelihoods[0]), "before_weekend": float(likelihoods[1])})
