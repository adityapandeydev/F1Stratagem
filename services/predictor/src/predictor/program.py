"""A pace-model run recorded as a fixed program, replayed by the native filter.

Which states exist, when they are added, parked, reset or removed, and which
states every observation touches depend only on the sessions and on whether the
weekend and affinity terms are on, never on the settings' values. `record`
walks the sessions once with the same bookkeeping as `pace.Filter` and emits
flat arrays; `predictor_native.log_likelihood` then runs the Kalman filter for
any settings without Python in the loop. The objective is the same quantity as
`PaceModel.run(...).objective` with the default objective (differences are
rounding only, checked by the tests).
"""
from dataclasses import dataclass

import numpy as np

from .rules import REGULATION_CHANGES

KINDS = ("car", "driver", "offset", "weekend", "affinity")
CAR, DRIVER, OFFSET, WEEKEND, AFFINITY = range(5)
DAY_SCALE = 14.0

# opcodes; every op is one row of `ops` (code, a, b, c) plus optional slices of the shared arrays
ADVANCE = 0   # a = steps (index into floats), b = ordinary season starts crossed, c = regulation season starts crossed
ADD = 1       # a = kind, b = parked slot or -1, c = accrual record (index into floats, then two counts in ints)
PARK = 2      # a = start in ints, b = count: pairs (state index, slot); the states are then removed
RESET = 3     # a = start in ints, b = count: weekend state indices returned to their prior
UPDATE = 4    # a = group index


@dataclass
class Program:
    ops: np.ndarray            # (n, 4) int64
    floats: np.ndarray         # steps for ADVANCE and accruals
    ints: np.ndarray           # index lists for PARK, RESET and accrual counts
    group_rows: np.ndarray     # per group: start in `entries`, number of drivers, noise type index
    row_starts: np.ndarray     # per driver row: start in `entries` (rows of a group are consecutive), plus the end
    entries: np.ndarray        # state indices of every design row
    values: np.ndarray         # observed log times x 100, one per driver row
    noise_types: tuple
    slots: int


class _Recorder:
    def __init__(self, weekend, affinity, noise_types):
        self.weekend, self.affinity = weekend, affinity
        self.noise_types = list(noise_types)
        self.index, self.kind = {}, []
        self.parked = {}      # key -> (slot, time parked)
        self.slot_of = {}
        self.time = None
        self.circuit = None
        self.ops, self.floats, self.ints = [], [], []
        self.groups, self.row_starts, self.entries, self.values = [], [], [], []

    def _float(self, value):
        self.floats.append(float(value))
        return len(self.floats) - 1

    @staticmethod
    def _seasons(start, end):
        years = range(start.year + 1, end.year + 1)
        regulation = sum(1 for y in years if y in REGULATION_CHANGES)
        return len(years) - regulation, regulation

    def column(self, key, kind):
        if (kind, key) not in self.index:
            slot, accrual = -1, -1
            if kind == "affinity" and key in self.parked:
                slot, since = self.parked.pop(key)
                accrual = len(self.ints)
                if since is None or self.time is None or self.time <= since:
                    self.ints += [-1, 0, 0]
                else:
                    ordinary, regulation = self._seasons(since, self.time)
                    steps = (self.time - since).total_seconds() / 86400 / DAY_SCALE
                    self.ints += [self._float(steps), ordinary, regulation]
            self.ops.append((ADD, KINDS.index(kind), slot, accrual))
            self.index[(kind, key)] = len(self.kind)
            self.kind.append(kind)
        return self.index[(kind, key)]

    def advance(self, time):
        if self.time is not None and time > self.time:
            ordinary, regulation = self._seasons(self.time, time)
            steps = (time - self.time).total_seconds() / 86400 / DAY_SCALE
            self.ops.append((ADVANCE, self._float(steps), ordinary, regulation))
        self.time = time if self.time is None else max(self.time, time)

    def start_event(self, circuit):
        if self.affinity:
            active = [i for i, kind in enumerate(self.kind) if kind == "affinity"]
            keys = sorted(self.index, key=self.index.get)
            if active:
                start = len(self.ints)
                for i in active:
                    key = keys[i][1]
                    slot = self.slot_of.setdefault(key, len(self.slot_of))
                    self.parked[key] = (slot, self.time)
                    self.ints += [i, slot]
                self.ops.append((PARK, start, len(active), 0))
                keep = [i for i in range(len(self.kind)) if i not in set(active)]
                self.index = {key: position for position, key in enumerate(keys[i] for i in keep)}
                self.kind = [self.kind[i] for i in keep]
            self.circuit = circuit
        if self.weekend:
            weekend = [i for i, kind in enumerate(self.kind) if kind == "weekend"]
            if weekend:
                start = len(self.ints)
                self.ints += weekend
                self.ops.append((RESET, start, len(weekend), 0))

    def update(self, cars, drivers, values, session_type):
        first_row = len(self.row_starts)
        for car, driver in zip(cars, drivers):
            row = [self.column(car, "car"), self.column(driver, "driver")]
            if session_type != "Q":
                row.append(self.column((car, session_type), "offset"))
            if self.weekend:
                row.append(self.column(car, "weekend"))
            if self.affinity and self.circuit is not None:
                row.append(self.column((car, self.circuit), "affinity"))
            self.row_starts.append(len(self.entries))
            self.entries += row
        self.values += [float(v) for v in values]
        self.ops.append((UPDATE, len(self.groups), 0, 0))
        self.groups.append((first_row, len(cars), self.noise_types.index(session_type)))


def record(model, view, sessions, weekend, affinity):
    """The program of `model.run(view, hyper, sessions)` for any hyper with the given
    weekend / affinity switches (default objective only)."""
    if model.objective != "all":
        raise ValueError("the native filter supports the default objective only")
    recorder = _Recorder(weekend, affinity, model.session_types)
    event = None
    for session in sessions.itertuples():
        recorder.advance(session.scheduled_start_utc)
        if session.event_key != event:
            event = session.event_key
            recorder.start_event(model.circuit(view, event))
        for cars, drivers, values in model.observations(view, session):
            recorder.update(cars, drivers, values, session.session_type)
    r = recorder
    return Program(ops=np.array(r.ops, dtype=np.int64).reshape(-1, 4), floats=np.array(r.floats, dtype=np.float64),
                   ints=np.array(r.ints, dtype=np.int64), group_rows=np.array(r.groups, dtype=np.int64).reshape(-1, 3),
                   row_starts=np.array(r.row_starts + [len(r.entries)], dtype=np.int64),
                   entries=np.array(r.entries, dtype=np.int64), values=np.array(r.values, dtype=np.float64),
                   noise_types=tuple(model.session_types), slots=len(r.slot_of))


def settings(hyper, noise_types):
    """Settings as the arrays the native filter reads."""
    priors = [hyper.car_prior, hyper.driver_prior, hyper.offset_prior, hyper.weekend_prior, hyper.affinity_prior]
    drifts = [hyper.car_drift, hyper.driver_drift, hyper.offset_drift, 0.0, hyper.affinity_drift]
    winters = [hyper.car_winter, hyper.driver_winter, 0.0, 0.0, hyper.affinity_winter]
    multiplied = [1.0, 0.0, 0.0, 0.0, 1.0]   # which kinds the regulation multiplier scales
    noise = [hyper.noise.get(t, 0.0) for t in noise_types]
    return (np.array(priors, dtype=np.float64), np.array(drifts, dtype=np.float64),
            np.array(winters, dtype=np.float64), np.array(multiplied, dtype=np.float64),
            float(hyper.regulation_multiplier), np.array(noise, dtype=np.float64))
