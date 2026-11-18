"""Lap-by-lap race simulation (race model stage 2).

Each draw runs the race lap by lap, every car at its race pace:

- **Start.** Cars leave the grid spaced START_SPACING seconds per slot, shuffled
  by a start noise (`start_noise`, fitted: launches, the first corners); pace then
  acts through racing like on any lap, so a fast car starting behind still has to
  pass.
- **Racing laps.** Every lap, lap 1 included, every car laps at its race pace plus lap noise (LAP_NOISE, %
  of a lap). Front to back, a car closes on or drops away from the car ahead; it
  passes with the probability of the battle model (`PassModel`: the gap at the
  start of the lap, the race-pace difference, the circuit), and otherwise is held
  at least MIN_FOLLOW seconds behind. At most one pass per car per lap.
- **Battles.** `battles(laps)` reads, from a dry race, every pair of consecutive
  cars within BATTLE_GAP seconds at the end of a green lap on which neither
  pitted, and whether the car behind was ahead at the end of the next green lap.
  The pass model is a logistic regression on the gap, log gap, the race-pace
  difference (s per lap, from `laps.race_pace`) and its product with the gap,
  plus the circuit's log pass propensity from earlier races, shrunk towards the
  typical one by PASS_PRIOR_PAIRS pairs of it.

Measured on 2018-2023 dry races: 41,340 battle pairs, 2,362 passes; a car held
behind keeps at least about 0.35 s (5% quantile); lap-1 gaps median 0.77 s; lap
noise about 0.55% of a lap; the circuit term raises the held-out (leave-one-
season-out) battle log-likelihood by 113.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .laps import race_pace, wet_session

BATTLE_GAP = 2.5        # s: pairs further apart are not battles
MIN_FOLLOW = 0.35       # s: closest a car is held behind the car ahead
START_SPACING = 0.25    # s per grid slot at the start
LAP_NOISE = 0.55        # % of a lap: lap-to-lap noise of a car's lap time
PASS_PRIOR_PAIRS = 400.0
DEFAULT_LAPS, DEFAULT_LAP = 57, 90.0


def battles(laps):
    """Battle pairs of a dry race: DataFrame gap, delta (s per lap, car behind minus
    car ahead; negative: the car behind is faster), passed. Empty for a wet race or
    one without race pace."""
    columns = ("Position", "LapNumber", "DriverNumber", "PitInTime", "PitOutTime", "TrackStatus", "Time", "LapTime")
    empty = pd.DataFrame(columns=["gap", "delta", "passed"])
    if any(c not in laps for c in columns) or wet_session(laps):
        return empty
    pace = race_pace(laps)
    if pace.empty:
        return empty
    pace = dict(zip(pace.driver_number.astype(str), pace.pace))
    reference = float(laps.LapTime.dt.total_seconds().median())
    laps = laps.assign(number=laps.DriverNumber.astype(str), time=laps.Time.dt.total_seconds(),
                       pit=laps.PitInTime.notna() | laps.PitOutTime.notna(),
                       green=laps.TrackStatus.astype(str).str.fullmatch("1+"))
    green = laps.groupby("LapNumber").green.all()
    by_lap = {lap: frame.set_index("number") for lap, frame in laps.groupby("LapNumber")}
    rows = []
    for lap in sorted(by_lap):
        following = by_lap.get(lap + 1)
        if lap < 2 or following is None or not green.get(lap, False) or not green.get(lap + 1, False):
            continue
        now = by_lap[lap]
        now = now[now.Position.notna()].sort_values("Position")
        order = list(now.index)
        for ahead, behind in zip(order, order[1:]):
            if ahead not in following.index or behind not in following.index or ahead not in pace or behind not in pace:
                continue
            if now.pit[ahead] or now.pit[behind] or following.pit[ahead] or following.pit[behind]:
                continue
            if now.Position[behind] != now.Position[ahead] + 1:
                continue
            gap = now.time[behind] - now.time[ahead]
            if not 0 < gap < BATTLE_GAP or pd.isna(following.Position[behind]) or pd.isna(following.Position[ahead]):
                continue
            rows.append((gap, (pace[behind] - pace[ahead]) * reference / 100,
                         bool(following.Position[behind] < following.Position[ahead])))
    return pd.DataFrame(rows, columns=["gap", "delta", "passed"]) if rows else empty


def _features(gap, delta):
    delta = np.clip(delta, -3.0, 3.0)
    return np.stack([np.ones_like(gap), gap, np.log(np.maximum(gap, 0.01)), delta, delta * gap], axis=-1)


class PassModel:
    """P(the car behind is ahead after the next lap | gap, pace difference, circuit)."""

    def __init__(self):
        self.coef = np.zeros(6)
        self.base = 0.0           # typical pass share of battle pairs

    def fit(self, frame, term):
        """`frame`: battles; `term`: each pair's circuit log pass propensity, from
        races other than its own season's (`circuit_propensity`)."""
        if frame.empty:
            self.coef = np.array([-10.0, 0, 0, 0, 0, 0])      # no battles seen: no passes
            return self
        x = np.column_stack([_features(frame.gap.to_numpy(float), frame.delta.to_numpy(float)),
                             np.asarray(term, dtype=float)])
        y = frame.passed.to_numpy(float)

        def loss(b):
            z = x @ b
            return float(np.sum(np.logaddexp(0, z) - y * z))
        self.coef = minimize(loss, np.zeros(x.shape[1]), method="BFGS").x
        self.base = float(y.mean())
        return self

    def probability(self, gap, delta, circuit_term=0.0):
        z = _features(gap, delta) @ self.coef[:5] + self.coef[5] * circuit_term
        return 1.0 / (1.0 + np.exp(-z))


def circuit_propensity(frame, base):
    """{circuit: log(shrunk pass share / typical)} of battle pairs in `frame`."""
    counts = frame.groupby("circuit").passed.agg(["sum", "count"])
    share = (counts["sum"] + PASS_PRIOR_PAIRS * base) / (counts["count"] + PASS_PRIOR_PAIRS)
    return np.log(share / base).to_dict()


def simulate(pace, grid, passes, circuit_term, laps, lap_seconds, start_noise, rng):
    """Finishing order (draws x entrants, 1 = first) of `laps` laps.

    `pace`: race pace draws (% of a lap, lower is faster); `grid`: slots, one row or
    one per draw; `passes`: a PassModel; `lap_seconds`: a typical lap."""
    draws, n = pace.shape
    seconds = pace * lap_seconds / 100                       # s per lap against the field's reference
    grid = np.broadcast_to(grid, pace.shape)
    time = START_SPACING * (grid - 1) + start_noise * rng.standard_normal(pace.shape)
    rows = np.arange(draws)
    for _ in range(laps):
        order = np.argsort(time, axis=1, kind="stable")        # car at each position
        before = np.take_along_axis(time, order, axis=1)
        lap = seconds + (LAP_NOISE * lap_seconds / 100) * rng.standard_normal(pace.shape)
        after = before + np.take_along_axis(lap, order, axis=1)
        own = np.take_along_axis(seconds, order, axis=1)
        luck = rng.random((draws, n))
        for k in range(1, n):
            ahead, behind = order[:, k - 1].copy(), order[:, k].copy()
            gap = before[:, k] - before[:, k - 1]
            delta = own[:, k] - own[:, k - 1]
            chance = np.where(gap < BATTLE_GAP, passes.probability(np.maximum(gap, 0.01), delta, circuit_term), 0.0)
            passed = luck[:, k] < chance
            # passed: swap; the passed car is held behind the passer
            t_ahead, t_behind = after[:, k - 1].copy(), after[:, k].copy()
            first = np.minimum(t_ahead, t_behind)
            if k >= 2:      # one place per lap: the passer stays behind the car two ahead
                first = np.maximum(first, after[:, k - 2] + MIN_FOLLOW)
            first = np.where(passed, first, t_ahead)
            second = np.where(passed, np.maximum(t_ahead, first + MIN_FOLLOW),
                              np.maximum(t_behind, t_ahead + MIN_FOLLOW))
            after[:, k - 1], after[:, k] = first, second
            order[:, k - 1] = np.where(passed, behind, ahead)
            order[:, k] = np.where(passed, ahead, behind)
            before[:, k - 1], before[:, k] = np.where(passed, before[:, k], before[:, k - 1]), \
                np.where(passed, before[:, k - 1], before[:, k])
            own[:, k - 1], own[:, k] = np.where(passed, own[:, k], own[:, k - 1]), np.where(passed, own[:, k - 1], own[:, k])
        time = np.empty_like(time)
        time[rows[:, None], order] = after
    order = np.argsort(time, axis=1, kind="stable")
    positions = np.empty_like(order)
    positions[rows[:, None], order] = np.arange(1, n + 1)
    return positions
