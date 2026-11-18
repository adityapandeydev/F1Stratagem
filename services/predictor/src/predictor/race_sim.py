"""Lap-by-lap race simulation (race model stage 2).

Each draw runs the race lap by lap, every car at its race pace:

- **Start.** Cars leave the grid spaced START_SPACING seconds per slot, shuffled
  by a start noise (`start_noise`, fitted: launches, the first corners); pace then
  acts through racing like on any lap, so a fast car starting behind still has to
  pass.
- **Racing laps.** Every lap, lap 1 included, every car laps at its race pace plus
  lap noise (LAP_NOISE, % of a lap). Front to back, a car closes on or drops away from the car ahead; it
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
import predictor_race_native
from scipy.optimize import minimize

from .laps import race_pace, wet_session

BATTLE_GAP = 2.5        # s: pairs further apart are not battles
MIN_FOLLOW = 0.35       # s: closest a car is held behind the car ahead
START_SPACING = 0.25    # s per grid slot at the start
LAP_NOISE = 0.55        # % of a lap: lap-to-lap noise of a car's lap time
PASS_PRIOR_PAIRS = 400.0
DEFAULT_LAPS, DEFAULT_LAP = 57, 90.0
KINDS = ("sc", "vsc", "red")    # neutralisations, in the native module's order
SEVERITY = {"vsc": 0, "sc": 1, "red": 2}   # a run of mixed laps takes its most severe kind
SC_GAP = 0.52           # s: median gap between consecutive cars when a safety car ends (2018-2024)
SLOW_LAP = 1.10         # a lap whose field median is this much slower than the race's typical lap is neutralised
SLOW_FIRST_LAP = 1.25   # the same for lap 1, which includes the standing start
RECENT_SEASONS = 3      # seasons the base neutralisation rate is taken from
RACE_TO_QUALIFYING = 1.075   # typical green race lap / median best qualifying lap, without earlier races
MIN_GREEN_LAPS = 10     # racing laps a race needs to give a lap reference


def green_lap_seconds(laps):
    """Median time (s) of racing laps: green for every car, no pit stop, lap 1 left
    out. None with fewer than MIN_GREEN_LAPS such laps (a race run mostly behind the
    safety car)."""
    if any(c not in laps for c in ("LapNumber", "LapTime", "TrackStatus", "PitInTime", "PitOutTime")):
        return None
    status = laps.TrackStatus.astype(str).str.fullmatch("1+")
    green = status.groupby(laps.LapNumber).transform("all")
    keep = green & (laps.LapNumber > 1) & laps.PitInTime.isna() & laps.PitOutTime.isna() & laps.LapTime.notna()
    if laps.LapNumber[keep].nunique() < MIN_GREEN_LAPS:
        return None
    return float(laps.LapTime[keep].dt.total_seconds().median())


def qualifying_lap(laps):
    """Median of the drivers' best laps of a dry qualifying session (s); None for a
    wet or mixed session or one without lap times."""
    if laps.empty or "LapTime" not in laps or wet_session(laps):
        return None
    best = laps.dropna(subset=["LapTime"]).groupby("DriverNumber").LapTime.min().dt.total_seconds()
    return float(best.median()) if len(best) else None


def race_shape(laps):
    """(laps completed by the leader, green lap time in s or None) of a race; no lap
    time from a wet or mixed race (slower than a dry one)."""
    completed = int(laps.LapNumber.max()) if "LapNumber" in laps and len(laps) else 0
    if not len(laps) or wet_session(laps):
        return completed, None
    return completed, green_lap_seconds(laps)


def battles(laps):
    """Battle pairs of a dry race: DataFrame gap, delta (s per lap, car behind minus
    car ahead, from the race's own measured pace; negative: the car behind is
    faster), passed, ahead and behind (driver numbers), lap (the race's green lap,
    s). Empty for a wet race or one without race pace."""
    columns = ("Position", "LapNumber", "DriverNumber", "PitInTime", "PitOutTime", "TrackStatus", "Time", "LapTime")
    empty = pd.DataFrame(columns=["gap", "delta", "passed", "ahead", "behind", "lap"])
    if any(c not in laps for c in columns) or wet_session(laps):
        return empty
    pace = race_pace(laps)
    if pace.empty:
        return empty
    pace = dict(zip(pace.driver_number.astype(str), pace.pace))
    reference = green_lap_seconds(laps)
    if reference is None:
        return empty
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
                         bool(following.Position[behind] < following.Position[ahead]), ahead, behind, reference))
    return pd.DataFrame(rows, columns=list(empty.columns)) if rows else empty


def _features(gap, delta):
    delta = np.clip(delta, -3.0, 3.0)
    return np.stack([np.ones_like(gap), gap, np.log(np.maximum(gap, 0.01)), delta, delta * gap], axis=-1)


class PassModel:
    """P(the car behind is ahead after the next lap | gap, pace difference, circuit)."""

    def __init__(self):
        self.coef = np.zeros(6)
        self.base = 0.0           # typical pass share of battle pairs

    def fit(self, frame, term, weights=None):
        """`frame`: battles; `term`: each pair's circuit log pass propensity, from
        races other than its own season's (`circuit_propensity`); `weights`: per
        pair (default 1)."""
        if frame.empty:
            self.coef = np.array([-10.0, 0, 0, 0, 0, 0])      # no battles seen: no passes
            return self
        x = np.column_stack([_features(frame.gap.to_numpy(float), frame.delta.to_numpy(float)),
                             np.asarray(term, dtype=float)])
        y = frame.passed.to_numpy(float)
        w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)

        def loss(b):
            z = x @ b
            return float(np.sum(w * (np.logaddexp(0, z) - y * z)))
        self.coef = minimize(loss, np.zeros(x.shape[1]), method="BFGS").x
        self.base = float(np.sum(w * y) / np.sum(w))
        return self

    def probability(self, gap, delta, circuit_term=0.0, shift=0.0):
        """`shift`: added to the log-odds (a race-outcome calibration of passing).
        Computed natively, the same function the simulation uses."""
        gap, delta = np.broadcast_arrays(np.asarray(gap, dtype=float), np.asarray(delta, dtype=float))
        out = predictor_race_native.pass_probability(np.ascontiguousarray(gap.ravel()), np.ascontiguousarray(delta.ravel()),
                                                     np.asarray(self.coef, dtype=float), float(circuit_term), float(shift))
        return out.reshape(gap.shape)


def circuit_propensity(frame, base):
    """{circuit: log(shrunk pass share / typical)} of battle pairs in `frame`; every
    circuit neutral (0) when the typical share is not positive (no passes seen)."""
    if not base > 0:
        return {circuit: 0.0 for circuit in frame.circuit.unique()}
    counts = frame.groupby("circuit").passed.agg(["sum", "count"])
    share = (counts["sum"] + PASS_PRIOR_PAIRS * base) / (counts["count"] + PASS_PRIOR_PAIRS)
    return np.log(share / base).to_dict()


def simulate(pace, grid, passes, circuit_term, laps, lap_seconds, start_noise, rng, pass_shift=0.0, neutral=None):
    """Finishing order (draws x entrants, 1 = first) of `laps` laps, run natively
    (`predictor_race_native.simulate`, its own random numbers seeded from `rng`).

    `pace`: race pace draws (% of a lap, lower is faster); `grid`: slots, one row or
    one per draw; `passes`: a PassModel; `lap_seconds`: a typical lap; `pass_shift`:
    added to every pass's log-odds; `neutral`: `NeutralModel.race(...)` (None: no
    neutralisations)."""
    laps = int(laps)
    if neutral is None:
        neutral = {"hazard": np.zeros(laps), "kinds": np.array([1.0, 0.0, 0.0]),
                   "lengths": {kind: np.array([1]) for kind in KINDS}, "red_end": 0.0}
    seconds = np.ascontiguousarray(pace * lap_seconds / 100, dtype=float)   # s per lap against the field's reference
    grid = np.ascontiguousarray(np.broadcast_to(grid, pace.shape), dtype=float)
    seed = int(rng.integers(0, 2 ** 63))
    lengths = [np.asarray(neutral["lengths"][kind], dtype=np.int64) for kind in KINDS]
    return predictor_race_native.simulate(seconds, grid, np.asarray(passes.coef, dtype=float), float(circuit_term),
                                          float(pass_shift), laps, float(start_noise), START_SPACING,
                                          float(LAP_NOISE * lap_seconds / 100), BATTLE_GAP, MIN_FOLLOW, seed,
                                          np.ascontiguousarray(neutral["hazard"][:laps], dtype=float),
                                          np.asarray(neutral["kinds"], dtype=float), *lengths, SC_GAP,
                                          float(neutral["red_end"]))


def neutral_periods(laps):
    """Neutralisations of a race from its track status: [(first lap, kind, laps)],
    one per run of consecutive neutralised laps, its kind the most severe in the run
    (red flag, then safety car, then virtual safety car); None without track status."""
    if "TrackStatus" not in laps or laps.empty:
        return None
    status = laps.groupby("LapNumber").TrackStatus.agg(lambda s: "".join(str(x) for x in s.dropna()))
    kind = pd.Series(None, index=status.index, dtype=object)
    kind[status.str.contains("[67]")] = "vsc"
    kind[status.str.contains("4")] = "sc"
    kind[status.str.contains("5")] = "red"
    periods, run = [], None
    for lap, here in kind.items():
        if pd.isna(here):
            if run:
                periods.append(tuple(run))
            run = None
        elif run is None:
            run = [int(lap), here, 1]
        else:
            run[2] += 1
            if SEVERITY[here] > SEVERITY[run[1]]:
                run[1] = here
    if run:
        periods.append(tuple(run))
    return periods


def slow_lap_periods(times):
    """Neutralisations inferred from lap times alone (races without track status):
    `times` maps lap -> every car's lap time (s). A lap is neutralised when the
    field's median is SLOW_LAP times the race's typical lap (SLOW_FIRST_LAP on lap
    1); [(first lap, None, laps)]. On 2018, the one season with both, this gives
    the exact count in 18 of 20 races and the same total (27)."""
    median = pd.Series({lap: float(np.median(t)) for lap, t in times.items()}).sort_index()
    typical = median[median.index > 1].median()
    slow = median > SLOW_LAP * typical
    if 1 in slow.index:
        slow[1] = median[1] > SLOW_FIRST_LAP * typical
    periods, start = [], None
    for lap, here in slow.items():
        if here and start is None:
            start = int(lap)
        elif not here and start is not None:
            periods.append((start, None, int(lap) - start))
            start = None
    if start is not None:
        periods.append((start, None, int(slow.index[-1]) - start + 1))
    return periods


def history_neutralisations(root, cutoff, before_season=2018):
    """Races before `before_season` from Jolpica's lap times (1996 on) available at
    `cutoff`: DataFrame season, circuit, laps, n (neutralisations inferred by
    `slow_lap_periods`)."""
    from .archive import NOMINAL_DURATION, PUBLICATION_DELAY
    tables = root / "external" / "jolpica" / "tables"
    if not (tables / "laps.parquet").exists():
        return pd.DataFrame(columns=["season", "circuit", "laps", "n"])
    races = pd.read_parquet(tables / "races.parquet")
    races = races.assign(season=races.season.astype(int), round=races["round"].astype(int))
    start = pd.to_datetime(races.date + " " + races.time.fillna("12:00:00Z").str.rstrip("Z"), utc=True)
    races = races[(start + NOMINAL_DURATION["R"] + PUBLICATION_DELAY["R"] <= cutoff) & (races.season < before_season)]
    laps = pd.read_parquet(tables / "laps.parquet")
    laps = laps[laps.season.astype(int) < before_season]
    laps = laps.assign(season=laps.season.astype(int), round=laps["round"].astype(int), seconds=laps.time.map(_seconds))
    circuit = {(r.season, r.round): r.Circuit_circuitId for r in races.itertuples()}
    rows = []
    for (season, rnd), race in laps.groupby(["season", "round"]):
        if (season, rnd) not in circuit:
            continue
        times = {lap: g.seconds.to_numpy() for lap, g in race.groupby("lap")}
        rows.append((season, circuit[(season, rnd)], int(race.lap.max()), len(slow_lap_periods(times))))
    return pd.DataFrame(rows, columns=["season", "circuit", "laps", "n"])


def _seconds(text):
    total = 0.0
    for part in str(text).split(":"):
        total = 60 * total + float(part)
    return total


def eligible_laps(laps, periods):
    """Laps of a race on which a neutralisation could start: every lap but the
    continuation laps of one already out (the simulation draws a new one only then)."""
    covered = {lap for start, _, length in periods for lap in range(start + 1, start + length)}
    return set(range(1, int(laps) + 1)) - covered


class NeutralModel:
    """Per-lap hazard of a neutralisation: a base rate from the latest
    RECENT_SEASONS seasons with track status (laps 1-2, which hold about a third of
    safety cars, apart), counted over the laps on which one could start, times a
    circuit factor. The factor is the circuit's count
    against what its seasons' rates expect over all races known (track status from
    2018, lap times before), shrunk by a gamma prior whose spread is estimated from
    the same races (method of moments). Kind mix, lengths and the chance a red flag
    ends the race come from the races with track status."""

    def fit(self, history, recent):
        """`history`: DataFrame season, circuit, laps, n of every race; `recent`:
        [(season, laps, periods)] of races with track status."""
        self.factors, self.spread = {}, 0.0
        if not history.empty:
            rate = history.groupby("season").n.sum() / history.groupby("season").laps.sum()
            expected = history.laps * history.season.map(rate)
            by = pd.DataFrame({"obs": history.n, "exp": expected, "circuit": history.circuit}).groupby("circuit").sum()
            spread = float((((by.obs - by.exp) ** 2 - by.exp).sum()) / (by.exp ** 2).sum())
            self.spread = max(spread, 0.0)
            if self.spread > 0:
                k = 1.0 / self.spread
                self.factors = ((by.obs + k) / (by.exp + k)).to_dict()
        seasons = sorted({season for season, _, _ in recent})[-RECENT_SEASONS:]
        base = [(laps, periods) for season, laps, periods in recent if season in seasons]
        early = sum(1 for _, periods in base for start, _, _ in periods if start <= 2)
        later = sum(1 for _, periods in base for start, _, _ in periods if start > 2)
        open_early = sum(len(eligible_laps(laps, periods) & {1, 2}) for laps, periods in base)
        open_later = sum(len(eligible_laps(laps, periods) - {1, 2}) for laps, periods in base)
        self.early = early / max(open_early, 1)
        self.later = later / max(open_later, 1)
        every = [p for _, _, periods in recent for p in periods]
        counts = np.array([sum(1 for _, kind, _ in every if kind == k) for k in KINDS], dtype=float)
        self.kinds = counts / counts.sum() if counts.sum() else np.array([1.0, 0.0, 0.0])
        self.lengths = {k: np.array([length for _, kind, length in every if kind == k] or [1]) for k in KINDS}
        ended = sum(1 for _, laps, periods in recent for start, kind, length in periods
                    if kind == "red" and start + length - 1 >= laps)
        self.red_end = ended / counts[2] if counts[2] else 0.0
        return self

    def race(self, circuit, laps):
        """The neutralisations of one race of `laps` laps at `circuit`."""
        factor = self.factors.get(circuit, 1.0)
        hazard = np.full(int(laps), self.later * factor)
        hazard[:2] = self.early * factor
        return {"hazard": np.clip(hazard, 0.0, 0.9), "kinds": self.kinds, "lengths": self.lengths,
                "red_end": self.red_end, "factor": factor}
