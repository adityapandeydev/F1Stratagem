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

from .laps import DRY as DRY_COMPOUNDS, FUEL, clean_race_laps, practice_laps, race_pace, wet_session

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
PIT_LOSS, STOPS, WEAR = 23.0, 1.66, 0.05   # typical pit loss (s), stops per car, wear (% of a lap per lap of age)
SC_PIT_RATIO = 0.83     # loss of a stop under a safety car or virtual one, against a green-flag stop
STINT_NOISE = 3.0       # laps: spread of a planned stop around evenly spread stints (default; fitted in race.py)
STOP_WINDOW = 6         # laps: a stop due this soon is taken under a safety car
FIRST_STOP_SEASON = 2014     # Jolpica pit stops used from the hybrid era on
LAST_STOP_SEASON = 2024      # never later: 2025-2026 are the sealed holdout
STOP_BANDS = (8, 14)         # pace rank: front cars up to 8, middle up to 14, back after
TYRE_ERAS = (2022,)          # first seasons of new tyre eras (18-inch wheels from 2022)
MIN_ERA_RACES = 10           # fewer dry races in the era: tyre history from every era
MIN_WEAR_PAIRS = 15          # races with practice and race wear needed to calibrate weekend wear
CIRCUIT_PRIOR = 2.0
# Gap a held-up car keeps behind the car ahead (s, 5% quantiles; the first is the
# 0.35 s minimum). Measured 2018-2024 on 16,737 laps of runs of 5+ laps behind the
# same car that began within 1 s: median 0.98 s, a quarter beyond 1.75 s, a tenth
# beyond 3.2 s (followers drop back to save tyres and temperatures).
FOLLOW_GAPS = (0.35, 0.44, 0.53, 0.6, 0.65, 0.71, 0.76, 0.81, 0.86, 0.91, 0.98, 1.08, 1.21, 1.36, 1.54, 1.75,
               2.04, 2.5, 3.22, 4.88, 8.0)     # races: weight of the typical value in a circuit's pit loss, stops and wear
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


def simulate(pace, grid, passes, circuit_term, laps, lap_seconds, start_noise, rng, pass_shift=0.0, neutral=None,
             strategy=None, with_stops=False, trace=False, keyed=False, free_traffic=False, fixed_counts=None,
             stop_laps=None, follow=None):
    """Finishing order (draws x entrants, 1 = first) of `laps` laps, run natively
    (`predictor_race_native.simulate`, its own random numbers seeded from `rng`).

    `pace`: race pace draws (% of a lap, lower is faster); `grid`: slots, one row or
    one per draw; `passes`: a PassModel; `lap_seconds`: a typical lap; `pass_shift`:
    added to every pass's log-odds; `neutral`: `NeutralModel.race(...)` (None: no
    neutralisations); `strategy`: `StrategyModel.race(...)` (None: no stops or wear;
    with `deviate`, each car's own stop count by its pace rank). `with_stops`: also
    each car's pit stops per draw (a second array).

    Diagnostics (never used for forecasts): `trace` returns a dict of positions,
    stops and lap-by-lap arrays (draws x laps x entrants: time, tyre age, stop event
    0 none / 1 green / 2 neutralised / 3 called in for the race's stop / 4 new set
    under a red flag, time lost held up) and laps run per draw; `keyed`: random
    numbers keyed by draw, lap, car and purpose, so variants share their scenarios;
    `free_traffic`: nobody is held up or passes; `fixed_counts`: each car's planned
    stop count per draw (draws x entrants); `stop_laps`: each entrant's planned stop
    laps (entrants x k, 1-based, padded with 0). `follow`: the quantiles of the gap a
    held-up car keeps behind the car ahead (FOLLOW_GAPS; None: held at MIN_FOLLOW)."""
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
                                          float(neutral["red_end"]),
                                          *((float(strategy["wear"]), float(strategy["pit_loss"]), float(strategy["sc_ratio"]),
                                             float(strategy["stops"]), float(strategy.get("stint_noise", STINT_NOISE)),
                                             STOP_WINDOW,
                                             np.asarray(strategy["residuals"], dtype=float),
                                             car_deviations(strategy.get("deviate"), pace),
                                             np.asarray(strategy.get("team", np.zeros(0)), dtype=np.int64),
                                             np.asarray(strategy.get("team_mean", np.zeros(0)), dtype=float),
                                             np.asarray(strategy.get("team_sd", np.zeros(0)), dtype=float),
                                             np.asarray(strategy.get("pit_coef", np.zeros(0)), dtype=float),
                                             float(strategy.get("passing", 0.0)), float(lap_seconds))
                                            if strategy else (0.0, 0.0, 1.0, 1.0, 0.0, 0, np.zeros(0), np.zeros(0),
                                                              np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0),
                                                              np.zeros(0), 0.0, float(lap_seconds))), bool(with_stops),
                                          bool(trace), bool(keyed), bool(free_traffic),
                                          np.ascontiguousarray(np.zeros((0, 0)) if fixed_counts is None else fixed_counts,
                                                               dtype=np.int64),
                                          np.ascontiguousarray(np.zeros((0, 0)) if stop_laps is None else stop_laps,
                                                               dtype=np.int64),
                                          np.asarray(() if follow is None else follow, dtype=float))


def car_deviations(table, pace):
    """Each entrant's three chances of another stop count (flat, entrant-major) from
    the per-band `table` and its rank by mean race pace; empty without a table."""
    if table is None:
        return np.zeros(0)
    ranks = np.argsort(np.argsort(np.asarray(pace).mean(axis=0))) + 1
    return np.ascontiguousarray(np.asarray(table, dtype=float)[stop_band(ranks)]).ravel()


def stop_band(rank):
    """0 front, 1 middle, 2 back for a 1-based rank (pace or grid; 0 = pit lane)."""
    rank = np.asarray(rank)
    return np.where((rank <= 0) | (rank > STOP_BANDS[1]), 2, np.where(rank <= STOP_BANDS[0], 0, 1))


def history_stops(root, before_season):
    """Jolpica pit stops of the dry-looking races (1 or 2 stops most common among
    finishers) from FIRST_STOP_SEASON to before `before_season`, never after
    LAST_STOP_SEASON: dict residuals (each stop's pit-lane time against its race's
    median, s) and deviate (3 x 3: per grid band, the chance of one more stop in a
    one-stop race, one fewer and one more in a race of two or more); None without
    the tables."""
    tables = root / "external" / "jolpica" / "tables"
    if not (tables / "pitstops.parquet").exists() or not (tables / "results.parquet").exists():
        return None
    last = min(int(before_season) - 1, LAST_STOP_SEASON)
    stops = pd.read_parquet(tables / "pitstops.parquet", columns=["season", "round", "driverId", "duration"])
    stops = stops[stops.season.astype(int).between(FIRST_STOP_SEASON, last)]
    if stops.empty:
        return None
    stops = stops.assign(season=stops.season.astype(int), round=stops["round"].astype(int),
                         d=pd.to_numeric(stops.duration, errors="coerce"))     # m:ss = red-flag waits: dropped
    timed = stops.dropna(subset=["d"])
    residuals = (timed.d - timed.groupby(["season", "round"]).d.transform("median")).to_numpy(dtype=float)
    results = pd.read_parquet(tables / "results.parquet",
                              columns=["season", "round", "grid", "laps", "status", "Driver_driverId"])
    results = results.assign(season=results.season.astype(int), round=results["round"].astype(int),
                             laps=results.laps.astype(int), grid=results.grid.astype(int))
    results = results[results.season.between(FIRST_STOP_SEASON, last)]
    full = results.groupby(["season", "round"]).laps.transform("max")
    finished = results[(results.laps >= full - 2) & results.status.astype(str).str.match(r"^(Finished|\+)")]
    count = stops.groupby(["season", "round", "driverId"]).size().rename("stops")
    raced = set(zip(stops.season, stops["round"]))
    finished = finished[[k in raced for k in zip(finished.season, finished["round"])]]
    finished = finished.join(count, on=["season", "round", "Driver_driverId"]).fillna({"stops": 0})
    plan = finished.groupby(["season", "round"]).stops.transform(lambda s: s.mode().min())
    finished = finished.assign(plan=plan, band=stop_band(finished.grid))[plan.between(1, 2)]
    deviate = np.zeros((3, 3))
    for band in range(3):
        one = finished[(finished.band == band) & (finished.plan == 1)]
        two = finished[(finished.band == band) & (finished.plan == 2)]
        deviate[band] = [(one.stops > 1).mean() if len(one) else 0.0,
                         (two.stops < 2).mean() if len(two) else 0.0,
                         (two.stops > 2).mean() if len(two) else 0.0]
    return {"residuals": residuals, "deviate": deviate}


def stop_record(laps):
    """Pit stops of a dry race: dict green (each green-flag stop's loss against the
    cars not stopping on those laps, s), neutral (the same under a safety car or
    virtual one), stops_per_car (all stops over all cars), modal (the most common
    count among the cars running to the end, None without any); None for a wet race
    or one without laps."""
    if laps.empty or "PitInTime" not in laps or wet_session(laps):
        return None
    periods = neutral_periods(laps) or []
    neutral = {lap for start, _, length in periods for lap in range(start, start + length)}
    laps = laps.assign(t=laps.LapTime.dt.total_seconds(), stop=laps.PitInTime.notna(), out=laps.PitOutTime.notna())
    field = laps[~laps.stop & ~laps.out].groupby("LapNumber").t.median()
    by_car = {d: g.set_index("LapNumber").t for d, g in laps.groupby("DriverNumber")}
    green, under, stops = [], [], 0
    last = int(laps.LapNumber.max())
    for row in laps[laps.stop].itertuples():
        lap = int(row.LapNumber)
        if lap >= last:
            continue
        stops += 1
        out = by_car[row.DriverNumber].get(lap + 1, np.nan)
        if lap not in field.index or lap + 1 not in field.index or not np.isfinite(row.t) or not np.isfinite(out):
            continue
        loss = row.t + out - field[lap] - field[lap + 1]
        if np.isfinite(loss):
            (under if lap in neutral or lap + 1 in neutral else green).append(float(loss))
    running = laps.groupby("DriverNumber").LapNumber.max() >= last - 2
    counts = (laps[laps.stop & (laps.LapNumber < last)].groupby("DriverNumber").size()
              .reindex(running[running].index, fill_value=0))
    modal = int(counts.mode().min()) if len(counts) else None
    return {"green": green, "neutral": under, "stops_per_car": stops / max(laps.DriverNumber.nunique(), 1),
            "modal": modal}


def tyre_era(season):
    """0 for seasons before the first TYRE_ERAS boundary, 1 after it, ..."""
    return sum(int(season) >= first for first in TYRE_ERAS)


PRACTICE_STINT_LAPS, PRACTICE_STINT_AGES, PRACTICE_STINTS = 5, 4, 3


def practice_wear(laps):
    """Field tyre wear of a dry practice session: the median over its long-run
    stints (PRACTICE_STINT_LAPS laps, PRACTICE_STINT_AGES tyre ages) of the slope
    of 100 log lap time on tyre age, fuel burn added back (% of a lap per lap of
    age); (wear, stints), or None with fewer than PRACTICE_STINTS stints. Per stint,
    not per driver: most drivers run one compound, which leaves a driver + compound
    + wear fit unidentified (it fails in more than half of the sessions)."""
    if laps.empty or wet_session(laps):
        return None
    frame = practice_laps(laps)
    frame = frame[frame.run == "long"]
    slopes = [np.polyfit(g.TyreLife.astype(float), 100 * np.log(g.seconds.to_numpy()), 1)[0]
              for _, g in frame.groupby(["driver_number", "Stint"])
              if len(g) >= PRACTICE_STINT_LAPS and g.TyreLife.nunique() >= PRACTICE_STINT_AGES]
    return (float(np.median(slopes)) - FUEL, len(slopes)) if len(slopes) >= PRACTICE_STINTS else None


def shrunk_mean(values, typical, prior=CIRCUIT_PRIOR):
    """Mean of `values` shrunk towards `typical` by `prior` pseudo-values."""
    values = [v for v in values if v is not None and np.isfinite(v)]
    return (sum(values) + prior * typical) / (len(values) + prior)


class WeekendWear:
    """Race wear from this weekend's practice long-run wear and the circuit's past
    wear: least squares race = a + b practice + c past over earlier races (measured
    2018-2024 with `practice_wear`, 132 of 134 dry weekends: practice wear
    correlates 0.51 with race wear on 18-inch tyres, 0.38 on 13-inch)."""

    def fit(self, pairs):
        """`pairs`: [(practice, past, race)] of earlier races."""
        rows = np.array([p for p in pairs if all(v is not None and np.isfinite(v) for v in p)], dtype=float)
        self.coef = None
        if len(rows) >= MIN_WEAR_PAIRS:
            x = np.c_[np.ones(len(rows)), rows[:, 0], rows[:, 1]]
            self.coef, *_ = np.linalg.lstsq(x, rows[:, 2], rcond=None)
        return self

    def predict(self, practice, past):
        """Race wear (%/lap of age) or None without a calibration or practice."""
        if self.coef is None or practice is None or not np.isfinite(practice):
            return None
        return float(np.clip(self.coef @ [1.0, practice, past], 0.0, 0.3))


TRAFFIC_GAP = 1.0            # s to the car ahead at a lap's end: a lap in traffic
TEAM_STINT_LAPS = 6          # a stint needs this many clean laps (and 4 tyre ages) for team wear
TEAM_LAPS = 60               # a team needs this many such laps in a race for its wear deviation
TEAM_DECAYS = (1.0, 0.8, 0.6)   # recency weights per race back tried for team wear
MIN_TEAM_RACES = 20          # earlier races needed to fit the team wear model


def traffic_laps(laps):
    """(driver number, lap) of the laps run in traffic, from every car's line
    crossings in session-time order whatever its lap: crossing within TRAFFIC_GAP
    behind any car (a car lapping a backmarker is held up like one following a car
    on its own lap), or within TRAFFIC_GAP ahead of a car on a later lap (being
    lapped, under blue flags)."""
    ends = laps.dropna(subset=["Time"]).assign(end=lambda f: f.Time.dt.total_seconds()).sort_values("end")
    end, lap = ends.end.to_numpy(), ends.LapNumber.to_numpy()
    behind = np.r_[np.inf, np.diff(end)] < TRAFFIC_GAP
    lapped = np.r_[(np.diff(end) < TRAFFIC_GAP) & (lap[1:] > lap[:-1]), False]
    flag = behind | lapped
    return set(zip(ends.DriverNumber.to_numpy()[flag], ends.LapNumber.to_numpy()[flag]))


ESTIMABLE = 1e-6        # a contrast this close to the design's row space is estimable


def _team_fit(f, teams):
    """Least squares of the team model on `teams`: (deviations, their covariance,
    estimability error per team). A team's deviation (against the mean of the
    teams) is estimable when its contrast lies in the row space of the design: else
    it is confounded, e.g. with the wear of a compound only that team ran."""
    columns = [pd.get_dummies(f.LapNumber, prefix="lap", drop_first=True, dtype=float),
               pd.get_dummies(f.stint, prefix="st", dtype=float)]
    for c in sorted(f.compound.unique()):
        columns.append(pd.DataFrame({f"wear:{c}": (f.compound == c).to_numpy(float) * f.age.to_numpy()}, index=f.index))
    for t in teams[:-1]:          # effect coding: the last team is minus the sum of the others
        code = (f.lineage == t).astype(float) - (f.lineage == teams[-1]).astype(float)
        columns.append(pd.DataFrame({f"dev:{t}": code.to_numpy() * f.age.to_numpy()}, index=f.index))
    x = pd.concat(columns, axis=1)
    X, y = x.to_numpy(), f.y.to_numpy()
    pinv = np.linalg.pinv(X)
    beta = pinv @ y
    residual = y - X @ beta
    sigma2 = residual @ residual / max(len(y) - np.linalg.matrix_rank(X), 1)
    index = [list(x.columns).index(f"dev:{t}") for t in teams[:-1]]
    k = len(teams)
    a = np.vstack([np.eye(k - 1), -np.ones(k - 1)])
    contrasts = np.zeros((k, X.shape[1]))
    contrasts[:, index] = a
    error = np.linalg.norm(contrasts @ (pinv @ X) - contrasts, axis=1) / np.linalg.norm(contrasts, axis=1)
    cov = sigma2 * (contrasts @ (pinv @ pinv.T) @ contrasts.T)        # (X'X)^+ = X^+ X^+'
    return contrasts @ beta, cov, error


def team_wear_deviations(laps, lineage_of, min_laps=None):
    """Each team's tyre wear in a dry race against the field (% of a lap per lap of
    age) with its standard error: DataFrame lineage, dev, se, laps; empty without.

    100 log lap time = lap effect + stint effect + wear(compound) x age + team
    deviation x age + noise, on clean laps out of traffic (`traffic_laps`). The lap
    effect takes fuel burn and track evolution (shared by the field); the stint
    effect each driver's stint pace. The field's own wear is not identified this
    way (age is lap minus the stint's start), but a team's departure from it is,
    when the team shares compounds with others: a team whose deviation is not
    estimable is left out (the most confounded first) and the race is refitted;
    fewer than three teams left, no estimate. Deviations and their covariance are
    centred together on the precision-weighted field. `lineage_of`: driver number
    -> team (any grouping: driver -> driver gives driver deviations); `min_laps`:
    clean laps a group needs (TEAM_LAPS)."""
    empty = pd.DataFrame(columns=["lineage", "dev", "se", "laps"])
    if laps.empty or wet_session(laps):
        return empty
    traffic = traffic_laps(laps)
    f = clean_race_laps(laps)
    if f.empty:
        return empty
    f = f[[(d, n) not in traffic for d, n in zip(f.DriverNumber, f.LapNumber)]]
    f = f.assign(lineage=f.driver_number.map(lineage_of), stint=f.driver_number + "-" + f.Stint.astype(str),
                 age=f.TyreLife.astype(float), y=100 * np.log(f.seconds.to_numpy()))
    f = f[f.lineage.notna()]
    f = f.groupby("stint").filter(lambda g: len(g) >= TEAM_STINT_LAPS and g.age.nunique() >= 4)
    counts = f.lineage.value_counts()
    teams = sorted(counts[counts >= (TEAM_LAPS if min_laps is None else min_laps)].index)
    while len(teams) >= 3:
        here = f[f.lineage.isin(teams)]
        dev, cov, error = _team_fit(here, teams)
        if error.max() <= ESTIMABLE:
            break
        teams = [t for i, t in enumerate(teams) if i != int(np.argmax(error))]
    else:
        return empty
    w = 1 / np.diag(cov)
    centre = np.eye(len(teams)) - np.outer(np.ones(len(teams)), w / w.sum())
    dev, cov = centre @ dev, centre @ cov @ centre.T
    return pd.DataFrame({"lineage": teams, "dev": dev, "se": np.sqrt(np.clip(np.diag(cov), 0, None)),
                         "laps": [int(counts[t]) for t in teams]})


class TeamWear:
    """Each team's wear deviation from the field in its next race, as a
    distribution: a pool of its earlier races in the tyre era (recency weight per
    race back), shrunk towards the field by the persistent spread between teams,
    plus the race-to-race spread a team shows beyond it.

    Measured 2018-2024 (133 dry races): persistent spread 0.010% of a lap per lap
    of age, race-specific 0.020, estimation error 0.010; earlier races predict the
    next race's deviation beyond noise (correlation 0.24). The recency weight is
    chosen per season by one-step predictive likelihood over the earlier races."""

    def __init__(self):
        self.persistent = self.specific = 0.0
        self.decay = None

    def fit(self, races):
        """`races`: [deviations from `team_wear_deviations`] of the earlier dry races
        of the tyre era, in order."""
        self.persistent = self.specific = 0.0
        self.decay = None
        parts = [d.assign(order=i) for i, d in enumerate(races) if not d.empty]
        if len(parts) < MIN_TEAM_RACES:
            return self
        frame = pd.concat(parts, ignore_index=True)
        nxt = frame.groupby("lineage").dev.shift(-1)
        pairs = frame.assign(nxt=nxt).dropna(subset=["nxt"])
        self.persistent = max(float(np.cov(pairs.dev, pairs.nxt)[0, 1]), 0.0)
        self.specific = max(float(frame.dev.var() - (frame.se ** 2).mean()) - self.persistent, 1e-6)
        if self.persistent <= 0:
            return self
        best = None
        orders = sorted(frame.order.unique())
        for decay in TEAM_DECAYS:
            self.decay, total = decay, 0.0
            for order in orders[MIN_TEAM_RACES // 2:]:
                prior, now = frame[frame.order < order], frame[frame.order == order]
                for r in now.itertuples():
                    mean, var = self._pool(prior[prior.lineage == r.lineage])
                    s2 = r.se ** 2 + self.specific + var
                    total += -0.5 * (np.log(s2) + (r.dev - mean) ** 2 / s2)
            if best is None or total > best[0]:
                best = (total, decay)
        self.decay = best[1]
        return self

    def _pool(self, earlier):
        """(mean, variance) of a team's persistent deviation from its earlier races
        (in order); the field's (0, persistent spread) without any."""
        if earlier.empty:
            return 0.0, self.persistent
        weights = self.decay ** np.arange(len(earlier))[::-1] / (earlier.se.to_numpy() ** 2 + self.specific)
        mean = float(np.sum(weights * earlier.dev.to_numpy()) / weights.sum())
        shrink = self.persistent / (self.persistent + 1 / weights.sum())
        return shrink * mean, self.persistent * (1 - shrink)

    def predict(self, lineages, races):
        """(team index per entrant, mean and spread per team of the wear deviation
        in % of a lap per lap of age) for the next race; `races`: the earlier
        races' deviations in order. None while the model is not fitted."""
        if self.decay is None:
            return None
        parts = [d for d in races if not d.empty]
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["lineage", "dev", "se"])
        teams = sorted({l for l in lineages if l is not None})
        means, sds = [], []
        for team in teams:
            mean, var = self._pool(frame[frame.lineage == team])
            means.append(mean)
            sds.append(float(np.sqrt(var + self.specific)))
        if any(l is None for l in lineages):          # unknown team: the field's spread
            means.append(0.0)
            sds.append(float(np.sqrt(self.persistent + self.specific)))
        index = [teams.index(l) if l is not None else len(teams) for l in lineages]
        return index, np.array(means), np.array(sds)


AGE_EDGES = (5, 10, 15, 20, 25, 30, 40)    # tyre-age bands of the pit hazard (laps)
PIT_FEATURES = tuple(f"age>{e}" for e in AGE_EDGES) + (
    "first_stint", "frac", "frac2", "end", "neutral", "neutral_old", "position", "lost", "clean_air",
    "gap_ahead", "gap_behind", "cover_behind", "cover_ahead", "pace_adv", "passing", "pace_x_passing",
    "lost_x_passing", "tyre_loss", "tyre_loss2", "wear")
PIT_FIRST_LAP, PIT_LAST_LAPS = 2, 3        # the hazard acts from lap 2 to 3 laps from the end
END_LAPS, OLD_TYRES, COVER_GAP, GAP_CAP = 8, 10, 3.0, 20.0
MIN_PIT_RACES = 15                          # dry races of the tyre era needed for an era-only fit


def pit_features(age, stops, lap, laps, neutral, position, lost, clean_air, gap_ahead, gap_behind,
                 cover_behind, cover_ahead, pace_adv, passing, wear):
    """The pit hazard's features (PIT_FEATURES order) for arrays of car-laps; the
    native simulator computes the same ones (`predictor_race_native`, keep in step).
    Gaps in s, `lost`: cars that would pass during the stop, `pace_adv`: the pace of
    the car it would rejoin behind minus its own (% of a lap; + = faster than it),
    `passing`: the circuit's centred log pass rate, `wear`: the car's tyre wear (% of
    a lap per lap of age; `tyre_loss` = wear x age, what the tyres cost now)."""
    age, lap = np.asarray(age, dtype=float), np.asarray(lap, dtype=float)
    frac = lap / laps
    lost = np.asarray(lost, dtype=float)
    adv = np.clip(np.asarray(pace_adv, dtype=float), -3.0, 3.0)
    neutral = np.asarray(neutral, dtype=float)
    columns = [(age > e).astype(float) - (age > nxt).astype(float) if nxt is not None else (age > e).astype(float)
               for e, nxt in zip(AGE_EDGES, AGE_EDGES[1:] + (None,))]
    columns += [(np.asarray(stops) == 0).astype(float), frac, frac ** 2, (laps - lap <= END_LAPS).astype(float),
                neutral, neutral * (age >= OLD_TYRES), np.asarray(position, dtype=float) / 20, lost / 10,
                np.log1p(np.minimum(clean_air, GAP_CAP)), np.log1p(np.minimum(gap_ahead, GAP_CAP)),
                np.log1p(np.minimum(gap_behind, GAP_CAP)), np.asarray(cover_behind, dtype=float),
                np.asarray(cover_ahead, dtype=float), adv, np.full(len(age), passing), adv * passing,
                lost / 10 * passing]
    wear = np.broadcast_to(np.clip(np.asarray(wear, dtype=float), 0.0, 0.3), age.shape)
    columns += [wear * age, (wear * age) ** 2, wear]
    return np.column_stack(columns)


def pit_decisions(laps, loss, pace, wear=None, sc_ratio=SC_PIT_RATIO):
    """Every car's racing laps of a dry race, each with whether the car pitted at
    the lap's end and the state it was decided from: DataFrame of `pit_features`
    inputs plus `pit`; empty for a wet race. The state is the one the simulator
    decides from (`predictor_race_native`): the field as it crossed the line at the
    end of the lap before (no consequence of this lap's stop can enter it), the
    car's tyre age on this lap (TyreLife), whether this lap is neutralised (then
    places lost and clean air use `sc_ratio` x `loss`), and who pitted at the end of
    the lap before. The simulator charges a stop in full at the end of its in-lap,
    while timing splits it between the in-lap and the out-lap: a car that pitted at
    the end of the lap before is placed at its out-lap crossing less that lap's
    typical lap (the median of the cars neither entering nor leaving the pits), its
    stop then wholly behind it as in the simulator. `loss`: the race's green pit loss (s); `pace`: driver -> race
    pace (%); `wear`: the race's tyre wear (%/lap of age; WEAR without)."""
    if laps.empty or wet_session(laps) or "PitInTime" not in laps:
        return pd.DataFrame()
    total = int(laps.LapNumber.max())
    f = laps.dropna(subset=["Time"]).assign(end=lambda x: x.Time.dt.total_seconds(), d=lambda x: x.DriverNumber.astype(str))
    pitted = set(zip(f.d[f.PitInTime.notna()], f.LapNumber[f.PitInTime.notna()]))
    status = laps.groupby("LapNumber").TrackStatus.agg(lambda v: "".join(str(x) for x in v.dropna()))
    neutral = {n for n, v in status.items() if any(c in v for c in "4567")}
    by_lap = {n: g for n, g in f.groupby("LapNumber")}
    rows = []
    for lap in range(PIT_FIRST_LAP, total - PIT_LAST_LAPS + 1):
        if lap - 1 not in by_lap or lap not in by_lap:
            continue
        field = by_lap[lap - 1].copy()
        stopped = field.PitInTime.notna()
        if stopped.any():
            out = by_lap[lap].set_index("d")
            clean = out[out.PitInTime.isna() & out.PitOutTime.isna()].LapTime.dt.total_seconds()
            typical = float(clean.median()) if len(clean) else np.nan
            for i in field.index[stopped]:
                d = field.at[i, "d"]
                if d in out.index and np.isfinite(typical):
                    crossing = out.loc[d, "end"]
                    field.at[i, "end"] = float(crossing.iloc[0] if isinstance(crossing, pd.Series) else crossing) - typical
        field = field.sort_values("end")
        ends, drivers = field.end.to_numpy(), field.d.to_numpy()
        now = by_lap[lap].set_index("d")
        expected = loss * (sc_ratio if lap in neutral else 1.0)
        for i, d in enumerate(drivers):
            if d not in now.index:
                continue
            row = now.loc[d]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if pd.isna(row.TyreLife) or str(row.Compound).upper() not in DRY_COMPOUNDS:
                continue
            t, rejoin = ends[i], ends[i] + expected
            others = np.arange(len(ends)) != i
            passing_cars = others & (ends > t) & (ends <= rejoin)
            before = ends[others & (ends <= rejoin)]
            after = ends[others & (ends > rejoin)]
            target = drivers[passing_cars][-1] if passing_cars.any() else None
            rows.append({"driver": d, "lap": lap, "age": float(row.TyreLife),
                         "stops": max(int(row.Stint) - 1, 0) if pd.notna(row.Stint) else 0,
                         "neutral": lap in neutral, "position": i + 1, "lost": int(passing_cars.sum()),
                         "clean_air": min(rejoin - before.max() if len(before) else GAP_CAP,
                                          after.min() - rejoin if len(after) else GAP_CAP),
                         "gap_ahead": t - ends[i - 1] if i else GAP_CAP,
                         "gap_behind": ends[i + 1] - t if i + 1 < len(ends) else GAP_CAP,
                         "cover_behind": i + 1 < len(ends) and (drivers[i + 1], lap - 1) in pitted and ends[i + 1] - t < COVER_GAP,
                         "cover_ahead": i > 0 and (drivers[i - 1], lap - 1) in pitted and t - ends[i - 1] < COVER_GAP,
                         "pace_adv": (pace.get(target, np.nan) - pace.get(d, np.nan)) if target else 0.0,
                         "pit": (d, lap) in pitted})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pace_adv"] = out.pace_adv.fillna(0.0)
        out["laps"] = total
        out["wear"] = WEAR if wear is None or not np.isfinite(wear) else float(wear)
    return out


class PitModel:
    """Per-lap pit hazard: logistic regression of a car pitting at the end of a lap
    on `pit_features`, over the earlier dry races of the tyre era (every era while
    it has fewer than MIN_PIT_RACES). Measured 2018-2024 (137,157 car-laps, 4,327
    stops): fitted within the era it is calibrated on later seasons (riskiest tenth
    of laps 0.138 predicted, 0.131 observed; pooled across eras 0.097); clean air,
    covering a rival who pitted and pace with circuit passing each predict held-out
    decisions beyond noise. Coefficients are on the raw features (intercept first)."""

    def __init__(self):
        self.coef = None
        self.centre = 0.0          # mean log pass rate of the training races

    def fit(self, races):
        """`races`: [(decisions from `pit_decisions`, circuit log pass rate)]."""
        from sklearn.linear_model import LogisticRegression
        frames = [d.assign(passing=rate) for d, rate in races if not d.empty and np.isfinite(rate)]
        self.coef = None
        if len(frames) < MIN_PIT_RACES // 3:
            return self
        f = pd.concat(frames, ignore_index=True)
        self.centre = float(f.passing.mean())
        x = pit_features(f.age, f.stops, f.lap, f.laps.to_numpy(float), f.neutral, f.position, f.lost, f.clean_air,
                         f.gap_ahead, f.gap_behind, f.cover_behind, f.cover_ahead, f.pace_adv, 0.0, f.wear)
        x[:, PIT_FEATURES.index("passing")] = f.passing - self.centre
        x[:, PIT_FEATURES.index("pace_x_passing")] = x[:, PIT_FEATURES.index("pace_adv")] * (f.passing - self.centre)
        x[:, PIT_FEATURES.index("lost_x_passing")] = x[:, PIT_FEATURES.index("lost")] * (f.passing - self.centre)
        mu, sd = x.mean(axis=0), x.std(axis=0)
        sd[sd == 0] = 1.0
        model = LogisticRegression(C=1.0, max_iter=3000).fit((x - mu) / sd, f.pit.to_numpy())
        raw = model.coef_[0] / sd
        self.coef = np.r_[model.intercept_[0] - raw @ mu, raw]
        return self

    def probability(self, features):
        return 1 / (1 + np.exp(-(self.coef[0] + features @ self.coef[1:])))


def log_pass_rate(laps):
    """log(passes per racing lap + 0.001) of a dry race; None without a count."""
    from .laps import race_passes
    counted = race_passes(laps)
    if not counted or not counted[1]:
        return None
    return float(np.log(counted[0] / counted[1] + 1e-3))


def race_wear(laps):
    """Mean tyre wear of a dry race over its compounds (% of a lap per lap of
    age, from `race_pace`); None without a fit."""
    pace = race_pace(laps)
    if pace.empty:
        return None
    wear = [v for k, v in pace.attrs["corrections"].items() if k.startswith("wear:") and k != "wear:shared"]
    return float(np.mean(wear)) if wear else None


class StrategyModel:
    """Pit loss, stops per car and tyre wear per circuit: the mean over earlier races
    at the circuit, shrunk towards the typical race by CIRCUIT_PRIOR races
    (measured 2018-2024: a circuit's pit loss repeats at the next visit with
    correlation 0.54, its stops 0.39, its hard-tyre wear 0.60). Slow stops are the
    residuals of green-flag stops against their race's median; a stop under a safety
    car or virtual one costs the observed ratio of the two losses."""

    def fit(self, races, stops=None):
        """`races`: [(circuit, record from `stop_record`, wear)] of earlier dry races;
        `stops`: `history_stops` (pit-lane slow stops and per-car stop counts), or
        None for slow stops from the races' own losses and the plan for every car."""
        records = [(c, r, w) for c, r, w in races if r is not None]
        green = [np.median(r["green"]) for _, r, _ in records if len(r["green"]) >= 4]
        self.pit_loss = float(np.median(green)) if green else PIT_LOSS
        # with per-car deviations the shared plan is the race's most common count (the
        # deviations were measured against it); else the mean, which already holds them
        planned = stops is not None and stops.get("deviate") is not None
        key = "modal" if planned else "stops_per_car"
        records = [(c, {**r, "plan": r.get(key) if r.get(key) is not None else r["stops_per_car"]}, w)
                   for c, r, w in records]
        self.stops = float(np.mean([r["plan"] for _, r, _ in records])) if records else (1.0 if planned else STOPS)
        wears = [w for _, _, w in records if w is not None and np.isfinite(w)]
        self.wear = float(np.median(wears)) if wears else WEAR
        residuals = [x - np.median(r["green"]) for _, r, _ in records if len(r["green"]) >= 4 for x in r["green"]]
        self.residuals = np.array(residuals) if residuals else np.zeros(1)
        self.deviate = None
        if stops is not None and stops.get("residuals") is not None and len(stops["residuals"]):
            self.residuals = np.asarray(stops["residuals"], dtype=float)
        if stops is not None and stops.get("deviate") is not None:
            self.deviate = np.asarray(stops["deviate"], dtype=float)
        under = [x for _, r, _ in records for x in r["neutral"]]
        all_green = [x for _, r, _ in records for x in r["green"]]
        self.sc_ratio = float(np.median(under) / np.median(all_green)) if under and all_green else SC_PIT_RATIO
        self.circuits = {}
        for circuit in {c for c, _, _ in records}:
            here = [(r, w) for c, r, w in records if c == circuit]
            losses = [np.median(r["green"]) for r, _ in here if len(r["green"]) >= 4]
            wear = [w for _, w in here if w is not None and np.isfinite(w)]
            self.circuits[circuit] = {
                "pit_loss": (sum(losses) + CIRCUIT_PRIOR * self.pit_loss) / (len(losses) + CIRCUIT_PRIOR),
                "stops": (sum(r["plan"] for r, _ in here) + CIRCUIT_PRIOR * self.stops) / (len(here) + CIRCUIT_PRIOR),
                "wear": (sum(wear) + CIRCUIT_PRIOR * self.wear) / (len(wear) + CIRCUIT_PRIOR)}
        return self

    def race(self, circuit, lap_seconds):
        """Strategy inputs of one race at `circuit` (wear in s per lap of age);
        `deviate`: the per-band chances of another stop count, if fitted."""
        here = self.circuits.get(circuit, {"pit_loss": self.pit_loss, "stops": self.stops, "wear": self.wear})
        out = {"pit_loss": here["pit_loss"], "stops": max(here["stops"], 1.0), "wear": here["wear"] * lap_seconds / 100,
               "sc_ratio": self.sc_ratio, "residuals": self.residuals}
        if self.deviate is not None:
            out["deviate"] = self.deviate
        return out


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
