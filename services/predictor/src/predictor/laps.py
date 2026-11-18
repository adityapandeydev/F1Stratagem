"""Race pace corrected lap by lap: tyres, tyre age and fuel.

A dry race's clean laps (green track, no pit in or out lap, not lap 1, not deleted
or inaccurate, within 107% of the session's median) are fitted together:

    100 log(lap time) - FUEL x laps run = driver pace + compound offset
                                          + wear(compound) x tyre age + noise

Fuel burns at the same rate for every car; within a stint tyre age and laps run
rise together, so a race alone cannot tell fuel from wear and a fitted fuel slope
moved driver pace by up to 0.3%. Fuel is therefore a fixed physical constant
(about 0.06 s per lap at 90 s), which also cannot leak. Compound offsets and wear
are estimated per race; a compound with too few laps or tyre ages shares one wear
rate with the other such compounds. Split strategies need no special handling.
Laps far from the fit (traffic, mistakes) are trimmed on a robust scale and the
fit repeated on the final set.

A race is not used as dry pace when the track was wet for part of it (wet tyres on
more than WET_SHARE of timed laps, or fewer than CLEAN_SHARE of them clean: the slick
laps of a drying track are not dry pace), or when the fit cannot tell drivers apart.
A driver needs MIN_LAPS clean laps.

A wet or mixed race is read with `drying=True`. Tyres fall in three classes (slicks,
intermediates, full wets), and the gap between classes swings by seconds a lap
around a crossover: before it the old class is seconds faster, after it the new one,
and a lap late costs more than a lap early (slicks lose more on a wet track than
intermediates on a dry one). No shape is assumed: every lap has an effect shared by
the field (the track, fuel, a slow lap for all) and, on a lap where two classes run
together, each class but the lap's most used has its own effect for that lap:

    100 log(lap time) = driver pace + lap effect + class gap on that lap
                        + compound offset within its class + wear(compound) x tyre age + noise

A car on the other class at a crossover lap is thus compared only through that
lap's class gap, never read as slow or fast driving. Laps are kept within 107% of
the median of their lap and tyre class. "Lap" here is race time (the leader's lap in
progress at the middle of the car's lap, so a lapped car meets the track the leader
did), while fuel follows the car's own laps.

The result per driver is the corrected race pace (log time x 100, relative: only
differences between drivers carry meaning) and its standard error relative to the
field mean.
"""

import numpy as np
import pandas as pd

FUEL = -0.065                  # log time x 100 per lap run (fuel burn), the same for every car
SLOW_CUTOFF = 1.07
TRIM_SIGMA = 3.0
TRIM_ROUNDS = 3
MIN_LAPS = 20                  # a driver needs this many clean laps to get a pace
MIN_WEAR_LAPS = 30             # a compound needs this many laps (and two tyre ages) for its own wear rate
WET_SHARE = 0.05
CLEAN_SHARE = 0.5
DRY = ("SOFT", "MEDIUM", "HARD", "HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SUPERHARD")
WET = ("INTERMEDIATE", "WET")
EMPTY = ["driver_number", "pace", "se", "laps"]


def wet_session(laps):
    """Whether a session ran in wet or mixed conditions: wet tyres on more than
    WET_SHARE of its timed laps (False without timed laps)."""
    if "LapTime" not in laps or "Compound" not in laps:
        return False
    timed = laps[laps.LapTime.notna()]
    if timed.empty:
        return False
    return float(timed.Compound.fillna("").astype(str).str.upper().isin(WET).mean()) > WET_SHARE


def wet_race(laps):
    """Whether the track was wet for part of the race: wet tyres on more than
    WET_SHARE of its timed laps."""
    timed = laps[laps.LapTime.notna()]
    if timed.empty:
        return True
    return float(timed.Compound.fillna("").astype(str).str.upper().isin(WET).mean()) > WET_SHARE


def clean_race_laps(laps, drying=False):
    """Laps of a race usable for pace: green, timed, not lap 1, not an in or out lap,
    accurate, not deleted, dry tyres with known age, within 107% of the median. With
    `drying`, wet tyres too and within 107% of the median of their lap and tyre class."""
    frame = laps[laps.LapTime.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna() & (laps.LapNumber > 1)]
    if "TrackStatus" in frame:
        frame = frame[frame.TrackStatus.fillna("").astype(str) == "1"]
    if "IsAccurate" in frame:
        frame = frame[frame.IsAccurate.fillna(False).astype(bool)]
    if "Deleted" in frame:
        frame = frame[~frame.Deleted.fillna(False).astype(bool)]
    compound = frame.Compound.fillna("").astype(str).str.upper()
    frame = frame[compound.isin(DRY + WET if drying else DRY) & frame.TyreLife.notna()].copy()
    if frame.empty:
        return frame.assign(seconds=pd.Series(dtype=float), driver_number=pd.Series(dtype=str),
                            compound=pd.Series(dtype=str))
    frame["seconds"] = frame.LapTime.dt.total_seconds()
    if drying:
        frame["track_lap"] = track_laps(laps, frame)
        # a car on the other tyre class at a crossover may be seconds off: its own median
        tyres = frame.Compound.astype(str).str.upper().map(tyre_class)
        median = frame.groupby([frame.track_lap, tyres]).seconds.transform("median")
    else:
        median = frame.seconds.median()
    frame = frame[frame.seconds <= median * SLOW_CUTOFF]
    return frame.assign(driver_number=frame.DriverNumber.astype(str),
                        compound=frame.Compound.astype(str).str.upper())


def race_pace(laps, drying=False):
    """Corrected pace per driver (see the module docstring). DataFrame:
    driver_number, pace, se, laps; empty when the race is not used. `.attrs` holds
    the fitted corrections, the noise scale and lap counts. With `drying`, the pace
    of a wet or mixed race (empty for a dry one)."""
    empty = pd.DataFrame(columns=EMPTY)
    if wet_race(laps) != drying:
        return empty
    frame = clean_race_laps(laps, drying)
    if len(frame) < CLEAN_SHARE * laps.LapTime.notna().sum():
        return empty
    # trim, then drop drivers left with fewer than MIN_LAPS laps and refit the rest,
    # until every driver in the fit keeps MIN_LAPS laps
    while True:
        counts = frame.driver_number.value_counts()
        frame = frame[frame.driver_number.isin(counts[counts >= MIN_LAPS].index)]
        drivers = sorted(frame.driver_number.unique())
        if len(drivers) < 2:
            return empty
        y = 100 * np.log(frame.seconds.to_numpy()) - FUEL * frame.LapNumber.to_numpy(dtype=float)
        keep = np.ones(len(frame), dtype=bool)
        for _ in range(TRIM_ROUNDS):
            x, names = _design(frame[keep], drivers, drying=drying)
            beta, *_ = np.linalg.lstsq(x, y[keep], rcond=None)
            residual = y - _design(frame, drivers, names)[0] @ beta
            scale = 1.4826 * float(np.median(np.abs(residual[keep] - np.median(residual[keep]))))
            trimmed = np.abs(residual) <= TRIM_SIGMA * scale
            if (trimmed == keep).all() or trimmed.sum() <= len(names):
                break
            keep = trimmed
        kept = pd.Series(keep).groupby(frame.driver_number.to_numpy()).sum()
        if (kept >= MIN_LAPS).all():
            break
        frame = frame[frame.driver_number.isin(kept[kept >= MIN_LAPS].index)]
    x, names = _design(frame[keep], drivers, drying=drying)     # the final fit, on the final lap set
    n = len(drivers)
    rank = np.linalg.matrix_rank(x)
    centre = np.eye(n) - np.ones((n, n)) / n                    # relative to the field mean
    if drying:
        # a compound the whole field ran at the same time cannot be told from those laps'
        # effects, nor the field's level from the lap effects: only each driver against
        # the field must be estimable (in the row space of the design)
        contrasts = np.hstack([centre, np.zeros((n, x.shape[1] - n))])
        separable = np.allclose(contrasts @ np.linalg.pinv(x) @ x, contrasts, atol=1e-8)
    else:
        separable = rank == x.shape[1]
    if keep.sum() <= rank or not separable:
        return empty                                            # drivers or tyres not separable
    beta, *_ = np.linalg.lstsq(x, y[keep], rcond=None)
    residual = y[keep] - x @ beta
    sigma = float(np.sqrt(residual @ residual / max(len(residual) - rank, 1)))
    # a generalised inverse: drivers against the field are estimable, so their covariance is exact
    cov = sigma ** 2 * (np.linalg.pinv(x.T @ x) if drying else np.linalg.inv(x.T @ x))
    result = pd.DataFrame({"driver_number": drivers, "pace": beta[:n],
                           "se": np.sqrt(np.diag(centre @ cov[:n, :n] @ centre.T)),
                           "laps": [int((keep & (frame.driver_number == d).to_numpy()).sum()) for d in drivers]})
    result.attrs = {"corrections": dict(zip(names[n:], beta[n:].tolist())), "noise": sigma,
                    "laps_used": int(keep.sum()), "laps_clean": len(frame),
                    "covariance": centre @ cov[:n, :n] @ centre.T}     # of the paces, drivers fitted together
    return result


def track_laps(laps, frame):
    """The leader's lap in progress at the middle of each lap of `frame`: track
    conditions follow race time, and a lapped car runs its lap 30 on the track of the
    leader's lap 31. Without session times, the driver's own lap number."""
    columns = ("Time", "LapStartTime")
    if any(c not in laps or c not in frame for c in columns) or frame.LapStartTime.isna().any():
        return frame.LapNumber.to_numpy(dtype=float)
    # when the first car completed each lap number (every timed lap counts, not only clean ones)
    ends = laps[laps.Time.notna()].groupby("LapNumber").Time.min().sort_index()
    ends = np.maximum.accumulate(ends.dt.total_seconds().to_numpy())
    middle = (frame.LapStartTime + frame.LapTime / 2).dt.total_seconds().to_numpy()
    return (np.searchsorted(ends, middle, side="left") + 1).astype(float)


def tyre_class(compound):
    """Slick ("dry"), "INTERMEDIATE" or "WET"."""
    return compound if compound in WET else "dry"


def _design(frame, drivers, names=None, drying=False):
    """Columns: one per driver (pace), compound offsets against the first compound,
    wear per compound with enough laps and ages, one shared wear for the rest. With
    `drying`, compound offsets only within a tyre class, an effect per lap number but
    the first, and a class gap per lap where two classes run (each class but the lap's
    most used). With `names`, the same columns as a previous call (for residuals of
    trimmed laps)."""
    number = frame.driver_number.to_numpy()
    compound = frame.compound.to_numpy()
    age = frame.TyreLife.to_numpy(dtype=float)
    # track effects and class gaps follow race time (the leader's lap), fuel the car's own laps
    lap = frame.track_lap.to_numpy(dtype=float) if "track_lap" in frame else \
        frame.LapNumber.to_numpy(dtype=float) if "LapNumber" in frame else np.zeros(len(frame))
    classes = np.array([tyre_class(c) for c in compound], dtype=object)
    if names is None:
        present = sorted(set(compound))
        own = [c for c in present if (compound == c).sum() >= MIN_WEAR_LAPS and len(set(age[compound == c])) >= 2]
        offsets = present[1:]
        if drying:              # across classes the lap effects and class gaps take the difference
            offsets = [c for c in present if c != min(p for p in present if tyre_class(p) == tyre_class(c))]
        names = [f"driver:{d}" for d in drivers] + [f"compound:{c}" for c in offsets] \
            + [f"wear:{c}" for c in own] + (["wear:shared"] if len(own) < len(present) else [])
        if drying:
            names += [f"lap:{int(n)}" for n in sorted(set(lap))[1:]]
            for n in sorted(set(lap)):
                counts = pd.Series(classes[lap == n]).value_counts()
                names += [f"gap:{c}@{int(n)}" for c in sorted(counts.index[1:])]
    own = {name[5:] for name in names if name.startswith("wear:") and name != "wear:shared"}
    columns = []
    for name in names:
        kind, value = name.split(":", 1)
        if kind == "driver":
            columns.append((number == value).astype(float))
        elif kind == "compound":
            columns.append((compound == value).astype(float))
        elif kind == "lap":
            columns.append((lap == float(value)).astype(float))
        elif kind == "gap":
            group, n = value.split("@")
            columns.append(((classes == group) & (lap == float(n))).astype(float))
        elif value == "shared":
            columns.append(~np.isin(compound, list(own)) * age)
        else:
            columns.append((compound == value) * age)
    return (np.column_stack(columns) if len(frame) else np.zeros((0, len(names)))), names


# Practice: teams alternate low-fuel push runs (qualifying practice) and long runs on
# race fuel; fuel loads are not published. Laps are classified within each stint:
REPRESENTATIVE = 1.03          # within 3% of the stint's best: not an out, in or cool-down lap
LONG_RUN_LAPS = 5              # a stint with this many representative laps is a long run
PUSH_TYRE_AGE = 4              # a push lap is on tyres at most this old
PUSH_SHARE = 1.01              # a push lap is within 1% of the driver's best on its compound
PUSH_FUEL = 1.02               # ... and within 2% of the driver's best of the session
MIN_LONG_RUN_LAPS = 5          # a driver needs this many long-run laps for a long-run pace


def practice_laps(laps):
    """A practice session's clean dry laps, each classified "push" (low-fuel run),
    "long" (long run) or None (neither)."""
    frame = laps[laps.LapTime.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna()]
    if "TrackStatus" in frame:
        frame = frame[frame.TrackStatus.fillna("").astype(str) == "1"]
    if "IsAccurate" in frame:
        frame = frame[frame.IsAccurate.fillna(False).astype(bool)]
    if "Deleted" in frame:
        frame = frame[~frame.Deleted.fillna(False).astype(bool)]
    compound = frame.Compound.fillna("").astype(str).str.upper()
    frame = frame[compound.isin(DRY) & frame.TyreLife.notna() & frame.Stint.notna()].copy()
    if frame.empty:
        return frame.assign(seconds=pd.Series(dtype=float), driver_number=pd.Series(dtype=str),
                            compound=pd.Series(dtype=str), run=pd.Series(dtype=object))
    frame["seconds"] = frame.LapTime.dt.total_seconds()
    frame["driver_number"] = frame.DriverNumber.astype(str)
    frame["compound"] = frame.Compound.astype(str).str.upper()
    stint = frame.groupby(["driver_number", "Stint"]).seconds
    frame = frame[frame.seconds <= stint.transform("min") * REPRESENTATIVE].copy()
    # a push lap: fresh tyres, within PUSH_SHARE of the driver's best on that compound
    # (often the start of a stint that becomes a long run); labelled first
    best = frame.groupby(["driver_number", "compound"]).seconds.transform("min")
    overall = frame.groupby("driver_number").seconds.transform("min")
    # ... and within PUSH_FUEL of the driver's best overall: race fuel costs 2-4%, a
    # compound 0.5-1%, so the first laps of a long run do not pass as push laps
    push = (frame.TyreLife <= PUSH_TYRE_AGE) & (frame.seconds <= best * PUSH_SHARE)         & (frame.seconds <= overall * PUSH_FUEL)
    frame["run"] = np.where(push, "push", None)
    # a long run: a stint with LONG_RUN_LAPS representative laps besides its push laps
    rest = (~push).groupby([frame.driver_number, frame.Stint]).transform("sum")
    frame.loc[~push & (rest >= LONG_RUN_LAPS), "run"] = "long"
    return frame


def push_pace(laps):
    """Low-fuel pace per driver from a practice session: each driver's best push lap
    on each compound, fitted as driver pace + compound offset, so a best lap on
    mediums counts like one on softs. Same columns as `race_pace`."""
    frame = practice_laps(laps)
    frame = frame[frame.run == "push"]
    if frame.empty:
        return pd.DataFrame(columns=EMPTY)
    best = frame.groupby(["driver_number", "compound"]).seconds.min().reset_index()
    # no trimming: most drivers have one row per compound, and trimming a driver's only
    # row removes the driver
    fitted = _fit(best, 100 * np.log(best.seconds.to_numpy()), wear=False, trim=False)
    if not fitted.empty:
        return fitted
    # compounds cannot be told from drivers (no driver pushed on two of them): compare
    # the drivers who pushed on the session's most used compound, as they are
    common = best.compound.value_counts().index[0]
    same = best[best.compound == common]
    if len(same) < 2:
        return pd.DataFrame(columns=EMPTY)
    pace = 100 * np.log(same.seconds.to_numpy())
    return pd.DataFrame({"driver_number": same.driver_number.to_numpy(), "pace": pace - pace.mean(),
                         "se": np.nan, "laps": 1})


def long_run_pace(laps):
    """Race-fuel pace per driver from a practice session's long runs: driver pace +
    compound offset + wear x tyre age (fuel burn within the run is in the wear
    term; each team's unknown fuel load is left to the model's per-team offset).
    Same columns as `race_pace`."""
    frame = practice_laps(laps)
    frame = frame[frame.run == "long"]
    counts = frame.driver_number.value_counts()
    frame = frame[frame.driver_number.isin(counts[counts >= MIN_LONG_RUN_LAPS].index)]
    # a compound only one driver ran cannot be told from that driver: leave it out
    shared = frame.groupby("compound").driver_number.nunique()
    frame = frame[frame.compound.isin(shared[shared >= 2].index)]
    if frame.empty:
        return pd.DataFrame(columns=EMPTY)
    return _fit(frame, 100 * np.log(frame.seconds.to_numpy()), wear=True)


def _fit(frame, y, wear, trim=True):
    """Least squares of `y` on driver + compound (+ wear) with robust trimming; the
    columns are fixed from the full set of laps. Pace per driver, se relative to the
    field mean."""
    drivers = sorted(frame.driver_number.unique())
    if len(drivers) < 2:
        return pd.DataFrame(columns=EMPTY)
    if "TyreLife" not in frame:
        frame = frame.assign(TyreLife=0.0)
    names = _design(frame, drivers)[1]
    if not wear:
        names = [n for n in names if not n.startswith("wear:")]
    x_all = _design(frame, drivers, names)[0]
    keep = np.ones(len(frame), dtype=bool)
    for _ in range(TRIM_ROUNDS if trim else 0):
        beta, *_ = np.linalg.lstsq(x_all[keep], y[keep], rcond=None)
        residual = y - x_all @ beta
        scale = 1.4826 * float(np.median(np.abs(residual[keep] - np.median(residual[keep]))))
        trimmed = np.abs(residual) <= TRIM_SIGMA * scale if scale > 0 else keep
        if (trimmed == keep).all() or trimmed.sum() < x_all.shape[1]:
            break
        keep = trimmed
    x = x_all[keep]
    if keep.sum() < x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        return pd.DataFrame(columns=EMPTY)
    beta, *_ = np.linalg.lstsq(x, y[keep], rcond=None)
    residual = y[keep] - x @ beta
    sigma = float(np.sqrt(residual @ residual / max(len(residual) - x.shape[1], 1)))
    n = len(drivers)
    cov = sigma ** 2 * np.linalg.pinv(x.T @ x)
    centre = np.eye(n) - np.ones((n, n)) / n
    result = pd.DataFrame({"driver_number": drivers, "pace": beta[:n],
                           "se": np.sqrt(np.clip(np.diag(centre @ cov[:n, :n] @ centre.T), 0, None)),
                           "laps": [int((keep & (frame.driver_number == d).to_numpy()).sum()) for d in drivers]})
    result.attrs = {"corrections": dict(zip(names[n:], beta[n:].tolist())), "noise": sigma, "laps_used": int(keep.sum())}
    return result


# Track evolution within a wet qualifying. On a drying (or worsening) track the lap that
# counts is worth more the later (or earlier) it was set: a driver who crossed the line
# last on a drying track beats a faster driver who ran earlier. `qualifying_trend` fits
# the track's condition through the session from every push lap, so each segment time
# can be read at a common track: driver + a smooth trend per tyre class (dry,
# intermediate, wet: their gap moves through the crossover) over session time, a
# slope that may bend every TREND_KNOT seconds with a ridge weight TREND_BEND on each
# bend. The trend runs on through breaks and red flags (the track keeps drying), and a
# class run only late is tied to the rest by its drivers' earlier laps. Laps more than
# TREND_SLOW slower than the fit (out, in and cool-down laps) leave the fit.
TREND_KNOT = 180.0
TREND_BEND = 1.0               # ridge weight on each bend (change of slope, per minute): with
                               # few push laps per driver a free bend trades against driver pace
TREND_SLOW = 1.5               # log time x 100 above the fitted lap
TREND_START = 2.0              # the first fit: laps within this of the driver's own best
TREND_ROUNDS = 4
TREND_CLASS_LAPS = 10          # a tyre class needs this many laps for its own trend
MATCH_SECONDS = 0.0015         # a segment time and its lap agree to the timing resolution
SEGMENTS_Q = ("Q1", "Q2", "Q3")


def _trend_columns(t, cls, classes):
    """(columns, bends): per tyre class a level (all but the first), a slope and a
    hinge every TREND_KNOT seconds within the class's laps; `bends` marks the hinges."""
    columns, bends = [], []
    start = t.min()
    for k, c in enumerate(classes):
        inside = cls == c
        since = (t - start) * inside
        if k:
            columns.append(inside.astype(float))
            bends.append(False)
        columns.append(since / 60.0)
        bends.append(False)
        first, last = since[inside].min(), since[inside].max()
        for knot in np.arange(TREND_KNOT, last, TREND_KNOT):
            if knot > first:
                columns.append(np.maximum(since - knot, 0.0) * inside / 60.0)
                bends.append(True)
    return np.column_stack(columns), np.array(bends, dtype=bool)


def qualifying_trend(laps, results):
    """{segment: {driver number: track condition at the time of his segment time, log
    time x 100, centred over the segment's drivers}} for one qualifying, from
    segments whose differences of condition the laps can estimate and whose every
    segment time is found among the fitted laps (a time left unread would compare one
    track with another). Subtracting it from each segment time reads the segment on one
    track. Empty without lap times."""
    columns = ("LapTime", "Time", "DriverNumber", "Compound")
    if any(c not in laps for c in columns) or any(s not in results for s in SEGMENTS_Q):
        return {}
    clean = laps[laps.LapTime.notna() & laps.Time.notna()]
    for flag in ("PitInTime", "PitOutTime"):
        if flag in clean:
            clean = clean[clean[flag].isna()]
    if len(clean) < 2 * TREND_CLASS_LAPS:
        return {}
    seconds = clean.LapTime.dt.total_seconds().to_numpy()
    y = 100 * np.log(seconds)
    t = clean.Time.dt.total_seconds().to_numpy() - seconds / 2
    numbers = clean.DriverNumber.astype(str).to_numpy()
    cls = clean.Compound.fillna("").astype(str).str.upper().map(tyre_class).to_numpy()
    counts = pd.Series(cls).value_counts()
    keep = (y <= y.min() + 100 * np.log(1.25)) & np.isin(cls, counts[counts >= TREND_CLASS_LAPS].index)
    y, t, numbers, cls, seconds = y[keep], t[keep], numbers[keep], cls[keep], seconds[keep]
    if len(y) < 2 * TREND_CLASS_LAPS:
        return {}
    classes = list(pd.Series(cls).value_counts().index)        # the most used class first
    drivers = sorted(set(numbers))
    trend, bends = _trend_columns(t, cls, classes)
    n = len(drivers)
    x = np.column_stack([(numbers == d).astype(float) for d in drivers] + [trend])
    penalty = np.sqrt(TREND_BEND) * np.eye(x.shape[1])[np.r_[np.zeros(n, dtype=bool), bends]]
    used = y <= pd.Series(y).groupby(numbers).transform("min").to_numpy() + TREND_START
    for _ in range(TREND_ROUNDS):
        beta = np.linalg.lstsq(np.vstack([x[used], penalty]), np.r_[y[used], np.zeros(len(penalty))],
                               rcond=None)[0]
        used = y - x @ beta < TREND_SLOW
    fit = np.vstack([x[used], penalty])                         # a bend without laps is held by its penalty
    projector = np.linalg.pinv(fit) @ fit
    out = {}
    for segment in SEGMENTS_Q:
        best = pd.to_timedelta(results[segment]).dt.total_seconds()
        rows, names = [], []
        for number, value in zip(results.DriverNumber.astype(str), best):
            if pd.isna(value):
                continue
            match = np.flatnonzero((numbers == number) & (np.abs(seconds - value) < MATCH_SECONDS))
            if len(match) != 1:
                rows = []
                break
            rows.append(match[0])
            names.append(number)
        if len(rows) < 2:
            continue
        contrasts = np.zeros((len(rows), x.shape[1]))
        contrasts[:, n:] = trend[rows] - trend[rows].mean(axis=0)
        if not np.allclose(contrasts @ projector, contrasts, atol=1e-6):
            continue                                            # condition differences not estimable
        out[segment] = dict(zip(names, (contrasts @ beta).tolist()))
    return out


def segment_wet_shares(laps, results):
    """{segment: share of its segment times set on wet-weather tyres} for one
    qualifying, over the times found among the laps on a known compound. A segment
    where fewer than half its times have a known compound is left out, so the caller
    keeps the session's own classification: a missing compound is no evidence of
    slicks. A wet qualifying often dries: Q1 on intermediates, Q3 on slicks."""
    if any(c not in laps for c in ("LapTime", "DriverNumber", "Compound")):
        return {}
    timed = laps[laps.LapTime.notna()]
    seconds = timed.LapTime.dt.total_seconds().to_numpy()
    numbers = timed.DriverNumber.astype(str).to_numpy()
    compound = timed.Compound.fillna("").astype(str).str.upper()
    wet = compound.isin(WET).to_numpy()
    known = compound.isin(DRY + WET).to_numpy()
    out = {}
    for segment in SEGMENTS_Q:
        if segment not in results:
            continue
        found, times = [], 0
        for number, value in zip(results.DriverNumber.astype(str), pd.to_timedelta(results[segment]).dt.total_seconds()):
            if pd.isna(value):
                continue
            times += 1
            match = np.flatnonzero((numbers == number) & (np.abs(seconds - value) < MATCH_SECONDS))
            if len(match) == 1 and known[match[0]]:
                found.append(bool(wet[match[0]]))
        if found and 2 * len(found) >= times:
            out[segment] = float(np.mean(found))
    return out


PASS_MIN_LAPS = 20   # racing laps a race needs for its pass rate


def race_passes(laps):
    """(passes, racing laps) of a dry race: pairs of cars that swapped order between
    two consecutive green laps on which neither car pitted, lap 1 excluded. A lap is
    green only if every car's track status on it is confirmed green ("1" only): a
    yellow flag, safety car, virtual safety car, red flag or a missing status rules
    it out. None for a wet or mixed race, a race without positions, or one with too
    few racing laps."""
    columns = ("Position", "LapNumber", "DriverNumber", "PitInTime", "PitOutTime", "TrackStatus")
    if any(c not in laps for c in columns) or wet_session(laps):
        return None
    laps = laps[laps.Position.notna()]
    if laps.empty:
        return None
    position = laps.pivot_table(index="LapNumber", columns="DriverNumber", values="Position")
    pitted = (laps.PitInTime.notna() | laps.PitOutTime.notna()).astype(float)
    pit = laps.assign(pit=pitted).pivot_table(index="LapNumber", columns="DriverNumber", values="pit", aggfunc="max")
    pit = pit.reindex(index=position.index, columns=position.columns).fillna(1.0) > 0
    green = laps.TrackStatus.astype(str).str.fullmatch("1+")
    neutral = (~green).groupby(laps.LapNumber).max()
    neutral = neutral.reindex(position.index).fillna(True).astype(bool)
    passes, counted = 0, 0
    numbers = list(position.index)
    for previous, lap in zip(numbers, numbers[1:]):
        if previous < 2 or lap != previous + 1 or neutral[lap] or neutral[previous]:
            continue
        clean = (~pit.loc[lap] & ~pit.loc[previous] & position.loc[lap].notna() & position.loc[previous].notna()).to_numpy()
        before, after = position.loc[previous].to_numpy()[clean], position.loc[lap].to_numpy()[clean]
        passes += int(((before[:, None] < before[None, :]) & (after[:, None] > after[None, :])).sum())
        counted += 1
    return (passes, counted) if counted >= PASS_MIN_LAPS else None
