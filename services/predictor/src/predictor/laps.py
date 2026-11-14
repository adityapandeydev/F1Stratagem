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

A race is not used when the track was wet for part of it (wet tyres on more than
WET_SHARE of timed laps, or fewer than CLEAN_SHARE of them clean: the slick laps of
a drying track are not dry pace), or when the fit cannot tell drivers apart. A
driver needs MIN_LAPS clean laps.

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


def wet_race(laps):
    """Whether the track was wet for part of the race: wet tyres on more than
    WET_SHARE of its timed laps."""
    timed = laps[laps.LapTime.notna()]
    if timed.empty:
        return True
    return float(timed.Compound.fillna("").astype(str).str.upper().isin(WET).mean()) > WET_SHARE


def clean_race_laps(laps):
    """Laps of a race usable for pace: green, timed, not lap 1, not an in or out lap,
    accurate, not deleted, dry tyres with known age, within 107% of the median."""
    frame = laps[laps.LapTime.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna() & (laps.LapNumber > 1)]
    if "TrackStatus" in frame:
        frame = frame[frame.TrackStatus.fillna("").astype(str) == "1"]
    if "IsAccurate" in frame:
        frame = frame[frame.IsAccurate.fillna(False).astype(bool)]
    if "Deleted" in frame:
        frame = frame[~frame.Deleted.fillna(False).astype(bool)]
    compound = frame.Compound.fillna("").astype(str).str.upper()
    frame = frame[compound.isin(DRY) & frame.TyreLife.notna()].copy()
    if frame.empty:
        return frame.assign(seconds=pd.Series(dtype=float), driver_number=pd.Series(dtype=str),
                            compound=pd.Series(dtype=str))
    frame["seconds"] = frame.LapTime.dt.total_seconds()
    frame = frame[frame.seconds <= frame.seconds.median() * SLOW_CUTOFF]
    return frame.assign(driver_number=frame.DriverNumber.astype(str),
                        compound=frame.Compound.astype(str).str.upper())


def race_pace(laps):
    """Corrected pace per driver (see the module docstring). DataFrame:
    driver_number, pace, se, laps; empty when the race is not used. `.attrs` holds
    the fitted corrections, the noise scale and lap counts."""
    empty = pd.DataFrame(columns=EMPTY)
    if wet_race(laps):
        return empty
    frame = clean_race_laps(laps)
    if len(frame) < CLEAN_SHARE * laps.LapTime.notna().sum():
        return empty
    counts = frame.driver_number.value_counts()
    frame = frame[frame.driver_number.isin(counts[counts >= MIN_LAPS].index)]
    drivers = sorted(frame.driver_number.unique())
    if len(drivers) < 2:
        return empty
    y = 100 * np.log(frame.seconds.to_numpy()) - FUEL * frame.LapNumber.to_numpy(dtype=float)
    keep = np.ones(len(frame), dtype=bool)
    for _ in range(TRIM_ROUNDS):
        x, names = _design(frame[keep], drivers)
        beta, *_ = np.linalg.lstsq(x, y[keep], rcond=None)
        residual = y - _design(frame, drivers, names)[0] @ beta
        scale = 1.4826 * float(np.median(np.abs(residual[keep] - np.median(residual[keep]))))
        trimmed = np.abs(residual) <= TRIM_SIGMA * scale
        if (trimmed == keep).all() or trimmed.sum() <= len(names):
            break
        keep = trimmed
    x, names = _design(frame[keep], drivers)                    # the final fit, on the final lap set
    if keep.sum() <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        return empty                                            # drivers or tyres not separable
    beta, *_ = np.linalg.lstsq(x, y[keep], rcond=None)
    residual = y[keep] - x @ beta
    sigma = float(np.sqrt(residual @ residual / max(len(residual) - x.shape[1], 1)))
    cov = sigma ** 2 * np.linalg.inv(x.T @ x)
    n = len(drivers)
    centre = np.eye(n) - np.ones((n, n)) / n                    # relative to the field mean
    result = pd.DataFrame({"driver_number": drivers, "pace": beta[:n],
                           "se": np.sqrt(np.diag(centre @ cov[:n, :n] @ centre.T)),
                           "laps": [int((keep & (frame.driver_number == d).to_numpy()).sum()) for d in drivers]})
    result.attrs = {"corrections": dict(zip(names[n:], beta[n:].tolist())), "noise": sigma,
                    "laps_used": int(keep.sum()), "laps_clean": len(frame)}
    return result


def _design(frame, drivers, names=None):
    """Columns: one per driver (pace), compound offsets against the first compound,
    wear per compound with enough laps and ages, one shared wear for the rest. With
    `names`, the same columns as a previous call (for residuals of trimmed laps)."""
    number = frame.driver_number.to_numpy()
    compound = frame.compound.to_numpy()
    age = frame.TyreLife.to_numpy(dtype=float)
    if names is None:
        present = sorted(set(compound))
        own = [c for c in present if (compound == c).sum() >= MIN_WEAR_LAPS and len(set(age[compound == c])) >= 2]
        names = [f"driver:{d}" for d in drivers] + [f"compound:{c}" for c in present[1:]] \
            + [f"wear:{c}" for c in own] + (["wear:shared"] if len(own) < len(present) else [])
    own = {name[5:] for name in names if name.startswith("wear:") and name != "wear:shared"}
    columns = []
    for name in names:
        kind, value = name.split(":", 1)
        if kind == "driver":
            columns.append((number == value).astype(float))
        elif kind == "compound":
            columns.append((compound == value).astype(float))
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
