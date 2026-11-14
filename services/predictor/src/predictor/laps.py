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
        if (trimmed == keep).all():
            break
        keep = trimmed
    x, names = _design(frame[keep], drivers)                    # the final fit, on the final lap set
    if np.linalg.matrix_rank(x) < x.shape[1]:
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
