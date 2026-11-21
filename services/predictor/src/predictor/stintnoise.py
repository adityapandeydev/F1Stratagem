"""How far a race pace's reported uncertainty understates its real error.

`laps.race_pace` reports a covariance that assumes independent laps. A stint shares
its fuel and tyre phase, management and traffic, so its laps' errors do not average
out. Measured by splitting a driver: in one fit of the race, k contiguous laps of one
of his stints (k = 5, 7, 10, 14, 20) and his laps of his other stints enter as two
separate drivers (the rest of the cut stint is left out, so the two share no stint).
Their pace difference has true mean zero; its reported variance comes from that
fit's joint covariance (shared compound, wear and field effects included, and the
two parts disjoint, so no overlap with a reference). The scale is the summed squared
difference over the summed reported variance.

`scale_by_season` turns those rows into the factor on the estimation variance used for
the forecasts of each season, from earlier seasons only (pace.STINT_SCALE holds the
values measured on 2018-2024 dry races with seed 7). It assumes one factor for both
parts of a split. Never run on sealed seasons.

An earlier version compared the cut with the driver's full-race pace, which contains
the cut laps: the overlap halves the scale for independent laps (review 2026-11-22).
"""
import numpy as np
import pandas as pd

from . import laps as L

CUT_LAPS = (5, 7, 10, 14, 20)
MIN_REST_LAPS = 10           # the driver's laps in his other stints
DRIVERS_PER_RACE = 4
REST = "~rest"               # suffix of the pseudo-driver holding a split driver's other stints


def split_difference(laps, driver, cut, rest, read):
    """(pace of `cut` minus pace of `rest`, its reported variance, laps of each) from one
    fit in which `driver`'s laps at index `cut` keep his number, those at `rest` become
    a pseudo-driver and his other laps are left out; None when either part is unfitted."""
    mine = laps.index[laps.DriverNumber.astype(str) == str(driver)]
    split = laps.drop(index=mine.difference(cut.union(rest))).copy()
    split.loc[rest, "DriverNumber"] = f"{driver}{REST}"
    fit = L.race_pace(split, **read)
    if fit.empty:
        return None
    order = list(fit.driver_number)
    a, b = str(driver), f"{driver}{REST}"
    if a not in order or b not in order:
        return None
    i, j = order.index(a), order.index(b)
    cov = np.asarray(fit.attrs["covariance"])
    variance = cov[i, i] + cov[j, j] - 2 * cov[i, j]
    return float(fit.pace.iloc[i] - fit.pace.iloc[j]), float(variance), int(fit.laps.iloc[i]), int(fit.laps.iloc[j])


def race_rows(laps, rng, read=None, drivers=DRIVERS_PER_RACE, cuts=CUT_LAPS):
    """Split-driver rows of one race (see the module docstring): driver, k, laps_used,
    rest_laps, err, var (reported variance of err), se (sqrt of var). `read`:
    laps.race_pace options (default the champion's reading, partial, min_laps 5)."""
    read = {"partial": True, "min_laps": 5, **(read or {})}
    if laps.empty or L.wet_race(laps):
        return []
    clean = L.clean_race_laps(laps)
    if clean.empty:
        return []
    candidates = []
    for d, mine in clean.groupby("driver_number"):
        if mine.Stint.nunique() >= 2:
            candidates.append(d)
    rows = []
    if not candidates:
        return rows
    for driver in rng.choice(sorted(candidates), size=min(drivers, len(candidates)), replace=False):
        mine = clean[clean.driver_number == driver].sort_values("LapNumber")
        for k in cuts:
            stints = [(s, g) for s, g in mine.groupby("Stint")
                      if len(g) >= k and (mine.Stint != s).sum() >= MIN_REST_LAPS]
            if not stints:
                continue
            stint, laps_of = stints[rng.integers(len(stints))]
            start = rng.integers(0, len(laps_of) - k + 1)
            result = split_difference(laps, driver, laps_of.index[start:start + k], mine.index[mine.Stint != stint],
                                      read)
            if result is None or not result[1] > 0:
                continue
            err, var, used, rest = result
            rows.append({"driver": driver, "k": k, "laps_used": used, "rest_laps": rest,
                         "rest_stints": int(mine.Stint[mine.Stint != stint].nunique()), "err": err, "var": var,
                         "se": float(np.sqrt(var))})
    return rows


def cut_stint_rows(view, seasons, seed=7, read=None):
    """`race_rows` of every dry race of `seasons` in `view`, with season and event."""
    rng = np.random.default_rng(seed)
    races = view.sessions[(view.sessions.session_type == "R") & view.sessions.season.astype(int).isin(seasons)]
    rows = []
    for race in races.itertuples():
        for row in race_rows(view.laps(race.session_id), rng, read):
            rows.append({"season": int(race.season), "event": race.event_key, **row})
    return pd.DataFrame(rows)


def scale(rows):
    """The factor c with err ~ N(0, c var), by maximum likelihood (a weighted mean of
    err^2 / var: the squared-error ratio would let a few large errors in long stints
    set it). Checked on 2018-2024: a further per-stint term added to c var fits to
    about zero by likelihood, and held-out z-score sd is 0.89-1.12 for k 5-20."""
    ratio = (rows.err ** 2 / rows["var"]).to_numpy()
    return float(ratio.mean())


def scale_by_season(rows, seasons):
    """{season: c from rows of earlier seasons only}; seasons without earlier rows are left out."""
    return {s: scale(rows[rows.season < s]) for s in seasons if (rows.season < s).any()}


def honesty(rows, scales):
    """z-score sd (1 = honest) by season and k, raw and with each season's own scale."""
    out = []
    for (season, k), part in rows.groupby(["season", "k"]):
        if season not in scales:
            continue
        z = part.err / part.se
        out.append({"season": season, "k": k, "n": len(part), "z_raw": float(z.std()),
                    "z_scaled": float((z / np.sqrt(scales[season])).std())})
    return pd.DataFrame(out)
