"""How far a race pace's reported uncertainty understates its real error.

`laps.race_pace` reports a standard error that assumes independent laps. A stint
shares its fuel and tyre phase, management and traffic, so its laps' errors do not
average out. Measured by cutting: a driver with at least 25 clean laps is cut to k
contiguous laps of one stint (k = 5, 7, 10, 14, 20), the race is refitted, and the cut
pace is compared with the full-lap pace (each relative to the drivers in both fits).

`scale_by_season` turns those rows into the factor on the estimation variance used for
the forecasts of each season, from earlier seasons only (pace.STINT_SCALE holds the
values measured on 2018-2024 dry races with seed 7). Never run on sealed seasons.
"""
import numpy as np
import pandas as pd

from . import laps as L

CUT_LAPS = (5, 7, 10, 14, 20)
MIN_FULL_LAPS = 25
DRIVERS_PER_RACE = 4


def cut_stint_rows(view, seasons, seed=7, read=None):
    """One row per cut: season, event, driver, k, laps_used, err (cut minus full, % of a
    lap), se (reported for the cut), se_full, full_laps. Dry races of `seasons` in
    `view`; races read with `read` (laps.race_pace options; default the champion's
    reading, partial, min_laps 5 so a cut driver keeps a pace)."""
    read = {"partial": True, "min_laps": 5, **(read or {})}
    rng = np.random.default_rng(seed)
    races = view.sessions[(view.sessions.session_type == "R") & view.sessions.season.astype(int).isin(seasons)]
    rows = []
    for race in races.itertuples():
        laps = view.laps(race.session_id)
        if laps.empty or L.wet_race(laps):
            continue
        full = L.race_pace(laps, **read)
        if full.empty or len(full) < 6:
            continue
        clean = L.clean_race_laps(laps)
        full = full.set_index("driver_number")
        long = [d for d in full.index if (clean.driver_number == d).sum() >= MIN_FULL_LAPS]
        if not long:
            continue
        for driver in rng.choice(long, size=min(DRIVERS_PER_RACE, len(long)), replace=False):
            mine = clean[clean.driver_number == driver].sort_values("LapNumber")
            for k in CUT_LAPS:
                stints = [g for _, g in mine.groupby("Stint") if len(g) >= k]
                if not stints:
                    continue
                stint = stints[rng.integers(len(stints))]
                start = rng.integers(0, len(stint) - k + 1)
                kept = stint.index[start:start + k]
                cut = L.race_pace(laps.drop(index=mine.index.difference(kept)), **read)
                if cut.empty or driver not in set(cut.driver_number):
                    continue
                cut = cut.set_index("driver_number")
                common = full.index.intersection(cut.index)
                err = (cut.pace[driver] - cut.pace[common].mean()) - (full.pace[driver] - full.pace[common].mean())
                rows.append({"season": int(race.season), "event": race.event_key, "driver": driver, "k": k,
                             "laps_used": int(cut.laps[driver]), "err": float(err), "se": float(cut.se[driver]),
                             "se_full": float(full.se[driver]), "full_laps": int(full.laps[driver])})
    return pd.DataFrame(rows)


def scale(rows):
    """The factor c with E[err^2] = c se^2: summed squared errors over summed squared
    reported se."""
    return float((rows.err ** 2).sum() / (rows.se ** 2).sum())


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
