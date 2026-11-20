"""Clean-air race pace: the pace model with race evidence from laps out of dirty air.

A car running close behind another is slowed by its wake and is not showing its own
pace: a car stuck in a midfield train looks slower than it is, a leader in free air
does not. `CleanAirPaceModel` reads a race's corrected pace (`laps.race_pace`) only
from laps on which the car crossed the line at least `min_gap` seconds behind the car
before it (any car, any lap: lapping traffic counts too); everything else is the pace
model as it is.

The module is not part of the fit cache's code fingerprint: the variant's fits are
stored under its own model name (`with_clean_air`), which carries VERSION; change
VERSION whenever this module's behaviour changes, so no stale fit is reused."""
import numpy as np
import pandas as pd

from .laps import race_pace
from .pace import PaceModel, base_type, race_observations

VERSION = 1


def dirty_air_removed(laps, min_gap):
    """`laps` with the lap time of every lap ended within `min_gap` s of the car
    crossing the line before it set to missing (race pace then leaves it out)."""
    if laps.empty or "Time" not in laps:
        return laps
    ends = laps.dropna(subset=["Time"]).assign(end=lambda f: f.Time.dt.total_seconds()).sort_values("end")
    gap = np.r_[np.inf, np.diff(ends.end.to_numpy())]
    dirty = ends.index[gap < min_gap]
    out = laps.copy()
    out.loc[dirty, "LapTime"] = pd.NaT
    return out


class CleanAirPaceModel(PaceModel):
    """`PaceModel` whose race observations come from clean-air laps only."""

    min_gap = 1.5

    def observations(self, view, session):
        if base_type(session.session_type) == "R" and session.session_id not in self._observations:
            laps = dirty_air_removed(view.laps(session.session_id), self.min_gap)
            groups, covariances = race_observations(laps, view.entrants(session.session_id), with_covariance=True,
                                                    drying=session.session_type != "R")
            self._observations[session.session_id] = groups
            self._covariances[session.session_id] = covariances
        return super().observations(view, session)


def with_clean_air(model, min_gap):
    """`model` (a PaceModel) reading race pace from laps at least `min_gap` s behind
    the car ahead; its name (and fit-cache key) gains the gap and VERSION."""
    model.__class__ = CleanAirPaceModel
    model.min_gap = float(min_gap)
    model.name = f"{model.name}-air{min_gap:g}v{VERSION}"
    return model


CORRECTED_VERSION = 4   # 4: start gaps across lap counts (race_sim.start_state)


def start_gaps(laps):
    """Each lap's gap at its start to the car ahead (`race_sim.start_state`, the
    previous lap's crossings in their order)."""
    from .race_sim import start_state
    return start_state(laps).start_gap


SUPPORT = 2.0          # a band is corrected only when its race penalty is identified and this many standard errors above 0


def following_correction(laps):
    """Per-car following corrections of a dry race, by empirical Bayes within the
    race. For each band whose race penalty (`race_sim.following_penalty`, all laps)
    is identified and SUPPORT standard errors above 0: the spread of the true penalty
    between drivers, tau^2 = mean((own - race)^2) - mean(own se^2) (0 at least), from
    the drivers whose own penalty is identified (`driver_following_penalties`, 3 or
    more); each car's correction is the race value pulled towards its own estimate by
    tau^2 / (tau^2 + se^2), and its remaining variance is tau^2 se^2 / (tau^2 + se^2),
    or tau^2 for a car whose own penalty is not identified (a car in traffic all
    stint: its clean-air pace rests on the race value, with that uncertainty).
    {"race": (penalties, se) with unsupported bands 0, "cars": {driver: (corrections,
    variances)}}; None when no band is supported."""
    from .race_sim import driver_following_penalties, following_penalty
    fitted = following_penalty(laps)
    if fitted is None or laps.empty:
        return None
    penalty, _, se = (np.asarray(v, dtype=float) for v in fitted)
    supported = np.isfinite(penalty) & np.isfinite(se) & (penalty > SUPPORT * se)
    if not supported.any():
        return None
    race_penalty, race_se = np.where(supported, penalty, 0.0), np.where(supported, se, 0.0)
    own = driver_following_penalties(laps)
    own = own if own is not None else pd.DataFrame(columns=["driver", "band", "penalty", "se", "laps"])
    tau2 = np.zeros(3)
    for band in np.flatnonzero(supported):
        rows = own[own.band == band]
        if len(rows) >= 3:
            tau2[band] = max(0.0, float(np.mean((rows.penalty - penalty[band]) ** 2) - np.mean(rows.se ** 2)))
    cars = {}
    for driver in laps.DriverNumber.dropna().astype(str).unique():
        correction, variance = race_penalty.copy(), np.where(supported, tau2, 0.0)
        mine = own[own.driver == driver]
        for row in mine.itertuples():
            band = int(row.band)
            if supported[band] and tau2[band] + row.se ** 2 > 0:
                w = tau2[band] / (tau2[band] + row.se ** 2)
                correction[band] = penalty[band] + w * (row.penalty - penalty[band])
                variance[band] = w * row.se ** 2
        cars[driver] = (correction, variance)
    return {"race": (race_penalty, race_se), "cars": cars}


def _apply(laps, correction):
    from .race_sim import PENALTY_BANDS
    gaps = start_gaps(laps).to_numpy()
    edges = (0.0,) + PENALTY_BANDS
    drivers = laps.DriverNumber.astype(str).to_numpy()
    cut = np.zeros(len(laps))
    for driver, (values, _) in correction["cars"].items():
        mine = drivers == driver
        for band, (lo, hi) in enumerate(zip(edges, edges[1:])):
            cut[mine & (gaps >= lo) & (gaps < hi)] = values[band]
    out = laps.copy()
    out["LapTime"] = out.LapTime * np.exp(-cut / 100)
    return out


def following_corrected(laps):
    """`laps` with each lap's time freed of its car's following correction for its
    start gap band (`following_correction`); unchanged when no band is supported."""
    correction = following_correction(laps)
    return laps if correction is None else _apply(laps, correction)


def correction_covariance(laps, correction, drivers):
    """Covariance (% of a lap squared) that the following correction adds to the
    race paces of `drivers` (in that order), centred on the field mean: each car's
    share of its clean laps per band times the race penalty's error (shared by all
    cars) and its own remaining variance (its alone)."""
    from .laps import clean_race_laps
    from .race_sim import PENALTY_BANDS
    clean = clean_race_laps(laps)
    gaps = start_gaps(laps).loc[clean.index].to_numpy()
    edges = (0.0,) + PENALTY_BANDS
    share = np.zeros((len(drivers), 3))
    for i, driver in enumerate(drivers):
        mine = (clean.driver_number == driver).to_numpy()
        if mine.any():
            share[i] = [np.mean((gaps[mine] >= lo) & (gaps[mine] < hi)) for lo, hi in zip(edges, edges[1:])]
    _, race_se = correction["race"]
    own = np.array([correction["cars"].get(d, (None, np.zeros(3)))[1] for d in drivers]).reshape(len(drivers), 3)
    added = share @ np.diag(race_se ** 2) @ share.T + np.diag(np.sum(share ** 2 * own, axis=1))
    n = len(drivers)
    centre = np.eye(n) - np.ones((n, n)) / n
    return centre @ added @ centre.T


class CorrectedPaceModel(PaceModel):
    """`PaceModel` whose race observations come from every lap, each freed of its
    car's following correction (`following_correction`), with the correction's
    uncertainty added to the race paces' covariance (`correction_covariance`)."""

    def observations(self, view, session):
        if base_type(session.session_type) == "R" and session.session_id not in self._observations:
            raw = view.laps(session.session_id)
            correction = following_correction(raw)
            laps = raw if correction is None else _apply(raw, correction)
            entrants = view.entrants(session.session_id)
            drying = session.session_type != "R"
            groups, covariances = race_observations(laps, entrants, with_covariance=True, drying=drying)
            if correction is not None and covariances:
                # the correction's own uncertainty, for the drivers in the group (race_observations' order)
                pace = race_pace(laps, drying)
                ids = entrants.set_index("driver_number")
                keep = [pd.notna(ids.lineage_id.get(n) if "lineage_id" in ids else None) and pd.notna(ids.driver_id.get(n))
                        for n in pace.driver_number]
                drivers = [d for d, k in zip(pace.driver_number, keep) if k]
                covariances = [covariances[0] + correction_covariance(laps, correction, drivers)] + covariances[1:]
            self._observations[session.session_id] = groups
            self._covariances[session.session_id] = covariances
        return super().observations(view, session)


def with_corrected_pace(model):
    """`model` (a PaceModel) reading race pace from every lap freed of the following
    penalty; its name (and fit-cache key) gains CORRECTED_VERSION."""
    model.__class__ = CorrectedPaceModel
    model.name = f"{model.name}-fp{CORRECTED_VERSION}"
    return model
