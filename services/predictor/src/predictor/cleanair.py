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
