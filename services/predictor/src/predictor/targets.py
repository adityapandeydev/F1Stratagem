"""Forecast targets: the final classification of a qualifying, sprint or race session.

Every entrant gets a position in the full order (FastF1's `Position`, which ranks
non-classified drivers by distance covered). Race and sprint entrants also get an
outcome:

    win, other_points, outside_points        classified finishers
    mechanical, incident_or_other            retirements with a stated cause
    retired_unknown                          retirements recorded only as "Retired"
    unmapped                                 retirements with a status not yet mapped
    excluded                                 did not start, withdrawn, disqualified

From 2023 the sources record almost every retirement as "Retired". Causes
reviewed by hand from race reports (`f1data labels`) replace the recorded status
where they exist; otherwise the split is scored only where a cause is stated.
Unknown stays unknown; it is never imputed.
"""

import functools

import pandas as pd
from f1data.labels import load_labels

from .rules import points_positions

HOLDOUT_SEASONS = frozenset({2025, 2026})
OUTCOMES = ("win", "other_points", "outside_points", "mechanical", "incident_or_other")

# Statuses describing a car failure.
MECHANICAL = frozenset({
    "Battery", "Brakes", "Clutch", "Cooling system", "Differential", "Driveshaft", "Electrical", "Electronics",
    "ERS", "Engine", "Exhaust", "Fuel leak", "Fuel pressure", "Fuel pump", "Fuel system", "Gearbox", "Halfshaft",
    "Hydraulics", "Mechanical", "MGU-H", "MGU-K", "Oil leak", "Oil pressure", "Overheating", "Power loss",
    "Power Unit", "Radiator", "Rear wing", "Steering", "Suspension", "Technical", "Throttle", "Transmission",
    "Turbo", "Undertray", "Vibrations", "Water leak", "Water pressure", "Water pump", "Wheel bearing",
})
# Statuses describing a collision, driver error, damage or an operational problem.
INCIDENT_OR_OTHER = frozenset({
    "Accident", "Collision", "Collision damage", "Damage", "Debris", "Front wing", "Illness", "Out of fuel",
    "Puncture", "Spun off", "Tyre", "Wheel", "Wheel nut",
})
GENERIC_RETIREMENT = frozenset({"Retired"})
# Non-classified, non-excluded entrants: every one of these is a retirement.
RETIREMENTS = ("mechanical", "incident_or_other", "retired_unknown", "unmapped")
EXCLUDED_CODES = frozenset({"D", "E", "F", "W"})  # disqualified, excluded, failed to qualify, withdrawn


# Reviewed causes (f1data labels) -> outcome. The finer cause is kept for incident models.
LABEL_OUTCOME = {"mechanical": "mechanical", "unknown": "retired_unknown"}


@functools.lru_cache(maxsize=1)
def reviewed_causes():
    return {key: label["cause"] for key, label in load_labels()["retirements"].items()}


class HoldoutError(PermissionError):
    """Targets of held-out seasons were requested without explicit permission."""


def retirement_cause(status):
    if status in MECHANICAL:
        return "mechanical"
    if status in INCIDENT_OR_OTHER:
        return "incident_or_other"
    if status in GENERIC_RETIREMENT:
        return "retired_unknown"
    return None


def classify(results, season, session_type):
    """Positions and outcomes for one session's results table."""
    frame = pd.DataFrame({
        "driver_number": results.DriverNumber.astype(str),
        "position": results.Position,
        "status": results.Status,
        "classified_position": results.ClassifiedPosition.astype("string"),
        "laps": results.Laps if "Laps" in results else pd.Series(pd.NA, index=results.index),
    })
    if session_type in ("R", "S") and frame.position.isna().any():
        # The source occasionally leaves a starter's position blank (e.g. a crash on the
        # opening lap). They are placed behind every positioned entrant, by laps completed.
        blank = frame.position.isna() & ~frame.classified_position.fillna("").isin(EXCLUDED_CODES)
        order = frame[blank].sort_values("laps", ascending=False, na_position="last").index
        frame.loc[order, "position"] = range(int(frame.position.max()) + 1, int(frame.position.max()) + 1 + len(order))
    frame = frame.dropna(subset=["position"]).drop(columns="laps").copy()
    frame["position_repaired"] = False
    if frame.position.duplicated().any():
        # The source occasionally repeats a position (e.g. two drivers 15th). Re-rank by the
        # published position, then by the session's own best times; the raw data is unchanged.
        best = [results.get(column) for column in ("Q3", "Q2", "Q1") if column in results]
        tiebreak = pd.concat(best, axis=1).loc[frame.index] if best else pd.DataFrame(index=frame.index)
        keys = pd.concat([frame.position, tiebreak], axis=1)
        order = keys.sort_values(list(keys.columns), na_position="last").index
        first = int(frame.position.min())
        repaired = pd.Series(range(first, first + len(frame)), index=order)
        frame["position_repaired"] = repaired != frame.position
        frame["position"] = repaired
    frame["position"] = frame.position.astype(int)
    if session_type not in ("R", "S"):
        frame["outcome"] = pd.NA
        frame["excluded"] = False
        return frame.sort_values("position").reset_index(drop=True)
    code = frame.classified_position.fillna("")
    classified = code.str.isdigit()
    excluded = code.isin(EXCLUDED_CODES) | frame.status.isin(["Did not start", "Withdrew", "Disqualified"])
    paying = points_positions(season, session_type)
    outcome = []
    for is_classified, is_excluded, position, status in zip(classified, excluded, frame.position, frame.status):
        if is_excluded:
            outcome.append("excluded")
        elif is_classified:
            outcome.append("win" if position == 1 else "other_points" if position <= paying else "outside_points")
        else:
            outcome.append(retirement_cause(status) or "unmapped")
    frame["outcome"] = outcome
    frame["excluded"] = excluded.values
    return frame.sort_values("position").reset_index(drop=True)


def session_target(archive, event_key, session_type, allow_holdout=False):
    """Final classification of `session_type` at `event_key`, joined to canonical IDs."""
    season = int(event_key[:4])
    if season in HOLDOUT_SEASONS and not allow_holdout:
        raise HoldoutError(f"{season} is held out until model selection; pass allow_holdout=True deliberately")
    sessions = archive.event_sessions(event_key)
    match = sessions[sessions.session_type == session_type]
    if match.empty:
        return None
    session_id = match.session_id.iloc[0]
    target = classify(archive._table(session_id, "results"), season, session_type)
    columns = [c for c in ("driver_number", "driver_id", "team_id", "lineage_id") if c in archive.entrants]
    ids = archive.entrants[archive.entrants.session_id == session_id][columns]
    target = target.merge(ids, on="driver_number", how="left")
    target.insert(0, "session_id", session_id)
    if session_type in ("R", "S"):
        causes = reviewed_causes()
        unmatched = [key for key in causes if key.startswith(f"{session_id}/")
                     and key.split("/", 1)[1] not in set(target.driver_number)]
        if unmatched:
            raise ValueError(f"reviewed labels with no entrant in {session_id}: {unmatched}")
        target["reviewed_cause"] = [causes.get(f"{session_id}/{number}") for number in target.driver_number]
        relabel = target.outcome.isin(RETIREMENTS) & target.reviewed_cause.notna()
        target.loc[relabel, "outcome"] = target.loc[relabel, "reviewed_cause"].map(
            lambda cause: LABEL_OUTCOME.get(cause, "incident_or_other"))
    return target
