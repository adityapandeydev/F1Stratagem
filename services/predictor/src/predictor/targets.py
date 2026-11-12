"""Forecast targets: the final classification of a qualifying, sprint or race session.

Every entrant gets a position in the full order (FastF1's `Position`, which ranks
non-classified drivers by distance covered). Race and sprint entrants also get an
outcome:

    win, other_points, outside_points        classified finishers
    mechanical, incident_or_other            retirements with a stated cause
    retired_unknown                          retirements recorded only as "Retired"
    excluded                                 did not start, withdrawn, disqualified

From 2023 the sources record almost every retirement as "Retired", so the
mechanical / incident split is scored only where a cause is stated. Unknown stays
unknown; it is never imputed.
"""

import pandas as pd

from .rules import points_positions

HOLDOUT_SEASONS = frozenset({2025, 2026})
OUTCOMES = ("win", "other_points", "outside_points", "mechanical", "incident_or_other")

# Statuses describing a car failure.
MECHANICAL = frozenset({
    "Battery", "Brakes", "Clutch", "Cooling system", "Differential", "Driveshaft", "Electrical", "Electronics",
    "ERS", "Engine", "Exhaust", "Fuel leak", "Fuel pressure", "Fuel pump", "Fuel system", "Gearbox", "Halfshaft",
    "Hydraulics", "Mechanical", "MGU-H", "MGU-K", "Oil leak", "Oil pressure", "Overheating", "Power loss",
    "Power Unit", "Radiator", "Rear wing", "Steering", "Suspension", "Technical", "Throttle", "Transmission",
    "Turbo", "Undertray", "Vibrations", "Water leak", "Water pressure", "Wheel bearing",
})
# Statuses describing a collision, driver error, damage or an operational problem.
INCIDENT_OR_OTHER = frozenset({
    "Accident", "Collision", "Collision damage", "Damage", "Debris", "Front wing", "Illness", "Out of fuel",
    "Puncture", "Spun off", "Tyre", "Wheel", "Wheel nut",
})
GENERIC_RETIREMENT = frozenset({"Retired"})
EXCLUDED_CODES = frozenset({"D", "E", "F", "W"})  # disqualified, excluded, failed to qualify, withdrawn


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
    })
    frame = frame.dropna(subset=["position"]).copy()
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
    ids = archive.entrants[archive.entrants.session_id == session_id][["driver_number", "driver_id", "team_id"]]
    target = target.merge(ids, on="driver_number", how="left")
    target.insert(0, "session_id", session_id)
    return target
