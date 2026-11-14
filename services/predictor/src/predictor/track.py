"""Corners and track profiles from telemetry.

A session's clean fast laps (dry tyres, green track, one per driver) are put on a
common grid of fractions of the lap; their median racing line (position) and
median speed and throttle form the session's reference lap. Corners are found
once on that reference, from the line's curvature, not from speed, so flat-out
and lift corners count (Maggotts, 130R). A corner is split where its direction of
turn changes, and its turn is the total heading change through it.

A track profile is a set of numbers, not a category, so a mixed track (Sepang,
Silverstone) is simply a mixed profile:

- time shares that sum to 1: straights, and corners by speed band (slow under
  about 120 km/h, medium 120-200, fast over 200). Bands are soft: a corner taken
  at 120 km/h counts half slow, half medium, so a few km/h between years cannot
  flip it from one band to the other;
- full-throttle share of the lap;
- per km: heavy braking zones (a slow-down of more than 100 km/h), hairpins (under
  90 km/h with 120 degrees or more of turn) and chicanes (two short corners of
  opposite direction close together, paired once each).

Every threshold is soft (a logistic weight), so a corner near one counts partly
and cannot flip a count between years. A corner made of joined pieces (same
direction, close together) is counted once, but its time is banded by each
piece's own minimum speed, and the hairpin test applies to each piece: whether two
pieces happen to join must not move a fast corner into the slow band.

Each grid cell belongs to one corner (cells between corners split at the
midpoint) or to a straight, so nothing is counted twice.

Sessions where anyone ran intermediate or wet tyres are not used: the fastest
slick laps of a drying track (Singapore 2022, Montreal 2023) understate corner
speeds.

Speeds, and so the bands, describe the track as the cars of the time drive it: a
corner at 205 km/h in 2019 may be 195 km/h under the 2021-2022 rules. That is
what a forecast needs, so `layout_profile` takes the median of the most recent
sessions of a layout (`circuit_layouts.csv`) that the cutoff allows: earlier
visits, or this weekend's practice once it has run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

STEP_M = 5.0                 # distance between grid points of the reference lap
MIN_RADIUS_M = 350.0         # tighter than this is a corner
MIN_TURN_DEG = 20.0          # pieces turning less than this in total are kinks
MERGE_M = 40.0               # pieces of one corner (same direction) closer than this are joined
CORNER_MARGIN_M = 25.0       # braking and exit around a corner's curved part
BAND_WIDTH_KMH = 7.5         # softness of the band edges (logistic scale)
HAIRPIN = (90.0, 120.0)      # slower than this (km/h) with at least this turn (degrees)
CHICANE = {"gap_m": 60.0, "piece_m": 120.0, "below_kmh": 200.0, "max_turn_deg": 100.0}
HEAVY_BRAKING_KMH = 100.0
RECOVERY_KMH = 5.0           # a braking zone ends once speed recovers this much above its minimum
FULL_THROTTLE = 98.0
PADDING = pd.Timedelta(seconds=1)
MAX_GAP_S = 1.5              # a lap whose telemetry has a longer gap is not used
MIN_LAPS = 3
RECENT_SESSIONS = 3          # a layout's profile is the median of its latest sessions
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
FEATURES = ("share_straight", "share_slow", "share_medium", "share_fast", "share_full_throttle",
            "heavy_braking_per_km", "hairpins_per_km", "chicanes_per_km")


def _smooth(values, window):
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def lap_trace(car, pos, start, end):
    """One lap on distance from its start: distance (m), x, y (m), speed (km/h),
    throttle (%), time (s); None if the telemetry does not cover the lap or has a gap.
    `car` and `pos` may extend beyond the lap (they are cut to [start, end])."""
    car = car.drop_duplicates("Date").sort_values("Date")
    pos = pos.drop_duplicates("Date").sort_values("Date")
    if car.empty or pos.empty or car.Date.iloc[0] > start or car.Date.iloc[-1] < end \
            or pos.Date.iloc[0] > start or pos.Date.iloc[-1] < end:
        return None
    for frame in (car, pos):
        inside = frame[(frame.Date >= start - PADDING) & (frame.Date <= end + PADDING)].Date
        if inside.diff().dt.total_seconds().max() > MAX_GAP_S:
            return None
    t = (car.Date - start).dt.total_seconds().to_numpy()
    v = car.Speed.to_numpy(dtype=float)
    d = np.concatenate([[0.0], np.cumsum(np.diff(t) * (v[1:] + v[:-1]) / 2 / 3.6)])
    lap_end = (end - start).total_seconds()
    d0, d1 = np.interp(0.0, t, d), np.interp(lap_end, t, d)
    if d1 - d0 < 1000.0:
        return None
    grid = np.arange(0.0, d1 - d0, STEP_M)
    time = np.interp(grid + d0, d, t)
    pt = (pos.Date - start).dt.total_seconds().to_numpy()
    return pd.DataFrame({
        "distance": grid,
        "x": np.interp(time, pt, pos.X.to_numpy(dtype=float)) / 10.0,     # decimetres
        "y": np.interp(time, pt, pos.Y.to_numpy(dtype=float)) / 10.0,
        "speed": np.interp(time, t, v),
        "throttle": np.interp(time, t, car.Throttle.to_numpy(dtype=float)),
        "time": time,
    })


def reference_lap(traces):
    """Median of several laps on a common grid of lap fractions, scaled to the
    median lap length."""
    length = float(np.median([trace.distance.iloc[-1] for trace in traces]))
    grid = np.arange(0.0, length, STEP_M)
    columns = {}
    for name in ("x", "y", "speed", "throttle"):
        columns[name] = np.median([np.interp(grid / length, trace.distance / trace.distance.iloc[-1], trace[name])
                                   for trace in traces], axis=0)
    speed = np.maximum(columns["speed"], 1.0)
    time = np.concatenate([[0.0], np.cumsum(STEP_M / ((speed[1:] + speed[:-1]) / 2 / 3.6))])
    return pd.DataFrame({"distance": grid, **columns, "time": time})


def corners(trace):
    """Corners of a lap or reference lap, in lap order."""
    x, y = _smooth(trace.x.to_numpy(), 7), _smooth(trace.y.to_numpy(), 7)
    dx, dy = np.gradient(x), np.gradient(y)
    moving = np.hypot(dx, dy) > 1e-6
    heading = np.unwrap(np.arctan2(dy, dx))
    curvature = np.where(moving, _smooth(np.gradient(heading) / STEP_M, 5), 0.0)
    distance, speed = trace.distance.to_numpy(), trace.speed.to_numpy()
    sign = np.sign(curvature) * (np.abs(curvature) > 1.0 / MIN_RADIUS_M)
    pieces, i = [], 0
    while i < len(distance):
        if sign[i] != 0:
            j = i
            while j + 1 < len(distance) and sign[j + 1] == sign[i]:
                j += 1
            pieces.append([i, j, sign[i]])
            i = j + 1
        else:
            i += 1
    joined = []
    for a, b, s in pieces:
        if joined and s == joined[-1][2] and distance[a] - distance[joined[-1][1]] < MERGE_M:
            joined[-1][1] = b
        else:
            joined.append([a, b, s])
    turn = np.abs(np.diff(heading, prepend=heading[0]))
    rows = []
    for a, b, s in joined:
        gross = float(np.degrees(turn[a:b + 1].sum()))
        if gross < MIN_TURN_DEG:
            continue
        parts = [(distance[pa], distance[pb], float(np.degrees(turn[pa:pb + 1].sum())), float(speed[pa:pb + 1].min()))
                 for pa, pb, _ in pieces if pa >= a and pb <= b]
        rows.append({"start": distance[a], "end": distance[b], "turn_deg": gross, "direction": "L" if s > 0 else "R",
                     "min_speed": float(speed[a:b + 1].min()), "pieces": parts,
                     "hairpin": max(_hairpin(t, v) for _, _, t, v in parts)})
    table = pd.DataFrame(rows, columns=["start", "end", "turn_deg", "direction", "min_speed", "pieces", "hairpin"])
    table["chicane"] = _chicanes(table)
    return table


def _logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def _hairpin(turn_deg, min_speed):
    """Soft: under HAIRPIN[0] km/h with at least HAIRPIN[1] degrees of turn."""
    return float(_logistic((HAIRPIN[0] - min_speed) / BAND_WIDTH_KMH) * _logistic((turn_deg - HAIRPIN[1]) / 10.0))


def _pair_weight(p, c):
    """Soft: two short corners of opposite direction, close together, slow or medium,
    neither a hairpin-sized turn."""
    if p.direction == c.direction:
        return 0.0
    weight = _logistic((CHICANE["gap_m"] - (c.start - p.end)) / 10.0)
    for corner in (p, c):
        weight *= _logistic((CHICANE["piece_m"] - (corner.end - corner.start)) / 15.0)
        weight *= _logistic((CHICANE["max_turn_deg"] - corner.turn_deg) / 10.0)
    return float(weight * _logistic((CHICANE["below_kmh"] - max(p.min_speed, c.min_speed)) / BAND_WIDTH_KMH))


def _chicanes(table):
    """Chicane weight per corner: adjacent pairs taken in lap order, each corner in at
    most one pair (L-R-L esses give one chicane, not two); a pair counts when its
    weight exceeds 0.5 and contributes that weight."""
    weights = np.zeros(len(table))
    k = 1
    while k < len(table):
        w = _pair_weight(table.iloc[k - 1], table.iloc[k])
        if w > 0.5:
            weights[k - 1] = weights[k] = w
            k += 2
        else:
            k += 1
    return weights


def band_weights(speed):
    """Soft membership of a minimum corner speed in the slow, medium and fast bands."""
    slow = 1.0 / (1.0 + np.exp((speed - 120.0) / BAND_WIDTH_KMH))
    fast = 1.0 / (1.0 + np.exp(-(speed - 200.0) / BAND_WIDTH_KMH))
    return {"slow": slow, "medium": max(0.0, 1.0 - slow - fast), "fast": fast}


def braking_zones(trace):
    """Slow-downs on the speed trace, each weighted softly by its size around
    HEAVY_BRAKING_KMH. A zone runs from a peak to the lowest speed before the car
    recovers RECOVERY_KMH above it, so a brief uptick does not split it."""
    speed = _smooth(trace.speed.to_numpy(), 3)
    total, top, low = 0.0, speed[0], speed[0]

    def zone(drop):
        return float(_logistic((drop - HEAVY_BRAKING_KMH) / 10.0)) if drop > 20 else 0.0

    for v in speed[1:]:
        if v > low + RECOVERY_KMH and low < top - RECOVERY_KMH:    # recovered from a slow-down: close it
            total += zone(top - low)
            top = low = v
        elif v < low:
            low = v
        elif v > top:                                               # still climbing to the next peak
            top = low = v
    return total + zone(top - low)


def profile(trace, table=None):
    """The profile of a lap or reference lap (see the module docstring)."""
    table = corners(trace) if table is None else table
    distance, time = trace.distance.to_numpy(), trace.time.to_numpy()
    dt = np.diff(time, append=time[-1])
    lap = float(dt.sum())
    owner = np.full(len(distance), -1)
    for k, row in enumerate(table.itertuples()):
        a, b = row.start - CORNER_MARGIN_M, row.end + CORNER_MARGIN_M
        if k > 0:
            a = max(a, (table.end.iloc[k - 1] + row.start) / 2)
        if k + 1 < len(table):
            b = min(b, (row.end + table.start.iloc[k + 1]) / 2)
        owner[(distance >= a) & (distance < b)] = k
    shares = {"slow": 0.0, "medium": 0.0, "fast": 0.0}
    for k, row in enumerate(table.itertuples()):
        cells = np.flatnonzero(owner == k)
        if not len(cells):
            continue
        centres = np.array([(a + b) / 2 for a, b, _, _ in row.pieces])
        nearest = np.abs(distance[cells][:, None] - centres[None, :]).argmin(axis=1)
        for piece, (_, _, _, piece_speed) in enumerate(row.pieces):
            weights = band_weights(piece_speed)
            time_in = dt[cells[nearest == piece]].sum() / lap
            for band in shares:
                shares[band] += weights[band] * time_in
    km = distance[-1] / 1000.0
    return {"share_straight": float(dt[owner < 0].sum() / lap),
            **{f"share_{band}": float(value) for band, value in shares.items()},
            "share_full_throttle": float(dt[trace.throttle.to_numpy() >= FULL_THROTTLE].sum() / lap),
            "heavy_braking_per_km": braking_zones(trace) / km,
            "hairpins_per_km": float(table.hairpin.sum()) / km,
            "chicanes_per_km": float(table.chicane.sum()) / 2 / km,
            "corners": int(len(table)), "length_m": float(distance[-1])}


def clean_laps(laps, count=None):
    """The fastest clean laps of a session, one per driver: timed, accurate, not
    deleted, not an in or out lap, on dry tyres, with the track green throughout."""
    clean = laps[laps.LapTime.notna() & laps.LapStartDate.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna()]
    if "IsAccurate" in clean:
        clean = clean[clean.IsAccurate.fillna(False).astype(bool)]
    if "Deleted" in clean:
        clean = clean[~clean.Deleted.fillna(False).astype(bool)]
    if "Compound" in clean:
        compound = clean.Compound.fillna("").astype(str).str.upper()
        clean = clean[(compound != "") & ~compound.isin(WET_COMPOUNDS) & (compound != "UNKNOWN")]
    if "TrackStatus" in clean:
        clean = clean[clean.TrackStatus.fillna("").astype(str) == "1"]
    best = clean.sort_values("LapTime").drop_duplicates("DriverNumber")
    return best if count is None else best.head(count)


def lap_telemetry(view, session_id, lap):
    """The trace of one lap, or None if its telemetry is missing or has gaps."""
    car, pos = (view.telemetry(session_id, lap.DriverNumber, kind) for kind in ("car_data", "pos_data"))
    if car is None or pos is None:
        return None
    start, end = lap.LapStartDate, lap.LapStartDate + lap.LapTime
    window = (lambda frame: frame[(frame.Date >= start - PADDING) & (frame.Date <= end + PADDING)])
    return lap_trace(window(car), window(pos), start, end)


def wet_running(laps):
    """Whether anyone ran intermediate or wet tyres in the session."""
    return "Compound" in laps and laps.Compound.fillna("").astype(str).str.upper().isin(WET_COMPOUNDS).any()


def session_profile(view, session_id, count=5):
    """The profile of a session's reference lap; None for a session with wet running
    or fewer than MIN_LAPS clean laps with usable telemetry. Reads only what `view`
    allows."""
    laps = view.laps(session_id)
    if wet_running(laps):
        return None
    traces = []
    for lap in clean_laps(laps).itertuples():     # fastest first, one per driver
        trace = lap_telemetry(view, session_id, lap)
        if trace is not None:
            traces.append(trace)
            if len(traces) == count:
                break
    if len(traces) < MIN_LAPS:
        return None
    return {**profile(reference_lap(traces)), "laps": len(traces)}


def layout_profile(profiles):
    """Median of the latest RECENT_SESSIONS usable session profiles of one layout
    (`profiles` in time order, each a dict from `session_profile` or None); the time
    shares are renormalised to sum to 1."""
    usable = [p for p in profiles if p is not None][-RECENT_SESSIONS:]
    if not usable:
        return None
    median = pd.DataFrame(usable)[list(FEATURES)].median().to_dict()
    shares = [f"share_{b}" for b in ("straight", "slow", "medium", "fast")]
    total = sum(median[f] for f in shares)
    return {**median, **{f: median[f] / total for f in shares}}


LAYOUTS_FILE = Path(__file__).resolve().parent / "circuit_layouts.csv"


def layouts():
    """Circuit layouts used in F1 from 2018: circuit, layout name, first and last
    season, events it was used at when that differs from its seasons (Bahrain's outer
    loop at the 2020 Sakhir GP), official turn count and its source."""
    table = pd.read_csv(LAYOUTS_FILE, dtype={"events": str}, keep_default_na=False)
    table["events"] = [tuple(e.split(";")) if e else () for e in table.events]
    return table


def layout_of(circuit_id, season, event_key, table=None):
    """The layout an event ran on, as "circuit_id/layout"; None if the table has none."""
    table = layouts() if table is None else table
    rows = table[(table.circuit_id == circuit_id) & (table.layout_from <= season) & (table.layout_to >= season)]
    named = rows[[event_key in events for events in rows.events]]
    rows = named if len(named) else rows[[not events for events in rows.events]]
    if len(rows) != 1:
        return None
    return f"{circuit_id}/{rows.layout.iloc[0]}"
