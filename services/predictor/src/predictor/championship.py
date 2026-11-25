"""Championship standings from classifications (stage 5).

Points per classified position come from `rules.points_table` (Grand Prix and sprint, by
season). On top of that, the rules that only matter for standings:

- Fastest lap: one point to the driver with the race's fastest lap if he is classified in
  the top ten, Grands Prix 2019-2024 (none in sprints).
- Reduced points: a Grand Prix that does not reach the distance its season's rules require
  scores a fraction. 2019-2021: half points below 75% of the distance (no fastest-lap point);
  from 2022 a sliding scale applies to races that are suspended and not resumed. The share
  is an input of a session (`share`), never looked up from its date: a historical replay
  passes the recorded share (`recorded_share`; only Belgium 2021 in 2019-2024), a
  simulated race its own scenario's (1 for a full race).
- Constructor deductions decided by the FIA (CONSTRUCTOR_DEDUCTIONS), each from the time
  its decision was published: standings as of a cutoff include only deductions published
  by then, each applied once.
- Ties on points: the driver (constructor) with more Grand Prix wins ranks higher, then more
  second places, and so on (countback over Grand Prix results; sprints are not counted).

`standings` builds the tables from per-session classifications; `points` scores one
session and works on arrays, so simulated seasons can use it.
"""

import numpy as np
import pandas as pd

from .rules import points_table

FASTEST_LAP_SEASONS = frozenset(range(2019, 2025))
REDUCED = {(2021, 12): 0.5}                       # Belgium 2021: one lap behind the safety car
# (season, constructor) -> (points, decision published); brake ducts: the FIA decision of
# 2020-08-07 (between the 70th Anniversary GP's practice and qualifying)
CONSTRUCTOR_DEDUCTIONS = {(2020, "racing_point"): (15.0, pd.Timestamp("2020-08-07 18:00", tz="UTC"))}


def recorded_share(season, round_, session_type="R"):
    """The points share a past session was actually scored with."""
    return REDUCED.get((season, round_), 1.0) if session_type == "R" else 1.0


def points(season, session_type, position, fastest=None, share=1.0):
    """Points for classified positions (int array, 0 or NaN = not classified or
    disqualified) of one session; `fastest` marks the fastest-lap driver; `share` the
    session's points share (1 = full points; a reduced race gets no fastest-lap point)."""
    position = np.nan_to_num(np.asarray(position, dtype=float), nan=0.0).astype(int)
    table = np.asarray((0,) + tuple(points_table(season, session_type)), dtype=float)
    scored = table[np.where((position >= 1) & (position < len(table)), position, 0)]
    scored = scored * share
    if session_type == "R" and season in FASTEST_LAP_SEASONS and fastest is not None and share == 1.0:
        fastest = np.asarray(fastest, dtype=bool)
        scored = scored + ((fastest & (position >= 1) & (position <= 10)) * 1.0)
    return scored


def _ranked(totals, finishes):
    """Rank by points, then countback: more wins, more second places, ..."""
    positions = sorted(p for p in finishes.position.dropna().unique() if p >= 1)
    counts = (finishes[finishes.position >= 1].groupby(["key", "position"]).size().unstack(fill_value=0)
              .reindex(index=totals.index, columns=positions, fill_value=0))
    order = totals.to_frame("points").join(counts)
    order = order.sort_values(["points", *positions], ascending=False)
    order["rank"] = np.arange(1, len(order) + 1)
    order["wins"] = counts.get(1, pd.Series(0, index=counts.index)).reindex(order.index).fillna(0).astype(int)
    return order[["points", "wins", "rank"]]


def scored(entries):
    """`entries` with each row's points (see `standings` for the columns; "share" defaults
    to full points)."""
    rows = []
    for (season, round_, kind), g in entries.groupby(["season", "round", "session_type"]):
        share = float(g.share.iloc[0]) if "share" in g else 1.0
        rows.append(g.assign(points=points(int(season), kind, g.position.to_numpy(), g.fastest.to_numpy(), share)))
    return pd.concat(rows)


def standings(entries, as_of, carried=None):
    """Driver and constructor standings of one season as of the time `as_of` (UTC).
    `entries`: one row per car per scoring session: season, round, session_type ("R" or
    "S"), driver, team, position (classified position, NaN when not classified or
    disqualified), fastest (bool), optionally share (points share, default 1).
    `carried`: standings so far (the (drivers, constructors) of an earlier call, with
    their Grand Prix finishes and time) to which these entries' points are added. A
    constructor deduction applies when its decision is published by `as_of` and was not
    yet by the carried standings' time."""
    seasons = entries.season.unique()
    if len(seasons) != 1:
        raise ValueError("one season at a time")
    season = int(seasons[0])
    rows = scored(entries)
    grand_prix = rows[rows.session_type == "R"]
    drivers = rows.groupby("driver").points.sum()
    teams = rows.groupby("team").points.sum()
    finishes_d = grand_prix.assign(key=grand_prix.driver)[["key", "position"]]
    finishes_c = grand_prix.assign(key=grand_prix.team)[["key", "position"]]
    as_of = pd.Timestamp(as_of)
    since = None
    if carried is not None:
        (old_d, old_c) = carried
        if old_c.attrs["as_of"] > as_of:
            raise ValueError("carried standings are later than as_of")
        since = old_c.attrs["as_of"]
        drivers = drivers.add(old_d.points, fill_value=0.0)
        teams = teams.add(old_c.points, fill_value=0.0)
        finishes_d = pd.concat([old_d.attrs["finishes"], finishes_d])
        finishes_c = pd.concat([old_c.attrs["finishes"], finishes_c])
    for (s, team), (deduction, published) in CONSTRUCTOR_DEDUCTIONS.items():
        if s == season and published <= as_of and (since is None or published > since):
            teams[team] = teams.get(team, 0.0) - deduction
    out_d, out_c = _ranked(drivers, finishes_d), _ranked(teams, finishes_c)
    for out, finishes in ((out_d, finishes_d), (out_c, finishes_c)):
        out.attrs["finishes"], out.attrs["as_of"] = finishes, as_of
    return out_d, out_c
