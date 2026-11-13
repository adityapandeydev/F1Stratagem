"""Sporting rules that change by season: points and points-paying positions.

Values were checked against the archived classifications for 2018–2026. The
fastest-lap bonus point (2019–2024) and one-off reduced-points races (Belgium
2021) are not part of expected points.
"""

RACE_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
SPRINT_POINTS = {2021: (3, 2, 1)}
SPRINT_POINTS_DEFAULT = (8, 7, 6, 5, 4, 3, 2, 1)
FIRST_SPRINT_SEASON = 2021
# Seasons that began under new technical regulations (2009: slicks and the aero reset;
# 2014: hybrid power units; 2017: wider cars and tyres; 2022: ground effect; 2026).
REGULATION_CHANGES = frozenset({2009, 2014, 2017, 2022, 2026})


def low_fuel_segments(season):
    """Qualifying segments run on low fuel, so lap times compare car and driver pace.

    2003-2005 qualifying laps were run on race fuel, as was Q3 from 2006 to 2009
    (in the data Q3 is 0.6-1.9% slower than Q2 then, 0.2-0.4% faster from 2010).
    The FastF1 archive (2018+) has every segment on low fuel."""
    if season >= 2010:
        return ("Q1", "Q2", "Q3")
    if season >= 2006:
        return ("Q1", "Q2")
    return ()


def points_table(season, session_type):
    """Points by finishing position (index 0 = P1) for a race ('R') or sprint ('S')."""
    if session_type == "R":
        return RACE_POINTS
    if session_type == "S":
        if season < FIRST_SPRINT_SEASON:
            raise ValueError(f"no sprints in {season}")
        return SPRINT_POINTS.get(season, SPRINT_POINTS_DEFAULT)
    raise ValueError(f"no points for session type {session_type}")


def points_positions(season, session_type):
    """Number of positions that score points."""
    return len(points_table(season, session_type))


def points_for(position, season, session_type):
    table = points_table(season, session_type)
    return table[position - 1] if 1 <= position <= len(table) else 0
