"""Sporting rules that change by season: points and points-paying positions.

Values were checked against the archived classifications for 2018–2026. The
fastest-lap bonus point (2019–2024) and one-off reduced-points races (Belgium
2021) are not part of expected points.
"""

RACE_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
SPRINT_POINTS = {2021: (3, 2, 1)}
SPRINT_POINTS_DEFAULT = (8, 7, 6, 5, 4, 3, 2, 1)
FIRST_SPRINT_SEASON = 2021
# Seasons that began under new technical regulations.
REGULATION_CHANGES = frozenset({2014, 2017, 2022, 2026})


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
