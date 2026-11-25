import numpy as np
import pandas as pd
import pytest

from predictor.championship import points, recorded_share, standings

END = pd.Timestamp("2030-01-01", tz="UTC")


def test_grand_prix_points_fastest_lap_and_unclassified():
    got = points(2022, "R", [1, 2, 10, 11, np.nan], fastest=[False, False, True, False, False])
    assert got.tolist() == [25, 18, 2, 0, 0]          # P10 with the fastest lap: 1 + 1
    got = points(2022, "R", [11, 1], fastest=[True, False])
    assert got.tolist() == [0, 25]                    # fastest lap outside the top ten: nothing
    got = points(2018, "R", [10], fastest=[True])
    assert got.tolist() == [1]                        # no fastest-lap point before 2019


def test_sprint_points_by_season_and_no_fastest_lap():
    assert points(2021, "S", [1, 3, 4], fastest=[True, False, False]).tolist() == [3, 1, 0]
    assert points(2023, "S", [1, 8, 9]).tolist() == [8, 1, 0]
    with pytest.raises(ValueError):
        points(2020, "S", [1])


def test_reduced_points_race_halves_and_drops_fastest_lap():
    share = recorded_share(2021, 12)
    got = points(2021, "R", [1, 2, 10], fastest=[False, False, True], share=share)
    assert share == 0.5 and got.tolist() == [12.5, 9.0, 0.5]


def test_simulated_race_at_a_reduced_round_scores_full_points():
    # a forecast before Spa 2021 simulates a full race there: 25 and the fastest-lap point
    got = points(2021, "R", [1, 10], fastest=[True, False])
    assert got.tolist() == [26, 1]


def _entries(rows):
    return pd.DataFrame(rows, columns=["season", "round", "session_type", "driver", "team", "position", "fastest"])


def test_countback_on_grand_prix_results_only():
    # level on 52 points: b has a Grand Prix win, a two sprint wins and no Grand Prix win;
    # counting sprints would put a first, the rules put b first
    e = _entries([
        (2023, 1, "R", "a", "x", 2, False), (2023, 1, "R", "b", "y", 1, False),
        (2023, 1, "S", "a", "x", 1, False), (2023, 1, "S", "b", "y", 3, False),
        (2023, 2, "R", "a", "x", 2, False), (2023, 2, "R", "b", "y", 3, False),
        (2023, 2, "S", "a", "x", 1, False), (2023, 2, "S", "b", "y", 3, False),
    ])
    drivers, _ = standings(e, END)
    assert drivers.loc["a", "points"] == drivers.loc["b", "points"] == 52
    assert drivers.loc["b", "rank"] == 1 and drivers.loc["a", "rank"] == 2
    assert drivers.loc["b", "wins"] == 1 and drivers.loc["a", "wins"] == 0


def test_carried_standings_equal_the_whole_season():
    rng = np.random.default_rng(0)
    rows = []
    for round_ in range(1, 7):
        order = rng.permutation(6) + 1
        fastest = rng.integers(6)
        for i, d in enumerate("abcdef"):
            rows.append((2022, round_, "R", d, "t" + str(i // 2), order[i], i == fastest))
    e = _entries(rows)
    whole = standings(e, END)
    for split in range(1, 6):
        carried = standings(e[e["round"] > split], END, carried=standings(e[e["round"] <= split], END))
        for a, b in zip(whole, carried):
            pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_constructor_deduction_from_its_publication_once():
    e = _entries([(2020, 1, "R", "p", "racing_point", 4, False), (2020, 1, "R", "q", "other", 1, False)])
    july = pd.Timestamp("2020-07-05 23:59", tz="UTC")
    _, constructors = standings(e, july)
    assert constructors.loc["racing_point", "points"] == 12          # decision published 2020-08-07
    _, constructors = standings(e, END)
    assert constructors.loc["racing_point", "points"] == 12 - 15
    # carried from July: the later ruling is applied once when the standings pass its date
    later = e.assign(round=2)
    _, c = standings(later, pd.Timestamp("2020-07-12 23:59", tz="UTC"), carried=standings(e, july))
    assert c.loc["racing_point", "points"] == 24
    _, c = standings(later, END, carried=standings(e, july))
    assert c.loc["racing_point", "points"] == 24 - 15
    _, c = standings(later, END, carried=standings(e, END))
    assert c.loc["racing_point", "points"] == 24 - 15
    with pytest.raises(ValueError):
        standings(later, july, carried=standings(e, END))
