import pandas as pd

from f1data.labels import problem_candidates, retirements


def rows():
    return pd.DataFrame({
        "session": ["Race"] * 5,
        "car": ["1", "2", "3", "4", "5"],
        "grid": [1.0, 3.0, 12.0, 5.0, 2.0],
        "finish_position": [1.0, 14.0, 13.0, 19.0, 20.0],
        "laps_completed": [57, 57, 56, 40, 0],
        "race_laps": [57] * 5,
        "recorded_status": ["Finished", "Finished", "Lapped", "Retired", "Did not start"],
        "classified": ["1", "14", "13", "R", "W"],
    })


def test_retirements_exclude_classified_and_non_starters():
    assert retirements(rows()).car.tolist() == ["4"]


def test_problem_candidates_flag_unexpected_results_only():
    flagged = problem_candidates(rows()).set_index("car").why_flagged
    assert set(flagged.index) == {"2"}
    assert "outside the points" in flagged["2"] and "lost 11 places" in flagged["2"]
