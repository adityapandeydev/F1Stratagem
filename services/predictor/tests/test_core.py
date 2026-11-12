import numpy as np
import pandas as pd
import pytest

from predictor.forecast import Forecast
from predictor.rules import points_for, points_positions
from predictor.simulate import log_likelihood, sample_orders
from predictor.targets import OUTCOMES, classify


def test_points_rules():
    assert points_for(1, 2018, "R") == 25 and points_for(10, 2026, "R") == 1 and points_for(11, 2026, "R") == 0
    assert points_positions(2021, "S") == 3 and points_positions(2022, "S") == 8
    with pytest.raises(ValueError):
        points_positions(2019, "S")


def test_orders_are_permutations_and_follow_strengths():
    rng = np.random.default_rng(0)
    positions = sample_orders(np.log([4.0, 2.0, 1.0, 1.0]), 40000, rng)
    assert all(sorted(row) == [1, 2, 3, 4] for row in positions[:100])
    win = (positions == 1).mean(axis=0)
    assert np.allclose(win, np.array([4, 2, 1, 1]) / 8, atol=0.01)


def test_plackett_luce_log_likelihood():
    strengths = np.array([3.0, 2.0, 1.0])
    expected = np.log(3 / 6) + np.log(2 / 3) + np.log(1 / 1)
    assert log_likelihood(np.log(strengths), np.array([1, 2, 3])) == pytest.approx(expected)


def fields():
    return dict(event_key="2023-01", cutoff_label="after_Q", cutoff_time="t", driver_numbers=["1", "2", "3"],
                driver_ids=["a", "b", "c"], model="test", seed=0)


def test_forecast_from_draws_is_consistent():
    positions = np.array([[1, 2, 3], [2, 1, 3], [1, 3, 2], [3, 2, 1]])
    win, other = OUTCOMES.index("win"), OUTCOMES.index("other_points")
    outcomes = np.where(positions == 1, win, other)
    forecast = Forecast.from_draws(positions=positions, outcomes=outcomes, season=2023, session_type="R", **fields())
    assert forecast.position_probs[0].tolist() == [0.5, 0.25, 0.25]
    assert forecast.top(1).sum() == pytest.approx(1)
    assert forecast.expected_points[0] == pytest.approx((25 + 18 + 25 + 15) / 4)
    lower, upper = forecast.interval()
    assert lower[0] == 1 and upper[0] == 3


def test_forecast_check_rejects_inconsistent_probabilities():
    forecast = Forecast.from_draws(positions=np.array([[1, 2, 3]]), season=2023, session_type="Q", **fields())
    forecast.position_probs = np.full((3, 3), 0.5)
    with pytest.raises(ValueError):
        forecast.check()


def results(rows):
    return pd.DataFrame(rows, columns=["DriverNumber", "Position", "ClassifiedPosition", "Status"])


def test_race_outcomes():
    frame = classify(results([
        ("1", 1, "1", "Finished"), ("2", 8, "8", "+1 Lap"), ("3", 12, "12", "Lapped"),
        ("4", 17, "R", "Engine"), ("5", 18, "R", "Collision"), ("6", 19, "R", "Retired"),
        ("7", 20, "W", "Did not start"), ("8", 16, "R", "Something new"),
    ]), 2023, "R")
    outcome = dict(zip(frame.driver_number, frame.outcome))
    assert outcome == {"1": "win", "2": "other_points", "3": "outside_points", "4": "mechanical",
                       "5": "incident_or_other", "6": "retired_unknown", "7": "excluded", "8": "unmapped"}
    assert frame.set_index("driver_number").excluded.to_dict()["7"]


def test_sprint_points_positions_follow_the_season():
    frame = classify(results([("1", 1, "1", "Finished"), ("2", 4, "4", "Finished")]), 2021, "S")
    assert dict(zip(frame.driver_number, frame.outcome)) == {"1": "win", "2": "outside_points"}


def test_qualifying_has_positions_but_no_outcomes():
    frame = classify(results([("1", 1, "", ""), ("2", 2, "", ""), ("3", None, "", "")]), 2023, "Q")
    assert frame.position.tolist() == [1, 2] and frame.outcome.isna().all()
