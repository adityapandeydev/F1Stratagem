import json

import numpy as np
import pandas as pd
import pytest

from predictor.archive import Archive, CutoffViolation
from predictor.baselines import Example, RankBaseline
from predictor.evaluation import score, walk_forward
from predictor.forecast import Forecast
from predictor.targets import HoldoutError, session_target

UTC = "UTC"


def ts(text):
    return pd.Timestamp(text, tz=UTC)


@pytest.fixture
def archive(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    sessions = pd.DataFrame([
        {"session_id": "2023-01-4", "event_key": "2023-01", "season": 2023, "session_type": "Q",
         "scheduled_start_utc": ts("2023-03-04 15:00"), "last_lap_end_utc": ts("2023-03-04 16:00")},
        {"session_id": "2023-01-5", "event_key": "2023-01", "season": 2023, "session_type": "R",
         "scheduled_start_utc": ts("2023-03-05 15:00"), "last_lap_end_utc": pd.NaT},
        {"session_id": "2025-01-5", "event_key": "2025-01", "season": 2025, "session_type": "R",
         "scheduled_start_utc": ts("2025-03-16 04:00"), "last_lap_end_utc": ts("2025-03-16 05:45")},
    ])
    sessions.to_parquet(reference / "sessions.parquet")
    pd.DataFrame({"event_key": ["2023-01", "2025-01"], "season": [2023, 2025], "is_testing": [False, False],
                  "round": [1, 1]}).to_parquet(reference / "events.parquet")
    pd.DataFrame({"session_id": ["2023-01-4"] * 2 + ["2023-01-5"] * 2, "driver_number": ["1", "2"] * 2,
                  "driver_id": ["a", "b"] * 2, "team_id": ["x", "y"] * 2}).to_parquet(reference / "entrants.parquet")
    (reference / "manifest.json").write_text(json.dumps({"built_at_utc": "now"}))
    for session_id in ("2023-01-4", "2023-01-5"):
        folder = tmp_path / "sessions" / "2023" / session_id / "normalized"
        folder.mkdir(parents=True)
        pd.DataFrame({"DriverNumber": ["1", "2"], "Position": [2.0, 1.0], "ClassifiedPosition": ["2", "1"],
                      "Status": ["Finished", "Finished"], "Points": [18.0, 25.0]}).to_parquet(folder / "results.parquet")
    return Archive(tmp_path)


def test_availability_uses_last_lap_or_nominal_duration(archive):
    available = archive.sessions.set_index("session_id").available_at_utc
    assert available["2023-01-4"] == ts("2023-03-04 16:30")          # last lap + 30 min
    assert available["2023-01-5"] == ts("2023-03-05 19:00")          # start + 3 h + 1 h


def test_view_refuses_data_after_its_cutoff(archive):
    after_quali = next(c for c in archive.cutoffs("2023-01") if c.label == "after_Q")
    view = archive.view(after_quali)
    assert view.results("2023-01-4").Position.tolist() == [2.0, 1.0]
    with pytest.raises(CutoffViolation):
        view.results("2023-01-5")
    with pytest.raises(ValueError):
        archive.view(pd.Timestamp("2023-03-06"))                        # naive time


def test_cutoffs_in_session_order(archive):
    assert [c.label for c in archive.cutoffs("2023-01")] == ["pre_weekend", "after_Q", "after_R"]


def test_standings_only_count_available_races(archive):
    assert archive.view(ts("2023-03-05 12:00")).standings(2023).empty
    assert archive.view(ts("2023-03-06")).standings(2023).to_dict() == {"b": 25.0, "a": 18.0}


def test_held_out_seasons_need_explicit_permission(archive):
    with pytest.raises(HoldoutError):
        session_target(archive, "2025-01", "R")
    assert session_target(archive, "2023-01", "R").position.tolist() == [1, 2]


def example(positions, ranks, available, cutoff):
    n = len(positions)
    return Example(event_key=f"e{available}", season=2023, session_type="Q", cutoff_label="after_FP3",
                   cutoff_time=pd.Timestamp(cutoff, tz=UTC), target_available_at=pd.Timestamp(available, tz=UTC),
                   driver_numbers=[str(i) for i in range(n)], driver_ids=list("abcdef")[:n],
                   positions=np.array(positions), outcomes=[None] * n, excluded=np.zeros(n, dtype=bool),
                   ranks={"qualifying": np.array(ranks, dtype=float), "uniform": np.zeros(n)})


def test_perfect_forecast_scores_zero_error():
    ex = example([1, 2, 3], [1, 2, 3], "2023-01-01", "2023-01-01")
    forecast = Forecast.from_draws(positions=np.array([[1, 2, 3]]), season=2023, session_type="Q",
                                   event_key="e", cutoff_label="c", cutoff_time="t", driver_numbers=["0", "1", "2"],
                                   driver_ids=["a", "b", "c"], model="perfect", seed=0)
    row = score(forecast, ex)
    assert row["rps"] == 0 and row["position_mae"] == 0 and row["coverage_80"] == 1 and row["winner_log_loss"] == 0


def test_beta_is_fitted_only_on_earlier_events_and_reflects_informative_ranks():
    history = [example([1, 2, 3, 4], [1, 2, 3, 4], f"2023-0{i}-01", f"2023-0{i}-01") for i in range(1, 8)]
    model = RankBaseline("qualifying").fit(history)
    assert model.beta > 0.5
    first = history[0]
    scores = walk_forward([first], ["qualifying"])
    assert scores.beta.iloc[0] == 0.0          # nothing finished before the first cutoff
