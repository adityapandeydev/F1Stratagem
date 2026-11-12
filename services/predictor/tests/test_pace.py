import json

import numpy as np
import pytest
import pandas as pd

from predictor.archive import Archive
from predictor.baselines import Example
from predictor.pace import Filter, Hyper, PaceModel

UTC = "UTC"
CARS = {"fast": -1.0, "mid": 0.0, "slow": 1.0}
DRIVERS = {"a": -0.2, "b": 0.2, "c": -0.1, "d": 0.1, "e": 0.0, "f": 0.3}
LINEUP = [("a", "fast"), ("b", "fast"), ("c", "mid"), ("d", "mid"), ("e", "slow"), ("f", "slow")]


def test_filter_recovers_car_and_driver_pace_from_qualifying_segments():
    rng = np.random.default_rng(1)
    state = Filter(Hyper(noise={"Q": 0.1}, car_drift=1e-4, driver_drift=1e-4, car_prior=2.0, driver_prior=0.5))
    start = pd.Timestamp("2023-01-01", tz=UTC)
    for week in range(30):
        state.advance(start + pd.Timedelta(weeks=week))
        truth = np.array([CARS[c] + DRIVERS[d] for d, c in LINEUP])
        state.update([c for _, c in LINEUP], [d for d, _ in LINEUP], truth + rng.normal(0, 0.1, len(LINEUP)))
    estimate = state.design([c for _, c in LINEUP], [d for d, _ in LINEUP]) @ state.mean
    relative = estimate - estimate.mean()
    truth = np.array([CARS[c] + DRIVERS[d] for d, c in LINEUP])
    assert np.allclose(relative, truth - truth.mean(), atol=0.06)


def write_archive(root, later_time_q1):
    """Two qualifying sessions; the second one is after the forecast cutoff."""
    reference = root / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    rows = []
    for session_id, start in (("2023-01-4", "2023-03-04 15:00"), ("2023-02-4", "2023-03-18 17:00")):
        rows.append({"session_id": session_id, "event_key": session_id[:7], "season": 2023, "session_type": "Q",
                     "scheduled_start_utc": pd.Timestamp(start, tz=UTC),
                     "last_lap_end_utc": pd.Timestamp(start, tz=UTC) + pd.Timedelta(hours=1)})
    pd.DataFrame(rows).to_parquet(reference / "sessions.parquet")
    pd.DataFrame({"event_key": ["2023-01", "2023-02"], "season": [2023, 2023], "is_testing": [False, False],
                  "round": [1, 2]}).to_parquet(reference / "events.parquet")
    entrants = [{"session_id": s, "driver_number": str(i), "driver_id": d, "team_id": c, "lineage_id": c}
                for s in ("2023-01-4", "2023-02-4") for i, (d, c) in enumerate(LINEUP)]
    pd.DataFrame(entrants).to_parquet(reference / "entrants.parquet")
    (reference / "manifest.json").write_text(json.dumps({"built_at_utc": "now"}))
    for session_id, base in (("2023-01-4", 90.0), ("2023-02-4", later_time_q1)):
        folder = root / "sessions" / "2023" / session_id / "normalized"
        folder.mkdir(parents=True, exist_ok=True)
        times = [base * (1 + (CARS[c] + DRIVERS[d]) / 100) for d, c in LINEUP]
        pd.DataFrame({"DriverNumber": [str(i) for i in range(len(LINEUP))], "Position": range(1, 7),
                      "Q1": pd.to_timedelta(times, unit="s"),
                      "Q2": pd.Series([pd.NaT] * 6, dtype="timedelta64[ns]"),
                      "Q3": pd.Series([pd.NaT] * 6, dtype="timedelta64[ns]")}).to_parquet(
            folder / "results.parquet")


def forecast_after_first_session(root):
    archive = Archive(root)
    cutoff = pd.Timestamp("2023-03-10", tz=UTC)
    example = Example(event_key="2023-03", season=2023, session_type="Q", cutoff_label="pre_weekend",
                      cutoff_time=cutoff, target_available_at=cutoff + pd.Timedelta(days=1),
                      driver_numbers=[str(i) for i in range(6)], driver_ids=[d for d, _ in LINEUP],
                      positions=np.arange(1, 7), outcomes=[None] * 6, excluded=np.zeros(6, dtype=bool),
                      band_rank=np.arange(1, 7, dtype=float), lineage_ids=[c for _, c in LINEUP])
    model = PaceModel()
    model.hyper, model.fitted_season = Hyper(), 2023   # skip fitting in this test
    return model.predict(archive.view(cutoff), example, draws=2000)


def test_data_after_the_cutoff_cannot_change_a_forecast(tmp_path):
    write_archive(tmp_path / "one", later_time_q1=90.0)
    write_archive(tmp_path / "two", later_time_q1=200.0)
    first = forecast_after_first_session(tmp_path / "one")
    second = forecast_after_first_session(tmp_path / "two")
    assert np.array_equal(first.position_probs, second.position_probs)
    assert first.position_probs[0, 0] > first.position_probs[5, 0]   # fast car, faster driver leads


def test_team_session_offset_absorbs_practice_flattering():
    """One team is always 0.6% quicker in FP2 than its qualifying pace (e.g. engine
    modes). The offset learns it, so the qualifying estimate is not fooled."""
    rng = np.random.default_rng(2)
    noise = {"Q": 0.1, "FP2": 0.15}
    state = Filter(Hyper(noise=noise, car_drift=1e-4, driver_drift=1e-4, offset_drift=1e-5,
                         car_prior=2.0, driver_prior=0.5, offset_prior=1.0))
    cars, drivers = [c for _, c in LINEUP], [d for d, _ in LINEUP]
    truth = np.array([CARS[c] + DRIVERS[d] for d, c in LINEUP])
    flatter = np.array([-0.6 if c == "fast" else 0.0 for _, c in LINEUP])
    start = pd.Timestamp("2023-01-01", tz=UTC)
    for week in range(40):
        state.advance(start + pd.Timedelta(weeks=week))
        state.update(cars, drivers, truth + flatter + rng.normal(0, 0.15, 6), "FP2")
        state.update(cars, drivers, truth + rng.normal(0, 0.1, 6), "Q")
    offsets = {c: state.mean[state.index[("offset", (c, "FP2"))]] for c in CARS}
    assert offsets["fast"] - np.mean([offsets["mid"], offsets["slow"]]) == pytest.approx(-0.6, abs=0.1)
    estimate = state.design(cars, drivers, "Q") @ state.mean
    assert np.allclose(estimate - estimate.mean(), truth - truth.mean(), atol=0.08)


def test_winter_inflation_is_larger_at_regulation_changes():
    def car_variance_after(year_from, year_to):
        state = Filter(Hyper(car_drift=0.0, driver_drift=0.0, car_prior=1.0, driver_prior=1.0,
                             car_winter=0.5, driver_winter=0.05, regulation_multiplier=3.0))
        state.column("fast", "car")
        state.advance(pd.Timestamp(f"{year_from}-11-01", tz=UTC))
        state.advance(pd.Timestamp(f"{year_to}-03-01", tz=UTC))
        return state.cov[0, 0]
    assert car_variance_after(2020, 2021) == pytest.approx(1.0 + 0.5)       # ordinary winter
    assert car_variance_after(2021, 2022) == pytest.approx(1.0 + 1.5)       # new regulations
    assert car_variance_after(2021, 2021) == pytest.approx(1.0)             # no season boundary


def test_qualifying_objective_counts_only_qualifying_likelihood(tmp_path):
    from predictor.pace import ALL_SESSIONS
    write_archive(tmp_path, later_time_q1=90.0)
    archive = Archive(tmp_path)
    view = archive.view(pd.Timestamp("2023-04-01", tz=UTC))
    sessions = view.sessions[view.sessions.session_type == "Q"].sort_values("scheduled_start_utc")
    everything = PaceModel(ALL_SESSIONS, "all").run(view, Hyper(), sessions)
    qualifying = PaceModel(ALL_SESSIONS, "q", objective="qualifying").run(view, Hyper(), sessions)
    assert everything.objective == pytest.approx(everything.log_likelihood)
    # Two horizons for qualifying: before the weekend and after its practice (here identical).
    assert qualifying.objective == pytest.approx(2 * everything.log_likelihood)
    assert np.allclose(qualifying.mean, everything.mean)


def test_weekend_effect_informs_that_weekend_only():
    """Each event the cars swing around constant long-run pace. Practice reveals the
    swing to that weekend's qualifying, but long-run pace stays put and the next event
    starts from it."""
    rng = np.random.default_rng(3)
    hyper = Hyper(noise={"Q": 0.05, "FP2": 0.05}, car_drift=1e-4, driver_drift=1e-5, offset_drift=1e-6,
                  car_prior=2.0, driver_prior=0.5, offset_prior=0.01, weekend_prior=0.5)
    state = Filter(hyper)
    cars, drivers = [c for _, c in LINEUP], [d for d, _ in LINEUP]
    truth = np.array([CARS[c] + DRIVERS[d] for d, c in LINEUP])
    start = pd.Timestamp("2023-01-01", tz=UTC)
    for event in range(30):
        swing = {c: rng.normal(0, 0.5) for c in CARS}
        weekend = np.array([swing[c] for _, c in LINEUP])
        state.advance(start + pd.Timedelta(weeks=2 * event))
        state.start_event(event)
        state.update(cars, drivers, truth + weekend + rng.normal(0, 0.05, 6), "FP2")
        if event == 29:
            after_practice = state.design(cars, drivers, "Q") @ state.mean
            assert np.allclose(after_practice - after_practice.mean(),
                               (truth + weekend) - (truth + weekend).mean(), atol=0.15)
        state.update(cars, drivers, truth + weekend + rng.normal(0, 0.05, 6), "Q")
    state.start_event(30)
    long_run = state.design(cars, drivers, "Q") @ state.mean
    assert np.allclose(long_run - long_run.mean(), truth - truth.mean(), atol=0.25)
    weekend_columns = [i for i, kind in enumerate(state.kind) if kind == "weekend"]
    assert np.allclose(state.mean[weekend_columns], 0) and np.allclose(
        np.diag(state.cov)[weekend_columns], hyper.weekend_prior ** 2)


def affinity_hyper(**overrides):
    values = dict(noise={"Q": 0.05}, car_drift=1e-5, driver_drift=1e-6, car_prior=2.0, driver_prior=0.5,
                  weekend_prior=0.05, affinity_prior=1.0, affinity_drift=1e-5, affinity_winter=0.01,
                  car_winter=0.01, driver_winter=0.001, regulation_multiplier=3.0)
    values.update(overrides)
    return Hyper(**values)


def test_affinity_persists_between_visits_and_shapes_the_next_forecast():
    """The fast car is always 0.8% slower at 'marina_bay' than elsewhere. Visits alternate
    with another circuit; before the next Marina Bay weekend the forecast expects it."""
    rng = np.random.default_rng(4)
    state = Filter(affinity_hyper())
    cars, drivers = [c for _, c in LINEUP], [d for d, _ in LINEUP]
    truth = np.array([CARS[c] + DRIVERS[d] for d, c in LINEUP])
    penalty = np.array([0.8 if c == "fast" else 0.0 for _, c in LINEUP])
    time = pd.Timestamp("2019-03-01", tz=UTC)
    for event in range(24):
        circuit = "marina_bay" if event % 2 else "monza"
        time += pd.Timedelta(weeks=8)
        state.advance(time)
        state.start_event(event, circuit)
        state.update(cars, drivers, truth + (penalty if circuit == "marina_bay" else 0) + rng.normal(0, 0.05, 6), "Q")
    state.advance(time + pd.Timedelta(weeks=8))
    state.start_event(99, "marina_bay")
    at_marina = state.design(cars, drivers, "Q") @ state.mean
    state.start_event(100, "monza")
    at_monza = state.design(cars, drivers, "Q") @ state.mean
    gap = lambda x: x[0] - x[2]                   # fast car's lead driver vs the mid car's
    assert gap(at_marina) - gap(at_monza) == pytest.approx(0.8, abs=0.15)


def test_regulation_change_inflates_parked_affinity_but_keeps_its_value():
    state = Filter(affinity_hyper(affinity_drift=0.0))
    state.advance(pd.Timestamp("2021-09-01", tz=UTC))
    state.start_event(1, "marina_bay")
    state.design(["fast"], ["a"], "Q")
    index = state.index[("affinity", ("fast", "marina_bay"))]
    state.mean[index], state.cov[index, index] = 0.7, 0.04
    state.advance(pd.Timestamp("2021-10-01", tz=UTC))
    state.start_event(2, "monza")                  # parked
    state.advance(pd.Timestamp("2022-09-01", tz=UTC))
    state.start_event(3, "marina_bay")
    state.design(["fast"], ["a"], "Q")
    index = state.index[("affinity", ("fast", "marina_bay"))]
    assert state.mean[index] == pytest.approx(0.7)                       # value kept
    assert state.cov[index, index] == pytest.approx(0.04 + 0.01 * 3.0)   # 2022 winter x regulation multiplier
