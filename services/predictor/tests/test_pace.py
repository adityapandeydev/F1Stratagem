import copy
import json
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from predictor.archive import Archive
from predictor.baselines import Example
from predictor.pace import ALL_SESSIONS, Hyper, PaceModel
from predictor import program
from predictor.program import Recorder

UTC = "UTC"
CARS = {"fast": -1.0, "mid": 0.0, "slow": 1.0}
DRIVERS = {"a": -0.2, "b": 0.2, "c": -0.1, "d": 0.1, "e": 0.0, "f": 0.3}
LINEUP = [("a", "fast"), ("b", "fast"), ("c", "mid"), ("d", "mid"), ("e", "slow"), ("f", "slow")]


def estimate(recorder, hyper, cars, drivers, session_type="Q"):
    """Mean pace of each (car, driver) under `hyper` at the recorder's current point."""
    rows = recorder.columns(cars, drivers, session_type)
    state = recorder.replay(hyper)
    return state.design(rows) @ state.mean, state


def test_filter_recovers_car_and_driver_pace_from_qualifying_segments():
    rng = np.random.default_rng(1)
    hyper = Hyper(noise={"Q": 0.1}, car_drift=1e-4, driver_drift=1e-4, car_prior=2.0, driver_prior=0.5)
    state = Recorder(("Q",))
    start = pd.Timestamp("2023-01-01", tz=UTC)
    for week in range(30):
        state.advance(start + pd.Timedelta(weeks=week))
        truth = np.array([CARS[c] + DRIVERS[d] for d, c in LINEUP])
        state.update([c for _, c in LINEUP], [d for d, _ in LINEUP], truth + rng.normal(0, 0.1, len(LINEUP)))
    pace, _ = estimate(state, hyper, [c for _, c in LINEUP], [d for d, _ in LINEUP])
    relative = pace - pace.mean()
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
    hyper = Hyper(noise=noise, car_drift=1e-4, driver_drift=1e-4, offset_drift=1e-5,
                  car_prior=2.0, driver_prior=0.5, offset_prior=1.0)
    state = Recorder(("Q", "FP2"))
    cars, drivers = [c for _, c in LINEUP], [d for d, _ in LINEUP]
    truth = np.array([CARS[c] + DRIVERS[d] for d, c in LINEUP])
    flatter = np.array([-0.6 if c == "fast" else 0.0 for _, c in LINEUP])
    start = pd.Timestamp("2023-01-01", tz=UTC)
    for week in range(40):
        state.advance(start + pd.Timedelta(weeks=week))
        state.update(cars, drivers, truth + flatter + rng.normal(0, 0.15, 6), "FP2")
        state.update(cars, drivers, truth + rng.normal(0, 0.1, 6), "Q")
    pace, result = estimate(state, hyper, cars, drivers)
    offsets = {c: result.mean[result.index[("offset", (c, "FP2"))]] for c in CARS}
    assert offsets["fast"] - np.mean([offsets["mid"], offsets["slow"]]) == pytest.approx(-0.6, abs=0.1)
    assert np.allclose(pace - pace.mean(), truth - truth.mean(), atol=0.08)


def test_winter_inflation_is_larger_at_regulation_changes():
    def car_variance_after(year_from, year_to):
        hyper = Hyper(car_drift=0.0, driver_drift=0.0, car_prior=1.0, driver_prior=1.0,
                      car_winter=0.5, driver_winter=0.05, regulation_multiplier=3.0)
        state = Recorder(("Q",))
        state.column("fast", "car")
        state.advance(pd.Timestamp(f"{year_from}-11-01", tz=UTC))
        state.advance(pd.Timestamp(f"{year_to}-03-01", tz=UTC))
        return state.replay(hyper).cov[0, 0]
    assert car_variance_after(2020, 2021) == pytest.approx(1.0 + 0.5)       # ordinary winter
    assert car_variance_after(2021, 2022) == pytest.approx(1.0 + 1.5)       # new regulations
    assert car_variance_after(2021, 2021) == pytest.approx(1.0)             # no season boundary


def test_qualifying_objective_counts_only_qualifying_likelihood(tmp_path):
    write_archive(tmp_path, later_time_q1=90.0)
    archive = Archive(tmp_path)
    view = archive.view(pd.Timestamp("2023-04-01", tz=UTC))
    sessions = view.sessions[view.sessions.session_type == "Q"].sort_values("scheduled_start_utc")
    everything = PaceModel(ALL_SESSIONS, "all").run(view, Hyper(), sessions)
    qualifying = PaceModel(ALL_SESSIONS, "q", objective="qualifying").run(view, Hyper(), sessions)
    # Two horizons for qualifying: before the weekend and after its practice (here identical).
    assert qualifying.horizons["before_weekend"] == pytest.approx(everything.objective)
    assert qualifying.objective == pytest.approx(2 * everything.objective)
    assert np.allclose(qualifying.mean, everything.mean)


def test_weekend_effect_informs_that_weekend_only():
    """Each event the cars swing around constant long-run pace. Practice reveals the
    swing to that weekend's qualifying, but long-run pace stays put and the next event
    starts from it."""
    rng = np.random.default_rng(3)
    hyper = Hyper(noise={"Q": 0.05, "FP2": 0.05}, car_drift=1e-4, driver_drift=1e-5, offset_drift=1e-6,
                  car_prior=2.0, driver_prior=0.5, offset_prior=0.01, weekend_prior=0.5)
    state = Recorder(("Q", "FP2"), weekend=True)
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
            after_practice, _ = estimate(state, hyper, cars, drivers)
            assert np.allclose(after_practice - after_practice.mean(),
                               (truth + weekend) - (truth + weekend).mean(), atol=0.15)
        state.update(cars, drivers, truth + weekend + rng.normal(0, 0.05, 6), "Q")
    state.start_event(30)
    long_run, result = estimate(state, hyper, cars, drivers)
    assert np.allclose(long_run - long_run.mean(), truth - truth.mean(), atol=0.25)
    weekend_columns = [i for i, kind in enumerate(result.kind) if kind == "weekend"]
    assert np.allclose(result.mean[weekend_columns], 0) and np.allclose(
        np.diag(result.cov)[weekend_columns], hyper.weekend_prior ** 2)


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
    hyper = affinity_hyper()
    state = Recorder(("Q",), weekend=True, affinity=True)
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
    at_marina, _ = estimate(state, hyper, cars, drivers)
    state.start_event(100, "monza")
    at_monza, _ = estimate(state, hyper, cars, drivers)
    gap = lambda x: x[0] - x[2]                   # fast car's lead driver vs the mid car's
    assert gap(at_marina) - gap(at_monza) == pytest.approx(0.8, abs=0.15)


def test_regulation_change_inflates_parked_affinity_but_keeps_its_value():
    hyper = affinity_hyper(affinity_drift=0.0)
    key = ("affinity", ("fast", "marina_bay"))
    state = Recorder(("Q",), weekend=True, affinity=True)
    state.advance(pd.Timestamp("2021-09-01", tz=UTC))
    state.start_event(1, "marina_bay")
    state.update(["fast", "mid"], ["a", "c"], np.array([90.0, 90.9]))
    visited = state.replay(hyper)
    value, variance = visited.mean[visited.index[key]], visited.cov[visited.index[key]][visited.index[key]]
    assert value != 0
    state.advance(pd.Timestamp("2021-10-01", tz=UTC))
    state.start_event(2, "monza")                  # parked
    state.advance(pd.Timestamp("2022-09-01", tz=UTC))
    state.start_event(3, "marina_bay")
    state.columns(["fast"], ["a"], "Q")
    resumed = state.replay(hyper)
    index = resumed.index[key]
    assert resumed.mean[index] == pytest.approx(value)                              # value kept
    assert resumed.cov[index, index] == pytest.approx(variance + 0.01 * 3.0)        # 2022 winter x regulation multiplier


def test_optimiser_reaches_the_minimum_of_a_curved_valley():
    from predictor.pace import _optimise

    def rosenbrock(x):
        return float(sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    x, value, converged = _optimise(rosenbrock, np.full(8, -1.5))
    assert converged
    assert value < 1e-3
    assert np.allclose(x, 1.0, atol=0.05)


def write_circuit_archive(root):
    """Three weekends (circuits A, B, A) across a regulation change, with practice; a fourth (B) ahead."""
    reference = root / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    events = [("2021-01", "A", "2021-07-02"), ("2021-02", "B", "2021-07-16"), ("2022-01", "A", "2022-03-18")]
    sessions, entrants = [], []
    for event, _, day in events:
        for number, (kind, hour) in enumerate((("FP1", 10), ("Q", 15)), start=1):
            session_id = f"{event}-{number}"
            start = pd.Timestamp(f"{day} {hour}:00", tz=UTC)
            sessions.append({"session_id": session_id, "event_key": event, "season": int(event[:4]),
                             "session_type": kind, "scheduled_start_utc": start,
                             "last_lap_end_utc": start + pd.Timedelta(hours=1)})
            entrants += [{"session_id": session_id, "driver_number": str(i), "driver_id": d, "team_id": c,
                          "lineage_id": c} for i, (d, c) in enumerate(LINEUP)]
            folder = root / "sessions" / event[:4] / session_id / "normalized"
            folder.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(len(sessions))
            times = [90 * (1 + (CARS[c] + DRIVERS[d] + rng.normal(0, 0.3)) / 100) for d, c in LINEUP]
            if kind == "Q":
                pd.DataFrame({"DriverNumber": [str(i) for i in range(6)], "Position": range(1, 7),
                              "Q1": pd.to_timedelta(times, unit="s"),
                              "Q2": pd.Series([pd.NaT] * 6, dtype="timedelta64[ns]"),
                              "Q3": pd.Series([pd.NaT] * 6, dtype="timedelta64[ns]")}).to_parquet(folder / "results.parquet")
            else:
                pd.DataFrame({"DriverNumber": [str(i) for i in range(6)], "Position": range(1, 7)}).to_parquet(
                    folder / "results.parquet")
                pd.DataFrame({"DriverNumber": [str(i) for i in range(6)], "LapTime": pd.to_timedelta(times, unit="s"),
                              "PitInTime": pd.Series([pd.NaT] * 6, dtype="timedelta64[ns]"),
                              "PitOutTime": pd.Series([pd.NaT] * 6, dtype="timedelta64[ns]"),
                              "Deleted": [False] * 6}).to_parquet(folder / "laps.parquet")
    pd.DataFrame(sessions).to_parquet(reference / "sessions.parquet")
    # a fourth weekend at B with no sessions yet: forecast before it, B's parked affinity resumes
    pd.DataFrame({"event_key": [e for e, _, _ in events] + ["2022-02"], "season": [2021, 2021, 2022, 2022],
                  "is_testing": False, "round": [1, 2, 1, 2], "circuit_id": [c for _, c, _ in events] + ["B"]}
                 ).to_parquet(reference / "events.parquet")
    pd.DataFrame(entrants).to_parquet(reference / "entrants.parquet")
    (reference / "manifest.json").write_text(json.dumps({"built_at_utc": "now"}))


GOLDEN = json.loads((Path(__file__).parent / "data" / "pace_golden.json").read_text(encoding="utf-8"))
GOLDEN_SETTINGS = dict(car_winter=0.3, driver_winter=0.04, regulation_multiplier=2.5, weekend_prior=0.3,
                       affinity_prior=0.25, affinity_drift=0.002, affinity_winter=0.05, car_drift=0.01)
NO_AFFINITY = dict(affinity_prior=0.0, affinity_drift=0.0, affinity_winter=0.0)
GOLDEN_CASES = {
    "pace3wa": (dict(winter=True, weekend=True, affinity=True), GOLDEN_SETTINGS),
    "pace3w": (dict(winter=True, weekend=True), {**GOLDEN_SETTINGS, **NO_AFFINITY}),
    "pace3q": (dict(winter=True, objective="qualifying"), {**GOLDEN_SETTINGS, **NO_AFFINITY, "weekend_prior": 0.0}),
}
GOLDEN_EXAMPLES = {  # event, cutoff label, cutoff time
    "before_B": ("2022-02", "pre_weekend", "2022-03-25 00:00"),
    "after_FP1_A": ("2022-01", "after_FP1", "2022-03-18 12:00"),
}


def golden_example(event, label, cutoff):
    cutoff = pd.Timestamp(cutoff, tz=UTC)
    drivers = [d for d, _ in LINEUP] + [None, "g"]          # an unknown driver, a new driver in an unknown car
    cars = [c for _, c in LINEUP] + ["mid", None]
    return Example(event_key=event, season=int(event[:4]), session_type="Q", cutoff_label=label, cutoff_time=cutoff,
                   target_available_at=cutoff + pd.Timedelta(days=1), driver_numbers=[str(i) for i in range(8)],
                   driver_ids=drivers, positions=np.arange(1, 9), outcomes=[None] * 8,
                   excluded=np.zeros(8, dtype=bool), band_rank=np.arange(1, 9, dtype=float), lineage_ids=cars)


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_native_filter_reproduces_the_python_filter(tmp_path, name):
    """Reference values written by the Python filter before it was replaced by the
    native one (tests/data/pace_golden.json): log-likelihood, final state and
    forecast distributions, across parking, resuming, weekend resets, a regulation
    winter, unknown entrants and the qualifying objective's two tracks."""
    expected = GOLDEN[name]
    write_circuit_archive(tmp_path)
    archive = Archive(tmp_path)
    options, settings = GOLDEN_CASES[name]
    model, hyper = PaceModel(ALL_SESSIONS, name, **options), Hyper(**settings)
    view = archive.view(pd.Timestamp("2023-01-01", tz=UTC))
    sessions = model.sessions(view)
    state = model.run(view, hyper, sessions)
    assert state.objective == pytest.approx(expected["objective"], rel=1e-12)
    assert model.objective_function(view, sessions, hyper)(hyper) == pytest.approx(expected["objective"], rel=1e-12)
    if options.get("objective") == "qualifying":
        for horizon, value in expected["horizons"].items():
            assert state.horizons[horizon] == pytest.approx(value, rel=1e-12)
    assert [repr(k) for k in state.keys] == expected["keys"]
    assert np.allclose(state.mean, expected["mean"], rtol=0, atol=1e-9)
    assert np.allclose(state.cov, expected["cov"], rtol=0, atol=1e-9)
    model.hyper = hyper
    for label, args in GOLDEN_EXAMPLES.items():
        example = golden_example(*args)
        mean, cov = model.distribution(archive.view(example.cutoff_time), example)
        assert np.allclose(mean, expected["forecasts"][label]["mean"], rtol=0, atol=1e-9)
        assert np.allclose(cov, expected["forecasts"][label]["cov"], rtol=0, atol=1e-9)


def test_recorded_program_exercises_every_operation(tmp_path):
    write_circuit_archive(tmp_path)
    view = Archive(tmp_path).view(pd.Timestamp("2023-01-01", tz=UTC))
    model = PaceModel(ALL_SESSIONS, winter=True, weekend=True, affinity=True, objective="qualifying")
    p = model.record(view, model.sessions(view), True, True).program()
    assert {program.ADVANCE, program.ADD, program.PARK, program.RESET, program.UPDATE, program.SNAPSHOT} \
        <= set(p.ops[:, 0])
    assert ((p.ops[:, 0] == program.ADD) & (p.ops[:, 3] >= 0)).any()      # a parked affinity resumed
    assert ((p.ops[:, 0] == program.ADD) & (p.ops[:, 1] == 1)).any()      # the before-weekend track adds states


def test_settings_that_switch_a_recorded_term_are_rejected():
    state = Recorder(("Q",), weekend=True)
    state.update(["fast", "mid"], ["a", "c"], np.array([90.0, 90.5]))
    with pytest.raises(ValueError):
        state.replay(Hyper())


def write_grid_archive(root):
    """One qualifying session: 20 drivers, the same lap times in Q1, Q2 and Q3."""
    reference = root / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2023-03-04 15:00", tz=UTC)
    pd.DataFrame([{"session_id": "2023-01-4", "event_key": "2023-01", "season": 2023, "session_type": "Q",
                   "scheduled_start_utc": start, "last_lap_end_utc": start + pd.Timedelta(hours=1)}]).to_parquet(
        reference / "sessions.parquet")
    pd.DataFrame({"event_key": ["2023-01"], "season": [2023], "is_testing": [False], "round": [1]}).to_parquet(
        reference / "events.parquet")
    pd.DataFrame([{"session_id": "2023-01-4", "driver_number": str(i), "driver_id": f"d{i}", "team_id": f"c{i // 2}",
                   "lineage_id": f"c{i // 2}"} for i in range(20)]).to_parquet(reference / "entrants.parquet")
    (reference / "manifest.json").write_text(json.dumps({"built_at_utc": "now"}))
    folder = root / "sessions" / "2023" / "2023-01-4" / "normalized"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    base = 90 + rng.normal(0, 0.5, 20)
    segments = {s: pd.to_timedelta(base, unit="s") for s in ("Q1", "Q2", "Q3")}
    pd.DataFrame({"DriverNumber": [str(i) for i in range(20)], "Position": range(1, 21), **segments}).to_parquet(
        folder / "results.parquet")


def test_a_numerically_failing_candidate_is_rejected_not_fatal(tmp_path):
    """At noise near the lower bound (exp(-20)) the covariance update loses its
    precision and an innovation covariance stops being positive definite. The
    optimiser reaches such candidates from an ordinary start; the fit must reject
    them and keep the best finite settings."""
    write_grid_archive(tmp_path)
    view = Archive(tmp_path).view(pd.Timestamp("2023-04-01", tz=UTC))
    model = PaceModel()
    sessions = model.sessions(view)
    recorder = model.record(view, sessions, False, False)
    with pytest.raises(ArithmeticError):
        recorder.objective(Hyper(noise={"Q": np.exp(-20)}))
    with pytest.raises(ArithmeticError):                     # a forecast never reads a failed state
        recorder.replay(Hyper(noise={"Q": np.exp(-20)}))
    start = Hyper()
    hyper = model.fit_hyper(view, sessions, start)
    assert model.last_fit["rejected"] > 0
    assert np.isfinite(model.last_fit["objective"])
    assert model.last_fit["objective"] >= recorder.objective(start)
    assert model.last_fit["objective"] == pytest.approx(recorder.objective(hyper))


def write_history(root):
    """Two qualifying sessions of 2017 in the reference history (all segments low fuel)."""
    rows = []
    for event, circuit, day in (("2017-01", "A", "2017-03-25"), ("2017-02", "B", "2017-04-08")):
        rng = np.random.default_rng(int(event[-1]))
        for d, c in LINEUP:
            seconds = 90 * (1 + (CARS[c] + DRIVERS[d] + rng.normal(0, 0.3, 3)) / 100)
            rows.append({"event_key": event, "season": 2017, "circuit_id": circuit, "driver_id": d, "lineage_id": c,
                         "q1_seconds": seconds[0], "q2_seconds": seconds[1], "q3_seconds": seconds[2],
                         "start_utc": pd.Timestamp(f"{day} 06:00", tz=UTC)})
    pd.DataFrame(rows).to_parquet(root / "reference" / "qualifying_history.parquet")


@pytest.mark.parametrize("mode", ["shared", "noise", "state", "weighted"])
def test_history_modes_read_history_as_qualifying_without_an_offset(tmp_path, mode):
    write_circuit_archive(tmp_path)
    write_history(tmp_path)
    view = Archive(tmp_path).view(pd.Timestamp("2023-01-01", tz=UTC))
    model = PaceModel(ALL_SESSIONS, winter=True, weekend=True, history=mode)
    sessions = model.sessions(view)
    history = sessions[sessions.session_id.str.startswith("history:")]
    assert len(history) == 2 and set(history.session_type) == {"H" if mode in ("noise", "weighted") else "Q"}
    recorder = model.record(view, sessions, True, False)
    assert not any(kind == "offset" and key[1] in ("Q", "H") for kind, key in recorder.keys)
    if mode in ("noise", "weighted"):   # historical rows read car and driver only
        assert all(len(recorder.entries[a:b]) == 2 for a, b in zip(recorder.row_starts[:36], recorder.row_starts[1:37]))
    updates = recorder.program().ops
    updates = updates[updates[:, 0] == program.UPDATE]
    counted = updates[:, 3]
    groups = 2 * 3                                     # two sessions, three segments each
    assert list(counted[:groups]) == [0 if mode in ("state", "weighted") else 1] * groups
    assert counted[groups:].all()
    hyper = model.fit_hyper(view, sessions, Hyper(**GOLDEN_SETTINGS | NO_AFFINITY))
    assert ("H" in model.last_fit["fitted"]) == (mode in ("noise", "weighted"))
    assert np.isfinite(model.last_fit["objective"]) and hyper.noise["Q"] > 0


def test_state_only_history_fits_only_what_the_counted_sessions_identify(tmp_path):
    write_circuit_archive(tmp_path)
    write_history(tmp_path)                                              # 2017 at circuits A and B
    view = Archive(tmp_path).view(pd.Timestamp("2022-01-01", tz=UTC))   # counted data: 2021 at A and B
    fitted = {}
    for mode in (False, "shared", "state"):
        model = PaceModel(ALL_SESSIONS, winter=True, weekend=True, affinity=True, history=mode)
        model.fit_hyper(view, model.sessions(view), Hyper(**GOLDEN_SETTINGS))
        fitted[mode] = set(model.last_fit["fitted"])
    # without history: one season, no season start crossed, every circuit seen once
    assert not fitted[False] & {"car_winter", "driver_winter", "affinity_prior"}
    # shared history is counted: it identifies everything, its own first sights included
    assert {"car_winter", "driver_winter", "car_prior", "driver_prior", "affinity_prior"} <= fitted["shared"]
    # state-only history: the 2021 sessions revisit history's circuits, so the affinity is
    # fitted; they cross no season start and show no new car or driver, so winter terms and
    # priors keep their starting values (regularisation: from history alone they are weak)
    assert {"affinity_prior", "affinity_drift"} <= fitted["state"]
    assert not fitted["state"] & {"car_winter", "driver_winter", "affinity_winter", "car_prior", "driver_prior",
                                  "regulation_multiplier"}


def write_history_at(root, circuits):
    rows = []
    for i, circuit in enumerate(circuits):
        for d, c in LINEUP:
            rows.append({"event_key": f"2017-{i + 1:02d}", "season": 2017, "circuit_id": circuit, "driver_id": d,
                         "lineage_id": c, "q1_seconds": 90 * (1 + (CARS[c] + DRIVERS[d]) / 100), "q2_seconds": np.nan,
                         "q3_seconds": np.nan, "start_utc": pd.Timestamp(f"2017-0{i + 4}-01 06:00", tz=UTC)})
    pd.DataFrame(rows).to_parquet(root / "reference" / "qualifying_history.parquet")


def test_state_only_history_keeps_affinity_identified_by_modern_repeat_visits(tmp_path):
    write_circuit_archive(tmp_path)                    # modern: A (2021), B (2021), A (2022)
    write_history_at(tmp_path, ["C", "D"])             # history at other circuits only
    view = Archive(tmp_path).view(pd.Timestamp("2023-01-01", tz=UTC))
    model = PaceModel(ALL_SESSIONS, winter=True, weekend=True, affinity=True, history="state")
    model.fit_hyper(view, model.sessions(view), Hyper(**GOLDEN_SETTINGS))
    assert "affinity_prior" in model.last_fit["fitted"]


def test_weighted_history_noise_reaches_the_objective_only_through_the_state(tmp_path):
    """The history noise reaches the counted objective only through the state: with
    trusted history the history's content moves the objective; with a huge history
    noise it barely does (history then only leaves vaguer states entering 2018+)."""
    write_circuit_archive(tmp_path)
    write_history(tmp_path)
    view = Archive(tmp_path).view(pd.Timestamp("2023-01-01", tz=UTC))
    settings = Hyper(**GOLDEN_SETTINGS | NO_AFFINITY)

    def objective(shuffled):
        model = PaceModel(ALL_SESSIONS, winter=True, weekend=True, history="weighted")
        sessions = model.sessions(view)
        if shuffled:   # history with its times reversed within each group: different content
            for session in sessions[sessions.session_id.str.startswith("history:")].itertuples():
                model._observations[session.session_id] = [(c, d, v[::-1]) for c, d, v in
                                                            model.observations(view, session)]
        return model.objective_function(view, sessions, settings)

    original, shuffled = objective(False), objective(True)
    trust = {h: Hyper(**GOLDEN_SETTINGS | NO_AFFINITY, noise={**Hyper().noise, "H": h}) for h in (0.3, 50.0)}
    assert abs(original(trust[0.3]) - shuffled(trust[0.3])) > 0.5
    assert abs(original(trust[50.0]) - shuffled(trust[50.0])) < 0.05


def test_stage_comparison_reports_each_stage_of_the_season():
    from predictor import evaluation
    rows = [{"event_key": f"2021-{r:02d}", "season": 2021, "model": m,
             "rps": 0.1 + (0.01 if m == "b" and r >= 15 else 0) + 0.001 * (r % 3)}
            for r in range(1, 23) for m in ("a", "b")]
    stages = evaluation.stage_comparison(pd.DataFrame(rows), "a")
    assert list(stages.stage) == ["4-8", "9-14", "15+"]          # rounds 1-3: fewer than five events
    late = stages[stages.stage == "15+"].iloc[0]
    assert late.worse_beyond_noise and late.events == 8


def test_history_noise_is_fitted_above_its_floor(tmp_path):
    from predictor.pace import HISTORY_NOISE_FLOOR
    write_circuit_archive(tmp_path)
    write_history(tmp_path)
    view = Archive(tmp_path).view(pd.Timestamp("2023-01-01", tz=UTC))
    model = PaceModel(ALL_SESSIONS, winter=True, weekend=True, history="weighted")
    start = Hyper(**GOLDEN_SETTINGS | NO_AFFINITY, noise={**Hyper().noise, "H": 1e-4})   # below the floor
    hyper = model.fit_hyper(view, model.sessions(view), start)
    assert "H" in model.last_fit["fitted"] and hyper.noise["H"] >= HISTORY_NOISE_FLOOR * (1 - 1e-9)


def write_session_manifests(root, seasons):
    for season in seasons:
        for folder in (root / "sessions" / str(season)).glob("*"):
            (folder / "manifest.json").write_text(json.dumps(
                {"files": {"normalized/results.parquet": {"sha256": "a"}, "source/weather.parquet": {"sha256": "b"}}}))


@pytest.mark.parametrize("options", [dict(winter=True, weekend=True), dict(shared_from_qualifying=True),
                                     dict(winter=True, weekend=True, affinity=True)])
def test_a_cached_fit_forecasts_exactly_as_a_fresh_fit(tmp_path, monkeypatch, options):
    from predictor import pace
    from predictor.fitcache import FitCache
    monkeypatch.setattr(pace, "MIN_QUALIFYING_SESSIONS", 1)       # the fixture has two qualifying sessions
    root = tmp_path / "archive"
    write_circuit_archive(root)
    write_session_manifests(root, (2021, 2022))
    archive = Archive(root)
    examples = [golden_example("2022-02", "pre_weekend", "2022-03-25 00:00"),
                golden_example("2022-02", "pre_weekend", "2022-03-25 00:00")]
    examples[1] = examples[1].__class__(**{**examples[1].__dict__, "season": 2023})   # a second season: the chain

    def forecasts(cache):
        model = PaceModel(ALL_SESSIONS, "pace:test", **options)
        model.fit_cache = cache
        out = []
        for example in examples:
            view = archive.view(example.cutoff_time)
            model._ensure_hyper(view, example.season)
            out.append((copy.deepcopy(model.hyper), copy.deepcopy(model._qualifying_hyper), model.fit_log[example.season],
                        *model.distribution(view, example)))
        return out

    fresh = forecasts(None)
    cache = FitCache(tmp_path / "fits")
    first = forecasts(cache)                                      # fits and saves both seasons
    assert len(list((tmp_path / "fits").rglob("*.json"))) == 2
    monkeypatch.setattr(PaceModel, "fit_hyper", lambda *a, **k: (_ for _ in ()).throw(AssertionError("refitted")))
    second = forecasts(cache)                                     # reads both saved fits, fits nothing
    for a, b, c in zip(fresh, first, second):
        assert a[0] == b[0] == c[0] and a[1] == b[1] == c[1] and a[2] == b[2] == c[2]
        for x, y, z in zip(a[3:], b[3:], c[3:]):
            assert np.array_equal(x, y) and np.array_equal(x, z)


def test_fit_cache_key_follows_model_start_and_data(tmp_path):
    from predictor.fitcache import FitCache
    root = tmp_path / "archive"
    write_circuit_archive(root)
    write_session_manifests(root, (2021, 2022))
    reference = root / "reference" / "manifest.json"
    reference.write_text(json.dumps({"built_at_utc": "now", "files": {"events": {"sha256": "e"}}}))
    model = PaceModel(ALL_SESSIONS, "pace:test", winter=True, weekend=True)

    def key(model=model, start=Hyper()):
        return FitCache(tmp_path / "fits").key(model, 2022, start, None, root)

    base = key()
    assert key() == base
    assert key(start=Hyper(car_drift=0.02)) != base                                   # another start
    assert key(model=PaceModel(ALL_SESSIONS, "pace:test", winter=True)) != base      # other options
    reference.write_text(json.dumps({"built_at_utc": "later", "files": {"events": {"sha256": "e"}}}))
    assert key() == base                                                              # rebuild, same tables
    reference.write_text(json.dumps({"built_at_utc": "later", "files": {"events": {"sha256": "f"}}}))
    assert key() != base                                                              # other reference data
    base = key()
    later = next((root / "sessions" / "2022").glob("*/manifest.json"))
    later.write_text(json.dumps({"files": {"normalized/results.parquet": {"sha256": "z"}}}))
    assert key() == base                                    # a 2022 session cannot change a fit for 2022
    earlier = next((root / "sessions" / "2021").glob("*/manifest.json"))
    earlier.write_text(json.dumps({"files": {"normalized/results.parquet": {"sha256": "a"},
                                             "source/weather.parquet": {"sha256": "changed"}}}))
    assert key() == base                                    # a file the fit does not read
    earlier.write_text(json.dumps({"files": {"normalized/results.parquet": {"sha256": "a"},
                                             "source/weather.parquet": {"sha256": "b"},
                                             "source/car_data/driver-1.parquet": {"sha256": "c"}}}))
    assert key() != base                                    # telemetry (track profiles)
    base = key()
    earlier.write_text(json.dumps({"files": {"normalized/results.parquet": {"sha256": "changed"}}}))
    assert key() != base                                    # an earlier session the fit reads


def test_a_fit_is_not_saved_if_the_data_changed_while_it_ran(tmp_path):
    from predictor.fitcache import FitCache
    cache = FitCache(tmp_path / "fits")
    path = tmp_path / "fits" / "m" / "2022-x.json"
    with pytest.warns(UserWarning, match="not saved"):
        entry, cached = cache.get_or_fit(path, lambda: {"hyper": 1}, still_valid=lambda: False)
    assert entry == {"hyper": 1} and not cached and not path.exists()
    entry, cached = cache.get_or_fit(path, lambda: {"hyper": 2})
    assert entry == {"hyper": 2} and not cached and path.exists()
    assert cache.get_or_fit(path, lambda: {"hyper": 3}) == ({"hyper": 2}, True)
    assert not list(path.parent.glob("*.lock")) and not list(path.parent.glob("*.tmp"))


def test_a_screen_fits_only_what_the_model_adds(tmp_path, monkeypatch):
    from predictor import pace
    monkeypatch.setattr(pace, "MIN_QUALIFYING_SESSIONS", 1)
    write_circuit_archive(tmp_path)
    view = Archive(tmp_path).view(pd.Timestamp("2023-01-01", tz=UTC))
    base = PaceModel(ALL_SESSIONS, "pace:base", winter=True, weekend=True)
    model = PaceModel(ALL_SESSIONS, "pace:candidate", winter=True, weekend=True, affinity=True)
    model.screen_base = base
    model._ensure_hyper(view, 2023)
    assert set(model.last_fit["fitted"]) == {"affinity_prior", "affinity_drift", "affinity_winter"}
    for name in Hyper.SCALARS:
        if not name.startswith("affinity"):
            assert getattr(model.hyper, name) == getattr(base.hyper, name), name
    assert model.hyper.noise == {**model.hyper.noise, **base.hyper.noise}
    assert model.options()["screen_base"][0] == "pace:base"


def test_cache_paths_are_valid_folder_names(tmp_path):
    from predictor.fitcache import FitCache
    model = PaceModel(ALL_SESSIONS, "pace:v3whw|screen")
    path = FitCache(tmp_path / "fits").path(model, 2022, "0" * 64)
    assert path.parent.name == "pace_v3whw_screen"
    path.parent.mkdir(parents=True)


def test_a_screen_with_nothing_to_fit_keeps_the_base_settings(tmp_path, monkeypatch):
    from predictor import pace
    monkeypatch.setattr(pace, "MIN_QUALIFYING_SESSIONS", 1)
    write_circuit_archive(tmp_path)
    view = Archive(tmp_path).view(pd.Timestamp("2022-01-01", tz=UTC))       # one season: affinity not identified
    base = PaceModel(ALL_SESSIONS, "pace:base", winter=True, weekend=True)
    model = PaceModel(ALL_SESSIONS, "pace:candidate", winter=True, weekend=True, affinity=True)
    model.screen_base = base
    model._ensure_hyper(view, 2022)
    assert model.last_fit["fitted"] == [] and model.hyper.affinity_prior == 0.0
    assert model.hyper.noise["Q"] == base.hyper.noise["Q"] and model.hyper.car_drift == base.hyper.car_drift


def test_a_screen_freezes_what_the_base_holds_and_refuses_non_extensions(tmp_path, monkeypatch):
    from predictor import pace
    monkeypatch.setattr(pace, "MIN_QUALIFYING_SESSIONS", 1)
    write_circuit_archive(tmp_path)
    write_history(tmp_path)
    view = Archive(tmp_path).view(pd.Timestamp("2023-01-01", tz=UTC))
    base = PaceModel(ALL_SESSIONS, "pace:base", winter=True, weekend=True)
    model = PaceModel(ALL_SESSIONS, "pace:candidate", winter=True, weekend=True, history="shared")
    model.screen_base = base
    model._ensure_hyper(view, 2023)
    # counted history crosses more season starts, but the winter terms belong to the base: frozen
    assert not set(model.last_fit["fitted"]) & {"car_winter", "driver_winter", "regulation_multiplier", "car_drift"}
    for name in ("car_winter", "driver_winter", "regulation_multiplier", "weekend_prior"):
        assert getattr(model.hyper, name) == getattr(base.hyper, name), name
    narrower = PaceModel(ALL_SESSIONS, "pace:narrower", winter=True)
    narrower.screen_base = base
    with pytest.raises(ValueError, match="weekend"):
        narrower._ensure_hyper(view, 2023)


def test_season_form_resets_each_season_and_the_career_level_carries_on():
    hyper = Hyper(form_prior=0.3, driver_drift=0.0, car_drift=0.0)
    state = Recorder(("Q",), form=True)
    state.advance(pd.Timestamp("2021-07-01", tz=UTC))
    for _ in range(5):          # driver "a" much faster than his teammate "b" in the same car
        state.update(["fast", "fast"], ["a", "b"], [0.0, 1.0])
    replayed = state.replay(hyper)
    form = [i for i, (kind, key) in enumerate(replayed.keys) if kind == "form"]
    driver = [i for i, (kind, key) in enumerate(replayed.keys) if kind == "driver"]
    assert np.abs(replayed.mean[form]).max() > 0.01            # this season's evidence moved the form
    state.advance(pd.Timestamp("2022-03-01", tz=UTC))           # a new season
    after = state.replay(hyper)
    assert np.allclose(after.mean[form], 0.0) and np.allclose(np.diag(after.cov)[form], 0.3 ** 2)
    assert np.allclose(after.cov[np.ix_(form, driver)], 0.0)
    assert np.allclose(after.mean[driver], replayed.mean[driver])     # the career level is kept
    assert program.RESET in set(state.program().ops[:, 0])


def test_season_form_is_fitted_only_once_the_counted_data_crosses_a_season(tmp_path, monkeypatch):
    from predictor import pace
    monkeypatch.setattr(pace, "MIN_QUALIFYING_SESSIONS", 1)
    write_circuit_archive(tmp_path)
    archive = Archive(tmp_path)
    fitted = {}
    for season in (2022, 2023):
        model = PaceModel(ALL_SESSIONS, winter=True, weekend=True, form=True)
        view = archive.view(pd.Timestamp(f"{season}-01-01", tz=UTC))
        hyper = model.fit_hyper(view, model.sessions(view), model._default_start())
        fitted[season] = ("form_prior" in model.last_fit["fitted"], hyper.form_prior)
    assert fitted[2022] == (False, 0.0)          # one season: form off
    assert fitted[2023][0] and fitted[2023][1] > 0


def test_weighted_entries_match_a_direct_kalman_update(monkeypatch):
    """One weighted observation against the textbook update in numpy."""
    hyper = Hyper(form_prior=0.4, driver_drift=0.0, car_drift=0.0)
    weights = {"a": 0.25, "b": 0.75}
    original = Recorder.columns

    def weighted(self, cars, drivers, session_type="Q", track=0):
        rows = original(self, cars, drivers, session_type, track)
        form = {self.tracks[track].index[("form", d)]: d for d in drivers}
        return [[(i, weights[form[i]]) if i in form else i for i in row] for row in rows]

    monkeypatch.setattr(Recorder, "columns", weighted)
    state = Recorder(("Q",), form=True)
    state.advance(pd.Timestamp("2021-07-01", tz=UTC))
    state.update(["c", "c"], ["a", "b"], [0.0, 0.6])
    replayed = state.replay(hyper)
    keys = replayed.keys
    prior = np.diag([hyper.car_prior ** 2 if k == "car" else hyper.driver_prior ** 2 if k == "driver"
                     else 0.4 ** 2 for k, _ in keys])
    h = np.zeros(len(keys))
    for (kind, key), i in zip(keys, range(len(keys))):
        if kind == "driver":
            h[i] = 1.0 if key == "b" else -1.0
        if kind == "form":
            h[i] = weights[key] if key == "b" else -weights[key]
    s = h @ prior @ h + 2 * hyper.noise["Q"] ** 2
    gain = prior @ h / s
    assert np.allclose(replayed.mean, gain * 0.6)
    assert np.allclose(replayed.cov, prior - np.outer(gain, h @ prior))
    assert replayed.objective == pytest.approx(-0.5 * 0.6 ** 2 / s - 0.5 * np.log(2 * np.pi * s))


def test_a_zero_weight_term_changes_nothing(monkeypatch):
    original = Recorder.columns

    def zero_form(self, cars, drivers, session_type="Q", track=0):
        rows = original(self, cars, drivers, session_type, track)
        form = {self.tracks[track].index[("form", d)] for d in drivers}
        return [[(i, 0.0) if i in form else i for i in row] for row in rows]

    def run(form):
        state = Recorder(("Q",), form=form)
        state.advance(pd.Timestamp("2021-07-01", tz=UTC))
        state.update(["c", "c", "d"], ["a", "b", "e"], [0.0, 0.6, 1.1])
        return state.replay(Hyper(form_prior=0.4) if form else Hyper())

    plain = run(False)
    monkeypatch.setattr(Recorder, "columns", zero_form)
    weighted = run(True)
    assert weighted.objective == plain.objective


def test_two_weighted_updates_of_three_drivers_match_numpy(monkeypatch):
    """Weights other than 0 and 1, a non-zero prior mean at the second update and a
    2x2 innovation covariance: mean, covariance, objective and forecasts against numpy."""
    weights = {"a": 0.25, "b": 0.75, "e": 1.6}
    original = Recorder.columns

    def weighted(self, cars, drivers, session_type="Q", track=0):
        rows = original(self, cars, drivers, session_type, track)
        form = {self.tracks[track].index[("form", d)]: d for d in drivers}
        return [[(i, weights[form[i]]) if i in form else i for i in row] for row in rows]

    monkeypatch.setattr(Recorder, "columns", weighted)
    hyper = Hyper(form_prior=0.4)
    recorder = Recorder(("Q",), form=True)
    recorder.advance(pd.Timestamp("2021-07-01", tz=UTC))
    observations = [(["c", "c", "d"], ["a", "b", "e"], [0.0, 0.6, 1.1]),
                    (["d", "c", "c"], ["e", "b", "a"], [0.3, -0.2, 0.5])]
    for observation in observations:
        recorder.update(*observation)
    state = recorder.replay(hyper)
    index = {key: i for i, key in enumerate(state.keys)}
    spread = {"car": hyper.car_prior, "driver": hyper.driver_prior, "form": 0.4}
    cov = np.diag([spread[kind] ** 2 for kind, _ in state.keys])
    mean, objective, noise = np.zeros(len(index)), 0.0, hyper.noise["Q"] ** 2
    for cars, drivers, values in observations:
        rows = np.zeros((3, len(index)))
        for i, (car, driver) in enumerate(zip(cars, drivers)):
            rows[i, index[("car", car)]] += 1
            rows[i, index[("driver", driver)]] += 1
            rows[i, index[("form", driver)]] += weights[driver]
        h = rows[1:] - rows[0]
        innovation = np.array(values[1:]) - values[0] - h @ mean
        s = h @ cov @ h.T + noise * (np.eye(2) + 1)
        gain = cov @ h.T @ np.linalg.inv(s)
        objective += -0.5 * innovation @ np.linalg.solve(s, innovation) - 0.5 * np.log(np.linalg.det(s)) \
            - np.log(2 * np.pi)
        mean, cov = mean + gain @ innovation, cov - gain @ h @ cov
    assert np.allclose(state.mean, mean, atol=1e-12) and np.allclose(state.cov, cov, atol=1e-12)
    assert state.objective == pytest.approx(objective, rel=1e-12)
    forecast = state.design(recorder.columns(["c", "d"], ["a", "e"])) @ state.mean
    expected = [mean[index[("car", "c")]] + mean[index[("driver", "a")]] + 0.25 * mean[index[("form", "a")]],
                mean[index[("car", "d")]] + mean[index[("driver", "e")]] + 1.6 * mean[index[("form", "e")]]]
    assert np.allclose(forecast, expected, atol=1e-12)


def test_the_native_filter_rejects_mismatched_weights():
    recorder = Recorder(("Q",))
    recorder.advance(pd.Timestamp("2021-07-01", tz=UTC))
    recorder.update(["c", "d"], ["a", "b"], [0.0, 0.5])
    program = recorder.program()
    program.weights = program.weights[:-1]
    with pytest.raises(ValueError, match="weights"):
        recorder.objective(Hyper())


def test_skills_read_centred_shares_of_earlier_visits_only(tmp_path, monkeypatch):
    """Each driver's pace reads skill states weighted by how the track differs from
    the reference; shares are fixed per event and come from earlier visits only."""
    from predictor import track
    write_circuit_archive(tmp_path)
    profiles = {"2021-01-2": {**dict.fromkeys(track.FEATURES, 0.0), "share_straight": 0.3, "share_slow": 0.4,
                              "share_medium": 0.2, "share_fast": 0.1}}
    monkeypatch.setattr(track, "session_profile", lambda view, session_id, count=5: profiles.get(session_id))
    monkeypatch.setattr(track, "layout_of", lambda circuit, season, event_key, table=None: circuit and f"{circuit}/gp")
    archive = Archive(tmp_path)
    view = archive.view(pd.Timestamp("2023-01-01", tz=UTC))
    model = PaceModel(ALL_SESSIONS, winter=True, weekend=True, skills=("fast",),
                      season_skills=("straight", "slow", "medium", "fast"))
    assert model.shares(view, "2021-01") == {f"share_{b}": 0.0 for b in track.BANDS}       # first visit to A
    revisit = model.shares(view, "2022-01")                                               # A again, after 2021-01
    assert revisit["share_slow"] == pytest.approx(0.4 - track.REFERENCE_SHARES["slow"])
    assert sum(revisit.values()) == pytest.approx(0.0)
    hyper = Hyper(**GOLDEN_SETTINGS | NO_AFFINITY, skill_prior=0.3, skill_drift=0.001, season_skill_prior=0.2)
    recorder = model.record(view, model.sessions(view), True, False, skills=True, season_skills=True)
    keys = recorder.keys
    assert ("skill", ("a", "fast")) in keys and ("season_skill", ("a", "slow")) in keys
    assert np.isfinite(recorder.objective(hyper))
    # equal season skills in every band cancel: a season-wide level is not a skill
    rows = Recorder.columns(recorder, ["fast"], ["a"], "Q")
    weights = [w for entry in rows[0] if isinstance(entry, tuple) for _, w in [entry]]
    assert sum(weights[1:]) == pytest.approx(0.0)


def test_screens_need_the_base_skills():
    base = PaceModel(ALL_SESSIONS, "pace:base", winter=True, skills=("fast",))
    other = PaceModel(ALL_SESSIONS, "pace:other", winter=True, skills=("slow",))
    with pytest.raises(ValueError, match="skills"):
        other._check_extends(base)
    PaceModel(ALL_SESSIONS, "pace:more", winter=True, skills=("fast", "slow"))._check_extends(base)


def test_an_unwritable_cache_fits_without_it(tmp_path, monkeypatch):
    from predictor import fitcache
    cache = fitcache.FitCache(tmp_path / "fits")
    path = tmp_path / "fits" / "m" / "2022-x.json"
    calls = []

    def refuse(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(fitcache.os, "open", refuse)           # lock creation refused, no lock present
    with pytest.warns(UserWarning, match="unavailable"):
        entry, cached = cache.get_or_fit(path, lambda: calls.append(1) or {"hyper": 1})
    assert entry == {"hyper": 1} and not cached and calls == [1]
    monkeypatch.undo()
    monkeypatch.setattr(type(path), "mkdir", refuse)            # the folder cannot be created
    with pytest.warns(UserWarning, match="unavailable"):
        entry, cached = fitcache.FitCache(tmp_path / "other").get_or_fit(tmp_path / "other" / "m" / "x.json",
                                                                         lambda: {"hyper": 2})
    assert entry == {"hyper": 2} and not cached


def test_skills_are_off_until_a_layout_is_revisited(tmp_path, monkeypatch):
    from predictor import pace, track
    monkeypatch.setattr(pace, "MIN_QUALIFYING_SESSIONS", 1)
    write_circuit_archive(tmp_path)                    # A (2021), B (2021), A (2022)
    profile = {**dict.fromkeys(track.FEATURES, 0.0), "share_straight": 0.3, "share_slow": 0.4,
               "share_medium": 0.2, "share_fast": 0.1}
    monkeypatch.setattr(track, "session_profile", lambda view, session_id, count=5: profile)
    monkeypatch.setattr(track, "layout_of", lambda circuit, season, event_key, table=None: circuit and f"{circuit}/gp")
    archive = Archive(tmp_path)
    for cutoff, revisit in (("2022-01-01", False), ("2023-01-01", True)):
        model = PaceModel(ALL_SESSIONS, winter=True, weekend=True, skills=("fast",))
        view = archive.view(pd.Timestamp(cutoff, tz=UTC))
        hyper = model.fit_hyper(view, model.sessions(view), model._default_start())
        assert ("skill_prior" in model.last_fit["fitted"]) == revisit
        assert (hyper.skill_prior > 0) == revisit


def test_a_lock_left_by_a_dead_process_is_taken_over(tmp_path):
    import os
    import subprocess
    import sys
    from predictor import fitcache
    finished = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead = int(finished.stdout.strip())
    assert fitcache.process_alive(os.getpid()) and not fitcache.process_alive(dead)
    path = tmp_path / "fits" / "m" / "2022-x.json"
    path.parent.mkdir(parents=True)
    path.with_suffix(".lock").write_text(f"{dead}-0")
    entry, cached = fitcache.FitCache(tmp_path / "fits").get_or_fit(path, lambda: {"hyper": 1})
    assert entry == {"hyper": 1} and not cached and path.exists() and not path.with_suffix(".lock").exists()


def test_a_driver_race_offset_is_read_only_in_races():
    recorder = Recorder(("Q", "R"), driver_offsets=("R",))
    recorder.advance(pd.Timestamp("2021-07-01", tz=UTC))
    race = recorder.columns(["c"], ["a"], "R")[0]
    qualifying = recorder.columns(["c"], ["a"], "Q")[0]
    keys = recorder.keys
    assert ("offset", ("a", "R")) in keys and ("offset", ("c", "R")) in keys
    assert len(race) == len(qualifying) + 2          # car and driver race offsets


def test_a_failed_owner_write_leaves_no_lock(tmp_path, monkeypatch):
    from predictor import fitcache
    path = tmp_path / "fits" / "m" / "2022-x.json"

    def full(handle, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(fitcache.os, "write", full)
    with pytest.warns(UserWarning, match="unavailable"):
        entry, cached = fitcache.FitCache(tmp_path / "fits").get_or_fit(path, lambda: {"hyper": 1})
    assert entry == {"hyper": 1} and not cached
    assert not path.with_suffix(".lock").exists()


def test_an_ownerless_lock_goes_stale_after_a_grace_period(tmp_path):
    import os
    import time
    from predictor import fitcache
    lock = tmp_path / "x.lock"
    lock.write_text("")
    assert not fitcache.lock_owner_gone(lock)                     # just created: its owner may be writing
    old = time.time() - 2 * fitcache.EMPTY_LOCK_GRACE_S
    os.utime(lock, (old, old))
    assert fitcache.lock_owner_gone(lock)


def test_push_pace_and_long_runs_together_keep_both(tmp_path, monkeypatch):
    """Long-run sessions come from the practice sessions before any of them is
    relabelled as push pace."""
    from predictor import pace
    write_circuit_archive(tmp_path)
    view = Archive(tmp_path).view(pd.Timestamp("2023-01-01", tz=UTC))
    monkeypatch.setattr(PaceModel, "_push_covers", lambda self, view, session: True)
    kinds = {}
    for push, long in ((False, True), ("own", False), ("own", True)):
        model = PaceModel(ALL_SESSIONS, winter=True, weekend=True, practice_push=push, long_runs=long)
        kinds[(push, long)] = set(model.sessions(view).session_type)
    assert "L" in kinds[(False, True)] and "FP1" in kinds[(False, True)]
    assert "FP1P" in kinds[("own", False)] and "L" not in kinds[("own", False)]
    assert {"FP1P", "L"} <= kinds[("own", True)]


def test_every_model_draws_the_same_random_numbers_for_an_example():
    from predictor.baselines import seed_for
    example = golden_example(*GOLDEN_EXAMPLES["before_B"])
    assert seed_for(example) == seed_for(example)
    import inspect
    assert "seed = seed_for(example)" in inspect.getsource(PaceModel.predict)


def test_a_groups_estimation_covariance_enters_its_noise():
    """A group carrying the estimation covariance C of its values: the innovation
    covariance gains scale x cov(v_i - v_0, v_k - v_0), checked against numpy."""
    covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    hyper = Hyper(estimate_scale=1.7)
    recorder = Recorder(("Q",))
    recorder.advance(pd.Timestamp("2021-07-01", tz=UTC))
    recorder.update(["c", "c", "d"], ["a", "b", "e"], [0.0, 0.6, 1.1], covariance=covariance)
    state = recorder.replay(hyper)
    index = {key: i for i, key in enumerate(state.keys)}
    prior = np.diag([hyper.car_prior ** 2 if kind == "car" else hyper.driver_prior ** 2 for kind, _ in state.keys])
    rows = np.zeros((3, len(index)))
    for i, (car, driver) in enumerate(zip(["c", "c", "d"], ["a", "b", "e"])):
        rows[i, index[("car", car)]] = rows[i, index[("driver", driver)]] = 1
    h = rows[1:] - rows[0]
    difference = np.array([[-1, 1, 0], [-1, 0, 1]], dtype=float)
    s = h @ prior @ h.T + hyper.noise["Q"] ** 2 * (np.eye(2) + 1) + 1.7 * difference @ covariance @ difference.T
    innovation = np.array([0.6, 1.1])
    gain = prior @ h.T @ np.linalg.inv(s)
    assert np.allclose(state.mean, gain @ innovation, atol=1e-12)
    assert state.objective == pytest.approx(-0.5 * innovation @ np.linalg.solve(s, innovation)
                                            - 0.5 * np.log(np.linalg.det(s)) - np.log(2 * np.pi), rel=1e-12)
    plain = Recorder(("Q",))
    plain.advance(pd.Timestamp("2021-07-01", tz=UTC))
    plain.update(["c", "c", "d"], ["a", "b", "e"], [0.0, 0.6, 1.1], covariance=covariance)
    assert plain.replay(Hyper()).objective == recorder.replay(Hyper(estimate_scale=0.0)).objective   # scale 0: unused
