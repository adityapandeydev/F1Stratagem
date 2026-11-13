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
