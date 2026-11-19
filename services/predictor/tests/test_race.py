import types

import numpy as np
import pandas as pd

from predictor.race import RaceModel, grid_at


def _view(season, sessions, results):
    """A view of one event: `sessions` the types classified at the cutoff, `results`
    {type: {driver number: position}}."""
    table = pd.DataFrame({"event_key": "e", "session_type": sessions, "session_id": [f"e-{t}" for t in sessions]})
    return types.SimpleNamespace(
        sessions=table,
        results=lambda sid: pd.DataFrame({"DriverNumber": list(results[sid[2:]]),
                                          "Position": list(results[sid[2:]].values())}))


def _example(season, numbers=("1", "2", "3")):
    return types.SimpleNamespace(event_key="e", season=season, driver_numbers=list(numbers))


def test_the_grid_is_the_qualifying_order_once_qualifying_is_classified():
    view = _view(2024, ["FP1", "Q"], {"Q": {"2": 1, "3": 2, "1": 3}})
    assert list(grid_at(view, _example(2024))) == [3, 1, 2]


def test_before_qualifying_there_is_no_known_grid():
    assert grid_at(_view(2024, ["FP1"], {}), _example(2024)) is None


def test_in_2021_and_2022_the_sprint_sets_the_grid_once_it_has_run():
    results = {"Q": {"1": 1, "2": 2, "3": 3}, "S": {"3": 1, "1": 2, "2": 3}}
    assert list(grid_at(_view(2022, ["Q", "S"], results), _example(2022))) == [2, 3, 1]
    assert list(grid_at(_view(2022, ["Q"], results), _example(2022))) == [1, 2, 3]     # sprint not run yet
    assert list(grid_at(_view(2023, ["Q", "S"], results), _example(2023))) == [1, 2, 3]  # from 2023: qualifying


def test_a_driver_missing_from_qualifying_starts_at_the_back():
    view = _view(2024, ["Q"], {"Q": {"2": 1, "3": 2}})
    assert list(grid_at(view, _example(2024))) == [3, 1, 2]


def test_kappa_trades_race_pace_against_grid_position():
    model = RaceModel.__new__(RaceModel)
    race = np.array([[0.0, -0.3]])            # driver 2 is faster in the race
    grid = np.array([[1.0, 2.0]])             # driver 1 starts ahead
    assert list(model.positions(race, grid, 0.0, 0.0, 1)[0]) == [2, 1]   # pace alone decides
    assert list(model.positions(race, grid, 0.5, 0.0, 1)[0]) == [1, 2]   # the grid slot is worth more


def _race_laps(swaps, laps=25):
    """Three cars running 1-2-3; at each lap in `swaps` (lap: kind) cars 1 and 2 swap
    for good: kind "green", "pit" (car 2 pits that lap), "sc" (safety car), "yellow"
    (a yellow flag on car 3's record only) or "unknown" (no status recorded)."""
    order, rows = ["1", "2", "3"], []
    for lap in range(1, laps + 1):
        kind = swaps.get(lap)
        if kind:
            order = [order[1], order[0], order[2]]
        for position, number in enumerate(order, start=1):
            rows.append({"LapNumber": lap, "DriverNumber": number, "Position": float(position),
                         "PitInTime": pd.Timedelta(seconds=1) if kind == "pit" and number == "2" else pd.NaT,
                         "PitOutTime": pd.NaT,
                         "TrackStatus": {"sc": "4", "unknown": None}.get(kind, "12" if kind == "yellow" and number == "3"
                                                                         else "1"),
                         "LapTime": pd.Timedelta(seconds=90), "Compound": "SOFT"})
    return pd.DataFrame(rows)


def test_only_swaps_on_green_laps_without_a_pit_stop_are_passes():
    from predictor.laps import race_passes
    passes, counted = race_passes(_race_laps({2: "green", 5: "green", 8: "pit", 11: "sc"}))
    # lap 2 compares with lap 1 (excluded), lap 8 involves a stop, lap 11 is neutralised
    assert passes == 1
    # lap pairs (2, 3) .. (24, 25), less the two around the safety car; a stop leaves out the
    # car that stopped, not the lap
    assert counted == 23 - 2


def test_a_short_race_has_no_pass_rate():
    from predictor.laps import race_passes
    assert race_passes(_race_laps({}, laps=15)) is None


def test_a_swap_under_a_yellow_flag_or_an_unknown_status_is_not_a_pass():
    from predictor.laps import race_passes
    for kind in ("yellow", "unknown"):
        passes, counted = race_passes(_race_laps({5: kind}))
        assert passes == 0, kind
        assert counted == 23 - 2, kind


def test_settings_are_fitted_per_cutoff_from_races_forecast_from_that_cutoff():
    model = RaceModel.__new__(RaceModel)
    model.overtaking, model.settings = True, {}
    earlier = [types.SimpleNamespace(season=2018, cutoff_label="last", session_type="R")]
    model._history = earlier
    model._fit_season(2019, "pre_weekend")          # no pre-weekend history: the neutral settings
    assert model.settings[(2019, "pre_weekend")] == {"kappa": 0.0, "sigma": 0.0, "rps": None, "races": 0,
                                                     "gamma": 0.0}


def _passing(always):
    """A battle model where a faster car within range always passes (`always`), or
    nobody ever does."""
    from predictor.race_sim import PassModel
    model = PassModel()
    model.coef = np.array([0.0, 0, 0, -1000.0, 0, 0]) if always else np.array([-1000.0, 0, 0, 0, 0, 0])
    return model


def _run(always, laps, pace=(0.0, 1.0, -2.0), grid=(1, 2, 3)):
    from predictor import race_sim
    import predictor.race_sim as sim
    keep = sim.LAP_NOISE
    sim.LAP_NOISE = 0.0
    try:
        return race_sim.simulate(np.array([pace]), np.array([grid], dtype=float), _passing(always), 0.0, laps, 90.0,
                                 0.0, np.random.default_rng(0))[0]
    finally:
        sim.LAP_NOISE = keep


def test_without_passing_a_faster_car_stays_held_behind():
    # car 3 is 2% faster but starts last; nobody can pass
    assert list(_run(False, laps=30)) == [1, 2, 3]


def test_a_faster_car_passes_one_place_per_lap():
    assert list(_run(True, laps=1)) == [1, 3, 2]          # one lap: past car 2 only
    assert list(_run(True, laps=30)) == [2, 3, 1]         # then past the leader too; car 2 (slower) passed by 3


def test_the_battle_model_learns_that_faster_closer_cars_pass():
    from predictor.race_sim import PassModel
    rng = np.random.default_rng(1)
    gap = rng.uniform(0.1, 2.4, 4000)
    delta = rng.normal(0, 0.6, 4000)
    truth = 1 / (1 + np.exp(-(-1.0 - 2.5 * gap - 1.5 * delta)))
    frame = pd.DataFrame({"gap": gap, "delta": delta, "passed": rng.random(4000) < truth})
    model = PassModel().fit(frame, np.zeros(4000))
    close_fast = model.probability(np.array([0.3]), np.array([-1.0]))[0]
    far_slow = model.probability(np.array([2.0]), np.array([0.5]))[0]
    assert close_fast > 0.4 and far_slow < 0.01


def test_a_shortened_race_does_not_shorten_the_next_one():
    from predictor.race import scheduled_laps
    earlier = [("spa/2019", 44), ("spa/2019", 3)]            # the second was stopped after three laps
    assert scheduled_laps(earlier, "spa/2019") == 44
    assert scheduled_laps([("spa/2019", 44), ("spa/2022", 3)], "spa/2022") == 3     # same layout first
    assert scheduled_laps([("spa/2019", 44)], "spa/2022") == 44                      # else the circuit
    assert scheduled_laps([], "spa/2022") is None


def test_the_lap_reference_skips_laps_behind_the_safety_car():
    from predictor.race_sim import green_lap_seconds
    laps = _race_laps({}, laps=20)
    laps.loc[laps.LapNumber <= 4, "TrackStatus"] = "4"
    laps.loc[laps.LapNumber <= 4, "LapTime"] = pd.Timedelta(seconds=150)
    assert green_lap_seconds(laps) == 90.0
    laps.loc[laps.LapNumber <= 12, "TrackStatus"] = "4"        # too few racing laps left
    assert green_lap_seconds(laps) is None
    laps["TrackStatus"] = "4"
    assert green_lap_seconds(laps) is None


def test_a_history_without_passes_gives_neutral_circuits():
    from predictor.race_sim import circuit_propensity
    frame = pd.DataFrame({"circuit": ["a", "a", "b"], "passed": [False, False, False]})
    assert circuit_propensity(frame, 0.0) == {"a": 0.0, "b": 0.0}


def test_the_simulation_is_reproducible_from_its_seed():
    """The same seed gives the same races (golden orders recorded from the native
    simulation; it matched the NumPy version it replaced in distribution: position
    probabilities within Monte Carlo noise, no bias, on 100,000 draws)."""
    import hashlib
    from predictor import race_sim
    model = race_sim.PassModel()
    model.coef = np.array([-0.8, -1.9, -0.3, -1.6, 0.2, 0.7])
    rng = np.random.default_rng(3)
    pace = rng.normal(0, 0.8, (300, 10))
    grid = np.argsort(rng.random((300, 10)), axis=1) + 1.0
    got = race_sim.simulate(pace, grid, model, 0.4, 25, 95.0, 0.5, np.random.default_rng(4), -0.5)
    again = race_sim.simulate(pace, grid, model, 0.4, 25, 95.0, 0.5, np.random.default_rng(4), -0.5)
    assert np.array_equal(got, again)
    assert got[:2].tolist() == [[9, 1, 8, 4, 5, 7, 2, 6, 3, 10], [7, 4, 8, 2, 1, 5, 9, 3, 10, 6]]
    assert hashlib.sha256(got.astype(np.int64).tobytes()).hexdigest() ==         "8a19c38b82274093dc4edee338c5cbef5d90b1481451d21b56f0396e0eeeedca"


def test_race_champions_route_by_forecast_moment():
    from predictor.evaluation import RACE_CHAMPIONS, model_registry
    router = model_registry()["race_champion"]()
    assert router.name == "race_champion" and router.champions == RACE_CHAMPIONS
    before = types.SimpleNamespace(cutoff_label="pre_weekend")
    after = types.SimpleNamespace(cutoff_label="last")
    assert router.models[router.route(before)].name == "race:v2a"
    assert router.models[router.route(after)].name == "race:v1a"


def test_the_battle_model_can_read_forecast_pace_differences():
    from predictor.race import BATTLE_DRAWS, SimulatedRace
    model = SimulatedRace.__new__(SimulatedRace)
    example = types.SimpleNamespace(event_key="e", season=2020, cutoff_label="last", driver_numbers=["1", "2", "3"],
                                    session_type="R")
    model._history = [example]
    race = np.tile([[0.0, 1.0, -1.0]], (50, 1))            # forecast: car 3 fastest, car 2 slowest (% of a lap)
    model._draws = {("e", "last", 2021): (race, None, None)}
    frame = pd.DataFrame({"gap": [0.5, 1.0, 0.8], "delta": [9.0, 9.0, 9.0], "passed": [True, False, True],
                          "ahead": ["1", "2", "9"], "behind": ["2", "3", "1"], "lap": [100.0] * 3,
                          "event_key": ["e", "e", "e"]})
    out, term, weights = model._forecast_deltas(frame, np.array([0.1, 0.2, 0.3]), 2021, "last")
    assert len(out) == 2 * BATTLE_DRAWS                     # the pair with an unknown driver is left out
    assert np.allclose(out.delta[:BATTLE_DRAWS], 1.0) and np.allclose(out.delta[BATTLE_DRAWS:], -2.0)
    assert np.allclose(term, [0.1] * BATTLE_DRAWS + [0.2] * BATTLE_DRAWS) and np.allclose(weights, 1 / BATTLE_DRAWS)
    assert list(out.passed[::BATTLE_DRAWS]) == [True, False]


def test_neutral_periods_merge_a_run_and_keep_its_most_severe_kind():
    from predictor.race_sim import neutral_periods
    status = {1: "1", 2: "4", 3: "45", 4: "4", 5: "1", 6: "6", 7: "67", 8: "1", 9: "4"}
    laps = pd.DataFrame({"LapNumber": list(status), "TrackStatus": list(status.values())})
    assert neutral_periods(laps) == [(2, "red", 3), (6, "vsc", 2), (9, "sc", 1)]


def test_slow_field_laps_mark_a_neutralisation_without_track_status():
    from predictor.race_sim import slow_lap_periods
    times = {lap: np.full(10, 90.0) for lap in range(1, 21)}
    times[1] = np.full(10, 100.0)                    # standing start: slower, not neutralised
    for lap in (8, 9, 10):
        times[lap] = np.full(10, 125.0)              # behind the safety car
    assert slow_lap_periods(times) == [(8, None, 3)]


def test_circuit_factors_are_shrunk_and_the_first_laps_have_their_own_rate():
    from predictor.race_sim import NeutralModel
    history = pd.DataFrame({"season": [2010] * 8 + [2011] * 8, "circuit": ["a", "b", "c", "d"] * 4,
                            "laps": [60] * 16, "n": [4, 0, 1, 1] * 4})
    recent = [(2011, 60, [(1, "sc", 4), (30, "vsc", 2)]), (2011, 60, []), (2011, 60, [(40, "red", 2)])]
    model = NeutralModel().fit(history, recent)
    assert model.factors["a"] > 1 > model.factors["b"]
    assert model.factors["a"] < 4 / 1.5                       # shrunk towards 1 (raw ratio 2.67)
    race = model.race("a", 60)
    assert race["hazard"][0] == race["hazard"][1] > race["hazard"][2]
    assert np.allclose(model.kinds, [1 / 3, 1 / 3, 1 / 3]) and model.red_end == 0.0
    assert model.race("unseen", 60)["factor"] == 1.0


def _neutral(hazard, kind, length, red_end=0.0, laps=30):
    from predictor.race_sim import KINDS
    kinds = np.array([1.0 if k == kind else 0.0 for k in KINDS])
    return {"hazard": np.r_[hazard, np.zeros(laps - 1)], "kinds": kinds,
            "lengths": {k: np.array([length]) for k in KINDS}, "red_end": red_end}


def test_a_safety_car_freezes_the_order_and_a_red_flag_can_end_the_race():
    from predictor import race_sim
    pace, grid = np.array([[0.0, 1.0, -2.0]]), np.array([[1.0, 2.0, 3.0]])
    run = lambda neutral: list(race_sim.simulate(pace, grid, _passing(True), 0.0, 30, 90.0, 0.0,
                                                 np.random.default_rng(0), neutral=neutral)[0])
    assert run(None) == [2, 3, 1]                                         # the fastest car comes through
    assert run(_neutral(1.0, "sc", 30)) == [1, 2, 3]                      # safety car to the flag
    assert run(_neutral(1.0, "red", 1, red_end=1.0)) == [1, 2, 3]         # stopped on lap 1, not restarted
    assert run(_neutral(0.0, "sc", 5)) == run(None)                       # no hazard: the same race


def test_without_hazards_the_simulation_draws_the_same_races():
    from predictor import race_sim
    rng = np.random.default_rng(5)
    pace, grid = rng.normal(0, 0.5, (200, 8)), np.arange(1, 9, dtype=float)[None]
    model = race_sim.PassModel()
    model.coef = np.array([-0.8, -1.9, -0.3, -1.6, 0.2, 0.7])
    plain = race_sim.simulate(pace, grid, model, 0.0, 20, 90.0, 0.3, np.random.default_rng(1))
    zero = race_sim.simulate(pace, grid, model, 0.0, 20, 90.0, 0.3, np.random.default_rng(1),
                             neutral=_neutral(0.0, "sc", 3, laps=20))
    assert np.array_equal(plain, zero)


def test_a_safety_car_followed_by_a_virtual_one_is_one_safety_car():
    from predictor.race_sim import neutral_periods
    status = {1: "1", 2: "4", 3: "4", 4: "6", 5: "67", 6: "1"}
    laps = pd.DataFrame({"LapNumber": list(status), "TrackStatus": list(status.values())})
    assert neutral_periods(laps) == [(2, "sc", 4)]


def test_hazards_count_only_laps_on_which_a_neutralisation_could_start():
    from predictor.race_sim import NeutralModel
    empty = pd.DataFrame(columns=["season", "circuit", "laps", "n"])
    later = NeutralModel().fit(empty, [(2020, 60, [(10, "sc", 5), (30, "sc", 5)])])
    assert np.isclose(later.later, 2 / 50) and later.early == 0.0     # laps 3-60 less 2 x 4 continuation laps
    early = NeutralModel().fit(empty, [(2020, 60, [(1, "sc", 4)])])
    assert early.early == 1.0                                          # lap 2 continues lap 1's: only lap 1 counts


def test_the_qualifying_lap_is_the_median_of_the_drivers_best_laps_when_dry():
    from predictor.race_sim import qualifying_lap
    laps = pd.DataFrame({"DriverNumber": ["1", "1", "2", "3"], "Compound": ["SOFT"] * 4,
                         "LapTime": pd.to_timedelta([80.0, 79.0, 80.0, 81.0], unit="s")})
    assert qualifying_lap(laps) == 80.0
    wet = laps.assign(Compound="WET")
    assert qualifying_lap(wet) is None


def test_a_stop_costs_its_in_and_out_laps_against_the_cars_staying_out():
    from predictor.race_sim import stop_record
    laps = _race_laps({}, laps=12)
    laps["PitInTime"] = pd.Series(pd.NaT, index=laps.index, dtype="timedelta64[ns]")
    laps.loc[(laps.DriverNumber == "2") & (laps.LapNumber == 5), "PitInTime"] = pd.Timedelta(seconds=1)
    laps.loc[(laps.DriverNumber == "2") & (laps.LapNumber == 5), "LapTime"] = pd.Timedelta(seconds=100)
    laps.loc[(laps.DriverNumber == "2") & (laps.LapNumber == 6), "LapTime"] = pd.Timedelta(seconds=103)
    record = stop_record(laps)
    assert record["green"] == [23.0] and record["neutral"] == [] and np.isclose(record["stops_per_car"], 1 / 3)


def test_circuit_strategy_values_are_shrunk_towards_the_typical_race():
    from predictor.race_sim import StrategyModel
    rec = lambda loss, stops: {"green": [loss] * 6, "neutral": [loss * 0.8], "stops_per_car": stops}
    races = [("a", rec(30.0, 3.0), 0.10), ("b", rec(20.0, 1.0), 0.02), ("c", rec(22.0, 1.5), 0.04)]
    model = StrategyModel().fit(races)
    a = model.race("a", 100.0)
    assert model.pit_loss < a["pit_loss"] < 30.0 and model.stops < a["stops"] < 3.0
    assert np.isclose(model.sc_ratio, 0.8)
    assert model.race("unseen", 100.0)["pit_loss"] == model.pit_loss
    assert np.isclose(a["wear"], (0.10 + 2 * 0.04) / 3)          # % of a 100 s lap = s


def test_strategy_runs_reproducibly_and_off_means_the_same_races():
    from predictor import race_sim
    rng = np.random.default_rng(5)
    pace, grid = rng.normal(0, 0.5, (300, 8)), np.arange(1, 9, dtype=float)[None]
    model = race_sim.PassModel()
    model.coef = np.array([-0.8, -1.9, -0.3, -1.6, 0.2, 0.7])
    strategy = {"pit_loss": 22.0, "stops": 1.6, "wear": 0.05, "sc_ratio": 0.8, "residuals": np.array([0.0, 4.0])}
    run = lambda s: race_sim.simulate(pace, grid, model, 0.0, 40, 90.0, 0.3, np.random.default_rng(1), strategy=s)
    assert np.array_equal(run(strategy), run(strategy))
    assert not np.array_equal(run(strategy), run(None))
