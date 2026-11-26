import types

import numpy as np
import pandas as pd

from predictor.race import RaceModel, grid_at, grid_observed


def _view(season, sessions, results, scheduled=None):
    """A view of one event: `sessions` the types classified at the cutoff, `results`
    {type: {driver number: position}}, `scheduled` the weekend's session types (default
    `sessions`)."""
    table = pd.DataFrame({"event_key": "e", "session_type": sessions, "session_id": [f"e-{t}" for t in sessions]})
    plan = pd.DataFrame({"session_type": list(scheduled if scheduled is not None else sessions)})
    return types.SimpleNamespace(
        sessions=table, archive=types.SimpleNamespace(event_sessions=lambda event_key: plan),
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


def test_an_empty_classification_is_no_grid():
    """Review 2026-11-26: no classified position gave the entrants' roster order as slots."""
    assert grid_at(_view(2024, ["Q"], {"Q": {}}), _example(2024)) is None
    assert grid_at(_view(2024, ["Q"], {"Q": {"1": None, "2": float("nan"), "3": None}}), _example(2024)) is None
    assert not grid_observed(_view(2024, ["Q"], {"Q": {}}), _example(2024))
    assert not grid_observed(_view(2024, ["Q"], {"Q": {"1": None, "2": None, "3": None}}), _example(2024))


def test_the_grid_is_observed_once_the_scheduled_grid_session_is_classified():
    """The session that sets the grid comes from the weekend's schedule: a 2021-22 sprint
    weekend after Friday qualifying, before the sprint, has no observed grid."""
    results = {"Q": {"1": 1, "2": 2, "3": 3}, "S": {"3": 1, "1": 2, "2": 3}}
    sprint = ["FP1", "Q", "FP2", "S"]
    assert not grid_observed(_view(2021, ["FP1", "Q"], results, sprint), _example(2021))
    assert grid_observed(_view(2021, ["FP1", "Q", "FP2", "S"], results, sprint), _example(2021))
    assert grid_observed(_view(2024, ["FP1", "Q"], results), _example(2024))
    assert not grid_observed(_view(2024, ["FP1"], results, ["FP1", "Q"]), _example(2024))
    assert grid_observed(_view(2023, ["Q"], results, ["FP1", "Q", "SQ", "S"]), _example(2023))   # from 2023: qualifying
    assert not grid_observed(_view(2024, ["Q"], {"Q": {"1": 1, "2": 2}}), _example(2024))      # fewer than three
    assert grid_observed(_view(2024, ["Q"], {"Q": {"1": 1, "2": 2}}), _example(2024, ("1", "2")))


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
    moment = lambda label: types.SimpleNamespace(cutoff_label=label, event_key="2023-05", season=2023,
                                                 driver_numbers=["1", "11", "16"])
    before, after, grid, practice = (moment(x) for x in ("pre_weekend", "last", "after_grid", "after_FP2"))

    def view_with(types_run, positions=(2, 1, 3)):     # the event's sessions classified at the cutoff
        sessions = pd.DataFrame({"event_key": "2023-05", "session_type": types_run,
                                 "session_id": [f"2023-05-{t}" for t in types_run]})
        results = pd.DataFrame({"DriverNumber": ["1", "11", "16"], "Position": list(positions)})
        plan = pd.DataFrame({"session_type": ["FP1", "FP2", "FP3", "Q"]})
        return types.SimpleNamespace(sessions=sessions, results=lambda session_id: results,
                                     archive=types.SimpleNamespace(event_sessions=lambda event_key: plan))
    qualified, unqualified = view_with(["FP1", "FP2", "FP3", "Q"]), view_with(["FP1", "FP2", "FP3"])
    unclassified = view_with(["FP1", "FP2", "FP3", "Q"], (None, None, None))
    assert router.models[router.route(after, unclassified)].name == "race:mix3ffnps51aps5o"
    assert router.models[router.route(before, qualified)].name == "race:mix3fwp2apk"
    # step 9b serves the cutoff it was tested at only, with the grid observed; a practice
    # cutoff, or the last cutoff before qualifying is classified, keeps the old champion
    assert router.models[router.route(after, qualified)].name == "race:mix3ffnps51aps5of"
    assert router.models[router.route(after, unqualified)].name == "race:mix3ffnps51aps5o"
    assert router.models[router.route(practice, qualified)].name == "race:mix3ffnps51aps5o"
    assert router.models[router.route(grid, qualified)].name == "race:mix3ffnps51aps5go"
    last = router.models["last"]
    assert [c.name for c in last.components] == ["race:v3ffnps5o", "race:v1aps5of"]
    assert list(last.weights) == [0.5, 0.5]
    assert [c.front_bonus for c in last.components] == [False, True]
    assert not any(c.front_bonus for k in ("after_session", "after_grid") for c in router.models[k].components)
    # the stable settings fit before the weekend only; the old fit after qualifying
    stable = {k: all(c.stable_fit for c in m.components) for k, m in router.models.items()}
    assert stable == {"pre_weekend": True, "after_session": False, "last": False, "after_grid": False}
    # the qualifying champions have no 'after_grid' entry: the session's champion serves it
    from predictor.evaluation import ChampionRouter
    assert ChampionRouter().route(grid, qualified) == "after_session"


def test_qualifying_champions_route_by_forecast_moment_and_condition():
    from predictor.evaluation import CHAMPIONS, ChampionRouter
    router = ChampionRouter(CHAMPIONS)
    moment = lambda label, condition="dry": types.SimpleNamespace(cutoff_label=label, condition=condition)
    assert router.models[router.route(moment("pre_weekend"), None)].name == "pace:v3wruqwt-rps"
    assert router.models[router.route(moment("last"), None)].name == "pace:v3wrqwt-rp"
    assert router.models[router.route(moment("pre_weekend", "wet"), None)].name == "pace:v3wrqwt-rpc"
    assert router.models[router.route(moment("last", "wet"), None)].name == "pace:v3wrqwt-rpc"


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
    assert record["modal"] == 0                                    # two of the three cars ran without a stop


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


def test_cars_take_their_own_stop_count_by_pace_band():
    from predictor import race_sim
    pace = np.tile(np.arange(20, dtype=float), (5, 1))       # car 0 fastest
    table = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    flat = race_sim.car_deviations(table, pace).reshape(20, 3)
    assert np.allclose(flat[0], table[0]) and np.allclose(flat[8], table[1]) and np.allclose(flat[19], table[2])
    assert race_sim.car_deviations(None, pace).size == 0
    assert list(race_sim.stop_band([0, 1, 8, 9, 14, 15])) == [2, 0, 0, 1, 1, 2]


def test_an_extra_stop_without_wear_costs_its_cars_places():
    from predictor import race_sim
    rng = np.random.default_rng(3)
    pace = np.linspace(-0.4, 0.4, 16)[None] + rng.normal(0, 0.05, (400, 16))
    grid = np.arange(1, 17, dtype=float)[None]
    model = race_sim.PassModel()
    model.coef = np.array([-0.8, -1.9, -0.3, -1.6, 0.2, 0.7])
    base = {"pit_loss": 22.0, "stops": 1.0, "wear": 0.0, "sc_ratio": 0.8, "residuals": np.zeros(1), "stint_noise": 0.0}
    middle = np.zeros((3, 3)); middle[1, 0] = 1.0                 # pace ranks 9-14 stop twice in a one-stop race
    run = lambda s: race_sim.simulate(pace, grid, model, 0.0, 40, 90.0, 0.0, np.random.default_rng(1), strategy=s)
    usual = run(base)
    assert np.array_equal(usual, run(base))                        # no table: no extra random numbers
    extra = run({**base, "deviate": middle})
    assert extra[:, 8:14].mean() > usual[:, 8:14].mean() + 0.5
    assert extra[:, :8].mean() <= usual[:, :8].mean()


def test_stop_history_never_reads_past_2024(tmp_path):
    from predictor import race_sim
    tables = tmp_path / "external" / "jolpica" / "tables"
    tables.mkdir(parents=True)
    rows = [(s, 1, d, "22.0" if d != "c" else "26.0") for s in (2023, 2024, 2025) for d in ("a", "b", "c")]
    pd.DataFrame(rows, columns=["season", "round", "driverId", "duration"]).to_parquet(tables / "pitstops.parquet")
    res = [(str(s), "1", str(g), "50", "Finished", d) for s in (2023, 2024, 2025) for g, d in enumerate("abc", 1)]
    pd.DataFrame(res, columns=["season", "round", "grid", "laps", "status", "Driver_driverId"]).to_parquet(
        tables / "results.parquet")
    assert len(race_sim.history_stops(tmp_path, 2026)["residuals"]) == 6     # 2023-2024 only
    assert len(race_sim.history_stops(tmp_path, 2024)["residuals"]) == 3
    assert race_sim.history_stops(tmp_path, 2023) is None


def test_the_plan_and_the_deviations_reproduce_the_observed_stop_counts(tmp_path):
    """Four one-stop finishers and one two-stop: mean 1.20. The shared plan is the
    modal count, so the deviations are not counted twice."""
    import predictor_race_native
    from predictor import race_sim
    tables = tmp_path / "external" / "jolpica" / "tables"
    tables.mkdir(parents=True)
    stops = [(2023, 1, d, 1, "22.0") for d in "abcde"] + [(2023, 1, "e", 2, "22.0")]
    pd.DataFrame(stops, columns=["season", "round", "driverId", "stop", "duration"]).to_parquet(tables / "pitstops.parquet")
    results = [("2023", "1", str(g), "50", "Finished", d) for g, d in enumerate("abcde", 1)]
    pd.DataFrame(results, columns=["season", "round", "grid", "laps", "status", "Driver_driverId"]).to_parquet(
        tables / "results.parquet")
    history = race_sim.history_stops(tmp_path, 2024)
    assert np.isclose(history["deviate"][0, 0], 0.2)
    record = {"green": [22.0] * 6, "neutral": [], "stops_per_car": 1.2, "modal": 1}
    strategy = race_sim.StrategyModel().fit([("x", record, 0.05)], history).race("x", 90.0)
    assert strategy["stops"] == 1.0
    pace = np.tile(np.arange(5, dtype=float), (3, 1))
    counts = predictor_race_native.stop_counts(strategy["stops"], race_sim.car_deviations(strategy["deviate"], pace),
                                               5, 40000, 11)
    assert abs(counts.mean() - 1.2) < 0.01
    assert set(np.unique(counts)) == {1, 2}
    # without deviations the plan stays the mean, as before
    assert race_sim.StrategyModel().fit([("x", record, 0.05)]).race("x", 90.0)["stops"] == 1.2


def test_pit_noise_scales_slow_stops_and_stop_changes():
    from predictor.race import SimulatedRace
    strategy = {"pit_loss": 22.0, "stops": 1.0, "wear": 0.05, "sc_ratio": 0.8, "residuals": np.array([2.0, -1.0]),
                "deviate": np.full((3, 3), 0.2)}
    half = SimulatedRace._race_strategy(strategy, {"stint_noise": 0.0, "pit_noise": 0.5})
    assert np.allclose(half["residuals"], [1.0, -0.5]) and np.allclose(half["deviate"], 0.1)
    plain = SimulatedRace._race_strategy(strategy, {"stint_noise": 0.0})
    assert plain["residuals"] is strategy["residuals"] and plain["deviate"] is strategy["deviate"]
    assert SimulatedRace._race_strategy(None, {}) is None


def test_tyre_eras_and_weekend_wear_calibration():
    from predictor import race_sim
    assert [race_sim.tyre_era(s) for s in (2019, 2021, 2022, 2026)] == [0, 0, 1, 1]
    rng = np.random.default_rng(0)
    practice, past = rng.uniform(0.0, 0.2, 40), rng.uniform(0.02, 0.08, 40)
    race = 0.01 + 0.3 * practice + 0.5 * past
    model = race_sim.WeekendWear().fit(list(zip(practice, past, race)))
    assert np.isclose(model.predict(0.1, 0.05), 0.01 + 0.03 + 0.025)
    assert model.predict(None, 0.05) is None
    assert race_sim.WeekendWear().fit(list(zip(practice, past, race))[:5]).predict(0.1, 0.05) is None   # too few races
    assert np.isclose(race_sim.shrunk_mean([0.1, None, np.nan], 0.04, prior=2.0), (0.1 + 0.08) / 3)


def test_a_seasons_forecasts_do_not_depend_on_seasons_evaluated_before_it():
    """Fitting 2024's settings forecasts 2023 races with 2024's models: the strategy,
    the wear calibration and the race-to-qualifying ratio alike, whether or not 2023
    was evaluated first (standalone run against multi-season run)."""
    from predictor import race_sim
    from predictor.race import SimulatedRace
    rec = {"green": [22.0] * 6, "neutral": [], "stops_per_car": 1.5, "modal": 1}
    def season_models(wear, coef, ratio):
        strategy = race_sim.StrategyModel().fit([("x", rec, wear)])
        calibration = race_sim.WeekendWear(); calibration.coef = np.array(coef)
        return strategy, calibration, ratio
    models = {2023: season_models(0.03, [0.0, 0.2, 0.5], 1.06), 2024: season_models(0.07, [0.01, 0.4, 0.3], 1.08)}
    def model(seasons):
        m = SimulatedRace(None, "race:test", strategy=True, weekend_lap=True, weekend_wear=True)
        for s in seasons:
            m._strategies[s], m._wear_models[s], m._ratios[s] = models[s]
            m._activate(s)
        m._practice_wear = lambda view, event_key: 0.12
        return m
    standalone, multi = model([2024]), model([2023, 2024])
    example = types.SimpleNamespace(season=2023, event_key="2023-05")
    a = standalone._strategy_inputs(None, example, "x", 90.0)
    b = multi._strategy_inputs(None, example, "x", 90.0)
    expected = models[2024][1].predict(0.12, models[2024][0].circuits["x"]["wear"]) * 90.0 / 100
    assert np.isclose(a["wear"], expected) and np.isclose(b["wear"], expected)
    assert standalone._ratio == multi._ratio == 1.08
    multi._activate(2023)
    assert multi._ratio == 1.06 and not np.isclose(multi._strategy_inputs(None, example, "x", 90.0)["wear"], expected)


def _team_wear_race(devs, seed=0, laps=50, compounds=None):
    """Two cars per team, two stints (soft then hard unless `compounds` gives a team
    its own pair), each team's wear the field's plus its deviation; cars far apart
    (no traffic)."""
    rng = np.random.default_rng(seed)
    rows, lineage_of = [], {}
    for k, (team, dev) in enumerate(devs.items()):
        first, second = (compounds or {}).get(team, ("SOFT", "HARD"))
        for j in range(2):
            number = str(10 * k + j + 1)
            lineage_of[number] = team
            stop = 20 + 4 * k + 3 * j
            clock = 8.0 * (2 * k + j)
            for lap in range(1, laps + 1):
                stint, age = (1, lap) if lap <= stop else (2, lap - stop)
                wear = (0.08 if stint == 1 else 0.04) + dev
                y = 0.3 * k + 0.1 * j - 0.06 * lap + wear * age + rng.normal(0, 0.05)
                seconds = 90 * np.exp(y / 100)
                clock += seconds
                rows.append({"LapNumber": lap, "DriverNumber": number, "LapTime": pd.Timedelta(seconds=seconds),
                             "Time": pd.Timedelta(seconds=clock), "Stint": stint, "TyreLife": float(age),
                             "Compound": first if stint == 1 else second, "TrackStatus": "1", "IsAccurate": True,
                             "Deleted": False, "PitInTime": pd.NaT, "PitOutTime": pd.NaT,
                             "Position": float(2 * k + j + 1)})
    return pd.DataFrame(rows), lineage_of


def test_team_wear_deviations_recover_each_teams_wear_against_the_field():
    from predictor import race_sim
    truth = {"a": 0.02, "b": -0.01, "c": 0.0, "d": -0.01}
    laps, lineage_of = _team_wear_race(truth)
    devs = race_sim.team_wear_deviations(laps, lineage_of).set_index("lineage")
    centre = np.mean(list(truth.values()))
    for team, dev in truth.items():
        assert abs(devs.dev[team] - (dev - centre)) < 3 * devs.se[team] + 1e-3
    assert devs.se.max() < 0.005


def test_team_wear_pools_earlier_races_and_shrinks_towards_the_field():
    from predictor import race_sim
    rng = np.random.default_rng(1)
    persistent = {"a": 0.02, "b": -0.02, "c": 0.0}
    races = [pd.DataFrame({"lineage": list(persistent), "dev": [v + rng.normal(0, 0.01) for v in persistent.values()],
                           "se": 0.005}) for _ in range(30)]
    model = race_sim.TeamWear().fit(races)
    assert model.decay in race_sim.TEAM_DECAYS and model.persistent > 0
    index, mean, sd = model.predict(["a", "b", None], races)
    assert mean[0] > 0.01 and mean[1] < -0.01 and abs(mean[0]) < 0.02 + 1e-9      # shrunk, right sign
    assert index == [0, 1, 2] and mean[2] == 0.0                                    # unknown team: the field
    few = race_sim.TeamWear().fit(races[:5])
    assert few.decay is None and few.predict(["a"], races) is None


def test_a_team_confounded_with_its_own_compound_is_left_out():
    """Team d alone runs mediums then hards: its wear cannot be told from the
    medium's. With no team effect at all, no spurious deviations either."""
    from predictor import race_sim
    truth = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    laps, lineage_of = _team_wear_race(truth, compounds={"d": ("MEDIUM", "MEDIUM")})
    devs = race_sim.team_wear_deviations(laps, lineage_of)
    assert "d" not in set(devs.lineage) and len(devs) == 3
    assert (devs.dev.abs() < 4 * devs.se + 1e-3).all()


def test_team_wear_deviations_do_not_depend_on_team_labels():
    from predictor import race_sim
    truth = {"a": 0.02, "b": -0.01, "c": 0.0, "d": -0.01}
    laps, lineage_of = _team_wear_race(truth, compounds={"c": ("MEDIUM", "HARD")})
    one = race_sim.team_wear_deviations(laps, lineage_of).set_index("lineage")
    rename = {"a": "z", "b": "y", "c": "x", "d": "w"}
    two = race_sim.team_wear_deviations(laps, {k: rename[v] for k, v in lineage_of.items()}).set_index("lineage")
    for team, other in rename.items():
        assert np.isclose(one.dev[team], two.dev[other], atol=1e-9) and np.isclose(one.se[team], two.se[other], atol=1e-9)


def test_the_returned_errors_describe_the_centred_deviations():
    """With no team effect the centred deviations scatter around zero by their
    returned standard errors (over many simulated races)."""
    from predictor import race_sim
    z = []
    for seed in range(40):
        laps, lineage_of = _team_wear_race({t: 0.0 for t in "abcde"}, seed=seed,
                                           compounds={"e": ("MEDIUM", "HARD")})
        devs = race_sim.team_wear_deviations(laps, lineage_of)
        z.extend(devs.dev / devs.se)
    assert len(z) == 200 and 0.8 < np.std(z) < 1.25 and abs(np.mean(z)) < 0.2


def test_traffic_covers_cars_a_lap_apart():
    from predictor import race_sim
    t = lambda s: pd.Timedelta(seconds=s)
    laps = pd.DataFrame({"DriverNumber": ["1", "2", "3", "1"], "LapNumber": [10, 9, 9, 11],
                         "Time": [t(1000.0), t(1000.5), t(1050.0), t(1090.2)]})
    traffic = race_sim.traffic_laps(laps)
    # car 2 (a lap down) crosses 0.5 s behind the leader on its next lap: in traffic;
    # the leader, 0.5 s ahead of a car a lap down, is not being lapped: not in traffic
    assert ("2", 9) in traffic and ("1", 10) not in traffic and ("3", 9) not in traffic
    being_lapped = pd.DataFrame({"DriverNumber": ["9", "1"], "LapNumber": [20, 21], "Time": [t(500.0), t(500.6)]})
    assert race_sim.traffic_laps(being_lapped) == {("9", 20), ("1", 21)}


def test_a_mixture_pools_whole_forecasts_and_reports_its_components():
    from predictor.evaluation import ForecastMixture
    from predictor.forecast import Forecast
    def stub(name, positions):
        class Stub:
            def fit(self, history, view_for):
                return self
            def predict(self, view, example, **kwargs):
                return Forecast.from_draws(positions=np.array(positions), season=2024, session_type="R",
                                           event_key="e", cutoff_label="last", cutoff_time="t",
                                           driver_numbers=["1", "2"], driver_ids=["a", "b"], model=name, seed=0)
        return Stub
    registry = {"one": stub("one", [[1, 2], [1, 2]]), "two": stub("two", [[2, 1], [2, 1]])}
    mix = ForecastMixture(["one", "two"], [1, 3], "mix", registry=registry)
    forecast = mix.predict(None, None)
    one, two = forecast.metadata["components"]
    assert np.allclose(forecast.position_probs, 0.25 * one.position_probs + 0.75 * two.position_probs)
    assert forecast.model == "mix" and [c.model for c in (one, two)] == ["one", "two"]


def test_the_simulator_computes_the_pit_features_it_was_fitted_on():
    import predictor_race_native
    from predictor import race_sim
    rng = np.random.default_rng(4)
    for _ in range(200):
        args = dict(age=float(rng.integers(0, 50)), stops=int(rng.integers(0, 3)), lap=int(rng.integers(2, 60)), laps=60,
                    neutral=bool(rng.random() < 0.3), position=int(rng.integers(1, 21)), lost=float(rng.integers(0, 12)),
                    clean_air=float(rng.uniform(0, 30)), gap_ahead=float(rng.uniform(0, 30)), gap_behind=float(rng.uniform(0, 30)),
                    cover_behind=bool(rng.random() < 0.2), cover_ahead=bool(rng.random() < 0.2),
                    pace_adv=float(rng.normal(0, 2)), passing=float(rng.normal(0, 0.5)), wear=float(rng.uniform(-0.05, 0.4)))
        native = predictor_race_native.pit_features(**args)
        python = race_sim.pit_features(*[np.array([args[k]]) for k in ("age", "stops", "lap")], args["laps"],
                                       *[np.array([args[k]]) for k in ("neutral", "position", "lost", "clean_air", "gap_ahead",
                                                                        "gap_behind", "cover_behind", "cover_ahead", "pace_adv")],
                                       args["passing"], np.array([args["wear"]]))[0]
        assert len(native) == len(race_sim.PIT_FEATURES) and np.allclose(native, python)


def test_the_pit_model_learns_that_losing_places_deters_a_stop():
    from predictor import race_sim
    rng = np.random.default_rng(5)
    n = 30000
    f = pd.DataFrame({"age": rng.integers(0, 40, n).astype(float), "stops": 0, "lap": rng.integers(2, 50, n), "laps": 55,
                      "neutral": False, "position": rng.integers(1, 21, n), "lost": rng.integers(0, 10, n).astype(float),
                      "clean_air": rng.uniform(0, 10, n), "gap_ahead": rng.uniform(0, 10, n), "gap_behind": rng.uniform(0, 10, n),
                      "cover_behind": False, "cover_ahead": False, "pace_adv": 0.0, "wear": 0.05})
    z = -4 + 0.08 * f.age - 0.3 * f.lost
    f["pit"] = rng.random(n) < 1 / (1 + np.exp(-z))
    model = race_sim.PitModel().fit([(f, -1.0)] * 1 + [(f.iloc[:10], -1.2)] * 5)
    lost = model.coef[1 + race_sim.PIT_FEATURES.index("lost")]
    assert model.coef is not None and lost < -1.5          # per 10 cars: about -3


def _pit_race(offsets=(0.0, 8.0, 18.0, 30.0), stop_lap=6, stopper="1", delay=25.0, neutral_laps=(), laps=12,
              out_delay=0.0):
    """Cars at fixed gaps (s, at lap 1's end), 90 s laps; `stopper` pits at the end
    of `stop_lap`, its in-lap `delay` s slower and its out-lap `out_delay` s."""
    rows = []
    for k, offset in enumerate(offsets):
        d = str(k + 1)
        clock, age, stint = offset, 0, 1
        for lap in range(1, laps + 1):
            age += 1
            seconds = 90.0 + (delay if d == stopper and lap == stop_lap else 0.0)                 + (out_delay if d == stopper and lap == stop_lap + 1 else 0.0)
            clock += seconds
            rows.append({"LapNumber": lap, "DriverNumber": d, "Time": pd.Timedelta(seconds=clock),
                         "LapTime": pd.Timedelta(seconds=seconds), "TyreLife": float(age), "Stint": stint,
                         "Compound": "MEDIUM", "TrackStatus": "4" if lap in neutral_laps else "1",
                         "PitInTime": pd.Timedelta(seconds=clock) if d == stopper and lap == stop_lap else pd.NaT,
                         "PitOutTime": pd.Timedelta(seconds=clock - 85.0) if d == stopper and lap == stop_lap + 1 else pd.NaT})
            if d == stopper and lap == stop_lap:
                age, stint = 0, 2
    return pd.DataFrame(rows)


def test_pit_features_are_decided_before_the_stop_happens():
    """The stopping car's in-lap delay must not change the state its stop is
    labelled from (the field at the end of the lap before)."""
    from predictor import race_sim
    rows = lambda delay: race_sim.pit_decisions(_pit_race(delay=delay), 22.0, {}).query("driver == '1' and lap == 6")
    quick, slow = rows(5.0), rows(40.0)
    assert quick.pit.iloc[0] and slow.pit.iloc[0]
    columns = ["age", "stops", "position", "lost", "clean_air", "gap_ahead", "gap_behind"]
    assert quick[columns].equals(slow[columns])
    assert quick.position.iloc[0] == 1 and quick.age.iloc[0] == 6.0


def test_a_neutralised_lap_counts_places_at_the_neutralised_loss():
    """Car 1 has car 2 8 s and car 3 18 s behind: a 22 s green stop loses both, an
    0.8 x 22 = 17.6 s stop under a safety car only car 2."""
    from predictor import race_sim
    green = race_sim.pit_decisions(_pit_race(), 22.0, {}, sc_ratio=0.8).query("driver == '1' and lap == 6")
    under = race_sim.pit_decisions(_pit_race(neutral_laps=(6,)), 22.0, {}, sc_ratio=0.8).query("driver == '1' and lap == 6")
    assert green.lost.iloc[0] == 2 and under.lost.iloc[0] == 1 and under.neutral.iloc[0]


def test_training_and_the_simulator_build_the_same_features_from_one_race_state():
    import predictor_race_native
    from predictor import race_sim
    laps = _pit_race(neutral_laps=(6,))
    pace = {"1": -0.3, "2": 0.1, "3": 0.0, "4": 0.4}
    rows = race_sim.pit_decisions(laps, 22.0, pace, wear=0.06, sc_ratio=0.8)
    for lap in (4, 6, 7):            # 7: the lap after car 1's stop, its loss wholly charged
        here = rows[rows.lap == lap].sort_values("driver")
        before = laps[laps.LapNumber == lap - 1].sort_values("DriverNumber")
        python = race_sim.pit_features(here.age, here.stops, here.lap, 12, here.neutral, here.position, here.lost,
                                       here.clean_air, here.gap_ahead, here.gap_behind, here.cover_behind, here.cover_ahead,
                                       here.pace_adv, 0.25, here.wear)
        lap_seconds = 90.0
        native = predictor_race_native.pit_state_features(
            lap, 12, bool(here.neutral.iloc[0]), 22.0 * (0.8 if here.neutral.iloc[0] else 1.0),
            before.Time.dt.total_seconds().tolist(), [pace[d] * lap_seconds / 100 for d in before.DriverNumber],
            here.age.tolist(), [0.06 * lap_seconds / 100] * 4, here.stops.astype(int).tolist(),
            [bool(((laps.LapNumber == lap - 1) & (laps.DriverNumber == d) & laps.PitInTime.notna()).any())
             for d in before.DriverNumber], 0.25, lap_seconds)
        assert np.allclose(np.array(native), python)


def _strategy_race():
    from predictor import race_sim
    rng = np.random.default_rng(6)
    pace = np.linspace(-0.5, 0.5, 10)[None] + rng.normal(0, 0.1, (60, 10))
    model = race_sim.PassModel()
    model.coef = np.array([-0.8, -1.9, -0.3, -1.6, 0.2, 0.7])
    neutral = {"hazard": np.full(30, 0.03), "kinds": np.array([0.6, 0.3, 0.1]),
               "lengths": {"sc": np.array([3, 4]), "vsc": np.array([2]), "red": np.array([2])}, "red_end": 0.2}
    strategy = {"pit_loss": 22.0, "stops": 1.5, "wear": 0.05, "sc_ratio": 0.8, "residuals": np.array([0.0, 3.0])}
    return pace, np.arange(1, 11, dtype=float)[None], model, neutral, strategy


def test_the_trace_only_observes():
    """Tracing a race must not change any finishing position or stop."""
    from predictor import race_sim
    pace, grid, model, neutral, strategy = _strategy_race()
    run = lambda **k: race_sim.simulate(pace, grid, model, 0.0, 30, 90.0, 0.3, np.random.default_rng(2), neutral=neutral,
                                        strategy=strategy, **k)
    plain, stops = run(with_stops=True)
    traced = run(trace=True)
    assert np.array_equal(plain, traced["positions"]) and np.array_equal(stops, traced["stops"])
    assert traced["time"].shape == (60, 30, 10)
    visits = ((traced["event"] == 1) | (traced["event"] == 2)).sum(axis=1)
    red = (traced["event"] == 4).any(axis=1)
    # every pit visit is in the trace; a planned stop due at a red flag is a free tyre
    # change there (event 4), counted as a stop but not as a pit visit
    assert (visits <= stops).all() and (visits[~red] == stops[~red]).all()


def test_keyed_variants_share_their_scenarios():
    """With keyed random numbers, a race without traffic and one with it differ
    only through traffic: no traffic and no stops gives each car its own laps."""
    from predictor import race_sim
    pace, grid, model, neutral, strategy = _strategy_race()
    run = lambda **k: race_sim.simulate(pace, grid, model, 0.0, 30, 90.0, 0.3, np.random.default_rng(2), neutral=neutral,
                                        strategy=strategy, trace=True, keyed=True, **k)
    one, two = run(), run()
    assert np.array_equal(one["time"], two["time"])                     # reproducible
    free = run(free_traffic=True)
    assert np.abs(free["held"]).max() == 0.0 and np.abs(one["held"]).max() > 0.0
    fixed = run(fixed_counts=np.full((60, 10), 2))
    assert (fixed["stops"] == 2).all()                           # each car's count as given


def test_the_state_after_a_stop_has_the_whole_stop_behind_the_car():
    """Timing splits a stop between in-lap and out-lap; the simulator charges it at
    the in-lap's end. The next decisions see the same field either way."""
    from predictor import race_sim
    columns = ["driver", "lap", "age", "stops", "position", "lost", "clean_air", "gap_ahead", "gap_behind",
               "cover_behind", "cover_ahead"]
    whole = race_sim.pit_decisions(_pit_race(delay=25.0), 22.0, {})
    split = race_sim.pit_decisions(_pit_race(delay=5.0, out_delay=20.0), 22.0, {})
    after = lambda f: f[f.lap >= 7][columns].sort_values(["lap", "driver"]).reset_index(drop=True)
    assert after(whole).equals(after(split))
    # 25 s behind its old place: car 1 now runs behind car 3 (18 s back) and ahead of car 4 (30 s back)
    assert after(whole).query("lap == 7 and driver == '1'").position.iloc[0] == 3


def test_race_labels_map_to_the_nominated_compound():
    from predictor import tyres
    t = tyres.compounds()
    assert t.season.between(2018, 2024).all() and t.event_key.is_unique
    assert t.groupby("season").size().to_dict() == {2018: 21, 2019: 21, 2020: 17, 2021: 22, 2022: 22, 2023: 22, 2024: 24}
    # hard is the hardest of the three nominated, soft the softest, at every race
    assert (t.apply(lambda r: tyres.softness(r.hard) < tyres.softness(r.medium) < tyres.softness(r.soft), axis=1)).all()
    key = t[(t.season == 2023) & (t.event_name.str.startswith("Bahrain"))].event_key.iloc[0]
    assert tyres.identity(key, "SOFT")[:3] == ("18in", "C3", 3) and tyres.identity(key, "hard")[:3] == ("18in", "C1", 1)
    monaco = t[(t.season == 2018) & (t.event_name.str.startswith("Monaco"))].event_key.iloc[0]
    assert tyres.identity(monaco, "SOFT")[:3] == ("2018", "hypersoft", 6)
    assert tyres.identity(key, "INTERMEDIATE") is None


def test_a_held_up_car_keeps_a_gap_of_its_own():
    """With FOLLOW_GAPS a held car settles at a drawn gap (median about 1 s) instead
    of exactly MIN_FOLLOW behind; without it, races are unchanged."""
    from predictor import race_sim
    pace, grid, model, neutral, strategy = _strategy_race()
    model.coef = np.array([-8.0, -1.9, -0.3, -1.6, 0.2, 0.7])      # passing all but impossible: trains form
    run = lambda **k: race_sim.simulate(pace, grid, model, 0.0, 30, 90.0, 0.3, np.random.default_rng(2), neutral=neutral,
                                        strategy=strategy, trace=True, keyed=True, **k)
    table = np.linspace(0.35, 3.0, 21)
    glued, spaced = run(), run(follow=table)
    gaps = lambda t: np.diff(np.sort(t[:, 10:], axis=2), axis=2).ravel()
    at_minimum = lambda g: np.mean(np.abs(g - race_sim.MIN_FOLLOW) < 1e-6)
    assert at_minimum(gaps(glued["time"])) > 0.2 and at_minimum(gaps(spaced["time"])) < at_minimum(gaps(glued["time"])) / 3
    assert np.array_equal(run()["time"], glued["time"])


def test_the_2022_c1_and_the_2023_c0_are_one_tyre():
    """Pirelli renamed the 2022 C1 the C0 for 2023 and introduced a new C1."""
    from predictor import tyres
    t = tyres.compounds()
    first = lambda season: t[t.season == season].sort_values("round").event_key.iloc[0]
    assert tyres.identity(first(2022), "HARD")[1] == "C1" and tyres.identity(first(2023), "HARD")[1] == "C1"
    assert tyres.identity(first(2022), "HARD")[3] != tyres.identity(first(2023), "HARD")[3]
    assert tyres.spec(2022, "C1") == tyres.spec(2023, "C0") != tyres.spec(2024, "C1") == tyres.spec(2023, "C1")
    assert tyres.spec(2022, "C3") == tyres.spec(2024, "C3") != tyres.spec(2021, "C3")


def test_following_gaps_come_from_runs_behind_the_same_car():
    """Car 2 follows car 1 within 1 s for 8 laps, then drops back; car 3 is never
    within a second. Only car 2's run counts."""
    from predictor import race_sim
    rows = []
    for lap in range(1, 13):
        t1 = 90.0 * lap
        gap2 = 0.8 if lap <= 8 else 0.8 + 0.5 * (lap - 8)
        for d, t, pos in (("1", t1, 1), ("2", t1 + gap2, 2), ("3", t1 + gap2 + 6.0, 3)):
            rows.append({"LapNumber": lap, "DriverNumber": d, "Time": pd.Timedelta(seconds=t), "Position": float(pos),
                         "LapTime": pd.Timedelta(seconds=90.0), "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TrackStatus": "1",
                         "IsAccurate": True, "Deleted": False, "Compound": "MEDIUM", "TyreLife": float(lap)})
    gaps = race_sim.following_gaps(pd.DataFrame(rows))
    assert len(gaps) == 11 and np.allclose(gaps[:7], 0.8)            # car 2 from lap 2 (lap 1 is not clean)
    # only a follower faster than the car ahead is held up
    assert len(race_sim.following_gaps(pd.DataFrame(rows), {"1": 0.0, "2": -0.2, "3": 0.5})) == 11
    assert len(race_sim.following_gaps(pd.DataFrame(rows), {"1": 0.0, "2": 0.3, "3": 0.5})) == 0
    assert race_sim.follow_quantiles(gaps) is None                    # too few laps to learn from
    q = race_sim.follow_quantiles(np.r_[np.full(800, 0.2), np.full(800, 2.0)])
    assert q[0] == race_sim.MIN_FOLLOW and q[-1] == 2.0 and len(q) == 21


def test_clean_air_pace_leaves_out_laps_in_dirty_air():
    from predictor import cleanair
    from predictor.evaluation import model_registry
    t = lambda s: pd.Timedelta(seconds=s)
    laps = pd.DataFrame({"DriverNumber": ["1", "2", "3"], "LapNumber": [5, 5, 5], "Time": [t(500.0), t(500.6), t(503.0)],
                         "LapTime": [t(90.0), t(90.1), t(91.0)]})
    out = cleanair.dirty_air_removed(laps, 1.0)
    assert out.LapTime.notna().tolist() == [True, False, True]       # car 2 crossed 0.6 s behind car 1
    model = model_registry()["race1aa1"]()
    assert isinstance(model.pace, cleanair.CleanAirPaceModel) and model.pace.min_gap == 1.0
    assert model.pace.name.endswith(f"-air1v{cleanair.VERSION}")


def test_corrected_race_pace_keeps_every_lap_and_frees_following_laps():
    from predictor import cleanair
    from predictor.evaluation import model_registry
    t = lambda s: pd.Timedelta(seconds=s)
    laps = pd.DataFrame({"DriverNumber": ["1", "2", "1", "2"], "LapNumber": [1, 1, 2, 2], "Position": [1.0, 2.0, 1.0, 2.0],
                         "Time": [t(100.0), t(100.5), t(190.0), t(190.6)], "LapTime": [t(100.0), t(100.5), t(90.0), t(90.1)]})
    gaps = cleanair.start_gaps(laps)
    assert np.isinf(gaps.iloc[2]) and np.isclose(gaps.iloc[3], 0.5) and np.isnan(gaps.iloc[1])   # lap 1 has no start gap
    model = model_registry()["race1afp"]()
    assert isinstance(model.pace, cleanair.CorrectedPaceModel) and model.pace.name.endswith(f"-fp{cleanair.CORRECTED_VERSION}")


def _synthetic_race(stops, wear=0.0, laps=30, spacing=0.5, compounds=("MEDIUM", "MEDIUM")):
    """Eight cars `spacing` s apart, lap times 100 s * exp((wear * tyre age - 0.03 *
    lap) / 100) plus a driver offset; car i stops at the end of lap stops[i] (in and
    out laps marked), onto compounds[1]."""
    t, rows = pd.Timedelta, []
    for i, stop in enumerate(stops):
        clock = i * spacing
        for lap in range(1, laps + 1):
            stint = 1 if lap <= stop else 2
            age = lap if stint == 1 else lap - stop
            seconds = 100 * np.exp((wear * age - 0.03 * lap + 0.1 * i) / 100)
            clock += seconds
            rows.append({"DriverNumber": str(i + 1), "LapNumber": lap, "LapTime": t(seconds=seconds),
                         "Time": t(seconds=clock), "Position": float(i + 1), "TrackStatus": "1",
                         "PitInTime": t(seconds=clock) if lap == stop else pd.NaT,
                         "PitOutTime": t(seconds=clock - seconds) if lap == stop + 1 else pd.NaT,
                         "Compound": compounds[stint - 1], "TyreLife": float(age), "Stint": float(stint)})
    return pd.DataFrame(rows)


def test_start_gap_follows_the_previous_laps_crossing_order_after_a_pass():
    from predictor import cleanair, race_sim
    t = lambda s: pd.Timedelta(seconds=s)
    # car 2 passes car 1 on lap 2; positions are end-of-lap
    laps = pd.DataFrame({"DriverNumber": ["1", "2", "1", "2", "1", "2"], "LapNumber": [1, 1, 2, 2, 3, 3],
                         "Position": [1.0, 2.0, 2.0, 1.0, 2.0, 1.0],
                         "Time": [t(100.0), t(100.5), t(190.5), t(190.0), t(280.5), t(280.0)],
                         "LapTime": [t(100.0), t(100.5), t(90.5), t(89.5), t(90.0), t(90.0)]})
    state = race_sim.start_state(laps)
    assert np.isinf(state.start_gap.iloc[2]) and np.isclose(state.start_gap.iloc[3], 0.5)   # lap 2: car 1 led into it
    # lap 3: car 2 now leads; the car last across before it is car 1 a lap earlier (90 s): on track, not inf
    assert np.isclose(state.start_gap.iloc[4], 0.5) and np.isclose(state.start_gap.iloc[5], 90.0)
    assert state.ahead.iloc[3] == "1" and state.ahead.iloc[4] == "2"
    assert cleanair.start_gaps(laps).equals(state.start_gap)


def test_a_following_penalty_confounded_with_stint_effects_is_not_measured_or_removed():
    from predictor import cleanair, race_sim
    laps = _synthetic_race([15] * 8)            # cars start 0.5 s apart and drift back 0.1 s a lap; no penalty
    penalty, counts, se = race_sim.following_penalty(laps)
    assert counts[0] > 0 and np.isnan(penalty[:2]).all()          # the close bands: confounded with the stints
    assert np.all(np.isnan(penalty) | (np.abs(penalty) < 1e-6))    # a band that is identified reads its true 0
    out = cleanair.following_corrected(laps)
    assert np.allclose(out.LapTime.dt.total_seconds(), laps.LapTime.dt.total_seconds())


def test_lap_effect_wear_is_none_when_stops_are_synchronised_and_recovered_when_staggered():
    from predictor import race_sim
    assert race_sim.lap_effect_wear(_synthetic_race([15] * 8, compounds=("SOFT", "HARD"))) is None
    assert race_sim.lap_effect_wear(_synthetic_race([15] * 8, wear=0.05, compounds=("SOFT", "HARD"))) is None
    staggered = race_sim.lap_effect_wear(_synthetic_race([10, 11, 12, 14, 16, 18, 19, 20], wear=0.05))
    assert staggered is not None and np.isclose(staggered, 0.05, atol=1e-6)
    staggered = race_sim.lap_effect_wear(_synthetic_race([10, 11, 12, 14, 16, 18, 19, 20], wear=0.0))
    assert staggered is not None and abs(staggered) < 1e-6
    changed = race_sim.lap_effect_wear(_synthetic_race([10, 11, 12, 14, 16, 18, 19, 20], wear=0.05, compounds=("SOFT", "HARD")))
    assert changed is not None and np.isclose(changed, 0.05, atol=1e-6)      # staggered stops onto another compound


def test_a_car_in_traffic_all_race_keeps_the_race_penalty_with_its_full_uncertainty(monkeypatch):
    from predictor import cleanair, race_sim
    laps = _synthetic_race([15] * 8)
    monkeypatch.setattr(race_sim, "following_penalty",
                        lambda laps, pace=None: (np.array([0.4, 0.2, 0.1]), np.array([50, 50, 50]), np.array([0.05, 0.05, 0.2])))
    own = pd.DataFrame([{"driver": "1", "band": 0, "penalty": 0.8, "se": 0.1, "laps": 20},
                        {"driver": "2", "band": 0, "penalty": 0.0, "se": 0.1, "laps": 20},
                        {"driver": "3", "band": 0, "penalty": 0.4, "se": 0.1, "laps": 20}])
    monkeypatch.setattr(race_sim, "driver_following_penalties", lambda laps: own)
    c = cleanair.following_correction(laps)
    race, race_se = c["race"]
    assert np.allclose(race, [0.4, 0.2, 0.0]) and np.allclose(race_se, [0.05, 0.05, 0.0])   # band 3: not 2 se above 0
    tau2 = (0.16 + 0.16 + 0.0) / 3 - 0.01
    w = tau2 / (tau2 + 0.01)
    car1, var1 = c["cars"]["1"]
    assert np.isclose(car1[0], 0.4 + w * 0.4) and np.isclose(var1[0], w * 0.01)              # its own evidence, shrunk
    car5, var5 = c["cars"]["5"]
    assert np.isclose(car5[0], 0.4) and np.isclose(var5[0], tau2)                             # none: race value, tau^2
    assert car5[2] == 0.0 and var5[2] == 0.0                                                  # unsupported band: untouched


def test_the_correction_widens_the_pace_error_of_a_car_that_spent_its_race_in_traffic():
    from predictor import cleanair
    laps = _synthetic_race([15] * 8, spacing=4.0)          # 4 s apart: everyone in clean air ...
    t = pd.Timedelta
    stuck = laps.DriverNumber == "2"                       # ... except car 2, 0.5 s behind car 1 all race
    laps.loc[stuck, "Time"] = laps.loc[laps.DriverNumber == "1", "Time"].to_numpy() + t(seconds=0.5)
    correction = {"race": (np.array([0.4, 0.0, 0.0]), np.array([0.05, 0.0, 0.0])),
                  "cars": {d: (np.array([0.4, 0.0, 0.0]), np.array([0.09, 0.0, 0.0])) for d in map(str, range(1, 9))}}
    drivers = [str(d) for d in range(1, 9)]
    added = cleanair.correction_covariance(laps, correction, drivers)
    variance = np.diag(added)
    assert variance[1] > 10 * variance[0] and np.allclose(added, added.T)
    assert np.isclose(added.sum(), 0.0, atol=1e-12)        # centred on the field


def test_fresh_tyre_peak_is_recovered_after_staggered_stops_and_pooled_within_an_era():
    from predictor import race_sim
    laps = _synthetic_race([10, 11, 12, 14, 16, 18, 19, 20], wear=0.05, spacing=4.0)
    t = laps.LapTime.dt.total_seconds().to_numpy()
    out = laps[laps.PitOutTime.notna()].set_index("DriverNumber").LapNumber
    into = laps.LapNumber - laps.DriverNumber.map(out) + 1
    boost = np.select([into == 2, into == 3, into == 4], [0.3, 0.2, 0.1], 0.0)
    laps["LapTime"] = pd.to_timedelta(t * np.exp(-boost / 100), unit="s")
    peak = race_sim.fresh_peak(laps)
    assert np.allclose(peak, [0.3, 0.2, 0.1], atol=1e-6)
    assert race_sim.pooled_fresh_peak([peak] * (race_sim.MIN_FRESH_RACES - 1)) is None
    assert np.allclose(race_sim.pooled_fresh_peak([peak] * race_sim.MIN_FRESH_RACES), [0.3, 0.2, 0.1])


def test_start_gap_sees_a_car_on_another_lap():
    from predictor import race_sim
    t = lambda s: pd.Timedelta(seconds=s)
    # A laps B: A ends lap 11 at 1000 s, B ends lap 10 at 1000.5 s; B's lap 11 starts 0.5 s behind A
    laps = pd.DataFrame({"DriverNumber": ["A", "B", "A", "B", "A"], "LapNumber": [10, 10, 11, 11, 12],
                         "Time": [t(900.0), t(1000.5), t(1000.0), t(1100.5), t(1100.0)],
                         "LapTime": [t(90.0), t(100.0), t(100.0), t(100.0), t(100.0)]})
    state = race_sim.start_state(laps)
    assert np.isclose(state.start_gap.iloc[3], 0.5) and state.ahead.iloc[3] == "A"
    # A's lap 12 starts at 1000 s: no other car crossed before it in these laps
    assert np.isinf(state.start_gap.iloc[4]) and state.ahead.iloc[4] is None


def test_fresh_peak_support_counts_identified_estimates_per_lap():
    from predictor import race_sim
    nan = [np.nan] * 3
    assert race_sim.pooled_fresh_peak([[0.3, 0.2, 0.1]] + [nan] * (race_sim.MIN_FRESH_RACES - 1)) is None
    some = [[0.3, 0.2, np.nan]] * race_sim.MIN_FRESH_RACES
    assert np.allclose(race_sim.pooled_fresh_peak(some), [0.3, 0.2, 0.0])        # lap 3 unsupported: none
    gap = [[0.3, np.nan, 0.1]] * race_sim.MIN_FRESH_RACES
    assert np.allclose(race_sim.pooled_fresh_peak(gap), [0.3, 0.0, 0.0])         # nothing after an unsupported lap


def test_stop_windows_are_learned_from_green_flag_stops_of_finishers():
    from predictor import race_sim
    laps = _synthetic_race([10, 11, 12, 14, 16, 18, 19, 20], laps=40)
    windows = race_sim.stop_windows(laps)
    assert sorted(windows) == [1] and len(windows[1]) == 8
    assert np.isclose(np.mean(windows[1]), np.mean([10, 11, 12, 14, 16, 18, 19, 20]) / 40)
    pooled = race_sim.pooled_windows([windows] * 4)            # 32 one-stop cars, no two-stop ones
    assert np.allclose(pooled, [np.mean(windows[1])])
    assert race_sim.pooled_windows([windows] * 3) is None       # 24 cars: too few


def test_start_roles_and_their_window_shifts_are_learned():
    from predictor import race_sim
    laps = _synthetic_race([10, 11, 12, 14, 16, 18, 19, 26], laps=40)
    laps.loc[laps.DriverNumber == "8", "Compound"] = "HARD"         # car 8 starts on the hard role, stops late
    laps.loc[laps.DriverNumber != "8", "Compound"] = "MEDIUM"
    grid = {str(i): float(i) if i < 8 else 18.0 for i in range(1, 9)}
    roles = race_sim.start_roles(laps, grid)
    assert list(roles.role) == [1] * 7 + [2] and list(roles.band) == [0, 0, 0, 1, 1, 1, 1, 3]
    mean = np.mean([10, 11, 12, 14, 16, 18, 19, 26]) / 40
    assert np.isclose(roles.shifts.iloc[7][0], 26 / 40 - mean)
    chances, shifts = race_sim.pooled_roles([roles] * race_sim.MIN_ROLE_RACES, np.array([0.4]))
    assert np.allclose(chances.sum(axis=1), 1) and chances[3].argmax() == 2 and chances[0].argmax() == 1
    assert np.isclose(shifts[2], 26 / 40 - mean) and shifts[0] == 0.0       # no soft starters: no shift
    assert race_sim.pooled_roles([roles] * (race_sim.MIN_ROLE_RACES - 1), np.array([0.4])) is None


def test_gamble_model_learns_more_stops_for_high_wear_teams():
    from predictor import race_sim
    rng = np.random.default_rng(0)
    n = 4000
    wear = rng.normal(0, 0.03, n)
    band = rng.integers(0, 4, n)
    z = -1.8 + 25 * wear
    more = rng.uniform(size=n) < 1 / (1 + np.exp(-z))
    fewer = ~more & (rng.uniform(size=n) < 1 / (1 + np.exp(-(-2.5 - 25 * wear))))
    rows = pd.DataFrame({"modal": 2, "dev": np.where(more, 1, np.where(fewer, -1, 0)), "band": band, "wear": wear})
    model = race_sim.GambleModel().fit(rows)
    c = model.chances(np.array([5.0, 5.0]), np.array([-0.05, 0.05])).reshape(2, 3)
    assert c[1, 2] > 2 * c[0, 2] and c[0, 1] > 2 * c[1, 1]       # high wear: more; gentle: fewer
    assert np.all(c[:, 0] == 0)                                    # no one-stop races seen: no chance there
    assert list(race_sim.grid_band([1, 3, 4, 10, 11, 16, 0])) == [0, 0, 1, 1, 2, 3, 3]


def test_gamble_rows_of_a_season_do_not_depend_on_which_seasons_ran_before(monkeypatch):
    from predictor import race_sim
    from predictor.race import SimulatedRace
    import types
    sessions = pd.DataFrame({"session_id": ["r1"], "session_type": ["R"], "season": [2019], "event_key": ["2019-01"],
                             "scheduled_start_utc": [pd.Timestamp("2019-03-17", tz="UTC")]})
    view = types.SimpleNamespace(sessions=sessions, results=lambda s: pd.DataFrame({"DriverNumber": ["1"], "GridPosition": [1.0]}),
                                 entrants=lambda s: pd.DataFrame({"driver_number": ["1"], "lineage_id": ["t"]}),
                                 laps=lambda s: pd.DataFrame())
    model = SimulatedRace.__new__(SimulatedRace)
    model.gambles, model._gamble_rows, model._team_devs = True, {}, {}
    model._archive = types.SimpleNamespace(view=lambda cutoff: view)
    model._era_team_devs_before = lambda view, race: []
    seen = []
    monkeypatch.setattr(race_sim, "gamble_rows", lambda laps, grid, wear: seen.append(dict(wear)) or None)
    class Team:
        def __init__(self, value): self.value = value
        def predict(self, lineages, races): return [0], np.array([self.value]), np.array([0.0])
    model._team_models = {2020: Team(1.0), 2021: Team(10.0)}
    model._fit_gambles(2020); model._fit_gambles(2021)
    sequential = seen[-1]
    seen.clear(); model._gamble_rows = {}
    model._fit_gambles(2021)
    assert sequential == seen[-1] == {"1": 10.0}      # 2021 uses its own team model, run alone or after 2020


def test_randomised_pit_is_uniform_under_a_calibrated_forecast():
    from predictor import calibration
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(10), size=4000)
    actual = np.array([rng.choice(10, p=p) + 1 for p in probs])
    pit = calibration.randomised_pit(probs, actual, rng)
    assert abs(pit.mean() - 0.5) < 0.02 and abs(pit.var() - 1 / 12) < 0.005
    s = calibration.summary(pit, np.repeat(np.arange(200), 20), rng, resamples=200).set_index("statistic")
    assert s.low["variance"] < 1 / 12 < s.high["variance"]
    sure = np.eye(10)[actual - 1]                       # all mass on the actual position: PIT spread over its slot
    assert 0 <= calibration.randomised_pit(sure, actual, rng).min() and calibration.randomised_pit(sure, actual, rng).max() <= 1


def test_batch_mean_averages_per_race_over_seed_batches_on_common_events():
    from predictor import evaluation
    a = pd.DataFrame({"event_key": ["2019-01", "2019-01", "2019-02"], "model": ["x", "y", "x"], "season": 2019,
                      "rps": [0.10, 0.20, 0.30]})
    b = pd.DataFrame({"event_key": ["2019-01", "2019-01"], "model": ["x", "y"], "season": 2019, "rps": [0.14, 0.22]})
    m = evaluation.batch_mean([a, b]).set_index(["event_key", "model"])
    assert list(m.index) == [("2019-01", "x"), ("2019-01", "y")]          # 2019-02 only in one batch
    assert np.isclose(m.rps[("2019-01", "x")], 0.12) and np.isclose(m.rps[("2019-01", "y")], 0.21)
    assert (m.batches == 2).all()


def test_race_forecast_draws_follow_the_module_setting_at_call_time():
    import inspect
    from predictor import race
    assert inspect.signature(race.RaceModel.predict).parameters["draws"].default is None


def _pace_race(stops, paces, track, laps=40):
    """Laps of a dry race: car i at pace paces[i] (% of a lap), stopping at stops[i]
    onto a fresh set, 0.05% wear per lap of age, the track's lap effects `track(n)`
    (% of a lap) and a little noise."""
    rng = np.random.default_rng(3)
    t, rows = pd.Timedelta, []
    for i, (stop, pace) in enumerate(zip(stops, paces)):
        for n in range(1, laps + 1):
            stint = 1 if n <= stop else 2
            age = n if stint == 1 else n - stop
            log = 100 * np.log(90.0) + pace + 0.05 * age + track(n) + rng.normal(0, 0.02)
            rows.append({"DriverNumber": str(i + 1), "LapNumber": n, "LapTime": t(seconds=float(np.exp(log / 100))),
                         "TrackStatus": "1", "PitInTime": t(seconds=1) if n == stop else pd.NaT,
                         "PitOutTime": t(seconds=1) if n == stop + 1 else pd.NaT,
                         "Compound": "MEDIUM" if stint == 1 else "HARD", "TyreLife": float(age), "Stint": float(stint)})
    return pd.DataFrame(rows)


def test_race_pace_defaults_are_unchanged_by_the_new_options():
    import importlib.util, pathlib, os
    from predictor import laps as now
    before_path = pathlib.Path(os.environ.get("RACE_PACE_BEFORE", "")) if os.environ.get("RACE_PACE_BEFORE") else None
    race = _pace_race([12, 14, 16, 18, 20, 22, 24, 26], [0.0, 0.1, 0.3, 0.35, 0.5, 0.8, 0.9, 1.2],
                      lambda n: -0.065 * n + (0.4 if n % 6 == 0 else 0.0))
    a = now.race_pace(race)
    b = now.race_pace(race, lap_effects=False, partial=False, min_laps=now.MIN_LAPS)
    pd.testing.assert_frame_equal(a, b)
    if before_path is not None and before_path.exists():      # against the code before the options (CI: optional)
        spec = importlib.util.spec_from_file_location("laps_before", before_path)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        pd.testing.assert_frame_equal(a, old.race_pace(race))


def test_lap_effects_recover_driver_pace_when_the_fuel_rate_is_wrong():
    from predictor import laps
    paces = [0.0, 0.1, 0.3, 0.35, 0.5, 0.8, 0.9, 1.2]
    # fuel and track: 0.12% a lap faster (not the fixed 0.065) plus slow laps every sixth lap
    race = _pace_race([12, 14, 16, 18, 20, 22, 24, 26], paces, lambda n: -0.12 * n + (0.4 if n % 6 == 0 else 0.0))
    fit = laps.race_pace(race, lap_effects=True).set_index("driver_number").pace
    truth = pd.Series(paces, index=[str(i + 1) for i in range(8)])
    err = (fit - fit.mean()) - (truth - truth.mean())
    assert err.abs().max() < 0.02, err


def test_partial_keeps_the_drivers_the_laps_connect():
    from predictor import laps
    paces = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    race = _pace_race([10, 12, 14, 16, 18, 20], paces, lambda n: -0.065 * n)
    # car 6 runs only laps the others do not, on a compound nobody else uses: it shares no lap effect or tyre
    car6 = race.DriverNumber == "6"
    race.loc[car6, "LapNumber"] += 100
    race.loc[car6, "Compound"] = "SOFT"
    assert laps.race_pace(race, lap_effects=True).empty                         # not all separable: dropped
    kept = laps.race_pace(race, lap_effects=True, partial=True)
    assert sorted(kept.driver_number) == ["1", "2", "3", "4", "5"]


def test_gate_b_models_read_race_pace_with_the_connected_group_only():
    from predictor import evaluation
    reg = evaluation.model_registry()
    for name in ("pace3wrqwtrp", "race1ap", "race2ap", "race3ffnp", "race3fwp", "race_mix3ffnp1ap", "race_mix3fwp2ap"):
        assert name in reg
    assert reg["pace3wrqwtrp"]().name == "pace:v3wrqwt-rp"
    pace = evaluation._pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True, robust=True,
                                  race_read={"partial": True})
    assert pace.race_read == {"partial": True}
    assert pace.name.endswith("-rp")


def test_stint_scale_uses_earlier_seasons_only_and_measures_the_understatement():
    from predictor import stintnoise
    rng = np.random.default_rng(0)
    var = rng.uniform(0.01, 0.09, 4000)
    season = np.repeat([2018, 2019, 2020, 2021], 1000)
    c = np.where(season == 2018, 4.0, 9.0)                       # later seasons understated more
    rows = pd.DataFrame({"season": season, "k": 5, "var": var, "se": np.sqrt(var),
                         "err": rng.normal(0, np.sqrt(c * var))})
    scales = stintnoise.scale_by_season(rows, [2018, 2019, 2021])
    assert 2018 not in scales                                    # nothing earlier
    assert abs(scales[2019] - 4.0) < 0.5                         # 2018 only: never sees the 9s
    assert 5.0 < scales[2021] < 8.5                              # 2018-2020
    h = stintnoise.honesty(rows, {2019: 9.0})
    assert abs(h.z_scaled.iloc[0] - 1.0) < 0.1 and h.z_raw.iloc[0] > 2.5


def _stint_race(rng, stint_sd, cars=10, laps=40, noise=0.3):
    """A dry race: cars stop once (laps 15-25), MEDIUM then HARD, iid lap noise and a
    random offset per stint of sd `stint_sd` (% of a lap)."""
    t, rows = pd.Timedelta, []
    for i in range(cars):
        stop = int(rng.integers(15, 26))
        offsets = rng.normal(0, stint_sd, 2)
        pace = 0.1 * i
        for n in range(1, laps + 1):
            stint = 1 if n <= stop else 2
            age = n if stint == 1 else n - stop
            log = 100 * np.log(90.0) + pace + 0.05 * age - 0.065 * n + offsets[stint - 1] + rng.normal(0, noise)
            rows.append({"DriverNumber": str(i + 1), "LapNumber": n, "LapTime": t(seconds=float(np.exp(log / 100))),
                         "TrackStatus": "1", "PitInTime": t(seconds=1) if n == stop else pd.NaT,
                         "PitOutTime": t(seconds=1) if n == stop + 1 else pd.NaT,
                         "Compound": "MEDIUM" if stint == 1 else "HARD", "TyreLife": float(age), "Stint": float(stint)})
    return pd.DataFrame(rows)


def test_stint_scale_is_one_for_independent_laps_and_above_one_with_stint_errors():
    # review case: a cut compared against a reference containing it gave 0.50 for
    # independent laps; split drivers in one fit (disjoint parts, joint covariance) give 1
    from predictor import stintnoise
    rng = np.random.default_rng(11)
    iid = pd.DataFrame([r for _ in range(40) for r in stintnoise.race_rows(_stint_race(rng, 0.0), rng, drivers=3,
                                                                          cuts=(5, 10, 14))])
    assert len(iid) > 200
    assert 0.8 < stintnoise.scale(iid) < 1.25
    stint = pd.DataFrame([r for _ in range(40) for r in stintnoise.race_rows(_stint_race(rng, 0.25), rng, drivers=3,
                                                                            cuts=(5, 10, 14))])
    assert stintnoise.scale(stint) > 1.8
    assert (stint.err.abs() > 0).all() and (stint.rest_laps >= stintnoise.MIN_REST_LAPS).all()


def test_stint_scale_table_is_walk_forward_and_sets_the_season_fit_start():
    from predictor import pace, evaluation
    assert pace.stint_scale(2019) == pace.STINT_SCALE[2019]
    assert pace.stint_scale(2026) == pace.STINT_SCALE_LATER and pace.stint_scale(2018) == pace.STINT_SCALE[2019]
    m = evaluation.model_registry()["pace3wrqwtrps5"]()
    assert m.race_estimates and m.stint_scale and m.race_read == {"partial": True, "min_laps": 5}
    # the scale is held, not fitted: estimates are off in the fitted settings of a stint-scale model
    seen = {}

    def fake_fit_hyper(earlier, sessions, start, frozen=()):
        seen["scale"] = start.estimate_scale
        return start
    m.fit_hyper = fake_fit_hyper
    m.shared_from_qualifying = False

    class V:
        class archive:
            @staticmethod
            def view(t):
                return V
        sessions = pd.DataFrame({"session_id": ["a"] * 30, "session_type": ["Q"] * 30, "event_key": ["2018-01"] * 30})
    m.sessions = lambda view: V.sessions
    m._fit_season(V, 2021, m._default_start())
    assert seen["scale"] == pace.STINT_SCALE[2021] and m.hyper.estimate_scale == pace.STINT_SCALE[2021]


def test_calibration_handles_races_of_different_field_sizes():
    from predictor import calibration
    forecasts = pd.DataFrame({
        "model": "m", "excluded": False, "event_key": ["2019-01", "2019-01", "2019-02", "2019-02", "2019-02"],
        "actual": [1, 2, 1, 3, 2],
        "probs": [np.array([0.7, 0.3]), np.array([0.3, 0.7]),
                  np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5]), np.array([0.3, 0.4, 0.3])]})
    overall, _ = calibration.diagnose(forecasts, "m", resamples=50)
    assert np.isfinite(overall.value).all()
    # one field size: the same PIT values as a single call
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    same = forecasts.iloc[2:]
    a = calibration.forecast_pit(same.probs.to_numpy(), same.actual.to_numpy(), rng_a)
    b = calibration.randomised_pit(np.stack(same.probs.to_numpy()), same.actual.to_numpy(), rng_b)
    assert np.allclose(a, b)
    # mixed sizes: each row's PIT within its own CDF step
    pit = calibration.forecast_pit(forecasts.probs.to_numpy(), forecasts.actual.to_numpy(), np.random.default_rng(0))
    assert 0.0 <= pit[0] <= 0.7 and 0.3 <= pit[1] <= 1.0 and 0.5 <= pit[3] <= 1.0


def _grid_archive(tmp_path, rows):
    (tmp_path / "reference").mkdir()
    pd.DataFrame(rows, columns=["event_key", "grid_position", "status", "driver_number", "published_utc",
                                "doc_number"]).assign(
        published_utc=lambda g: pd.to_datetime(g.published_utc, utc=True)).to_parquet(
        tmp_path / "reference" / "starting_grids.parquet")
    return types.SimpleNamespace(root=tmp_path)


def test_the_published_grid_replaces_qualifying_order_once_out(tmp_path):
    from predictor.race import grid_published, published_grid_at
    archive = _grid_archive(tmp_path, [
        ("2023-01", 1, "grid", "16", "2023-03-04 22:00", 27), ("2023-01", 2, "grid", "1", "2023-03-04 22:00", 27),
        ("2023-01", 3, "grid", "44", "2023-03-04 22:00", 27), ("2023-01", 4, "grid", "63", "2023-03-04 22:00", 27),
        # final grid: car 16 starts from the pit lane, car 63 is withdrawn
        ("2023-01", 1, "grid", "1", "2023-03-05 13:00", 40), ("2023-01", 2, "grid", "44", "2023-03-05 13:00", 40),
        ("2023-01", 3, "pit_lane", "16", "2023-03-05 13:00", 40),
        ("2023-01", None, "withdrawn", "63", "2023-03-05 13:00", 40)])
    example = types.SimpleNamespace(event_key="2023-01", driver_numbers=[16, 44, 1, 63, 77])
    view = lambda t: types.SimpleNamespace(archive=archive, cutoff=pd.Timestamp(t, tz="UTC"))
    assert published_grid_at(view("2023-03-04 21:00"), example) is None
    slots, withdrawn = published_grid_at(view("2023-03-04 23:00"), example)
    assert slots.tolist() == [1, 3, 2, 4, 5] and not withdrawn.any()      # car 77 not on it: behind
    slots, withdrawn = published_grid_at(view("2023-03-05 14:00"), example)
    assert slots.tolist() == [3, 2, 1, 5, 4] and withdrawn.tolist() == [False, False, False, True, False]
    assert grid_published(archive, "2023-01", pd.Timestamp("2023-03-05 15:00", tz="UTC")) == \
        pd.Timestamp("2023-03-05 13:00", tz="UTC")
    assert grid_published(archive, "2023-01", pd.Timestamp("2023-03-05 12:00", tz="UTC")) == \
        pd.Timestamp("2023-03-04 22:00", tz="UTC")
    assert grid_published(archive, "2023-02", pd.Timestamp("2023-03-05 15:00", tz="UTC")) is None


def test_withdrawn_entrants_finish_behind_everyone_and_free_their_places():
    from predictor.race import withdraw
    from predictor.targets import OUTCOMES
    example = types.SimpleNamespace(season=2023, session_type="R")
    final = np.array([[1, 2, 3, 4], [2, 1, 4, 3]])
    win, pts, mech = (OUTCOMES.index(o) for o in ("win", "other_points", "mechanical"))
    outcomes = np.array([[win, pts, pts, pts], [pts, win, mech, pts]])
    moved, again = withdraw(final, outcomes, np.array([True, False, False, False]), example)
    assert moved.tolist() == [[4, 1, 2, 3], [4, 1, 3, 2]]
    assert again[0, 1] == win and again[1, 1] == win and again[1, 2] == mech
    # the withdrawn car drawn as the winner does not start: no win left on it
    assert (again[:, 0] == OUTCOMES.index("incident_or_other")).all()


def test_grid_models_change_only_the_grid_they_read():
    from predictor import evaluation
    reg = evaluation.model_registry()
    model = reg["race_mix3ffnps51aps5g"]()
    assert model.name == "race:mix3ffnps51aps5g"
    assert [c.name for c in model.components] == ["race:v3ffnps5g", "race:v1aps5g"]
    assert all(c.published_grid for c in model.components)
    assert not any(c.published_grid for c in reg["race_mix3ffnps51aps5"]().components)


def _grid(rule, penalties, n=6):
    """Grid order (qualifying slots, pole first) from qualifying order 1..n."""
    from predictor.gridrules import apply_grid_penalties
    drivers = [str(q) for q in range(1, n + 1)]
    slots = apply_grid_penalties(np.arange(1, n + 1), drivers, penalties, rule)[0]
    return [int(q) for q in np.argsort(slots) + 1]


def test_empty_grid_procedure_of_2024_article_42_3():
    from predictor.gridrules import GridPenalty as P
    for rule in ("art42", "drops"):
        # the first two qualifiers each get 3 places: temporary slots 4 and 5, the
        # unpenalised drivers fill 1-3 and 6
        assert _grid(rule, {"1": P(3), "2": P(3)}) == [3, 4, 5, 1, 2, 6]
        # a shared temporary slot (3): the slower qualifier keeps it, the other goes just ahead
        assert _grid(rule, {"1": P(2), "2": P(1)}) == [3, 1, 2, 4, 5, 6]
        # a temporary slot past the end: penalised drivers move up into the gap
        assert _grid(rule, {"5": P(10)}) == [1, 2, 3, 4, 6, 5]
        assert _grid(rule, {}) == [1, 2, 3, 4, 5, 6]
    # Monza 2022's provisional grid: Verstappen (2nd, 5 places) starts 7th
    quali = ["16", "1", "55", "11", "44", "63", "4", "3", "10", "14", "31", "77", "45", "24", "22", "6", "5", "18",
             "20", "47"]
    monza = {"1": P(5), "11": P(10), "31": P(5), "77": P(15), "20": P(15), "47": P(15), "55": P(back=True),
             "44": P(back=True), "22": P(back=True)}
    order = [quali[q - 1] for q in _grid("drops", {str(quali.index(d) + 1): p for d, p in monza.items()}, 20)]
    assert order == ["16", "63", "4", "3", "10", "14", "1", "45", "24", "6", "5", "18", "11", "31", "77", "20", "47",
                     "55", "44", "22"]


def test_more_than_15_places_is_the_back_of_the_grid_in_qualifying_order():
    from predictor.gridrules import GridPenalty as P, procedure
    n = 20
    # from October 2022: 75 places (car 2), 20 (car 5) and the back of the grid (car 9)
    # line up at the back in qualifying order; exactly 15 (car 1) is still a drop
    penalties = {"2": P(75), "5": P(20), "9": P(back=True), "1": P(15)}
    order = _grid("art42", penalties, n)
    assert order[-3:] == [2, 5, 9]
    assert order.index(1) < 17
    # before, drops adding up past 15 stayed drops (Bottas, Belgium 2022): only the
    # decision sending car 9 back puts it behind
    order = _grid("drops", penalties, n)
    assert order[-1] == 9 and order[-3:-1] == [5, 2]               # temporary slots 25 and 77
    assert _grid("art42", {"3": P(20), "4": P(3)}, 6)[-1] == 3
    assert _grid("drops", {"3": P(20), "4": P(3)}, 6)[-1] == 3        # alone at the end either way
    # pit-lane starters behind the back of the grid
    assert _grid("art42", {"1": P(pit_lane=True), "2": P(back=True)}) == [3, 4, 5, 6, 2, 1]
    ts = lambda t: pd.Timestamp(t, tz="UTC")
    assert procedure(2021, ts("2021-12-12")) == "drops"
    assert procedure(2022, ts("2022-10-09")) == "drops"
    assert procedure(2022, ts("2022-10-22")) == "art42"
    assert procedure(2024, ts("2024-03-02")) == "art42"


def test_known_penalties_are_those_served_in_the_race_and_published_by_the_cutoff(tmp_path):
    from predictor.gridrules import GridPenalty as P
    from predictor.race import known_penalties
    (tmp_path / "reference").mkdir()
    pd.DataFrame({"event_key": ["2023-02"] * 4 + ["2023-03"], "served": ["R", "R", "R", "S", "R"],
                  "driver_number": ["16", "16", "4", "77", "1"],
                  "places": pd.array([10, 5, None, 3, 3], dtype="Int64"),
                  "back_of_grid": [False, False, True, False, False],
                  "pit_lane": [False] * 5, "doc_number": [1, 2, 3, 4, 5],
                  "published_utc": pd.to_datetime(["2023-03-17 16:51", "2023-03-18 12:00", "2023-03-18 20:00",
                                                   "2023-03-18 09:00", "2023-03-17 10:00"], utc=True)}).to_parquet(
        tmp_path / "reference" / "grid_penalties.parquet")
    view = lambda t: types.SimpleNamespace(archive=types.SimpleNamespace(root=tmp_path), cutoff=pd.Timestamp(t, tz="UTC"))
    assert known_penalties(view("2023-03-17 12:00"), "2023-02") == {}
    assert known_penalties(view("2023-03-18 13:00"), "2023-02") == {"16": P(15, first=0)}
    assert known_penalties(view("2023-03-19 13:00"), "2023-02") == {"16": P(15, first=0), "4": P(0, True, first=2)}
    # the sprint's penalty (car 77) is served in the sprint, not the race
    assert known_penalties(view("2023-03-19 13:00"), "2023-02", served="S") == {"77": P(3, first=0)}


def test_penalty_models_set_both_grid_flags():
    from predictor import evaluation
    reg = evaluation.model_registry()
    for key, name in (("race_mix3ffnps51aps5k", "race:mix3ffnps51aps5k"), ("race_mix3fwp2apk", "race:mix3fwp2apk")):
        model = reg[key]()
        assert model.name == name
        assert all(c.published_grid and c.known_penalties and c.name.endswith("k") for c in model.components)


def test_a_withdrawn_car_takes_no_part_in_the_simulated_race():
    from predictor import race_sim
    from predictor.race import SimulatedRace
    model = SimulatedRace.__new__(SimulatedRace)
    model.passes = race_sim.PassModel()
    model.passes.coef = np.array([-0.8, -1.9, -0.3, -1.6, 0.2, 0.7])
    model._follow = model._peak = model._window = model._role = model._penalty = None
    model._release = 0.0
    rng = np.random.default_rng(6)
    pace = rng.normal(0, 0.4, (300, 6))
    grid = np.array([[3.0, 1.0, 2.0, 4.0, 5.0, 6.0]])
    strategy = {"pit_loss": 22.0, "stops": 1.6, "wear": 0.05, "sc_ratio": 0.8, "residuals": np.array([0.0, 4.0]),
                "team": np.array([0, 1, 1, 2, 2, 0]), "team_mean": np.array([0.01, -0.02, 0.0]),
                "team_sd": np.array([0.0, 0.0, 0.0])}
    withdrawn = np.array([False, False, False, False, False, True])
    context = {"circuit_term": 0.0, "laps": 40, "lap_seconds": 90.0, "strategy": strategy, "withdrawn": withdrawn}
    setting = {"sigma": 0.0, "start_noise": 0.3}
    runs = []
    for withdrawn_pace in (-3.0, 0.0, 3.0):                 # the withdrawn car very fast, level, very slow
        pace[:, 5] = withdrawn_pace
        runs.append(model.run(pace.copy(), grid, setting, context, 1))
    assert all(np.array_equal(runs[0], other) for other in runs[1:])
    assert (runs[0][:, 5] == 6).all() and (np.sort(runs[0][:, :5], axis=1) == np.arange(1, 6)).all()


def test_a_replaced_penalty_counts_until_its_replacement_is_published(tmp_path):
    from predictor.gridrules import GridPenalty as P
    from predictor.race import known_penalties
    # Belgium 2022, Bottas: 15 places (decision 18) and ten (decision 26, 15:01) on
    # Friday; decision 42 replaced the ten with five on Saturday at 09:35
    (tmp_path / "reference").mkdir()
    pd.DataFrame({"event_key": ["2022-14"] * 3, "served": ["R"] * 3, "driver_number": ["77"] * 3,
                  "places": pd.array([15, 10, 5], dtype="Int64"), "back_of_grid": [False] * 3,
                  "pit_lane": [False] * 3, "doc_number": [18, 26, 42],
                  "published_utc": pd.to_datetime(["2022-08-26 14:18", "2022-08-26 15:01", "2022-08-27 09:35"], utc=True),
                  "superseded_utc": pd.to_datetime([None, "2022-08-27 09:35", None], utc=True)}).to_parquet(
        tmp_path / "reference" / "grid_penalties.parquet")
    view = lambda t: types.SimpleNamespace(archive=types.SimpleNamespace(root=tmp_path), cutoff=pd.Timestamp(t, tz="UTC"))
    assert known_penalties(view("2022-08-26 16:00"), "2022-14") == {"77": P(25, first=0)}
    assert known_penalties(view("2022-08-27 09:34"), "2022-14") == {"77": P(25, first=0)}
    assert known_penalties(view("2022-08-27 09:35"), "2022-14") == {"77": P(20, first=0)}


def test_keyed_draws_keep_every_other_event_when_only_stop_timing_changes():
    from predictor import race_sim
    # identical cars, no tyre wear, a negligible pit loss (stops only happen with a loss
    # above zero; two zero cost residuals, so each stop still draws a number): moving the
    # stops changes nothing real, so with keyed random numbers every order stays the same
    pace, grid = np.zeros((300, 8)), np.arange(1, 9, dtype=float)[None]
    model = race_sim.PassModel()
    model.coef = np.array([-0.8, -1.9, -0.3, -1.6, 0.2, 0.7])
    run = lambda stint, keyed: race_sim.simulate(
        pace, grid, model, 0.0, 40, 90.0, 0.3, np.random.default_rng(1), keyed=keyed, with_stops=True,
        strategy={"pit_loss": 1e-9, "stops": 1.6, "wear": 0.0, "sc_ratio": 1.0, "residuals": np.array([0.0, 0.0]),
                  "stint_noise": stint})
    keyed = run(0.0, True)
    assert keyed[1].mean() > 1                                       # the cars do stop
    assert np.array_equal(keyed[0], run(3.0, True)[0])
    sequential = (run(0.0, False)[0] != run(3.0, False)[0]).any(axis=1)
    assert sequential.mean() > 0.9                                   # the sequential stream shifts


def test_the_stable_fit_smooths_the_choice_over_neighbouring_settings():
    from predictor.race import fit_seed, smoothed
    settings = [{"a": a, "b": b} for a in (0, 1, 2) for b in (0, 1)]
    # a lone low candidate (a=2, b=1) among high neighbours loses to the low region (a=0)
    means = np.array([0.09, 0.09, 0.11, 0.12, 0.12, 0.085])
    assert int(means.argmin()) == 5
    assert int(smoothed(settings, means).argmin()) == 0
    assert np.allclose(smoothed(settings, np.full(6, 0.1)), 0.1)
    example = types.SimpleNamespace(event_key="2020-01", session_type="R", cutoff_label="last")
    from predictor.baselines import seed_for
    assert fit_seed(example, own=False) == seed_for(example) != fit_seed(example)


def test_old_fit_variants_switch_off_the_stable_fit():
    from predictor import evaluation
    reg = evaluation.model_registry()
    for key in ("race_mix3ffnps51aps5o", "race_mix3fwp2apko", "race_mix3ffnps51aps5go"):
        old = reg[key]()
        assert old.name.endswith("o") and all(not c.stable_fit for c in old.components)
        new = reg[key[:-1]]()
        assert all(c.stable_fit for c in new.components)


def test_both_race_models_choose_by_the_smoothed_mean_unless_on_the_old_fit():
    from predictor.race import RaceModel, SimulatedRace
    settings = [{"x": x} for x in (0, 1, 2, 3)]
    means = np.array([0.10, 0.10, 0.12, 0.09])      # a lone low candidate at the edge
    for cls in (RaceModel, SimulatedRace):
        model = cls.__new__(cls)
        assert model._choose(settings, means) in (0, 1)    # smoothed: the low region wins
        model.stable_fit = False
        assert model._choose(settings, means) == 3         # the old fit: the lowest mean


def test_split_grid_weights_qualifying_and_start_slots_apart():
    from predictor.race import RaceModel, RHOS
    model = RaceModel.__new__(RaceModel)
    rng = np.random.default_rng(3)
    race = rng.normal(0, 0.3, (500, 4))
    qualified = np.array([1.0, 2.0, 3.0, 4.0])
    grid = np.array([[4.0, 1.0, 2.0, 3.0]])                    # the pole-sitter penalised to the back
    plain = {"kappa": 0.2, "sigma": 0.1}
    base = model.order(race, grid, plain, {"difficulty": 1.0}, 7)
    # rho 1 is exactly the single grid term; without the qualifying slots the term is the single one
    assert np.array_equal(model.order(race, grid, {**plain, "rho": 1.0}, {"difficulty": 1.0, "qualified": qualified}, 7), base)
    assert np.array_equal(model.order(race, grid, {**plain, "rho": 0.0}, {"difficulty": 1.0, "qualified": None}, 7), base)
    # rho 0: the start slot adds nothing to the qualifying slot
    split = model.order(race, grid, {**plain, "rho": 0.0}, {"difficulty": 1.0, "qualified": qualified}, 7)
    assert np.array_equal(split, model.order(race, qualified[None, :], plain, {"difficulty": 1.0}, 7))
    assert (split[:, 0] == 1).mean() > (base[:, 0] == 1).mean()
    # a simulated grid (many rows) is never split
    many = np.tile(grid, (500, 1))
    assert np.array_equal(model.order(race, many, {**plain, "rho": 0.0}, {"difficulty": 1.0, "qualified": qualified}, 7),
                          model.order(race, many, plain, {"difficulty": 1.0}, 7))
    # candidates: rho outermost, 1 first (kept on ties), only with the split
    model.overtaking, model.split_grid = False, True
    tried = list(model.candidates())
    assert [c["rho"] for c in tried[:: len(tried) // len(RHOS)]] == list(RHOS) and RHOS[0] == 1.0
    assert model.neutral()["rho"] == 1.0
    model.split_grid = False
    assert all("rho" not in c for c in model.candidates()) and "rho" not in model.neutral()


def test_split_grid_ties_keep_the_single_grid_term():
    from predictor.race import RaceModel, RHOS
    model = RaceModel.__new__(RaceModel)
    model.overtaking, model.split_grid = False, True
    settings = list(model.candidates())
    # no earlier race separates qualifying from start: every rho scores the same
    per = {(s["kappa"], s["sigma"]): 0.1 + 0.001 * s["kappa"] + 0.0001 * s["sigma"] for s in settings}
    means = np.array([per[(s["kappa"], s["sigma"])] for s in settings])
    for stable in (True, False):
        model.stable_fit = stable
        assert settings[model._choose(settings, means)]["rho"] == RHOS[0] == 1.0


def test_split_grid_drops_withdrawn_cars_from_the_qualifying_slots():
    from predictor.race import RaceModel
    model = RaceModel.__new__(RaceModel)
    seen = {}
    model.order = lambda race, grid, setting, context, seed: seen.update(q=context["qualified"], g=grid) or \
        np.tile(np.arange(1, grid.shape[1] + 1), (race.shape[0], 1))
    withdrawn = np.array([False, True, False])
    model.run(np.zeros((2, 3)), np.array([[1.0, 3.0, 2.0]]), {"kappa": 0.1, "sigma": 0.0, "rho": 0.5},
              {"difficulty": 1.0, "withdrawn": withdrawn, "qualified": np.array([1.0, 2.0, 3.0])}, 1)
    assert seen["q"].tolist() == [1.0, 3.0] and seen["g"].tolist() == [[1.0, 2.0]]


def test_split_grid_variants_set_the_split_on_the_plain_model_only():
    from predictor import evaluation
    from predictor.race import SimulatedRace
    model = evaluation.model_registry()["race_mix3ffnps51aps5kqo"]()
    for part in model.components:
        assert part.split_grid == (not isinstance(part, SimulatedRace))
        assert part.known_penalties and part.published_grid and not part.stable_fit
    assert not any(c.split_grid for c in evaluation.model_registry()["race_mix3ffnps51aps5ko"]().components)


def test_smoothing_without_rho_is_unchanged_and_rho_slices_are_smoothed_alone():
    from predictor.race import smoothed
    settings = [{"a": a, "b": b} for a in (0, 1, 2) for b in (0, 1)]
    means = np.array([0.09, 0.09, 0.11, 0.12, 0.12, 0.085])
    assert np.array_equal(smoothed(settings, means), smoothed(settings, means, ()))
    with_rho = [{**s, "rho": r} for r in (1.0, 0.5) for s in settings]
    out = smoothed(with_rho, np.concatenate([means, means + 1]))
    assert np.array_equal(out[:6], smoothed(settings, means)) and np.allclose(out[6:], out[:6] + 1)


def test_season_settings_score_with_retirements_fitted_at_the_season_start(monkeypatch):
    """Final review 2026-11-26: the settings for a season must not depend on how many of
    its races were in the history when it was first forecast."""
    class Retirements:
        def __init__(self):
            self.seasons = None

        def fit(self, history):
            self.seasons = sorted({e.season for e in history})
    model = RaceModel.__new__(RaceModel)
    races = [types.SimpleNamespace(season=s, cutoff_label="last", session_type="R") for s in (2022, 2023, 2024, 2024)]
    model._history, model.retirements = races, Retirements()
    model.retirements.fit(races)                         # as in fit(): rounds of 2024 included
    seen = []
    monkeypatch.setattr(RaceModel, "_fit_season_settings",
                        lambda self, season, cutoff, earlier: seen.append(self.retirements.seasons))
    model._fit_season(2024, "last")
    assert seen == [[2022, 2023]]
    assert model.retirements.seasons == [2022, 2023, 2024]     # the current model is left as it was


def test_the_native_simulator_inputs_are_checked_before_the_call():
    """Final review 2026-11-26: NaN or no entrants panicked in the kernel; a negative
    team index was silently coerced. The wrapper raises a ValueError first."""
    import pytest
    from predictor import race_sim
    passes = _passing(False)
    pace = np.zeros((50, 4))
    ok = race_sim.simulate(pace, np.arange(1, 5.0), passes, 0.0, 10, 90.0, 0.4, np.random.default_rng(0))
    assert ok.shape == (50, 4)
    bad = pace.copy()
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        race_sim.simulate(bad, np.arange(1, 5.0), passes, 0.0, 10, 90.0, 0.4, np.random.default_rng(0))
    with pytest.raises(ValueError, match="one draw and one entrant"):
        race_sim.simulate(np.zeros((50, 0)), np.zeros(0), passes, 0.0, 10, 90.0, 0.4, np.random.default_rng(0))
    with pytest.raises(ValueError, match="not finite"):
        race_sim.simulate(pace, np.arange(1, 5.0), passes, 0.0, 10, float("nan"), 0.4, np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least one lap"):
        race_sim.simulate(pace, np.arange(1, 5.0), passes, 0.0, 0, 90.0, 0.4, np.random.default_rng(0))
    with pytest.raises(ValueError, match="negative"):
        race_sim.simulate(pace, np.arange(1, 5.0), passes, 0.0, 10, 90.0, 0.4, np.random.default_rng(0),
                          fixed_counts=-np.ones((50, 4), dtype=np.int64))
