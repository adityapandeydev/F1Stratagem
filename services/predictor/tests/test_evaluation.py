

def test_the_roster_is_known_at_the_cutoff():
    """The entry list is the roster; a regular driver who ran the car's latest session
    in place of the entered driver is the one known at the cutoff (forecast, not
    scored); practice-only stand-ins do not replace the entered driver; positions are
    kept as they are."""
    import types
    import pandas as pd
    from predictor.evaluation import _roster_at_cutoff
    t = pd.Timestamp("2021-07-09 12:00", tz="UTC")
    sessions = pd.DataFrame({"session_id": ["p-4", "e-1", "e-4"], "event_key": ["p", "e", "e"],
                             "session_type": ["Q", "FP1", "Q"],
                             "available_at_utc": [t - pd.Timedelta(days=7), t, t + pd.Timedelta(days=1)]})
    entrants = pd.DataFrame({
        "session_id": ["p-4"] * 3 + ["e-1"] * 3,
        "driver_number": ["1", "2", "3", "1", "2", "7"],
        "driver_id": ["a", "b", "c", "a", "b", "x"],          # x: a practice-only stand-in in car y
        "lineage_id": ["x", "x", "y", "x", "x", "y"]})
    archive = types.SimpleNamespace(
        sessions=sessions, entrants=entrants,
        events=pd.DataFrame({"event_key": ["p", "e"], "season": [2021, 2021], "round": [1, 2], "is_testing": False}))
    # 9 (s) replaced b in car x after the cutoff; c drives car y in qualifying although x ran practice
    target = pd.DataFrame({"session_id": "e-4", "driver_number": ["1", "3", "9"], "driver_id": ["a", "c", "s"],
                           "lineage_id": ["x", "y", "x"], "position": [2, 3, 1], "outcome": None, "excluded": False})
    roster = _roster_at_cutoff(archive, "e", t + pd.Timedelta(hours=1), target)
    assert list(roster.driver_id) == ["a", "b", "c"]          # b known at the cutoff, s not; c kept despite x
    assert list(roster.excluded) == [False, True, False]
    assert list(roster.position) == [2, 0, 3]                   # absolute positions kept
    before = _roster_at_cutoff(archive, "e", t - pd.Timedelta(hours=1), target)
    assert list(before.driver_id) == ["a", "b", "c"]          # before any session: s, not a regular of x, is a later substitute


def test_a_withdrawn_driver_is_not_a_winner_in_baseline_fitting():
    import numpy as np
    from predictor.baselines import RankBaseline
    from types import SimpleNamespace
    example = SimpleNamespace(positions=np.array([0, 1, 2]), excluded=np.array([True, False, False]))
    model = RankBaseline("uniform")
    model.ranker = "championship"
    model.ranks = lambda view, e: np.array([3.0, 1.0, 2.0])
    model.retirements = SimpleNamespace(fit=lambda history: None)
    model.fit([example], lambda e: None)
    assert model.beta > 0                                     # ranks agree with the classified drivers


def test_block_bootstrap_is_wider_for_correlated_differences_and_margin_is_reported():
    import numpy as np
    import pandas as pd
    from predictor import evaluation
    rng = np.random.default_rng(0)
    events = [f"2021-{i:02d}" for i in range(1, 41)]
    trend = np.repeat(rng.normal(0, 0.002, 10), 4)                  # differences correlated in runs of 4
    rows = [{"event_key": e, "season": 2021, "model": m, "rps": 0.1 + (d if m == "b" else 0.0)}
            for e, d in zip(events, trend - 0.0005) for m in ("a", "b")]
    result = evaluation.compare(pd.DataFrame(rows), "a").iloc[0]
    assert result.block_90_high - result.block_90_low > result.interval_90_high - result.interval_90_low
    assert result.within_margin == (result.block_90_high < evaluation.DEGRADATION_MARGIN)


def test_a_repeated_position_is_repaired_without_misaligned_rows():
    """Rows out of position order with a repeated position: the repair re-ranks by the
    published position, then the best time, aligned row by row."""
    import pandas as pd
    from predictor.targets import classify
    results = pd.DataFrame({"DriverNumber": ["1", "2", "3", "4"], "Position": [3.0, 1.0, 2.0, 2.0],
                            "Status": "Finished", "ClassifiedPosition": ["3", "1", "2", "2"],
                            "Q1": pd.to_timedelta([92.0, 90.0, 91.2, 91.0], unit="s")})
    frame = classify(results, 2024, "Q").set_index("driver_number")
    assert frame.position.to_dict() == {"1": 4, "2": 1, "3": 3, "4": 2}


def test_the_champion_router_serves_each_cutoff_with_its_champion():
    import types
    from predictor.evaluation import ChampionRouter

    class Stub:
        def __init__(self, name):
            self.name, self.fitted = name, 0

        def fit(self, history, view_for):
            self.fitted += 1

        def predict(self, view, example, **kwargs):
            return types.SimpleNamespace(model=self.name, metadata={})

    registry = {"a": lambda: Stub("a"), "b": lambda: Stub("b")}
    router = ChampionRouter({"pre_weekend": "a", "after_session": "b"}, registry)
    router.fit([], None)
    assert all(m.fitted == 1 for m in router.models.values())
    for label, served in (("pre_weekend", "a"), ("after_FP1", "b"), ("last", "b")):
        forecast = router.predict(None, types.SimpleNamespace(cutoff_label=label))
        assert forecast.metadata["served_by"] == served and forecast.model == "champion"
    same = ChampionRouter({"pre_weekend": "a", "after_session": "a"}, registry)
    assert same.models["pre_weekend"] is same.models["after_session"]
    registry["c"] = lambda: Stub("c")
    four = ChampionRouter({"pre_weekend": "a", "after_session": "b", "pre_weekend_wet": "c",
                           "after_session_wet": "c"}, registry)
    assert four.models["pre_weekend_wet"] is four.models["after_session_wet"]
    for label, condition, served in (("pre_weekend", "wet", "c"), ("last", "wet", "c"), ("last", "dry", "b"),
                                     ("pre_weekend", None, "a")):
        example = types.SimpleNamespace(cutoff_label=label, condition=condition)
        assert four.predict(None, example).metadata["served_by"] == served
    assert router.route(types.SimpleNamespace(cutoff_label="last", condition="wet"), None) == "after_session"


def test_a_practice_only_reserve_is_not_a_regular_driver():
    """A reserve who runs FP1 at consecutive weekends does not displace the entered
    driver: regulars come from the previous event's competitive sessions."""
    import types
    import pandas as pd
    from predictor.evaluation import _roster_at_cutoff
    t = pd.Timestamp("2022-10-21 12:00", tz="UTC")
    sessions = pd.DataFrame({"session_id": ["p-1", "p-4", "e-1", "e-4"], "event_key": ["p", "p", "e", "e"],
                             "session_type": ["FP1", "Q", "FP1", "Q"],
                             "available_at_utc": [t - pd.Timedelta(days=8), t - pd.Timedelta(days=7), t,
                                                  t + pd.Timedelta(days=1)]})
    entrants = pd.DataFrame({
        "session_id": ["p-1", "p-4", "e-1"],
        "driver_number": ["2", "6", "2"], "driver_id": ["reserve", "regular", "reserve"],
        "lineage_id": ["w", "w", "w"]})
    archive = types.SimpleNamespace(sessions=sessions, entrants=entrants,
                                    events=pd.DataFrame({"event_key": ["p", "e"], "season": [2022, 2022],
                                                         "round": [19, 20], "is_testing": False}))
    target = pd.DataFrame({"session_id": "e-4", "driver_number": ["6"], "driver_id": ["regular"],
                           "lineage_id": ["w"], "position": [1], "outcome": None, "excluded": False})
    roster = _roster_at_cutoff(archive, "e", t + pd.Timedelta(hours=1), target)
    assert list(roster.driver_id) == ["regular"] and not roster.excluded.any()


def test_too_few_events_give_no_block_verdict():
    import numpy as np
    import pandas as pd
    from predictor import evaluation
    rows = [{"event_key": f"2021-{i:02d}", "season": 2021, "model": m, "rps": 0.1 + (d if m == "b" else 0.0)}
            for i, d in enumerate([-0.004, -0.001, 0.002, -0.003], start=1) for m in ("a", "b")]
    result = evaluation.compare(pd.DataFrame(rows), "a").iloc[0]
    assert np.isnan(result.block_90_low) and not result.better_beyond_noise and not result.within_margin


def test_the_entry_list_published_before_the_cutoff_is_the_roster():
    import types
    import numpy as np
    import pandas as pd
    from predictor.evaluation import _roster_at_cutoff
    t = pd.Timestamp("2022-10-28 12:00", tz="UTC")
    sessions = pd.DataFrame({"session_id": ["p-4", "e-4"], "event_key": ["p", "e"], "session_type": ["Q", "Q"],
                             "available_at_utc": [t - pd.Timedelta(days=7), t + pd.Timedelta(days=1)]})
    entrants = pd.DataFrame({"session_id": ["p-4"] * 3, "driver_number": ["1", "2", "3"],
                             "driver_id": ["a", "b", "c"], "lineage_id": ["x", "x", "y"]})
    lists = pd.DataFrame({"event_key": "e", "driver_id": ["a", "b", "c", "r"] + ["z"] * 10,
                          "role": ["race"] * 3 + ["practice"] + ["race"] * 10,
                          "published_utc": t - pd.Timedelta(hours=2), "doc_number": 10})
    lists.loc[4:, "event_key"] = "other"
    archive = types.SimpleNamespace(sessions=sessions, entrants=entrants, entry_lists=lists,
                                    events=pd.DataFrame({"event_key": ["p", "e"], "season": 2022, "round": [1, 2],
                                                         "is_testing": False}))
    target = pd.DataFrame({"session_id": "e-4", "driver_number": ["1", "3", "9"], "driver_id": ["a", "c", "s"],
                           "lineage_id": ["x", "y", "x"], "position": [2, 3, 1], "outcome": None, "excluded": False})
    # the list for "e" has 3 race drivers: below 10, so it is not trusted and the proxy is used
    proxy = _roster_at_cutoff(archive, "e", t, target)
    assert list(proxy.driver_id) == ["a", "b", "c"]            # no trusted list: regulars before the weekend
    lists["event_key"] = "e"
    lists["driver_id"] = ["a", "b", "c", "r"] + [f"d{i}" for i in range(10)]
    roster = _roster_at_cutoff(archive, "e", t, target)
    assert "s" not in set(roster.driver_id)                    # a late substitute is not forecast
    assert "r" not in set(roster.driver_id)                    # a practice-only driver is not entered
    assert roster.set_index("driver_id").loc["b", "excluded"]  # entered, then withdrew: forecast, not scored
    assert roster.set_index("driver_id").loc["a", "position"] == 2
    earlier = _roster_at_cutoff(archive, "e", t - pd.Timedelta(hours=3), target)
    assert list(earlier.driver_id) == ["a", "b", "c"]          # before the list was published


def test_a_listed_drivers_seat_is_known_at_the_cutoff():
    """No later session decides a withdrawn driver's seat; a driver with no archived
    seat yet keeps his place on the roster with the number printed on the list."""
    import types
    import pandas as pd
    from predictor.evaluation import _latest_seat
    t = pd.Timestamp("2022-10-28 12:00", tz="UTC")
    sessions = pd.DataFrame({"session_id": ["old", "later"], "available_at_utc": [t - pd.Timedelta(days=7),
                                                                               t + pd.Timedelta(days=2)]})
    entrants = pd.DataFrame({"session_id": ["old", "later"], "driver_number": ["2", "2"], "driver_id": ["b", "b"],
                             "lineage_id": ["x", "future_team"]})
    archive = types.SimpleNamespace(sessions=sessions, entrants=entrants)
    assert _latest_seat(archive, "b", t) == ("2", "x")                      # not the later team
    assert _latest_seat(archive, "rookie", t, "631") == ("63", None)         # footnote marker removed
    assert _latest_seat(archive, "rookie", t, "44") == ("44", None)
    assert _latest_seat(archive, "rookie", t) is None


def test_before_the_weekend_without_a_list_a_later_substitute_is_not_forecast():
    import types
    import pandas as pd
    from predictor.evaluation import _roster_at_cutoff
    t = pd.Timestamp("2024-03-07 13:00", tz="UTC")
    sessions = pd.DataFrame({"session_id": ["p-4", "e-1"], "event_key": ["p", "e"], "session_type": ["Q", "FP1"],
                             "available_at_utc": [t - pd.Timedelta(days=7), t + pd.Timedelta(days=1)]})
    entrants = pd.DataFrame({"session_id": ["p-4", "p-4"], "driver_number": ["55", "16"],
                             "driver_id": ["sainz", "leclerc"], "lineage_id": ["maranello", "maranello"]})
    archive = types.SimpleNamespace(sessions=sessions, entrants=entrants, entry_lists=None,
                                    events=pd.DataFrame({"event_key": ["p", "e"], "season": 2024, "round": [1, 2],
                                                         "is_testing": False}))
    target = pd.DataFrame({"session_id": "e-4", "driver_number": ["38", "16"], "driver_id": ["bearman", "leclerc"],
                           "lineage_id": ["maranello", "maranello"], "position": [7, 2], "outcome": None,
                           "excluded": False})
    roster = _roster_at_cutoff(archive, "e", t, target).set_index("driver_id")
    assert set(roster.index) == {"sainz", "leclerc"} and roster.loc["sainz", "excluded"]


def test_too_few_events_say_no_verdict():
    import pandas as pd
    from predictor import evaluation
    rows = [{"event_key": f"2024-{i:02d}", "season": 2024, "model": m, "rps": 0.1 + (0.01 if m == "b" else 0.0)}
            for i in range(1, 6) for m in ("a", "b")]
    assert evaluation.compare(pd.DataFrame(rows), "a").iloc[0].verdict == "none"
