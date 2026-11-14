

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
    assert list(before.driver_id) == ["a", "c", "s"]          # before any session: the entry-list proxy


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
