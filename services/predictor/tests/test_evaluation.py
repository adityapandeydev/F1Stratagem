

def test_the_roster_is_known_at_the_cutoff():
    """After a session of the event, its entrants are the roster: a withdrawn driver is
    forecast but not scored, a late substitute is not forecast, positions re-ranked."""
    import types
    import numpy as np
    import pandas as pd
    from predictor.evaluation import _roster_at_cutoff
    t = pd.Timestamp("2021-07-02 12:00", tz="UTC")
    archive = types.SimpleNamespace(
        sessions=pd.DataFrame({"session_id": ["e-1", "e-4"], "event_key": ["e", "e"],
                               "available_at_utc": [t, t + pd.Timedelta(days=1)]}),
        entrants=pd.DataFrame({"session_id": ["e-1"] * 3, "driver_number": ["1", "2", "3"],
                               "driver_id": ["a", "b", "c"], "team_id": ["x", "x", "y"], "lineage_id": ["x", "x", "y"]}))
    target = pd.DataFrame({"session_id": "e-4", "driver_number": ["1", "3", "9"], "driver_id": ["a", "c", "s"],
                           "team_id": ["x", "y", "x"], "lineage_id": ["x", "y", "x"], "position": [2, 3, 1],
                           "outcome": None, "excluded": False})
    roster = _roster_at_cutoff(archive, "e", t + pd.Timedelta(hours=1), target)
    assert list(roster.driver_number) == ["1", "2", "3"]               # 2 withdrew, 9 (a substitute) unknown
    assert list(roster.excluded) == [False, True, False]
    assert list(roster.position[~roster.excluded]) == [1, 2]            # re-ranked among those forecast
    before = _roster_at_cutoff(archive, "e", t - pd.Timedelta(hours=1), target)
    assert list(before.driver_number) == ["1", "3", "9"]               # before the weekend: the entry-list proxy


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
