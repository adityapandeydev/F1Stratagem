import types

import numpy as np
import pandas as pd

from predictor.retirements import DetailedRetirements, classify_status


def test_status_text_is_read_as_a_cause():
    assert classify_status("Finished") == "finished" and classify_status("+1 Lap") == "finished"
    assert classify_status("Collision") == "incident" and classify_status("Spun off") == "incident"
    assert classify_status("Engine") == "mechanical" and classify_status("Gearbox") == "mechanical"
    assert classify_status("Did not start") is None and classify_status("Retired") is None


def _race(season, event, outcomes, lineages):
    n = len(outcomes)
    return types.SimpleNamespace(season=season, event_key=event, session_type="R", outcomes=outcomes,
                                 excluded=np.zeros(n, bool), lineage_ids=lineages, driver_numbers=[str(i) for i in range(n)])


def test_an_unreliable_team_and_a_crash_prone_circuit_raise_their_chances():
    lineages = ["a", "a", "b", "b"]
    history = []
    for i in range(12):
        event = f"e{i}"
        outcomes = ["mechanical" if i % 2 == 0 else "other_points", "other_points",      # team a breaks down often
                    "incident_or_other" if event in ("e0", "e4", "e8") else "other_points", "outside_points"]
        history.append(_race(2020, event, outcomes, lineages))
    circuits = {f"e{i}": ("monaco" if i % 4 == 0 else f"c{i}") for i in range(12)}
    older = pd.DataFrame({"season": [2010, 2010, 2011, 2011], "circuit": ["monaco", "c1", "monaco", "c1"],
                          "starts": [20, 20, 20, 20], "incidents": [6, 1, 5, 1]})
    model = DetailedRetirements(circuits, older).fit(history)
    incident, mechanical = model.chances(_race(2020, "e0", [None] * 4, lineages))
    assert mechanical[0] > mechanical[2]                 # team a less reliable than b
    assert model.circuit_factor["monaco"] > 1 > model.circuit_factor["c1"]
    elsewhere, _ = model.chances(_race(2020, "e1", [None] * 4, lineages))
    assert incident[0] > elsewhere[0]
    # a new season keeps part of last season's deviation
    _, next_season = model.chances(_race(2021, "e0", [None] * 4, lineages))
    assert mechanical[0] > next_season[0] > next_season[2]


def test_applied_retirements_go_behind_the_finishers():
    model = DetailedRetirements()
    model.incident_rate, model.mechanical_rate = 0.5, 0.0
    example = _race(2020, "e", [None] * 4, ["a", "b", "c", "d"])
    final, outcomes = model.apply(np.tile([[1, 2, 3, 4]], (200, 1)), example, np.random.default_rng(0))
    assert all(sorted(row) == [1, 2, 3, 4] for row in final)
    retired = outcomes == 4                              # incident_or_other
    assert 0.35 < retired.mean() < 0.65
    assert all((final[d][retired[d]] > final[d][~retired[d]].max(initial=0)).all() for d in range(200))


def test_refitting_on_history_without_circuit_differences_clears_old_factors():
    circuits = {"e0": "monaco", "e1": "c1"}
    skewed = [_race(2020, "e0", ["incident_or_other"] * 4, list("abcd")),
              _race(2020, "e1", ["other_points"] * 4, list("abcd"))]
    model = DetailedRetirements(circuits).fit(skewed)
    assert model.circuit_factor
    balanced = [_race(2020, e, ["incident_or_other", "other_points", "other_points", "other_points"], list("abcd"))
                for e in ("e0", "e1")]
    assert model.fit(balanced).circuit_factor == DetailedRetirements(circuits).fit(balanced).circuit_factor == {}


def test_a_race_model_with_detailed_retirements_forecasts_without_history():
    from predictor.race import RaceModel
    model = RaceModel.__new__(RaceModel)
    RaceModel.__init__(model, None, "race:test", detailed_retirements=True)
    incident, mechanical = model.retirements.chances(_race(2020, "e", [None] * 3, list("abc")))
    assert np.allclose(incident, 0.0) and np.allclose(mechanical, 0.0)
