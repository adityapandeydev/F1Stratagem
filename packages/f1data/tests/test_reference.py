import pandas as pd

from f1data.reference import lineage_candidates, resolve_entrants


def entrants(rows):
    frame = pd.DataFrame(rows, columns=["session_id", "event_key", "season", "DriverNumber", "Abbreviation",
                                        "DriverId", "TeamId", "TeamName", "FullName"])
    frame["classified_in_results"] = True
    return frame.astype({"DriverId": "string", "TeamId": "string"})


def test_ids_fill_from_the_same_event_then_the_same_season():
    frame = resolve_entrants(entrants([
        ("2024-01-5", "2024-01", 2024, "1", "VER", "max_verstappen", "red_bull", "Red Bull Racing", "Max Verstappen"),
        ("2024-01-1", "2024-01", 2024, "1", "VER", None, None, "Red Bull Racing", "Max Verstappen"),
        ("2024-T1-1", "2024-T1", 2024, "1", "VER", None, None, None, None),
    ]))
    by_session = frame.set_index("session_id")
    assert by_session.loc["2024-01-1", "driver_source"] == "same_event"
    assert by_session.loc["2024-T1-1", "driver_id"] == "max_verstappen"
    assert by_session.loc["2024-T1-1", "driver_source"] == "same_season"
    assert by_session.loc["2024-T1-1", "team_id"] == "red_bull"          # team of the season's first event


def test_ambiguous_candidates_stay_empty():
    frame = resolve_entrants(entrants([
        ("2024-05-5", "2024-05", 2024, "40", "LAW", "lawson", "rb", "RB", "Liam Lawson"),
        ("2024-05-4", "2024-05", 2024, "40", "LAW", "someone_else", "rb", "RB", "Liam Lawson"),
        ("2024-05-1", "2024-05", 2024, "40", "LAW", None, None, "RB", "Liam Lawson"),
    ]))
    assert pd.isna(frame.set_index("session_id").loc["2024-05-1", "driver_id"])


def test_practice_only_driver_keeps_empty_id_but_gets_the_team():
    frame = resolve_entrants(entrants([
        ("2024-03-5", "2024-03", 2024, "44", "HAM", "hamilton", "mercedes", "Mercedes", "Lewis Hamilton"),
        ("2024-03-1", "2024-03", 2024, "12", "ANT", None, None, "Mercedes", "Andrea Kimi Antonelli"),
    ]))
    rookie = frame.set_index("session_id").loc["2024-03-1"]
    assert pd.isna(rookie.driver_id) and rookie.team_id == "mercedes"


def test_lineage_candidates_pair_ending_and_starting_constructors():
    teams = pd.DataFrame({"season": [2019, 2019, 2020, 2020], "constructor_id": ["toro_rosso", "ferrari",
                                                                                "alphatauri", "ferrari"]})
    assert lineage_candidates(teams) == [{"ends": "toro_rosso", "last_season": 2019,
                                          "starting_next_season": ["alphatauri"]}]
