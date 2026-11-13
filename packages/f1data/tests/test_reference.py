import pandas as pd

from f1data.reference import resolve_entrants


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


def test_lineage_maps_every_constructor_season_once(tmp_path):
    import json
    import pytest
    from f1data.reference import lineage_table
    config = tmp_path / "lineage.json"
    config.write_text(json.dumps({"lineages": {"faenza": {"members": [
        {"constructor_id": "toro_rosso", "from": 2018, "to": 2019}, {"constructor_id": "alphatauri", "from": 2020}]}}}))
    constructors = pd.DataFrame({"season": [2019, 2020, 2021], "constructor_id": ["toro_rosso", "alphatauri", "alphatauri"]})
    table = lineage_table(constructors, config)
    assert set(table[table.season == 2021].constructor_id) == {"alphatauri"}
    with pytest.raises(ValueError, match="without a lineage"):
        lineage_table(pd.concat([constructors, pd.DataFrame({"season": [2021], "constructor_id": ["cadillac"]})]),
                      config)


def test_qualifying_history_keeps_times_lineage_and_an_estimated_start(tmp_path):
    from f1data.reference import qualifying_history
    tables = tmp_path / "external" / "jolpica" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame({"season": ["2010", "2010", "2018"], "round": ["1", "1", "1"],
                  "circuit_id": ["bahrain", "bahrain", "albert_park"],
                  "Driver_driverId": ["vettel", "alonso", "hamilton"],
                  "Constructor_constructorId": ["red_bull", "ferrari", "mercedes"], "number": ["5", "8", "44"],
                  "position": ["1", "2", "1"], "Q1": ["1:55.029", "1:54.612", "1:22.824"],
                  "Q2": ["1:53.883", None, "1:22.051"], "Q3": ["1:54.101", None, "1:21.164"]}).to_parquet(
        tables / "qualifying.parquet")
    races = pd.DataFrame({"season": [2010, 2018], "round": [1, 1], "date": ["2010-03-14", "2018-03-25"],
                          "time": ["12:00:00Z", "05:10:00Z"], "Qualifying_date": [None, "2018-03-24"]})
    lineage = pd.DataFrame({"season": [2010, 2010], "constructor_id": ["red_bull", "ferrari"],
                            "lineage_id": ["milton_keynes", "maranello"]})
    history = qualifying_history(tmp_path, races, lineage).set_index("driver_id")
    assert list(history.index) == ["vettel", "alonso"]                 # 2018+ comes from the FastF1 archive
    assert history.loc["vettel", "q1_seconds"] == 115.029
    assert pd.isna(history.loc["alonso", "q2_seconds"])
    assert history.loc["vettel", "lineage_id"] == "milton_keynes"
    assert history.loc["vettel", "start_utc"] == pd.Timestamp("2010-03-13 12:00", tz="UTC")   # day before the race
