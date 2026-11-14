import numpy as np
import pandas as pd
import pytest

from predictor import laps as L

PACE = {"1": 0.0, "2": 0.4, "3": 0.9, "4": 1.3}           # true pace, log time x 100
OFFSET = {"SOFT": 0.0, "MEDIUM": 0.5, "HARD": 0.9}
WEAR = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.03}
FUEL = L.FUEL
STRATEGY = {"1": [("SOFT", 15), ("HARD", 40)], "2": [("MEDIUM", 25), ("HARD", 30)],   # split strategies
            "3": [("HARD", 30), ("MEDIUM", 25)], "4": [("SOFT", 20), ("MEDIUM", 35)]}


def synthetic_race(noise=0.05, seed=0, extra=()):
    rng = np.random.default_rng(seed)
    rows = []
    for driver, stints in STRATEGY.items():
        lap = 1
        for compound, length in stints:
            for age in range(1, length + 1):
                log_time = 100 * np.log(90.0) + PACE[driver] + OFFSET[compound] + WEAR[compound] * age + FUEL * lap \
                    + rng.normal(0, noise)
                rows.append({"DriverNumber": driver, "LapNumber": float(lap), "Compound": compound, "TyreLife": float(age),
                             "LapTime": pd.to_timedelta(np.exp(log_time / 100), unit="s"),
                             "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TrackStatus": "1",
                             "IsAccurate": True, "Deleted": False})
                lap += 1
    return pd.DataFrame(rows + list(extra))


def test_race_pace_recovers_driver_pace_through_tyres_wear_and_fuel():
    pace = L.race_pace(synthetic_race()).set_index("driver_number").pace
    relative = pace - pace["1"]
    for driver, truth in PACE.items():
        assert relative[driver] == pytest.approx(truth, abs=0.03)
    corrections = L.race_pace(synthetic_race()).attrs["corrections"]
    for compound in ("SOFT", "MEDIUM", "HARD"):
        assert corrections[f"wear:{compound}"] == pytest.approx(WEAR[compound], abs=0.01)
    assert corrections["compound:MEDIUM"] == pytest.approx(OFFSET["MEDIUM"] - OFFSET["HARD"], abs=0.05)  # vs HARD


def test_unusable_laps_are_left_out_and_outliers_trimmed():
    traffic = {"DriverNumber": "2", "LapNumber": 30.0, "Compound": "HARD", "TyreLife": 5.0,
               "LapTime": pd.to_timedelta(93.0, unit="s"), "PitInTime": pd.NaT, "PitOutTime": pd.NaT,
               "TrackStatus": "1", "IsAccurate": True, "Deleted": False}
    safety_car = {**traffic, "LapTime": pd.to_timedelta(120.0, unit="s"), "TrackStatus": "4"}
    wet = {**traffic, "Compound": "INTERMEDIATE"}
    race = synthetic_race(extra=[traffic, safety_car, wet])
    clean = L.clean_race_laps(race)
    assert len(clean) == len(race) - 2 - len(STRATEGY)      # safety-car, wet and lap-1 laps gone; traffic kept, then trimmed
    pace = L.race_pace(race)
    assert pace.attrs["laps_used"] == len(clean) - 1
    assert (pace.set_index("driver_number").pace - pace.set_index("driver_number").pace["1"])["2"] \
        == pytest.approx(PACE["2"], abs=0.03)


def test_a_wet_or_drying_race_gives_no_pace():
    race = synthetic_race()
    assert L.race_pace(race.assign(Compound="INTERMEDIATE")).empty
    drying = race.copy()
    drying.loc[drying.LapNumber <= 8, "Compound"] = "INTERMEDIATE"      # wet start, slicks after
    assert L.wet_race(drying) and L.race_pace(drying).empty
    assert not L.wet_race(race)


def test_a_driver_with_few_laps_is_left_out_and_se_is_relative():
    race = synthetic_race()
    early_retirement = race[(race.DriverNumber != "4") | (race.LapNumber <= 10)]
    pace = L.race_pace(early_retirement)
    assert set(pace.driver_number) == {"1", "2", "3"}
    assert (pace.se < 0.05).all()                     # relative to the field mean: small with 50+ clean laps each


def test_2018_compound_names_count_as_dry():
    race = synthetic_race().replace({"Compound": {"SOFT": "SUPERSOFT", "HARD": "MEDIUM"}})
    assert not L.race_pace(race).empty


def test_a_race_that_cannot_tell_drivers_apart_gives_no_pace():
    race = synthetic_race()
    race.loc[race.DriverNumber == "4", "Compound"] = "HARD"
    race.loc[race.DriverNumber != "4", "Compound"] = race.loc[race.DriverNumber != "4", "Compound"].replace(
        {"HARD": "MEDIUM"})
    # driver 4 alone on hards: its pace and the hard offset are the same column
    assert L.race_pace(race).empty
