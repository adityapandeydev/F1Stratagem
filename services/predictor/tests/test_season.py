import numpy as np
import pandas as pd

from predictor.championship import standings
from predictor.season import SeasonSimulator, teams_at

END = pd.Timestamp("2030-01-01", tz="UTC")


def _entries(rows):
    return pd.DataFrame(rows, columns=["season", "round", "session_type", "driver", "team", "position", "fastest"])


def _sim():
    return SeasonSimulator(model=None, setting={}, fastest=np.zeros(31), draws=3)


def test_driver_missing_from_the_roster_keeps_points_and_title_chance():
    # 2020 before Sakhir: Hamilton leads but is not in the next race's roster
    carried = standings(_entries([
        (2020, 1, "R", "hamilton", "mercedes", 1, False), (2020, 1, "R", "bottas", "mercedes", 2, False),
        (2020, 1, "R", "russell", "williams", 3, False)]), END)
    roster = ["bottas", "russell"]
    add = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 30.0]])          # three simulated seasons
    gp = np.zeros((3, 2, 3))
    out = _sim()._tables(add, gp, roster, carried, {"bottas": "mercedes", "russell": "mercedes"})
    assert out["drivers"] == ["bottas", "russell", "hamilton"]
    i = out["drivers"].index("hamilton")
    assert (out["points"][:, i] == 25).all()
    assert out["rank"][:, i].tolist() == [1, 2, 2]     # Bottas 18 + 0 | 18 + 10 | 18, Russell 15 | 15 | 15 + 30
    assert out["points"][:, 1].tolist() == [15.0, 15.0, 45.0]


def test_known_seat_change_wins_over_the_last_result():
    # 2019 before Belgium: Albon has moved to Red Bull, Gasly to Toro Rosso
    teams = pd.DataFrame({"season": [2019, 2019], "lineage_id": ["milton_keynes", "faenza"],
                          "constructor_id": ["red_bull", "toro_rosso"]})
    seats = teams_at(["albon", "gasly", "kubica"], ["milton_keynes", "faenza", None], 2019, teams,
                     {"albon": "toro_rosso", "gasly": "red_bull", "kubica": "williams"})
    assert seats == {"albon": "red_bull", "gasly": "toro_rosso", "kubica": "williams"}
    carried = standings(_entries([
        (2019, 1, "R", "albon", "toro_rosso", 1, False), (2019, 1, "R", "gasly", "red_bull", 2, False)]), END)
    out = _sim()._tables(np.array([[18.0, 0.0]]), np.zeros((1, 2, 3)), ["albon", "gasly"], carried, seats)
    c = dict(zip(out["constructors"], out["constructor_points"][0]))
    assert c == {"red_bull": 18.0 + 18.0, "toro_rosso": 25.0}         # Albon's new points go to Red Bull
