from datetime import date
import json

import pandas as pd
import pytest

from f1data import openmeteo


def write_races(root, rows):
    (root / "tables").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "tables" / "races.parquet")


RACE_2024 = {"season": "2024", "round": "6", "raceName": "Miami Grand Prix", "date": "2024-05-05",
             "Circuit_circuitId": "miami", "Circuit_Location_lat": "25.9581", "Circuit_Location_long": "-80.2389",
             "FirstPractice_date": "2024-05-03", "Sprint_date": "2024-05-04", "Qualifying_date": "2024-05-04"}
RACE_1955 = {"season": "1955", "round": "1", "raceName": "Argentine Grand Prix", "date": "1955-01-16",
             "Circuit_circuitId": "galvez", "Circuit_Location_lat": "-34.6943", "Circuit_Location_long": "-58.4593",
             "FirstPractice_date": None, "Sprint_date": None, "Qualifying_date": None}


class FakeClient:
    def __init__(self, refuse=()):
        self.calls, self.refuse = [], set(refuse)

    def get(self, url, params, weight):
        self.calls.append((url, params, weight))
        if params["start_date"] in self.refuse:
            raise openmeteo.RequestRefused("HTTP 400: out of range")
        variables = params["hourly"].split(",")
        hours = ["2024-05-02T00:00", "2024-05-02T01:00"]
        return {"latitude": 25.97, "longitude": -80.26, "elevation": 5.0,
                "hourly": {"time": hours, **{v: [1.0, None] for v in variables}}}


def test_call_weight_follows_documented_rule():
    assert openmeteo.call_weight(10, 14) == 1.0
    assert openmeteo.call_weight(15, 14) == 1.5
    assert openmeteo.call_weight(10, 28) == 2.0
    assert openmeteo.call_weight(64, 6) == pytest.approx(6.4)


def test_limiter_waits_until_weight_leaves_the_hour(tmp_path):
    now = [10_000.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = openmeteo.WeightedLimiter(tmp_path / "rate.json", per_hour=10, per_day=100, min_interval=0,
                                        clock=lambda: now[0], sleep=sleep)
    limiter.acquire(6)
    limiter.acquire(4)
    limiter.acquire(1)
    assert slept and slept[0] == pytest.approx(3601)
    restored = openmeteo.WeightedLimiter(tmp_path / "rate.json", clock=lambda: now[0])
    assert len(restored.calls) == 3


def test_weekend_window_covers_sessions_and_utc_spill(tmp_path):
    write_races(tmp_path, [RACE_2024, RACE_1955])
    old, miami = openmeteo.weekends(tmp_path)
    assert (old["start_date"], old["end_date"]) == ("1955-01-13", "1955-01-17")
    assert (miami["start_date"], miami["end_date"]) == ("2024-05-02", "2024-05-06")
    assert miami["latitude"] == pytest.approx(25.9581)


def test_download_skips_present_unsettled_and_known_failures(tmp_path):
    jolpica, root = tmp_path / "jolpica", tmp_path / "open-meteo"
    future = {**RACE_2024, "season": "2026", "round": "23", "date": "2026-12-06", "FirstPractice_date": None,
              "Sprint_date": None, "Qualifying_date": None}
    write_races(jolpica, [RACE_1955, RACE_2024, future])
    client = FakeClient(refuse={"1955-01-13"})
    today = date(2026, 11, 10)
    counts = openmeteo.download(root, client, ["observed", "forecast-runs"], today=today, jolpica_root=jolpica)
    assert counts == {"fetched": 2, "present": 0, "not_settled": 2, "failed": 1, "skipped_failed": 0}
    # forecast-runs starts in 2021, so 1955 has only an observed request
    assert sorted(json.loads((root / "failed.json").read_text())["entries"]) == ["observed/1955-01"]
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["entries"]["forecast-runs/2024-06"]
    assert entry["fill"]["precipitation_previous_day3"] == 1
    assert (root / entry["file"]).is_file()

    again = openmeteo.download(root, FakeClient(), ["observed", "forecast-runs"], today=today, jolpica_root=jolpica)
    assert again == {"fetched": 0, "present": 2, "not_settled": 2, "failed": 0, "skipped_failed": 1}
    retried = openmeteo.download(root, FakeClient(), ["observed"], today=today, retry_failed=True,
                                 jolpica_root=jolpica)
    assert retried["fetched"] == 1
    assert json.loads((root / "failed.json").read_text())["entries"] == {}


def test_tables_split_variable_and_lead(tmp_path):
    jolpica, root = tmp_path / "jolpica", tmp_path / "open-meteo"
    write_races(jolpica, [RACE_2024])
    openmeteo.download(root, FakeClient(), ["observed", "forecast-runs"], today=date(2026, 11, 10),
                       jolpica_root=jolpica)
    written = openmeteo.build_tables(root)
    assert written["observed_hourly"]["rows"] == 2
    forecasts = pd.read_parquet(root / "tables" / "forecast_runs_hourly.parquet")
    # nulls dropped: one value per variable and lead
    assert len(forecasts) == len(openmeteo.FORECAST_VARIABLES) * len(openmeteo.FORECAST_LEADS)
    rain = forecasts[forecasts.variable == "rain"]
    assert sorted(rain.lead_days) == list(range(8))
    assert str(forecasts.time.dt.tz) == "UTC"
