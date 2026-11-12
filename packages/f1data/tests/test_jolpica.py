import datetime

import pandas as pd
import pytest

from f1data import jolpica


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self.payload, self.status_code, self.headers = payload, status, headers or {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def mr(total, offset, **table):
    return {"MRData": {"total": str(total), "limit": "100", "offset": str(offset), **table}}


def race(rnd, rows_key=None, rows=()):
    data = {"season": "2023", "round": str(rnd), "raceName": f"GP {rnd}", "date": "2023-03-05",
            "Circuit": {"circuitId": f"c{rnd}", "Location": {"lat": "1.0", "long": "2.0"}}}
    if rows_key:
        data[rows_key] = list(rows)
    return data


class FakeSession:
    """Serves 2 races; 150 results rows to force two pages."""

    def __init__(self):
        self.headers, self.calls = {}, []

    def get(self, url, params, timeout):
        path = url.split("/f1/")[1].removesuffix(".json")
        offset = params["offset"]
        self.calls.append((path, offset))
        if path == "2023/races":
            return FakeResponse(mr(2, offset, RaceTable={"Races": [race(1), race(2)]}))
        if path == "2023/results":
            rows = [{"number": str(i), "position": str(i % 20 + 1), "points": "0",
                     "Driver": {"driverId": f"d{i}"}, "Constructor": {"constructorId": "t"}} for i in range(150)]
            page = rows[offset:offset + 100]
            return FakeResponse(mr(150, offset, RaceTable={"Races": [race(1, "Results", page)]}))
        if path == "2023/driverstandings":
            return FakeResponse(mr(1, offset, StandingsTable={"StandingsLists": [
                {"season": "2023", "round": "2", "DriverStandings": [
                    {"position": "1", "points": "50", "wins": "2", "Driver": {"driverId": "d0"}}]}]}))
        if path.startswith("2023/") and path.endswith("/pitstops"):
            rnd = int(path.split("/")[1])
            return FakeResponse(mr(1, offset, RaceTable={"Races": [race(rnd, "PitStops", [
                {"driverId": "d0", "lap": "10", "stop": "1", "duration": "22.1"}])]}))
        return FakeResponse(mr(0, offset))


class Clock:
    def __init__(self):
        self.t, self.slept = 1_000_000.0, []

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


def client(tmp_path, session=None, per_hour=250):
    clock = Clock()
    limiter = jolpica.RateLimiter(tmp_path / "rate.json", per_hour=per_hour, clock=clock, sleep=clock.sleep)
    return jolpica.Client(limiter, session=session or FakeSession(), sleep=clock.sleep), clock


def test_season_download_paginates_saves_raw_and_builds_tables(tmp_path):
    c, _ = client(tmp_path)
    past = datetime.date(2026, 11, 10)
    manifest = jolpica.download_season(tmp_path, 2023, c, today=past)
    assert manifest["status"] == "complete" and manifest["final"]
    assert manifest["totals"]["results"] == 150 and manifest["totals"]["pitstops/02"] == 1
    assert (tmp_path / "raw/2023/results/offset-00000.json").is_file()
    assert (tmp_path / "raw/2023/results/offset-00100.json").is_file()
    assert (tmp_path / "raw/2023/pitstops/round-01/offset-00000.json").is_file()
    assert ("2023/results", 100) in c.session.calls
    assert c.session.headers["User-Agent"] == jolpica.USER_AGENT
    jolpica.build_tables(tmp_path)
    results = pd.read_parquet(tmp_path / "tables/results.parquet")
    assert len(results) == 150 and results.Driver_driverId.iloc[0] == "d0"
    assert set(pd.read_parquet(tmp_path / "tables/pitstops.parquet")["round"]) == {1, 2}
    standings = pd.read_parquet(tmp_path / "tables/driver_standings.parquet")
    assert standings.points.tolist() == ["50"]


def test_final_season_is_skipped_without_requests_and_damaged_pages_are_refetched(tmp_path):
    c, _ = client(tmp_path)
    today = datetime.date(2026, 11, 10)
    jolpica.download_season(tmp_path, 2023, c, today=today)
    c.session.calls.clear()
    jolpica.download_season(tmp_path, 2023, c, today=today)
    assert c.session.calls == []
    (tmp_path / "raw/2023/results/offset-00100.json").write_text("{}")
    jolpica.download_season(tmp_path, 2023, c, today=today)
    assert c.session.calls == [("2023/results", 100)]


def test_current_season_is_provisional_and_refetched(tmp_path):
    c, _ = client(tmp_path)
    during = datetime.date(2023, 7, 1)
    manifest = jolpica.download_season(tmp_path, 2023, c, today=during)
    assert not manifest["final"]
    calls = len(c.session.calls)
    jolpica.download_season(tmp_path, 2023, c, today=during)
    assert len(c.session.calls) == 2 * calls


def test_rate_limiter_spaces_requests_and_waits_out_the_hourly_budget(tmp_path):
    clock = Clock()
    limiter = jolpica.RateLimiter(tmp_path / "rate.json", per_hour=3, min_interval=1.0, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire()
    assert clock.slept == [1.0, 1.0]
    limiter.acquire()
    assert clock.slept[-1] == pytest.approx(3600 - 2 + 1)
    restored = jolpica.RateLimiter(tmp_path / "rate.json", per_hour=3, clock=clock, sleep=clock.sleep)
    assert restored.timestamps == [clock.t - 3599, clock.t]  # older calls have left the hour


def test_throttling_and_server_errors_are_retried(tmp_path):
    class Flaky(FakeSession):
        def __init__(self):
            super().__init__()
            self.failures = [FakeResponse({}, 429, {"Retry-After": "7"}), FakeResponse({}, 503)]

        def get(self, url, params, timeout):
            if self.failures:
                return self.failures.pop(0)
            return super().get(url, params, timeout)

    c, clock = client(tmp_path, session=Flaky())
    payload = c.get("2023/races", 0)
    assert payload["MRData"]["total"] == "2"
    assert 7 in clock.slept and 120 in clock.slept
