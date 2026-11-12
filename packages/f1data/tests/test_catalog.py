import pandas as pd
import pytest

from f1data.catalog import select_schedule_sessions, ingestion_batch
from f1data.sessions import discover


def schedule():
    return pd.DataFrame([
        {"RoundNumber": 1, "EventName": "First GP", "Location": "Track A", "Country": "A",
         "EventFormat": "conventional", "F1ApiSupport": True,
         "Session1": "Practice 1", "Session1DateUtc": "2023-03-01T10:00Z",
         "Session5": "Race", "Session5DateUtc": "2023-03-03T10:00Z"},
        {"RoundNumber": 2, "EventName": "Sprint GP", "Location": "Track B", "Country": "B",
         "EventFormat": "sprint_shootout", "F1ApiSupport": True,
         "Session1": "Practice 1", "Session1DateUtc": "2023-04-01T10:00Z",
         "Session3": "Sprint Shootout", "Session3DateUtc": "2023-04-02T10:00Z",
         "Session4": "Sprint", "Session4DateUtc": "2023-04-02T14:00Z",
         "Session5": "Race", "Session5DateUtc": "2023-04-03T10:00Z"},
    ])


def test_whole_season_discovers_dynamic_formats_and_preserves_provider_context():
    items, excluded = select_schedule_sessions(schedule(), 2023, now="2024-01-01T00:00Z")
    assert len(items) == 6 and not excluded
    assert {i["session_name"] for i in items} == {"Practice 1", "Sprint Shootout", "Sprint", "Race"}
    race = next(i for i in items if i["session_id"] == "2023-02-5")
    assert race["location"] == "Track B" and race["event_format"] == "sprint_shootout"


def test_future_unsupported_and_unknown_dates_are_recorded_separately():
    frame = schedule()
    frame.loc[0, "F1ApiSupport"] = False
    frame.loc[1, "Session3DateUtc"] = None
    frame.loc[1, "Session4DateUtc"] = "2025-01-01T00:00Z"
    items, excluded = select_schedule_sessions(frame, 2023, now="2024-01-01T00:00Z")
    assert {i["session_id"] for i in items} == {"2023-02-1", "2023-02-5"}
    assert {i["reason"] for i in excluded} == {"api_unsupported", "date_unknown", "not_yet_finished"}


def test_live_or_recent_sessions_are_not_assumed_finished():
    frame = schedule().iloc[:1].copy()
    frame["Session5DateUtc"] = "2023-03-03T10:00Z"
    items, excluded = select_schedule_sessions(frame, 2023, now="2023-03-03T11:00Z")
    assert not any(i["session_name"] == "Race" for i in items)
    assert excluded[-1]["reason"] == "not_yet_finished"


def test_pre_2018_or_invalid_year_is_rejected():
    for year in (1950, 2017, -1, 2023.5, True):
        with pytest.raises(ValueError):
            select_schedule_sessions(schedule(), year, now="2024-01-01T00:00Z")


def test_explicit_weekend_selection_still_validates_event_identity():
    with pytest.raises(ValueError, match="identity"):
        select_schedule_sessions(schedule(), 2023, weekends=[{"round": 1, "event_name": "Wrong GP", "purpose": "test"}], now="2024-01-01T00:00Z")


def test_batch_limit_counts_attempts_not_skips(tmp_path, monkeypatch):
    items = [{"session_id": str(i), "year": 2023} for i in range(5)]
    calls = []
    monkeypatch.setattr("f1data.catalog.present", lambda path, telemetry: path.name in {"0", "1"})
    def ingest(item, root, telemetry, attempts):
        calls.append(item["session_id"])
        return True
    result = ingestion_batch(items, tmp_path, False, 3, max_sessions=1, ingest_session=ingest)
    assert calls == ["0", "1", "2"]
    assert result["attempted"] == 1 and result["skipped"] == 2
    assert result["deferred"] == ["3", "4"]


def test_full_season_discovery_merges_prior_inventory_and_deduplicates_years(tmp_path, monkeypatch):
    from f1data.sessions import merge_inventory
    import json
    merge_inventory(tmp_path, [{"year": 2022, "round": 1, "session_id": "2022-01-5"}])
    called = []
    def fetch(year, **kwargs):
        called.append(year)
        return schedule()
    monkeypatch.setattr("fastf1.get_event_schedule", fetch)
    items = discover({"years": [2023, 2023]}, tmp_path)
    assert called == [2023] and len(items) == 6
    assert len(json.loads((tmp_path/"inventory.json").read_text())) == 7
    coverage = json.loads((tmp_path/"catalog/2023.json").read_text())
    assert coverage["discovered_sessions"] == 6


def test_auth_stop_preserves_completed_later_sessions_without_listing_them_as_pending(tmp_path, monkeypatch):
    from f1data.sessions import ProviderAccessError
    items = [{"session_id": str(i), "year": 2023} for i in range(4)]
    monkeypatch.setattr("f1data.catalog.present", lambda path, telemetry: path.name == "2")
    def blocked(*args):
        raise ProviderAccessError("HTTP 401")
    result = ingestion_batch(items, tmp_path, False, 3, ingest_session=blocked)
    assert result["failed"] == ["0"]
    assert result["deferred"] == ["1", "3"]
    assert result["stopped_reason"] == "HTTP 401"


def test_bounded_batch_keeps_looking_for_completed_sessions_after_the_limit(tmp_path, monkeypatch):
    items = [{"session_id": str(i), "year": 2023} for i in range(4)]
    monkeypatch.setattr("f1data.catalog.present", lambda path, telemetry: path.name == "3")
    called = []
    def ingest(item, *args):
        called.append(item["session_id"])
        return True
    result = ingestion_batch(items, tmp_path, False, 3, max_sessions=1, ingest_session=ingest)
    assert called == ["0", "3"] and result["deferred"] == ["1", "2"]


def with_testing():
    tests = pd.DataFrame([
        {"RoundNumber": 0, "EventName": "Pre-Season Testing", "Location": "Track T", "Country": "T",
         "EventFormat": "testing", "F1ApiSupport": True,
         "Session1": "Practice 1", "Session1DateUtc": "2023-02-20T08:00Z",
         "Session2": "Practice 2", "Session2DateUtc": "2023-02-21T08:00Z"},
        {"RoundNumber": 0, "EventName": "Pre-Season Testing 2", "Location": "Track T", "Country": "T",
         "EventFormat": "testing", "F1ApiSupport": True,
         "Session1": "Practice 1", "Session1DateUtc": "2023-02-25T08:00Z"},
    ])
    return pd.concat([tests, schedule()], ignore_index=True)


def test_testing_is_opt_in_and_numbered_in_season_order():
    items, _ = select_schedule_sessions(with_testing(), 2023, now="2024-01-01T00:00Z")
    assert not any(i.get("is_testing") for i in items)
    items, _ = select_schedule_sessions(with_testing(), 2023, now="2024-01-01T00:00Z", testing=True)
    testing = [i for i in items if i.get("is_testing")]
    assert [i["session_id"] for i in testing] == ["2023-T1-1", "2023-T1-2", "2023-T2-1"]
    assert {i["test_number"] for i in testing} == {1, 2} and {i["round"] for i in testing} == {0}
    assert len(items) == 9


def test_competitive_telemetry_scope_skips_practice_streams(tmp_path, monkeypatch):
    from f1data.catalog import COMPETITIVE_SESSIONS
    names = ["Practice 1", "Qualifying", "Sprint Shootout", "Sprint Qualifying", "Sprint", "Race"]
    items = [{"session_id": str(i), "year": 2023, "session_name": name} for i, name in enumerate(names)]
    monkeypatch.setattr("f1data.catalog.present", lambda path, telemetry: False)
    requested = {}
    def ingest(item, root, telemetry, attempts):
        requested[item["session_name"]] = telemetry
        return True
    ingestion_batch(items, tmp_path, True, 3, ingest_session=ingest, telemetry_scope=COMPETITIVE_SESSIONS)
    assert requested == {"Practice 1": False, "Qualifying": True, "Sprint Shootout": True,
                         "Sprint Qualifying": True, "Sprint": True, "Race": True}


def test_rate_limit_wait_retries_the_same_session_then_continues(tmp_path, monkeypatch):
    from f1data.sessions import RateLimitReached
    items = [{"session_id": str(i), "year": 2023} for i in range(3)]
    monkeypatch.setattr("f1data.catalog.present", lambda path, telemetry: False)
    calls, slept = [], []
    def ingest(item, *args):
        calls.append(item["session_id"])
        if calls.count("1") == 1 and item["session_id"] == "1":
            raise RateLimitReached("500 calls/h")
        return True
    result = ingestion_batch(items, tmp_path, False, 3, ingest_session=ingest, rate_limit_wait=1800, sleep=slept.append)
    assert calls == ["0", "1", "1", "2"] and slept == [1800]
    assert result["succeeded"] == ["0", "1", "2"] and result["stopped_reason"] is None


def test_rate_limit_without_wait_still_stops_with_resume_message(tmp_path, monkeypatch):
    from f1data.sessions import RateLimitReached
    items = [{"session_id": str(i), "year": 2023} for i in range(2)]
    monkeypatch.setattr("f1data.catalog.present", lambda path, telemetry: False)
    def limited(*args):
        raise RateLimitReached("500 calls/h")
    result = ingestion_batch(items, tmp_path, False, 3, ingest_session=limited)
    assert result["failed"] == ["0"] and result["deferred"] == ["1"]
    assert result["stopped_reason"] == "500 calls/h"


def test_rate_limit_wait_cap_is_per_session_and_eventually_stops(tmp_path, monkeypatch):
    from f1data.sessions import RateLimitReached
    items = [{"session_id": str(i), "year": 2023} for i in range(3)]
    monkeypatch.setattr("f1data.catalog.present", lambda path, telemetry: False)
    seen = {}
    def ingest(item, *args):
        seen[item["session_id"]] = seen.get(item["session_id"], 0) + 1
        if item["session_id"] == "2" or seen[item["session_id"]] <= 2:
            raise RateLimitReached("500 calls/h")
        return True
    slept = []
    result = ingestion_batch(items, tmp_path, False, 3, ingest_session=ingest, rate_limit_wait=60,
                             max_rate_limit_waits=2, sleep=slept.append)
    assert result["succeeded"] == ["0", "1"] and result["failed"] == ["2"]
    assert len(slept) == 6 and result["stopped_reason"] == "500 calls/h"


def test_computed_wait_overrides_fixed_wait_and_state_is_saved_each_attempt(tmp_path, monkeypatch):
    from f1data.sessions import RateLimitReached
    items = [{"session_id": "0", "year": 2023}]
    monkeypatch.setattr("f1data.catalog.present", lambda path, telemetry: False)
    attempts, slept, saves = [], [], []
    def ingest(item, *args):
        attempts.append(1)
        if len(attempts) == 1:
            raise RateLimitReached("500 calls/h")
        return True
    ingestion_batch(items, tmp_path, False, 3, ingest_session=ingest, rate_limit_wait=1800,
                    sleep=slept.append, compute_wait=lambda: 2500.4, after_session=lambda: saves.append(1))
    assert slept == [2500] and len(saves) == 2


def write_manifest(root, session_id, **manifest):
    import json
    directory = root / "sessions" / "2018" / session_id
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"session_id": session_id, **manifest}))


def test_known_gaps_are_skipped_and_listed_but_transient_failures_are_retried(tmp_path, monkeypatch):
    write_manifest(tmp_path, "gap", status="failed", error={"type": "ValueError"})
    write_manifest(tmp_path, "notel", status="partial", telemetry=True, missing_tables=["pos_data/driver-1"])
    write_manifest(tmp_path, "limited", status="failed", error={"type": "RateLimitExceededError"})
    write_manifest(tmp_path, "offline", status="failed", error={"type": "ConnectionError"})
    write_manifest(tmp_path, "interrupted", status="running")
    write_manifest(tmp_path, "coreonly", status="complete", telemetry=False)
    ids = ["gap", "notel", "limited", "offline", "interrupted", "coreonly", "new"]
    items = [{"session_id": i, "year": 2018} for i in ids]
    monkeypatch.setattr("f1data.catalog.present", lambda path, telemetry: False)
    called = []
    def ingest(item, *args):
        called.append(item["session_id"])
        return True
    result = ingestion_batch(items, tmp_path, True, 3, ingest_session=ingest)
    assert called == ["limited", "offline", "interrupted", "coreonly", "new"]
    assert set(result["unresolved_skipped"]) == {"gap", "notel"}
    assert result["unresolved_skipped"]["gap"] == "failed: ValueError"
    called.clear()
    result = ingestion_batch(items, tmp_path, True, 3, ingest_session=ingest, retry_failed=True)
    assert called == ids and result["unresolved_skipped"] == {}


def test_partial_core_without_telemetry_request_counts_as_unresolved(tmp_path, monkeypatch):
    write_manifest(tmp_path, "s", status="partial", telemetry=False, missing_tables=["race_control"])
    monkeypatch.setattr("f1data.catalog.present", lambda path, telemetry: False)
    result = ingestion_batch([{"session_id": "s", "year": 2018}], tmp_path, False, 3,
                             ingest_session=lambda *a: pytest.fail("known gap retried"))
    assert "s" in result["unresolved_skipped"]
