import json
from types import SimpleNamespace

import pandas as pd
import pytest

from f1data.paths import default_data_root, repository_root
from f1data.sessions import (ProviderAccessError, check_access, export_session,
                             ingest, report, verified, validate, merge_inventory, read_telemetry)
from f1data.storage import write_table


def session():
    return SimpleNamespace(
        laps=pd.DataFrame({"DriverNumber": ["1", "2"], "LapNumber": [1., 1.],
                           "LapTime": pd.to_timedelta([90, None], unit="s")}),
        results=pd.DataFrame({"DriverNumber": ["1", "2"], "Position": [1., 2.]}),
        weather_data=pd.DataFrame({"Time": pd.to_timedelta([1], unit="s"), "Rainfall": [False]}),
        track_status=pd.DataFrame({"Time": pd.to_timedelta([1], unit="s"), "Status": ["1"]}),
        session_status=pd.DataFrame({"Time": pd.to_timedelta([1], unit="s"), "Status": ["Started"]}),
        race_control_messages=pd.DataFrame({"Time": pd.to_datetime(["2023-03-05T15:00:00"]), "Message": ["CLEAR"]}),
    )


def test_duration_and_null_roundtrip(tmp_path):
    frame = session().laps
    write_table(frame, tmp_path / "laps.parquet", {"session_id": "2023-01-5"})
    restored = pd.read_parquet(tmp_path / "laps.parquet")
    assert restored.LapTime.iloc[0] == pd.Timedelta(seconds=90)
    assert pd.isna(restored.LapTime.iloc[1])
    assert restored.session_id.tolist() == ["2023-01-5"] * 2


def test_rejects_duplicate_laps():
    frame = session().laps
    with pytest.raises(ValueError, match="duplicate"):
        validate("laps", pd.concat([frame, frame]))


def test_checksums_detect_corruption_and_missing_artifacts(tmp_path):
    manifest = export_session(session(), tmp_path, {"session_id": "2023-01-5"}, False)
    assert manifest["status"] == "complete"
    assert verified(tmp_path, False)
    (tmp_path / "normalized" / "laps.parquet").write_bytes(b"damaged")
    assert not verified(tmp_path, False)


def test_missing_optional_table_is_partial_and_not_skipped(tmp_path):
    data = session()
    data.weather_data = None
    manifest = export_session(data, tmp_path, {"session_id": "2023-01-5"}, False)
    assert manifest["status"] == "partial"
    assert "weather" in manifest["missing_tables"]
    assert not verified(tmp_path, False)


def test_missing_laps_never_marks_complete(tmp_path):
    data = session()
    data.laps = pd.DataFrame()
    with pytest.raises(ValueError, match="laps"):
        export_session(data, tmp_path, {"session_id": "2023-01-5"}, False)
    assert not verified(tmp_path, False)


def test_core_export_does_not_satisfy_telemetry_request(tmp_path):
    export_session(session(), tmp_path, {"session_id": "2023-01-5"}, False)
    assert not verified(tmp_path, True)


def test_missing_manifest_is_not_complete(tmp_path):
    assert not verified(tmp_path, False)


def test_access_rejection_is_actionable(monkeypatch):
    monkeypatch.setattr("fastf1.Cache.requests_get", lambda *a, **kw: SimpleNamespace(status_code=401))
    with pytest.raises(ProviderAccessError, match="HTTP 401"):
        check_access(SimpleNamespace(api_path="/static/example/"), 1)


def test_completed_ingestion_skips_network(monkeypatch, tmp_path):
    item = {"year": 2023, "round": 1, "slot": 5, "session_id": "2023-01-5", "session_name": "Race"}
    directory = tmp_path / "sessions" / "2023" / item["session_id"]
    export_session(session(), directory, item, False)
    def unexpected(*a, **kw):
        raise AssertionError("Network must not be accessed for a complete session")
    monkeypatch.setattr("fastf1.get_session", unexpected)
    assert ingest(item, tmp_path, False, 1)


def test_ingestion_records_failure_and_can_resume(monkeypatch, tmp_path):
    item = {"year": 2023, "round": 1, "slot": 5, "session_id": "2023-01-5", "session_name": "Race", "event_name": "Bahrain"}
    data = session()
    def broken(**kwargs):
        raise RuntimeError("interrupted download")
    data.load = broken
    monkeypatch.setattr("fastf1.get_session", lambda *a: data)
    monkeypatch.setattr("f1data.sessions.check_access", lambda *a: None)
    assert not ingest(item, tmp_path, False, 1)
    manifest_path = tmp_path / "sessions/2023/2023-01-5/manifest.json"
    assert json.loads(manifest_path.read_text())["status"] == "failed"
    data.load = lambda **kwargs: None
    assert ingest(item, tmp_path, False, 1)
    assert verified(manifest_path.parent, False)


def test_report_distinguishes_unattempted_sessions(tmp_path):
    (tmp_path / "inventory.json").write_text(json.dumps([{"session_id": "2023-01-5"}]))
    result = report(tmp_path)
    assert result["expected_sessions"] == 1
    assert result["not_attempted"] == ["2023-01-5"]


def test_native_telemetry_preserves_samples_and_driver_identity(tmp_path):
    data = session()
    data.car_data = {d: pd.DataFrame({"Speed": [100, 200], "Time": pd.to_timedelta([1, 2], unit="s")}) for d in ("1", "2")}
    data.pos_data = {d: pd.DataFrame({"X": [1, 2], "Y": [3, 4]}) for d in ("1", "2")}
    manifest = export_session(data, tmp_path, {"session_id": "2023-01-5"}, True)
    assert manifest["status"] == "complete"
    assert verified(tmp_path, True)
    assert not (tmp_path / "normalized/car_data").exists()
    frame = read_telemetry(tmp_path, "car_data", "1")
    assert frame.driver_number.tolist() == ["1", "1"]
    assert frame.session_id.tolist() == ["2023-01-5"] * 2
    assert frame.Speed.tolist() == [100, 200]
    source = pd.read_parquet(tmp_path / "source/car_data/driver-1.parquet")
    assert list(source.columns) == ["Speed", "Time"]
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "running"}))
    assert not verified(tmp_path, False)


def test_default_data_root_is_in_repository_not_working_directory(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.delenv("F1_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    expected = Path(__file__).resolve().parents[3] / "data"  # tests -> f1data -> packages -> repository
    assert default_data_root() == expected


def test_repository_root_is_not_found_outside_a_checkout(tmp_path):
    assert repository_root(tmp_path) is None


def test_data_root_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("F1_DATA_ROOT", str(tmp_path / "custom"))
    assert default_data_root() == tmp_path / "custom"


def test_new_inventory_preserves_earlier_races(tmp_path):
    earlier = {"session_id": "2023-01-5", "event_name": "Bahrain Grand Prix"}
    newer = {"session_id": "2024-01-5", "event_name": "Bahrain Grand Prix"}
    merge_inventory(tmp_path, [earlier])
    merge_inventory(tmp_path, [newer])
    combined = json.loads((tmp_path / "inventory.json").read_text())
    assert {item["session_id"] for item in combined} == {"2023-01-5", "2024-01-5"}
    merge_inventory(tmp_path, [{**earlier, "purpose": "updated"}])
    combined = json.loads((tmp_path / "inventory.json").read_text())
    assert len(combined) == 2
    assert next(item for item in combined if item["session_id"] == earlier["session_id"])["purpose"] == "updated"


def test_progress_distinguishes_core_and_telemetry(tmp_path):
    item = {"year": 2023, "round": 1, "slot": 5, "session_id": "2023-01-5", "event_name": "Bahrain Grand Prix", "session_name": "Race"}
    merge_inventory(tmp_path, [item])
    export_session(session(), tmp_path / "sessions/2023/2023-01-5", item, False)
    report(tmp_path)
    ledger = json.loads((tmp_path / "progress.json").read_text())
    assert ledger["sessions"][0]["core_status"] == "complete"
    assert ledger["sessions"][0]["telemetry_status"] == "not_started"
    assert ledger["weekends"][0]["core_completed"] == 1
    assert ledger["weekends"][0]["telemetry_completed"] == 0


def test_ingestion_records_failure_in_progress_immediately(monkeypatch, tmp_path):
    item = {"year": 2023, "round": 1, "slot": 5, "session_id": "2023-01-5", "session_name": "Race", "event_name": "Bahrain"}
    def broken(*args):
        raise RuntimeError("network interrupted")
    monkeypatch.setattr("fastf1.get_session", broken)
    assert not ingest(item, tmp_path, False, 1)
    ledger = json.loads((tmp_path / "progress.json").read_text())
    assert ledger["sessions"][0]["core_status"] == "failed"
    assert ledger["sessions"][0]["error"]["message"] == "network interrupted"


def test_partial_telemetry_does_not_invalidate_complete_core(tmp_path):
    data = session()
    data.car_data = {"1": pd.DataFrame({"Speed": [100]})}
    data.pos_data = {"1": pd.DataFrame({"X": [1]})}
    item = {"year": 2023, "round": 1, "slot": 5, "session_id": "2023-01-5", "event_name": "Bahrain", "session_name": "Race"}
    directory = tmp_path / "sessions/2023/2023-01-5"
    manifest = export_session(data, directory, item, True)
    assert manifest["status"] == "partial"
    assert verified(directory, False)
    assert not verified(directory, True)
    report(tmp_path)
    ledger = json.loads((tmp_path / "progress.json").read_text())
    assert ledger["sessions"][0]["core_status"] == "complete"
    assert ledger["sessions"][0]["telemetry_status"] == "partial"


def test_testing_session_without_classification_is_complete_not_partial(tmp_path):
    data = session()
    data.results = pd.DataFrame()
    item = {"year": 2023, "round": 0, "slot": 1, "session_id": "2023-T1-1", "session_name": "Practice 1",
            "is_testing": True, "test_number": 1}
    manifest = export_session(data, tmp_path, item, False)
    assert manifest["status"] == "complete"
    assert manifest["not_applicable_tables"] == ["results"] and manifest["missing_tables"] == []
    assert verified(tmp_path, False)


def test_empty_results_still_fail_a_championship_session(tmp_path):
    data = session()
    data.results = pd.DataFrame()
    with pytest.raises(ValueError, match="results"):
        export_session(data, tmp_path, {"session_id": "2023-01-5"}, False)


def test_testing_sessions_load_through_the_testing_api(monkeypatch, tmp_path):
    item = {"year": 2023, "round": 0, "slot": 2, "session_id": "2023-T1-2", "session_name": "Practice 2",
            "event_name": "Pre-Season Testing", "is_testing": True, "test_number": 1}
    data = session()
    data.load = lambda **kwargs: None
    calls = []
    monkeypatch.setattr("fastf1.get_testing_session", lambda *a: calls.append(a) or data)
    monkeypatch.setattr("fastf1.get_session", lambda *a: pytest.fail("championship API used for testing"))
    monkeypatch.setattr("f1data.sessions.check_access", lambda *a: None)
    assert ingest(item, tmp_path, False, 1)
    assert calls == [(2023, 1, 2)]
    ledger = json.loads((tmp_path / "progress.json").read_text())
    assert ledger["weekends"][0]["test_number"] == 1


def test_compact_removes_only_provably_redundant_storage(tmp_path):
    from f1data.sessions import compact
    data = session()
    data.car_data = {d: pd.DataFrame({"Speed": [100, 200]}) for d in ("1", "2")}
    data.pos_data = {d: pd.DataFrame({"X": [1, 2]}) for d in ("1", "2")}
    directory = tmp_path / "sessions/2023/2023-01-5"
    export_session(data, directory, {"session_id": "2023-01-5"}, True)
    # Recreate the legacy layout: a normalized copy with constant metadata columns.
    manifest = json.loads((directory / "manifest.json").read_text())
    for stream in ("car_data", "pos_data"):
        for d in ("1", "2"):
            name = f"normalized/{stream}/driver-{d}.parquet"
            frame = pd.read_parquet(directory / f"source/{stream}/driver-{d}.parquet")
            manifest["files"][name] = write_table(frame, directory / name, {"session_id": "2023-01-5", "driver_number": d})
    (directory / "manifest.json").write_text(json.dumps(manifest))
    cache = tmp_path / "cache"
    (cache / "2023/x").mkdir(parents=True)
    (cache / "fastf1_http_cache.sqlite").write_bytes(b"raw")
    (cache / "2023/x/car_data.ff1pkl").write_bytes(b"parsed")
    result = compact(tmp_path)
    assert result["normalized_telemetry_files"] == 4 and result["parsed_cache_files"] == 1
    assert not (directory / "normalized/car_data").exists()
    assert (cache / "fastf1_http_cache.sqlite").read_bytes() == b"raw"
    assert verified(directory, True) and verified(directory, False)
    assert read_telemetry(directory, "pos_data", "2").X.tolist() == [1, 2]
    assert compact(tmp_path)["normalized_telemetry_files"] == 0


def test_compact_refuses_when_source_copy_is_corrupt(tmp_path):
    from f1data.sessions import compact
    data = session()
    data.car_data = {"1": pd.DataFrame({"Speed": [1]}), "2": pd.DataFrame({"Speed": [2]})}
    data.pos_data = {"1": pd.DataFrame({"X": [1]}), "2": pd.DataFrame({"X": [2]})}
    directory = tmp_path / "sessions/2023/2023-01-5"
    export_session(data, directory, {"session_id": "2023-01-5"}, True)
    manifest = json.loads((directory / "manifest.json").read_text())
    name = "normalized/car_data/driver-1.parquet"
    manifest["files"][name] = write_table(pd.DataFrame({"Speed": [1]}), directory / name, {})
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "source/car_data/driver-1.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        compact(tmp_path)
    assert (directory / name).exists()


def test_403_on_a_missing_session_is_not_treated_as_a_login_problem(monkeypatch):
    from f1data.sessions import SessionUnavailable, REFERENCE_SESSION_PATH
    def get(url, **kw):
        return SimpleNamespace(status_code=200 if REFERENCE_SESSION_PATH in str(url) else 403)
    monkeypatch.setattr("fastf1.Cache.requests_get", get)
    with pytest.raises(SessionUnavailable):
        check_access(SimpleNamespace(api_path="/static/2019/cancelled/"), 1)


def test_missing_session_is_recorded_as_a_known_gap(monkeypatch, tmp_path):
    from f1data.sessions import unresolved_attempt
    item = {"year": 2019, "round": 17, "slot": 3, "session_id": "2019-17-3", "session_name": "Practice 3", "event_name": "Japan"}
    monkeypatch.setattr("fastf1.get_session", lambda *a: SimpleNamespace(api_path="/static/x/"))
    monkeypatch.setattr("fastf1.Cache.requests_get",
                        lambda url, **kw: SimpleNamespace(status_code=403 if "/static/x/" in str(url) else 200))
    assert not ingest(item, tmp_path, False, 1)
    assert unresolved_attempt(tmp_path / "sessions/2019/2019-17-3", True) == "failed: SessionUnavailable"


def test_fast_skip_check_catches_missing_or_truncated_files_and_verify_catches_corruption(tmp_path):
    from f1data.sessions import present, verify_all
    directory = tmp_path / "sessions/2023/2023-01-5"
    export_session(session(), directory, {"session_id": "2023-01-5"}, False)
    assert present(directory, False) and verify_all(tmp_path) == []
    laps = directory / "source/laps.parquet"
    original = laps.read_bytes()
    laps.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))  # same size, different content
    assert present(directory, False)
    assert verify_all(tmp_path) == ["2023-01-5"]
    laps.write_bytes(original[:-5])
    assert not present(directory, False)


def test_missed_sessions_file_lists_known_gaps_only(tmp_path):
    from f1data.sessions import write_progress
    good = tmp_path / "sessions/2023/2023-01-5"
    export_session(session(), good, {"session_id": "2023-01-5", "year": 2023, "round": 1}, False)
    gap = tmp_path / "sessions/2019/2019-17-3"
    gap.mkdir(parents=True)
    (gap / "manifest.json").write_text(json.dumps({"session_id": "2019-17-3", "status": "failed",
        "session_name": "Practice 3", "error": {"type": "SessionUnavailable", "message": "no data (reference checked uncached)"}}))
    limited = tmp_path / "sessions/2019/2019-18-1"
    limited.mkdir(parents=True)
    (limited / "manifest.json").write_text(json.dumps({"session_id": "2019-18-1", "status": "failed",
        "error": {"type": "RateLimitExceededError", "message": "500 calls/h"}}))
    write_progress(tmp_path)
    missed = json.loads((tmp_path / "missed-sessions.json").read_text())
    assert missed["count"] == 1
    assert missed["sessions"][0]["session_id"] == "2019-17-3"
    assert missed["sessions"][0]["reason"] == "failed: SessionUnavailable"


def test_expired_login_is_not_mistaken_for_missing_sessions(monkeypatch):
    """The reference session's 200 sits in the HTTP cache; only a live probe shows the login has expired."""
    import fastf1
    from f1data.sessions import ProviderAccessError, REFERENCE_SESSION_PATH
    def get(url, **kw):
        cached = REFERENCE_SESSION_PATH in str(url) and not fastf1.Cache._http_tmp_disabled
        return SimpleNamespace(status_code=200 if cached else 403)
    monkeypatch.setattr("fastf1.Cache.requests_get", get)
    before = fastf1.Cache._http_tmp_disabled
    with pytest.raises(ProviderAccessError):
        check_access(SimpleNamespace(api_path="/static/2022/race/"), 1)
    assert fastf1.Cache._http_tmp_disabled == before


def test_gaps_recorded_by_the_cached_probe_are_attempted_again(tmp_path):
    from f1data.sessions import unresolved_attempt
    directory = tmp_path / "sessions/2022/2022-01-5"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"status": "failed", "error": {
        "type": "SessionUnavailable", "message": "HTTP 403 ... while a known session is accessible; no data"}}))
    assert unresolved_attempt(directory, True) is None


def test_sessions_lost_upstream_are_served_by_the_fastf1_mirror(monkeypatch):
    from f1data.sessions import LIVETIMING_MIRROR
    def get(url, **kw):
        urls = [url] if isinstance(url, str) else list(url)
        # like FastF1: first URL answering below 400 wins
        return SimpleNamespace(status_code=200 if any(u.startswith(LIVETIMING_MIRROR) for u in urls) else 403,
                               raise_for_status=lambda: None)
    monkeypatch.setattr("fastf1.Cache.requests_get", get)
    check_access(SimpleNamespace(api_path="/static/2022/2022-03-20_Bahrain_Grand_Prix/2022-03-20_Race/"), 1)


def test_session_missing_on_both_servers_is_a_known_gap(monkeypatch):
    """F1 answers 403 for a missing session, the mirror 404; FastF1 returns the mirror's response."""
    from f1data.sessions import SessionUnavailable, REFERENCE_SESSION_PATH
    def get(url, **kw):
        return SimpleNamespace(status_code=200 if REFERENCE_SESSION_PATH in str(url) else 404)
    monkeypatch.setattr("fastf1.Cache.requests_get", get)
    with pytest.raises(SessionUnavailable):
        check_access(SimpleNamespace(api_path="/static/2019/cancelled/"), 1)


def test_verify_does_not_count_tables_the_source_never_had(tmp_path):
    """A partial session without a weather feed (2022 pre-season test 2) is intact."""
    from f1data.sessions import verify_all
    directory = tmp_path / "sessions/2022/2022-T2-1"
    export_session(session(), directory, {"session_id": "2022-T2-1"}, False)
    manifest = json.loads((directory / "manifest.json").read_text())
    for layer in ("source", "normalized"):
        manifest["files"].pop(f"{layer}/weather.parquet", None)
        (directory / f"{layer}/weather.parquet").unlink(missing_ok=True)
    manifest["status"], manifest["missing_tables"] = "partial", ["weather"]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    assert verify_all(tmp_path) == []
