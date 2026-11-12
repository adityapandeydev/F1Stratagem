import json
import pandas as pd
import pytest

from test_sessions import session
from f1data.sessions import export_session, ingest
from f1data.storage import checksum
from f1data.snapshots import record_snapshot, snapshot_asof


def test_snapshot_is_immutable_idempotent_and_not_backdated(tmp_path):
    directory = tmp_path / "sessions/2023/2023-01-5"
    item = {"session_id": "2023-01-5", "year": 2023, "round": 1, "slot": 5, "session_name": "Race"}
    export_session(session(), directory, item, False)
    output = record_snapshot(tmp_path, directory)
    original = checksum(output / "manifest.json")
    assert record_snapshot(tmp_path, directory) == output
    assert checksum(output / "manifest.json") == original
    assert snapshot_asof(tmp_path, item["session_id"], pd.Timestamp("2023-03-05T00:00Z")) is None
    assert snapshot_asof(tmp_path, item["session_id"], pd.Timestamp.now(tz="UTC")+pd.Timedelta(minutes=1)) == output
    with pytest.raises(ValueError, match="timezone"):
        snapshot_asof(tmp_path, item["session_id"], pd.Timestamp("2023-03-05"))


def test_updated_source_creates_a_new_version_preserving_the_old_one(tmp_path):
    directory = tmp_path / "sessions/2023/2023-01-5"
    item = {"session_id": "2023-01-5", "year": 2023, "round": 1, "slot": 5, "session_name": "Race"}
    data = session()
    export_session(data, directory, item, False)
    first = record_snapshot(tmp_path, directory)
    first_hash = checksum(first / "normalized/laps.parquet")
    data.laps.loc[0, "LapTime"] = pd.Timedelta(seconds=91)
    export_session(data, directory, item, False)
    second = record_snapshot(tmp_path, directory)
    assert first != second and checksum(first / "normalized/laps.parquet") == first_hash
    assert len(json.loads((tmp_path / "snapshots/index.json").read_text())["receipts"]) == 2


def test_ingestion_snapshots_old_core_before_a_telemetry_upgrade(monkeypatch, tmp_path):
    directory = tmp_path / "sessions/2023/2023-01-5"
    item = {"session_id": "2023-01-5", "year": 2023, "round": 1, "slot": 5,
            "session_name": "Race", "event_name": "Test GP"}
    data = session()
    export_session(data, directory, item, False)
    original = checksum(directory / "normalized/laps.parquet")
    data.laps.loc[0, "LapTime"] = pd.Timedelta(seconds=91)
    data.car_data = {n: pd.DataFrame({"Speed": [100]}) for n in ("1", "2")}
    data.pos_data = {n: pd.DataFrame({"X": [1]}) for n in ("1", "2")}
    data.load = lambda **kwargs: None
    monkeypatch.setattr("fastf1.get_session", lambda *a: data)
    monkeypatch.setattr("f1data.sessions.check_access", lambda *a: None)
    assert ingest(item, tmp_path, True, 1)
    receipts = json.loads((tmp_path / "snapshots/index.json").read_text())["receipts"]
    assert len(receipts) == 2
    assert any(checksum(tmp_path/r["directory"]/"normalized/laps.parquet") == original for r in receipts)
