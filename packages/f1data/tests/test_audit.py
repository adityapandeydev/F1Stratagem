import hashlib
import json

import pytest

from f1data import audit
from f1data.cli import build_parser, main


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def archive(tmp_path):
    jolpica = tmp_path / "external" / "jolpica"
    write(jolpica / "raw/2020/races/offset-00000.json", b"{}")
    (jolpica / "raw/2020/manifest.json").write_text(json.dumps(
        {"pages": {"raw/2020/races/offset-00000.json": {"sha256": sha(b"{}")}}}))
    meteo = tmp_path / "external" / "open-meteo"
    write(meteo / "raw/observed/2020/2020-01.json", b"[]")
    (meteo / "manifest.json").write_text(json.dumps(
        {"entries": {"observed/2020-01": {"file": "raw/observed/2020/2020-01.json", "sha256": sha(b"[]")}}}))
    fia = tmp_path / "external" / "fia"
    write(fia / "pdf/2020/austria/doc-001.pdf", b"%PDF")
    (fia / "manifest.json").write_text(json.dumps(
        {"files": {"pdf/2020/austria/doc-001.pdf": {"sha256": sha(b"%PDF")}}}))
    return tmp_path


@pytest.mark.parametrize("source", sorted(audit.SOURCES))
def test_clean_source_verifies(archive, source):
    result = audit.verify_source(source, archive)
    assert result["ok"] and result["files_checked"] == 1


def test_changed_missing_and_unlisted_files_fail(archive):
    fia = archive / "external" / "fia"
    (fia / "pdf/2020/austria/doc-001.pdf").write_bytes(b"changed")
    write(fia / "pdf/2020/austria/doc-002.pdf", b"%PDF")
    result = audit.verify_source("fia", archive)
    assert result["mismatched"] == ["pdf/2020/austria/doc-001.pdf"]
    assert result["unlisted"] == ["pdf/2020/austria/doc-002.pdf"]
    assert not result["ok"]
    (fia / "pdf/2020/austria/doc-001.pdf").unlink()
    assert audit.verify_source("fia", archive)["missing"] == ["pdf/2020/austria/doc-001.pdf"]


def test_source_with_nothing_recorded_does_not_pass(archive):
    (archive / "external" / "fia" / "manifest.json").write_text(json.dumps({"files": {}}))
    (archive / "external" / "fia" / "pdf/2020/austria/doc-001.pdf").unlink()
    assert not audit.verify_source("fia", archive)["ok"]


def test_table_hashes_are_checked(archive):
    meteo = archive / "external" / "open-meteo"
    write(meteo / "tables/observed_hourly.parquet", b"table")
    (meteo / "tables/manifest.json").write_text(json.dumps(
        {"files": {"observed_hourly": {"rows": 1, "sha256": sha(b"other")}}}))
    assert audit.verify_source("openmeteo", archive)["mismatched"] == ["tables/observed_hourly.parquet"]


def test_cli_exposes_every_source_and_verifies_external_sources(archive, capsys):
    parser = build_parser()
    for source in ("fastf1", "jolpica", "openmeteo", "fia", "verify"):
        assert parser.parse_args([source] + ([] if source == "verify" else ["verify"])).handler
    assert main(["jolpica", "verify", "--data-root", str(archive)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
