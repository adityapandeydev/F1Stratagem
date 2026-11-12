import json

import pytest
import requests

from f1data import fia

ROW = """<li class="document-row key-{n}">  <a href="/sites/default/files/decision-document/2024 Abu Dhabi Grand Prix - {title}.pdf" download target="_blank">
  <div class="file-type"><div class="pdf"></div></div>
  <div class="title">
  {inner}Doc {n} - {title}{close}
  </div>
  <div class="published">
  Published on <span  class="date-display-single">{date}</span> CET  </div>
  </a>
</li>"""


def row(n, title, date="06.12.24 07:57", wrapped=False):
    inner = '<div class="field"><div class="field-items"><div class="field-item even">' if wrapped else ""
    close = "</div></div></div>" if wrapped else ""
    return ROW.format(n=n, title=title, date=date, inner=inner, close=close)


SEASON_PAGE = f"""
<select><option value="/documents/championships/fia-formula-one-world-championship-14/season/season-2025-2071">SEASON 2025</option>
<option value="/documents/championships/fia-formula-one-world-championship-14/season/season-2019-971">SEASON 2019</option></select>
<ul class="event-wrapper"><li>
  <div class="event-title active">Abu Dhabi Grand Prix</div>
  <ul class="document-type-wrapper open data-id-50086"><li><ul class="document-row-wrapper non-categorised">
  {row(12, "Car Presentation Submissions")}
  </ul></li></ul></li></ul>
<ul class="event-wrapper"><li>
  <a href="/decision-document-list/nojs/50085" class="event-title data-id-50085  use-ajax">
          Qatar Grand Prix        </a>
  <ul class="document-type-wrapper  data-id-50085"></ul></li></ul>
"""


def test_season_page_lists_seasons_and_events():
    assert fia.season_urls(SEASON_PAGE) == {
        2025: "/documents/championships/fia-formula-one-world-championship-14/season/season-2025-2071",
        2019: "/documents/championships/fia-formula-one-world-championship-14/season/season-2019-971"}
    assert fia.events(SEASON_PAGE) == {50086: "Abu Dhabi Grand Prix", 50085: "Qatar Grand Prix"}


def test_documents_parse_both_title_layouts_and_paris_time():
    # 06.07.24 is summer time (CEST, UTC+2) although the site labels it CET
    found = fia.documents(row(9, "PU elements used per driver up to now") +
                          row(24, "Decision - Car 44 - Alleged impeding of car 4", "06.07.24 12:49", wrapped=True))
    assert [(d["doc_number"], d["category"]) for d in found] == [(9, "pu_elements"), (24, "decision")]
    assert found[0]["published_utc"] == "2024-12-06T06:57:00+00:00"
    assert found[1]["published_utc"] == "2024-07-06T10:49:00+00:00"
    assert found[1]["title"] == "Decision - Car 44 - Alleged impeding of car 4"
    assert found[0]["url"].endswith("PU elements used per driver up to now.pdf")


@pytest.mark.parametrize("title, category", [
    ("Infringement - Car 16 - PU Elements", "infringement"),
    ("New PU elements for this Competition - Corrected", "pu_elements"),
    ("New RNCs for this Competition", "components"),
    ("Event Notes - Pirelli Preview", "tyres"),
    ("Offence - Car 44 - Speeding in the pit lane", "decision"),
    ("Summons - Car 44 - Alleged impeding of Car 4", "summons"),
    ("Parts and parameters been replaced and or changed during Parc Fermé", "technical"),
    ("Final Starting Grid", "starting_grid"),
    ("Post-Race Procedure", "other"),
])
def test_categories(title, category):
    assert fia.categorise(title) == category


class FakeResponse:
    def __init__(self, status=200, content=b"", payload=None, text=""):
        self.status_code, self.content, self._payload, self.text = status, content, payload, text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeClient:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def request(self, method, path):
        self.calls.append((method, path))
        response = self.pages[path]
        response.raise_for_status()
        return response


def test_index_then_download_is_resumable(tmp_path):
    ajax = [{"command": "settings"}, {"command": "insert", "data": row(5, "New PU elements for this Competition")},
            {"command": "insert", "data": row(6, "Post-Race Procedure")}]
    good = "https://www.fia.com/sites/default/files/decision-document/2024 Abu Dhabi Grand Prix - New PU elements for this Competition.pdf"
    client = FakeClient({
        "/season-2024": FakeResponse(text=SEASON_PAGE),
        "/decision-document-list/ajax/50085": FakeResponse(payload=ajax),
        "/decision-document-list/ajax/50086": FakeResponse(payload=[{"command": "insert", "data": row(12, "Car Presentation Submissions")}]),
        good: FakeResponse(content=b"%PDF-1.7 pu"),
        "https://www.fia.com/sites/default/files/decision-document/2024 Abu Dhabi Grand Prix - Car Presentation Submissions.pdf":
            FakeResponse(status=404),
    })
    index = fia.index_season(tmp_path, client, 2024, "/season-2024", final=True)
    assert index["events"] == 2 and len(index["documents"]) == 3
    calls = len(client.calls)
    fia.index_season(tmp_path, client, 2024, "/season-2024", final=True)
    assert len(client.calls) == calls  # final season: no refetch

    counts = fia.download(tmp_path, client, {"pu_elements", "car_presentation"})
    assert counts == {"fetched": 1, "present": 0, "failed": 1, "skipped_failed": 0}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    (key, entry), = manifest["files"].items()
    assert key == "pdf/2024/qatar-grand-prix/doc-005-new-pu-elements-for-this-competition.pdf"
    assert entry["bytes"] == len(b"%PDF-1.7 pu") and entry["category"] == "pu_elements"
    again = fia.download(tmp_path, client, {"pu_elements", "car_presentation"})
    assert again == {"fetched": 0, "present": 1, "failed": 0, "skipped_failed": 1}
    assert fia.build_index_table(tmp_path)["rows"] == 3


def test_non_pdf_response_is_a_failure(tmp_path):
    (tmp_path / "index").mkdir()
    doc = {"season": 2024, "event": "Qatar Grand Prix", "event_id": 1, **fia.documents(row(9, "PU elements used"))[0]}
    (tmp_path / "index" / "2024.json").write_text(json.dumps({"documents": [doc]}))
    client = FakeClient({doc["url"]: FakeResponse(content=b"<html>maintenance</html>")})
    assert fia.download(tmp_path, client, {"pu_elements"})["failed"] == 1
    assert not list((tmp_path / "pdf").rglob("*.pdf")) if (tmp_path / "pdf").exists() else True


def test_doc_number_falls_back_to_row_key():
    untitled = row(33, "Final Starting Grid").replace("Doc 33 - ", "")
    found = fia.documents(untitled + row(58, "Championship Points"))
    assert [(d["doc_number"], d["doc_number_source"]) for d in found] == [(33, "row_key"), (58, "title")]
    assert found[0]["title"] == "Final Starting Grid"


def test_connection_dropped_mid_download_is_retried():
    class Limiter:
        def acquire(self):
            pass

    class Session:
        def __init__(self):
            self.headers, self.calls = {}, 0

        def request(self, method, url, timeout):
            self.calls += 1
            if self.calls == 1:
                raise requests.exceptions.ChunkedEncodingError("IncompleteRead")
            return FakeResponse(content=b"%PDF ok")

    session = Session()
    client = fia.Client(Limiter(), session=session, sleep=lambda s: None)
    assert client.request("GET", "https://www.fia.com/x.pdf").content == b"%PDF ok"
    assert session.calls == 2


def test_base64_served_pdf_is_decoded():
    import base64
    raw = b"%PDF-1.4\n%binary\n"
    assert fia.pdf_bytes(raw) == (raw, None)
    assert fia.pdf_bytes(base64.b64encode(raw) + b"\r\n") == (raw, "base64")
    with pytest.raises(ValueError):
        fia.pdf_bytes(b"<html>maintenance</html>")
