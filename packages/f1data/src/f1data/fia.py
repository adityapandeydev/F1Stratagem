"""FIA Formula 1 event documents (fia.com), 2019 onward.

Three stages, each resumable and offline after the first:

  index     season pages and per-event document lists -> one row per document
            (season, event, doc number, title, category, publication time, URL)
  download  the PDFs of the selected categories, unchanged, with SHA-256
  extract   per-category parsers -> tables (see fia_extract.py)

    data/external/fia/
      raw/pages/<season>/season.html          season page as served
      raw/pages/<season>/event-<id>.json      per-event document list (Drupal AJAX response)
      index/<season>.json                     parsed document rows for the season
      pdf/<season>/<event-slug>/doc-NNN-<name>.pdf
      manifest.json                           per PDF: URL, SHA-256, bytes, fetched_at
      failed.json                             downloads that failed; retried with --retry-failed
      rate-state.json                         request log (keeps the pace across runs)
      tables/documents.parquet                the index as a table

fia.com has documents from 2019 (plus a few from 2015). robots.txt allows these
paths and asks for a 10 s crawl delay, which the client keeps. Publication
times are shown as "dd.mm.yy HH:MM CET"; they are read as Paris local time
(CET/CEST) and converted to UTC.

Terms: the FIA website terms reserve all rights in site content (sections 3.5,
3.9). The PDFs stay private research copies and are never republished; the
product links to fia.com. Use of extracted facts needs FIA consent or legal
advice before a commercial launch.
"""

import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .jolpica import RateLimiter
from .storage import atomic_json, checksum, utcnow, write_table

BASE_URL = "https://www.fia.com"
SEASON_LIST_PATH = "/documents/championships/fia-formula-one-world-championship-14/season/season-2024-2043"
USER_AGENT = "f1data/0.1.0"
CRAWL_DELAY = 10.0
PARIS = ZoneInfo("Europe/Paris")

# Title -> category. First match wins; order matters (an "Infringement - Car 16 -
# PU Elements" notice is an infringement, not a PU usage sheet).
CATEGORIES = (
    ("decision", r"^decision\b|^offence\b|\bdecision\s*-\s*car\b"),
    ("infringement", r"^infringement\b"),
    ("summons", r"^summons\b"),
    ("car_presentation", r"car presentation"),
    ("pu_elements", r"\bpu elements?\b|power unit elements?"),
    ("components", r"\brncs?\b|gearbox|regulated non"),
    ("tyres", r"pirelli|tyre"),
    ("entry_list", r"entry list"),
    ("starting_grid", r"starting grid"),
    ("technical", r"technical delegate|scrutineering|parc ferm|fuel sample|car weights|weighing|checks on car"),
    ("classification", r"classification|lap analysis|lap chart|history chart|championship points|speed trap|fastest laps|best sector"),
    ("timing", r"timetable|schedule"),
)
DOWNLOAD_CATEGORIES = ("decision", "infringement", "car_presentation", "pu_elements", "components", "tyres",
                       "entry_list", "starting_grid", "technical")


def categorise(title):
    text = title.lower()
    for name, pattern in CATEGORIES:
        if re.search(pattern, text):
            return name
    return "other"


class Client:
    def __init__(self, limiter, session=None, attempts=4, sleep=time.sleep):
        self.limiter, self.attempts, self.sleep = limiter, attempts, sleep
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def request(self, method, path):
        url = path if path.startswith("http") else BASE_URL + path
        for attempt in range(self.attempts):
            self.limiter.acquire()
            try:
                response = self.session.request(method, url, timeout=120)
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):  # dropped mid-body too
                if attempt + 1 == self.attempts:
                    raise
                self.sleep(min(60 * 2 ** attempt, 900))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 == self.attempts:
                    response.raise_for_status()
                print(f"FIA HTTP {response.status_code} for {url}; retrying", flush=True)
                self.sleep(min(120 * 2 ** attempt, 1800))
                continue
            response.raise_for_status()
            return response
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------- parsing

SEASON_OPTION = re.compile(r'value="(/documents/championships/fia-formula-one-world-championship-14/season/season-(\d{4})-\d+)"')
EVENT_LINK = re.compile(r'<a href="/decision-document-list/nojs/(\d+)" class="event-title[^"]*">\s*(.*?)\s*</a>', re.S)
EVENT_OPEN = re.compile(r'<div class="event-title[^"]*">\s*(.*?)\s*</div>\s*<ul class="document-type-wrapper[^"]*data-id-(\d+)"', re.S)
DOCUMENT_ROW = re.compile(r'<li class="document-row(?: key-(\d+))?[^"]*">(.*?)</li>', re.S)
HREF = re.compile(r'<a href="([^"]+)"')
TITLE = re.compile(r'<div class="title">(.*?)</div>\s*(?:</div>\s*)*<div class="published">', re.S)
PUBLISHED = re.compile(r'date-display-single">\s*([^<]+?)\s*<')
DOC_NUMBER = re.compile(r"^Doc\s+(\d+)\s*-\s*", re.I)


def clean(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def season_urls(page):
    return {int(year): path for path, year in SEASON_OPTION.findall(page)}


def events(page):
    found = {int(i): clean(name) for i, name in EVENT_LINK.findall(page)}
    found.update({int(i): clean(name) for name, i in EVENT_OPEN.findall(page)})
    return found


def published_utc(text):
    try:
        local = datetime.strptime(text, "%d.%m.%y %H:%M").replace(tzinfo=PARIS)
    except ValueError:
        return None
    return local.astimezone(timezone.utc).isoformat()


def documents(fragment):
    rows = []
    for key, block in DOCUMENT_ROW.findall(fragment):
        href, title, published = HREF.search(block), TITLE.search(block), PUBLISHED.search(block)
        if not href or not title:
            continue
        name = clean(title.group(1))
        number = DOC_NUMBER.match(name)
        published_text = published.group(1) if published else None
        # About a third of titles lack "Doc N - "; the row key carries the same
        # number (key-58 is "Doc 58") and is used when the title has none.
        rows.append({
            "doc_number": int(number.group(1)) if number else (int(key) if key else None),
            "doc_number_source": "title" if number else ("row_key" if key else None),
            "title": DOC_NUMBER.sub("", name), "category": categorise(DOC_NUMBER.sub("", name)),
            "published_local": published_text, "published_utc": published_utc(published_text or ""),
            "url": BASE_URL + html.unescape(href.group(1)),
        })
    return rows


def ajax_fragment(payload):
    return "".join(command.get("data") for command in payload if isinstance(command.get("data"), str))


# ---------------------------------------------------------------- index

def slug(text, limit=80):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].strip("-")


def index_season(root, client, season, path, final):
    out = root / "index" / f"{season}.json"
    if final and out.is_file():
        print(f"SKIP index {season}", flush=True)
        return json.loads(out.read_text(encoding="utf-8"))
    pages = root / "raw" / "pages" / str(season)
    page = client.request("GET", path).text
    pages.mkdir(parents=True, exist_ok=True)
    (pages / "season.html").write_text(page, encoding="utf-8")
    rows = []
    for event_id, event_name in sorted(events(page).items()):
        cached = pages / f"event-{event_id}.json"
        if final and cached.is_file():
            payload = json.loads(cached.read_text(encoding="utf-8"))
        else:
            payload = client.request("POST", f"/decision-document-list/ajax/{event_id}").json()
            atomic_json(cached, payload)
        found = documents(ajax_fragment(payload))
        print(f"INDEX {season} {event_name}: {len(found)} documents", flush=True)
        rows += [{"season": season, "event": event_name, "event_id": event_id, **row} for row in found]
    result = {"season": season, "final": final, "indexed_at_utc": utcnow(), "source": BASE_URL + path,
              "events": len({r["event_id"] for r in rows}), "documents": rows}
    atomic_json(out, result)
    return result


def build_index_table(root):
    rows = []
    for path in sorted((root / "index").glob("*.json")):
        rows += json.loads(path.read_text(encoding="utf-8"))["documents"]
    if not rows:
        return None
    frame = pd.DataFrame(rows).astype({"title": "string", "category": "string", "event": "string", "url": "string"})
    return write_table(frame, root / "tables" / "documents.parquet", {})


# ---------------------------------------------------------------- download

def pdf_bytes(content):
    """The PDF to store, and how it was served. fia.com serves at least one file
    base64-encoded ("JVBERi0" is base64 for "%PDF-": 2019 Australia, Offence -
    Car 18); it is decoded and stored as the PDF it contains."""
    if content.startswith(b"%PDF"):
        return content, None
    if content.lstrip().startswith(b"JVBERi0"):
        decoded = base64.b64decode(b"".join(content.split()), validate=True)
        if decoded.startswith(b"%PDF"):
            return decoded, "base64"
    raise ValueError("response is not a PDF")


def pdf_path(row):
    # The doc number makes names unique within an event (titles repeat, e.g.
    # "New PU elements for this Competition" on Friday and Saturday).
    name = slug(row["title"], 70)
    number = f"{row['doc_number']:03d}" if row["doc_number"] is not None else "xxx"
    return Path("pdf") / str(row["season"]) / slug(row["event"]) / f"doc-{number}-{name}.pdf"


def download(root, client, categories, seasons=None, retry_failed=False):
    manifest_path, failed_path = root / "manifest.json", root / "failed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"files": {}}
    failed = json.loads(failed_path.read_text(encoding="utf-8"))["entries"] if failed_path.is_file() else {}
    counts = {"fetched": 0, "present": 0, "failed": 0, "skipped_failed": 0}
    for path in sorted((root / "index").glob("*.json")):
        for row in json.loads(path.read_text(encoding="utf-8"))["documents"]:
            if row["category"] not in categories or (seasons and row["season"] not in seasons):
                continue
            relative = pdf_path(row)
            key = relative.as_posix()
            entry = manifest["files"].get(key)
            if entry and entry["url"] == row["url"] and (root / relative).is_file() \
                    and (root / relative).stat().st_size == entry["bytes"]:
                counts["present"] += 1
                continue
            if key in failed and not retry_failed:
                counts["skipped_failed"] += 1
                continue
            try:
                content, encoding = pdf_bytes(client.request("GET", row["url"]).content)
            except (requests.HTTPError, ValueError) as error:
                failed[key] = {**row, "error": str(error), "attempted_at_utc": utcnow()}
                atomic_json(failed_path, {"note": "Rerun download with --retry-failed.", "entries": failed})
                counts["failed"] += 1
                print(f"FAILED {key}: {error}", flush=True)
                continue
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".pdf.tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
            manifest["files"][key] = {**row, "sha256": checksum(target), "bytes": target.stat().st_size,
                                      "fetched_at_utc": utcnow(), **({"served_as": encoding} if encoding else {})}
            atomic_json(manifest_path, manifest)
            if failed.pop(key, None) is not None:
                atomic_json(failed_path, {"note": "Rerun download with --retry-failed.", "entries": failed})
            counts["fetched"] += 1
            print(f"OK {key}", flush=True)
    return counts


def register(commands, common):
    parser = commands.add_parser("fia", parents=[common], help="FIA documents: index, download and table extraction (2019+)",
                                 description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["index", "download", "extract", "report", "verify"])
    parser.add_argument("--seasons", type=int, nargs="+", help="seasons (default: every season fia.com lists)")
    parser.add_argument("--categories", nargs="+", default=list(DOWNLOAD_CATEGORIES),
                        choices=[name for name, _ in CATEGORIES] + ["other"])
    parser.add_argument("--retry-failed", action="store_true")
    parser.set_defaults(handler=run, parser=parser)


def run(args):
    root = args.data_root / "external" / "fia"
    if args.command == "extract":
        from .fia_extract import extract
        written, counts, problems = extract(root, set(args.seasons or []))
        print(json.dumps({"documents": counts, "rows": {k: v["rows"] for k, v in written.items()},
                          "problems": len(problems)}, indent=2), flush=True)
        return 0
    if args.command in ("report", "verify"):
        from . import audit
        status = audit.run_source("fia", args.data_root, args.command)
        if args.command == "report":
            print_index_summary(root)
        return status
    client = Client(RateLimiter(root / "rate-state.json", per_hour=int(3600 / CRAWL_DELAY), min_interval=CRAWL_DELAY))
    if args.command == "index":
        listing = {2024: SEASON_LIST_PATH, **season_urls(client.request("GET", SEASON_LIST_PATH).text)}
        current = datetime.now(timezone.utc).year
        for season in sorted(args.seasons or listing):
            if season not in listing:
                args.parser.error(f"fia.com lists no documents for {season}; available: {sorted(listing)}")
            index_season(root, client, season, listing[season], final=season < current)
        build_index_table(root)
    else:
        print(json.dumps(download(root, client, set(args.categories), set(args.seasons or []),
                                  args.retry_failed)), flush=True)
    print_index_summary(root)
    return 0


def print_index_summary(root):
    """Documents indexed per category and season."""
    rows = []
    for path in sorted((root / "index").glob("*.json")):
        rows += json.loads(path.read_text(encoding="utf-8"))["documents"]
    if rows:
        frame = pd.DataFrame(rows)
        print(frame.pivot_table(index="category", columns="season", values="url", aggfunc="count",
                                fill_value=0).to_string(), flush=True)
