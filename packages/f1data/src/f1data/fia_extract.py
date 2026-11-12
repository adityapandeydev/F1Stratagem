"""Structured tables from downloaded FIA documents (see fia.py).

Parsers, one per document type; each row keeps its provenance (file, SHA-256,
document number, publication time, page):

  pu_usage          "PU elements used per driver up to now": cumulative count per
                    driver and power-unit element (ICE, TC, MGU-H, MGU-K, ES, CE, EX).
  rnc_usage         "RNCs used per driver up to now": gearbox case/cassette and
                    driveline counts.
  car_updates       "Car Presentation Submissions": one row per declared upgrade
                    (team, component, primary reason, geometry change, description).
  tyre_nominations  "Pirelli Preview": the three nominated slick compounds (C1-C6)
                    and minimum starting pressures front/rear.

The PDFs are text-based (not scans). Layout-dependent parsers check what they
read (row counts, value ranges) and report problems instead of guessing; a
document that fails a check is listed in tables/extract-problems.json.
"""

import json
import re

import pandas as pd
import pdfplumber

from .paths import CONFIG_DIR
from .storage import atomic_json, utcnow, write_table

# ---------------------------------------------------------------- usage tables

# number, optional name (words not starting with a digit), counts. The name is
# empty when team and driver wrap above and below the numbers (2019 sheets).
USAGE_ROW = re.compile(r"^(\d{1,2})((?:\s+[^\d\s]\S*)*)((?:\s+\d{1,2})+)\s*$")


class ExtractionProblem(ValueError):
    pass


# A driver row whose car number is missing or garbled in the source (2019 Abu
# Dhabi prints no number for Sainz; 2026 Mexico prints "i" for Stroll).
UNNUMBERED_ROW = re.compile(r"^(?:\S{1,2}\s+)?([A-Z][^\d]*?[a-z][^\d]*?)((?:\s+\d{1,2})+)\s*$")


def usage_rows(lines, columns, warnings=None):
    """Rows as published. A row without a readable car number keeps its name and
    gets driver_number None; a driver-like row with the wrong number of values
    is reported in `warnings` instead of being dropped silently."""
    rows = []
    for line in lines:
        text = line.strip()
        match = USAGE_ROW.match(text)
        number = int(match.group(1)) if match else None
        if not match:
            match = UNNUMBERED_ROW.match(text)
            if not match or len(match.group(2).split()) != len(columns):
                continue
            name, values = match.group(1), match.group(2)
        else:
            name, values = match.group(2), match.group(3)
        counts = [int(v) for v in values.split()]
        if len(counts) != len(columns):
            if warnings is not None and name.strip() and len(counts) >= len(columns) - 2:
                warnings.append(f"row has {len(counts)} values for {len(columns)} columns: {text}")
            continue
        rows.append({"driver_number": number, "car_driver": name.strip(), **dict(zip(columns, counts))})
    return rows


def check_usage(rows, columns, what):
    if not 18 <= len(rows) <= 24:
        raise ExtractionProblem(f"{what}: {len(rows)} driver rows (expected 18-24)")
    # Keyed on number and name: sheets occasionally print a wrong number (2025
    # China lists Hadjar under Lawson's 30); such rows are kept as published.
    if len({(r["driver_number"], r["car_driver"]) for r in rows}) != len(rows):
        raise ExtractionProblem(f"{what}: duplicate driver rows")
    if any(not 0 <= r[c] <= 30 for r in rows for c in columns):
        raise ExtractionProblem(f"{what}: count outside 0-30")


def _stack(parts):
    """Join heading pieces stacked in one column: "MGU" over "-K" -> "MGU-K"."""
    text = ""
    for part in parts:
        text = text + part if (text.endswith("-") or part.startswith("-") or not text) else f"{text} {part}"
    return text


def pu_columns(words):
    """Element headings right of "Driver", read by position because they wrap
    (2026: "MGU" / "-K", "PU-" / "CE"). Returns (columns, header top) or None."""
    drivers = [w for w in words if w["text"] == "Driver"]
    for driver in drivers:
        parts = [w for w in words if w["x0"] > driver["x1"] and abs(w["top"] - driver["top"]) <= 14
                 and re.fullmatch(r"[A-Z-]{1,8}", w["text"])]
        clusters = []
        for w in sorted(parts, key=lambda w: (w["x0"] + w["x1"]) / 2):
            centre = (w["x0"] + w["x1"]) / 2
            if clusters and abs(centre - clusters[-1][0]) <= 9:
                clusters[-1][1].append(w)
            else:
                clusters.append([centre, [w]])
        columns = [_stack([w["text"] for w in sorted(ws, key=lambda w: w["top"])]) for _, ws in clusters]
        if len(columns) >= 5:
            return columns
    return None


def parse_pu_usage(pdf, warnings=None):
    for page_number, page in enumerate(pdf.pages, 1):
        # tight letter spacing: 2019 sheets otherwise glue two headings ("MGU-MGU-")
        columns = pu_columns(page.extract_words(x_tolerance=1.5))
        if not columns:
            continue
        rows = []
        # the table can run onto the next page
        for number, later in enumerate(pdf.pages[page_number - 1:], page_number):
            rows += [{**r, "page": number}
                     for r in usage_rows((later.extract_text() or "").splitlines(), columns, warnings)]
        check_usage(rows, columns, "PU usage")
        return rows, columns
    raise ExtractionProblem("PU usage: header row not found")


def parse_rnc_usage(pdf, warnings=None):
    """Column headings wrap over several lines, so their order is read from word
    positions: 'case' (case and cassette) versus 'driveline'."""
    for page_number, page in enumerate(pdf.pages, 1):
        words = page.extract_words()
        x = {w["text"].strip(",").lower(): w["x0"] for w in words}
        if "driveline" not in x or "case" not in x:
            continue
        columns = sorted([("gearbox_case_cassette", x["case"]), ("gearbox_driveline", x["driveline"])],
                         key=lambda item: item[1])
        columns = [name for name, _ in columns]
        rows = usage_rows((page.extract_text() or "").splitlines(), columns, warnings)
        check_usage(rows, columns, "RNC usage")
        return [{**r, "page": page_number} for r in rows], columns
    raise ExtractionProblem("RNC usage: column headings not found")


# ---------------------------------------------------------------- car presentation

CAR_COLUMNS = (("component", "updated"), ("primary_reason", "primary"),
               ("geometric_difference", "geometric"), ("description", "brief"))
NO_UPDATES = "no updates submitted"
# Column starts (points) on 2024 sheets, used when a page draws its headings as an image.
DEFAULT_EDGES = {"component": 94.0, "primary_reason": 170.0, "geometric_difference": 270.0, "description": 516.0}


def _lines(page):
    return [line.strip() for line in (page.extract_text() or "").splitlines() if line.strip()]


HEADING_STARTS = ("updated", "primary", "geometric", "brief", "component", "for update", "version")

# Entrant names change with sponsors ("Visa Cash App RB", "Stake F1 Team Kick
# Sauber"); map them to Jolpica constructor IDs. First match wins.
TEAM_IDS = (("red bull", "red_bull"), ("visa cash app", "rb"), ("racing bulls", "rb"), ("alphatauri", "alphatauri"),
            ("mercedes", "mercedes"), ("ferrari", "ferrari"), ("mclaren", "mclaren"), ("aston martin", "aston_martin"),
            ("alpine", "alpine"), ("williams", "williams"), ("sauber", "sauber"), ("alfa romeo", "alfa"),
            ("haas", "haas"), ("cadillac", "cadillac"), ("audi", "audi"))


def team_id(name):
    lowered = (name or "").lower()
    return next((team for key, team in TEAM_IDS if key in lowered), None)


def _team_name(lines):
    """The team is the line just above the table headings (whose lines come in
    varying order) or the "no updates" note. Styles vary: "SCUDERIA FERRARI",
    "*Mercedes-AMG PETRONAS F1 Team*". A continuation page has neither and keeps
    the previous team."""
    # only pages that carry the table headings (or the "no updates" note) name a
    # team; descriptions on continuation pages may start with "Updated ..."
    if not any("brief description" in line.lower() or NO_UPDATES in line.lower() for line in lines):
        return None
    for i, line in enumerate(lines):
        lowered = line.lower()
        if lowered.startswith(HEADING_STARTS) or NO_UPDATES in lowered:
            j = i - 1
            while j >= 0 and lines[j].lower().startswith(HEADING_STARTS):
                j -= 1
            if j < 0 or lines[j].lower().startswith("car presentation"):
                return None
            return lines[j].strip("* ").strip()
    return None


def _column_edges(words):
    """x-position where each column starts, from the header words."""
    starts = {}
    for name, anchor in CAR_COLUMNS:
        hits = [w for w in words if w["text"].lower() == anchor]
        if not hits:
            return None, None
        starts[name] = min(hits, key=lambda w: w["top"])
    # headings wrap onto a second line ("component", "for update", "version")
    second = [w for w in words if w["text"].lower() in ("component", "version", "works")
              and 0 < w["top"] - starts["component"]["top"] < 20]
    header_bottom = max(w["bottom"] for w in [*starts.values(), *second]) + 1
    # cell text can start left of its heading; the item number sits further left
    return {name: w["x0"] - (14 if name == "component" else 10) for name, w in starts.items()}, header_bottom


def _column_of(x, edges):
    column = None
    for name, start in sorted(edges.items(), key=lambda item: item[1]):
        if x >= start:
            column = name
    return column  # None: the item-number column left of the first heading


def _join(words):
    """Group words into lines (tops within 3 pt), then read each line left to
    right. Sorting by rounded top alone put a slightly lower dash ahead of the
    words on its line (2024 Belgium, Alpine)."""
    lines = []
    for w in sorted(words, key=lambda w: w["top"]):
        if lines and abs(w["top"] - lines[-1][0]) <= 3:
            lines[-1][1].append(w)
        else:
            lines.append([w["top"], [w]])
    return " ".join(" ".join(w["text"] for w in sorted(ws, key=lambda w: w["x0"])) for _, ws in lines).strip()


def _centre(word):
    return (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2


def _rows_from_rules(page, words, numbers):
    """Read items using the table's drawn dividers: thin filled rectangles (< 2 pt
    tall), one segment per column. Background strips behind each text line are
    tall rectangles and are ignored, so a row is everything between two
    dividers, per column. Where a column has no divider between two items but
    the number column does, the cell is shared (merged); where no column has
    one, the divider is missing and the rows split halfway between the numbers.
    Returns None unless the dividers show five columns (number + four)."""
    rules = [r for r in page.rects if r["height"] <= 2 and r["width"] > 15]
    if not numbers:
        return None
    # Columns come from the vertical borders. Horizontal dividers are drawn either
    # per column or as one line across several columns (2026 Haas), so a divider
    # bounds every column it crosses.
    verticals = sorted(r["x0"] for r in page.rects if r["width"] <= 2 and r["height"] > 8)
    edges = []
    for x in verticals:
        if not edges or x - edges[-1] > 3:
            edges.append(x)
    if len(edges) == len(CAR_COLUMNS) + 2:
        columns = [[a, b] for a, b in zip(edges, edges[1:])]
    else:
        # no usable vertical borders: one column per group of divider segments
        columns = []
        for rule in sorted(rules, key=lambda r: r["x0"]):
            if columns and abs(rule["x0"] - columns[-1][0]) <= 3:
                columns[-1][1] = min(columns[-1][1], rule["x1"])
            else:
                columns.append([rule["x0"], rule["x1"]])
    if len(columns) != len(CAR_COLUMNS) + 1:
        return None
    boundaries = []
    for x0, x1 in columns:
        middle = (x0 + x1) / 2
        boundaries.append(sorted({round(r["top"], 1) for r in rules if r["x0"] - 2 <= middle <= r["x1"] + 2}))
    centres = sorted(_centre(n)[1] for n in numbers)
    for upper, lower in zip(centres, centres[1:]):
        if not any(upper < y < lower for y in boundaries[0]):  # divider missing in every column
            for ys in boundaries:
                if not any(upper < y < lower for y in ys):
                    ys.append((upper + lower) / 2)
    boundaries = [sorted(ys) for ys in boundaries]
    rows = []
    for number in numbers:
        y = _centre(number)[1]
        record = {}
        for (name, _), (x0, x1), ys in zip(CAR_COLUMNS, columns[1:], boundaries[1:]):
            above, below = [b for b in ys if b <= y], [b for b in ys if b > y]
            if not above or not below:
                return None
            top, bottom = above[-1], below[0]
            record[name] = _join([w for w in words if x0 - 1 <= _centre(w)[0] <= x1 + 1
                                  and top - 1 <= _centre(w)[1] <= bottom + 1])
        rows.append((int(number["text"]), record))
    return rows


def _rows_from_cells(table, words, numbers):
    """Read each item from the table's own cells. The row's height is its
    number cell; each column's text is every word inside that column's cells
    overlapping the row. Cell borders, not headings, bound the columns, so text
    that starts left of its heading stays in its column, and a merged cell
    (one description shared by several items) is given to each item it spans.
    Returns None when the cells don't show the expected four columns, and the
    caller falls back to heading positions."""
    cells = [c for c in table.cells if c[2] - c[0] > 15]  # drop the thin spacer cells
    rows = []
    for number in numbers:
        x, y = _centre(number)
        holders = [c for c in cells if c[0] <= x <= c[2] and c[1] <= y <= c[3]]
        if not holders:
            return None
        band = min(holders, key=lambda c: c[3] - c[1])
        middle = (band[1] + band[3]) / 2
        in_row = [c for c in cells if c[0] >= band[2] - 1 and c[1] <= middle <= c[3]]
        spans = sorted({(round(c[0], 1), round(c[2], 1)) for c in in_row})
        if len(spans) != len(CAR_COLUMNS):
            return None
        record = {}
        for (name, _), (x0, x1) in zip(CAR_COLUMNS, spans):
            column = [c for c in cells if abs(c[0] - x0) <= 1.5 and abs(c[2] - x1) <= 1.5
                      and c[3] > band[1] + 1 and c[1] < band[3] - 1]
            top, bottom = min(c[1] for c in column), max(c[3] for c in column)
            record[name] = _join([w for w in words if x0 - 1 <= _centre(w)[0] <= x1 + 1
                                  and top - 1 <= _centre(w)[1] <= bottom + 1])
        rows.append((int(number["text"]), record))
    return rows


def parse_car_presentation(pdf):
    """Rows are vertically centred in their cells, so a row's band is taken from
    pdfplumber's table rows: a band starts at a table row whose item-number cell
    holds a number and runs until the next such row."""
    updates, team, edges, no_update_teams, unparsed = [], None, None, [], []
    for page_number, page in enumerate(pdf.pages, 1):
        lines = _lines(page)
        if page_number == 1 or not lines:
            continue  # cover page, or a photo page
        # A team page opens with the "Car Presentation - <event>" title and the
        # team on the next line; continuation pages have no title.
        new_team_page = lines[0].lower().startswith("car presentation")
        named = None
        if new_team_page and len(lines) > 1 and not lines[1].lower().startswith(HEADING_STARTS):
            named = lines[1].strip("* ").strip()
        named = named or _team_name(lines)
        if named:
            team = named
        if any(NO_UPDATES in line.lower() for line in lines):
            no_update_teams.append(team)
            continue
        words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
        page_edges, header_bottom = _column_edges(words)
        if page_edges:
            edges = page_edges
        elif new_team_page:
            # headings drawn as an image: use the standard column positions
            edges, header_bottom = dict(DEFAULT_EDGES), 0
        elif edges is None:
            continue
        else:
            header_bottom = 0  # continuation page without headings
        tables = page.find_tables()
        if not tables:
            if new_team_page:
                unparsed.append({"team": team, "page": page_number,
                                 "reason": "team page has no text table (scanned image); read the page and enter it as a correction"})
            continue
        table = max(tables, key=lambda t: (t.bbox[3] - t.bbox[1]) * (t.bbox[2] - t.bbox[0]))
        body = [w for w in words if w["top"] >= header_bottom and table.bbox[1] - 2 <= w["top"] <= table.bbox[3] + 2]
        # pdfplumber lists a merged row and also the stripes inside it; an item
        # starts at the top of the outermost table row holding its number.
        numbers = [w for w in body if _column_of(w["x0"], edges) is None and w["text"].isdigit()]
        starts = []
        for number in sorted(numbers, key=lambda w: w["top"]):
            # rows holding this number but no other item number
            holders = [row.bbox[1] for row in table.rows if row.bbox[1] - 1 <= number["top"] <= row.bbox[3] + 1
                       and not any(other is not number and row.bbox[1] - 1 <= other["top"] <= row.bbox[3] + 1
                                   for other in numbers)]
            top = min(holders) if holders else number["top"] - 4
            if starts and top <= starts[-1][0]:
                top = number["top"] - 4
            starts.append((top, int(number["text"])))
        if not starts and updates and updates[-1]["team"] == team:
            # continuation of the previous page's last row
            last = updates[-1]
            for name, _ in CAR_COLUMNS:
                text = _join([w for w in body if _column_of(w["x0"], edges) == name])
                if text:
                    last[name] = f"{last[name]} {text}".strip()
            continue
        ordered = sorted(numbers, key=lambda w: w["top"])
        for method, reader in (("rules", lambda: _rows_from_rules(page, words, ordered)),
                               ("cells", lambda: _rows_from_cells(table, words, ordered))):
            found = reader()
            if found is not None:
                for item, record in found:
                    updates.append({"team": team, "item": item, "page": page_number, "columns_inferred": False,
                                    "method": method, **record})
                break
        if found is not None:
            continue
        bands = [(top, starts[i + 1][0] if i + 1 < len(starts) else table.bbox[3] + 2, item)
                 for i, (top, item) in enumerate(starts)]
        for top, bottom, item in bands:
            inside = [w for w in body if top - 1 <= w["top"] < bottom - 1]
            record = {"team": team, "item": item, "page": page_number, "columns_inferred": True,
                      "method": "headings"}
            for name, _ in CAR_COLUMNS:
                record[name] = _join([w for w in inside if _column_of(w["x0"], edges) == name])
            updates.append(record)
    if not updates and not no_update_teams:
        raise ExtractionProblem("car presentation: no team pages recognised")
    # a numbered row with no text in any column is empty in the source itself
    # (Red Bull, 2026 Barcelona, item 3); log it rather than fail the document
    empty = [r for r in updates if not any(r[name] for name, _ in CAR_COLUMNS)]
    unparsed += [{"team": r["team"], "page": r["page"], "reason": f"item {r['item']} is empty in the source"}
                 for r in empty]
    updates = [r for r in updates if r not in empty]
    for record in updates:
        if not record["component"] or not record["team"]:
            raise ExtractionProblem(f"car presentation: incomplete row {record}")
        record["primary_reason"] = re.sub(r"\s*-\s*", " - ", record["primary_reason"]).strip()
    return updates, no_update_teams, unparsed


# ---------------------------------------------------------------- Pirelli preview

PRESSURE_ROW = re.compile(r"\b(Front|Rear)\s+(\d{2}(?:\.\d)?)\s*psi\s+(\d{2}(?:\.\d)?)\s*psi\s+(\d{2}(?:\.\d)?)\s*psi")
CAMBER = re.compile(r"Camber limit\s+(-?\d+(?:\.\d+)?)\s*°\s+(-?\d+(?:\.\d+)?)\s*°")


def parse_pirelli(path):
    """The preview is a graphic layout that pdfplumber reads letter-spaced and
    incomplete; PDFium's text layer is clean ("Compound C2 C1 C3", "Front 26.0
    psi 25.0 psi 22.0 psi"). Camber limits are kept as printed (two values; the
    sheet does not label which axle each belongs to in the text layer)."""
    import pypdfium2
    document = pypdfium2.PdfDocument(str(path))
    try:
        text = " ".join(document[i].get_textpage().get_text_range() for i in range(len(document)))
    finally:
        document.close()
    text = re.sub(r"\s+", " ", text)
    # The selection block runs from "Compound" to the pressure chart's axis
    # ("10 12 14 ..."); its labels come in varying order and repeat (Q3 tyre,
    # mandatory race tyres), so the nomination is the set of compounds in it.
    block = re.search(r"Compound\b(.*?)\b10 12 14\b", text)
    if block:
        numbers = sorted({int(n) for n in re.findall(r"\bC([1-6])\b", block.group(1))})
    else:
        # some sheets (2023 Spain) lack the labels in PDFium's text layer, while
        # pdfplumber reads them as lines "C 1", "C 2", "C 3"
        with pdfplumber.open(path) as pdf:
            lines = [line.strip() for page in pdf.pages for line in (page.extract_text() or "").splitlines()]
        numbers = sorted({int(m.group(1)) for line in lines if (m := re.fullmatch(r"C\s*([1-6])", line))})
        if not numbers:
            raise ExtractionProblem("Pirelli preview: compound selection not found")
    if len(numbers) != 3:
        raise ExtractionProblem(f"Pirelli preview: compounds C{numbers} are not three distinct")
    record = {"hard": f"C{numbers[0]}", "medium": f"C{numbers[1]}", "soft": f"C{numbers[2]}"}
    for axle, slick, inter, wet in PRESSURE_ROW.findall(text)[:2]:
        side = axle.lower()
        record.update({f"min_pressure_{side}_slick_psi": float(slick), f"min_pressure_{side}_inter_psi": float(inter),
                       f"min_pressure_{side}_wet_psi": float(wet)})
    if not all(15 <= v <= 35 for k, v in record.items() if k.startswith("min_pressure")):
        raise ExtractionProblem("Pirelli preview: pressure outside 15-35 psi")
    camber = CAMBER.search(text)
    if camber:
        record["camber_limits_deg"] = f"{camber.group(1)},{camber.group(2)}"
    return record


# ---------------------------------------------------------------- driver

def kind_of(entry):
    title = entry["title"].lower()
    if entry["category"] == "pu_elements" and "used per driver" in title:
        return "pu_usage"
    if entry["category"] == "components" and "used per driver" in title:
        return "rnc_usage"
    if entry["category"] == "car_presentation" and "procedure" not in title:
        return "car_updates"
    # pre-season testing previews list every compound, not a race nomination
    if entry["category"] == "tyres" and "pirelli preview" in title and "test" not in entry["event"].lower():
        return "tyre_nominations"
    return None


PROVENANCE = ("season", "event", "event_id", "doc_number", "title", "published_utc", "url", "sha256")
CORRECTIONS = CONFIG_DIR / "fia-corrections.json"


def load_corrections(path=CORRECTIONS):
    """Manual readings of pages without a text layer, keyed by (file, page).
    Kept in the repository: they are human work, unlike the re-downloadable data."""
    if not path.is_file():
        # fail loudly: silently skipping would drop verified manual readings
        raise FileNotFoundError(f"FIA corrections file not found: {path}")
    entries = json.loads(path.read_text(encoding="utf-8")).get("car_presentation", [])
    return {(e["file"], e["page"]): e for e in entries}


def extract(root, seasons=None):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["files"]
    tables = {"pu_usage": [], "rnc_usage": [], "car_updates": [], "car_updates_none": [], "tyre_nominations": []}
    problems, counts, applied = [], {}, 0
    corrections = load_corrections()
    for key, entry in sorted(manifest.items()):
        kind = kind_of(entry)
        if kind is None or (seasons and entry["season"] not in seasons):
            continue
        source = {name: entry.get(name) for name in PROVENANCE} | {"file": key}
        try:
            with pdfplumber.open(root / key) as pdf:
                if kind in ("pu_usage", "rnc_usage"):
                    warnings = []
                    rows, columns = (parse_pu_usage if kind == "pu_usage" else parse_rnc_usage)(pdf, warnings)
                    problems += [{"file": key, "kind": kind, "error": f"warning: {w}"} for w in warnings]
                    for row in rows:
                        for column in columns:
                            tables[kind].append({**source, "page": row["page"], "driver_number": row["driver_number"],
                                                 "car_driver": row["car_driver"], "element": column,
                                                 "count": row[column]})
                elif kind == "car_updates":
                    rows, none, unparsed = parse_car_presentation(pdf)
                    for u in unparsed:
                        fix = corrections.get((key, u["page"]))
                        if fix is None:
                            problems.append({"file": key, "kind": kind, "team": u["team"], "page": u["page"],
                                             "error": u["reason"]})
                            continue
                        applied += 1
                        if fix["status"] == "no_updates":
                            none.append(u["team"])
                        else:
                            rows += [{"team": u["team"], "page": u["page"], "columns_inferred": False,
                                      "method": "manual", **update} for update in fix["updates"]]
                    unknown = {r["team"] for r in rows if team_id(r["team"]) is None} | \
                              {team for team in none if team_id(team) is None}
                    if unknown:
                        raise ExtractionProblem(f"car presentation: unrecognised team names {sorted(map(str, unknown))}")
                    tables["car_updates"] += [{**source, "team_id": team_id(row["team"]), **row} for row in rows]
                    tables["car_updates_none"] += [{**source, "team_id": team_id(team), "team": team} for team in none]
            if kind == "tyre_nominations":
                tables["tyre_nominations"].append({**source, **parse_pirelli(root / key)})
            counts[kind] = counts.get(kind, 0) + 1
        except (ExtractionProblem, ValueError, KeyError) as error:
            problems.append({"file": key, "kind": kind, "error": str(error)})
            print(f"PROBLEM {key}: {error}", flush=True)
    written = {}
    for name, rows in tables.items():
        if rows:
            frame = pd.DataFrame(rows)
            if name in ("pu_usage", "rnc_usage"):
                # Counts are cumulative, so a drop means the FIA corrected an earlier
                # sheet (e.g. Norris EX 3 -> 2, France 2022). Values stay as published.
                frame = frame.sort_values(["published_utc", "doc_number"], kind="stable")
                previous = frame.groupby(["season", "driver_number", "car_driver", "element"])["count"].shift()
                frame["below_previous_sheet"] = frame["count"] < previous
            frame = frame.astype({c: "string" for c in frame.columns if frame[c].dtype == object})
            written[name] = write_table(frame, root / "tables" / f"{name}.parquet", {})
    atomic_json(root / "tables" / "extract-problems.json",
                {"extracted_at_utc": utcnow(), "documents_parsed": counts, "corrections_applied": applied, "problems": problems})
    return written, counts, problems
