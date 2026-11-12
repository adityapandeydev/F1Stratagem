"""Workbooks for hand-labelling what the sources do not record.

From 2023 the timing sources record almost every retirement only as "Retired".
`f1data labels export` writes a workbook with one row per race and sprint
retirement (and a second sheet of finishers whose result suggests a problem),
prefilled with everything the archive knows. A person fills in the cause from
race reports; `f1data labels import` reads the reviewed rows into
`config/race-incident-labels.json`, version-controlled like the FIA corrections.
"""

import json

import pandas as pd
from openpyxl import load_workbook
from openpyxl import Workbook

from .paths import CONFIG_DIR
from .storage import atomic_json, utcnow
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

CAUSES = {
    "mechanical": "car or power-unit failure (including a failure that forced a retirement after damage-free running)",
    "driver_error": "solo crash, spin or off by this driver",
    "collision_at_fault": "this driver caused a collision",
    "collision_victim": "this driver was hit by another driver",
    "collision_racing_incident": "contact with no clear fault",
    "damage": "puncture or debris damage not from contact",
    "team_error": "pit stop, unsafe release, fuel or strategy error",
    "withdrawn_by_team": "retired by the team as a precaution or to save components",
    "illness_or_other": "illness or anything else",
    "unknown": "no reliable source found",
}
PROBLEM_ANSWERS = ("yes_mechanical", "yes_damage", "no", "unknown")
FINISHED = r"^(Finished|\+\d+ Laps?|Lapped)$"
LABEL_FILL = PatternFill("solid", fgColor="FFF2CC")
LABELS_PATH = CONFIG_DIR / "race-incident-labels.json"
SESSION_CODES = {"Race": "R", "Sprint": "S"}


def race_rows(root, seasons):
    """Every race and sprint entrant of `seasons` with grid, laps and recorded status."""
    reference = root / "reference"
    sessions = pd.read_parquet(reference / "sessions.parquet")
    events = pd.read_parquet(reference / "events.parquet")[["event_key", "event_name"]]
    entrants = pd.read_parquet(reference / "entrants.parquet")
    sessions = sessions[sessions.session_type.isin(["R", "S"]) & sessions.season.isin(seasons)]
    frames = []
    for session in sessions.itertuples():
        results = pd.read_parquet(root / "sessions" / str(session.season) / session.session_id / "normalized"
                                  / "results.parquet")
        frame = pd.DataFrame({
            "session_id": session.session_id, "event_key": session.event_key, "season": session.season,
            "round": session.round, "session": "Race" if session.session_type == "R" else "Sprint",
            "car": results.DriverNumber.astype(str), "driver": results.Abbreviation,
            "grid": results.GridPosition, "laps_completed": results.Laps, "race_laps": results.Laps.max(),
            "recorded_status": results.Status, "classified": results.ClassifiedPosition.astype(str),
            "finish_position": results.Position,
        })
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True).merge(events, on="event_key", how="left")
    ids = entrants[["session_id", "driver_number", "driver_id", "team_name", "team_id"]].rename(
        columns={"driver_number": "car"})
    return rows.merge(ids, on=["session_id", "car"], how="left").sort_values(["season", "round", "session_id",
                                                                             "finish_position"])


def retirements(rows):
    excluded = rows.classified.isin(["D", "E", "F", "W"]) | rows.recorded_status.isin(
        ["Did not start", "Withdrew", "Disqualified"])
    return rows[~rows.classified.str.isdigit() & ~excluded]


def problem_candidates(rows):
    """Classified finishers whose result suggests a problem: started in the top 10 and
    finished outside the points, lost 8+ places from the grid, or finished 2+ laps down
    after starting in the top 12. Candidates only; confirmed from race reports."""
    finished = rows[rows.classified.str.isdigit() & rows.recorded_status.str.match(FINISHED)].copy()
    grid = finished.grid.where(finished.grid > 0)
    laps_down = finished.race_laps - finished.laps_completed
    paying = finished.session.map({"Race": 10, "Sprint": 8})
    reasons = []
    for g, pos, down, pay in zip(grid, finished.finish_position, laps_down, paying):
        why = []
        if g <= 10 and pos > pay:
            why.append(f"started P{g:.0f}, finished outside the points")
        if g == g and pos - g >= 8:
            why.append(f"lost {pos - g:.0f} places")
        if g <= 12 and down >= 2:
            why.append(f"{down:.0f} laps down")
        reasons.append("; ".join(why))
    finished["why_flagged"] = reasons
    return finished[finished.why_flagged != ""]


def _sheet(book, title, frame, columns, label_columns, validation=None, first=False):
    sheet = book.active if first else book.create_sheet()
    sheet.title = title
    headers = [header for header, _ in columns] + label_columns
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for record in frame.itertuples(index=False):
        sheet.append([None if pd.isna(getattr(record, field)) else getattr(record, field) for _, field in columns]
                     + [None] * len(label_columns))
    first_label = len(columns) + 1
    for row in sheet.iter_rows(min_row=2, min_col=first_label, max_col=len(headers)):
        for cell in row:
            cell.fill = LABEL_FILL
    if validation is not None and len(frame):
        column = sheet.cell(row=1, column=first_label).column_letter
        validation.add(f"{column}2:{column}{len(frame) + 1}")
        sheet.add_data_validation(validation)
    sheet.freeze_panes = sheet.cell(row=2, column=first_label)
    for index, header in enumerate(headers, start=1):
        sheet.column_dimensions[sheet.cell(row=1, column=index).column_letter].width = max(10, min(40, len(header) + 4))
    return sheet


def export(root, seasons, path):
    rows = race_rows(root, seasons)
    retired = retirements(rows)
    candidates = problem_candidates(rows)
    common = [("season", "season"), ("round", "round"), ("event", "event_name"), ("session", "session"),
              ("car", "car"), ("driver", "driver"), ("driver_id", "driver_id"), ("team", "team_name"),
              ("grid", "grid"), ("laps_completed", "laps_completed"), ("race_laps", "race_laps"),
              ("recorded_status", "recorded_status")]
    book = Workbook()
    causes = DataValidation(type="list", formula1='"' + ",".join(CAUSES) + '"', allow_blank=True)
    _sheet(book, "Retirements", retired, common, ["cause", "other_driver", "notes", "source_url"], causes, first=True)
    answers = DataValidation(type="list", formula1='"' + ",".join(PROBLEM_ANSWERS) + '"', allow_blank=True)
    _sheet(book, "Finished with problems", candidates,
           common + [("finish_position", "finish_position"), ("why_flagged", "why_flagged")],
           ["had_problem", "description", "source_url"], answers)
    guide = book.create_sheet("Instructions")
    guide.append(["Fill the yellow columns. One row per entrant; leave a cell empty if unsure."])
    guide.append([])
    guide.append(["Retirements: cause", "meaning"])
    for cause, meaning in CAUSES.items():
        guide.append([cause, meaning])
    guide.append([])
    guide.append(["Retirements: other_driver", "for collisions, the abbreviation of the other driver (e.g. VER)"])
    guide.append([])
    guide.append(["Finished with problems: had_problem", "yes_mechanical, yes_damage, no or unknown"])
    guide.append(["", "rows were flagged from the data only; most will be 'no'"])
    guide.column_dimensions["A"].width = 38
    guide.column_dimensions["B"].width = 90
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    return {"retirements": len(retired), "candidates": len(candidates),
            "retirements_by_season": retired.groupby("season").size().to_dict(), "path": str(path)}


def _reviewed_rows(sheet):
    rows = list(sheet.iter_rows(values_only=True))
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    return frame[frame.verified.astype(str).str.strip().str.lower() == "yes"]


def import_workbook(root, workbook, path=LABELS_PATH):
    """Merge the reviewed rows of a filled workbook into the labels file.

    Rows are keyed by session ID and car number; a re-import replaces earlier
    labels for the same rows and keeps everything else."""
    sessions = pd.read_parquet(root / "reference" / "sessions.parquet")
    session_ids = {(int(r.season), int(r.round), r.session_type): r.session_id
                   for r in sessions[sessions.session_type.isin(["R", "S"])].itertuples()}
    book = load_workbook(workbook, read_only=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"retirements": {}, "problems": {}}

    def key(row):
        session_id = session_ids[(int(row.season), int(row.round), SESSION_CODES[row.session])]
        return f"{session_id}/{row.car}", session_id

    def text(value):
        return None if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)

    counts = {}
    for sheet, section, fields in (("Retirements", "retirements", ("cause", "other_driver", "notes")),
                                   ("Finished with problems", "problems", ("had_problem", "description"))):
        reviewed = _reviewed_rows(book[sheet])
        for row in reviewed.itertuples():
            label_key, session_id = key(row)
            existing[section][label_key] = {
                "session_id": session_id, "driver_number": str(row.car), "driver": row.driver,
                **{field: text(getattr(row, field)) for field in fields},
                "source_url": text(row.source_url), "confidence": text(row.confidence),
            }
        counts[section] = len(reviewed)
    allowed = {"retirements": ("cause", CAUSES), "problems": ("had_problem", PROBLEM_ANSWERS)}
    for section, (field, values) in allowed.items():
        bad = {k: v[field] for k, v in existing[section].items() if v[field] not in values}
        if bad:
            raise ValueError(f"{section}: unknown {field} values {bad}")
    existing.update({"schema_version": 1, "updated_at_utc": utcnow(),
                     "note": "Hand-reviewed causes of race and sprint retirements and of poor finishes, "
                             "from race reports. Target labels only; never a pre-race feature."})
    atomic_json(path, existing)
    return counts


def load_labels(path=LABELS_PATH):
    """Reviewed labels as {"retirements": {...}, "problems": {...}}, keyed "<session_id>/<car>"."""
    if not path.is_file():
        return {"retirements": {}, "problems": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def register(commands, common):
    parser = commands.add_parser("labels", parents=[common], help="workbooks for hand-labelled data",
                                 description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["export", "import"])
    parser.add_argument("--seasons", type=int, nargs="+", help="export: seasons to include")
    parser.add_argument("--workbook", help="import: the filled workbook")
    parser.set_defaults(handler=run, parser=parser)


def run(args):
    if args.command == "import":
        if not args.workbook:
            args.parser.error("import needs --workbook")
        from pathlib import Path
        print(json.dumps(import_workbook(args.data_root, Path(args.workbook)), indent=2))
        return 0
    if not args.seasons:
        args.parser.error("export needs --seasons")
    seasons = sorted(args.seasons)
    path = args.data_root / "labels" / f"retirements-{seasons[0]}-{seasons[-1]}.xlsx"
    print(json.dumps(export(args.data_root, seasons, path), indent=2, default=str))
    return 0
