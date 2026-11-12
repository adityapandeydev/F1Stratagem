import pytest

from f1data import fia_extract as fx

PU_LINES = """Technical Delegate's Report
N° Car Driver ICE TC MGU-H MGU-K ES CE EX
01 RBR Honda RBPT Max Verstappen 4 4 4 4 2 2 5
63 Mercedes George Russell 2 2 2 2 1 1 2
27 Haas Ferrari Nico Hülkenberg 2 2 2 2 2 2 3""".splitlines()


def test_usage_rows_split_number_name_and_counts():
    columns = ["ICE", "TC", "MGU-H", "MGU-K", "ES", "CE", "EX"]
    rows = fx.usage_rows(PU_LINES, columns)
    assert rows[0] == {"driver_number": 1, "car_driver": "RBR Honda RBPT Max Verstappen", "ICE": 4, "TC": 4,
                       "MGU-H": 4, "MGU-K": 4, "ES": 2, "CE": 2, "EX": 5}
    assert [r["driver_number"] for r in rows] == [1, 63, 27]


def test_usage_check_rejects_partial_tables():
    rows = fx.usage_rows(PU_LINES, ["ICE", "TC", "MGU-H", "MGU-K", "ES", "CE", "EX"])
    with pytest.raises(fx.ExtractionProblem, match="3 driver rows"):
        fx.check_usage(rows, ["ICE"], "PU usage")


@pytest.mark.parametrize("name, team", [
    ("ORACLE RED BULL RACING", "red_bull"), ("Red Bull Racing", "red_bull"),
    ("VISA CASH APP RB FORMULA ONE TEAM", "rb"), ("*Mercedes-AMG PETRONAS F1 Team*", "mercedes"),
    ("Stake F1 Team KICK Sauber", "sauber"), ("MONEYGRAM HAAS F1 TEAM", "haas"),
    ("ASTON MARTIN ARAMCO COGNIZANT F1 TEAM", "aston_martin"), ("Primary reason", None),
])
def test_team_ids(name, team):
    assert fx.team_id(name) == team


def test_team_line_sits_above_headings_in_any_order():
    lines = ["Car Presentation – Japanese Grand Prix", "McLaren Formula 1 Team",
             "Primary reason Geometric differences compared to previous", "Updated Brief description", "1 Floor Body ..."]
    assert fx._team_name(lines) == "McLaren Formula 1 Team"
    assert fx._team_name(["the rear corner allows an increased expansion,"]) is None
    # a description line starting "Updated" on a continuation page is not a heading
    assert fx._team_name(["The bib geometry has been updated", "Performance -", "Updated bib and keel"]) is None
    assert fx._team_name(["Car Presentation – X", "WILLIAMS RACING", "No updates submitted for this event."]) \
        == "WILLIAMS RACING"


def test_column_assignment_keeps_item_numbers_separate():
    edges = dict(fx.DEFAULT_EDGES)
    assert fx._column_of(84, edges) is None          # item number
    assert fx._column_of(106, edges) == "component"  # text starting left of its heading
    assert fx._column_of(196, edges) == "primary_reason"
    assert fx._column_of(531, edges) == "description"


def test_pirelli_patterns():
    text = ("Compound C2 C1 C3 10 12 Slicks Intermediate Wet Front 26.0 psi 25.0 psi 22.0 psi "
            "Rear 22.5 psi 23.5 psi 20.5 psi Camber limit -1.5° -2.75°")
    assert fx.PRESSURE_ROW.findall(text) == [("Front", "26.0", "25.0", "22.0"), ("Rear", "22.5", "23.5", "20.5")]
    assert fx.CAMBER.search(text).groups() == ("-1.5", "-2.75")


def test_kind_of_skips_procedures_and_testing():
    base = {"category": "car_presentation", "event": "Saudi Arabian Grand Prix"}
    assert fx.kind_of({**base, "title": "Car Presentation Submissions"}) == "car_updates"
    assert fx.kind_of({**base, "title": "Car Presentation and Display Procedure"}) is None
    tyres = {"category": "tyres", "title": "Event Notes - Pirelli Preview"}
    assert fx.kind_of({**tyres, "event": "Bahrain Grand Prix"}) == "tyre_nominations"
    assert fx.kind_of({**tyres, "event": "Bahrain Tests Season"}) is None


class FakeTable:
    def __init__(self, cells):
        self.cells = cells


def word(text, x0, top, width=20):
    return {"text": text, "x0": x0, "x1": x0 + width, "top": top, "bottom": top + 8}


def test_cells_keep_overflowing_words_in_their_column_and_share_merged_cells():
    """Columns are bounded by cell borders, not headings: 'Revised' starts left of the
    geometry heading but inside the geometry cell. Items 2 and 3 share one merged
    description cell (as on Ferrari's 2024 Las Vegas sheet)."""
    cells = [
        (73, 176, 100, 234), (100, 176, 179, 234), (179, 176, 277, 234), (277, 176, 526, 234),
        (531, 176, 764, 205), (531, 205, 764, 234),                      # description stripes, item 1
        (73, 234, 100, 275), (100, 234, 179, 275), (179, 234, 277, 275), (277, 234, 526, 275),
        (73, 275, 100, 315), (100, 275, 179, 315), (179, 275, 277, 315), (277, 275, 526, 315),
        (526, 234, 769, 315),                                            # merged description, items 2-3
        (100, 176, 106, 234),                                            # thin spacer, ignored
    ]
    words = [word("1", 84, 200, 5), word("Front", 110, 200), word("Performance", 185, 195, 50),
             word("Revised", 282, 195, 30), word("wing", 290, 210), word("Own", 540, 185), word("text", 540, 215),
             word("2", 84, 250, 5), word("Floor", 110, 250), word("Flow", 185, 250), word("Fences", 290, 250),
             word("Shared", 540, 280),
             word("3", 84, 290, 5), word("Edge", 110, 290), word("Flow", 185, 290), word("Floor", 290, 290)]
    numbers = [w for w in words if w["text"] in ("1", "2", "3")]
    rows = dict(fx._rows_from_cells(FakeTable(cells), words, numbers))
    assert rows[1] == {"component": "Front", "primary_reason": "Performance", "geometric_difference": "Revised wing",
                       "description": "Own text"}
    assert rows[2]["description"] == rows[3]["description"] == "Shared"
    assert rows[3]["geometric_difference"] == "Floor"


def test_cells_without_four_columns_fall_back():
    cells = [(73, 176, 100, 234), (100, 176, 526, 234)]
    assert fx._rows_from_cells(FakeTable(cells), [word("1", 84, 200, 5)], [word("1", 84, 200, 5)]) is None


class FakePage:
    def __init__(self, rects):
        self.rects = rects


def rule(top, x0, x1):
    return {"top": top, "height": 0.7, "x0": x0, "x1": x1, "width": x1 - x0}


COLUMNS = [(73, 103), (103, 172), (172, 267), (267, 521), (521, 770)]


def test_rules_split_rows_and_handle_merged_and_missing_dividers():
    """Item 1 spans several text lines (strips are ignored: only thin rules count).
    Items 2-3: the description column has no divider between them but the number
    column does -> shared cell. Items 3-4: no divider in any column (as on Haas's
    2024 China sheet) -> split halfway between the item numbers."""
    rects = [rule(y, x0, x1) for y in (154, 185, 230, 360) for x0, x1 in COLUMNS]
    rects += [rule(274, x0, x1) for x0, x1 in COLUMNS[:4]]           # no divider in description column
    rects += [{"top": 190, "height": 14, "x0": 521, "x1": 770, "width": 249}]  # background strip
    words = [word("1", 84, 203, 5), word("Floor", 110, 203), word("Flow", 180, 203), word("Improved", 280, 203),
             word("Line1", 530, 189), word("Line2", 530, 204), word("Line3", 530, 218),
             word("2", 84, 248, 5), word("Edge", 110, 248), word("Load", 180, 248), word("Shared", 530, 240),
             word("3", 84, 292, 5), word("Cover", 110, 292), word("Drag", 180, 292), word("Slim", 280, 292),
             word("Own3", 530, 285),
             word("4", 84, 330, 5), word("Rear", 110, 330), word("Flow", 180, 330), word("Own4", 530, 335)]
    numbers = [w for w in words if w["text"] in "1234" and w["x0"] < 100]
    rows = dict(fx._rows_from_rules(FakePage(rects), words, numbers))
    assert rows[1]["description"] == "Line1 Line2 Line3"
    assert rows[2]["description"].startswith("Shared")       # merged cell spans items 2 and 3
    assert rows[3]["component"] == "Cover" and rows[4]["component"] == "Rear"
    assert rows[4]["description"] == "Own4"                  # missing divider: split between 3 and 4
    assert "Own4" not in rows[3]["description"]


def test_rules_need_five_columns():
    assert fx._rows_from_rules(FakePage([rule(154, 73, 770)]), [], [word("1", 84, 200, 5)]) is None


def test_rows_without_names_and_wrong_numbers_are_kept():
    # 2019: team and driver wrap above and below the numbers
    assert fx.usage_rows(["07 1 1 1 1 1 1"], ["ICE", "TC", "MGU-H", "MGU-K", "ES", "CE"])[0]["car_driver"] == ""
    # 2025 China prints Hadjar under Lawson's 30: distinct rows, not duplicates
    rows = [{"driver_number": 30, "car_driver": n, "ICE": 1} for n in ("Liam Lawson", "Isack Hadjar")]
    rows += [{"driver_number": i, "car_driver": str(i), "ICE": 1} for i in range(1, 19)]
    fx.check_usage(rows, ["ICE"], "PU usage")


def hw(text, x0, top):
    return {"text": text, "x0": x0, "x1": x0 + 6 * len(text), "top": top, "bottom": top + 8}


def test_pu_headings_stack_across_lines():
    """2026 sheets wrap headings: "MGU" over "-K", "PU-" over "CE"/"ANC", with the
    second line up to 11 pt below "Driver"."""
    words = [hw("Driver", 146, 388), hw("ICE", 310, 388), hw("TC", 341, 388), hw("EXH", 373, 388),
             hw("MGU", 417, 388), hw("ES", 455, 388), hw("PU-", 479, 388), hw("PU-", 509, 388),
             hw("-K", 420, 399), hw("CE", 482, 399), hw("ANC", 509, 399)]
    assert fx.pu_columns(words) == ["ICE", "TC", "EXH", "MGU-K", "ES", "PU-CE", "PU-ANC"]


def test_rows_with_missing_or_garbled_numbers_are_kept_and_short_rows_reported():
    columns = ["ICE", "TC", "EXH", "MGU-K", "ES", "PU-CE", "PU-ANC"]
    lines = ["McLaren Renault Carlos Sainz 6 5 5 5 5 4 4",                 # no number printed
             "i Aston Martin Aramco Honda Lance Stroll 6 7 3 5 8 7 9",      # number printed as "i"
             "31 Haas Ferrari Esteban Ocon 4 4 4 3 3 3",                    # one value missing
             "From The FIA Formula One Technical Delegate Document 17"]
    warnings = []
    rows = fx.usage_rows(lines, columns, warnings)
    assert [(r["driver_number"], r["car_driver"]) for r in rows] == [
        (None, "McLaren Renault Carlos Sainz"), (None, "Aston Martin Aramco Honda Lance Stroll")]
    assert len(warnings) == 1 and "Esteban Ocon" in warnings[0]


def test_corrections_file_is_valid_and_keyed_by_file_and_page():
    fixes = fx.load_corrections()
    assert fixes, "config/fia-corrections.json should exist"
    for (file, page), entry in fixes.items():
        assert file.startswith("pdf/") and page >= 1
        assert entry["status"] in ("no_updates", "updates")
        for update in entry.get("updates", []):
            assert set(update) == {"item", "component", "primary_reason", "geometric_difference", "description"}


def test_join_reads_each_line_left_to_right():
    words = [word("Performance", 185, 204.0, 50), word("-", 245, 205.2, 3), word("Flow", 185, 218.0)]
    assert fx._join(words) == "Performance - Flow"


def test_missing_corrections_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        fx.load_corrections(tmp_path / "absent.json")
