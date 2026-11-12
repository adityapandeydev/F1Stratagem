import pandas as pd

from f1data.labels import problem_candidates, retirements


def rows():
    return pd.DataFrame({
        "session": ["Race"] * 5,
        "car": ["1", "2", "3", "4", "5"],
        "grid": [1.0, 3.0, 12.0, 5.0, 2.0],
        "finish_position": [1.0, 14.0, 13.0, 19.0, 20.0],
        "laps_completed": [57, 57, 56, 40, 0],
        "race_laps": [57] * 5,
        "recorded_status": ["Finished", "Finished", "Lapped", "Retired", "Did not start"],
        "classified": ["1", "14", "13", "R", "W"],
    })


def test_retirements_exclude_classified_and_non_starters():
    assert retirements(rows()).car.tolist() == ["4"]


def test_problem_candidates_flag_unexpected_results_only():
    flagged = problem_candidates(rows()).set_index("car").why_flagged
    assert set(flagged.index) == {"2"}
    assert "outside the points" in flagged["2"] and "lost 11 places" in flagged["2"]


def test_import_keeps_reviewed_rows_and_rejects_unknown_causes(tmp_path):
    import json
    import pytest
    from openpyxl import Workbook
    from f1data.labels import import_workbook
    (tmp_path / "reference").mkdir()
    pd.DataFrame({"season": [2023], "round": [1], "session_type": ["R"], "session_id": ["2023-01-5"]}).to_parquet(
        tmp_path / "reference" / "sessions.parquet")
    book = Workbook()
    retired = book.active
    retired.title = "Retirements"
    retired.append(["season", "round", "session", "car", "driver", "cause", "other_driver", "notes", "source_url",
                    "confidence", "verified"])
    retired.append([2023, 1, "Race", "18", "STR", "mechanical", None, "rear wing broke", "https://x", "high", "yes"])
    retired.append([2023, 1, "Race", "2", "SAR", "driver_error", None, None, None, "low", None])   # not reviewed
    problems = book.create_sheet("Finished with problems")
    problems.append(["season", "round", "session", "car", "driver", "had_problem", "description", "source_url",
                     "confidence", "verified"])
    workbook, labels = tmp_path / "w.xlsx", tmp_path / "labels.json"
    book.save(workbook)
    assert import_workbook(tmp_path, workbook, labels) == {"retirements": 1, "problems": 0}
    stored = json.loads(labels.read_text())["retirements"]
    assert stored["2023-01-5/18"]["cause"] == "mechanical" and "2023-01-5/2" not in stored
    retired.cell(row=2, column=6, value="aliens")
    book.save(workbook)
    with pytest.raises(ValueError, match="unknown cause"):
        import_workbook(tmp_path, workbook, labels)
