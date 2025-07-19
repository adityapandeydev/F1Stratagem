import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache('cache')

REPORT_FILE = "race_pace_report.txt"

def extract_existing_report(team, year):
    if not os.path.exists(REPORT_FILE):
        return None

    header = f"{team} Race Pace {year} Season"
    block_start = None
    block_end = None

    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if header in line:
            block_start = i
            break

    if block_start is None:
        return None

    for j in range(block_start + 1, len(lines)):
        if "Race Pace" in lines[j]:  
            block_end = j
            break


    if block_end is None:
        block_end = len(lines)

    report_text = "".join(lines[block_start:block_end])
    return report_text


def append_to_report(text):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n\n")  


def get_team_race_pace(year, team):
    schedule = fastf1.get_event_schedule(year)

    schedule = schedule[schedule["EventName"] != "Pre-Season Test"]
    schedule = schedule[schedule["EventFormat"] != "Non-Championship"]

    race_weekends = schedule.reset_index(drop=True)

    team_paces = []

    for _, race in race_weekends.iterrows():
        event_name = race["EventName"]
        print(f"Loading {event_name}...")

        try:
            session = fastf1.get_session(year, event_name, "R")
            session.load()
        except Exception as e:
            print(f"❌ Failed to load {event_name}: {e}")
            continue

        laps = session.laps
        team_laps = laps[laps["Team"] == team]

        if len(team_laps) == 0:
            continue

        clean_laps = team_laps.pick_wo_box()

        if len(clean_laps) == 0:
            continue

        avg_pace = clean_laps["LapTime"].mean()

        if pd.isna(avg_pace):
            pace_sec = None
        else:
            pace_sec = avg_pace.total_seconds()

        team_paces.append((event_name, pace_sec))

    return team_paces

def generate_report(year, team):
    existing = extract_existing_report(team, year)

    if existing:
        print("✔ Loaded existing report from race_pace_report.txt\n")
        print(existing)
        return

    print("⏳ No existing report found — calculating...")

    race_paces = get_team_race_pace(year, team)

    lines = []
    header = f"{team} Race Pace {year} Season"
    lines.append(header)

    valid_paces = []

    for race, pace in race_paces:
        if pace is None:
            lines.append(f"- {race}: N/A")
        else:
            valid_paces.append(pace)
            lines.append(f"- {race}: {pace:.3f} s")


    if valid_paces:
        season_avg = sum(valid_paces) / len(valid_paces)
        lines.append(f"- Season Average: {season_avg:.3f} s")
    else:
        lines.append(f"- Season Average: N/A")


    final_text = "\n".join(lines)

    append_to_report(final_text)

    print("\n✔ Report generated and saved to race_pace_report.txt\n")
    print(final_text)