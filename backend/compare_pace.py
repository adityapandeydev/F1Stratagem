import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

REPORT_FILE = "race_pace_report.txt"

def load_team_report(team, year):
    header = f"{team} Race Pace {year} Season"

    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the block for the team
    match = re.search(rf"{header}(.+?)(?=\w+ Race Pace|\Z)", content, re.S)
    if not match:
        raise ValueError(f"No data found for {team} {year}")

    block = match.group(1)

    races = []
    paces = []

    for line in block.split("\n"):
        if ":" not in line or "Season Average" in line:
            continue

        race, pace = line.split(":")
        race = race.replace("-", "").strip()

        if "N/A" in pace:
            continue

        pace_val = float(pace.replace("s", "").strip())

        races.append(race)
        paces.append(pace_val)

    return pd.DataFrame({
        "Race": races,
        "Pace": paces,
        "Team": team
    })


def compare_teams(year, team1, team2):
    df1 = load_team_report(team1, year)
    df2 = load_team_report(team2, year)

    df = pd.concat([df1, df2])

    sns.set(style="whitegrid")

    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=df,
        x="Race",
        y="Pace",
        hue="Team",
        marker="o"
    )

    plt.xticks(rotation=45)
    plt.title(f"Race Pace Comparison - {year} Season")
    plt.tight_layout()
    plt.show()


# Example usage
compare_teams(2021, "Mercedes", "Red Bull Racing")
