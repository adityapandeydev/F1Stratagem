import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

REPORT_FILE = "race_pace_report.txt"

def load_report(team, year):
    header = f"{team} Race Pace {year} Season"

    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_lines = []
    inside_block = False

    for line in lines:
        if header in line:
            inside_block = True
            continue

        # next header encountered → stop
        if inside_block and "Race Pace" in line:
            break

        if inside_block:
            data_lines.append(line.strip())

    # Convert bullet lines → tidy DataFrame
    races = []
    paces = []

    for ln in data_lines:
        if ln.startswith("-"):
            ln = ln[2:]   # remove '- '

            name, value = ln.split(":")

            name = name.strip()
            value = value.strip()

            if value == "N/A":
                continue  # skip cancelled races

            if "Full Season" in name:
                continue  # ignore season avg row

            races.append(name)
            paces.append(float(value.replace(" s", "")))

    df = pd.DataFrame({"Race": races, "Pace": paces})
    return df


def plot_race_pace(team, year):
    df = load_report(team, year)

    if df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=df,
        x="Race",
        y="Pace",
        marker="o"
    )

    plt.title(f"{team} Race Pace Trend ({year})", fontsize=15)
    plt.xlabel("Race")
    plt.ylabel("Avg Pace (s)")
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    plt.show()