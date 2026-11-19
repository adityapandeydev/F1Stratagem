"""Pirelli's dry compounds per Grand Prix (`tyre_compounds.csv`, 2018-2024).

A race's soft, medium and hard are labels for three compounds Pirelli nominates
from its range: the C3 can be the hard at one track and the soft at another. A
compound's identity is its place in the range of its tyre generation:
- "2018": superhard (hardest) to hypersoft (softest), seven compounds;
- "13in": C1 (hardest) to C5, 13-inch tyres 2019-2021;
- "18in": C0/C1 (hardest) to C5, 18-inch tyres from 2022.
A compound with the same name in two generations is a different tyre, and a name
can change tyre within a generation: Pirelli's 2023 notice says "Last year's C1 tyre
has become the C0 for 2023", with a new C1 between the old C1 and the C2; its 2024
notice omits the C0 ("All five compounds homologated for 2024 - with the C0 now
having been omitted"). `spec` follows the tyre, not the name: the 2022 C1 and the
2023 C0 are one specification, the 2023-2024 C1 another. C2-C5 of 2022-2024 are
taken as one specification each (Pirelli says the C2 remains; it states no change
to C3-C5), and so are C1-C5 of 2019-2021 (assumed: no documented renaming).

Sources (per row): the FIA event notes with Pirelli's preview (2022-2024 where
published), Pirelli's press previews, race articles and news reports; rows from a
single news report are marked so. Nothing after 2024 is in the table."""
import functools
from pathlib import Path

import pandas as pd

TABLE = Path(__file__).with_name("tyre_compounds.csv")
RANGE_2018 = ("superhard", "hard", "medium", "soft", "supersoft", "ultrasoft", "hypersoft")
LABELS = ("HARD", "MEDIUM", "SOFT")


@functools.lru_cache(maxsize=1)
def compounds():
    """The nominations: event_key, season, round, event_name, hard, medium, soft, source."""
    return pd.read_csv(TABLE)


def generation(season):
    return "2018" if season <= 2018 else "13in" if season <= 2021 else "18in"


def softness(name):
    """Place in the range, 0 = hardest (C0 is 0, C1 1 ...; 2018 names 0-6)."""
    name = str(name)
    return RANGE_2018.index(name) if name in RANGE_2018 else int(name[1:])


def spec(season, name):
    """A stable identifier of the tyre a compound name stood for in `season`."""
    season, name = int(season), str(name)
    gen = generation(season)
    if gen == "18in" and ((season == 2022 and name == "C1") or (season >= 2023 and name == "C0")):
        return "18in-C1-2022"          # the 2022 C1, renamed C0 for 2023
    if gen == "18in" and season >= 2023 and name == "C1":
        return "18in-C1-2023"          # new for 2023
    return f"{gen}-{name}"


def identity(event_key, label):
    """(generation, compound name, softness, specification) of a race's
    "HARD"/"MEDIUM"/"SOFT" label; None for another label or an event outside the
    table."""
    label = str(label).upper()
    table = compounds()
    row = table[table.event_key == event_key]
    if label not in LABELS or row.empty:
        return None
    row = row.iloc[0]
    name = row[label.lower()]
    return generation(int(row.season)), name, softness(name), spec(int(row.season), name)
