"""The FIA's procedure for turning a qualifying order and grid penalties into a grid.

From the Formula 1 Sporting Regulations (2020-2024 issues), checked against the
FIA provisional starting grids of 2020-2024 (2019's are images):

- **Empty grid** (2024 Art. 42.3; written into the rules on 19 October 2022, but
  the grids show it in use before): (a) each penalised driver gets the temporary
  slot qualifying position + places; drivers sharing one are ordered by
  qualifying, the slowest keeping it and the others taking the slots just ahead;
  (b) unpenalised drivers fill the empty slots in qualifying order; (c) penalised
  drivers move up into any gaps; (d) drivers sent to the back go behind, in their
  qualifying order, however many places they had (every season 2019-2024; 2018
  used the order of the offences).
- **More than 15 places.** From 19 October 2022 (WMSC, effective from the 2022
  United States GP), more than 15 cumulative places is the back of the grid
  (Art. 42.3 d). Before, only a penalty itself "exceeding fifteen grid places"
  (a power unit penalty) sent a driver back, by its own decision; drops adding up
  past 15 stayed drops (Bottas, Belgium 2022: 15 + 5 places, last of the
  droppers, ahead of the six sent to the back).
- **Pit lane.** Drivers required to start from the pit lane start behind the
  grid (in qualifying order here).

The 2019-2022 texts say penalties are applied "in the order the offences were
committed" (2022: higher qualifier first). Dropping drivers one after another in
that order matches as many 2020-2021 grids as the empty grid (36 of 38) and fewer
in 2022 (14 of 18 against 15) and after (40 of 49 against 44), so the empty grid
is used throughout. Exact provisional grids from qualifying and the decisions
published before them: 94 of 105 (89.5%; 54 with qualifying order alone), the
rest explained by decisions published late or missing, qualifying
disqualifications and drivers without a time.

The procedure puts the unpenalised drivers in qualifying order around the
penalised ones, so where everyone ends up depends only on the penalised drivers'
qualifying slots: `allocate` works on those and is cached.
"""
import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd

BACK_OF_GRID_ABOVE = 15                                   # cumulative places; more is the back (from 2022-10-19)
EMPTY_GRID_FROM = pd.Timestamp("2022-10-19", tz="UTC")   # WMSC decision, effective immediately


@dataclass(frozen=True)
class GridPenalty:
    places: int = 0          # grid places, summed over decisions
    back: bool = False       # sent to the back of the grid
    pit_lane: bool = False   # required to start from the pit lane
    first: int = 0           # rank of the driver's first decision (kept for the record)


def procedure(season, when):
    """The procedure in force at `when` (UTC): "art42" (more than 15 cumulative places
    is the back of the grid) from 19 October 2022, "drops" before (only decisions
    sending a driver to the back do)."""
    return "art42" if when >= EMPTY_GRID_FROM else "drops"


@functools.lru_cache(maxsize=100_000)
def allocate(n, entries, rule):
    """Final grid slot of every qualifying slot 1..n (array index slot - 1) under
    `rule` (`procedure`), for `entries`: ((qualifying slot, places, back, pit lane),
    ...) of the penalised drivers."""
    penalised = {e[0]: e for e in entries}
    pit = sorted(q for q, e in penalised.items() if e[3])
    back = sorted(q for q, e in penalised.items()
                  if not e[3] and (e[2] or (rule == "art42" and e[1] > BACK_OF_GRID_ABOVE)))
    drops = {q: e[1] for q, e in penalised.items() if q not in pit and q not in back and e[1] > 0}
    order = _empty_grid(n, drops, set(pit) | set(back)) + back + pit
    final = np.empty(n, dtype=int)
    final[np.array(order) - 1] = np.arange(1, n + 1)
    return final


def _empty_grid(n, drops, behind):
    """Qualifying slots in grid order of the drivers not in `behind`, by Art. 42.3 a-c;
    `drops`: {qualifying slot: places}."""
    taken = {}
    # (a) temporary slots; the slowest of those sharing a slot keeps it, the others go
    # just ahead (to the nearest free slot ahead, should that one be taken too)
    for q in sorted(drops, key=lambda q: (q + drops[q], q), reverse=True):
        slot = q + drops[q]
        while slot in taken and slot > 1:
            slot -= 1
        while slot in taken:
            slot += 1
        taken[slot] = q
    # (b) unpenalised drivers fill the free slots in qualifying order
    free = (s for s in range(1, 2 * n + 2) if s not in taken)
    for q in range(1, n + 1):
        if q not in drops and q not in behind:
            taken[next(free)] = q
    # (c) penalised drivers move up into any gaps: the occupied slots in order
    return [taken[s] for s in sorted(taken)]


def apply_grid_penalties(slots, driver_numbers, penalties, rule):
    """Grid slots (draws x entrants, 1 = pole) from qualifying slots after
    `penalties` ({driver number: GridPenalty}) under `rule` (`procedure`)."""
    slots = np.atleast_2d(slots)
    out = slots.astype(float).copy()
    columns = [(i, penalties[str(d)]) for i, d in enumerate(driver_numbers) if str(d) in penalties]
    if not columns:
        return out
    n = slots.shape[1]
    q = np.rint(slots).astype(int)
    for row in range(len(q)):
        entries = tuple(sorted((int(q[row, i]), p.places, p.back, p.pit_lane) for i, p in columns))
        out[row] = allocate(n, entries, rule)[q[row] - 1]
    return out
