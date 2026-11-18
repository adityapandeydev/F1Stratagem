"""Retirement model by circuit and team (race model).

Each driver's chance of retiring from a race is an incident chance times the
circuit's incident factor plus a mechanical chance times the team's reliability
factor:

- **Base rates** from the races of the latest RECENT_SEASONS seasons in the
  history (unknown causes shared out in proportion to the known ones).
- **Circuit incident factor.** The circuit's incidents against what its seasons'
  rates expect, over Jolpica's classified results before 2018 (from the
  retirement status: accidents, collisions, spins, damage) and the history's
  labelled races from 2018, shrunk towards 1 by a gamma prior whose spread is
  estimated from the same races. Measured 1996-2022: circuits differ far beyond
  chance (p = 2e-11) and keep their rank between 1996-2009 and 2010-2022
  (correlation 0.78): Monaco about 1.8 times the typical rate, Bahrain 0.5.
- **Team reliability factor.** The lineage's mechanical failures this season
  against the base rate, shrunk towards its previous season's shrunk factor.
  Measured 2014-2022: a team's mechanical rate repeats between odd and even rounds
  of a season (correlation 0.25) and from one season to the next (0.42); the true
  spread between team-seasons is about 0.04 per start around 0.10. A season's
  prior keeps CARRY of last season's deviation from 1.
"""
import numpy as np
import pandas as pd

from .rules import points_positions
from .targets import OUTCOMES, RETIREMENTS

RECENT_SEASONS = 3
TEAM_PRIOR = 4.0        # expected failures: weight of the season's prior factor (spread 0.4 of the rate)
CARRY = 0.42            # share of last season's deviation kept into the next (measured 2014-2022)
FIRST_STATUS_SEASON, LAST_STATUS_SEASON = 1996, 2017


def classify_status(status):
    """Jolpica result status -> "finished", "incident", "mechanical" or None
    (not started, excluded, or no cause given)."""
    text = str(status or "").lower()
    if text.startswith("finished") or text.startswith("+") or text == "lapped":
        return "finished"
    if any(word in text for word in ("not start", "qualify", "withdrew", "excluded", "disqualified", "107")):
        return None
    if any(word in text for word in ("accident", "collision", "spun", "damage", "crash", "debris", "puncture")):
        return "incident"
    if text in ("", "retired"):
        return None
    return "mechanical"


def history_incidents(root):
    """Starts before 2018 from Jolpica's results: DataFrame season, circuit,
    starts, incidents."""
    path = root / "external" / "jolpica" / "tables" / "results.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["season", "circuit", "starts", "incidents"])
    results = pd.read_parquet(path, columns=["season", "round", "circuit_id", "status"])
    results = results[results.season.astype(int).between(FIRST_STATUS_SEASON, LAST_STATUS_SEASON)]
    kind = results.status.map(classify_status)
    results = results.assign(season=results.season.astype(int), kind=kind)[kind.notna()]
    return (results.assign(incident=(results.kind == "incident").astype(int))
            .groupby(["season", "circuit_id"]).agg(starts=("incident", "size"), incidents=("incident", "sum"))
            .reset_index().rename(columns={"circuit_id": "circuit"}))


def _shrunk(obs, exp, k):
    return (obs + k) / (exp + k)


class DetailedRetirements:
    """Model interface of `baselines.RetirementModel`: fit(history), apply(positions,
    example, rng). `circuits`: event key -> circuit; `older`: `history_incidents`."""

    def __init__(self, circuits=None, older=None):
        self.circuits = circuits or {}
        self.older = older if older is not None else pd.DataFrame(columns=["season", "circuit", "starts", "incidents"])
        self.incident_rate, self.mechanical_rate = 0.0, 0.0
        self.circuit_factor, self.team_factor = {}, {}
        self.retire_rate, self.mechanical_share = 0.0, 0.5     # summaries, as RetirementModel reports

    def fit(self, history):
        rows = []
        for e in history:
            if e.session_type != "R":
                continue
            lineages = e.lineage_ids if len(e.lineage_ids) == len(e.outcomes) else [None] * len(e.outcomes)
            for outcome, excluded, lineage in zip(e.outcomes, e.excluded, lineages):
                if outcome is None or excluded:
                    continue
                rows.append((e.season, e.event_key, self.circuits.get(e.event_key), lineage, outcome))
        if not rows:
            self.incident_rate = self.mechanical_rate = self.retire_rate = 0.0
            self.circuit_factor, self.team_factor = {}, {}
            return self
        starts = pd.DataFrame(rows, columns=["season", "event", "circuit", "lineage", "outcome"])
        latest = sorted(starts.season.unique())[-RECENT_SEASONS:]
        recent = starts[starts.season.isin(latest)]
        retired = recent.outcome.isin(RETIREMENTS)
        mechanical = (recent.outcome == "mechanical").sum()
        incident = (recent.outcome == "incident_or_other").sum()
        known = mechanical + incident
        share = mechanical / known if known else 0.5
        self.retire_rate = float(retired.mean())
        self.mechanical_share = float(share)
        self.mechanical_rate = self.retire_rate * share
        self.incident_rate = self.retire_rate * (1 - share)
        self._fit_circuits(starts)
        self._fit_teams(starts)
        return self

    def _fit_circuits(self, starts):
        self.circuit_factor = {}            # neutral unless this history supports differences
        labelled = starts[starts.outcome != "retired_unknown"]
        newer = (labelled.assign(incident=(labelled.outcome == "incident_or_other").astype(int))
                 .groupby(["season", "circuit"]).agg(starts=("incident", "size"), incidents=("incident", "sum"))
                 .reset_index())
        races = pd.concat([self.older[self.older.season < starts.season.min()], newer], ignore_index=True)
        races = races[races.circuit.notna()]
        if races.empty:
            return
        rate = races.groupby("season").incidents.sum() / races.groupby("season").starts.sum()
        races = races.assign(expected=races.starts * races.season.map(rate))
        by = races.groupby("circuit")[["incidents", "expected"]].sum()
        spread = float((((by.incidents - by.expected) ** 2 - by.expected).sum()) / (by.expected ** 2).sum())
        if spread > 0:
            self.circuit_factor = _shrunk(by.incidents, by.expected, 1.0 / spread).to_dict()

    def _fit_teams(self, starts):
        """Per (season, lineage): this season's mechanical failures against the base
        rate, shrunk towards the previous season's factor."""
        starts = starts[starts.lineage.notna()]
        season_rate = starts.groupby("season").outcome.apply(lambda o: (o == "mechanical").mean())
        counts = (starts.assign(mech=(starts.outcome == "mechanical").astype(int))
                  .groupby(["season", "lineage"]).agg(obs=("mech", "sum"), n=("mech", "size")).reset_index())
        counts["exp"] = counts.n * counts.season.map(season_rate)
        self.team_factor = {}
        for season in sorted(counts.season.unique()):
            for row in counts[counts.season == season].itertuples():
                prior = self._prior(season, row.lineage)
                factor = (row.obs + TEAM_PRIOR * prior) / (row.exp + TEAM_PRIOR) if row.exp > 0 else prior
                self.team_factor[(season, row.lineage)] = float(factor)

    def _prior(self, season, lineage):
        return 1.0 + CARRY * (self.team_factor.get((season - 1, lineage), 1.0) - 1.0)

    def _team(self, season, lineage):
        if (season, lineage) in self.team_factor:
            return self.team_factor[(season, lineage)]
        return self._prior(season, lineage)            # no race of this season yet

    def chances(self, example):
        """(incident, mechanical) retirement chance per entrant."""
        n = len(example.driver_numbers)
        circuit = self.circuit_factor.get(self.circuits.get(example.event_key), 1.0)
        lineages = example.lineage_ids if len(example.lineage_ids) == n else [None] * n
        team = np.array([self._team(example.season, l) if l is not None else 1.0 for l in lineages])
        return np.full(n, self.incident_rate * circuit), np.clip(self.mechanical_rate * team, 0.0, 0.9)

    def apply(self, positions, example, rng):
        draws, n = positions.shape
        incident, mechanical = self.chances(example)
        total = np.clip(incident + mechanical, 0.0, 0.95)
        retired = rng.random((draws, n)) < total
        key = positions + retired * (n + 1)          # finishers keep their order; retirements behind them
        order = np.argsort(key, axis=1)
        final = np.empty_like(positions)
        final[np.arange(draws)[:, None], order] = np.arange(1, n + 1)
        paying = points_positions(example.season, example.session_type)
        outcomes = np.where(final == 1, OUTCOMES.index("win"),
                            np.where(final <= paying, OUTCOMES.index("other_points"), OUTCOMES.index("outside_points")))
        share = np.divide(mechanical, total, out=np.full(n, 0.5), where=total > 0)
        is_mechanical = rng.random((draws, n)) < share
        outcomes = np.where(retired, np.where(is_mechanical, OUTCOMES.index("mechanical"),
                                              OUTCOMES.index("incident_or_other")), outcomes)
        return final, outcomes
