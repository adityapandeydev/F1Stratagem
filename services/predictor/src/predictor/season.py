"""The rest of a season, simulated (stage 5).

From a cutoff before an event (`next_example`, its pre-weekend forecast), every remaining
Grand Prix and sprint of the season is simulated with the simple race model
(race.RaceModel: race pace, simulated grid, one grid weight and race-day noise,
retirements), and the points are added to the standings at the cutoff
(championship.standings).

What a simulated season shares: the pace model's state at the cutoff splits into states
that carry from event to event (car, driver, session offsets, wet ability: random walks
with the fitted drift per 14 days) and the weekend effects, which return to their prior
at every event's start. A simulated season draws the carried states once from the
state's distribution at the next event, then adds each later event's drift; the weekend
effects are drawn afresh at each event. Qualifying, race-day noise, retirements and the
fastest lap are drawn per event, exactly as the race model draws them for one race
(`RaceModel.draws`), but conditional on the season's sampled state: strength
uncertainty is sampled once, not again inside each weekend.

`shared=False` draws each event's state independently from that event's own
distribution (the same marginal per event, drift included) with the same event-level
random numbers: the difference is what carrying strength across a season does.

Assumptions (documented, not fitted): the roster at the cutoff races every remaining
event, and a driver already in the standings but not in that roster keeps his points and
results and starts no more races; a driver's points go to the team the cutoff roster puts
him in (`teams_at`); the calendar and sprint schedule are the events of the season in the archive;
sprints use the race model with a grid of their own and a third of the race retirement
rate; a retired car is not classified; the pace model has no circuit-specific state
(checked).
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .championship import points, standings
from .pace import knockout_positions
from .program import DAY_SCALE
from .targets import OUTCOMES

CARRIED = ("car", "driver", "offset", "wet")
FRESH = ("weekend",)
SPRINT_RETIREMENT_SHARE = 1 / 3
CLASSIFIED = OUTCOMES.index("outside_points")     # outcome indices up to this one are finishers


class _Scaled:
    """A retirement model whose rate is scaled (sprints)."""

    def __init__(self, base, share):
        self.base, self.share = base, share

    def apply(self, positions, example, rng):
        rate = self.base.retire_rate
        try:
            self.base.retire_rate = rate * self.share
            return self.base.apply(positions, example, rng)
        finally:
            self.base.retire_rate = rate


def teams_at(driver_ids, lineage_ids, season, teams, last_team=None):
    """Driver id -> constructor id at a cutoff: the cutoff roster's seat (its car lineage,
    mapped through the season's teams table: season, lineage_id, constructor_id) first; the
    team of the driver's last result (`last_team`) only for a driver the roster cannot place."""
    out = {}
    table = teams[teams.season == season].drop_duplicates("lineage_id").set_index("lineage_id").constructor_id
    for driver, lineage in zip(driver_ids, lineage_ids):
        if lineage is not None and not pd.isna(lineage) and lineage in table.index:
            out[driver] = table[lineage]
        elif last_team and driver in last_team:
            out[driver] = last_team[driver]
    return out


def fastest_lap_table(results, before_season, first=2014):
    """P(fastest lap | finishing position) from Grands Prix of seasons [first, before_season)
    (Jolpica results: season, position, positionText, FastestLap_rank)."""
    r = results[(results.season >= first) & (results.season < before_season)]
    r = r[r.positionText.astype(str).str.fullmatch(r"\d+").fillna(False)]
    position = pd.to_numeric(r.position).astype(int)
    fastest = (r.FastestLap_rank.astype(str) == "1")
    counts = fastest.groupby(position).sum()
    total = position.value_counts().sort_index()
    rate = (counts.reindex(total.index, fill_value=0) + 0.5) / (total + 1.0)
    out = np.zeros(31)
    out[rate.index.to_numpy()[rate.index.to_numpy() < 31]] = rate.to_numpy()[rate.index.to_numpy() < 31]
    return out


class SeasonSimulator:
    def __init__(self, model, setting, fastest, draws=2000, shared=True):
        self.model, self.setting, self.fastest = model, setting, np.asarray(fastest, dtype=float)
        self.draws, self.shared = draws, shared

    # --- state -----------------------------------------------------------------------
    def _state(self, view, example):
        pace = self.model.pace
        pace._ensure_hyper(view, example.season)
        hyper = pace.hyper
        for name in ("affinity_prior", "driver_affinity_prior", "skill_prior", "season_skill_prior", "form_prior"):
            if getattr(hyper, name, 0.0) > 0:
                raise ValueError(f"the season simulator assumes no circuit or season-form state ({name} > 0)")
        state, designs = pace.state_at(view, example, ("Q", "R"))
        used = np.flatnonzero(np.abs(np.vstack([designs["Q"], designs["R"]])).sum(axis=0) > 0)
        kinds = np.array([state.kind[i] for i in used])
        unknown = set(kinds) - set(CARRIED) - set(FRESH)
        if unknown:
            raise ValueError(f"state kinds the season simulator does not handle: {unknown}")
        drift = {"car": hyper.car_drift, "driver": hyper.driver_drift, "offset": hyper.offset_drift,
                 "wet": getattr(hyper, "wet_drift", 0.0)}
        return {"mean": state.mean[used], "cov": state.cov[np.ix_(used, used)], "kinds": kinds,
                "Q": designs["Q"][:, used], "R": designs["R"][:, used],
                "drift": np.array([drift.get(k, 0.0) for k in kinds]),
                "fresh": np.isin(kinds, FRESH), "hyper": hyper}

    def state_paths(self, s, times, rng):
        """draws x events x states: the season's states at each event (times: UTC of each
        event's cutoff; the first is the state's own time)."""
        n, k = self.draws, len(s["mean"])
        fresh, carried = s["fresh"], ~s["fresh"]
        cov = s["cov"]
        if fresh.any() and np.abs(cov[np.ix_(fresh, carried)]).max() > 1e-9:
            raise ValueError("weekend effects are correlated with the carried states at an event's start")
        steps = np.array([(t - times[0]).total_seconds() / 86400 / DAY_SCALE for t in times])
        out = np.empty((n, len(times), k))
        chol = np.linalg.cholesky(cov + 1e-12 * np.eye(k))
        for j, step in enumerate(steps):
            # this event's own distribution: drift added to the carried states, weekend afresh
            if j == 0 or not self.shared:
                c = cov + np.diag(np.where(carried, s["drift"] * step, 0.0))
                cj = np.linalg.cholesky(c + 1e-12 * np.eye(k)) if j else chol
                out[:, j] = s["mean"] + rng.standard_normal((n, k)) @ cj.T
            else:
                previous = out[:, j - 1].copy()
                walk = np.sqrt(np.maximum(s["drift"] * (step - steps[j - 1]), 0.0))
                previous[:, carried] += rng.standard_normal((n, carried.sum())) * walk[carried]
                if fresh.any():
                    cf = np.linalg.cholesky(cov[np.ix_(fresh, fresh)] + 1e-12 * np.eye(fresh.sum()))
                    previous[:, fresh] = s["mean"][fresh] + rng.standard_normal((n, fresh.sum())) @ cf.T
                out[:, j] = previous
        return out

    # --- one weekend -------------------------------------------------------------------
    def weekend(self, s, x, example, rng_seed, sprint=False):
        """Race (and sprint) of one event given the sampled states x (draws x states):
        (final positions, classified, fastest) per session, as RaceModel.draws/predict."""
        hyper = s["hyper"]
        pace = self.model.pace
        n, size = x.shape[0], s["R"].shape[0]
        qualifying, race_pace = x @ s["Q"].T, x @ s["R"].T
        dof = hyper.robust_dof if pace.robust else 0.0
        out = {}
        for kind in (("S", "R") if sprint else ("R",)):
            seed = [rng_seed, 0 if kind == "R" else 1]
            noise = hyper.noise["R"] * np.random.default_rng([*seed, 4]).standard_normal((n, size))
            if dof > 0:
                noise /= np.sqrt(np.random.default_rng([*seed, 5]).chisquare(dof, (n, 1)) / dof)
            grid = knockout_positions(qualifying, hyper.noise["Q"], np.random.default_rng([*seed, 6]), dof=dof,
                                      scale_rng=np.random.default_rng([*seed, 7]))
            context = {"difficulty": 1.0, "withdrawn": None}
            order = self.model.run(race_pace + noise, np.atleast_2d(grid).astype(float), self.setting, context,
                                   int(np.random.default_rng([*seed, 8]).integers(2**31)))
            retirements = self.model.retirements if kind == "R" else _Scaled(self.model.retirements,
                                                                               SPRINT_RETIREMENT_SHARE)
            final, outcomes = retirements.apply(order, replace(example, session_type=kind),
                                                np.random.default_rng([*seed, 9]))
            classified = outcomes <= CLASSIFIED
            fastest = np.zeros((n, size), dtype=bool)
            if kind == "R":
                weight = np.where(classified, self.fastest[np.clip(final, 0, len(self.fastest) - 1)], 0.0)
                total = weight.sum(axis=1, keepdims=True)
                pick = (np.random.default_rng([*seed, 10]).random((n, 1)) * total > np.cumsum(weight, axis=1)).sum(axis=1)
                fastest[np.arange(n), np.minimum(pick, size - 1)] = total[:, 0] > 0
            out[kind] = (final, classified, fastest)
        return out

    # --- a season ----------------------------------------------------------------------
    def simulate(self, view, next_example, events, carried, teams, seed):
        """events: DataFrame of the remaining events in order (event_key, round, time: the
        event's pre-weekend cutoff, sprint: bool), the first being next_example's;
        carried: (drivers, constructors) standings at the cutoff (championship.standings), or
        None before the first race; teams: driver id -> constructor id (`teams_at`). Returns
        final points and ranks per draw for every driver in the roster or the standings."""
        s = self._state(view, next_example)
        rng = np.random.default_rng([seed, 100 + int(self.shared)])
        paths = self.state_paths(s, list(events.time), rng)
        drivers = list(next_example.driver_ids)
        season = next_example.season
        size, n = len(drivers), self.draws
        add = np.zeros((n, size))
        gp_counts = np.zeros((n, size, size + 1))
        for j, ev in enumerate(events.itertuples()):
            example = replace(next_example, event_key=ev.event_key, cutoff_time=ev.time)
            sessions = self.weekend(s, paths[:, j], example, [seed, j], sprint=bool(ev.sprint))
            for kind, (final, classified, fastest) in sessions.items():
                position = np.where(classified, final, 0)
                add += points(season, kind, position.ravel(), fastest.ravel()).reshape(n, size)
                if kind == "R":
                    np.add.at(gp_counts, (np.arange(n)[:, None], np.arange(size)[None, :], position), 1)
        return self._tables(add, gp_counts, drivers, carried, teams)

    def _tables(self, add, gp_counts, drivers, carried, teams):
        """Final tables per draw from the points (`add`, draws x roster) and Grand Prix
        finishes (`gp_counts`, draws x roster x position) the remaining events gave the
        roster, on the carried standings. A driver in the standings but not in the roster
        keeps his points and results and adds nothing."""
        if carried is None:
            carried = _empty_standings()
        n = add.shape[0]
        absent = [d for d in carried[0].index if d not in set(drivers)]
        if absent:
            add = np.hstack([add, np.zeros((n, len(absent)))])
            gp_counts = np.concatenate([gp_counts, np.zeros((n, len(absent), gp_counts.shape[2]))], axis=1)
            drivers = list(drivers) + absent
        old_d, old_c = carried
        n, size = add.shape
        # countback reaches the deepest finish, earlier or simulated (a field that shrinks
        # must not drop an earlier P20)
        deepest = [gp_counts.shape[2] - 1]
        for table in (old_d, old_c):
            p = table.attrs["finishes"].position
            if len(p) and p.notna().any():
                deepest.append(int(p.max()))
        positions = max(deepest)
        if positions + 1 > gp_counts.shape[2]:
            gp_counts = np.concatenate([gp_counts, np.zeros((n, size, positions + 1 - gp_counts.shape[2]))], axis=2)
        base = old_d.points.reindex(drivers).fillna(0.0).to_numpy()
        finish = old_d.attrs["finishes"]
        past = np.zeros((size, positions + 1))
        for i, d in enumerate(drivers):
            p = finish[(finish.key == d) & (finish.position >= 1)].position.astype(int)
            for k in p[p <= positions]:
                past[i, k] += 1
        total = base[None, :] + add
        counts = gp_counts + past[None]
        # countback: points, then wins, second places, ... (np.lexsort: last key first)
        keys = [counts[:, :, k] for k in range(positions, 0, -1)] + [total]
        rank = np.empty((n, size), dtype=int)
        for i in range(n):
            order = np.lexsort([-k[i] for k in keys])
            rank[i, order] = np.arange(1, size + 1)
        # constructors: their drivers' added points on the carried table
        names = sorted(set(old_c.index) | {teams.get(d) for d in drivers if teams.get(d)})
        team_of = np.array([names.index(teams[d]) if teams.get(d) in names else -1 for d in drivers])
        c_total = np.tile(old_c.points.reindex(names).fillna(0.0).to_numpy(), (n, 1))
        c_counts = np.zeros((n, len(names), positions + 1))
        finish_c = old_c.attrs["finishes"]
        for t, name in enumerate(names):
            p = finish_c[(finish_c.key == name) & (finish_c.position >= 1)].position.astype(int)
            for k in p[p <= positions]:
                c_counts[:, t, k] += 1
        for i, t in enumerate(team_of):
            if t >= 0:
                c_total[:, t] += add[:, i]
                c_counts[:, t] += gp_counts[:, i]
        c_keys = [c_counts[:, :, k] for k in range(positions, 0, -1)] + [c_total]
        c_rank = np.empty((n, len(names)), dtype=int)
        for i in range(n):
            order = np.lexsort([-k[i] for k in c_keys])
            c_rank[i, order] = np.arange(1, len(names) + 1)
        return {"drivers": drivers, "points": total, "rank": rank,
                "constructors": names, "constructor_points": c_total, "constructor_rank": c_rank}


def _empty_standings():
    d = pd.DataFrame(columns=["points", "wins", "rank"], dtype=float)
    c = pd.DataFrame(columns=["points", "wins", "rank"], dtype=float)
    for t in (d, c):
        t.attrs["finishes"] = pd.DataFrame({"key": pd.Series(dtype=object), "position": pd.Series(dtype=float)})
    return d, c
