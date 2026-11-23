"""Race forecasts from the pace model.

Stage 1a: each simulated race orders the drivers by a race score,

    score = race pace + kappa x (grid slot - 1) + sigma x race-day noise,

lower is better. Race pace is the pace model's forecast for a race (session type
"R": the fuel- and tyre-corrected race pace it learns from every race, with the
race observation noise), the grid is the one known at the cutoff, and retirements
come from the historical rate (`baselines.RetirementModel`), as for the baselines.

- **Grid.** Once the session that sets the race grid has been classified at the
  cutoff (qualifying; the sprint in 2021-2022), its order is the grid. Before that
  the grid is simulated from the same pace draws through the qualifying knockout
  (`pace.knockout_positions`), so a driver fast in qualifying is fast in the race
  in the same draw. With `published_grid` (stage 4), the latest FIA starting grid
  published by the cutoff (grid penalties applied; `published_grid_at`) replaces
  the qualifying order once it is out: pit-lane starters behind it, and the cars it
  lists as withdrawn classified behind everyone in every draw (`withdraw`). With
  `known_penalties`, until then the grid penalties the stewards have published by
  the cutoff to be served in this race (not its sprint) are applied to the
  qualifying order, or to each simulated grid, by the FIA's allocation procedure
  (`gridrules`); grid penalties not yet decided are not forecast.
- **kappa, sigma.** kappa is what a grid slot is worth in pace units (% of a lap):
  0 lets the race pace alone decide, a large kappa freezes the grid. sigma is race
  noise beyond the pace model's (starts, strategy, incidents). Both are fitted per
  season and cutoff on the races of earlier seasons, by mean ranked probability
  score, on a fixed grid of values with common random numbers.

Stage 1b (`overtaking=True`): a grid slot is worth more where passing is hard,

    kappa at a circuit = kappa x (typical pass rate / circuit pass rate) ^ gamma,

with each circuit's on-track pass rate (`laps.race_passes`: passes per green lap,
dry races) from the races finished before the cutoff, shrunk towards the typical
rate by PASS_PRIOR_RACES races' worth of it; a circuit without history gets the
typical rate. gamma is fitted with kappa and sigma (0 is stage 1a).
"""
import functools
import zlib

import numpy as np
import pandas as pd

from .baselines import RetirementModel, seed_for
from .forecast import Forecast
from .gridrules import GridPenalty, apply_grid_penalties, procedure
from .laps import race_passes
from .pace import knockout_positions
from .rules import points_positions
from .targets import OUTCOMES, RETIREMENTS
from . import race_sim, track
from .retirements import DetailedRetirements, history_incidents

KAPPAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.4)
DRAWS = 10000
FIT_DRAWS = 1000
STICK_SE = 1.645   # the simple model keeps last season's settings unless beaten by this many paired SEs
GRID_FROM_SPRINT = frozenset({2021, 2022})   # seasons whose sprint set the race grid
GAMMAS = (0.0, 0.5, 1.0, 1.5, 2.0)
PASS_PRIOR_RACES = 1.0      # weight of the typical pass rate in a circuit's, in races


def _slots(order, driver_numbers):
    """Grid slots 1..n of `driver_numbers` by their value in `order` (missing: behind)."""
    values = np.array([order.get(str(n), np.nan) for n in driver_numbers], dtype=float)
    values[np.isnan(values)] = np.inf
    slots = np.empty(len(values))
    slots[np.argsort(values, kind="stable")] = np.arange(1, len(values) + 1)
    return slots


@functools.lru_cache(maxsize=4)
def starting_grids(root):
    """The reference table of FIA race starting grids under data root `root` (empty if
    it has not been built): event_key, grid_position, status (grid, pit_lane,
    withdrawn), driver_number, published_utc, doc_number."""
    path = root / "reference" / "starting_grids.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=["event_key", "grid_position", "status", "driver_number", "published_utc",
                                     "doc_number"])
    grids = pd.read_parquet(path)
    grids["published_utc"] = pd.to_datetime(grids.published_utc, utc=True)
    grids["driver_number"] = grids.driver_number.astype(str)
    if "status" not in grids:
        grids["status"] = "grid"
    return grids


def grid_published(archive, event_key, before):
    """Publication time of the latest starting grid of `event_key`'s race published
    before `before`, or None."""
    grids = starting_grids(archive.root)
    times = grids.published_utc[(grids.event_key == event_key) & (grids.published_utc < before)]
    return times.max() if len(times) else None


def published_grid_at(view, example):
    """(grid slots, withdrawn) of the example's entrants from the latest FIA starting
    grid of its race published by the view's cutoff (provisional or final: grid
    penalties applied), or None if none is out. Pit-lane starters take the slots
    after the grid's, in the order listed; withdrawn: a boolean per entrant, the cars
    the document lists as withdrawn (they get the last slots and take no part). An
    entrant the document does not mention starts behind those it lists."""
    grids = starting_grids(view.archive.root)
    out = grids[(grids.event_key == example.event_key) & (grids.published_utc <= view.cutoff)]
    if out.empty:
        return None
    latest = out[(out.published_utc == out.published_utc.max())]
    latest = latest[latest.doc_number == latest.doc_number.max()]
    starting = latest[latest.status != "withdrawn"]
    gone = set(latest.driver_number[latest.status == "withdrawn"])
    withdrawn = np.array([str(n) in gone for n in example.driver_numbers])
    order = {str(d): 1e5 for d in example.driver_numbers}          # not mentioned: behind those listed
    order.update({d: float(p) for d, p in zip(starting.driver_number, starting.grid_position)})
    order.update({d: 1e6 for d in gone})                             # withdrawn: the last slots
    return _slots(order, example.driver_numbers), withdrawn


@functools.lru_cache(maxsize=4)
def grid_penalty_table(root):
    """The reference table of stewards' grid penalties under data root `root` (empty
    if not built): event_key and served (the session it is served in: "S" sprint, "R"
    race), driver_number, places, back_of_grid, pit_lane, published_utc, superseded_utc
    (when a replacing decision was published; NaT while the decision is in force)."""
    path = root / "reference" / "grid_penalties.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=["event_key", "served", "driver_number", "places", "back_of_grid", "pit_lane",
                                     "published_utc", "doc_number", "superseded_utc"])
    table = pd.read_parquet(path)
    table["published_utc"] = pd.to_datetime(table.published_utc, utc=True)
    table["driver_number"] = table.driver_number.astype(str)
    if "served" not in table:
        table["served"] = "R"
    if "pit_lane" not in table:
        table["pit_lane"] = False
    table["superseded_utc"] = (pd.to_datetime(table.superseded_utc, utc=True) if "superseded_utc" in table
                               else pd.Series(pd.NaT, index=table.index, dtype="datetime64[ns, UTC]"))
    return table


def known_penalties(view, event_key, served="R"):
    """{driver number: GridPenalty} from the grid penalties served in `event_key`'s
    race (`served` "R"; "S" its sprint) published by the view's cutoff and not yet
    replaced by then: drops add up, `first` orders the drivers by their earliest
    decision (offence order)."""
    table = grid_penalty_table(view.archive.root)
    rows = table[(table.event_key == event_key) & (table.served == served) & (table.published_utc <= view.cutoff)
                 & ~(table.superseded_utc <= view.cutoff)]
    penalties = {}
    for rank, row in enumerate(rows.sort_values(["published_utc", "doc_number"]).itertuples()):
        old = penalties.get(row.driver_number, GridPenalty(first=rank))
        penalties[row.driver_number] = GridPenalty(
            old.places + (0 if pd.isna(row.places) else int(row.places)), old.back or bool(row.back_of_grid),
            old.pit_lane or bool(row.pit_lane), old.first)
    return penalties


def withdraw(final, outcomes, withdrawn, example):
    """(final positions, outcomes) with the `withdrawn` entrants moved behind everyone,
    the others moved up, and the finishers' outcomes (win, points) read again from
    their new positions; a withdrawn entrant does not start (incident_or_other, unscored)."""
    draws, n = final.shape
    key = final + withdrawn[None, :] * (n + 1)
    order = np.argsort(key, axis=1, kind="stable")
    moved = np.empty_like(final)
    moved[np.arange(draws)[:, None], order] = np.arange(1, n + 1)
    finished = ~np.isin(outcomes, [OUTCOMES.index(o) for o in RETIREMENTS if o in OUTCOMES])
    paying = points_positions(example.season, example.session_type)
    again = np.where(moved == 1, OUTCOMES.index("win"),
                     np.where(moved <= paying, OUTCOMES.index("other_points"), OUTCOMES.index("outside_points")))
    outcomes = np.where(finished, again, outcomes)
    return moved, np.where(withdrawn[None, :], OUTCOMES.index("incident_or_other"), outcomes)


def grid_at(view, example):
    """Grid slots (1 = pole) of the example's entrants from the session that sets the
    race grid, if it is classified at the cutoff; None otherwise. An entrant missing
    from that classification starts behind those in it."""
    sessions = view.sessions[view.sessions.event_key == example.event_key]
    types = set(sessions.session_type)
    source = "S" if example.season in GRID_FROM_SPRINT and "S" in types else "Q"
    if source not in types:
        return None
    results = view.results(sessions[sessions.session_type == source].session_id.iloc[0])
    order = dict(zip(results.DriverNumber.astype(str), pd.to_numeric(results.Position, errors="coerce")))
    return _slots(order, example.driver_numbers)


class RaceModel:
    """Model interface: fit(history, view_for) then predict(view, example). Races only."""

    # True: the stable settings fit (2026-11-23): keyed simulator random numbers, fit seeds
    # of their own, and a steadier choice (`_choose`: last season's settings kept unless
    # clearly beaten; the simulator smooths over neighbouring settings). False: the fit
    # before then (sequential numbers, forecast seeds, lowest mean), for comparison
    stable_fit = True

    def __init__(self, pace, name, overtaking=False, detailed_retirements=False):
        self.pace = pace
        self.name = name
        self.overtaking = overtaking
        self.detailed_retirements = detailed_retirements
        self.published_grid = False  # True: the published starting grid replaces the qualifying order once out
        self.known_penalties = False  # True: grid penalties published by the cutoff applied until then
        self.grid_inputs_in_fit = True  # False (diagnosis): settings fitted without the two grid inputs above
        self._passes = {}           # race session -> race_passes(...)
        self.retirements = DetailedRetirements() if detailed_retirements else RetirementModel()
        self.settings = {}          # (season, cutoff label) -> {"kappa", "sigma", "rps", ...}
        self._history, self._view_for = [], None
        self._draws = {}            # (event, cutoff, season) -> draws(...) at FIT_DRAWS, while fitting

    @property
    def fit_cache(self):
        return self.pace.fit_cache

    @fit_cache.setter
    def fit_cache(self, cache):
        self.pace.fit_cache = cache

    def fit(self, history, view_for):
        self._history = [e for e in history if e.session_type == "R"]
        self._view_for = view_for
        if self.detailed_retirements and history and not self.retirements.circuits:
            archive = view_for(history[0]).archive     # circuits are reference data; older results end in 2017
            self.retirements = DetailedRetirements(dict(zip(archive.events.event_key, archive.events.circuit_id)),
                                                   history_incidents(archive.root))
        self.retirements.fit(history)
        return self

    def draws(self, view, example, count, seed):
        """(race pace, grid): race pace draws x entrants with the race observation noise,
        and grid slots, either the known grid (one row) or simulated (draws x entrants)."""
        pace = self.pace
        rng = np.random.default_rng(seed)
        state, designs = pace.state_at(view, example, ("Q", "R"))
        design = np.vstack([designs["Q"], designs["R"]])
        size = len(example.driver_numbers)
        mean, cov = design @ state.mean, design @ state.cov @ design.T
        joint = rng.multivariate_normal(mean, cov + 1e-9 * np.eye(2 * size), size=count, method="cholesky")
        qualifying, race = joint[:, :size], joint[:, size:]
        dof = pace.hyper.robust_dof if pace.robust else 0.0
        noise = pace.hyper.noise["R"] * np.random.default_rng([seed, 4]).standard_normal((count, size))
        if dof > 0:          # the robust model's race noise: one Student-t scale per race
            noise /= np.sqrt(np.random.default_rng([seed, 5]).chisquare(dof, (count, 1)) / dof)
        race = race + noise
        published = published_grid_at(view, example) if self.published_grid else None
        if published is not None:
            return race, np.atleast_2d(published[0]).astype(float)
        grid = grid_at(view, example)
        if grid is None:
            grid = knockout_positions(qualifying, pace.hyper.noise["Q"], np.random.default_rng([seed, 6]), dof=dof,
                                      scale_rng=np.random.default_rng([seed, 7]))
        if self.known_penalties:
            penalties = known_penalties(view, example.event_key)
            if penalties:
                grid = apply_grid_penalties(grid, example.driver_numbers, penalties,
                                            procedure(example.season, example.cutoff_time))
        return race, np.atleast_2d(grid).astype(float)

    def start_context(self, view, example):
        """`context` plus "withdrawn": the entrants a published starting grid lists as
        withdrawn (None when there are none or the model does not read the grid)."""
        context = self.context(view, example)
        published = published_grid_at(view, example) if self.published_grid else None
        withdrawn = published[1] if published is not None and published[1].any() else None
        return {**context, "withdrawn": withdrawn}

    def difficulty(self, view, example):
        """(typical pass rate / this circuit's pass rate) from the dry races finished
        before the cutoff, the circuit's rate shrunk towards the typical one; 1 without
        history."""
        circuit = self.pace.circuit(view, example.event_key)
        races = view.sessions[(view.sessions.session_type == "R") & (view.sessions.event_key != example.event_key)]
        rates, here = [], []
        for race in races.itertuples():
            if race.session_id not in self._passes:
                self._passes[race.session_id] = race_passes(view.laps(race.session_id))
            counted = self._passes[race.session_id]
            if counted is None:
                continue
            rate = counted[0] / counted[1]
            rates.append(rate)
            if circuit is not None and self.pace.circuit(view, race.event_key) == circuit:
                here.append(rate)
        if not rates:
            return 1.0
        typical = float(np.mean(rates))
        rate = (sum(here) + PASS_PRIOR_RACES * typical) / (len(here) + PASS_PRIOR_RACES)
        return typical / max(rate, 1e-3)

    def candidates(self):
        """Settings tried by the fit, in order (the first best is kept)."""
        for gamma in (GAMMAS if self.overtaking else (0.0,)):
            for kappa in KAPPAS:
                for sigma in SIGMAS:
                    yield {"kappa": kappa, "sigma": sigma, **({"gamma": gamma} if self.overtaking else {})}

    def neutral(self):
        """Settings without any earlier race to fit them on."""
        return {"kappa": 0.0, "sigma": 0.0, **({"gamma": 0.0} if self.overtaking else {})}

    def prepare_season(self, season, cutoff):
        """Anything fitted per season and cutoff before the settings (nothing here)."""

    def select(self, key):
        """Use what `prepare_season` fitted for `key` (nothing here)."""

    def context(self, view, example):
        """What a forecast needs from the archive besides the draws."""
        return {"difficulty": self.difficulty(view, example) if self.overtaking else 1.0}

    def order(self, race, grid, setting, context, seed):
        kappa = setting["kappa"] * context["difficulty"] ** setting.get("gamma", 0.0)
        return self.positions(race, grid, kappa, setting["sigma"], seed)

    def run(self, race, grid, setting, context, seed):
        """`order` over the cars that start: the entrants context["withdrawn"] marks
        take no part (no passes, blocking or pit traffic) and fill the last places;
        `forecast` then classifies them behind everyone."""
        withdrawn = context.get("withdrawn")
        if withdrawn is None or not withdrawn.any():
            return self.order(race, grid, setting, context, seed)
        active = ~withdrawn
        sub = dict(context)
        strategy = context.get("strategy")
        if strategy and strategy.get("team") is not None:     # the only per-car strategy input
            sub["strategy"] = {**strategy, "team": np.asarray(strategy["team"])[active]}
        positions = self.order(race[:, active], grid[:, active], setting, sub, seed)
        full = np.empty((len(positions), len(withdrawn)), dtype=positions.dtype)
        full[:, active] = positions
        full[:, withdrawn] = np.arange(active.sum() + 1, len(withdrawn) + 1)
        return full

    def positions(self, race, grid, kappa, sigma, seed):
        """Finishing order before retirements, draws x entrants (1 = first)."""
        score = race + kappa * (grid - 1)
        if sigma > 0:
            score = score + sigma * np.random.default_rng([seed, 8]).standard_normal(race.shape)
        order = np.argsort(score, axis=1)
        positions = np.empty_like(order)
        positions[np.arange(len(score))[:, None], order] = np.arange(1, score.shape[1] + 1)
        return positions

    def forecast(self, example, positions, seed, metadata, withdrawn=None):
        """Forecast from finishing orders before retirements; `withdrawn` (a boolean
        per entrant) are known not to start: classified behind everyone in every draw
        (not scored, but the places they would take go to the others)."""
        rng = np.random.default_rng([seed, 9])
        final, outcomes = self.retirements.apply(positions, example, rng)
        if withdrawn is not None and withdrawn.any():
            final, outcomes = withdraw(final, outcomes, withdrawn, example)
        return Forecast.from_draws(
            positions=final, outcomes=outcomes, season=example.season, session_type=example.session_type,
            event_key=example.event_key, cutoff_label=example.cutoff_label, cutoff_time=example.cutoff_time.isoformat(),
            driver_numbers=list(example.driver_numbers), driver_ids=list(example.driver_ids), model=self.name,
            seed=seed, metadata=metadata)

    def _fit_inputs(self, view, example):
        """(race pace, grid, context) of an earlier race for fitting the settings. With
        `grid_inputs_in_fit` False they are read without the published grid and the
        known penalties, so the settings are those of the model without them (the
        diagnosis of the inputs' own effect, separate from refitting)."""
        saved = self.published_grid, self.known_penalties
        if not self.grid_inputs_in_fit:
            self.published_grid = self.known_penalties = False
        try:
            return (*self.draws(view, example, FIT_DRAWS, fit_seed(example, self.stable_fit)),
                    self.start_context(view, example))
        finally:
            self.published_grid, self.known_penalties = saved

    def _fit_season(self, season, cutoff):
        """The settings for `season` at the cutoff labelled `cutoff`, from the races of
        earlier seasons in the history forecast from the same cutoff."""
        from .evaluation import score
        earlier = [e for e in self._history if e.season < season and e.cutoff_label == cutoff]
        if not earlier:
            self.settings[(season, cutoff)] = {**self.neutral(), "rps": None, "races": 0}
            return
        prepared = []
        for example in earlier:
            key = (example.event_key, example.cutoff_label, season)
            if key not in self._draws:
                view = self._view_for(example)
                self._draws[key] = self._fit_inputs(view, example)
            prepared.append((example, *self._draws[key]))
        surface = []
        for setting in self.candidates():
            per_race = []
            for example, race, grid, context in prepared:
                seed = fit_seed(example, self.stable_fit)
                forecast = self.forecast(example, self.run(race, grid, setting, context, seed), seed, {},
                                         context.get("withdrawn"))
                per_race.append(score(forecast, example)["rps"])
            surface.append((setting, per_race))
        means = np.array([np.mean(r) for _, r in surface])
        chosen = self._choose(surface, means, self.settings.get((season - 1, cutoff)))
        best = (float(means[chosen]), surface[chosen][0])
        # every candidate's RPS per earlier race (diagnosis of the settings fit; not saved)
        self.fit_surface = {**getattr(self, "fit_surface", {}),
                            (season, cutoff): ([e.event_key for e, *_ in prepared], surface)}
        self.settings[(season, cutoff)] = {**best[1], "rps": best[0], "races": len(prepared)}
        self._draws = {}             # tied to this season's pace settings

    def _choose(self, surface, means, previous):
        """Index of the chosen candidate. Old fit: the lowest mean RPS. Stable fit (rule R3,
        chosen on the 2020-22 replay, docs of 2026-11-23): last season's setting unless
        the lowest beats it by more than STICK_SE paired standard errors per race."""
        best = int(means.argmin())
        if not self.stable_fit or not previous:
            return best
        names = [k for k in surface[0][0]]
        kept = next((i for i, (s, _) in enumerate(surface) if all(s[n] == previous.get(n) for n in names)), None)
        if kept is None:
            return best
        d = np.asarray(surface[kept][1]) - np.asarray(surface[best][1])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        return kept if d.mean() <= STICK_SE * se else best

    def predict(self, view, example, draws=None):
        draws = DRAWS if draws is None else draws          # read at call time: `evaluate --draws` sets it
        if example.session_type != "R":
            return None
        self.pace._ensure_hyper(view, example.season)
        self._archive = view.archive
        key = (example.season, example.cutoff_label)
        if key not in self.settings:
            self.prepare_season(*key)
            self._fit_season(*key)
        self.select(key)
        chosen = self.settings[key]
        seed = seed_for(example)
        race, grid = self.draws(view, example, draws, seed)
        context = self.start_context(view, example)
        positions = self.run(race, grid, chosen, context, seed)
        known_grid = grid.shape[0] == 1
        withdrawn = context.pop("withdrawn", None)
        return self.forecast(example, positions, seed, {
            "grid": "known" if known_grid else "simulated",
            **({"withdrawn": [str(n) for n, w in zip(example.driver_numbers, withdrawn) if w]}
               if withdrawn is not None else {}),
            "context": {k: (v["factor"] if k == "neutral" else
                            {x: (y.tolist() if isinstance(y, np.ndarray) else y) for x, y in v.items() if x != "residuals"}
                            if k == "strategy" else v)
                        for k, v in context.items()},
            "hyper": {"race": chosen, "noise": dict(self.pace.hyper.noise), "fit": self.pace.fit_log.get(example.season)},
            "retire_rate": self.retirements.retire_rate, "mechanical_share": self.retirements.mechanical_share,
            **({"retire_factors": dict(zip(("incident", "mechanical"), map(list, self.retirements.chances(example))))}
               if self.detailed_retirements else {})}, withdrawn)


def fit_seed(example, own=True):
    """Random seed of an earlier race when settings are fitted on it: its own, apart
    from the seed of its forecast (`seed_for`), so a fit is not tuned to the draws
    the race is later scored with; shared by every candidate setting (paired).
    own=False: the forecast seed (the fit before 2026-11-23)."""
    return zlib.crc32(f"fit|{seed_for(example)}".encode()) if own else seed_for(example)


def smoothed(settings, means):
    """Each candidate's mean RPS averaged with the candidates one step away in one
    setting on the grid of tried values, so a lone lucky candidate on a flat surface
    cannot win (rule R2, the simulator's choice: SimulatedRace._choose)."""
    names = [k for k in settings[0] if k not in ("rps", "races")]
    levels = {n: sorted({s[n] for s in settings}) for n in names}
    at = np.array([[levels[n].index(s[n]) for n in names] for s in settings])
    near = np.abs(at[:, None, :] - at[None, :, :]).sum(axis=2) <= 1
    return (near * np.asarray(means)[None, :]).sum(axis=1) / near.sum(axis=1)


def scheduled_laps(earlier, layout):
    """Race length in laps: the most laps completed in an earlier race on the same
    layout (a shortened race only ever has fewer; the distance rule is fixed per
    layout), else on any layout of the circuit. `earlier`: (layout, laps
    completed) of earlier races at the circuit. None without any."""
    same = [laps for name, laps in earlier if layout is not None and name == layout]
    pool = same or [laps for _, laps in earlier]
    return max(pool) if pool else None


START_NOISES = (0.0, 0.25, 0.5, 1.0, 2.0)   # s: start noise of a car's time
SIM_SIGMAS = (0.0, 0.15, 0.3, 0.45, 0.6)     # % of a lap: race-day pace noise beyond the pace model's
PASS_SHIFTS = (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0)   # log-odds added to every pass (pass_shift option)
STINT_NOISES = (0.0, 1.5, 3.0)   # laps: spread of each car's planned stops (strategy option)
PIT_NOISES = (0.0, 0.5, 1.0)     # share kept of slow-stop residuals and per-car stop changes (pit_noise option)
BATTLE_DRAWS = 10       # forecast draws per battle pair when the battle model reads forecast pace


class SimulatedRace(RaceModel):
    """Stage 2: the race run lap by lap (`race_sim.simulate`): start, racing laps,
    passes from the battle model. Fitted per season: the battle model on the dry
    races of earlier seasons; then lap-1 noise and race-day noise by mean RPS.

    `forecast_battles`: the battle model reads, for each pair, the pace difference
    the model forecast at the cutoff (BATTLE_DRAWS draws of its race pace, the input
    the simulation gives it) instead of the race's own measured pace, which is
    taken from the same race and so is slowed by being held up.

    `neutralisations`: safety cars, virtual safety cars and red flags
    (`race_sim.NeutralModel`, fitted per season on the races before it).

    `weekend_lap`: once this weekend's qualifying has run (dry), the typical race
    lap is its median best lap times the race-to-qualifying ratio of earlier races
    (median 1.075, spread 1.5% over 2018-2024). It predicts the race's green lap to
    1.0% (median; 2.0% at the 90th percentile), against 1.5% (10.8%) from the
    latest earlier race at the circuit, which stays the source before qualifying."""

    def __init__(self, pace, name, pass_shift=False, forecast_battles=False, neutralisations=False,
                 weekend_lap=False, detailed_retirements=False, strategy=False, pit_lane=False,
                 car_strategies=False, pit_noise=False, weekend_wear=False, team_wear=False, pit_model=False,
                 following=False, follow_penalty=False, lap_effect_wear=False, fresh_peak=False, windows=False,
                 start_roles=False, gambles=False):
        super().__init__(pace, name, detailed_retirements=detailed_retirements)
        self.strategy = strategy
        self.pit_lane, self.car_strategies = pit_lane, car_strategies
        self.pit_noise = pit_noise
        self.weekend_wear = weekend_wear
        self._wear_models, self._practice = {}, {}
        self._ratio, self._wear = race_sim.RACE_TO_QUALIFYING, None
        self.team_wear = team_wear
        self._team_devs, self._team_models, self._team = {}, {}, None
        self.pit_model = pit_model
        self.following = following
        self._gaps, self._follow_tables, self._follow = {}, {}, None
        self._runs, self._release = {}, 0.0
        self.follow_penalty = follow_penalty
        self.lap_effect_wear = lap_effect_wear
        self.fresh_peak = fresh_peak
        self._peaks, self._peak_tables, self._peak = {}, {}, None
        self.windows = windows
        self._windows, self._window_tables, self._window = {}, {}, None
        self.start_roles = start_roles
        self._roles, self._role_tables, self._role = {}, {}, None
        self.gambles = gambles            # True: grid band and team wear; "grid": grid band only; "pace": by pace rank
        self._gamble_rows, self._gamble_models, self._gamble = {}, {}, None
        self._penalties, self._penalty_tables, self._penalty = {}, {}, None
        self._pit_rows, self._pass_rates, self._pit_models, self._pit = {}, {}, {}, None
        self._strategies, self._stops, self._strategy = {}, {}, None
        self.weekend_lap = weekend_lap
        self._qualifying, self._ratios = {}, {}
        self.pass_shift = pass_shift
        self.forecast_battles = forecast_battles
        self.neutralisations = neutralisations
        self._neutral_models, self._periods, self._neutral = {}, {}, None
        self.passes = race_sim.PassModel()
        self._pass_models = {}      # (season, cutoff) -> PassModel
        self._battles, self._race_shape = {}, {}
        self._archive = None

    def fit(self, history, view_for):
        super().fit(history, view_for)
        if history and self._archive is None:
            self._archive = view_for(history[0]).archive
        return self

    def _choose(self, surface, means, previous):
        """Index of the chosen candidate. Old fit: the lowest mean RPS. Stable fit: the
        lowest after smoothing over neighbouring settings (rule R2, chosen on the 2020-22
        replay, docs of 2026-11-23: the simulator's surface is flat)."""
        if not self.stable_fit:
            return int(means.argmin())
        return int(smoothed([s for s, _ in surface], means).argmin())

    def candidates(self):
        for shift in (PASS_SHIFTS if self.pass_shift else (0.0,)):
            for stint in (STINT_NOISES if self.strategy else (None,)):
                for pit in (PIT_NOISES if self.pit_noise else (None,)):
                    for start_noise in START_NOISES:
                        for sigma in SIM_SIGMAS:
                            yield {"start_noise": start_noise, "sigma": sigma,
                                   **({"pass_shift": shift} if self.pass_shift else {}),
                                   **({"stint_noise": stint} if self.strategy else {}),
                                   **({"pit_noise": pit} if self.pit_noise else {})}

    def neutral(self):
        return {"start_noise": 1.0, "sigma": 0.0}

    def _race_battles(self, view, race):
        if race.session_id not in self._battles:
            frame = race_sim.battles(view.laps(race.session_id))
            self._battles[race.session_id] = frame.assign(circuit=self.pace.circuit(view, race.event_key),
                                                          season=int(race.season), event_key=race.event_key)
        return self._battles[race.session_id]

    def _battles_in(self, view, exclude=None):
        races = view.sessions[(view.sessions.session_type == "R") & (view.sessions.event_key != exclude)]
        frames = [self._race_battles(view, race) for race in races.itertuples()]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["gap", "delta", "passed", "ahead", "behind", "lap", "circuit", "season", "event_key"])

    def select(self, key):
        self.passes = self._pass_models[key]
        self._activate(key[0])

    def _activate(self, season):
        """Make `season`'s fitted models the active ones. Every race forecast while
        they are active uses them, including the earlier seasons' races that fit the
        season's settings, so a season's forecasts do not depend on which seasons
        were evaluated before it."""
        self._neutral = self._neutral_models.get(season)
        self._strategy = self._strategies.get(season)
        self._ratio = self._ratios.get(season, race_sim.RACE_TO_QUALIFYING)
        self._wear = self._wear_models.get(season) if self.weekend_wear else None
        self._team = self._team_models.get(season) if self.team_wear else None
        self._pit = self._pit_models.get(season) if self.pit_model else None
        follow = self._follow_tables.get(season) if self.following else None
        self._follow, self._release = (follow if follow is not None else (None, 0.0))
        self._penalty = self._penalty_tables.get(season) if self.follow_penalty else None
        self._peak = self._peak_tables.get(season) if self.fresh_peak else None
        self._window = self._window_tables.get(season) if self.windows else None
        self._role = self._role_tables.get(season) if self.start_roles else None
        self._gamble = self._gamble_models.get(season) if self.gambles else None

    def _fit_strategy(self, season):
        """Pit loss, stops and wear per circuit from the dry races before `season`."""
        view = self._archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        races, rows = [], view.sessions[view.sessions.session_type == "R"]
        if self.weekend_wear:       # tyre history from the season's tyre era when it has enough races
            same = rows[rows.season.map(race_sim.tyre_era) == race_sim.tyre_era(season)]
            rows = same if len(same) >= race_sim.MIN_ERA_RACES else rows
        for race in rows.sort_values("scheduled_start_utc").itertuples():
            if race.session_id not in self._stops:
                laps = view.laps(race.session_id)
                record = race_sim.stop_record(laps)
                wear = race_sim.lap_effect_wear if self.lap_effect_wear else race_sim.race_wear
                self._stops[race.session_id] = (record, wear(laps) if record is not None else None)
            record, wear = self._stops[race.session_id]
            races.append((self.pace.circuit(view, race.event_key), record, wear))
        if self.weekend_wear:
            self._wear_models[season] = self._fit_weekend_wear(view, rows, races)
        if self.pit_model:
            self._pit_models[season] = self._fit_pit(view, season)
        if self.team_wear:
            # the spreads and recency weight from the tyre era, or every era while it is young;
            # a team's own pool is always its era's races (`_strategy_inputs`)
            devs = self._era_team_devs(view, season)
            if sum(not d.empty for d in devs) < race_sim.MIN_TEAM_RACES:
                every = view.sessions[view.sessions.session_type == "R"].sort_values("scheduled_start_utc")
                devs = [self._race_team_devs(view, race) for race in every.itertuples()]
            self._team_models[season] = race_sim.TeamWear().fit(devs)
        stops = None
        if self.pit_lane or self.car_strategies:
            stops = race_sim.history_stops(self._archive.root, season)
            if stops is not None:
                stops = {"residuals": stops["residuals"] if self.pit_lane else None,
                         "deviate": stops["deviate"] if self.car_strategies else None}
        return race_sim.StrategyModel().fit(races, stops)

    def _fit_following(self, season):
        """Quantiles of the gaps held-up cars (faster than the car ahead on race
        pace) kept in the dry races before `season`."""
        view = self._archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        gaps = []
        for race in view.sessions[view.sessions.session_type == "R"].itertuples():
            if race.session_id not in self._gaps:
                laps = view.laps(race.session_id)
                pace = race_sim.race_pace(laps)
                pace = dict(zip(pace.driver_number.astype(str), pace.pace)) if not pace.empty else {}
                self._gaps[race.session_id] = race_sim.following_gaps(laps, pace)
                self._runs[race.session_id] = race_sim.held_run_lengths(laps, pace)
            gaps.append(self._gaps[race.session_id])
        table = race_sim.follow_quantiles(np.concatenate(gaps) if gaps else np.zeros(0))
        if table is None:
            return None
        runs = np.concatenate([self._runs[r.session_id] for r in view.sessions[view.sessions.session_type == "R"].itertuples()])
        return table, (1.0 / float(runs.mean()) if len(runs) else 0.0)   # a hold ends on average after its mean length

    def _fit_penalty(self, season):
        """The following penalty of cars not held up, from the dry races before `season`."""
        view = self._archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        for race in view.sessions[view.sessions.session_type == "R"].itertuples():
            if race.session_id not in self._penalties:
                laps = view.laps(race.session_id)
                pace = race_sim.race_pace(laps)
                pace = dict(zip(pace.driver_number.astype(str), pace.pace)) if not pace.empty else {}
                self._penalties[race.session_id] = race_sim.following_penalty(laps, pace)
        races = [self._penalties[r.session_id] for r in view.sessions[view.sessions.session_type == "R"].itertuples()]
        return race_sim.pooled_penalty(races)

    def _fit_windows(self, season):
        """Learned pit windows from the dry races before `season`."""
        view = self._archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        races = view.sessions[view.sessions.session_type == "R"]
        for race in races.itertuples():
            if race.session_id not in self._windows:
                self._windows[race.session_id] = race_sim.stop_windows(view.laps(race.session_id))
        return race_sim.pooled_windows([self._windows[r.session_id] for r in races.itertuples()])

    def _fit_gambles(self, season):
        """The gamble model from the dry races before `season`: each finisher's stop
        count against its race's most common, by grid band and its team's expected
        wear before that race (the season's team wear model on the tyre era's races
        before it)."""
        view = self._archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        races = view.sessions[view.sessions.session_type == "R"].sort_values("scheduled_start_utc")
        team_model = self._team_models.get(season) if self.gambles != "grid" else None
        frames = []
        for race in races.itertuples():
            # keyed by the target season too: a row's team wear comes from that season's team wear model
            key = (season, race.session_id)
            if key not in self._gamble_rows:
                results = view.results(race.session_id)
                grid = ({} if "GridPosition" not in results else
                        {d: float(g) for d, g in zip(results.DriverNumber.astype(str),
                                                     pd.to_numeric(results.GridPosition, errors="coerce")) if pd.notna(g)})
                wear = {}
                if team_model is not None:
                    entrants = view.entrants(race.session_id)
                    numbers = list(entrants.driver_number.astype(str))
                    teams = team_model.predict(list(entrants.lineage_id),
                                               self._era_team_devs_before(view, race))
                    if teams is not None:
                        index, mean, _ = teams
                        wear = {d: float(mean[i]) for d, i in zip(numbers, index)}
                self._gamble_rows[key] = race_sim.gamble_rows(view.laps(race.session_id), grid, wear)
            frames.append(self._gamble_rows[key])
        frames = [f for f in frames if f is not None]
        return race_sim.GambleModel().fit(pd.concat(frames, ignore_index=True) if frames else None)

    def _era_team_devs_before(self, view, race):
        """Team wear deviations of the races of `race`'s tyre era before it, in order."""
        season = int(race.season)
        rows = view.sessions[(view.sessions.session_type == "R")
                             & (view.sessions.scheduled_start_utc < race.scheduled_start_utc)]
        rows = rows[rows.season.map(race_sim.tyre_era) == race_sim.tyre_era(season)]
        return [self._race_team_devs(view, r) for r in rows.sort_values("scheduled_start_utc").itertuples()]

    def _fit_roles(self, season):
        """Starting tyre roles from the dry races before `season` of its tyre era."""
        view = self._archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        races = view.sessions[(view.sessions.session_type == "R")
                              & (view.sessions.season.astype(int).map(race_sim.tyre_era) == race_sim.tyre_era(season))]
        for race in races.itertuples():
            if race.session_id not in self._roles:
                results = view.results(race.session_id)
                grid = ({} if "GridPosition" not in results else
                        dict(zip(results.DriverNumber.astype(str), pd.to_numeric(results.GridPosition, errors="coerce"))))
                self._roles[race.session_id] = race_sim.start_roles(view.laps(race.session_id), grid)
        return race_sim.pooled_roles([self._roles[r.session_id] for r in races.itertuples()],
                                     self._window_tables.get(season))

    def _fit_peak(self, season):
        """The fresh-tyre peak from the dry races before `season` of its tyre era."""
        view = self._archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        races = view.sessions[(view.sessions.session_type == "R")
                              & (view.sessions.season.astype(int).map(race_sim.tyre_era) == race_sim.tyre_era(season))]
        for race in races.itertuples():
            if race.session_id not in self._peaks:
                self._peaks[race.session_id] = race_sim.fresh_peak(view.laps(race.session_id))
        return race_sim.pooled_fresh_peak([self._peaks[r.session_id] for r in races.itertuples()])

    def _pass_rate(self, view, race):
        if race.session_id not in self._pass_rates:
            self._pass_rates[race.session_id] = race_sim.log_pass_rate(view.laps(race.session_id))
        return self._pass_rates[race.session_id]

    def _circuit_passing(self, view, circuit, exclude):
        """Mean log pass rate of the circuit's dry races readable in `view` other than
        `exclude` (an event key); None without any."""
        races = view.sessions[(view.sessions.session_type == "R") & (view.sessions.event_key != exclude)]
        rates = [self._pass_rate(view, r) for r in races.itertuples() if self.pace.circuit(view, r.event_key) == circuit]
        rates = [r for r in rates if r is not None]
        return float(np.mean(rates)) if rates else None

    def _fit_pit(self, view, season):
        """The pit hazard from the dry races before `season` in its tyre era (every
        era while that has fewer than MIN_PIT_RACES)."""
        races = view.sessions[view.sessions.session_type == "R"]
        same = races[races.season.map(race_sim.tyre_era) == race_sim.tyre_era(season)]
        races = same if len(same) >= race_sim.MIN_PIT_RACES else races
        data = []
        for race in races.sort_values("scheduled_start_utc").itertuples():
            if race.session_id not in self._pit_rows:
                laps = view.laps(race.session_id)
                record = race_sim.stop_record(laps)
                loss = float(np.median(record["green"])) if record and len(record["green"]) >= 4 else race_sim.PIT_LOSS
                pace = race_sim.race_pace(laps)
                pace = dict(zip(pace.driver_number.astype(str), pace.pace)) if not pace.empty else {}
                ratio = (float(np.median(record["neutral"]) / np.median(record["green"]))
                         if record and record["neutral"] and len(record["green"]) >= 4 else race_sim.SC_PIT_RATIO)
                self._pit_rows[race.session_id] = race_sim.pit_decisions(laps, loss, pace, race_sim.race_wear(laps), ratio)
            rate = self._circuit_passing(view, self.pace.circuit(view, race.event_key), race.event_key)
            data.append((self._pit_rows[race.session_id], np.nan if rate is None else rate))
        return race_sim.PitModel().fit(data)

    def _race_team_devs(self, view, race):
        """A race's team wear deviations (cached; a dry race's laps are final)."""
        if race.session_id not in self._team_devs:
            entrants = view.entrants(race.session_id)
            lineage_of = dict(zip(entrants.driver_number.astype(str), entrants.lineage_id))
            self._team_devs[race.session_id] = race_sim.team_wear_deviations(view.laps(race.session_id), lineage_of)
        return self._team_devs[race.session_id]

    def _era_team_devs(self, view, season, exclude=None):
        """Team wear deviations of the races readable in `view` in `season`'s tyre
        era, in order (`exclude`: an event key left out)."""
        races = view.sessions[(view.sessions.session_type == "R") & (view.sessions.event_key != exclude)]
        races = races[races.season.map(race_sim.tyre_era) == race_sim.tyre_era(season)]
        return [self._race_team_devs(view, race) for race in races.sort_values("scheduled_start_utc").itertuples()]

    def _practice_wear(self, view, event_key):
        """Field wear of this event's dry practice long runs readable in `view`
        (sessions weighted by their stints)."""
        practice = view.sessions[(view.sessions.event_key == event_key)
                                 & view.sessions.session_type.isin(("FP1", "FP2", "FP3"))]
        values = []
        for session in practice.session_id:
            if session not in self._practice:
                self._practice[session] = race_sim.practice_wear(view.laps(session))
            if self._practice[session] is not None:
                values.append(self._practice[session])
        return float(np.average([w for w, _ in values], weights=[n for _, n in values])) if values else None

    def _fit_weekend_wear(self, view, rows, races):
        """Calibrate race wear on practice wear and the circuit's past wear over the
        earlier races (in order: each race's past is the races before it)."""
        typical = [w for _, r, w in races if r is not None and w is not None and np.isfinite(w)]
        typical = float(np.median(typical)) if typical else race_sim.WEAR
        seen, pairs = {}, []
        for race, (circuit, record, wear) in zip(rows.sort_values("scheduled_start_utc").itertuples(), races):
            if record is not None and wear is not None and np.isfinite(wear):
                past = race_sim.shrunk_mean(seen.get(circuit, []), typical)
                pairs.append((self._practice_wear(view, race.event_key), past, wear))
                seen.setdefault(circuit, []).append(wear)
        return race_sim.WeekendWear().fit(pairs)

    def _race_periods(self, view, race):
        if race.session_id not in self._periods:
            table = view.laps(race.session_id)
            periods = race_sim.neutral_periods(table)
            self._periods[race.session_id] = None if periods is None else (int(table.LapNumber.max()), periods)
        return self._periods[race.session_id]

    def _weekend_qualifying(self, view, event_key):
        """This event's dry qualifying median best lap if classified at the cutoff."""
        q = view.sessions[(view.sessions.event_key == event_key) & (view.sessions.session_type == "Q")]
        if q.empty:
            return None
        session = q.session_id.iloc[0]
        if session not in self._qualifying:
            self._qualifying[session] = race_sim.qualifying_lap(view.laps(session))
        return self._qualifying[session]

    def _fit_ratio(self, season):
        """Median green race lap / median best qualifying lap over the races before
        `season`; RACE_TO_QUALIFYING without any."""
        view = self._archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        ratios = []
        for race in view.sessions[view.sessions.session_type == "R"].itertuples():
            _, green = self._shape(view, race)
            qualifying = self._weekend_qualifying(view, race.event_key)
            if green is not None and qualifying:
                ratios.append(green / qualifying)
        return float(np.median(ratios)) if ratios else race_sim.RACE_TO_QUALIFYING

    def _fit_neutral(self, season):
        """The neutralisation model for `season`, from the races before it."""
        view = self._archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        races = view.sessions[view.sessions.session_type == "R"]
        recent, rows = [], []
        for race in races.itertuples():
            shape = self._race_periods(view, race)
            if shape is None:
                continue
            laps, periods = shape
            recent.append((int(race.season), laps, periods))
            rows.append((int(race.season), self.pace.circuit(view, race.event_key), laps, len(periods)))
        history = pd.concat([race_sim.history_neutralisations(self._archive.root, view.cutoff),
                             pd.DataFrame(rows, columns=["season", "circuit", "laps", "n"])], ignore_index=True)
        return race_sim.NeutralModel().fit(history, recent)

    def _forecast_deltas(self, frame, term, season, cutoff):
        """The battle pairs of races forecast from `cutoff` in earlier seasons, each
        repeated BATTLE_DRAWS times with the pace difference of one forecast draw;
        (frame, term, weights)."""
        parts, terms = [], []
        for example in (e for e in self._history if e.season < season and e.cutoff_label == cutoff):
            rows = np.flatnonzero((frame.event_key == example.event_key).to_numpy())
            if not len(rows):
                continue
            key = (example.event_key, example.cutoff_label, season)
            if key not in self._draws:
                view = self._view_for(example)
                self._draws[key] = self._fit_inputs(view, example)
            race = self._draws[key][0]
            column = {str(n): i for i, n in enumerate(example.driver_numbers)}
            pairs = frame.iloc[rows]
            known = (pairs.ahead.map(column).notna() & pairs.behind.map(column).notna()).to_numpy()
            pairs, rows = pairs[known], rows[known]
            if pairs.empty:
                continue
            ahead = pairs.ahead.map(column).to_numpy(int)
            behind = pairs.behind.map(column).to_numpy(int)
            pick = np.random.default_rng([fit_seed(example, self.stable_fit), 11]).integers(0, len(race), (len(pairs), BATTLE_DRAWS))
            delta = (race[pick, behind[:, None]] - race[pick, ahead[:, None]]) * pairs.lap.to_numpy(float)[:, None] / 100
            parts.append(pd.DataFrame({"gap": np.repeat(pairs.gap.to_numpy(float), BATTLE_DRAWS),
                                       "delta": delta.ravel(),
                                       "passed": np.repeat(pairs.passed.to_numpy(bool), BATTLE_DRAWS)}))
            terms.append(np.repeat(term[rows], BATTLE_DRAWS))
        if not parts:
            return frame.iloc[:0], term[:0], None
        expanded = pd.concat(parts, ignore_index=True)
        return expanded, np.concatenate(terms), np.full(len(expanded), 1.0 / BATTLE_DRAWS)

    def prepare_season(self, season, cutoff):
        """The battle model for `season` at `cutoff`, from the races before the
        season; each pair's circuit term from the other seasons' battles. Also the
        neutralisation model for the season."""
        if self.neutralisations and season not in self._neutral_models:
            self._neutral_models[season] = self._fit_neutral(season)
        if self.weekend_lap and season not in self._ratios:
            self._ratios[season] = self._fit_ratio(season)
        if self.following and season not in self._follow_tables:
            self._follow_tables[season] = self._fit_following(season)
        if self.follow_penalty and season not in self._penalty_tables:
            self._penalty_tables[season] = self._fit_penalty(season)
        if self.fresh_peak and season not in self._peak_tables:
            self._peak_tables[season] = self._fit_peak(season)
        if self.windows and season not in self._window_tables:
            self._window_tables[season] = self._fit_windows(season)
        if self.start_roles and season not in self._role_tables:
            self._role_tables[season] = self._fit_roles(season)
        if self.strategy and season not in self._strategies:
            self._strategies[season] = self._fit_strategy(season)
        if self.gambles and season not in self._gamble_models:      # after the strategy: needs the team wear model
            self._gamble_models[season] = self._fit_gambles(season)
        self._activate(season)
        view = self._archive.view(pd.Timestamp(f"{season}-01-01", tz="UTC"))
        frame = self._battles_in(view)
        term = np.zeros(len(frame))
        if not frame.empty:
            for year in frame.season.unique():
                other = frame[frame.season != year]
                if other.empty:
                    continue
                propensity = race_sim.circuit_propensity(other, float(other.passed.mean()))
                rows = (frame.season == year).to_numpy()
                term[rows] = frame[rows].circuit.map(propensity).fillna(0.0).to_numpy(float)
        weights = None
        if self.forecast_battles:
            frame, term, weights = self._forecast_deltas(frame, term, season, cutoff)
        self._pass_models[(season, cutoff)] = self.passes = race_sim.PassModel().fit(frame, term, weights)

    def context(self, view, example):
        """The circuit's pass propensity; the scheduled race length (`scheduled_laps`)
        and a typical green lap from the latest earlier dry race with enough racing
        laps, on the same layout if there is one."""
        frame = self._battles_in(view, exclude=example.event_key)
        circuit = self.pace.circuit(view, example.event_key)
        term = 0.0
        if not frame.empty and circuit is not None:
            term = float(race_sim.circuit_propensity(frame, float(frame.passed.mean())).get(circuit, 0.0))
        laps, lap = race_sim.DEFAULT_LAPS, race_sim.DEFAULT_LAP
        races = view.sessions[(view.sessions.session_type == "R") & (view.sessions.event_key != example.event_key)]
        same = sorted((r for r in races.itertuples() if circuit is not None
                       and self.pace.circuit(view, r.event_key) == circuit), key=lambda r: r.scheduled_start_utc)
        if same:
            layout = track.layout_of(circuit, int(example.season), example.event_key)
            shapes = [(track.layout_of(circuit, int(r.season), r.event_key), *self._shape(view, r)) for r in same]
            laps = scheduled_laps([(name, completed) for name, completed, _ in shapes], layout) or laps
            green = [seconds for name, _, seconds in shapes if seconds is not None]
            green_same = [seconds for name, _, seconds in shapes if seconds is not None and layout is not None
                          and name == layout]
            lap = (green_same or green or [lap])[-1]
        if self.weekend_lap:
            qualifying = self._weekend_qualifying(view, example.event_key)
            if qualifying:
                lap = qualifying * self._ratio
        context = {"circuit_term": term, "laps": laps, "lap_seconds": lap}
        if self.neutralisations and self._neutral is not None:
            context["neutral"] = self._neutral.race(circuit, laps)
        if self.strategy and self._strategy is not None:
            context["strategy"] = self._strategy_inputs(view, example, circuit, lap)
        return context

    def _strategy_inputs(self, view, example, circuit, lap):
        """The active strategy model's inputs for this race, its tyre wear from this
        weekend's practice under the active wear calibration when there is one."""
        strategy = self._strategy.race(circuit, lap)
        if self._wear is not None:
            past = self._strategy.circuits.get(circuit, {}).get("wear", self._strategy.wear)
            weekend = self._wear.predict(self._practice_wear(view, example.event_key), past)
            if weekend is not None:
                strategy["wear"] = weekend * lap / 100
        if self._pit is not None and self._pit.coef is not None:
            rate = self._circuit_passing(view, circuit, example.event_key)
            strategy.update(pit_coef=self._pit.coef, passing=0.0 if rate is None else rate - self._pit.centre)
        if self._gamble is not None:
            strategy["gambles"] = self._gamble
            strategy["gamble_wear"] = self.gambles != "grid"
            strategy["gamble_by_pace"] = self.gambles == "pace"   # the band of the car's mean race pace rank
        if self._team is not None:
            n = len(example.driver_numbers)
            lineages = list(example.lineage_ids) if len(example.lineage_ids) == n else [None] * n
            teams = self._team.predict(lineages, self._era_team_devs(view, int(example.season), example.event_key))
            if teams is not None:
                index, mean, sd = teams
                strategy.update(team=np.asarray(index), team_mean=mean * lap / 100, team_sd=sd * lap / 100)
        return strategy

    def _shape(self, view, race):
        if race.session_id not in self._race_shape:
            self._race_shape[race.session_id] = race_sim.race_shape(view.laps(race.session_id))
        return self._race_shape[race.session_id]

    def order(self, race, grid, setting, context, seed):
        if setting["sigma"] > 0:
            race = race + setting["sigma"] * np.random.default_rng([seed, 8]).standard_normal(race.shape)
        return race_sim.simulate(race, grid, self.passes, context["circuit_term"], context["laps"],
                                 context["lap_seconds"], setting["start_noise"], np.random.default_rng([seed, 10]),
                                 setting.get("pass_shift", 0.0), context.get("neutral"),
                                 self._race_strategy(context.get("strategy"), setting),
                                 follow=self._follow, follow_release=self._release, follow_penalty=self._penalty,
                                 fresh_peak=self._peak, windows=self._window, roles=self._role, keyed=self.stable_fit)

    @staticmethod
    def _race_strategy(strategy, setting):
        """The race's strategy inputs under `setting`: its stint noise, and its
        pit_noise share of the slow-stop residuals and per-car stop changes."""
        if not strategy:
            return None
        out = {**strategy, "stint_noise": setting.get("stint_noise", race_sim.STINT_NOISE)}
        if "pit_noise" in setting:
            share = float(setting["pit_noise"])
            out["residuals"] = np.asarray(strategy["residuals"], dtype=float) * share
            if strategy.get("deviate") is not None:
                out["deviate"] = np.asarray(strategy["deviate"], dtype=float) * share
            if strategy.get("gambles") is not None:
                out["gamble_share"] = share
        return out
