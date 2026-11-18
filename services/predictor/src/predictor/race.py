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
  in the same draw. Grid penalties are not modelled yet.
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
import numpy as np
import pandas as pd

from .baselines import RetirementModel, seed_for
from .forecast import Forecast
from .laps import race_passes
from .pace import knockout_positions
from . import race_sim, track

KAPPAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.4)
DRAWS = 10000
FIT_DRAWS = 1000
GRID_FROM_SPRINT = frozenset({2021, 2022})   # seasons whose sprint set the race grid
GAMMAS = (0.0, 0.5, 1.0, 1.5, 2.0)
PASS_PRIOR_RACES = 1.0      # weight of the typical pass rate in a circuit's, in races


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
    values = np.array([order.get(str(n), np.nan) for n in example.driver_numbers], dtype=float)
    values[np.isnan(values)] = np.inf
    slots = np.empty(len(values))
    slots[np.argsort(values, kind="stable")] = np.arange(1, len(values) + 1)
    return slots


class RaceModel:
    """Model interface: fit(history, view_for) then predict(view, example). Races only."""

    def __init__(self, pace, name, overtaking=False):
        self.pace = pace
        self.name = name
        self.overtaking = overtaking
        self._passes = {}           # race session -> race_passes(...)
        self.retirements = RetirementModel()
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
        grid = grid_at(view, example)
        if grid is None:
            grid = knockout_positions(qualifying, pace.hyper.noise["Q"], np.random.default_rng([seed, 6]), dof=dof,
                                      scale_rng=np.random.default_rng([seed, 7]))
        return race, np.atleast_2d(grid).astype(float)

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

    def prepare_season(self, season):
        """Anything fitted per season before the settings (nothing here)."""

    def context(self, view, example):
        """What a forecast needs from the archive besides the draws."""
        return {"difficulty": self.difficulty(view, example) if self.overtaking else 1.0}

    def order(self, race, grid, setting, context, seed):
        kappa = setting["kappa"] * context["difficulty"] ** setting.get("gamma", 0.0)
        return self.positions(race, grid, kappa, setting["sigma"], seed)

    def positions(self, race, grid, kappa, sigma, seed):
        """Finishing order before retirements, draws x entrants (1 = first)."""
        score = race + kappa * (grid - 1)
        if sigma > 0:
            score = score + sigma * np.random.default_rng([seed, 8]).standard_normal(race.shape)
        order = np.argsort(score, axis=1)
        positions = np.empty_like(order)
        positions[np.arange(len(score))[:, None], order] = np.arange(1, score.shape[1] + 1)
        return positions

    def forecast(self, example, positions, seed, metadata):
        rng = np.random.default_rng([seed, 9])
        final, outcomes = self.retirements.apply(positions, example, rng)
        return Forecast.from_draws(
            positions=final, outcomes=outcomes, season=example.season, session_type=example.session_type,
            event_key=example.event_key, cutoff_label=example.cutoff_label, cutoff_time=example.cutoff_time.isoformat(),
            driver_numbers=list(example.driver_numbers), driver_ids=list(example.driver_ids), model=self.name,
            seed=seed, metadata=metadata)

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
                self._draws[key] = (*self.draws(view, example, FIT_DRAWS, seed_for(example)),
                                    self.context(view, example))
            prepared.append((example, *self._draws[key]))
        best = None
        for setting in self.candidates():
            total = 0.0
            for example, race, grid, context in prepared:
                seed = seed_for(example)
                forecast = self.forecast(example, self.order(race, grid, setting, context, seed), seed, {})
                total += score(forecast, example)["rps"]
            rps = total / len(prepared)
            if best is None or rps < best[0]:
                best = (rps, setting)
        self.settings[(season, cutoff)] = {**best[1], "rps": best[0], "races": len(prepared)}
        self._draws = {}             # tied to this season's pace settings

    def predict(self, view, example, draws=DRAWS):
        if example.session_type != "R":
            return None
        self.pace._ensure_hyper(view, example.season)
        self._archive = view.archive
        key = (example.season, example.cutoff_label)
        if key not in self.settings:
            self.prepare_season(example.season)
            self._fit_season(*key)
        chosen = self.settings[key]
        seed = seed_for(example)
        race, grid = self.draws(view, example, draws, seed)
        context = self.context(view, example)
        positions = self.order(race, grid, chosen, context, seed)
        known_grid = grid.shape[0] == 1
        return self.forecast(example, positions, seed, {
            "grid": "known" if known_grid else "simulated", "context": context,
            "hyper": {"race": chosen, "noise": dict(self.pace.hyper.noise), "fit": self.pace.fit_log.get(example.season)},
            "retire_rate": self.retirements.retire_rate, "mechanical_share": self.retirements.mechanical_share})


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


class SimulatedRace(RaceModel):
    """Stage 2: the race run lap by lap (`race_sim.simulate`): start, racing laps,
    passes from the battle model. Fitted per season: the battle model on the dry
    races of earlier seasons; then lap-1 noise and race-day noise by mean RPS."""

    def __init__(self, pace, name, pass_shift=False):
        super().__init__(pace, name)
        self.pass_shift = pass_shift
        self.passes = race_sim.PassModel()
        self._battles, self._race_shape = {}, {}
        self._archive = None

    def fit(self, history, view_for):
        super().fit(history, view_for)
        if history and self._archive is None:
            self._archive = view_for(history[0]).archive
        return self

    def candidates(self):
        for shift in (PASS_SHIFTS if self.pass_shift else (0.0,)):
            for start_noise in START_NOISES:
                for sigma in SIM_SIGMAS:
                    yield {"start_noise": start_noise, "sigma": sigma, **({"pass_shift": shift} if self.pass_shift else {})}

    def neutral(self):
        return {"start_noise": 1.0, "sigma": 0.0}

    def _race_battles(self, view, race):
        if race.session_id not in self._battles:
            frame = race_sim.battles(view.laps(race.session_id))
            self._battles[race.session_id] = frame.assign(circuit=self.pace.circuit(view, race.event_key),
                                                          season=int(race.season))
        return self._battles[race.session_id]

    def _battles_in(self, view, exclude=None):
        races = view.sessions[(view.sessions.session_type == "R") & (view.sessions.event_key != exclude)]
        frames = [self._race_battles(view, race) for race in races.itertuples()]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["gap", "delta", "passed", "circuit", "season"])

    def prepare_season(self, season):
        """The battle model from the races before `season`; each pair's circuit term
        from the other seasons' battles."""
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
        self.passes.fit(frame, term)

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
        return {"circuit_term": term, "laps": laps, "lap_seconds": lap}

    def _shape(self, view, race):
        if race.session_id not in self._race_shape:
            self._race_shape[race.session_id] = race_sim.race_shape(view.laps(race.session_id))
        return self._race_shape[race.session_id]

    def order(self, race, grid, setting, context, seed):
        if setting["sigma"] > 0:
            race = race + setting["sigma"] * np.random.default_rng([seed, 8]).standard_normal(race.shape)
        return race_sim.simulate(race, grid, self.passes, context["circuit_term"], context["laps"],
                                 context["lap_seconds"], setting["start_noise"], np.random.default_rng([seed, 10]),
                                 setting.get("pass_shift", 0.0))
