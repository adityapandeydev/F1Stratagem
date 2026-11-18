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
"""
import numpy as np
import pandas as pd

from .baselines import RetirementModel, seed_for
from .forecast import Forecast
from .pace import knockout_positions

KAPPAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.4)
DRAWS = 10000
FIT_DRAWS = 1000
GRID_FROM_SPRINT = frozenset({2021, 2022})   # seasons whose sprint set the race grid


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

    def __init__(self, pace, name):
        self.pace = pace
        self.name = name
        self.retirements = RetirementModel()
        self.settings = {}          # season -> {"kappa", "sigma", "rps"}
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

    def _fit_season(self, season):
        """kappa and sigma for `season`, from the races of earlier seasons in the history."""
        from .evaluation import score
        earlier = [e for e in self._history if e.season < season]
        if not earlier:
            self.settings[season] = {"kappa": 0.0, "sigma": 0.0, "rps": None, "races": 0}
            return
        prepared = []
        for example in earlier:
            key = (example.event_key, example.cutoff_label, season)
            if key not in self._draws:
                self._draws[key] = self.draws(self._view_for(example), example, FIT_DRAWS, seed_for(example))
            prepared.append((example, *self._draws[key]))
        best = None
        for kappa in KAPPAS:
            for sigma in SIGMAS:
                total = 0.0
                for example, race, grid in prepared:
                    seed = seed_for(example)
                    forecast = self.forecast(example, self.positions(race, grid, kappa, sigma, seed), seed, {})
                    total += score(forecast, example)["rps"]
                rps = total / len(prepared)
                if best is None or rps < best[0]:
                    best = (rps, kappa, sigma)
        self.settings[season] = {"kappa": best[1], "sigma": best[2], "rps": best[0], "races": len(prepared)}
        self._draws = {}             # tied to this season's pace settings

    def predict(self, view, example, draws=DRAWS):
        if example.session_type != "R":
            return None
        self.pace._ensure_hyper(view, example.season)
        if example.season not in self.settings:
            self._fit_season(example.season)
        chosen = self.settings[example.season]
        seed = seed_for(example)
        race, grid = self.draws(view, example, draws, seed)
        positions = self.positions(race, grid, chosen["kappa"], chosen["sigma"], seed)
        known_grid = grid.shape[0] == 1
        return self.forecast(example, positions, seed, {
            "grid": "known" if known_grid else "simulated",
            "hyper": {"race": chosen, "noise": dict(self.pace.hyper.noise), "fit": self.pace.fit_log.get(example.season)},
            "retire_rate": self.retirements.retire_rate, "mechanical_share": self.retirements.mechanical_share})
