"""Walk-forward evaluation: score forecasts against the final classification.

For every event in the evaluated seasons, a model is fitted on events whose
target session finished before the event's cutoff, then forecasts the target
session from data available at the cutoff. Seasons 2025–2026 are held out until
model selection (see targets.HOLDOUT_SEASONS).

Metrics, each averaged over scored entrants (excluded entrants are not scored):

    winner_log_loss    -log p(actual winner); pole for qualifying
    rps                ranked probability score of the position distribution
    brier_top3/top10   Brier score of finishing in the top 3 / top 10
    position_mae       |expected position - actual position|, overall and by band
                       (bands by championship rank at the cutoff: 1-5, 6-12, 13+)
    coverage_80        share of actual positions inside the central 80% interval
    interval_mass      probability the forecast itself puts inside that interval;
                       calibration is the gap between coverage and mass
    interval_width     mean width of that interval, in positions
    outcome_log_loss   five-way outcome, entrants with a known outcome only
    retirement_brier   any retirement (cause known or not)
"""

import json
import subprocess

from dataclasses import replace

import numpy as np
import pandas as pd

from .archive import Archive
from .baselines import RANKERS, Example, RankBaseline, dense_ranks, rank_championship
from .targets import HOLDOUT_SEASONS, OUTCOMES, RETIREMENTS, session_target

EPS = 1e-12  # safety only; forecasts carry a prior pseudo-draw, so no probability is zero
BANDS = ((1, 5, "front"), (6, 12, "midfield"), (13, 99, "back"))
ID_COLUMNS = ("event_key", "season", "model", "served_by")   # served_by: a router's champion of the race


def score(forecast, example):
    keep = ~example.excluded
    actual = example.positions[keep]
    p = forecast.position_probs[keep]
    n = forecast.position_probs.shape[1]
    winners = np.flatnonzero((example.positions == 1) & ~example.excluded)
    winner = int(winners[0]) if len(winners) else None     # None: the winner was not forecast (late substitute)
    cdf = np.cumsum(p, axis=1)
    observed = (np.arange(1, n + 1)[None, :] >= np.minimum(actual, n)[:, None]).astype(float)
    lower, upper = forecast.interval()
    lower, upper = lower[keep], upper[keep]
    inside = (np.arange(1, n + 1)[None, :] >= lower[:, None]) & (np.arange(1, n + 1)[None, :] <= upper[:, None])
    expected = forecast.expected_position[keep]
    band_rank = example.band_rank[keep]
    row = {
        "winner_log_loss": -np.log(max(forecast.position_probs[winner, 0], EPS)) if winner is not None else np.nan,
        "rps": float(np.mean(np.sum((cdf - observed) ** 2, axis=1) / (n - 1))),
        "brier_top3": float(np.mean((forecast.top(3)[keep] - (actual <= 3)) ** 2)),
        "brier_top10": float(np.mean((forecast.top(10)[keep] - (actual <= 10)) ** 2)),
        "position_mae": float(np.mean(np.abs(expected - actual))),
        "coverage_80": float(np.mean((actual >= lower) & (actual <= upper))),
        "interval_mass": float(np.mean(np.sum(p * inside, axis=1))),
        "interval_width": float(np.mean(upper - lower + 1)),
    }
    for low, high, band in BANDS:
        in_band = (band_rank >= low) & (band_rank < high + 1)
        row[f"position_mae_{band}"] = float(np.mean(np.abs(expected - actual)[in_band])) if in_band.any() else np.nan
    if forecast.outcome_probs is not None:
        outcomes = [o for o, x in zip(example.outcomes, example.excluded) if not x]
        probs = forecast.outcome_probs[keep]
        known = [(i, OUTCOMES.index(o)) for i, o in enumerate(outcomes) if o in OUTCOMES]
        row["outcome_log_loss"] = float(np.mean([-np.log(max(probs[i, k], EPS)) for i, k in known])) if known else np.nan
        retired = np.array([o in RETIREMENTS for o in outcomes], dtype=float)
        p_retired = probs[:, OUTCOMES.index("mechanical")] + probs[:, OUTCOMES.index("incident_or_other")]
        row["retirement_brier"] = float(np.mean((p_retired - retired) ** 2))
    return row


def _pace_model(stage, objective="all", weekend=False, shared=False, affinity=False, history=False, form=False,
                skills=(), season_skills=(), races=False, driver_race=False, push=False, long_runs=False,
                race_estimates=False, knockout=False, failures=False, wet_noise=False, robust=False,
                wet_forecast=False, wet_ability=False, wet_races=False, wet_evolution=False,
                wet_segments=False, driver_affinity=False, race_read=None, stint_scale=False):
    from .pace import ALL_SESSIONS, QUALIFYING_ONLY, WITH_RACES, PaceModel
    if stage == 1:
        return PaceModel(QUALIFYING_ONLY, "pace:v1")
    suffix = ("q" if objective == "qualifying" else "") + ("w" if weekend else "") + ("s" if shared else "") \
        + ("a" if affinity else "") + ("h" if history else "") \
        + {"noise": "n", "state": "s", "weighted": "w"}.get(history, "") + ("f" if form else "") + ("k" if skills else "") + ("b" if season_skills else "") \
        + ("r" if races else "") + ("d" if driver_race else "") \
        + ("l" if long_runs else "") + ("u" if race_estimates else "") + ("q" if knockout else "") + ("f" if failures else "") + ("w" if wet_noise else "") + ("t" if robust else "") \
        + ("p" if push else "") + ("n" if push == "own" else "") \
        + ("x" if wet_races else "") + ("e" if wet_evolution else "") + ("g" if wet_segments else "") \
        + ("i" if driver_affinity else "")
    forecast = ("c" if wet_forecast and not wet_ability else "") + ("y" if wet_ability else "")
    model = PaceModel(WITH_RACES if races else ALL_SESSIONS, f"pace:v{stage}{suffix}", winter=stage >= 3, objective=objective,
                     weekend=weekend, shared_from_qualifying=shared, affinity=affinity, history=history, form=form,
                     skills=skills, season_skills=season_skills, driver_offsets=("R",) if driver_race else (),
                     practice_push=push, long_runs=long_runs, race_estimates=race_estimates,
                     knockout=knockout, failures=failures, wet_noise=wet_noise, robust=robust,
                     wet_forecast=wet_forecast, wet_ability=wet_ability, wet_races=wet_races,
                     wet_evolution=wet_evolution, wet_segments=wet_segments, driver_affinity=driver_affinity)
    if race_read:
        # how race laps are read (`laps.race_pace` options); its own fits: the name carries it
        model.race_read = dict(race_read)
        model.name += "-r" + "".join({"lap_effects": "e", "partial": "p"}.get(k, "") for k, v in sorted(race_read.items())
                                     if v is True) + (f"m{race_read['min_laps']}" if "min_laps" in race_read else "")
    if stint_scale:
        # the race estimation covariance scaled by the measured stint factor (pace.stint_scale)
        assert race_estimates, "the stint scale scales the race estimation covariance"
        model.stint_scale = True
        model.name += "s"
    # the wet forecast mode changes no fit: a model with it shares the fits of the one without
    if forecast == "c":
        model.fit_name = model.name
    model.name += forecast
    return model


def _clean_air(model, min_gap):
    from .cleanair import with_clean_air
    return with_clean_air(model, min_gap)


def _corrected(model):
    from .cleanair import with_corrected_pace
    return with_corrected_pace(model)


def _race_model(pace, name, simulated=False, **options):
    from .race import RaceModel, SimulatedRace
    if simulated:
        return SimulatedRace(model_registry()[pace](), name, **options)
    return RaceModel(model_registry()[pace](), name, **options)


def _grid_input(key, suffix, **flags):
    """Registry model `key` (a race model or a mixture of them) with the grid flags of
    race.RaceModel set on it and its components (`published_grid`: the published
    starting grid once out; `known_penalties`: published grid penalties applied until
    then; `split_grid`: qualifying and start slots weighted apart, set on the plain
    race models only: the simulator has no grid term); `suffix` ends its name and its
    components' names."""
    from .race import SimulatedRace
    model = model_registry()[key]()
    for part in getattr(model, "components", [model]):
        for flag, value in flags.items():
            if flag != "split_grid" or not isinstance(part, SimulatedRace):
                setattr(part, flag, value)
        if part is not model:
            part.name += suffix
    model.name += suffix
    return model


# The champion at each forecast moment (per-horizon promotion, design/prediction-engine.md).
# Before the weekend: no session of the event has run. Once any has, the last-cutoff
# champion serves; after FP1 and after FP2 that routing is provisional until those
# cutoffs have been evaluated. A forecast for a wet qualifying (the wet side of the
# dry/wet toggle) is served by the "_wet" champion of its moment: the moment's dry
# champion forecasting with the fitted wet qualifying noise (dry forecasts identical).
# v3wrqwc: wet calibration 0.75-0.79 -> 0.88-0.92 coverage against 0.86 predicted, RPS
# within noise on 15 wet events. v3wrqwtc: wet RPS 0.1370 -> 0.1328 (2019-23) and
# 0.1400 -> 0.1373 (2024) against v3wrqwt, coverage 0.76 -> 0.90 against 0.87.
# After a session the robust models serve too: v3wrqwtc against v3wrqwc at the last
# cutoff -0.00125 [-0.00225, -0.00020] on 2019-23 (15+ better), -0.00226 on 2024
# (15+ not worse), 2024 coverage 0.90 against 0.87 (v3wrqwc 0.93 against 0.87).
# Gate B (2026-11-21, 3 seed batches): race pace keeping a race's largest connected
# group (v3wrqwt-rp). Before the weekend -0.000069 [-0.000149, -0.000001], better
# beyond noise, 15+ and 2024 (-0.000098) fine: promoted. After the session -0.000020,
# noise, nothing worse beyond noise, 2024 -0.000061: kept as a tie (tie-break: races
# with an unseparable driver are used instead of dropped). Gate A (lap effects,
# v3wrqwt-re) was not adopted: +0.000082 after the session, -0.000012 before. The
# wet champion with the same reading (v3wrqwt-rpc), scored on the 15 wet qualifyings:
# +0.000003 [-0.000026, +0.000036] after the session, +0.000027 [-0.000021, +0.000078]
# before (90% bootstrap, 3 batches): level, kept as a tie so every forecast reads race
# pace the same way.
CHAMPIONS = {"pre_weekend": "pace3wrqwtrps", "after_session": "pace3wrqwtrp",
             "pre_weekend_wet": "pace3wrqwtcrp", "after_session_wet": "pace3wrqwtcrp"}

# Race champions (2019-24, 128 races, RPS against race:v1a). Before the weekend the
# lap-by-lap simulation: race:v2a -0.00030 [-0.00049, -0.00011], 15+ within noise,
# 2024 -0.00029. At the last cutoff it is worse beyond noise (+0.00073), and a fitted
# pass shift (race:v2as) only reaches -0.00020 (noise): race:v1a stays.
# Stage 2b (race:v2b, neutralisations) is level with race:v2af at both cutoffs and
# not better than either champion. race:v2bw (+ the lap from this weekend's
# qualifying) is level with v2b at the last cutoff (-0.00004, noise). Retirements by
# circuit and team (v1ar, v2ar, v2bwr) are within noise against their bases at both
# cutoffs (better on 2024 and in retirement Brier); race:v2bwr is the simulation base.
# Stage 3 (tyres and strategy): race:v3f (team tyre wear from race stints) is the
# development simulator; alone it is not better beyond noise than either champion.
# Its 50/50 mixture with race:v2a (race:mix3f, frozen: weights not searched) before
# the weekend: -0.00031 [-0.00051, -0.00011] against race:v2a, 15+ better, every
# season better, 2024 -0.00023: promoted. Its mixture with race:v1a (race:mix3f1a)
# after qualifying: -0.00038 [-0.00073, -0.00003], 15+ within noise, but 2024 level
# (+0.00002): not promoted under the rule (2024 must confirm); race:v1a stays.
# race:v3ff (held-up cars keep a gap of their own) after qualifying: -0.00058 against
# race:v1a (noise), 2024 -0.0019. Its 50/50 mixture with race:v1a (race:mix3ff1a,
# frozen): -0.00055 [-0.00086, -0.00025], 15+ better, 2024 -0.00132 (rule amended
# beforehand: 2024 within +0.0001): promoted, then WITHDRAWN on review: its following
# gaps were measured on 2018-2024 itself (leaking into the evaluated seasons) and a
# held car did not keep its gap. race:v1a served until the corrected race:v3ff passed.
# Corrected (gaps learned per season from earlier races, from held-up cars only; the
# gap kept while held), race:mix3ff1a after qualifying: -0.00047 [-0.00083, -0.00011]
# against race:v1a, 15+ better (-0.00053), 2024 -0.00109: promoted. Most of the gain is
# the mixture (race:v3f with race:v1a: -0.00038); race:v3ff alone is level with v1a
# after qualifying and worse than race:v3f before the weekend (+0.00052). With held cars
# able to leave their gap and attack (re-gated): -0.00060 [-0.00093, -0.00028], 15+
# -0.00065, 2024 -0.00119, winner log loss better: confirmed.
# Gate B (2026-11-21, 3 seed batches, race pace keeping a race's largest connected group
# in every component): before the weekend race:mix3fwp2ap -0.000099 [-0.000196,
# -0.000010], outcome and winner log loss better beyond noise, 15+ -0.000094 (noise),
# 2024 -0.00019: promoted. After qualifying race:mix3ffnp1ap -0.000045 [-0.000096,
# +0.000004] (noise), outcome and winner better beyond noise, 15+ -0.000006, 2024
# -0.000015: kept as a tie. Gate A (lap effects) was not adopted: 2024 +0.000104
# before the weekend, +0.00023 after qualifying (limit +0.0001).
# Gate C (2026-11-21, 3 seed batches): race pace with its measured uncertainty
# (estimation covariance x pace.STINT_SCALE) and 5 clean laps enough. After qualifying
# race:mix3ffnps51aps5 -0.000194 [-0.000333, -0.000049], winner log loss better beyond
# noise, 15+ -0.000256 (better), 2024 -0.00037: promoted. Before the weekend
# race:mix3fwps52aps5 +0.000071 (noise), outcome and winner log loss worse beyond noise:
# not adopted. On qualifying it was not adopted at either cutoff (15+ worse).
# Rerun with the corrected scale (2026-11-22, split-driver measurement, about 7 in
# place of 4.4): after qualifying race:mix3ffnps51aps5 -0.000340 [-0.000519,
# -0.000162], winner better beyond noise, 15+ -0.00043 (better), 2024 -0.00067:
# confirmed. Before the weekend race:mix3fwps52aps5 -0.000013 (noise), outcome and
# winner log loss worse beyond noise: not adopted. Five clean laps admitted with the
# independent-lap covariance only (no scale; race:mix3ffnpm51apm5, mix3fwpm52apm5):
# +0.000032 (noise, winner worse) after qualifying, +0.000146 (worse) before the
# weekend: not adopted. Qualifying: after the session every stint-scale variant is
# worse beyond noise (v3wruqwt-rps +0.000045) and admission alone (v3wrqwt-rpm5,
# +0.000026) is worse at 15+; before the weekend v3wruqwt-rps is level (-0.000010,
# 15+ noise, 2024 -0.000018, winner log loss better beyond noise): not adopted, the
# tie-break left to the owner; adopted by the owner (2026-11-22) as a tie: the forecast
# made before the weekend reads race pace with its measured uncertainty, as the race
# forecast after qualifying does. The wet champion is unchanged (scale measured dry).
# Stage 4 (2026-11-23, 3 seed batches, after the review fixes: withdrawn cars sit out
# the simulation, a replaced decision counts until its replacement is published).
# Published starting grid at the 'after_grid' cutoff (race_mix3ffnps51aps5g, grid
# changed in 67 races): -0.000179 [-0.000561, +0.000203] (noise, within margin), 15+
# -0.000089, 2024 -0.000002, outcome and winner log loss level. Stewards' grid
# penalties known before the weekend (race_mix3fwp2apk, 12 races changed): -0.000011
# [-0.000043, +0.000014], winner log loss -0.00089 (block interval better, plain
# interval touches 0), 15+ +0.000003, 2024 +0.000015. Both adopted by the owner as
# ties (published facts in place of assumptions), never as gains. Penalties known at
# the last cutoff after qualifying (race_mix3ffnps51aps5k, 47 races changed): +0.000236
# [-0.000124, +0.000626], 2024 +0.00095 (over the limit): not adopted, under diagnosis;
# its 81 races with unchanged grids are worse (+0.00018 beyond noise).
# Settings fit re-baseline (2026-11-24, 3 seed batches): the stable fit (race.RaceModel.stable_fit)
# passes before the weekend, -0.000073 [-0.000120, -0.000030], 15+ -0.000100, 2024 -0.000091,
# and is served there; after qualifying (+0.000202, 15+ worse, 2024 +0.00042) and once the
# grid is published (+0.000259, winner and 15+ worse, 2024 +0.00042) it fails, so those keep
# the fit of before 2026-11-23 (the "o" variants).
# Step 9b (2026-11-26, 3 seed batches): the simple half with a fitted race-score bonus for
# the cars starting 1-3 (race_mix3ffnps51aps5of), at the last cutoff before the race:
# RPS -0.000133 [-0.000232, -0.000032], 15+ -0.00008 (noise), 2024 -0.000127; the
# mixture's outcome log loss -0.00063 [-0.00165, +0.00046] (pre-registered limit +0.0025).
# Tested at that cutoff only, with the grid classified (all 128 races), so it serves 'last'
# alone and only with an observed grid: forecasts at an earlier session cutoff, or at 'last'
# before the grid-setting session is classified (the bonus would act on a simulated grid),
# keep the 'after_session' champion.
RACE_CHAMPIONS = {"pre_weekend": "race_mix3fwp2apk", "after_session": "race_mix3ffnps51aps5o",
                  "last": "race_mix3ffnps51aps5of", "after_grid": "race_mix3ffnps51aps5go"}


class ChampionRouter:
    """Serves each forecast with the champion of its forecast moment: one model
    instance per champion, fitted and asked only for the examples it serves."""

    def __init__(self, champions=None, registry=None, name="champion"):
        self.name = name
        self.champions = dict(champions or CHAMPIONS)
        registry = registry or model_registry()
        instances = {name: registry[name]() for name in dict.fromkeys(self.champions.values())}
        self.models = {key: instances[name] for key, name in self.champions.items()}   # one fit per model

    @property
    def fit_cache(self):
        return getattr(self.models["pre_weekend"], "fit_cache", None)

    @fit_cache.setter
    def fit_cache(self, cache):
        for model in self.models.values():
            if hasattr(model, "fit_cache"):
                model.fit_cache = cache

    def route(self, example, view):
        """The champion's key: 'pre_weekend'; 'after_grid' (once the FIA starting grid is
        out) when the router has one; 'last' (the latest session cutoff before the target)
        when the router has one and the grid-setting session is classified in `view`;
        otherwise 'after_session'; '_wet' added for a wet example when the router has
        that key."""
        if example.cutoff_label == "pre_weekend":
            moment = "pre_weekend"
        elif example.cutoff_label == "after_grid" and "after_grid" in self.champions:
            moment = "after_grid"
        elif example.cutoff_label == "last" and "last" in self.champions and self._grid_observed(view, example):
            moment = "last"
        else:
            moment = "after_session"
        wet = f"{moment}_wet"
        return wet if getattr(example, "condition", None) == "wet" and wet in self.champions else moment

    def fit(self, history, view_for):
        for model in {id(m): m for m in self.models.values()}.values():
            model.fit(history, view_for)
        return self

    @staticmethod
    def _grid_observed(view, example):
        from .race import grid_observed
        return grid_observed(view, example)

    def predict(self, view, example, **kwargs):
        key = self.route(example, view)
        forecast = self.models[key].predict(view, example, **kwargs)
        if forecast is not None:
            # scored under the router's name only: components differ between routes, so
            # reporting them would leave no event common to all names (final review)
            metadata = {k: v for k, v in (forecast.metadata or {}).items() if k != "components"}
            forecast.metadata = {**metadata, "served_by": self.champions[key]}
            forecast.model = self.name
        return forecast


class ForecastMixture:
    """A weighted mixture of complete forecasts: as if each component's simulated
    finishing orders were pooled in proportion to its weight, so every position and
    outcome probability is the weighted mean of the components' (each entrant's
    whole distribution is mixed, not its expected position). Each component is
    fitted once and also reported on its own (`components` in the metadata)."""

    def __init__(self, components, weights, name, registry=None):
        registry = registry or model_registry()
        self.name = name
        self.components = [registry[c]() for c in components]
        self.weights = np.asarray(weights, dtype=float) / float(np.sum(weights))

    @property
    def fit_cache(self):
        return getattr(self.components[0], "fit_cache", None)

    @fit_cache.setter
    def fit_cache(self, cache):
        for model in self.components:
            if hasattr(model, "fit_cache"):
                model.fit_cache = cache

    def fit(self, history, view_for):
        for model in self.components:
            model.fit(history, view_for)
        return self

    def predict(self, view, example, **kwargs):
        forecasts = [model.predict(view, example, **kwargs) for model in self.components]
        if any(f is None for f in forecasts):
            return None
        if any(f.driver_numbers != forecasts[0].driver_numbers for f in forecasts):
            raise ValueError("components forecast different entrants")
        mix = lambda field: (None if getattr(forecasts[0], field) is None else
                             sum(w * getattr(f, field) for w, f in zip(self.weights, forecasts)))
        return replace(forecasts[0], position_probs=mix("position_probs"), outcome_probs=mix("outcome_probs"),
                       expected_points=mix("expected_points"), model=self.name,
                       metadata={"weights": self.weights.tolist(), "components": forecasts,
                                 "served_by": [f.model for f in forecasts]})


def model_registry():
    return {**{name: (lambda name=name: RankBaseline(name)) for name in RANKERS},
            "pace": lambda: _pace_model(1), "pace2": lambda: _pace_model(2),
            "pace3": lambda: _pace_model(3), "pace2q": lambda: _pace_model(2, "qualifying"),
            "pace3q": lambda: _pace_model(3, "qualifying"), "pace2w": lambda: _pace_model(2, weekend=True),
            "pace3w": lambda: _pace_model(3, weekend=True), "pace2s": lambda: _pace_model(2, shared=True),
            "pace3wa": lambda: _pace_model(3, weekend=True, affinity=True),
            "pace3wh": lambda: _pace_model(3, weekend=True, history=True),
            "pace3wah": lambda: _pace_model(3, weekend=True, affinity=True, history=True),
            "pace3whn": lambda: _pace_model(3, weekend=True, history="noise"),
            "pace3whs": lambda: _pace_model(3, weekend=True, history="state"),
            "pace3wahs": lambda: _pace_model(3, weekend=True, affinity=True, history="state"),
            "pace3whw": lambda: _pace_model(3, weekend=True, history="weighted"),
            "pace3wf": lambda: _pace_model(3, weekend=True, form=True),
            "pace3whsf": lambda: _pace_model(3, weekend=True, history="state", form=True),
            "pace3wk": lambda: _pace_model(3, weekend=True, skills=("fast",)),
            "pace3wr": lambda: _pace_model(3, weekend=True, races=True),
            "pace3wrd": lambda: _pace_model(3, weekend=True, races=True, driver_race=True),
            "pace3wrp": lambda: _pace_model(3, weekend=True, races=True, push=True),
            "pace3wrpn": lambda: _pace_model(3, weekend=True, races=True, push="own"),
            "pace3wrl": lambda: _pace_model(3, weekend=True, races=True, long_runs=True),
            "pace3wru": lambda: _pace_model(3, weekend=True, races=True, race_estimates=True),
            "champion": lambda: ChampionRouter(),
            "race_champion": lambda: ChampionRouter(RACE_CHAMPIONS, name="race_champion"),
            "pace3wrq": lambda: _pace_model(3, weekend=True, races=True, knockout=True),
            "pace3wrqf": lambda: _pace_model(3, weekend=True, races=True, knockout=True, failures=True),
            "pace3wrqw": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True),
            "pace3wrqwt": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True, robust=True),
            # race pace read with lap effects (no fixed fuel rate), then also keeping the connected group
            "pace3wrqwtre": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True, robust=True,
                                                race_read={"lap_effects": True}),
            "pace3wrqwtrep": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                 robust=True, race_read={"lap_effects": True, "partial": True}),
            # race pace from clean-air laps only (cleanair.py): 1.0 s and 1.5 s to the car ahead
            "pace3wrqwta1": lambda: _clean_air(_pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                           robust=True), 1.0),
            "pace3wrqwta15": lambda: _clean_air(_pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                            robust=True), 1.5),
            # race pace from every lap, each freed of the race's following penalty (cleanair.py)
            "pace3wrqwtfp": lambda: _corrected(_pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                           robust=True)),
            "pace3wrlqwt": lambda: _pace_model(3, weekend=True, races=True, long_runs=True, knockout=True, wet_noise=True,
                                               robust=True),
            "pace3wrqwp": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True, push="split"),
            # name pace:v3wrqwp ("split" adds no letter)
            # forecasts by the target's condition (c), with each driver's wet ability (y)
            "pace3wrqwc": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                              wet_forecast=True),
            "pace3wrqwy": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                              wet_ability=True),
            # wet and mixed races read with a drying curve (x)
            "pace3wrqwx": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                              wet_races=True),
            # wet targets forecast with the wet noise: against v3wrqwxy it isolates wet ability
            "pace3wrqwxc": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                               wet_races=True, wet_forecast=True),
            "pace3wrqwxy": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                               wet_races=True, wet_ability=True),
            # the robust (Student-t) pre-weekend champion with the wet toggle
            "pace3wrqwtc": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                               robust=True, wet_forecast=True),
            "pace3wrqwtxy": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                robust=True, wet_races=True, wet_ability=True),
            # each driver's own affinity for a circuit, against his teammates (i)
            "pace3wrqwti": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                               robust=True, driver_affinity=True),
            "pace3wrqwtic": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                robust=True, driver_affinity=True, wet_forecast=True),
            # each segment of a wet qualifying typed by the tyres its times were set on (g)
            "pace3wrqwtgc": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                robust=True, wet_segments=True, wet_forecast=True),
            # wet qualifying read at one track per segment (e)
            "pace3wrqwec": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                               wet_evolution=True, wet_forecast=True),
            "pace3wrqwtec": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                robust=True, wet_evolution=True, wet_forecast=True),
            # race forecasts (race.py) on the champion's pace model; fits shared with it
            "race1a": lambda: _race_model("pace3wrqwt", "race:v1a"),
            # the race champions on race pace read with lap effects (every component switched)
            "race1ae": lambda: _race_model("pace3wrqwtre", "race:v1ae"),
            "race2ae": lambda: _race_model("pace3wrqwtre", "race:v2ae", simulated=True),
            "race3ffne": lambda: _race_model("pace3wrqwtre", "race:v3ffne", simulated=True, forecast_battles=True,
                                             neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                             strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                             team_wear=True, following=True, windows=True),
            "race3fwe": lambda: _race_model("pace3wrqwtre", "race:v3fwe", simulated=True, forecast_battles=True,
                                            neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                            strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                            team_wear=True, lap_effect_wear=True),
            "race_mix3ffne1ae": lambda: ForecastMixture(["race3ffne", "race1ae"], [0.5, 0.5], "race:mix3ffne1ae"),
            "race_mix3fwe2ae": lambda: ForecastMixture(["race3fwe", "race2ae"], [0.5, 0.5], "race:mix3fwe2ae"),
            # gate B: the champions on race pace that keeps a race's largest connected group (no lap effects)
            "pace3wrqwtrp": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                robust=True, race_read={"partial": True}),
            # C: race pace with its measured uncertainty (estimation covariance x stint scale),
            # then with fewer clean laps needed (10, 5)
            "pace3wrqwtrps": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                 robust=True, race_estimates=True, stint_scale=True,
                                                 race_read={"partial": True}),
            "pace3wrqwtrps10": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                   robust=True, race_estimates=True, stint_scale=True,
                                                   race_read={"partial": True, "min_laps": 10}),
            "pace3wrqwtrps5": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                  robust=True, race_estimates=True, stint_scale=True,
                                                  race_read={"partial": True, "min_laps": 5}),
            # C split (review): 5 clean laps enough, without the estimation covariance (every
            # driver's race pace at the session noise, as before C): admission without weighting
            "pace3wrqwtrpm5": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                  robust=True, race_read={"partial": True, "min_laps": 5}),
            "race1apm5": lambda: _race_model("pace3wrqwtrpm5", "race:v1apm5"),
            "race2apm5": lambda: _race_model("pace3wrqwtrpm5", "race:v2apm5", simulated=True),
            "race3ffnpm5": lambda: _race_model("pace3wrqwtrpm5", "race:v3ffnpm5", simulated=True, forecast_battles=True,
                                               neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                               strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                               team_wear=True, following=True, windows=True),
            "race3fwpm5": lambda: _race_model("pace3wrqwtrpm5", "race:v3fwpm5", simulated=True, forecast_battles=True,
                                              neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                              strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                              team_wear=True, lap_effect_wear=True),
            "race_mix3ffnpm51apm5": lambda: ForecastMixture(["race3ffnpm5", "race1apm5"], [0.5, 0.5],
                                                            "race:mix3ffnpm51apm5"),
            "race_mix3fwpm52apm5": lambda: ForecastMixture(["race3fwpm5", "race2apm5"], [0.5, 0.5],
                                                           "race:mix3fwpm52apm5"),
            # C on races: the champions' components on the 5-lap stint-scale pace
            "race1aps5": lambda: _race_model("pace3wrqwtrps5", "race:v1aps5"),
            "race2aps5": lambda: _race_model("pace3wrqwtrps5", "race:v2aps5", simulated=True),
            "race3ffnps5": lambda: _race_model("pace3wrqwtrps5", "race:v3ffnps5", simulated=True, forecast_battles=True,
                                               neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                               strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                               team_wear=True, following=True, windows=True),
            "race3fwps5": lambda: _race_model("pace3wrqwtrps5", "race:v3fwps5", simulated=True, forecast_battles=True,
                                              neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                              strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                              team_wear=True, lap_effect_wear=True),
            "race_mix3ffnps51aps5": lambda: ForecastMixture(["race3ffnps5", "race1aps5"], [0.5, 0.5],
                                                            "race:mix3ffnps51aps5"),
            "race_mix3fwps52aps5": lambda: ForecastMixture(["race3fwps5", "race2aps5"], [0.5, 0.5],
                                                           "race:mix3fwps52aps5"),
            "pace3wrqwtcrp": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True,
                                                 robust=True, wet_forecast=True, race_read={"partial": True}),
            "race1ap": lambda: _race_model("pace3wrqwtrp", "race:v1ap"),
            "race2ap": lambda: _race_model("pace3wrqwtrp", "race:v2ap", simulated=True),
            "race3ffnp": lambda: _race_model("pace3wrqwtrp", "race:v3ffnp", simulated=True, forecast_battles=True,
                                             neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                             strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                             team_wear=True, following=True, windows=True),
            "race3fwp": lambda: _race_model("pace3wrqwtrp", "race:v3fwp", simulated=True, forecast_battles=True,
                                            neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                            strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                            team_wear=True, lap_effect_wear=True),
            # stage 4: the champions reading the published starting grid (g) and published grid penalties (k)
            "race_mix3ffnps51aps5g": lambda: _grid_input("race_mix3ffnps51aps5", "g", published_grid=True),
            "race_mix3ffnps51aps5k": lambda: _grid_input("race_mix3ffnps51aps5", "k", published_grid=True,
                                                         known_penalties=True),
            "race_mix3fwp2apk": lambda: _grid_input("race_mix3fwp2ap", "k", published_grid=True, known_penalties=True),
            # the race champions with the settings fit before 2026-11-23 (sequential random numbers,
            # forecast seeds, lowest mean RPS), for the re-baseline of the stable fit
            "race_mix3ffnps51aps5o": lambda: _grid_input("race_mix3ffnps51aps5", "o", stable_fit=False),
            "race_mix3fwp2apko": lambda: _grid_input("race_mix3fwp2apk", "o", stable_fit=False),
            "race_mix3ffnps51aps5go": lambda: _grid_input("race_mix3ffnps51aps5g", "o", stable_fit=False),
            # penalties after qualifying on the served old fit, without and with the split grid
            # term of the simple model (qualifying slot apart from start slot); the simple model
            # alone as well, for a quick read of the split
            "race_mix3ffnps51aps5ko": lambda: _grid_input("race_mix3ffnps51aps5k", "o", stable_fit=False),
            "race_mix3ffnps51aps5kqo": lambda: _grid_input("race_mix3ffnps51aps5", "kqo", published_grid=True,
                                                           known_penalties=True, split_grid=True, stable_fit=False),
            "race3ffnps5ko": lambda: _grid_input("race3ffnps5", "ko", published_grid=True, known_penalties=True,
                                                 stable_fit=False),
            "race1aps5o": lambda: _grid_input("race1aps5", "o", stable_fit=False),
            # step 9a: the simple half with the recent retirement level and the starting-slot
            # retirement effect (the simulator half's arm failed docs/step9/level_check.py)
            "race1aps5or": lambda: _grid_input("race1aps5", "or", stable_fit=False, grid_retirements=True,
                                               recent_retirements=True),
            # step 9b: the simple half with a fitted bonus for the cars starting 1-3
            "race1aps5of": lambda: _grid_input("race1aps5", "of", stable_fit=False, front_bonus=True),
            "race3ffnps5o": lambda: _grid_input("race3ffnps5", "o", stable_fit=False),
            "race_mix3ffnps51aps5of": lambda: ForecastMixture(["race3ffnps5o", "race1aps5of"], [0.5, 0.5],
                                                              "race:mix3ffnps51aps5of"),
            "race1aps5ko": lambda: _grid_input("race1aps5", "ko", published_grid=True, known_penalties=True,
                                               stable_fit=False),
            "race1aps5kqo": lambda: _grid_input("race1aps5", "kqo", published_grid=True, known_penalties=True,
                                                split_grid=True, stable_fit=False),
            "race_mix3ffnp1ap": lambda: ForecastMixture(["race3ffnp", "race1ap"], [0.5, 0.5], "race:mix3ffnp1ap"),
            "race_mix3fwp2ap": lambda: ForecastMixture(["race3fwp", "race2ap"], [0.5, 0.5], "race:mix3fwp2ap"),
            "race1b": lambda: _race_model("pace3wrqwt", "race:v1b", overtaking=True),
            "race1aa1": lambda: _race_model("pace3wrqwta1", "race:v1aa1"),
            "race1aa15": lambda: _race_model("pace3wrqwta15", "race:v1aa15"),
            "race1afp": lambda: _race_model("pace3wrqwtfp", "race:v1afp"),
            "race2a": lambda: _race_model("pace3wrqwt", "race:v2a", simulated=True),
            "race2as": lambda: _race_model("pace3wrqwt", "race:v2as", simulated=True, pass_shift=True),
            "race2af": lambda: _race_model("pace3wrqwt", "race:v2af", simulated=True, forecast_battles=True),
            "race2afs": lambda: _race_model("pace3wrqwt", "race:v2afs", simulated=True, forecast_battles=True,
                                            pass_shift=True),
            "race2b": lambda: _race_model("pace3wrqwt", "race:v2b", simulated=True, forecast_battles=True,
                                          neutralisations=True),
            "race2bw": lambda: _race_model("pace3wrqwt", "race:v2bw", simulated=True, forecast_battles=True,
                                           neutralisations=True, weekend_lap=True),
            "race1ar": lambda: _race_model("pace3wrqwt", "race:v1ar", detailed_retirements=True),
            "race2ar": lambda: _race_model("pace3wrqwt", "race:v2ar", simulated=True, detailed_retirements=True),
            "race2bwr": lambda: _race_model("pace3wrqwt", "race:v2bwr", simulated=True, forecast_battles=True,
                                            neutralisations=True, weekend_lap=True, detailed_retirements=True),
            "race2bwl": lambda: _race_model("pace3wrlqwt", "race:v2bwl", simulated=True, forecast_battles=True,
                                            neutralisations=True, weekend_lap=True, detailed_retirements=True),
            "race3a": lambda: _race_model("pace3wrqwt", "race:v3a", simulated=True, forecast_battles=True,
                                          neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                          strategy=True),
            "race3b": lambda: _race_model("pace3wrqwt", "race:v3b", simulated=True, forecast_battles=True,
                                          neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                          strategy=True, pit_lane=True),
            "race3c": lambda: _race_model("pace3wrqwt", "race:v3c", simulated=True, forecast_battles=True,
                                          neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                          strategy=True, pit_lane=True, car_strategies=True),
            "race3d": lambda: _race_model("pace3wrqwt", "race:v3d", simulated=True, forecast_battles=True,
                                          neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                          strategy=True, pit_lane=True, car_strategies=True, pit_noise=True),
            "race3e": lambda: _race_model("pace3wrqwt", "race:v3e", simulated=True, forecast_battles=True,
                                          neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                          strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True),
            "race3f": lambda: _race_model("pace3wrqwt", "race:v3f", simulated=True, forecast_battles=True,
                                          neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                          strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                          team_wear=True),
            # the development simulator mixed 50/50 with a champion (frozen: no weight search);
            # race:mix3f is the pre-weekend champion, race:mix3f1a the after-qualifying candidate.
            # (Both were evaluated as one mixture with the champion router, which served race:v2a
            # before the weekend and race:v1a after qualifying.)
            "race_mix3f": lambda: ForecastMixture(["race3f", "race2a"], [0.5, 0.5], "race:mix3f"),
            "race_mix3f1a": lambda: ForecastMixture(["race3f", "race1a"], [0.5, 0.5], "race:mix3f1a"),
            "race_mix3ff1a": lambda: ForecastMixture(["race3ff", "race1a"], [0.5, 0.5], "race:mix3ff1a"),
            # gambles (one stop more or fewer) from grid band and team wear, on the after-qualifying champion
            "race3ffng": lambda: _race_model("pace3wrqwt", "race:v3ffng", simulated=True, forecast_battles=True,
                                             neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                             strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                             team_wear=True, following=True, windows=True, gambles=True),
            "race_mix3ffng1a": lambda: ForecastMixture(["race3ffng", "race1a"], [0.5, 0.5], "race:mix3ffng1a"),
            # diagnostic: the gamble model on grid bands only (no team wear term)
            "race3ffngg": lambda: _race_model("pace3wrqwt", "race:v3ffngg", simulated=True, forecast_battles=True,
                                              neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                              strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                              team_wear=True, following=True, windows=True, gambles="grid"),
            "race_mix3ffngg1a": lambda: ForecastMixture(["race3ffngg", "race1a"], [0.5, 0.5], "race:mix3ffngg1a"),
            # the gamble model applied by race pace rank (as the pace-band table), with the team wear term
            "race3ffngp": lambda: _race_model("pace3wrqwt", "race:v3ffngp", simulated=True, forecast_battles=True,
                                              neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                              strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                              team_wear=True, following=True, windows=True, gambles="pace"),
            "race_mix3ffngp1a": lambda: ForecastMixture(["race3ffngp", "race1a"], [0.5, 0.5], "race:mix3ffngp1a"),
            # starting tyre roles on the after-qualifying champion's simulator (learned windows)
            "race3ffnr": lambda: _race_model("pace3wrqwt", "race:v3ffnr", simulated=True, forecast_battles=True,
                                             neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                             strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                             team_wear=True, following=True, windows=True, start_roles=True),
            "race_mix3ffnr1a": lambda: ForecastMixture(["race3ffnr", "race1a"], [0.5, 0.5], "race:mix3ffnr1a"),
            # pre-weekend cross-check: the following model, with learned windows, with lap-effect wear
            "race3ffwn": lambda: _race_model("pace3wrqwt", "race:v3ffwn", simulated=True, forecast_battles=True,
                                             neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                             strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                             team_wear=True, following=True, lap_effect_wear=True, windows=True),
            "race_mix3ff2a": lambda: ForecastMixture(["race3ff", "race2a"], [0.5, 0.5], "race:mix3ff2a"),
            "race_mix3ffn2a": lambda: ForecastMixture(["race3ffn", "race2a"], [0.5, 0.5], "race:mix3ffn2a"),
            "race_mix3ffwn2a": lambda: ForecastMixture(["race3ffwn", "race2a"], [0.5, 0.5], "race:mix3ffwn2a"),
            "race3g": lambda: _race_model("pace3wrqwt", "race:v3g", simulated=True, forecast_battles=True,
                                          neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                          strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                          team_wear=True, pit_model=True),
            "race3ff": lambda: _race_model("pace3wrqwt", "race:v3ff", simulated=True, forecast_battles=True,
                                           neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                           strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                           team_wear=True, following=True),
            "race3fp": lambda: _race_model("pace3wrqwt", "race:v3fp", simulated=True, forecast_battles=True,
                                           neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                           strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                           team_wear=True, following=True, follow_penalty=True),
            "race_mix3fp1a": lambda: ForecastMixture(["race3fp", "race1a"], [0.5, 0.5], "race:mix3fp1a"),
            # tyre wear from lap effects (no assumed fuel rate) on the two champions' simulators
            "race3ffw": lambda: _race_model("pace3wrqwt", "race:v3ffw", simulated=True, forecast_battles=True,
                                            neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                            strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                            team_wear=True, following=True, lap_effect_wear=True),
            "race3fw": lambda: _race_model("pace3wrqwt", "race:v3fw", simulated=True, forecast_battles=True,
                                           neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                           strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                           team_wear=True, lap_effect_wear=True),
            "race_mix3ffw1a": lambda: ForecastMixture(["race3ffw", "race1a"], [0.5, 0.5], "race:mix3ffw1a"),
            "race_mix3fw2a": lambda: ForecastMixture(["race3fw", "race2a"], [0.5, 0.5], "race:mix3fw2a"),
            # corrected all-laps race pace (clean-air pace) with the following penalty in the simulator
            "race3fpc": lambda: _race_model("pace3wrqwtfp", "race:v3fpc", simulated=True, forecast_battles=True,
                                            neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                            strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                            team_wear=True, following=True, follow_penalty=True),
            "race_mix3fpc1a": lambda: ForecastMixture(["race3fpc", "race1a"], [0.5, 0.5], "race:mix3fpc1a"),
            "race_mix3fpc2a": lambda: ForecastMixture(["race3fpc", "race2a"], [0.5, 0.5], "race:mix3fpc2a"),
            # learned pit windows, alone and with the fresh-tyre peak, on each champion's simulator
            "race3ffn": lambda: _race_model("pace3wrqwt", "race:v3ffn", simulated=True, forecast_battles=True,
                                            neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                            strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                            team_wear=True, following=True, windows=True),
            "race3ffnk": lambda: _race_model("pace3wrqwt", "race:v3ffnk", simulated=True, forecast_battles=True,
                                             neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                             strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                             team_wear=True, following=True, windows=True, fresh_peak=True),
            "race3fwn": lambda: _race_model("pace3wrqwt", "race:v3fwn", simulated=True, forecast_battles=True,
                                            neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                            strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                            team_wear=True, lap_effect_wear=True, windows=True),
            "race3fwnk": lambda: _race_model("pace3wrqwt", "race:v3fwnk", simulated=True, forecast_battles=True,
                                             neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                             strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                             team_wear=True, lap_effect_wear=True, windows=True, fresh_peak=True),
            "race_mix3ffn1a": lambda: ForecastMixture(["race3ffn", "race1a"], [0.5, 0.5], "race:mix3ffn1a"),
            "race_mix3ffnk1a": lambda: ForecastMixture(["race3ffnk", "race1a"], [0.5, 0.5], "race:mix3ffnk1a"),
            "race_mix3fwn2a": lambda: ForecastMixture(["race3fwn", "race2a"], [0.5, 0.5], "race:mix3fwn2a"),
            "race_mix3fwnk2a": lambda: ForecastMixture(["race3fwnk", "race2a"], [0.5, 0.5], "race:mix3fwnk2a"),
            # the fresh-tyre peak after a stop, on each champion's simulator
            "race3ffk": lambda: _race_model("pace3wrqwt", "race:v3ffk", simulated=True, forecast_battles=True,
                                            neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                            strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                            team_wear=True, following=True, fresh_peak=True),
            "race3fwk": lambda: _race_model("pace3wrqwt", "race:v3fwk", simulated=True, forecast_battles=True,
                                            neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                            strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                            team_wear=True, lap_effect_wear=True, fresh_peak=True),
            "race_mix3ffk1a": lambda: ForecastMixture(["race3ffk", "race1a"], [0.5, 0.5], "race:mix3ffk1a"),
            "race_mix3fwk2a": lambda: ForecastMixture(["race3fwk", "race2a"], [0.5, 0.5], "race:mix3fwk2a"),
            # the same on the pre-weekend champion's lap-effect wear
            "race3fpcw": lambda: _race_model("pace3wrqwtfp", "race:v3fpcw", simulated=True, forecast_battles=True,
                                             neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                             strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                             team_wear=True, following=True, follow_penalty=True,
                                             lap_effect_wear=True),
            "race_mix3fpcw2a": lambda: ForecastMixture(["race3fpcw", "race2a"], [0.5, 0.5], "race:mix3fpcw2a"),
            "race3fl": lambda: _race_model("pace3wrlqwt", "race:v3fl", simulated=True, forecast_battles=True,
                                           neutralisations=True, weekend_lap=True, detailed_retirements=True,
                                           strategy=True, pit_lane=True, car_strategies=True, weekend_wear=True,
                                           team_wear=True),
            "pace3wkb": lambda: _pace_model(3, weekend=True, skills=("fast",),
                                            season_skills=("straight", "slow", "medium", "fast"))}


def select_cutoff(archive, event_key, cutoff_label, target_start):
    """The named cutoff, or with 'last' the latest cutoff before the target session.
    'after_grid' (races): when the latest FIA starting grid before the race was
    published (grid penalties applied), not before the last session's cutoff; at that
    cutoff where no grid is on file (2019 grids are images)."""
    before = [c for c in archive.cutoffs(event_key) if c.time < target_start]
    if cutoff_label == "after_grid":
        from .archive import Cutoff
        from .race import grid_published
        if not before:
            return None
        last = before[-1]
        published = grid_published(archive, event_key, target_start)
        time = max(last.time, published) if published is not None else last.time
        return Cutoff(event_key, "after_grid", time, last.anchor_session)
    if cutoff_label == "last":
        return before[-1] if before else None
    return next((c for c in before if c.label == cutoff_label), None)


def prepare(archive, session_type, cutoff_label, seasons, allow_holdout=False):
    """Examples (target and cutoff) for every championship event in `seasons`."""
    examples = []
    events = archive.events[~archive.events.is_testing & archive.events.season.isin(seasons)]
    for event_key in events.sort_values(["season", "round"]).event_key:
        target = session_target(archive, event_key, session_type, allow_holdout=allow_holdout)
        if target is None or target.empty:
            continue
        target_session = archive.sessions[archive.sessions.session_id == target.session_id.iloc[0]].iloc[0]
        cutoff = select_cutoff(archive, event_key, cutoff_label, target_session.scheduled_start_utc)
        if cutoff is None:
            continue
        target = target.sort_values("driver_number", key=lambda s: s.astype(int)).reset_index(drop=True)
        target = _roster_at_cutoff(archive, event_key, cutoff.time, target)
        from .laps import wet_session
        # the condition the forecast is asked for: in evaluation the one the session ran in,
        # so a dry and a wet forecast are each scored on their own condition
        condition = "wet" if wet_session(archive._table(target_session.session_id, "laps")) else "dry"
        example = Example(
            event_key=event_key, season=int(event_key[:4]), session_type=session_type, cutoff_label=cutoff_label,
            cutoff_time=cutoff.time, target_available_at=target_session.available_at_utc,
            driver_numbers=target.driver_number.tolist(), driver_ids=target.driver_id.tolist(),
            positions=target.position.to_numpy(),
            outcomes=[None if pd.isna(o) else o for o in target.outcome],
            excluded=target.excluded.to_numpy(dtype=bool), band_rank=np.array([]),
            cutoff_session=cutoff.anchor_session, team_ids=target.team_id.tolist(),
            lineage_ids=target.lineage_id.tolist() if "lineage_id" in target else [], condition=condition)
        example.band_rank = dense_ranks(rank_championship(archive.view(cutoff), example))
        examples.append(example)
    return examples


def _roster_at_cutoff(archive, event_key, cutoff_time, target):
    """The drivers a forecast is made for, as known at the cutoff, aligned with their
    actual results.

    The roster is the event's latest FIA entry list published before the cutoff (its
    race drivers): a target driver it does not list is a late substitute and is not
    forecast, and a listed driver without a classification withdrew and is forecast
    but not scored. Without such a list (or with a name it could not resolve) the
    target's entrants stand in for it, corrected by `_substitutions_seen`.

    Actual classification positions are kept as they are (absolute positions are what
    a forecast is scored against)."""
    target = target.assign(driver_number=target.driver_number.astype(str))
    entered = _entry_list_at(archive, event_key, cutoff_time)
    if entered is None:
        rows = _substitutions_seen(archive, event_key, cutoff_time, target)
    else:
        rows = [row._asdict() for row in target.itertuples(index=False) if row.driver_id in entered]
        missing = set(entered) - set(target.driver_id)
        for driver in sorted(missing):
            seat = _latest_seat(archive, driver, cutoff_time, entered[driver])
            if seat is not None:
                rows.append({**{c: None for c in target.columns}, "session_id": target.session_id.iloc[0],
                             "driver_number": seat[0], "driver_id": driver, "lineage_id": seat[1],
                             "position": np.nan, "outcome": None, "excluded": True})
    roster = pd.DataFrame(rows, columns=target.columns)
    roster = roster.sort_values("driver_number", key=lambda s: s.astype(int)).reset_index(drop=True)
    roster["excluded"] = roster.excluded.fillna(False).astype(bool) | roster.position.isna()
    roster["position"] = roster.position.fillna(0).astype(int)        # 0: not scored (excluded)
    return roster


def _entry_list_at(archive, event_key, cutoff_time):
    """{driver ID: number printed on the list} of the race drivers of the event's latest
    entry list published at or before the cutoff; None without one or when a name on
    it is unresolved."""
    lists = getattr(archive, "entry_lists", None)
    if lists is None:
        return None
    published = lists[(lists.event_key == event_key) & (lists.published_utc <= cutoff_time)]
    if published.empty:
        return None
    latest = published[published.doc_number == published.sort_values(["published_utc", "doc_number"]).doc_number.iloc[-1]]
    race = latest[latest.role == "race"]
    if race.driver_id.isna().any() or len(race) < 10:
        return None
    numbers = race.raw_number if "raw_number" in race else pd.Series([None] * len(race), index=race.index)
    return dict(zip(race.driver_id, numbers))


def _latest_seat(archive, driver_id, cutoff_time, printed=None):
    """(driver number, lineage) of a listed driver as known at the cutoff: his latest
    archived session available by then; the number printed on the entry list when he
    has none (footnote markers follow the number: "631" is car 63), lineage unknown."""
    seen = archive.entrants[archive.entrants.driver_id == driver_id].drop(columns=["event_key"], errors="ignore").merge(
        archive.sessions[["session_id", "available_at_utc"]], on="session_id")
    seen = seen[seen.available_at_utc <= cutoff_time].sort_values("available_at_utc")
    if not seen.empty:
        last = seen.iloc[-1]
        return str(last.driver_number), last.lineage_id
    if printed is None or pd.isna(printed):
        return None
    digits = str(printed)
    return (digits[:-1] if len(digits) > 2 or int(digits) > 99 else digits), None


def _substitutions_seen(archive, event_key, cutoff_time, target):
    """Rows of the target's entrants, a car's entered driver replaced by its regular
    driver when the substitution is visible at the cutoff: the entered driver has run
    no session of the event yet and a regular of the car (`_regular_drivers`) ran its
    latest session instead (Stroll at the 2020 Eifel GP, before Hulkenberg was called
    in). Practice-only reserves are never regulars, and a driver who missed practice
    stays entered."""
    sessions = archive.sessions
    run = sessions[(sessions.event_key == event_key) & (sessions.available_at_utc <= cutoff_time)]
    if "lineage_id" not in target:
        return [row._asdict() for row in target.itertuples(index=False)]
    if run.empty:
        return _regulars_before_the_weekend(archive, event_key, cutoff_time, target)
    entrants = archive.entrants
    ran = set(entrants[entrants.session_id.isin(run.session_id)].driver_id.dropna())
    latest = run.sort_values("available_at_utc").iloc[-1]
    in_latest = entrants[entrants.session_id == latest.session_id]
    regulars = _regular_drivers(archive, event_key)
    rows = []
    for row in target.itertuples(index=False):
        replacement = None
        if row.driver_id not in ran and pd.notna(row.lineage_id):
            stand = in_latest[(in_latest.lineage_id == row.lineage_id) & ~in_latest.driver_id.isin(set(target.driver_id))]
            regular = stand[stand.driver_id.isin(regulars.get(row.lineage_id, set()))]
            if not regular.empty:
                replacement = regular.iloc[0]
        if replacement is None:
            rows.append(row._asdict())
        else:
            rows.append({**row._asdict(), "driver_number": str(replacement.driver_number),
                         "driver_id": replacement.driver_id, "position": np.nan, "outcome": None, "excluded": True})
    return rows


def _regulars_before_the_weekend(archive, event_key, cutoff_time, target):
    """Before any session of the event, without an entry list: a target driver who is
    not a regular of his car is a substitute announced later (Bearman for Sainz,
    Saudi Arabia 2024), so the car's regular is forecast instead, excluded from
    scoring."""
    regulars = _regular_drivers(archive, event_key)
    entered = set(target.driver_id)
    rows = []
    for row in target.itertuples(index=False):
        car = regulars.get(row.lineage_id, set()) if pd.notna(row.lineage_id) else set()
        missing = sorted(car - entered)
        seat = _latest_seat(archive, missing[0], cutoff_time) if car and row.driver_id not in car and missing else None
        if seat is None:
            rows.append(row._asdict())
        else:
            entered.add(missing[0])
            rows.append({**row._asdict(), "driver_number": seat[0], "driver_id": missing[0], "position": np.nan,
                         "outcome": None, "excluded": True})
    return rows


REGULAR_SESSIONS = ("FP2", "FP3", "Q", "SQ", "S", "R")     # first practice is where reserves drive


def _regular_drivers(archive, event_key):
    """lineage -> its regular drivers: those who drove it in second or third practice
    or a competitive session at the car's most recent earlier event of the season
    (reserves drive first practice only). Looking back to the car's latest event, not
    a fixed one, follows a replacement from its first full weekend on."""
    events = archive.events[~archive.events.is_testing].sort_values(["season", "round"]).reset_index(drop=True)
    keys = list(events.event_key)
    if event_key not in keys:
        return {}
    here = keys.index(event_key)
    season = events.season.iloc[here]
    earlier = [keys[i] for i in range(here - 1, -1, -1) if events.season.iloc[i] == season]
    sessions = archive.sessions[archive.sessions.event_key.isin(earlier)
                                & archive.sessions.session_type.isin(REGULAR_SESSIONS)][["session_id", "event_key"]]
    entrants = archive.entrants.drop(columns=["event_key"], errors="ignore").merge(sessions, on="session_id")
    regulars = {}
    for event in earlier:                                 # most recent first
        for lineage, drivers in entrants[entrants.event_key == event].groupby("lineage_id").driver_id:
            regulars.setdefault(lineage, set(drivers.dropna()))
    return regulars


def walk_forward(examples, models, archive, train_examples=(), fit_cache=None, screen_base=None,
                 keep_forecasts=False):
    """Before each example, fit every model on targets that finished before its cutoff,
    then forecast from an ArchiveView at the cutoff. Model instances persist through the
    run, so a model can decide how often to refit its own parameters."""
    pool = sorted([*train_examples, *examples], key=lambda e: e.target_available_at)
    registry = model_registry()
    instances = {name: registry[name]() for name in models}
    base = None
    if screen_base is not None:
        if fit_cache is None:
            raise ValueError("a screen reads the base model's fits from the fit cache")
        base = registry[screen_base]()
        base.fit_cache = fit_cache
    for name, model in instances.items():
        if fit_cache is not None and hasattr(model, "fit_cache"):
            model.fit_cache = fit_cache
        if base is not None and name != screen_base and hasattr(model, "screen_base"):
            # quick screen: the base's settings frozen, only this model's own settings fitted
            model.screen_base = base
            model.name = f"{model.name}|screen"
            if getattr(model, "fit_name", None):
                model.fit_name = f"{model.fit_name}|screen"

    def view_for(example):
        return archive.view(example.cutoff_time)

    rows, settings, kept = [], {}, []
    for example in examples:
        history = [e for e in pool if e.target_available_at < example.cutoff_time]
        view = view_for(example)
        for name, model in instances.items():
            model.fit(history, view_for)
            forecast = model.predict(view, example)
            if forecast is None:
                continue
            # a mixture also reports its components, each scored as its own model
            for one in [forecast, *forecast.metadata.pop("components", [])]:
                fitted = one.metadata.get("hyper") or {k: v for k, v in one.metadata.items()
                                                       if k in ("beta", "retire_rate", "mechanical_share")}
                settings.setdefault(one.model, {}).setdefault(str(example.season), fitted)
                rows.append({"event_key": example.event_key, "season": example.season, "model": one.model,
                             **({"served_by": one.metadata["served_by"]} if "served_by" in one.metadata else {}),
                             **score(one, example)})
                if keep_forecasts:
                    kept.extend(forecast_rows(one, example))
    scores = pd.DataFrame(rows)
    scores.attrs["settings"] = settings
    scores.attrs["forecasts"] = pd.DataFrame(kept) if keep_forecasts else None
    return scores


def forecast_rows(forecast, example):
    """One row per entrant: its position distribution, expected position and spread,
    and what happened (for error analysis)."""
    n = forecast.position_probs.shape[1]
    places = np.arange(1, n + 1)
    expected = forecast.position_probs @ places
    spread = np.sqrt(np.clip(forecast.position_probs @ places ** 2 - expected ** 2, 0, None))
    return [{"event_key": example.event_key, "season": example.season, "model": forecast.model,
             "driver_number": str(d), "actual": int(a), "excluded": bool(x), "expected": float(e), "spread": float(s),
             "probs": p.tolist()}
            for d, a, x, e, s, p in zip(forecast.driver_numbers, example.positions, example.excluded, expected, spread,
                                        forecast.position_probs)]


def common_events(scores):
    """Only events every model forecast, so means compare like with like."""
    models = scores.model.nunique()
    counts = scores.groupby("event_key").model.nunique()
    return scores[scores.event_key.isin(counts[counts == models].index)]


def summarise(scores):
    common = common_events(scores)
    metrics = [c for c in scores.columns if c not in ID_COLUMNS]
    overall = common.groupby("model")[metrics].mean()
    overall.insert(0, "events", common.groupby("model").event_key.nunique())
    by_season = common.groupby(["model", "season"])[metrics].mean()
    return overall, by_season


BLOCK_EVENTS = 4             # consecutive events resampled together (shared team form and model state)
DEGRADATION_MARGIN = 3e-4    # RPS: a model is "not acceptably worse" if its 90% upper bound is below this
MIN_BLOCKS = 3               # fewer independent blocks than this: the block interval is withheld


def _block_means(diff, resamples, rng, block):
    """Bootstrap means resampling blocks of `block` consecutive events (circular)."""
    n = diff.size
    starts = rng.integers(0, n, size=(resamples, -(-n // block)))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return diff[index.reshape(resamples, -1)[:, :n]].mean(axis=1)


def compare(scores, reference, metric="rps", lower_is_better=True, resamples=2000, seed=0):
    """Paired per-event difference of each model against `reference` on common events,
    in event order, with 90% intervals from an event bootstrap and from a block
    bootstrap (BLOCK_EVENTS consecutive events: neighbouring races share team form and
    model state, so single-event resampling overstates certainty).

    Verdicts use the block interval: better beyond noise when it lies below zero,
    worse beyond noise when it lies above zero, and within the degradation margin when
    its upper end is below DEGRADATION_MARGIN ("not significantly worse" alone does not
    show that a loss is small)."""
    common = common_events(scores)
    table = common.pivot_table(index="event_key", columns="model", values=metric).sort_index()
    base = f"baseline:{reference}" if f"baseline:{reference}" in table else reference
    rows = []
    for model in table.columns:
        if model == base:
            continue
        diff = (table[model] - table[base]).dropna().to_numpy()
        if not lower_is_better:
            diff = -diff
        rng = np.random.default_rng(seed)
        events = rng.choice(diff, size=(resamples, diff.size)).mean(axis=1)
        low, high = np.quantile(events, [0.05, 0.95])
        if diff.size >= MIN_BLOCKS * BLOCK_EVENTS:
            blocks = _block_means(diff, resamples, np.random.default_rng(seed), BLOCK_EVENTS)
            block_low, block_high = np.quantile(blocks, [0.05, 0.95])
        else:                         # too few independent blocks: no block verdict
            block_low = block_high = np.nan
        rows.append({"model": model, "reference": base, "metric": metric, "events": diff.size,
                     "mean_difference": diff.mean(), "interval_90_low": low, "interval_90_high": high,
                     "block_90_low": block_low, "block_90_high": block_high,
                     "verdict": ("none" if np.isnan(block_low) else "better" if block_high < 0
                                 else "worse" if block_low > 0 else "noise"),
                     "better_beyond_noise": bool(block_high < 0), "worse_beyond_noise": bool(block_low > 0),
                     "within_margin": bool(block_high < DEGRADATION_MARGIN),     # NaN compares False: no verdict
                     "share_of_events_better": float(np.mean(diff < 0))})
    return pd.DataFrame(rows)


SEASON_STAGES = ((1, 3), (4, 8), (9, 14), (15, 99))   # rounds; the last stage is checked on its own
LATE_STAGE = "15+"
SEED_BATCHES = 3             # a gain under DEGRADATION_MARGIN counts only on the mean over this many seed batches


def batch_mean(scores_list):
    """Scores averaged per (event, model) over seed batches (each a run's
    scores.csv frame); only events and models present in every batch. The protocol
    judges a small gain on these per-race means (common random numbers within each
    batch), so one lucky batch cannot promote a model."""
    if not scores_list:
        return pd.DataFrame()
    keys = ["event_key", "model"]
    common = set.intersection(*[set(map(tuple, s[keys].itertuples(index=False))) for s in scores_list])
    frames = [s[[tuple(k) in common for k in s[keys].itertuples(index=False)]] for s in scores_list]
    stacked = pd.concat(frames, ignore_index=True)
    numeric = [c for c in stacked.columns if c not in keys + ["season"] and pd.api.types.is_numeric_dtype(stacked[c])]
    out = stacked.groupby(keys + ["season"], as_index=False)[numeric].mean()
    out["batches"] = len(scores_list)
    return out


def stage_comparison(scores, reference):
    """`compare` within each stage of the season (by round). A model must not be worse
    beyond noise in the late stage at either cutoff, where the season's evidence is
    fullest. Stages are small (15-34 events): this guards against harm; it is not
    evidence of a gain."""
    rounds = scores.event_key.str[-2:].astype(int)
    tables = []
    for first, last in SEASON_STAGES:
        part = scores[(rounds >= first) & (rounds <= last)]
        if part.event_key.nunique() >= 5:
            label = f"{first}+" if last == 99 else f"{first}-{last}"
            tables.append(compare(part, reference).assign(stage=label))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def git_revision():
    """HEAD, and whether tracked files had uncommitted changes when the run started."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, check=True).stdout.strip() != ""
        return {"head": head, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return None


def run(session_type, cutoff_label, seasons, models, warmup=(2018,), allow_holdout=False, archive=None, out=None,
        reference="uniform", fit_cache=None, screen_base=None, keep_forecasts=False):
    archive = archive or Archive()
    revision = git_revision()   # the code the run starts with; the tree may change while it runs
    if set(seasons) & HOLDOUT_SEASONS and not allow_holdout:
        raise PermissionError(f"seasons {sorted(set(seasons) & HOLDOUT_SEASONS)} are held out")
    if fit_cache is not None:
        fit_cache.prime(archive.root, seasons)
    # the history reaches the first evaluated season whatever `seasons` is (final review
    # 2026-11-26: a run of 2025-26 alone otherwise learnt from 2018 only)
    warmup = tuple(sorted(set(warmup) | set(range(min(warmup), min(seasons)))))
    train = prepare(archive, session_type, cutoff_label, warmup, allow_holdout=False)
    examples = prepare(archive, session_type, cutoff_label, seasons, allow_holdout=allow_holdout)
    scores = walk_forward(examples, models, archive, train, fit_cache, screen_base, keep_forecasts)
    if out is not None:             # saved first: a failing comparison must not lose the run
        out.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out / "scores.csv", index=False)
        if scores.attrs.get("forecasts") is not None:
            scores.attrs["forecasts"].to_parquet(out / "forecasts.parquet", index=False)
    if reference != "uniform" and reference not in set(scores.model):
        raise ValueError(f"reference {reference} was not evaluated: add its model to --models"
                         + (f" (scores saved in {out})" if out is not None else ""))
    overall, by_season = summarise(scores)
    comparison = compare(scores, reference)
    stages = stage_comparison(scores, reference)
    from . import baselines as _baselines, race as race_module
    manifest = {
        "session_type": session_type, "cutoff": cutoff_label, "seasons": list(seasons), "warmup": list(warmup),
        "models": list(models), "events": len(examples), "git_revision": revision,
        "fit_cache": None if fit_cache is None else str(fit_cache.folder), "screen_base": screen_base,
        "reference_manifest": json.loads((archive.root / "reference" / "manifest.json").read_text(encoding="utf-8"))
        .get("built_at_utc"), "allow_holdout": allow_holdout,
        # module settings that change the scores (set by the CLI), for compare-batches
        "seed_batch": _baselines.SEED_BATCH, "race_draws": race_module.DRAWS, "race_fit_draws": race_module.FIT_DRAWS,
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out / "scores.csv", index=False)
        overall.to_csv(out / "overall.csv")
        by_season.to_csv(out / "by_season.csv")
        comparison.to_csv(out / "comparison.csv", index=False)
        stages.to_csv(out / "stages.csv", index=False)
        (out / "settings.json").write_text(json.dumps(scores.attrs.get("settings", {}), indent=2, default=float),
                                           encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return scores, overall, by_season, comparison, manifest
