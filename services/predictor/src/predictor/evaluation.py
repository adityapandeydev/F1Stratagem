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

import numpy as np
import pandas as pd

from .archive import Archive
from .baselines import RANKERS, Example, RankBaseline, dense_ranks, rank_championship
from .targets import HOLDOUT_SEASONS, OUTCOMES, RETIREMENTS, session_target

EPS = 1e-12  # safety only; forecasts carry a prior pseudo-draw, so no probability is zero
BANDS = ((1, 5, "front"), (6, 12, "midfield"), (13, 99, "back"))
ID_COLUMNS = ("event_key", "season", "model")


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
                race_estimates=False, knockout=False, failures=False, wet_noise=False, robust=False):
    from .pace import ALL_SESSIONS, QUALIFYING_ONLY, WITH_RACES, PaceModel
    if stage == 1:
        return PaceModel(QUALIFYING_ONLY, "pace:v1")
    suffix = ("q" if objective == "qualifying" else "") + ("w" if weekend else "") + ("s" if shared else "") \
        + ("a" if affinity else "") + ("h" if history else "") \
        + {"noise": "n", "state": "s", "weighted": "w"}.get(history, "") + ("f" if form else "") + ("k" if skills else "") + ("b" if season_skills else "") \
        + ("r" if races else "") + ("d" if driver_race else "") + ("p" if push else "") + ("n" if push == "own" else "") \
        + ("l" if long_runs else "") + ("u" if race_estimates else "") + ("q" if knockout else "") + ("f" if failures else "") + ("w" if wet_noise else "") + ("t" if robust else "")
    return PaceModel(WITH_RACES if races else ALL_SESSIONS, f"pace:v{stage}{suffix}", winter=stage >= 3, objective=objective,
                     weekend=weekend, shared_from_qualifying=shared, affinity=affinity, history=history, form=form,
                     skills=skills, season_skills=season_skills, driver_offsets=("R",) if driver_race else (),
                     practice_push=push, long_runs=long_runs, race_estimates=race_estimates,
                     knockout=knockout, failures=failures, wet_noise=wet_noise, robust=robust)


# The champion at each forecast moment (per-horizon promotion, design/prediction-engine.md).
# Before the weekend: no session of the event has run. Once any has, the last-cutoff
# champion serves; after FP1 and after FP2 that routing is provisional until those
# cutoffs have been evaluated.
CHAMPIONS = {"pre_weekend": "pace3wrqw", "after_session": "pace3wrqw"}


class ChampionRouter:
    """Serves each forecast with the champion of its forecast moment: one model
    instance per champion, fitted and asked only for the examples it serves."""

    name = "champion"

    def __init__(self, champions=None, registry=None):
        self.champions = dict(champions or CHAMPIONS)
        registry = registry or model_registry()
        self.models = {key: registry[name]() for key, name in self.champions.items()}
        if self.champions["pre_weekend"] == self.champions["after_session"]:
            self.models["after_session"] = self.models["pre_weekend"]      # one fit serves both

    @property
    def fit_cache(self):
        return getattr(self.models["pre_weekend"], "fit_cache", None)

    @fit_cache.setter
    def fit_cache(self, cache):
        for model in self.models.values():
            if hasattr(model, "fit_cache"):
                model.fit_cache = cache

    def route(self, example):
        return "pre_weekend" if example.cutoff_label == "pre_weekend" else "after_session"

    def fit(self, history, view_for):
        for model in {id(m): m for m in self.models.values()}.values():
            model.fit(history, view_for)
        return self

    def predict(self, view, example, **kwargs):
        forecast = self.models[self.route(example)].predict(view, example, **kwargs)
        if forecast is not None:
            forecast.metadata = {**(forecast.metadata or {}), "served_by": self.champions[self.route(example)]}
            forecast.model = self.name
        return forecast


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
            "pace3wrq": lambda: _pace_model(3, weekend=True, races=True, knockout=True),
            "pace3wrqf": lambda: _pace_model(3, weekend=True, races=True, knockout=True, failures=True),
            "pace3wrqw": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True),
            "pace3wrqwt": lambda: _pace_model(3, weekend=True, races=True, knockout=True, wet_noise=True, robust=True),
            "pace3wkb": lambda: _pace_model(3, weekend=True, skills=("fast",),
                                            season_skills=("straight", "slow", "medium", "fast"))}


def select_cutoff(archive, event_key, cutoff_label, target_start):
    """The named cutoff, or with 'last' the latest cutoff before the target session."""
    before = [c for c in archive.cutoffs(event_key) if c.time < target_start]
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
        example = Example(
            event_key=event_key, season=int(event_key[:4]), session_type=session_type, cutoff_label=cutoff_label,
            cutoff_time=cutoff.time, target_available_at=target_session.available_at_utc,
            driver_numbers=target.driver_number.tolist(), driver_ids=target.driver_id.tolist(),
            positions=target.position.to_numpy(),
            outcomes=[None if pd.isna(o) else o for o in target.outcome],
            excluded=target.excluded.to_numpy(dtype=bool), band_rank=np.array([]),
            cutoff_session=cutoff.anchor_session, team_ids=target.team_id.tolist(),
            lineage_ids=target.lineage_id.tolist() if "lineage_id" in target else [])
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
    if run.empty or "lineage_id" not in target:
        return [row._asdict() for row in target.itertuples(index=False)]
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


def walk_forward(examples, models, archive, train_examples=(), fit_cache=None, screen_base=None):
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

    def view_for(example):
        return archive.view(example.cutoff_time)

    rows, settings = [], {}
    for example in examples:
        history = [e for e in pool if e.target_available_at < example.cutoff_time]
        view = view_for(example)
        for name, model in instances.items():
            model.fit(history, view_for)
            forecast = model.predict(view, example)
            if forecast is None:
                continue
            fitted = forecast.metadata.get("hyper") or {k: v for k, v in forecast.metadata.items()
                                                         if k in ("beta", "retire_rate", "mechanical_share")}
            settings.setdefault(forecast.model, {}).setdefault(str(example.season), fitted)
            rows.append({"event_key": example.event_key, "season": example.season, "model": forecast.model,
                         **score(forecast, example)})
    scores = pd.DataFrame(rows)
    scores.attrs["settings"] = settings
    return scores


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
                     "better_beyond_noise": bool(block_high < 0), "worse_beyond_noise": bool(block_low > 0),
                     "within_margin": bool(block_high < DEGRADATION_MARGIN),     # NaN compares False: no verdict
                     "share_of_events_better": float(np.mean(diff < 0))})
    return pd.DataFrame(rows)


SEASON_STAGES = ((1, 3), (4, 8), (9, 14), (15, 99))   # rounds; the last stage is checked on its own
LATE_STAGE = "15+"


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
        reference="uniform", fit_cache=None, screen_base=None):
    archive = archive or Archive()
    revision = git_revision()   # the code the run starts with; the tree may change while it runs
    if fit_cache is not None:
        fit_cache.prime(archive.root, seasons)
    if set(seasons) & HOLDOUT_SEASONS and not allow_holdout:
        raise PermissionError(f"seasons {sorted(set(seasons) & HOLDOUT_SEASONS)} are held out")
    train = prepare(archive, session_type, cutoff_label, warmup, allow_holdout=False)
    examples = prepare(archive, session_type, cutoff_label, seasons, allow_holdout=allow_holdout)
    scores = walk_forward(examples, models, archive, train, fit_cache, screen_base)
    overall, by_season = summarise(scores)
    comparison = compare(scores, reference)
    stages = stage_comparison(scores, reference)
    manifest = {
        "session_type": session_type, "cutoff": cutoff_label, "seasons": list(seasons), "warmup": list(warmup),
        "models": list(models), "events": len(examples), "git_revision": revision,
        "fit_cache": None if fit_cache is None else str(fit_cache.folder), "screen_base": screen_base,
        "reference_manifest": json.loads((archive.root / "reference" / "manifest.json").read_text(encoding="utf-8"))
        .get("built_at_utc"), "allow_holdout": allow_holdout,
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
