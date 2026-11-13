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
    winner = int(np.flatnonzero(example.positions == 1)[0])
    cdf = np.cumsum(p, axis=1)
    observed = (np.arange(1, n + 1)[None, :] >= actual[:, None]).astype(float)
    lower, upper = forecast.interval()
    lower, upper = lower[keep], upper[keep]
    inside = (np.arange(1, n + 1)[None, :] >= lower[:, None]) & (np.arange(1, n + 1)[None, :] <= upper[:, None])
    expected = forecast.expected_position[keep]
    band_rank = example.band_rank[keep]
    row = {
        "winner_log_loss": -np.log(max(forecast.position_probs[winner, 0], EPS)),
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


def _pace_model(stage, objective="all", weekend=False, shared=False, affinity=False, history=False):
    from .pace import ALL_SESSIONS, QUALIFYING_ONLY, PaceModel
    if stage == 1:
        return PaceModel(QUALIFYING_ONLY, "pace:v1")
    suffix = ("q" if objective == "qualifying" else "") + ("w" if weekend else "") + ("s" if shared else "") \
        + ("a" if affinity else "") + ("h" if history else "") \
        + {"noise": "n", "state": "s", "weighted": "w"}.get(history, "")
    return PaceModel(ALL_SESSIONS, f"pace:v{stage}{suffix}", winter=stage >= 3, objective=objective,
                     weekend=weekend, shared_from_qualifying=shared, affinity=affinity, history=history)


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
            "pace3whw": lambda: _pace_model(3, weekend=True, history="weighted")}


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


def walk_forward(examples, models, archive, train_examples=(), fit_cache=None):
    """Before each example, fit every model on targets that finished before its cutoff,
    then forecast from an ArchiveView at the cutoff. Model instances persist through the
    run, so a model can decide how often to refit its own parameters."""
    pool = sorted([*train_examples, *examples], key=lambda e: e.target_available_at)
    registry = model_registry()
    instances = {name: registry[name]() for name in models}
    for model in instances.values():
        if fit_cache is not None and hasattr(model, "fit_cache"):
            model.fit_cache = fit_cache

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


def compare(scores, reference, metric="rps", lower_is_better=True, resamples=2000, seed=0):
    """Paired per-event difference of each model against `reference` on common events,
    with a bootstrap 90% interval. A component is better beyond noise when the
    whole interval is on the improving side of zero, and fails the "no worse" gate when
    the whole interval is on the worsening side."""
    common = common_events(scores)
    table = common.pivot_table(index="event_key", columns="model", values=metric)
    base = f"baseline:{reference}" if f"baseline:{reference}" in table else reference
    rng = np.random.default_rng(seed)
    rows = []
    for model in table.columns:
        if model == base:
            continue
        diff = (table[model] - table[base]).dropna().to_numpy()
        if not lower_is_better:
            diff = -diff
        samples = rng.choice(diff, size=(resamples, diff.size)).mean(axis=1)
        low, high = np.quantile(samples, [0.05, 0.95])
        rows.append({"model": model, "reference": base, "metric": metric, "events": diff.size,
                     "mean_difference": diff.mean(), "interval_90_low": low, "interval_90_high": high,
                     "better_beyond_noise": bool(high < 0), "worse_beyond_noise": bool(low > 0),
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
        reference="uniform", fit_cache=None):
    archive = archive or Archive()
    revision = git_revision()   # the code the run starts with; the tree may change while it runs
    if fit_cache is not None:
        fit_cache.prime(archive.root, seasons)
    if set(seasons) & HOLDOUT_SEASONS and not allow_holdout:
        raise PermissionError(f"seasons {sorted(set(seasons) & HOLDOUT_SEASONS)} are held out")
    train = prepare(archive, session_type, cutoff_label, warmup, allow_holdout=False)
    examples = prepare(archive, session_type, cutoff_label, seasons, allow_holdout=allow_holdout)
    scores = walk_forward(examples, models, archive, train, fit_cache)
    overall, by_season = summarise(scores)
    comparison = compare(scores, reference)
    stages = stage_comparison(scores, reference)
    manifest = {
        "session_type": session_type, "cutoff": cutoff_label, "seasons": list(seasons), "warmup": list(warmup),
        "models": list(models), "events": len(examples), "git_revision": revision,
        "fit_cache": None if fit_cache is None else str(fit_cache.folder),
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
