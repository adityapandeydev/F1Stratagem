"""Walk-forward evaluation: score forecasts against the final classification.

For every event in the evaluated seasons, a model is fitted on events whose
target session finished before the event's cutoff, then forecasts the target
session from data available at the cutoff. Seasons 2025–2026 are held out until
model selection (see targets.HOLDOUT_SEASONS).

Metrics, each averaged over scored entrants (excluded entrants are not scored):

    winner_log_loss    -log p(actual winner); pole for qualifying
    rps                ranked probability score of the position distribution
    brier_top3/top10   Brier score of finishing in the top 3 / top 10
    position_mae       |expected position - actual position|
    coverage_80        share of actual positions inside the central 80% interval
    interval_width     mean width of that interval, in positions
    outcome_log_loss   five-way outcome, entrants with a known outcome only
    retirement_brier   any retirement (cause known or not)
"""

import json
import subprocess

import numpy as np
import pandas as pd

from .archive import Archive
from .baselines import RANKERS, Example, RankBaseline, _dense_ranks
from .targets import HOLDOUT_SEASONS, OUTCOMES, session_target

EPS = 1e-6
BANDS = ((1, 5, "front"), (6, 12, "midfield"), (13, 99, "back"))
RETIRED = ("mechanical", "incident_or_other", "retired_unknown")


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
    expected = forecast.expected_position[keep]
    row = {
        "winner_log_loss": -np.log(max(forecast.position_probs[winner, 0], EPS)),
        "rps": float(np.mean(np.sum((cdf - observed) ** 2, axis=1) / (n - 1))),
        "brier_top3": float(np.mean((forecast.top(3)[keep] - (actual <= 3)) ** 2)),
        "brier_top10": float(np.mean((forecast.top(10)[keep] - (actual <= 10)) ** 2)),
        "position_mae": float(np.mean(np.abs(expected - actual))),
        "coverage_80": float(np.mean((actual >= lower) & (actual <= upper))),
        "interval_width": float(np.mean(upper - lower + 1)),
    }
    for low, high, band in BANDS:
        in_band = (actual >= low) & (actual <= high)
        row[f"position_mae_{band}"] = float(np.mean(np.abs(expected - actual)[in_band])) if in_band.any() else np.nan
    if forecast.outcome_probs is not None:
        outcomes = [o for o, x in zip(example.outcomes, example.excluded) if not x]
        probs = forecast.outcome_probs[keep]
        known = [(i, OUTCOMES.index(o)) for i, o in enumerate(outcomes) if o in OUTCOMES]
        row["outcome_log_loss"] = float(np.mean([-np.log(max(probs[i, k], EPS)) for i, k in known])) if known else np.nan
        retired = np.array([o in RETIRED for o in outcomes], dtype=float)
        p_retired = probs[:, OUTCOMES.index("mechanical")] + probs[:, OUTCOMES.index("incident_or_other")]
        row["retirement_brier"] = float(np.mean((p_retired - retired) ** 2))
    return row


def prepare(archive, session_type, cutoff_label, seasons, allow_holdout=False):
    """Examples (target + rankings at the cutoff) for every championship event in `seasons`."""
    examples = []
    events = archive.events[~archive.events.is_testing & archive.events.season.isin(seasons)]
    for event_key in events.sort_values(["season", "round"]).event_key:
        cutoff = next((c for c in archive.cutoffs(event_key) if c.label == cutoff_label), None)
        target = session_target(archive, event_key, session_type, allow_holdout=allow_holdout)
        if cutoff is None or target is None or target.empty:
            continue
        target_session = archive.sessions[archive.sessions.session_id == target.session_id.iloc[0]].iloc[0]
        if cutoff.time >= target_session.scheduled_start_utc:
            continue  # the cutoff must precede the session being forecast
        target = target.sort_values("driver_number", key=lambda s: s.astype(int)).reset_index(drop=True)
        view = archive.view(cutoff)
        example = Example(
            event_key=event_key, season=int(event_key[:4]), session_type=session_type, cutoff_label=cutoff_label,
            cutoff_time=cutoff.time, target_available_at=target_session.available_at_utc,
            driver_numbers=target.driver_number.tolist(), driver_ids=target.driver_id.tolist(),
            positions=target.position.to_numpy(),
            outcomes=[None if pd.isna(o) else o for o in target.outcome],
            excluded=target.excluded.to_numpy(dtype=bool), ranks={})
        for name, ranker in RANKERS.items():
            values = ranker(view, example)
            example.ranks[name] = None if values is None else _dense_ranks(values)
        examples.append(example)
    return examples


def walk_forward(examples, models, train_examples=()):
    """Fit each model on everything finished before each example's cutoff, then score it."""
    pool = sorted([*train_examples, *examples], key=lambda e: e.target_available_at)
    rows = []
    for example in examples:
        history = [e for e in pool if e.target_available_at < example.cutoff_time]
        for name in models:
            model = RankBaseline(name).fit(history)
            forecast = model.predict(example)
            if forecast is None:
                continue
            rows.append({"event_key": example.event_key, "season": example.season, "model": forecast.model,
                         "beta": model.beta, **score(forecast, example)})
    return pd.DataFrame(rows)


def summarise(scores):
    metrics = [c for c in scores.columns if c not in ("event_key", "season", "model", "beta")]
    overall = scores.groupby("model")[metrics].mean()
    by_season = scores.groupby(["model", "season"])[metrics].mean()
    return overall, by_season


def git_revision():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(session_type, cutoff_label, seasons, models, warmup=(2018,), allow_holdout=False, archive=None, out=None):
    archive = archive or Archive()
    if set(seasons) & HOLDOUT_SEASONS and not allow_holdout:
        raise PermissionError(f"seasons {sorted(set(seasons) & HOLDOUT_SEASONS)} are held out")
    train = prepare(archive, session_type, cutoff_label, warmup, allow_holdout=False)
    examples = prepare(archive, session_type, cutoff_label, seasons, allow_holdout=allow_holdout)
    scores = walk_forward(examples, models, train)
    overall, by_season = summarise(scores)
    manifest = {
        "session_type": session_type, "cutoff": cutoff_label, "seasons": list(seasons), "warmup": list(warmup),
        "models": list(models), "events": len(examples), "git_revision": git_revision(),
        "reference_manifest": json.loads((archive.root / "reference" / "manifest.json").read_text(encoding="utf-8"))
        .get("built_at_utc"), "allow_holdout": allow_holdout,
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out / "scores.csv", index=False)
        overall.to_csv(out / "overall.csv")
        by_season.to_csv(out / "by_season.csv")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return scores, overall, by_season, manifest
