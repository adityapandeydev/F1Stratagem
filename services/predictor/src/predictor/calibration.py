"""Calibration diagnostics of saved position forecasts (a run's forecasts.parquet).

Randomised PIT of a discrete finishing position y under a forecast with CDF F:
u = F(y - 1) + V (F(y) - F(y - 1)), V uniform on [0, 1). Under a calibrated forecast
u is uniform. Reported: mean (uniform 0.5), variance (1/12: above it the forecast is
overconfident, below underconfident), the decile histogram (1 = flat) and both
tails (share below 0.05 and above 0.95; 0.05 each), each with a 90% interval from
resampling whole races (a race's drivers are not independent). Matching the
uniform's mean and variance does not establish calibration on its own; read the
histogram and the tails. Groups that condition on what happened in the race
(safety car, weather) are descriptive: an unconditionally calibrated forecast can
look too wide in the calm-race subset and too narrow in the chaotic one.
"""
import numpy as np
import pandas as pd

RESAMPLES = 1000


def randomised_pit(probs, actual, rng):
    """PIT values of each row's actual position (1-based; beyond the last slot
    counts as the last) under its position probabilities."""
    probs = np.asarray(probs, dtype=float)
    y = np.minimum(np.asarray(actual, dtype=int), probs.shape[1])
    cdf = np.cumsum(probs, axis=1)
    rows = np.arange(len(y))
    below = np.where(y > 1, cdf[rows, np.maximum(y - 2, 0)], 0.0)
    at = cdf[rows, y - 1]
    return below + rng.uniform(size=len(y)) * (at - below)


def _stats(pit):
    hist = np.histogram(pit, bins=10, range=(0, 1))[0] / max(len(pit), 1) * 10
    return np.concatenate([[pit.mean(), pit.var(), np.mean(pit < 0.05), np.mean(pit > 0.95)], hist])


NAMES = ["mean", "variance", "tail_low", "tail_high"] + [f"decile_{k}" for k in range(1, 11)]
UNIFORM = [0.5, 1 / 12, 0.05, 0.05] + [1.0] * 10


def summary(pit, events, rng, resamples=RESAMPLES):
    """DataFrame (statistic, value, low, high, uniform) of `pit` with 90% intervals
    from resampling whole races (`events`: each value's race)."""
    pit, events = np.asarray(pit, dtype=float), np.asarray(events)
    groups = [np.flatnonzero(events == e) for e in np.unique(events)]
    point = _stats(pit)
    draws = np.array([_stats(pit[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])])
                      for _ in range(resamples)])
    return pd.DataFrame({"statistic": NAMES, "value": point, "low": np.percentile(draws, 5, axis=0),
                         "high": np.percentile(draws, 95, axis=0), "uniform": UNIFORM})


def diagnose(forecasts, model, groups=None, seed=0, resamples=RESAMPLES):
    """(overall summary, per-group table) of `model`'s rows in `forecasts` (a run's
    saved forecasts; excluded and unclassified rows left out). `groups`: {name:
    Series aligned with the forecast rows} to split by; per group: n, mean,
    variance and both tails with race-resampled 90% intervals of the variance."""
    rng = np.random.default_rng(seed)
    f = forecasts[(forecasts.model == model) & ~forecasts.excluded & (forecasts.actual > 0)]
    pit = randomised_pit(np.stack(f.probs.to_numpy()), f.actual.to_numpy(), rng)
    overall = summary(pit, f.event_key.to_numpy(), rng, resamples)
    rows = []
    for name, labels in (groups or {}).items():
        labels = pd.Series(labels).reindex(f.index)
        for value in labels.dropna().unique():
            mask = (labels == value).to_numpy()
            s = summary(pit[mask], f.event_key.to_numpy()[mask], rng, resamples).set_index("statistic")
            rows.append({"group": name, "value": value, "n": int(mask.sum()), "mean": s.value["mean"],
                         "variance": s.value["variance"], "variance_low": s.low["variance"],
                         "variance_high": s.high["variance"], "tail_low": s.value["tail_low"],
                         "tail_high": s.value["tail_high"]})
    return overall, pd.DataFrame(rows)
