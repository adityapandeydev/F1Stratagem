# predictor

The F1Stratagem prediction engine. Design:
[design/prediction-engine.md](../../design/prediction-engine.md).

Implemented: the evaluation foundation and the qualifying pace model.

| Module | Role |
|---|---|
| `archive.py` | Cutoff-gated reads. An `ArchiveView` only exposes sessions available at its cutoff; any other read raises `CutoffViolation`. |
| `targets.py` | Final classification and race outcomes per session. Seasons 2025–2026 are held out and need an explicit flag. |
| `rules.py` | Points tables by season. |
| `forecast.py` | The forecast object every model produces, built from joint simulation draws and checked for consistency. |
| `simulate.py` | Plackett-Luce ranking draws and likelihood. |
| `baselines.py` | Reference models: uniform, championship order, persistence, practice pace, qualifying order. |
| `pace.py` | Pace model: Kalman filter over car, driver, team × session offsets, weekend effect, circuit affinity and winter reset; settings fit; forecasts. |
| `program.py` | The filter's bookkeeping: records a run (which states exist, when they are added, parked, reset, observed) as a flat program. |
| `native/` | The filter's arithmetic in Rust (`predictor_native`, built with maturin): replays a program for given settings and returns the log-likelihood and final state. |
| `evaluation.py` | Walk-forward evaluation, metrics, paired comparisons. |

Models (`predictor evaluate --models ...`):

| Name | Model |
|---|---|
| `uniform`, `championship`, `persistence`, `practice`, `qualifying` | Baselines |
| `pace` | v1: qualifying evidence only |
| `pace2` | v2: + practice, sprint qualifying, testing, team × session offsets |
| `pace3w` | v3w: + weekend effect and winter reset (current) |
| `pace3wa` | v3wa: v3w + car × circuit affinity (tested, not adopted) |
| `pace3`, `pace2q`, `pace3q`, `pace2w`, `pace2s` | Development variants kept for comparison |

## Availability in backtests

Historical data has no publication timestamps, so a session counts as available
once its last lap has ended plus a publication delay (30 minutes for practice and
qualifying, 60 for sprints and races). Results tables can include corrections made
after the session; backtests built on them are retrospective (`archive_proxy`).

## Evaluation

```powershell
make predictor-install                        # the package, then the native module (needs a Rust toolchain)
f1data reference build                        # canonical IDs, once per archive update
predictor evaluate --session Q --cutoff pre_weekend --reference pace:v3w --models pace3w pace3wa
predictor evaluate --session Q --cutoff last --reference pace:v3w --models pace3w pace3wa
```

`--cutoff` takes `pre_weekend`, `after_<session>` (e.g. `after_FP2`) or `last`
(the latest cutoff before the target session). `--reference` names the model the
others are compared against.

Seasons default to 2019–2023 (development); 2018 is warm-up data only. Baselines
are refitted before each event on events that finished before its cutoff; the pace
model refits its settings once per season on data available before the season and
filters every session up to each forecast's cutoff. Runs are
written to `services/predictor/data/runs/` with per-event scores, summaries, a
paired comparison, each model's fitted settings per season (`settings.json`) and a
manifest (code revision and whether it had uncommitted changes, reference build).

The printed summary includes the paired difference against the reference with a
bootstrap 90% interval (`better_beyond_noise`, `worse_beyond_noise`), the same
comparison on each season's first three rounds, and RPS by season. A new component
must not be `worse_beyond_noise` at any cutoff.

Metrics: winner (or pole) log loss, ranked probability score, Brier scores for
top 3 and top 10, expected-position error overall and by band (championship rank
1–5, 6–12, 13+ at the cutoff), 80% interval coverage and width, five-way outcome log loss on entrants with
a known outcome, and retirement Brier score.

## Tests

```powershell
python -m pytest services/predictor
```
