# predictor

The F1Stratagem prediction engine. Design:
[design/prediction-engine.md](../../design/prediction-engine.md).

Implemented: the evaluation foundation, the qualifying pace model and the first
stages of the race model.

| Module | Role |
|---|---|
| `archive.py` | Cutoff-gated reads. An `ArchiveView` only exposes sessions available at its cutoff; any other read raises `CutoffViolation`. |
| `targets.py` | Final classification and race outcomes per session. Seasons 2025–2026 are held out and need an explicit flag. |
| `rules.py` | Points tables by season. |
| `forecast.py` | The forecast object every model produces, built from joint simulation draws and checked for consistency. |
| `simulate.py` | Plackett-Luce ranking draws and likelihood. |
| `baselines.py` | Reference models: uniform, championship order, persistence, practice pace, qualifying order. |
| `pace.py` | Pace model: Kalman filter over car, driver, team × session offsets, weekend effect, circuit affinities, wet ability and winter reset; settings fit; qualifying forecasts (Q1/Q2/Q3 simulation); state draws for the race model. |
| `program.py` | The filter's bookkeeping: records a run (which states exist, when they are added, parked, reset, observed) as a flat program. |
| `native/` | The filter's arithmetic in Rust (`predictor_native`, built with maturin): replays a program for given settings and returns the log-likelihood and final state. |
| `laps.py` | Lap-level corrections: corrected race pace (tyres, tyre age, fuel; drying races lap by lap) with its estimation covariance; practice push laps and long runs; wet track evolution and segment wetness in qualifying; green-lap pass counts. |
| `race.py` | Race model: race pace and grid draws from the pace model, the grid known at the cutoff, retirements, settings fitted per season and cutoff by RPS; the lap-by-lap variant. |
| `race_sim.py` | Battles from position data, the battle (pass) model, race length and lap reference, neutralisations (track status from 2018, inferred from Jolpica lap times before) and their hazard model, the lap-by-lap simulation driver. |
| `race_native/` | The race simulation in Rust (`predictor_race_native`): start, laps, passes, neutralisations and their random numbers, one generator per draw, draws in parallel. |
| `track.py` | Corners and track profiles from telemetry; the circuit layout table (`circuit_layouts.csv`). |
| `fitcache.py` | Fitted settings saved per model and season under a key covering model, start, code and data; reused across cutoffs and runs. |
| `evaluation.py` | Walk-forward evaluation, metrics, paired comparisons (event and block bootstrap), results by stage of the season, quick screens. |

Models (`predictor evaluate --models ...`):

| Name | Model |
|---|---|
| `uniform`, `championship`, `persistence`, `practice`, `qualifying` | Baselines |
| `pace` | v1: qualifying evidence only |
| `pace2` | v2: + practice, sprint qualifying, testing, team × session offsets |
| `pace3w` | v3w: + weekend effect and winter reset |
| `pace3wr` | v3wr: v3w + corrected race pace |
| `pace3wrq` | v3wrq: + Q1/Q2/Q3 simulation |
| `pace3wrqw` | v3wrqw: + wet sessions as their own evidence |
| `pace3wrqwt` | v3wrqwt: + robust Student-t noise (qualifying champion at both cutoffs) |
| `pace3wrqwc`, `pace3wrqwtc` | Wet forecast mode (wet targets forecast with wet noise); `pace3wrqwtc` serves the wet toggle |
| `champion` | Router: the champion of each forecast moment (before the weekend, after a session) and condition (dry, wet) |
| `pace3wrqwx`, `pace3wrqwxc`, `pace3wrqwxy`, `pace3wrqwy`, `pace3wrqwtxy`, `pace3wrqwec`, `pace3wrqwtec`, `pace3wrqwtgc`, `pace3wrqwti`, `pace3wrqwtic`, `pace3wrqwp` | Tested, not adopted, kept for the re-test after the holdout check: drying curve (x), wet ability (y), wet track evolution (e), wet segments by tyres (g), driver × circuit affinity (i), practice push pace (p) |
| `race1a` | race:v1a: race pace, grid at the cutoff, retirements (last-cutoff race champion) |
| `race1b` | race:v1b: grid weight by circuit overtaking (tested, not adopted) |
| `race2a` | race:v2a: lap-by-lap simulation with the battle model (pre-weekend race champion) |
| `race2as` | race:v2as: v2a + a fitted pass shift (tested, not adopted) |
| `race2af`, `race2afs` | race:v2af: battle model fitted on forecast pace differences; v2afs adds the pass shift (not adopted) |
| `race2b` | race:v2b: v2af + safety cars, virtual safety cars and red flags (simulation base for the next stages) |
| `race_champion` | Router: the race champion of each forecast moment |
| `pace3wrpn`, `pace3wru`, `pace3wrqf` | Tested, not adopted (push pace challenger, race estimation covariance, failed laps) |
| `pace3wa` | v3wa: v3w + car × circuit affinity (tested, not adopted) |
| `pace3`, `pace2q`, `pace3q`, `pace2w`, `pace2s` | Development variants kept for comparison |

## Availability in backtests

Historical data has no publication timestamps, so a session counts as available
once its last lap has ended plus a publication delay (30 minutes for practice and
qualifying, 60 for sprints and races). Results tables can include corrections made
after the session; backtests built on them are retrospective (`archive_proxy`).

## Evaluation

```powershell
make predictor-install                        # the package, then both native modules (needs a Rust toolchain)
f1data reference build                        # canonical IDs, once per archive update
predictor evaluate --session Q --cutoff pre_weekend --reference pace:v3w --models pace3w pace3wa
predictor evaluate --session Q --cutoff last --reference pace:v3w --models pace3w pace3wa
predictor evaluate --session R --cutoff last --seasons 2019 2020 2021 2022 2023 2024 --reference race:v1a --models race1a race2a
```

Rebuild a native module after changing it: `make predictor-native`. Rebuilding
`native/` changes the fit cache key, so every pace model refits; `race_native/`
does not affect cached fits.

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
