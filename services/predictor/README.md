# predictor

The F1Stratagem prediction engine. Design:
[design/prediction-engine.md](../../design/prediction-engine.md).

Implemented so far: the foundation every model component is evaluated on.

| Module | Role |
|---|---|
| `archive.py` | Cutoff-gated reads. An `ArchiveView` only exposes sessions available at its cutoff; any other read raises `CutoffViolation`. |
| `targets.py` | Final classification and race outcomes per session. Seasons 2025–2026 are held out and need an explicit flag. |
| `rules.py` | Points tables by season. |
| `forecast.py` | The forecast object every model produces, built from joint simulation draws and checked for consistency. |
| `simulate.py` | Plackett-Luce ranking draws and likelihood. |
| `baselines.py` | Reference models: uniform, championship order, practice pace, qualifying order. |
| `evaluation.py` | Walk-forward evaluation and metrics. |

## Availability in backtests

Historical data has no publication timestamps, so a session counts as available
once its last lap has ended plus a publication delay (30 minutes for practice and
qualifying, 60 for sprints and races). Results tables can include corrections made
after the session; backtests built on them are retrospective (`archive_proxy`).

## Evaluation

```powershell
uv pip install -e "services/predictor[dev]"
f1data reference build                        # canonical IDs, once per archive update
predictor evaluate --session Q --cutoff after_FP3
predictor evaluate --session R --cutoff after_Q --models uniform championship qualifying
```

Seasons default to 2019–2023 (development); 2018 is warm-up data only. Every model
is refitted before each event on events that finished before its cutoff. Runs are
written to `services/predictor/data/runs/` with per-event scores, summaries and a
manifest (code revision, reference build, settings).

Metrics: winner (or pole) log loss, ranked probability score, Brier scores for
top 3 and top 10, expected-position error overall and by band (positions 1–5, 6–12,
13+), 80% interval coverage and width, five-way outcome log loss on entrants with
a known outcome, and retirement Brier score.

## Tests

```powershell
python -m pytest services/predictor
```
