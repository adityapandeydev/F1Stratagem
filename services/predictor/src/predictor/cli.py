"""`predictor`: evaluate models walk-forward.

    predictor evaluate --session R --cutoff after_Q --models uniform championship qualifying
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

from . import evaluation
from .evaluation import model_registry
from .fitcache import FitCache

SERVICE_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_SEASONS = (2019, 2020, 2021, 2022, 2023)


def _claim(folder):
    """Create `folder` if it does not exist yet; False if another run took it."""
    try:
        folder.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        return False


def check_batches(manifests):
    """Problems that make runs unfit to average as seed batches (final review 2026-11-26):
    runs differing in what they forecast or in their draws, or repeating a seed batch;
    runs from before the manifest recorded its seed batch are reported, not refused."""
    errors, notes = [], []
    for key in ("session_type", "cutoff", "seasons", "warmup", "models", "events", "allow_holdout",
                "race_draws", "race_fit_draws", "qualifying_draws"):
        values = {json.dumps(m.get(key), sort_keys=True) for m in manifests if key in m}
        if len(values) > 1:
            errors.append(f"runs differ in {key}: {sorted(values)}")
    batches = [m.get("seed_batch") for m in manifests]
    if None in batches:
        notes.append("seed batch not recorded in some manifests (runs before 2026-11-26): not checked")
    known = [b for b in batches if b is not None]
    if len(set(known)) < len(known):
        errors.append(f"a seed batch is repeated: {batches}")
    if len({json.dumps(m.get("git_revision"), sort_keys=True) for m in manifests}) > 1:
        notes.append("runs come from different code revisions")
    return errors, notes


def _compare_batches(args):
    runs = [Path(r) if Path(r).is_absolute() else SERVICE_ROOT / r for r in args.runs]
    manifests = [json.loads((r / "manifest.json").read_text(encoding="utf-8")) for r in runs]
    errors, notes = check_batches(manifests)
    for note in notes:
        print(f"note: {note}")
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1
    scores = evaluation.batch_mean([pd.read_csv(r / "scores.csv") for r in runs])
    print(f"{len(runs)} seed batches, {scores.event_key.nunique()} common events; per-race scores averaged over the batches")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        for metric in ("rps", "outcome_log_loss", "winner_log_loss"):
            if metric in scores:
                c = evaluation.compare(scores, args.reference, metric=metric)
                print(f"\n[{metric}] vs {args.reference} (negative = better):")
                print(c.drop(columns=["reference", "metric"]).to_string(index=False))
        stages = evaluation.stage_comparison(scores, args.reference)
        if not stages.empty and "model" in stages:
            print(f"\nBy stage ({evaluation.LATE_STAGE} must not be worse beyond noise):")
            print(stages[["stage", "model", "events", "mean_difference", "block_90_low", "block_90_high",
                          "verdict"]].to_string(index=False))
        p = scores.pivot_table(index="event_key", columns="model", values="rps")
        if args.reference in p:
            d = p.sub(p[args.reference], axis=0).drop(columns=[args.reference])
            print("\nRPS difference by season:")
            print(d.groupby(d.index.str[:4]).mean().round(6).to_string())
    return 0


def _calibrate(args):
    from . import calibration
    run = Path(args.run)
    if not run.is_absolute():
        run = SERVICE_ROOT / run
    forecasts = pd.read_parquet(run / "forecasts.parquet")
    with pd.option_context("display.width", 200, "display.precision", 4):
        for model in args.models:
            overall, _ = calibration.diagnose(forecasts, model)
            print(f"{model}: randomised PIT, 90% intervals resampling whole races")
            print(overall.to_string(index=False))
            print()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="predictor", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="walk-forward evaluation of baselines")
    evaluate.add_argument("--session", choices=["Q", "SQ", "S", "R"], required=True)
    evaluate.add_argument("--cutoff", required=True, help="pre_weekend, after_<session> (e.g. after_FP2) or last (latest before the target)")
    evaluate.add_argument("--models", nargs="+", choices=sorted(model_registry()), default=["uniform", "championship"])
    evaluate.add_argument("--seasons", type=int, nargs="+", default=list(DEVELOPMENT_SEASONS))
    evaluate.add_argument("--reference", default="uniform", help="model the others are compared against")
    evaluate.add_argument("--allow-holdout", action="store_true",
                          help="evaluate held-out seasons (2025-2026); only at model selection")
    evaluate.add_argument("--save-forecasts", action="store_true",
                          help="also write every forecast per entrant (forecasts.parquet) for error analysis")
    evaluate.add_argument("--no-fit-cache", action="store_true",
                          help="fit every season afresh instead of reusing saved fits (data/fits)")
    evaluate.add_argument("--screen-base", choices=sorted(n for n in model_registry() if n.startswith("pace")),
                          help="quick screen: take this model's fitted settings, fit only what each other "
                               "model adds (never a basis for adoption)")
    evaluate.add_argument("--seed-batch", type=int,
                          help="another set of common random numbers (shared by every model), to check that "
                               "a difference is not Monte Carlo variation")
    evaluate.add_argument("--draws", type=int, help="simulated races per race forecast (default race.DRAWS); "
                          "qualifying forecasts always use pace.DRAWS")
    evaluate.add_argument("--fit-draws", type=int, help="simulated races per race when fitting settings (default race.FIT_DRAWS)")
    batches = commands.add_parser("compare-batches", help="compare models on the mean of several seed batches' runs")
    batches.add_argument("--runs", nargs="+", required=True, help="run folders (data/runs/...), one per seed batch")
    batches.add_argument("--reference", required=True)
    calibrate = commands.add_parser("calibrate", help="randomised PIT calibration of a run's saved forecasts")
    calibrate.add_argument("--run", required=True, help="run folder (data/runs/...) with forecasts.parquet")
    calibrate.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "compare-batches":
        return _compare_batches(args)
    if args.draws is not None or args.fit_draws is not None:
        from . import race
        race.DRAWS = args.draws or race.DRAWS
        race.FIT_DRAWS = args.fit_draws or race.FIT_DRAWS
    if args.seed_batch is not None:
        from . import baselines
        baselines.SEED_BATCH = args.seed_batch
    if args.screen_base and args.no_fit_cache:
        parser.error("--screen-base reads the base model's fits from the fit cache")
    fit_cache = None if args.no_fit_cache else FitCache(SERVICE_ROOT / "data" / "fits")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = SERVICE_ROOT / "data" / "runs" / f"{stamp}-{args.session}-{args.cutoff}{'-screen' if args.screen_base else ''}"
    while out.exists() or not _claim(out):          # two runs started in the same second must not share a folder
        out = out.with_name(out.name + "-b")
    scores, overall, _, comparison, manifest = evaluation.run(args.session, args.cutoff, args.seasons, args.models,
                                                         allow_holdout=args.allow_holdout, out=out,
                                                         reference=args.reference, fit_cache=fit_cache,
                                                         screen_base=args.screen_base,
                                                         keep_forecasts=args.save_forecasts)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(f"{manifest['events']} events, session {args.session}, cutoff {args.cutoff}")
        print(overall.T)
        print()
        if not comparison.empty:
            print(f"Paired RPS difference vs {args.reference} (negative = better), 90% intervals (event and "
                  f"block bootstrap; verdicts use the block interval; margin {evaluation.DEGRADATION_MARGIN}):")
            print(comparison.drop(columns=["reference", "metric"]).to_string(index=False))
        stages = evaluation.stage_comparison(scores, args.reference)
        if not stages.empty and "model" in stages:
            print()
            print(f"By stage of the season (rounds); a model must not be worse beyond noise in "
                  f"{evaluation.LATE_STAGE} at either cutoff:")
            print(stages[["stage", "model", "events", "mean_difference", "block_90_low", "block_90_high",
                          "verdict"]].to_string(index=False))
        print()
        print("RPS by season:")
        print(evaluation.common_events(scores).pivot_table(index="season", columns="model", values="rps").round(4))
    print(f"Run: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
