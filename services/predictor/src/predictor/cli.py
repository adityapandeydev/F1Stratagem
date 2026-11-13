"""`predictor`: evaluate models walk-forward.

    predictor evaluate --session R --cutoff after_Q --models uniform championship qualifying
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

from . import evaluation
from .evaluation import model_registry

SERVICE_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_SEASONS = (2019, 2020, 2021, 2022, 2023)


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
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = SERVICE_ROOT / "data" / "runs" / f"{stamp}-{args.session}-{args.cutoff}"
    scores, overall, _, comparison, manifest = evaluation.run(args.session, args.cutoff, args.seasons, args.models,
                                                         allow_holdout=args.allow_holdout, out=out,
                                                         reference=args.reference)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(f"{manifest['events']} events, session {args.session}, cutoff {args.cutoff}")
        print(overall.T)
        print()
        print(f"Paired RPS difference vs {args.reference} (negative = better), 90% bootstrap interval:")
        print(comparison.drop(columns=["reference", "metric"]).to_string(index=False))
        stages = evaluation.stage_comparison(scores, args.reference)
        if not stages.empty:
            print()
            print(f"By stage of the season (rounds); a model must not be worse beyond noise in "
                  f"{evaluation.LATE_STAGE} at either cutoff:")
            print(stages[["stage", "model", "events", "mean_difference", "interval_90_low", "interval_90_high",
                          "better_beyond_noise", "worse_beyond_noise"]].to_string(index=False))
        print()
        print("RPS by season:")
        print(evaluation.common_events(scores).pivot_table(index="season", columns="model", values="rps").round(4))
    print(f"Run: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
