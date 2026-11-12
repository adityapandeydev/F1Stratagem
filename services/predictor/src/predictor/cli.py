"""`predictor`: evaluate models walk-forward.

    predictor evaluate --session R --cutoff after_Q --models uniform championship qualifying
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

from . import evaluation
from .baselines import RANKERS

SERVICE_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_SEASONS = (2019, 2020, 2021, 2022, 2023)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="predictor", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="walk-forward evaluation of baselines")
    evaluate.add_argument("--session", choices=["Q", "SQ", "S", "R"], required=True)
    evaluate.add_argument("--cutoff", required=True, help="e.g. pre_weekend, after_FP2, after_Q")
    evaluate.add_argument("--models", nargs="+", choices=sorted(RANKERS), default=sorted(RANKERS))
    evaluate.add_argument("--seasons", type=int, nargs="+", default=list(DEVELOPMENT_SEASONS))
    evaluate.add_argument("--allow-holdout", action="store_true",
                          help="evaluate held-out seasons (2025-2026); only at model selection")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = SERVICE_ROOT / "data" / "runs" / f"{stamp}-{args.session}-{args.cutoff}"
    _, overall, _, manifest = evaluation.run(args.session, args.cutoff, args.seasons, args.models,
                                             allow_holdout=args.allow_holdout, out=out)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(f"{manifest['events']} events, session {args.session}, cutoff {args.cutoff}")
        print(overall.T)
    print(f"Run: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
