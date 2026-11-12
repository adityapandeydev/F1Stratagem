"""`f1data`: build, verify and inspect the F1Stratagem data archive.

    f1data fastf1    discover | ingest | report | compact | verify
    f1data jolpica   download | tables | report | verify
    f1data openmeteo download | tables | report | verify
    f1data fia       index | download | extract | report | verify
    f1data reference build | report | verify
    f1data labels    export --seasons ...
    f1data verify    every source

This is the only process that writes the archive's source data (the prediction
engine owns forecasts/). Never run two at once on one data root.
"""

import argparse
import json
from pathlib import Path
import sys

from . import audit, fia, jolpica, labels, openmeteo, reference, sessions
from .paths import default_data_root


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path,
                        help="archive directory (default: $F1_DATA_ROOT, else <repository>/data)")
    parser = argparse.ArgumentParser(prog="f1data", description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    commands = parser.add_subparsers(dest="source", required=True, metavar="SOURCE")
    for module in (sessions, jolpica, openmeteo, fia, reference, labels):
        module.register(commands, common)
    everything = commands.add_parser("verify", parents=[common], help="verify every source")
    everything.set_defaults(handler=verify_all)
    return parser


def verify_all(args):
    problems = sessions.verify_all(args.data_root)
    results = {"fastf1": {"checksum_failures": problems, "ok": not problems}}
    for source in audit.SOURCES:
        results[source] = audit.verify_source(source, args.data_root)
    print(json.dumps(results, indent=2), flush=True)
    return 0 if all(result["ok"] for result in results.values()) else 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.data_root = (args.data_root or default_data_root()).resolve()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
