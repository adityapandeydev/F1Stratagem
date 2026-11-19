# Analysis service

Measurements computed from the archive's raw data (laps, timing, telemetry,
weather). Every number describes the recorded data of the session(s) it covers:
no priors, no weighting or smoothing across sessions. Smoothed estimates are the
prediction engine's job.

Written in Rust. It reads the archive (`data/`, written by `packages/f1data`)
and never writes to it.

## Build

```powershell
cargo build --release
cargo test --release
```

The archive is found at `--data-root`, else `F1_DATA_ROOT`, else the
repository's `data/`.

## Team race pace

The standard recipe (`pace::team_race_pace`) measures each team's race pace from
one race's own laps:

- Laps: dry tyres, green throughout, after lap 1, no pit in- or out-laps, more
  than 2 s behind the car ahead at the line.
- Fit: lap time = driver + compound offset + tyre-wear rate per compound x tyre
  age + a time for every lap number shared by all cars. The lap-number terms take
  up fuel burn, rubbering-in and changing weather lap by lap; tyre age resets at
  stops, so wear is separable without assuming a fuel rate. Laps more than three
  robust standard deviations from the fit are dropped once and the fit repeated.
- Each driver's pace is the fitted lap on the most-run compound at the median tyre
  age and lap; a team's pace is its quicker driver's. Gaps are given in percent of
  the quickest team's lap, or in seconds.
- A season figure is the mean of a team's per-race gaps over the races it was
  measured in.

## Modules

| Module | Does |
|---|---|
| `table` | Column access over Parquet files (arrow-rs) |
| `archive` | Session lookup from `reference/sessions.parquet`, per-session tables |
| `laps` | Laps in seconds, green/slick/pit tests, gap to the car ahead from lap completion times |
| `pace` | Team race pace from a per-race lap-time regression (compound, tyre age, lap trend or a time per lap number, traffic), quicker car, teammate mean or per-lap quickest car |
| `season` | One recipe over every race of a season, mean gap per team |
| `calibrate` | Runs every team-pace recipe on an event and ranks them by agreement with a reference table |

## Commands

```powershell
analysis calibrate --event 2024-20 --reference table.csv [--top 25] [--detail 3] [--filter text]
```

`table.csv` holds `team,gap` rows, the reference gap of each team to the quickest,
in percent, or in seconds when the file starts with `unit,s`.

```powershell
analysis season --season 2024
```

Runs the standard recipe on every race of the season and prints each race's gaps
and the mean per team. Flags change one part of the recipe: `--compounds
all|slick|inter`, `--min-gap S` (0 keeps all laps), `--trim K` (0 = no trim),
`--trend linear|lapxcomp|perlap`, `--team-wear`, `--traffic`,
`--skip-after-neutral N`, `--agg best|mean|lapbest`.
