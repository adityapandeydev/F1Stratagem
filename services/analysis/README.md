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

## Methodology

Two views of race pace, each documented in full:

- [Team race pace](../../design/analysis/team-race-pace.md): each car with its
  quicker driver, measured from one race's own laps on equal tyres and fuel, in
  clear air.
- [Car-only race pace](../../design/analysis/car-only-race-pace.md): the same
  measurement with each driver's contribution removed, using teammate
  comparisons across seasons; each race measured with earlier races only.

## Modules

| Module | Does |
|---|---|
| `table` | Column access over Parquet files (arrow-rs) |
| `archive` | Session lookup from `reference/sessions.parquet`, per-session tables |
| `laps` | Laps in seconds, green/slick/pit tests, gap to the car ahead from lap completion times |
| `pace` | Team race pace from a per-race lap-time regression, stint-resampled intervals, per-driver evidence |
| `season` | A recipe over every race of a season: per-race results, mean and median summaries over all and common races |
| `car_pace` | Car-only race pace: driver skills per season from teammate comparisons, shrunk towards the average, removed race by race |
| `calibrate` | Runs every team-pace recipe on an event and ranks them by agreement with a reference table |

## Commands

```powershell
analysis calibrate --event 2024-20 --reference table.csv [--top 25] [--detail 3] [--filter text]
```

`table.csv` holds `team,gap` rows, the reference gap of each team to the quickest,
in percent, or in seconds when the file starts with `unit,s`.

```powershell
analysis race --event 2024-20 [--resamples 200] [--json race.json]
analysis season --season 2024 [--resamples 100] [--json season.json]
```

`race` prints one race's teams and drivers; `season` prints each race's team gaps
and the season summaries. `--json` writes the full result. Optional flags change
one part of the standard recipe for comparisons: `--compounds all|slick|inter`,
`--min-gap S` (0 keeps all laps), `--trim K` (0 = no
trim), `--trend linear|lapxcomp|perlap|timeblocks`, `--team-wear`,
`--traffic-term`, `--skip-after-neutral N`, `--agg best|mean|lapbest`.

```powershell
analysis car-race --event 2024-20 [--window 6]
analysis car-season --season 2024 [--window 6] [--json car-season.json]
```

`car-race` prints one race's car-only figures for dry and wet laps; `car-season`
prints each race's car-only gaps and the season summaries. `--window` sets the
seasons driver skills come from (the season and those before it). The internal
driver skill table is printed with `--show-skills`; it is not part of the
published outputs.
