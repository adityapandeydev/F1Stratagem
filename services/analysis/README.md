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
analysis season --season 2024 [--compounds slick] [--min-gap 2] [--trim 3] [--trend perlap] [--team-wear] [--agg best]
```

Runs one recipe on every race of the season and prints each race's gaps and the
mean per team.
