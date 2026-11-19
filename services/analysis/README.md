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
| `pace` | Team pace of one session: one-lap best, race lap statistics, lap-by-lap relative pace, per-race lap-time regression (compound, tyre age, lap number, traffic) |
| `calibrate` | Runs every team-pace recipe on an event and ranks them against a published table |

## Commands

```powershell
analysis calibrate --event 2024-20 --reference table.csv [--top 25] [--detail 3] [--filter text]
```

`table.csv` holds `team,pct` rows: a published gap of each team to the quickest,
in percent.
