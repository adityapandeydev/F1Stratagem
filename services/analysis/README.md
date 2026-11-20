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

Each team's race pace is measured from one race's own laps by a least-squares
fit: lap time = driver + compound offset + tyre-wear rate per compound x tyre age
+ a time for every lap number shared by all cars. The lap-number terms take up
fuel burn, rubbering-in and changing conditions; tyre age resets at stops, so wear
is separable without assuming a fuel rate. Laps more than three robust standard
deviations from the fit are dropped once and the fit repeated. Each driver is
compared at a common reference (most-run compound, median tyre age and lap, free
air); only differences between cars are reported, anchored to the race's median
kept lap, and only between cars the laps connect. A team's figure is its quicker
driver: the car's best demonstrated pace.

Two recipes are kept side by side (`pace::Recipe`):

| | v1 | v2 |
|---|---|---|
| Laps | green, after lap 1, no pit laps, known tyre (dry, intermediate, wet) with a recorded age | same |
| Clear air | more than 2 s behind the car physically ahead at the finish line | more than 2 s clear at both sector lines and the finish line; laps whose traffic is unknown are left out |
| Lap minimum | 5 laps per driver before trimming | rechecked after trimming |
| Single-stint races | wear dropped silently when it cannot be measured | the race is marked not tyre-corrected and does not count in a season |

v2 also reports, per race: each driver's gap, laps, stints and laps shared with
other cars; an 80% interval for every gap from resampling whole stints (gaps to
the measurement's quickest team); and each team's chance of being the quickest.

A season gives, per team, the mean and the median of its per-race gaps, in
percent and seconds, as measured and rebased so the best team is zero, over every
race that counts and over the common races where every team was measured. A race
counts when at least half of its teams are measured (and, in v2, it is
tyre-corrected).

## Modules

| Module | Does |
|---|---|
| `table` | Column access over Parquet files (arrow-rs) |
| `archive` | Session lookup from `reference/sessions.parquet`, per-session tables |
| `laps` | Laps in seconds, green/slick/pit tests, gap to the car ahead from lap completion times |
| `pace` | Team race pace from a per-race lap-time regression: recipes v1 and v2, stint-resampled intervals, per-driver evidence |
| `season` | A recipe over every race of a season: per-race results, mean and median summaries over all and common races |
| `calibrate` | Runs every team-pace recipe on an event and ranks them by agreement with a reference table |

## Commands

```powershell
analysis calibrate --event 2024-20 --reference table.csv [--top 25] [--detail 3] [--filter text]
```

`table.csv` holds `team,gap` rows, the reference gap of each team to the quickest,
in percent, or in seconds when the file starts with `unit,s`.

```powershell
analysis race --event 2024-20 [--recipe v1|v2] [--resamples 200] [--json race.json]
analysis season --season 2024 [--recipe v1|v2] [--resamples 100] [--json season.json]
```

`race` prints one race's teams and drivers; `season` prints each race's team gaps
and the season summaries. `--json` writes the full result. Optional flags change
one part of the chosen recipe for comparisons: `--compounds all|slick|inter`,
`--min-gap S` (0 keeps all laps), `--traffic finish|lines`, `--trim K` (0 = no
trim), `--trend linear|lapxcomp|perlap|timeblocks`, `--team-wear`,
`--traffic-term`, `--skip-after-neutral N`, `--agg best|mean|lapbest`.
