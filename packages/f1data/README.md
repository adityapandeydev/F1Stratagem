# f1data

Builds, verifies and reads the F1Stratagem data archive: FastF1 sessions
(2018+), Jolpica results history (1950+), Open-Meteo weather and FIA documents.
It is the only code that writes the archive's source data; the analysis service
and the prediction engine read it. The one other writer is the prediction engine,
which appends to `data/forecasts/` and nothing else. Design: [design/data-archive.md](../../design/data-archive.md).

## Install

Python 3.10+.

```powershell
uv pip install -e "packages/f1data[dev]"
```

The archive lives in `<repository>/data` unless `F1_DATA_ROOT` or `--data-root`
points elsewhere.

## Rebuild the archive from scratch

FastF1 needs an F1TV account. Authenticate once on the machine; credentials never
go in the repository.

```powershell
python -m fastf1 auth f1tv --authenticate

f1data fastf1 ingest --years 2018 2019 2020 2021 2022 2023 2024 2025 2026 --testing --telemetry --wait-on-rate-limit 30
f1data fastf1 ingest --years 2018 2019 2020 2021 2022 2023 2024 2025 2026 --testing --telemetry --retry-failed
f1data jolpica download --laps
f1data openmeteo download
f1data fia index
f1data fia download
f1data fia extract
f1data reference build
f1data verify
```

Every command resumes: verified files are skipped and cached responses reused.
Each source respects its provider's rate limit and keeps its request window
across restarts (FastF1 500 calls/h, Jolpica ≤ 400 requests/h, fia.com one
request per 10 s). A full FastF1 download takes one to two days.

## Commands

| Command | Does |
|---|---|
| `f1data fastf1 discover\|ingest\|report\|compact\|verify` | Session discovery, download, ledger report, lossless cleanup, checksum audit |
| `f1data jolpica download\|tables\|report\|verify` | API pages, flattened tables, inventory, checksum audit |
| `f1data openmeteo download\|tables\|report\|verify` | Observed weather and forecasts as issued |
| `f1data fia index\|download\|extract\|report\|verify` | Document index, PDFs, parsed tables (PU and gearbox usage, car updates, tyre nominations) |
| `f1data reference build\|report\|verify` | Canonical events, sessions, entrants, drivers, teams, circuits and team lineages (Ergast IDs) |
| `f1data labels export --seasons ...` | Workbook of race and sprint retirements and suspicious finishes, to label by hand |
| `f1data labels import --workbook FILE` | Merge the reviewed rows into `config/race-incident-labels.json` |
| `f1data verify` | Checksum audit of every source; exits non-zero on any mismatch, missing or unlisted file |

`f1data fastf1 ingest` takes `--years` (whole seasons) or `--weekends FILE` (a
selection like `src/f1data/config/pilot-weekends.json`).

## Configuration

- `src/f1data/config/fia-corrections.json`: manual readings of FIA pages without a
  text layer, applied on top of parsed data. Hand-verified; keep it under version
  control.
- `src/f1data/config/pilot-weekends.json`: example weekend selection.
- `src/f1data/config/team-lineage.json`: constructor IDs grouped into team
  lineages since 1996; every constructor-season must map to exactly one.
- `src/f1data/config/race-incident-labels.json`: hand-reviewed causes of race and
  sprint retirements and poor finishes, with sources. Reviewed human work; keep it
  under version control.

## Rules

- One `f1data` writer per data root. Never run two `f1data` downloads against
  the same archive at once, and never write to `forecasts/` from here.
- Values are stored as published. Source errors are flagged, not corrected.

## Tests

```powershell
python -m pytest packages/f1data
```

All tests run offline.
