# F1Stratagem

F1Stratagem is a Formula 1 analysis, prediction and community platform. It
combines a permanent archive of F1 timing, telemetry, results, weather and
official documents with session analysis tools, a probabilistic prediction
engine and a calendar builder where fans design, simulate and share their own
seasons.

The project is under active development. The data archive is complete and
verified; the prediction engine is the current focus; the analysis frontend and
API are being rebuilt around the archive.

## Status

| Area | State |
|---|---|
| Data archive: FastF1 2018–2026, Jolpica 1950–2026, Open-Meteo, FIA documents 2019–2026 | Built, verified (37 GB) |
| Session analysis and multi-driver telemetry comparison | Prototype, being rebuilt on the archive |
| Go API | Prototype, being reworked as the orchestrator of all services |
| Prediction engine | Qualifying pace model implemented and evaluated walk-forward; race model next |
| Calendar builder, community | Planned |
| News and contextual intelligence | Planned |

## Features

### Session and race analysis

Explore every session from 2018 onward: telemetry overlays for up to four
drivers, lap-by-lap and sector analysis, race pace and tyre degradation, pit
stops, qualifying evolution, practice programmes, and historical comparisons
across drivers, teams, circuits and regulation eras. The full catalogue of
metrics and views is in [DataUsePlan.md](DataUsePlan.md).

### Predictions

Probabilistic forecasts for qualifying, sprint and Grand Prix for every driver:
the full finishing-position distribution, pole and win probabilities, a likely
range ("P3–P7"), retirement risk and expected points. Forecasts are issued at
every stage of a weekend (after testing, each practice session, qualifying and
grid penalties) and update live during a race.

- Performance transfers between circuits through measured characteristics, not
  results: corners are classified from telemetry, each circuit is a mix of speed
  regimes, and each car has a measured performance envelope.
- Championship odds are recomputed after every race, for the real calendar and
  for calendars built by users.
- Every published forecast is frozen and hashed before its session starts and
  scored afterwards, so the track record is public and verifiable.

Design: [design/prediction-engine.md](design/prediction-engine.md).

### Strategy and scenarios

Tyre strategy, pit windows, undercut and overcut, degradation and weather
crossover windows, modelled from historical data and the remaining tyre sets of
each driver. Users can change race conditions (rain mid-race, a different
starting compound, safety-car likelihood, track temperature, aggressive or
conservative strategy) and see the simulated outcome.

### Driver and car profiling

Profiles built from telemetry and history: performance by corner type and
mini-sector, braking and throttle application, tyre management, wet-weather
pace, qualifying versus race execution, consistency and risk. Profiles relate
driver style to car characteristics, track layouts and regulation eras, and
track strengths, weaknesses and trends over time.

### Calendar builder

Design a season: choose the number of races, add or remove circuits on a 3D
globe, order them, define rotating slots (for example Monza in even years and
Imola in odd years), and generate multi-year schedules. Calendars are scored for
freight distance, backtracking, regional clustering, timezone transitions,
environmental impact and seasonal weather suitability, with suggestions for
better sequencing. Any calendar can be published, compared with others, and used
to simulate a championship.

### News and contextual intelligence

- Official evidence for the prediction engine: FIA documents, team and Pirelli
  releases, extracted into structured, timestamped records. Upgrade effects are
  measured from timing data, never taken from claims. Design:
  [design/news-intelligence.md](design/news-intelligence.md).
- A retrieval layer over news, FIA documents, technical directives and
  interviews to explain forecasts and track technical changes. Language models
  extract and explain; they never produce forecast numbers.
- A community news feed linking to the original articles, with discussion.

### Community

Public profiles, calendar sharing, voting and discussions.

## Architecture

A monorepo of independently deployable services. A Go API is the single public
entry point; Python services handle analysis and prediction; the data archive is
shared by every service, and each dataset in it has exactly one writer.

| Component | Path | Technology |
|---|---|---|
| Frontend | `frontend/` | React 19, TypeScript, Vite, Bun, Tailwind CSS, ECharts, Three.js |
| API | `api/` | Go, PostgreSQL (pgx) |
| Analysis service | `services/analysis/` | Python, FastAPI, FastF1, pandas |
| Prediction engine | `services/predictor/` | Python, NumPy, SciPy; Rust (PyO3) for the pace model's Kalman filter |
| Data archive | `packages/f1data/` | Python, Parquet, DuckDB |

Under consideration, adopted only where they prove their value: Redis for
caching, gradient boosting or neural models where they beat simpler models in
walk-forward evaluation, and a vector store for the retrieval layer.

System context, deployment, network, sequence and schema diagrams are in
[Architecture.md](Architecture.md).

## Repository layout

```
F1Stratagem/
├── frontend/              web application
├── api/                   Go API
├── services/
│   └── analysis/          session analysis service
├── packages/
│   └── f1data/            data archive: collection, verification, readers
├── design/                subsystem designs
├── data/                  the archive (not in version control)
├── Architecture.md        platform architecture and diagrams
└── DataUsePlan.md         catalogue of metrics and views
```

## Getting started

Requirements: Go 1.27+, Python 3.10+ with [uv](https://docs.astral.sh/uv/),
Bun, PostgreSQL, a Rust toolchain for the prediction engine, and optionally GNU
make (the Makefile wraps the commands below).

```powershell
# configuration
copy .env.example .env                      # then set DATABASE_URL and the DB_* values

# database schema (a migration runner is planned)
psql -d f1stratagem -f api/migrations/001_core_schema.up.sql

# data archive tooling
uv pip install -e "packages/f1data[dev]"
f1data verify                               # audit an existing archive in data/

# prediction engine: the Python package, then its Rust module
uv pip install -e "services/predictor[dev]"
cd services/predictor/native; maturin develop --release

# services, each in its own terminal
cd api; go run ./cmd/server                                   # :8080
cd services/analysis; pip install -r requirements.txt
python -m uvicorn src.main:app --port 8081 --reload           # :8081
cd frontend; bun install; bun run dev                          # :5173
```

The archive is not distributed with the repository. Building it from scratch
takes one to two days and requires an F1TV account for FastF1; see
[packages/f1data/README.md](packages/f1data/README.md).

Tests:

```powershell
cd api; go test ./...
python -m pytest packages/f1data
python -m pytest services/predictor
```

## Data sources

| Source | Data | Licence |
|---|---|---|
| [FastF1](https://github.com/theOehrly/Fast-F1) | Timing, telemetry, weather and race control from the F1 live timing archive, 2018+ | Library MIT; data subject to Formula 1's terms |
| [Jolpica](https://github.com/jolpica/jolpica-f1) | Results, standings, pit stops and lap times, 1950+ | CC BY-NC-SA 4.0 |
| [Open-Meteo](https://open-meteo.com/) | Observed weather and forecasts as issued for every race weekend | CC BY 4.0 |
| [FIA](https://www.fia.com/documents) | Power unit and gearbox usage, car presentations, tyre nominations | All rights reserved; documents are not redistributed |

Weather data by Open-Meteo.com. Commercial use of Jolpica, Open-Meteo's API and
FIA documents requires the providers' permission and will be arranged before any
commercial release.

## Documentation

| Document | Contents |
|---|---|
| [Architecture.md](Architecture.md) | Services, data stores, deployment, request flows, schema, roadmap |
| [design/data-archive.md](design/data-archive.md) | Sources, layout, integrity, extraction, time semantics |
| [design/prediction-engine.md](design/prediction-engine.md) | Model components, simulators, forecast ledger, evaluation |
| [design/news-intelligence.md](design/news-intelligence.md) | News pipeline and upgrade measurement |
| [DataUsePlan.md](DataUsePlan.md) | Metrics and views the platform will offer |
| [packages/f1data/README.md](packages/f1data/README.md) | Building and verifying the archive |

## Roadmap

| Phase | Focus |
|---|---|
| 1 | Core analysis: session browser and telemetry comparison |
| 1b | Monorepo, data archive and canonical driver, team, circuit and lineage identifiers (done); analysis on the archive; typed service contracts |
| 2 | Prediction engine, built and evaluated one component at a time (in progress: qualifying); forecast ledger; championship odds after every race |
| 3 | Driver, car and circuit profiling |
| 4 | Calendar builder, accounts and community; championship simulation on user calendars |
| 5 | Live forecasting and replay; news and contextual intelligence |
| 6 | Strategy tools and user-defined scenarios |

Details per phase: [Architecture.md](Architecture.md#roadmap).

## License

No licence has been chosen yet.

## Disclaimer

F1Stratagem is an unofficial project and is not associated in any way with the
Formula 1 companies. F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD
CHAMPIONSHIP, GRAND PRIX and related marks are trade marks of Formula One
Licensing B.V.

---

Built by a Formula 1 fan, for Formula 1 fans.
