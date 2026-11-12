# Prediction engine

The prediction engine produces probabilistic forecasts for every entrant of an F1
weekend: qualifying (including sprint grid sessions), sprint and Grand Prix. It runs
as a Python service called by the platform's Go API; it holds no user state.

## Status

| Component | State |
|---|---|
| Data archive (FastF1, Jolpica, Open-Meteo, FIA documents), see [data-archive.md](data-archive.md) | Implemented |
| Pace model, circuit/car profiles, hazards | Planned |
| Car performance envelope and lap simulator | Planned |
| Practice run classification, clean-air tagging | Planned |
| Learned race events, weather scenarios, grid-penalty forecasts, tyre strategy | Planned |
| Qualifying and race simulators | Planned |
| Championship simulation (real and user-built calendars) | Planned |
| Forecast ledger | Planned |
| Live replay and live inference | Planned |

## Outputs

Per entrant, per forecast:

- Full finishing-position distribution and expected position.
- Qualifying: pole probability, position distribution, central 80% interval (e.g. P3–P7).
- Race: five-way primary outcome — `win`, `other_points_finish`,
  `outside_points_finish`, `mechanical_retirement`, `incident_or_other_retirement`
  — summing to 1. DNS/DSQ are explicit exclusions, not forced into a category.
- Podium, top-5, top-10, points and finish probabilities; expected points under
  that season's points rules.
- Championship: drivers' and constructors' title and final-position
  probabilities, for the real calendar and for calendars built by users.

All quantities are derived from one joint simulation per event, so they are
consistent by construction: one winner per simulated race, win ≤ podium, grid-wide
win probabilities sum to 1, each position column sums to 1.

Every forecast records its cutoff, the sources it used, model version, data
freshness and mode (`LIVE`, `REPLAY`, `OFFLINE`).

## Forecast cutoffs

Before the weekend, after pre-season testing, after each practice session, after
sprint grid session and sprint, after qualifying, after published grid penalties,
before the race, and live (per lap and on events: pit stops, SC/VSC, red flags,
rain, incidents, penalties). A forecast may only read data whose availability time
is at or before its cutoff.

## Components

### 1. Pace model

Lap time is decomposed into additive effects:

```
lap_time = session/track + car(team, t) + driver(t) + compound × tyre_age
         + fuel (lap-number proxy) + track_evolution + traffic + noise
```

- Car and driver strength are latent states that evolve over time, not fixed
  per-season constants. Each new session (testing, FP, qualifying, race) updates the
  estimate and its uncertainty.
- Across a winter break the state carries over with inflated variance; inflation is
  larger at regulation changes (2022, 2026). The amount is fitted from history,
  which captures that the front of the field is more stable year to year than the
  midfield. Consequence: early in a season, testing and practice move estimates a
  lot; later, one unusual session moves them little. No hand-set "after race N"
  switch is needed.
- Team × session-type transfer offsets (practice → qualifying, qualifying → race)
  are estimated with shrinkage toward zero. This captures systematic patterns such
  as a team running higher engine modes in practice, only when history supports it.
- Effects are partially pooled, so sparse histories (rookies, new teams) stay
  uncertain instead of extreme.

### 2. Circuit and car profiles

Performance transfers between circuits through measured characteristics, not
through results.

- **Corner classification.** Corner positions from FastF1 `get_circuit_info()`;
  entry, minimum and exit speeds from representative fast laps across seasons.
  Corners are classed slow / medium / fast from measured speed. Layout changes
  produce a new layout ID. Manual review only for anomalies (e.g. corners taken
  flat in qualifying but not in the race).
- **Track profile.** Share of lap time in each regime: full-throttle straights,
  slow, medium and fast corners. A track is a weighted mix, not a single label.
- **Car profile.** Per team per session: time gained or lost to the field in each
  regime, top speed, speed traps, acceleration out of slow corners. This separates
  low drag (high top speed, weak corners) from traction (fast exits that carry onto
  the straight).
- **Expected pace at a circuit** = car regime strengths weighted by the track
  profile, adjusted by recent form and upgrades.

### 2a. Car performance envelope and lap simulator

The regime profile above stays as the simpler baseline and cross-check. On top of
it:

- **Envelope.** Each car's physical limits are fitted per team per weekend from
  telemetry and position data: lateral grip as a function of speed (aero),
  traction out of slow corners, braking, and the power/drag balance seen in
  acceleration and top speed. From 2026, energy deployment limits are part of the
  envelope (speed fade at the end of long straights).
- **Lap simulator.** A quasi-steady-state simulator drives each envelope around a
  layout's geometry (curvature from position data) to give expected lap-time
  differences between cars, broken down by corner and straight.
- **Why it matters.** It explains why a car is fast, transfers performance to
  layouts that changed or have no recent history, and supplies circuit pace for
  user-built calendars.

### 2b. Practice run classification

Fuel load, engine modes and run plans are not observed. Each practice run is
classified (qualifying simulation, long run, short run, programme/aero work) from
stint length, lap-time pattern, compound and timing within the session. The run
type sets the fuel-and-mode uncertainty of every lap entering the pace model, and
pace comparisons are made like for like. This addresses practice pace that does
not carry into qualifying at its source; the team × session offsets in (1) absorb
what remains.

### 2c. Clean-air pace

The gap to the car ahead, from position data, tags each lap as free air or
traffic. The pace model is fitted on free-air laps, with a learned dirty-air
penalty applied to the rest.

### 3. Hazards and circuit priors

- Mechanical retirement hazard: team-season reliability, component penalties.
- Incident hazard: grid position, start, circuit incident rate, driver history.
- Circuit priors: SC/VSC/red-flag rates, pit loss, overtaking difficulty, typical
  strategies by compound.
- Unknown retirement causes stay unknown; cause-specific models train only on
  evidence-resolved labels.
- SC/VSC/red-flag risk is a per-lap hazard, including the lap-1 spike.

### 3a. Learned race events

Each race-simulator rule is a model fitted from data rather than a fixed rule:

- **Start:** positions gained or lost on lap 1 by grid slot, grid side and team
  launch performance.
- **Overtaking:** pass probability learned from real wheel-to-wheel battles in
  position data, as a function of pace delta, circuit, tyre offset and DRS (or the
  2026 overtake mode).
- **Pit decisions:** pit-stop hazard conditioned on tyre age, compound, gaps and
  neutralisations; undercut and overcut gains measured per circuit.

### 3b. Weather scenarios

Each simulated race samples a weather scenario (onset time and intensity of rain,
or none) instead of a single rain flag. A scenario changes lap times, the
slick-to-intermediate crossover lap, pit timing, neutralisation and incident rates,
and applies learned per-driver and per-team wet-pace effects. Track temperature
feeds tyre degradation; wind enters through the regime profile.

- **Before the session:** the forecast as issued at the cutoff, from the same
  provider as the training data (archived forecast runs 0–7 days ahead). Raw
  forecast probabilities are calibrated against observed weather per circuit
  before use.
- **During the race:** three inputs, refreshed every 10–15 minutes rather than per
  lap: the circuit weather station in the timing feed (rain flag, air and track
  temperature), a short-range forecast or nowcast, and track evidence (lap-time
  drops, switches to intermediates). Track evidence dominates once conditions
  change.
- **Fetching:** forecasts are fetched by the server on a schedule and stored with
  their issue time in the forecast ledger; every user reads the stored copy.

### 3c. Grid-penalty forecasts

Each driver has a per-season allocation of every power-unit element and gearbox
component, and a pool of the elements introduced so far. Rotating between
introduced elements is free. A grid penalty applies once per additional element,
at the first event where it is used.

- Input: FIA usage sheets (elements introduced per driver, per event) and the
  per-event notices of which elements are fitted.
- Allocations change by season and are read from a per-season configuration taken
  from the sporting regulations (2026 includes a bonus element for the new rules).
- Output: per driver and upcoming event, the probability of introducing an element
  beyond the allocation, from element ages, failure history and how teams time
  penalties (low-penalty circuits, back-of-grid starts). Pre-weekend and
  championship forecasts sample these penalties.
- Because the penalty counts once per element, one wrong usage figure misplaces a
  penalty in two events: the event where it is wrongly applied and the later
  event where it should have been. FIA sheets are occasionally corrected after
  publication.
  - **Upcoming events:** before a forecast is published, each penalty the model
    is about to apply is shown to an operator, who confirms or rejects it against
    current news. The decision is stored with the forecast.
  - **Backtests:** historical sheets are used as published (later sheets already
    contain the corrections), with no confirmation step.

### 3d. Tyre allocation and strategy

- **Sets available:** after qualifying and after the sprint, each driver's
  remaining sets by compound and new/used state, from the weekend allocation
  (per event; sprint and alternative-allocation weekends differ) minus sets used in
  practice and qualifying.
- **Compounds:** each event's nomination (C1–C6) from the Pirelli preview, mapped
  to stable compound identities so a track's history can be adjusted when this
  year's choice is harder or softer.
- **Degradation, not habit:** stop counts come from measured degradation per track
  and compound (fuel-corrected lap-time slope per stint, and whether it
  accelerates into a cliff), not from last year's stop count. Low degradation with
  harder compounds can turn a two-stop race into a one-stop; accelerating
  degradation keeps it a two-stop.
- **Strategy options:** feasible strategies per driver from the sets left, the
  degradation curves, pit loss, safety-car likelihood, weather scenarios and grid
  position. The race simulator samples from these, and last year's optimum at the
  track is kept for comparison.

### 4. Qualifying simulator

Simulates Q1/Q2/Q3 knockouts (or sprint grid equivalents) from the pace model:
session lap-time spread, track evolution, mistakes, red flags. 10k+ seeded
simulations give the position distribution and intervals.

### 5. Race simulator

Lap-level Monte Carlo over the whole field:

- Lap time from pace state, tyre degradation, fuel, dirty-air penalty.
- Overtaking probability from pace delta and circuit difficulty.
- Sampled strategies; pit loss; SC/VSC/red flags shared by all cars; multi-car
  incidents; per-lap retirement hazards.
- Starts, overtakes and pit decisions come from the learned event models (3a);
  weather from the scenarios (3b); grid penalties from (3c); strategies and tyre
  sets from (3d).
- Classification rules (90% distance) and season points rules applied per draw.

The pre-race forecast is the simulator started at lap 0. Live forecasting starts
the same simulator from the observed race state, so pre-race and live share one
model.

### 6. External evidence

Upgrade and team news arrive from the news service as timestamped structured
records. Their effect is estimated from subsequent timing data (see
[news-intelligence.md](news-intelligence.md)); claimed gains are not used as
numbers. Effects are measured as changes in the car envelope (2a) and regime
profile (2).

### 7. Championship simulation

The season simulator runs the weekend simulator over every remaining event from
the current standings, with car and driver states evolving between events
(including expected development and upgrade uncertainty) and grid penalties
sampled.

- Rerun after every race, and at the season start after testing. Uncertainty
  narrows as the season progresses; the history of title probabilities across
  the season is stored and shown, with points scenarios for each contender.
- **User-built calendars:** the same simulator runs over a calendar made in the
  platform's calendar builder, using the lap simulator (2a) for circuit pace and
  the circuit priors for event hazards. Circuits without recent F1 data are
  flagged with wider uncertainty. The calendar builder's core purpose is picking
  and sharing calendars; championship prediction is an optional layer on top.
- **Weather for user calendars:** dates are usually beyond any forecast, so they
  use climatology: a per-circuit, per-day-of-year table built once from decades of
  reanalysis (rain probability in a race window, temperature distribution,
  probability of freezing, snow or extreme heat). It needs no API calls per user.
  - Blocked dates come from thresholds on this table, not hand-picked months.
    Borderline dates show a warning.
  - A race at risk of being unrunnable (snow, extreme heat) is cancelled in that
    share of simulated seasons and scores nothing there, so the title odds carry
    the risk.
- **Cost:** results are cached by calendar, and per-circuit, per-date work is
  reused across calendars. Simulations per user per day are limited by compute.

### 8. Forecast ledger

Every published forecast is frozen before its session starts: payload, model
version, training-data fingerprint and input cutoff are written to the archive and
identified by a SHA-256 hash. Forecasts are never edited after publication;
corrections are new versions.

- Published: frozen forecasts, calibration and interval coverage, and the
  methodology. Fixed seeds make any forecast reproducible from its recorded inputs.
- Betting-market odds, where licensable, are a benchmark only, never a model input.
- Results far from the forecast are surfaced as "against the forecast" moments
  (a race win from a low pre-race probability, a title won against a season of
  low title probability), shown with the full frozen probability history so the
  story is verifiable and shareable to the community.

## Training and evaluation

- Walk-forward over races. 2018 is warm-up; 2019–2023 development and calibration;
  2024 validation; 2025–2026 held out until model selection. 2026 doubles as a
  regulation-change stress test.
- Whole events stay together; nothing from the target event or later is used.
- Baselines: uniform, qualifying order, championship order.
- Metrics: log loss, Brier, ranked probability score, calibration, position MAE,
  interval coverage (an 80% interval should contain the result ~80% of the time),
  expected-points error, retirement calibration. Reported by season, circuit,
  weather, cutoff and position band (front, midfield, back), with uncertainty.
- The model is built one component or feature at a time. Each addition is
  evaluated walk-forward over hundreds of events and is kept if it improves the
  overall score beyond run-to-run noise. It is also kept if it clearly improves a
  class of events (wet races, safety-car races, high-degradation tracks, street
  circuits) without hurting the overall score. Otherwise it is dropped and logged
  with its result.
- A new version is promoted only if it beats the current one on held-out events.

## Known constraints

- Historical archive data has no verified publication timestamps; backtests on it
  run in `archive_proxy` mode and are labelled as retrospective. Data captured live
  from go-live onward supports strict backtests.
- Fuel load, engine modes and run plans are not observed; the model carries them
  as uncertainty, not invented values.
- 2026 grid has 22 cars; entrant and points rules are versioned by season.
- Open: whether "open" means published methodology plus reproducible frozen
  forecasts, or also open-sourcing the model code.
