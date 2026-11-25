# Prediction engine

The prediction engine produces probabilistic forecasts for every entrant of an F1
weekend: qualifying (including sprint grid sessions), sprint and Grand Prix. It runs
as a Python service called by the platform's Go API; it holds no user state.

## Status

| Component | State |
|---|---|
| Data archive (FastF1, Jolpica, Open-Meteo, FIA documents), see [data-archive.md](data-archive.md) | Implemented |
| Cutoff-gated archive access, targets, forecast object, baselines, walk-forward evaluation | Implemented |
| Pace model for qualifying: car, driver, team × session offsets, weekend effect, winter reset | Implemented (v3w) |
| Corrected race pace as evidence (tyres, tyre age, fuel lap by lap) | Implemented (v3wr) |
| Qualifying simulated through Q1, Q2 and Q3 (eliminations, a lap per segment) | Implemented (v3wrq) |
| Wet and mixed sessions as their own evidence types (own noise) | Implemented (v3wrqw) |
| Robust Student-t observation noise | Implemented; in every champion (v3wrqwt) |
| Race pace read from the best-connected group of cars; measured estimation uncertainty with a fitted stint scale | Implemented; qualifying champions v3wrqwt-rp (after a session), v3wrqwt-rps (before the weekend), v3wrqwt-rpc (wet); pace of the race champions |
| Wet forecast mode: wet qualifying forecast with wet-session noise | Implemented; serves the wet toggle (v3wrqwtc) |
| Wet races read with a drying curve and tyre-class crossover (x), driver wet ability (y), wet track evolution in qualifying (e), wet segments typed by tyres (g), driver × circuit affinity (i) | Tested, not adopted; kept for the re-test after the holdout check |
| Champion router: a model per forecast moment and condition | Implemented |
| Fits drop each state after its last use | Implemented (identical results; 2.3–2.5× faster fits with long history) |
| Race forecasts from the pace model: race pace, grid at the cutoff, retirements (stage 1a) | Implemented; half of the after-qualifying and published-grid race mixtures |
| Grid slot weighted by circuit overtaking difficulty (stage 1b) | Tested, not adopted |
| Lap-by-lap race simulation with a battle model learned from position data (stage 2a) | Implemented; half of the pre-weekend race mixture |
| Native race simulation (Rust, PyO3) | Implemented (same forecasts; 27–41× faster) |
| Forecast roster from the FIA entry list known at the cutoff | Implemented |
| Native Kalman filter (Rust, PyO3): weighted observations, per-group estimation covariance, Student-t updates | Implemented |
| Fit cache (fits reused across cutoffs and runs) and quick screens | Implemented |
| Track profiles from telemetry, circuit layout table | Implemented |
| Practice run classification (push laps, long runs) | Implemented; per-driver push pace (v3wrqwp) in evaluation |
| Race pace estimation uncertainty in the filter (v3wru), failed laps in qualifying (v3wrqf) | Tested, not adopted |
| Qualifying history 1996–2017, driver season form, car × circuit affinity, driver corner skills | Tested, not adopted (see Training and evaluation) |
| Retirement causes for 2023–2024 from reviewed race reports | Implemented |
| Safety cars, virtual safety cars and red flags in the race simulation (stage 2b) | Implemented; simulation base (race:v2b) |
| Tyre wear and pit stops in the race simulation (stage 3a) | Implemented (race:v3a) |
| Stage 3: per-car stop counts, pit-lane slow stops, team tyre wear, wear from lap effects, following gaps, learned pit windows | Implemented; the simulator half of every race champion |
| Learned per-lap pit decisions, learned gambles, start-tyre roles, following penalty and corrected race pace, fresh-tyre peak, clean-air race pace, circuit pass slope | Tested, not adopted; kept in code (parked, see Results 2026-11-19 to 2026-11-25) |
| Forecast mixtures (frozen 50/50 of the simulator and a simpler model) | Implemented; every race champion is one |
| Published starting grid and grid penalties known at the cutoff (stage 4a) | Implemented; served once the grid is published and before the weekend; penalties after qualifying not adopted |
| Stable settings fit (keyed random numbers, separate fit seeds, choice smoothed over neighbouring settings) | Implemented; serves before the weekend |
| Car profile from telemetry (straight-line speed, acceleration, corner bands) | Measured: repeats beyond lap time; not adopted in the battle model; pace-model test next |
| Weather in the race simulation, wet crossover (stage 4b) | Planned |
| Circuit/car profiles, hazards | Planned |
| Car performance envelope and lap simulator | Planned |
| Practice run classification, clean-air tagging | Planned |
| Learned race events, weather scenarios, tyre sets | Planned |
| Power-unit penalty forecasts | Dropped (penalised cars finish near their qualifying place; see Results 2026-11-19 to 2026-11-25) |
| Championship standings (points, fastest lap, reduced-points races, timed constructor deductions, countback) | Implemented; official standings 2019–2024 reproduced exactly |
| Championship simulation on the real calendar (stage 5): simple race model, season strength shared within a simulated season | Implemented; beats points-per-race projection in every season |
| Championship simulation on user-built calendars | Planned |
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

Logical output model (field names indicative):

```mermaid
classDiagram
    direction LR
    class Forecast {
        +forecast_id
        +event
        +session: qualifying | sprint_grid | sprint | race
        +cutoff
        +mode: LIVE | REPLAY | OFFLINE
        +model_version
        +data_fingerprint
        +sources
        +seed
        +sha256
        +supersedes
    }
    class EntrantForecast {
        +driver
        +team
        +position_distribution
        +expected_position
        +interval_80
    }
    class QualifyingOutcome {
        +pole_probability
    }
    class RaceOutcome {
        +win
        +other_points_finish
        +outside_points_finish
        +mechanical_retirement
        +incident_or_other_retirement
        +podium, top5, top10, points, finish
        +expected_points
    }
    class AppliedPenalty {
        +driver
        +element
        +grid_places
        +operator_decision
    }
    class ChampionshipForecast {
        +season
        +calendar: real | user-built
        +after_round
        +title_probability
        +final_position_distribution
        +sha256
    }
    Forecast "1" *-- "20..22" EntrantForecast : entrants
    EntrantForecast "1" *-- "0..1" QualifyingOutcome
    EntrantForecast "1" *-- "0..1" RaceOutcome
    Forecast "1" *-- "0..*" AppliedPenalty : penalties applied
    ChampionshipForecast ..> Forecast : built from weekend simulations
```

## Forecast cutoffs

Before the weekend, after pre-season testing, after each practice session, after
sprint grid session and sprint, after qualifying, after published grid penalties,
before the race, and live (per lap and on events: pit stops, SC/VSC, red flags,
rain, incidents, penalties). A forecast may only read data whose availability time
is at or before its cutoff.

```mermaid
flowchart LR
    prior["Season prior<br/>last season + winter variance"] --> test["After testing"]
    test --> pre["Before the weekend"]
    pre --> fp1["After FP1"]
    fp1 --> fp2["After FP2"] --> fp3["After FP3"] --> quali["After qualifying"]
    fp1 -->|"sprint weekend"| sq["After sprint grid session"] --> sprint["After sprint"] --> quali
    quali --> pen["After grid penalties"] --> race["Before the race"]
    race --> live["Live: per lap and on events"]
    live --> review["Post-race review<br/>state update"]
    review -->|"next event"| pre
```

## Components

Data flow through the components below (section numbers in brackets):

```mermaid
flowchart TB
    subgraph inputs["Archive inputs (as of the cutoff)"]
        tel["Sessions: laps, telemetry, position"]
        hist["Jolpica history, pit stops"]
        wx["Weather: forecasts as issued, observed"]
        fiat["FIA: starting grids, penalty decisions, car updates, tyre nominations"]
        newsr["News records (6)"]
    end

    subgraph measure["Measured profiles"]
        corners["Corner classes, track profiles (2)"]
        carprof["Car regime profiles (2)"]
        env["Car envelope and lap simulator (2a)"]
        runs["Practice run classification (2b)"]
        air["Clean-air tagging (2c)"]
        deg["Degradation curves, tyre sets left (3d)"]
    end

    pace["Pace model: car and driver state, session offsets (1)"]

    subgraph events["Event models"]
        hz["Hazards and circuit priors (3)"]
        learned["Starts, overtakes, pit decisions (3a)"]
        scen["Weather scenarios (3b)"]
        pens["Starting grid and known grid penalties (3c)"]
        strat["Strategy options (3d)"]
    end

    subgraph sims["Simulators"]
        qsim["Qualifying simulator (4)"]
        rsim["Race simulator (5)"]
        csim["Championship simulator (7)"]
    end

    ledger[("Forecast ledger (8)")]

    tel --> corners & carprof & env & runs & air & deg
    hist --> hz & learned
    newsr --> carprof & env
    corners --> carprof
    carprof & env & runs & air --> pace
    wx --> scen
    fiat --> pens & deg
    deg --> strat
    scen --> strat
    pace --> qsim & rsim
    hz & learned & scen & pens & strat --> rsim
    pens --> qsim
    qsim --> rsim
    rsim --> csim
    qsim & rsim & csim --> ledger
```

### 1. Pace model

Lap time is decomposed into additive effects:

```
lap_time = session/track + car(team, t) + driver(t) + compound × tyre_age
         + fuel (lap-number proxy) + track_evolution + traffic + noise
```

**Implemented for qualifying (`pace.py`).** Pace is log lap time × 100 (about
percent of a lap; lower is faster). A driver's pace in a session is

```
car(lineage, t) + driver(t) + offset(lineage, session type)
    + weekend(lineage, event) + affinity(lineage, circuit) + noise(session type)
```

- **States.** A Kalman filter carries all states jointly. Car (per team lineage)
  and driver are random walks whose variance grows with days elapsed. Teammates
  share the car state, which separates car from driver.
- **Evidence.** Qualifying and sprint-qualifying segments, each practice
  session's best clean lap per driver, and testing days. Every driver is compared
  with a reference driver in the same segment or session, so track evolution
  between segments does not leak in; laps beyond 107% of the group's best are
  dropped. Qualifying is the reference session type; every other type has its own
  noise and a per-lineage offset with a zero-mean prior (a team flattered by its
  practice running gets a negative offset only as far as the data supports it).
- **Weekend effect.** Each lineage has an effect shared by every session of one
  event and reset at the next. It holds track-specific and one-off swings (setup,
  conditions, upgrades), so they do not move long-run pace. Before the weekend it
  is unknown (prior spread, shared by teammates); after practice it carries what
  the weekend's practice showed into the same weekend's qualifying.
- **Car × circuit affinity (tested, not adopted).** A persistent per-circuit effect for
  each lineage, kept between visits. Its variance grows with time and at each
  season start, multiplied at regulation changes; its value is kept, so recent
  visits weigh more and an earlier regulation era still counts. Only the current
  circuit's affinities are active in the joint state; others are parked with mean
  and variance and resume with the variance accrued while parked.
- **Driver × circuit affinity (tested, not adopted).** The same structure per
  driver and circuit, without the regulation multiplier (a driver's liking for a
  track does not reset with the cars). Measured before modelling: a driver's
  deviation from their usual teammate gap repeats at the next visit to the same
  circuit (correlation 0.09, 90% interval 0.04–0.15; about 0.11% of a lap). The
  fitted spread is small (0.05–0.06) and forecasts were slightly better at both
  cutoffs but within noise; it waits for the re-test with more visits.
- **Winter reset.** Each season start adds variance to cars and drivers, multiplied
  for cars in seasons that begin under new technical regulations (2014, 2017, 2022,
  2026). Early-season sessions therefore move estimates a lot and mid-season ones
  little; there is no hand-set "after race N" switch.
- **Settings.** Noise levels, drift rates, prior spreads, winter terms and the
  regulation multiplier are fitted by maximum one-step predictive likelihood on
  sessions available before each evaluated season. A setting is fitted only when
  the data can identify it (offsets only with non-qualifying evidence; the
  regulation multiplier only once the data crosses a regulation change; affinity
  only once the data spans two seasons, so circuits have been revisited).
- **Fitting.** Fit quality matters more than run time. Settings are fitted as
  logs within bounds. Each round runs Powell, then L-BFGS-B and adaptive
  Nelder-Mead from the best point so far, and rounds repeat until one gains less
  than 0.01 in log-likelihood; methods stall in different places, so alternating
  them escapes false optima. Fits start from the previous season's settings and
  from defaults; a model that extends another (v3wa extends v3w) also starts from
  the smaller model's optimum with the extra terms barely on, so its fit can
  never end below the model it extends. Every fit records its convergence,
  evaluations and the log-likelihood reached from each start.
- **Forecast.** States are drawn jointly from the posterior (10,000 draws) with
  qualifying noise added, and each draw is ranked.
- **Retired states.** A fit drops every state after its last use (a driver or
  team that has left), so long histories do not carry dead states through every
  update. Dropping a state at the right moment leaves the likelihood exactly
  unchanged (tested); fits with qualifying history from 1996 run 2.3–2.5× faster.
  Forecasts keep every state, since a driver can return.

```mermaid
flowchart LR
    subgraph evidence["Evidence up to the cutoff"]
        q["Qualifying and sprint-qualifying segments"]
        fp["Practice: best clean lap per driver"]
        t["Testing days"]
    end
    subgraph state["Joint state (Kalman filter)"]
        car["Car pace per lineage<br/>random walk + winter reset"]
        drv["Driver pace<br/>random walk + winter reset"]
        off["Team x session offsets<br/>FP1, FP2, FP3, SQ, testing"]
        wk["Weekend effect<br/>reset each event"]
        aff["Car x circuit affinity<br/>kept between visits"]
    end
    q & fp & t -->|"relative laps within a group"| state
    state --> draws["10,000 joint draws + qualifying noise"]
    draws --> rank["Ranked qualifying orders"]
    rank --> fc["Forecast: position distributions, pole, 80% intervals"]
```

Lessons from development:

- Adding practice without a weekend effect made forecasts before the weekend
  worse. With nowhere to put track-specific swings, the fitted car drift rose
  about fivefold and the model forgot too fast. The weekend effect brought the
  drift back down (0.10 to 0.007 per 14 days) and fixed it. Fitting the settings
  on qualifying likelihood alone did not.
- The winter reset had no effect while car drift was inflated; once drift was
  fixed it improved the 2022 regulation-change season at both cutoffs.
- The first fitter (Nelder-Mead alone) stopped short of the optimum, by up to
  10 log-likelihood. With one season of history it gave the affinity a drift so
  large that forecasts were near-uniform. The current fitter reaches the optimum
  from every start tried; refitting v3w with it improved RPS before the weekend
  beyond noise (0.1040 to 0.1038) and was within noise at the last cutoff
  (0.0990 to 0.0989).
- Car × circuit affinity (v3wa), fairly fitted, gains +3.6 log-likelihood on
  2018–2022 for three extra settings, with a small fitted spread (0.09) and
  almost no growth over time. Forecasts were not better: +0.0003 RPS before the
  weekend (within noise; worse beyond noise on each season's first three rounds)
  and +0.0003 at the last cutoff (worse beyond noise). With 2018 onwards most
  car–circuit pairs have only a few visits; the retest adds Jolpica qualifying
  from 1996, and circuit characteristics from telemetry are a second route to
  the same effect.

Not yet modelled: compound and tyre age, fuel, traffic, and races.

**Python and Rust.** Rust is chosen per component on function, not to grow its
share of the code. A component moves to Rust when that makes the model faster or
better and its forecasts stay the same: identical outputs where the same random
numbers can be fed in, otherwise a walk-forward gate no worse at either cutoff.
Equal performance is still enough, because a compiled language catches more
mistakes before a run. Python stays where it is clearly the better tool:
optimisers (SciPy), data handling (pandas), evaluation and tests. When Rust
replaces Python, the Python version is deleted and reference values it produced
stay in the tests. Two modules exist:

| Module | Role | Why Rust |
|---|---|---|
| `predictor_native` (`services/predictor/kernels/pace`) | Kalman filter replay for the pace model | Fits replay the filter thousands of times: 4.3–6.4× faster fits |
| `predictor_race_native` (`services/predictor/kernels/race`) | The race simulation for every draw, random numbers included | The lap loop was 90% of a race-model evaluation: 27–41× faster |

They are separate because the fit cache fingerprints `predictor_native`:
rebuilding it invalidates every cached pace fit (hours of refitting), so race
code changes must not touch it.

Planned moves:

- **Race model:** later stages (safety cars, tyres and strategy, championship
  simulation) are written in Rust from the start, in `predictor_race_native`.
- **Qualifying model:** the Q1/Q2/Q3 simulation, the sampling of forecast draws
  and the recording of the filter program are candidates. They are moved together
  in one rebuild of `predictor_native` when the race work is carried back into
  qualifying, so the pace fits are refitted once and checked for identical
  results. The settings optimiser stays in SciPy unless a Rust optimiser reaches
  optima as good.

The filter is split in two. Which states exist, when they
are added, parked, reset or removed, and which states each observation touches
depend on the sessions only, never on the settings. Python records that once
as a flat program (`program.py`); a Rust extension module, `predictor_native`
(`services/predictor/kernels/pace`, built with maturin and PyO3), replays it for a
set of settings and returns the log-likelihood and the final mean and
covariance. A settings fit records once and replays the program for every
candidate; a forecast records up to its cutoff and samples from the returned
state in Python. Everything that changes often (evidence, features, model
variants, evaluation) stays in Python; only the arithmetic, which rarely
changes, is in Rust.

Rust replaced the Python filter after a side-by-side comparison. Forecasts
were the same: on the 2019–2023 gate both versions gave identical verdicts and
the largest per-event RPS difference was 0.000001; on real data before 2023 the
final state and every 2023 forecast agree to 1e-15. Fits were faster:

| Fit | Python | Rust | Speed-up |
|---|---:|---:|---:|
| v3w, before 2020 | 185 s | 36 s | 5.1× |
| v3wa, before 2020 | 1,332 s | 209 s | 6.4× |
| v3w, before 2023 | 1,238 s | 287 s | 4.3× |
| v3wa, before 2023 | 12,511 s | 2,172 s | 5.8× |
| Full gate (v3w + v3wa, both cutoffs) | 6 h 40 min | 1 h 14 min | 5.4× |

A fit spends almost all its time in sequential optimiser steps, so running
candidates on several cores did not help (192 s against 193 s); making each
evaluation cheaper did. Reference values written by the Python filter before it
was removed are kept in the tests (`tests/data/pace_golden.json`).

```mermaid
flowchart LR
    subgraph py["predictor (Python)"]
        arch["archive.py<br/>cutoff-gated reads"]
        pace["pace.py<br/>evidence, model variants"]
        rec["program.py<br/>Recorder: state bookkeeping"]
        opt["scipy optimiser<br/>settings fit"]
        fc["Forecast: joint draws,<br/>Q1/Q2/Q3 simulation"]
        race["race.py<br/>race settings fit, context"]
        sim["race_sim.py<br/>battle model, race length and lap"]
    end
    subgraph rs["predictor_native (Rust, PyO3)"]
        kf["Kalman filter replay"]
    end
    subgraph rrs["predictor_race_native (Rust, PyO3)"]
        lap["Race for every draw:<br/>start, laps, passes, random numbers"]
    end
    arch --> pace --> rec
    rec -->|"program (numpy arrays)"| kf
    opt -->|"candidate settings"| kf
    kf -->|"log-likelihood"| opt
    kf -->|"mean, covariance"| fc
    fc -->|"race pace and grid draws"| race
    race --> sim
    sim -->|"pace, grid, battle model, seed"| lap
    lap -->|"finishing orders"| sim
```

The native race simulation, against the NumPy version it replaced (20 cars, 57
laps):

| Simulated races | NumPy | Rust, NumPy random numbers | Rust | Speed-up |
|---|---:|---:|---:|---:|
| 1,000 | 0.10 s | 0.025 s | 0.004 s | 27× |
| 10,000 | 1.3 s | 0.18 s | 0.032 s | 41× |

It moved in two steps. First the lap alone, fed NumPy's random numbers, so the
finishing orders were identical to the NumPy version. Then the random numbers:
each draw has its own PCG64 generator seeded from the forecast's seed and the
draw index, so a forecast is reproducible and does not depend on the number of
threads (draws run in parallel). Its races differ from NumPy's draw by draw but
not in distribution: on 100,000 simulated races, position probabilities differ
by no more than two NumPy runs with different seeds (z-scores of the differences
spread 0.87, none beyond 2.4, mean −0.03).

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

### 3c. Starting grids and grid penalties

**Implemented (stage 4a).** The FIA starting grid (provisional about three hours
after qualifying, final about two hours before the race) is parsed for 2020–2024
and becomes its own cutoff, `after_grid`, once published. Before it is out, grid
penalty decisions published by the cutoff (decisions and infringements, with the
session that serves them, replaced decisions dropped) are applied to the
qualifying order with the grid rules of their season: the empty-grid procedure,
and back-of-grid groups for more than 15 places from late 2022. Qualifying plus
decisions rebuilds 96 of 105 provisional grids exactly. A car withdrawn on the
published grid sits out every simulated race.

Served: the published grid, and penalties known before the weekend. Penalties
known after qualifying are not served: penalised cars finish near where they
qualified, and neither one grid weight nor a split grid term (qualifying slot as
pace signal, starting slot as track position) turned that into a better forecast.

**Dropped: power-unit penalty forecasts.** The design forecast, per driver and
event, the chance of introducing an element beyond the allocation (from FIA usage
sheets, element ages and how teams time penalties), confirmed by an operator
before publication. It adds little to the finishing order for the same reason, so
penalties are to act through learned recovery and the car's pace instead.

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

The race model is built in stages, each gated walk-forward against the stage
before (race cutoffs: before the weekend, and the last cutoff before the race).
A race forecast is 10,000 simulated races; each uses its own luck (starts, lap
noise, passes, retirements) from a seed fixed per race, so forecasts are
reproducible and every model sees the same luck for the same race.

**Stage 1a: race pace, grid, retirements (implemented; race champion
`race:v1a`).** Each draw takes the pace model's race-pace state for every driver,
plus race noise (Student-t when the model is robust). The grid is the qualifying
classification once it is known at the cutoff (in 2021–2022 the sprint set the
grid once it had run), with drivers missing from it at the back; before
qualifying the grid is simulated jointly from the same draws through Q1, Q2 and
Q3. A driver's race score is race pace + kappa × (grid slot − 1) + sigma × noise,
and the order of scores is the finishing order; retirements come from the
retirement model (historical rate and mechanical share). Kappa and sigma are
fitted per season and cutoff by ranked probability score on earlier seasons'
races forecast from the same cutoff. Race pace forecasts are calibrated as
race-time predictions (slope 0.95, correlation 0.9 against measured race pace).

**Stage 1b: overtaking difficulty by circuit (not adopted).** The green-lap
pass rate of a circuit repeats from year to year (correlation 0.48 across
seasons; Monaco 0.04 passes per lap pair against Spa 0.92), but scaling the grid
weight by it was within noise and its exponent was unstable (0–2).

**Stage 2a: lap-by-lap simulation (implemented; pre-weekend race champion
`race:v2a`).** Each draw runs
the race lap by lap:

- **Start.** Cars leave the grid spaced 0.25 s per slot, shuffled by a fitted
  start noise; pace then acts through racing, so a fast car starting behind still
  has to pass.
- **Racing laps.** Every lap each car laps at its race pace plus lap noise (0.55%
  of a lap). Front to back, a car within 2.5 s of the car ahead passes with the
  battle model's probability, otherwise it is held at least 0.35 s behind; at
  most one place per car per lap.
- **Battle model.** Learned from real battles: every pair of consecutive cars
  within 2.5 s at the end of a green lap on which neither pitted, and whether the
  car behind was ahead after the next green lap (41,340 pairs and 2,362 passes in
  dry races 2018–2023). It is a logistic regression on the gap, log gap, race-pace
  difference and its product with the gap, plus the circuit's pass propensity
  from other seasons (shrunk towards the typical circuit by 400 pairs; it raises
  the held-out log-likelihood by 113). Fitted per season on earlier races.
- **Race length and lap.** The scheduled length is the most laps completed in an
  earlier race on the same layout (a shortened race only ever has fewer). The lap
  time that converts pace into seconds is, once this weekend's qualifying has run
  dry (v2bw), its median best lap times the race-to-qualifying ratio of earlier
  races (1.075, spread 1.5%, steady from 2018 to 2024); before that, the median
  green racing lap of the latest earlier dry race with at least 10 such laps, same
  layout first. Against the race's actual green lap the weekend lap is off by 1.0%
  (median; 2.0% at the 90th percentile, 3.9% at most), the earlier race by 1.5%
  (10.8%; 63% for 2020 Sakhir, the first race on the outer loop, and up to 29% at
  first visits with no earlier race).
- **Fitted per season and cutoff:** start noise and race-day noise (and, in
  `race:v2as`, a shift of every pass's log-odds) by ranked probability score.
- **Battle model input (v2af).** Each battle's pace difference can be taken from
  the forecast at the cutoff instead of the race's own measured pace; the measured
  pace comes from the same race, so a car held up looks slower than it was.

**Retirements by circuit and team (`retirements.py`, v1ar, v2ar, v2bwr).** Each
driver's chance of retiring is an incident chance times the circuit's incident
factor plus a mechanical chance times the team's reliability factor. Base rates
come from the last three seasons. The circuit factor is the circuit's incidents
against what its seasons' rates expect, over Jolpica's results from 1996 to 2017
(cause read from the status text) and the labelled races from 2018, shrunk towards
1 by a gamma prior whose spread is estimated from the same races: circuits differ
far beyond chance (p = 2e-11) and keep their rank between 1996–2009 and
2010–2022 (correlation 0.78), from Monaco about 1.7 to Bahrain about 0.6 after
shrinkage. The team factor is the team's mechanical failures this season, shrunk
towards a prior that keeps 42% of last season's deviation (measured 2014–2022: a
team's rate repeats between odd and even rounds with correlation 0.25 and between
seasons with 0.42; true spread about 0.04 per start around 0.10).

**Stage 3a: tyre wear and pit stops (`race:v3a`, in evaluation).** Measured on
the dry races of 2018–2024: a green-flag stop costs 22.7 s against the cars
staying out (circuit medians 19–28 s; a circuit's value repeats at the next visit
with correlation 0.54), one under a safety car or virtual one 0.83 of that; 25%
of stops fall under one while 9% of laps are neutralised; 12% of stops are more
than 5 s slower than their race's median; cars make 1.66 stops (circuit means
1.1–3.3, next-visit correlation 0.39); tyres lose about 0.05% of a lap per lap of
age (next-visit correlation 0.26–0.60 by compound). In the simulation every car
of a draw makes the same number of stops (the circuit's count rounded at random
once per race: drawn per car, a one-stop and a two-stop car end some 13 s apart,
far more than real strategies differ, and the last cutoff got worse beyond noise),
at evenly spread laps moved by a fitted spread (the data choose none at the last
cutoff); tyre age slows a car and counts in the pace difference that decides
passes; a stop costs the circuit's pit loss plus a slow-stop residual; when a
safety car or virtual one comes out, cars with a stop due within 6 laps take it
at once; a red flag gives everyone new tyres. Outcome probabilities improve
beyond noise at both cutoffs, but at the last cutoff the finishing-order ranking
is worse than without stops. The slow-stop residuals come from in- and out-lap
times, which also hold traffic and lap noise, and likely overstate how much stops
reorder the field; pit-lane times would measure that directly.

**Other history before 2018, measured and not used:**

- **Battles 2011–2017** (46,144 pairs from Jolpica lap times and pit stops): the
  circuits' pass factors agree with 2018–2024 (correlation 0.82), but adding them
  lowers the held-out likelihood of 2018–2024 passes (down-weighted, at best +2
  against the +156 the circuit factor already gives); the battle model itself
  needs forecast pace, which exists only from 2018.
- **Starts 1996–2018** (8,534): how much lap 1 reshuffles the field by circuit
  does not persist between eras (correlation 0.12); no grid-side advantage
  (0.02); a team's launch repeats within a season only weakly (odd against even
  rounds 0.24; about 0.5 places per start against 3.1 places of noise), so an
  in-season estimate explains under 1% of a start. Kept for the 2026 work, when
  launches differed more.

**Next stages:**

- **Shortened races** beyond a red flag that ends the race (time limits) as
  their own scenario; a shortened race never sets the scheduled distance.
**Stage 2b: neutralisations (implemented; simulation base `race:v2b`).**

- **History.** Track status gives every safety car, virtual safety car and red
  flag from 2018 (a run of neutralised laps is one period, its kind the most
  severe: red flag, then safety car, then virtual safety car). Before 2018 they
  are inferred from Jolpica's lap times: a lap is neutralised when the field's
  median is 10% slower than the race's typical lap (25% on lap 1, which includes
  the standing start). On 2018, the season with both, this gives the exact count
  in 18 of 20 races and the same total.
- **Circuits.** On 2018–2024 alone circuits differ no more than chance (p = 0.71
  for safety cars); over 1996–2024 (543 races) they clearly do (p < 0.0001):
  Montreal and Singapore about 1.8 times the typical rate, Melbourne and
  Interlagos 1.6, Monaco 1.5; Barcelona 0.4, the Hungaroring 0.5. The rate also
  rose over time: 0.54 neutralisations per race in 1996–2005, 0.84 in 2006–2014,
  1.0 in 2015–2017, 1.46 in 2018–2024.
- **Hazard.** Per lap: a base rate from the last three seasons with track status,
  laps 1–2 apart (32% of races have one there), counted over the laps on which one
  can start, times a circuit factor shrunk towards 1 by a gamma prior whose spread
  is estimated from the same races. Kind mix, lengths (safety car median 5 laps,
  virtual 3) and the chance a red flag ends the race (1 of 19 in 2018–2024) come
  from the races with track status.
- **In the simulation.** A safety car or red flag closes the field up to 0.52 s
  per car (the median gap when a safety car ends); a virtual safety car keeps the
  gaps; nobody passes while it lasts.
- **Result.** Level with the same simulation without neutralisations at both
  cutoffs: they matter for strategy (a cheap stop under a safety car), which
  stage 3 adds, more than for the finishing order alone.
- **3: tyres and strategy:** done (see Results 2026-11-19 to 2026-11-25); the
  wet crossover moved to stage 4 with the in-race weather.
- **4: grid penalties and weather scenarios.**
- **5: championship simulation after every race.**
- **2026 rules (2026–2029).** Under the 2026 power units, battery deployment
  produced pass-and-repass battles early in the 2026 season: a car that passes
  spends its energy and is repassed on the next lap, for the first laps of a stint
  and again when cars rejoin close after stops, until the faster car settles
  ahead; launch procedures also differed by team. The battle model above counts
  a pass after one lap, so it would read such swaps as easy, lasting passes. The
  plan: refit the battle model within the season, weighting races of the current
  rules; model a repass chance that depends on the laps since the start or the
  last stop; and give each team a start effect learned within the season. The
  next engine rules are agreed in principle for 2030 (to be confirmed by the
  World Motor Sport Council in December 2026), so these rules run four seasons.

The full race simulator, once complete:

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

**Implemented (stage 5, `predictor.championship`, `predictor.season`).**

- **Standings.** Points per classified position by season (Grand Prix and
  sprint), the fastest-lap point (2019–2024, top ten only), reduced points as an
  input of the session (a replay passes the recorded share, Belgium 2021; a
  simulated race scores full points), FIA constructor deductions from the time
  their decision was published, and countback over Grand Prix results only.
  Standings carry forward from any round. Replayed on every Grand Prix and
  sprint of 2019–2024: official points per driver, standings after every round
  and final standings (31 drivers level on points included) reproduced exactly.
- **What a simulated season shares.** The pace model's state at the cutoff splits
  into carried states (car, driver, session offsets, wet ability: random walks
  with the fitted drift) and weekend effects, which return to their prior at
  every event. Each simulated season draws the carried states once and adds the
  drift between events; weekend effects, qualifying, race-day noise,
  retirements and the fastest lap are drawn per event as the race model draws a
  single race, conditional on the season's state. Redrawing strength
  independently every weekend averages away the uncertainty about which car is
  really faster and concentrates the title odds.
- **Race engine.** The simple race model (race pace, simulated grid, grid
  weight and race-day noise, retirements), fast enough for thousands of seasons
  over twenty-odd remaining events; sprints use a grid of their own and a third
  of the race retirement rate.
- **Roster.** The roster at the cutoff races every remaining event; a driver
  already in the standings but missing from it keeps his points and results;
  each driver scores for the team of his seat in the cutoff roster.
- **Checks.** Drift three events ahead equals the pace model's own state; one
  simulated weekend reproduces the race model's forecast within Monte Carlo
  error.

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

Forecast lifecycle. Backtest forecasts (`REPLAY`, `OFFLINE`) skip operator
confirmation and are never published live.

```mermaid
stateDiagram-v2
    [*] --> Computed: cutoff reached, simulation run
    Computed --> AwaitingConfirmation: upcoming event with grid penalties
    AwaitingConfirmation --> Computed: penalty rejected, rerun
    AwaitingConfirmation --> Frozen: all penalties confirmed
    Computed --> Frozen: no penalties to confirm
    Frozen --> Published: before the session starts
    Published --> Superseded: next cutoff forecast published
    Published --> Corrected: error found, new version issued
    Superseded --> Reviewed: result known, misses attributed
    Published --> Reviewed: result known, misses attributed
    Corrected --> Reviewed
    Reviewed --> [*]
```

## Training and evaluation

- Walk-forward over races. 2018 is warm-up; 2019–2023 development and calibration;
  2024 validation; 2025–2026 held out until model selection. 2026 doubles as a
  regulation-change stress test.
- Whole events stay together; nothing from the target event or later is used.
- Baselines: uniform, championship order, persistence (the latest qualifying
  order), practice pace, qualifying order.
- Cutoffs evaluated for qualifying: `pre_weekend` (before the first session) and
  `last` (latest data before qualifying: after FP3, or after FP1 on sprint
  weekends). For races: `pre_weekend` and `last` (after qualifying, or after the
  sprint where it sets the grid); race baselines are championship order before
  the weekend and qualifying order at the last cutoff.
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
- Comparisons are paired by event, with a bootstrap 90% interval on the mean
  difference. A component is better beyond noise when the whole interval is on the
  improving side; it fails the gate when the whole interval is on the worsening
  side against the model it replaces, at any cutoff.
- Every run records the code revision (and uncommitted changes), the reference
  build, and each model's fitted settings and fit status per season.
- A new version is promoted only if it beats the current one on held-out events.

Qualifying results, 2019–2023 (104 events, walk-forward, ranked probability score;
lower is better):

| Model | Before the weekend | Last cutoff |
|---|---:|---:|
| Uniform | 0.175 | 0.175 |
| Championship order | 0.117 | 0.117 |
| Practice pace | — | 0.114 |
| v1: qualifying evidence only | 0.108 | 0.108 |
| v2: + practice, testing, team × session offsets | 0.112 | 0.102 |
| v3w: + weekend effect, winter reset (current) | 0.104 | 0.099 |
| v3wa: v3w + car × circuit affinity (not adopted) | 0.104 | 0.099 |
| v3wr: v3w + corrected race pace | 0.1035 | 0.0984 |
| v3wrq: + Q1/Q2/Q3 simulation | 0.1027 | 0.0974 |
| v3wrqw: + wet sessions as their own evidence | 0.1020 | 0.0960 |
| v3wrqwt: + robust Student-t noise (champion at both cutoffs) | 0.1012 | 0.0951 |

v3w's pole log loss is 1.61 at the last cutoff against 2.08 for championship
order. Its 80% intervals contain the result 89–91% of the time against an
expected 87%, so the forecasts are slightly underconfident.

```mermaid
flowchart TB
    cand["Candidate component or feature"] --> wf["Walk-forward evaluation<br/>hundreds of events, whole events kept together"]
    wf --> overall{"Overall score better<br/>beyond run-to-run noise?"}
    overall -->|"yes"| keep["Keep"]
    overall -->|"no"| cls{"Clearly better on a class of events<br/>(wet, safety car, high degradation, street)<br/>without hurting overall?"}
    cls -->|"yes"| keep
    cls -->|"no"| drop["Drop and log the result"]
    keep --> version["Candidate model version"]
    version --> promo{"Beats the current version<br/>on held-out events?"}
    promo -->|"yes"| promote["Promote"]
    promo -->|"no"| stay["Current version stays"]
```

### After the holdout check (planned)

The 2025–2026 seasons are opened once, to score the frozen champions. After that:

1. **They become training data.** The model is walk-forward, so every forecast
   trains on all sessions before it; from the 2027 season on that includes
   2025–2026, wet races among them.
2. **One re-test batch, fixed in advance.** Features kept in the code but not
   adopted for lack of evidence are re-tested once, walk-forward over 2019–2026,
   under a rule written down before the run (the per-horizon promotion rule; wet
   features judged on wet events and on interval calibration):
   - wet and mixed races read with a drying curve and the tyre-class crossover
     (`wet_races`, the "x" variants);
   - driver wet ability (`wet_ability`, the "y" variants);
   - wet qualifying track evolution (`wet_evolution`, "e") and wet segments
     typed by their tyres (`wet_segments`, "g");
   - per-driver practice push pace (`practice_push`, v3wrqwp), which passed
     development and failed 2024 validation;
   - driver × circuit affinity (`driver_affinity`, the "i" variants): slightly
     better in every comparison but within noise, with few visits per circuit to
     learn from;
   - any wet feature that was within noise before the holdout.

   These features are short of data (about 15 wet qualifying sessions in
   2019–2024), which is why they wait for the extra seasons.
3. **How 2026 is read in the check.** 2026 is the first season of new rules,
   and its first races raced differently (pass-and-repass battles, see the race
   simulator). The check reports 2026 in two parts, roughly its first six races
   and the rest, and judges the race model mainly on how it adapts within the
   season. Early-2026 races that look like outliers do not on their own reverse
   decisions made on 2019–2024; the answer to them is learning within the season,
   which serves the rules through 2029.
4. **Shadow mode from the 2027 season.** No unseen history remains once the
   holdout is used, so re-testing on the same seasons again would select winners
   by chance. Candidates instead forecast every live weekend alongside the
   champions without being shown, and are promoted once they beat a champion on
   races that did not exist when they were chosen.

### Results since the native filter (2026-11-13 to 2026-11-17)

Promotion now follows **per-horizon champions**: each forecast moment (before the
weekend, last cutoff) has its own model, replaced only by a candidate that is
better beyond noise at that cutoff (block bootstrap over 4 consecutive events)
and not worse beyond noise in rounds 15+. Promotions are confirmed on 2024, a
validation season no model was selected on; 2025–2026 stay sealed for one final
check of the frozen policy. Forecasts of different models share their random
draws, and the forecast roster is the entry list known at the cutoff (the
target's entrants stand in for it, with substitutions counted once visible).

| Change | Result (development 2019–2023, RPS difference) | Decision |
|---|---|---|
| Qualifying history 1996–2017 read as qualifying (v3wh) | worse beyond noise at both cutoffs: the old era pulled the shared settings | not adopted |
| History with its own noise (v3whn) | worse beyond noise | not adopted |
| History in the state only (v3whs) | level overall; better in rounds 1–3, worse in rounds 15+ | not adopted |
| History with a learned trust (v3whw) | level; worse in rounds 15+ before the weekend | not adopted |
| Car × circuit affinity with history (v3wahs) | within noise, slightly worse | not adopted |
| Driver season form (v3wf) | level; fitted spread 0.004–0.034 against a driver prior near 0.3 | not adopted |
| Driver fast-corner skill from telemetry (v3wk) | level (the specialty is real and persists across seasons, but tracks differ too little for it to move qualifying order) | not adopted; kept for the race model |
| Corrected race pace as evidence (v3wr) | −0.00031 before the weekend, −0.00052 at the last cutoff, both better beyond noise; better in rounds 15+; 2024: −0.00126 and −0.00075, both better beyond noise | **adopted at both cutoffs** |
| Practice push pace, compound-corrected (v3wrp) | within noise | not adopted |
| Practice long runs as evidence (v3wrl) | worse beyond noise at the last cutoff (team fuel loads vary by session) | not adopted; kept for the race model |
| Push pace with its own noise (v3wrpn) | last cutoff −0.00204 better beyond noise; 2024: +0.00055 | not promoted; last-cutoff challenger |
| Race pace estimation covariance in the noise (v3wru) | level (−0.00004, −0.00001): with 20+ clean laps the estimates' own error is small next to race noise | not adopted |
| Qualifying simulated through Q1, Q2 and Q3 (v3wrq) | −0.00084 and −0.00087, better beyond noise (also under the entry-list roster); 2024: −0.00204 and −0.00190 | **adopted at both cutoffs** |
| Failed laps, driver-specific (v3wrqf) | worse beyond noise before the weekend (+0.00043): failures are mostly not driver traits | not adopted |
| Wet and mixed sessions as their own types (v3wrqw) | −0.00053 and −0.00136, better beyond noise; 2024: −0.00144 and −0.00435 | **adopted at both cutoffs**; last-cutoff champion |
| Robust Student-t noise (v3wrqwt) | −0.00074 before the weekend, better beyond noise; last cutoff within noise; 2024 before the weekend −0.00074 | **pre-weekend champion** (confirmed after aligning its forecast draws with the fitted distribution: −0.00074, 2024 −0.00073) |

Lessons:

- New information helps; re-reading the same qualifying times does not.
  Corrected race pace was the first adoption: race noise fits at 0.37–0.40,
  below practice noise (0.49–0.56).
- History pulls shared settings towards the old era; kept out of the fit it
  helps early rounds and hurts late ones. Settings are fitted by one-step
  likelihood over all sessions, which favours early-season gains; tuning against
  ranking is planned.
- A promising development result can fail on a new season (v3wrpn): promotions
  are confirmed on 2024 before they count.
- Errors concentrate in wet and mixed qualifying (12 of 104 sessions): ranking
  error about 50% higher and intervals too narrow (78% coverage against 87%
  predicted), while dry sessions are slightly underconfident (91% against 87%).
  A global spread change would therefore be wrong. Treating wet sessions as weaker
  evidence (v3wrqw) lowered the fitted dry qualifying noise from about 0.45 to
  0.39 and improved every season. Wet qualifying is now forecast in its own mode
  (see the results below).
- Simulating qualifying segment by segment helped more than any evidence change:
  lap noise acts in each segment, which a single sorted draw understated.
- Heavy-tailed noise fits lap data far better (log-likelihood +780 on data before
  2021 at 2–5 degrees of freedom), but forecasts must draw from the same
  distribution the filter fits (one shared scale per segment), or they become
  overconfident.
- The roster comes from the FIA entry list published before the cutoff (125 of 128
  pre-weekend forecasts 2019–2024): withdrawn drivers are forecast but not scored,
  late substitutes are not forecast, and a seat is taken only from sessions
  available at the cutoff.

**Track profiles.** Corners are found on each session's reference lap (the
median line of its clean dry laps) from the racing line's curvature and banded
softly by minimum speed (slow under about 120 km/h, medium, fast over 200 km/h);
a track profile is the share of lap time on straights and in each band, the
full-throttle share and hairpins, chicanes and heavy braking zones per km. A
share varies about six times more between layouts than between years of one
layout. `circuit_layouts.csv` lists every layout from 2018 (official turn counts
and sources), including layout changes.

**Corrected race pace.** Each dry race's clean laps are fitted together: driver
pace, compound offsets and wear per compound, with fuel burn a fixed constant
(a race alone cannot separate fuel from wear). Drying races, drivers with fewer
than 20 clean laps and unseparable fits are left out. Corrected teammate gaps
correlate 0.48 with qualifying teammate gaps, against 0.39 for raw laps.

### Results 2026-11-17 to 2026-11-18

Qualifying (development 2019–2023 unless stated; RPS difference, negative is
better; "noise" means the 90% block interval spans zero):

| Change | Result | Decision |
|---|---|---|
| Wet forecast mode: wet targets forecast with wet-session noise (v3wrqwc) | overall within noise (dry forecasts identical); on the 15 wet events 80% intervals cover 92% (before the weekend) and 88% (last cutoff) against 79% and 75% before, for about 86% predicted | **serves the wet toggle**; robust version (v3wrqwtc) at both cutoffs |
| Driver wet ability (v3wrqwy) | within noise; slightly worse on 2024 dry events | not adopted; re-test |
| Wet races read with a drying curve and the crossover between tyre classes (v3wrqwx) | all 11 wet races 2018–2023 now fitted; −0.00001 and −0.00006, within noise | not adopted; re-test |
| Wet track evolution within qualifying (v3wrqwec) | +0.00018, worse beyond noise; wet events +0.002 | not adopted; re-test |
| Robust noise at the last cutoff (v3wrqwt vs v3wrqw) | −0.00125, better beyond noise, better in rounds 15+; 2024 −0.00226 (noise), calibration gap halved (0.032 against 0.060) | **promoted at the last cutoff**: v3wrqwt is champion at both |
| Wet segments typed by their tyres (v3wrqwtgc) | +0.00007 and +0.00010 (noise); worse in rounds 15+ before the weekend; 2024 +0.00034 | not adopted; re-test |
| Missed FP1, first races in a new car (measured, no model) | missed FP1 (98 cases): +0.016% ± 0.038 of a lap against the teammate; first race in a new car +0.10% ± 0.13, later races none | no model change |
| Driver × circuit affinity (v3wrqwti) | −0.00014 and −0.00013 (noise); 2024 −0.00044 and −0.00069 (noise) | not adopted; re-test |

Lessons:

- Wet data are few (15 wet qualifying sessions in 2019–2024), so wet features are
  judged on wet events and calibration as well as overall; most wait for the
  re-test with 2025–2026.
- The timing feed's rainfall flag is unreliable (a dry Monaco race was flagged
  wet on 83% of its laps); tyres are the better evidence. 9 of the 15 wet-typed
  qualifying sessions have a segment run on slicks.
- Missing FP1 or changing teams does not cost measurable qualifying pace, so no
  penalty was added.

Races (2019–2024, 128 races):

| Model | Before the weekend | Last cutoff | Decision |
|---|---:|---:|---|
| Championship order | 0.143 | — | baseline |
| Qualifying order | — | 0.135 | baseline |
| race:v1a: race pace, grid, retirements | 0.121 | 0.112 | **last-cutoff race champion**; better than the baselines in every stage of the season and every season, 2024 included |
| race:v1b: grid weight by circuit overtaking | +0.00002 | −0.00004 | not adopted (noise) |
| race:v2a: lap-by-lap simulation | −0.00030, better beyond noise; 2024 −0.00029 | +0.00073, worse beyond noise | **pre-weekend race champion** |
| race:v2as: v2a + a fitted shift of every pass's log-odds | +0.00006 (noise) | −0.00020 (noise); 2024 −0.00105 | not adopted |
| race:v2af: v2a with the battle model fitted on forecast pace differences | −0.00012 (noise); +0.00018 against v2a (noise) | −0.00026 (noise); −0.00099 against v2a, better beyond noise; winner log loss 1.36 → 1.32 | not a champion; **base of the next stage** |
| race:v2afs: v2af + pass shift | −0.00001 (noise) | −0.00040 (noise) | not adopted |
| race:v2b: v2af + safety cars, virtual safety cars and red flags | −0.00007 (noise); +0.00016 against v2af (noise) | −0.00029 (noise); −0.00099 against v2a, better beyond noise; −0.00003 against v2af (noise) | not a champion (stages 3–4 need neutralisations) |
| race:v2bw: v2b + the typical lap from this weekend's qualifying | identical to v2b (no session yet) | −0.00032 (noise); −0.00004 against v2b (noise) | not a champion |
| race:v3a: v2bwr + tyre wear and pit stops (stage 3a) | −0.00005 against v2bwr, level with v2a (noise); outcome log loss −0.0052 against v2bwr, better beyond noise | +0.00037 against v2bwr, worse beyond noise; level with v1a; outcome log loss −0.0094 and winner log loss −0.029 against v1a, both better beyond noise | not a champion; open: ranking against outcome probabilities |
| race:v1ar, v2ar, v2bwr: retirements by circuit and team, on v1a, v2a and v2bw | against their own base +0.00001, +0.00010, −0.00016 (noise) | −0.00028, −0.00015, −0.00012 (noise); 2024 −0.0017, −0.0009, −0.0009; v1ar better beyond noise in rounds 15+; retirement Brier −0.0003 for each (noise) | champions unchanged; **v2bwr is the simulation base** |

Differences are against race:v1a. The first v2a results were discarded: a
shortened race could set the next year's race length and lap time (fixed before
the run above).

v1a's fitted grid weight is 0.05–0.075% of a lap per grid slot and its race-day
noise 0.45–0.6%. At the last cutoff, v2as fits a pass shift of −1.5 to −3 in
every season: the battle model passes too easily when the grid is known. It
learns from each race's measured pace, which is taken from the same race (a car
stuck behind looks slower because it was stuck), while the simulation feeds it
forecast pace. Fitting the battle model on the pace difference forecast at the
cutoff (v2af: for each battle, 10 draws of that race's forecast, the input the
simulation gives it) halves the fitted shift (−0.5 to −1.5) and improves the
simulation at the last cutoff beyond noise; some over-passing remains. Stage 2b
builds on v2af.

### Results 2026-11-19 to 2026-11-25

Races 2019–2024 (128 races) unless stated. RPS differences per race, negative is
better; "noise" means the 90% block interval spans zero. 2024 is development
evidence; 2025–2026 stay sealed.

#### Stage 3: tyres and strategy

Against the simulation base of the time (race:v2bwr) or the serving champion:

| Change | Before the weekend | Last cutoff | Decision |
|---|---|---|---|
| race:v3a, stops shared by every car of a draw | −0.00019, better beyond noise | +0.00043, worse beyond noise; outcome log loss better | in code; base stays v2bwr |
| race:v3b pit-lane slow stops, race:v3c per-car stop counts (from 2014–2024 deviations by pace band) | −0.00000 / −0.00014 (noise) | +0.00024 / +0.00048, worse beyond noise; winner log loss −0.043 against race:v1a | v3c carried into v3e and v3f |
| race:v3d fitted share of strategy randomness | noise | +0.00037, worse; the fitted share flips between seasons | not adopted |
| race:v3e this weekend's tyre wear and tyre-era history | −0.00016 (noise) | +0.00044, worse | not adopted |
| race:v3f team tyre wear from race stints, drawn per simulated race | −0.00039, better beyond noise (first strategy model to beat the base) | +0.00031 (noise) | **development simulator** |
| race:mix3f, v3f and race:v2a 50/50 | −0.00031 against the champion, better in every season | −0.00038 against race:v1a, 2024 level | **promoted before the weekend** |
| race:v3g learned per-lap pit decisions | +0.0013, worse | +0.0025, worse | parked (diagnosis below) |
| race:v3ff held cars keep a learned gap (walk-forward, held-up cars only), attack release | +0.00037 against v3f (noise) | race:mix3ff1a −0.00060 [−0.00093, −0.00028] against race:v1a, 15+ and 2024 better | **promoted after qualifying** |
| Following penalty (race:v3fp), race pace corrected for it (race:v1afp), both (race:v3fpc), per-car correction | all noise or worse; 2024 worse | all noise; 2024 worse | not adopted; kept for the analysis service |
| Clean-air race pace (gap ahead ≥ 1.0 or 1.5 s) | noise | 1.5 s: −0.00027, better beyond noise; 2024 worse at every threshold | not adopted |
| Wear from lap effects instead of a fixed fuel rate (race:mix3fw2a) | −0.000024, better (one batch); a tie over three batches | level | **served before the weekend** (tie-break: no assumed fuel rate) |
| Learned pit windows (race:mix3ffn1a) | +0.000052, worse | −0.000086 (one batch); a tie over three batches | **served after qualifying** (tie-break: measured stop timing) |
| Fresh-tyre peak after a stop (first laps 0.34% faster in the 13-inch era, 0.11% in the 18-inch era) | level | level | not adopted |
| Start-tyre roles shift the windows | not tested | +0.000121, worse | not adopted |
| Learned one-stop-more or one-stop-fewer gambles (three variants) | not tested | +0.00022 to +0.00033, worse | not adopted; the pace-band table stays |
| Circuit-specific pass slope (held-out passes) | — | −0.000088 log loss per pass (about 0.05%) | real but negligible; not gated |
| Learned mixture weights (walk-forward) | level | level; the optimum is flat | 50/50 stays |

The learned pit-decision model (race:v3g) was diagnosed before anything else
changed. On real race states its hazard predicts too few stops (0.90 of the real ones on
ordinary laps, 0.62 after red flags, 0.63–0.65 on tyres up to 10 laps old) and cannot tell one-stop
from two-stop races (correlation 0.09 per race). In paired simulations
switching traffic off made it worse, so traffic is not the damage. Stops off the
planned timing do not pay back: once comparable cars have stopped, a car's place
has moved by 3.4 on average, against 2.2 in reality. The tyre-payback mechanics are exact (a
controlled two-car test gives the implied 0.90 s per lap); the simulator's wear
rate is about 25% low in the 13-inch era and right in the 18-inch era.

Before the weekend the following model itself hurts (race:mix3ff2a: winner log loss
worse beyond noise, 2023–2024 worse): a sticky order on an uncertain pace and grid.
From then on strategy refinements target the forecast after qualifying, and the
pre-weekend simulator stays simpler.

Measured and closed without a model change: used tyre sets (26.7% of stints after a
stop) are not slower once tyre age is counted (−0.12%, se 0.016); compound identity
(C1–C5) predicts per-compound wear no better than the weekend's hard, medium and soft
labels. The nominations table (`tyre_compounds.csv`, 149 Grands Prix) stays for
strategy and the wet crossover.

#### Calibration and seed noise

- Randomised PIT variance 0.0747 after qualifying and 0.0769 before the weekend
  (0.0833 is ideal): slightly underconfident, no bias. Wet races are overconfident
  (the simulation is dry). Front-runners (grid 1–3) finish better than forecast:
  P(top 3) 0.59 forecast against 0.69. The simulator and race:v1a share this bias,
  so it comes from the shared pace model or grid weighting, not from pit cycles.
- Row sharpening chosen walk-forward gains −0.0003 (noise) but breaks
  position consistency; coherent recalibration waits for ranking tuning.
- **Seed noise.** The pit-window promotion measured −0.000086, −0.000152 and
  +0.000112 in three seed batches: Monte Carlo noise between batches (about
  ±0.00013) exceeds small effects, and the bootstrap intervals do not include it.
  The promotion protocol now averages three seed batches with 4,000 fitting draws
  and 20,000 forecast draws, and a tie may be broken in favour of the more
  realistic mechanism. Both small promotions above became ties under it.

#### Race pace reading (gates A, B and C, three seed batches)

| Option | Qualifying | Races | Decision |
|---|---|---|---|
| A: lap effects per lap number instead of a fixed fuel rate | last +0.000082, pre −0.000012 (noise) | ties; 2024 +0.000104 and +0.000233 (over the +0.0001 limit) | not adopted |
| B: keep the best-connected group of cars when a race is not fully identified (+4 races) | before the weekend −0.000069, better beyond noise; after a session a tie | before the weekend −0.000099, better beyond noise; after qualifying a tie | **adopted** |
| C: races of 5–19 laps, with a stint scale on the estimation variance (the reported variance understated the real error; scale 5.9–7.6 by maximum likelihood, split-driver fit) | before the weekend a tie (adopted); after a session worse | after qualifying −0.000340, better beyond noise, 2024 −0.00067; before the weekend outcome and winner log loss worse | **adopted after qualifying** and for qualifying before the weekend |

#### Stage 4a: starting grids and grid penalties

The qualifying order differs from the starting grid in 69% of races (141 drops of
three places or more, 42 pit-lane starts). Starting grids are parsed from the FIA
documents for 2020–2024 (2019 is published as images; it falls back to the
qualifying order). Penalty decisions (214 rows 2019–2024) are applied with the
grid rules of their season (empty-grid procedure, back-of-grid groups); qualifying
plus decisions rebuilds 96 of 105 provisional grids exactly (qualifying alone: 54).

| Input | Result | Decision |
|---|---|---|
| Published grid, a new cutoff once it is out (race:mix3ffnps51aps5g) | −0.000179 [−0.000561, +0.000203], a tie | **adopted** |
| Penalties known before the weekend (race:mix3fwp2apk) | −0.000011, a tie; 12 races change | **adopted** |
| Penalties known after qualifying (race:mix3ffnps51aps5k) | +0.000236 (noise), 2024 +0.00095 | not adopted |

#### Settings-fit stability

The diagnosis of the after-qualifying penalty gate found that refitting the
simulator's settings on slightly different grids flipped them between distant
values, differently per seed batch. Three causes:

1. **Sequential random numbers.** A setting that moved a stop shifted which numbers
   later became lap noise, passes or safety cars. Fixed: numbers keyed by draw, lap
   and car (in a controlled test 299 of 300 finishing orders changed with the old
   stream, 0 with keyed numbers); settings are fitted on seeds of their own.
2. **Winner-take-all choice over a flat surface.** In every season 15–70 of the 75
   simulator candidates lie within paired noise of the best. Fixed: the choice is
   the lowest mean after averaging each candidate with its neighbours on the
   settings grid, chosen by a time-separated replay of the 2020–2022 fits
   (best score for the simulator; no rule better beyond noise).
3. **Compensating settings** (race-day noise and stop-timing noise both spread the
   order): not separable from aggregate RPS; accepted.

Re-baseline (three batches): before the weekend the stable fit is better beyond
noise (−0.000073 [−0.000120, −0.000030], 15+ better, 2024 −0.000091) and serves.
After qualifying (+0.000202, 15+ worse, 2024 +0.00042) and with the published grid
(+0.000259) it fails; the old fit serves there. A "keep last season's settings
unless clearly beaten" rule froze the simple model's 2019 settings until 2023 and
was removed.

#### Penalties after qualifying: split grid term

Penalised cars finish near where they qualified: 91 cars moved back qualified 10.8,
started 16.3 and finished 10.66 (forecast 10.40 ignoring the penalty, 11.63 with
it; win chance 5.5% actual, 5.6% and 1.9%). One grid weight did two jobs: the
qualifying slot as a pace signal and the starting slot as track position. The
simple model now separates them, `kappa × ((qualified − 1) + rho × (start −
qualified))`, rho fitted from {1, 0.75, 0.5, 0.25, 0}. This removes the refit damage
entirely (races with an unchanged grid score exactly as served), but the changed
races stay worse: the mixture with penalties and the split scores +0.000181
[−0.000143, +0.000525], 2024 +0.00072. Not adopted. Whether the simulator recovers
out-of-position cars correctly is not shown either way (the check's pace proxy was
too weak). Power-unit penalty forecasts are dropped:
penalties should act through learned recovery and the car's pace, not through a
forecast of who takes one.

#### Car profile from telemetry

From every dry practice and qualifying session 2018–2024 (491 sessions, 146
weekends), each driver's best clean lap and best free-air lap (no car within 2 s
ahead at the start line or either sector line), relative to the session median:
times on straights and in slow, medium and fast corners, minimum speed per corner
band, top speed, speed-trap speed, traction (100–180 km/h, full throttle at both
ends) and high-speed acceleration (250–300 km/h, DRS closed at both ends).

- **Repeatability.** Every measure repeats within a weekend beyond what lap time
  already says (rank partial correlation, top speed 0.65, high-speed acceleration
  0.60, traction 0.23), between teammates (0.56, 0.49), and to the next
  consecutive weekend (0.44, 0.29). This is a shared team signal; it does not
  separate the car from setup, driver or conditions.
- **Battles.** Adding free-air top speed and high-speed acceleration to the battle
  model (walk-forward 2020–2024, 87 test races): −0.00018 [−0.00056, +0.00020] log
  loss per pair, better in 5 of 5 seasons but within noise. Not adopted; the
  acceleration coefficient is positive in every season. Every measure at once, and
  gradient-boosted or neural versions, are worse or unvalidated.

#### Weather, first check

Wet and mixed races (16 of 128) carry 15% of the error. Blending a known-wet
forecast towards uniform does not help (+0.00015 after qualifying, +0.00010 before
the weekend). This tests one adjustment only; rain can also act through tyre
choice, pit losses, interruptions and recovery chances.

#### Fundamentals audit and the pace error split

| Forecast (last cutoff) | RPS |
|---|---|
| uniform | 0.1738 |
| practice order (fitted strength) | 0.1445 |
| championship order | 0.1429 |
| qualifying order | 0.1355 |
| served after-qualifying mixture | 0.1111 |
| served simple model race:v1aps5o | 0.1116 |
| the simple model with the race's own measured pace (hindsight) | 0.0993 |

The served forecast beats the qualifying order by 0.024 (91% of races, every
season). Knowing the race's measured pace improves the simple model by 0.012
(86% of races, every season): about a hundred times the recent feature gains.
The hindsight forecast is optimistic (measured pace also holds race-day damage and
traffic) and its remaining 0.099 is not shown to be irreducible: it also holds the
simple model's limits and missing measurements.

The split replaces only part of the pace with the measured one: each team's level
(teammates' forecast difference kept), or the difference between teammates (the
team level as forecast). Three seed batches, settings refitted per arm:

| Cutoff | Served simple model | Full | Team level | Teammate difference |
|---|---|---|---|---|
| before the weekend | 0.1213 | −0.0214 | −0.0120 | −0.0059 |
| after first practice | 0.1202 | −0.0208 | −0.0120 | −0.0062 |
| last | 0.1116 | −0.0123 | −0.0064 | −0.0041 |

All beyond noise; the team share is the larger in every season before the weekend
and in five of six after qualifying (2019 the exception). Review corrections:

- When only one teammate has a measured pace (163 team-races in 85 races) the team
  arm moved the whole team by that driver and the driver arm changed nothing: the
  comparison was asymmetric. Corrected run: both arms, and the full one, change
  only teams with two measured drivers.
- Each arm refitted its grid weight and race-day noise (2024: kappa 0.075 served,
  0.02 full, 0.05 team and driver), so the table answers "how well could this model
  do adapted to hindsight pace", not how its error divides. Corrected run: the
  served model's settings held fixed, next to refitted ones.
- A team's mean measured pace also holds shared tyre, strategy, traffic and weather
  effects, and the teammate difference is not pure driver skill. The arms are not
  additive shares (RPS is nonlinear; the arms refit separately). The split ran on
  the simple model at every cutoff, not on the served pre-weekend simulator
  mixture.
- After first practice has 127 races (2020-11 is missing); on common races it beats
  before the weekend by 0.00076. The pace extraction measures 2,261 of 2,560
  driver entries; five races have too few for substitution.

Corrected run (batch 3; teams with two measured drivers only in every arm; a team
with one leaves the forecast unchanged; the served model reproduces to 1e-16):

| Cutoff | Arm | Served settings fixed | Settings refitted |
|---|---|---|---|
| before the weekend | full | −0.0157 [−0.0178, −0.0136] | −0.0200 |
| | team level | −0.0097 [−0.0114, −0.0079] | −0.0113 |
| | teammate difference | −0.0049 [−0.0056, −0.0042] | −0.0060 |
| last | full | −0.0086 [−0.0101, −0.0071] | −0.0117 |
| | team level | −0.0048 [−0.0061, −0.0036] | −0.0058 |
| | teammate difference | −0.0035 [−0.0041, −0.0028] | −0.0042 |

Before the weekend the team level stays about twice the teammate difference with
the settings fixed, in every season (2021–2024 −0.011 to −0.012 against −0.005 to
−0.007). After qualifying it is 1.4 times, in five of six seasons (2019: −0.0007
against −0.0030; 2024 close, −0.0040 against −0.0035). Races with a one-measured
team (85) split like the others. Refitting the settings added 15–35% to every arm.

#### Stage 5: championship simulation

Walk-forward over 122 cutoffs (before rounds 2 to the last of 2019–2024), 2,000
simulated seasons each, the realised calendar assumed known; seasons weighted
equally:

| | Shared season state | Redrawn every weekend | Points per Grand Prix so far |
|---|---|---|---|
| Drivers, final-rank RPS | 0.0408 | 0.0426 | 0.0728 |
| Drivers, final points error | 18.2 | 18.2 | 21.6 |
| Constructors, final-rank RPS | 0.0421 | 0.0449 | 0.0695 |
| Constructors, final points error | 30.6 | 30.7 | 36.6 |

The shared state beats the projection in every season on all four measures and
the independent redraw on rank RPS in five of six seasons per championship
(expected points are the same by construction). With one to seven races left
the projection's points error is slightly lower (drivers 6.8 against 7.4): the
weekend forecast under-rates the top three drivers by about 1.1 points a race in
five of six seasons, the front-runner bias already seen in stage 3's
calibration. It belongs to the race model's ranking tuning, not to the season
simulator. Title probabilities stay descriptive (six champions).

#### Served champions (2026-11-25)

| Forecast moment | Qualifying | Race |
|---|---|---|
| Before the weekend | pace:v3wrqwt-rps (wet: -rpc) | race:mix3fwp2apk: race:v3fwp and race:v2ap 50/50, known penalties, stable fit |
| After a session (to qualifying) | pace:v3wrqwt-rp (wet: -rpc) | race:mix3ffnps51aps5o: race:v3ffnps5 and race:v1aps5 50/50, old fit |
| Starting grid published | — | race:mix3ffnps51aps5go: the same with the published grid |

#### Parked features

Kept in code, not served, each with what would bring it back:

| Feature | Result | Re-test when |
|---|---|---|
| Car profile in the battle model (top speed, high-speed acceleration) | −0.00018 per pair, noise, 5 of 5 seasons | after the car profile enters the pace model |
| Learned per-lap pit decisions (race:v3g) | worse at both cutoffs | the room test below; tyre sets and compounds |
| Penalties after qualifying with the split grid term | +0.000181, 2024 over the limit | the room test; after the pace model improves |
| Following penalty, corrected race pace, clean-air pace | noise; 2024 worse | the room test |
| Fresh-tyre peak, start-tyre roles, learned gambles | level or worse | the room test |
| Wet pieces: drying curve and tyre-class crossover, driver wet ability, wet track evolution, wet segments by tyres; driver × circuit affinity | noise | after the 2025–2026 check (more wet data) |
| Lap effects per lap number (gate A) | ties, 2024 over the limit | with the next pace-model change |
| Car profile in the pace model (car x track suitability; this weekend's profile change) | simple model with fixed settings: suitability -0.00018 before the weekend, noise later; weekend change +0.00030 after first practice, worse | like-for-like sessions (FP1 against earlier FP1s, Q against earlier Qs); 2025–2026 data |
| Team residual carried to the next race | squared team error +0.0073, worse in every season; previous-race correlation +0.03 | closed |
| Slick-intermediate crossover trend | with inputs timed correctly, no better than the last gap; 7 races in 2018–2024 with any mixed interval, practice none | after the 2025–2026 check |
| In-race rain | deferred, untested | after the crossover |

#### Plan from 2026-11-25 (as written that day; steps 2–6 were run the same day: the car profile and the team residual did not convert, the crossover is parked, and stage 5 followed)

The audit moved the priority from race mechanisms to the pace forecast: the
reachable error is in race pace, mostly at team level, and race-mechanism features
were judged on top of a pace forecast that misses by an order of magnitude more
than they can gain. The data are the same; the change is how they are used.

1. **Corrected split.** Done (above): the team share holds with the fixes, most
   clearly before the weekend.
2. **Bounded car-profile pace check.** Profiles strictly before each cutoff, a few
   regularised car × track-mix terms, judged on held-out race pace error first:
   does a better car map convert into better forecasts, not only a smaller gap to
   hindsight?
3. **Car profile in the pace model, full gate.** Settings fitted, race and
   qualifying champions, three seed batches, the promotion protocol.
4. **Teammate difference from the same data.** With the car mapped, the gap between
   teammates becomes a residual that the corner-band, straight-line and session
   measures can explain: where one driver loses to the other, not only by how
   much. Gated like any feature, and only if step 1 supports the priority.
5. **Room test.** The parked race features re-tested on the improved pace (and
   under hindsight pace), to see which were masked rather than wrong.
6. **Stage 4 continued.** A small weather uncertainty term (wet and mixed
   scenarios), then the in-race rain model with the wet crossover; penalties back
   through learned recovery on the improved pace.

## Known constraints

- Historical archive data has no verified publication timestamps; backtests on it
  run in `archive_proxy` mode and are labelled as retrospective. Data captured live
  from go-live onward supports strict backtests.
- Fuel load, engine modes and run plans are not observed; the model carries them
  as uncertainty, not invented values.
- From 2023 the timing sources record almost every retirement only as "Retired".
  Causes for 2023–2024 come from race reports, reviewed by hand; 2025–2026 will be
  labelled the same way. Unknown causes stay unknown.
- Circuits are identified by circuit ID, not by layout version, so a reprofiled
  track still counts as the same circuit for affinity.
- A thorough settings fit takes minutes per season for v3w and up to about
  40 minutes for v3wa (before 2023, native filter); fits run once per season.
- 2026 grid has 22 cars; entrant and points rules are versioned by season.
- Before the weekend the race simulation takes its typical lap from an earlier
  race at the circuit, so a first race on a new layout borrows another layout's
  lap until its qualifying has run.
- Open: whether "open" means published methodology plus reproducible frozen
  forecasts, or also open-sourcing the model code.
