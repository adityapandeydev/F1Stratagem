# News and upgrade intelligence (planned)

A separate service that collects F1 news and official documents, publishes verified
items to the platform, and supplies structured, timestamped evidence to the
prediction engine. Not implemented.

## Responsibilities

1. Collect: team press releases, Pirelli press releases (compound nominations
   before 2022, which the FIA did not publish), then selected reputable outlets.
   FIA documents (upgrade sheets, PU usage, decisions, Pirelli previews) are
   collected and parsed by the data archive pipeline, see
   [data-archive.md](data-archive.md); this service reads those tables, not
   the PDFs.
2. Deduplicate and attribute: one item per underlying fact, with every source and
   its publication time.
3. Extract: an LLM converts text into structured records. It classifies and
   extracts; it does not produce the numbers the prediction engine uses.
4. Publish: headline, short summary, source link and publication time to the
   platform. Full articles are not republished.
5. Feed the engine: structured records with availability times.

## Upgrade record

| Field | Example |
|---|---|
| team | McLaren |
| component | floor |
| type | performance / circuit-specific / reliability (as declared) |
| stated purpose | "local load" |
| geometry change | "revised floor edge and fences" |
| first event | 2024-06 |
| affected regimes | slow corners, medium corners (inferred, with confidence) |
| sources | FIA car presentation sheet (2024+), publication time |
| confidence | official / reported / rumour |

From 2024 the official record comes straight from the archive's `car_updates`
table. For 2022–2023, and for anything outside the official sheets, records come
from news and carry a lower confidence.

## Measuring upgrade effects

Claimed gains are not trusted. After an upgrade appears, the car profile (time
gained or lost per speed regime, see [prediction-engine.md](prediction-engine.md))
is compared before and after across practice and qualifying, adjusted for the
circuit's regime mix. The estimated effect and its uncertainty enter the pace
model; the claim itself only marks when and where to look.

## Rules

- A historical forecast may only use records published before its cutoff.
- Missing news or LLM credentials must not break forecasts; the engine runs
  without this input.
- Interpretive signals ("reading between the lines") are optional features that
  must show predictive value in walk-forward evaluation before use.
- No fabricated upgrades or effects.
