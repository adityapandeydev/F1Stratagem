# Team race pace

Team race pace answers one question for every Grand Prix: on equal tyres and
fuel, in clear air, how far was each car from the quickest car in that race?

It is a measurement, not a model of the season. Every number comes from the
recorded laps of the race it describes. Nothing is carried over from other
races, nothing is weighted by hand, and the same rules apply to every team. Run
the same laps through the method and you get the same numbers.

The figure describes a car together with its quicker driver: the best race pace
the team demonstrated that day. A separate view, [car-only race
pace](car-only-race-pace.md), takes the drivers out.

## Why raw lap times are not enough

A raw lap time mixes the car with conditions that have nothing to do with it:

- **Tyre age.** A lap on a 25-lap-old set is slower than one on a fresh set.
- **Fuel.** Early in the race a car carries far more fuel than at the end.
- **Track.** The circuit gets quicker as rubber goes down, and changes with the
  weather.
- **Traffic.** A car close behind another loses time in disturbed air.
- **Compound.** Soft, medium, hard, intermediate and wet tyres have different
  pace.

Two cars on different strategies drive their laps under different conditions,
so averaging their raw laps compares the strategies as much as the cars. Team
race pace separates these effects using the race's own laps.

## Which laps count

A lap is used when all of these hold:

| Rule | Reason |
|---|---|
| Green flag for the whole lap | Safety cars, virtual safety cars and red flags change pace for everyone |
| Not the first lap | Standing start and first-lap congestion |
| Not a pit in-lap or out-lap | Pit lane time is not race pace |
| A known tyre compound (dry, intermediate or wet) with a recorded tyre age | Both enter the fit |
| Not deleted, and marked accurate by the timing feed | Unreliable timing is left out |
| Clear air at all three timing lines | Following another car costs time |

**Clear air** means more than 2 seconds behind the car physically ahead at
each of the three timing lines of the lap: the two sector lines and the finish
line. The car ahead is whichever car crossed that line last, whatever its race
position or lap count, so lapping a slower car or being lapped counts as
traffic. A car finishing an in-lap is in the pit lane and does not count. A lap
whose traffic cannot be determined, because a crossing time is missing, is left
out.

A driver needs at least five such laps to be measured.

## The fit

All kept laps of the race are fitted together by least squares:

> lap time = driver + compound offset + tyre wear rate (per compound) × tyre age + track term (per lap number)

- **Driver** is the pace of that driver and car.
- **Compound offset** is the difference between each compound and the most-used
  one.
- **Tyre wear** is the time lost per lap of tyre age, one rate per compound,
  measured from cars running different tyre ages on the same lap.
- **Track term** is a separate time for every lap number, shared by all cars. It
  absorbs fuel burn, track evolution and changing weather lap by lap.

Tyre age resets at every stop while the fuel load keeps falling, so the race
separates the two by itself: no fuel consumption figure is assumed.

Laps that sit far from the fit (a mistake, a lock-up, a lap spent behind a car
that was not caught by the clear-air rule) are removed once: anything more than
three robust standard deviations away. The fit is then repeated. The five-lap
minimum is checked again after this step.

## Comparing cars

Each driver is compared at the same reference point: the most-used compound, the
median tyre age, the median lap and clear air. Only the differences between
drivers are reported. They are anchored to the race's median kept lap so that
percentages are always taken against a real lap time.

Two checks protect the comparison:

- **Connected cars only.** Cars can only be compared through laps they share.
  If some cars ran only at the start of a race and others only at the end, they
  cannot be placed on one scale, and only the larger connected group is
  reported.
- **Tyre wear must be measurable.** When every car runs a single stint, tyre
  wear cannot be told apart from the track terms. The race is then reported as
  not tyre-corrected and does not count towards season figures.

## The team figure

A team's figure is its quicker driver: the car's best demonstrated pace in that
race. Both drivers' figures are reported alongside it.

Gaps are given as a percentage of the quickest car's lap and in seconds.

## How firm a gap is

Every race also reports how much each gap depends on which laps happened to be
clean. The race is rebuilt from its own stints, drawn at random with
replacement (200 times for a race report, 100 per race in a season run), and the
same fit is run on each rebuild. The middle 80% of the
results is the range shown for each gap, measured against the same reference
car each time. The share of rebuilds in which each car comes out quickest is
reported as well.

These figures describe the precision of a gap. They never change the gap
itself, and no information from outside the race enters them.

For each driver the evidence behind the number is reported too: laps used,
stints, and laps shared with other measured cars.

## Season figures

A season figure summarises the race figures; it is never fitted across races.
For each team:

| Figure | Meaning |
|---|---|
| Mean | Average gap across the races measured |
| Median | Gap in a typical race, less affected by one unusual weekend |
| As measured | The season's best car is above zero when it was not the quickest in every race |
| Rebased | The season's best car is set to zero |
| All races | Every race that counts, each team over the races it was measured in |
| Common races | Only the races in which every team was measured |

Common races exist because a team that could not be measured in a difficult
race (both cars out early, or too few clean laps) would otherwise have a
flattering season figure. A race counts when it is tyre-corrected and at least
half of its teams are measured. Each figure is given in percent and in seconds,
with the number of races behind it.

## What the figure does not show

- **Chosen pace, not maximum pace.** Drivers manage tyres, protect gaps and save
  energy. Lap times show the pace a team chose to run.
- **The driver is included.** A strong driver lifts the car's figure. The
  car-only view addresses this.
- **Wet and mixed races are harder.** The track terms absorb changing conditions
  lap by lap, but grip changes within a lap and differences in how each compound
  copes with a drying track are only partly captured.
- **Some races cannot be measured.** A race run under neutralisation, or with
  too few clean laps, has no figure rather than a guessed one.

## Reproducing a figure

The method is implemented in [`services/analysis`](../../services/analysis):

```powershell
analysis race --event 2024-20
analysis season --season 2024
```

`--json` writes the full result, including every driver's figures, the ranges
and the evidence counts.
