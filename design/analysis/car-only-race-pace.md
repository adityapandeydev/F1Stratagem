# Car-only race pace

Car-only race pace answers the question behind every pace table: how fast was
the car itself, with the drivers taken out?

The [team race pace](team-race-pace.md) measures a car together with its
quicker driver. That is the right figure for what happened in a race, but a
team with an exceptional driver looks faster than its car is, and a team with
two inexperienced drivers looks slower. Car-only race pace measures each
driver's contribution from evidence and removes it.

It is published as a separate view next to the team race pace, never in place
of it, and it is clearly labelled as using information from more than one race.

## A note on this view

This view is an experiment, built to the best of my ability. For some seasons
its results run against the general consensus, and in places turn it upside
down. Every choice in the method below was made because I believed it would best
represent the pace of the car alone. None was made to show any driver or team as
better or worse than they were or are.

That is why I still consider the team race pace the more reliable figure, and
why this view should be read as an experiment rather than a verdict. If you
disagree with the method, I would welcome your suggestions on how car-only pace
could be measured more accurately.

## The idea

Teammates drive the same car in the same race. The gap between them is
therefore a difference between drivers, not cars. Collected over many races,
these teammate gaps measure every driver's race pace relative to their teammates.

Teammate gaps alone only compare drivers within a team. Drivers change teams,
and every move links two teams: if Driver A beat Driver B at Team X, and Driver
B later matched Driver C at Team Y, then A, B and C are on one scale. Over
several seasons these links connect the whole grid, and every driver gets a
skill on a common scale.

A car's pace in a race is then each driver's measured gap with that driver's skill
removed.

## Step 1: every driver's gap in every race

The team race pace method is run on every race with dry-tyre laps and
wet-tyre laps measured separately, so a driver's pace in the wet is kept apart
from the same driver's pace in the dry. Each driver's gap comes with its precision from the
stint rebuilds of that method. More precise gaps count for more in the next
step.

## Step 2: driver skills from teammate comparisons

For every team in every race, each driver's gap is compared with the team's
average gap. The car is the same for both, so it cancels out and what remains is
the difference between the drivers. Every such comparison asks for:

> driver's skill − team average of skills = driver's gap − team average gap

Skills are found that satisfy all comparisons as closely as possible, with
precise comparisons weighted more.

**Skills change, but gradually.** Each driver has one skill per season, and a
driver's skill in one season is expected to be close to the previous season's.
How close is not chosen by us. Several allowances are tried, from almost no
change to large changes, and the one that best predicts teammate gaps in races
held out of the fit is used. A driver who improves over the years is followed
without letting a single season distort everything linked to that driver.

**The window** is the season being measured and the five before it. Several
seasons are needed because driver pairings often stay unchanged for years, and
only moves between teams link those teams to the rest of the grid.

**The average driver is zero.** Skills are stated against the average driver of
the grid. Moving that zero moves every car by the same amount, so it never
changes a gap between cars.

## Wet skill

Some drivers are distinctly stronger or weaker in the wet. A driver gets a
separate wet adjustment only when all of these hold:

- at least five wet races against a teammate;
- wet races against at least two different teammates, so that a pair who only
  ever meet each other in the wet cannot set each other's level;
- the adjustment is larger than twice its own uncertainty.

Otherwise the driver's dry skill is used in the wet as well.

## Measured with what was known at the time

A race's car-only figure is computed only from races before it: earlier
seasons in the window and earlier races of the same season. It is never
recomputed with later results.

This matters because drivers improve. If later races were used, a driver's
improvement over the season would be applied to races run before the improvement,
and too much would be removed from those races. Each race is measured with the
driver as known at that point, and its figure stays fixed once
published.

Season figures are the average of these fixed race figures, updated after every
race.

## Shrinking skills towards the average

Every skill is measured with some uncertainty. Ranking many uncertain
measurements makes the extremes look more extreme than they are: the quickest
measured driver is usually a little less quick than measured, and the slowest a
little less slow. Left alone, this overstates how much drivers differ and
removes too much from the cars at both ends of the grid.

Skills are therefore pulled towards the average driver of the race by an amount
measured from the data:

> kept share = real spread / (real spread + driver's measurement uncertainty)

The real spread is how much the grid's skills differ beyond what their
uncertainty explains. A driver with plenty of evidence keeps most of their
distance from the average. A driver with little evidence is pulled in more.
Nobody is moved past the average, and the order of drivers with similar
evidence is unchanged. Wet adjustments are shrunk the same way by their own
uncertainty, so a clear wet specialist keeps most of the adjustment.

## The car figure

For each team in each race, each driver's gap minus that driver's shrunk skill estimates
the car. Both drivers measure the same car, so their estimates are combined,
weighted by precision. Unlike the team race pace, there is no choice of the
quicker driver: with the drivers removed, both are evidence for the car.

Gaps are given against the quickest car of the race, in percent and in
seconds, with an approximate 80% range combining the uncertainty of the gaps and
of the skills.

## Rookies and drivers without enough data

A driver's skill is used only when the driver is established:

- at least eight races with a measured teammate within the window, and
- linked to the rest of the grid through teammates.

When one driver of a team is not established, the race uses only the
established teammate, and the figure is marked: *rookie in the car: not enough
comparative data to remove the driver; measured from the established teammate
only.* When neither driver is established, the team has no car-only figure for
that race.

## Example

At a race, Team A's drivers are 0.10% and 0.60% behind the quickest car; Team
B's are 0.30% and 0.40%.

- Before the race, Team A's first driver has a shrunk skill of −0.40% (faster
  than average) and the teammate +0.10%. Team B's drivers are at 0.00% and
  +0.10%.
- Team A's car: 0.10 − (−0.40) = 0.50 and 0.60 − 0.10 = 0.50, so 0.50.
- Team B's car: 0.30 − 0.00 = 0.30 and 0.40 − 0.10 = 0.30, so 0.30.

On team race pace, Team A was quicker by 0.20%. With the drivers removed, Team
B's car was quicker by 0.20%: Team A's figure was carried by its first driver.

## Season figures

As for team race pace: mean and median, as measured and rebased, over all
counted races and over the common races in which every team was measured. A
race counts when at least half of its teams have a car-only figure. When a race
has both dry and wet running, the condition with more measured laps is the
race's figure; both are reported.

## What the figure does not show

Separating car from driver is an estimate, and it has limits we state openly:

- **Car and driver interact.** A car built around one driver's style, or a team
  giving one car new parts first, shows up as a difference between drivers.
- **Skills rest on links.** Teams are placed on one scale through drivers who
  changed teams. Where those links are few, skills are less certain, which the
  ranges and the shrinking reflect.
- **Early in a season the evidence is thinner.** The first races of a season
  rely mostly on earlier seasons, and ranges are wider.
- **Our records start in 2018.** Windows for the first seasons are shorter and
  some teams can only be linked in part of a season.
- **Race pace only.** Qualifying is not part of a driver's skill here.

## Reproducing a figure

The method is implemented in [`services/analysis`](../../services/analysis):

```powershell
analysis car-race --event 2024-20
analysis car-season --season 2024
```

`--json` writes the full result, including every race's figures, the drivers
used and the notes.
