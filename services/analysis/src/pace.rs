//! Team pace of one session, measured from that session's own laps.
//!
//! Every function here is a calculation on recorded laps: no priors, no weighting
//! across sessions. Results are a representative lap time per team, turned into a
//! percentage gap to the quickest team with [`gaps_pct`].

use std::collections::{BTreeMap, HashMap};

use nalgebra::{DMatrix, DVector};

use crate::laps::{self, Lap};

/// Team -> representative lap time in seconds.
pub type TeamTimes = BTreeMap<String, f64>;

/// How a team's two (or more) drivers combine into one number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agg {
    /// The quicker driver.
    Best,
    /// The mean of the drivers.
    Mean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stat {
    Fastest,
    Median,
    Mean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compounds {
    All,
    Slicks,
    Inters,
}

/// Which race laps count.
#[derive(Debug, Clone, Copy)]
pub struct RaceFilter {
    pub compounds: Compounds,
    /// Laps completed this close or closer to the car ahead are left out.
    pub min_gap_s: Option<f64>,
    /// Laps up to and including this number are left out (the start).
    pub skip_laps: u32,
    /// Keep only laps the timing feed marks accurate.
    pub accurate_only: bool,
}

impl Default for RaceFilter {
    fn default() -> Self {
        Self { compounds: Compounds::All, min_gap_s: None, skip_laps: 1, accurate_only: true }
    }
}

/// Terms of the per-race lap-time regression.
#[derive(Debug, Clone, Copy)]
pub struct RaceModel {
    /// Tyre-age slope per team (else one per compound).
    pub team_wear: bool,
    /// Lap-number slope per compound (else one for the race). The lap-number term
    /// carries fuel burn and track evolution; tyre age resets at stops, so the two
    /// are separable within one race without assuming a fuel rate.
    pub lap_slope_per_compound: bool,
    /// Drop laps whose residual exceeds this many robust standard deviations, then refit.
    pub trim_sigma: Option<f64>,
    /// Gap to the car ahead as a term (seconds lost when within 2 s), instead of filtering.
    pub traffic_term: bool,
}

pub fn gaps_pct(times: &TeamTimes) -> BTreeMap<String, f64> {
    let best = times.values().copied().fold(f64::INFINITY, f64::min);
    times.iter().map(|(team, t)| (team.clone(), (t - best) / best * 100.0)).collect()
}

fn combine(per_driver: &HashMap<String, f64>, teams: &HashMap<String, String>, agg: Agg) -> TeamTimes {
    let mut grouped: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (driver, t) in per_driver {
        if let Some(team) = teams.get(driver) {
            grouped.entry(team.clone()).or_default().push(*t);
        }
    }
    grouped
        .into_iter()
        .map(|(team, ts)| {
            let v = match agg {
                Agg::Best => ts.iter().copied().fold(f64::INFINITY, f64::min),
                Agg::Mean => ts.iter().sum::<f64>() / ts.len() as f64,
            };
            (team, v)
        })
        .collect()
}

fn statistic(values: &mut [f64], stat: Stat) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(match stat {
        Stat::Fastest => values.iter().copied().fold(f64::INFINITY, f64::min),
        Stat::Mean => values.iter().sum::<f64>() / values.len() as f64,
        Stat::Median => median(values),
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Qualifying (or any one-lap session): each driver's best valid lap.
pub fn one_lap_team_times(session: &[Lap], agg: Agg) -> TeamTimes {
    let mut best: HashMap<String, f64> = HashMap::new();
    for lap in session.iter().filter(|l| !l.deleted) {
        if let Some(t) = lap.lap_time {
            let e = best.entry(lap.driver.clone()).or_insert(f64::INFINITY);
            *e = e.min(t);
        }
    }
    combine(&best, &laps::driver_teams(session), agg)
}

/// The race laps a filter keeps, with each lap's gap to the car ahead.
pub fn race_laps<'a>(race: &'a [Lap], filter: &RaceFilter) -> Vec<(&'a Lap, f64, Option<f64>)> {
    let gaps = laps::gaps_ahead(race);
    race.iter()
        .zip(gaps)
        .filter_map(|(lap, gap)| {
            let t = lap.lap_time?;
            let compound_ok = match filter.compounds {
                Compounds::All => lap.compound.is_some(),
                Compounds::Slicks => lap.is_slick(),
                Compounds::Inters => lap.compound.as_deref() == Some("INTERMEDIATE"),
            };
            let keep = compound_ok
                && lap.is_green()
                && !lap.is_pit_lap()
                && !lap.deleted
                && lap.lap_number > filter.skip_laps
                && (!filter.accurate_only || lap.accurate)
                && filter.min_gap_s.is_none_or(|min| gap.is_none_or(|g| g > min));
            keep.then_some((lap, t, gap))
        })
        .collect()
}

/// A plain statistic of each driver's kept race laps.
pub fn race_stat_team_times(race: &[Lap], filter: &RaceFilter, stat: Stat, agg: Agg, min_laps: usize) -> TeamTimes {
    let mut per_driver: HashMap<String, Vec<f64>> = HashMap::new();
    for (lap, t, _) in race_laps(race, filter) {
        per_driver.entry(lap.driver.clone()).or_default().push(t);
    }
    let values: HashMap<String, f64> = per_driver
        .into_iter()
        .filter(|(_, ts)| ts.len() >= min_laps)
        .filter_map(|(d, mut ts)| statistic(&mut ts, stat).map(|v| (d, v)))
        .collect();
    combine(&values, &laps::driver_teams(race), agg)
}

/// What a lap is compared with on the same lap number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LapBase {
    /// The quickest kept lap of that lap number.
    Fastest,
    /// The median kept lap of that lap number.
    Median,
}

/// Lap-by-lap relative pace: every kept lap as a ratio to the field on the same lap
/// number (so changing conditions affect everyone alike), then a statistic of each
/// driver's ratios. Returned as a time-like value (ratio x the race's median kept lap)
/// so [`gaps_pct`] gives the same percentages.
pub fn race_per_lap_team_times(
    race: &[Lap],
    filter: &RaceFilter,
    base: LapBase,
    stat: Stat,
    agg: Agg,
    min_laps: usize,
    min_cars: usize,
) -> TeamTimes {
    let kept = race_laps(race, filter);
    let mut by_lap: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for (lap, t, _) in &kept {
        by_lap.entry(lap.lap_number).or_default().push(*t);
    }
    let bases: HashMap<u32, f64> = by_lap
        .into_iter()
        .filter(|(_, ts)| ts.len() >= min_cars)
        .map(|(n, mut ts)| {
            let b = match base {
                LapBase::Fastest => ts.iter().copied().fold(f64::INFINITY, f64::min),
                LapBase::Median => median(&mut ts),
            };
            (n, b)
        })
        .collect();
    let mut all: Vec<f64> = kept.iter().map(|(_, t, _)| *t).collect();
    if all.is_empty() {
        return TeamTimes::new();
    }
    let scale = median(&mut all);
    let mut per_driver: HashMap<String, Vec<f64>> = HashMap::new();
    for (lap, t, _) in &kept {
        if let Some(b) = bases.get(&lap.lap_number) {
            per_driver.entry(lap.driver.clone()).or_default().push(t / b);
        }
    }
    let values: HashMap<String, f64> = per_driver
        .into_iter()
        .filter(|(_, rs)| rs.len() >= min_laps)
        .filter_map(|(d, mut rs)| statistic(&mut rs, stat).map(|v| (d, v * scale)))
        .collect();
    combine(&values, &laps::driver_teams(race), agg)
}

/// Driver effects from a least-squares fit of every kept lap:
/// lap time = driver + compound offset + wear x tyre age + slope x lap number (+ traffic).
/// A driver's time is the fitted lap at a common reference (most-run compound, median
/// tyre age and lap number), so drivers are compared on equal tyres and fuel.
pub fn race_model_team_times(race: &[Lap], filter: &RaceFilter, model: &RaceModel, agg: Agg, min_laps: usize) -> TeamTimes {
    let kept = race_laps(race, filter);
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (lap, _, _) in &kept {
        *counts.entry(lap.driver.as_str()).or_default() += 1;
    }
    let mut rows: Vec<(&Lap, f64, Option<f64>)> =
        kept.into_iter().filter(|(l, _, _)| counts[l.driver.as_str()] >= min_laps).collect();
    if rows.is_empty() {
        return TeamTimes::new();
    }
    let teams = laps::driver_teams(race);

    let mut drivers: Vec<String> = rows.iter().map(|(l, _, _)| l.driver.clone()).collect();
    drivers.sort();
    drivers.dedup();
    let mut compound_counts: BTreeMap<String, usize> = BTreeMap::new();
    for (l, _, _) in &rows {
        *compound_counts.entry(l.compound.clone().unwrap_or_default()).or_default() += 1;
    }
    let reference_compound = compound_counts.iter().max_by_key(|(_, n)| **n).map(|(c, _)| c.clone()).unwrap();
    let compounds: Vec<String> = compound_counts.keys().cloned().collect();
    let mut team_names: Vec<String> = drivers.iter().filter_map(|d| teams.get(d).cloned()).collect();
    team_names.sort();
    team_names.dedup();

    let mut fit = None;
    for _pass in 0..2 {
        let design = Design::new(&drivers, &compounds, &reference_compound, &team_names, model);
        let (x, y) = design.matrix(&rows, &teams);
        let Ok(beta) = x.clone().svd(true, true).solve(&y, 1e-10) else { return TeamTimes::new() };
        let residuals = &y - &x * &beta;
        fit = Some((design, beta));
        let Some(k) = model.trim_sigma else { break };
        let mut abs: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
        let sigma = 1.4826 * median(&mut abs);
        let keep: Vec<bool> = residuals.iter().map(|r| r.abs() <= k * sigma).collect();
        if keep.iter().all(|&k| k) {
            break;
        }
        rows = rows.into_iter().zip(keep).filter_map(|(r, k)| k.then_some(r)).collect();
    }
    let (design, beta) = fit.unwrap();

    let mut ages: Vec<f64> = rows.iter().filter_map(|(l, _, _)| l.tyre_life).collect();
    let mut lap_numbers: Vec<f64> = rows.iter().map(|(l, _, _)| l.lap_number as f64).collect();
    let age_ref = if ages.is_empty() { 0.0 } else { median(&mut ages) };
    let lap_ref = median(&mut lap_numbers);

    let per_driver: HashMap<String, f64> = drivers
        .iter()
        .filter_map(|d| {
            let team = teams.get(d)?;
            let reference = Reference { driver: d, team, compound: &reference_compound, age: age_ref, lap: lap_ref };
            Some((d.clone(), design.predict(&beta, &reference)))
        })
        .collect();
    combine(&per_driver, &teams, agg)
}

struct Reference<'a> {
    driver: &'a str,
    team: &'a str,
    compound: &'a str,
    age: f64,
    lap: f64,
}

/// Column layout of the regression.
struct Design<'a> {
    drivers: &'a [String],
    compounds: &'a [String],
    reference_compound: &'a str,
    teams: &'a [String],
    model: &'a RaceModel,
}

impl<'a> Design<'a> {
    fn new(drivers: &'a [String], compounds: &'a [String], reference_compound: &'a str, teams: &'a [String], model: &'a RaceModel) -> Self {
        Self { drivers, compounds, reference_compound, teams, model }
    }

    fn offsets(&self) -> usize {
        self.drivers.len()
    }
    fn wear_start(&self) -> usize {
        self.offsets() + self.compounds.len() - 1
    }
    fn wear_columns(&self) -> usize {
        if self.model.team_wear { self.teams.len() } else { self.compounds.len() }
    }
    fn lap_start(&self) -> usize {
        self.wear_start() + self.wear_columns()
    }
    fn lap_columns(&self) -> usize {
        if self.model.lap_slope_per_compound { self.compounds.len() } else { 1 }
    }
    fn traffic_column(&self) -> usize {
        self.lap_start() + self.lap_columns()
    }
    fn width(&self) -> usize {
        self.traffic_column() + usize::from(self.model.traffic_term)
    }

    fn row(&self, driver: &str, team: &str, compound: &str, age: f64, lap: f64, gap: Option<f64>) -> Vec<f64> {
        let mut r = vec![0.0; self.width()];
        r[self.drivers.iter().position(|d| d == driver).unwrap()] = 1.0;
        let others: Vec<&String> = self.compounds.iter().filter(|c| *c != self.reference_compound).collect();
        if let Some(j) = others.iter().position(|c| *c == compound) {
            r[self.offsets() + j] = 1.0;
        }
        let ci = self.compounds.iter().position(|c| c == compound).unwrap_or(0);
        let wear = if self.model.team_wear { self.teams.iter().position(|t| t == team).unwrap_or(0) } else { ci };
        r[self.wear_start() + wear] = age;
        let lap_col = if self.model.lap_slope_per_compound { ci } else { 0 };
        r[self.lap_start() + lap_col] = lap;
        if self.model.traffic_term {
            // Seconds of following: 2 s and more counts as free air.
            r[self.traffic_column()] = gap.map_or(0.0, |g| (2.0 - g).max(0.0));
        }
        r
    }

    fn matrix(&self, rows: &[(&Lap, f64, Option<f64>)], teams: &HashMap<String, String>) -> (DMatrix<f64>, DVector<f64>) {
        let width = self.width();
        let mut data = Vec::with_capacity(rows.len() * width);
        for (lap, _, gap) in rows {
            let team = teams.get(&lap.driver).map(String::as_str).unwrap_or("");
            data.extend(self.row(
                &lap.driver,
                team,
                lap.compound.as_deref().unwrap_or(""),
                lap.tyre_life.unwrap_or(0.0),
                lap.lap_number as f64,
                *gap,
            ));
        }
        let x = DMatrix::from_row_slice(rows.len(), width, &data);
        let y = DVector::from_iterator(rows.len(), rows.iter().map(|(_, t, _)| *t));
        (x, y)
    }

    fn predict(&self, beta: &DVector<f64>, at: &Reference) -> f64 {
        let r = self.row(at.driver, at.team, at.compound, at.age, at.lap, None);
        r.iter().zip(beta.iter()).map(|(a, b)| a * b).sum()
    }
}
