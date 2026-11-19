//! Team race pace of one race, measured from that race's own laps.
//!
//! A least-squares fit of the kept laps separates the car from tyre compound, tyre
//! age, fuel and track evolution, all measured within the race: no priors, no
//! weighting across races. Results are a representative lap time per team, turned
//! into gaps to the quickest team with [`gaps_pct`] or [`gaps_s`].

use std::collections::{BTreeMap, BTreeSet, HashMap};

use nalgebra::{DMatrix, DVector};

use crate::laps::{self, Lap};

/// Team -> representative lap time in seconds.
pub type TeamTimes = BTreeMap<String, f64>;

/// How a team's cars combine into one number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agg {
    /// Fit each driver, keep the quicker driver.
    Best,
    /// Fit each driver, take the mean of the drivers.
    Mean,
    /// On every lap number keep the team's quickest kept lap (either car), then fit
    /// the team on those laps.
    LapBest,
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
    /// Laps this many or fewer after a safety car, virtual safety car or red flag
    /// are left out (restarts: cold tyres, bunched field). 0 keeps them.
    pub skip_after_neutral: u32,
}

impl Default for RaceFilter {
    fn default() -> Self {
        Self { compounds: Compounds::All, min_gap_s: None, skip_laps: 1, accurate_only: true, skip_after_neutral: 0 }
    }
}

/// How lap number enters the fit: fuel burn and track evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LapTrend {
    /// One slope for the race.
    Linear,
    /// One slope per compound.
    LinearPerCompound,
    /// A separate time for every lap number, shared by all cars: absorbs any change
    /// of conditions (drying, rain, rubber) lap by lap.
    PerLap,
}

/// Terms of the per-race lap-time regression.
#[derive(Debug, Clone, Copy)]
pub struct RaceModel {
    /// Tyre-age slope per team (else one per compound).
    pub team_wear: bool,
    pub lap_trend: LapTrend,
    /// Drop laps whose residual exceeds this many robust standard deviations, then refit.
    pub trim_sigma: Option<f64>,
    /// Gap to the car ahead as a term (seconds lost when within 2 s), instead of filtering.
    pub traffic_term: bool,
}

/// Laps a driver (or team) needs to be measured.
pub const MIN_LAPS: usize = 5;

impl RaceFilter {
    /// The standard team race pace laps: dry tyres, green, after lap 1, no pit laps,
    /// more than 2 s behind the car ahead.
    pub fn standard() -> Self {
        Self { compounds: Compounds::Slicks, min_gap_s: Some(2.0), ..Self::default() }
    }
}

impl RaceModel {
    /// The standard team race pace fit: a time per lap number shared by all cars,
    /// one tyre-wear rate per compound, one 3-sigma trim and refit.
    pub fn standard() -> Self {
        Self { team_wear: false, lap_trend: LapTrend::PerLap, trim_sigma: Some(3.0), traffic_term: false }
    }
}

/// Standard team race pace of one race: [`RaceFilter::standard`] laps,
/// [`RaceModel::standard`] fit, the quicker driver of each team.
pub fn team_race_pace(race: &[Lap]) -> TeamTimes {
    race_model_team_times(race, &RaceFilter::standard(), &RaceModel::standard(), Agg::Best, MIN_LAPS)
}

pub fn gaps_pct(times: &TeamTimes) -> BTreeMap<String, f64> {
    let best = times.values().copied().fold(f64::INFINITY, f64::min);
    times.iter().map(|(team, t)| (team.clone(), (t - best) / best * 100.0)).collect()
}

pub fn gaps_s(times: &TeamTimes) -> BTreeMap<String, f64> {
    let best = times.values().copied().fold(f64::INFINITY, f64::min);
    times.iter().map(|(team, t)| (team.clone(), t - best)).collect()
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

fn is_neutralised(lap: &Lap) -> bool {
    lap.track_status.as_deref().is_some_and(|s| s.contains(['4', '5', '6', '7']))
}

/// The race laps a filter keeps, with each lap's gap to the car ahead.
pub fn race_laps<'a>(race: &'a [Lap], filter: &RaceFilter) -> Vec<(&'a Lap, f64, Option<f64>)> {
    let gaps = laps::gaps_ahead(race);
    let neutral: BTreeSet<u32> = race.iter().filter(|l| is_neutralised(l)).map(|l| l.lap_number).collect();
    let after_neutral = |n: u32| {
        filter.skip_after_neutral > 0
            && neutral.range(n.saturating_sub(filter.skip_after_neutral)..n).next().is_some()
    };
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
                && !after_neutral(lap.lap_number)
                && filter.min_gap_s.is_none_or(|min| gap.is_none_or(|g| g > min));
            keep.then_some((lap, t, gap))
        })
        .collect()
}

/// One lap entering the fit. `entity` is the driver, or the team for [`Agg::LapBest`].
#[derive(Debug, Clone)]
struct Obs {
    entity: String,
    team: String,
    compound: String,
    age: f64,
    lap: u32,
    time: f64,
    gap: Option<f64>,
}

fn observations(race: &[Lap], filter: &RaceFilter, agg: Agg, min_laps: usize) -> Vec<Obs> {
    let teams = laps::driver_teams(race);
    let mut obs: Vec<Obs> = race_laps(race, filter)
        .into_iter()
        .map(|(lap, time, gap)| {
            let team = teams.get(&lap.driver).cloned().unwrap_or_default();
            Obs {
                entity: if agg == Agg::LapBest { team.clone() } else { lap.driver.clone() },
                team,
                compound: lap.compound.clone().unwrap_or_default(),
                age: lap.tyre_life.unwrap_or(0.0),
                lap: lap.lap_number,
                time,
                gap,
            }
        })
        .collect();
    if agg == Agg::LapBest {
        let mut quickest: BTreeMap<(String, u32), Obs> = BTreeMap::new();
        for o in obs {
            let key = (o.team.clone(), o.lap);
            match quickest.get(&key) {
                Some(kept) if kept.time <= o.time => {}
                _ => {
                    quickest.insert(key, o);
                }
            }
        }
        obs = quickest.into_values().collect();
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for o in &obs {
        *counts.entry(o.entity.clone()).or_default() += 1;
    }
    obs.retain(|o| counts[&o.entity] >= min_laps);
    obs
}

/// Least-squares fit of every kept lap:
/// lap time = entity + compound offset + wear x tyre age + lap trend (+ traffic).
/// Entities are compared at a common reference (most-run compound, median tyre age
/// and lap number, free air), so cars are compared on equal tyres and fuel. The
/// returned time is the race's median kept lap plus the entity's fitted difference
/// to the most comparable entity.
///
/// Tyre wear is measured from cars running different tyre ages on the same lap.
/// When every car ran a single dry stint (stops a lap or two apart) wear cannot be
/// told apart from the lap terms and the gaps would depend on it, so the fit is
/// repeated without the wear term and the version that compares more cars is kept.
pub fn race_model_team_times(race: &[Lap], filter: &RaceFilter, model: &RaceModel, agg: Agg, min_laps: usize) -> TeamTimes {
    let obs = observations(race, filter, agg, min_laps);
    if obs.is_empty() {
        return TeamTimes::new();
    }
    let (with_wear, all_compared) = fit_cars(obs.clone(), model, true);
    let cars = if all_compared {
        with_wear
    } else {
        let (without_wear, _) = fit_cars(obs, model, false);
        if without_wear.len() > with_wear.len() { without_wear } else { with_wear }
    };

    let mut by_team: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (team, t) in cars {
        by_team.entry(team).or_default().push(t);
    }
    by_team
        .into_iter()
        .map(|(team, ts)| {
            let v = match agg {
                Agg::Best | Agg::LapBest => ts.iter().copied().fold(f64::INFINITY, f64::min),
                Agg::Mean => ts.iter().sum::<f64>() / ts.len() as f64,
            };
            (team, v)
        })
        .collect()
}

/// One fit: (team, time) per comparable entity, and whether every entity was comparable.
fn fit_cars(mut obs: Vec<Obs>, model: &RaceModel, wear: bool) -> (Vec<(String, f64)>, bool) {
    let mut fit = None;
    for pass in 0..2 {
        let design = Design::new(&obs, model, wear);
        let (x, y) = design.matrix(&obs);
        let svd = x.clone().svd(true, true);
        let Ok(beta) = svd.solve(&y, 1e-10) else { return (Vec::new(), false) };
        let residuals = &y - &x * &beta;
        fit = Some((design, beta, svd));
        // Trim once, then refit; the final fit and its laps stay together.
        let (Some(k), 0) = (model.trim_sigma, pass) else { break };
        let mut abs: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
        let sigma = 1.4826 * median(&mut abs);
        let keep: Vec<bool> = residuals.iter().map(|r| r.abs() <= k * sigma).collect();
        if keep.iter().all(|&k| k) {
            break;
        }
        obs = obs.into_iter().zip(keep).filter_map(|(o, k)| k.then_some(o)).collect();
    }
    let (design, beta, svd) = fit.expect("fitted at least once");

    let mut ages: Vec<f64> = obs.iter().map(|o| o.age).collect();
    let mut laps_run: Vec<f64> = obs.iter().map(|o| o.lap as f64).collect();
    let mut kept_times: Vec<f64> = obs.iter().map(|o| o.time).collect();
    let reference_age = median(&mut ages);
    let reference_lap = design.nearest_lap(median(&mut laps_run).round() as u32);
    let base = median(&mut kept_times);

    let mut counts: BTreeMap<&str, (usize, &str)> = BTreeMap::new();
    for o in &obs {
        let e = counts.entry(o.entity.as_str()).or_insert((0, o.team.as_str()));
        e.0 += 1;
    }
    let at = |entity: &str, team: &str| Obs {
        entity: entity.to_string(),
        team: team.to_string(),
        compound: design.reference_compound.clone(),
        age: reference_age,
        lap: reference_lap,
        time: 0.0,
        gap: None,
    };
    let row_space = RowSpace::new(&svd);
    let rows: BTreeMap<&str, Vec<f64>> = counts.iter().map(|(e, (_, team))| (*e, design.row(&at(e, team)))).collect();
    let comparable = |a: &[f64], b: &[f64]| {
        let contrast: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        row_space.contains(&contrast)
    };
    // Cars can only be compared through laps they share (directly or via others). The
    // reference is the car comparable with the most others, then the most measured.
    let reference_entity = counts
        .iter()
        .max_by_key(|(e, (n, _))| (rows.values().filter(|r| comparable(&rows[*e], r)).count(), *n))
        .map(|(e, _)| *e)
        .expect("laps kept");
    let reference_row = rows[reference_entity].clone();
    let reference_time = design.predict(&beta, &at(reference_entity, counts[reference_entity].1));

    // Only differences between cars are reported, anchored to the race's median kept
    // lap: when every car stops on the same lap, tyre age and the lap terms cannot be
    // told apart and the fitted absolute lap time is arbitrary, but the gaps between
    // cars are not. A car the laps do not connect to the reference is left out.
    let fitted: Vec<(String, f64)> = counts
        .iter()
        .filter(|(e, _)| comparable(&rows[*e], &reference_row))
        .map(|(e, (_, team))| (team.to_string(), base + design.predict(&beta, &at(e, team)) - reference_time))
        .collect();
    let all_compared = fitted.len() == counts.len();
    (fitted, all_compared)
}

/// The row space of a design matrix: the linear functions of the coefficients
/// that the laps determine.
struct RowSpace {
    basis: DMatrix<f64>,
}

impl RowSpace {
    fn new(svd: &nalgebra::SVD<f64, nalgebra::Dyn, nalgebra::Dyn>) -> Self {
        let v_t = svd.v_t.as_ref().expect("svd computed with V");
        let largest = svd.singular_values.iter().copied().fold(0.0, f64::max);
        let rows: Vec<usize> =
            (0..svd.singular_values.len()).filter(|&i| svd.singular_values[i] > 1e-9 * largest).collect();
        Self { basis: v_t.select_rows(&rows) }
    }

    fn contains(&self, c: &[f64]) -> bool {
        let c = DVector::from_column_slice(c);
        let norm2 = c.norm_squared();
        if norm2 == 0.0 {
            return true;
        }
        let projected = (&self.basis * &c).norm_squared();
        norm2 - projected <= 1e-8 * norm2
    }
}

/// Column layout of the regression.
struct Design<'a> {
    model: &'a RaceModel,
    /// Whether tyre age enters the fit.
    wear: bool,
    entities: Vec<String>,
    compounds: Vec<String>,
    reference_compound: String,
    teams: Vec<String>,
    laps: Vec<u32>,
}

impl<'a> Design<'a> {
    fn new(obs: &[Obs], model: &'a RaceModel, wear: bool) -> Self {
        let unique = |f: &dyn Fn(&Obs) -> String| -> Vec<String> {
            obs.iter().map(f).collect::<BTreeSet<_>>().into_iter().collect()
        };
        let mut compound_counts: BTreeMap<&str, usize> = BTreeMap::new();
        for o in obs {
            *compound_counts.entry(o.compound.as_str()).or_default() += 1;
        }
        let reference_compound =
            compound_counts.iter().max_by_key(|(_, n)| **n).map(|(c, _)| c.to_string()).unwrap_or_default();
        Self {
            model,
            wear,
            entities: unique(&|o| o.entity.clone()),
            compounds: unique(&|o| o.compound.clone()),
            reference_compound,
            teams: unique(&|o| o.team.clone()),
            laps: obs.iter().map(|o| o.lap).collect::<BTreeSet<_>>().into_iter().collect(),
        }
    }

    fn nearest_lap(&self, lap: u32) -> u32 {
        *self.laps.iter().min_by_key(|l| l.abs_diff(lap)).unwrap_or(&lap)
    }

    fn offsets_start(&self) -> usize {
        self.entities.len()
    }
    fn wear_start(&self) -> usize {
        self.offsets_start() + self.compounds.len() - 1
    }
    fn wear_columns(&self) -> usize {
        if !self.wear {
            0
        } else if self.model.team_wear { self.teams.len() } else { self.compounds.len() }
    }
    fn trend_start(&self) -> usize {
        self.wear_start() + self.wear_columns()
    }
    fn trend_columns(&self) -> usize {
        match self.model.lap_trend {
            LapTrend::Linear => 1,
            LapTrend::LinearPerCompound => self.compounds.len(),
            // The first lap number is the baseline; entity columns carry the level.
            LapTrend::PerLap => self.laps.len().saturating_sub(1),
        }
    }
    fn traffic_column(&self) -> usize {
        self.trend_start() + self.trend_columns()
    }
    fn width(&self) -> usize {
        self.traffic_column() + usize::from(self.model.traffic_term)
    }

    fn row(&self, o: &Obs) -> Vec<f64> {
        let mut r = vec![0.0; self.width()];
        if let Some(i) = self.entities.iter().position(|e| *e == o.entity) {
            r[i] = 1.0;
        }
        let others: Vec<&String> = self.compounds.iter().filter(|c| **c != self.reference_compound).collect();
        if let Some(j) = others.iter().position(|c| **c == o.compound) {
            r[self.offsets_start() + j] = 1.0;
        }
        let ci = self.compounds.iter().position(|c| *c == o.compound).unwrap_or(0);
        let wear = if self.model.team_wear { self.teams.iter().position(|t| *t == o.team).unwrap_or(0) } else { ci };
        if self.wear {
            r[self.wear_start() + wear] = o.age;
        }
        match self.model.lap_trend {
            LapTrend::Linear => r[self.trend_start()] = o.lap as f64,
            LapTrend::LinearPerCompound => r[self.trend_start() + ci] = o.lap as f64,
            LapTrend::PerLap => {
                if let Some(k) = self.laps.iter().position(|l| *l == o.lap).filter(|&k| k > 0) {
                    r[self.trend_start() + k - 1] = 1.0;
                }
            }
        }
        if self.model.traffic_term {
            // Seconds of following: 2 s and more counts as free air.
            r[self.traffic_column()] = o.gap.map_or(0.0, |g| (2.0 - g).max(0.0));
        }
        r
    }

    fn matrix(&self, obs: &[Obs]) -> (DMatrix<f64>, DVector<f64>) {
        let width = self.width();
        let mut data = Vec::with_capacity(obs.len() * width);
        for o in obs {
            data.extend(self.row(o));
        }
        let x = DMatrix::from_row_slice(obs.len(), width, &data);
        let y = DVector::from_iterator(obs.len(), obs.iter().map(|o| o.time));
        (x, y)
    }

    fn predict(&self, beta: &DVector<f64>, at: &Obs) -> f64 {
        self.row(at).iter().zip(beta.iter()).map(|(a, b)| a * b).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lap(driver: &str, team: &str, n: u32, time: f64, compound: &str, age: f64) -> Lap {
        Lap {
            driver: driver.into(),
            team: team.into(),
            lap_number: n,
            lap_time: Some(time),
            time: None,
            stint: None,
            pit_in: false,
            pit_out: false,
            compound: Some(compound.into()),
            tyre_life: Some(age),
            fresh_tyre: None,
            track_status: Some("1".into()),
            position: None,
            deleted: false,
            accurate: true,
        }
    }

    /// Cars with known pace, two compounds, different stop laps, tyre wear and a
    /// track whose speed changes irregularly from lap to lap.
    fn synthetic_race() -> Vec<Lap> {
        let cars = [("A1", "Alpha", 0.0), ("A2", "Alpha", 0.3), ("B1", "Beta", 0.5), ("B2", "Beta", 0.8), ("C1", "Gamma", 1.2)];
        let mut laps = Vec::new();
        for (i, (driver, team, pace)) in cars.iter().enumerate() {
            let stop = 20 + 3 * i as u32;
            for n in 1..=50u32 {
                let (compound, age, offset, wear) =
                    if n <= stop { ("MEDIUM", n as f64, 0.0, 0.05) } else { ("HARD", (n - stop) as f64, 0.4, 0.03) };
                let track = 90.0 - 0.06 * n as f64 + if n % 7 == 0 { 1.5 } else { 0.0 };
                laps.push(lap(driver, team, n, track + pace + offset + wear * age, compound, age));
            }
        }
        laps
    }

    #[test]
    fn recovers_known_car_gaps() {
        let gaps = gaps_s(&team_race_pace(&synthetic_race()));
        assert!(gaps["Alpha"].abs() < 1e-6);
        assert!((gaps["Beta"] - 0.5).abs() < 1e-6, "Beta {}", gaps["Beta"]);
        assert!((gaps["Gamma"] - 1.2).abs() < 1e-6, "Gamma {}", gaps["Gamma"]);
    }

    #[test]
    fn same_lap_stops_keep_gaps_and_a_real_percent_base() {
        // Every car stops on lap 25: tyre age equals lap number minus a constant for
        // everyone, so wear and the lap terms are not separable. Gaps must survive and
        // percentages must use a real lap time as base.
        let mut race = Vec::new();
        for (driver, team, pace) in [("A1", "Alpha", 0.0), ("B1", "Beta", 0.9), ("C1", "Gamma", 1.8)] {
            for n in 1..=50u32 {
                let (compound, age) = if n <= 25 { ("MEDIUM", n as f64) } else { ("HARD", (n - 25) as f64) };
                let time = 90.0 + pace + 0.05 * age + if n % 5 == 0 { 0.7 } else { 0.0 };
                race.push(lap(driver, team, n, time, compound, age));
            }
        }
        let times = team_race_pace(&race);
        let s = gaps_s(&times);
        assert!((s["Beta"] - 0.9).abs() < 1e-6 && (s["Gamma"] - 1.8).abs() < 1e-6, "{s:?}");
        let pct = gaps_pct(&times);
        assert!((pct["Gamma"] - 1.8 / 90.0 * 100.0).abs() < 0.1, "{pct:?}");
    }

    #[test]
    fn separate_running_groups_keep_the_larger_group() {
        // Two cars on slicks early only, three cars on slicks late only: the groups
        // share no laps, so only the larger group is compared.
        let mut race = Vec::new();
        for (driver, team, pace, laps) in [
            ("E1", "Early", 0.0, 2..=10u32),
            ("E2", "Early2", 0.2, 2..=10u32),
            ("L1", "Late", 0.5, 30..=45u32),
            ("L2", "Late2", 0.9, 30..=45u32),
            ("L3", "Late3", 1.4, 30..=45u32),
        ] {
            for n in laps {
                race.push(lap(driver, team, n, 90.0 + pace + 0.03 * n as f64, "MEDIUM", n as f64));
            }
        }
        let s = gaps_s(&team_race_pace(&race));
        assert_eq!(s.len(), 3, "{s:?}");
        assert!((s["Late2"] - 0.4).abs() < 1e-6 && (s["Late3"] - 0.9).abs() < 1e-6, "{s:?}");
    }

    #[test]
    fn single_stints_fall_back_to_no_wear() {
        // Every car runs one dry stint from a slightly different stop lap: wear is not
        // separable, so the fit drops it and still compares every car.
        let mut race = Vec::new();
        for (driver, team, pace, stop) in [("A1", "Alpha", 0.0, 40u32), ("B1", "Beta", 0.4, 41), ("C1", "Gamma", 0.8, 42)] {
            for n in (stop + 1)..=52 {
                race.push(lap(driver, team, n, 88.0 + pace - 0.1 * n as f64, "MEDIUM", (n - stop) as f64));
            }
        }
        let s = gaps_s(&team_race_pace(&race));
        assert_eq!(s.len(), 3, "{s:?}");
        assert!((s["Beta"] - 0.4).abs() < 1e-6 && (s["Gamma"] - 0.8).abs() < 1e-6, "{s:?}");
    }

    #[test]
    fn teammate_mean_averages_drivers() {
        let times = race_model_team_times(&synthetic_race(), &RaceFilter::standard(), &RaceModel::standard(), Agg::Mean, MIN_LAPS);
        let gaps = gaps_s(&times);
        assert!((gaps["Beta"] - 0.5).abs() < 1e-6, "Beta {}", gaps["Beta"]);
        assert!((gaps["Gamma"] - 1.05).abs() < 1e-6, "Gamma {}", gaps["Gamma"]);
    }

    #[test]
    fn neutralised_and_close_laps_are_left_out() {
        let mut race = synthetic_race();
        race[10].track_status = Some("14".into());
        let filter = RaceFilter { skip_after_neutral: 2, ..RaceFilter::standard() };
        let kept: Vec<u32> = race_laps(&race, &filter).iter().filter(|(l, _, _)| l.driver == "A1").map(|(l, _, _)| l.lap_number).collect();
        assert!(!kept.contains(&1) && !kept.contains(&11) && !kept.contains(&12) && !kept.contains(&13));
        assert!(kept.contains(&14));
    }
}
