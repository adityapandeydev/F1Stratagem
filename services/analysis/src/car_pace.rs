//! Car-only race pace: team race pace with each driver's skill removed.
//!
//! Built on the standard team race pace and never changing it. Two stages:
//!
//! 1. Every driver's gap in every race, from the standard recipe run separately on
//!    dry-tyre and on wet-tyre laps (so wet skill stays apart from dry skill), with
//!    the precision of each gap from its stint-resampled interval.
//! 2. Across a window of seasons: gap = car (this team, this race, this condition)
//!    + driver skill (this season) + noise. The car term is removed by comparing
//!    teammates within each team and race. Driver skills are linked across teams by
//!    drivers who changed teams and across seasons by letting each driver's skill
//!    change from one season to the next only as much as the data supports: the
//!    allowed change per season is chosen by how well it predicts teammate gaps in
//!    races left out (cross-validation). Skills are relative to the average driver,
//!    which shifts every car alike and leaves car gaps unchanged.
//!
//! A driver gets a separate wet adjustment only with enough wet teammate races;
//! otherwise his dry skill is used in the wet too.
//!
//! For each race, skills come from the other races of the window only. A team's car
//! pace in a race is the precision-weighted mean of its drivers' gaps minus their
//! skills. A driver without enough teammate comparisons in the window, or not
//! linked to the rest of the grid, is not established: the race uses only the
//! established teammate and is marked; with no established driver the team has no
//! car-only figure for that race.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use serde::Serialize;

use crate::archive::Archive;
use crate::pace::{self, Compounds, RaceFilter, RacePace, Recipe};
use crate::season::{SeasonTable, SeasonTeam, Summary};

/// Default seasons in the window: the race's season and the ones before it.
pub const WINDOW_SEASONS: i64 = 6;
/// Races with a teammate comparison a driver needs in the window to be established.
pub const MIN_TEAMMATE_RACES: usize = 8;
/// Wet races with a teammate comparison a driver needs for a separate wet skill.
pub const MIN_WET_TEAMMATE_RACES: usize = 5;
/// Candidate standard deviations (percent per season) of a driver's skill change
/// from one season to the next; the best by cross-validation is used.
pub const SKILL_DRIFT_CANDIDATES: [f64; 9] = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 3.0];
/// Folds (by race) of the cross-validation choosing the skill drift.
const DRIFT_FOLDS: usize = 5;
/// Stint resamples behind each stage-1 gap's precision.
const STAGE1_RESAMPLES: usize = 60;
/// Floor on a stage-1 gap's standard deviation (percent), so no single race dominates.
const MIN_SD_PCT: f64 = 0.03;
/// Normal quantile of the 10th/90th percentiles (80% intervals).
const Z80: f64 = 1.2816;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Condition {
    Dry,
    Wet,
}

/// One driver's measured gap (percent, to the race's quickest team) in one race and
/// condition, with its standard deviation.
#[derive(Debug, Clone)]
pub struct DriverObs {
    pub race: String,
    pub season: i64,
    pub condition: Condition,
    pub driver: String,
    pub team: String,
    pub gap_pct: f64,
    pub sd_pct: f64,
}

/// One race measured in one condition by the standard recipe.
#[derive(Debug, Clone)]
pub struct RaceRun {
    pub event_key: String,
    pub condition: Condition,
    pub pace: RacePace,
}

fn season_of(event_key: &str) -> i64 {
    event_key.split('-').next().and_then(|s| s.parse().ok()).unwrap_or(0)
}

fn condition_recipe(condition: Condition) -> Recipe {
    let mut recipe = Recipe::standard();
    recipe.filter = RaceFilter {
        compounds: match condition {
            Condition::Dry => Compounds::Slicks,
            Condition::Wet => Compounds::Inters,
        },
        ..recipe.filter
    };
    recipe
}

/// Every race of a season measured by the standard recipe on dry-tyre and on
/// wet-tyre laps separately; a run counts under the same rule as the main season
/// (tyre-corrected, at least half of the teams measured).
pub fn season_runs(archive: &Archive, season: i64) -> Result<Vec<RaceRun>> {
    let mut runs = Vec::new();
    for event_key in archive.season_events(season)? {
        let Some(race) = archive.session_of_type(&event_key, "R")? else { continue };
        let laps = archive.laps(&race.session_id)?;
        for condition in [Condition::Dry, Condition::Wet] {
            let any = laps.iter().any(|l| match condition {
                Condition::Dry => l.is_slick(),
                Condition::Wet => l.is_wet_tyre(),
            });
            if !any {
                continue;
            }
            let seed = pace::seed_for(&format!("{}-{condition:?}", race.session_id));
            let Some(p) = pace::race_pace(&laps, &condition_recipe(condition), STAGE1_RESAMPLES, seed) else { continue };
            if p.tyre_corrected && 2 * p.teams.len() >= p.teams_in_race {
                runs.push(RaceRun { event_key: event_key.clone(), condition, pace: p });
            }
        }
    }
    Ok(runs)
}

fn observations(runs: &[&RaceRun]) -> Vec<DriverObs> {
    let mut out = Vec::new();
    for run in runs {
        for d in &run.pace.drivers {
            let sd = d.interval_pct.map_or(f64::NAN, |[lo, hi]| (hi - lo) / (2.0 * Z80));
            out.push(DriverObs {
                race: run.event_key.clone(),
                season: season_of(&run.event_key),
                condition: run.condition,
                driver: d.driver.clone(),
                team: d.team.clone(),
                gap_pct: d.gap_pct,
                sd_pct: if sd.is_finite() { sd.max(MIN_SD_PCT) } else { f64::NAN },
            });
        }
    }
    // A gap without an interval gets the median precision of the others.
    let mut sds: Vec<f64> = out.iter().map(|o| o.sd_pct).filter(|s| s.is_finite()).collect();
    let fallback = if sds.is_empty() { 0.2 } else { pace::median(&mut sds) };
    for o in &mut out {
        if !o.sd_pct.is_finite() {
            o.sd_pct = fallback;
        }
    }
    out
}

/// Driver skills (percent, relative to the average driver; lower is faster) per
/// season, fitted from teammate comparisons.
#[derive(Debug, Clone, Default)]
pub struct Skills {
    /// Dry skill per driver and season with comparisons.
    dry: HashMap<(String, i64), f64>,
    /// Wet minus dry, for drivers with enough wet teammate races.
    wet_delta: HashMap<String, f64>,
    variance: HashMap<(String, i64, Condition), f64>,
    /// Drivers linked to the largest connected part of the grid.
    linked: BTreeSet<String>,
    /// Races in which each driver had a measured teammate.
    teammate_races: HashMap<String, usize>,
    /// The skill drift used (percent per season).
    pub drift: f64,
}

impl Skills {
    pub fn established(&self, driver: &str) -> bool {
        self.linked.contains(driver) && self.teammate_races.get(driver).copied().unwrap_or(0) >= MIN_TEAMMATE_RACES
    }

    /// The driver's season with a fitted skill nearest to `season`.
    fn nearest_season(&self, driver: &str, season: i64) -> Option<i64> {
        self.dry.keys().filter(|(d, _)| d == driver).map(|(_, s)| *s).min_by_key(|s| (s.abs_diff(season), -*s))
    }

    pub fn skill(&self, driver: &str, season: i64, condition: Condition) -> f64 {
        let Some(s) = self.nearest_season(driver, season) else { return 0.0 };
        let dry = self.dry[&(driver.to_string(), s)];
        match condition {
            Condition::Dry => dry,
            Condition::Wet => dry + self.wet_delta.get(driver).copied().unwrap_or(0.0),
        }
    }

    fn variance(&self, driver: &str, season: i64, condition: Condition) -> f64 {
        let Some(s) = self.nearest_season(driver, season) else { return 0.0 };
        self.variance.get(&(driver.to_string(), s, condition)).copied().unwrap_or(0.0)
    }
}

/// Teammate groups: drivers of one team in one race and condition, at least two.
fn teammate_groups<'a>(obs: &[&'a DriverObs]) -> Vec<Vec<&'a DriverObs>> {
    let mut groups: BTreeMap<(&str, Condition, &str), Vec<&DriverObs>> = BTreeMap::new();
    for o in obs {
        groups.entry((o.race.as_str(), o.condition, o.team.as_str())).or_default().push(o);
    }
    groups.into_values().filter(|g| g.len() >= 2).collect()
}

/// Column layout: dry skill per (driver, season), then a wet adjustment per driver
/// with enough wet teammate races.
struct Columns {
    dry: BTreeMap<(String, i64), usize>,
    wet: BTreeMap<String, usize>,
}

impl Columns {
    fn new(groups: &[Vec<&DriverObs>]) -> Self {
        let mut dry = BTreeMap::new();
        let mut wet_races: BTreeMap<String, BTreeSet<&str>> = BTreeMap::new();
        for g in groups {
            for o in g {
                dry.entry((o.driver.clone(), o.season)).or_insert(0);
                if o.condition == Condition::Wet {
                    wet_races.entry(o.driver.clone()).or_default().insert(o.race.as_str());
                }
            }
        }
        for (i, v) in dry.values_mut().enumerate() {
            *v = i;
        }
        let n = dry.len();
        let wet = wet_races
            .into_iter()
            .filter(|(_, r)| r.len() >= MIN_WET_TEAMMATE_RACES)
            .enumerate()
            .map(|(i, (d, _))| (d, n + i))
            .collect();
        Self { dry, wet }
    }

    fn width(&self) -> usize {
        self.dry.len() + self.wet.len()
    }

    fn of(&self, o: &DriverObs) -> Vec<(usize, f64)> {
        let mut c = vec![(self.dry[&(o.driver.clone(), o.season)], 1.0)];
        if o.condition == Condition::Wet {
            if let Some(&w) = self.wet.get(&o.driver) {
                c.push((w, 1.0));
            }
        }
        c
    }
}

/// Weighted rows of one teammate group: each driver's gap minus the group's
/// precision-weighted mean gap = his skill minus the weighted mean skill (the car
/// term cancels).
fn group_rows(g: &[&DriverObs], cols: &Columns) -> Vec<(Vec<(usize, f64)>, f64, f64)> {
    let weights: Vec<f64> = g.iter().map(|o| 1.0 / (o.sd_pct * o.sd_pct)).collect();
    let total: f64 = weights.iter().sum();
    let mean_gap: f64 = g.iter().zip(&weights).map(|(o, w)| o.gap_pct * w).sum::<f64>() / total;
    let mut mean_x: BTreeMap<usize, f64> = BTreeMap::new();
    for (o, w) in g.iter().zip(&weights) {
        for (c, v) in cols.of(o) {
            *mean_x.entry(c).or_default() += v * w / total;
        }
    }
    g.iter()
        .zip(&weights)
        .map(|(o, w)| {
            let mut x: BTreeMap<usize, f64> = mean_x.iter().map(|(c, v)| (*c, -v)).collect();
            for (c, v) in cols.of(o) {
                *x.entry(c).or_default() += v;
            }
            (x.into_iter().collect(), o.gap_pct - mean_gap, *w)
        })
        .collect()
}

/// Penalty rows: a driver's skill changes between consecutive seasons with a
/// standard deviation of `drift` per season elapsed.
fn drift_rows(cols: &Columns, drift: f64) -> Vec<(Vec<(usize, f64)>, f64, f64)> {
    let mut seasons: BTreeMap<&str, Vec<(i64, usize)>> = BTreeMap::new();
    for ((d, s), c) in &cols.dry {
        seasons.entry(d.as_str()).or_default().push((*s, *c));
    }
    let mut rows = Vec::new();
    for list in seasons.values() {
        for pair in list.windows(2) {
            let ((s0, c0), (s1, c1)) = (pair[0], pair[1]);
            let gap = (s1 - s0) as f64;
            rows.push((vec![(c1, 1.0), (c0, -1.0)], 0.0, 1.0 / (drift * drift * gap)));
        }
    }
    rows
}

struct Solved {
    beta: DVector<f64>,
    /// Pseudo-inverse of the weighted normal matrix, and the data rows' dispersion.
    inverse: DMatrix<f64>,
    dispersion: f64,
}

fn solve(data: &[(Vec<(usize, f64)>, f64, f64)], penalty: &[(Vec<(usize, f64)>, f64, f64)], width: usize) -> Solved {
    let rows: Vec<&(Vec<(usize, f64)>, f64, f64)> = data.iter().chain(penalty).collect();
    let m = rows.len();
    let mut x = DMatrix::zeros(m, width);
    let mut y = DVector::zeros(m);
    for (i, (cols, target, w)) in rows.iter().enumerate() {
        let s = w.sqrt();
        for (c, v) in cols {
            x[(i, *c)] += v * s;
        }
        y[i] = target * s;
    }
    let svd = x.clone().svd(true, true);
    let beta = svd.solve(&y, 1e-10).unwrap_or_else(|_| DVector::zeros(width));
    let largest = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let v_t = svd.v_t.as_ref().expect("svd computed with V");
    let mut inverse = DMatrix::zeros(width, width);
    for k in 0..svd.singular_values.len() {
        let sv = svd.singular_values[k];
        if sv > 1e-9 * largest {
            let v = v_t.row(k).transpose();
            inverse += &v * v.transpose() / (sv * sv);
        }
    }
    let residual = &y - &x * &beta;
    let data_rss: f64 = (0..data.len()).map(|i| residual[i] * residual[i]).sum();
    let dispersion = if data.len() > width { data_rss / (data.len() - width) as f64 } else { 1.0 };
    Solved { beta, inverse, dispersion: dispersion.max(1e-12) }
}

/// The skill drift that best predicts teammate gaps in races left out of the fit.
pub fn choose_drift(obs: &[&DriverObs]) -> f64 {
    let groups = teammate_groups(obs);
    let races: Vec<&str> = groups.iter().map(|g| g[0].race.as_str()).collect::<BTreeSet<_>>().into_iter().collect();
    let fold_of: HashMap<&str, usize> = races.iter().enumerate().map(|(i, r)| (*r, i % DRIFT_FOLDS)).collect();
    let mut best = (f64::INFINITY, SKILL_DRIFT_CANDIDATES[0]);
    for &drift in &SKILL_DRIFT_CANDIDATES {
        let mut error = 0.0;
        for fold in 0..DRIFT_FOLDS {
            let train: Vec<Vec<&DriverObs>> = groups.iter().filter(|g| fold_of[g[0].race.as_str()] != fold).cloned().collect();
            let test: Vec<&Vec<&DriverObs>> = groups.iter().filter(|g| fold_of[g[0].race.as_str()] == fold).collect();
            let cols = Columns::new(&train);
            let data: Vec<_> = train.iter().flat_map(|g| group_rows(g, &cols)).collect();
            let solved = solve(&data, &drift_rows(&cols, drift), cols.width());
            let skills = to_skills(&cols, &solved, &train, drift);
            for g in test {
                // Predicted teammate deviations from the held-out race.
                let weights: Vec<f64> = g.iter().map(|o| 1.0 / (o.sd_pct * o.sd_pct)).collect();
                let total: f64 = weights.iter().sum();
                let gap_mean: f64 = g.iter().zip(&weights).map(|(o, w)| o.gap_pct * w).sum::<f64>() / total;
                let skill_mean: f64 =
                    g.iter().zip(&weights).map(|(o, w)| skills.skill(&o.driver, o.season, o.condition) * w).sum::<f64>() / total;
                for (o, w) in g.iter().zip(&weights) {
                    let predicted = skills.skill(&o.driver, o.season, o.condition) - skill_mean;
                    error += w * (o.gap_pct - gap_mean - predicted).powi(2);
                }
            }
        }
        if error < best.0 {
            best = (error, drift);
        }
    }
    best.1
}

fn to_skills(cols: &Columns, solved: &Solved, groups: &[Vec<&DriverObs>], drift: f64) -> Skills {
    let cov = |a: usize, b: usize| solved.inverse[(a, b)] * solved.dispersion;
    let mut skills = Skills { drift, ..Skills::default() };
    for ((d, s), &c) in &cols.dry {
        skills.dry.insert((d.clone(), *s), solved.beta[c]);
        skills.variance.insert((d.clone(), *s, Condition::Dry), cov(c, c));
        let wet_var = match cols.wet.get(d) {
            Some(&w) => cov(c, c) + cov(w, w) + 2.0 * cov(c, w),
            None => cov(c, c),
        };
        skills.variance.insert((d.clone(), *s, Condition::Wet), wet_var);
    }
    for (d, &w) in &cols.wet {
        skills.wet_delta.insert(d.clone(), solved.beta[w]);
    }

    // Links: teammates in a race join their drivers; keep the part with most comparisons.
    let drivers: Vec<&str> = cols.dry.keys().map(|(d, _)| d.as_str()).collect::<BTreeSet<_>>().into_iter().collect();
    let index: HashMap<&str, usize> = drivers.iter().enumerate().map(|(i, d)| (*d, i)).collect();
    let mut parent: Vec<usize> = (0..drivers.len()).collect();
    fn find(parent: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while parent[r] != r {
            r = parent[r];
        }
        let mut c = i;
        while parent[c] != r {
            let next = parent[c];
            parent[c] = r;
            c = next;
        }
        r
    }
    let mut races_of: HashMap<&str, BTreeSet<&str>> = HashMap::new();
    for g in groups {
        let first = index[g[0].driver.as_str()];
        for o in g {
            let (a, b) = (find(&mut parent, first), find(&mut parent, index[o.driver.as_str()]));
            parent[a] = b;
            races_of.entry(o.driver.as_str()).or_default().insert(o.race.as_str());
        }
    }
    let mut size: HashMap<usize, usize> = HashMap::new();
    for d in &drivers {
        let root = find(&mut parent, index[d]);
        *size.entry(root).or_default() += races_of[d].len();
    }
    let main = size.iter().max_by_key(|(_, s)| **s).map(|(r, _)| *r);
    for d in &drivers {
        if Some(find(&mut parent, index[d])) == main {
            skills.linked.insert(d.to_string());
        }
        skills.teammate_races.insert(d.to_string(), races_of[d].len());
    }
    skills
}

/// Fit driver skills per season from teammate comparisons with the given drift.
pub fn fit_skills(obs: &[&DriverObs], drift: f64) -> Skills {
    let groups = teammate_groups(obs);
    if groups.is_empty() {
        return Skills { drift, ..Skills::default() };
    }
    let cols = Columns::new(&groups);
    let data: Vec<_> = groups.iter().flat_map(|g| group_rows(g, &cols)).collect();
    let solved = solve(&data, &drift_rows(&cols, drift), cols.width());
    to_skills(&cols, &solved, &groups, drift)
}

/// One team's car-only pace in one race and condition.
#[derive(Debug, Clone, Serialize)]
pub struct CarTeam {
    pub team: String,
    /// Gap to the quickest car, driver skills removed (percent and seconds).
    pub gap_pct: f64,
    pub gap_s: f64,
    /// Approximate 80% interval of the gap (percent).
    pub interval_pct: Option<[f64; 2]>,
    /// Drivers whose laps measured the car.
    pub drivers: Vec<String>,
    /// Set when a driver could not be removed (not enough comparative data).
    pub note: Option<String>,
}

/// Car-only pace of one race in one condition.
#[derive(Debug, Clone, Serialize)]
pub struct CarRun {
    pub condition: Condition,
    /// Quickest car first.
    pub teams: Vec<CarTeam>,
    /// Teams measured by the main recipe but without an established driver.
    pub not_measured: Vec<String>,
}

pub const ROOKIE_NOTE: &str = "rookie in the car: not enough comparative data to remove the driver; measured from the established teammate only";

/// Car-only pace of one race and condition from its drivers' gaps and the skills
/// fitted without this race. `base_lap_s` converts percent to seconds.
pub fn car_run(condition: Condition, race_obs: &[&DriverObs], skills: &Skills, base_lap_s: f64) -> CarRun {
    let mut by_team: BTreeMap<&str, Vec<&DriverObs>> = BTreeMap::new();
    for o in race_obs {
        by_team.entry(o.team.as_str()).or_default().push(o);
    }
    let mut cars: Vec<(String, f64, f64, Vec<String>, Option<String>)> = Vec::new();
    let mut not_measured = Vec::new();
    for (team, drivers) in by_team {
        let established: Vec<&&DriverObs> = drivers.iter().filter(|o| skills.established(&o.driver)).collect();
        if established.is_empty() {
            not_measured.push(team.to_string());
            continue;
        }
        let (mut sum, mut weight) = (0.0, 0.0);
        for o in &established {
            let var = o.sd_pct * o.sd_pct + skills.variance(&o.driver, o.season, condition);
            sum += (o.gap_pct - skills.skill(&o.driver, o.season, condition)) / var;
            weight += 1.0 / var;
        }
        let note = (established.len() < drivers.len()).then(|| ROOKIE_NOTE.to_string());
        cars.push((team.to_string(), sum / weight, 1.0 / weight, established.iter().map(|o| o.driver.clone()).collect(), note));
    }
    let Some((best, best_var)) = cars.iter().min_by(|a, b| a.1.total_cmp(&b.1)).map(|c| (c.1, c.2)) else {
        return CarRun { condition, teams: Vec::new(), not_measured };
    };
    let mut teams: Vec<CarTeam> = cars
        .into_iter()
        .map(|(team, value, var, drivers, note)| {
            let gap = value - best;
            let interval = (gap > 0.0).then(|| {
                let half = Z80 * (var + best_var).sqrt();
                [gap - half, gap + half]
            });
            CarTeam { team, gap_pct: gap, gap_s: gap / 100.0 * base_lap_s, interval_pct: interval, drivers, note }
        })
        .collect();
    teams.sort_by(|a, b| a.gap_pct.total_cmp(&b.gap_pct));
    CarRun { condition, teams, not_measured }
}

/// Car-only pace of one race.
#[derive(Debug, Clone, Serialize)]
pub struct CarRace {
    pub event_key: String,
    /// The condition with more kept laps; its table is the race's headline.
    pub headline: Option<Condition>,
    pub runs: Vec<CarRun>,
    pub counted: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CarSeason {
    pub season: i64,
    /// First and last season of the window driver skills come from.
    pub window: [i64; 2],
    pub min_teammate_races: usize,
    /// Allowed change of a driver's skill per season (percent), chosen by cross-validation.
    pub skill_drift: f64,
    pub all_races: SeasonTable,
    pub common_races: SeasonTable,
    pub races: Vec<CarRace>,
}

fn window_runs(season: i64, runs: &BTreeMap<i64, Vec<RaceRun>>, window_seasons: i64) -> Vec<&RaceRun> {
    runs.iter().filter(|(s, _)| **s > season - window_seasons && **s <= season).flat_map(|(_, r)| r).collect()
}

/// Car-only pace of every race of `season`, from the runs of every season of its
/// window of `window_seasons` seasons (`runs[season]` for each).
pub fn car_season(season: i64, runs: &BTreeMap<i64, Vec<RaceRun>>, window_seasons: i64) -> CarSeason {
    let window = window_runs(season, runs, window_seasons);
    let all_obs = observations(&window);
    let drift = choose_drift(&all_obs.iter().collect::<Vec<_>>());
    let target: Vec<&RaceRun> = runs.get(&season).map(|r| r.iter().collect()).unwrap_or_default();
    let mut events: Vec<String> = Vec::new();
    for r in &target {
        if !events.contains(&r.event_key) {
            events.push(r.event_key.clone());
        }
    }
    let mut races = Vec::new();
    for event in events {
        let others: Vec<&DriverObs> = all_obs.iter().filter(|o| o.race != event).collect();
        let skills = fit_skills(&others, drift);
        let event_runs: Vec<&&RaceRun> = target.iter().filter(|r| r.event_key == event).collect();
        let mut car_runs = Vec::new();
        for run in &event_runs {
            let obs: Vec<&DriverObs> =
                all_obs.iter().filter(|o| o.race == event && o.condition == run.condition).collect();
            car_runs.push(car_run(run.condition, &obs, &skills, run.pace.base_lap_s));
        }
        let headline_run = event_runs.iter().max_by_key(|r| r.pace.laps.kept);
        let headline = headline_run.map(|r| r.condition);
        let teams_in_race = headline_run.map_or(0, |r| r.pace.teams_in_race);
        let measured = car_runs.iter().find(|c| Some(c.condition) == headline).map_or(0, |c| c.teams.len());
        let reason = (2 * measured < teams_in_race).then(|| "fewer than half of the teams measured".to_string());
        races.push(CarRace { event_key: event, headline, runs: car_runs, counted: reason.is_none(), reason });
    }
    let headline_teams = |r: &CarRace| -> Vec<(String, f64, f64)> {
        r.runs
            .iter()
            .find(|c| Some(c.condition) == r.headline)
            .map(|c| c.teams.iter().map(|t| (t.team.clone(), t.gap_pct, t.gap_s)).collect())
            .unwrap_or_default()
    };
    let counted: Vec<(String, Vec<(String, f64, f64)>)> =
        races.iter().filter(|r| r.counted).map(|r| (r.event_key.clone(), headline_teams(r))).collect();
    let season_teams: BTreeSet<String> = counted.iter().flat_map(|(_, t)| t.iter().map(|x| x.0.clone())).collect();
    let common: Vec<(String, Vec<(String, f64, f64)>)> = counted
        .iter()
        .filter(|(_, t)| season_teams.iter().all(|s| t.iter().any(|x| &x.0 == s)))
        .cloned()
        .collect();
    CarSeason {
        season,
        window: [season - window_seasons + 1, season],
        min_teammate_races: MIN_TEAMMATE_RACES,
        skill_drift: drift,
        all_races: summarise(&counted),
        common_races: summarise(&common),
        races,
    }
}

/// Season summary (mean and median, as measured and rebased) of per-race gaps.
fn summarise(races: &[(String, Vec<(String, f64, f64)>)]) -> SeasonTable {
    let mut pct: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut secs: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (_, teams) in races {
        for (team, p, s) in teams {
            pct.entry(team.clone()).or_default().push(*p);
            secs.entry(team.clone()).or_default().push(*s);
        }
    }
    let stats = |v: &[f64]| (v.iter().sum::<f64>() / v.len() as f64, pace::median(&mut v.to_vec()));
    let raw: Vec<(String, usize, (f64, f64), (f64, f64))> =
        pct.iter().map(|(t, p)| (t.clone(), p.len(), stats(p), stats(&secs[t]))).collect();
    let min = |f: &dyn Fn(&(String, usize, (f64, f64), (f64, f64))) -> f64| raw.iter().map(f).fold(f64::INFINITY, f64::min);
    let (pm, pmed, sm, smed) = (min(&|r| r.2 .0), min(&|r| r.2 .1), min(&|r| r.3 .0), min(&|r| r.3 .1));
    let mut teams: Vec<SeasonTeam> = raw
        .into_iter()
        .map(|(team, n, (p_mean, p_med), (s_mean, s_med))| SeasonTeam {
            team,
            races_measured: n,
            pct: Summary { mean: p_mean, median: p_med, mean_rebased: p_mean - pm, median_rebased: p_med - pmed },
            s: Summary { mean: s_mean, median: s_med, mean_rebased: s_mean - sm, median_rebased: s_med - smed },
        })
        .collect();
    teams.sort_by(|a, b| a.pct.mean.total_cmp(&b.pct.mean));
    SeasonTable { races: races.iter().map(|r| r.0.clone()).collect(), teams }
}

/// Internal driver skill table of a window (not part of the published outputs).
pub struct SkillRow {
    pub driver: String,
    /// Dry skill per season of the window (percent vs the average linked driver of
    /// the latest season; negative is faster); `None` without comparisons that season.
    pub dry: Vec<Option<f64>>,
    pub wet_delta: Option<f64>,
    pub teammate_races: usize,
    pub established: bool,
}

pub fn skills_table(season: i64, runs: &BTreeMap<i64, Vec<RaceRun>>, window_seasons: i64) -> (Vec<i64>, f64, Vec<SkillRow>) {
    let obs = observations(&window_runs(season, runs, window_seasons));
    let refs: Vec<&DriverObs> = obs.iter().collect();
    let drift = choose_drift(&refs);
    let skills = fit_skills(&refs, drift);
    let seasons: Vec<i64> = ((season - window_seasons + 1)..=season).collect();
    let latest: Vec<f64> = skills
        .dry
        .iter()
        .filter(|((d, s), _)| *s == season && skills.linked.contains(d))
        .map(|(_, v)| *v)
        .collect();
    let mean = if latest.is_empty() { 0.0 } else { latest.iter().sum::<f64>() / latest.len() as f64 };
    let drivers: BTreeSet<&String> = skills.dry.keys().map(|(d, _)| d).collect();
    let mut rows: Vec<SkillRow> = drivers
        .into_iter()
        .map(|d| SkillRow {
            driver: d.clone(),
            dry: seasons.iter().map(|s| skills.dry.get(&(d.clone(), *s)).map(|v| v - mean)).collect(),
            wet_delta: skills.wet_delta.get(d).copied(),
            teammate_races: skills.teammate_races[d],
            established: skills.established(d),
        })
        .collect();
    let last = |r: &SkillRow| r.dry.iter().rev().flatten().next().copied().unwrap_or(f64::INFINITY);
    rows.sort_by(|a, b| last(a).total_cmp(&last(b)));
    (seasons, drift, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(race: &str, driver: &str, team: &str, gap: f64) -> DriverObs {
        DriverObs {
            race: race.into(),
            season: season_of(race),
            condition: Condition::Dry,
            driver: driver.into(),
            team: team.into(),
            gap_pct: gap,
            sd_pct: 0.1,
        }
    }

    /// Three teams over three seasons of 10 races; drivers move between teams each
    /// season, so skills are linked across the grid. Each race's gaps are car +
    /// skill, measured to the quickest. `drift` changes driver B's skill by season.
    fn grid(b_by_season: [f64; 3]) -> Vec<DriverObs> {
        let mut out = Vec::new();
        for (si, season) in [2021, 2022, 2023].iter().enumerate() {
            let skill: HashMap<&str, f64> =
                [("A", -0.3), ("B", b_by_season[si]), ("C", 0.0), ("D", 0.4), ("E", -0.1), ("F", 0.2)].into();
            for r in 0..10 {
                let lineup: [(&str, &str, &str); 3] = match si {
                    0 => [("Red", "A", "B"), ("Blue", "C", "D"), ("Green", "E", "F")],
                    1 => [("Red", "A", "C"), ("Blue", "B", "E"), ("Green", "D", "F")],
                    _ => [("Red", "A", "D"), ("Blue", "C", "F"), ("Green", "B", "E")],
                };
                let car: HashMap<&str, f64> = [("Red", 0.0), ("Blue", 0.5 + 0.02 * r as f64), ("Green", 1.0)].into();
                let race = format!("{season}-{r:02}");
                let raw: Vec<(&str, &str, f64)> = lineup
                    .iter()
                    .flat_map(|(t, a, b)| [(*t, *a, car[t] + skill[a]), (*t, *b, car[t] + skill[b])])
                    .collect();
                let best = raw.iter().map(|x| x.2).fold(f64::INFINITY, f64::min);
                for (t, d, v) in raw {
                    out.push(obs(&race, d, t, v - best));
                }
            }
        }
        out
    }

    fn car_gaps(all: &[DriverObs], race: &str, drift: f64) -> CarRun {
        let others: Vec<&DriverObs> = all.iter().filter(|o| o.race != race).collect();
        let skills = fit_skills(&others, drift);
        let mine: Vec<&DriverObs> = all.iter().filter(|o| o.race == race).collect();
        car_run(Condition::Dry, &mine, &skills, 90.0)
    }

    #[test]
    fn recovers_car_gaps_with_the_driver_removed() {
        let run = car_gaps(&grid([0.1, 0.1, 0.1]), "2023-05", 0.1);
        let gap = |t: &str| run.teams.iter().find(|c| c.team == t).unwrap().gap_pct;
        assert!(gap("Red").abs() < 1e-3, "{run:?}");
        assert!((gap("Blue") - 0.6).abs() < 1e-3, "{run:?}");
        assert!((gap("Green") - 1.0).abs() < 1e-3, "{run:?}");
        assert!(run.teams.iter().all(|t| t.note.is_none()));
    }

    #[test]
    fn a_driver_whose_form_changes_between_seasons_is_followed() {
        // B is much slower in his first season than later. With few links, teammate gaps
        // cannot fully tell "B improved" from "his teammates got slower" (the smallest
        // total change splits it between them), but a season skill with the drift chosen
        // by cross-validation must still beat one fixed skill clearly.
        let all = grid([0.8, 0.1, 0.1]);
        let refs: Vec<&DriverObs> = all.iter().collect();
        let drift = choose_drift(&refs);
        assert!(drift >= 0.3, "drift {drift}");
        let error = |drift: f64| {
            let run = car_gaps(&all, "2023-05", drift);
            let gap = |t: &str| run.teams.iter().find(|c| c.team == t).unwrap().gap_pct;
            (gap("Blue") - 0.6).abs() + (gap("Green") - 1.0).abs()
        };
        let (chosen, fixed) = (error(drift), error(SKILL_DRIFT_CANDIDATES[0]));
        assert!(chosen < 0.8 * fixed, "chosen {chosen} fixed {fixed}");
    }

    #[test]
    fn a_rookie_is_not_removed_and_the_race_is_marked() {
        let mut all = grid([0.1, 0.1, 0.1]);
        // G replaces F in one race only: no comparisons elsewhere.
        for o in all.iter_mut().filter(|o| o.race == "2023-05" && o.driver == "F") {
            o.driver = "G".into();
        }
        let run = car_gaps(&all, "2023-05", 0.1);
        let blue = run.teams.iter().find(|c| c.team == "Blue").unwrap();
        assert_eq!(blue.drivers, vec!["C".to_string()]);
        assert_eq!(blue.note.as_deref(), Some(ROOKIE_NOTE));
        assert!((blue.gap_pct - 0.6).abs() < 1e-3);
    }

    #[test]
    fn unlinked_teams_are_not_established() {
        let mut all = grid([0.1, 0.1, 0.1]);
        for season in [2021, 2022, 2023] {
            for r in 0..10 {
                for (d, v) in [("X", 0.0), ("Y", 0.2)] {
                    all.push(obs(&format!("{season}-{r:02}"), d, "Solo", 2.0 + v));
                }
            }
        }
        let refs: Vec<&DriverObs> = all.iter().collect();
        let skills = fit_skills(&refs, 0.1);
        assert!(skills.established("A"));
        assert!(!skills.established("X"));
    }

    #[test]
    fn wet_skill_needs_enough_wet_teammate_races() {
        let mut all = grid([0.1, 0.1, 0.1]);
        for o in all.iter_mut().filter(|o| o.race == "2022-03" || o.race == "2022-04") {
            o.condition = Condition::Wet;
        }
        let refs: Vec<&DriverObs> = all.iter().collect();
        let skills = fit_skills(&refs, 0.1);
        assert!(skills.wet_delta.is_empty(), "two wet races are not enough");
    }
}
