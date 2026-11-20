//! Car-only race pace: team race pace with each driver's skill removed.
//!
//! Built on the standard team race pace and never changing it. Two stages:
//!
//! 1. Every driver's gap in every race, from the standard recipe run separately on
//!    dry-tyre and on wet-tyre laps (so wet skill stays apart from dry skill), with
//!    the precision of each gap from its stint-resampled interval.
//! 2. Across a window of seasons: gap = car (this team, this race, this condition)
//!    + driver skill + noise. The car term is removed by comparing teammates within
//!    each team and race; driver skills are linked across teams by drivers who
//!    changed teams, and fitted by weighted least squares. Skills are relative to
//!    the average driver, which shifts every car alike and leaves car gaps unchanged.
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
pub const WINDOW_SEASONS: i64 = 3;
/// Races with a teammate comparison a driver needs in the window to be established.
pub const MIN_TEAMMATE_RACES: usize = 8;
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

fn observations(runs: &[RaceRun]) -> Vec<DriverObs> {
    let mut out = Vec::new();
    for run in runs {
        for d in &run.pace.drivers {
            let sd = d.interval_pct.map_or(f64::NAN, |[lo, hi]| (hi - lo) / (2.0 * Z80));
            out.push(DriverObs {
                race: run.event_key.clone(),
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

/// Driver skills (percent, relative to the average driver; lower is faster) fitted
/// from teammate comparisons.
#[derive(Debug, Clone, Default)]
pub struct Skills {
    dry: HashMap<String, f64>,
    wet_delta: HashMap<String, f64>,
    variance: HashMap<(String, Condition), f64>,
    /// Drivers linked to the largest connected part of the grid.
    linked: BTreeSet<String>,
    /// Races in which each driver had a measured teammate.
    teammate_races: HashMap<String, usize>,
}

impl Skills {
    pub fn established(&self, driver: &str) -> bool {
        self.linked.contains(driver) && self.teammate_races.get(driver).copied().unwrap_or(0) >= MIN_TEAMMATE_RACES
    }

    pub fn skill(&self, driver: &str, condition: Condition) -> f64 {
        let dry = self.dry.get(driver).copied().unwrap_or(0.0);
        match condition {
            Condition::Dry => dry,
            Condition::Wet => dry + self.wet_delta.get(driver).copied().unwrap_or(0.0),
        }
    }

    fn variance(&self, driver: &str, condition: Condition) -> f64 {
        self.variance.get(&(driver.to_string(), condition)).copied().unwrap_or(0.0)
    }
}

/// Fit driver skills from teammate comparisons: within each team, race and
/// condition, every driver's gap minus the team's precision-weighted mean gap equals
/// his skill minus his teammates' weighted mean skill (the car term cancels).
pub fn fit_skills(obs: &[&DriverObs]) -> Skills {
    let mut groups: BTreeMap<(&str, Condition, &str), Vec<&DriverObs>> = BTreeMap::new();
    for o in obs {
        groups.entry((o.race.as_str(), o.condition, o.team.as_str())).or_default().push(o);
    }
    groups.retain(|_, g| g.len() >= 2);

    let drivers: Vec<String> =
        groups.values().flatten().map(|o| o.driver.clone()).collect::<BTreeSet<_>>().into_iter().collect();
    let index: HashMap<&str, usize> = drivers.iter().enumerate().map(|(i, d)| (d.as_str(), i)).collect();
    let n = drivers.len();
    if n == 0 {
        return Skills::default();
    }

    // Columns: dry skill per driver, then wet-minus-dry per driver.
    let column = |driver: &str, condition: Condition| -> Vec<(usize, f64)> {
        let i = index[driver];
        match condition {
            Condition::Dry => vec![(i, 1.0)],
            Condition::Wet => vec![(i, 1.0), (n + i, 1.0)],
        }
    };
    let mut rows: Vec<(Vec<f64>, f64, f64)> = Vec::new();
    for g in groups.values() {
        let weights: Vec<f64> = g.iter().map(|o| 1.0 / (o.sd_pct * o.sd_pct)).collect();
        let total: f64 = weights.iter().sum();
        let mean_gap: f64 = g.iter().zip(&weights).map(|(o, w)| o.gap_pct * w).sum::<f64>() / total;
        let mut mean_x = vec![0.0; 2 * n];
        for (o, w) in g.iter().zip(&weights) {
            for (c, v) in column(&o.driver, o.condition) {
                mean_x[c] += v * w / total;
            }
        }
        for (o, w) in g.iter().zip(&weights) {
            let mut x: Vec<f64> = mean_x.iter().map(|m| -m).collect();
            for (c, v) in column(&o.driver, o.condition) {
                x[c] += v;
            }
            rows.push((x, o.gap_pct - mean_gap, *w));
        }
    }
    let m = rows.len();
    let mut data = Vec::with_capacity(m * 2 * n);
    let mut y = Vec::with_capacity(m);
    for (x, target, w) in &rows {
        let s = w.sqrt();
        data.extend(x.iter().map(|v| v * s));
        y.push(target * s);
    }
    let x = DMatrix::from_row_slice(m, 2 * n, &data);
    let y = DVector::from_vec(y);
    let svd = x.clone().svd(true, true);
    let beta = svd.solve(&y, 1e-10).unwrap_or_else(|_| DVector::zeros(2 * n));

    // Covariance: dispersion x pseudo-inverse of X'WX.
    let largest = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let keep: Vec<usize> = (0..svd.singular_values.len()).filter(|&i| svd.singular_values[i] > 1e-9 * largest).collect();
    let rank = keep.len();
    let residual = &y - &x * &beta;
    let dispersion = if m > rank { residual.norm_squared() / (m - rank) as f64 } else { 1.0 };
    let v_t = svd.v_t.as_ref().expect("svd computed with V");
    let cov = |a: usize, b: usize| -> f64 {
        keep.iter().map(|&k| v_t[(k, a)] * v_t[(k, b)] / svd.singular_values[k].powi(2)).sum::<f64>() * dispersion
    };

    let mut skills = Skills::default();
    for (i, d) in drivers.iter().enumerate() {
        skills.dry.insert(d.clone(), beta[i]);
        skills.wet_delta.insert(d.clone(), beta[n + i]);
        skills.variance.insert((d.clone(), Condition::Dry), cov(i, i));
        skills.variance.insert((d.clone(), Condition::Wet), cov(i, i) + cov(n + i, n + i) + 2.0 * cov(i, n + i));
    }

    // Links: teammates in a race join their drivers; keep the part with most comparisons.
    let mut parent: Vec<usize> = (0..n).collect();
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
    for g in groups.values() {
        let first = index[g[0].driver.as_str()];
        for o in g {
            let (a, b) = (find(&mut parent, first), find(&mut parent, index[o.driver.as_str()]));
            parent[a] = b;
            races_of.entry(o.driver.as_str()).or_default().insert(o.race.as_str());
        }
    }
    let mut size: HashMap<usize, usize> = HashMap::new();
    for d in &drivers {
        let root = find(&mut parent, index[d.as_str()]);
        *size.entry(root).or_default() += races_of[d.as_str()].len();
    }
    let main = size.iter().max_by_key(|(_, s)| **s).map(|(r, _)| *r);
    for d in &drivers {
        if Some(find(&mut parent, index[d.as_str()])) == main {
            skills.linked.insert(d.clone());
        }
        skills.teammate_races.insert(d.clone(), races_of[d.as_str()].len());
    }
    skills
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
            let var = o.sd_pct * o.sd_pct + skills.variance(&o.driver, condition);
            sum += (o.gap_pct - skills.skill(&o.driver, condition)) / var;
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
    pub all_races: SeasonTable,
    pub common_races: SeasonTable,
    pub races: Vec<CarRace>,
}

/// Car-only pace of every race of `season`, from the runs of every season of its
/// window of `window_seasons` seasons (`runs[season]` for each).
pub fn car_season(season: i64, runs: &BTreeMap<i64, Vec<RaceRun>>, window_seasons: i64) -> CarSeason {
    let window: Vec<&RaceRun> =
        runs.iter().filter(|(s, _)| **s > season - window_seasons && **s <= season).flat_map(|(_, r)| r).collect();
    let all_obs: Vec<DriverObs> = observations(&window.iter().map(|r| (*r).clone()).collect::<Vec<_>>());
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
        let skills = fit_skills(&others);
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

/// Internal: every driver's fitted skill over a window (percent, relative to the
/// average driver; negative is faster), dry and wet, with teammate races. Not part
/// of the published outputs.
pub fn skills_table(runs: &[&RaceRun]) -> Vec<(String, f64, f64, usize, bool)> {
    let obs = observations(&runs.iter().map(|r| (*r).clone()).collect::<Vec<_>>());
    let refs: Vec<&DriverObs> = obs.iter().collect();
    let skills = fit_skills(&refs);
    let linked: Vec<&String> = skills.linked.iter().collect();
    let mean = linked.iter().map(|d| skills.skill(d, Condition::Dry)).sum::<f64>() / linked.len().max(1) as f64;
    let mut out: Vec<(String, f64, f64, usize, bool)> = skills
        .dry
        .keys()
        .map(|d| {
            (
                d.clone(),
                skills.skill(d, Condition::Dry) - mean,
                skills.skill(d, Condition::Wet) - mean,
                skills.teammate_races[d],
                skills.established(d),
            )
        })
        .collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(race: &str, driver: &str, team: &str, gap: f64) -> DriverObs {
        DriverObs {
            race: race.into(),
            condition: Condition::Dry,
            driver: driver.into(),
            team: team.into(),
            gap_pct: gap,
            sd_pct: 0.1,
        }
    }

    /// Three teams over 30 races; drivers move between teams, so skills are linked
    /// across the grid. Each race's gaps are car + skill, measured to the quickest.
    fn grid() -> Vec<DriverObs> {
        let skill: HashMap<&str, f64> = [("A", -0.3), ("B", 0.1), ("C", 0.0), ("D", 0.4), ("E", -0.1), ("F", 0.2)].into();
        let mut out = Vec::new();
        for r in 0..30 {
            // Pairs change at race 10 and 20.
            let lineup: [(&str, &str, &str); 3] = match r {
                0..=9 => [("Red", "A", "B"), ("Blue", "C", "D"), ("Green", "E", "F")],
                10..=19 => [("Red", "A", "C"), ("Blue", "B", "E"), ("Green", "D", "F")],
                _ => [("Red", "A", "D"), ("Blue", "C", "F"), ("Green", "B", "E")],
            };
            let car: HashMap<&str, f64> = [("Red", 0.0), ("Blue", 0.5 + 0.02 * r as f64), ("Green", 1.0)].into();
            let race = format!("R{r:02}");
            let raw: Vec<(&str, &str, f64)> = lineup
                .iter()
                .flat_map(|(t, a, b)| [(*t, *a, car[t] + skill[a]), (*t, *b, car[t] + skill[b])])
                .collect();
            let best = raw.iter().map(|x| x.2).fold(f64::INFINITY, f64::min);
            for (t, d, v) in raw {
                out.push(obs(&race, d, t, v - best));
            }
        }
        out
    }

    #[test]
    fn recovers_car_gaps_with_the_driver_removed() {
        let all = grid();
        let race = "R25";
        let others: Vec<&DriverObs> = all.iter().filter(|o| o.race != race).collect();
        let skills = fit_skills(&others);
        let mine: Vec<&DriverObs> = all.iter().filter(|o| o.race == race).collect();
        let run = car_run(Condition::Dry, &mine, &skills, 90.0);
        let gap = |t: &str| run.teams.iter().find(|c| c.team == t).unwrap().gap_pct;
        assert!(gap("Red").abs() < 1e-6);
        assert!((gap("Blue") - (0.5 + 0.02 * 25.0)).abs() < 1e-6, "{run:?}");
        assert!((gap("Green") - 1.0).abs() < 1e-6, "{run:?}");
        assert!(run.teams.iter().all(|t| t.note.is_none()));
    }

    #[test]
    fn a_rookie_is_not_removed_and_the_race_is_marked() {
        let mut all = grid();
        // G replaces F in race 25 only: no comparisons elsewhere.
        for o in all.iter_mut().filter(|o| o.race == "R25" && o.driver == "F") {
            o.driver = "G".into();
        }
        let others: Vec<&DriverObs> = all.iter().filter(|o| o.race != "R25").collect();
        let skills = fit_skills(&others);
        assert!(!skills.established("G"));
        let mine: Vec<&DriverObs> = all.iter().filter(|o| o.race == "R25").collect();
        let run = car_run(Condition::Dry, &mine, &skills, 90.0);
        let blue = run.teams.iter().find(|c| c.team == "Blue").unwrap();
        assert_eq!(blue.drivers, vec!["C".to_string()]);
        assert_eq!(blue.note.as_deref(), Some(ROOKIE_NOTE));
        assert!((blue.gap_pct - (0.5 + 0.02 * 25.0)).abs() < 1e-6);
    }

    #[test]
    fn unlinked_teams_are_not_established() {
        // Two teams whose drivers never meet anyone else: they form their own part.
        let mut all = grid();
        for r in 0..30 {
            for (d, v) in [("X", 0.0), ("Y", 0.2)] {
                all.push(obs(&format!("R{r:02}"), d, "Solo", 2.0 + v));
            }
        }
        let refs: Vec<&DriverObs> = all.iter().collect();
        let skills = fit_skills(&refs);
        assert!(skills.established("A"));
        assert!(!skills.established("X"));
    }
}
