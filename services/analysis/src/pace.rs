//! Team race pace of one race, measured from that race's own laps.
//!
//! A least-squares fit of the kept laps separates each driver-car from tyre
//! compound, tyre age, fuel and track evolution, all measured within the race: no
//! priors, no weighting across races. Two recipes exist side by side:
//!
//! - [`Recipe::v1`], the first adopted recipe, kept unchanged so results can be
//!   reproduced and compared;
//! - [`Recipe::v2`], v1 with stricter evidence rules: clear air checked at all three
//!   timing lines, laps with unknown traffic left out, the five-lap minimum
//!   rechecked after trimming, and a race whose cars can only be compared without the
//!   tyre-wear term marked as not tyre-corrected instead of silently using that fit.
//!
//! [`race_pace`] gives the full result of one race: every driver and team with
//! their gaps, an interval from resampling whole stints, and the evidence behind
//! each number.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use nalgebra::{DMatrix, DVector};
use serde::Serialize;

use crate::laps::{self, Lap};

/// Team -> representative lap time in seconds.
pub type TeamTimes = BTreeMap<String, f64>;

/// How a team's cars combine into one number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agg {
    /// Fit each driver, keep the quicker driver: the car's best demonstrated pace.
    Best,
    /// Fit each driver, take the mean of the drivers.
    Mean,
    /// On every lap number keep the team's quickest kept lap (either car), then fit
    /// the team on those laps.
    LapBest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compounds {
    /// Dry, intermediate and wet tyres (laps with no known compound are left out).
    All,
    Slicks,
    /// Intermediate and wet tyres.
    Inters,
}

/// Where clear air is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Traffic {
    /// At the finish line only.
    FinishLine,
    /// At both sector lines and the finish line.
    TimingLines,
}

/// Which race laps count.
#[derive(Debug, Clone, Copy)]
pub struct RaceFilter {
    pub compounds: Compounds,
    /// Laps completed this close or closer to the car ahead are left out.
    pub min_gap_s: Option<f64>,
    pub traffic: Traffic,
    /// Keep laps whose traffic cannot be determined (a crossing time is missing).
    pub keep_unknown_traffic: bool,
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
        Self {
            compounds: Compounds::All,
            min_gap_s: None,
            traffic: Traffic::FinishLine,
            keep_unknown_traffic: true,
            skip_laps: 1,
            accurate_only: true,
            skip_after_neutral: 0,
        }
    }
}

/// How lap number and race time enter the fit: fuel burn and track evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LapTrend {
    /// One slope for the race.
    Linear,
    /// One slope per compound.
    LinearPerCompound,
    /// A separate time for every lap number, shared by all cars.
    PerLap,
    /// A separate time for every block of race time (one median lap long), shared by
    /// all cars, for track conditions; plus one slope on each car's own lap count for
    /// fuel. A lapped car's lap meets the conditions of the moment it was driven.
    TimeBlocks,
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
    /// Recheck the lap minimum after trimming and refit without drivers below it.
    pub recheck_min_laps: bool,
}

/// Laps a driver (or team) needs to be measured.
pub const MIN_LAPS: usize = 5;

impl RaceFilter {
    /// v1 laps: any known tyre (dry, intermediate, wet), green, after lap 1, no pit
    /// laps, more than 2 s behind the car ahead at the finish line.
    pub fn standard() -> Self {
        Self { compounds: Compounds::All, min_gap_s: Some(2.0), ..Self::default() }
    }

    /// v2 laps: as v1, but clear by more than 2 s at all three timing lines, and laps
    /// whose traffic is unknown are left out.
    pub fn standard_v2() -> Self {
        Self { traffic: Traffic::TimingLines, keep_unknown_traffic: false, ..Self::standard() }
    }
}

impl RaceModel {
    /// v1 fit: a time per lap number shared by all cars, one tyre-wear rate per
    /// compound, one 3-sigma trim and refit.
    pub fn standard() -> Self {
        Self {
            team_wear: false,
            lap_trend: LapTrend::PerLap,
            trim_sigma: Some(3.0),
            traffic_term: false,
            recheck_min_laps: false,
        }
    }

    /// v2 fit: as v1, with the lap minimum rechecked after trimming.
    pub fn standard_v2() -> Self {
        Self { recheck_min_laps: true, ..Self::standard() }
    }
}

/// A complete team race pace recipe.
#[derive(Debug, Clone, Copy)]
pub struct Recipe {
    pub name: &'static str,
    pub filter: RaceFilter,
    pub model: RaceModel,
    pub agg: Agg,
    pub min_laps: usize,
    /// A race measured only without the tyre-wear term does not count in a season.
    pub require_tyre_corrected: bool,
}

impl Recipe {
    pub fn v1() -> Self {
        Self {
            name: "v1",
            filter: RaceFilter::standard(),
            model: RaceModel::standard(),
            agg: Agg::Best,
            min_laps: MIN_LAPS,
            require_tyre_corrected: false,
        }
    }

    pub fn v2() -> Self {
        Self {
            name: "v2",
            filter: RaceFilter::standard_v2(),
            model: RaceModel::standard_v2(),
            agg: Agg::Best,
            min_laps: MIN_LAPS,
            require_tyre_corrected: true,
        }
    }

    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "v1" => Some(Self::v1()),
            "v2" => Some(Self::v2()),
            _ => None,
        }
    }
}

/// v1 team race pace of one race (the quicker driver of each team).
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

pub(crate) fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

fn is_neutralised(lap: &Lap) -> bool {
    lap.track_status.as_deref().is_some_and(|s| s.contains(['4', '5', '6', '7']))
}

/// How the race's laps were used.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct LapCounts {
    /// Laps passing every rule except traffic.
    pub candidates: usize,
    /// Of those, laps whose traffic could not be determined.
    pub unknown_traffic: usize,
    /// Laps kept for the fit (before tyre age, lap minimum and trimming).
    pub kept: usize,
}

/// The race laps a filter keeps, with each lap's gap to the car ahead at the line.
pub fn race_laps<'a>(race: &'a [Lap], filter: &RaceFilter) -> Vec<(&'a Lap, f64, Option<f64>)> {
    select_laps(race, filter).0
}

fn select_laps<'a>(race: &'a [Lap], filter: &RaceFilter) -> (Vec<(&'a Lap, f64, Option<f64>)>, LapCounts) {
    let finish_gaps = laps::gaps_ahead(race);
    let line_gaps = (filter.traffic == Traffic::TimingLines).then(|| laps::gaps_at_timing_lines(race));
    let neutral: BTreeSet<u32> = race.iter().filter(|l| is_neutralised(l)).map(|l| l.lap_number).collect();
    let after_neutral = |n: u32| {
        filter.skip_after_neutral > 0
            && neutral.range(n.saturating_sub(filter.skip_after_neutral)..n).next().is_some()
    };
    let mut counts = LapCounts::default();
    let mut kept = Vec::new();
    for (i, lap) in race.iter().enumerate() {
        let Some(t) = lap.lap_time else { continue };
        let compound_ok = match filter.compounds {
            Compounds::All => lap.is_slick() || lap.is_wet_tyre(),
            Compounds::Slicks => lap.is_slick(),
            Compounds::Inters => lap.is_wet_tyre(),
        };
        let eligible = compound_ok
            && lap.is_green()
            && !lap.is_pit_lap()
            && !lap.deleted
            && lap.lap_number > filter.skip_laps
            && (!filter.accurate_only || lap.accurate)
            && !after_neutral(lap.lap_number);
        if !eligible {
            continue;
        }
        counts.candidates += 1;
        let clear = match (filter.min_gap_s, &line_gaps) {
            (None, _) => true,
            (Some(min), None) => finish_gaps[i].is_none_or(|g| g > min),
            (Some(min), Some(lines)) => {
                let status: Vec<Option<bool>> = lines[i].iter().map(|g| g.clear_of(min)).collect();
                if status.contains(&Some(false)) {
                    false
                } else if status.contains(&None) {
                    counts.unknown_traffic += 1;
                    filter.keep_unknown_traffic
                } else {
                    true
                }
            }
        };
        if clear {
            kept.push((lap, t, finish_gaps[i]));
        }
    }
    counts.kept = kept.len();
    (kept, counts)
}

/// One lap entering the fit. `entity` is the driver, or the team for [`Agg::LapBest`].
#[derive(Debug, Clone)]
struct Obs {
    entity: String,
    team: String,
    compound: String,
    age: f64,
    lap: u32,
    stint: u32,
    /// Session time at the middle of the lap (NaN when unknown).
    clock: f64,
    time: f64,
    gap: Option<f64>,
}

fn observations(race: &[Lap], filter: &RaceFilter, agg: Agg, min_laps: usize) -> (Vec<Obs>, LapCounts) {
    let teams = laps::driver_teams(race);
    let (kept, counts) = select_laps(race, filter);
    let mut obs: Vec<Obs> = kept
        .into_iter()
        // Tyre age is a fit input: a lap without it cannot be placed on the wear curve.
        .filter(|(lap, _, _)| lap.tyre_life.is_some())
        .map(|(lap, time, gap)| {
            let team = teams.get(&lap.driver).cloned().unwrap_or_default();
            Obs {
                entity: if agg == Agg::LapBest { team.clone() } else { lap.driver.clone() },
                team,
                compound: lap.compound.clone().unwrap_or_default(),
                age: lap.tyre_life.expect("filtered above"),
                lap: lap.lap_number,
                stint: lap.stint.unwrap_or(0),
                clock: lap.time.map_or(f64::NAN, |end| end - time / 2.0),
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
    retain_min_laps(&mut obs, min_laps);
    (obs, counts)
}

fn retain_min_laps(obs: &mut Vec<Obs>, min_laps: usize) -> bool {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for o in obs.iter() {
        *counts.entry(o.entity.clone()).or_default() += 1;
    }
    let before = obs.len();
    obs.retain(|o| counts[&o.entity] >= min_laps);
    obs.len() != before
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
/// told apart from the lap terms; the fit is then repeated without the wear term and
/// the version that compares more cars is kept (see [`race_pace`] for how v2 marks it).
pub fn race_model_team_times(race: &[Lap], filter: &RaceFilter, model: &RaceModel, agg: Agg, min_laps: usize) -> TeamTimes {
    let (obs, _) = observations(race, filter, agg, min_laps);
    if obs.is_empty() {
        return TeamTimes::new();
    }
    let measured = measure(obs, model, min_laps);
    team_times(&measured.fit.cars, agg)
}

/// A fitted driver (or team for [`Agg::LapBest`]).
#[derive(Debug, Clone)]
struct CarFit {
    entity: String,
    team: String,
    time: f64,
}

struct FitResult {
    cars: Vec<CarFit>,
    all_compared: bool,
    /// The laps the final fit used.
    obs: Vec<Obs>,
}

struct Measured {
    fit: FitResult,
    tyre_corrected: bool,
}

fn measure(obs: Vec<Obs>, model: &RaceModel, min_laps: usize) -> Measured {
    let with_wear = fit_cars(obs.clone(), model, true, min_laps);
    if with_wear.all_compared {
        return Measured { fit: with_wear, tyre_corrected: true };
    }
    let without_wear = fit_cars(obs, model, false, min_laps);
    if without_wear.cars.len() > with_wear.cars.len() {
        Measured { fit: without_wear, tyre_corrected: false }
    } else {
        Measured { fit: with_wear, tyre_corrected: true }
    }
}

fn team_times(cars: &[CarFit], agg: Agg) -> TeamTimes {
    let mut by_team: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for c in cars {
        by_team.entry(c.team.clone()).or_default().push(c.time);
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

/// One fit: each comparable entity's time, and whether every entity was comparable.
fn fit_cars(mut obs: Vec<Obs>, model: &RaceModel, wear: bool, min_laps: usize) -> FitResult {
    if model.lap_trend == LapTrend::TimeBlocks {
        obs.retain(|o| o.clock.is_finite());
    }
    let empty = || FitResult { cars: Vec::new(), all_compared: false, obs: Vec::new() };
    if obs.is_empty() {
        return empty();
    }
    let mut trimmed = false;
    let (design, beta, svd) = loop {
        let design = Design::new(&obs, model, wear);
        let (x, y) = design.matrix(&obs);
        let svd = x.clone().svd(true, true);
        let Ok(beta) = svd.solve(&y, 1e-10) else { return empty() };
        // Trim once, then refit; the final fit and its laps stay together.
        if let (Some(k), false) = (model.trim_sigma, trimmed) {
            trimmed = true;
            let residuals = &y - &x * &beta;
            let mut abs: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
            let sigma = 1.4826 * median(&mut abs);
            let keep: Vec<bool> = residuals.iter().map(|r| r.abs() <= k * sigma).collect();
            if !keep.iter().all(|&k| k) {
                obs = obs.into_iter().zip(keep).filter_map(|(o, k)| k.then_some(o)).collect();
                if model.recheck_min_laps {
                    retain_min_laps(&mut obs, min_laps);
                }
                if obs.is_empty() {
                    return empty();
                }
                continue;
            }
        }
        break (design, beta, svd);
    };

    let mut ages: Vec<f64> = obs.iter().map(|o| o.age).collect();
    let mut laps_run: Vec<f64> = obs.iter().map(|o| o.lap as f64).collect();
    let mut clocks: Vec<f64> = obs.iter().map(|o| o.clock).filter(|c| c.is_finite()).collect();
    let mut kept_times: Vec<f64> = obs.iter().map(|o| o.time).collect();
    let reference_age = median(&mut ages);
    let reference_lap = design.nearest_lap(median(&mut laps_run).round() as u32);
    let reference_clock = if clocks.is_empty() { f64::NAN } else { median(&mut clocks) };
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
        stint: 0,
        clock: reference_clock,
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
    let cars: Vec<CarFit> = counts
        .iter()
        .filter(|(e, _)| comparable(&rows[*e], &reference_row))
        .map(|(e, (_, team))| CarFit {
            entity: e.to_string(),
            team: team.to_string(),
            time: base + design.predict(&beta, &at(e, team)) - reference_time,
        })
        .collect();
    let all_compared = cars.len() == counts.len();
    let compared: BTreeSet<&str> = cars.iter().map(|c| c.entity.as_str()).collect();
    let obs = obs.into_iter().filter(|o| compared.contains(o.entity.as_str())).collect();
    FitResult { cars, all_compared, obs }
}

/// Uncertainty of a gap: the 10th and 90th percentiles over stint resamples of the
/// gap to the measurement's quickest team.
pub const INTERVAL: (f64, f64) = (0.10, 0.90);

/// One driver's measured race pace.
#[derive(Debug, Clone, Serialize)]
pub struct DriverPace {
    pub driver: String,
    pub team: String,
    pub gap_pct: f64,
    pub gap_s: f64,
    pub interval_pct: Option<[f64; 2]>,
    pub interval_s: Option<[f64; 2]>,
    /// Laps in the final fit.
    pub laps: usize,
    /// Stints those laps come from.
    pub stints: usize,
    /// Of those laps, how many share their lap number with another measured car.
    pub overlap_laps: usize,
}

/// One team's race pace: its quicker driver.
#[derive(Debug, Clone, Serialize)]
pub struct TeamPace {
    pub team: String,
    pub driver: String,
    pub gap_pct: f64,
    pub gap_s: f64,
    /// Interval of the gap to the quickest team of the measurement.
    pub interval_pct: Option<[f64; 2]>,
    pub interval_s: Option<[f64; 2]>,
    /// Share of stint resamples in which this team was the quickest.
    pub p_quickest: Option<f64>,
}

/// The full race pace result of one race.
#[derive(Debug, Clone, Serialize)]
pub struct RacePace {
    pub recipe: String,
    /// The fit could compare the cars with the tyre-wear term.
    pub tyre_corrected: bool,
    /// Median kept lap time: the base of the percentages.
    pub base_lap_s: f64,
    pub laps: LapCounts,
    /// Teams that took part in the race.
    pub teams_in_race: usize,
    /// Stint resamples drawn, and those that contained the quickest team (used).
    pub resamples: usize,
    pub resamples_used: usize,
    /// Quickest team first.
    pub teams: Vec<TeamPace>,
    /// Quickest driver first.
    pub drivers: Vec<DriverPace>,
}

/// Full race pace of one race under a recipe, with `resamples` stint resamples for
/// the intervals (deterministic for a given `seed`).
pub fn race_pace(race: &[Lap], recipe: &Recipe, resamples: usize, seed: u64) -> Option<RacePace> {
    let teams_in_race = laps::driver_teams(race).into_values().collect::<BTreeSet<_>>().len();
    let (obs, counts) = observations(race, &recipe.filter, recipe.agg, recipe.min_laps);
    if obs.is_empty() {
        return None;
    }
    let measured = measure(obs.clone(), &recipe.model, recipe.min_laps);
    if measured.fit.cars.is_empty() {
        return None;
    }
    let teams = team_times(&measured.fit.cars, recipe.agg);
    let quickest = teams.values().copied().fold(f64::INFINITY, f64::min);
    let mut base_laps: Vec<f64> = measured.fit.obs.iter().map(|o| o.time).collect();
    let base = median(&mut base_laps);
    let pct = |t: f64, best: f64| (t - best) / best * 100.0;

    // Stint resamples: whole stints drawn with replacement, the same fit repeated.
    let quickest_team = teams.iter().min_by(|a, b| a.1.total_cmp(b.1)).map(|(t, _)| t.clone()).unwrap_or_default();
    let mut team_draws: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut driver_draws: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut quickest_count: BTreeMap<String, usize> = BTreeMap::new();
    let mut used = 0usize;
    if resamples > 0 {
        let mut stints: BTreeMap<(String, u32), Vec<Obs>> = BTreeMap::new();
        for o in &obs {
            stints.entry((o.entity.clone(), o.stint)).or_default().push(o.clone());
        }
        let stints: Vec<Vec<Obs>> = stints.into_values().collect();
        let mut rng = SplitMix64(seed);
        for _ in 0..resamples {
            let mut sample = Vec::with_capacity(obs.len());
            for _ in 0..stints.len() {
                sample.extend(stints[rng.below(stints.len())].iter().cloned());
            }
            retain_min_laps(&mut sample, recipe.min_laps);
            if sample.is_empty() {
                continue;
            }
            let fit = fit_cars(sample, &recipe.model, measured.tyre_corrected, recipe.min_laps);
            if fit.cars.is_empty() {
                continue;
            }
            let draw_teams = team_times(&fit.cars, recipe.agg);
            // Gaps are measured to the same reference team in every resample (the
            // quickest in the measurement itself); a resample without it is skipped.
            let Some(reference) = draw_teams.get(&quickest_team).copied() else { continue };
            used += 1;
            let draw_best = draw_teams.iter().min_by(|a, b| a.1.total_cmp(b.1)).map(|(t, _)| t.clone());
            if let Some(best) = draw_best {
                *quickest_count.entry(best).or_default() += 1;
            }
            for (team, t) in &draw_teams {
                team_draws.entry(team.clone()).or_default().push(t - reference);
            }
            for c in &fit.cars {
                driver_draws.entry(c.entity.clone()).or_default().push(c.time - reference);
            }
        }
    }
    let interval = |draws: Option<&Vec<f64>>| -> (Option<[f64; 2]>, Option<[f64; 2]>) {
        match draws {
            Some(d) if d.len() >= 10 => {
                let mut d = d.clone();
                d.sort_by(|a, b| a.total_cmp(b));
                let s = [quantile(&d, INTERVAL.0), quantile(&d, INTERVAL.1)];
                (Some([s[0] / quickest * 100.0, s[1] / quickest * 100.0]), Some(s))
            }
            _ => (None, None),
        }
    };

    let mut team_rows: Vec<TeamPace> = teams
        .iter()
        .map(|(team, t)| {
            let driver = measured
                .fit
                .cars
                .iter()
                .filter(|c| &c.team == team)
                .min_by(|a, b| a.time.total_cmp(&b.time))
                .map(|c| c.entity.clone())
                .unwrap_or_default();
            let (interval_pct, interval_s) = interval(team_draws.get(team));
            let p_quickest = (used > 0).then(|| quickest_count.get(team).copied().unwrap_or(0) as f64 / used as f64);
            TeamPace {
                team: team.clone(),
                driver,
                gap_pct: pct(*t, quickest),
                gap_s: t - quickest,
                interval_pct,
                interval_s,
                p_quickest,
            }
        })
        .collect();
    team_rows.sort_by(|a, b| a.gap_s.total_cmp(&b.gap_s));

    let final_obs = &measured.fit.obs;
    let mut cars_on_lap: HashMap<u32, BTreeSet<&str>> = HashMap::new();
    for o in final_obs {
        cars_on_lap.entry(o.lap).or_default().insert(o.entity.as_str());
    }
    let mut driver_rows: Vec<DriverPace> = measured
        .fit
        .cars
        .iter()
        .map(|c| {
            let mine: Vec<&Obs> = final_obs.iter().filter(|o| o.entity == c.entity).collect();
            let stints = mine.iter().map(|o| o.stint).collect::<BTreeSet<_>>().len();
            let overlap_laps = mine.iter().filter(|o| cars_on_lap[&o.lap].len() > 1).count();
            let (interval_pct, interval_s) = interval(driver_draws.get(&c.entity));
            DriverPace {
                driver: c.entity.clone(),
                team: c.team.clone(),
                gap_pct: pct(c.time, quickest),
                gap_s: c.time - quickest,
                interval_pct,
                interval_s,
                laps: mine.len(),
                stints,
                overlap_laps,
            }
        })
        .collect();
    driver_rows.sort_by(|a, b| a.gap_s.total_cmp(&b.gap_s));

    Some(RacePace {
        recipe: recipe.name.to_string(),
        tyre_corrected: measured.tyre_corrected,
        base_lap_s: base,
        laps: counts,
        teams_in_race,
        resamples,
        resamples_used: used,
        teams: team_rows,
        drivers: driver_rows,
    })
}

/// A small deterministic random number generator for resampling.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A stable seed from a text key (FNV-1a), so a race's resamples never change.
pub fn seed_for(key: &str) -> u64 {
    key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
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
    /// Time blocks ([`LapTrend::TimeBlocks`]): start clock and block length.
    block_origin: f64,
    block_length: f64,
    blocks: Vec<i64>,
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
        let (block_origin, block_length) = if model.lap_trend == LapTrend::TimeBlocks {
            let mut times: Vec<f64> = obs.iter().map(|o| o.time).collect();
            let origin = obs.iter().map(|o| o.clock).fold(f64::INFINITY, f64::min);
            (origin, median(&mut times))
        } else {
            (0.0, 1.0)
        };
        let mut design = Self {
            model,
            wear,
            entities: unique(&|o| o.entity.clone()),
            compounds: unique(&|o| o.compound.clone()),
            reference_compound,
            teams: unique(&|o| o.team.clone()),
            laps: obs.iter().map(|o| o.lap).collect::<BTreeSet<_>>().into_iter().collect(),
            block_origin,
            block_length,
            blocks: Vec::new(),
        };
        if model.lap_trend == LapTrend::TimeBlocks {
            design.blocks = obs.iter().map(|o| design.raw_block(o.clock)).collect::<BTreeSet<_>>().into_iter().collect();
        }
        design
    }

    fn raw_block(&self, clock: f64) -> i64 {
        ((clock - self.block_origin) / self.block_length).floor() as i64
    }

    /// Index of the block a clock time falls in (the nearest block that has laps).
    fn block_index(&self, clock: f64) -> usize {
        let b = self.raw_block(clock);
        (0..self.blocks.len()).min_by_key(|&i| self.blocks[i].abs_diff(b)).unwrap_or(0)
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
        } else if self.model.team_wear {
            self.teams.len()
        } else {
            self.compounds.len()
        }
    }
    fn trend_start(&self) -> usize {
        self.wear_start() + self.wear_columns()
    }
    fn trend_columns(&self) -> usize {
        match self.model.lap_trend {
            LapTrend::Linear => 1,
            LapTrend::LinearPerCompound => self.compounds.len(),
            // The first lap number (or block) is the baseline; entity columns carry the level.
            LapTrend::PerLap => self.laps.len().saturating_sub(1),
            LapTrend::TimeBlocks => self.blocks.len().saturating_sub(1) + 1,
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
            LapTrend::TimeBlocks => {
                let k = self.block_index(o.clock);
                if k > 0 {
                    r[self.trend_start() + k - 1] = 1.0;
                }
                // Fuel: the car's own lap count.
                r[self.trend_start() + self.blocks.len().saturating_sub(1)] = o.lap as f64;
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
            sector1_time: None,
            sector2_time: None,
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
                let (compound, age, offset, wear, stint) =
                    if n <= stop { ("MEDIUM", n as f64, 0.0, 0.05, 1) } else { ("HARD", (n - stop) as f64, 0.4, 0.03, 2) };
                let track = 90.0 - 0.06 * n as f64 + if n % 7 == 0 { 1.5 } else { 0.0 };
                let mut l = lap(driver, team, n, track + pace + offset + wear * age, compound, age);
                l.stint = Some(stint);
                laps.push(l);
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

    /// v2 without the traffic rule: synthetic laps carry no crossing times.
    fn v2_untimed() -> Recipe {
        Recipe { filter: RaceFilter { min_gap_s: None, ..RaceFilter::standard_v2() }, ..Recipe::v2() }
    }

    /// Every car runs one dry stint from a different stop lap, with real tyre wear.
    fn single_stint_race() -> Vec<Lap> {
        let mut race = Vec::new();
        for (driver, team, pace, stop) in [("A1", "Alpha", 0.0, 35u32), ("B1", "Beta", 0.4, 40), ("C1", "Gamma", 0.8, 45)] {
            for n in (stop + 1)..=52 {
                let age = (n - stop) as f64;
                race.push(lap(driver, team, n, 88.0 + pace - 0.1 * n as f64 + 0.05 * age, "MEDIUM", age));
            }
        }
        race
    }

    #[test]
    fn v2_marks_races_compared_without_the_wear_term() {
        // Wear cannot be separated from the lap terms with one stint per car: the cars
        // are still compared, but the race is reported as not tyre-corrected.
        let pace = race_pace(&single_stint_race(), &v2_untimed(), 0, 1).expect("measured");
        assert!(!pace.tyre_corrected);
        assert_eq!(pace.teams.len(), 3);
    }

    #[test]
    fn v1_still_falls_back_silently() {
        assert_eq!(gaps_s(&team_race_pace(&single_stint_race())).len(), 3);
    }

    #[test]
    fn multi_stint_races_are_tyre_corrected_with_intervals() {
        let pace = race_pace(&synthetic_race(), &v2_untimed(), 50, 7).expect("measured");
        assert!(pace.tyre_corrected);
        let gamma = pace.teams.iter().find(|t| t.team == "Gamma").unwrap();
        assert!((gamma.gap_s - 1.2).abs() < 1e-6);
        let [lo, hi] = gamma.interval_s.expect("interval");
        assert!(lo <= 1.2 + 1e-6 && hi >= 1.2 - 1e-6, "{lo} {hi}");
        let b2 = pace.drivers.iter().find(|d| d.driver == "B2").unwrap();
        assert_eq!((b2.laps, b2.stints), (49, 2));
    }

    #[test]
    fn lap_minimum_is_rechecked_after_trimming() {
        // C1 has five laps, one of them clearly slow: trimming leaves four, below five.
        // Realistic lap-to-lap noise, so the robust spread is not zero.
        let mut race = synthetic_race();
        for (i, l) in race.iter_mut().enumerate() {
            l.lap_time = l.lap_time.map(|t| t + 0.15 * ((i as f64) * 1.7).sin());
        }
        race.retain(|l| l.driver != "C1" || l.lap_number <= 6);
        for l in race.iter_mut().filter(|l| l.driver == "C1" && l.lap_number == 6) {
            l.lap_time = l.lap_time.map(|t| t + 0.8);
        }
        let v1 = race_model_team_times(&race, &RaceFilter::standard(), &RaceModel::standard(), Agg::Best, MIN_LAPS);
        let v2 = race_model_team_times(&race, &v2_untimed().filter, &RaceModel::standard_v2(), Agg::Best, MIN_LAPS);
        assert!(v1.contains_key("Gamma"));
        assert!(!v2.contains_key("Gamma"));
    }

    #[test]
    fn time_blocks_follow_the_clock_not_the_lap_count() {
        // A lapped car's lap 20 is driven during the leader's lap 21; conditions change
        // by the clock. Time blocks recover the true gap.
        let mut race = Vec::new();
        for (driver, team, pace, behind) in [("A1", "Alpha", 0.0, 0.0), ("B1", "Beta", 2.0, 92.0), ("C1", "Gamma", 1.0, 40.0)] {
            let mut clock = 3600.0 + behind;
            for n in 1..=40u32 {
                let grip = if clock > 3600.0 + 92.0 * 20.0 { -1.5 } else { 0.0 };
                let t = 92.0 + pace + grip - 0.05 * n as f64;
                clock += t;
                let mut l = lap(driver, team, n, t, "MEDIUM", n as f64);
                l.time = Some(clock);
                race.push(l);
            }
        }
        let model = RaceModel { lap_trend: LapTrend::TimeBlocks, trim_sigma: None, ..RaceModel::standard_v2() };
        let s = gaps_s(&race_model_team_times(&race, &RaceFilter::default(), &model, Agg::Best, MIN_LAPS));
        assert!((s["Gamma"] - 1.0).abs() < 0.2 && (s["Beta"] - 2.0).abs() < 0.2, "{s:?}");
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
