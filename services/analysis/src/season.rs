//! A team race pace recipe run on every race of a season, summarised per team.
//!
//! Every race keeps its own measurement. The season summary gives, per team, the
//! mean and the median of its per-race gaps, in percent and in seconds, both as
//! measured (the season's best car is above zero when it was not the quickest in
//! every race) and rebased so the best car is zero. It is given twice: over every
//! race that counts, and over the common races where every team was measured, so
//! teams missing difficult races are not flattered.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use serde::Serialize;

use crate::archive::Archive;
use crate::pace::{self, RacePace, Recipe};

/// One race of the season and whether it counts.
#[derive(Debug, Clone, Serialize)]
pub struct RaceEntry {
    pub event_key: String,
    pub session_id: String,
    pub counted: bool,
    /// Why the race does not count.
    pub reason: Option<String>,
    pub pace: Option<RacePace>,
}

/// Mean and median of a team's per-race gaps, as measured and rebased so the best
/// team in the table is zero.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Summary {
    pub mean: f64,
    pub median: f64,
    pub mean_rebased: f64,
    pub median_rebased: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeasonTeam {
    pub team: String,
    /// Races of the table this team was measured in.
    pub races_measured: usize,
    pub pct: Summary,
    pub s: Summary,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeasonTable {
    /// Event keys of the races in this table.
    pub races: Vec<String>,
    /// Lowest mean percentage gap first.
    pub teams: Vec<SeasonTeam>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeasonPace {
    pub season: i64,
    pub recipe: String,
    /// Percentiles of the stint resamples behind each race's intervals.
    pub interval: [f64; 2],
    /// Every race that counts; a team's figures use the races it was measured in.
    pub all_races: SeasonTable,
    /// Only races where every team of the season was measured.
    pub common_races: SeasonTable,
    pub races: Vec<RaceEntry>,
}

pub fn season_pace(archive: &Archive, season: i64, recipe: &Recipe, resamples: usize) -> Result<SeasonPace> {
    let mut races = Vec::new();
    for event_key in archive.season_events(season)? {
        let Some(race) = archive.session_of_type(&event_key, "R")? else { continue };
        let laps = archive.laps(&race.session_id)?;
        let pace = pace::race_pace(&laps, recipe, resamples, pace::seed_for(&race.session_id));
        let reason = match &pace {
            None => Some("no laps could be measured".to_string()),
            Some(p) if recipe.require_tyre_corrected && !p.tyre_corrected => {
                Some("cars could only be compared without the tyre-wear term".to_string())
            }
            Some(p) if 2 * p.teams.len() < p.teams_in_race => Some("fewer than half of the teams measured".to_string()),
            Some(_) => None,
        };
        races.push(RaceEntry { event_key, session_id: race.session_id, counted: reason.is_none(), reason, pace });
    }
    let counted: Vec<&RaceEntry> = races.iter().filter(|r| r.counted).collect();
    let season_teams: BTreeSet<&str> =
        counted.iter().flat_map(|r| r.pace.as_ref().unwrap().teams.iter().map(|t| t.team.as_str())).collect();
    let common: Vec<&RaceEntry> = counted
        .iter()
        .copied()
        .filter(|r| {
            let measured: BTreeSet<&str> = r.pace.as_ref().unwrap().teams.iter().map(|t| t.team.as_str()).collect();
            season_teams.is_subset(&measured)
        })
        .collect();
    Ok(SeasonPace {
        season,
        recipe: recipe.name.to_string(),
        interval: [pace::INTERVAL.0, pace::INTERVAL.1],
        all_races: table(&counted),
        common_races: table(&common),
        races,
    })
}

fn table(races: &[&RaceEntry]) -> SeasonTable {
    let mut pct: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut secs: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for race in races {
        for t in &race.pace.as_ref().expect("counted races are measured").teams {
            pct.entry(t.team.clone()).or_default().push(t.gap_pct);
            secs.entry(t.team.clone()).or_default().push(t.gap_s);
        }
    }
    let stats = |values: &[f64]| {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        (mean, pace::median(&mut values.to_vec()))
    };
    let raw: Vec<(String, usize, (f64, f64), (f64, f64))> =
        pct.iter().map(|(team, p)| (team.clone(), p.len(), stats(p), stats(&secs[team]))).collect();
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
    SeasonTable { races: races.iter().map(|r| r.event_key.clone()).collect(), teams }
}
