//! One team-pace recipe run on every race of a season.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;

use crate::archive::Archive;
use crate::laps;
use crate::pace::{self, Agg, RaceFilter, RaceModel};

/// One race's percentage gaps to the quickest team.
pub struct RaceGaps {
    pub event_key: String,
    pub gaps_pct: BTreeMap<String, f64>,
}

/// Per-race gaps for every race of the season that is archived. A race counts only
/// when at least half of the teams that took part are measured: gaps to the quickest
/// of a few teams would hand those teams a free near-zero. Races that fail (e.g. a
/// wet race measured on slick laps) give no gaps.
pub fn race_gaps(
    archive: &Archive,
    season: i64,
    filter: &RaceFilter,
    model: &RaceModel,
    agg: Agg,
    min_laps: usize,
) -> Result<Vec<RaceGaps>> {
    let mut out = Vec::new();
    for event_key in archive.season_events(season)? {
        let Some(race) = archive.session_of_type(&event_key, "R")? else { continue };
        let laps = archive.laps(&race.session_id)?;
        let times = pace::race_model_team_times(&laps, filter, model, agg, min_laps);
        let teams: BTreeSet<String> = laps::driver_teams(&laps).into_values().collect();
        let gaps_pct = if 2 * times.len() >= teams.len() { pace::gaps_pct(&times) } else { BTreeMap::new() };
        out.push(RaceGaps { event_key, gaps_pct });
    }
    Ok(out)
}

/// Team -> (mean percentage gap over the races it was measured in, races counted).
pub fn mean_gaps(races: &[RaceGaps]) -> BTreeMap<String, (f64, usize)> {
    let mut sums: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for race in races {
        for (team, g) in &race.gaps_pct {
            let e = sums.entry(team.clone()).or_default();
            e.0 += g;
            e.1 += 1;
        }
    }
    sums.into_iter().map(|(team, (s, n))| (team, (s / n as f64, n))).collect()
}
